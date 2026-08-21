using CommunityToolkit.Maui.Core;
using System.Diagnostics;
using SkiaSharp;
using FaceAttendanceApp.Services;
using FaceAttendanceApp.Model;

namespace FaceAttendanceApp
{
    public partial class RegisterWorkerPage : ContentPage
    {
        private readonly FaceDetectionService _detectionService;
        private readonly FaceRecognitionService _recognitionService;
        private readonly MaskHelmetDetectionService _ppeService;

        private WorkerDatabase? _database;
        private float[]? _lastEmbedding;

        // How many good captures to collect per label before averaging. 3 is a reasonable
        // floor — enough to cancel out per-shot noise (blink, tiny head-angle shift, momentary
        // focus hunt) without making enrollment feel tedious. Every sample is checked against
        // the SAME quality/duplicate/PPE rules as a single capture; only successful ones count.
        private const int SamplesRequired = 3;

        // Accumulates embeddings for the PRIMARY (bare-face) enrollment currently in progress.
        // Cleared on retake, on successful enroll, and whenever a duplicate is detected.
        private readonly List<float[]> _primarySamples = new();

        // Accumulates embeddings for the VARIANT (masked/spectacles/etc.) capture currently in
        // progress. Cleared whenever a variant session starts, completes, or is abandoned.
        private readonly List<float[]> _variantSamples = new();

        private CameraInfo? _frontCamera;
        private CameraInfo? _backCamera;
        private bool _usingFrontCamera = false;

        // Generic "capturing an extra face variant" state — replaces the old masked-only flow.
        // _pendingVariantLabel holds the label being captured (e.g. "With Mask"); null means
        // we're not currently in variant-capture mode.
        private string? _pendingVariantLabel = null;
        private int? _pendingVariantWorkerId = null;
        private string _pendingVariantWorkerName = string.Empty;

        private static readonly string[] SuggestedLabels =
        {
            "With Mask",
            "With Spectacles",
            "With Mask + Spectacles",
            "Custom..."
        };

        public RegisterWorkerPage(FaceDetectionService detectionService, FaceRecognitionService recognitionService, MaskHelmetDetectionService ppeService)
        {
            InitializeComponent();
            _detectionService = detectionService;
            _recognitionService = recognitionService;
            _ppeService = ppeService;
        }

        protected override async void OnAppearing()
        {
            base.OnAppearing();

            LoadingOverlay.IsVisible = true;
            LoadingLabel.Text = "Requesting camera permission...";

            var status = await Permissions.RequestAsync<Permissions.Camera>();

            if (status == PermissionStatus.Granted)
            {
                StatusLabel.Text = "Camera ready";
            }
            else
            {
                StatusLabel.Text = "Camera permission denied";
                LoadingLabel.Text = "Camera permission denied";
                LoadingSpinner.IsRunning = false;
                return;
            }

            var dbPath = Path.Combine(FileSystem.AppDataDirectory, "faceattendance.db3");
            _database = new WorkerDatabase(dbPath);
            Debug.WriteLine($"[RegisterWorkerPage] Database initialized at {dbPath}");

            // Only load models the FIRST time — since services are now registered as
            // singletons in MauiProgram.cs, IsLoaded stays true across page visits, so this
            // skips reloading from disk (and re-initializing the ONNX session) on subsequent
            // visits to this page.
            if (!_detectionService.IsLoaded || !_recognitionService.IsLoaded || !_ppeService.IsLoaded)
            {
                LoadingLabel.Text = "Loading face recognition models...";

                // Anti-spoof model is NOT loaded here — registration is a static, in-person
                // enrollment step, so liveness checking is not required. PPE model IS loaded,
                // though, so variant captures (e.g. "With Mask") can be verified against what's
                // actually visible in the photo.
                var detectionModelPath = await EnsureModelFileAsync("face_detection_yunet_2023mar.onnx");
                var recognitionModelPath = await EnsureModelFileAsync("face_recognition_sface_2021dec.onnx");
                var ppeModelPath = await EnsureModelFileAsync("ppe_detection.onnx");

                if (!_detectionService.IsLoaded)
                    await Task.Run(() => _detectionService.LoadModel(detectionModelPath));

                if (!_recognitionService.IsLoaded)
                    await Task.Run(() => _recognitionService.LoadModel(recognitionModelPath));

                if (!_ppeService.IsLoaded)
                    await Task.Run(() => _ppeService.LoadModel(ppeModelPath));
            }

            LoadingLabel.Text = "Starting camera...";

            try
            {
                var cameras = await MainCamera.GetAvailableCameras(CancellationToken.None);

                _frontCamera = cameras.FirstOrDefault(c => c.Position == CameraPosition.Front);
                _backCamera = cameras.FirstOrDefault(c => c.Position == CameraPosition.Rear);

                MainCamera.SelectedCamera = _backCamera ?? cameras.FirstOrDefault();

                Debug.WriteLine($"[RegisterWorkerPage] Cameras found — front: {_frontCamera != null}, back: {_backCamera != null}");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[RegisterWorkerPage] Camera enumeration FAILED: {ex}");
                LoadingLabel.Text = "Camera setup failed";
                LoadingSpinner.IsRunning = false;
                return;
            }

            LoadingOverlay.IsVisible = false;
        }

        /// <summary>
        /// Copies a bundled model file from the app package into app data storage (if not
        /// already there), and returns the plain file path. This is MAUI-specific logic, so it
        /// lives here in the UI project — Core's LoadModel(string) only ever receives the
        /// finished path string and knows nothing about FileSystem/app packaging.
        /// </summary>
        private static async Task<string> EnsureModelFileAsync(string fileName)
        {
            var modelPath = Path.Combine(FileSystem.AppDataDirectory, fileName);

            if (!File.Exists(modelPath))
            {
                using var assetStream = await FileSystem.OpenAppPackageFileAsync(fileName);
                using var fileStream = File.Create(modelPath);
                await assetStream.CopyToAsync(fileStream);
            }

            return modelPath;
        }

        private void OnSwitchCameraClicked(object? sender, EventArgs e)
        {
            _usingFrontCamera = !_usingFrontCamera;

            var target = _usingFrontCamera ? _frontCamera : _backCamera;

            if (target != null)
            {
                MainCamera.SelectedCamera = target;
                Debug.WriteLine($"[RegisterWorkerPage] Switched to {(_usingFrontCamera ? "front" : "back")} camera");
            }
            else
            {
                Debug.WriteLine("[RegisterWorkerPage] Switch failed — target camera not available");
            }
        }

        /// <summary>
        /// Averages a set of already-normalized embeddings and re-normalizes the result.
        /// Averaging cancels out per-shot noise (blink, slight head angle, momentary focus
        /// hunt) that would otherwise lower match scores even for a correct match — this is
        /// the same technique used for occluded-reference matching, just applied at enrollment
        /// time instead of match time.
        /// </summary>
        private static float[] AverageEmbeddings(List<float[]> embeddings)
        {
            int dims = embeddings[0].Length;
            var avg = new float[dims];

            foreach (var e in embeddings)
            {
                for (int i = 0; i < dims; i++)
                {
                    avg[i] += e[i];
                }
            }

            float norm = 0f;
            for (int i = 0; i < dims; i++)
            {
                avg[i] /= embeddings.Count;
                norm += avg[i] * avg[i];
            }

            norm = MathF.Sqrt(norm);
            if (norm > 0f)
            {
                for (int i = 0; i < dims; i++)
                {
                    avg[i] /= norm;
                }
            }

            return avg;
        }

        private void ShowResultWithBoxes(SKBitmap original, List<(FaceDetection face, string label)> results, bool allowEnroll = true)
        {
            using var surface = SKSurface.Create(new SKImageInfo(original.Width, original.Height));
            var canvas = surface.Canvas;
            canvas.DrawBitmap(original, 0, 0);

            using var font = new SKFont
            {
                Size = Math.Max(24, original.Width / 30f)
            };

            foreach (var (face, label) in results)
            {
                bool isDuplicate = label.Contains("Already registered");

                using var boxPaint = new SKPaint
                {
                    Color = isDuplicate ? SKColors.Orange : SKColors.LimeGreen,
                    Style = SKPaintStyle.Stroke,
                    StrokeWidth = Math.Max(4, original.Width / 200f)
                };

                using var textPaint = new SKPaint
                {
                    Color = isDuplicate ? SKColors.Orange : SKColors.LimeGreen,
                    IsAntialias = true
                };

                canvas.DrawRect(
                    SKRect.Create(face.X1, face.Y1, face.X2 - face.X1, face.Y2 - face.Y1),
                    boxPaint);

                canvas.DrawText(label, face.X1, face.Y1 - 10, font, textPaint);
            }

            using var image = surface.Snapshot();
            using var data = image.Encode(SKEncodedImageFormat.Jpeg, 90);
            var bytes = data.ToArray();

            MainThread.BeginInvokeOnMainThread(() =>
            {
                ResultImage.Source = ImageSource.FromStream(() => new MemoryStream(bytes));
                ResultImage.IsVisible = true;
                MainCamera.IsVisible = false;
                RetakeBtn.IsVisible = true;
                SwitchCameraBtn.IsVisible = false;
                CaptureBtn.IsVisible = false;
                WorkerNameEntry.IsVisible = results.Count > 0 && allowEnroll;
                EnrollBtn.IsVisible = results.Count > 0 && allowEnroll;
            });
        }

        private void OnRetakeClicked(object? sender, EventArgs e)
        {
            ResultImage.IsVisible = false;
            MainCamera.IsVisible = true;
            RetakeBtn.IsVisible = false;
            SwitchCameraBtn.IsVisible = true;
            CaptureBtn.IsVisible = true;
            WorkerNameEntry.IsVisible = false;
            EnrollBtn.IsVisible = false;
            WorkerNameEntry.Text = string.Empty;
            _lastEmbedding = null;
            _primarySamples.Clear();
            StatusLabel.Text = "Camera ready";
        }

        private async void OnEnrollClicked(object? sender, EventArgs e)
        {
            if (_database == null || _lastEmbedding == null)
            {
                Debug.WriteLine("[RegisterWorkerPage] Enroll failed — no database or embedding available");
                return;
            }

            var name = WorkerNameEntry.Text?.Trim();

            if (string.IsNullOrWhiteSpace(name))
            {
                await DisplayAlert("Missing name", "Please enter the worker's name before enrolling.", "OK");
                return;
            }

            var worker = new Worker
            {
                Name = name,
                EmbeddingCsv = Worker.SerializeEmbedding(_lastEmbedding)
            };

            var id = await _database.SaveWorkerAsync(worker);

            Debug.WriteLine($"[RegisterWorkerPage] Worker enrolled — id={id}, name={name} (averaged from {_primarySamples.Count} samples)");

            _primarySamples.Clear();

            OnRetakeClicked(sender!, e);

            await OfferAnotherVariantAsync(id, name);
        }

        /// <summary>
        /// Offers to capture an additional face variant (mask, spectacles, any combo) for the
        /// just-enrolled worker, and loops — a worker can register as many looks as they
        /// actually need, not just one fixed "masked" slot like before.
        /// </summary>
        private async Task OfferAnotherVariantAsync(int workerId, string workerName)
        {
            bool wantsVariant = await DisplayAlert(
                "Enrolled",
                $"{workerName} has been enrolled successfully.\n\nWould you like to also register another look (e.g. with a mask or spectacles) to improve recognition accuracy?",
                "Yes, add a variant", "Done");

            if (!wantsVariant) return;

            var choice = await DisplayActionSheet("What will the worker be wearing?", "Cancel", null, SuggestedLabels);
            if (string.IsNullOrWhiteSpace(choice) || choice == "Cancel") return;

            string label = choice;
            if (choice == "Custom...")
            {
                label = await DisplayPromptAsync("Custom Label", "Describe this variant (e.g. 'With Hard Hat')");
                if (string.IsNullOrWhiteSpace(label)) return;
            }

            _pendingVariantLabel = label;
            _pendingVariantWorkerId = workerId;
            _pendingVariantWorkerName = workerName;
            _variantSamples.Clear();

            MaskedModeBanner.IsVisible = true;
            MaskedModeLabel.Text = $"Put on: {label} — capturing for {workerName}";
            StatusLabel.Text = $"Put on: {label}. Capture {SamplesRequired} shots (0/{SamplesRequired})";
        }

        private void RunInference(string imagePath)
        {
            if (!_detectionService.IsLoaded)
            {
                Debug.WriteLine("[RegisterWorkerPage] Inference skipped — detection model not loaded yet");
                return;
            }

            try
            {
                var stopwatch = Stopwatch.StartNew();

                using var originalForSize = DecodeWithOrientation(imagePath);

                var faces = _detectionService.Detect(originalForSize);

                stopwatch.Stop();

                Debug.WriteLine($"[RegisterWorkerPage] Detection completed in {stopwatch.ElapsedMilliseconds} ms");
                Debug.WriteLine($"[RegisterWorkerPage] Faces detected: {faces.Count}");

                if (_pendingVariantLabel != null && _pendingVariantWorkerId.HasValue)
                {
                    RunVariantCapture(originalForSize, faces);
                    return;
                }

                RunPrimaryCapture(originalForSize, faces);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[RegisterWorkerPage] Inference FAILED: {ex}");
            }
        }

        /// <summary>
        /// Primary (bare-face) enrollment path with multi-sample averaging. Duplicate checking
        /// only happens on the FIRST sample of a session — if the first shot doesn't match any
        /// existing worker, we trust that identity for the remaining samples in this session
        /// (re-checking every sample would be redundant and could occasionally flip on a noisy
        /// frame). Once SamplesRequired good samples are collected, they're averaged into
        /// _lastEmbedding and the Enroll button becomes available.
        /// </summary>
        private void RunPrimaryCapture(SKBitmap originalForSize, List<FaceDetection> faces)
        {
            if (faces.Count == 0)
            {
                Debug.WriteLine("[RegisterWorkerPage] No face detected — nothing to enroll");
                MainThread.BeginInvokeOnMainThread(() =>
                {
                    StatusLabel.Text = "No face detected — try again";
                });
                return;
            }

            if (faces.Count > 1)
            {
                Debug.WriteLine("[RegisterWorkerPage] Multiple faces detected — using the first for enrollment");
            }

            var face = faces[0];

            var alignStopwatch = Stopwatch.StartNew();
            using var aligned = _recognitionService.AlignFace(originalForSize, face.Landmarks);
            var embedding = _recognitionService.GetEmbedding(aligned);
            alignStopwatch.Stop();

            if (embedding == null)
            {
                Debug.WriteLine("[RegisterWorkerPage] Embedding failed — try again");
                MainThread.BeginInvokeOnMainThread(() =>
                {
                    StatusLabel.Text = "Capture failed — try again";
                });
                return;
            }

            Debug.WriteLine($"[RegisterWorkerPage] Embedding computed in {alignStopwatch.ElapsedMilliseconds} ms");

            // Duplicate check — only on the very first sample of this session.
            if (_primarySamples.Count == 0)
            {
                var workers = _database?.GetAllWorkersAsync().Result ?? new List<Worker>();
                Debug.WriteLine($"[RegisterWorkerPage] Checking against {workers.Count} enrolled workers for duplicates");

                var (matchedWorker, score) = _recognitionService.FindBestMatch(workers, embedding);

                if (matchedWorker != null)
                {
                    Debug.WriteLine($"[RegisterWorkerPage]   DUPLICATE — matches existing worker {matchedWorker.Name}, score={score:F3}");

                    var label = $"Already registered: {matchedWorker.Name} ({score:P0})";
                    ShowResultWithBoxes(originalForSize, new List<(FaceDetection, string)> { (face, label) }, allowEnroll: false);

                    var duplicateName = matchedWorker.Name;
                    MainThread.BeginInvokeOnMainThread(async () =>
                    {
                        await DisplayAlert("Already Registered", $"This face is already enrolled as {duplicateName}.", "OK");
                    });
                    return;
                }
            }

            _primarySamples.Add(embedding);
            Debug.WriteLine($"[RegisterWorkerPage] Primary sample {_primarySamples.Count}/{SamplesRequired} captured");

            if (_primarySamples.Count < SamplesRequired)
            {
                // Not enough samples yet — stay on the live camera and ask for another shot,
                // rather than dropping into the Retake/Enroll result screen prematurely.
                int remaining = SamplesRequired - _primarySamples.Count;
                MainThread.BeginInvokeOnMainThread(() =>
                {
                    StatusLabel.Text = $"Captured {_primarySamples.Count}/{SamplesRequired} — capture {remaining} more (keep facing camera)";
                });
                return;
            }

            // Got enough samples — average them and show the final result with Enroll enabled.
            _lastEmbedding = AverageEmbeddings(_primarySamples);
            Debug.WriteLine($"[RegisterWorkerPage] Averaged {_primarySamples.Count} samples into final embedding");

            var finalLabel = $"Ready to enroll ({_primarySamples.Count} samples averaged)";
            ShowResultWithBoxes(originalForSize, new List<(FaceDetection, string)> { (face, finalLabel) }, allowEnroll: true);
        }

        /// <summary>
        /// Checks that the captured face actually shows the PPE implied by a variant label, so
        /// "With Mask" can't be saved from a bare face by mistake. See the identical method in
        /// WorkerEditPage.xaml.cs for the full explanation — kept in sync with that one.
        /// </summary>
        private (bool ok, string message) VerifyVariantAgainstLabel(SKBitmap original, FaceDetection face, string label)
        {
            if (!_ppeService.IsLoaded)
            {
                return (true, string.Empty);
            }

            var lowerLabel = label.ToLowerInvariant();
            bool labelWantsMask = lowerLabel.Contains("mask");
            bool labelWantsSpectacles = lowerLabel.Contains("spectacle") || lowerLabel.Contains("glasses");

            if (!labelWantsMask && !labelWantsSpectacles)
            {
                return (true, string.Empty);
            }

            var (hasHelmet, hasMask) = _ppeService.CheckHelmetAndMask(original, face.X1, face.Y1, face.X2, face.Y2);
            var (hasGoggles, hasEyeglasses) = _ppeService.CheckGogglesAndEyeglasses(original, face.X1, face.Y1, face.X2, face.Y2);

            if (labelWantsMask && !hasMask)
            {
                return (false, $"'{label}' expects a mask, but no mask was detected. Please put on a mask and try again.");
            }

            if (labelWantsSpectacles && !hasEyeglasses && !hasGoggles)
            {
                return (false, $"'{label}' expects spectacles, but none were detected. Please put on spectacles and try again.");
            }

            return (true, string.Empty);
        }

        /// <summary>
        /// Generic path for capturing an extra face-variant embedding (mask, spectacles, any
        /// combo) for an already-enrolled worker, now with the same multi-sample averaging as
        /// the primary path. Each sample is independently verified against the PPE rules for
        /// the label before being accepted — a bad shot (PPE not visible) doesn't count toward
        /// the total and doesn't get silently averaged in.
        /// </summary>
        private void RunVariantCapture(SKBitmap originalForSize, List<FaceDetection> faces)
        {
            if (faces.Count == 0)
            {
                Debug.WriteLine("[RegisterWorkerPage] Variant capture — no face detected, retry");
                MainThread.BeginInvokeOnMainThread(() =>
                {
                    StatusLabel.Text = "No face detected — try again";
                });
                return;
            }

            var face = faces[0];

            if (_pendingVariantLabel != null)
            {
                var (ok, message) = VerifyVariantAgainstLabel(originalForSize, face, _pendingVariantLabel);
                if (!ok)
                {
                    Debug.WriteLine($"[RegisterWorkerPage] Variant verification FAILED: {message}");
                    MainThread.BeginInvokeOnMainThread(async () =>
                    {
                        StatusLabel.Text = "Verification failed — try again";
                        await DisplayAlert("PPE Not Detected", message, "OK");
                    });
                    return;
                }
            }

            using var aligned = _recognitionService.AlignFace(originalForSize, face.Landmarks);
            var embedding = _recognitionService.GetEmbedding(aligned);

            if (embedding == null)
            {
                Debug.WriteLine("[RegisterWorkerPage] Variant capture — embedding failed, retry");
                MainThread.BeginInvokeOnMainThread(() =>
                {
                    StatusLabel.Text = "Capture failed — try again";
                });
                return;
            }

            _variantSamples.Add(embedding);
            Debug.WriteLine($"[RegisterWorkerPage] Variant sample {_variantSamples.Count}/{SamplesRequired} captured for label '{_pendingVariantLabel}'");

            if (_variantSamples.Count < SamplesRequired)
            {
                int remaining = SamplesRequired - _variantSamples.Count;
                MainThread.BeginInvokeOnMainThread(() =>
                {
                    StatusLabel.Text = $"Captured {_variantSamples.Count}/{SamplesRequired} — capture {remaining} more (keep the {_pendingVariantLabel} look on)";
                });
                return;
            }

            var averagedVariantEmbedding = AverageEmbeddings(_variantSamples);
            Debug.WriteLine($"[RegisterWorkerPage] Averaged {_variantSamples.Count} variant samples for label '{_pendingVariantLabel}'");

            MainThread.BeginInvokeOnMainThread(async () =>
            {
                if (_database == null || !_pendingVariantWorkerId.HasValue || _pendingVariantLabel == null)
                {
                    Debug.WriteLine("[RegisterWorkerPage] Variant capture — missing pending state");
                    _pendingVariantLabel = null;
                    _pendingVariantWorkerId = null;
                    _variantSamples.Clear();
                    MaskedModeBanner.IsVisible = false;
                    return;
                }

                var workerId = _pendingVariantWorkerId.Value;
                var label = _pendingVariantLabel;
                var workerName = _pendingVariantWorkerName;

                await _database.AddFaceReferenceAsync(workerId, label, averagedVariantEmbedding);

                Debug.WriteLine($"[RegisterWorkerPage] Face variant '{label}' saved for {workerName} (id={workerId}), averaged from {_variantSamples.Count} samples");

                _pendingVariantLabel = null;
                _pendingVariantWorkerId = null;
                _variantSamples.Clear();
                MaskedModeBanner.IsVisible = false;
                StatusLabel.Text = "Camera ready";

                // Loop — ask if they want to add yet another variant, so a worker can register
                // as many looks as needed in one enrollment session.
                await OfferAnotherVariantAsync(workerId, workerName);
            });
        }

        private async void OnCaptureClicked(object? sender, EventArgs e)
        {
            try
            {
                Debug.WriteLine("[RegisterWorkerPage] Capture button clicked");
                StatusLabel.Text = "Capturing...";

                await MainCamera.CaptureImage(CancellationToken.None);

                Debug.WriteLine("[RegisterWorkerPage] CaptureImage call completed without exception");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[RegisterWorkerPage] Capture EXCEPTION: {ex}");
                StatusLabel.Text = $"Capture failed: {ex.Message}";
            }
        }

        private async void MainCamera_MediaCaptured(object sender, MediaCapturedEventArgs e)
        {
            Debug.WriteLine("[RegisterWorkerPage] MediaCaptured event FIRED");

            try
            {
                var fileName = $"capture_{DateTime.Now:yyyyMMdd_HHmmss}.jpg";
                var filePath = Path.Combine(FileSystem.CacheDirectory, fileName);

                using (var fileStream = File.Create(filePath))
                {
                    await e.Media.CopyToAsync(fileStream);
                }

                Debug.WriteLine($"[RegisterWorkerPage] Saved successfully to {filePath}");

                await Task.Run(() => RunInference(filePath));
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[RegisterWorkerPage] Save EXCEPTION: {ex}");

                MainThread.BeginInvokeOnMainThread(() =>
                {
                    StatusLabel.Text = $"Save failed: {ex.Message}";
                });
            }
        }

        private SKBitmap DecodeWithOrientation(string path)
        {
            SKEncodedOrigin orientation;
            using (var stream = File.OpenRead(path))
            using (var codec = SKCodec.Create(stream))
            {
                orientation = codec.EncodedOrigin;
            }

            using var decodeStream = File.OpenRead(path);
            var bitmap = SKBitmap.Decode(decodeStream);

            if (orientation == SKEncodedOrigin.Default)
                return bitmap;

            bool swapDims = orientation == SKEncodedOrigin.RightTop || orientation == SKEncodedOrigin.LeftBottom;
            var target = new SKBitmap(swapDims ? bitmap.Height : bitmap.Width, swapDims ? bitmap.Width : bitmap.Height);
            using var canvas = new SKCanvas(target);

            switch (orientation)
            {
                case SKEncodedOrigin.RightTop:
                    canvas.Translate(target.Width, 0);
                    canvas.RotateDegrees(90);
                    break;
                case SKEncodedOrigin.BottomRight:
                    canvas.RotateDegrees(180, bitmap.Width / 2f, bitmap.Height / 2f);
                    break;
                case SKEncodedOrigin.LeftBottom:
                    canvas.Translate(0, target.Height);
                    canvas.RotateDegrees(-90);
                    break;
            }
            canvas.DrawBitmap(bitmap, 0, 0);
            bitmap.Dispose();
            return target;
        }
    }
}