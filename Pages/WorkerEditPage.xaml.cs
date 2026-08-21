using CommunityToolkit.Maui.Core;
using FaceAttendanceApp.Model;
using FaceAttendanceApp.Services;
using SkiaSharp;
using System.Diagnostics;

namespace FaceAttendanceApp
{
    [QueryProperty(nameof(WorkerId), "WorkerId")]
    public partial class WorkerEditPage : ContentPage
    {
        private readonly FaceDetectionService _detectionService;
        private readonly FaceRecognitionService _recognitionService;
        private readonly MaskHelmetDetectionService _ppeService;

        private WorkerDatabase? _database;
        private Worker? _worker;
        private List<WorkerFaceReference> _faceReferences = new();

        // "primary" recaptures the main Worker.EmbeddingCsv. Any other value is treated as the
        // label for a new/updated WorkerFaceReference row (e.g. "With Mask", "With Spectacles").
        private string _recaptureTarget = "primary";

        // How many good captures to collect before averaging into the final saved embedding.
        // Same reasoning as RegisterWorkerPage — cancels per-shot noise (blink, tiny angle
        // shift) that would otherwise lower match scores even for a correct recapture.
        private const int SamplesRequired = 3;

        // Accumulates embeddings for whichever recapture session is currently active (primary
        // or a specific variant label). Cleared on cancel, on successful save, and whenever a
        // new recapture session starts.
        private readonly List<float[]> _recaptureSamples = new();

        // Suggested labels shown in the picker — free-text "Custom..." is also offered so this
        // isn't a hard-coded enum and new combos don't need a code change later.
        private static readonly string[] SuggestedLabels =
        {
            "With Mask",
            "With Spectacles",
            "With Mask + Spectacles",
            "Custom..."
        };

        private CameraInfo? _frontCamera;
        private CameraInfo? _backCamera;
        private bool _usingFrontCamera = false;

        public string WorkerId { get; set; } = string.Empty;

        public WorkerEditPage(FaceDetectionService detectionService, FaceRecognitionService recognitionService, MaskHelmetDetectionService ppeService)
        {
            InitializeComponent();
            _detectionService = detectionService;
            _recognitionService = recognitionService;
            _ppeService = ppeService;
        }

        protected override async void OnAppearing()
        {
            base.OnAppearing();

            if (!int.TryParse(WorkerId, out var id))
            {
                Debug.WriteLine("[WorkerEditPage] Invalid WorkerId query parameter");
                return;
            }

            var dbPath = Path.Combine(FileSystem.AppDataDirectory, "faceattendance.db3");
            _database = new WorkerDatabase(dbPath);

            var workers = await _database.GetAllWorkersAsync();
            _worker = workers.FirstOrDefault(w => w.Id == id);

            if (_worker == null)
            {
                Debug.WriteLine($"[WorkerEditPage] Worker id={id} not found");
                await DisplayAlert("Not Found", "This worker could not be found.", "OK");
                await Shell.Current.GoToAsync("..");
                return;
            }

            Title = $"Edit: {_worker.Name}";
            NameEntry.Text = _worker.Name;

            await RefreshFaceReferencesAsync();
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

        private async Task RefreshFaceReferencesAsync()
        {
            if (_database == null || _worker == null) return;

            _faceReferences = await _database.GetFaceReferencesAsync(_worker.Id);
            FaceReferencesList.ItemsSource = _faceReferences;
            NoVariantsLabel.IsVisible = _faceReferences.Count == 0;
        }

        private async void OnSaveNameClicked(object? sender, EventArgs e)
        {
            if (_worker == null || _database == null) return;

            var newName = NameEntry.Text?.Trim();

            if (string.IsNullOrWhiteSpace(newName))
            {
                await DisplayAlert("Missing name", "Please enter a name.", "OK");
                return;
            }

            _worker.Name = newName;
            await _database.UpdateWorkerAsync(_worker);
            Title = $"Edit: {_worker.Name}";

            Debug.WriteLine($"[WorkerEditPage] Worker id={_worker.Id} renamed to '{newName}'");
            await DisplayAlert("Saved", "Worker name updated.", "OK");
        }

        /// <summary>
        /// Averages a set of already-normalized embeddings and re-normalizes the result.
        /// Identical technique to RegisterWorkerPage.AverageEmbeddings — kept as a private
        /// copy here rather than shared, since the two pages don't currently share a base
        /// class or helper module.
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

        private async void OnRecaptureFaceClicked(object? sender, EventArgs e)
        {
            _recaptureTarget = "primary";
            _recaptureSamples.Clear();
            RecaptureBannerLabel.Text = "Recapturing primary face reference";
            RecaptureStatusLabel.Text = $"Position face in frame (0/{SamplesRequired})";
            await StartRecaptureAsync();
        }

        /// <summary>
        /// Lets the person pick a suggested label (or type a custom one) for a new face
        /// variant, then starts the camera to capture it. This replaces the old single
        /// hard-coded "masked reference" button — any combination of PPE can be registered.
        /// </summary>
        private async void OnAddFaceVariantClicked(object? sender, EventArgs e)
        {
            var choice = await DisplayActionSheet("Add face variant — what will the worker be wearing?", "Cancel", null, SuggestedLabels);

            if (string.IsNullOrWhiteSpace(choice) || choice == "Cancel")
                return;

            string label = choice;

            if (choice == "Custom...")
            {
                label = await DisplayPromptAsync("Custom Label", "Describe this variant (e.g. 'With Hard Hat')");
                if (string.IsNullOrWhiteSpace(label))
                    return;
            }

            _recaptureTarget = label;
            _recaptureSamples.Clear();
            RecaptureBannerLabel.Text = $"Recapturing — {label}";
            RecaptureStatusLabel.Text = $"Put on: {label} — then position face in frame (0/{SamplesRequired})";
            await StartRecaptureAsync();
        }

        private async void OnDeleteFaceReferenceClicked(object? sender, EventArgs e)
        {
            if (_database == null) return;
            if (sender is not Button button || button.CommandParameter is not WorkerFaceReference reference) return;

            bool confirm = await DisplayAlert("Delete Variant", $"Delete the '{reference.Label}' face reference?", "Delete", "Cancel");
            if (!confirm) return;

            await _database.DeleteFaceReferenceAsync(reference);
            await RefreshFaceReferencesAsync();

            Debug.WriteLine($"[WorkerEditPage] Deleted face reference '{reference.Label}' (id={reference.Id})");
        }

        private async Task StartRecaptureAsync()
        {
            var status = await Permissions.RequestAsync<Permissions.Camera>();
            if (status != PermissionStatus.Granted)
            {
                await DisplayAlert("Camera permission denied", "Camera access is required to recapture.", "OK");
                return;
            }

            // Only load models the FIRST time — since services are registered as singletons in
            // MauiProgram.cs, IsLoaded stays true across page visits, so this skips reloading
            // from disk (and re-initializing each ONNX session) on subsequent visits.
            if (!_detectionService.IsLoaded || !_recognitionService.IsLoaded || !_ppeService.IsLoaded)
            {
                RecaptureStatusLabel.Text = "Loading models...";

                var detectionModelPath = await EnsureModelFileAsync("face_detection_yunet_2023mar.onnx");
                var recognitionModelPath = await EnsureModelFileAsync("face_recognition_sface_2021dec.onnx");
                var ppeModelPath = await EnsureModelFileAsync("ppe_detection.onnx");

                if (!_detectionService.IsLoaded)
                    await Task.Run(() => _detectionService.LoadModel(detectionModelPath));

                if (!_recognitionService.IsLoaded)
                    await Task.Run(() => _recognitionService.LoadModel(recognitionModelPath));

                if (!_ppeService.IsLoaded)
                    await Task.Run(() => _ppeService.LoadModel(ppeModelPath));

                RecaptureStatusLabel.Text = $"Position face in frame (0/{SamplesRequired})";
            }

            EditFormView.IsVisible = false;
            RecaptureView.IsVisible = true;

            try
            {
                var cameras = await RecaptureCamera.GetAvailableCameras(CancellationToken.None);
                _frontCamera = cameras.FirstOrDefault(c => c.Position == CameraPosition.Front);
                _backCamera = cameras.FirstOrDefault(c => c.Position == CameraPosition.Rear);
                _usingFrontCamera = false;
                RecaptureCamera.SelectedCamera = _backCamera ?? cameras.FirstOrDefault();
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[WorkerEditPage] Camera enumeration FAILED: {ex}");
            }
        }

        // Same switch-camera pattern as RegisterWorkerPage/ScanAttendancePage.
        private void OnSwitchCameraClicked(object? sender, EventArgs e)
        {
            _usingFrontCamera = !_usingFrontCamera;
            var target = _usingFrontCamera ? _frontCamera : _backCamera;

            if (target != null)
            {
                RecaptureCamera.SelectedCamera = target;
                Debug.WriteLine($"[WorkerEditPage] Switched to {(_usingFrontCamera ? "front" : "back")} camera");
            }
            else
            {
                Debug.WriteLine("[WorkerEditPage] Switch failed — target camera not available");
            }
        }

        private void OnRecaptureCancelClicked(object? sender, EventArgs e)
        {
            _recaptureSamples.Clear();
            RecaptureView.IsVisible = false;
            EditFormView.IsVisible = true;
        }

        private async void OnRecaptureCaptureClicked(object? sender, EventArgs e)
        {
            try
            {
                RecaptureStatusLabel.Text = "Capturing...";
                await RecaptureCamera.CaptureImage(CancellationToken.None);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[WorkerEditPage] Capture EXCEPTION: {ex}");
                RecaptureStatusLabel.Text = $"Capture failed: {ex.Message}";
            }
        }

        private async void RecaptureCamera_MediaCaptured(object sender, MediaCapturedEventArgs e)
        {
            try
            {
                var fileName = $"recapture_{DateTime.Now:yyyyMMdd_HHmmss}.jpg";
                var filePath = Path.Combine(FileSystem.CacheDirectory, fileName);

                using (var fileStream = File.Create(filePath))
                {
                    await e.Media.CopyToAsync(fileStream);
                }

                await Task.Run(() => ProcessRecapture(filePath));

                try { File.Delete(filePath); } catch { /* best-effort cleanup */ }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[WorkerEditPage] MediaCaptured EXCEPTION: {ex}");
            }
        }

        /// <summary>
        /// Checks that the captured face actually shows the PPE implied by a variant label,
        /// so "With Mask" can't be saved from a bare face by mistake. Matches keywords in the
        /// label against the PPE model's detections — "mask" requires a detected mask, and
        /// "spectacles"/"glasses" requires detected eyeglasses (goggles also accepted, since the
        /// model can't always tell them apart perfectly). Labels that don't mention either
        /// keyword (custom labels like "With Hard Hat") skip verification entirely, since we
        /// have no PPE class to check them against.
        /// </summary>
        private (bool ok, string message) VerifyVariantAgainstLabel(SKBitmap original, FaceDetection face, string label)
        {
            if (!_ppeService.IsLoaded)
            {
                // Fail open — don't block saving if the PPE model didn't load for some reason.
                return (true, string.Empty);
            }

            var lowerLabel = label.ToLowerInvariant();
            bool labelWantsMask = lowerLabel.Contains("mask");
            bool labelWantsSpectacles = lowerLabel.Contains("spectacle") || lowerLabel.Contains("glasses");

            if (!labelWantsMask && !labelWantsSpectacles)
            {
                // Custom label with no recognizable keyword — nothing to verify against.
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
        /// Now collects SamplesRequired good captures before saving, averaging them into the
        /// final embedding — same reasoning as RegisterWorkerPage.RunPrimaryCapture /
        /// RunVariantCapture. Each sample is independently checked (face detected, PPE verified
        /// for variants, embedding computed) before being counted; a failed sample doesn't
        /// consume one of the required slots.
        /// </summary>
        private void ProcessRecapture(string imagePath)
        {
            if (_worker == null || _database == null) return;

            try
            {
                using var original = DecodeWithOrientation(imagePath);
                var faces = _detectionService.Detect(original);

                if (faces.Count == 0)
                {
                    MainThread.BeginInvokeOnMainThread(() =>
                    {
                        RecaptureStatusLabel.Text = "No face detected — try again";
                    });
                    return;
                }

                var face = faces[0];

                // Only variant captures get PPE-verified — the primary/bare-face capture has
                // nothing to check against (there's no keyword implying PPE should be present).
                if (_recaptureTarget != "primary")
                {
                    var (ok, message) = VerifyVariantAgainstLabel(original, face, _recaptureTarget);
                    if (!ok)
                    {
                        Debug.WriteLine($"[WorkerEditPage] Variant verification FAILED: {message}");
                        MainThread.BeginInvokeOnMainThread(async () =>
                        {
                            RecaptureStatusLabel.Text = "Verification failed — try again";
                            await DisplayAlert("PPE Not Detected", message, "OK");
                        });
                        return;
                    }
                }

                using var aligned = _recognitionService.AlignFace(original, face.Landmarks);
                var embedding = _recognitionService.GetEmbedding(aligned);

                if (embedding == null)
                {
                    MainThread.BeginInvokeOnMainThread(() =>
                    {
                        RecaptureStatusLabel.Text = "Capture failed — try again";
                    });
                    return;
                }

                _recaptureSamples.Add(embedding);
                Debug.WriteLine($"[WorkerEditPage] Recapture sample {_recaptureSamples.Count}/{SamplesRequired} captured for target '{_recaptureTarget}'");

                if (_recaptureSamples.Count < SamplesRequired)
                {
                    int remaining = SamplesRequired - _recaptureSamples.Count;
                    MainThread.BeginInvokeOnMainThread(() =>
                    {
                        RecaptureStatusLabel.Text = $"Captured {_recaptureSamples.Count}/{SamplesRequired} — capture {remaining} more (hold the same pose)";
                    });
                    return;
                }

                var averagedEmbedding = AverageEmbeddings(_recaptureSamples);
                Debug.WriteLine($"[WorkerEditPage] Averaged {_recaptureSamples.Count} samples for target '{_recaptureTarget}'");

                MainThread.BeginInvokeOnMainThread(async () =>
                {
                    if (_recaptureTarget == "primary")
                    {
                        _worker.EmbeddingCsv = Worker.SerializeEmbedding(averagedEmbedding);
                        await _database.UpdateWorkerAsync(_worker);
                        Debug.WriteLine($"[WorkerEditPage] Primary reference updated for worker id={_worker.Id} (averaged from {_recaptureSamples.Count} samples)");
                    }
                    else
                    {
                        await _database.AddFaceReferenceAsync(_worker.Id, _recaptureTarget, averagedEmbedding);
                        Debug.WriteLine($"[WorkerEditPage] Face variant '{_recaptureTarget}' added for worker id={_worker.Id} (averaged from {_recaptureSamples.Count} samples)");
                        await RefreshFaceReferencesAsync();
                    }

                    var savedLabel = _recaptureTarget == "primary" ? "Primary face reference" : _recaptureTarget;
                    _recaptureSamples.Clear();

                    RecaptureView.IsVisible = false;
                    EditFormView.IsVisible = true;

                    await DisplayAlert("Saved", $"{savedLabel} updated for {_worker.Name}.", "OK");
                });
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[WorkerEditPage] ProcessRecapture FAILED: {ex}");
            }
        }

        private async void OnBackClicked(object? sender, EventArgs e)
        {
            await Shell.Current.GoToAsync("..");
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