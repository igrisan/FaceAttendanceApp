using CommunityToolkit.Maui.Core;
using System.Diagnostics;
using SkiaSharp;
using FaceAttendanceApp.Services;
using FaceAttendanceApp.Model;

namespace FaceAttendanceApp
{
    public partial class ScanAttendancePage : ContentPage
    {
        private readonly FaceDetectionService _detectionService = new();
        private readonly FaceRecognitionService _recognitionService = new();
        private readonly LivenessService _livenessService = new();
        private readonly MaskHelmetDetectionService _ppeService = new();

        private WorkerDatabase? _database;
        private AttendanceDatabase? _attendanceDatabase;

        private List<WorkerEmbeddingCandidate> _cachedCandidates = new();

        private static readonly TimeSpan DuplicateSuppressWindow = TimeSpan.FromMinutes(2);

        // Target capture resolution now square-ish so the sensor buffer itself is close to 1:1
        // instead of the default 4:3/16:9 — combined with CenterCropToSquare below, this keeps
        // the actual crop small (or a no-op if the camera can supply a native square-ish mode).
        // Lowered from 1200x1200 — decode+crop time scales with pixel count (measured ~130ms
        // combined at 1088x1088 in the logs) and a kiosk face doesn't need more than ~720px
        // across for reliable detection/recognition at normal standing distance.
        private static readonly Microsoft.Maui.Graphics.Size TargetCaptureResolution = new(720, 720);

        // Detection runs on a downscaled copy of the (already-cropped) square frame using the
        // Detect(bitmap, fullWidth, fullHeight) overload already implemented in
        // FaceDetectionService — it internally maps boxes/landmarks back to full-res coordinates,
        // so nothing downstream (liveness, alignment, PPE) needs to change.
        private const int DetectionInputSize = 480;

        // Reserved space above/below the square preview box, in device-independent points.
        // Mirrors the pill (top) and result-card/stop-button zone (bottom) so the square preview
        // never overlaps them. Tune these if you resize those elements in XAML.
        private const double TopReservedSpace = 60;
        private const double BottomReservedSpace = 140;

        // Detection crop zone: fraction of the SQUARE captured frame's height to discard from the
        // top and bottom before running any inference. Kept small now since CenterCropToSquare
        // already removes most of the excess — this is just fine-tuning for the kiosk's mounting
        // height/distance if faces near the edges still get clipped.
        private const float CropTopFraction = 0.0f;
        private const float CropBottomFraction = 0.0f;

        private CameraInfo? _frontCamera;
        private CameraInfo? _backCamera;
        private bool _usingFrontCamera = false;

        private bool _cameraReady = false;

        private bool _isScanning = false;
        private bool _isBusy = false;

        private static readonly TimeSpan ResultDisplayDelay = TimeSpan.FromSeconds(5);

        // Controls the independent overlay-hide timer. Cancelled/replaced every time a new
        // overlay is shown, so the capture loop is never gated on this delay anymore — capture
        // and display are decoupled.
        private CancellationTokenSource? _overlayHideCts;

        // ---- alignment + mirror state ----
        // The size of the (now square) preview surface currently on screen (in device-independent
        // points), captured every time it changes, so we can correctly map captured-image pixel
        // coordinates back onto what the user is actually looking at.
        private double _previewWidth;
        private double _previewHeight;

        public ScanAttendancePage()
        {
            InitializeComponent();

            // Recompute the square preview box whenever the page/root grid resizes (rotation,
            // window resize, etc.), and keep _previewWidth/_previewHeight in sync with it.
            RootGrid.SizeChanged += (_, _) => ApplySquarePreviewBounds();

            MainCamera.SizeChanged += (_, _) =>
            {
                _previewWidth = MainCamera.Width;
                _previewHeight = MainCamera.Height;
            };
        }

        /// <summary>
        /// Sizes MainCamera + OverlayImage to a centered square box that fits within the space
        /// left after the top pill and bottom result-card/button zone. CameraView has no
        /// Aspect/PreviewAspect property — it simply fills whatever WidthRequest/HeightRequest
        /// box it's given — so making that box square is what actually fixes the stretched
        /// "3:4-looking" preview instead of a true 1:1.
        /// </summary>
        private void ApplySquarePreviewBounds()
        {
            double availableW = RootGrid.Width;
            double availableH = RootGrid.Height - TopReservedSpace - BottomReservedSpace;

            if (availableW <= 0 || availableH <= 0)
            {
                return;
            }

            double side = Math.Min(availableW, availableH);

            MainCamera.WidthRequest = side;
            MainCamera.HeightRequest = side;
            OverlayImage.WidthRequest = side;
            OverlayImage.HeightRequest = side;

            // Center vertically within the reserved band (not just the whole page), by nudging
            // with a Margin. VerticalOptions="Center" on its own centers within the WHOLE grid,
            // which would push the square down into the bottom reserved zone since that zone is
            // taller than the top one. This keeps it centered in the actual visible gap.
            double verticalOffset = (TopReservedSpace - BottomReservedSpace) / 2.0;
            MainCamera.Margin = new Thickness(0, verticalOffset, 0, 0);
            OverlayImage.Margin = new Thickness(0, verticalOffset, 0, 0);

            _previewWidth = side;
            _previewHeight = side;

            Debug.WriteLine($"[ScanAttendancePage] Square preview bounds applied: {side}x{side} (available {availableW}x{availableH})");
        }

        protected override async void OnAppearing()
        {
            base.OnAppearing();

            ApplySquarePreviewBounds();

            var status = await Permissions.RequestAsync<Permissions.Camera>();

            if (status != PermissionStatus.Granted)
            {
                ScanStateLabel.Text = "Camera permission denied";
                return;
            }

            var dbPath = Path.Combine(FileSystem.AppDataDirectory, "faceattendance.db3");
            _database = new WorkerDatabase(dbPath);

            var attendanceDbPath = Path.Combine(FileSystem.AppDataDirectory, "attendance.db3");
            _attendanceDatabase = new AttendanceDatabase(attendanceDbPath);

            Debug.WriteLine($"[ScanAttendancePage] Databases initialized — workers: {dbPath}, attendance: {attendanceDbPath}");

            ScanStateLabel.Text = "Loading models...";

            await Task.Run(_detectionService.LoadModel);
            await Task.Run(_recognitionService.LoadModel);
            await Task.Run(_livenessService.LoadModel);
            await Task.Run(_ppeService.LoadModel);

            await RefreshWorkerCacheAsync();

            try
            {
                var cameras = await MainCamera.GetAvailableCameras(CancellationToken.None);
                _frontCamera = cameras.FirstOrDefault(c => c.Position == CameraPosition.Front);
                _backCamera = cameras.FirstOrDefault(c => c.Position == CameraPosition.Rear);
                MainCamera.SelectedCamera = _backCamera ?? cameras.FirstOrDefault();

                ApplyCaptureResolution();

                Debug.WriteLine($"[ScanAttendancePage] Cameras found — front: {_frontCamera != null}, back: {_backCamera != null}");

                _cameraReady = true;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[ScanAttendancePage] Camera enumeration FAILED: {ex}");
            }

            ApplySquarePreviewBounds();

            ScanStateLabel.Text = "Scanning...";
            _isScanning = true;
            StartScanPulse();

            await CaptureNextFrame();
        }

        private void ApplyCaptureResolution()
        {
            var supported = MainCamera.SelectedCamera?.SupportedResolutions;

            if (supported == null || supported.Count == 0)
            {
                Debug.WriteLine("[ScanAttendancePage] ApplyCaptureResolution skipped — no SupportedResolutions reported, leaving sensor default");
                return;
            }

            var best = supported
                .OrderBy(size => Math.Abs(size.Width - TargetCaptureResolution.Width)
                               + Math.Abs(size.Height - TargetCaptureResolution.Height))
                .First();

            MainCamera.ImageCaptureResolution = best;
            Debug.WriteLine($"[ScanAttendancePage] Capture resolution set to {best.Width}x{best.Height} (target was {TargetCaptureResolution.Width}x{TargetCaptureResolution.Height})");
        }

        private async void StartScanPulse()
        {
            while (_isScanning)
            {
                await ScanPulseDot.FadeTo(0.3, 500);
                if (!_isScanning) break;
                await ScanPulseDot.FadeTo(1.0, 500);
            }
        }

        private async Task RefreshWorkerCacheAsync()
        {
            if (_database == null) return;

            var stopwatch = Stopwatch.StartNew();
            _cachedCandidates = await _database.GetAllMatchCandidatesAsync();
            stopwatch.Stop();

            var workerCount = _cachedCandidates.Select(c => c.Worker.Id).Distinct().Count();
            Debug.WriteLine($"[ScanAttendancePage] Worker cache refreshed — {workerCount} workers, {_cachedCandidates.Count} total face candidates loaded in {stopwatch.ElapsedMilliseconds} ms");
        }

        private async void OnRefreshWorkersClicked(object? sender, EventArgs e)
        {
            await RefreshWorkerCacheAsync();
        }

        private async void OnSwitchCameraClicked(object? sender, EventArgs e)
        {
            if (!_cameraReady)
            {
                Debug.WriteLine("[ScanAttendancePage] Switch ignored — camera not ready yet");
                return;
            }

            _cameraReady = false;

            _usingFrontCamera = !_usingFrontCamera;

            var target = _usingFrontCamera ? _frontCamera : _backCamera;

            if (target != null)
            {
                MainCamera.SelectedCamera = target;
                ApplyCaptureResolution();
                Debug.WriteLine($"[ScanAttendancePage] Switched to {(_usingFrontCamera ? "front" : "back")} camera");
            }
            else
            {
                Debug.WriteLine("[ScanAttendancePage] Switch failed — target camera not available");
            }

            await Task.Delay(800);
            _cameraReady = true;
        }

        protected override void OnDisappearing()
        {
            base.OnDisappearing();
            _isScanning = false;
            _cameraReady = false;
            _overlayHideCts?.Cancel();
        }

        private async void OnStopScanClicked(object? sender, EventArgs e)
        {
            _isScanning = false;
            _overlayHideCts?.Cancel();
            ScanStateLabel.Text = "Scanning stopped";
            await Shell.Current.GoToAsync("..");
        }

        private async Task CaptureNextFrame()
        {
            if (!_isScanning || _isBusy)
            {
                return;
            }

            try
            {
                _isBusy = true;
                await MainCamera.CaptureImage(CancellationToken.None);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[ScanAttendancePage] Capture EXCEPTION: {ex}");
                _isBusy = false;

                if (_isScanning)
                {
                    await Task.Delay(500);
                    await CaptureNextFrame();
                }
            }
        }

        private async void MainCamera_MediaCaptured(object sender, MediaCapturedEventArgs e)
        {
            try
            {
                using var memoryStream = new MemoryStream();
                await e.Media.CopyToAsync(memoryStream);
                memoryStream.Position = 0;

                await Task.Run(async () => await RunInferenceAsync(memoryStream));
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[ScanAttendancePage] MediaCaptured EXCEPTION: {ex}");
            }

            _isBusy = false;

            if (_isScanning)
            {
                // No artificial delay here — next capture starts immediately after detection
                // + overlay is drawn (see RunInferenceAsync: the box now goes up BEFORE the
                // slow recognition/liveness/PPE work runs), so the perceived loop is much tighter.
                await CaptureNextFrame();
            }
        }

        private async Task RunInferenceAsync(Stream imageStream)
        {
            if (!_detectionService.IsLoaded)
            {
                Debug.WriteLine("[ScanAttendancePage] Inference skipped — detection model not loaded yet");
                return;
            }

            var frameSw = Stopwatch.StartNew();
            var stageSw = new Stopwatch();

            try
            {
                stageSw.Restart();
                using var decoded = DecodeWithOrientation(imageStream);
                stageSw.Stop();
                Debug.WriteLine($"[Timing] Decode + orientation fix: {stageSw.ElapsedMilliseconds} ms — {decoded.Width}x{decoded.Height}");

                // Center-crop the raw captured frame to a square BEFORE the detection-zone crop,
                // so what the model sees (and what the overlay is drawn against) matches the
                // square preview box the user is looking at, instead of the sensor's native
                // 4:3/16:9 aspect ratio.
                stageSw.Restart();
                using var squared = CenterCropToSquare(decoded);
                stageSw.Stop();
                Debug.WriteLine($"[Timing] Center crop to square: {stageSw.ElapsedMilliseconds} ms — {squared.Width}x{squared.Height} (from {decoded.Width}x{decoded.Height})");

                stageSw.Restart();
                using var originalForSize = CropToDetectionZone(squared);
                stageSw.Stop();
                Debug.WriteLine($"[Timing] Crop to detection zone: {stageSw.ElapsedMilliseconds} ms — {originalForSize.Width}x{originalForSize.Height} (from {squared.Width}x{squared.Height})");

                stageSw.Restart();
                List<FaceDetection> faces;
                if (originalForSize.Width > DetectionInputSize)
                {
                    float detectScale = (float)DetectionInputSize / originalForSize.Width;
                    int detectHeight = (int)Math.Round(originalForSize.Height * detectScale);

                    using var detectFrame = new SKBitmap(DetectionInputSize, detectHeight);
                    using (var c = new SKCanvas(detectFrame))
                    {
                        c.DrawBitmap(originalForSize,
                            new SKRect(0, 0, originalForSize.Width, originalForSize.Height),
                            new SKRect(0, 0, DetectionInputSize, detectHeight));
                    }

                    faces = _detectionService.Detect(detectFrame, originalForSize.Width, originalForSize.Height);
                }
                else
                {
                    faces = _detectionService.Detect(originalForSize);
                }
                stageSw.Stop();
                Debug.WriteLine($"[Timing] Face detection: {stageSw.ElapsedMilliseconds} ms — faces found: {faces.Count}");

                if (faces.Count == 0)
                {
                    MainThread.BeginInvokeOnMainThread(() =>
                    {
                        _overlayHideCts?.Cancel();
                        OverlayImage.IsVisible = false;
                        UpdateResultCard(new List<(FaceDetection, string)>());
                    });
                    Debug.WriteLine($"[Timing] TOTAL frame time: {frameSw.ElapsedMilliseconds} ms (no faces)");
                    return;
                }

                // ---- FAST PATH: draw boxes immediately with a "..." placeholder label so the
                // user sees a box glued to their face right away, instead of waiting for the
                // full liveness + recognition + PPE pipeline to finish first. ----
                var placeholderResults = faces
                    .Select(f => (face: f, label: "Processing..."))
                    .ToList();
                ShowOverlay(originalForSize, placeholderResults, TargetCaptureResolution: (originalForSize.Width, originalForSize.Height));

                var candidates = _cachedCandidates;
                var displayResults = new List<(FaceDetection face, string label)>();

                foreach (var face in faces)
                {
                    stageSw.Restart();
                    var (isLive, liveConfidence) = _livenessService.CheckLiveness(originalForSize, face);
                    stageSw.Stop();
                    Debug.WriteLine($"[Timing] Liveness check: {stageSw.ElapsedMilliseconds} ms — live={isLive}, confidence={liveConfidence:F3}");

                    if (!isLive)
                    {
                        var spoofLabel = $"SPOOF ({liveConfidence:P0})";
                        Debug.WriteLine($"[ScanAttendancePage] SPOOF flagged — live confidence {liveConfidence:P3}");
                        displayResults.Add((face, spoofLabel));
                        continue;
                    }

                    stageSw.Restart();
                    using var aligned = _recognitionService.AlignFace(originalForSize, face.Landmarks);
                    var embedding = _recognitionService.GetEmbedding(aligned);
                    stageSw.Stop();
                    Debug.WriteLine($"[Timing] Face embedding (align + SFace run): {stageSw.ElapsedMilliseconds} ms");

                    Worker? matchedWorker = null;
                    string matchedLabel = string.Empty;
                    float score = 0f;

                    if (embedding != null)
                    {
                        stageSw.Restart();
                        (matchedWorker, matchedLabel, score) = _recognitionService.FindBestMatch(candidates, embedding);
                        stageSw.Stop();
                        Debug.WriteLine($"[Timing] Worker matching ({candidates.Count} candidates): {stageSw.ElapsedMilliseconds} ms");
                    }

                    string label = matchedWorker != null
                        ? $"{matchedWorker.Name} ({score:P0})"
                        : $"Unknown ({score:P0})";

                    bool hasHelmet = false;
                    bool hasMask = false;

                    if (matchedWorker != null)
                    {
                        bool isDuplicate = await IsDuplicateAttendanceAsync(matchedWorker.Id);

                        if (!isDuplicate)
                        {
                            if (_ppeService.IsLoaded)
                            {
                                stageSw.Restart();
                                (hasHelmet, hasMask) = _ppeService.CheckHelmetAndMask(
                                    originalForSize, face.X1, face.Y1, face.X2, face.Y2);
                                stageSw.Stop();
                                Debug.WriteLine($"[Timing] PPE check: {stageSw.ElapsedMilliseconds} ms — helmet={hasHelmet}, mask={hasMask}");
                            }

                            Debug.WriteLine($"[ScanAttendancePage] MATCHED: {matchedWorker.Name} via '{matchedLabel}', score={score:F3}");

                            await SaveAttendanceAsync(matchedWorker, score, hasHelmet, hasMask);
                        }
                        else
                        {
                            Debug.WriteLine($"[ScanAttendancePage] MATCHED (duplicate, PPE skipped): {matchedWorker.Name} via '{matchedLabel}', score={score:F3}");
                        }
                    }

                    if (hasHelmet)
                    {
                        label += " [Helmet]";
                    }
                    if (hasMask)
                    {
                        label += " [Mask]";
                    }

                    displayResults.Add((face, label));
                }

                // ---- Final pass: replace the "Processing..." boxes with the real labels.
                // The box position itself doesn't move (same frame, same coordinates) — only
                // the text/color updates once recognition finishes. ----
                ShowOverlay(originalForSize, displayResults, TargetCaptureResolution: (originalForSize.Width, originalForSize.Height));

                frameSw.Stop();
                Debug.WriteLine($"[Timing] TOTAL frame time: {frameSw.ElapsedMilliseconds} ms — faces processed: {faces.Count}");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[ScanAttendancePage] Inference FAILED: {ex}");
            }
        }

        private async Task<bool> IsDuplicateAttendanceAsync(int workerId)
        {
            if (_attendanceDatabase == null) return false;

            var last = await _attendanceDatabase.GetLastRecordForWorkerAsync(workerId);

            if (last != null && DateTime.UtcNow - last.TimestampUtc < DuplicateSuppressWindow)
            {
                Debug.WriteLine($"[ScanAttendancePage] Duplicate within {DuplicateSuppressWindow.TotalMinutes:F0} min window (last: {last.TimestampUtc:u}) — skipping PPE + save");
                return true;
            }

            return false;
        }

        private async Task SaveAttendanceAsync(Worker worker, float score, bool hasHelmet, bool hasMask)
        {
            if (_attendanceDatabase == null) return;

            try
            {
                double? lat = null;
                double? lon = null;

                try
                {
                    var location = await Geolocation.Default.GetLastKnownLocationAsync()
                                    ?? await Geolocation.Default.GetLocationAsync(
                                           new GeolocationRequest(GeolocationAccuracy.Medium, TimeSpan.FromSeconds(5)));
                    if (location != null)
                    {
                        lat = location.Latitude;
                        lon = location.Longitude;
                    }
                }
                catch (Exception geoEx)
                {
                    Debug.WriteLine($"[ScanAttendancePage] Geolocation FAILED (continuing without it): {geoEx}");
                }

                var record = new AttendanceRecord
                {
                    WorkerId = worker.Id,
                    WorkerName = worker.Name,
                    TimestampUtc = DateTime.UtcNow,
                    MatchScore = score,
                    HasHelmet = hasHelmet,
                    HasMask = hasMask,
                    Latitude = lat,
                    Longitude = lon,
                    Synced = false
                };

                var id = await _attendanceDatabase.SaveRecordAsync(record);
                Debug.WriteLine($"[ScanAttendancePage] Attendance SAVED — id={id}, worker={worker.Name}, score={score:F3}, helmet={hasHelmet}, mask={hasMask}, lat={lat}, lon={lon}");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[ScanAttendancePage] Attendance save FAILED for {worker.Name}: {ex}");
            }
        }

        /// <summary>
        /// Draws the bounding boxes for the CURRENT captured frame's pixel space (original.Width x
        /// original.Height), which is exactly what OverlayImage will display via AspectFit. Because
        /// OverlayImage is now sized to the SAME square box as MainCamera, and both use AspectFit
        /// against the SAME (now square) source image dimensions, the boxes drawn here land in the
        /// correct relative position. We additionally mirror horizontally for the front camera,
        /// since the live preview is mirrored but the captured buffer typically is not.
        /// </summary>
        private void ShowOverlay(SKBitmap original, List<(FaceDetection face, string label)> results,
            (int Width, int Height) TargetCaptureResolution)
        {
            using var surface = SKSurface.Create(new SKImageInfo(original.Width, original.Height));
            var canvas = surface.Canvas;
            canvas.Clear(SKColors.Transparent);

            using var font = new SKFont
            {
                Size = Math.Max(24, original.Width / 30f)
            };

            bool mirror = _usingFrontCamera;

            foreach (var (face, label) in results)
            {
                bool isSpoof = label.Contains("SPOOF");
                bool isPlaceholder = label == "Processing...";

                var color = isSpoof ? SKColors.Red
                          : isPlaceholder ? SKColors.Gray
                          : SKColors.LimeGreen;

                using var boxPaint = new SKPaint
                {
                    Color = color,
                    Style = SKPaintStyle.Stroke,
                    StrokeWidth = Math.Max(4, original.Width / 200f)
                };

                using var textPaint = new SKPaint
                {
                    Color = color,
                    IsAntialias = true
                };

                // Mirror the box horizontally for the front camera so it matches the mirrored
                // preview the user is looking at (captured buffer is normally NOT pre-mirrored
                // even though the live preview is).
                float x1 = mirror ? original.Width - face.X2 : face.X1;
                float x2 = mirror ? original.Width - face.X1 : face.X2;

                canvas.DrawRect(
                    SKRect.Create(x1, face.Y1, x2 - x1, face.Y2 - face.Y1),
                    boxPaint);

                canvas.DrawText(label, x1, face.Y1 - 10, font, textPaint);
            }

            using var image = surface.Snapshot();
            using var data = image.Encode(SKEncodedImageFormat.Png, 100);
            var bytes = data.ToArray();

            MainThread.BeginInvokeOnMainThread(() =>
            {
                OverlayImage.Source = ImageSource.FromStream(() => new MemoryStream(bytes));
                OverlayImage.IsVisible = true;

                // Only update the info card with real (non-placeholder) results, so it doesn't
                // flash "Processing..." into the name/score labels.
                if (!results.Any(r => r.label == "Processing..."))
                {
                    UpdateResultCard(results);
                }
            });

            ScheduleOverlayHide();
        }

        private void ScheduleOverlayHide()
        {
            _overlayHideCts?.Cancel();
            var cts = new CancellationTokenSource();
            _overlayHideCts = cts;
            _ = HideOverlayAfterDelayAsync(cts.Token);
        }

        private async Task HideOverlayAfterDelayAsync(CancellationToken token)
        {
            try
            {
                await Task.Delay(ResultDisplayDelay, token);
            }
            catch (TaskCanceledException)
            {
                return;
            }

            if (token.IsCancellationRequested) return;

            MainThread.BeginInvokeOnMainThread(() =>
            {
                OverlayImage.IsVisible = false;
                if (_isScanning)
                {
                    ScanStateLabel.Text = "Scanning...";
                }
            });
        }

        private void UpdateResultCard(List<(FaceDetection face, string label)> results)
        {
            if (results.Count == 0)
            {
                ResultIcon.Text = "👤";
                ResultNameLabel.Text = "Waiting for face...";
                ResultScoreLabel.Text = "";
                HelmetChip.IsVisible = false;
                MaskChip.IsVisible = false;
                SpoofChip.IsVisible = false;
                return;
            }

            var spoofMatch = results.FirstOrDefault(r => r.label.Contains("SPOOF"));
            var namedMatch = results.FirstOrDefault(r => !r.label.StartsWith("Unknown") && !r.label.Contains("SPOOF"));

            (FaceDetection face, string label) chosen;
            if (spoofMatch.label != null)
                chosen = spoofMatch;
            else if (namedMatch.label != null)
                chosen = namedMatch;
            else
                chosen = results[0];

            bool isSpoof = chosen.label.Contains("SPOOF");
            bool isUnknown = chosen.label.StartsWith("Unknown");
            bool hasHelmet = chosen.label.Contains("[Helmet]");
            bool hasMask = chosen.label.Contains("[Mask]");

            var cleanLabel = chosen.label
                .Replace(" [Helmet]", "")
                .Replace(" [Mask]", "");

            SpoofChip.IsVisible = isSpoof;
            HelmetChip.IsVisible = hasHelmet;
            MaskChip.IsVisible = hasMask;

            if (isSpoof)
            {
                ResultIcon.Text = "⚠";
                ResultNameLabel.Text = "Spoof attempt blocked";
                ResultScoreLabel.Text = "";
            }
            else if (isUnknown)
            {
                ResultIcon.Text = "❓";
                ResultNameLabel.Text = "Unknown person";
                ResultScoreLabel.Text = ExtractScore(cleanLabel);
            }
            else
            {
                ResultIcon.Text = "✅";
                ResultNameLabel.Text = ExtractName(cleanLabel);
                ResultScoreLabel.Text = ExtractScore(cleanLabel);
                ResultTimestampLabel.Text = $"Logged at {DateTime.Now:HH:mm:ss}";
            }
        }

        private static string ExtractName(string label)
        {
            var idx = label.IndexOf(" (");
            return idx > 0 ? label[..idx] : label;
        }

        private static string ExtractScore(string label)
        {
            var start = label.IndexOf('(');
            var end = label.IndexOf(')');
            return start >= 0 && end > start ? label[start..(end + 1)] : "";
        }

        /// <summary>
        /// Center-crops the raw captured bitmap (native sensor aspect ratio, e.g. 4:3 or 16:9)
        /// down to a square, matching what the square preview box on screen actually shows.
        /// Without this, the model would run on the FULL non-square sensor frame while the user
        /// only sees a square window into it, causing detection coordinates to not line up with
        /// what the overlay draws, and wasting inference time on pixels the user can't see.
        /// </summary>
        private static SKBitmap CenterCropToSquare(SKBitmap source)
        {
            int side = Math.Min(source.Width, source.Height);

            if (side == source.Width && side == source.Height)
            {
                var same = new SKBitmap(source.Width, source.Height);
                using (var c = new SKCanvas(same))
                {
                    c.DrawBitmap(source, 0, 0);
                }
                return same;
            }

            int x = (source.Width - side) / 2;
            int y = (source.Height - side) / 2;

            var cropRect = new SKRectI(x, y, x + side, y + side);
            var cropped = new SKBitmap(side, side);

            using (var canvas = new SKCanvas(cropped))
            {
                canvas.DrawBitmap(source, cropRect, new SKRect(0, 0, side, side));
            }

            return cropped;
        }

        /// <summary>
        /// Crops out the top CropTopFraction and bottom CropBottomFraction of the (now square)
        /// frame before any inference runs, for fine-tuning only — CenterCropToSquare already
        /// does the heavy lifting. Both fractions default to 0 now that the frame is square; only
        /// increase them if faces near the top/bottom edge of the square still get clipped for
        /// your specific kiosk mounting height or standing distance.
        /// </summary>
        private static SKBitmap CropToDetectionZone(SKBitmap source)
        {
            int top = (int)(source.Height * CropTopFraction);
            int bottom = (int)(source.Height * CropBottomFraction);
            int cropHeight = source.Height - top - bottom;

            if (cropHeight <= 0 || cropHeight == source.Height)
            {
                // Degenerate config (fractions too large, or both zero) — fall back to full frame
                // rather than producing an invalid/empty bitmap.
                var fallback = new SKBitmap(source.Width, source.Height);
                using (var c = new SKCanvas(fallback))
                {
                    c.DrawBitmap(source, 0, 0);
                }
                return fallback;
            }

            var cropRect = new SKRectI(0, top, source.Width, top + cropHeight);
            var cropped = new SKBitmap(cropRect.Width, cropRect.Height);

            using (var canvas = new SKCanvas(cropped))
            {
                canvas.DrawBitmap(source, cropRect, new SKRect(0, 0, cropRect.Width, cropRect.Height));
            }

            return cropped;
        }

        private SKBitmap DecodeWithOrientation(Stream stream)
        {
            using var codec = SKCodec.Create(stream);
            var info = codec.Info;
            var bitmap = new SKBitmap(info.Width, info.Height);
            codec.GetPixels(bitmap.Info, bitmap.GetPixels());

            var orientation = codec.EncodedOrigin;
            if (orientation == SKEncodedOrigin.Default)
                return bitmap;

            bool swapDims = orientation == SKEncodedOrigin.RightTop || orientation == SKEncodedOrigin.LeftBottom;
            var target = new SKBitmap(swapDims ? info.Height : info.Width, swapDims ? info.Width : info.Height);
            using var canvas = new SKCanvas(target);

            switch (orientation)
            {
                case SKEncodedOrigin.RightTop:
                    canvas.Translate(target.Width, 0);
                    canvas.RotateDegrees(90);
                    break;
                case SKEncodedOrigin.BottomRight:
                    canvas.RotateDegrees(180, info.Width / 2f, info.Height / 2f);
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