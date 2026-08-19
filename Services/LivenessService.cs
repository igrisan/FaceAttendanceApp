using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using FaceAttendanceApp.Helpers;

namespace FaceAttendanceApp.Services
{
    public class LivenessService
    {
        /// <summary>
        /// Which output index represents "real/live" for this onnx export.
        /// Confirmed from the official source (minivision-ai/Silent-Face-Anti-Spoofing, test.py):
        /// "label = np.argmax(prediction); if label == 1: ... Real Face". Index 1 = real.
        /// </summary>
        private const int LiveClassIndex = 1;

        /// <summary>
        /// Set to true to re-enable the per-frame pixel-value logging and PNG crop dump used
        /// while debugging the original tensor-value mismatch. Left OFF by default for
        /// production: writing a PNG to disk on every single frame adds real per-frame overhead
        /// (encode + file I/O) for no benefit once the model's input format is confirmed working,
        /// which it now is (see LiveClassIndex comment above). Flip back to true only if you need
        /// to inspect what's actually being fed to the model again.
        /// </summary>
        private const bool EnableDiagnostics = false;

        private InferenceSession? _antiSpoofSession;

        public bool IsLoaded => _antiSpoofSession != null;

        public void LoadModel()
        {
            try
            {
                var modelPath = Path.Combine(FileSystem.AppDataDirectory, "2.7_80x80_MiniFASNetV2.onnx");

                using (var assetStream = FileSystem.OpenAppPackageFileAsync("2.7_80x80_MiniFASNetV2.onnx").Result)
                using (var fileStream = File.Create(modelPath))
                {
                    assetStream.CopyTo(fileStream);
                }

                _antiSpoofSession = new InferenceSession(modelPath);
                Debug.WriteLine("[LivenessService] Anti-spoof model loaded successfully");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[LivenessService] Anti-spoof model load FAILED: {ex}");
            }
        }

        /// <summary>
        /// Crops a region 2.7x the size of the detected face box, centered on the face,
        /// draws directly into an 80x80 destination, applies lighting normalization, then
        /// converts to BGR channel order, raw 0-255 range (matches the model's expected input —
        /// see the /255 vs raw testing note that led to LiveClassIndex being confirmed).
        /// </summary>
        private DenseTensor<float> PreprocessForAntiSpoof(SKBitmap original, FaceDetection face, float cropScale = 2.7f)
        {
            float boxW = face.X2 - face.X1;
            float boxH = face.Y2 - face.Y1;
            float centerX = face.X1 + boxW / 2f;
            float centerY = face.Y1 + boxH / 2f;

            float maxScaleX = original.Width / boxW;
            float maxScaleY = original.Height / boxH;
            float effectiveScale = Math.Min(cropScale, Math.Min(maxScaleX, maxScaleY));

            float newWidth = boxW * effectiveScale;
            float newHeight = boxH * effectiveScale;

            float left = centerX - newWidth / 2f;
            float top = centerY - newHeight / 2f;
            float right = centerX + newWidth / 2f;
            float bottom = centerY + newHeight / 2f;

            if (left < 0) { right -= left; left = 0; }
            if (top < 0) { bottom -= top; top = 0; }
            if (right > original.Width) { left -= (right - original.Width); right = original.Width; }
            if (bottom > original.Height) { top -= (bottom - original.Height); bottom = original.Height; }

            left = Math.Max(0, left);
            top = Math.Max(0, top);
            right = Math.Min(original.Width, right);
            bottom = Math.Min(original.Height, bottom);

            var cropRect = new SKRectI((int)left, (int)top, (int)right, (int)bottom);

            if (EnableDiagnostics)
            {
                Debug.WriteLine($"[LivenessService] Original size: {original.Width}x{original.Height}, face box: ({face.X1:F0},{face.Y1:F0})-({face.X2:F0},{face.Y2:F0}), cropRect: {cropRect}");
            }

            // Draw directly from the source crop region into an 80x80 destination in one step.
            // A separate crop + .Resize() call was producing corrupted/garbled output on some
            // crop sizes. Drawing straight into the final-size canvas with the canvas cleared
            // first (same pattern used in FaceDetectionService/FaceRecognitionService) avoids that.
            using var resizedRaw = new SKBitmap(80, 80);
            using (var canvas = new SKCanvas(resizedRaw))
            {
                canvas.Clear(SKColors.Black);
                var sampling = new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Linear);
                canvas.DrawBitmap(original, cropRect, new SKRect(0, 0, 80, 80), sampling);
            }

            // Lighting normalization: percentile-stretch contrast on the 80x80 crop before it
            // goes into the model. Helps outdoor/backlit/low-light captures where the raw crop
            // is either blown out or crushed in shadow — the anti-spoof model is sensitive to
            // exactly this kind of exposure variance (see the low-confidence 61.4% outlier case
            // this was added to address).
            using var resized = ImageQualityHelper.NormalizeLighting(resizedRaw);

            if (EnableDiagnostics)
            {
                Debug.WriteLine($"[LivenessService] resized.ColorType={resized.ColorType}, AlphaType={resized.AlphaType}");
                for (int y = 0; y < 80; y += 20)
                {
                    var p = resized.GetPixel(40, y);
                    Debug.WriteLine($"[LivenessService] pixel(40,{y}) = R={p.Red} G={p.Green} B={p.Blue}");
                }

                try
                {
                    var debugPath = Path.Combine(FileSystem.CacheDirectory, "antispoof_debug_crop.png");
                    using var debugData = resized.Encode(SKEncodedImageFormat.Png, 100);
                    using var debugStream = File.Create(debugPath);
                    debugData.SaveTo(debugStream);
                    Debug.WriteLine($"[LivenessService] DEBUG crop saved to {debugPath}");
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[LivenessService] DEBUG crop save FAILED: {ex}");
                }
            }

            var tensor = new DenseTensor<float>(new[] { 1, 3, 80, 80 });

            // FAST PATH: direct byte-buffer copy into the tensor's backing buffer instead of
            // 6,400 individual GetPixel calls — same technique already used in
            // FaceDetectionService/FaceRecognitionService/MaskHelmetDetectionService. GetPixel is
            // a slow managed call per invocation; this replaces it with one Marshal.Copy plus a
            // straight span write. Channel order (BGR) and value range (raw 0-255, no /255) are
            // preserved exactly as before — only the access pattern changed, not the math.
            int byteCount = resized.ByteCount;
            byte[] pixelBytes = new byte[byteCount];
            Marshal.Copy(resized.GetPixels(), pixelBytes, 0, byteCount);
            int rowBytes = resized.RowBytes;

            var tensorSpan = tensor.Buffer.Span;
            const int size = 80;
            int planeSize = size * size;

            for (int y = 0; y < size; y++)
            {
                int rowOffset = y * rowBytes;
                int rowBase = y * size;
                for (int x = 0; x < size; x++)
                {
                    int offset = rowOffset + x * 4;
                    int pixelIdx = rowBase + x;
                    byte r = pixelBytes[offset];
                    byte g = pixelBytes[offset + 1];
                    byte b = pixelBytes[offset + 2];

                    // BGR order — plane 0 = Blue, plane 1 = Green, plane 2 = Red (matches the
                    // original tensor[0,0,y,x]=Blue / [0,1,y,x]=Green / [0,2,y,x]=Red layout).
                    tensorSpan[pixelIdx] = b;
                    tensorSpan[planeSize + pixelIdx] = g;
                    tensorSpan[2 * planeSize + pixelIdx] = r;
                }
            }

            if (EnableDiagnostics)
            {
                Debug.WriteLine($"[LivenessService] tensor sample: B[40,40]={tensor[0, 0, 40, 40]:F4} G[40,40]={tensor[0, 1, 40, 40]:F4} R[40,40]={tensor[0, 2, 40, 40]:F4}");
            }

            return tensor;
        }

        /// <summary>
        /// Runs the anti-spoofing model and returns (isLive, liveConfidence).
        /// Class mapping confirmed from official source (minivision-ai/Silent-Face-Anti-Spoofing
        /// test.py): index 1 = real face, indices 0 and 2 = fake.
        /// Returns (true, 0f) if the model isn't loaded, so callers fail open rather than blocking
        /// attendance entirely.
        /// </summary>
        public (bool isLive, float liveConfidence) CheckLiveness(SKBitmap original, FaceDetection face)
        {
            if (_antiSpoofSession == null)
            {
                Debug.WriteLine("[LivenessService] Liveness check skipped — model not loaded");
                return (true, 0f);
            }

            try
            {
                var tensor = PreprocessForAntiSpoof(original, face);

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("input", tensor)
                };

                using var results = _antiSpoofSession.Run(inputs);
                var outputTensor = results.First(r => r.Name == "output").AsTensor<float>();

                float[] rawScores = { outputTensor[0, 0], outputTensor[0, 1], outputTensor[0, 2] };

                float maxScore = rawScores.Max();
                float[] expScores = rawScores.Select(s => MathF.Exp(s - maxScore)).ToArray();
                float sumExp = expScores.Sum();
                float[] probabilities = expScores.Select(e => e / sumExp).ToArray();

                Debug.WriteLine($"[LivenessService] Raw scores — idx0={rawScores[0]:F3} idx1={rawScores[1]:F3} idx2={rawScores[2]:F3}");
                Debug.WriteLine($"[LivenessService] Probabilities — idx0={probabilities[0]:P1} idx1={probabilities[1]:P1} idx2={probabilities[2]:P1}");

                float liveProb = probabilities[LiveClassIndex];
                float bestOtherProb = probabilities
                    .Where((_, i) => i != LiveClassIndex)
                    .Max();

                Debug.WriteLine($"[LivenessService] Using idx{LiveClassIndex} as live — live={liveProb:P1}, bestOther={bestOtherProb:P1}");

                bool isLive = liveProb > bestOtherProb;

                return (isLive, liveProb);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[LivenessService] Liveness check FAILED: {ex}");
                return (true, 0f);
            }
        }
    }
}