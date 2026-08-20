using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace FaceAttendanceApp.Services
{
    // Confidence = objectness * classScore (standard YOLOX decode; used for NMS ranking and for
    // goggles/eyeglasses, which decode reliably this way).
    // ClassScore = the raw per-class sigmoid score alone, with no objectness multiplied in. Used
    // for mask/helmet instead, because objectness is unreliable at the anchors those classes land
    // on for this exported model (see CheckHelmetAndMask for the full explanation).
    public record PpeDetection(float X1, float Y1, float X2, float Y2, float Confidence, float ClassScore, int ClassId, string ClassName);

    public class MaskHelmetDetectionService
    {
        private const int ModelInputSize = 640;

        private static readonly string[] ClassNames = { "goggles", "eyeglasses", "helmet", "mask" };
        private const int GogglesClassId = 0;
        private const int EyeglassesClassId = 1;
        private const int HelmetClassId = 2;
        private const int MaskClassId = 3;

        private InferenceSession? _session;
        private (int stride, int gridW, int gridH)[]? _gridMeta;

        public bool IsLoaded => _session != null;

        public void LoadModel()
        {
            try
            {
                var modelPath = Path.Combine(FileSystem.AppDataDirectory, "ppe_detection.onnx");

                using (var assetStream = FileSystem.OpenAppPackageFileAsync("ppe_detection.onnx").Result)
                using (var fileStream = File.Create(modelPath))
                {
                    assetStream.CopyTo(fileStream);
                }

                var options = new SessionOptions();
                options.IntraOpNumThreads = 2;
                options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;

                _session = new InferenceSession(modelPath, options);
                Debug.WriteLine($"[MaskHelmetDetectionService] PPE model (YOLOX-Nano) loaded successfully ({options.IntraOpNumThreads} threads)");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[MaskHelmetDetectionService] PPE model load FAILED: {ex}");
            }
        }

        public List<PpeDetection> Detect(SKBitmap original, float confThreshold = 0.5f, float nmsThreshold = 0.45f)
        {
            if (_session == null)
            {
                Debug.WriteLine("[MaskHelmetDetectionService] Detect skipped — model not loaded yet");
                return new List<PpeDetection>();
            }

            var tensor = PreprocessImage(original, out float scale);

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", tensor)
            };

            using var results = _session.Run(inputs);

            return DecodeDetections(results, scale, original.Width, original.Height, confThreshold, nmsThreshold);
        }

        private const float MaskThreshold = 0.35f;
        private const float HelmetThreshold = 0.5f;
        private const float RawDetectionThreshold = 0.1f;

        private const float CropPaddingSides = 0.3f;
        private const float CropPaddingTop = 0.6f;
        private const float CropPaddingBottom = 0.2f;

        // Standard YOLOX decoding scores a detection as objectness * classScore. That works fine
        // for goggles/eyeglasses in this model, but diagnostic logging showed it badly
        // under-reports mask AND helmet: raw class scores of 0.75-0.85 (mask genuinely visible)
        // were collapsing to combined confidences as low as 0.16, because objectness is
        // miscalibrated at the grid anchors those two classes land on (likely because mask/helmet
        // sit near the top/bottom edge of the padded face crop, a region this export wasn't
        // confidently trained on for "is there anything here at all"). The class-score branch
        // itself is reliable — it's specifically the objectness multiplier that's broken for
        // these two classes — so mask/helmet are decided on ClassScore alone, while
        // goggles/eyeglasses keep using the combined Confidence that already works for them.
        public (bool hasHelmet, bool hasMask) CheckHelmetAndMask(SKBitmap original, float faceX1, float faceY1, float faceX2, float faceY2)
        {
            using var faceCrop = CropWithPadding(original, faceX1, faceY1, faceX2, faceY2);

            var detections = Detect(faceCrop, RawDetectionThreshold);

            var bestMask = detections.Where(d => d.ClassId == MaskClassId).OrderByDescending(d => d.ClassScore).FirstOrDefault();
            var bestHelmet = detections.Where(d => d.ClassId == HelmetClassId).OrderByDescending(d => d.ClassScore).FirstOrDefault();
            Debug.WriteLine($"[MaskHelmetDetectionService] Best kept mask candidate: {(bestMask == null ? "none" : $"classScore={bestMask.ClassScore:F3}")} (threshold={MaskThreshold}); best helmet candidate: {(bestHelmet == null ? "none" : $"classScore={bestHelmet.ClassScore:F3}")} (threshold={HelmetThreshold})");

            bool hasHelmet = bestHelmet != null && bestHelmet.ClassScore >= HelmetThreshold;
            bool hasMask = bestMask != null && bestMask.ClassScore >= MaskThreshold;
            return (hasHelmet, hasMask);
        }

        public (bool hasHelmet, bool hasMask) CheckHelmetAndMask(SKBitmap original)
        {
            var detections = Detect(original, RawDetectionThreshold);
            bool hasHelmet = detections.Any(d => d.ClassId == HelmetClassId && d.ClassScore >= HelmetThreshold);
            bool hasMask = detections.Any(d => d.ClassId == MaskClassId && d.ClassScore >= MaskThreshold);
            return (hasHelmet, hasMask);
        }

        public (bool hasGoggles, bool hasEyeglasses) CheckGogglesAndEyeglasses(SKBitmap original, float faceX1, float faceY1, float faceX2, float faceY2)
        {
            using var faceCrop = CropWithPadding(original, faceX1, faceY1, faceX2, faceY2);
            var detections = Detect(faceCrop, RawDetectionThreshold);

            bool hasGoggles = detections.Any(d => d.ClassId == GogglesClassId && d.Confidence >= HelmetThreshold);
            bool hasEyeglasses = detections.Any(d => d.ClassId == EyeglassesClassId && d.Confidence >= HelmetThreshold);
            return (hasGoggles, hasEyeglasses);
        }

        private SKBitmap CropWithPadding(SKBitmap original, float x1, float y1, float x2, float y2)
        {
            float faceWidth = x2 - x1;
            float faceHeight = y2 - y1;

            int cropX1 = (int)Math.Max(0, x1 - faceWidth * CropPaddingSides);
            int cropY1 = (int)Math.Max(0, y1 - faceHeight * CropPaddingTop);
            int cropX2 = (int)Math.Min(original.Width, x2 + faceWidth * CropPaddingSides);
            int cropY2 = (int)Math.Min(original.Height, y2 + faceHeight * CropPaddingBottom);

            int cropWidth = Math.Max(1, cropX2 - cropX1);
            int cropHeight = Math.Max(1, cropY2 - cropY1);

            using var surface = SKSurface.Create(new SKImageInfo(cropWidth, cropHeight));
            var canvas = surface.Canvas;
            var srcRect = SKRect.Create(cropX1, cropY1, cropWidth, cropHeight);
            var destRect = SKRect.Create(0, 0, cropWidth, cropHeight);
            canvas.DrawBitmap(original, srcRect, destRect);

            using var image = surface.Snapshot();
            return SKBitmap.FromImage(image);
        }

        private DenseTensor<float> PreprocessImage(SKBitmap original, out float scale)
        {
            scale = Math.Min((float)ModelInputSize / original.Width, (float)ModelInputSize / original.Height);
            int scaledWidth = (int)(original.Width * scale);
            int scaledHeight = (int)(original.Height * scale);

            using var resized = original.Resize(new SKImageInfo(scaledWidth, scaledHeight), new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None));

            var canvasInfo = new SKImageInfo(ModelInputSize, ModelInputSize, SKColorType.Rgba8888, SKAlphaType.Unpremul);
            using var canvasBitmap = new SKBitmap(canvasInfo);
            using (var canvas = new SKCanvas(canvasBitmap))
            {
                canvas.Clear(new SKColor(114, 114, 114));
                canvas.DrawBitmap(resized, 0, 0);
            }

            var tensor = new DenseTensor<float>(new[] { 1, 3, ModelInputSize, ModelInputSize });

            int byteCount = canvasBitmap.ByteCount;
            byte[] pixelBytes = new byte[byteCount];
            Marshal.Copy(canvasBitmap.GetPixels(), pixelBytes, 0, byteCount);
            int rowBytes = canvasBitmap.RowBytes;

            // FAST PATH v2: direct span write into tensor buffer, no indexer stride math per element.
            var tensorSpan = tensor.Buffer.Span;
            int planeSize = ModelInputSize * ModelInputSize;

            for (int y = 0; y < ModelInputSize; y++)
            {
                int rowOffset = y * rowBytes;
                int rowBase = y * ModelInputSize;
                for (int x = 0; x < ModelInputSize; x++)
                {
                    int offset = rowOffset + x * 4;
                    int pixelIdx = rowBase + x;
                    tensorSpan[pixelIdx] = pixelBytes[offset];                     // R plane
                    tensorSpan[planeSize + pixelIdx] = pixelBytes[offset + 1];     // G plane
                    tensorSpan[2 * planeSize + pixelIdx] = pixelBytes[offset + 2]; // B plane
                }
            }

            return tensor;
        }

        private (int stride, int gridW, int gridH)[] GetGridMeta()
        {
            if (_gridMeta != null) return _gridMeta;

            var strides = new[] { 8, 16, 32 };
            var meta = new List<(int, int, int)>();
            foreach (var stride in strides)
            {
                int gridSize = ModelInputSize / stride;
                for (int gy = 0; gy < gridSize; gy++)
                {
                    for (int gx = 0; gx < gridSize; gx++)
                    {
                        meta.Add((stride, gx, gy));
                    }
                }
            }
            _gridMeta = meta.ToArray();
            return _gridMeta;
        }

        private List<PpeDetection> DecodeDetections(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results,
            float scale, int originalWidth, int originalHeight,
            float confThreshold, float nmsThreshold)
        {
            var output = results.First().AsTensor<float>();
            int numAnchors = output.Dimensions[1];
            int numAttrs = output.Dimensions[2];

            var gridMeta = GetGridMeta();
            if (gridMeta.Length != numAnchors)
            {
                Debug.WriteLine($"[MaskHelmetDetectionService] WARNING: grid meta count ({gridMeta.Length}) != model anchors ({numAnchors}). Decoding will likely be wrong — check ModelInputSize / stride config.");
            }

            var candidates = new List<PpeDetection>();
            var maxScorePerClass = new float[ClassNames.Length];

            for (int i = 0; i < numAnchors; i++)
            {
                float rawCx = output[0, i, 0];
                float rawCy = output[0, i, 1];
                float rawW = output[0, i, 2];
                float rawH = output[0, i, 3];
                float objectness = output[0, i, 4];

                var (stride, gx, gy) = i < gridMeta.Length ? gridMeta[i] : (8, 0, 0);
                float cx = (rawCx + gx) * stride;
                float cy = (rawCy + gy) * stride;
                float w = MathF.Exp(rawW) * stride;
                float h = MathF.Exp(rawH) * stride;

                float x1 = cx - w / 2f;
                float y1 = cy - h / 2f;
                float x2 = cx + w / 2f;
                float y2 = cy + h / 2f;

                // Multi-label decode: goggles/eyeglasses/helmet/mask are NOT mutually
                // exclusive (a face can show a mask AND eyeglasses at the same time),
                // so every class is evaluated independently against confThreshold
                // instead of only taking the anchor's single highest-scoring class
                // (argmax). Argmax silently drops a real detection whenever two
                // classes both score highly at the same anchor — e.g. mask=0.76
                // losing to eyeglasses=0.73 and never being recorded at all, even
                // though mask's own objectness*score would have cleared the
                // threshold on its own.
                for (int c = 0; c < ClassNames.Length; c++)
                {
                    float clsScore = output[0, i, 5 + c];
                    if (clsScore > maxScorePerClass[c]) maxScorePerClass[c] = clsScore;

                    float finalScore = objectness * clsScore;

                    // Gate on EITHER the combined (objectness*classScore) score or the raw
                    // classScore alone. Mask/helmet rely on ClassScore downstream (see
                    // CheckHelmetAndMask) because objectness collapses their combined score to
                    // near-zero even for genuine detections — if we only gated on finalScore
                    // here, those candidates would be discarded before ClassScore ever gets
                    // checked, which is exactly what was happening before this fix (mask
                    // candidates disappearing entirely instead of showing a low score).
                    if (finalScore < confThreshold && clsScore < confThreshold)
                        continue;

                    candidates.Add(new PpeDetection(x1, y1, x2, y2, finalScore, clsScore, c, ClassNames[c]));
                }
            }

            Debug.WriteLine($"[MaskHelmetDetectionService] All max scores — " +
                string.Join(", ", ClassNames.Select((name, idx) => $"{name}={maxScorePerClass[idx]:F3}")));

            var kept = NonMaxSuppressionPerClass(candidates, nmsThreshold);

            var mapped = new List<PpeDetection>();
            foreach (var det in kept)
            {
                float x1 = Math.Clamp(det.X1 / scale, 0, originalWidth);
                float y1 = Math.Clamp(det.Y1 / scale, 0, originalHeight);
                float x2 = Math.Clamp(det.X2 / scale, 0, originalWidth);
                float y2 = Math.Clamp(det.Y2 / scale, 0, originalHeight);

                mapped.Add(new PpeDetection(x1, y1, x2, y2, det.Confidence, det.ClassScore, det.ClassId, det.ClassName));
            }

            return mapped;
        }

        private List<PpeDetection> NonMaxSuppressionPerClass(List<PpeDetection> boxes, float iouThreshold)
        {
            var kept = new List<PpeDetection>();

            foreach (var group in boxes.GroupBy(b => b.ClassId))
            {
                var sorted = group.OrderByDescending(b => b.ClassScore).ToList();

                while (sorted.Count > 0)
                {
                    var best = sorted[0];
                    kept.Add(best);
                    sorted.RemoveAt(0);

                    sorted.RemoveAll(b => IoU(best, b) > iouThreshold);
                }
            }

            return kept;
        }

        private float IoU(PpeDetection a, PpeDetection b)
        {
            float interX1 = Math.Max(a.X1, b.X1);
            float interY1 = Math.Max(a.Y1, b.Y1);
            float interX2 = Math.Min(a.X2, b.X2);
            float interY2 = Math.Min(a.Y2, b.Y2);

            float interWidth = Math.Max(0, interX2 - interX1);
            float interHeight = Math.Max(0, interY2 - interY1);
            float interArea = interWidth * interHeight;

            float areaA = (a.X2 - a.X1) * (a.Y2 - a.Y1);
            float areaB = (b.X2 - b.X1) * (b.Y2 - b.Y1);

            float unionArea = areaA + areaB - interArea;

            return unionArea <= 0 ? 0 : interArea / unionArea;
        }
    }
}