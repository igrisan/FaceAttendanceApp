using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace FaceAttendanceApp.Services
{
    public record FaceDetection(float X1, float Y1, float X2, float Y2, float Confidence, float[] Landmarks);

    public class FaceDetectionService
    {
        private const int ModelInputSize = 640;
        private static readonly int[] Strides = { 8, 16, 32 };

        private InferenceSession? _session;

        public bool IsLoaded => _session != null;

        public void LoadModel()
        {
            try
            {
                var modelPath = Path.Combine(FileSystem.AppDataDirectory, "face_detection_yunet_2023mar.onnx");

                using (var assetStream = FileSystem.OpenAppPackageFileAsync("face_detection_yunet_2023mar.onnx").Result)
                using (var fileStream = File.Create(modelPath))
                {
                    assetStream.CopyTo(fileStream);
                }

                var options = new SessionOptions();
                options.IntraOpNumThreads = 4;
                options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;

                _session = new InferenceSession(modelPath, options);
                Debug.WriteLine($"[FaceDetectionService] YuNet model loaded successfully ({options.IntraOpNumThreads} threads)");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[FaceDetectionService] YuNet model load FAILED: {ex}");
            }
        }

        /// <summary>
        /// Detects faces on a possibly-downscaled copy of the frame, then maps boxes/landmarks
        /// back to fullResWidth/fullResHeight coordinates. Pass the SAME bitmap for both
        /// detectionInput and the full-res dimensions if you're not using a separate downscaled
        /// copy for detection speed.
        /// </summary>
        public List<FaceDetection> Detect(SKBitmap detectionInput, int fullResWidth, int fullResHeight, float confThreshold = 0.6f, float nmsThreshold = 0.3f)
        {
            var results = Detect(detectionInput, confThreshold, nmsThreshold);

            if (detectionInput.Width == fullResWidth && detectionInput.Height == fullResHeight)
                return results;

            float scaleX = (float)fullResWidth / detectionInput.Width;
            float scaleY = (float)fullResHeight / detectionInput.Height;

            var rescaled = new List<FaceDetection>();
            foreach (var det in results)
            {
                var landmarks = new float[10];
                for (int i = 0; i < 5; i++)
                {
                    landmarks[i * 2] = det.Landmarks[i * 2] * scaleX;
                    landmarks[i * 2 + 1] = det.Landmarks[i * 2 + 1] * scaleY;
                }

                rescaled.Add(new FaceDetection(
                    det.X1 * scaleX, det.Y1 * scaleY,
                    det.X2 * scaleX, det.Y2 * scaleY,
                    det.Confidence, landmarks));
            }

            return rescaled;
        }

        public List<FaceDetection> Detect(SKBitmap original, float confThreshold = 0.6f, float nmsThreshold = 0.3f)
        {
            if (_session == null)
            {
                Debug.WriteLine("[FaceDetectionService] Detect skipped — model not loaded yet");
                return new List<FaceDetection>();
            }

            var tensor = PreprocessImage(original, out float scale, out int padX, out int padY);

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", tensor)
            };

            using var results = _session.Run(inputs);

            return DecodeDetections(results, scale, padX, padY, original.Width, original.Height, confThreshold, nmsThreshold);
        }

        private DenseTensor<float> PreprocessImage(SKBitmap original, out float scale, out int padX, out int padY)
        {
            scale = Math.Min((float)ModelInputSize / original.Width, (float)ModelInputSize / original.Height);
            int scaledWidth = (int)Math.Round(original.Width * scale);
            int scaledHeight = (int)Math.Round(original.Height * scale);

            int localPadX = (ModelInputSize - scaledWidth) / 2;
            int localPadY = (ModelInputSize - scaledHeight) / 2;

            using var resized = original.Resize(new SKImageInfo(scaledWidth, scaledHeight), new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None));

            var canvasInfo = new SKImageInfo(ModelInputSize, ModelInputSize, SKColorType.Rgba8888, SKAlphaType.Unpremul);
            using var canvasBitmap = new SKBitmap(canvasInfo);
            using (var canvas = new SKCanvas(canvasBitmap))
            {
                canvas.Clear(SKColors.Black);
                canvas.DrawBitmap(resized, localPadX, localPadY);
            }

            var tensor = new DenseTensor<float>(new[] { 1, 3, ModelInputSize, ModelInputSize });

            int byteCount = canvasBitmap.ByteCount;
            byte[] pixelBytes = new byte[byteCount];
            Marshal.Copy(canvasBitmap.GetPixels(), pixelBytes, 0, byteCount);
            int rowBytes = canvasBitmap.RowBytes;

            // FAST PATH v2: write directly into the tensor's backing buffer instead of using
            // the DenseTensor indexer (tensor[0,0,y,x]), which recomputes strides on every
            // single call. This is a straight span write, no index math per element.
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

            padX = localPadX;
            padY = localPadY;

            return tensor;
        }

        private List<FaceDetection> DecodeDetections(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results,
            float scale, int padX, int padY,
            int originalWidth, int originalHeight,
            float confThreshold, float nmsThreshold)
        {
            var candidates = new List<FaceDetection>();

            foreach (var stride in Strides)
            {
                var clsTensor = results.First(r => r.Name == $"cls_{stride}").AsTensor<float>();
                var objTensor = results.First(r => r.Name == $"obj_{stride}").AsTensor<float>();
                var bboxTensor = results.First(r => r.Name == $"bbox_{stride}").AsTensor<float>();
                var kpsTensor = results.First(r => r.Name == $"kps_{stride}").AsTensor<float>();

                int gridSize = ModelInputSize / stride;

                for (int row = 0; row < gridSize; row++)
                {
                    for (int col = 0; col < gridSize; col++)
                    {
                        int idx = row * gridSize + col;

                        float clsScore = Math.Clamp(clsTensor[0, idx, 0], 0f, 1f);
                        float objScore = Math.Clamp(objTensor[0, idx, 0], 0f, 1f);
                        float confidence = MathF.Sqrt(clsScore * objScore);

                        if (confidence < confThreshold)
                            continue;

                        float cx = (col + bboxTensor[0, idx, 0]) * stride;
                        float cy = (row + bboxTensor[0, idx, 1]) * stride;
                        float w = MathF.Exp(bboxTensor[0, idx, 2]) * stride;
                        float h = MathF.Exp(bboxTensor[0, idx, 3]) * stride;

                        float x1 = cx - w / 2f;
                        float y1 = cy - h / 2f;
                        float x2 = x1 + w;
                        float y2 = y1 + h;

                        var landmarks = new float[10];
                        for (int p = 0; p < 5; p++)
                        {
                            landmarks[p * 2] = (col + kpsTensor[0, idx, p * 2]) * stride;
                            landmarks[p * 2 + 1] = (row + kpsTensor[0, idx, p * 2 + 1]) * stride;
                        }

                        candidates.Add(new FaceDetection(x1, y1, x2, y2, confidence, landmarks));
                    }
                }
            }

            var kept = NonMaxSuppression(candidates, nmsThreshold);

            var mapped = new List<FaceDetection>();
            foreach (var det in kept)
            {
                float x1 = Math.Clamp((det.X1 - padX) / scale, 0, originalWidth);
                float y1 = Math.Clamp((det.Y1 - padY) / scale, 0, originalHeight);
                float x2 = Math.Clamp((det.X2 - padX) / scale, 0, originalWidth);
                float y2 = Math.Clamp((det.Y2 - padY) / scale, 0, originalHeight);

                var mappedLandmarks = new float[10];
                for (int i = 0; i < 5; i++)
                {
                    mappedLandmarks[i * 2] = (det.Landmarks[i * 2] - padX) / scale;
                    mappedLandmarks[i * 2 + 1] = (det.Landmarks[i * 2 + 1] - padY) / scale;
                }

                mapped.Add(new FaceDetection(x1, y1, x2, y2, det.Confidence, mappedLandmarks));
            }

            return mapped;
        }

        private List<FaceDetection> NonMaxSuppression(List<FaceDetection> boxes, float iouThreshold)
        {
            var sorted = boxes.OrderByDescending(b => b.Confidence).ToList();
            var kept = new List<FaceDetection>();

            while (sorted.Count > 0)
            {
                var best = sorted[0];
                kept.Add(best);
                sorted.RemoveAt(0);

                sorted.RemoveAll(b => IoU(best, b) > iouThreshold);
            }

            return kept;
        }

        private float IoU(FaceDetection a, FaceDetection b)
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