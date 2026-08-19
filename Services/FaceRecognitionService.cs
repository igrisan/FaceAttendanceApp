using System.Diagnostics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using FaceAttendanceApp.Model;

namespace FaceAttendanceApp.Services
{
    public class FaceRecognitionService
    {
        private const int EmbedInputSize = 112;
        private const float MatchThreshold = 0.65f;

        // Lower threshold used when matching against any OCCLUDED reference embedding (masked,
        // spectacles, any combo — anything that isn't the primary "Bare Face" reference).
        // Occluded embeddings carry less facial detail, so similarity scores run lower even for
        // a correct match — 0.65 was tuned for full-face comparisons and is too strict here.
        // This value is a starting point; it should be revisited once there's real occluded-
        // match data to tune against, the same way the PPE mask-detection threshold was tuned
        // from device logs.
        private const float MaskedMatchThreshold = 0.50f;

        private static readonly SKPoint RefRightEye = new(38.2946f, 51.6963f);
        private static readonly SKPoint RefLeftEye = new(73.5318f, 51.5014f);

        private InferenceSession? _sfaceSession;

        public bool IsLoaded => _sfaceSession != null;

        public void LoadModel()
        {
            try
            {
                var modelPath = Path.Combine(FileSystem.AppDataDirectory, "face_recognition_sface_2021dec.onnx");

                using (var assetStream = FileSystem.OpenAppPackageFileAsync("face_recognition_sface_2021dec.onnx").Result)
                using (var fileStream = File.Create(modelPath))
                {
                    assetStream.CopyTo(fileStream);
                }

                var options = new SessionOptions();
                // NOTE: was Environment.ProcessorCount (8 threads on this device). SFace runs on
                // a tiny 112x112 single-face crop — thread-launch/scheduling overhead for 8
                // threads on a model this small outweighs any parallelism gained, especially
                // since this stage runs sequentially after detection/liveness (nothing else is
                // competing for the CPU at the same time anyway). 2 threads matches what's
                // already used for the similarly-small PPE model and measured faster in testing
                // on this class of ARM SoC.
                options.IntraOpNumThreads = 2;
                options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;

                _sfaceSession = new InferenceSession(modelPath, options);
                Debug.WriteLine($"[FaceRecognitionService] SFace model loaded successfully ({options.IntraOpNumThreads} threads)");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[FaceRecognitionService] SFace model load FAILED: {ex}");
            }
        }

        public SKBitmap AlignFace(SKBitmap original, float[] landmarks)
        {
            var rightEye = new SKPoint(landmarks[0], landmarks[1]);
            var leftEye = new SKPoint(landmarks[2], landmarks[3]);

            float srcDx = leftEye.X - rightEye.X;
            float srcDy = leftEye.Y - rightEye.Y;
            float srcDist = MathF.Sqrt(srcDx * srcDx + srcDy * srcDy);
            float srcAngle = MathF.Atan2(srcDy, srcDx);

            float refDx = RefLeftEye.X - RefRightEye.X;
            float refDy = RefLeftEye.Y - RefRightEye.Y;
            float refDist = MathF.Sqrt(refDx * refDx + refDy * refDy);
            float refAngle = MathF.Atan2(refDy, refDx);

            float scale = refDist / srcDist;
            float rotation = refAngle - srcAngle;

            var matrix = SKMatrix.CreateTranslation(-rightEye.X, -rightEye.Y);
            matrix = matrix.PostConcat(SKMatrix.CreateScale(scale, scale));
            matrix = matrix.PostConcat(SKMatrix.CreateRotation(rotation));
            matrix = matrix.PostConcat(SKMatrix.CreateTranslation(RefRightEye.X, RefRightEye.Y));

            var aligned = new SKBitmap(EmbedInputSize, EmbedInputSize);

            using (var canvas = new SKCanvas(aligned))
            {
                canvas.Clear(SKColors.Black);
                canvas.SetMatrix(matrix);
                canvas.DrawBitmap(original, 0, 0);
            }

            return aligned;
        }

        public float[]? GetEmbedding(SKBitmap alignedFace)
        {
            if (_sfaceSession == null)
            {
                Debug.WriteLine("[FaceRecognitionService] Embedding skipped — SFace model not loaded yet");
                return null;
            }

            try
            {
                // FAST PATH: direct byte-buffer copy into the tensor's backing buffer, same
                // technique used in FaceDetectionService/MaskHelmetDetectionService, instead of
                // calling alignedFace.GetPixel(x, y) + tensor[0,c,y,x] per pixel (12,544 slow
                // managed calls + stride recompute each). This was the single biggest cost in
                // the whole pipeline (254-456ms out of ~1s+ total frame time).
                var tensor = new DenseTensor<float>(new[] { 1, 3, EmbedInputSize, EmbedInputSize });

                int byteCount = alignedFace.ByteCount;
                byte[] pixelBytes = new byte[byteCount];
                Marshal.Copy(alignedFace.GetPixels(), pixelBytes, 0, byteCount);
                int rowBytes = alignedFace.RowBytes;

                var tensorSpan = tensor.Buffer.Span;
                int planeSize = EmbedInputSize * EmbedInputSize;

                // SFace was trained/exported via OpenCV's reference pipeline, which loads images
                // with cv::imread (BGR order) and feeds them straight into the model with no
                // color-space conversion anywhere (alignCrop and feature() both leave channel
                // order untouched). So the ONNX graph expects BGR, not RGB.
                //
                // SkiaSharp on Android/iOS defaults SKBitmap to Rgba8888, meaning offset+0 is R,
                // offset+1 is G, offset+2 is B in the raw pixel buffer. To match the model's
                // expected BGR input, plane 0 must be filled with B, plane 2 with R (plane 1/G
                // is unaffected since it's the middle channel either way).
                //
                // No additional scaling or mean subtraction is applied here — the reference
                // pipeline also passes raw 0-255 values straight through, so this matches that
                // contract exactly.
                for (int y = 0; y < EmbedInputSize; y++)
                {
                    int rowOffset = y * rowBytes;
                    int rowBase = y * EmbedInputSize;
                    for (int x = 0; x < EmbedInputSize; x++)
                    {
                        int offset = rowOffset + x * 4;
                        int pixelIdx = rowBase + x;
                        tensorSpan[pixelIdx] = pixelBytes[offset + 2];
                        tensorSpan[planeSize + pixelIdx] = pixelBytes[offset + 1];
                        tensorSpan[2 * planeSize + pixelIdx] = pixelBytes[offset];
                    }
                }

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("data", tensor)
                };

                using var results = _sfaceSession.Run(inputs);
                var embeddingTensor = results.First(r => r.Name == "fc1").AsTensor<float>();

                var embedding = new float[128];
                for (int i = 0; i < 128; i++)
                {
                    embedding[i] = embeddingTensor[0, i];
                }

                float norm = MathF.Sqrt(embedding.Sum(v => v * v));
                for (int i = 0; i < embedding.Length; i++)
                {
                    embedding[i] /= norm;
                }

                return embedding;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[FaceRecognitionService] Embedding FAILED: {ex}");
                return null;
            }
        }

        /// <summary>
        /// Cosine similarity between two ALREADY-NORMALIZED embeddings — this is just a dot
        /// product. Uses TensorPrimitives.Dot (System.Numerics.Tensors) instead of a manual
        /// for-loop so the CPU can process several floats at once (SIMD) instead of one at a
        /// time. Same math, same result, just faster per comparison.
        /// </summary>
        public float CosineSimilarity(float[] a, float[] b)
        {
            return TensorPrimitives.Dot((ReadOnlySpan<float>)a, (ReadOnlySpan<float>)b);
        }

        /// <summary>
        /// Legacy single-embedding-per-worker matcher. Kept only for callers that still pass a
        /// plain List&lt;Worker&gt; (e.g. RegisterWorkerPage's duplicate-check, which only ever
        /// needs to compare against each worker's primary/bare-face embedding). Scan matching
        /// should use the candidate-list overload below instead, since it covers every
        /// registered variant, not just the primary one.
        /// </summary>
        public (Worker? worker, float score) FindBestMatch(List<Worker> workers, float[] embedding)
        {
            Worker? best = null;
            float bestScore = -1f;

            foreach (var worker in workers)
            {
                float score = CosineSimilarity(embedding, worker.GetEmbedding());
                if (score > bestScore)
                {
                    bestScore = score;
                    best = worker;
                }
            }

            if (best != null && bestScore >= MatchThreshold)
            {
                return (best, bestScore);
            }

            return (null, bestScore);
        }

        /// <summary>
        /// Finds the best-matching worker across ALL of their registered face references —
        /// primary/bare-face plus every labeled variant (masked, spectacles, any combo).
        /// Compares the captured embedding against every candidate and returns whichever scores
        /// highest overall, so a worker matches correctly regardless of which look they're
        /// wearing right now, without needing to know in advance which variant to check.
        ///
        /// Threshold: the primary "Bare Face" reference uses the stricter MatchThreshold since
        /// it carries full facial detail. Any other variant (an occluded reference, by
        /// definition — masked, spectacles, etc.) uses the more forgiving MaskedMatchThreshold,
        /// since occluded embeddings run lower similarity even for a correct match. This mirrors
        /// the reasoning the old isMasked-specific threshold used, just generalized to any label.
        /// </summary>
        public (Worker? worker, string matchedLabel, float score) FindBestMatch(List<WorkerEmbeddingCandidate> candidates, float[] embedding)
        {
            WorkerEmbeddingCandidate? best = null;
            float bestScore = -1f;

            // ReadOnlySpan over the incoming embedding once, outside the loop, so we're not
            // re-wrapping it on every single comparison.
            ReadOnlySpan<float> embeddingSpan = embedding;

            foreach (var candidate in candidates)
            {
                float score = TensorPrimitives.Dot(embeddingSpan, (ReadOnlySpan<float>)candidate.Embedding);
                if (score > bestScore)
                {
                    bestScore = score;
                    best = candidate;
                }
            }

            if (best == null)
            {
                return (null, string.Empty, bestScore);
            }

            float threshold = best.Label == "Bare Face" ? MatchThreshold : MaskedMatchThreshold;

            if (bestScore >= threshold)
            {
                return (best.Worker, best.Label, bestScore);
            }

            return (null, string.Empty, bestScore);
        }
    }
}