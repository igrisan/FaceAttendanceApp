using SkiaSharp;

namespace FaceAttendanceApp.Helpers
{
    /// <summary>
    /// Pre-inference quality gates: rejects blurry frames before they ever reach
    /// YuNet/SFace/anti-spoof, and normalizes lighting so outdoor/low-light shots
    /// look more like the conditions the models were trained on.
    ///
    /// Why this matters: a blurry or badly-lit frame doesn't just fail silently —
    /// it produces a LOW-CONFIDENCE WRONG answer (a spoofed-looking liveness score,
    /// a mediocre match score) instead of no answer at all. Catching it here means
    /// you retry the capture instead of logging a shaky 0.74 match against a real
    /// worker.
    /// </summary>
    public static class ImageQualityHelper
    {
        /// <summary>
        /// Below this, a frame is considered too blurry to trust for recognition.
        /// This threshold is scale/resolution dependent — tune against your own
        /// device + lighting conditions using LogSharpness() below before locking
        /// it in. 60-80 is a reasonable starting point for a face-sized crop.
        /// </summary>
        public const double BlurVarianceThreshold = 60.0;

        /// <summary>
        /// Computes the variance of the Laplacian (edge-detection) response over a
        /// grayscale version of the image. Sharp images have lots of local
        /// intensity variation (high variance); blurry images are smooth (low
        /// variance). This is the standard, cheap, no-ML way to detect blur.
        /// </summary>
        public static double ComputeBlurVariance(SKBitmap bitmap)
        {
            // Downscale for speed — blur detection doesn't need full resolution,
            // and this keeps the cost low enough to run on every captured frame.
            const int workSize = 256;
            using var gray = ToGrayscaleResized(bitmap, workSize);

            int width = gray.Width;
            int height = gray.Height;
            var pixels = gray.Pixels;

            // 3x3 Laplacian kernel: [0 1 0; 1 -4 1; 0 1 0]
            double sum = 0;
            double sumSq = 0;
            int count = 0;

            for (int y = 1; y < height - 1; y++)
            {
                for (int x = 1; x < width - 1; x++)
                {
                    int idx = y * width + x;
                    byte center = pixels[idx].Red;
                    byte up = pixels[idx - width].Red;
                    byte down = pixels[idx + width].Red;
                    byte left = pixels[idx - 1].Red;
                    byte right = pixels[idx + 1].Red;

                    double laplacian = up + down + left + right - 4.0 * center;

                    sum += laplacian;
                    sumSq += laplacian * laplacian;
                    count++;
                }
            }

            if (count == 0) return 0;

            double mean = sum / count;
            double variance = (sumSq / count) - (mean * mean);
            return variance;
        }

        /// <summary>
        /// True if the frame is sharp enough to run through recognition/liveness.
        /// Call this on the cropped face region right after detection, before
        /// liveness/embedding — it's cheap (256x256 grayscale) relative to the
        /// ONNX inference stages downstream.
        /// </summary>
        public static bool IsSharpEnough(SKBitmap faceCrop, out double variance)
        {
            variance = ComputeBlurVariance(faceCrop);
            return variance >= BlurVarianceThreshold;
        }

        /// <summary>
        /// Applies simple contrast-limited histogram stretching per RGB channel.
        /// Not full CLAHE (SkiaSharp doesn't expose tiled histogram equalization
        /// natively), but a global percentile stretch that meaningfully helps
        /// underexposed outdoor/backlit or dim indoor shots without needing an
        /// extra native dependency. Run this on the face crop before liveness
        /// and before embedding — both are sensitive to exposure.
        /// </summary>
        public static SKBitmap NormalizeLighting(SKBitmap source)
        {
            var result = new SKBitmap(source.Width, source.Height, source.ColorType, source.AlphaType);
            var srcPixels = source.Pixels;
            var dstPixels = new SKColor[srcPixels.Length];

            // Find 1st/99th percentile per channel to clip outliers before stretching
            // (avoids a few blown-out highlight pixels compressing the whole range).
            var rVals = new byte[srcPixels.Length];
            var gVals = new byte[srcPixels.Length];
            var bVals = new byte[srcPixels.Length];

            for (int i = 0; i < srcPixels.Length; i++)
            {
                rVals[i] = srcPixels[i].Red;
                gVals[i] = srcPixels[i].Green;
                bVals[i] = srcPixels[i].Blue;
            }

            Array.Sort(rVals);
            Array.Sort(gVals);
            Array.Sort(bVals);

            int lowIdx = (int)(srcPixels.Length * 0.01);
            int highIdx = (int)(srcPixels.Length * 0.99);

            byte rLow = rVals[lowIdx], rHigh = rVals[highIdx];
            byte gLow = gVals[lowIdx], gHigh = gVals[highIdx];
            byte bLow = bVals[lowIdx], bHigh = bVals[highIdx];

            // Guard against a near-flat channel (would divide by ~0)
            int rRange = Math.Max(1, rHigh - rLow);
            int gRange = Math.Max(1, gHigh - gLow);
            int bRange = Math.Max(1, bHigh - bLow);

            for (int i = 0; i < srcPixels.Length; i++)
            {
                var p = srcPixels[i];
                byte r = (byte)Math.Clamp((p.Red - rLow) * 255 / rRange, 0, 255);
                byte g = (byte)Math.Clamp((p.Green - gLow) * 255 / gRange, 0, 255);
                byte b = (byte)Math.Clamp((p.Blue - bLow) * 255 / bRange, 0, 255);
                dstPixels[i] = new SKColor(r, g, b, p.Alpha);
            }

            result.Pixels = dstPixels;
            return result;
        }

        private static SKBitmap ToGrayscaleResized(SKBitmap source, int targetSize)
        {
            float scale = (float)targetSize / Math.Max(source.Width, source.Height);
            int w = Math.Max(1, (int)(source.Width * scale));
            int h = Math.Max(1, (int)(source.Height * scale));

            using var resized = source.Resize(new SKImageInfo(w, h), new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None));

            var grayInfo = new SKImageInfo(w, h, SKColorType.Gray8, SKAlphaType.Opaque);
            var gray = new SKBitmap(grayInfo);

            using var canvas = new SKCanvas(gray);
            using var paint = new SKPaint();
            paint.ColorFilter = SKColorFilter.CreateColorMatrix(new float[]
            {
                0.299f, 0.587f, 0.114f, 0, 0,
                0.299f, 0.587f, 0.114f, 0, 0,
                0.299f, 0.587f, 0.114f, 0, 0,
                0,      0,      0,      1, 0
            });
            canvas.DrawBitmap(resized, 0, 0, paint);

            return gray;
        }
    }
}