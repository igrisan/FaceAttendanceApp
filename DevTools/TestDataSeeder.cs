using System.Diagnostics;
using SkiaSharp;
using FaceAttendanceApp.Model;
using FaceAttendanceApp.Services;

namespace FaceAttendanceApp.DevTools
{
    /// <summary>
    /// DEV-ONLY tool for load-testing ScanAttendancePage against 5,000–10,000+ enrolled
    /// workers, matching the app's stated requirement to work at that scale.
    ///
    /// Two separate seeding modes, because they test two separate things:
    ///
    ///   1) SeedRandomWorkersAsync — fills the DB with random normalized embeddings.
    ///      Good for: DB load time, cache-refresh time, per-frame matching time, memory.
    ///      USELESS for: accuracy. Random noise vectors don't share the similarity
    ///      distribution of real face embeddings, so false-match-rate numbers from this
    ///      data are meaningless.
    ///
    ///   2) SeedFromImageFolderAsync — runs real photos through your ACTUAL detection +
    ///      recognition pipeline (same services ScanAttendancePage uses) and inserts the
    ///      real resulting embeddings. Use a public face dataset (e.g. a subset of LFW)
    ///      for this. This is what actually tells you whether match scores/thresholds
    ///      hold up once the candidate pool is huge.
    ///
    /// Recommended combo for a realistic 10k-scale test:
    ///   - Seed ~9,950 random distractors (fast, tests scale)
    ///   - Seed ~50 real faces including your own (tests that YOUR face still matches
    ///     correctly and the score doesn't degrade with a huge pool)
    ///
    /// IMPORTANT: Never ship access to this in a release build. Wrap any UI entry point
    /// to these methods in #if DEBUG, or otherwise keep it unreachable in production —
    /// this can silently balloon or corrupt the real worker table if triggered by
    /// accident on a device in the field.
    /// </summary>
    public class TestDataSeeder
    {
        private readonly WorkerDatabase _db;
        private static readonly Random _rng = new();

        /// <summary>
        /// Prefix used to tag every worker this seeder creates, so they can be found and
        /// wiped later with ClearSeededTestDataAsync without touching real enrollments.
        /// </summary>
        public const string TestWorkerNamePrefix = "TestWorker_";

        public TestDataSeeder(WorkerDatabase db)
        {
            _db = db;
        }

        /// <summary>
        /// Inserts `count` workers with random normalized embeddings, for pure scale/speed
        /// testing (DB load, cache refresh, per-frame matching loop, memory).
        ///
        /// embeddingDim must match whatever your FaceRecognitionService actually produces.
        /// Don't assume this — log embedding.Length once from a real GetEmbedding() call in
        /// your pipeline and confirm before running a large seed, or a dimension mismatch
        /// will silently produce garbage matches instead of an error.
        ///
        /// Uses a single transaction via WorkerDatabase.RunBulkInsertAsync — inserting 10k
        /// rows one at a time through SaveWorkerAsync would itself take long enough to skew
        /// the very timing numbers you're trying to measure.
        /// </summary>
        /// <param name="count">How many synthetic workers to insert (e.g. 9950).</param>
        /// <param name="embeddingDim">Dimension of your recognition model's embedding vector.</param>
        /// <param name="progress">Optional callback invoked every 500 records, useful for a progress bar.</param>
        public async Task SeedRandomWorkersAsync(int count, int embeddingDim, IProgress<int>? progress = null)
        {
            var sw = Stopwatch.StartNew();
            var workers = new List<Worker>(count);

            for (int i = 0; i < count; i++)
            {
                workers.Add(new Worker
                {
                    Name = $"{TestWorkerNamePrefix}{i:D5}",
                    EmbeddingCsv = Worker.SerializeEmbedding(RandomNormalizedEmbedding(embeddingDim)),
                    EnrolledAtUtc = DateTime.UtcNow
                });

                if ((i + 1) % 500 == 0)
                {
                    progress?.Report(i + 1);
                }
            }

            Debug.WriteLine($"[TestDataSeeder] Generated {count} random embeddings in {sw.ElapsedMilliseconds} ms — inserting...");

            sw.Restart();
            await _db.RunBulkInsertAsync(workers);
            sw.Stop();

            Debug.WriteLine($"[TestDataSeeder] Bulk-inserted {count} synthetic workers in {sw.ElapsedMilliseconds} ms");
        }

        /// <summary>
        /// Runs every image in `folderPath` through the app's REAL detection + recognition
        /// services and inserts the resulting real embeddings. This is what gives you
        /// meaningful accuracy/threshold numbers at scale — random vectors cannot.
        ///
        /// Pass in the SAME FaceDetectionService/FaceRecognitionService instances the rest
        /// of the app uses (already-loaded models), not fresh ones, so the embeddings you
        /// seed are produced identically to how a real scan would produce them.
        ///
        /// Images with no detected face, or where embedding extraction fails, are skipped
        /// and counted — check the returned SeedResult to see how many actually made it in.
        /// </summary>
        /// <param name="folderPath">Folder of face images, one identity per image (e.g. an LFW subset).</param>
        /// <param name="detector">Already-loaded FaceDetectionService instance.</param>
        /// <param name="recognizer">Already-loaded FaceRecognitionService instance.</param>
        /// <param name="progress">Optional callback invoked after each processed file with the running count.</param>
        public async Task<SeedResult> SeedFromImageFolderAsync(
            string folderPath,
            FaceDetectionService detector,
            FaceRecognitionService recognizer,
            IProgress<int>? progress = null)
        {
            if (!detector.IsLoaded || !recognizer.IsLoaded)
            {
                throw new InvalidOperationException(
                    "Detector/recognizer models must be loaded before seeding real faces. " +
                    "Call LoadModel() on both first (same as OnAppearing does for the real pages).");
            }

            var files = Directory.EnumerateFiles(folderPath, "*.*")
                .Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase)
                         || f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase)
                         || f.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
                .ToList();

            var workers = new List<Worker>();
            int noFace = 0;
            int embeddingFailed = 0;
            int processed = 0;

            var sw = Stopwatch.StartNew();

            foreach (var file in files)
            {
                processed++;
                try
                {
                    using var bmp = SKBitmap.Decode(file);
                    if (bmp == null)
                    {
                        Debug.WriteLine($"[TestDataSeeder] Skipped (decode failed): {file}");
                        continue;
                    }

                    var faces = detector.Detect(bmp);
                    if (faces.Count == 0)
                    {
                        noFace++;
                        continue;
                    }

                    using var aligned = recognizer.AlignFace(bmp, faces[0].Landmarks);
                    var embedding = recognizer.GetEmbedding(aligned);

                    if (embedding == null)
                    {
                        embeddingFailed++;
                        continue;
                    }

                    workers.Add(new Worker
                    {
                        Name = $"{TestWorkerNamePrefix}{Path.GetFileNameWithoutExtension(file)}",
                        EmbeddingCsv = Worker.SerializeEmbedding(embedding),
                        EnrolledAtUtc = DateTime.UtcNow
                    });
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[TestDataSeeder] FAILED on {file}: {ex}");
                }

                if (processed % 50 == 0)
                {
                    progress?.Report(processed);
                }
            }

            sw.Stop();
            Debug.WriteLine(
                $"[TestDataSeeder] Real-face import: {workers.Count} succeeded, " +
                $"{noFace} had no detected face, {embeddingFailed} failed embedding, " +
                $"out of {files.Count} files, in {sw.ElapsedMilliseconds} ms");

            if (workers.Count > 0)
            {
                await _db.RunBulkInsertAsync(workers);
            }

            return new SeedResult(
                TotalFiles: files.Count,
                Imported: workers.Count,
                NoFaceDetected: noFace,
                EmbeddingFailed: embeddingFailed,
                ElapsedMs: sw.ElapsedMilliseconds);
        }

        /// <summary>
        /// Deletes every worker whose name starts with TestWorkerNamePrefix (i.e. everything
        /// this seeder ever created), leaving real enrollments untouched. Run this after a
        /// scale test to clean up before going back to normal use/demoing the app.
        /// </summary>
        public async Task<int> ClearSeededTestDataAsync()
        {
            var all = await _db.GetAllWorkersAsync();
            var testWorkers = all.Where(w => w.Name.StartsWith(TestWorkerNamePrefix)).ToList();

            Debug.WriteLine($"[TestDataSeeder] Clearing {testWorkers.Count} seeded test workers...");

            var sw = Stopwatch.StartNew();
            foreach (var worker in testWorkers)
            {
                await _db.DeleteWorkerAsync(worker);
            }
            sw.Stop();

            Debug.WriteLine($"[TestDataSeeder] Cleared {testWorkers.Count} test workers in {sw.ElapsedMilliseconds} ms");
            return testWorkers.Count;
        }

        private static float[] RandomNormalizedEmbedding(int dim)
        {
            var v = new float[dim];
            float norm = 0f;

            for (int i = 0; i < dim; i++)
            {
                v[i] = (float)(_rng.NextDouble() * 2 - 1);
                norm += v[i] * v[i];
            }

            norm = MathF.Sqrt(norm);
            if (norm > 0f)
            {
                for (int i = 0; i < dim; i++)
                {
                    v[i] /= norm;
                }
            }

            return v;
        }
    }

    /// <summary>
    /// Outcome of a SeedFromImageFolderAsync run, so a caller (dev UI, log, etc.) can report
    /// exactly what happened instead of just a raw count.
    /// </summary>
    public record SeedResult(
        int TotalFiles,
        int Imported,
        int NoFaceDetected,
        int EmbeddingFailed,
        long ElapsedMs);
}