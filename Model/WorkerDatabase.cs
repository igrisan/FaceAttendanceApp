using SQLite;
using SQLitePCL;

namespace FaceAttendanceApp.Model
{
    public class Worker
    {
        [PrimaryKey, AutoIncrement]
        public int Id { get; set; }

        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// The primary/bare-face embedding, captured during initial enrollment. This one is
        /// always required. Additional variants (masked, spectacles, combos, etc.) live in the
        /// WorkerFaceReference table instead of as fixed columns here, so a worker can register
        /// as many looks as they actually need rather than being limited to 1-2 hardcoded types.
        /// </summary>
        public string EmbeddingCsv { get; set; } = string.Empty;

        public DateTime EnrolledAtUtc { get; set; }

        // Parsed-embedding cache for the primary embedding. See GetEmbedding() below for why
        // this exists — same reasoning as before, just no longer duplicated per-variant here
        // since variants are cached on the WorkerFaceReference objects themselves now.
        private float[]? _cachedEmbedding;

        public float[] GetEmbedding()
        {
            _cachedEmbedding ??= EmbeddingCsv
                .Split(',', StringSplitOptions.RemoveEmptyEntries)
                .Select(float.Parse)
                .ToArray();
            return _cachedEmbedding;
        }

        /// <summary>
        /// Call this if EmbeddingCsv changes after the cache has already been populated (e.g.
        /// right after re-capturing the primary reference for this same in-memory instance).
        /// Not needed for normal load-from-DB usage since a fresh Worker instance has an empty
        /// cache by default.
        /// </summary>
        public void InvalidateEmbeddingCache()
        {
            _cachedEmbedding = null;
        }

        public static string SerializeEmbedding(float[] embedding) =>
            string.Join(',', embedding);
    }

    /// <summary>
    /// One additional face reference for a worker, beyond their primary bare-face embedding.
    /// A worker can have any number of these — e.g. "With Mask", "With Spectacles",
    /// "With Mask + Spectacles" — registered whenever that look is relevant for them.
    /// Matching at scan time compares the captured face against ALL of a worker's references
    /// (primary + every WorkerFaceReference row) and takes whichever scores highest.
    /// </summary>
    public class WorkerFaceReference
    {
        [PrimaryKey, AutoIncrement]
        public int Id { get; set; }

        [Indexed]
        public int WorkerId { get; set; }

        /// <summary>
        /// Free-text label shown in the UI, e.g. "With Mask", "With Spectacles",
        /// "With Mask + Spectacles". Not an enum, so new combos don't need a code change.
        /// </summary>
        public string Label { get; set; } = string.Empty;

        public string EmbeddingCsv { get; set; } = string.Empty;

        public DateTime CreatedAtUtc { get; set; }

        // Parsed-embedding cache, same reasoning as Worker.GetEmbedding() — this object is
        // expected to live in memory across many scan comparisons rather than being re-parsed
        // from CSV on every single frame.
        private float[]? _cachedEmbedding;

        public float[] GetEmbedding()
        {
            _cachedEmbedding ??= EmbeddingCsv
                .Split(',', StringSplitOptions.RemoveEmptyEntries)
                .Select(float.Parse)
                .ToArray();
            return _cachedEmbedding;
        }

        public void InvalidateEmbeddingCache()
        {
            _cachedEmbedding = null;
        }
    }

    /// <summary>
    /// Convenience bundle used by scan matching: a worker plus every embedding available for
    /// them (primary + all registered variants), each tagged with a label so the scan result
    /// can show which variant matched (e.g. "ashraff — matched via 'With Mask'").
    /// </summary>
    public record WorkerEmbeddingCandidate(Worker Worker, string Label, float[] Embedding);

    public class WorkerDatabase
    {
        private readonly SQLiteAsyncConnection _connection;

        public WorkerDatabase(string dbPath)
        {
            Batteries_V2.Init();

            _connection = new SQLiteAsyncConnection(dbPath);
            _connection.CreateTableAsync<Worker>().Wait();
            _connection.CreateTableAsync<WorkerFaceReference>().Wait();
        }

        public async Task<int> SaveWorkerAsync(Worker worker)
        {
            worker.EnrolledAtUtc = DateTime.UtcNow;
            await _connection.InsertAsync(worker);
            return worker.Id;
        }

        public Task<int> UpdateWorkerAsync(Worker worker)
        {
            worker.InvalidateEmbeddingCache();
            return _connection.UpdateAsync(worker);
        }

        public Task<List<Worker>> GetAllWorkersAsync()
        {
            return _connection.Table<Worker>().ToListAsync();
        }

        public Task<int> DeleteWorkerAsync(Worker worker)
        {
            // Face references are deleted alongside the worker so we don't leave orphaned rows
            // behind — SQLite-net doesn't cascade-delete automatically, so this is explicit.
            return DeleteWorkerAndReferencesAsync(worker);
        }

        private async Task<int> DeleteWorkerAndReferencesAsync(Worker worker)
        {
            var references = await GetFaceReferencesAsync(worker.Id);
            foreach (var reference in references)
            {
                await _connection.DeleteAsync(reference);
            }
            return await _connection.DeleteAsync(worker);
        }

        /// <summary>
        /// All additional face-reference variants for one worker (e.g. "With Mask",
        /// "With Spectacles"). Does NOT include the worker's primary bare-face embedding —
        /// that one lives on the Worker object itself via GetEmbedding().
        /// </summary>
        public Task<List<WorkerFaceReference>> GetFaceReferencesAsync(int workerId)
        {
            return _connection.Table<WorkerFaceReference>()
                .Where(r => r.WorkerId == workerId)
                .ToListAsync();
        }

        /// <summary>
        /// All face references for ALL workers in one query, grouped by worker. Used when
        /// building the full in-memory candidate list for scan matching, so we don't run one
        /// query per worker (which would be slow at 5,000+ workers).
        /// </summary>
        public async Task<Dictionary<int, List<WorkerFaceReference>>> GetAllFaceReferencesGroupedAsync()
        {
            var all = await _connection.Table<WorkerFaceReference>().ToListAsync();
            return all.GroupBy(r => r.WorkerId)
                .ToDictionary(g => g.Key, g => g.ToList());
        }

        public async Task<int> AddFaceReferenceAsync(int workerId, string label, float[] embedding)
        {
            var reference = new WorkerFaceReference
            {
                WorkerId = workerId,
                Label = label,
                EmbeddingCsv = Worker.SerializeEmbedding(embedding),
                CreatedAtUtc = DateTime.UtcNow
            };
            await _connection.InsertAsync(reference);
            return reference.Id;
        }

        public Task<int> DeleteFaceReferenceAsync(WorkerFaceReference reference)
        {
            return _connection.DeleteAsync(reference);
        }

        public Task<int> UpdateFaceReferenceAsync(WorkerFaceReference reference)
        {
            reference.InvalidateEmbeddingCache();
            return _connection.UpdateAsync(reference);
        }

        /// <summary>
        /// Builds the full candidate list for scan matching: every worker's primary embedding
        /// PLUS every registered variant, each tagged with a label. Call this once when
        /// refreshing the in-memory worker cache (same place GetAllWorkersAsync() is currently
        /// called from in ScanAttendancePage), not per-frame.
        /// </summary>
        public async Task<List<WorkerEmbeddingCandidate>> GetAllMatchCandidatesAsync()
        {
            var workers = await GetAllWorkersAsync();
            var referencesByWorker = await GetAllFaceReferencesGroupedAsync();

            var candidates = new List<WorkerEmbeddingCandidate>();
            foreach (var worker in workers)
            {
                candidates.Add(new WorkerEmbeddingCandidate(worker, "Bare Face", worker.GetEmbedding()));

                if (referencesByWorker.TryGetValue(worker.Id, out var references))
                {
                    foreach (var reference in references)
                    {
                        candidates.Add(new WorkerEmbeddingCandidate(worker, reference.Label, reference.GetEmbedding()));
                    }
                }
            }
            return candidates;
        }
    }
}