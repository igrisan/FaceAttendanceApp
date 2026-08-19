using SQLite;
using SQLitePCL;

namespace FaceAttendanceApp.Model
{
    public class AttendanceRecord
    {
        [PrimaryKey, AutoIncrement]
        public int Id { get; set; }

        public int WorkerId { get; set; }

        public string WorkerName { get; set; } = string.Empty;

        public DateTime TimestampUtc { get; set; }

        public float MatchScore { get; set; }

        public bool HasHelmet { get; set; }

        public bool HasMask { get; set; }

        // GPS — left nullable for now since capture isn't wired up yet. Populate these once
        // Geolocation.GetLocationAsync() is added to the scan flow.
        public double? Latitude { get; set; }

        public double? Longitude { get; set; }

        // Offline-sync support: false until this record has been successfully pushed to a
        // server. A background/foreground sync routine should query for Synced == false and
        // upload those, then flip this to true — nothing currently sets it besides the default.
        public bool Synced { get; set; } = false;
    }

    public class AttendanceDatabase
    {
        private readonly SQLiteAsyncConnection _connection;

        public AttendanceDatabase(string dbPath)
        {
            Batteries_V2.Init();

            _connection = new SQLiteAsyncConnection(dbPath);
            _connection.CreateTableAsync<AttendanceRecord>().Wait();
        }

        public Task<int> SaveRecordAsync(AttendanceRecord record)
        {
            return _connection.InsertAsync(record);
        }

        public Task<List<AttendanceRecord>> GetAllRecordsAsync()
        {
            return _connection.Table<AttendanceRecord>()
                .OrderByDescending(r => r.TimestampUtc)
                .ToListAsync();
        }

        public Task<List<AttendanceRecord>> GetUnsyncedRecordsAsync()
        {
            return _connection.Table<AttendanceRecord>()
                .Where(r => !r.Synced)
                .ToListAsync();
        }

        /// <summary>
        /// Most recent record for a worker, used for duplicate-scan suppression — so someone
        /// standing in frame across multiple capture cycles doesn't get logged repeatedly.
        /// </summary>
        public async Task<AttendanceRecord?> GetLastRecordForWorkerAsync(int workerId)
        {
            var records = await _connection.Table<AttendanceRecord>()
                .Where(r => r.WorkerId == workerId)
                .OrderByDescending(r => r.TimestampUtc)
                .Take(1)
                .ToListAsync();

            return records.FirstOrDefault();
        }

        public Task<int> MarkSyncedAsync(AttendanceRecord record)
        {
            record.Synced = true;
            return _connection.UpdateAsync(record);
        }
    }
}