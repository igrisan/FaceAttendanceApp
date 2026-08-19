using System.Collections.ObjectModel;
using System.Diagnostics;
using FaceAttendanceApp.Model;

namespace FaceAttendanceApp
{
    /// <summary>
    /// Display-friendly wrapper around AttendanceRecord for binding in the CollectionView.
    /// Keeps formatting logic (icons, date strings) out of the XAML.
    /// </summary>
    public class AttendanceRecordDisplay
    {
        public int Id { get; set; }
        public int WorkerId { get; set; }
        public string WorkerName { get; set; } = string.Empty;

        // Falls back to "Worker #<id>" if WorkerName is somehow blank, so the card never
        // shows nothing where a name should be.
        public string WorkerNameDisplay => string.IsNullOrWhiteSpace(WorkerName) ? $"Worker #{WorkerId}" : WorkerName;

        public DateTime TimestampUtc { get; set; }
        public string TimestampDisplay => TimestampUtc.ToLocalTime().ToString("ddd, dd MMM yyyy — HH:mm:ss");
        public string DayGroupKey => TimestampUtc.ToLocalTime().ToString("dddd, dd MMM yyyy");
        public float MatchScore { get; set; }
        public string ScoreDisplay => MatchScore > 0 ? $"{MatchScore:P0}" : "";
        public bool HasHelmet { get; set; }
        public bool HasMask { get; set; }
        public bool Synced { get; set; }
    }

    /// <summary>
    /// Group wrapper CollectionView needs for IsGrouped="True" binding — GroupName is shown in
    /// the header template, and the collection itself supplies the items in that group.
    /// </summary>
    public class AttendanceGroup : ObservableCollection<AttendanceRecordDisplay>
    {
        public string GroupName { get; }

        public AttendanceGroup(string groupName, IEnumerable<AttendanceRecordDisplay> items) : base(items)
        {
            GroupName = groupName;
        }
    }

    public partial class AttendanceHistoryPage : ContentPage
    {
        private AttendanceDatabase? _attendanceDatabase;
        private List<AttendanceRecordDisplay> _allRecords = new();

        public AttendanceHistoryPage()
        {
            InitializeComponent();
            GroupByPicker.SelectedIndex = 0; // Default to "Day"
        }

        protected override async void OnAppearing()
        {
            base.OnAppearing();
            await LoadRecordsAsync();
        }

        private async Task LoadRecordsAsync()
        {
            try
            {
                var dbPath = Path.Combine(FileSystem.AppDataDirectory, "attendance.db3");
                _attendanceDatabase = new AttendanceDatabase(dbPath);

                var records = await _attendanceDatabase.GetAllRecordsAsync();

                _allRecords = records.Select(r => new AttendanceRecordDisplay
                {
                    Id = r.Id,
                    WorkerId = r.WorkerId,
                    WorkerName = r.WorkerName,
                    TimestampUtc = r.TimestampUtc,
                    MatchScore = r.MatchScore,
                    HasHelmet = r.HasHelmet,
                    HasMask = r.HasMask,
                    Synced = r.Synced
                }).ToList();

                // Debug check — confirms whether WorkerName is actually populated in the DB.
                foreach (var r in _allRecords.Take(5))
                {
                    Debug.WriteLine($"[AttendanceHistoryPage] Record id={r.Id} workerId={r.WorkerId} workerName='{r.WorkerName}'");
                }

                RecordCountLabel.Text = _allRecords.Count == 1
                    ? "1 record"
                    : $"{_allRecords.Count} records";

                ApplyGrouping();

                Debug.WriteLine($"[AttendanceHistoryPage] Loaded {_allRecords.Count} attendance records");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[AttendanceHistoryPage] Load FAILED: {ex}");
                RecordCountLabel.Text = "Failed to load records";
            }
            finally
            {
                HistoryRefreshView.IsRefreshing = false;
            }
        }

        private void ApplyGrouping()
        {
            var mode = GroupByPicker.SelectedItem as string ?? "Day";

            IEnumerable<AttendanceGroup> grouped;

            switch (mode)
            {
                case "Worker Name":
                    grouped = _allRecords
                        .GroupBy(r => r.WorkerNameDisplay)
                        .OrderBy(g => g.Key)
                        .Select(g => new AttendanceGroup(g.Key, g.OrderByDescending(r => r.TimestampUtc)));
                    break;

                case "None (flat list)":
                    grouped = new List<AttendanceGroup>
                    {
                        new AttendanceGroup("All Records", _allRecords.OrderByDescending(r => r.TimestampUtc))
                    };
                    break;

                case "Day":
                default:
                    grouped = _allRecords
                        .GroupBy(r => r.DayGroupKey)
                        .Select(g => new AttendanceGroup(g.Key, g.OrderByDescending(r => r.TimestampUtc)))
                        .OrderByDescending(g => g.Max(r => r.TimestampUtc));
                    break;
            }

            HistoryCollectionView.ItemsSource = grouped.ToList();
        }

        private void OnGroupByChanged(object? sender, EventArgs e)
        {
            if (_allRecords.Count > 0 || GroupByPicker.SelectedItem != null)
            {
                ApplyGrouping();
            }
        }

        private async void OnRefreshing(object? sender, EventArgs e)
        {
            await LoadRecordsAsync();
        }

        private async void OnBackClicked(object? sender, EventArgs e)
        {
            await Shell.Current.GoToAsync("..");
        }
    }
}