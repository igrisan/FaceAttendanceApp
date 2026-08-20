#if DEBUG
using FaceAttendanceApp.DevTools;
using FaceAttendanceApp.Model;
using FaceAttendanceApp.Services;
using System.Diagnostics;

namespace FaceAttendanceApp
{
    /// <summary>
    /// DEV-ONLY page for triggering TestDataSeeder from the UI during scale testing.
    /// Compiled ONLY in DEBUG builds — the #if DEBUG around the whole file means this
    /// class doesn't exist at all in a Release build, so there's no way to accidentally
    /// reach it (or even reference it) in production.
    ///
    /// You still need a minimal .xaml for this to bind to (a ContentPage with a
    /// StatusLabel, three buttons, and an Entry for the image folder path is enough —
    /// wire it up in the designer or just build the same page in code-behind).
    ///
    /// Navigate here manually during testing, e.g. temporarily add one debug-only
    /// button on a settings page:
    ///   #if DEBUG
    ///   await Navigation.PushAsync(new SeedTestDataPage());
    ///   #endif
    /// </summary>
    public partial class SeedTestDataPage : ContentPage
    {
        private readonly FaceDetectionService _detectionService = new();
        private readonly FaceRecognitionService _recognitionService = new();
        private WorkerDatabase? _database;
        private TestDataSeeder? _seeder;

        // Confirm this matches what your FaceRecognitionService.GetEmbedding() actually
        // returns — log embedding.Length once from a real capture before trusting this.
        private const int EmbeddingDim = 128;

        public SeedTestDataPage()
        {
            InitializeComponent();
        }

        protected override async void OnAppearing()
        {
            base.OnAppearing();

            StatusLabel.Text = "Loading models...";

            var dbPath = Path.Combine(FileSystem.AppDataDirectory, "faceattendance.db3");
            _database = new WorkerDatabase(dbPath);
            _seeder = new TestDataSeeder(_database);

            await Task.Run(_detectionService.LoadModel);
            await Task.Run(_recognitionService.LoadModel);

            var count = (await _database.GetAllWorkersAsync()).Count;
            StatusLabel.Text = $"Ready. Current worker count: {count}";
        }

        private async void OnSeedRandomClicked(object? sender, EventArgs e)
        {
            if (_seeder == null) return;

            int count = 9950;
            if (int.TryParse(RandomCountEntry.Text, out var parsed))
            {
                count = parsed;
            }

            StatusLabel.Text = $"Seeding {count} random workers...";
            SeedRandomBtn.IsEnabled = false;

            var progress = new Progress<int>(n =>
            {
                MainThread.BeginInvokeOnMainThread(() => StatusLabel.Text = $"Seeded {n}/{count}...");
            });

            var sw = Stopwatch.StartNew();
            await _seeder.SeedRandomWorkersAsync(count, EmbeddingDim, progress);
            sw.Stop();

            var total = (await _database!.GetAllWorkersAsync()).Count;
            StatusLabel.Text = $"Done — inserted {count} in {sw.ElapsedMilliseconds} ms. Total workers now: {total}";
            SeedRandomBtn.IsEnabled = true;
        }

        private async void OnSeedRealFacesClicked(object? sender, EventArgs e)
        {
            if (_seeder == null) return;

            var folder = RealFacesFolderEntry.Text?.Trim();
            if (string.IsNullOrWhiteSpace(folder) || !Directory.Exists(folder))
            {
                await DisplayAlert("Invalid folder", "Enter a valid folder path containing face images.", "OK");
                return;
            }

            StatusLabel.Text = "Importing real faces...";
            SeedRealFacesBtn.IsEnabled = false;

            var progress = new Progress<int>(n =>
            {
                MainThread.BeginInvokeOnMainThread(() => StatusLabel.Text = $"Processed {n} files...");
            });

            var result = await _seeder.SeedFromImageFolderAsync(
                folder, _detectionService, _recognitionService, progress);

            StatusLabel.Text =
                $"Done — imported {result.Imported}/{result.TotalFiles} " +
                $"({result.NoFaceDetected} no face, {result.EmbeddingFailed} embedding failed) " +
                $"in {result.ElapsedMs} ms";
            SeedRealFacesBtn.IsEnabled = true;
        }

        private async void OnClearTestDataClicked(object? sender, EventArgs e)
        {
            if (_seeder == null) return;

            bool confirm = await DisplayAlert(
                "Clear test data",
                "This deletes every worker whose name starts with 'TestWorker_'. Real enrollments are not touched. Continue?",
                "Yes, clear", "Cancel");

            if (!confirm) return;

            StatusLabel.Text = "Clearing...";
            var cleared = await _seeder.ClearSeededTestDataAsync();

            var total = (await _database!.GetAllWorkersAsync()).Count;
            StatusLabel.Text = $"Cleared {cleared} test workers. Total workers now: {total}";
        }
    }
}
#endif