using System.Diagnostics;
using FaceAttendanceApp.Model;

namespace FaceAttendanceApp
{
    public class WorkerListItem
    {
        public int Id { get; set; }
        public string Name { get; set; } = string.Empty;
        public string IdDisplay => $"Worker ID: {Id}";

        // Replaces the old single HasMaskedReference flag — a worker can now have any number
        // of registered variants (masked, spectacles, any combo), so we show a count instead.
        public int VariantCount { get; set; }
        public bool HasVariants => VariantCount > 0;
        public string VariantSummary => VariantCount == 1
            ? "1 additional face variant"
            : $"{VariantCount} additional face variants";
    }

    public partial class WorkerListPage : ContentPage
    {
        private WorkerDatabase? _database;

        public WorkerListPage()
        {
            InitializeComponent();
        }

        protected override async void OnAppearing()
        {
            base.OnAppearing();
            await LoadWorkersAsync();
        }

        private async Task LoadWorkersAsync()
        {
            try
            {
                var dbPath = Path.Combine(FileSystem.AppDataDirectory, "faceattendance.db3");
                _database = new WorkerDatabase(dbPath);

                var workers = await _database.GetAllWorkersAsync();

                // One query for every worker's variants, grouped by worker id — avoids N+1
                // queries when building the list (matters once this scales to 5,000 workers).
                var referencesByWorker = await _database.GetAllFaceReferencesGroupedAsync();

                var items = workers
                    .OrderBy(w => w.Name)
                    .Select(w => new WorkerListItem
                    {
                        Id = w.Id,
                        Name = w.Name,
                        VariantCount = referencesByWorker.TryGetValue(w.Id, out var refs) ? refs.Count : 0
                    })
                    .ToList();

                WorkerCollectionView.ItemsSource = items;

                WorkerCountLabel.Text = items.Count == 1
                    ? "1 worker enrolled"
                    : $"{items.Count} workers enrolled";

                Debug.WriteLine($"[WorkerListPage] Loaded {items.Count} workers");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[WorkerListPage] Load FAILED: {ex}");
                WorkerCountLabel.Text = "Failed to load workers";
            }
            finally
            {
                ListRefreshView.IsRefreshing = false;
            }
        }

        private async void OnRefreshing(object? sender, EventArgs e)
        {
            await LoadWorkersAsync();
        }

        /// <summary>
        /// Single "⋮" icon per card opens a native action sheet with Edit/Delete choices,
        /// instead of showing multiple always-visible icon buttons.
        /// </summary>
        private async void OnMoreOptionsClicked(object? sender, EventArgs e)
        {
            if (sender is not Button { CommandParameter: int workerId } || _database == null)
            {
                return;
            }

            var workers = await _database.GetAllWorkersAsync();
            var worker = workers.FirstOrDefault(w => w.Id == workerId);

            if (worker == null)
            {
                return;
            }

            string action = await DisplayActionSheet(worker.Name, "Cancel", null, "Edit", "Delete");

            switch (action)
            {
                case "Edit":
                    await Shell.Current.GoToAsync($"{nameof(WorkerEditPage)}?WorkerId={workerId}");
                    break;

                case "Delete":
                    bool confirmDelete = await DisplayAlert("Delete Worker", $"Delete {worker.Name}? This cannot be undone.", "Delete", "Cancel");

                    if (confirmDelete)
                    {
                        await _database.DeleteWorkerAsync(worker);
                        Debug.WriteLine($"[WorkerListPage] Worker deleted — id={worker.Id}, name={worker.Name}");
                        await LoadWorkersAsync();
                    }
                    break;
            }
        }
    }
}