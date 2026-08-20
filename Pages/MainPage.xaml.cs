using FaceAttendanceApp.Model;
using System.Diagnostics;

namespace FaceAttendanceApp
{
    public partial class MainPage : ContentPage
    {
        public MainPage()
        {
            InitializeComponent();
        }

        private async void OnRegisterWorkerClicked(object? sender, EventArgs e)
        {
            await Shell.Current.GoToAsync(nameof(RegisterWorkerPage));
        }

        private async void OnScanAttendanceClicked(object? sender, EventArgs e)
        {
            await Shell.Current.GoToAsync(nameof(ScanAttendancePage));
        }

        private async void OnListWorkersClicked(object? sender, EventArgs e)
        {
            await Shell.Current.GoToAsync(nameof(WorkerListPage));
        }

        private async void OnAttendanceHistoryClicked(object? sender, EventArgs e)
        {
            await Shell.Current.GoToAsync(nameof(AttendanceHistoryPage));
        }

        private async void OnSeedTestDataClicked(object? sender, EventArgs e)
        {
#if DEBUG
            await Shell.Current.GoToAsync(nameof(SeedTestDataPage));
#else
    Debug.WriteLine("[MainPage] Seed test data blocked — not a debug build");
#endif
        }
    }
}