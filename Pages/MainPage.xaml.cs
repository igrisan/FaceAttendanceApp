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
    }
}