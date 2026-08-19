using FaceAttendanceApp;

namespace FaceAttendanceApp
{
    public partial class AppShell : Shell
    {
        public AppShell()
        {
            InitializeComponent();

            Routing.RegisterRoute(nameof(RegisterWorkerPage), typeof(RegisterWorkerPage));
            Routing.RegisterRoute(nameof(ScanAttendancePage), typeof(ScanAttendancePage));
            Routing.RegisterRoute(nameof(AttendanceHistoryPage), typeof(AttendanceHistoryPage));
            Routing.RegisterRoute(nameof(WorkerListPage), typeof(WorkerListPage));
            Routing.RegisterRoute(nameof(WorkerEditPage), typeof(WorkerEditPage));
        }
    }
}