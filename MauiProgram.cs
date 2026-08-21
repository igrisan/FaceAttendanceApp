using Microsoft.Extensions.Logging;
using CommunityToolkit.Maui;
using Syncfusion.Maui.Toolkit.Hosting;
using FaceAttendanceApp.Services;

namespace FaceAttendanceApp
{
    public static class MauiProgram
    {
        public static MauiApp CreateMauiApp()
        {
            var builder = MauiApp.CreateBuilder();
            builder
                .UseMauiApp<App>()
                .UseMauiCommunityToolkit()
                .UseMauiCommunityToolkitCamera()
                .ConfigureSyncfusionToolkit()
                .ConfigureFonts(fonts =>
                {
                    fonts.AddFont("OpenSans-Regular.ttf", "OpenSansRegular");
                    fonts.AddFont("OpenSans-Semibold.ttf", "OpenSansSemibold");
                });

            builder.Services.AddSingleton<FaceDetectionService>();
            builder.Services.AddSingleton<FaceRecognitionService>();
            builder.Services.AddSingleton<MaskHelmetDetectionService>();
            builder.Services.AddSingleton<LivenessService>();

            builder.Services.AddTransient<RegisterWorkerPage>();
            builder.Services.AddTransient<ScanAttendancePage>();
            builder.Services.AddTransient<WorkerListPage>();
            builder.Services.AddTransient<WorkerEditPage>();
            builder.Services.AddTransient<AttendanceHistoryPage>();

#if DEBUG
            builder.Logging.AddDebug();
#endif

            return builder.Build();
        }
    }
}