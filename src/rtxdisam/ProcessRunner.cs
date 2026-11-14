using System.Diagnostics;

namespace rtxdisam;

/// <summary>
/// Small helper for running an external executable and capturing both
/// stdout and stderr without deadlocking on the child process pipes.
/// </summary>
public class ProcessRunner
{
    public struct ExecResult
    {
        public string Output;
        public string Error;
        public int ExitCode;
    }

    public async Task<ExecResult> ExecAll(
        string fileName,
        string arguments)
    {
        using var process = new Process();

        process.StartInfo = new ProcessStartInfo
        {
            FileName = fileName,
            Arguments = arguments,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        if (!process.Start())
        {
            throw new InvalidOperationException(
                $"Failed to start process: {fileName}");
        }

        // Read both streams concurrently to prevent deadlocks.
        Task<string> outputTask =
            process.StandardOutput.ReadToEndAsync();

        Task<string> errorTask =
            process.StandardError.ReadToEndAsync();

        await process.WaitForExitAsync();

        return new ExecResult
        {
            Output = await outputTask,
            Error = await errorTask,
            ExitCode = process.ExitCode
        };
    }
}
