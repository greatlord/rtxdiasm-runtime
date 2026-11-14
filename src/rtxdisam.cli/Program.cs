using System.Text.Json;
using rtxdisam;

namespace rtxdisam.cli;

internal static class Program
{
    private static int Main(string[] args)
    {
        JsonOutput.Configure((value, indented) => JsonSerializer.Serialize(
            value, new JsonSerializerOptions { WriteIndented = indented }));
        return RtxDisamCli.Run(args);
    }
}
