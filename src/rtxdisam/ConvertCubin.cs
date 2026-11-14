using System;
using System.IO;
using nvDismDecode;

namespace rtxdisam;

internal static partial class Program
{
    private static int ConvertCubin(string cubinPath, string targetArchitecture, string outputPath, string? irOutput, string? toolkit, string? fatbinOutput, FatbinCompression compression)
    {
        string tempSass = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".sass");
        try
        {
            byte[] data = File.ReadAllBytes(cubinPath);
            CubinFile cubin = CubinFile.Load(data, cubinPath);
            cubin.Toolkit = cubin.ToolkitFromAbi;
            TextWriter oldOut = Console.Out;
            using (StringWriter sw = new StringWriter())
            {
                Console.SetOut(sw);
                try
                {
                    UseCubinArchitecture(cubin);
                    if (!WriteCubinText(cubin, null, null, false))
                    {
                        Console.SetOut(oldOut);
                        throw new InvalidOperationException("Failed to disassemble");
                    }
                    string disasm = sw.ToString();
                    Console.SetOut(oldOut);
                    File.WriteAllText(tempSass, disasm);
                }
                catch
                {
                    Console.SetOut(oldOut);
                    throw;
                }
            }
            SassAssembler.Convert(tempSass, cubin.ArchitectureKey, targetArchitecture, outputPath, irOutput, toolkit);
            if (fatbinOutput != null)
            {
                CubinFile assembled = CubinFile.Load(outputPath);
                FatbinWriter.Write(fatbinOutput, File.ReadAllBytes(outputPath), assembled.Architecture, compression);
            }
            Console.WriteLine(outputPath);
            return 0;
        }
        finally
        {
            try { if (File.Exists(tempSass)) File.Delete(tempSass); } catch { }
        }
    }
}
