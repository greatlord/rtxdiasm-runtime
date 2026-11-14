using System.Buffers.Binary;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using nvDismDecode;

namespace rtxdisam;

internal static partial class Program
{
    private const string Version = "2.3.0";

    private static bool _noDataflow;
    private static bool _preserveRelocations;
    private static HashSet<uint>? _functionIndexes;
    private static FatbinCompression _fatbinCompression = FatbinCompression.None;
    private static string? _toolkit;

    internal static int Run(string[] args)
    {
        string? binaryPath = null;
        string? cubinPath = null;
        string? hexInstruction = null;
        string? asmPath = null;
        string? asmOutput = null;
        string? asmArchitecture = null;
        string? irOutput = null;
        string? fatbinOutput = null;
        ulong baseAddress = 0;
        string binaryArchitecture = "sm_86";
        bool emitJson = false;
        bool nvJson = false;
    bool annotateJson = false;
        bool showFields = false;
        bool extractPtx = false;
        bool extractCubin = false;
        bool extractAll = false;
        bool showEncoding = false;
        bool listOnly = false;
        string? functionFilter = null;
        string? outputDir = null;
        bool force = false;
        string? liftPtxPath = null;
        string? liftPtxEntry = null;
        string? liftPtxOutput = null;
        var liftPtxParams = new List<PtxParameter>();
        bool liftPtxFunctions = false;
        string? runCubinPath = null;
        string? runPtxPath = null;
        var runBuffers = new List<string>();
        var runBufferHexes = new List<string>();
        var runScalars = new List<string>();
        string? runBlock = null;
        bool runNoBuffer = false;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {

                case "--binary":
                case "-b":
                    if (++i >= args.Length)
                    {
                        Console.Error.WriteLine("--binary requires a file path.");
                        return 1;
                    }
                    // Accept nvdisasm's "--binary <arch> <file>"
                    // spelling as well as "--binary <file>".
                    if (IsArchitectureToken(args[i]))
                    {
                        if (!TryParseArchitecture(args[i], out binaryArchitecture))
                        {
                            Console.Error.WriteLine(
                                $"Unsupported architecture '{args[i]}'; " +
                                "this disassembler supports SM30, SM32, SM35, SM37, SM50, SM52, SM53, " +
                                "SM60, SM61, SM62, SM70, SM72, SM75, SM80, SM86, SM87, " +
                                "SM88, SM89, SM90, SM90a, SM100, SM101, SM103, " +
                                "SM107, SM110, SM120 and SM121.");
                            return 1;
                        }
                        if (++i >= args.Length)
                        {
                            Console.Error.WriteLine(
                                "--binary requires a file path.");
                            return 1;
                        }
                    }
                    binaryPath = args[i];
                    break;

                case "-asmfile":
                case "--asmfile":
                    if (++i >= args.Length)
                    {
                        Console.Error.WriteLine("-asmfile requires a SASS file path.");
                        return 1;
                    }
                    asmPath = args[i];
                    break;

                case "-arch":
                case "--arch":
                case "-convert-to":
                case "--convert-to":
                case "-target":
                case "--target":
                case "-to":
                    if (++i >= args.Length)
                    {
                        Console.Error.WriteLine("-arch/--convert-to requires an architecture.");
                        return 1;
                    }
                    if (!TryParseArchitecture(args[i], out string parsedArch))
                    {
                        Console.Error.WriteLine(
                            $"Unsupported architecture '{args[i]}'.");
                        return 1;
                    }
                    asmArchitecture = parsedArch;
                    break;

                case "--toolkit":
                    if (++i >= args.Length ||
                        !InstructionDecoder.TryNormalizeToolkit(
                            args[i], out string parsedToolkit))
                    {
                        Console.Error.WriteLine(
                            "--toolkit requires 10.2 or 13.4.");
                        return 1;
                    }
                    _toolkit = parsedToolkit;
                    break;

                case "-emit-ir":
                case "--emit-ir":
                    if (++i >= args.Length)
                    {
                        Console.Error.WriteLine("-emit-ir requires an output path.");
                        return 1;
                    }
                    irOutput = args[i];
                    break;

                case "-emit-fatbin":
                case "--emit-fatbin":
                    if (++i >= args.Length)
                    {
                        Console.Error.WriteLine("-emit-fatbin requires an output path.");
                        return 1;
                    }
                    fatbinOutput = args[i];
                    break;

                case "--fatbin-compression":
                    if (++i >= args.Length ||
                        !TryParseFatbinCompression(
                            args[i], out _fatbinCompression))
                    {
                        Console.Error.WriteLine(
                            "--fatbin-compression requires none, deflate, lz4, lz4v2 or zstd.");
                        return 1;
                    }
                    break;

                case "-o":
                case "--output":
                    if (++i >= args.Length)
                    {
                        Console.Error.WriteLine("-o requires an output path.");
                        return 1;
                    }
                    asmOutput = args[i];
                    break;

                case "--cubin":
                case "--fatbin":
                    if (++i >= args.Length)
                    {
                        Console.Error.WriteLine($"{args[i - 1]} requires a file path.");
                        return 1;
                    }
                    cubinPath = args[i];
                    break;

                case "--base-address":
                case "-base":
                    if (++i >= args.Length || !TryParseU64(args[i], out baseAddress))
                    {
                        Console.Error.WriteLine("--base-address requires a hexadecimal value.");
                        return 1;
                    }
                    break;

                case "--emit-json":
                case "-json":
                    emitJson = true;
                    break;

                case "--nv-json":
                    nvJson = true;
                    break;

                case "--annotate":
                case "-annotate":
                    annotateJson = true;
                    break;

                case "--fields":
                    showFields = true;
                    break;

                case "--print-instruction-encoding":
                case "--hex":
                case "-hex":
                    showEncoding = true;
                    break;

                // nvdisasm compatibility no-ops: rtxdisam always emits plain
                // code sections.
                case "--print-code":
                case "-c":
                case "--no-dataflow":
                case "-ndf":
                    _noDataflow = true;
                    break;

                case "--no-vliw":
                case "-novliw":
                    break;

                case "--annotate-spills":
                    // SpillRefill annotations are emitted by default; keep
                    // the flag as a compatibility alias.
                    break;

                case "--annotate-brx":
                    // BRX annotations are emitted by default; keep the flag
                    // as a compatibility alias.
                    break;

                case "--not-nvdisasm-compatible":
                case "-not-nvdisasm-compatible":
                case "--preserve-relocations":
                    // Emit relocation metadata that nvdisasm's text syntax
                    // cannot represent directly.  The resulting SASS is
                    // reassemblable but not an nvdisasm-compatible listing.
                    _preserveRelocations = true;
                    break;

                case "--list":
                case "-list":
                    listOnly = true;
                    break;

                case "--function":
                case "-fun":
                    if (++i >= args.Length)
                    {
                        Console.Error.WriteLine(
                            "--function requires a name or symbol index list.");
                        return 1;
                    }
                    if (TryParseFunctionIndexes(
                            args[i], out HashSet<uint>? indexes))
                    {
                        _functionIndexes = indexes;
                        functionFilter = null;
                    }
                    else
                    {
                        _functionIndexes = null;
                        functionFilter = args[i];
                    }
                    break;

                case "-extract-ptx":
                case "--extract-ptx":
                    extractPtx = true;
                    break;

                case "-extract-cubin":
                case "--extract-cubin":
                    extractCubin = true;
                    break;

                case "-extract-all":
                case "--extract-all":
                    extractAll = true;
                    break;

                case "--output-dir":
                    if (++i >= args.Length)
                    {
                        Console.Error.WriteLine("--output-dir requires a directory.");
                        return 1;
                    }
                    outputDir = args[i];
                    break;

                case "--force":
                    force = true;
                    break;

                case "--lift-ptx":
                    if (++i >= args.Length)
                    {
                        Console.Error.WriteLine("--lift-ptx requires a SASS file path.");
                        return 1;
                    }
                    liftPtxPath = args[i];
                    break;

                case "--ptx-entry":
                    if (++i >= args.Length)
                    {
                        Console.Error.WriteLine("--ptx-entry requires a name.");
                        return 1;
                    }
                    liftPtxEntry = args[i];
                    break;

                case "--ptx-output":
                    if (++i >= args.Length)
                    {
                        Console.Error.WriteLine("--ptx-output requires a path.");
                        return 1;
                    }
                    liftPtxOutput = args[i];
                    break;

                case "--ptx-param":
                    if (++i >= args.Length ||
                        !TryParsePtxParameter(args[i], out PtxParameter? ptxParam))
                    {
                        Console.Error.WriteLine(
                            "--ptx-param expects name:type:offset (type u32/s32/u64).");
                        return 1;
                    }
                    liftPtxParams.Add(ptxParam!);
                    break;

                case "--ptx-functions":
                    liftPtxFunctions = true;
                    break;

                case "--run-cubin":
                    if (++i >= args.Length)
                    {
                        Console.Error.WriteLine("--run-cubin requires a cubin path.");
                        return 1;
                    }
                    runCubinPath = args[i];
                    break;

                case "--run-ptx":
                    if (++i >= args.Length)
                    {
                        Console.Error.WriteLine("--run-ptx requires a PTX path.");
                        return 1;
                    }
                    runPtxPath = args[i];
                    break;

                case "--run-buffer":
                    if (++i >= args.Length)
                    {
                        Console.Error.WriteLine(
                            "--run-buffer requires size[:initial].");
                        return 1;
                    }
                    runBuffers.Add(args[i]);
                    break;

                case "--run-buffer-hex":
                    if (++i >= args.Length)
                    {
                        Console.Error.WriteLine(
                            "--run-buffer-hex requires size:hex.");
                        return 1;
                    }
                    runBufferHexes.Add(args[i]);
                    break;

                case "--run-block":
                    if (++i >= args.Length)
                    {
                        Console.Error.WriteLine("--run-block requires a size.");
                        return 1;
                    }
                    runBlock = args[i];
                    break;

                case "--run-no-buffer":
                    runNoBuffer = true;
                    break;

                case "--run-scalar":
                    if (++i >= args.Length)
                    {
                        Console.Error.WriteLine("--run-scalar requires a u32.");
                        return 1;
                    }
                    runScalars.Add(args[i]);
                    break;

                case "--version":
                case "-V":
                    Console.WriteLine($"rtxdisam {Version} (SM30/SM32/SM35/SM37/SM50/SM52/SM53/SM60/SM61/SM62/SM70/SM72/SM75/SM80/SM86/SM87/SM88/SM89/SM90/SM90a/SM100/SM101/SM103/SM107/SM110/SM120/SM121)");
                    return 0;

                case "--help":
                case "-h":
                    PrintHelp();
                    return 0;

                default:
                    if (args[i].StartsWith("-", StringComparison.Ordinal))
                    {
                        Console.Error.WriteLine($"Unknown argument: {args[i]}");
                        return 1;
                    }
                    if (hexInstruction == null)
                        hexInstruction = args[i];
                    else
                    {
                        Console.Error.WriteLine($"Unknown argument: {args[i]}");
                        return 1;
                    }
                    break;
            }
        }

        if (annotateJson && !nvJson)
        {
            Console.Error.WriteLine(
                "--annotate requires --nv-json for nvdisasm-shaped output.");
            return 1;
        }

        // Validate explicit command-line target selection before parsing or
        // writing any source/output files.
        if (asmPath != null && asmArchitecture != null &&
            !ValidateTarget(asmArchitecture, _toolkit, out string asmTargetError))
        {
            Console.Error.WriteLine(asmTargetError);
            return 1;
        }
        if (binaryPath != null &&
            !ValidateTarget(
                binaryArchitecture, _toolkit, out string binaryTargetError))
        {
            Console.Error.WriteLine(binaryTargetError);
            return 1;
        }

        if (listOnly)
        {
            string? listInput = cubinPath;
            if (listInput == null && hexInstruction != null &&
                File.Exists(hexInstruction) && !LooksLikeHex(hexInstruction))
                listInput = hexInstruction;

            if (listInput == null)
            {
                Console.Error.WriteLine("--list requires a cubin or fatbin file path.");
                return 1;
            }

            return ListInput(listInput, emitJson);
        }

        if (extractAll)
        {
            string? allInput = cubinPath;
            if (allInput == null && hexInstruction != null &&
                File.Exists(hexInstruction) && !LooksLikeHex(hexInstruction))
                allInput = hexInstruction;

            if (allInput == null)
            {
                Console.Error.WriteLine(
                    "-extract-all requires a cubin or fatbin file path.");
                return 1;
            }

            return ExtractAll(allInput, emitJson, outputDir, force);
        }

        if (extractCubin)
        {
            string? cubinInput = cubinPath;
            if (cubinInput == null && hexInstruction != null &&
                File.Exists(hexInstruction) && !LooksLikeHex(hexInstruction))
                cubinInput = hexInstruction;

            if (cubinInput == null)
            {
                Console.Error.WriteLine(
                    "-extract-cubin requires a cubin or fatbin file path.");
                return 1;
            }

            return ExtractCubins(cubinInput, emitJson, outputDir, force);
        }

        if (extractPtx)
        {
            string? ptxInput = cubinPath;
            if (ptxInput == null && hexInstruction != null &&
                File.Exists(hexInstruction) && !LooksLikeHex(hexInstruction))
                ptxInput = hexInstruction;

            if (ptxInput == null)
            {
                Console.Error.WriteLine(
                    "-extract-ptx requires a cubin or fatbin file path.");
                return 1;
            }

            return ExtractPtx(ptxInput, emitJson, outputDir, force);
        }

        if (runCubinPath != null)
            return RunCubin(
                runCubinPath, liftPtxEntry, runBuffers, runBufferHexes,
                runScalars, runBlock, runNoBuffer);
        if (runPtxPath != null)
            return RunPtx(
                runPtxPath, liftPtxEntry, runBuffers, runBufferHexes,
                runScalars, runBlock, runNoBuffer);

        if (liftPtxPath != null)
        {
            if (Path.GetExtension(liftPtxPath).Equals(
                    ".cubin", StringComparison.OrdinalIgnoreCase))
            {
                return LiftCubin(liftPtxPath, liftPtxEntry, liftPtxParams,
                    liftPtxOutput, asmArchitecture, liftPtxFunctions);
            }
            return LiftPtx(liftPtxPath, liftPtxEntry, liftPtxParams,
                liftPtxOutput, asmArchitecture, null, 0, null, 0,
                liftPtxFunctions);
        }

        if (asmPath != null)
        {
            string output = asmOutput ??
                Path.ChangeExtension(asmPath, ".cubin");
            try
            {
                SassAssembler.Assemble(
                    asmPath, asmArchitecture, output, irOutput, _toolkit);
                if (fatbinOutput != null)
                {
                    CubinFile assembled = CubinFile.Load(output);
                    FatbinWriter.Write(
                        fatbinOutput, File.ReadAllBytes(output),
                        assembled.Architecture, _fatbinCompression);
                }
                Console.WriteLine(output);
                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Cannot assemble '{asmPath}': {ex.Message}");
                return 1;
            }
        }

        if (binaryPath != null)
            return DecodeBinary(
                binaryPath, binaryArchitecture, baseAddress, emitJson, nvJson,
                showEncoding, _toolkit);

        if (cubinPath != null)
        {
            if (asmArchitecture != null)
            {
                try
                {
                    return ConvertCubin(cubinPath, asmArchitecture, asmOutput ?? Path.ChangeExtension(cubinPath, "." + asmArchitecture + ".cubin"), irOutput, _toolkit, fatbinOutput, _fatbinCompression);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Cannot convert '{cubinPath}': {ex.Message}");
                    return 1;
                }
            }
            return DecodeCubinFile(
                cubinPath, emitJson, nvJson, annotateJson, functionFilter,
                showEncoding);
        }

        if (hexInstruction != null)
        {
            // A positional path is treated as a cubin; anything else must be
            // a 128-bit hexadecimal instruction.
            if (File.Exists(hexInstruction) && !LooksLikeHex(hexInstruction))
                return DecodeCubinFile(
                    hexInstruction, emitJson, nvJson, annotateJson,
                    functionFilter, showEncoding);

            return DecodeHex(
                hexInstruction, baseAddress, emitJson, showFields, showEncoding);
        }

        PrintHelp();
        return 1;
    }

    private static int DecodeBinary(
        string path, string architecture, ulong baseAddress, bool emitJson,
        bool nvJson, bool showEncoding, string? toolkit = null)
    {
        byte[] bytes;
        try
        {
            bytes = File.ReadAllBytes(path);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Cannot read '{path}': {ex.Message}");
            return 1;
        }

        if (bytes.Length % 16 != 0 && ArchitectureNumber(architecture) >= 70)
        {
            Console.Error.WriteLine(
                $"'{path}' is {bytes.Length} bytes; a raw SASS file must be a multiple of 16 bytes.");
            return 1;
        }
        if (ArchitectureNumber(architecture) < 70)
        {
            int bundleSize = PreVoltaDecoder.GetBundleSize(architecture);
            if (bytes.Length % bundleSize != 0)
            {
                Console.Error.WriteLine(
                    $"'{path}' is {bytes.Length} bytes; a pre-Volta raw SASS file " +
                    $"for {architecture} must be a multiple of {bundleSize} bytes.");
                return 1;
            }
        }
        else if (bytes.Length % 16 != 0)
        {
            Console.Error.WriteLine(
                $"'{path}' is {bytes.Length} bytes; a raw SASS file must be a multiple of 16 bytes.");
            return 1;
        }

        bool preVolta = ArchitectureNumber(architecture) < 70;
        if (preVolta)
        {
            try
            {
                PreVoltaDecoder.UseArchitecture(architecture, toolkit);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(ex.Message);
                return 1;
            }
        }
        else
        {
            try
            {
                InstructionDecoder.UseArchitecture(architecture, toolkit);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(ex.Message);
                return 1;
            }
        }

        var instructions = new List<object>();
        var nvInstructions = new List<object>();
        var nvSection = new CubinSection();
        var nvLabels = new Dictionary<(CubinSection, ulong), string>();
        if (!emitJson && !nvJson)
        {
            uint architectureNumber = ArchitectureNumber(architecture);
            string target = $"EF_CUDA_SM{architectureNumber}";
            string accelerators = architecture == "sm_90a"
                ? " EF_CUDA_ACCELERATORS"
                : "";
            string headerFlags =
                $"\t.headerflags\t@\"EF_CUDA_64BIT_ADDRESS{accelerators} " +
                $"{target} EF_CUDA_VIRTUAL_SM({target})\"";
            Console.WriteLine(headerFlags);
            Console.WriteLine(headerFlags);
        }

        if (preVolta)
        {
            if (nvJson)
            {
                Console.Error.WriteLine(
                    "nvdisasm-style JSON output is not supported for pre-Volta raw SASS.");
                return 1;
            }
            var decoded = PreVoltaDecoder
                .DecodeSection(bytes, baseAddress)
                .ToList();
            while (decoded.Count > 0 &&
                decoded[^1].Instruction.Mnemonic == "NOP")
            {
                decoded.RemoveAt(decoded.Count - 1);
            }
            if (emitJson)
            {
                foreach (var (_, instruction) in decoded)
                    instructions.Add(ToJson(instruction));
                WriteJson(baseAddress, instructions, architecture);
                return 0;
            }
            foreach (var (address, instruction) in decoded)
            {
                string encoding = showEncoding
                    ? $"                 /* 0x{instruction.Low:x16} */"
                    : "";
                Console.WriteLine(
                    FormatInstructionPrefix(address, instruction.Text) +
                    $"{instruction.Text};{encoding}");
            }
            return 0;
        }

        for (int offset = 0; offset < bytes.Length; offset += 16)
        {
            ulong low = BitConverter.ToUInt64(bytes, offset);
            ulong high = BitConverter.ToUInt64(bytes, offset + 8);
            ulong address = baseAddress + (ulong)offset;

            DecodedInstruction instruction = InstructionDecoder.Decode(low, high, address);

            if (nvJson)
            {
                nvInstructions.Add(NvInstruction(
                    nvSection, (offset, address, instruction), nvLabels));
            }
            else if (emitJson)
                instructions.Add(ToJson(instruction));
            else
            {
                string encoding = showEncoding
                    ? $"                 /* 0x{instruction.Low:x16} */"
                    : "";
                Console.WriteLine(
                    FormatInstructionPrefix(address, instruction.Text) +
                    $"{instruction.Text};{encoding}");
            }
        }

        if (nvJson)
            WriteNvRawJson(bytes.Length, nvInstructions, architecture);
        else if (emitJson)
            WriteJson(baseAddress, instructions, architecture);

        return 0;
    }

    private static int ListInput(string path, bool emitJson)
    {
        byte[] data;
        try
        {
            data = File.ReadAllBytes(path);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Cannot read '{path}': {ex.Message}");
            return 1;
        }

        var fatbins = new List<FatbinFile>();
        var cubins = new List<CubinFile>();
        try
        {
            if (FatbinFile.IsFatbin(data))
            {
                fatbins.Add(FatbinFile.Parse(data));
            }
            else if (IsElf(data))
            {
                CubinFile cubin = CubinFile.Load(data, path);
                if (cubin.Sections.Count > 0)
                    cubins.Add(cubin);
                fatbins.AddRange(ParseFatbinBlobs(cubin.EmbeddedFatbins));
            }
            else if (PeFatbinReader.IsPe(data))
            {
                fatbins.AddRange(ParseFatbinBlobs(
                    PeFatbinReader.FindEmbeddedFatbins(data)));
            }
            else if (CoffFatbinReader.IsCoff(data))
            {
                fatbins.AddRange(ParseFatbinBlobs(
                    CoffFatbinReader.FindEmbeddedFatbins(data)));
            }
            else
            {
                Console.Error.WriteLine(
                    $"'{path}' is neither an ELF/cubin nor a CUDA fatbin.");
                return 1;
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Cannot read '{path}': {ex.Message}");
            return 1;
        }

        var entries = fatbins.SelectMany(f => f.Entries).ToList();
        foreach (FatbinEntry entry in entries)
        {
            if (!entry.IsCubin ||
                (entry.Architecture != 0 &&
                    !IsSupportedArchitecture(entry.Architecture)))
                continue;
            try
            {
                cubins.Add(CubinFile.Load(entry.Payload, path));
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(
                    $"Cannot read fatbin ELF entry for sm_{entry.Architecture}: {ex.Message}");
            }
        }

        if (entries.Count == 0 && cubins.Count == 0)
        {
            Console.Error.WriteLine($"'{path}' contains no device code.");
            return 1;
        }

        if (emitJson)
        {
            Console.WriteLine(JsonOutput.Serialize(new
            {
                file = Path.GetFileName(path),
                fatbinEntries = entries.Select((e, index) => new
                {
                    index,
                    kind = e.KindName,
                    arch = FatbinEntryArchitecture(e),
                    identifier = e.Identifier,
                    ptxasOptions = e.PtxasOptions,
                    compression = e.Compression,
                    storedPayloadSize = e.StoredPayloadSize,
                    compressedSize = e.CompressedSize,
                    uncompressedSize = e.UncompressedSize > 0
                        ? e.UncompressedSize
                        : (ulong)e.Payload.Length,
                }),
                toolkit = cubins
                    .Select(c => c.ToolkitFromAbi ?? "unknown")
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .ToArray(),
                sections = cubins.SelectMany(c => c.Sections).Select(s => new
                {
                    name = s.Name,
                    address = $"0x{s.Address:x}",
                    size = $"0x{s.Size:x}",
                    functions = s.Functions.Select(f => new
                    {
                        name = f.Name,
                        index = f.Index,
                        address = $"0x{f.Value:x}",
                        size = $"0x{f.Size:x}",
                    }),
                }),
            }, indented: true));
            return 0;
        }

        Console.WriteLine($"file: {Path.GetFileName(path)}");
        if (entries.Count > 0)
        {
            Console.WriteLine("fatbin entries:");
            for (int i = 0; i < entries.Count; i++)
            {
                FatbinEntry entry = entries[i];
                string kind = entry.KindName;
                Console.WriteLine(
                    $"  [{i}] {kind} {FatbinEntryArchitecture(entry)} {entry.Compression} " +
                    (entry.Identifier.Length > 0 ? $"id={entry.Identifier} " : "") +
                    (entry.PtxasOptions.Length > 0
                        ? $"opts=\"{entry.PtxasOptions}\" "
                        : "") +
                    $"stored=0x{entry.StoredPayloadSize:x} " +
                    $"compressed=0x{entry.CompressedSize:x} " +
                    $"uncompressed=0x{entry.UncompressedSize:x}");
            }
        }

        if (cubins.Count > 0)
        {
            string[] toolkitLabels = cubins
                .Select(c => c.ToolkitFromAbi ?? "unknown")
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();
            Console.WriteLine($"  toolkit: {string.Join(", ", toolkitLabels)}");
            Console.WriteLine("sections:");
            foreach (CubinFile cubin in cubins)
            {
                foreach (CubinSection section in cubin.Sections)
                {
                    Console.WriteLine(
                        $"  {section.Name} addr=0x{section.Address:x} " +
                        $"size=0x{section.Size:x}");
                    foreach (CubinSymbol symbol in section.Functions)
                    {
                        Console.WriteLine(
                            $"    {symbol.Name} idx={symbol.Index} " +
                    $"0x{symbol.Value:x} 0x{symbol.Size:x}");
                    }
                }
            }
        }

        return 0;
    }

    private static int ExtractAll(
        string path, bool emitJson, string? outputDir, bool force)
    {
        byte[] data;
        try
        {
            data = File.ReadAllBytes(path);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Cannot read '{path}': {ex.Message}");
            return 1;
        }

        var fatbins = new List<FatbinFile>();
        try
        {
            if (FatbinFile.IsFatbin(data))
            {
                fatbins.Add(FatbinFile.Parse(data));
            }
            else if (IsElf(data))
            {
                CubinFile cubin = CubinFile.Load(data, path);
                fatbins.AddRange(ParseFatbinBlobs(cubin.EmbeddedFatbins));
            }
            else if (PeFatbinReader.IsPe(data))
            {
                fatbins.AddRange(ParseFatbinBlobs(
                    PeFatbinReader.FindEmbeddedFatbins(data)));
            }
            else if (CoffFatbinReader.IsCoff(data))
            {
                fatbins.AddRange(ParseFatbinBlobs(
                    CoffFatbinReader.FindEmbeddedFatbins(data)));
            }
            else
            {
                Console.Error.WriteLine(
                    $"'{path}' is neither an ELF/cubin nor a CUDA fatbin.");
                return 1;
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Cannot parse '{path}': {ex.Message}");
            return 1;
        }

        var entries = fatbins.SelectMany(f => f.Entries).ToList();
        if (entries.Count == 0)
        {
            Console.Error.WriteLine($"'{path}' contains no fatbin entries.");
            return 1;
        }

        string fullPath = Path.GetFullPath(path);
        string directory;
        try
        {
            directory = outputDir != null
                ? Path.GetFullPath(outputDir)
                : Path.GetDirectoryName(fullPath) ?? ".";
            Directory.CreateDirectory(directory);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Cannot create output directory: {ex.Message}");
            return 1;
        }

        string baseName = Path.GetFileNameWithoutExtension(fullPath);
        if (baseName.Length == 0)
            baseName = "extracted";

        var outputs = new List<(FatbinEntry Entry, string Path, byte[] Payload, string Token)>();
        var usedNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        for (int index = 0; index < entries.Count; index++)
        {
            FatbinEntry entry = entries[index];
            (string token, string extension) = DescribeKind(entry.Kind);
            string arch = FatbinEntryArchitecture(entry);
            string identifier = SafeFileNameStem(entry.Identifier);
            string stem = identifier.Length > 0 ? identifier : baseName;
            string indexed = $"{stem}.{arch}.{token}.{index}.{extension}";
            string fileName = identifier.Length == 0 && entries.Count > 1
                ? indexed
                : $"{stem}.{arch}.{token}.{extension}";
            if (!usedNames.Add(fileName))
                fileName = indexed;
            byte[] payload = entry.Kind == 1
                ? TrimTrailingZeros(entry.Payload)
                : entry.Payload;
            outputs.Add((entry, Path.Combine(directory, fileName), payload, token));
        }

        if (!force)
        {
            string? existing = outputs.FirstOrDefault(o => File.Exists(o.Path)).Path;
            if (existing != null)
            {
                Console.Error.WriteLine(
                    $"'{existing}' already exists; use --force to overwrite.");
                return 1;
            }
        }

        foreach (var output in outputs)
        {
            try
            {
                File.WriteAllBytes(output.Path, output.Payload);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(
                    $"Cannot write '{output.Path}': {ex.Message}");
                return 1;
            }
        }

        if (emitJson)
        {
            Console.WriteLine(JsonOutput.Serialize(new
            {
                file = Path.GetFileName(path),
                entries = outputs.Select(o => new
                {
                    kind = o.Entry.KindName,
                    arch = FatbinEntryArchitecture(o.Entry),
                    compression = o.Entry.Compression,
                    size = o.Payload.Length,
                    path = o.Path,
                }),
            }, indented: true));
        }
        else
        {
            foreach (var output in outputs)
                Console.WriteLine(output.Path);
        }

        return 0;
    }

    private static string SafeFileNameStem(string identifier)
    {
        if (string.IsNullOrWhiteSpace(identifier))
            return "";
        char[] invalid = Path.GetInvalidFileNameChars();
        string safe = new string(identifier
            .Select(c => Array.IndexOf(invalid, c) >= 0 ? '_' : c)
            .ToArray());
        return safe.Trim().TrimEnd('.');
    }

    private static (string Token, string Extension) DescribeKind(ushort kind) =>
        kind switch
        {
            1 => ("ptx", "ptx"),
            2 => ("elf", "cubin"),
            4 => ("oldcubin", "cubin"),
            8 => ("nvvm", "ltoir"),
            0x10 => ("capmerc", "elf"),
            0x20 => ("index", "index"),
            0x40 => ("relocptx", "bin"),
            0x80 => ("tileir", "tileir"),
            0x100 => ("concat", "bin"),
            _ => ($"kind{kind:x}", "bin"),
        };

    private static int ExtractCubins(
        string path, bool emitJson, string? outputDir, bool force)
    {
        byte[] data;
        try
        {
            data = File.ReadAllBytes(path);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Cannot read '{path}': {ex.Message}");
            return 1;
        }

        var fatbins = new List<FatbinFile>();
        try
        {
            if (FatbinFile.IsFatbin(data))
            {
                fatbins.Add(FatbinFile.Parse(data));
            }
            else if (IsElf(data))
            {
                CubinFile cubin = CubinFile.Load(data, path);
                fatbins.AddRange(ParseFatbinBlobs(cubin.EmbeddedFatbins));
            }
            else if (PeFatbinReader.IsPe(data))
            {
                fatbins.AddRange(ParseFatbinBlobs(
                    PeFatbinReader.FindEmbeddedFatbins(data)));
            }
            else if (CoffFatbinReader.IsCoff(data))
            {
                fatbins.AddRange(ParseFatbinBlobs(
                    CoffFatbinReader.FindEmbeddedFatbins(data)));
            }
            else
            {
                Console.Error.WriteLine(
                    $"'{path}' is neither an ELF/cubin nor a CUDA fatbin.");
                return 1;
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Cannot parse '{path}': {ex.Message}");
            return 1;
        }

        var entries = fatbins.SelectMany(f => f.Entries).Where(e => e.IsCubin).ToList();
        if (entries.Count == 0)
        {
            Console.Error.WriteLine($"'{path}' contains no ELF cubin entries.");
            return 1;
        }

        string fullPath = Path.GetFullPath(path);
        string directory;
        try
        {
            directory = outputDir != null
                ? Path.GetFullPath(outputDir)
                : Path.GetDirectoryName(fullPath) ?? ".";
            Directory.CreateDirectory(directory);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Cannot create output directory: {ex.Message}");
            return 1;
        }
        string baseName = Path.GetFileNameWithoutExtension(fullPath);
        if (baseName.Length == 0)
            baseName = "extracted";

        var outputs = new List<(FatbinEntry Entry, string Path)>();
        var usedNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        for (int index = 0; index < entries.Count; index++)
        {
            FatbinEntry entry = entries[index];
            string arch = FatbinEntryArchitecture(entry);
            string identifier = SafeFileNameStem(entry.Identifier);
            string stem = identifier.Length > 0 ? identifier : baseName;
            string indexed = $"{stem}.{arch}.{index}.cubin";
            string fileName = identifier.Length == 0 && entries.Count > 1
                ? indexed
                : $"{stem}.{arch}.cubin";
            if (!usedNames.Add(fileName))
                fileName = indexed;
            outputs.Add((entry, Path.Combine(directory, fileName)));
        }

        if (!force)
        {
            string? existing = outputs.FirstOrDefault(o => File.Exists(o.Path)).Path;
            if (existing != null)
            {
                Console.Error.WriteLine(
                    $"'{existing}' already exists; use --force to overwrite.");
                return 1;
            }
        }

        foreach (var output in outputs)
        {
            try
            {
                File.WriteAllBytes(output.Path, output.Entry.Payload);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(
                    $"Cannot write '{output.Path}': {ex.Message}");
                return 1;
            }
        }

        if (emitJson)
        {
            Console.WriteLine(JsonOutput.Serialize(new
            {
                file = Path.GetFileName(path),
                cubins = outputs.Select(o => new
                {
                    arch = FatbinEntryArchitecture(o.Entry),
                    size = o.Entry.Payload.Length,
                    path = o.Path,
                }),
            }, indented: true));
        }
        else
        {
            foreach (var output in outputs)
                Console.WriteLine(output.Path);
        }

        return 0;
    }

    private static int ExtractPtx(
        string path, bool emitJson, string? outputDir, bool force)
    {
        byte[] data;
        try
        {
            data = File.ReadAllBytes(path);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Cannot read '{path}': {ex.Message}");
            return 1;
        }

        var fatbins = new List<FatbinFile>();
        try
        {
            if (FatbinFile.IsFatbin(data))
            {
                fatbins.Add(FatbinFile.Parse(data));
            }
            else if (IsElf(data))
            {
                CubinFile cubin = CubinFile.Load(data, path);
                foreach (byte[] blob in cubin.EmbeddedFatbins)
                    fatbins.Add(FatbinFile.Parse(blob));
            }
            else if (PeFatbinReader.IsPe(data))
            {
                foreach (byte[] blob in PeFatbinReader.FindEmbeddedFatbins(data))
                    fatbins.Add(FatbinFile.Parse(blob));
            }
            else if (CoffFatbinReader.IsCoff(data))
            {
                foreach (byte[] blob in CoffFatbinReader.FindEmbeddedFatbins(data))
                    fatbins.Add(FatbinFile.Parse(blob));
            }
            else
            {
                Console.Error.WriteLine(
                    $"'{path}' is neither an ELF/cubin nor a CUDA fatbin.");
                return 1;
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Cannot parse '{path}': {ex.Message}");
            return 1;
        }

        var entries = fatbins.SelectMany(f => f.Entries).Where(e => e.IsPtx).ToList();
        if (entries.Count == 0)
        {
            Console.Error.WriteLine($"'{path}' contains no PTX entries.");
            return 1;
        }

        string fullPath = Path.GetFullPath(path);
        string directory;
        try
        {
            directory = outputDir != null
                ? Path.GetFullPath(outputDir)
                : Path.GetDirectoryName(fullPath) ?? ".";
            Directory.CreateDirectory(directory);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Cannot create output directory: {ex.Message}");
            return 1;
        }
        string baseName = Path.GetFileNameWithoutExtension(fullPath);
        if (baseName.Length == 0)
            baseName = "extracted";

        var outputs = new List<(FatbinEntry Entry, string Path, byte[] Payload)>();
        var usedNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        for (int index = 0; index < entries.Count; index++)
        {
            FatbinEntry entry = entries[index];
            string arch = FatbinEntryArchitecture(entry);
            string identifier = SafeFileNameStem(entry.Identifier);
            string stem = identifier.Length > 0 ? identifier : baseName;
            string indexed = $"{stem}.{arch}.{index}.ptx";
            string fileName = identifier.Length == 0 && entries.Count > 1
                ? indexed
                : $"{stem}.{arch}.ptx";
            if (!usedNames.Add(fileName))
                fileName = indexed;
            outputs.Add((
                entry,
                Path.Combine(directory, fileName),
                TrimTrailingZeros(entry.Payload)));
        }

        if (!force)
        {
            string? existing = outputs.FirstOrDefault(o => File.Exists(o.Path)).Path;
            if (existing != null)
            {
                Console.Error.WriteLine(
                    $"'{existing}' already exists; use --force to overwrite.");
                return 1;
            }
        }

        foreach (var output in outputs)
        {
            try
            {
                File.WriteAllBytes(output.Path, output.Payload);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(
                    $"Cannot write '{output.Path}': {ex.Message}");
                return 1;
            }
        }

        if (emitJson)
        {
            Console.WriteLine(JsonOutput.Serialize(new
            {
                file = Path.GetFileName(path),
                ptx = outputs.Select(o => new
                {
                    arch = FatbinEntryArchitecture(o.Entry),
                    codeVersionMajor = o.Entry.CodeVersionMajor,
                    codeVersionMinor = o.Entry.CodeVersionMinor,
                    size = o.Entry.UncompressedSize > 0
                        ? o.Entry.UncompressedSize
                        : (ulong)o.Entry.Payload.Length,
                    path = o.Path,
                }),
            }, indented: true));
        }
        else
        {
            foreach (var output in outputs)
                Console.WriteLine(output.Path);
        }

        return 0;
    }

    private static byte[] TrimTrailingZeros(byte[] data)
    {
        int length = data.Length;
        while (length > 0 && data[length - 1] == 0)
            length--;
        return length == data.Length ? data : data[..length];
    }

    private static string FatbinEntryArchitecture(FatbinEntry entry)
    {
        if (entry.IsCubin)
        {
            try
            {
                return CubinFile.Load(entry.Payload).ArchitectureKey;
            }
            catch
            {
                // Fall back to the fatbin header architecture below.
            }
        }
        else if (entry.IsPtx)
        {
            string text = Encoding.ASCII.GetString(entry.Payload);
            Match target = Regex.Match(text,
                @"\.target\s+(sm_\d+[a-z]?)",
                RegexOptions.IgnoreCase);
            if (target.Success)
                return target.Groups[1].Value.ToLowerInvariant();
        }
        return entry.Architecture == 0
            ? "unknown"
            : $"sm_{entry.Architecture}";
    }

    private static List<FatbinFile> ParseFatbinBlobs(IReadOnlyList<byte[]> blobs)
    {
        var parsed = new List<FatbinFile>();
        foreach (byte[] blob in blobs)
        {
            try
            {
                parsed.Add(FatbinFile.Parse(blob));
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Cannot parse embedded fatbin: {ex.Message}");
            }
        }
        return parsed;
    }

    private static int DecodeCubinFile(
        string path, bool emitJson, bool nvJson, bool annotateJson,
        string? functionFilter, bool showEncoding)
    {
        byte[] data;
        try
        {
            data = File.ReadAllBytes(path);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Cannot read '{path}': {ex.Message}");
            return 1;
        }

        return DecodeContainer(
            data, path, emitJson, nvJson, annotateJson,
            functionFilter, showEncoding);
    }

    internal static readonly object ConsoleOutputLock = new();

    internal static string ContainerToText(
        byte[] data, string name, string? functionFilter, bool showEncoding,
        bool preserveRelocations = false)
    {
        bool previousRelocations = _preserveRelocations;
        _preserveRelocations |= preserveRelocations;
        lock (ConsoleOutputLock)
        {
            try
            {
            var originalOut = Console.Out;
            var originalError = Console.Error;
            using var output = new StringWriter();
            using var error = new StringWriter();
            Console.SetOut(output);
            Console.SetError(error);
            int result;
            try
            {
                result = DecodeContainer(
                    data, name, false, false, false,
                    functionFilter, showEncoding);
            }
            finally
            {
                Console.SetOut(originalOut);
                Console.SetError(originalError);
            }
            if (result != 0)
            {
                string message = error.ToString().Trim();
                if (message.Length == 0)
                    message = $"container decode failed with status {result}";
                throw new InvalidDataException(message);
            }
            string text = output.ToString();
            if (text.Length == 0)
                throw new InvalidDataException("container decode produced no text");
            return text;
            }
            finally
            {
                _preserveRelocations = previousRelocations;
            }
        }
    }

    internal static int DecodeContainer(
        byte[] data, string path, bool emitJson, bool nvJson, bool annotateJson,
        string? functionFilter, bool showEncoding)
    {
        if (nvJson && !IsElf(data))
        {
            Console.Error.WriteLine(
                "--nv-json supports a single cubin, not a fatbin or host binary.");
            return 1;
        }

        if (FatbinFile.IsFatbin(data))
            return DecodeFatbin(data, path, emitJson, functionFilter, showEncoding);

        if (!IsElf(data))
        {
            if (PeFatbinReader.IsPe(data))
            {
                try
                {
                    var blobs = PeFatbinReader.FindEmbeddedFatbins(data);
                    var parsed = ParseFatbinBlobs(blobs);
                    if (parsed.Count > 0)
                        return DecodeFatbins(
                            parsed, path, emitJson, functionFilter, showEncoding);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine(
                        $"Cannot read PE fatbins in '{path}': {ex.Message}");
                    return 1;
                }
            }
            else if (CoffFatbinReader.IsCoff(data))
            {
                try
                {
                    var blobs = CoffFatbinReader.FindEmbeddedFatbins(data);
                    var parsed = ParseFatbinBlobs(blobs);
                    if (parsed.Count > 0)
                        return DecodeFatbins(
                            parsed, path, emitJson, functionFilter, showEncoding);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine(
                        $"Cannot read COFF fatbins in '{path}': {ex.Message}");
                    return 1;
                }
            }

            Console.Error.WriteLine(
                $"'{path}' is neither an ELF/cubin nor a CUDA fatbin.");
            return 1;
        }

        CubinFile cubin;
        try
        {
            cubin = CubinFile.Load(data, path);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Cannot read cubin '{path}': {ex.Message}");
            return 1;
        }
        cubin.Toolkit = cubin.ToolkitFromAbi;

        var fatbins = ParseFatbinBlobs(cubin.EmbeddedFatbins);

        if (cubin.Architecture != 0 &&
            !IsSupportedArchitecture(cubin.Architecture))
        {
            Console.Error.WriteLine(
                $"'{path}' targets sm_{cubin.Architecture}; " +
                "this disassembler supports sm_30, sm_32, sm_35, sm_37, sm_50, sm_52, sm_53, sm_60, sm_61, sm_62, sm_70, sm_72, sm_75, sm_80, sm_86, sm_87, sm_88, sm_89, sm_90, sm_90a, sm_100, sm_101, sm_103, sm_107, sm_110, sm_120 and sm_121.");
            return 1;
        }

        if (cubin.Architecture == 0 && fatbins.Count == 0 &&
            cubin.IsDeviceImage)
        {
            Console.Error.WriteLine(
                $"'{path}' does not declare a supported architecture " +
                "(old-format or unknown cubin); this disassembler " +
                "supports sm_70, sm_72, sm_75, sm_80, sm_86, sm_87, sm_88, sm_89, sm_90, sm_90a, sm_100, sm_101, sm_103, sm_107, sm_110, sm_120 and sm_121.");
            return 1;
        }

        if (fatbins.Count == 0 && !cubin.IsDeviceImage)
        {
            Console.Error.WriteLine(
                $"'{path}' is an ELF file but not a CUDA cubin " +
                "(no .nv.info or .note.nv.cuinfo section).");
            return 1;
        }

        if (cubin.Sections.Count == 0 && fatbins.Count == 0)
        {
            Console.Error.WriteLine($"'{path}' has no executable sections or fatbins.");
            return 1;
        }

        if (nvJson)
        {
            if (fatbins.Count > 0)
            {
                Console.Error.WriteLine(
                    "--nv-json supports a single cubin; extract an SM30/SM32/SM35/SM37/SM50/SM52/SM53/SM60/SM61/SM62/SM70/SM72/SM75/SM80/SM86/SM87/SM88/SM89/SM90/SM90a/SM100/SM101/SM103/SM107/SM110/SM120/SM121 " +
                    "cubin or use --emit-json for fatbins.");
                return 1;
            }
            if (!cubin.Sections.Any(s => SectionMatches(s, functionFilter)))
            {
                NoMatchingSections(path, functionFilter);
                return 1;
            }
            try
            {
                WriteNvJson(cubin, functionFilter, annotateJson);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(ex.Message);
                return 1;
            }
            return 0;
        }

        if (fatbins.Count == 0)
        {
            if (!cubin.Sections.Any(s => SectionMatches(s, functionFilter)))
            {
                NoMatchingSections(path, functionFilter);
                return 1;
            }

            try
            {
                // Validate the ABI before emitting any listing header.
                UseCubinArchitecture(cubin);
                if (emitJson)
                    WriteCubinJson(path, new[] { cubin }, null, functionFilter);
                else
                {
                    Console.WriteLine(
                        $"\t.target\t{cubin.ArchitectureKey}");
                    Console.WriteLine();
                    Console.WriteLine(
                        $"\t.elftype\t@\"{ElfTypeName(cubin.ElfType)}\"");
                    Console.WriteLine();
                    if (!WriteCubinText(
                            cubin, null, functionFilter, showEncoding))
                        return 1;
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(ex.Message);
                return 1;
            }
            return 0;
        }

        // The ELF may contain SASS sections *and* an embedded .nv_fatbin.
        // Decode both instead of assuming they are mutually exclusive.
        var cubins = new List<CubinFile>();
        if (cubin.Sections.Count > 0 && cubin.IsDeviceImage)
            cubins.Add(cubin);

        foreach (FatbinFile fatbin in fatbins)
        {
            foreach (FatbinEntry entry in fatbin.Entries)
            {
                if (!entry.IsCubin ||
                    (entry.Architecture != 0 &&
                        !IsSupportedArchitecture(entry.Architecture)))
                    continue;
                try
                {
                    CubinFile loaded = CubinFile.Load(entry.Payload, path);
                    if (loaded.Architecture != 0 &&
                        !IsSupportedArchitecture(loaded.Architecture))
                        continue;
                    loaded.Toolkit = loaded.ToolkitFromAbi;
                    cubins.Add(loaded);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine(
                        $"Cannot read fatbin ELF entry for sm_{entry.Architecture}: {ex.Message}");
                }
            }
        }

        if (!cubins.Any(c => c.Sections.Any(s => SectionMatches(s, functionFilter))))
        {
            NoMatchingSections(path, functionFilter);
            return 1;
        }

        if (emitJson)
        {
            WriteCubinJson(path, cubins, fatbins, functionFilter);
        }
        else
        {
            string? commonArchitecture = CommonArchitecture(cubins);
            if (commonArchitecture is string common)
                WriteTargetLine(common);

            int index = 0;
            if (cubin.Sections.Count > 0 && cubin.IsDeviceImage)
            {
                if (commonArchitecture == null &&
                    cubin.Architecture != 0)
                    WriteTargetLine(cubin.ArchitectureKey);
                WriteCubinText(cubin,
                    $"//--------------------- ELF: {Path.GetFileName(path)} --------------------------",
                    functionFilter, showEncoding);
            }

            foreach (CubinFile embedded in cubins)
            {
                if (ReferenceEquals(embedded, cubin))
                    continue;
                if (commonArchitecture == null &&
                    embedded.Architecture != 0)
                    WriteTargetLine(embedded.ArchitectureKey);
                string title = cubins.Count == 1
                    ? $"//--------------------- fatbin: {Path.GetFileName(path)} --------------------------"
                    : $"//--------------------- fatbin entry {index} " +
                        $"({embedded.ArchitectureKey}) --------------------------";
                WriteCubinText(embedded, title, functionFilter, showEncoding);
                index++;
            }
        }
        return 0;
    }

    private static int DecodeFatbin(
        byte[] data, string path, bool emitJson, string? functionFilter,
        bool showEncoding)
    {
        FatbinFile fatbin;
        try
        {
            fatbin = FatbinFile.Parse(data);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Cannot parse fatbin '{path}': {ex.Message}");
            return 1;
        }
        return DecodeFatbins(new[] { fatbin }, path, emitJson, functionFilter, showEncoding);
    }

    private static int DecodeFatbins(
        IReadOnlyList<FatbinFile> fatbins, string path, bool emitJson,
        string? functionFilter, bool showEncoding)
    {
        var cubins = new List<CubinFile>();
        var allEntries = new List<FatbinEntry>();
        foreach (FatbinFile fatbin in fatbins)
        {
            allEntries.AddRange(fatbin.Entries);
            foreach (FatbinEntry entry in fatbin.Entries)
            {
                if (!entry.IsCubin ||
                    (entry.Architecture != 0 &&
                        !IsSupportedArchitecture(entry.Architecture)))
                    continue;
                try
                {
                    CubinFile loaded = CubinFile.Load(entry.Payload, path);
                    if (loaded.Architecture != 0 &&
                        !IsSupportedArchitecture(loaded.Architecture))
                        continue;
                    loaded.Toolkit = loaded.ToolkitFromAbi;
                    cubins.Add(loaded);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine(
                        $"Cannot read fatbin ELF entry for sm_{entry.Architecture}: {ex.Message}");
                }
            }
        }

        if (cubins.Count == 0)
        {
            string available = string.Join(", ", allEntries.Select(
                e => $"{e.KindName} sm_{e.Architecture}"));
            Console.Error.WriteLine(
                $"'{path}' contains no supported SM70/SM72/SM75/SM80/SM86/SM87/SM88/SM89/SM90/SM90a/SM100/SM101/SM103/SM107/SM110/SM120/SM121 cubin entry ({available}).");
            return 1;
        }

        if (!cubins.Any(c => c.Sections.Any(s => SectionMatches(s, functionFilter))))
        {
            NoMatchingSections(path, functionFilter);
            return 1;
        }

        if (emitJson)
        {
            WriteCubinJson(path, cubins, fatbins, functionFilter);
        }
        else
        {
            string? commonArchitecture = CommonArchitecture(cubins);
            if (commonArchitecture is string common)
                WriteTargetLine(common);
            int index = 0;
            foreach (CubinFile cubin in cubins)
            {
                if (commonArchitecture == null && cubin.Architecture != 0)
                    WriteTargetLine(cubin.ArchitectureKey);
                string title = cubins.Count == 1
                    ? $"//--------------------- fatbin: {Path.GetFileName(path)} --------------------------"
                    : $"//--------------------- fatbin entry {index} " +
                        $"({cubin.ArchitectureKey}) --------------------------";
                if (WriteCubinText(cubin, title, functionFilter, showEncoding))
                    index++;
            }
        }
        return 0;
    }

    private static Dictionary<ulong, string> BuildGlobalSymbolMap(CubinFile cubin)
    {
        var symbols = new Dictionary<ulong, string>();
        foreach (CubinSection section in cubin.Sections)
        {
            foreach (CubinSymbol symbol in section.Functions)
            {
                ulong address = symbol.Value >= section.Address
                    ? symbol.Value
                    : section.Address + symbol.Value;
                symbols.TryAdd(address, symbol.Name);
            }
        }
        return symbols;
    }

    internal static bool ValidateTarget(
        string architecture, string? toolkit, out string error)
    {
        try
        {
            string? path = ArchitectureNumber(architecture) < 70
                ? PreVoltaDecoder.ResolveTablePath(architecture, toolkit)
                : InstructionDecoder.ResolveTablePath(architecture, toolkit);
            if (path != null)
            {
                error = "";
                return true;
            }
            string? normalized = InstructionDecoder.NormalizeToolkit(toolkit);
            error = $"No ISA table is available for {architecture}" +
                (normalized != null ? $" with toolkit {normalized}" : "") +
                ".";
            return false;
        }
        catch (Exception ex)
        {
            error = ex.Message;
            return false;
        }
    }

    private static void UseCubinArchitecture(CubinFile cubin)
    {
        string? toolkit = cubin.Toolkit;
        if (toolkit == null)
            throw new NotSupportedException(
                $"Unsupported cubin ABI for {cubin.ArchitectureKey} " +
                $"(OSABI 0x{cubin.ElfOsAbi:x2}, ABI version " +
                $"{cubin.ElfAbiVersion}); cannot decode SASS. " +
                "Extract PTX with --extract-ptx if it exists.");
        if (_toolkit != null && _toolkit != toolkit)
            throw new NotSupportedException(
                $"Cubin ABI is CUDA {toolkit}, but --toolkit {_toolkit} " +
                "was requested; refusing to decode with a mismatched ABI.");
        if (cubin.Architecture is > 0 and < 70)
        {
            PreVoltaDecoder.UseArchitecture(cubin.ArchitectureKey, toolkit);
            cubin.Toolkit = "102";
        }
        else
        {
            InstructionDecoder.UseArchitecture(cubin.ArchitectureKey, toolkit);
            cubin.Toolkit = InstructionDecoder.Table.Toolkit;
        }
    }

    private static string? ToolkitLabel(IReadOnlyList<CubinFile> cubins)
    {
        string[] toolkits = cubins
            .Select(cubin => cubin.Toolkit)
            .Where(toolkit => !string.IsNullOrEmpty(toolkit))
            .Select(toolkit => toolkit!)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        return toolkits.Length switch
        {
            0 => null,
            1 => toolkits[0],
            _ => "mixed",
        };
    }

    private static bool WriteCubinText(
        CubinFile cubin, string? title, string? functionFilter, bool showEncoding)
    {
        try
        {
            UseCubinArchitecture(cubin);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.Message);
            return false;
        }

        var original = Console.Out;
        using var buffer = new StringWriter();
        Console.SetOut(buffer);
        bool written;
        try
        {
            written = WriteCubinTextCore(
                cubin, title, functionFilter, showEncoding);
        }
        finally
        {
            Console.SetOut(original);
        }

        if (written)
            WriteAlignedCubinText(buffer.ToString());
        return written;
    }

    private static void WriteAlignedCubinText(string text)
    {
        var lines = new List<string>();
        using (var reader = new StringReader(text))
        {
            string? line;
            while ((line = reader.ReadLine()) != null)
                lines.Add(line);
        }

        int width = 0;
        foreach (string line in lines)
        {
            if (!line.Contains("/*", StringComparison.Ordinal))
                continue;
            int end = line.IndexOf(" (*\"", StringComparison.Ordinal);
            if (end < 0)
                end = line.IndexOf(';');
            if (end > width)
                width = end;
        }

        // nvdisasm leaves two columns before a trailing `;`, and aligns
        // instruction comments to that same column.
        int commentColumn = width + 2;
        foreach (string line in lines)
        {
            int annotation = line.IndexOf(" (*\"", StringComparison.Ordinal);
            if (annotation < 0)
            {
                Console.WriteLine(line);
                continue;
            }

            string prefix = line[..annotation];
            if (prefix.Length < commentColumn)
                prefix = prefix.PadRight(commentColumn);
            Console.WriteLine(prefix + line[annotation..]);
        }
    }

    private static bool WriteCubinTextCore(
        CubinFile cubin, string? title, string? functionFilter, bool showEncoding)
    {
        var sections = cubin.Sections
            .Where(s => SectionMatches(s, functionFilter))
            .ToList();
        if (sections.Count == 0)
            return false;

        var globalSymbols = BuildGlobalSymbolMap(cubin);
        var (localLabels, endLabels, decodedSections) = BuildLocalLabelMap(cubin);

        if (title != null)
            Console.WriteLine(title);

        foreach (CubinSection section in sections)
        {
            string displayName = section.DisplayName.Length > 0
                ? section.DisplayName
                : section.Name;
            Console.WriteLine($"//--------------------- {displayName} --------------------------");
            Console.WriteLine($"\t.section\t{displayName},\"ax\",@progbits");

            var decoded = decodedSections[section];
            uint? registerCount = null;
            foreach (CubinSymbol symbol in section.Functions)
            {
                if (cubin.RegisterCounts.TryGetValue(symbol.Index, out uint count))
                {
                    registerCount = count;
                    break;
                }
            }
            if (cubin.Architecture < 90)
            {
                registerCount = RegisterCountEstimator.Estimate(
                    decoded.Select(item => item.Instruction.Text),
                    preVolta: cubin.Architecture < 70);
            }
            // nvdisasm omits register-count section metadata for Hopper and
            // newer architectures.
            if (registerCount is uint registers && cubin.Architecture < 90)
                Console.WriteLine($"\t.sectioninfo\t@\"SHI_REGISTERS={registers}\"");
            if (section.Alignment > 1)
                Console.WriteLine($"\t.align\t{section.Alignment}");

            bool annotateSpills = !_noDataflow;
            bool annotateBrx = !_noDataflow;
            HashSet<(ulong Function, string Offset)> nonSpillSlots =
                annotateSpills
                    ? BuildNonSpillSlots(section, decoded)
                    : new HashSet<(ulong, string)>();
            var functionLabels = new Dictionary<ulong, string>();
            var functionSymbols = new Dictionary<ulong, CubinSymbol>();
            foreach (CubinSymbol symbol in section.Functions)
            {
                ulong offset = symbol.Value >= section.Address
                    ? symbol.Value - section.Address
                    : symbol.Value;
                if (section.PreVolta &&
                    offset % (ulong)PreVoltaDecoder.GetBundleSize(
                        section.ArchitectureKey) == 0)
                {
                    // A pre-Volta function starts at the bundle control word;
                    // nvdisasm prints the symbol immediately before the first
                    // instruction slot.  A symbol already inside a bundle (for
                    // example the branch-stack label L1 at slot 3) must not be
                    // shifted.
                    offset += 8;
                }
                functionLabels.TryAdd(offset, symbol.Name);
                functionSymbols.TryAdd(offset, symbol);
            }

            bool sectionLabelPrinted = false;
            if (section.Functions.Count == 0)
            {
                WriteSectionLabel(section);
                sectionLabelPrinted = true;
            }

            foreach (var item in decoded)
            {
                ulong offset = (ulong)item.Offset;
                if (endLabels.TryGetValue((section, item.Address), out string? endName) &&
                    !functionSymbols.ContainsKey(offset))
                    Console.WriteLine($"{endName}:");
                if (functionSymbols.TryGetValue(offset, out CubinSymbol? functionSymbol))
                {
                    if (functionSymbol.Binding is not (1 or 2) &&
                        !sectionLabelPrinted)
                    {
                        WriteSectionLabel(section);
                        sectionLabelPrinted = true;
                    }
                    WriteFunctionMetadata(
                        section, functionSymbol, endLabels, localLabels,
                        functionSymbols);
                    Console.WriteLine($"{functionSymbol.Name}:");
                    if (functionSymbol.Binding is 1 or 2 &&
                        !sectionLabelPrinted)
                    {
                        WriteSectionLabel(section);
                        sectionLabelPrinted = true;
                    }
                }
                if (localLabels.TryGetValue((section, item.Address), out string? localName))
                    Console.WriteLine($"{localName}:");

                string text = item.Instruction.Text;
                string? relocated = TryApplyRelocation(
                    section, item.Offset, text, localLabels);
                if (relocated != null)
                {
                    text = relocated;
                }
                else if (item.Instruction.BranchTarget is ulong target)
                {
                    ulong labelTarget =
                        SnapToInstructionAddress(section, target);
                    if (IsInstructionAddress(section, labelTarget))
                    {
                        ulong targetOffset = labelTarget - section.Address;
                        if (functionLabels.TryGetValue(targetOffset, out string? targetFunction))
                            text = ReplaceTarget(text, target, targetFunction);
                        else if (localLabels.TryGetValue((section, labelTarget), out string? targetLocal))
                            text = ReplaceTarget(text, target, targetLocal);
                    }
                    else if (globalSymbols.TryGetValue(target, out string? targetSymbol))
                    {
                        text = ReplaceTarget(text, target, targetSymbol);
                    }
                }

                string annotation = "";
                ulong functionStart = annotateSpills
                    ? FunctionStartAt(section, item.Address)
                    : 0;
                if (annotateSpills &&
                    TryGetR1Local(text, out string spillKey, out _, out _) &&
                    !nonSpillSlots.Contains((functionStart, spillKey)))
                {
                    annotation = " (*\"SpillRefill\"*)";
                }

                if (annotateBrx &&
                    section.IndirectBranchTargets.TryGetValue(
                        (ulong)item.Offset, out List<ulong>? branchTargets))
                {
                    var names = new List<string>();
                    foreach (ulong target in branchTargets)
                    {
                        if (localLabels.TryGetValue((section, target), out string? label))
                            names.Add(label);
                    }
                    if (names.Count > 0)
                    {
                        annotation +=
                            " (*\"BRANCH_TARGETS " +
                            string.Join(",", names.Distinct()) + "\"*)";
                    }
                }

                if (_preserveRelocations &&
                    section.Relocations.Any(
                        relocation => relocation.Offset == (ulong)item.Offset))
                {
                    annotation += PreservedRelocationAnnotations(
                        section, item.Offset);
                }

                string encoding = showEncoding
                    ? $"                 /* 0x{item.Instruction.Low:x16} */"
                    : "";
                Console.WriteLine(
                    FormatInstructionPrefix(item.Address, text) +
                    $"{text}{annotation};{encoding}");
            }

            if (endLabels.TryGetValue(
                    (section, section.Address + section.Size), out string? sectionEnd))
                Console.WriteLine($"{sectionEnd}:");

            Console.WriteLine();
        }

        if (cubin.ObjectSymbols.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine(
                "//--------------------- SYMBOLS --------------------------");
            Console.WriteLine();
            foreach (CubinSymbol symbol in cubin.ObjectSymbols)
            {
                if (symbol.Type == 1)
                {
                    Console.WriteLine(
                        $"\t.type\t\t{symbol.Name},@object");
                    Console.WriteLine(
                        $"\t.size\t\t{symbol.Name},0x{symbol.Size:x}");
                }
                else
                {
                    Console.WriteLine(
                        $"\t.type\t\t{symbol.Name},@\"STT_CUDA_OBJECT\"");
                    Console.WriteLine(
                        $"\t.other\t\t{symbol.Name}," +
                        "@\"STO_CUDA_SHARED STV_DEFAULT\"");
                }
            }
        }

        return true;
    }

    private static void WriteSectionLabel(CubinSection section)
    {
        string label = section.LabelName.Length > 0
            ? section.LabelName
            : section.Name;
        string original = section.LabelOriginalName.Length > 0
            ? section.LabelOriginalName
            : section.Name;
        if (label != original)
            Console.WriteLine($"        .map_symbolname {label} {original}");
        Console.WriteLine($"{label}:");
    }

    private static void WriteFunctionMetadata(
        CubinSection section, CubinSymbol symbol,
        Dictionary<(CubinSection, ulong), string> endLabels,
        Dictionary<(CubinSection, ulong), string> localLabels,
        IReadOnlyDictionary<ulong, CubinSymbol> functionSymbols)
    {
        if (symbol.Binding == 1)
            Console.WriteLine($"        .global         {symbol.Name}");
        else if (symbol.Binding == 2)
            Console.WriteLine($"        .weak           {symbol.Name}");
        Console.WriteLine($"        .type           {symbol.Name},@function");

        ulong start = symbol.Value >= section.Address
            ? symbol.Value - section.Address
            : symbol.Value;
        ulong endOffset = start + symbol.Size;
        ulong endAddress = section.Address + endOffset;
        string end = symbol.Size.ToString(CultureInfo.InvariantCulture);
        if (functionSymbols.TryGetValue(endOffset, out CubinSymbol? nextFunction))
            end = nextFunction.Name;
        else if (endLabels.TryGetValue((section, endAddress), out string? endLabel) ||
            localLabels.TryGetValue((section, endAddress), out endLabel))
            end = endLabel;
        Console.WriteLine(
            $"        .size           {symbol.Name},({end} - {symbol.Name})");

        if (symbol.Other != 0)
        {
            string visibility = (symbol.Other & 3) switch
            {
                1 => "STV_INTERNAL",
                2 => "STV_HIDDEN",
                3 => "STV_PROTECTED",
                _ => "STV_DEFAULT",
            };
            string cudaEntry = (symbol.Other & 0x10) != 0
                ? "STO_CUDA_ENTRY "
                : "";
            Console.WriteLine(
                $"        .other          {symbol.Name}," +
                $"@\"{cudaEntry}{visibility}\"");
        }

        if (symbol.Name != symbol.OriginalName)
            Console.WriteLine(
                $"        .map_symbolname {symbol.Name} {symbol.OriginalName}");
    }

    private static (Dictionary<(CubinSection, ulong), string> Labels,
        Dictionary<(CubinSection, ulong), string> EndLabels,
        Dictionary<CubinSection, List<(int Offset, ulong Address, DecodedInstruction Instruction)>> Decoded)
        BuildLocalLabelMap(CubinFile cubin)
    {
        var labels = new Dictionary<(CubinSection, ulong), string>();
        var endLabels = new Dictionary<(CubinSection, ulong), string>();
        var decodedSections = new Dictionary<
            CubinSection, List<(int Offset, ulong Address, DecodedInstruction Instruction)>>();
        int nextLabel = 0;

        // Label numbers are cubin-wide, not per section.  Iterate every
        // section in file order so filtering with --function does not
        // renumber the remaining labels.  Cubins normally have sh_addr == 0
        // for every section, so the section is part of the key.
        //
        // A pre-Volta function with control flow gets a local label at its
        // first instruction slot.  This label precedes branch-target labels
        // in nvdisasm's numbering.
        foreach (CubinSection section in cubin.Sections)
        {
            if (!section.PreVolta || section.Functions.Count == 0)
                continue;
            int bundleSize = PreVoltaDecoder.GetBundleSize(
                section.ArchitectureKey);
            foreach (CubinSymbol symbol in section.Functions
                .Where(symbol => symbol.Size > 0 &&
                    (symbol.Binding is 1 or 2))
                .Take(1))
            {
                ulong offset = symbol.Value >= section.Address
                    ? symbol.Value - section.Address
                    : symbol.Value;
                // A symbol at a bundle boundary points at the control word;
                // nvdisasm does not add a local label for that function.
                if (offset % (ulong)bundleSize == 0)
                    continue;
                ulong address = section.Address + offset;
                if (!labels.ContainsKey((section, address)))
                    labels[(section, address)] = $".L_{nextLabel++ + 1}";
            }
        }

        // nvdisasm allocates relocation targets first (ascending address per
        // section), then branch targets (ascending address per section), and
        // finally function-end labels.
        foreach (CubinSection section in cubin.Sections)
        {
            foreach (ulong target in section.RelocationTargets)
            {
                if (labels.ContainsKey((section, target)))
                    continue;
                labels[(section, target)] = $".L_{nextLabel++ + 1}";
            }
        }

        foreach (CubinSection section in cubin.Sections)
        {
            var functionStarts = new HashSet<ulong>();
            foreach (CubinSymbol symbol in section.Functions)
            {
                functionStarts.Add(symbol.Value >= section.Address
                    ? symbol.Value
                    : section.Address + symbol.Value);
            }

            var decoded = DecodeSection(section, cubin.Toolkit);
            decodedSections[section] = decoded;
            foreach (var item in decoded)
            {
                if (!_noDataflow && item.Instruction.Mnemonic == "BRX" &&
                    HasNearBrxTableRelocation(section, item.Offset))
                {
                    // nvdisasm consumes an extra dataflow label number while
                    // walking symbolic BRX jump tables, before the BRX
                    // instruction labels themselves are emitted.
                    nextLabel++;
                    continue;
                }

                if (item.Instruction.BranchTarget is not ulong target)
                    continue;
                if (!IsInstructionAddress(section, target))
                    continue;
                target = SnapToInstructionAddress(section, target);
                if (!IsInstructionAddress(section, target))
                    continue;
                if (functionStarts.Contains(target) ||
                    labels.ContainsKey((section, target)))
                    continue;
                labels[(section, target)] = $".L_{nextLabel++ + 1}";
            }
        }

        if (!_noDataflow)
        {
            int brxTargetEntries = 0;
            foreach (CubinSection section in cubin.Sections)
            {
                // A BRX instruction gets a dataflow label before its target
                // table labels, even when the instruction itself is not the
                // target of a normal branch.  nvdisasm also consumes a label
                // number for every target-table entry, including entries that
                // already have a label, which leaves gaps in the numbering.
                foreach (var item in decodedSections[section])
                {
                    if (item.Instruction.Mnemonic != "BRX")
                        continue;
                    var key = (section, item.Address);
                    if (!labels.ContainsKey(key))
                        labels[key] = $".L_{nextLabel++ + 1}";

                    if (!section.IndirectBranchTargets.TryGetValue(
                            (ulong)item.Offset, out List<ulong>? targets))
                        continue;
                    if (HasNearBrxTableRelocation(section, item.Offset))
                        brxTargetEntries += targets.Count;
                    foreach (ulong target in targets)
                    {
                        if (!IsInstructionAddress(section, target) ||
                            labels.ContainsKey((section, target)))
                        {
                            nextLabel++;
                            continue;
                        }
                        labels[(section, target)] = $".L_{nextLabel++ + 1}";
                    }
                }
            }
            // nvdisasm consumes one additional dataflow label per jump-table
            // entry after all target labels have been emitted.
            nextLabel += brxTargetEntries;
        }

        // nvdisasm allocates every function-end label after all branch-target
        // labels, in symbol-table order.  A function whose end already has a
        // label still consumes a number, which leaves gaps in the output.
        var sectionsByIndex = cubin.Sections.ToDictionary(s => s.Index);
        foreach (CubinSymbol symbol in cubin.FunctionSymbols)
        {
            if (symbol.Size == 0 ||
                !sectionsByIndex.TryGetValue(symbol.SectionIndex, out var section))
                continue;
            ulong start = symbol.Value >= section.Address
                ? symbol.Value - section.Address
                : symbol.Value;
            ulong end = start + symbol.Size;
            if (end > section.Size)
                continue;
            ulong address = section.Address + end;
            var key = (section, address);
            if (!labels.ContainsKey(key) && !endLabels.ContainsKey(key))
                endLabels[key] = $".L_{nextLabel++ + 1}";
            else
                nextLabel++;
        }

        return (labels, endLabels, decodedSections);
    }

    private static void WriteNvJson(
        CubinFile cubin, string? functionFilter, bool annotate)
    {
        UseCubinArchitecture(cubin);
        var (localLabels, _, decodedSections) = BuildLocalLabelMap(cubin);
        var globalSymbols = BuildGlobalSymbolMap(cubin);
        var functions = new List<object>();
        foreach (CubinSection section in cubin.Sections)
        {
            if (!SectionMatches(section, functionFilter))
                continue;

            var decoded = decodedSections[section];
            var nonSpillSlots = annotate
                ? BuildNonSpillSlots(section, decoded)
                : new HashSet<(ulong, string)>();
            var symbols = section.Functions.Count > 0
                ? section.Functions
                : new List<CubinSymbol>
                {
                    new(section.Name, 2, 0, 0, section.Size),
                };

            foreach (CubinSymbol symbol in symbols)
            {
                ulong start = symbol.Value >= section.Address
                    ? symbol.Value - section.Address
                    : symbol.Value;
                if (start >= section.Size)
                    continue;
                ulong length = symbol.Size > 0
                    ? symbol.Size
                    : section.Size - start;
                if (length > section.Size - start)
                    length = section.Size - start;

                // nvdisasm's JSON writer truncates a function at the next
                // symbol start even when the symbol's .size covers a nested
                // helper.  The reported length field still uses .size.
                ulong nextStart = section.Functions
                    .Select(candidate => candidate.Value >= section.Address
                        ? candidate.Value - section.Address
                        : candidate.Value)
                    .Where(offset => offset > start)
                    .DefaultIfEmpty(section.Size)
                    .Min();
                ulong instructionEnd = start + length;
                if (nextStart < instructionEnd)
                    instructionEnd = nextStart;

                var instructions = decoded
                    .Where(item => (ulong)item.Offset >= start &&
                        (ulong)item.Offset < instructionEnd)
                    .Select(item => NvInstruction(
                        section, item, localLabels, globalSymbols,
                        annotate && IsSpillRefill(section, item, nonSpillSlots)))
                    .ToList();
                if (instructions.Count == 0)
                    continue;

                functions.Add(new Dictionary<string, object?>
                {
                    ["function-name"] = symbol.Name,
                    ["start"] = start,
                    ["length"] = length,
                    ["other-attributes"] = Array.Empty<object>(),
                    ["sass-instructions"] = instructions,
                });
            }
        }

        var metadata = new Dictionary<string, object?>
        {
            ["ELF"] = new Dictionary<string, object?>
            {
                ["layout-id"] = LayoutId(cubin.Architecture),
                ["ei_osabi"] = cubin.ElfOsAbi,
                ["ei_abiversion"] = cubin.ElfAbiVersion,
            },
            ["SM"] = new Dictionary<string, object?>
            {
                ["version"] = new Dictionary<string, object?>
                {
                    ["major"] = cubin.Architecture / 10,
                    ["minor"] = cubin.Architecture % 10,
                },
            },
            ["SchemaVersion"] = new Dictionary<string, object?>
            {
                ["major"] = 13,
                ["minor"] = 4,
                ["revision"] = 0,
            },
            ["Producer"] = "rtxdisam",
            ["Description"] = "",
        };

        if (cubin.Notes.CuinfoPresent)
        {
            metadata[".note.nv.cuinfo"] = new Dictionary<string, object?>
            {
                ["nv_note_cuinfo"] = cubin.Notes.CuinfoFormat,
                ["nv_note_cuinfo_virt_sm"] = cubin.Notes.Architecture,
                ["nv_note_cuinfo_toolVersion"] =
                    cubin.Notes.CuinfoToolVersion,
            };
        }
        if (cubin.Notes.TkinfoPresent)
        {
            metadata[".note.nv.tkinfo"] = new Dictionary<string, object?>
            {
                ["nv_note_tkinfo"] = cubin.Notes.TkinfoVersion,
                ["tki_objFname"] = cubin.Notes.TkinfoObjectFileName,
                ["tki_toolName"] = cubin.Notes.TkinfoToolName,
                ["tki_toolVersion"] = cubin.Notes.TkinfoToolVersion,
                ["tki_toolBranch"] = cubin.Notes.TkinfoToolBranch,
                ["tki_toolOptions"] = cubin.Notes.TkinfoToolOptions,
            };
        }

        Console.WriteLine(JsonOutput.Serialize(
            new object[] { metadata, functions }));
    }

    private static void WriteNvRawJson(
        int length, List<object> instructions, string architecture)
    {
        uint architectureNumber = ArchitectureNumber(architecture);
        var metadata = new Dictionary<string, object?>
        {
            ["ELF"] = new Dictionary<string, object?>
            {
                ["layout-id"] = 4,
                ["ei_osabi"] = 51,
                ["ei_abiversion"] = 7,
            },
            ["SM"] = new Dictionary<string, object?>
            {
                ["version"] = new Dictionary<string, object?>
                {
                    ["major"] = architectureNumber / 10,
                    ["minor"] = architectureNumber % 10,
                },
            },
            ["SchemaVersion"] = new Dictionary<string, object?>
            {
                ["major"] = 13,
                ["minor"] = 4,
                ["revision"] = 0,
            },
            ["Producer"] = "rtxdisam",
            ["Description"] = "",
        };
        var function = new Dictionary<string, object?>
        {
            ["function-name"] = "Unknown",
            ["start"] = 0,
            ["length"] = length,
            ["other-attributes"] = Array.Empty<object>(),
            ["sass-instructions"] = instructions,
        };
        Console.WriteLine(JsonOutput.Serialize(
            new object[] { metadata, new object[] { function } }));
    }

    private static Dictionary<string, object?> NvInstruction(
        CubinSection section,
        (int Offset, ulong Address, DecodedInstruction Instruction) item,
        Dictionary<(CubinSection, ulong), string> localLabels,
        IReadOnlyDictionary<ulong, string>? globalSymbols = null,
        bool spillRefill = false)
    {
        string text = item.Instruction.Text;
        string? relocated = TryApplyRelocation(
            section, item.Offset, text, localLabels,
            numericTargets: true, nvJsonStyle: true);
        if (relocated != null)
            text = relocated;

        string? predicate = null;
        if (text.StartsWith("@", StringComparison.Ordinal))
        {
            int space = text.IndexOf(' ');
            if (space > 0)
            {
                predicate = text[..space];
                text = text[(space + 1)..];
            }
        }

        int separator = text.IndexOf(' ');
        string opcode = separator < 0 ? text : text[..separator];
        string? operands = separator < 0
            ? null
            : text[(separator + 1)..].Replace(", ", ",", StringComparison.Ordinal);

        if (operands != null)
        {
            // nvdisasm's JSON writer leaves a space after special float
            // literals; preserve that quirk for byte-for-byte parity.
            operands = operands
                .Replace("+INF,", "+INF ,", StringComparison.Ordinal)
                .Replace("-INF,", "-INF ,", StringComparison.Ordinal)
                .Replace("QNAN,", "QNAN ,", StringComparison.Ordinal)
                .Replace("] [", "][", StringComparison.Ordinal);
            if (operands.EndsWith("+INF", StringComparison.Ordinal) ||
                operands.EndsWith("-INF", StringComparison.Ordinal) ||
                operands.EndsWith("QNAN", StringComparison.Ordinal))
                operands += " ";

            if (operands.StartsWith("`(", StringComparison.Ordinal) &&
                operands.EndsWith(")", StringComparison.Ordinal))
            {
                operands = operands[2..^1];
            }
            else if (opcode.StartsWith("LD", StringComparison.Ordinal) &&
                operands.EndsWith("]", StringComparison.Ordinal))
            {
                int comma = operands.IndexOf(',');
                if (comma > 0 && comma + 1 < operands.Length &&
                    operands[comma + 1] == '[')
                    operands = operands[..comma] + operands[(comma + 1)..];
            }
            else if (opcode.StartsWith("RET", StringComparison.Ordinal))
            {
                int immediate = operands.IndexOf(" 0x", StringComparison.Ordinal);
                if (immediate > 0)
                    operands = operands[..immediate] + "," +
                        operands[(immediate + 1)..];
            }
            else if ((opcode.StartsWith("ATOM", StringComparison.Ordinal) ||
                opcode.StartsWith("RED", StringComparison.Ordinal) ||
                opcode.StartsWith("QSPC", StringComparison.Ordinal)) &&
                operands.Contains(",["))
            {
                int comma = operands.IndexOf(",[", StringComparison.Ordinal);
                operands = operands[..comma] + operands[(comma + 1)..];
            }
            else if (opcode == "P2R" && predicate == null)
            {
                string[] parts = operands.Split(',');
                int pr = Array.IndexOf(parts, "PR");
                if (pr >= 0)
                {
                    predicate = "PR";
                    operands = string.Join(",",
                        parts.Where((_, index) => index != pr));
                }
            }
            else if (opcode == "BRX" && !operands.Contains(','))
            {
                // nvdisasm emits the relative jump-table offset as a second
                // BRX operand.  When a nearby constant-bank relocation
                // establishes the jump-table base it resolves the table
                // against the function symbol, and its JSON writer keeps only
                // the trailing +0x10 table offset.  Otherwise the encoded
                // sImm*4 value is used directly.
                if (HasNearBrxTableRelocation(section, item.Offset))
                {
                    operands += ",0x10";
                }
                else
                {
                    ulong raw = ((item.Instruction.High & 0x3ffffUL) << 30) |
                        ((item.Instruction.Low >> 34) & 0x3fffffffUL);
                    long immediate = (long)(raw << 16) >> 16;
                    long scaled = immediate * 4;
                    operands += scaled < 0
                        ? $",-0x{-scaled:x}"
                        : $",0x{scaled:x}";
                }
            }
        }

        if (opcode.StartsWith("BREAK", StringComparison.Ordinal) &&
            operands != null)
        {
            string[] parts = operands.Split(',');
            if (parts.Length == 2 && parts[0].StartsWith("P", StringComparison.Ordinal) &&
                parts[1].StartsWith("B", StringComparison.Ordinal))
                operands = parts[1] + "," + parts[0];
        }

        if (relocated == null && operands != null &&
            item.Instruction.BranchTarget is ulong branchTarget &&
            (opcode.StartsWith("CALL", StringComparison.Ordinal) ||
                opcode.StartsWith("RET", StringComparison.Ordinal)))
        {
            string? targetSymbol = null;
            if (IsInstructionAddress(section, branchTarget))
            {
                ulong targetOffset = branchTarget - section.Address;
                foreach (CubinSymbol candidate in section.Functions)
                {
                    ulong offset = candidate.Value >= section.Address
                        ? candidate.Value - section.Address
                        : candidate.Value;
                    if (offset == targetOffset)
                    {
                        targetSymbol = candidate.Name;
                        break;
                    }
                }
            }
            else if (globalSymbols != null)
            {
                globalSymbols.TryGetValue(branchTarget, out targetSymbol);
            }

            if (targetSymbol != null)
            {
                int token = operands.LastIndexOf("0x", StringComparison.Ordinal);
                if (token >= 0)
                {
                    int end = token + 2;
                    while (end < operands.Length &&
                        Uri.IsHexDigit(operands[end]))
                        end++;
                    operands = operands[..token] + targetSymbol +
                        operands[end..];
                }
            }
        }

        var result = new Dictionary<string, object?>();
        if (predicate != null)
            result["predicate"] = predicate;
        result["opcode"] = opcode;
        if (!string.IsNullOrEmpty(operands))
            result["operands"] = operands;
        if (spillRefill)
        {
            result["annotations"] = new[]
            {
                new Dictionary<string, string>
                {
                    ["type"] = "program-analysis",
                    ["info"] = "SpillRefill",
                },
            };
        }

        var attributes = new Dictionary<string, string>();
        if (IsControlFlow(opcode))
            attributes["control-flow"] = "True";
        if (opcode.StartsWith("CALL", StringComparison.Ordinal) &&
            predicate == null)
            attributes["subroutine-call"] = "True";
        if (opcode.StartsWith("BAR", StringComparison.Ordinal))
            attributes["inter-warp-barrier"] = "True";
        if (attributes.Count > 0)
            result["other-attributes"] = attributes;
        return result;
    }

    private static bool IsControlFlow(string opcode) =>
        opcode.StartsWith("BRA", StringComparison.Ordinal) ||
        opcode.StartsWith("CALL", StringComparison.Ordinal) ||
        opcode.StartsWith("RET", StringComparison.Ordinal) ||
        opcode.StartsWith("EXIT", StringComparison.Ordinal) ||
        opcode.StartsWith("BSSY", StringComparison.Ordinal) ||
        opcode.StartsWith("BSYNC", StringComparison.Ordinal) ||
        opcode.StartsWith("BMOV", StringComparison.Ordinal) ||
        opcode.StartsWith("JMP", StringComparison.Ordinal) ||
        opcode.StartsWith("BRX", StringComparison.Ordinal) ||
        opcode.StartsWith("BPT", StringComparison.Ordinal) ||
        opcode.StartsWith("BREAK", StringComparison.Ordinal) ||
        opcode.StartsWith("WARPSYNC", StringComparison.Ordinal) ||
        opcode.StartsWith("YIELD", StringComparison.Ordinal) ||
        opcode.StartsWith("NANOSLEEP", StringComparison.Ordinal);

    private static void WriteCubinJson(
        string path, IReadOnlyList<CubinFile> cubins, IReadOnlyList<FatbinFile>? fatbins,
        string? functionFilter)
    {
        foreach (CubinFile cubin in cubins)
        {
            if (cubin.Toolkit == null)
                UseCubinArchitecture(cubin);
        }
        var sections = new List<object>();
        foreach (CubinFile cubin in cubins)
        {
            foreach (CubinSection section in cubin.Sections)
            {
                if (SectionMatches(section, functionFilter))
                    sections.Add(BuildSectionJson(
                        section, cubin.Toolkit ?? _toolkit));
            }
        }

        string architecture = ArchitectureLabel(cubins);
        object document;
        if (fatbins == null || fatbins.Count == 0)
        {
            document = new
            {
                architecture,
                toolkit = ToolkitLabel(cubins),
                file = Path.GetFileName(path),
                sections,
            };
        }
        else
        {
            document = new
            {
                architecture,
                toolkit = ToolkitLabel(cubins),
                file = Path.GetFileName(path),
                fatbin = new
                {
                    version = fatbins[0].Version,
                    containers = fatbins.Count,
                    entries = fatbins.SelectMany(f => f.Entries).Select(e => new
                    {
                        kind = e.KindName,
                        arch = FatbinEntryArchitecture(e),
                        identifier = e.Identifier,
                        ptxasOptions = e.PtxasOptions,
                        compression = e.Compression,
                        storedPayloadSize = e.StoredPayloadSize,
                        compressedSize = e.CompressedSize,
                        uncompressedSize = e.UncompressedSize > 0
                            ? e.UncompressedSize
                            : (ulong)e.Payload.Length,
                    }),
                },
                sections,
            };
        }

        Console.WriteLine(JsonOutput.Serialize(document, indented: true));
    }

    private static object BuildSectionJson(
        CubinSection section, string? toolkit = null)
    {
        var instructions = new List<object>();
        foreach (var item in DecodeSection(section, toolkit))
            instructions.Add(ToJson(item.Instruction));

        return new
        {
            name = section.Name,
            address = $"0x{section.Address:x}",
            size = $"0x{section.Size:x}",
            functions = section.Functions.Select(f => new
            {
                name = f.Name,
                index = f.Index,
                address = $"0x{f.Value:x}",
                size = $"0x{f.Size:x}",
            }),
            instructions,
        };
    }

    private static List<(int Offset, ulong Address, DecodedInstruction Instruction)>
        DecodeSection(CubinSection section, string? toolkit = null)
    {
        var decoded = new List<(int Offset, ulong Address, DecodedInstruction Instruction)>();
        string? effectiveToolkit = toolkit ?? _toolkit;
        if (section.PreVolta)
        {
            PreVoltaDecoder.UseArchitecture(
                section.ArchitectureKey, effectiveToolkit);
            foreach (var (address, instruction) in
                PreVoltaDecoder.DecodeSection(section.Data, section.Address))
            {
                int offset = (int)(address - section.Address);
                decoded.Add((offset, address, instruction));
            }
            // nvdisasm does not print trailing bundle padding.  CUDA 10.2
            // emits it either as NOP or as the all-zero MOV RZ, RZ form.
            // BRX sections retain their trailing slots because the indirect
            // branch table dataflow keeps them visible.
            if (!decoded.Any(item => item.Instruction.Mnemonic.StartsWith(
                    "BRX", StringComparison.OrdinalIgnoreCase)))
            {
                while (decoded.Count > 0 &&
                    (decoded[^1].Instruction.Mnemonic == "NOP" ||
                     decoded[^1].Instruction.Text == "MOV RZ, RZ"))
                {
                    decoded.RemoveAt(decoded.Count - 1);
                }
            }
            return decoded;
        }

        InstructionDecoder.UseArchitecture(
            section.ArchitectureKey, effectiveToolkit);
        int instructionCount = (int)(section.Size / 16);
        for (int index = 0; index < instructionCount; index++)
        {
            int offset = index * 16;
            ulong low = BitConverter.ToUInt64(section.Data, offset);
            ulong high = BitConverter.ToUInt64(section.Data, offset + 8);
            ulong address = section.Address + (ulong)offset;
            decoded.Add((offset, address, InstructionDecoder.Decode(low, high, address)));
        }
        return decoded;
    }

    private static bool IsElf(ReadOnlySpan<byte> data) =>
        data.Length >= 4 && data[0] == 0x7F && data[1] == (byte)'E' &&
        data[2] == (byte)'L' && data[3] == (byte)'F';

    private static bool SectionMatches(CubinSection section, string? functionFilter)
    {
        if (_functionIndexes != null)
            return section.Functions.Any(
                function => _functionIndexes.Contains(function.Index));
        if (functionFilter == null)
            return true;
        if (section.Name.Contains(functionFilter, StringComparison.OrdinalIgnoreCase) ||
            section.DisplayName.Contains(functionFilter, StringComparison.OrdinalIgnoreCase) ||
            section.LabelName.Contains(functionFilter, StringComparison.OrdinalIgnoreCase))
            return true;
        return section.Functions.Any(f =>
            f.Name.Contains(functionFilter, StringComparison.OrdinalIgnoreCase) ||
            f.OriginalName.Contains(functionFilter, StringComparison.OrdinalIgnoreCase));
    }

    private static void NoMatchingSections(string path, string? functionFilter)
    {
        string filter = _functionIndexes != null
            ? "symbol index " + string.Join(
                ",", _functionIndexes.OrderBy(index => index))
            : $"function '{functionFilter}'";
        Console.Error.WriteLine(
            $"'{path}' has no section matching {filter}.");
    }

    private static bool IsInstructionAddress(CubinSection section, ulong address) =>
        address >= section.Address &&
        address < section.Address + section.Size &&
        (address - section.Address) % (section.PreVolta ? 8UL : 16UL) == 0;

    // PreVolta bundles are 64 bytes: seven 8-byte instructions followed
    // by an 8-byte control word.  A PBK/BRK target can encode the
    // control-word slot, which carries no instruction to attach a label
    // to.  Move it to the first instruction of the next bundle, which is
    // where the branch actually resumes.
    private static ulong SnapToInstructionAddress(
        CubinSection section, ulong address)
    {
        if (!section.PreVolta || address < section.Address)
            return address;
        return (address - section.Address) % 64 == 0
            ? address + 8
            : address;
    }

    internal static bool TryGetR1Local(
        string text, out string key, out int width, out bool store)
    {
        key = "";
        width = 0;
        store = false;
        int start = 0;
        if (text.StartsWith("@", StringComparison.Ordinal))
        {
            int guardEnd = text.IndexOf(' ');
            if (guardEnd < 0)
                return false;
            start = guardEnd + 1;
        }

        int opcodeEnd = text.IndexOf(' ', start);
        string opcode = opcodeEnd < 0
            ? text[start..]
            : text[start..opcodeEnd];
        if (opcode.StartsWith("STL", StringComparison.Ordinal))
            store = true;
        else if (!opcode.StartsWith("LDL", StringComparison.Ordinal))
            return false;

        width = opcode.Contains(".128", StringComparison.Ordinal)
            ? 128
            : opcode.Contains(".64", StringComparison.Ordinal) ? 64 : 32;

        int open = text.IndexOf('[', start);
        if (open < 0 || open + 3 >= text.Length)
            return false;
        if (text[open + 1] != 'R' || text[open + 2] != '1')
            return false;
        char next = text[open + 3];
        if (next is not (']' or '+' or '.'))
            return false;

        int close = text.IndexOf(']', open + 3);
        if (close < 0)
            return false;
        key = text[(open + 3)..close];
        return true;
    }

    private static bool HasNearBrxTableRelocation(
        CubinSection section, int instructionOffset) =>
        section.Relocations.Any(
            r => (r.Type is 64 or 66) &&
                r.Offset <= (ulong)instructionOffset &&
                (ulong)instructionOffset - r.Offset <= 0x80);

    private static ulong FunctionStartAt(CubinSection section, ulong address)
    {
        ulong start = 0;
        foreach (CubinSymbol symbol in section.Functions)
        {
            ulong symbolAddress = symbol.Value >= section.Address
                ? symbol.Value
                : section.Address + symbol.Value;
            if (symbolAddress > address)
                break;
            start = symbolAddress;
        }
        return start;
    }

    private static bool IsSpillRefill(
        CubinSection section,
        (int Offset, ulong Address, DecodedInstruction Instruction) item,
        HashSet<(ulong, string)> nonSpillSlots)
    {
        if (!TryGetR1Local(
                item.Instruction.Text, out string key, out _, out _))
            return false;
        return !nonSpillSlots.Contains(
            (FunctionStartAt(section, item.Address), key));
    }

    internal static HashSet<(ulong Function, string Offset)> BuildNonSpillSlots(
        CubinSection section,
        List<(int Offset, ulong Address, DecodedInstruction Instruction)> decoded)
    {
        var parsed = new List<(
            ulong Function, string Key, int Width, bool Store, bool LastUse,
            ulong Address)>();
        foreach (var item in decoded)
        {
            if (!TryGetR1Local(
                    item.Instruction.Text, out string key, out int width,
                    out bool store))
                continue;
            parsed.Add((
                FunctionStartAt(section, item.Address), key, width, store,
                item.Instruction.Text.Contains("LDL.LU", StringComparison.Ordinal),
                item.Address));
        }

        // Slots that are only copied, zero-initialized, or used as local
        // arrays are not SpillRefill.  The observed nvdisasm pattern is:
        // vector spills are excluded, adjacent non-last-use copies are
        // excluded, and 64-bit slots with either no loads or balanced
        // multi-access traffic are local storage rather than spills.
        var slots = new Dictionary<(ulong, string),
            (bool Has128, bool Adjacent, bool HasLastUse, bool All64,
            int Stores, int Loads)>();
        for (int i = 0; i < parsed.Count; i++)
        {
            var current = parsed[i];
            var slot = (current.Function, current.Key);
            bool exists = slots.TryGetValue(slot, out var info);
            if (!exists)
                info.All64 = true;
            info.Has128 |= current.Width == 128;
            info.HasLastUse |= current.LastUse;
            info.All64 &= current.Width == 64;
            if (current.Store)
                info.Stores++;
            else
                info.Loads++;
            if (i + 1 < parsed.Count)
            {
                var next = parsed[i + 1];
                if (current.Store && !next.Store &&
                    current.Function == next.Function &&
                    current.Key == next.Key &&
                    next.Address == current.Address + 16)
                    info.Adjacent = true;
            }
            slots[slot] = info;
        }

        var hasVectorLocal = new HashSet<ulong>();
        var hasLastUse = new HashSet<ulong>();
        foreach (var pair in slots)
        {
            if (pair.Value.Has128)
                hasVectorLocal.Add(pair.Key.Item1);
            if (pair.Value.HasLastUse)
                hasLastUse.Add(pair.Key.Item1);
        }

        var slotCountByFunction = new Dictionary<ulong, int>();
        foreach (var pair in slots)
        {
            slotCountByFunction.TryGetValue(pair.Key.Item1, out int count);
            slotCountByFunction[pair.Key.Item1] = count + 1;
        }

        var baseNonSpill = new Dictionary<(ulong, string), bool>();
        var hasSpill = new HashSet<ulong>();
        var balanced64ByFunction =
            new Dictionary<ulong, List<(ulong, string)>>();
        foreach (var pair in slots)
        {
            var info = pair.Value;
            bool adjacentSingle = !info.Has128 && info.Adjacent &&
                !info.HasLastUse && info.Stores == 1 && info.Loads == 1;
            bool storesHeavy = !info.Has128 && !info.HasLastUse &&
                info.Stores > info.Loads &&
                !(info.Loads == 0 && info.All64);
            bool ambiguous = adjacentSingle || storesHeavy;
            bool vectorlessLocalFunction =
                !hasVectorLocal.Contains(pair.Key.Item1) &&
                !hasLastUse.Contains(pair.Key.Item1) &&
                slotCountByFunction[pair.Key.Item1] >= 10;
            bool adjacentCopy = info.Adjacent && !info.HasLastUse &&
                (adjacentSingle || info.Loads > info.Stores + 1);
            bool nonSpill = vectorlessLocalFunction || info.Has128 ||
                (info.Stores == 0 && !info.HasLastUse) ||
                (info.Loads == 0 && info.Stores >= 2) ||
                (info.Loads == 0 && info.All64) ||
                adjacentCopy ||
                (!info.HasLastUse && info.All64 &&
                    (info.Loads == 0 ||
                        (info.Stores == info.Loads && info.Stores >= 2)));

            // Functions that keep vector-sized local arrays but have no
            // last-use refills treat their other multi-access stack traffic
            // as local storage, not spill slots.
            if (!nonSpill && info.Stores >= 2 &&
                hasVectorLocal.Contains(pair.Key.Item1) &&
                !hasLastUse.Contains(pair.Key.Item1))
                nonSpill = true;

            if (ambiguous)
                nonSpill = true;
            baseNonSpill[pair.Key] = nonSpill;

            if (!nonSpill && !ambiguous)
                hasSpill.Add(pair.Key.Item1);

            if (!info.Has128 && !info.Adjacent && !info.HasLastUse &&
                info.All64 && info.Stores == 2 && info.Loads == 2)
            {
                if (!balanced64ByFunction.TryGetValue(
                        pair.Key.Item1, out var list))
                {
                    list = new List<(ulong, string)>();
                    balanced64ByFunction[pair.Key.Item1] = list;
                }
                list.Add(pair.Key);
            }
        }

        foreach (var pair in balanced64ByFunction)
        {
            if (pair.Value.Count < 3 || !hasSpill.Contains(pair.Key))
                continue;
            foreach (var key in pair.Value)
                baseNonSpill[key] = false;
        }

        var result = new HashSet<(ulong, string)>();
        foreach (var pair in baseNonSpill)
        {
            var info = slots[pair.Key];
            bool adjacentSingle = !info.Has128 && info.Adjacent &&
                !info.HasLastUse && info.Stores == 1 && info.Loads == 1;
            bool storesHeavy = !info.Has128 && !info.HasLastUse &&
                info.Stores > info.Loads &&
                !(info.Loads == 0 && info.All64);
            bool ambiguous = adjacentSingle || storesHeavy;
            bool vectorlessLocalFunction =
                !hasVectorLocal.Contains(pair.Key.Item1) &&
                !hasLastUse.Contains(pair.Key.Item1) &&
                slotCountByFunction[pair.Key.Item1] >= 10;
            bool functionHasOnlySlot =
                slotCountByFunction[pair.Key.Item1] == 1;
            bool forceSpill = !vectorlessLocalFunction &&
                ((ambiguous && hasSpill.Contains(pair.Key.Item1)) ||
                    (adjacentSingle && functionHasOnlySlot));
            bool nonSpill = pair.Value && !forceSpill;
            if (nonSpill)
                result.Add(pair.Key);
        }
        return result;
    }

    private static string PreservedRelocationAnnotations(
        CubinSection section, int instructionOffset)
    {
        var relocations = section.Relocations
            .Where(r => r.Offset == (ulong)instructionOffset)
            .ToList();
        if (relocations.Count == 0)
            return "";

        var text = new StringBuilder();
        foreach (CubinRelocation relocation in relocations)
        {
            string symbol = relocation.SymbolName.Length > 0
                ? relocation.SymbolName
                : "_";
            // This annotation deliberately uses text that nvdisasm does not
            // emit.  The assembler consumes it to restore relocations whose
            // operand syntax is not represented in a normal listing.
            text.Append(" (*\"")
                .Append(relocation.IsRel ? "RELOCREL" : "RELOC")
                .Append(' ')
                .Append(relocation.Type)
                .Append(' ')
                .Append(symbol)
                .Append(' ')
                .Append(relocation.Addend)
                .Append("\"*)");
        }
        return text.ToString();
    }

    internal static string? TryApplyRelocation(
        CubinSection section, int instructionOffset, string text,
        Dictionary<(CubinSection, ulong), string> labels,
        bool numericTargets = false, bool nvJsonStyle = false)
    {
        var candidates = section.Relocations
            .Where(r => r.Offset == (ulong)instructionOffset)
            .ToList();
        if (candidates.Count == 0)
            return null;

        CubinRelocation? relocation;
        if (text.Contains("CALL", StringComparison.Ordinal))
            relocation = candidates.FirstOrDefault(r => r.Type == 58)
                ?? candidates[0];
        else if (Regex.IsMatch(text, @"c\[0x[0-9a-fA-F]+\]\[U?R"))
            relocation = candidates.FirstOrDefault(r => r.Type == 66)
                ?? candidates[0];
        else if (text.Contains("c[0x0][0x0]", StringComparison.Ordinal))
            relocation = candidates.FirstOrDefault(r => r.Type == 64)
                ?? candidates[0];
        else
            relocation = candidates.FirstOrDefault(r => r.Type is 56 or 57)
                ?? candidates[0];

        if (text.Contains("CALL", StringComparison.Ordinal))
        {
            int comma = text.LastIndexOf(',');
            int space = text.LastIndexOf(' ');
            int separator = Math.Max(comma, space);
            if (separator < 0)
                return null;
            string tail = text[(separator + 1)..].Trim();
            // Register-indirect absolute calls can also carry CUDA
            // relocations; those relocations annotate the call site, not a
            // symbolic replacement for the register operand.  nvdisasm still
            // prints the relocation target after the register.
            if (Regex.IsMatch(tail, @"^(?:U?R\d+|RZ|URZ)$",
                    RegexOptions.IgnoreCase))
                return text + " `(" + relocation.SymbolName + ")";
            return text[..(separator + 1)] +
                "`(" + relocation.SymbolName + ")";
        }

        if (relocation.Type is 55 or 59)
        {
            string symbol = relocation.SymbolName.Length > 0
                ? relocation.SymbolName
                : "smem_full";
            int comma = text.LastIndexOf(',');
            if (comma >= 0)
                return text[..(comma + 1)] +
                    (nvJsonStyle ? symbol : " `(" + symbol + ")");
            int space = text.LastIndexOf(' ');
            if (space >= 0)
                return text[..(space + 1)] +
                    (nvJsonStyle ? symbol : "`(" + symbol + ")");
        }

        if (relocation.Type == 74 &&
            (text.Contains("LDS", StringComparison.Ordinal) ||
                text.Contains("STS", StringComparison.Ordinal)))
        {
            int open = text.LastIndexOf('[');
            int close = open < 0 ? -1 : text.IndexOf(']', open + 1);
            if (open >= 0 && close > open + 1)
            {
                string baseRegister = text[(open + 1)..close];
                string symbol = relocation.SymbolName.Length > 0
                    ? relocation.SymbolName
                    : "shrdMem32";
                string replacement = nvJsonStyle
                    ? relocation.Addend == 0
                        ? $"{baseRegister}+,{symbol}"
                        : $"{baseRegister}+0x{relocation.Addend:x}"
                    : relocation.Addend == 0
                        ? $"{baseRegister}+`({symbol})"
                        : $"{baseRegister}+`(({symbol} + 0x{relocation.Addend:x}))";
                return text[..(open + 1)] + replacement + text[close..];
            }
        }

        if (nvJsonStyle && relocation.Type == 64)
        {
            const string bank = "c[0x0][";
            int start = text.IndexOf(bank, StringComparison.Ordinal);
            if (start >= 0)
            {
                int close = text.IndexOf(']', start + bank.Length);
                if (close >= start + bank.Length)
                {
                    long value = (long)relocation.SymbolValue +
                        relocation.Addend;
                    string offset = value < 0
                        ? $"-0x{-value:x}"
                        : $"0x{value:x}";
                    return text[..start] + "c[," + offset + "]" +
                        text[(close + 1)..];
                }
            }
        }

        if (nvJsonStyle && relocation.Type == 66)
        {
            const string bank = "c[0x0][";
            int start = text.IndexOf(bank, StringComparison.Ordinal);
            if (start >= 0)
            {
                int close = text.IndexOf(']', start + bank.Length);
                if (close >= start + bank.Length)
                {
                    string register = text[(start + bank.Length)..close];
                    string addend = relocation.Addend < 0
                        ? $"+-0x{-relocation.Addend:x}"
                        : $"+0x{relocation.Addend:x}";
                    return text[..start] +
                        $"c[{register}{addend}]" +
                        text[(close + 1)..];
                }
            }
        }

        if (relocation.Type == 64)
        {
            const string zeroBank = "c[0x0][0x0]";
            int index = text.IndexOf(zeroBank, StringComparison.Ordinal);
            if (index >= 0)
            {
                return text[..index] +
                    "c[`(" + relocation.SymbolName + ")]" +
                    text[(index + zeroBank.Length)..];
            }
        }

        if (relocation.Type == 66)
        {
            Match bank = Regex.Match(
                text, @"c\[0x[0-9a-fA-F]+\]\[([^\]]+)\]");
            if (bank.Success)
            {
                string register = bank.Groups[1].Value;
                string addend = relocation.Addend < 0
                    ? $"-0x{-relocation.Addend:x}"
                    : $"0x{relocation.Addend:x}";
                string expression =
                    $"c[{register}+`(({relocation.SymbolName} + {addend}))]";
                return text[..bank.Index] + expression +
                    text[(bank.Index + bank.Length)..];
            }
        }

        if (relocation.Type is 56 or 57)
        {
            string kind = relocation.Type == 56 ? "lo" : "hi";
            ulong target = section.Address + (ulong)relocation.Addend;
            string targetText = numericTargets
                ? $"0x{relocation.Addend:x}"
                : labels.TryGetValue((section, target), out string? label)
                    ? $"{label}@srel"
                    : $"0x{relocation.Addend:x}";
            string expression =
                $"32@{kind}(({relocation.SymbolName} + {targetText}))";
            int comma = text.LastIndexOf(',');
            if (comma >= 0)
                return text[..(comma + 1)] + " " + expression;
        }

        return null;
    }

    private static string FormatInstructionPrefix(ulong address, string text)
    {
        string prefix = "        /*" + address.ToString("x4") + "*/";
        if (text.StartsWith("@", StringComparison.Ordinal))
        {
            int space = text.IndexOf(' ');
            int guardLength = space < 0 ? 0 : space;
            return prefix + new string(' ', Math.Max(1, 18 - guardLength));
        }
        return prefix + new string(' ', 19);
    }

    private static string ReplaceTarget(string text, ulong target, string label)
    {
        string token = "0x" + target.ToString("x", CultureInfo.InvariantCulture);
        int index = text.LastIndexOf(token, StringComparison.Ordinal);
        if (index < 0)
            return text;
        return text[..index] + "`(" + label + ")" + text[(index + token.Length)..];
    }

    private static bool LooksLikeHex(string text)
    {
        text = text.Trim();
        if (text.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
            text = text[2..];
        return text.Length is > 0 and <= 32 && text.All(Uri.IsHexDigit);
    }


    private static int DecodeHex(
        string text, ulong baseAddress, bool emitJson, bool showFields,
        bool showEncoding)
    {
        if (!TryParseU128(text, out UInt128 instruction))
        {
            Console.Error.WriteLine(
                $"'{text}' is not a valid 128-bit hexadecimal instruction.");
            return 1;
        }

        DecodedInstruction decoded = InstructionDecoder.Decode(instruction, baseAddress);

        if (emitJson)
        {
            WriteJson(baseAddress, new List<object> { ToJson(decoded) });
        }
        else
        {
            if (showFields)
            {
                Console.WriteLine($"Opcode: 0x{decoded.Opcode:X3}");
                Console.WriteLine($"Class:  {decoded.ClassName}");
                Console.WriteLine($"Guard:  {decoded.GuardName}" +
                    (decoded.GuardNot ? " (negated)" : ""));
                Console.WriteLine($"Rd:     {decoded.RdName}");
                Console.WriteLine($"Ra:     {decoded.RaName}");
                Console.WriteLine($"Rb:     {decoded.RbName}");
                Console.WriteLine($"Rc:     {decoded.RcName}");
                if (decoded.BranchTarget.HasValue)
                    Console.WriteLine($"Target: 0x{decoded.BranchTarget.Value:X}");
                if (showEncoding)
                    Console.WriteLine(
                        $"Encoding: 0x{decoded.High:x16}{decoded.Low:x16}");
                Console.WriteLine($"Text:   {decoded.Text}");
            }
            else
            {
                string encoding = showEncoding
                    ? $"                 /* 0x{decoded.Low:x16} */"
                    : "";
                Console.WriteLine(
                    FormatInstructionPrefix(baseAddress, decoded.Text) +
                    $"{decoded.Text};{encoding}");
            }
        }

        return 0;
    }

    private static object ToJson(DecodedInstruction instruction) => new
    {
        address = $"0x{instruction.Address:X}",
        low = $"0x{instruction.Low:X16}",
        high = $"0x{instruction.High:X16}",
        text = instruction.Text,
        mnemonic = instruction.Mnemonic,
        opcode = $"0x{instruction.Opcode:X}",
        className = instruction.ClassName,
        valid = instruction.IsValid,
        guard = instruction.GuardName,
        guardNot = instruction.GuardNot,
        rd = instruction.RdName,
        ra = instruction.RaName,
        rb = instruction.RbName,
        rc = instruction.RcName,
        branchTarget = instruction.BranchTarget.HasValue
            ? $"0x{instruction.BranchTarget.Value:X}"
            : null,
    };

    private static void WriteJson(
        ulong baseAddress, List<object> instructions, string architecture = "sm_86")
    {
        var document = new
        {
            architecture,
            baseAddress = $"0x{baseAddress:X}",
            instructions,
        };

        Console.WriteLine(JsonOutput.Serialize(document, indented: true));
    }

    private static string ElfTypeName(ushort elfType) =>
        elfType switch
        {
            1 => "ET_REL",
            2 => "ET_EXEC",
            3 => "ET_DYN",
            4 => "ET_CORE",
            _ => $"ET_{elfType}",
        };

    private static bool IsArchitectureToken(string text) =>
        text.StartsWith("sm", StringComparison.OrdinalIgnoreCase) &&
        text.Length > 2;

    private static bool TryParseArchitecture(string text, out string architecture)
    {
        string normalized = new string(
            text.Where(char.IsLetterOrDigit).ToArray()).ToLowerInvariant();
        architecture = "sm_86";
        if (!normalized.StartsWith("sm", StringComparison.Ordinal))
            return false;

        string body = normalized[2..];
        int digits = 0;
        while (digits < body.Length && char.IsDigit(body[digits]))
            digits++;
        if (digits == 0)
            return false;

        string key = "sm_" + body[..digits] + body[digits..];
        if (!IsSupportedArchitecture(key))
            return false;
        architecture = key;
        return true;
    }

    private static bool TryParseFatbinCompression(
        string text, out FatbinCompression compression)
    {
        switch (text.Trim().ToLowerInvariant())
        {
            case "none": compression = FatbinCompression.None; return true;
            case "deflate":
            case "zlib": compression = FatbinCompression.Deflate; return true;
            case "lz4": compression = FatbinCompression.Lz4; return true;
            case "lz4v2": compression = FatbinCompression.Lz4V2; return true;
            case "zstd": compression = FatbinCompression.Zstd; return true;
            default: compression = FatbinCompression.None; return false;
        }
    }

    private static bool IsSupportedArchitecture(uint architecture) =>
        architecture is 30 or 32 or 35 or 37 or 50 or 52 or 53 or
            60 or 61 or 62 or
            70 or 72 or 75 or 80 or 86 or 87 or 88 or 89 or 90 or
            100 or 101 or 103 or 107 or 110 or 120 or 121;

    private static bool IsSupportedArchitecture(string architecture) =>
        architecture is "sm_30" or "sm_32" or "sm_35" or "sm_37" or
            "sm_50" or "sm_52" or "sm_53" or
            "sm_60" or "sm_61" or "sm_62" or
            "sm_70" or "sm_72" or "sm_75" or "sm_80" or
            "sm_86" or "sm_87" or "sm_88" or "sm_89" or "sm_90" or "sm_90a" or
            "sm_100" or "sm_100a" or "sm_100f" or
            "sm_101" or "sm_101a" or "sm_101f" or
            "sm_103" or "sm_103a" or "sm_103f" or
            "sm_107" or "sm_107a" or "sm_107f" or
            "sm_110" or "sm_110a" or "sm_110f" or
            "sm_120" or "sm_120a" or "sm_120f" or
            "sm_121" or "sm_121a" or "sm_121f";

    private static int LayoutId(uint architecture) =>
        architecture >= 100 ? 2 : 4;

    private static uint ArchitectureNumber(string architecture)
    {
        int digits = 3;
        while (digits < architecture.Length &&
            char.IsDigit(architecture[digits]))
            digits++;
        return uint.Parse(
            architecture[3..digits], NumberStyles.None,
            CultureInfo.InvariantCulture);
    }

    private static string ArchitectureName(uint architecture) =>
        architecture == 0 ? "unknown" : $"sm_{architecture}";

    private static string ArchitectureLabel(IReadOnlyList<CubinFile> cubins)
    {
        var architectures = cubins
            .Select(c => c.ArchitectureKey)
            .Where(a => a.Length > 0)
            .Distinct()
            .OrderBy(a => a, StringComparer.Ordinal)
            .ToList();
        return architectures.Count == 0
            ? "unknown"
            : string.Join(",", architectures);
    }

    private static string? CommonArchitecture(IEnumerable<CubinFile> cubins)
    {
        var architectures = cubins
            .Select(c => c.ArchitectureKey)
            .Where(a => a.Length > 0)
            .Distinct()
            .Take(2)
            .ToList();
        return architectures.Count == 1 ? architectures[0] : null;
    }

    private static void WriteTargetLine(string architecture)
    {
        Console.WriteLine($"\t.target\t{architecture}");
        Console.WriteLine();
    }

    private static bool TryParseFunctionIndexes(
        string text, out HashSet<uint>? indexes)
    {
        indexes = null;
        string[] parts = text.Split(',');
        if (parts.Length == 0)
            return false;

        var result = new HashSet<uint>();
        foreach (string part in parts)
        {
            if (!uint.TryParse(
                    part.Trim(), NumberStyles.None,
                    CultureInfo.InvariantCulture, out uint index))
                return false;
            result.Add(index);
        }

        indexes = result;
        return true;
    }

    private static int RunCubin(
        string cubinPath, string? entry,
        IReadOnlyList<string> buffers, IReadOnlyList<string> bufferHexes,
        IReadOnlyList<string> scalars, string? block, bool noBuffer)
    {
        string ptx = Path.Combine(Path.GetTempPath(),
            "rtxdisam_run_" + Guid.NewGuid().ToString("N") + ".ptx");
        try
        {
            int lift = LiftCubin(
                cubinPath, entry, Array.Empty<PtxParameter>(), ptx, null);
            if (lift != 0)
                return lift;
            return RunPtx(
                ptx, entry, buffers, bufferHexes, scalars, block, noBuffer);
        }
        finally
        {
            if (File.Exists(ptx))
                File.Delete(ptx);
        }
    }

    private static int RunPtx(
        string ptxPath, string? entry,
        IReadOnlyList<string> buffers, IReadOnlyList<string> bufferHexes,
        IReadOnlyList<string> scalars, string? block, bool noBuffer)
    {
        string? root = FindRepoRoot();
        if (root == null)
        {
            Console.Error.WriteLine("cannot locate the repository root");
            return 1;
        }
        string script = Path.Combine(
            root, "tools", "dump_isa_102", "run_acuda.sh");
        if (!File.Exists(script))
        {
            Console.Error.WriteLine($"missing '{script}'");
            return 1;
        }
        entry ??= ExtractPtxEntry(ptxPath);
        if (string.IsNullOrWhiteSpace(entry))
        {
            Console.Error.WriteLine("cannot determine the PTX entry name");
            return 1;
        }
        bool windows = OperatingSystem.IsWindows();
        var start = new ProcessStartInfo(windows ? "wsl.exe" : "bash")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        if (windows)
        {
            start.ArgumentList.Add("-d");
            start.ArgumentList.Add("Ubuntu");
            start.ArgumentList.Add("--");
            // wsl.exe needs an explicit shell; on Linux the process already
            // is bash and adding this would run `bash bash <script>`.
            start.ArgumentList.Add("bash");
        }
        start.ArgumentList.Add(ToWslPath(script));
        start.ArgumentList.Add(ToWslPath(ptxPath));
        start.ArgumentList.Add(entry);
        if (noBuffer)
            start.ArgumentList.Add("--no-buffer");
        foreach (string buffer in buffers)
        {
            start.ArgumentList.Add("--buffer");
            start.ArgumentList.Add(buffer);
        }
        foreach (string bufferHex in bufferHexes)
        {
            start.ArgumentList.Add("--buffer-hex");
            start.ArgumentList.Add(bufferHex);
        }
        if (!string.IsNullOrWhiteSpace(block))
        {
            start.ArgumentList.Add("--block");
            start.ArgumentList.Add(block);
        }
        foreach (string scalar in scalars)
        {
            start.ArgumentList.Add("--scalar");
            start.ArgumentList.Add(scalar);
        }
        using Process process = Process.Start(start)
            ?? throw new InvalidOperationException("cannot start the ACuda runner");
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        Console.Write(stdout);
        Console.Error.Write(stderr);
        return process.ExitCode;
    }

    private static string? FindRepoRoot()
    {
        foreach (string start in new[]
        {
            AppContext.BaseDirectory,
            Directory.GetCurrentDirectory(),
        })
        {
            DirectoryInfo? directory = new(start);
            while (directory != null)
            {
                if (File.Exists(Path.Combine(
                        directory.FullName, "tools", "dump_isa_102",
                        "run_acuda.sh")))
                    return directory.FullName;
                directory = directory.Parent;
            }
        }
        return null;
    }

    private static string ExtractPtxEntry(string path)
    {
        foreach (string line in File.ReadLines(path))
        {
            Match match = Regex.Match(line,
                @"\.visible\s+\.entry\s+([^\s(]+)");
            if (match.Success)
                return match.Groups[1].Value;
        }
        return "";
    }

    private static string ToWslPath(string path)
    {
        string full = Path.GetFullPath(path);
        if (full.Length >= 2 && full[1] == ':')
        {
            char drive = char.ToLowerInvariant(full[0]);
            return "/mnt/" + drive + full[2..].Replace('\\', '/');
        }
        return full.Replace('\\', '/');
    }

    private static int LiftCubin(
        string cubinPath, string? entry, IReadOnlyList<PtxParameter> explicitParams,
        string? output, string? architecture, bool emitFunctions = false)
    {
        string cuda = Environment.GetEnvironmentVariable("CUDA102")
            ?? @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2";
        string nvdisasm = Path.Combine(cuda, "bin", "nvdisasm.exe");
        string cuobjdump = Path.Combine(cuda, "bin", "cuobjdump.exe");
        if (!File.Exists(nvdisasm) || !File.Exists(cuobjdump))
        {
            Console.Error.WriteLine(
                "CUDA 10.2 nvdisasm/cuobjdump were not found; set CUDA102.");
            return 1;
        }
        string sass = Path.Combine(Path.GetTempPath(),
            "rtxdisam_lift_" + Guid.NewGuid().ToString("N") + ".sass");
        try
        {
            bool hasRelocations = false;
            try
            {
                CubinFile probe = CubinFile.Load(cubinPath);
                hasRelocations = probe.Sections.Any(
                    section => section.Relocations.Count > 0);
            }
            catch
            {
                // Fall back to nvdisasm for containers it can read directly.
            }
            string sassText = hasRelocations
                ? ContainerToText(
                    File.ReadAllBytes(cubinPath), cubinPath, null, false, true)
                : RunNvdisasm(nvdisasm, cubinPath);
            File.WriteAllText(sass, sassText);
            string cuobjdumpElf = RunTool(cuobjdump, "-elf", cubinPath);
            IReadOnlyList<PtxParameter> parameters = explicitParams.Count > 0
                ? explicitParams
                : ParseCubinParameters(cuobjdumpElf);
            int sharedBytes = 0;
            try
            {
                Match shared = Regex.Match(
                    RunTool(cuobjdump, "-res-usage", cubinPath),
                    @"SHARED:(\d+)");
                if (shared.Success)
                {
                    sharedBytes = int.Parse(shared.Groups[1].Value,
                        CultureInfo.InvariantCulture);
                }
            }
            catch
            {
                // Shared size is optional; raw listings have no metadata.
            }
            Func<int, uint?>? constantResolver = null;
            Func<int, uint?>? constant2Resolver = null;
            int constant2Bytes = 0;
            try
            {
                CubinFile cubin = CubinFile.Load(cubinPath);
                string? constantEntry = entry;
                if (constantEntry == null)
                {
                    constantEntry = cubin.Sections
                        .FirstOrDefault(s => s.Name.StartsWith(
                            ".nv.constant0.", StringComparison.Ordinal))
                        ?.Name[".nv.constant0.".Length..];
                }
                if (constantEntry == null)
                {
                    constantEntry = cubin.Sections
                        .FirstOrDefault(s => s.Name.StartsWith(
                            ".text.", StringComparison.Ordinal))
                        ?.Name[".text.".Length..];
                }
                CubinSection? constants = constantEntry == null
                    ? null
                    : cubin.Sections.FirstOrDefault(s => s.Name.Equals(
                        ".nv.constant0." + constantEntry,
                        StringComparison.Ordinal));
                if (constants != null)
                {
                    constantResolver = offset =>
                        offset >= 0 && offset + 4 <= constants.Data.Length
                            ? BinaryPrimitives.ReadUInt32LittleEndian(
                                constants.Data.AsSpan(offset))
                            : null;
                }
                List<uint>? constant2Words = constantEntry == null
                    ? null
                    : ParseConstant2Words(cuobjdumpElf, constantEntry);
                if (constant2Words != null)
                {
                    constant2Bytes = constant2Words.Count * 4;
                    constant2Resolver = offset =>
                        offset >= 0 && offset % 4 == 0 &&
                        offset / 4 < constant2Words.Count
                            ? constant2Words[offset / 4]
                            : null;
                }
            }
            catch
            {
                // Constant resolution is optional; keep the explicit-param path.
            }
            return LiftPtx(
                sass, entry, parameters, output, architecture, constantResolver,
                sharedBytes, constant2Resolver, constant2Bytes, emitFunctions);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 1;
        }
        finally
        {
            if (File.Exists(sass))
                File.Delete(sass);
        }
    }

    private static string RunNvdisasm(string executable, string path)
    {
        var start = new ProcessStartInfo(executable)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        start.ArgumentList.Add("-c");
        start.ArgumentList.Add(path);
        using Process process = Process.Start(start)
            ?? throw new InvalidOperationException(
                $"cannot start '{executable}'");
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0 &&
            !stderr.Contains("warning", StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException(
                $"{Path.GetFileName(executable)} failed: {stderr.Trim()}");
        return stdout;
    }

    private static string RunTool(string executable, params string[] arguments)
    {
        var start = new ProcessStartInfo(executable)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        foreach (string argument in arguments)
            start.ArgumentList.Add(argument);
        using Process process = Process.Start(start)
            ?? throw new InvalidOperationException(
                $"cannot start '{executable}'");
        string stdout = process.StandardOutput.ReadToEnd();
        string stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0)
            throw new InvalidOperationException(
                $"{Path.GetFileName(executable)} failed: {stderr.Trim()}");
        return stdout;
    }

    private static List<uint>? ParseConstant2Words(
        string cuobjdumpText, string entry)
    {
        Match section = Regex.Match(cuobjdumpText,
            @"\.nv\.constant2\." + Regex.Escape(entry) +
            @"\s*\r?\n(?<body>(?:\s*0x[0-9A-Fa-f]{8}[^\r\n]*(?:\r?\n|$))+)",
            RegexOptions.Multiline);
        if (!section.Success)
            return null;
        var words = new List<uint>();
        foreach (Match word in Regex.Matches(section.Groups["body"].Value,
            @"0x([0-9A-Fa-f]{8})"))
        {
            words.Add(uint.Parse(word.Groups[1].Value,
                NumberStyles.HexNumber, CultureInfo.InvariantCulture));
        }
        return words.Count > 0 ? words : null;
    }

    private static IReadOnlyList<PtxParameter> ParseCubinParameters(
        string cuobjdumpText)
    {
        var parameters = new List<PtxParameter>();
        int baseOffset = 0;
        Match cbank = Regex.Match(cuobjdumpText,
            @"EIATTR_PARAM_CBANK\s+Format:\s+\S+\s+Value:\s+0x([0-9A-Fa-f]+)\s+0x([0-9A-Fa-f]+)",
            RegexOptions.Singleline);
        if (cbank.Success &&
            ulong.TryParse(cbank.Groups[2].Value, NumberStyles.HexNumber,
                CultureInfo.InvariantCulture, out ulong rawBase))
        {
            baseOffset = (int)(rawBase & 0xFFFF);
        }
        MatchCollection matches = Regex.Matches(cuobjdumpText,
            @"EIATTR_KPARAM_INFO\s+Format:\s+\S+\s+Value:\s+Index\s*:\s*0x([0-9A-Fa-f]+)\s+Ordinal\s*:\s*0x([0-9A-Fa-f]+)\s+Offset\s*:\s*0x([0-9A-Fa-f]+)\s+Size\s*:\s*0x([0-9A-Fa-f]+)",
            RegexOptions.Singleline);
        var entries = new List<(int Ordinal, int Offset, int Size)>();
        foreach (Match match in matches)
        {
            int ordinal = int.Parse(match.Groups[2].Value,
                NumberStyles.HexNumber, CultureInfo.InvariantCulture);
            int offset = int.Parse(match.Groups[3].Value,
                NumberStyles.HexNumber, CultureInfo.InvariantCulture);
            int size = int.Parse(match.Groups[4].Value,
                NumberStyles.HexNumber, CultureInfo.InvariantCulture);
            entries.Add((ordinal, offset, size));
        }
        entries.Sort((a, b) => a.Ordinal.CompareTo(b.Ordinal));
        foreach ((int _, int offset, int size) in entries)
        {
            parameters.Add(new PtxParameter(
                "p" + parameters.Count,
                size == 8
                    ? "u64"
                    : (size > 8
                        ? "u64x" + ((size + 7) / 8)
                        : "u32"),
                baseOffset + offset,
                size));
        }
        return parameters;
    }

    private static int LiftPtx(
        string path, string? entry, IReadOnlyList<PtxParameter> parameters,
        string? output, string? architecture,
        Func<int, uint?>? constantResolver = null, int sharedBytes = 0,
        Func<int, uint?>? constant2Resolver = null, int constant2Bytes = 0,
        bool emitFunctions = false)
    {
        try
        {
            SassProgram program = SassParser.Parse(path);
            List<SassSection> allTextSections = program.Sections.Where(s =>
                s.Name.StartsWith(".text", StringComparison.Ordinal)).ToList();
            IEnumerable<SassSection> textSections = allTextSections;
            if (entry != null)
            {
                textSections = textSections.Where(s =>
                    s.Name.Equals(".text." + entry,
                        StringComparison.Ordinal) ||
                    s.Name.Equals(".text", StringComparison.Ordinal));
            }
            SassSection? section = textSections
                .OrderByDescending(s => s.Nodes.Count)
                .FirstOrDefault();
            if (section == null)
            {
                Console.Error.WriteLine("no matching .text section");
                return 1;
            }
            string name = entry ?? "";
            if (string.IsNullOrEmpty(name))
            {
                name = section.Name.StartsWith(".text.", StringComparison.Ordinal)
                    ? section.Name[".text.".Length..]
                    : "kernel";
            }
            string target = architecture ?? "";
            if (string.IsNullOrEmpty(target))
                target = program.Architecture;
            if (string.IsNullOrEmpty(target))
                target = "sm_50";
            string ptx = SassToPtx.Translate(
                section, name!, parameters, target, constantResolver,
                sharedBytes, constant2Resolver, constant2Bytes,
                emitFunctions, allTextSections);
            if (output != null)
                File.WriteAllText(output, ptx);
            else
                Console.Write(ptx);
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 1;
        }
    }

    private static bool TryParsePtxParameter(
        string text, out PtxParameter? parameter)
    {
        parameter = null;
        string[] parts = text.Split(':');
        if (parts.Length != 3 || parts[0].Length == 0)
            return false;
        string type = parts[1].Trim().ToLowerInvariant();
        int size = type switch
        {
            "u64" => 8,
            "u32" or "s32" => 4,
            _ => 0,
        };
        if (size == 0 || !TryParseU64(parts[2], out ulong offset))
            return false;
        parameter = new PtxParameter(
            parts[0].Trim(), type, (int)offset, size);
        return true;
    }

    private static bool TryParseU64(string text, out ulong value)
    {
        text = text.Trim();
        if (text.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
            text = text[2..];
        return ulong.TryParse(text, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out value);
    }

    private static bool TryParseU128(string text, out UInt128 value)
    {
        text = text.Trim();
        if (text.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
            text = text[2..];
        return UInt128.TryParse(
            text, NumberStyles.HexNumber, CultureInfo.InvariantCulture, out value);
    }

    private static void PrintHelp()
    {
        Console.WriteLine("""
            rtxdisam - NVIDIA SM30/SM32/SM35/SM37/SM50/SM52/SM53/SM60/SM61/SM62/SM70/SM72/SM75/SM80/SM86/SM87/SM88/SM89/SM90/SM90a/SM100/SM101/SM103/SM107/SM110/SM120/SM121 SASS disassembler

            Usage:
              rtxdisam --binary [SM70|SM72|SM75|SM80|SM86|SM87|SM88|SM89|SM90|SM90a|SM100[*]|SM101[*]|SM103[*]|SM107[*]|SM110[*]|SM120[*]|SM121[*]] <file> [--toolkit <10.2|13.4>] [--base-address <hex>] [--emit-json]
              rtxdisam -asmfile <file.sass> [-arch <arch>] [--toolkit <10.2|13.4>] [-o <file.cubin>]
              rtxdisam --cubin <file> [--toolkit <10.2|13.4>] [--emit-json]
              rtxdisam --fatbin <file> [--emit-json]
              rtxdisam <cubin-or-fatbin-file> [--emit-json]
              rtxdisam <128-bit-hex> [--base-address <hex>] [--emit-json]

            Options:
              --binary [SM70|SM72|SM75|SM80|SM86|SM87|SM88|SM89|SM90|SM90a|SM100|SM101|SM103|SM107|SM110|SM120|SM121] <file>
                                     Decode a raw file containing 16-byte
                                     instructions.  The optional SM75, SM80, SM86,
                                     SM87, SM88, SM89, SM90, SM90a, SM100, SM101, SM103, SM107, SM110,
                                     SM120 or SM121 matches nvdisasm's spelling; SM86 is
                                     the default.
              -asmfile <file.sass>   Assemble SASS IR source into a cubin.
              -arch <arch>           Architecture for -asmfile (defaults to .target).
                                     SM101 is the legacy name for SM110; assembly emits sm_110 and a warning.
              --toolkit <10.2|13.4>  Select the toolkit/ISA profile explicitly.
                                     SM75 defaults to 13.4; use 10.2 for the
                                     legacy table.  Cubin decoding always uses
                                     the ELF ABI; a conflicting --toolkit is
                                     rejected.
              -o <file.cubin>        Output path for -asmfile.
              -emit-ir <file.json>   Write the parsed SassProgram IR as JSON.
              -emit-fatbin <file>    Also write a single-image fatbin.
              --fatbin-compression <none|deflate|lz4|lz4v2|zstd>
                                     Compression for -emit-fatbin.
              --cubin <file>         Decode a cubin or fatbin file.
              --fatbin <file>        Alias for --cubin (fatbin containers are auto-detected).
              -extract-ptx           Extract embedded PTX entries next to the input file
                                     (also accepted as --extract-ptx).
              -extract-cubin         Extract embedded ELF cubin entries next to the input
                                     file (also accepted as --extract-cubin).
              -extract-all           Extract every fatbin entry using a kind-specific
                                     extension (also accepted as --extract-all).
              --output-dir <dir>     Write extracted files into this directory.
              --force                Overwrite existing extracted files.
              --function <name|idx> Only disassemble sections matching a function name
                                     (substring) or comma-separated symbol indexes.
                                     Alias -fun.
              --list                 List fatbin entries, sections and functions
                                     without decoding instructions.
              --print-instruction-encoding
              --hex                  Show the 64-bit encoding after each instruction.
              --base-address <hex>   Address of the first instruction (default 0).
              --emit-json            Emit JSON instead of nvdisasm-style text.
              --nv-json              Emit nvdisasm-shaped function/instruction
                                     JSON (raw binary or single cubin).
              --annotate, -annotate  Include compiler annotations with --nv-json.
              --fields               Show the decoded guard/register fields (hex mode).
              --no-dataflow, -ndf    Disable SpillRefill and BRX dataflow
                                     annotations.
              --annotate-spills      Compatibility alias; SpillRefill
                                     annotations are emitted by default.
              --annotate-brx         Compatibility alias; BRX branch-target
                                     annotations are emitted by default.
              --not-nvdisasm-compatible, -not-nvdisasm-compatible
                                     Emit RELOC annotations for relocation
                                     metadata that nvdisasm text cannot
                                     represent.  The source remains
                                     reassemblable, but the listing is not
                                     nvdisasm-compatible.
              --preserve-relocations Alias for --not-nvdisasm-compatible.
              --version, -V          Show the rtxdisam version.
              --help                 Show this help.
            """);
    }
}
