using System.Buffers.Binary;
using nvDismDecode;

namespace rtxdisam;

[Flags]
public enum RtxDisamOutput
{
    Sass = 1,
    Ptx = 2,
    Both = Sass | Ptx,
}

public sealed record RtxDisamText(string? Sass, string? Ptx);

// Public process-internal API for hosts that embed rtxdisam.
// Cubin and fatbin inputs are accepted in memory.
public static class RtxDisamApi
{
    public static RtxDisamText ContainerToText(
        byte[] data, string? entry = null,
        IReadOnlyList<PtxParameter>? parameters = null,
        RtxDisamOutput output = RtxDisamOutput.Both,
        string? target = null, bool emitFunctions = false,
        int sharedBytes = 0, string? functionFilter = null,
        bool showEncoding = false, string? selectionTarget = null)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (data.Length == 0)
            throw new ArgumentException("input is empty", nameof(data));
        if (output == 0 || (output & ~RtxDisamOutput.Both) != 0)
            throw new ArgumentOutOfRangeException(nameof(output));

        EnsureTables();
        // A container that carries PTX text should be used verbatim when
        // there is no SASS image for the requested device.  Real CUDA
        // JIT-compiles the fatbin's PTX in that case: an sm_10..sm_30
        // cubin is not a match for an sm_86 device, so lifting the old
        // SASS would be both slower and less faithful.  PTX-only fatbins
        // keep the previous behaviour; mixed fatbins now follow the same
        // JIT rule instead of unconditionally preferring an older cubin.
        //
        // ACUDA_RTXDISAM_PTX_FIRST=0 forces the cubin/SASS path even when
        // the fatbin carries PTX.  Large PhysX modules compile far more
        // cheaply per entry than as one multi-megabyte module.
        bool preferPtx = Environment.GetEnvironmentVariable(
            "ACUDA_RTXDISAM_PTX_FIRST") is not ("0" or "false" or "no");
        if (preferPtx && (output & RtxDisamOutput.Ptx) != 0 &&
            FatbinFile.IsFatbin(data) &&
            (!HasCubinMember(data) ||
             !HasExactCubinMember(data, selectionTarget ?? target)))
        {
            string? fatbinPtx = SelectFatbinPtx(data, entry);
            if (fatbinPtx != null)
                return new RtxDisamText(null, fatbinPtx);
        }
        byte[] selectedData = SelectContainerData(data, entry, selectionTarget);
        string? sass = Program.ContainerToText(
            selectedData, "<memory>", functionFilter ?? entry, showEncoding,
            preserveRelocations: (output & RtxDisamOutput.Ptx) != 0);
        string? ptx = null;
        if ((output & RtxDisamOutput.Ptx) != 0)
        {
            IReadOnlyList<PtxParameter> effectiveParameters =
                parameters ?? Array.Empty<PtxParameter>();
            if (effectiveParameters.Count == 0)
                effectiveParameters = ExtractCubinParameters(selectedData, entry);
            if (sharedBytes == 0)
                sharedBytes = ExtractCubinSharedSize(selectedData, entry);
            // Kernel-specific compiler literals: bank 2 holds math
            // constants (sin/cos range reduction), bank 4 launch-size
            // descriptors and function tables.
            byte[]? bank2 = ExtractConstantBank(selectedData, entry, 2);
            byte[]? bank3 = ExtractConstantBank(selectedData, entry, 3);
            byte[]? bank4 = ExtractConstantBank(selectedData, entry, 4);
            IReadOnlyDictionary<int, string>? texrefs =
                ExtractTextureRefs(selectedData, entry);
            IReadOnlyDictionary<int, PtxGlobalImage>? globalImages =
                ExtractConstantBankGlobals(selectedData, entry, 4);
            ptx = SassTextToPtx(
                sass, entry, effectiveParameters,
                target, emitFunctions, sharedBytes,
                MakeWordResolver(bank2), bank2?.Length ?? 0,
                MakeWordResolver(bank4), bank4?.Length ?? 0,
                texrefs, globalImages,
                MakeWordResolver(bank3), bank3?.Length ?? 0);
        }
        if ((output & RtxDisamOutput.Sass) == 0)
            sass = null;
        return new RtxDisamText(sass, ptx);
    }

    public static string CubinToSass(
        byte[] data, string? functionFilter = null,
        bool showEncoding = false)
    {
        return ContainerToText(
            data, output: RtxDisamOutput.Sass,
            functionFilter: functionFilter, showEncoding: showEncoding).Sass!;
    }

    public static string CubinToPtx(
        byte[] data, string? entry = null,
        IReadOnlyList<PtxParameter>? parameters = null,
        string? target = null, bool emitFunctions = false,
        int sharedBytes = 0)
    {
        return ContainerToText(
            data, entry, parameters, RtxDisamOutput.Ptx, target,
            emitFunctions, sharedBytes).Ptx!;
    }

    public static RtxDisamText ContainerToText(
        string path, string? entry = null,
        IReadOnlyList<PtxParameter>? parameters = null,
        RtxDisamOutput output = RtxDisamOutput.Both,
        string? target = null, bool emitFunctions = false,
        int sharedBytes = 0, string? functionFilter = null,
        bool showEncoding = false, string? selectionTarget = null)
    {
        return ContainerToText(
            File.ReadAllBytes(path), entry, parameters, output, target,
            emitFunctions, sharedBytes, functionFilter, showEncoding,
            selectionTarget);
    }

    public static string CubinToSass(
        string path, string? functionFilter = null,
        bool showEncoding = false)
    {
        return ContainerToText(
            path, output: RtxDisamOutput.Sass,
            functionFilter: functionFilter, showEncoding: showEncoding).Sass!;
    }

    public static string CubinToPtx(
        string path, string? entry = null,
        IReadOnlyList<PtxParameter>? parameters = null,
        string? target = null, bool emitFunctions = false,
        int sharedBytes = 0)
    {
        return ContainerToText(
            path, entry, parameters, RtxDisamOutput.Ptx, target,
            emitFunctions, sharedBytes).Ptx!;
    }

    private const string ResourcePrefix = "rtxdisam.isa_json.";
    private static readonly object TableLock = new();
    private static bool _tablesReady;

    private static void EnsureTables()
    {
        if (_tablesReady)
            return;
        lock (TableLock)
        {
            if (_tablesReady)
                return;
            if (!string.IsNullOrEmpty(
                    nvDismDecode.InstructionDecoder.TableDirectoryOverride))
            {
                _tablesReady = true;
                return;
            }

            string? deployed = FindDeployedTableRoot();
            if (deployed != null)
            {
                nvDismDecode.InstructionDecoder.TableDirectoryOverride = deployed;
                _tablesReady = true;
                return;
            }

            var assembly = typeof(RtxDisamApi).Assembly;
            string[] resources = assembly.GetManifestResourceNames()
                .Where(name => name.StartsWith(
                    ResourcePrefix, StringComparison.Ordinal))
                .OrderBy(name => name, StringComparer.Ordinal)
                .ToArray();
            if (resources.Length == 0)
                throw new InvalidOperationException(
                    "rtxdisam embedded ISA tables are missing from this build");

            string signature = assembly.ManifestModule.ModuleVersionId
                .ToString("N");
            if (signature.Length > 16)
                signature = signature[..16];
            string root = Path.Combine(
                Path.GetTempPath(),
                $"rtxdisam-tables-{signature}-{Environment.ProcessId}");
            foreach (string resource in resources)
            {
                string relative = resource[ResourcePrefix.Length..];
                int separator = relative.IndexOf('.');
                if (separator <= 0 || separator == relative.Length - 1)
                    throw new InvalidDataException(
                        $"invalid embedded ISA resource '{resource}'");
                string directory = Path.Combine(
                    root, relative[..separator]);
                string file = Path.Combine(
                    directory, relative[(separator + 1)..]);
                using Stream input = assembly.GetManifestResourceStream(
                    resource) ?? throw new InvalidDataException(
                        $"cannot read embedded ISA resource '{resource}'");
                if (File.Exists(file) &&
                    new FileInfo(file).Length == input.Length)
                    continue;
                Directory.CreateDirectory(directory);
                using var output = new FileStream(
                    file, FileMode.Create, FileAccess.Write,
                    FileShare.Read);
                input.CopyTo(output);
            }

            nvDismDecode.InstructionDecoder.TableDirectoryOverride = root;
            _tablesReady = true;
        }
    }

    private static string? FindDeployedTableRoot()
    {
        foreach (string baseDirectory in new[]
                 {
                     AppContext.BaseDirectory,
                     Environment.CurrentDirectory,
                 })
        {
            string root = Path.Combine(baseDirectory, "isa_json");
            if (Directory.Exists(Path.Combine(root, "isa_102_json")) &&
                Directory.Exists(Path.Combine(root, "isa_134_json")))
                return root;
        }
        return null;
    }

    // Header-only walk of a fatbin's entry list: no payload decompression
    // or copying, so cubin-only containers cost almost nothing.  The layout
    // is the one FatbinFile.Parse reads: kind/version/header_size, then a
    // 64-bit stored size at +8 and the stored payload after the header.
    private static bool HasCubinMember(byte[] data)
    {
        if (data.Length < 16 ||
            BinaryPrimitives.ReadUInt32LittleEndian(data) != FatbinFile.Magic)
            return false;
        ushort headerSize = BinaryPrimitives.ReadUInt16LittleEndian(
            data.AsSpan(6));
        ulong totalSize = BinaryPrimitives.ReadUInt64LittleEndian(
            data.AsSpan(8));
        ulong end = (ulong)headerSize + totalSize;
        if (end > (ulong)data.Length)
            end = (ulong)data.Length;
        ulong offset = headerSize;
        while (offset + 64 <= end)
        {
            ushort kind = BinaryPrimitives.ReadUInt16LittleEndian(
                data.AsSpan((int)offset));
            uint entryHeaderSize = BinaryPrimitives.ReadUInt32LittleEndian(
                data.AsSpan((int)offset + 4));
            ulong storedSize = BinaryPrimitives.ReadUInt64LittleEndian(
                data.AsSpan((int)offset + 8));
            int compressedSize = BinaryPrimitives.ReadInt32LittleEndian(
                data.AsSpan((int)offset + 16));
            if (kind == 2 || kind == 4)
                return true;
            if (entryHeaderSize < 64)
                break;
            ulong advance = entryHeaderSize +
                (ulong)(compressedSize > 0
                    ? compressedSize
                    : (int)storedSize);
            if (advance == 0 || offset + advance <= offset)
                break;
            offset += advance;
        }
        return false;
    }

    // True when the fatbin carries a cubin whose architecture is exactly
    // the requested SM.  An older cubin is deliberately not a match: the
    // caller's device would have JIT-compiled the fatbin's PTX instead of
    // running a mismatched SASS image.
    private static bool HasExactCubinMember(byte[] data, string? target)
    {
        if (string.IsNullOrEmpty(target) || !FatbinFile.IsFatbin(data))
            return false;
        int wanted = ParseTargetArchitecture(target);
        if (wanted <= 0)
            return false;
        try
        {
            FatbinFile fatbin = FatbinFile.Parse(data);
            foreach (FatbinEntry candidate in fatbin.Entries)
            {
                if (candidate.IsCubin &&
                    candidate.Architecture == (uint)wanted)
                    return true;
            }
            return false;
        }
        catch
        {
            return false;
        }
    }

    // Return a fatbin PTX member verbatim.  Prefer the member that names
    // the requested entry, then the first PTX member (matching the C
    // parser's first-member behaviour).
    private static string? SelectFatbinPtx(byte[] data, string? entry)
    {
        try
        {
            FatbinFile fatbin = FatbinFile.Parse(data);
            string? first = null;
            foreach (FatbinEntry candidate in fatbin.Entries)
            {
                if (!candidate.IsPtx || candidate.Payload.Length == 0)
                    continue;
                string text = System.Text.Encoding.UTF8.GetString(
                    candidate.Payload);
                if (string.IsNullOrEmpty(entry) ||
                    text.Contains(entry, StringComparison.Ordinal))
                    return text;
                first ??= text;
            }
            return first;
        }
        catch
        {
            return null;
        }
    }

    // Prefer the fatbin cubin matching the caller's reported compute
    // capability. A target that is not present leaves the previous
    // first-member behavior intact, so architecture-specific fatbins keep
    // working on older devices when only a different SM image is available.
    private static byte[] SelectContainerData(
        byte[] data, string? entry, string? target)
    {
        if (string.IsNullOrEmpty(target) || !FatbinFile.IsFatbin(data))
            return data;
        int wanted = ParseTargetArchitecture(target);
        if (wanted <= 0)
            return data;
        try
        {
            FatbinFile fatbin = FatbinFile.Parse(data);
            byte[]? exact = null;
            byte[]? bestBelow = null;
            byte[]? bestAbove = null;
            uint belowArchitecture = 0;
            uint aboveArchitecture = uint.MaxValue;
            foreach (FatbinEntry candidate in fatbin.Entries)
            {
                if (!candidate.IsCubin)
                    continue;
                if (entry != null &&
                    !ContainsTextSection(candidate.Payload, entry))
                    continue;
                if (candidate.Architecture == (uint)wanted)
                {
                    exact = candidate.Payload;
                    break;
                }
                if (candidate.Architecture == 0)
                    continue;
                if (candidate.Architecture < (uint)wanted &&
                    candidate.Architecture > belowArchitecture)
                {
                    belowArchitecture = candidate.Architecture;
                    bestBelow = candidate.Payload;
                }
                else if (candidate.Architecture > (uint)wanted &&
                         candidate.Architecture < aboveArchitecture)
                {
                    aboveArchitecture = candidate.Architecture;
                    bestAbove = candidate.Payload;
                }
            }
            return exact ?? bestBelow ?? bestAbove ?? data;
        }
        catch
        {
            return data;
        }
    }

    // Static shared-memory size for one entry, from the cubin's
    // `.nv.shared.<entry>` section.  The section is NOBITS; only its
    // size is meaningful.  Kernels choose this themselves (0, 4 KiB,
    // 16 KiB, ...), so it cannot be assumed.
    private static int ExtractCubinSharedSize(byte[] data, string? entry)
    {
        if (string.IsNullOrEmpty(entry))
            entry = null;
        try
        {
            CubinFile? cubin = null;
            if (data.Length >= 4 && data[0] == 0x7f &&
                data[1] == (byte)'E' && data[2] == (byte)'L' &&
                data[3] == (byte)'F')
            {
                cubin = CubinFile.Load(data);
            }
            else if (FatbinFile.IsFatbin(data))
            {
                FatbinFile fatbin = FatbinFile.Parse(data);
                foreach (FatbinEntry candidate in fatbin.Entries)
                {
                    if (!candidate.IsCubin)
                        continue;
                    CubinFile parsed = CubinFile.Load(candidate.Payload);
                    if (entry == null ||
                        parsed.Sections.Any(section => section.Name.Equals(
                            ".text." + entry, StringComparison.Ordinal)))
                    {
                        cubin = parsed;
                        break;
                    }
                    cubin ??= parsed;
                }
            }
            if (cubin == null)
                return 0;
            string want = entry != null ? ".nv.shared." + entry : ".nv.shared";
            if (cubin.SharedSectionSizes.TryGetValue(want, out ulong size))
                return (int)size;
            return 0;
        }
        catch
        {
            return 0;
        }
    }

    private static bool ContainsTextSection(byte[] payload, string entry)
    {
        // Fast pre-filter before the full CubinFile.Load: the section name
        // string table contains ".text.<entry>\0" verbatim.  This avoids
        // parsing every cubin in a multi-architecture fatbin.
        byte[] needle = System.Text.Encoding.ASCII.GetBytes(
            ".text." + entry + "\0");
        return payload.AsSpan().IndexOf(needle) >= 0;
    }

    private static int ParseTargetArchitecture(string target)
    {
        string text = target.Trim();
        if (!text.StartsWith("sm_", StringComparison.OrdinalIgnoreCase))
            return 0;
        int end = 3;
        while (end < text.Length && char.IsDigit(text[end]))
            end++;
        return end > 3 &&
               int.TryParse(text[3..end], out int architecture)
            ? architecture
            : 0;
    }

    private static IReadOnlyList<PtxParameter> ExtractCubinParameters(
        byte[] data, string? entry)
    {
        if (string.IsNullOrEmpty(entry))
            entry = null;
        try
        {
            CubinFile? cubin = null;
            if (data.Length >= 4 && data[0] == 0x7f &&
                data[1] == (byte)'E' && data[2] == (byte)'L' &&
                data[3] == (byte)'F')
            {
                cubin = CubinFile.Load(data);
            }
            else if (FatbinFile.IsFatbin(data))
            {
                FatbinFile fatbin = FatbinFile.Parse(data);
                foreach (FatbinEntry candidate in fatbin.Entries)
                {
                    if (!candidate.IsCubin)
                        continue;
                    CubinFile parsed = CubinFile.Load(candidate.Payload);
                    if (entry == null ||
                        parsed.Sections.Any(section => section.Name.Equals(
                            ".text." + entry, StringComparison.Ordinal)))
                    {
                        cubin = parsed;
                        break;
                    }
                    cubin ??= parsed;
                }
            }
            if (cubin == null)
                return Array.Empty<PtxParameter>();
            return ParseCubinParameters(cubin, entry);
        }
        catch
        {
            return Array.Empty<PtxParameter>();
        }
    }

    private static IReadOnlyList<PtxParameter> ParseCubinParameters(
        CubinFile cubin, string? entry)
    {
        if (string.IsNullOrEmpty(entry))
            entry = null;
        int baseOffset = 0;
        var entries = new List<(int Ordinal, int Offset, int Size)>();
        string selected = ".nv.info." + (entry ?? "");
        bool parsedEntry = false;
        foreach (CubinInfoSection section in cubin.InfoSections)
        {
            if (!section.Name.StartsWith(".nv.info", StringComparison.Ordinal))
                continue;
            bool isGlobal = section.Name.Equals(
                ".nv.info", StringComparison.Ordinal);
            if (!isGlobal)
            {
                if (entry != null && !section.Name.Equals(
                        selected, StringComparison.Ordinal))
                    continue;
                if (entry == null && parsedEntry)
                    continue;
                parsedEntry = true;
            }
            byte[] info = section.Data;
            int offset = 0;
            while (offset + 4 <= info.Length)
            {
                byte format = info[offset];
                byte attribute = info[offset + 1];
                int length = BinaryPrimitives.ReadUInt16LittleEndian(
                    info.AsSpan(offset + 2));
                int record = format is 1 or 2 or 3 ? 4 : 4 + length;
                if (format is not (1 or 2 or 3 or 4) ||
                    record < 4 || offset + record > info.Length)
                    break;
                if (format == 4)
                {
                    ReadOnlySpan<byte> value =
                        info.AsSpan(offset + 4, length);
                    if (attribute == 0x0a && length >= 8)
                        baseOffset = (int)(BinaryPrimitives
                            .ReadUInt32LittleEndian(value[4..]) & 0xffff);
                    else if (attribute == 0x17 && length >= 12)
                    {
                        int ordinal = BinaryPrimitives.ReadUInt16LittleEndian(
                            value[4..]);
                        int paramOffset = BinaryPrimitives.ReadUInt16LittleEndian(
                            value[6..]);
                        int sizeCode = BinaryPrimitives.ReadUInt16LittleEndian(
                            value[10..]);
                        int size = sizeCode >> 2;
                        // By-value structs are not always a multiple of
                        // 8 bytes (e.g. cudnn's 260-byte
                        // GroupedDirectFpropParams).  Keep every record:
                        // dropping the struct shifts all later parameter
                        // offsets and the kernel reads zeros.
                        if (size >= 1)
                            entries.Add((ordinal, paramOffset, size));
                    }
                }
                offset += record;
            }
        }
        entries.Sort((a, b) => a.Ordinal.CompareTo(b.Ordinal));
        var parameters = new List<PtxParameter>(entries.Count);
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

    private static string SassTextToPtx(
        string sass, string? entry,
        IReadOnlyList<PtxParameter> parameters, string? target,
        bool emitFunctions, int sharedBytes,
        Func<int, uint?>? constant2Resolver = null, int constant2Bytes = 0,
        Func<int, uint?>? constant4Resolver = null, int constant4Bytes = 0,
        IReadOnlyDictionary<int, string>? texrefs = null,
        IReadOnlyDictionary<int, PtxGlobalImage>? globalImages = null,
        Func<int, uint?>? constant3Resolver = null, int constant3Bytes = 0)
    {
        string[] lines = sass.Replace("\r", "", StringComparison.Ordinal)
            .Split('\n');
        SassProgram program = SassParser.ParseLines(
            lines, "<cubin>", separateSections: true);
        List<SassSection> sections = program.Sections
            .Where(section => section.Name.StartsWith(
                ".text", StringComparison.Ordinal))
            .Where(section => section.Nodes.Count > 0)
            .ToList();
        List<SassSection> allSections = sections;
        if (!string.IsNullOrEmpty(entry))
        {
            List<SassSection> exact = sections.Where(section =>
                section.Name.Equals(
                    ".text." + entry, StringComparison.Ordinal)).ToList();
            if (exact.Count == 0 && sections.Count == 1 &&
                sections[0].Name.Equals(".text", StringComparison.Ordinal))
                exact = sections;
            if (exact.Count == 0)
                throw new InvalidDataException(
                    $"entry '{entry}' was not found in the decoded text sections");
            sections = exact;
        }
        if (sections.Count == 0)
            throw new InvalidDataException("no .text section was decoded");
        string? duplicate = sections
            .GroupBy(section => section.Name, StringComparer.Ordinal)
            .FirstOrDefault(group => group.Count() > 1)?.Key;
        if (duplicate != null)
            throw new NotSupportedException(
                $"PTX output is ambiguous: multiple '{duplicate}' sections " +
                "were decoded; use SASS output or select a single cubin");
        if (string.IsNullOrEmpty(entry) && sections.Count > 1)
            throw new NotSupportedException(
                "PTX output requires an entry name when the input contains " +
                "multiple text sections");

        SassSection selected = sections
            .OrderByDescending(section => section.Nodes.Count)
            .First();
        string name = entry ?? "";
        if (string.IsNullOrEmpty(name))
        {
            name = selected.Name.StartsWith(
                ".text.", StringComparison.Ordinal)
                ? selected.Name[".text.".Length..]
                : "kernel";
        }
        string selectedTarget = target ?? "";
        if (string.IsNullOrEmpty(selectedTarget))
            selectedTarget = selected.Architecture ?? "";
        if (string.IsNullOrEmpty(selectedTarget))
            selectedTarget = program.Architecture;
        if (string.IsNullOrEmpty(selectedTarget))
            selectedTarget = "sm_50";
        return SassToPtx.Translate(
            selected, name, parameters, selectedTarget, null, sharedBytes,
            constant2Resolver, constant2Bytes, emitFunctions, allSections,
            constant4Resolver, constant4Bytes, texrefs, globalImages,
            constant3Resolver, constant3Bytes);
    }

    private static Func<int, uint?>? MakeWordResolver(byte[]? data)
    {
        if (data == null || data.Length < 4)
            return null;
        return offset => offset >= 0 && offset + 4 <= data.Length
            ? BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset))
            : (uint?)null;
    }

    private static byte[]? ExtractConstantBank(
        byte[] data, string? entry, int bank)
    {
        try
        {
            CubinFile? cubin = LoadCubinForEntry(data, entry);
            if (cubin == null)
                return null;
            string suffix = string.IsNullOrEmpty(entry)
                ? "" : "." + entry;
            IEnumerable<CubinSection> all = cubin.Sections
                .Concat(cubin.DataSections);
            CubinSection? exact = all.FirstOrDefault(
                section => section.Name.Equals(
                    $".nv.constant{bank}{suffix}",
                    StringComparison.Ordinal));
            if (exact != null)
                return exact.Data;
            if (string.IsNullOrEmpty(entry))
            {
                // With no entry name, old cubins still have exactly one
                // per-kernel constant bank.  Prefer it over dropping the
                // bank entirely (which breaks FFMA immediates).
                List<CubinSection> perKernel = all.Where(
                    section => section.Name.StartsWith(
                        $".nv.constant{bank}.",
                        StringComparison.Ordinal)).ToList();
                if (perKernel.Count == 1)
                    return perKernel[0].Data;
            }
            List<CubinSection> globals = all.Where(
                section => section.Name.Equals(
                    $".nv.constant{bank}",
                    StringComparison.Ordinal)).ToList();
            return globals.Count == 1 ? globals[0].Data : null;
        }
        catch
        {
            return null;
        }
    }

    // Resolves `c[bank][offset]` slots that the cubin relocates against
    // module data (e.g. `__cudart_sin_cos_coeffs` in .nv.global.init).
    // The emitted PTX declares the bytes as a `.global` image and
    // materializes the symbol address for the relocated slot.
    private static IReadOnlyDictionary<int, PtxGlobalImage>?
        ExtractConstantBankGlobals(byte[] data, string? entry, int bank)
    {
        try
        {
            CubinFile? cubin = LoadCubinForEntry(data, entry);
            if (cubin == null)
                return null;
            string suffix = string.IsNullOrEmpty(entry)
                ? "" : "." + entry;
            List<CubinSection> all = cubin.Sections
                .Concat(cubin.DataSections)
                .GroupBy(section => section.Index)
                .Select(group => group.First())
                .ToList();
            CubinSection? bankSection = all.FirstOrDefault(
                section => section.Name.Equals(
                    $".nv.constant{bank}{suffix}",
                    StringComparison.Ordinal));
            if (bankSection == null && string.IsNullOrEmpty(entry))
            {
                List<CubinSection> perKernel = all.Where(
                    section => section.Name.StartsWith(
                        $".nv.constant{bank}.",
                        StringComparison.Ordinal)).ToList();
                if (perKernel.Count == 1)
                    bankSection = perKernel[0];
            }
            if (bankSection == null)
            {
                List<CubinSection> modules = all.Where(
                    section => section.Name.Equals(
                        $".nv.constant{bank}",
                        StringComparison.Ordinal)).ToList();
                if (modules.Count == 1)
                    bankSection = modules[0];
            }
            if (bankSection == null || bankSection.Relocations.Count == 0)
                return null;
            var result = new Dictionary<int, PtxGlobalImage>();
            foreach (CubinRelocation relocation in bankSection.Relocations)
            {
                if (string.IsNullOrEmpty(relocation.SymbolName))
                    continue;
                CubinSymbol? symbol = cubin.Symbols.FirstOrDefault(
                    candidate => candidate.Name.Equals(
                        relocation.SymbolName, StringComparison.Ordinal));
                if (symbol == null)
                    continue;
                CubinSection? target = all.FirstOrDefault(
                    section => section.Index == symbol.SectionIndex);
                if (target == null ||
                    !target.Name.Equals(
                        ".nv.global.init", StringComparison.Ordinal) &&
                    !target.Name.Equals(
                        ".nv.global", StringComparison.Ordinal))
                    continue;
                long start = (long)symbol.Value;
                long size = (long)symbol.Size;
                if (start < 0 || size <= 0 ||
                    start + size > target.Data.Length)
                    continue;
                var bytes = new byte[size];
                Array.Copy(target.Data, start, bytes, 0, size);
                result[(int)relocation.Offset] = new PtxGlobalImage(
                    symbol.Name, relocation.Addend, bytes);
            }
            return result.Count == 0 ? null : result;
        }
        catch
        {
            return null;
        }
    }

    private static CubinFile? LoadCubinForEntry(byte[] data, string? entry)
    {
        if (data.Length >= 4 && data[0] == 0x7f &&
            data[1] == (byte)'E' && data[2] == (byte)'L' &&
            data[3] == (byte)'F')
            return CubinFile.Load(data);
        if (!FatbinFile.IsFatbin(data))
            return null;
        FatbinFile fatbin = FatbinFile.Parse(data);
        CubinFile? fallback = null;
        foreach (FatbinEntry candidate in fatbin.Entries)
        {
            if (!candidate.IsCubin)
                continue;
            CubinFile parsed = CubinFile.Load(candidate.Payload);
            fallback ??= parsed;
            if (entry == null ||
                parsed.Sections.Any(section =>
                    section.Name.Equals(
                        ".text." + entry,
                        StringComparison.Ordinal)))
                return parsed;
        }
        return fallback;
    }

    // Recover the cubin's texture descriptors.  sm_30+ encodes the texref
    // as an index into the kernel's constant bank 0; .rel.nv.constant0
    // relocates the matching descriptor word against an
    // STT_CUDA_TEXTURE symbol.  The SASS TLD/TLDS immediate is the
    // descriptor byte offset divided by four.
    private static IReadOnlyDictionary<int, string>? ExtractTextureRefs(
        byte[] data, string? entry)
    {
        try
        {
            CubinFile? cubin = LoadCubinForEntry(data, entry);
            if (cubin == null)
                return null;
            string suffix = string.IsNullOrEmpty(entry)
                ? "" : "." + entry;
            IEnumerable<CubinSection> all = cubin.Sections
                .Concat(cubin.DataSections);
            CubinSection? section = all.FirstOrDefault(
                candidate => candidate.Name.Equals(
                    ".nv.constant0" + suffix,
                    StringComparison.Ordinal));
            if (section == null && string.IsNullOrEmpty(entry))
            {
                List<CubinSection> perKernel = all.Where(
                    candidate => candidate.Name.StartsWith(
                        ".nv.constant0.",
                        StringComparison.Ordinal)).ToList();
                if (perKernel.Count == 1)
                    section = perKernel[0];
            }
            if (section == null)
                return null;
            var map = new Dictionary<int, string>();
            foreach (CubinRelocation relocation in section.Relocations)
            {
                if (string.IsNullOrEmpty(relocation.SymbolName) ||
                    relocation.Offset % 4 != 0)
                    continue;
                map[(int)(relocation.Offset / 4)] =
                    relocation.SymbolName;
            }
            return map.Count == 0 ? null : map;
        }
        catch
        {
            return null;
        }
    }
}
