using System.Buffers.Binary;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace nvDismDecode;

// ---------------------------------------------------------------------------
// Minimal ELF64/cubin reader.
//
// A CUDA cubin is an ELF64 relocatable container.  The disassembler only
// needs the executable sections and the function symbols that name them; the
// relocation/debug/constant sections are left alone.
// ---------------------------------------------------------------------------

public sealed class CubinFile
{
    public string Path { get; init; } = "";
    public uint Architecture { get; init; }
    public string ArchitectureKey { get; init; } = "sm_86";
    // Toolkit/ISA profile inferred from the containing fatbin entry, if any.
    public string? Toolkit { get; set; }
    public CubinNoteInfo Notes { get; init; } = new();
    public ushort ElfType { get; init; }
    public byte ElfOsAbi { get; init; }
    public byte ElfAbiVersion { get; init; }
    // CUDA 10.x and older use OSABI 0x33 (the ABI version tracks the
    // toolkit, e.g. 4 for CUDA 3.x/4.x); CUDA 13.x uses the newer pair.
    public string? ToolkitFromAbi => ElfOsAbi switch
    {
        0x33 => "102",
        0x41 => "134",
        _ => null,
    };
    public bool IsDeviceImage { get; init; }
    public Dictionary<uint, uint> RegisterCounts { get; init; } = new();
    public List<CubinSection> Sections { get; init; } = new();
    // Non-executable per-kernel data (.nv.constant2/.nv.constant4).
    public List<CubinSection> DataSections { get; } = new();
    // Non-executable .nv.info* payloads (parameter and ABI metadata).
    public List<CubinInfoSection> InfoSections { get; } = new();
    // Static shared-memory size per entry.  These live in SHT_NOBITS
    // `.nv.shared.<entry>` sections: only the section size is known, and
    // every kernel may choose its own value.
    public Dictionary<string, ulong> SharedSectionSizes { get; } =
        new(StringComparer.Ordinal);
    public List<CubinSymbol> FunctionSymbols { get; } = new();
    public List<CubinSymbol> ObjectSymbols { get; } = new();
    // Full, preserve-order symbol table (functions, objects, sections).
    public List<CubinSymbol> Symbols { get; } = new();
    public List<byte[]> EmbeddedFatbins { get; init; } = new();

    public static CubinFile Load(string path) => Load(File.ReadAllBytes(path), path);

    public static CubinFile Load(byte[] data, string path = "")
    {
        if (data.Length < 52 || data[0] != 0x7F || data[1] != (byte)'E' ||
            data[2] != (byte)'L' || data[3] != (byte)'F')
            throw new InvalidDataException("Not an ELF/cubin file.");

        bool is64 = data[4] == 2;
        if (!is64 && data[4] != 1)
            throw new InvalidDataException("Unsupported ELF class.");
        if (data[5] != 1)
            throw new InvalidDataException("Only little-endian ELF/cubin files are supported.");

        ulong sectionHeaderOffset = is64
            ? BinaryPrimitives.ReadUInt64LittleEndian(data.AsSpan(0x28))
            : BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(0x20));
        ushort sectionHeaderSize = BinaryPrimitives.ReadUInt16LittleEndian(
            data.AsSpan(is64 ? 0x3A : 0x2E));
        ushort sectionCount = BinaryPrimitives.ReadUInt16LittleEndian(
            data.AsSpan(is64 ? 0x3C : 0x30));
        ushort stringSectionIndex = BinaryPrimitives.ReadUInt16LittleEndian(
            data.AsSpan(is64 ? 0x3E : 0x32));

        int minimumSectionHeaderSize = is64 ? 64 : 40;
        if (sectionHeaderSize < minimumSectionHeaderSize)
            throw new InvalidDataException("Invalid ELF section header size.");

        var headers = new List<SectionHeader>(sectionCount);
        for (int i = 0; i < sectionCount; i++)
        {
            ulong offset = sectionHeaderOffset + (ulong)(i * sectionHeaderSize);
            if (offset + (ulong)minimumSectionHeaderSize > (ulong)data.Length)
                throw new InvalidDataException("Truncated ELF section header table.");

            headers.Add(is64
                ? new SectionHeader
                {
                    NameOffset = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan((int)offset)),
                    Type = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan((int)offset + 4)),
                    Flags = BinaryPrimitives.ReadUInt64LittleEndian(data.AsSpan((int)offset + 8)),
                    Address = BinaryPrimitives.ReadUInt64LittleEndian(data.AsSpan((int)offset + 16)),
                    Offset = BinaryPrimitives.ReadUInt64LittleEndian(data.AsSpan((int)offset + 24)),
                    Size = BinaryPrimitives.ReadUInt64LittleEndian(data.AsSpan((int)offset + 32)),
                    Link = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan((int)offset + 40)),
                    Info = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan((int)offset + 44)),
                    Alignment = BinaryPrimitives.ReadUInt64LittleEndian(data.AsSpan((int)offset + 48)),
                    EntrySize = BinaryPrimitives.ReadUInt64LittleEndian(data.AsSpan((int)offset + 56)),
                }
                : new SectionHeader
                {
                    NameOffset = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan((int)offset)),
                    Type = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan((int)offset + 4)),
                    Flags = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan((int)offset + 8)),
                    Address = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan((int)offset + 12)),
                    Offset = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan((int)offset + 16)),
                    Size = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan((int)offset + 20)),
                    Link = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan((int)offset + 24)),
                    Info = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan((int)offset + 28)),
                    Alignment = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan((int)offset + 32)),
                    EntrySize = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan((int)offset + 36)),
                });
        }

        if (stringSectionIndex >= headers.Count)
            throw new InvalidDataException("Invalid ELF section-name table index.");

        string[] names = ReadSectionNames(data, headers, stringSectionIndex);
        var symbols = ReadSymbols(data, headers, is64);
        UniquifySymbolNames(symbols);
        string[] displayNames = BuildSectionDisplayNames(names);
        var sectionSymbols = new Dictionary<int, CubinSymbol>();
        foreach (CubinSymbol symbol in symbols)
        {
            if (symbol.Type == 3 && symbol.SectionIndex < headers.Count)
                sectionSymbols.TryAdd(symbol.SectionIndex, symbol);
        }
        CubinNoteInfo notes = ReadNotes(data, headers, names);
        // The cuinfo note records the virtual SM used by PTX, while e_flags
        // records the actual target.  They differ for code=sm_86 built from a
        // compute_75 virtual architecture.
        uint elfFlags = BinaryPrimitives.ReadUInt32LittleEndian(
            data.AsSpan(is64 ? 0x30 : 0x24));
        // CUDA 13.x stores the architecture in bits 15:8; older cubins
        // (SM30-SM70) store it in bits 7:0 with 4 in bits 15:8.
        uint lowArchitecture = elfFlags & 0xFF;
        uint shiftedArchitecture = (elfFlags >> 8) & 0xFF;
        uint elfArchitecture = lowArchitecture is 2 or 4
            ? shiftedArchitecture
            : lowArchitecture;
        uint architecture = elfArchitecture != 0
            ? elfArchitecture
            : notes.Architecture;
        string architectureKey = SelectArchitectureKey(
            architecture, notes.TkinfoToolOptions);
        var cubin = new CubinFile
        {
            Path = path,
            Architecture = architecture,
            ArchitectureKey = architectureKey,
            Notes = notes,
            ElfType = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(16)),
            ElfOsAbi = data[7],
            ElfAbiVersion = data[8],
            IsDeviceImage = names.Any(n =>
                n == ".note.nv.cuinfo" || n.StartsWith(".nv.info")),
            RegisterCounts = ReadRegisterCounts(data, headers, names),
        };
        cubin.Symbols.AddRange(symbols);
        var seenOffsets = new HashSet<int>();

        // A host ELF keeps its device code in .nv_fatbin.  Scan named
        // fatbin sections for one or more FatBinHeader blocks.
        for (int i = 0; i < headers.Count; i++)
        {
            SectionHeader header = headers[i];
            if (header.Type != 1 || header.Size == 0)
                continue;
            if (names[i].IndexOf("fatbin", StringComparison.OrdinalIgnoreCase) < 0)
                continue;
            ScanEmbeddedFatbins(data, header, cubin.EmbeddedFatbins, seenOffsets);
        }

        // Resolve .nvFatBinSegment wrappers as a second, independent path.
        ScanFatbinWrappers(data, headers, names, cubin.EmbeddedFatbins, seenOffsets);

        // If the section was stripped or renamed, fall back to scanning
        // non-executable PROGBITS sections for the FatBinHeader signature.
        if (cubin.EmbeddedFatbins.Count == 0)
        {
            for (int i = 0; i < headers.Count; i++)
            {
                SectionHeader header = headers[i];
                if (header.Type != 1 || header.Size == 0 ||
                    (header.Flags & 0x4) != 0)
                    continue;
                ScanEmbeddedFatbins(
                    data, header, cubin.EmbeddedFatbins, seenOffsets);
            }
        }

        var sectionMap = new Dictionary<int, CubinSection>();
        for (int i = 0; i < headers.Count; i++)
        {
            SectionHeader header = headers[i];
            if (header.Size > 0 &&
                names[i].StartsWith(".nv.info", StringComparison.Ordinal) &&
                header.Offset + header.Size <= (ulong)data.Length)
            {
                var infoBytes = new byte[header.Size];
                Array.Copy(data, (long)header.Offset, infoBytes, 0,
                    (long)header.Size);
                cubin.InfoSections.Add(new CubinInfoSection
                {
                    Name = names[i],
                    Data = infoBytes,
                });
            }
            if (header.Size > 0 &&
                names[i].StartsWith(".nv.shared", StringComparison.Ordinal))
                cubin.SharedSectionSizes[names[i]] = header.Size;
            if (header.Type != 1 || header.Size == 0)
                continue;
            bool executable = (header.Flags & 0x4) != 0;
            bool constant = names[i].StartsWith(
                ".nv.constant", StringComparison.Ordinal);
            // Module data referenced by constant-bank relocations (e.g.
            // libdevice's sin/cos coefficient tables).
            bool globalInit = names[i].Equals(
                ".nv.global.init", StringComparison.Ordinal);
            if (!executable && !constant && !globalInit)
                continue;
            if (header.Offset + header.Size > (ulong)data.Length)
                continue;

            var bytes = new byte[header.Size];
            Array.Copy(data, (long)header.Offset, bytes, 0, (long)header.Size);

            sectionSymbols.TryGetValue(i, out CubinSymbol? sectionSymbol);
            var section = new CubinSection
            {
                Index = i,
                Name = names[i],
                DisplayName = displayNames[i],
                LabelName = sectionSymbol?.Name ?? displayNames[i],
                LabelOriginalName = sectionSymbol?.OriginalName ?? names[i],
                Address = header.Address,
                Size = header.Size,
                Alignment = header.Alignment,
                ArchitectureKey = architectureKey,
                PreVolta = architecture is > 0 and < 70,
                Data = bytes,
            };

            foreach (var symbol in symbols)
            {
                if (symbol.SectionIndex != i || symbol.Size == 0)
                    continue;
                if (symbol.Type is not (2 or 0))
                    continue;
                section.Functions.Add(symbol);
            }
            section.Functions.Sort((a, b) => a.Value.CompareTo(b.Value));
            if (executable)
                cubin.Sections.Add(section);
            else
                cubin.DataSections.Add(section);
            sectionMap[i] = section;
        }

        // Preserve symbol-table order.  nvdisasm allocates function-end
        // labels in this order, which can differ from section order.
        foreach (CubinSymbol symbol in symbols)
        {
            if (symbol.Type == 13 && (symbol.Other & 0x40) != 0)
                cubin.ObjectSymbols.Add(symbol);
            else if (symbol.Type == 1 &&
                symbol.Name.StartsWith(".nv.reservedSmem", StringComparison.Ordinal))
                cubin.ObjectSymbols.Add(symbol);
            if (symbol.Size == 0 || symbol.Type is not (2 or 0))
                continue;
            cubin.FunctionSymbols.Add(symbol);
        }

        ReadRelocationTargets(data, headers, symbols, sectionMap, is64);
        ReadIndirectBranchTargets(data, headers, names, sectionMap);
        return cubin;
    }

    private static void ReadRelocationTargets(
        byte[] data, List<SectionHeader> headers, List<CubinSymbol> symbols,
        Dictionary<int, CubinSection> sectionMap, bool is64)
    {
        var symbolMap = new Dictionary<uint, CubinSymbol>();
        foreach (CubinSymbol symbol in symbols)
            symbolMap.TryAdd(symbol.Index, symbol);

        for (int i = 0; i < headers.Count; i++)
        {
            SectionHeader header = headers[i];
            bool withAddend = header.Type == 4; // SHT_RELA
            bool withoutAddend = header.Type == 9; // SHT_REL
            if ((!withAddend && !withoutAddend) || header.Size == 0)
                continue;
            if (!sectionMap.TryGetValue((int)header.Info, out CubinSection? section))
                continue;
            if (header.Offset + header.Size > (ulong)data.Length)
                continue;

            int minimumSize = is64
                ? (withAddend ? 24 : 16)
                : (withAddend ? 12 : 8);
            ulong entrySize = header.EntrySize == 0
                ? (ulong)minimumSize
                : header.EntrySize;
            if (entrySize < (ulong)minimumSize)
                continue;
            int count = (int)(header.Size / entrySize);
            for (int entry = 0; entry < count; entry++)
            {
                ulong offset = header.Offset + (ulong)(entry * (int)entrySize);
                if (offset + (ulong)minimumSize > (ulong)data.Length)
                    break;

                ulong relocationOffset;
                ulong relocationInfo;
                long addend;
                uint symbolIndex;
                uint relocationType;
                if (is64)
                {
                    relocationOffset = BinaryPrimitives.ReadUInt64LittleEndian(
                        data.AsSpan((int)offset));
                    relocationInfo = BinaryPrimitives.ReadUInt64LittleEndian(
                        data.AsSpan((int)offset + 8));
                    addend = withAddend
                        ? BinaryPrimitives.ReadInt64LittleEndian(
                            data.AsSpan((int)offset + 16))
                        : 0;
                    symbolIndex = (uint)(relocationInfo >> 32);
                    relocationType = (uint)(relocationInfo & 0xffffffff);
                }
                else
                {
                    relocationOffset = BinaryPrimitives.ReadUInt32LittleEndian(
                        data.AsSpan((int)offset));
                    relocationInfo = BinaryPrimitives.ReadUInt32LittleEndian(
                        data.AsSpan((int)offset + 4));
                    addend = withAddend
                        ? BinaryPrimitives.ReadInt32LittleEndian(
                            data.AsSpan((int)offset + 8))
                        : 0;
                    symbolIndex = (uint)(relocationInfo >> 8);
                    relocationType = (uint)(relocationInfo & 0xFF);
                }
                // Relocations may legitimately point past the section payload.
                // CUDA appends the texture-header table after .nv.constant0,
                // and the STT_CUDA_TEXTURE descriptors are relocated at
                // offsets just beyond the declared section size (e.g. 404..416
                // for a 404-byte .nv.constant0).  Dropping those would lose the
                // TLD/TLDS texref mapping entirely.
                if (relocationOffset > section.Size + 0x1000)
                    continue;

                if (!symbolMap.TryGetValue(symbolIndex, out CubinSymbol? symbol))
                    continue;

                section.Relocations.Add(new CubinRelocation
                {
                    Offset = relocationOffset,
                    Type = relocationType,
                    Addend = addend,
                    IsRel = withoutAddend,
                    SymbolName = symbol.Name,
                    SymbolValue = symbol.Value,
                    SymbolSectionIndex = symbol.SectionIndex,
                });

                if (!withAddend || symbol.SectionIndex != header.Info)
                    continue;

                long target = (long)symbol.Value + addend;
                if (target < 0 || (ulong)target >= section.Size)
                    continue;
                ulong address = section.Address + (ulong)target;
                if (!section.RelocationTargets.Contains(address))
                    section.RelocationTargets.Add(address);
            }
        }

        foreach (CubinSection section in sectionMap.Values)
            section.RelocationTargets.Sort();
    }

    private static string SelectArchitectureKey(
        uint architecture, string toolOptions)
    {
        string fallback = architecture != 0
            ? $"sm_{architecture}"
            : "sm_86";
        if (architecture == 0)
            return fallback;

        foreach (Match match in Regex.Matches(
            toolOptions ?? "", @"sm_(\d+)([a-z]+)",
            RegexOptions.IgnoreCase))
        {
            if (uint.TryParse(match.Groups[1].Value,
                    NumberStyles.None, CultureInfo.InvariantCulture,
                    out uint number) && number == architecture)
                return $"sm_{number}{match.Groups[2].Value.ToLowerInvariant()}";
        }
        return fallback;
    }

    private static void ReadIndirectBranchTargets(
        byte[] data, List<SectionHeader> headers, string[] names,
        Dictionary<int, CubinSection> sectionMap)
    {
        var queues = new Dictionary<string, Queue<CubinSection>>(
            StringComparer.Ordinal);
        foreach (CubinSection section in sectionMap.Values)
        {
            if (!section.Name.StartsWith(".text.", StringComparison.Ordinal))
                continue;
            string suffix = section.Name[".text.".Length..];
            if (!queues.TryGetValue(suffix, out Queue<CubinSection>? queue))
            {
                queue = new Queue<CubinSection>();
                queues[suffix] = queue;
            }
            queue.Enqueue(section);
        }

        for (int i = 0; i < headers.Count; i++)
        {
            string infoName = names[i];
            if (!infoName.StartsWith(".nv.info.", StringComparison.Ordinal))
                continue;
            string suffix = infoName[".nv.info.".Length..];
            if (!queues.TryGetValue(suffix, out Queue<CubinSection>? queue) ||
                queue.Count == 0)
                continue;

            SectionHeader header = headers[i];
            if (header.Size == 0 ||
                header.Offset + header.Size > (ulong)data.Length)
                continue;
            CubinSection section = queue.Dequeue();

            int offset = (int)header.Offset;
            int end = offset + (int)header.Size;
            while (offset + 4 <= end)
            {
                byte format = data[offset];
                byte attribute = data[offset + 1];
                ushort value = BinaryPrimitives.ReadUInt16LittleEndian(
                    data.AsSpan(offset + 2));
                int recordSize;
                if (format is 1 or 2 or 3)
                    recordSize = 4;
                else if (format == 4)
                    recordSize = 4 + value;
                else
                    break;
                if (offset + recordSize > end)
                    break;

                if (attribute == 102)
                    section.HasStackOffset = true;
                // EIATTR_FRAME_SIZE (0x11), EIATTR_MIN_STACK_SIZE (0x12),
                // EIATTR_MAX_STACK_SIZE (0x23) - format 1/2: value in record.
                if (format == 1 || format == 2) {
                    if (attribute == 0x11) section.FrameSize = value;
                    else if (attribute == 0x12) section.MinStackSize = value;
                    else if (attribute == 0x23) section.MaxStackSize = value;
                }

                if (format == 4 && attribute == 52 && value >= 12)
                {
                    int wordCount = value / 4;
                    int position = 0;
                    while (position + 3 <= wordCount)
                    {
                        int instructionOffset = (int)BinaryPrimitives
                            .ReadUInt32LittleEndian(
                                data.AsSpan(offset + 4 + position * 4));
                        uint count = BinaryPrimitives.ReadUInt32LittleEndian(
                            data.AsSpan(offset + 4 + (position + 2) * 4));
                        if (count > (uint)(wordCount - position - 3))
                            break;
                        if (instructionOffset >= 0 &&
                            (ulong)instructionOffset < section.Size)
                        {
                            var targets = new List<ulong>((int)count);
                            for (int t = 0; t < count; t++)
                            {
                                uint target = BinaryPrimitives
                                    .ReadUInt32LittleEndian(data.AsSpan(
                                        offset + 4 + (position + 3 + t) * 4));
                                if ((ulong)target < section.Size)
                                    targets.Add(section.Address + target);
                            }
                            section.IndirectBranchTargets[
                                (ulong)instructionOffset] = targets;
                        }
                        position += 3 + (int)count;
                    }
                }

                offset += recordSize;
            }
        }
    }

    private static CubinNoteInfo ReadNotes(
        byte[] data, List<SectionHeader> headers, string[] names)
    {
        var info = new CubinNoteInfo();
        for (int i = 0; i < headers.Count; i++)
        {
            SectionHeader header = headers[i];
            if (header.Type != 7 || header.Size < 16 ||
                header.Offset + header.Size > (ulong)data.Length)
                continue;

            string sectionName = names[i];
            bool cuinfo = sectionName == ".note.nv.cuinfo";
            bool tkinfo = sectionName == ".note.nv.tkinfo";
            if (!cuinfo && !tkinfo)
                continue;

            int start = (int)header.Offset;
            int end = start + (int)header.Size;
            int offset = start;
            while (offset + 12 <= end)
            {
                uint nameSize = BinaryPrimitives.ReadUInt32LittleEndian(
                    data.AsSpan(offset));
                uint descSize = BinaryPrimitives.ReadUInt32LittleEndian(
                    data.AsSpan(offset + 4));
                uint noteType = BinaryPrimitives.ReadUInt32LittleEndian(
                    data.AsSpan(offset + 8));
                if (nameSize > header.Size || descSize > header.Size)
                    break;
                int nameStart = offset + 12;
                int descStart = nameStart + Align4(nameSize);
                int next = descStart + Align4(descSize);
                if (next > end || next <= offset)
                    break;

                if (cuinfo && !info.CuinfoPresent &&
                    noteType == 0x3e8 && descSize >= 8)
                {
                    info.CuinfoPresent = true;
                    info.CuinfoFormat = BinaryPrimitives.ReadUInt16LittleEndian(
                        data.AsSpan(descStart));
                    info.Architecture = BinaryPrimitives.ReadUInt16LittleEndian(
                        data.AsSpan(descStart + 2));
                    info.CuinfoToolVersion = BinaryPrimitives
                        .ReadUInt16LittleEndian(data.AsSpan(descStart + 4));
                }
                else if (tkinfo && !info.TkinfoPresent &&
                    noteType == 2000 && descSize >= 24)
                {
                    info.TkinfoPresent = true;
                    info.TkinfoVersion = BinaryPrimitives.ReadUInt32LittleEndian(
                        data.AsSpan(descStart));
                    uint objectOffset = BinaryPrimitives.ReadUInt32LittleEndian(
                        data.AsSpan(descStart + 4));
                    uint nameOffset = BinaryPrimitives.ReadUInt32LittleEndian(
                        data.AsSpan(descStart + 8));
                    uint versionOffset = BinaryPrimitives.ReadUInt32LittleEndian(
                        data.AsSpan(descStart + 12));
                    uint branchOffset = BinaryPrimitives.ReadUInt32LittleEndian(
                        data.AsSpan(descStart + 16));
                    uint optionsOffset = BinaryPrimitives.ReadUInt32LittleEndian(
                        data.AsSpan(descStart + 20));
                    info.TkinfoObjectFileName = ReadNoteString(
                        data, descStart, (int)descSize, objectOffset);
                    info.TkinfoToolName = ReadNoteString(
                        data, descStart, (int)descSize, nameOffset);
                    info.TkinfoToolVersion = ReadNoteString(
                        data, descStart, (int)descSize, versionOffset);
                    info.TkinfoToolBranch = ReadNoteString(
                        data, descStart, (int)descSize, branchOffset);
                    info.TkinfoToolOptions = ReadNoteString(
                        data, descStart, (int)descSize, optionsOffset);
                }

                offset = next;
            }
        }
        return info;
    }

    private static string ReadNoteString(
        byte[] data, int descStart, int descSize, uint relativeOffset)
    {
        // tkinfo strings start 24 bytes into the descriptor payload.
        int start = descStart + 24 + (int)relativeOffset;
        int end = descStart + descSize;
        if (start < descStart || start >= end)
            return "";
        int length = 0;
        while (start + length < end && data[start + length] != 0)
            length++;
        return Encoding.UTF8.GetString(data, start, length);
    }

    private static Dictionary<uint, uint> ReadRegisterCounts(
        byte[] data, List<SectionHeader> headers, string[] names)
    {
        var result = new Dictionary<uint, uint>();
        for (int i = 0; i < headers.Count; i++)
        {
            SectionHeader header = headers[i];
            bool globalInfo = names[i] == ".nv.info";
            if ((!globalInfo && !names[i].StartsWith(
                    ".nv.info.", StringComparison.Ordinal)) ||
                header.Size == 0 ||
                header.Offset + header.Size > (ulong)data.Length)
                continue;

            int offset = (int)header.Offset;
            int end = offset + (int)header.Size;
            while (offset + 4 <= end)
            {
                byte format = data[offset];
                byte attribute = data[offset + 1];
                ushort value = BinaryPrimitives.ReadUInt16LittleEndian(
                    data.AsSpan(offset + 2));
                int recordSize;
                if (format is 1 or 2 or 3)
                {
                    // EIFMT_NVAL / EIFMT_BVAL / EIFMT_HVAL: inline u16 value.
                    recordSize = 4;
                }
                else if (format == 4)
                {
                    // EIFMT_SVAL: u16 byte length followed by the payload.
                    recordSize = 4 + value;
                }
                else
                {
                    break;
                }
                if (offset + recordSize > end)
                    break;

                // EIATTR_REGCOUNT (0x2f) carries a symbol index and the
                // maximum register count for that function.
                if (format == 4 && attribute == 0x2f && value >= 8)
                {
                    uint symbolIndex = BinaryPrimitives.ReadUInt32LittleEndian(
                        data.AsSpan(offset + 4));
                    uint count = BinaryPrimitives.ReadUInt32LittleEndian(
                        data.AsSpan(offset + 8));
                    // Per-function .nv.info.<name> sections carry the exact
                    // count; the aggregate .nv.info may list a larger value.
                    if (!globalInfo || !result.ContainsKey(symbolIndex))
                        result[symbolIndex] = count;
                }

                offset += recordSize;
            }
        }
        return result;
    }

    private static int Align4(uint value) => (int)((value + 3) & ~3u);

    private static void ScanEmbeddedFatbins(
        byte[] data, SectionHeader section, List<byte[]> result,
        HashSet<int> seenOffsets)
    {
        if (section.Offset + section.Size > (ulong)data.Length)
            return;
        FatbinScanner.Scan(
            data, (int)section.Offset, (int)(section.Offset + section.Size),
            result, seenOffsets);
    }

    private const uint WrapperMagic = 0x466243B1;

    private static void ScanFatbinWrappers(
        byte[] data, List<SectionHeader> headers, string[] names,
        List<byte[]> result, HashSet<int> seenOffsets)
    {
        for (int i = 0; i < headers.Count; i++)
        {
            SectionHeader section = headers[i];
            if (section.Type != 1 || section.Size < 24 ||
                section.Offset + section.Size > (ulong)data.Length)
                continue;
            if (names[i].IndexOf("segment", StringComparison.OrdinalIgnoreCase) < 0 &&
                names[i].IndexOf("fatbin", StringComparison.OrdinalIgnoreCase) < 0)
                continue;

            int start = (int)section.Offset;
            int end = start + (int)section.Size;
            for (int offset = start; offset + 24 <= end; offset += 8)
            {
                if (BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset)) != WrapperMagic)
                    continue;

                int version = BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(offset + 4));
                if (version is not (1 or 2))
                    continue;

                ulong pointer = BinaryPrimitives.ReadUInt64LittleEndian(data.AsSpan(offset + 8));
                if (pointer == 0 || !TryMapAddress(headers, pointer, out int fatbinOffset))
                    continue;
                if (fatbinOffset + 16 > data.Length ||
                    BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(fatbinOffset)) !=
                        FatbinFile.Magic)
                    continue;

                ushort headerSize = BinaryPrimitives.ReadUInt16LittleEndian(
                    data.AsSpan(fatbinOffset + 6));
                ulong totalSize = BinaryPrimitives.ReadUInt64LittleEndian(
                    data.AsSpan(fatbinOffset + 8));
                long length = headerSize + (long)totalSize;
                if (length < 16 || fatbinOffset + length > data.Length)
                    length = data.Length - fatbinOffset;

                if (seenOffsets.Add(fatbinOffset))
                {
                    var blob = new byte[length];
                    Array.Copy(data, fatbinOffset, blob, 0, (int)length);
                    result.Add(blob);
                }
            }
        }
    }

    private static bool TryMapAddress(
        List<SectionHeader> headers, ulong address, out int fileOffset)
    {
        foreach (SectionHeader section in headers)
        {
            if (section.Type != 1 || section.Size == 0)
                continue;
            if (address >= section.Address &&
                address < section.Address + section.Size)
            {
                fileOffset = (int)(section.Offset + (address - section.Address));
                return true;
            }
        }
        fileOffset = 0;
        return false;
    }

    private static string[] ReadSectionNames(
        byte[] data, List<SectionHeader> headers, int stringSectionIndex)
    {
        SectionHeader stringHeader = headers[stringSectionIndex];
        if (stringHeader.Offset + stringHeader.Size > (ulong)data.Length)
            throw new InvalidDataException("Truncated ELF section-name table.");

        var names = new string[headers.Count];
        for (int i = 0; i < headers.Count; i++)
            names[i] = ReadCString(
                data, stringHeader.Offset, stringHeader.Size, headers[i].NameOffset);
        return names;
    }

    private static List<CubinSymbol> ReadSymbols(
        byte[] data, List<SectionHeader> headers, bool is64)
    {
        var result = new List<CubinSymbol>();
        for (int i = 0; i < headers.Count; i++)
        {
            SectionHeader header = headers[i];
            if (header.Type != 2) // SHT_SYMTAB
                continue;
            if (header.Link >= headers.Count)
                continue;

            SectionHeader strings = headers[(int)header.Link];
            if (strings.Offset + strings.Size > (ulong)data.Length ||
                header.Offset + header.Size > (ulong)data.Length)
                continue;

            int minimumSize = is64 ? 24 : 16;
            ulong entrySize = header.EntrySize == 0
                ? (ulong)minimumSize
                : header.EntrySize;
            if (entrySize < (ulong)minimumSize)
                continue;
            int count = (int)(header.Size / entrySize);
            for (int entry = 0; entry < count; entry++)
            {
                ulong offset = header.Offset + (ulong)(entry * (int)entrySize);
                if (offset + (ulong)minimumSize > (ulong)data.Length)
                    break;

                uint nameOffset;
                byte info;
                byte other;
                ushort sectionIndex;
                ulong value;
                ulong size;
                if (is64)
                {
                    nameOffset = BinaryPrimitives.ReadUInt32LittleEndian(
                        data.AsSpan((int)offset));
                    info = data[(int)offset + 4];
                    other = data[(int)offset + 5];
                    sectionIndex = BinaryPrimitives.ReadUInt16LittleEndian(
                        data.AsSpan((int)offset + 6));
                    value = BinaryPrimitives.ReadUInt64LittleEndian(
                        data.AsSpan((int)offset + 8));
                    size = BinaryPrimitives.ReadUInt64LittleEndian(
                        data.AsSpan((int)offset + 16));
                }
                else
                {
                    nameOffset = BinaryPrimitives.ReadUInt32LittleEndian(
                        data.AsSpan((int)offset));
                    value = BinaryPrimitives.ReadUInt32LittleEndian(
                        data.AsSpan((int)offset + 4));
                    size = BinaryPrimitives.ReadUInt32LittleEndian(
                        data.AsSpan((int)offset + 8));
                    info = data[(int)offset + 12];
                    other = data[(int)offset + 13];
                    sectionIndex = BinaryPrimitives.ReadUInt16LittleEndian(
                        data.AsSpan((int)offset + 14));
                }

                string name = ReadCString(data, strings.Offset, strings.Size, nameOffset);
                if (name.Length == 0)
                    continue;

                result.Add(new CubinSymbol(
                    name, (byte)(info & 0x0F), sectionIndex, value, size,
                    (byte)(info >> 4), other, (uint)entry));
            }
        }
        return result;
    }

    private static string[] BuildSectionDisplayNames(string[] names)
    {
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        var result = new string[names.Length];
        for (int i = 0; i < names.Length; i++)
        {
            string name = names[i];
            if (counts.TryGetValue(name, out int count))
            {
                result[i] = name + "__" + count;
                counts[name] = count + 1;
            }
            else
            {
                result[i] = name;
                counts[name] = 1;
            }
        }
        return result;
    }

    private static void UniquifySymbolNames(List<CubinSymbol> symbols)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        int renameCounter = 0;
        for (int i = 0; i < symbols.Count; i++)
        {
            CubinSymbol symbol = symbols[i];
            if (seen.Add(symbol.Name))
                continue;
            string display = symbol.Name + "__" + renameCounter++;
            symbols[i] = new CubinSymbol(
                display, symbol.Type, symbol.SectionIndex, symbol.Value,
                symbol.Size, symbol.Binding, symbol.Other, symbol.Index,
                symbol.OriginalName);
        }
    }

    private static string ReadCString(byte[] data, ulong tableOffset, ulong tableSize, uint offset)
    {
        if (offset >= tableSize)
            return "";
        int start = (int)(tableOffset + offset);
        int end = start;
        int limit = (int)Math.Min((ulong)data.Length, tableOffset + tableSize);
        while (end < limit && data[end] != 0)
            end++;
        return Encoding.UTF8.GetString(data, start, end - start);
    }

    private sealed class SectionHeader
    {
        public uint NameOffset;
        public uint Type;
        public ulong Flags;
        public ulong Address;
        public ulong Offset;
        public ulong Size;
        public uint Link;
        public uint Info;
        public ulong Alignment;
        public ulong EntrySize;
    }
}

public sealed class CubinInfoSection
{
    public string Name { get; init; } = "";
    public byte[] Data { get; init; } = Array.Empty<byte>();
}

public sealed class CubinSection
{
    public int Index { get; init; }
    public string Name { get; init; } = "";
    public string DisplayName { get; init; } = "";
    public string LabelName { get; init; } = "";
    public string LabelOriginalName { get; init; } = "";
    public ulong Address { get; init; }
    public ulong Size { get; init; }
    public ulong Alignment { get; init; }
    public string ArchitectureKey { get; init; } = "";
    public bool PreVolta { get; init; }
    public byte[] Data { get; init; } = Array.Empty<byte>();
    public List<CubinSymbol> Functions { get; } = new();
    public List<CubinRelocation> Relocations { get; } = new();
    public List<ulong> RelocationTargets { get; } = new();
    public Dictionary<ulong, List<ulong>> IndirectBranchTargets { get; } =
        new();
    public bool HasStackOffset { get; set; }
    public ushort FrameSize { get; set; }
    public ushort MinStackSize { get; set; }
    public ushort MaxStackSize { get; set; }
}

public sealed class CubinNoteInfo
{
    public bool CuinfoPresent { get; set; }
    public uint Architecture { get; set; }
    public ushort CuinfoFormat { get; set; }
    public ushort CuinfoToolVersion { get; set; }
    public bool TkinfoPresent { get; set; }
    public uint TkinfoVersion { get; set; }
    public string TkinfoObjectFileName { get; set; } = "";
    public string TkinfoToolName { get; set; } = "";
    public string TkinfoToolVersion { get; set; } = "";
    public string TkinfoToolBranch { get; set; } = "";
    public string TkinfoToolOptions { get; set; } = "";
}

public sealed class CubinRelocation
{
    public ulong Offset { get; init; }
    public uint Type { get; init; }
    public long Addend { get; init; }
    public bool IsRel { get; init; }
    public string SymbolName { get; init; } = "";
    public ulong SymbolValue { get; init; }
    public ushort SymbolSectionIndex { get; init; }
}

public sealed class CubinSymbol
{
    public CubinSymbol(
        string name, byte type, ushort sectionIndex, ulong value, ulong size,
        byte binding = 0, byte other = 0, uint index = 0,
        string? originalName = null)
    {
        Name = name;
        OriginalName = originalName ?? name;
        Type = type;
        SectionIndex = sectionIndex;
        Value = value;
        Size = size;
        Binding = binding;
        Other = other;
        Index = index;
    }

    public string Name { get; }
    public string OriginalName { get; }
    public byte Type { get; }
    public ushort SectionIndex { get; }
    public ulong Value { get; }
    public ulong Size { get; }
    public byte Binding { get; }
    public byte Other { get; }
    public uint Index { get; }
}
