using System.Buffers.Binary;
using System.Text;

namespace rtxdisam;

public sealed class CubinImageSymbol
{
    public string Name { get; init; } = "";
    public int Section { get; init; }
    public ulong Value { get; init; }
    public ulong Size { get; init; }
    public int? RegisterCount { get; init; }
}

public sealed class CubinImageRelocation
{
    public ulong Offset { get; init; }
    public string Symbol { get; init; } = "";
    public int Type { get; init; }
    public long Addend { get; init; }
    public bool IsRel { get; init; }
}

public sealed class CubinImageSection
{
    public string Name { get; init; } = ".text";
    public uint Type { get; init; } = 1;
    public ulong Flags { get; init; } = 0x6;
    public int Alignment { get; init; } = 128;
    public uint Info { get; init; }
    public ulong EntrySize { get; init; }
    public byte[] Data { get; init; } = Array.Empty<byte>();
    public List<CubinImageRelocation> Relocations { get; init; } = new();
}

// Writes the minimal valid ELF64/cubin container needed by nvdisasm and by
// CubinReader.  The assembler supplies executable sections and function
// symbols.  The optional toolkit profile selects the legacy CUDA 10.x ELF
// ABI/tkinfo or the CUDA 13.x ABI/tkinfo.
public static class CubinWriter
{
    private const int ElfHeaderSize = 64;
    private const int SectionHeaderSize = 64;

    public static void Write(
        string path, uint architecture, string architectureName,
        IReadOnlyList<CubinImageSection> sections,
        IReadOnlyList<CubinImageSymbol> symbols, string? toolkit = null)
    {
        string? normalizedToolkit = toolkit == null
            ? null
            : nvDismDecode.InstructionDecoder.NormalizeToolkit(toolkit);
        if (normalizedToolkit == "134" && architecture < 75)
            throw new InvalidDataException(
                $"CUDA 13.x is not valid for pre-Volta SM{architecture}.");

        // Encoding must never invent an ABI for an architecture that has no
        // corresponding table.
        string? tablePath = architecture < 70
            ? nvDismDecode.PreVoltaDecoder.ResolveTablePath(
                architectureName, normalizedToolkit)
            : nvDismDecode.InstructionDecoder.ResolveTablePath(
                architectureName, normalizedToolkit);
        if (tablePath == null)
            throw new InvalidDataException(
                $"No ISA table is available for {architectureName}" +
                (normalizedToolkit != null
                    ? $" with toolkit {normalizedToolkit}"
                    : "") + ".");

        bool legacyCuda = normalizedToolkit == "102" ||
            (normalizedToolkit == null && architecture < 75);
        var allSymbols = new List<CubinImageSymbol>(symbols);
        var knownSymbols = new HashSet<string>(
            symbols.Select(symbol => symbol.Name), StringComparer.Ordinal);
        foreach (CubinImageRelocation relocation in sections
            .SelectMany(section => section.Relocations))
        {
            if (relocation.Symbol.Length == 0 ||
                !knownSymbols.Add(relocation.Symbol))
                continue;
            // Preserved hidden relocations may reference symbols that are not
            // declared by the SASS source.  Keep them as undefined symbols so
            // the relocation table retains its target name.
            allSymbols.Add(new CubinImageSymbol
            {
                Name = relocation.Symbol,
                Section = -1,
            });
        }

        var shstr = new StringTable();
        var sectionNames = sections.Select(s => shstr.Add(s.Name)).ToArray();
        int noteName = shstr.Add(".note.nv.cuinfo");
        int tkiName = shstr.Add(".note.nv.tkinfo");
        int compatName = shstr.Add(".nv.compat");
        int symtabName = shstr.Add(".symtab");
        int strtabName = shstr.Add(".strtab");
        int shstrtabName = shstr.Add(".shstrtab");
        int nvinfoName = shstr.Add(".nv.info");

        var str = new StringTable();
        var symbolNames = allSymbols.Select(s => str.Add(s.Name)).ToArray();

        int textCount = sections.Count;
        bool hasCompat = ArchitectureNumber(architectureName) >= 90;
        byte[] compat = hasCompat ? BuildCompatSection(architectureName) : Array.Empty<byte>();
        int noteIndex = 1 + textCount;
        int tkiIndex = noteIndex + 1;
        int compatIndex = tkiIndex + 1;
        int symtabIndex = (hasCompat ? compatIndex : tkiIndex) + 1;
        int strtabIndex = symtabIndex + 1;
        int shstrtabIndex = strtabIndex + 1;
        int nvinfoIndex = shstrtabIndex + 1;
        int sectionCount = nvinfoIndex + 1;

        byte[] note = BuildCuinfoNote(architecture);
        byte[] tki = BuildTkinfoNote(architectureName, legacyCuda);
        byte[] nvInfo = BuildRegisterCountInfo(symbols, textCount);
        byte[][] sectionData = sections
            .Select(section => section.Relocations.Count == 0
                ? section.Data
                : new byte[section.Relocations.Count *
                    (section.Type == 9 ? 16 : 24)])
            .ToArray();
        int offset = ElfHeaderSize;
        var dataOffsets = new int[textCount];
        for (int i = 0; i < textCount; i++)
        {
            offset = Align(offset, Math.Max(1, sections[i].Alignment));
            dataOffsets[i] = offset;
            offset += sectionData[i].Length;
        }

        offset = Align(offset, 4);
        int noteOffset = offset;
        offset += note.Length;
        offset = Align(offset, 4);
        int tkiOffset = offset;
        offset += tki.Length;
        int compatOffset = offset;
        if (hasCompat)
        {
            offset = Align(offset, 4);
            compatOffset = offset;
            offset += compat.Length;
        }

        offset = Align(offset, 8);
        int symtabOffset = offset;
        int symbolCount = 1 + textCount + allSymbols.Count;
        offset += symbolCount * 24;

        int strtabOffset = offset;
        offset += str.Bytes.Length;

        int shstrtabOffset = offset;
        offset += shstr.Bytes.Length;

        offset = Align(offset, 4);
        int nvInfoOffset = offset;
        offset += nvInfo.Length;

        offset = Align(offset, 8);
        int shoff = offset;
        offset += sectionCount * SectionHeaderSize;

        byte[] image = new byte[offset];
        WriteElfHeader(
            image, architecture, shoff, sectionCount, shstrtabIndex,
            legacyCuda);

        for (int i = 0; i < textCount; i++)
            sectionData[i].CopyTo(image, dataOffsets[i]);
        note.CopyTo(image, noteOffset);
        tki.CopyTo(image, tkiOffset);
        if (hasCompat)
            compat.CopyTo(image, compatOffset);

        int symbolOffset = symtabOffset;
        WriteSymbol(image, symbolOffset, 0, 0, 0, 0, 0, 0);
        symbolOffset += 24;
        for (int i = 0; i < textCount; i++)
        {
            WriteSymbol(image, symbolOffset, 0, 3, (ushort)(i + 1), 0, 0, 0);
            symbolOffset += 24;
        }
        for (int i = 0; i < allSymbols.Count; i++)
        {
            CubinImageSymbol symbol = allSymbols[i];
            if (symbol.Section < 0)
            {
                WriteSymbol(
                    image, symbolOffset, symbolNames[i], 0x21,
                    0, 0, 0, 0);
            }
            else
            {
                WriteSymbol(
                    image, symbolOffset, symbolNames[i], 0x12,
                    (ushort)(symbol.Section + 1), (ulong)symbol.Value,
                    symbol.Size, 0);
            }
            symbolOffset += 24;
        }

        var symbolIndexes = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < allSymbols.Count; i++)
            symbolIndexes[allSymbols[i].Name] = 1 + textCount + i;
        for (int i = 0; i < textCount; i++)
        {
            for (int j = 0; j < sections[i].Relocations.Count; j++)
            {
                CubinImageRelocation relocation = sections[i].Relocations[j];
                int target = symbolIndexes.TryGetValue(
                    relocation.Symbol, out int symbolIndex) ? symbolIndex : 0;
                ulong info = ((ulong)(uint)target << 32) |
                    (uint)relocation.Type;
                int entrySize = sections[i].Type == 9 ? 16 : 24;
                int entry = dataOffsets[i] + j * entrySize;
                BinaryPrimitives.WriteUInt64LittleEndian(
                    image.AsSpan(entry), relocation.Offset);
                BinaryPrimitives.WriteUInt64LittleEndian(
                    image.AsSpan(entry + 8), info);
                if (entrySize == 24)
                {
                    BinaryPrimitives.WriteInt64LittleEndian(
                        image.AsSpan(entry + 16), relocation.Addend);
                }
            }
        }

        str.Bytes.CopyTo(image, strtabOffset);
        shstr.Bytes.CopyTo(image, shstrtabOffset);
        nvInfo.CopyTo(image, nvInfoOffset);

        int sh = shoff;
        WriteSectionHeader(image, sh, 0, 0, 0, 0, 0, 0, 0, 0); sh += 64;
        for (int i = 0; i < textCount; i++)
        {
            uint sectionLink = sections[i].Type == 4
                ? (uint)symtabIndex : 0;
            WriteSectionHeader(
                image, sh, (uint)sectionNames[i], sections[i].Type,
                sections[i].Flags,
                (ulong)dataOffsets[i], (ulong)sectionData[i].Length,
                (ulong)Math.Max(1, sections[i].Alignment),
                sectionLink, sections[i].Info,
                sections[i].EntrySize); sh += 64;
        }

        WriteSectionHeader(
            image, sh, (uint)noteName, 7, 0x1000000, (ulong)noteOffset,
            (ulong)note.Length, 4, 0, 0);
        sh += 64;
        WriteSectionHeader(
            image, sh, (uint)tkiName, 7, 0x2000000, (ulong)tkiOffset,
            (ulong)tki.Length, 4, 0, 0);
        sh += 64;
        if (hasCompat)
        {
            WriteSectionHeader(
                image, sh, (uint)compatName, 0x70000086, 0,
                (ulong)compatOffset, (ulong)compat.Length, 4, 0, 0);
            sh += 64;
        }
        WriteSectionHeader(
            image, sh, (uint)symtabName, 2, 0, (ulong)symtabOffset,
            (ulong)(symbolCount * 24), 8, (uint)strtabIndex,
            (uint)(1 + textCount), 24);
        sh += 64;
        WriteSectionHeader(
            image, sh, (uint)strtabName, 3, 0, (ulong)strtabOffset,
            (ulong)str.Bytes.Length, 1, 0, 0); sh += 64;
        WriteSectionHeader(
            image, sh, (uint)shstrtabName, 3, 0, (ulong)shstrtabOffset,
            (ulong)shstr.Bytes.Length, 1, 0, 0); sh += 64;
        WriteSectionHeader(
            image, sh, (uint)nvinfoName, 0x70000000, 0,
            (ulong)nvInfoOffset, (ulong)nvInfo.Length, 4, 0, 0);

        File.WriteAllBytes(path, image);
    }

    private static void WriteElfHeader(
        byte[] data, uint architecture, int shoff, int sectionCount,
        int shstrtabIndex, bool legacyCuda)
    {
        data[0] = 0x7F;
        data[1] = (byte)'E';
        data[2] = (byte)'L';
        data[3] = (byte)'F';
        data[4] = 2; // ELFCLASS64
        data[5] = 1; // little endian
        data[6] = 1; // ELF version
        // CUDA 10.x and older use OSABI 0x33 / ABI 7 and encode the SM
        // number in the low and high bytes of e_flags.  CUDA 13.x uses
        // OSABI 0x41 / ABI 8 with the shifted architecture field.
        data[7] = legacyCuda ? (byte)0x33 : (byte)0x41;
        data[8] = legacyCuda ? (byte)7 : (byte)8;

        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(16), 2); // ET_EXEC
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(18), 190); // CUDA
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(20), 1);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(40), (ulong)shoff);
        BinaryPrimitives.WriteUInt32LittleEndian(
            data.AsSpan(0x30), ArchitectureFlags(architecture, legacyCuda));
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(52), 64);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(58), 64);
        BinaryPrimitives.WriteUInt16LittleEndian(
            data.AsSpan(60), (ushort)sectionCount);
        BinaryPrimitives.WriteUInt16LittleEndian(
            data.AsSpan(62), (ushort)shstrtabIndex);
    }

    private static byte[] BuildCuinfoNote(uint architecture)
    {
        byte[] name = Encoding.ASCII.GetBytes("NVIDIA Corp\0");
        const int noteHeader = 12;
        int nameAligned = (name.Length + 3) & ~3;
        byte[] note = new byte[noteHeader + nameAligned + 8];
        BinaryPrimitives.WriteUInt32LittleEndian(note.AsSpan(0), 12);
        BinaryPrimitives.WriteUInt32LittleEndian(note.AsSpan(4), 8);
        BinaryPrimitives.WriteUInt32LittleEndian(note.AsSpan(8), 0x3e8);
        name.CopyTo(note, noteHeader);
        int desc = noteHeader + nameAligned;
        BinaryPrimitives.WriteUInt16LittleEndian(
            note.AsSpan(desc), 2);
        BinaryPrimitives.WriteUInt16LittleEndian(
            note.AsSpan(desc + 2), (ushort)architecture);
        BinaryPrimitives.WriteUInt16LittleEndian(
            note.AsSpan(desc + 4), 0);
        BinaryPrimitives.WriteUInt16LittleEndian(
            note.AsSpan(desc + 6), 0);
        return note;
    }

    private static byte[] BuildCompatSection(string architectureName)
    {
        // CUDA 13.x compatibility records.  Blackwell 10.0/12.0 use an
        // additional record; sm_90a differs in the first record.
        const string prefix = "02090000020c000002020100030d0101";
        string hex;
        if (architectureName.StartsWith("sm_100", StringComparison.OrdinalIgnoreCase) ||
            architectureName.StartsWith("sm_120", StringComparison.OrdinalIgnoreCase))
        {
            hex = prefix + "0307010102030000040b0800" +
                (architectureName.StartsWith(
                    "sm_120", StringComparison.OrdinalIgnoreCase)
                    ? "50000000" : "09000000") +
                "00000000" + "030e0004";
        }
        else
        {
            hex = prefix + "02030000040b0800" +
                "00000000" + "00000000" + "030e0004";
        }
        byte[] data = Convert.FromHexString(hex);
        if (architectureName.Equals("sm_90a", StringComparison.OrdinalIgnoreCase))
            data[2] = 1;
        return data;
    }

    private static uint ArchitectureNumber(string architectureName)
    {
        int start = architectureName.StartsWith("sm_", StringComparison.OrdinalIgnoreCase)
            ? 3 : 0;
        int end = start;
        while (end < architectureName.Length &&
            char.IsDigit(architectureName[end]))
            end++;
        return uint.Parse(architectureName[start..end],
            System.Globalization.CultureInfo.InvariantCulture);
    }

    private static byte[] BuildTkinfoNote(
        string architectureName, bool legacyCuda)
    {
        // NVIDIA's note parser is strict about the 160-byte ptxas tkinfo
        // layout.  Keep the same descriptor/string offsets that nvcc emits.
        const int stringSize = 112;
        byte[] strings = new byte[stringSize];
        void Put(int offset, string value)
        {
            byte[] bytes = Encoding.ASCII.GetBytes(value);
            if (offset + bytes.Length >= strings.Length)
                throw new InvalidOperationException("tkinfo string too long");
            bytes.CopyTo(strings, offset);
        }

        Put(1, "ptxas");
        Put(7, legacyCuda
            ? "Cuda compilation tools, release 10.2, V10.2.89"
            : "Cuda compilation tools, release 13.4, V13.4.92");
        Put(54, legacyCuda
            ? "Build cuda_10.2.r10.2/compiler.29695070_0"
            : "Build cuda_13.4.r13.4/compiler.38855100_0");
        Put(96, $"-arch {architectureName} ");

        byte[] name = Encoding.ASCII.GetBytes("NVIDIA Corp\0");
        int nameAligned = (name.Length + 3) & ~3;
        const int descSize = 24 + stringSize;
        byte[] note = new byte[12 + nameAligned + descSize];
        BinaryPrimitives.WriteUInt32LittleEndian(note.AsSpan(0), 12);
        BinaryPrimitives.WriteUInt32LittleEndian(
            note.AsSpan(4), (uint)descSize);
        BinaryPrimitives.WriteUInt32LittleEndian(note.AsSpan(8), 2000);
        name.CopyTo(note, 12);

        int desc = 12 + nameAligned;
        BinaryPrimitives.WriteUInt32LittleEndian(note.AsSpan(desc), 2);
        BinaryPrimitives.WriteUInt32LittleEndian(note.AsSpan(desc + 4), 0);
        BinaryPrimitives.WriteUInt32LittleEndian(note.AsSpan(desc + 8), 1);
        BinaryPrimitives.WriteUInt32LittleEndian(note.AsSpan(desc + 12), 7);
        BinaryPrimitives.WriteUInt32LittleEndian(note.AsSpan(desc + 16), 54);
        BinaryPrimitives.WriteUInt32LittleEndian(note.AsSpan(desc + 20), 96);
        strings.CopyTo(note, desc + 24);
        return note;
    }

    private static byte[] BuildRegisterCountInfo(
        IReadOnlyList<CubinImageSymbol> symbols, int sectionCount)
    {
        var data = new List<byte>();
        var countedSections = new HashSet<int>();
        for (int i = 0; i < symbols.Count; i++)
        {
            if (symbols[i].RegisterCount is not int registerCount ||
                !countedSections.Add(symbols[i].Section))
                continue;
            data.Add(4); // format: byte payload
            data.Add(0x2f); // EIATTR_REGCOUNT
            data.Add(8);
            data.Add(0);
            AddWord(data, (uint)(1 + sectionCount + i));
            AddWord(data, (uint)registerCount);
        }
        return data.ToArray();
    }

    private static void AddWord(List<byte> data, uint value)
    {
        data.Add((byte)value);
        data.Add((byte)(value >> 8));
        data.Add((byte)(value >> 16));
        data.Add((byte)(value >> 24));
    }

    private static uint ArchitectureFlags(uint architecture, bool legacyCuda) =>
        legacyCuda
            ? ((architecture & 0xFF) << 16) | 0x0500 | (architecture & 0xFF)
            : 0x6000000u | ((architecture & 0xFF) << 8) |
                (architecture >= 100 ? 2u : 4u);

    private static void WriteSymbol(
        byte[] data, int offset, int name, byte info, ushort section,
        ulong value, ulong size, byte other = 0)
    {
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset), (uint)name);
        data[offset + 4] = info;
        data[offset + 5] = other;
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(offset + 6), section);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(offset + 8), value);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(offset + 16), size);
    }

    private static void WriteSectionHeader(
        byte[] data, int offset, uint name, uint type, ulong flags,
        ulong fileOffset, ulong size, ulong align, uint link,
        uint info = 0, ulong entrySize = 0)
    {
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset), name);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 4), type);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(offset + 8), flags);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(offset + 24), fileOffset);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(offset + 32), size);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 40), link);
        BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset + 44), info);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(offset + 48), align);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(offset + 56), entrySize);
    }

    private static int Align(int value, int alignment) =>
        (value + alignment - 1) / alignment * alignment;

    private sealed class StringTable
    {
        private readonly List<byte> _bytes = new() { 0 };
        private readonly Dictionary<string, int> _offsets = new(StringComparer.Ordinal);

        public byte[] Bytes => _bytes.ToArray();

        public int Add(string value)
        {
            if (_offsets.TryGetValue(value, out int offset))
                return offset;
            offset = _bytes.Count;
            _offsets[value] = offset;
            _bytes.AddRange(Encoding.ASCII.GetBytes(value));
            _bytes.Add(0);
            return offset;
        }
    }
}
