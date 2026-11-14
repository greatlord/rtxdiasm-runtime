using System.Buffers.Binary;
using System.Text;

namespace nvDismDecode;

// ---------------------------------------------------------------------------
// Minimal COFF object scanner for CUDA host object files.
//
// A relocatable host .obj has no MZ/PE header, but still carries the
// .nv_fatbin data section.  Wrapper pointers usually need relocations, so
// this reader scans section data for the FatBinHeader signature instead.
// ---------------------------------------------------------------------------

public static class CoffFatbinReader
{
    public static bool IsCoff(ReadOnlySpan<byte> data)
    {
        if (data.Length < 20 || data[0] == (byte)'M' && data[1] == (byte)'Z')
            return false;

        ushort machine = BinaryPrimitives.ReadUInt16LittleEndian(data);
        ushort sectionCount = BinaryPrimitives.ReadUInt16LittleEndian(data[2..]);
        ushort optionalSize = BinaryPrimitives.ReadUInt16LittleEndian(data[16..]);
        if (sectionCount == 0)
            return false;
        if (machine is not (0x014C or 0x8664 or 0xAA64 or 0x01C0 or 0x01C4))
            return false;

        long tableEnd = 20L + optionalSize + sectionCount * 40L;
        return tableEnd <= data.Length;
    }

    public static List<byte[]> FindEmbeddedFatbins(byte[] data)
    {
        if (!IsCoff(data))
            throw new InvalidDataException("Not a COFF object file.");

        ushort sectionCount = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(2));
        ushort optionalSize = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(16));
        uint symbolTableOffset = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(8));
        uint symbolCount = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(12));
        int sectionTable = 20 + optionalSize;

        var result = new List<byte[]>();
        var seenOffsets = new HashSet<int>();
        var sections = new List<CoffSection>();
        for (int i = 0; i < sectionCount; i++)
        {
            int offset = sectionTable + i * 40;
            if (offset + 40 > data.Length)
                break;

            string name = ReadSectionName(
                data, offset, symbolTableOffset, symbolCount);
            uint rawSize = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset + 16));
            uint rawPointer = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset + 20));
            if (rawSize == 0 || (ulong)rawPointer + rawSize > (ulong)data.Length)
                continue;

            sections.Add(new CoffSection
            {
                Name = name,
                RawPointer = rawPointer,
                RawSize = rawSize,
                IsExecutable = (BinaryPrimitives.ReadUInt32LittleEndian(
                    data.AsSpan(offset + 36)) & 0x20000000) != 0,
            });
        }

        foreach (CoffSection section in sections)
        {
            if (section.Name.Contains("fatb", StringComparison.OrdinalIgnoreCase))
            {
                FatbinScanner.Scan(
                    data, (int)section.RawPointer,
                    (int)(section.RawPointer + section.RawSize), result, seenOffsets);
            }
        }

        if (result.Count == 0)
        {
            foreach (CoffSection section in sections)
            {
                if (section.IsExecutable)
                    continue;
                FatbinScanner.Scan(
                    data, (int)section.RawPointer,
                    (int)(section.RawPointer + section.RawSize), result, seenOffsets);
            }
        }

        return result;
    }

    private static string ReadSectionName(
        byte[] data, int sectionOffset, uint symbolTableOffset, uint symbolCount)
    {
        int length = 0;
        while (length < 8 && data[sectionOffset + length] != 0)
            length++;
        string name = Encoding.ASCII.GetString(data, sectionOffset, length);
        if (!name.StartsWith('/') ||
            !int.TryParse(name.AsSpan(1), out int stringOffset))
            return name;

        long stringTable = symbolTableOffset + (long)symbolCount * 18;
        if (stringTable + 4 > data.Length)
            return name;
        if (stringTable + stringOffset >= data.Length)
            return name;

        int start = (int)(stringTable + stringOffset);
        int end = start;
        while (end < data.Length && data[end] != 0)
            end++;
        return Encoding.ASCII.GetString(data, start, end - start);
    }

    private sealed class CoffSection
    {
        public string Name { get; init; } = "";
        public uint RawPointer { get; init; }
        public uint RawSize { get; init; }
        public bool IsExecutable { get; init; }
    }
}
