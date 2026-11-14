using System.Buffers.Binary;
using System.Text;

namespace nvDismDecode;

// ---------------------------------------------------------------------------
// Minimal PE/COFF section scanner for Windows CUDA host binaries.
//
// CUDA embeds its fatbins in a section whose 8-byte PE name is normally
// ".nv_fatb" (the ELF name ".nv_fatbin" does not fit in a PE section name).
// The wrapper array is normally in ".nvFatBi" (truncated from
// ".nvFatBinSegment").
// ---------------------------------------------------------------------------

public static class PeFatbinReader
{
    private const uint WrapperMagic = 0x466243B1;

    public static bool IsPe(ReadOnlySpan<byte> data) =>
        data.Length >= 0x40 && data[0] == (byte)'M' && data[1] == (byte)'Z';

    public static List<byte[]> FindEmbeddedFatbins(byte[] data)
    {
        if (!IsPe(data))
            throw new InvalidDataException("Not a PE file.");

        int peOffset = BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(0x3C));
        if (peOffset < 0 || peOffset + 24 > data.Length ||
            data[peOffset] != (byte)'P' || data[peOffset + 1] != (byte)'E' ||
            data[peOffset + 2] != 0 || data[peOffset + 3] != 0)
            throw new InvalidDataException("Invalid PE signature.");

        int coff = peOffset + 4;
        int sectionCount = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(coff + 2));
        int optionalHeaderSize = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(coff + 16));
        int sectionTable = peOffset + 24 + optionalHeaderSize;
        ulong imageBase = ReadImageBase(data, peOffset + 24, optionalHeaderSize);

        var result = new List<byte[]>();
        var seenOffsets = new HashSet<int>();
        var sections = new List<PeSection>();
        for (int i = 0; i < sectionCount; i++)
        {
            int offset = sectionTable + i * 40;
            if (offset + 40 > data.Length)
                break;

            uint rawSize = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset + 16));
            uint rawPointer = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset + 20));
            if (rawSize == 0 || (ulong)rawPointer + rawSize > (ulong)data.Length)
                continue;

            sections.Add(new PeSection
            {
                Name = Encoding.ASCII.GetString(data, offset, 8).TrimEnd('\0'),
                VirtualAddress = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset + 12)),
                VirtualSize = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset + 8)),
                RawPointer = rawPointer,
                RawSize = rawSize,
                IsExecutable = (BinaryPrimitives.ReadUInt32LittleEndian(
                    data.AsSpan(offset + 36)) & 0x20000000) != 0,
            });
        }

        // Normal path: sections whose name still contains "fatb".
        foreach (PeSection section in sections)
        {
            if (section.Name.Contains("fatb", StringComparison.OrdinalIgnoreCase))
                FatbinScanner.Scan(
                    data, (int)section.RawPointer,
                    (int)(section.RawPointer + section.RawSize), result, seenOffsets);
        }

        // Independent path: resolve .nvFatBi wrapper pointers.
        if (imageBase != 0)
        {
            foreach (PeSection section in sections)
            {
                if (section.Name.IndexOf("fatbi", StringComparison.OrdinalIgnoreCase) < 0 &&
                    section.Name.IndexOf("segment", StringComparison.OrdinalIgnoreCase) < 0)
                    continue;
                ScanWrappers(data, sections, section, imageBase, result, seenOffsets);
            }
        }

        // If the section was renamed or uses a long COFF name, fall back to
        // scanning every section for the FatBinHeader signature.
        if (result.Count == 0)
        {
            foreach (PeSection section in sections)
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

    private static ulong ReadImageBase(byte[] data, int optionalHeader, int size)
    {
        if (size < 0x20 || optionalHeader + size > data.Length)
            return 0;
        ushort magic = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(optionalHeader));
        return magic switch
        {
            0x20B => BinaryPrimitives.ReadUInt64LittleEndian(data.AsSpan(optionalHeader + 0x18)),
            0x10B => BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(optionalHeader + 0x1C)),
            _ => 0,
        };
    }

    private static void ScanWrappers(
        byte[] data, List<PeSection> sections, PeSection wrapperSection,
        ulong imageBase, List<byte[]> result, HashSet<int> seenOffsets)
    {
        int start = (int)wrapperSection.RawPointer;
        int end = start + (int)wrapperSection.RawSize;
        for (int offset = start; offset + 24 <= end; offset += 8)
        {
            if (BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset)) != WrapperMagic)
                continue;

            int version = BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(offset + 4));
            if (version is not (1 or 2))
                continue;

            ulong pointer = BinaryPrimitives.ReadUInt64LittleEndian(data.AsSpan(offset + 8));
            if (pointer < imageBase)
                continue;
            ulong rva = pointer - imageBase;
            if (!TryMapRva(data, sections, rva, out int fatbinOffset))
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

    private static bool TryMapRva(
        byte[] data, List<PeSection> sections, ulong rva, out int fileOffset)
    {
        foreach (PeSection section in sections)
        {
            ulong span = Math.Max(section.VirtualSize, section.RawSize);
            if (rva >= section.VirtualAddress &&
                rva < section.VirtualAddress + span)
            {
                ulong offset = section.RawPointer + (rva - section.VirtualAddress);
                if (offset < (ulong)data.Length)
                {
                    fileOffset = (int)offset;
                    return true;
                }
            }
        }
        fileOffset = 0;
        return false;
    }

    private sealed class PeSection
    {
        public string Name { get; init; } = "";
        public uint VirtualAddress { get; init; }
        public uint VirtualSize { get; init; }
        public uint RawPointer { get; init; }
        public uint RawSize { get; init; }
        public bool IsExecutable { get; init; }
    }
}
