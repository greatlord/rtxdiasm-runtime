using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using K4os.Compression.LZ4.Streams;
using ZstdSharp;

namespace nvDismDecode;

// ---------------------------------------------------------------------------
// CUDA fatbin container.
//
// A fatbin is the multi-architecture wrapper that nvcc embeds in a host
// binary (normally in .nv_fatbin).  Its entries are either PTX text or ELF
// cubins.  This reader extracts and decompresses the payloads; cubin payloads
// are then handled by CubinFile.
// ---------------------------------------------------------------------------

public sealed class FatbinFile
{
    public int Version { get; init; }
    public List<FatbinEntry> Entries { get; init; } = new();

    public const uint Magic = 0xBA55ED50;
    public const uint OldStyleMagic = 0x1EE55A01;

    // bin_info compression bits.  The selected algorithm is always driven by
    // the entry flags rather than the CUDA version or file name.
    public const ulong CompressionDeflate = 0x1000;
    public const ulong CompressionLz4 = 0x2000;
    public const ulong CompressionLz4V2 = 0x4000;
    public const ulong CompressionZstd = 0x8000;

    public static bool IsFatbin(ReadOnlySpan<byte> data)
    {
        if (data.Length < 16)
            return false;
        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(data);
        return magic == Magic || magic == OldStyleMagic;
    }

    public static FatbinFile Parse(byte[] data)
    {
        if (data.Length >= 4 &&
            BinaryPrimitives.ReadUInt32LittleEndian(data) == OldStyleMagic)
            return ParseOldStyle(data);

        if (!IsFatbin(data))
            throw new InvalidDataException("Not a CUDA fatbin.");

        ushort version = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(4));
        ushort headerSize = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(6));
        ulong totalSize = BinaryPrimitives.ReadUInt64LittleEndian(data.AsSpan(8));

        if (headerSize < 16 || headerSize > data.Length)
            throw new InvalidDataException("Invalid fatbin header size.");

        // fatbin_size is measured from the end of the 16-byte header.
        ulong end = (ulong)headerSize + totalSize;
        if (end > (ulong)data.Length || end < (ulong)headerSize)
            end = (ulong)data.Length;

        var fatbin = new FatbinFile { Version = version };
        ulong offset = headerSize;
        while (offset + 64 <= end)
        {
            ushort kind = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan((int)offset));
            ushort entryVersion = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan((int)offset + 2));
            uint entryHeaderSize = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan((int)offset + 4));
            ulong storedPayloadSize = BinaryPrimitives.ReadUInt64LittleEndian(data.AsSpan((int)offset + 8));
            int compressedSize = BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan((int)offset + 16));
            uint ptxasOptionsOffset = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan((int)offset + 20));
            ushort codeVersionMinor = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan((int)offset + 24));
            ushort codeVersionMajor = BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan((int)offset + 26));
            uint arch = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan((int)offset + 28));
            uint identifierOffset = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan((int)offset + 32));
            uint identifierLength = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan((int)offset + 36));
            ulong binInfo = BinaryPrimitives.ReadUInt64LittleEndian(data.AsSpan((int)offset + 40));
            ulong uncompressedSize = BinaryPrimitives.ReadUInt64LittleEndian(data.AsSpan((int)offset + 56));

            if (entryHeaderSize < 64)
                break;

            string identifier = "";
            if (identifierOffset >= 64 && identifierLength > 0 &&
                identifierLength <= 1024 &&
                (ulong)identifierOffset + identifierLength <= entryHeaderSize &&
                offset + identifierOffset + identifierLength <= (ulong)data.Length)
            {
                identifier = Encoding.UTF8.GetString(
                    data, (int)(offset + identifierOffset), (int)identifierLength)
                    .TrimEnd('\0');
            }

            string ptxasOptions = "";
            if (kind == 1 && ptxasOptionsOffset >= 64 &&
                (ulong)ptxasOptionsOffset + 8 <= entryHeaderSize &&
                offset + ptxasOptionsOffset + 8 <= (ulong)data.Length)
            {
                uint stringOffset = BinaryPrimitives.ReadUInt32LittleEndian(
                    data.AsSpan((int)(offset + ptxasOptionsOffset)));
                uint stringLength = BinaryPrimitives.ReadUInt32LittleEndian(
                    data.AsSpan((int)(offset + ptxasOptionsOffset + 4)));
                if (stringLength > 0 && stringLength <= 4096 &&
                    (ulong)stringOffset + stringLength <= entryHeaderSize &&
                    offset + stringOffset + stringLength <= (ulong)data.Length)
                {
                    ptxasOptions = Encoding.UTF8.GetString(
                        data, (int)(offset + stringOffset), (int)stringLength)
                        .TrimEnd('\0');
                }
            }

            ulong payloadStart = offset + entryHeaderSize;
            ulong storedLength = storedPayloadSize;
            if (payloadStart + storedLength > (ulong)data.Length)
                break;

            // ZLUDA uses the offset-8 "payload_size" as the stored input
            // length for deflate entries; their compressed_size field is
            // normally zero.  The extra alignment bytes are harmless because
            // the zlib end marker terminates the stream.
            bool deflate = (binInfo & CompressionDeflate) != 0;
            int compressedLength = deflate
                ? (int)storedLength
                : compressedSize > 0 ? compressedSize : (int)storedLength;
            if (compressedLength < 0 || payloadStart + (ulong)compressedLength > (ulong)data.Length)
                break;

            byte[] compressed = new byte[compressedLength];
            Array.Copy(data, (long)payloadStart, compressed, 0, compressedLength);
            ulong expectedSize = uncompressedSize > 0 ? uncompressedSize : storedLength;

            string compression;
            byte[] payload;
            try
            {
                (compression, payload) = Decompress(compressed, binInfo, expectedSize);
            }
            catch (Exception ex)
            {
                throw new InvalidDataException(
                    $"Cannot decompress fatbin entry at 0x{offset:x}: {ex.Message}", ex);
            }

            fatbin.Entries.Add(new FatbinEntry
            {
                Kind = kind,
                Version = entryVersion,
                CodeVersionMajor = codeVersionMajor,
                CodeVersionMinor = codeVersionMinor,
                Architecture = arch,
                Identifier = identifier,
                PtxasOptions = ptxasOptions,
                BinInfo = binInfo,
                Compression = compression,
                Payload = payload,
                StoredPayloadSize = (int)storedLength,
                CompressedSize = compressedSize,
                UncompressedSize = uncompressedSize,
            });

            ulong next = offset + entryHeaderSize + storedLength;
            if (next <= offset)
                break;
            offset = next;
        }

        return fatbin;
    }

    // Old-style fatbins (nvcc 3.2 and earlier) do not use the modern
    // FatbinData records.  Their payloads are still ordinary ELF cubins or
    // PTX text, so recover those payloads by locating and sizing each ELF
    // image / PTX stream in the container.
    private static FatbinFile ParseOldStyle(byte[] data)
    {
        ushort version = data.Length >= 6
            ? BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(4))
            : (ushort)0;
        var fatbin = new FatbinFile { Version = version };
        var seen = new HashSet<int>();

        int offset = 0;
        while (offset + 4 <= data.Length)
        {
            if (!IsElfAt(data, offset))
            {
                offset++;
                continue;
            }

            if (!TryGetElfLength(data, offset, out int length) ||
                length <= 0 || offset + length > data.Length)
            {
                offset += 4;
                continue;
            }

            byte[] payload = data.AsSpan(offset, length).ToArray();
            fatbin.Entries.Add(new FatbinEntry
            {
                Kind = 2,
                Version = version,
                Architecture = ReadElfArchitecture(payload),
                Compression = "NONE",
                Payload = payload,
                StoredPayloadSize = payload.Length,
                UncompressedSize = (ulong)payload.Length,
            });
            seen.Add(offset);
            offset += length;
        }

        // PTX entries are plain text in old fatbins.  Only accept streams
        // that contain both a .version and a .target directive.
        int ptxOffset = 0;
        while (ptxOffset < data.Length)
        {
            int start = IndexOfAscii(data, ptxOffset, ".version");
            if (start < 0)
                break;

            int end = start;
            while (end < data.Length && data[end] != 0)
                end++;
            if (end > start + 16)
            {
                ReadOnlySpan<byte> candidate = data.AsSpan(start, end - start);
                string text = Encoding.ASCII.GetString(candidate);
                if (text.Contains(".target", StringComparison.Ordinal) &&
                    text.Contains(".entry", StringComparison.Ordinal) &&
                    !seen.Contains(start))
                {
                    fatbin.Entries.Add(new FatbinEntry
                    {
                        Kind = 1,
                        Version = version,
                        Architecture = ReadPtxArchitecture(text),
                        Compression = "NONE",
                        Payload = candidate.ToArray(),
                        StoredPayloadSize = candidate.Length,
                        UncompressedSize = (ulong)candidate.Length,
                    });
                    seen.Add(start);
                }
            }
            ptxOffset = Math.Max(start + 1, end);
        }

        return fatbin;
    }

    private static bool IsElfAt(byte[] data, int offset) =>
        offset + 4 <= data.Length &&
        data[offset] == 0x7F && data[offset + 1] == (byte)'E' &&
        data[offset + 2] == (byte)'L' && data[offset + 3] == (byte)'F';

    private static int IndexOfAscii(byte[] data, int start, string text)
    {
        byte[] needle = Encoding.ASCII.GetBytes(text);
        int end = data.Length - needle.Length;
        for (int i = Math.Max(0, start); i <= end; i++)
        {
            bool match = true;
            for (int j = 0; j < needle.Length; j++)
            {
                if (data[i + j] != needle[j])
                {
                    match = false;
                    break;
                }
            }
            if (match)
                return i;
        }
        return -1;
    }

    private static bool TryGetElfLength(
        byte[] data, int offset, out int length)
    {
        length = 0;
        int remaining = data.Length - offset;
        if (remaining < 52)
            return false;

        byte elfClass = data[offset + 4];
        ulong shoff;
        ushort shentsize;
        ushort shnum;
        if (elfClass == 2)
        {
            if (remaining < 64)
                return false;
            shoff = BinaryPrimitives.ReadUInt64LittleEndian(
                data.AsSpan(offset + 0x28));
            shentsize = BinaryPrimitives.ReadUInt16LittleEndian(
                data.AsSpan(offset + 0x3A));
            shnum = BinaryPrimitives.ReadUInt16LittleEndian(
                data.AsSpan(offset + 0x3C));
        }
        else if (elfClass == 1)
        {
            shoff = BinaryPrimitives.ReadUInt32LittleEndian(
                data.AsSpan(offset + 0x20));
            shentsize = BinaryPrimitives.ReadUInt16LittleEndian(
                data.AsSpan(offset + 0x2E));
            shnum = BinaryPrimitives.ReadUInt16LittleEndian(
                data.AsSpan(offset + 0x30));
        }
        else
        {
            return false;
        }

        if (shentsize == 0 || shnum == 0)
            return false;

        ulong tableEnd = shoff + (ulong)shentsize * shnum;
        if (tableEnd < shoff || tableEnd > (ulong)remaining)
            return false;

        ulong end = tableEnd;
        for (int i = 0; i < shnum; i++)
        {
            ulong header = shoff + (ulong)i * shentsize;
            if (header + 48 > (ulong)remaining)
                return false;
            ulong sectionOffset = elfClass == 2
                ? BinaryPrimitives.ReadUInt64LittleEndian(
                    data.AsSpan(offset + (int)header + 24))
                : BinaryPrimitives.ReadUInt32LittleEndian(
                    data.AsSpan(offset + (int)header + 16));
            ulong sectionSize = elfClass == 2
                ? BinaryPrimitives.ReadUInt64LittleEndian(
                    data.AsSpan(offset + (int)header + 32))
                : BinaryPrimitives.ReadUInt32LittleEndian(
                    data.AsSpan(offset + (int)header + 20));
            uint sectionType = BinaryPrimitives.ReadUInt32LittleEndian(
                data.AsSpan(offset + (int)header + 4));
            if (sectionType != 8 && sectionOffset + sectionSize > end)
                end = sectionOffset + sectionSize;
        }

        if (end == 0 || end > (ulong)remaining)
            return false;
        length = (int)end;
        return true;
    }

    private static uint ReadElfArchitecture(byte[] payload)
    {
        if (payload.Length < 52)
            return 0;
        bool elf64 = payload[4] == 2;
        int flagsOffset = elf64 ? 0x30 : 0x24;
        if (payload.Length < flagsOffset + 4)
            return 0;
        uint flags = BinaryPrimitives.ReadUInt32LittleEndian(
            payload.AsSpan(flagsOffset));
        uint low = flags & 0xFF;
        uint shifted = (flags >> 8) & 0xFF;
        return low is 2 or 4 ? shifted : low;
    }

    private static uint ReadPtxArchitecture(string text)
    {
        const string marker = ".target sm_";
        int start = text.IndexOf(marker, StringComparison.Ordinal);
        if (start < 0)
            return 0;
        start += marker.Length;
        int end = start;
        while (end < text.Length && char.IsDigit(text[end]))
            end++;
        return end > start && uint.TryParse(text[start..end], out uint value)
            ? value
            : 0;
    }

    private static (string Compression, byte[] Payload) Decompress(
        byte[] payload, ulong binInfo, ulong expectedSize)
    {
        // Compression is selected by the entry's bin_info flags.  Do not
        // assume an algorithm from the CUDA toolkit version: older tools emit
        // LZ4, newer tools emit ZSTD, and other producers emit DEFLATE or
        // LZ4-v2.
        if ((binInfo & CompressionZstd) != 0)
        {
            using var decompressor = new Decompressor();
            byte[] result = decompressor.Unwrap(payload).ToArray();
            return ("ZSTD", result);
        }

        if ((binInfo & CompressionDeflate) != 0)
            return ("DEFLATE", DecompressDeflate(payload));

        if ((binInfo & (CompressionLz4 | CompressionLz4V2)) != 0)
        {
            bool v2 = (binInfo & CompressionLz4V2) != 0;

            // LZ4 frames start with 0x184D2204; blocks start with a token.
            if (payload.Length >= 4 &&
                payload[0] == 0x04 && payload[1] == 0x22 &&
                payload[2] == 0x4D && payload[3] == 0x18)
            {
                return (v2 ? "LZ4v2-FRAME" : "LZ4-FRAME", DecompressLz4Frame(payload));
            }

            if (expectedSize == 0 || expectedSize > int.MaxValue)
                throw new InvalidDataException(
                    "LZ4 entry has no usable uncompressed size.");

            try
            {
                byte[] result = DecompressLz4Block(payload, (int)expectedSize);
                return (v2 ? "LZ4v2" : "LZ4", result);
            }
            catch (InvalidDataException)
            {
                // Some producers use a frame even without the magic being
                // recognized by the header check above.
                return (v2 ? "LZ4v2-FRAME" : "LZ4-FRAME", DecompressLz4Frame(payload));
            }
        }

        // Some producers leave the flag clear but still write a ZSTD frame.
        if (payload.Length >= 4 && payload[0] == 0x28 && payload[1] == 0xB5 &&
            payload[2] == 0x2F && payload[3] == 0xFD)
        {
            using var decompressor = new Decompressor();
            byte[] result = decompressor.Unwrap(payload).ToArray();
            return ("ZSTD", result);
        }

        return ("NONE", payload);
    }

    private static byte[] DecompressDeflate(byte[] payload)
    {
        // CUDA's COMPRESSED_DEFLATE entries are zlib-wrapped deflate streams.
        // Some producers leave payload_size at zero, so the slice may include
        // alignment zeros after the zlib end marker.
        try
        {
            using var input = new MemoryStream(payload);
            using var zlib = new ZLibStream(input, CompressionMode.Decompress);
            using var output = new MemoryStream();
            zlib.CopyTo(output);
            return output.ToArray();
        }
        catch (InvalidDataException)
        {
            // A few producers store raw deflate without the zlib header.
            using var input = new MemoryStream(payload);
            using var deflate = new DeflateStream(input, CompressionMode.Decompress);
            using var output = new MemoryStream();
            deflate.CopyTo(output);
            return output.ToArray();
        }
    }

    private static byte[] DecompressLz4Frame(byte[] payload)
    {
        using var input = new MemoryStream(payload);
        using var lz4 = LZ4Stream.Decode(input);
        using var output = new MemoryStream();
        lz4.CopyTo(output);
        return output.ToArray();
    }

    // LZ4 block format as used by CUDA's legacy fatbin compression.  It is
    // the standard token/literal/match encoding with 16-bit little-endian
    // offsets; matches may overlap.
    private static byte[] DecompressLz4Block(byte[] input, int outputSize)
    {
        var output = new byte[outputSize];
        int ipos = 0;
        int opos = 0;

        while (ipos < input.Length)
        {
            int token = input[ipos];
            int literalLength = token >> 4;
            int matchLength = 4 + (token & 0xF);

            if (literalLength == 0xF)
            {
                int value;
                do
                {
                    if (++ipos >= input.Length)
                        throw new InvalidDataException("Truncated LZ4 literal length.");
                    value = input[ipos];
                    literalLength += value;
                }
                while (value == 0xFF);
            }

            ipos++;
            if (ipos + literalLength > input.Length ||
                opos + literalLength > outputSize)
                throw new InvalidDataException("Truncated LZ4 literal data.");

            Buffer.BlockCopy(input, ipos, output, opos, literalLength);
            ipos += literalLength;
            opos += literalLength;

            if (ipos >= input.Length || opos >= outputSize)
                break;

            if (ipos + 2 > input.Length)
                throw new InvalidDataException("Truncated LZ4 match offset.");

            int backOffset = input[ipos] | (input[ipos + 1] << 8);
            ipos += 2;
            if (backOffset <= 0 || backOffset > opos)
                throw new InvalidDataException("Invalid LZ4 match offset.");

            if (matchLength == 0xF + 4)
            {
                int value;
                do
                {
                    if (ipos >= input.Length)
                        throw new InvalidDataException("Truncated LZ4 match length.");
                    value = input[ipos++];
                    matchLength += value;
                }
                while (value == 0xFF);
            }

            if (opos + matchLength > outputSize)
                throw new InvalidDataException("LZ4 match exceeds output size.");

            for (int i = 0; i < matchLength; i++)
                output[opos + i] = output[opos - backOffset + i];
            opos += matchLength;
        }

        if (opos != outputSize)
            throw new InvalidDataException(
                $"LZ4 decompressed to {opos} bytes, expected {outputSize}.");

        return output;
    }

    public static string ArchName(uint arch) => $"sm_{arch}";
}

public sealed class FatbinEntry
{
    public ushort Kind { get; init; }          // 1 = PTX, 2 = ELF cubin
    public ushort Version { get; init; }
    public ushort CodeVersionMajor { get; init; }
    public ushort CodeVersionMinor { get; init; }
    public uint Architecture { get; init; }
    public string Identifier { get; init; } = "";
    public string PtxasOptions { get; init; } = "";
    public ulong BinInfo { get; init; }
    public string Compression { get; init; } = "NONE";
    public int StoredPayloadSize { get; init; }
    public int CompressedSize { get; init; }
    public ulong UncompressedSize { get; init; }
    public byte[] Payload { get; init; } = Array.Empty<byte>();

    public bool IsCubin => (Kind is 2 or 4) && Payload.Length >= 4 &&
        Payload[0] == 0x7F && Payload[1] == (byte)'E' &&
        Payload[2] == (byte)'L' && Payload[3] == (byte)'F';

    public bool IsPtx => Kind == 1;
    public bool IsLtoIr => Kind == 8;
    public bool IsTileIr => Kind == 0x80;

    public string KindName => Kind switch
    {
        1 => "ptx",
        2 => "elf",
        4 => "old cubin",
        8 => "nvvm",
        0x10 => "capmerc",
        0x20 => "index",
        0x40 => "relocatable ptx",
        0x80 => "tile ir",
        0x100 => "concatenated entry",
        _ => $"unknown({Kind})",
    };
}

public static class FatbinScanner
{
    // Scan a byte range (normally one ELF or PE section) for one or more
    // FatBinHeader blocks.  The returned blobs are safe to pass to
    // FatbinFile.Parse.
    public static void Scan(
        byte[] data, int start, int end, List<byte[]> result,
        HashSet<int>? seenOffsets = null)
    {
        if (start < 0 || end > data.Length || start >= end)
            return;

        int offset = start;
        while (offset + 16 <= end)
        {
            uint magic = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset));
            bool oldStyle = magic == FatbinFile.OldStyleMagic;
            if (magic != FatbinFile.Magic && !oldStyle)
            {
                offset++;
                continue;
            }

            long length;
            if (oldStyle)
            {
                // The legacy container has no reliable structured size for
                // payload recovery; hand the rest of the host section to the
                // old-style parser.
                length = end - offset;
            }
            else
            {
                ushort headerSize = BinaryPrimitives.ReadUInt16LittleEndian(
                    data.AsSpan(offset + 6));
                ulong totalSize = BinaryPrimitives.ReadUInt64LittleEndian(
                    data.AsSpan(offset + 8));
                length = headerSize + (long)totalSize;
                if (length < 16 || offset + length > end)
                    length = end - offset;
            }

            if (seenOffsets == null || seenOffsets.Add(offset))
            {
                var blob = new byte[length];
                Array.Copy(data, offset, blob, 0, (int)length);
                result.Add(blob);
            }
            offset += (int)length;
        }
    }
}
