using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;
using K4os.Compression.LZ4;
using nvDismDecode;
using ZstdSharp;

namespace rtxdisam;

public enum FatbinCompression
{
    None = 0,
    Deflate = 0x1000,
    Lz4 = 0x2000,
    Lz4V2 = 0x4000,
    Zstd = 0x8000,
}

public enum FatbinPlatform : ulong
{
    Linux = 0x10,
    Windows = 0x40,
}

public enum FatbinImageKind : ushort
{
    Ptx = 1,
    Elf = 2,
    Ltoir = 8,
    Index = 0x20,
    TileIr = 0x80,
}

public enum FatbinToolkit
{
    Auto,
    Cuda102,
    Cuda134,
}

public sealed record FatbinImage(
    byte[] Payload,
    uint Architecture,
    FatbinImageKind Kind = FatbinImageKind.Elf,
    string? ArchitectureKey = null,
    string? Identifier = null,
    string? PtxasOptions = null);

// Writes CUDA fatbin containers.  The layout follows the CUDA 13.4
// FatBinHeader/FatBinEntry format; cubin entries can be byte-for-byte
// equivalent to `fatbinary --create ... --64`.
public static class FatbinWriter
{
    public static void Write(string path, byte[] cubin, uint architecture) =>
        Write(path, cubin, architecture, FatbinCompression.None);

    public static void Write(
        string path, byte[] cubin, uint architecture,
        FatbinCompression compression) =>
        Write(
            path,
            new[] { new FatbinImage(cubin, architecture) },
            compression);

    public static void Write(
        string path, IReadOnlyList<FatbinImage> images,
        FatbinCompression compression = FatbinCompression.None,
        bool is64Bit = true, FatbinPlatform? platform = null,
        FatbinToolkit toolkit = FatbinToolkit.Auto)
    {
        ArgumentNullException.ThrowIfNull(images);
        if (images.Count == 0)
            throw new InvalidDataException("A fatbin needs at least one image.");
        if (toolkit == FatbinToolkit.Cuda102 &&
            compression is FatbinCompression.Zstd or FatbinCompression.Lz4V2 or
                FatbinCompression.Deflate)
            throw new NotSupportedException(
                "CUDA 10.2 fatbins support only uncompressed or LZ4 entries.");

        FatbinPlatform selectedPlatform = platform ??
            (OperatingSystem.IsWindows()
                ? FatbinPlatform.Windows
                : FatbinPlatform.Linux);
        ulong baseFlags = (is64Bit ? 1UL : 0UL) | (ulong)selectedPlatform;

        var entries = new List<byte[]>(images.Count);
        ulong bodySize = 0;
        foreach (FatbinImage image in images)
        {
            byte[] entry = BuildEntry(
                image, compression, baseFlags, toolkit);
            entries.Add(entry);
            bodySize += (ulong)entry.Length;
            if (bodySize > int.MaxValue)
                throw new InvalidDataException("Fatbin is too large.");
        }

        const int fatbinHeaderSize = 16;
        byte[] data = new byte[fatbinHeaderSize + (int)bodySize];
        BinaryPrimitives.WriteUInt32LittleEndian(
            data.AsSpan(0), FatbinFile.Magic);
        BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(4), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(
            data.AsSpan(6), fatbinHeaderSize);
        BinaryPrimitives.WriteUInt64LittleEndian(data.AsSpan(8), bodySize);

        int offset = fatbinHeaderSize;
        foreach (byte[] entry in entries)
        {
            entry.CopyTo(data, offset);
            offset += entry.Length;
        }
        File.WriteAllBytes(path, data);
    }

    private static byte[] BuildEntry(
        FatbinImage image, FatbinCompression compression, ulong baseFlags,
        FatbinToolkit toolkit)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (image.Payload == null || image.Payload.Length == 0)
            throw new InvalidDataException("Fatbin image payload is empty.");

        bool elf = image.Kind == FatbinImageKind.Elf;
        bool ptx = image.Kind == FatbinImageKind.Ptx;
        bool ltoir = image.Kind == FatbinImageKind.Ltoir;
        bool tileir = image.Kind == FatbinImageKind.TileIr;
        bool index = image.Kind == FatbinImageKind.Index;
        if (!elf && !ptx && !ltoir && !tileir && !index)
            throw new InvalidDataException(
                $"Unsupported fatbin image kind {image.Kind}.");
        if ((elf || ptx || ltoir) && image.Architecture == 0)
            throw new InvalidDataException(
                "Fatbin image architecture is missing.");

        (uint architecture, char suffix) = ParseArchitectureKey(
            image.ArchitectureKey, image.Architecture);
        bool legacyCuda = false;
        bool modernCuda = false;
        ushort codeVersionMajor;
        ushort codeVersionMinor;
        if (elf)
        {
            byte[] cubin = image.Payload;
            bool isElf = cubin.Length >= 9 &&
                cubin[0] == 0x7F && cubin[1] == (byte)'E' &&
                cubin[2] == (byte)'L' && cubin[3] == (byte)'F';
            legacyCuda = isElf && cubin[7] == 0x33 && cubin[8] == 7;
            modernCuda = isElf && cubin[7] == 0x41 && cubin[8] == 8;
            if (!legacyCuda && !modernCuda)
                throw new InvalidDataException(
                    "Unsupported cubin ABI; cannot infer fatbin code version.");
            if (legacyCuda && architecture > 75)
                throw new InvalidDataException(
                    $"CUDA 10.x ABI is not valid for SM{architecture}.");
            if (modernCuda && architecture < 75)
                throw new InvalidDataException(
                    $"CUDA 13.x ABI is not valid for SM{architecture}.");
            codeVersionMajor = 1;
            codeVersionMinor = legacyCuda ? (ushort)7 : (ushort)8;
            if (toolkit == FatbinToolkit.Cuda102 && modernCuda)
                throw new InvalidDataException(
                    "CUDA 10.2 fatbins cannot contain CUDA 13.x ABI cubins.");
        }
        else if (ptx)
        {
            if (toolkit == FatbinToolkit.Cuda102 &&
                architecture > 75)
                throw new InvalidDataException(
                    "CUDA 10.2 PTX fatbins support architectures up to sm_75.");
            (codeVersionMajor, codeVersionMinor) =
                ParsePtxCodeVersion(image.Payload);
        }
        else if (ltoir)
        {
            if (image.Payload.Length < 4 ||
                !(image.Payload.AsSpan(0, 4).SequenceEqual(
                      new byte[] { 0x42, 0x43, 0xC0, 0xDE }) ||
                  (image.Payload[0] == (byte)'B' &&
                   image.Payload[1] == (byte)'C')))
                throw new InvalidDataException(
                    "LTOIR payload is not LLVM bitcode.");
            codeVersionMajor = 53;
            codeVersionMinor = 20;
        }
        else
        {
            codeVersionMajor = 1;
            codeVersionMinor = 8;
        }

        byte[] stored;
        ulong binInfo = baseFlags;
        if (elf && architecture >= 100)
        {
            binInfo |= 0x1000000UL;
            if (suffix == 'a')
                binInfo |= 0x100000UL;
            else if (suffix == 'f')
                binInfo |= 0x200000UL;
        }
        ulong uncompressedSize = 0;
        if (compression != FatbinCompression.None)
        {
            byte[] source = ptx
                ? EnsureNulTerminated(image.Payload)
                : image.Payload;
            stored = compression switch
            {
                FatbinCompression.Deflate => CompressDeflate(source),
                FatbinCompression.Lz4 => CompressLz4(source),
                FatbinCompression.Lz4V2 => CompressLz4(source),
                FatbinCompression.Zstd => CompressZstd(source),
                _ => source,
            };
            binInfo |= (ulong)compression;
            uncompressedSize = (ulong)source.Length;
        }
        else
        {
            stored = ptx ? EnsureNulTerminated(image.Payload) : image.Payload;
        }

        byte[] identifier = Encoding.ASCII.GetBytes(image.Identifier ?? "");
        int identifierStorage = identifier.Length == 0
            ? 0
            : Align8(identifier.Length + 1);
        byte[] options = Encoding.ASCII.GetBytes(
            elf || ltoir || tileir || index
                ? ""
                : image.PtxasOptions ?? "");
        bool extendedElf = elf && architecture >= 100;
        bool metadataEntry = ptx || ltoir;
        int optionsStorage = 0;
        if (metadataEntry)
        {
            optionsStorage = toolkit == FatbinToolkit.Cuda102
                ? (options.Length == 0 ? 0 : Align8(options.Length + 1))
                : Align8(Math.Max(1, options.Length + 1));
        }
        byte[] extended = extendedElf
            ? BuildBlackwellExtendedHeader(architecture)
            : Array.Empty<byte>();

        int headerSize;
        int ptxOptionsOffset = 0;
        int identifierOffset = 0;
        int extendedOffset = 0;
        if (elf)
        {
            headerSize = 64 + identifierStorage + extended.Length;
            if (identifier.Length > 0)
                identifierOffset = 64;
            if (extendedElf)
            {
                extendedOffset = 64 + identifierStorage;
                ptxOptionsOffset = extendedOffset;
                BinaryPrimitives.WriteUInt32LittleEndian(
                    extended.AsSpan(0), (uint)(extendedOffset + 8));
            }
        }
        else if (metadataEntry)
        {
            if (toolkit == FatbinToolkit.Cuda102)
            {
                headerSize = 64 + 8 + optionsStorage + identifierStorage;
                ptxOptionsOffset = 64;
                if (identifier.Length > 0)
                    identifierOffset = 72 + optionsStorage;
            }
            else
            {
                headerSize = 64 + identifierStorage + 8 + optionsStorage;
                ptxOptionsOffset = 64 + identifierStorage;
                if (identifier.Length > 0)
                    identifierOffset = 64;
            }
        }
        else
        {
            headerSize = 64 + identifierStorage;
            if (identifier.Length > 0)
                identifierOffset = 64;
        }

        int paddedPayload = Align8(stored.Length);
        byte[] entry = new byte[headerSize + paddedPayload];
        BinaryPrimitives.WriteUInt16LittleEndian(
            entry.AsSpan(0), (ushort)image.Kind);
        BinaryPrimitives.WriteUInt16LittleEndian(entry.AsSpan(2), 0x0101);
        BinaryPrimitives.WriteUInt32LittleEndian(
            entry.AsSpan(4), (uint)headerSize);
        BinaryPrimitives.WriteUInt64LittleEndian(
            entry.AsSpan(8), (ulong)paddedPayload);
        BinaryPrimitives.WriteInt32LittleEndian(
            entry.AsSpan(16),
            compression == FatbinCompression.None ? 0 : stored.Length);
        BinaryPrimitives.WriteUInt32LittleEndian(
            entry.AsSpan(20), (uint)ptxOptionsOffset);
        BinaryPrimitives.WriteUInt16LittleEndian(
            entry.AsSpan(24), codeVersionMinor);
        BinaryPrimitives.WriteUInt16LittleEndian(
            entry.AsSpan(26), codeVersionMajor);
        BinaryPrimitives.WriteUInt32LittleEndian(
            entry.AsSpan(28), architecture);
        BinaryPrimitives.WriteUInt32LittleEndian(
            entry.AsSpan(32), (uint)identifierOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(
            entry.AsSpan(36), (uint)identifier.Length);
        BinaryPrimitives.WriteUInt64LittleEndian(entry.AsSpan(40), binInfo);
        BinaryPrimitives.WriteUInt64LittleEndian(entry.AsSpan(48), 0);
        BinaryPrimitives.WriteUInt64LittleEndian(
            entry.AsSpan(56), uncompressedSize);

        if (identifier.Length > 0)
        {
            identifier.CopyTo(entry, identifierOffset);
        }
        if (extendedElf)
        {
            extended.CopyTo(entry, extendedOffset);
        }
        if (metadataEntry)
        {
            int descriptorOffset = ptxOptionsOffset;
            if (toolkit == FatbinToolkit.Cuda102)
            {
                if (options.Length > 0)
                {
                    BinaryPrimitives.WriteUInt32LittleEndian(
                        entry.AsSpan(descriptorOffset), 72u);
                    BinaryPrimitives.WriteUInt32LittleEndian(
                        entry.AsSpan(descriptorOffset + 4),
                        (uint)options.Length);
                    options.CopyTo(entry, 72);
                }
            }
            else
            {
                BinaryPrimitives.WriteUInt32LittleEndian(
                    entry.AsSpan(descriptorOffset),
                    (uint)(descriptorOffset + 8));
                BinaryPrimitives.WriteUInt32LittleEndian(
                    entry.AsSpan(descriptorOffset + 4),
                    (uint)options.Length);
                if (options.Length > 0)
                    options.CopyTo(entry, descriptorOffset + 8);
            }
        }
        Array.Copy(stored, 0, entry, headerSize, stored.Length);
        return entry;
    }

    private static (uint Architecture, char Suffix) ParseArchitectureKey(
        string? key, uint fallback)
    {
        if (string.IsNullOrEmpty(key))
            return (fallback, '\0');
        Match match = Regex.Match(
            key.Trim(), @"^sm_(\d+)([a-z]*)$",
            RegexOptions.IgnoreCase);
        if (!match.Success ||
            !uint.TryParse(match.Groups[1].Value, out uint architecture))
            throw new InvalidDataException(
                $"invalid architecture key '{key}'");
        if (architecture != fallback)
            throw new InvalidDataException(
                $"architecture key '{key}' does not match SM{fallback}");
        string suffix = match.Groups[2].Value.ToLowerInvariant();
        if (suffix.Length == 0)
            return (architecture, '\0');
        if (suffix is not ("a" or "f"))
            throw new InvalidDataException(
                $"unsupported architecture suffix in '{key}'");
        return (architecture, suffix[0]);
    }

    private static byte[] BuildBlackwellExtendedHeader(uint architecture)
    {
        // CUDA 13.4 capability records.  SM100/SM120 use the 120-byte form;
        // SM103/SM107/SM110/SM121 use the 112-byte form.
        string hex = architecture is 100 or 120
            ? architecture == 120
                ? "48000000280000000002090000020c000002020100030d01010307010102030000040b08005000000000000000030e000400000000000000"
                : "48000000280000000002090000020c000002020100030d01010307010102030000040b08000900000000000000030e000400000000000000"
            : "48000000240000000002090000020c000002020100030d010102030000040b08000000000000000000030e0004000000";
        return Convert.FromHexString(hex);
    }

    private static byte[] EnsureNulTerminated(byte[] payload)
    {
        if (payload.Length > 0 && payload[^1] == 0)
            return payload;
        byte[] result = new byte[payload.Length + 1];
        payload.CopyTo(result, 0);
        return result;
    }

    private static int Align8(int value) => (value + 7) & ~7;

    private static (ushort Major, ushort Minor) ParsePtxCodeVersion(
        byte[] payload)
    {
        int length = Math.Min(payload.Length, 4096);
        string text = Encoding.ASCII.GetString(payload, 0, length);
        Match match = Regex.Match(
            text, @"^\s*\.version\s+(\d+)\.(\d+)",
            RegexOptions.Multiline);
        if (!match.Success ||
            !ushort.TryParse(match.Groups[1].Value, out ushort major) ||
            !ushort.TryParse(match.Groups[2].Value, out ushort minor))
            return (1, 8);
        return (major, minor);
    }

    private static byte[] CompressDeflate(byte[] source)
    {
        using var output = new MemoryStream();
        using (var zlib = new ZLibStream(
            output, CompressionLevel.Optimal, leaveOpen: true))
            zlib.Write(source);
        return output.ToArray();
    }

    private static byte[] CompressLz4(byte[] source)
    {
        byte[] target = new byte[LZ4Codec.MaximumOutputSize(source.Length)];
        int length = LZ4Codec.Encode(
            source, 0, source.Length, target, 0, target.Length,
            LZ4Level.L09_HC);
        return target[..length];
    }

    private static byte[] CompressZstd(byte[] source)
    {
        using var compressor = new Compressor();
        return compressor.Wrap(source).ToArray();
    }
}
