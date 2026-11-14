using System.Globalization;

namespace nvDismDecode;

// Pre-Volta (SM30-SM62) instructions are 64-bit words.  The legacy CUDA 10.2
// table grammar is stored in isa_json/isa_102_json/; this entry point keeps
// that mode separate from the 128-bit Volta+ decoder.
public static class PreVoltaDecoder
{
    public static void UseArchitecture(
        string architecture, string? toolkit = null)
    {
        string path = ResolveTablePath(architecture, toolkit);
        InstructionDecoder.Load(path);
        // Shared tables name all architectures they cover.  Keep the
        // requested architecture visible to the decoder and callers.
        InstructionDecoder.Table.Architecture =
            "sm_" + Normalize(architecture);
        InstructionDecoder.Table.Toolkit = "102";
    }

    public static string ResolveTablePath(
        string architecture, string? toolkit = null)
    {
        string? normalizedToolkit = InstructionDecoder.NormalizeToolkit(toolkit);
        if (normalizedToolkit != null && normalizedToolkit != "102")
            throw new NotSupportedException(
                $"Pre-Volta {architecture} requires the CUDA 10.2 toolkit, " +
                $"not {normalizedToolkit}.");
        string key = Normalize(architecture);
        string root = Path.Combine(
            InstructionDecoder.TableRoot, "isa_102_json");
        if (Directory.Exists(root))
        {
            foreach (string file in Directory.EnumerateFiles(root, "*.json*")
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase))
            {
                string name = Path.GetFileName(file);
                if (!name.StartsWith(
                        "isa_102_", StringComparison.OrdinalIgnoreCase))
                    continue;
                string suffix;
                if (name.EndsWith(".json.zst", StringComparison.OrdinalIgnoreCase))
                    suffix = ".json.zst";
                else if (name.EndsWith(".json.gz", StringComparison.OrdinalIgnoreCase))
                    suffix = ".json.gz";
                else if (name.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
                    suffix = ".json";
                else
                    continue;
                string archList = name["isa_102_".Length..^suffix.Length];
                if (archList.Split(',').Any(arch =>
                        Normalize(arch) == key))
                    return file;
            }
        }
        throw new NotSupportedException(
            $"No CUDA 10.2 pre-Volta table is available for {architecture}.");
    }

    public static DecodedInstruction Decode(
        ulong word, ulong address = 0, int reuseCode = 0) =>
        InstructionDecoder.Decode(word, 0, address, reuseCode);

    public static DecodedInstruction Decode(
        ReadOnlySpan<byte> data, int offset = 0, ulong address = 0,
        int reuseCode = 0)
    {
        if (offset < 0 || offset + 8 > data.Length)
            throw new ArgumentOutOfRangeException(nameof(offset));
        ulong word = BitConverter.ToUInt64(data[offset..]);
        return Decode(word, address, reuseCode);
    }

    // Pre-Volta code sections use 32-byte bundles: one control word followed
    // by three 8-byte instruction slots.  nvdisasm reports addresses for the
    // instruction slots, not the control word.
    public static IEnumerable<(ulong Address, DecodedInstruction Instruction)>
        DecodeSection(byte[] data, ulong baseAddress = 0)
    {
        // Kepler (SM30-SM37) uses one control word per seven instructions.
        // Maxwell/Pascal (SM50-SM62) use one control word per three.
        int arch = ArchitectureNumber(InstructionDecoder.Table.Architecture);
        bool kepler = arch is >= 30 and <= 37;
        int bundleSize = GetBundleSize(InstructionDecoder.Table.Architecture);
        int slotCount = kepler ? 7 : 3;

        for (int bundle = 0; bundle + bundleSize <= data.Length; bundle += bundleSize)
        {
            ulong control = BitConverter.ToUInt64(data, bundle);
            // SM30 cubins can end with a control-word-free tail of eight
            // instructions.  Real SM30 control words carry the sched tag in
            // the top nibble (0x2...); a tail word such as the final BRA is
            // an instruction and must not be skipped as a control word.
            bool hasControl = arch switch
            {
                30 => control == 0 || ((control >> 60) & 0xF) == 0x2,
                >= 32 and <= 37 => control == 0 ||
                    ((control >> 60) & 0xF) == 0,
                _ => true,
            };
            int firstSlot = hasControl ? 1 : 0;
            int lastSlot = hasControl ? slotCount : 7;
            for (int slot = firstSlot; slot <= lastSlot; slot++)
            {
                int offset = bundle + slot * 8;
                ulong address = baseAddress + (ulong)offset;
                int reuseCode = 0;
                if (!kepler && hasControl)
                {
                    int shift = slot switch
                    {
                        1 => 17,
                        2 => 38,
                        _ => 59,
                    };
                    reuseCode = (int)((control >> shift) & 0xF);
                }
                yield return (address, Decode(data, offset, address, reuseCode));
            }
        }
    }

    public static int GetBundleSize(string architecture) =>
        ArchitectureNumber(architecture) is >= 30 and <= 37 ? 64 : 32;

    private static int ArchitectureNumber(string architecture)
    {
        string value = Normalize(architecture);
        int end = 0;
        while (end < value.Length && char.IsDigit(value[end]))
            end++;
        return int.TryParse(value[..end], out int number) ? number : 0;
    }

    private static string Normalize(string architecture)
    {
        string value = architecture.Trim().ToLowerInvariant();
        if (value.StartsWith("sm", StringComparison.Ordinal))
            value = value[2..];
        if (value.StartsWith("_", StringComparison.Ordinal))
            value = value[1..];
        return value;
    }
}
