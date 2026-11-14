using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using rtxdisam;

namespace rtxdisam.native;

public static unsafe class Exports
{
    [ThreadStatic]
    private static IntPtr _lastError;

    [UnmanagedCallersOnly(EntryPoint = "rtxdisam_cubin_to_text")]
    public static int CubinToText(
        byte* path, byte* entry, byte* parameters, int output,
        byte** sass, byte** ptx)
    {
        ClearOutputs(sass, ptx);
        return Run(() =>
        {
            string? pathText = Marshal.PtrToStringUTF8((IntPtr)path);
            if (string.IsNullOrEmpty(pathText))
                throw new ArgumentException("input path is empty");
            string? entryText = Marshal.PtrToStringUTF8((IntPtr)entry);
            string? parameterSpec = Marshal.PtrToStringUTF8((IntPtr)parameters);
            RtxDisamText result = RtxDisamApi.ContainerToText(
                pathText, entryText, ParseParameters(parameterSpec),
                (RtxDisamOutput)output);
            WriteResult(result, output, sass, ptx);
        });
    }

    [UnmanagedCallersOnly(EntryPoint = "rtxdisam_cubin_bytes_to_text")]
    public static int CubinBytesToText(
        byte* data, nuint size, byte* entry, byte* parameters, int output,
        byte** sass, byte** ptx)
    {
        return CubinBytesToTextCore(
            data, size, entry, parameters, null, output, sass, ptx);
    }

    [UnmanagedCallersOnly(
        EntryPoint = "rtxdisam_cubin_bytes_to_text_with_target")]
    public static int CubinBytesToTextWithTarget(
        byte* data, nuint size, byte* entry, byte* parameters, byte* target,
        int output, byte** sass, byte** ptx)
    {
        return CubinBytesToTextCore(
            data, size, entry, parameters, target, output, sass, ptx);
    }

    private static int CubinBytesToTextCore(
        byte* data, nuint size, byte* entry, byte* parameters, byte* target,
        int output, byte** sass, byte** ptx)
    {
        ClearOutputs(sass, ptx);
        return Run(() =>
        {
            if (data == null || size == 0)
                throw new ArgumentException("input buffer is empty");
            if (size > int.MaxValue)
                throw new ArgumentException("input buffer is too large");
            var bytes = new byte[(int)size];
            new ReadOnlySpan<byte>(data, checked((int)size)).CopyTo(bytes);
            string? entryText = Marshal.PtrToStringUTF8((IntPtr)entry);
            string? parameterSpec = Marshal.PtrToStringUTF8((IntPtr)parameters);
            string? targetText = Marshal.PtrToStringUTF8((IntPtr)target);
            RtxDisamText result = RtxDisamApi.ContainerToText(
                bytes, entryText, ParseParameters(parameterSpec),
                (RtxDisamOutput)output, selectionTarget: targetText);
            WriteResult(result, output, sass, ptx);
        });
    }

    [UnmanagedCallersOnly(EntryPoint = "rtxdisam_free")]
    public static void Free(void* memory)
    {
        if (memory != null)
            NativeMemory.Free(memory);
    }

    [UnmanagedCallersOnly(EntryPoint = "rtxdisam_last_error")]
    public static byte* LastError() => (byte*)_lastError;

    [UnmanagedCallersOnly(EntryPoint = "rtxdisam_version")]
    public static byte* Version() => Utf8("2.3.0");

    private static int Run(Action action)
    {
        try
        {
            action();
            SetError(null);
            return 0;
        }
        catch (Exception ex)
        {
            SetError(ex.ToString());
            return 1;
        }
    }

    private static void ClearOutputs(byte** sass, byte** ptx)
    {
        if (sass != null)
            *sass = null;
        if (ptx != null)
            *ptx = null;
    }

    private static void WriteResult(
        RtxDisamText result, int output, byte** sass, byte** ptx)
    {
        if (output <= 0 || (output & ~(int)RtxDisamOutput.Both) != 0)
            throw new ArgumentOutOfRangeException(nameof(output));
        bool wantSass = (output & (int)RtxDisamOutput.Sass) != 0;
        bool wantPtx = (output & (int)RtxDisamOutput.Ptx) != 0;
        if (wantSass && sass == null)
            throw new ArgumentException(
                "a SASS output pointer is required", nameof(sass));
        if (wantPtx && ptx == null)
            throw new ArgumentException(
                "a PTX output pointer is required", nameof(ptx));
        if (sass != null)
            *sass = null;
        if (ptx != null)
            *ptx = null;
        if (wantSass)
            *sass = Utf8(result.Sass ?? "");
        if (wantPtx)
            *ptx = Utf8(result.Ptx ?? "");
    }

    private static IReadOnlyList<PtxParameter> ParseParameters(string? spec)
    {
        if (string.IsNullOrWhiteSpace(spec))
            return Array.Empty<PtxParameter>();
        var parameters = new List<PtxParameter>();
        foreach (string raw in spec.Split(
                     new[] { ';', '\n', '\r' },
                     StringSplitOptions.RemoveEmptyEntries))
        {
            string text = raw.Trim();
            if (text.Length == 0)
                continue;
            string[] parts = text.Split(':');
            if (parts.Length is < 3 or > 4)
                throw new FormatException(
                    $"invalid parameter '{text}'; expected name:type:offset[:size]");
            string name = parts[0].Trim();
            string type = parts[1].Trim();
            if (name.Length == 0 ||
                type is not ("u32" or "s32" or "u64"))
                throw new FormatException($"invalid parameter '{text}'");
            int offset = ParseInteger(parts[2], text);
            int size = parts.Length == 4
                ? ParseInteger(parts[3], text)
                : type == "u64" ? 8 : 4;
            parameters.Add(new PtxParameter(name, type, offset, size));
        }
        return parameters;
    }

    private static int ParseInteger(string text, string source)
    {
        text = text.Trim();
        bool hex = text.StartsWith("0x", StringComparison.OrdinalIgnoreCase);
        if (!int.TryParse(
                hex ? text[2..] : text,
                hex ? NumberStyles.HexNumber : NumberStyles.Integer,
                CultureInfo.InvariantCulture, out int value))
            throw new FormatException($"invalid number in parameter '{source}'");
        return value;
    }

    private static byte* Utf8(string text)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(text);
        byte* memory = (byte*)NativeMemory.Alloc((nuint)bytes.Length + 1);
        bytes.AsSpan().CopyTo(new Span<byte>(memory, bytes.Length));
        memory[bytes.Length] = 0;
        return memory;
    }

    private static void SetError(string? message)
    {
        if (_lastError != IntPtr.Zero)
        {
            NativeMemory.Free((void*)_lastError);
            _lastError = IntPtr.Zero;
        }
        if (message != null)
            _lastError = (IntPtr)Utf8(message);
    }
}
