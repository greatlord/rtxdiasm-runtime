namespace rtxdisam;

// JSON output is a CLI concern.  The core library builds the document model,
// but the CLI supplies the System.Text.Json serializer so NativeAOT does not
// need reflection-based serialization.
internal static class JsonOutput
{
    private static Func<object, bool, string>? _serializer;

    public static void Configure(Func<object, bool, string> serializer)
    {
        ArgumentNullException.ThrowIfNull(serializer);
        _serializer = serializer;
    }

    public static string Serialize(object value, bool indented = false)
    {
        Func<object, bool, string> serializer = _serializer
            ?? throw new InvalidOperationException(
                "JSON output is only available through the rtxdisam CLI.");
        return serializer(value, indented);
    }
}
