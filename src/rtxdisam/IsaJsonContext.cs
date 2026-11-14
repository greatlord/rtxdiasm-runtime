using System.Text.Json.Serialization;

namespace nvDismDecode;

// NativeAOT disables reflection-based System.Text.Json by default.  Keep the
// ISA table reader working in the native shared library with a source-generated
// metadata context.
[JsonSourceGenerationOptions(
    PropertyNameCaseInsensitive = true,
    GenerationMode = JsonSourceGenerationMode.Metadata)]
[JsonSerializable(typeof(IsaTable))]
internal partial class IsaJsonContext : JsonSerializerContext
{
}
