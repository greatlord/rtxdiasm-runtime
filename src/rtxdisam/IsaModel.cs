using System.IO.Compression;
using System.Text.Json;
using System.Text.Json.Serialization;
using ZstdSharp;

namespace nvDismDecode;

// ---------------------------------------------------------------------------
// ISA description model.  Each architecture's JSON is generated from its
// corresponding NVIDIA SASS ISA tables (CLASS / ENCODING / FORMAT / OPCODES /
// enum / TABLES sections) by tools/regenerate_docs/gen_isa.py.  Keeping the description as
// data is what makes the decoder table driven instead of a pile of
// hand-written bit masks.
// ---------------------------------------------------------------------------

public sealed class IsaTable
{
    public string Architecture { get; set; } = "";
    // Toolkit/ISA profile that produced this table, e.g. "102" or "134".
    [JsonIgnore] public string Toolkit { get; set; } = "";
    public Dictionary<string, Dictionary<string, string>> Enums { get; set; } = new();
    public Dictionary<string, Dictionary<string, int>> TextToValue { get; set; } = new();
    public Dictionary<string, string> SpecialRegisters { get; set; } = new();
    public Dictionary<string, Dictionary<string, int[]>> Tables { get; set; } = new();
    public Dictionary<string, string> Parameters { get; set; } = new();
    public Dictionary<string, string> Constants { get; set; } = new();
    public Dictionary<string, string> StringMap { get; set; } = new();
    public List<string> SidlNames { get; set; } = new();
    public Dictionary<string, List<string>> TableSource { get; set; } = new();
    public IsaArchitectureInfo? ArchitectureInfo { get; set; }
    public List<string> OperationProperties { get; set; } = new();
    public List<string> OperationPredicates { get; set; } = new();
    public List<IsaFunit> Funit { get; set; } = new();
    public Dictionary<string, ulong> NopEncoding { get; set; } = new();
    public List<IsaClass> Classes { get; set; } = new();

    [JsonIgnore] public Dictionary<int, List<IsaClass>> ClassesByOpcode { get; } = new();
    [JsonIgnore] public int[][] OpcodeSegments { get; private set; } = { new[] { 91, 91 }, new[] { 11, 0 } };
    [JsonIgnore] public Dictionary<string, Dictionary<string, int>> EnumTextToValue { get; } = new();

    public void Prepare()
    {
        foreach (var (type, values) in Enums)
        {
            if (TextToValue.TryGetValue(type, out var aliases))
            {
                EnumTextToValue[type] = aliases;
                continue;
            }

            var inverse = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var (raw, text) in values)
            {
                if (int.TryParse(raw, out int value) && !inverse.ContainsKey(text))
                    inverse[text] = value;
            }
            EnumTextToValue[type] = inverse;
        }

        var opcodeField = Classes
            .SelectMany(cls => cls.Fields)
            .Where(field => field.Kind == "opcode")
            // Some SM75+ tables mix 12-bit and 13-bit opcode fields.  Use
            // the widest form so the high-opcode uniform-register classes
            // remain reachable; 12-bit instructions have bit 91 clear.
            .OrderByDescending(field => field.Segs.Sum(seg => seg[0] - seg[1] + 1))
            .FirstOrDefault();
        if (opcodeField != null)
            OpcodeSegments = opcodeField.Segs;

        foreach (var cls in Classes)
        {
            cls.Prepare();
            if (!ClassesByOpcode.TryGetValue(cls.Opcode, out var list))
                ClassesByOpcode[cls.Opcode] = list = new List<IsaClass>();
            list.Add(cls);
        }

        // Decide which fields can distinguish sibling classes.  A fixed
        // field that is identical in every sibling is an implied default
        // (notably the scheduling bits); a field that is variable or has a
        // different fixed value in a sibling must be read from the encoding.
        foreach (var group in ClassesByOpcode.Values)
        {
            var variants = new Dictionary<string, List<IsaField>>(StringComparer.Ordinal);
            foreach (var cls in group)
            {
                foreach (var field in cls.Fields)
                {
                    string key = field.Name + "|" + SegsKey(field.Segs);
                    if (!variants.TryGetValue(key, out var list))
                        variants[key] = list = new List<IsaField>();
                    list.Add(field);
                }
            }

            foreach (var list in variants.Values)
            {
                bool discriminates = list.Count > 1 && list.Any(f =>
                    f.Kind != "fixed" ||
                    f.Value != list[0].Value);
                if (!discriminates) continue;
                foreach (var field in list)
                    field.Discriminates = true;
            }
        }
    }

    private static string SegsKey(int[][] segs) =>
        string.Join(";", segs.Select(s => s[0] + ":" + s[1]));

    public static IsaTable Load(string path)
    {
        byte[] compressed = File.ReadAllBytes(path);
        using Stream stream = Decompress(path, compressed);
        var table = JsonSerializer.Deserialize(
                stream, IsaJsonContext.Default.IsaTable)
            ?? throw new InvalidDataException("The ISA JSON file is empty or invalid.");
        table.Prepare();
        return table;
    }

    private static Stream Decompress(string path, byte[] data)
    {
        if (path.EndsWith(".zst", StringComparison.OrdinalIgnoreCase))
        {
            using var decompressor = new Decompressor();
            return new MemoryStream(decompressor.Unwrap(data).ToArray());
        }
        if (path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase))
            return new GZipStream(
                new MemoryStream(data), CompressionMode.Decompress);
        return new MemoryStream(data);
    }
}

public sealed class IsaArchitectureInfo
{
    public string Name { get; set; } = "";
    public Dictionary<string, string> Settings { get; set; } = new();
    public List<string> Relocators { get; set; } = new();
}

public sealed class IsaFunit
{
    public string Name { get; set; } = "";
    public string? IssueSlots { get; set; }
    public int EncodingWidth { get; set; }
    public Dictionary<string, string> Masks { get; set; } = new();
}

public sealed class IsaClass
{
    public string Name { get; set; } = "";
    public string Mnemonic { get; set; } = "";
    public bool Alternate { get; set; }
    public int Opcode { get; set; }
    public List<string> OpcodeNames { get; set; } = new();
    public List<IsaField> Fields { get; set; } = new();
    public IsaFormat? Format { get; set; }
    public string FormatRaw { get; set; } = "";
    public string ConditionsRaw { get; set; } = "";
    public string? Remap { get; set; }
    public string? Alias { get; set; }
    public Dictionary<string, string> Properties { get; set; } = new();
    public Dictionary<string, string> Predicates { get; set; } = new();
    public List<IsaConstraint> Constraints { get; set; } = new();

    [JsonIgnore] public Dictionary<string, IsaField> FieldByName { get; } = new(StringComparer.Ordinal);
    [JsonIgnore] public Dictionary<string, IsaField> FieldByLogical { get; } = new(StringComparer.Ordinal);
    [JsonIgnore] public List<IsaField> TableFields { get; } = new();

    public void Prepare()
    {
        foreach (var field in Fields)
        {
            field.Prepare();
            if (!FieldByName.ContainsKey(field.Name))
                FieldByName[field.Name] = field;
            if (!string.IsNullOrEmpty(field.Logical) && !FieldByLogical.ContainsKey(field.Logical))
                FieldByLogical[field.Logical] = field;
            if (field.Kind == "table")
                TableFields.Add(field);
        }
    }

    public IsaField? FindField(string name)
    {
        // Legacy CUDA 10.2 tables expose constant-bank fields by internal
        // names while their FORMAT descriptors use modern logical aliases.
        if (name == "constBank") name = "Bcbank";
        else if (name == "immConstOffset") name = "Bcaddr";

        // FORMAT operands and modifiers refer to encoding logical names.  A
        // logical name can collide with a different encoding field's name
        // (SM75 HFMA2 has a field named Sb whose logical name is Sb1), so
        // prefer the logical mapping.
        if (FieldByLogical.TryGetValue(name, out var f)) return f;
        if (FieldByName.TryGetValue(name, out f)) return f;

        // Legacy pre-Volta classes often use RegA/RegB/RegC field names
        // while their FORMAT descriptors still say Ra/Rb/Rc.
        string legacy = name switch
        {
            "Ra" => "RegA",
            "Rb" => "RegB",
            "Rc" => "RegC",
            "size" => "LDInteger",
            "us32" => "IsSigned",
            "cop" => "LoadCacheOp",
            "e" => "E",
            "psl" => "PSL",
            "mrg" => "MRG",
            "xmode" => "X",
            "mufu" => "MufuOp",
            "TestCC" => "CCC_1",
            "CCTest" => "CCC_1",
            _ => "",
        };
        if (legacy.Length > 0)
        {
            if (FieldByLogical.TryGetValue(legacy, out f)) return f;
            if (FieldByName.TryGetValue(legacy, out f)) return f;
        }

        if (name == "size")
        {
            f = Fields.FirstOrDefault(field =>
                field.Logical is "LDInteger" or "CInteger" or "SQInteger" ||
                field.Name == "LSSize2");
            if (f != null) return f;
        }

        if (name is "us32" or "us32b")
        {
            f = Fields.FirstOrDefault(field =>
                field.Logical is "IsSigned" or "Integer32" or "us32");
            if (f != null) return f;
        }

        // Legacy format descriptors are often lower-cased versions of the
        // class field/logical names.
        f = Fields.FirstOrDefault(field =>
            field.Logical.Equals(name, StringComparison.OrdinalIgnoreCase) ||
            field.Name.Equals(name, StringComparison.OrdinalIgnoreCase));
        return f;
    }
}

public sealed class IsaField
{
    public string Name { get; set; } = "";
    public string Logical { get; set; } = "";
    public int Width { get; set; }
    public int[][] Segs { get; set; } = Array.Empty<int[]>();
    public string Kind { get; set; } = "var";
    public int? Value { get; set; }
    public string Expr { get; set; } = "";
    public string? Linked { get; set; }
    public string? Mod { get; set; }
    public int Scale { get; set; } = 1;

    [JsonIgnore] public string? TableName { get; private set; }
    [JsonIgnore] public string[]? TableArgs { get; private set; }

    // True when this field's encoding is used to choose between sibling
    // classes with the same opcode (for example tex_.scr vs tex_scr_.scr).
    // A fixed field that is the same in every sibling is an implied default
    // (the scheduling bits) and must not be compared against the bits.
    [JsonIgnore] public bool Discriminates { get; set; }

    public void Prepare()
    {
        if (Kind != "table") return;
        int l = Expr.IndexOf('(');
        int r = Expr.IndexOf(')', l + 1);
        if (l <= 0 || r <= l) return;
        TableName = Expr[..l].Trim();
        TableArgs = Expr[(l + 1)..r]
            .Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
    }
}

public sealed class IsaFormat
{
    public IsaGuard? Guard { get; set; }
    public List<IsaMod> Mods { get; set; } = new();
    public List<IsaOperand> Operands { get; set; } = new();

    // CUDA 10.2 control-word metadata (REQ, RD, WR, USCHED_INFO).  It is
    // kept separate from the printable operands so the decoder ignores it,
    // while the assembler/scheduler can consume it.
    public List<IsaControl> Control { get; set; } = new();
}

public sealed class IsaControl
{
    public string Raw { get; set; } = "";
    public List<IsaDescriptor> Descriptors { get; set; } = new();
}

public sealed class IsaGuard
{
    public bool NotMarker { get; set; }
    public string Type { get; set; } = "Predicate";
    public string Default { get; set; } = "PT";
    public string Field { get; set; } = "Pg";
}

public sealed class IsaMod
{
    public string Type { get; set; } = "";
    public string? Args { get; set; }
    public string Field { get; set; } = "";
}

public sealed class IsaDescriptor
{
    public string Type { get; set; } = "";
    public string? Args { get; set; }
    public bool Star { get; set; }
    public string Field { get; set; } = "";
    public string? Text { get; set; }
    public List<IsaMod> Mods { get; set; } = new();
    public List<string> Markers { get; set; } = new();
}

public sealed class IsaOperand
{
    public string Kind { get; set; } = "";
    public string? Label { get; set; }
    public string? Text { get; set; }
    public IsaDescriptor? Desc { get; set; }
    public List<IsaDescriptor>? Parts { get; set; }
    public IsaDescriptor? Bank { get; set; }
    public IsaDescriptor? Addr { get; set; }
    public IsaDescriptor? Ureg { get; set; }
    public IsaDescriptor? Offset { get; set; }
    public List<string> Markers { get; set; } = new();
}

public sealed class IsaConstraint
{
    public string Field { get; set; } = "";
    public string Op { get; set; } = "==";
    public long Value { get; set; }
    public int Score { get; set; }
}
