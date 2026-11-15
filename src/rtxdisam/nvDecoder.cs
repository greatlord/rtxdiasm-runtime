using System.Globalization;
using System.Text;

namespace nvDismDecode;

// ---------------------------------------------------------------------------
// SASS instruction representation.
//
// An SM75/SM86 instruction is 128 bits wide.  The low 64 bits (bits 0..63) are the
// classic instruction word; the high 64 bits (bits 64..127) contain the
// scheduling/control word *and* operand bits.  Several operands (notably Rc)
// live above bit 63, which is why reading Rc from bits 47:40 is wrong.
// ---------------------------------------------------------------------------

public sealed class DecodedInstruction
{
    public UInt128 Instruction { get; init; }
    public ulong Low { get; init; }
    public ulong High { get; init; }
    public ulong Address { get; init; }

    public bool IsValid { get; init; }
    public string Text { get; init; } = "";
    public string Mnemonic { get; init; } = "";
    public int Opcode { get; init; }
    public string? ClassName { get; init; }
    public IsaClass? Class { get; init; }
    public ulong? BranchTarget { get; init; }

    // Convenience fields for callers that only want the common registers.
    // These are extracted from the selected class, not from fixed bit slices.
    public int Guard { get; init; }
    public bool GuardNot { get; init; }
    public byte Rd { get; init; }
    public byte Ra { get; init; }
    public byte Rb { get; init; }
    public byte Rc { get; init; }

    public ulong InstructionWord => Low;
    public ulong ControlWord => High;

    public string GuardName => (Guard & 7) == 7 ? "PT" : $"P{Guard & 7}";
    public string RdName => RegisterName(Rd);
    public string RaName => RegisterName(Ra);
    public string RbName => RegisterName(Rb);
    public string RcName => RegisterName(Rc);

    public static string RegisterName(byte register) =>
        register == 255 ? "RZ" : $"R{register}";
}

// ---------------------------------------------------------------------------
// Table driven decoder/renderer.
//
// The algorithm is:
//   1. Extract the 13-bit opcode (bit 91 is the MSB, bits 11:0 the low bits).
//   2. Consider every CLASS with that opcode.
//   3. Reject classes whose fixed fields / single-value modifiers do not match.
//   4. Prefer non-alternate classes, then the class with the most matching
//      constraints, then the lexicographically first name.
//   5. Render the FORMAT using the field values from ENCODING.
//
// FORMAT descriptor arguments are defaults: if the encoded value equals the
// default, the operand/modifier is omitted.  Segments in BITS_* are written
// MSB-first, so get_bits shifts left for each segment.
// ---------------------------------------------------------------------------

public static class InstructionDecoder
{
    private static IsaTable? _table;
    private static readonly Dictionary<string, IsaTable> ArchitectureTables = new();

    public static IsaTable Table =>
        _table ??= LoadArchitecture("sm_86");

    public static void Load(string path) => _table = IsaTable.Load(path);

    // Embedding hosts can point the decoder at an extracted table directory.
    public static string? TableDirectoryOverride { get; set; }

    // Select the table used by subsequent Decode calls.  A fatbin can
    // contain more than one architecture, so callers switch this before each
    // cubin rather than relying on a process-wide default.
    public static void UseArchitecture(
        string architecture, string? toolkit = null) =>
        _table = LoadArchitecture(architecture, toolkit);

    public static void UseArchitecture(
        uint architecture, string? toolkit = null) =>
        UseArchitecture($"sm_{architecture}", toolkit);

    public static IsaTable LoadArchitecture(
        uint architecture, string? toolkit = null) =>
        LoadArchitecture($"sm_{architecture}", toolkit);

    public static IsaTable LoadArchitecture(
        string architecture, string? toolkit = null)
    {
        string? normalizedToolkit = NormalizeToolkit(toolkit);
        string cacheKey = architecture + "|" + (normalizedToolkit ?? "");
        if (ArchitectureTables.TryGetValue(cacheKey, out IsaTable? table))
            return table;

        string? path = FindTableFile(architecture, normalizedToolkit);
        if (path == null)
        {
            throw new NotSupportedException(
                $"No ISA table is available for {architecture}" +
                (normalizedToolkit != null
                    ? $" with toolkit {normalizedToolkit}"
                    : "") +
                $" (searched 'isa_json/isa_134_json' and " +
                $"'isa_json/isa_102_json').");
        }

        table = IsaTable.Load(path);
        // Shared tables contain a comma-separated architecture list.  Keep
        // the requested key visible to callers and tests.
        table.Architecture = architecture;
        table.Toolkit = normalizedToolkit ?? ToolkitFromPath(path);
        ArchitectureTables[cacheKey] = table;
        return table;
    }

    // Accept 10.2/102 and 13.4/134 (also cuda10.2/cuda13.4).
    public static string? NormalizeToolkit(string? toolkit)
    {
        if (string.IsNullOrWhiteSpace(toolkit))
            return null;
        string value = toolkit.Trim().ToLowerInvariant();
        if (value.StartsWith("cuda", StringComparison.Ordinal))
            value = value[4..];
        value = value.Replace(".", "").Replace("_", "").Replace("-", "");
        return value switch
        {
            "10" or "10x" or "102" => "102",
            "13" or "13x" or "134" => "134",
            _ => throw new NotSupportedException(
                $"Unknown toolkit '{toolkit}'; use 10.2 or 13.4."),
        };
    }

    public static bool TryNormalizeToolkit(
        string? toolkit, out string normalized)
    {
        try
        {
            normalized = NormalizeToolkit(toolkit) ?? "";
            return normalized.Length > 0;
        }
        catch (NotSupportedException)
        {
            normalized = "";
            return false;
        }
    }

    private static string? DefaultToolkitForArchitecture(string architecture)
    {
        string key = NormalizeArchitectureKey(architecture);
        if (!key.StartsWith("sm_", StringComparison.OrdinalIgnoreCase))
            return null;
        int end = 3;
        while (end < key.Length && char.IsDigit(key[end]))
            end++;
        if (end == 3 ||
            !int.TryParse(key[3..end], out int number))
            return null;
        return number < 75 ? "102" : "134";
    }

    private static string ToolkitFromPath(string path)
    {
        string name = Path.GetFileName(path);
        if (name.StartsWith("isa_", StringComparison.OrdinalIgnoreCase))
            name = name[4..];
        int separator = name.IndexOf('_');
        return separator > 0 && name[..separator].All(char.IsDigit)
            ? name[..separator]
            : "";
    }

    // Exposed for diagnostics and table-selection regression tests.
    public static string? ResolveTablePath(
        string architecture, string? toolkit = null) =>
        FindTableFile(architecture, NormalizeToolkit(toolkit));

    private static string NormalizeArchitectureKey(string key)
    {
        key = key.Trim().ToLowerInvariant();
        if (key.StartsWith("sm", StringComparison.Ordinal) &&
            key.Length > 2 && key[2] != '_')
            return "sm_" + key[2..];
        return key;
    }

    internal static string TableRoot =>
        TableDirectoryOverride ?? Path.Combine(AppContext.BaseDirectory, "isa_json");

    private static string? FindTableFile(
        string architecture, string? toolkit = null)
    {
        string normalized = NormalizeArchitectureKey(architecture);
        string[] tableDirectories =
        {
            Path.Combine(TableRoot, "isa_134_json"),
            Path.Combine(TableRoot, "isa_102_json"),
        };
        var matches = new List<(string Toolkit, string Path)>();
        foreach (string tableDirectory in tableDirectories
            .Distinct(StringComparer.OrdinalIgnoreCase))
        {
            if (!Directory.Exists(tableDirectory))
                continue;
            foreach (string file in Directory.EnumerateFiles(
                    tableDirectory, "isa_*")
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase))
            {
                string name = Path.GetFileName(file);
                if (name.EndsWith(".json.zst", StringComparison.OrdinalIgnoreCase))
                    name = name[..^9];
                else if (name.EndsWith(".json.gz", StringComparison.OrdinalIgnoreCase))
                    name = name[..^8];
                else if (name.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
                    name = name[..^5];
                else
                    continue;
                string keys = name.StartsWith("isa_", StringComparison.Ordinal)
                    ? name[4..]
                    : name;
                // Versioned tables use isa_<toolkit>_sm<arch>...
                string? version = null;
                int versionSeparator = keys.IndexOf('_');
                if (versionSeparator > 0 &&
                    keys[..versionSeparator].All(char.IsDigit))
                {
                    version = keys[..versionSeparator];
                    keys = keys[(versionSeparator + 1)..];
                }
                if (!keys.Split(',').Any(key =>
                        NormalizeArchitectureKey(key).Equals(
                            normalized, StringComparison.OrdinalIgnoreCase)))
                    continue;
                if (toolkit != null &&
                    !string.Equals(version, toolkit,
                        StringComparison.OrdinalIgnoreCase))
                    continue;
                matches.Add((version ?? "", file));
            }
        }

        if (matches.Count == 1)
            return matches[0].Path;
        if (matches.Count > 1)
        {
            // Keep architecture-only selection deterministic.  SM75+ defaults
            // to the current CUDA 13.4 profile; older architectures default
            // to CUDA 10.2.
            string? defaultToolkit =
                DefaultToolkitForArchitecture(architecture);
            (string Toolkit, string Path) preferred = matches
                .FirstOrDefault(match => string.Equals(
                    match.Toolkit, defaultToolkit,
                    StringComparison.OrdinalIgnoreCase));
            if (preferred.Path != null)
                return preferred.Path;

            string choices = string.Join(", ", matches
                .Select(match => match.Toolkit.Length > 0
                    ? match.Toolkit
                    : "unversioned")
                .Distinct(StringComparer.OrdinalIgnoreCase));
            throw new NotSupportedException(
                $"Multiple ISA tables match {architecture} ({choices}); " +
                "specify --toolkit 10.2 or --toolkit 13.4.");
        }

        // Compatibility fallback for a flat deployment layout.
        string flatName = $"isa_{normalized.Replace("_", "")}.json";
        string[] prefixes = toolkit != null
            ? new[] { $"isa_{toolkit}_" }
            : new[] { "isa_134_", "isa_102_", "isa_" };
        foreach (string root in new[] { AppContext.BaseDirectory, TableRoot }
            .Concat(tableDirectories))
        {
            foreach (string prefix in prefixes)
            {
                foreach (string suffix in new[] { ".zst", ".gz", "" })
                {
                    string candidate = Path.Combine(
                        root, prefix + flatName[4..] + suffix);
                    if (File.Exists(candidate)) return candidate;
                }
            }
        }
        return null;
    }

    // -- bit extraction -----------------------------------------------------

    public static ulong GetBits(ulong low, ulong high, int[][] segs)
    {
        ulong value = 0;
        foreach (int[] seg in segs)
        {
            int msb = seg[0];
            int lsb = seg[1];
            int width = msb - lsb + 1;
            ulong part;
            if (msb < 64)
            {
                part = (low >> lsb) & Mask(width);
            }
            else if (lsb >= 64)
            {
                part = (high >> (lsb - 64)) & Mask(width);
            }
            else
            {
                int lowWidth = 64 - lsb;
                part = (low >> lsb) & Mask(lowWidth);
                int highWidth = width - lowWidth;
                part |= (high & Mask(highWidth)) << lowWidth;
            }
            value = (value << width) | part;
        }
        return value;
    }

    private static ulong Mask(int width) =>
        width >= 64 ? ulong.MaxValue : (1UL << width) - 1;

    public static UInt128 GetBits(UInt128 value, int startBit, int width)
    {
        if (startBit < 0 || startBit > 127)
            throw new ArgumentOutOfRangeException(nameof(startBit));
        if (width < 1 || width > 128 - startBit)
            throw new ArgumentOutOfRangeException(nameof(width));
        if (width == 128) return value;
        return (value >> startBit) & ((UInt128.One << width) - 1);
    }

    public static long SignExtend(ulong value, int width)
    {
        if (width <= 0) return 0;
        if (width >= 64) return unchecked((long)value);
        ulong sign = 1UL << (width - 1);
        if ((value & sign) != 0)
            return unchecked((long)(value | (~0UL << width)));
        return (long)value;
    }

    // -- class selection ----------------------------------------------------

    public static IsaClass? SelectClass(ulong low, ulong high)
    {
        int opcode = (int)GetBits(low, high, Table.OpcodeSegments);
        IEnumerable<IsaClass> candidates;
        if (IsPreVoltaTable())
        {
            // Legacy tables mix 8-, 12- and 13-bit opcode fields.  Select by
            // each class's own opcode segments instead of forcing one global
            // width (which can alias MOV32I onto VSET, for example).
            candidates = Table.Classes.Where(c =>
            {
                IsaField? opcodeField = c.Fields.FirstOrDefault(
                    field => field.Kind == "opcode");
                return opcodeField != null &&
                    (int)GetBits(low, high, opcodeField.Segs) == c.Opcode;
            });
        }
        else
        {
            if (!Table.ClassesByOpcode.TryGetValue(opcode, out var list))
                return null;
            candidates = list;
        }

        IsaClass? best = null;
        int bestScore = int.MinValue;

        foreach (var candidate in candidates)
        {
            int score = 0;
            bool matches = true;

            foreach (var constraint in candidate.Constraints)
            {
                if (IsPreVoltaTable())
                {
                    IsaField? constraintField = candidate.FindField(constraint.Field);
                    if (constraintField != null &&
                        constraintField.Segs.All(seg => seg[1] >= 64))
                    {
                        // A pre-Volta instruction is 64 bits.  Sibling classes
                        // from wider encodings can synthesize scheduling
                        // fields in bits 64+; those bits are not present in a
                        // legacy word and must not reject an otherwise valid
                        // class.
                        continue;
                    }
                }
                long actual = GetConstraintValue(candidate, constraint.Field, low, high);
                if (IsPreVoltaTable())
                {
                    IsaField? fixedField = candidate.FindField(constraint.Field);
                    if (fixedField is { Kind: "fixed" } &&
                        fixedField.Segs.All(seg => seg[0] < 64))
                    {
                        // Sibling classes can be mixed with wider encodings,
                        // in which case Discriminates is not reliable for a
                        // fixed field.  Compare the legacy word bits directly.
                        actual = (long)GetBits(low, high, fixedField.Segs);
                    }
                }
                if (constraint.Op == "==" && actual != constraint.Value)
                {
                    matches = false;
                    break;
                }
                if (constraint.Op == "!=" && actual == constraint.Value)
                {
                    matches = false;
                    break;
                }
                score += constraint.Score;
            }

            if (!matches) continue;
            if (!MatchesSourceFormat(candidate, low, high)) continue;
            score += GetSpecializationScore(candidate, low, high);
            score -= 1000 * CountInvalidModifierValues(candidate, low, high);
            if (best == null || IsBetter(candidate, score, best, bestScore))
            {
                best = candidate;
                bestScore = score;
            }
        }

        if (best == null)
        {
            // Some CX constant forms carry non-default scheduling/predicate
            // bits that are not class discriminators.  Fall back to the CX
            // shape rather than emitting an unknown instruction.
            foreach (var candidate in candidates)
            {
                if (MatchesSourceFormat(candidate, low, high) &&
                    candidate.Format?.Operands.Any(
                        operand => operand.Kind == "CX") == true)
                    return candidate;
            }
        }

        return best;
    }

    public static bool IsPreVoltaTable()
    {
        string arch = Table.Architecture;
        int start = arch.IndexOf('_');
        if (start < 0)
            start = 0;
        else
            start++;
        int end = start;
        while (end < arch.Length && char.IsDigit(arch[end]))
            end++;
        return end > start &&
            int.TryParse(arch[start..end], out int number) &&
            number is >= 30 and < 70;
    }

    private static bool IsBetter(IsaClass candidate, int score, IsaClass best, int bestScore)
    {
        if (candidate.Alternate != best.Alternate &&
            candidate.Name != "CCTLT" && best.Name != "CCTLT")
            return !candidate.Alternate;
        if (score != bestScore)
            return score > bestScore;
        // CUDA 10.2 keeps several source-level text aliases in the legacy
        // tables.  They share an encoding with the hardware TEX/TLD form, so
        // prefer the real instruction when both classes are otherwise equal.
        bool candidateAlias = IsSourceAlias(candidate);
        bool bestAlias = IsSourceAlias(best);
        if (candidateAlias != bestAlias)
            return !candidateAlias;
        return string.CompareOrdinal(candidate.Name, best.Name) < 0;
    }

    private static bool IsSourceAlias(IsaClass cls) =>
        cls.Name is "LDG CT" or "LDG CG CS";

    private static bool IsCctltPattern(
        IsaClass cls, ulong low, ulong high)
    {
        long tid = GetFieldValue(cls, "TidB", low, high);
        long lod = GetFieldValue(cls, "LOD", low, high);
        long param = GetFieldValue(cls, "ParamA", low, high);
        long dest = GetFieldValue(cls, "Dest", low, high);
        long regA = GetFieldValue(cls, "RegA", low, high);
        long regB = GetFieldValue(cls, "RegB", low, high);
        long wmsk = GetFieldValue(cls, "Wmsk", low, high);
        return tid == 7 && lod == 1 && param == 0 &&
            dest == 255 && regA == 255 && regB == 255 && wmsk == 0;
    }

    // A fixed field that is identical in every sibling class is an implied
    // default; the bits in the instruction may be unrelated scheduling bits.
    // Fields that actually choose between siblings (scr, b, e_desc, ...)
    // must be compared against the encoding.
    private static long GetConstraintValue(
        IsaClass cls, string name, ulong low, ulong high)
    {
        IsaField? field = cls.FindField(name);
        if (field == null)
            return 0;
        if (field.Kind == "fixed")
            return field.Discriminates ? (long)GetBits(low, high, field.Segs) : field.Value ?? 0;
        return GetFieldValue(cls, name, low, high);
    }

    // Some ISA modifiers (XONLY, EXONLY, HIONLY, WIDEONLY) are not ordinary
    // constraints, but they distinguish otherwise similar classes.  Give a
    // class that actually has the requested bit a strong preference.
    private static int GetSpecializationScore(
        IsaClass cls, ulong low, ulong high)
    {
        int score = 0;
        if (cls.Format == null)
            return score;

        if (cls.Name.Contains("_pseudo", StringComparison.Ordinal))
            score += 100;

        // CCTLT is an ALTERNATE source class, but its constrained texture
        // encoding is what CUDA 10.2 nvdisasm prints for this bit pattern.
        if (cls.Name == "CCTLT" && IsCctltPattern(cls, low, high))
            score += 2000;

        foreach (IsaMod mod in cls.Format.Mods)
        {
            switch (mod.Type)
            {
                case "XONLY":
                case "EXONLY":
                case "HIONLY":
                case "HIONLY_lea":
                    if (GetFieldValue(cls, mod.Field, low, high) == 1)
                        score += 1000;
                    break;
                case "WIDEONLY":
                    if (cls.Name.Contains("_wide", StringComparison.Ordinal))
                        score += 1000;
                    break;
            }
        }

        return score;
    }

    // Sibling classes frequently share an opcode and differ only in the
    // legal range of a modifier enum (for example F2I's destination format).
    // Prefer the class whose format can actually represent the value.
    private static int CountInvalidModifierValues(
        IsaClass cls, ulong low, ulong high)
    {
        if (cls.Format == null)
            return 0;

        int count = 0;
        foreach (IsaMod mod in cls.Format.Mods)
        {
            if (mod.Type is "PSEUDO_OPCODE" or "REUSE" or "RelOpt") continue;
            if (mod.Args != null) continue;
            if (mod.Type.EndsWith("ONLY", StringComparison.Ordinal)) continue;
            if (!Table.Enums.TryGetValue(mod.Type, out var values)) continue;

            long value = GetFieldValue(cls, mod.Field, low, high);
            if (!values.ContainsKey(value.ToString(CultureInfo.InvariantCulture)))
                count++;
        }
        return count;
    }

    // Classes such as I2F have the same opcode and no fixed-field
    // constraints; their source-format modifier enum is the discriminator.
    private static bool MatchesSourceFormat(
        IsaClass cls, ulong low, ulong high)
    {
        if (cls.Format == null)
            return true;

        foreach (IsaMod mod in cls.Format.Mods)
        {
            if (mod.Field != "srcfmt")
                continue;
            // F32ONLY/U32ONLY/... are class-selection pseudo enums, not the
            // encoded source-format field.
            if (mod.Type.Contains("ONLY", StringComparison.Ordinal))
                continue;
            if (!Table.Enums.TryGetValue(mod.Type, out var values))
                continue;

            long value = GetFieldValue(cls, mod.Field, low, high);
            if (!values.ContainsKey(value.ToString(CultureInfo.InvariantCulture)))
                return false;
        }

        return true;
    }

    // -- field/table lookup -------------------------------------------------

    public static long GetFieldValue(IsaClass cls, string name, ulong low, ulong high)
    {
        IsaField? field = cls.FindField(name);

        if (field != null && field.Kind == "fixed")
            return field.Value ?? 0;

        if (field != null && field.Kind != "table")
        {
            long raw = (long)GetBits(low, high, field.Segs);
            // Legacy tables encode the Integer32 selector as an IsSigned
            // boolean, while the FORMAT enum uses 4=U32 and 5=S32.
            string logical = field.Logical ?? "";
            if (IsPreVoltaTable() &&
                (name is "us32" or "us32b" ||
                 logical is "us32" or "us32b" or "Integer32") &&
                raw is 0 or 1)
                return raw == 0 ? 4 : 5;
            if (IsPreVoltaTable() &&
                (((name == "srcfmt" || logical == "srcfmt") &&
                    cls.Mnemonic == "I2F") ||
                 ((name == "dstfmt" || logical == "dstfmt") &&
                    cls.Mnemonic == "F2I")))
            {
                // The legacy IntSize table differs by architecture: SM30 is
                // identity while SM35/SM37 permute the signed/size bits.
                if (Table.Tables.TryGetValue("IntSize", out var intSize) &&
                    intSize.TryGetValue(
                        raw.ToString(CultureInfo.InvariantCulture),
                        out int[]? mapped) &&
                    mapped.Length > 0)
                    return mapped[0];
                return raw;
            }
            return raw;
        }

        foreach (var tableField in cls.TableFields)
        {
            if (tableField.TableName == null || tableField.TableArgs == null)
                continue;
            int index = Array.IndexOf(tableField.TableArgs, name);
            if (index < 0) continue;

            long raw = (long)GetBits(low, high, tableField.Segs);
            if (Table.Tables.TryGetValue(tableField.TableName, out var map) &&
                map.TryGetValue(raw.ToString(CultureInfo.InvariantCulture), out int[]? values) &&
                index < values.Length)
            {
                return values[index];
            }
        }

        if (field != null)
            return (long)GetBits(low, high, field.Segs);

        return 0;
    }

    public static long? GetTableFlag(IsaClass cls, string? name, string suffix, ulong low, ulong high)
    {
        if (name == null) return null;
        string key = name + "@" + suffix;

        foreach (var tableField in cls.TableFields)
        {
            if (tableField.TableName == null || tableField.TableArgs == null)
                continue;
            int index = Array.IndexOf(tableField.TableArgs, key);
            if (index < 0) continue;

            long raw = (long)GetBits(low, high, tableField.Segs);
            if (Table.Tables.TryGetValue(tableField.TableName, out var map) &&
                map.TryGetValue(raw.ToString(CultureInfo.InvariantCulture), out int[]? values) &&
                index < values.Length)
            {
                return values[index];
            }
        }

        return null;
    }

    // Table-valued selector fields that are not declared as kind="table".
    // The legacy FORMAT grammar spells packed selectors as a call whose
    // arguments name the flags they carry, for example
    //   nAB = PSignFFMA(Ra@negate,Rb@negate)
    // where the table maps the encoded value to one flag per argument.
    // IsaField.Prepare only splits that form for kind="table", so these
    // fields never reach GetTableFlag and their flags are silently lost.
    private static long? GetTableExpressionFlag(
        IsaClass cls, string? name, string suffix, ulong low, ulong high)
    {
        if (name == null) return null;
        string key = name + "@" + suffix;

        foreach (var field in cls.Fields)
        {
            if (field.Kind == "table") continue;
            string expr = field.Expr;
            int open = expr.IndexOf('(');
            int close = expr.IndexOf(')', open + 1);
            if (open <= 0 || close <= open) continue;

            string[] args = expr[(open + 1)..close]
                .Split(',', StringSplitOptions.TrimEntries |
                    StringSplitOptions.RemoveEmptyEntries);
            int index = Array.IndexOf(args, key);
            if (index < 0) continue;

            string tableName = expr[..open].Trim();
            long raw = (long)GetBits(low, high, field.Segs);
            if (Table.Tables.TryGetValue(tableName, out var map) &&
                map.TryGetValue(raw.ToString(CultureInfo.InvariantCulture), out int[]? values) &&
                index < values.Length)
            {
                return values[index];
            }
        }

        return null;
    }

    private static List<int[]> UsedSegments(
        IsaClass cls, bool includeFormatMods)
    {
        var usedSegments = new List<int[]>();
        void AddUsed(string? name)
        {
            if (string.IsNullOrEmpty(name))
                return;
            IsaField? used = cls.FindField(name);
            if (used != null)
                usedSegments.AddRange(used.Segs);
        }
        if (cls.Format != null)
        {
            foreach (IsaOperand operand in cls.Format.Operands)
            {
                if (operand.Desc != null)
                    AddUsed(operand.Desc.Field);
                foreach (IsaDescriptor part in operand.Parts ??
                    new List<IsaDescriptor>())
                    AddUsed(part.Field);
                AddUsed(operand.Bank?.Field);
                AddUsed(operand.Addr?.Field);
                AddUsed(operand.Ureg?.Field);
                AddUsed(operand.Offset?.Field);
            }
            if (includeFormatMods)
            {
                foreach (IsaMod formatMod in cls.Format.Mods)
                    AddUsed(formatMod.Field);
            }
        }
        return usedSegments;
    }

    private static bool OverlapsUsed(IsaClass cls, IsaField field)
    {
        List<int[]> usedSegments = UsedSegments(cls, includeFormatMods: false);
        return field.Segs.Any(segment => usedSegments.Any(used =>
            !(segment[0] < used[1] || used[0] < segment[1])));
    }

    public static IsaField? FindLinked(IsaClass cls, string? baseField, string mod)
    {
        // Derived marker fields are authoritative.  Some legacy classes use
        // ordinary var fields instead, but those can overlap a register or
        // modifier field that is actually in use (FFMA's nA3 overlaps Rc).
        // Reject such overlapping aliases.
        List<int[]> usedSegments = UsedSegments(cls, includeFormatMods: true);
        bool Overlaps(IsaField field) =>
            field.Segs.Any(segment => usedSegments.Any(used =>
                !(segment[0] < used[1] || used[0] < segment[1])));
        return cls.Fields.FirstOrDefault(field =>
            field.Linked == baseField && field.Mod == mod &&
            (field.Kind == "derived" ||
             (field.Kind == "var" && !Overlaps(field))));
    }

    // -- enum helpers -------------------------------------------------------

    public static string? EnumText(string type, long value)
    {
        if (Table.Enums.TryGetValue(type, out var values) &&
            values.TryGetValue(value.ToString(CultureInfo.InvariantCulture), out var text))
            return text;
        return null;
    }

    public static long? DefaultValue(string type, string? args)
    {
        if (args == null) return null;
        string text = args.Trim();
        if (text.Length >= 2 && text[0] == '"' && text[^1] == '"')
            text = text[1..^1];
        int slash = text.IndexOf('/');
        if (slash >= 0)
            text = text[(slash + 1)..].Trim();

        if (Table.EnumTextToValue.TryGetValue(type, out var inverse) &&
            inverse.TryGetValue(text, out int value))
            return value;

        if (long.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out long numeric))
            return numeric;

        if (TryParseInt(text, out numeric))
            return numeric;

        return null;
    }

    private static (string? Width, string? Default) ParseArgs(string? args)
    {
        if (args == null) return (null, null);
        string text = args.Trim();
        int slash = text.IndexOf('/');
        if (slash < 0) return (text, null);
        return (text[..slash].Trim(), text[(slash + 1)..].Trim());
    }

    private static int ParseWidth(string? args)
    {
        var (width, _) = ParseArgs(args);
        if (width != null && int.TryParse(width.TrimEnd('*'), out int value))
            return value;
        return 0;
    }

    private static bool TryParseInt(string text, out long value)
    {
        text = text.Trim();
        if (text.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
            return long.TryParse(
                text[2..], NumberStyles.HexNumber, CultureInfo.InvariantCulture, out value);
        return long.TryParse(
            text, NumberStyles.Integer, CultureInfo.InvariantCulture, out value);
    }

    // -- rendering ----------------------------------------------------------

    private static IsaField? FindModField(IsaClass cls, IsaMod mod)
    {
        IsaField? field = cls.FindField(mod.Field);
        if (!IsPreVoltaTable())
            return field;

        // Legacy FORMAT selectors sometimes use a different name from the
        // encoding field.  Prefer the logical name, then the selector type.
        if (field == null)
        {
            IsaField? byLogical = cls.Fields.FirstOrDefault(candidate =>
                string.Equals(candidate.Logical, mod.Field,
                    StringComparison.OrdinalIgnoreCase));
            if (byLogical != null)
                return byLogical;
        }
        IsaField? byType = cls.Fields.FirstOrDefault(candidate =>
            string.Equals(candidate.Logical, mod.Type,
                StringComparison.OrdinalIgnoreCase));
        if (byType != null &&
            (field == null || byType.Width < field.Width))
            return byType;
        if (field != null)
            return field;

        // Only a few legacy texture FORMAT descriptors use short names that
        // do not match the encoding's logical field names.
        string? logical = mod.Field switch
        {
            "cop" => "LDGcop2",
            "e" => "LDG_E",
            "phase" => "TPhase",
            "sync" => "S",
            "clamp" => "CL",
            "hib" => "HILO",
            _ => null,
        };
        if (logical == null)
            return null;
        return cls.Fields.FirstOrDefault(candidate =>
            string.Equals(candidate.Logical, logical,
                StringComparison.OrdinalIgnoreCase) ||
            (logical == "LDGcop2" &&
             candidate.Logical?.StartsWith("LDGcop",
                 StringComparison.OrdinalIgnoreCase) == true));
    }

    private static string RenderModifier(IsaMod mod, IsaClass cls, ulong low, ulong high)
    {
        string type = mod.Type;
        if (type is "PSEUDO_OPCODE" or "REUSE")
            return "";
        // CCTLT only has the IVALL form; the flag is implied by its fixed
        // texture-ID encoding rather than a separate field.
        if (type == "OnlyIvall")
            return ".IVALL";
        // Kepler ISETP shares its xmode bit with an unrelated field in the
        // legacy table; nvdisasm does not print .X for ISETP.
        if (type == "X" && cls.Mnemonic == "ISETP")
            return "";
        if (type == "RelOpt")
            return ".REL";

        // Class-selection / pseudo modifiers.  XONLY, EXONLY, ... are real
        // printed modifiers in nvdisasm (for example HADD2.F32 and TEX.SCR)
        // while LOOnly/SIGNONLY only pick a class.
        switch (type)
        {
            case "LOOnly":
            case "SIGNONLY":
                return "";
        }

        if (type is "XONLY" or "EXONLY" or "HIONLY" or "HIONLY_lea" or
            "HIONLY_imad" or "ONLY64" or
            "WIDEONLY" or "SX32ONLY" or "SCRONLY" or "BONLY" or
            "F32ONLY" or "S32ONLY" or "U32ONLY" or
            "F16ONLY_f2fp" or "F32ONLY_f2fp" or
            "S32ONLY_i2i" or "U32ONLY_i2f")
        {
            if (!Table.Enums.TryGetValue(type, out var pseudoValues))
                return "";

            // These modifiers are implied by the class.  Their field is
            // often not present in the encoding at all.
            IsaField? pseudoField = cls.FindField(mod.Field);
            long? pseudoDefault = DefaultValue(type, mod.Args);
            if (pseudoField == null)
            {
                // An explicit default means the modifier is implied and
                // must not be printed (for example F2FP's /F32ONLY("F32")).
                if (pseudoDefault.HasValue)
                    return "";
                return pseudoValues.Count == 1
                    ? "." + pseudoValues.Values.First()
                    : "";
            }

            long pseudoValue = GetFieldValue(cls, mod.Field, low, high);
            // nvdisasm omits the implied .F32 on FRND, but keeps explicit
            // format modifiers such as F2FP.F16.F32.  Scope the default
            // suppression to FRND so unrelated pseudo modifiers are unchanged.
            if (cls.Mnemonic == "FRND" &&
                pseudoDefault.HasValue && pseudoValue == pseudoDefault.Value)
                return "";
            if (pseudoValues.TryGetValue(
                    pseudoValue.ToString(CultureInfo.InvariantCulture), out var pseudoText))
                return "." + pseudoText;
            return "";
        }

        if (type == "LDSInteger")
        {
            long ldsSize = GetFieldValue(cls, mod.Field, low, high);
            string? ldsText = LegacyLdsSizeText(ldsSize);
            return ldsText == null ? "" : "." + ldsText;
        }

        if (type == "SQInteger")
        {
            long sqSize = GetFieldValue(cls, mod.Field, low, high);
            if (Table.Architecture == "sm_30")
            {
                // SM30 uses the compact legacy SQInteger codes.
                if (sqSize is 4 or 15)
                    return "";
                if (sqSize == 5)
                    return ".U64";
                if (sqSize == 6)
                    return ".U128";
            }
            else
            {
                string? sqText = EnumText("SQInteger", sqSize);
                if (sqText is null or "32" or "U32")
                    return "";
                return "." + sqText;
            }
        }

        if (!Table.Enums.TryGetValue(type, out var values))
            return "";

        // Legacy CUDA 10.2 size enums are encoded as compact bit patterns;
        // the converted JSON values are not the numeric nvdisasm sizes.
        if (type is "LDInteger" or "CInteger")
        {
            IsaField? sizeField = FindModField(cls, mod);
            if (sizeField == null)
                return "";
            long rawSize = GetFieldValue(cls, mod.Field, low, high);
            string? sizeText = rawSize switch
            {
                0 => "U8",
                1 => "S8",
                2 => "U16",
                3 => "S16",
                4 => "32",
                5 => "64",
                6 => "128",
                7 => "U.128",
                _ => null,
            };
            if (sizeText == null || sizeText == "32")
                return "";
            return "." + sizeText;
        }

        // Compare selectors (FSETP/FSET/DSETP/HSETP2) live in a 4-bit
        // field that indexes the tail of the 6-bit "Test" space: selector
        // 1..15 is Test[selector + 31] and 0 is the always-false F.  The
        // 4-bit value never matches the enum on its own, which used to
        // leak "???1" and made the lifter treat every float compare as EQ.
        // Every entry was verified by patching the field of a real
        // instruction and reading the result back from nvdisasm.
        if (type == "Test")
        {
            IsaField? compareField = FindModField(cls, mod);
            if (compareField != null && compareField.Width == 4)
            {
                long selector = GetFieldValue(cls, mod.Field, low, high);
                if (selector == 0)
                    return ".F";
                string? compareText = EnumText("Test", selector + 31);
                return compareText != null
                    ? "." + compareText
                    : ".???" + selector.ToString(CultureInfo.InvariantCulture);
            }
        }

        IsaField? modField = FindModField(cls, mod);
        if (modField != null && IsPreVoltaTable() &&
            OverlapsUsed(cls, modField))
            return "";
        long value = modField != null
            ? GetFieldValue(cls, modField.Name, low, high)
            : GetFieldValue(cls, mod.Field, low, high);
        bool forceTypeModifier =
            IsPreVoltaTable() &&
            cls.Mnemonic is "I2F" or "F2I" &&
            type is "Float" or "Float32" or "FloatNo64" or
                "Integer" or "IntegerNo16" or "IntegerNo16No64" or
                "IntegerInv8No64";
        if (type == "VoteOp")
            forceTypeModifier = true;
        long? defaultValue = forceTypeModifier
            ? null
            : DefaultValue(type, mod.Args);
        if (defaultValue.HasValue && value == defaultValue.Value)
            return "";

        if (!values.TryGetValue(value.ToString(CultureInfo.InvariantCulture), out var text))
        {
            // A zero value is commonly the implicit/default selector in the
            // legacy tables; do not leak a synthetic ???0 modifier for it.
            if (value == 0)
                return "";
            return ".???" + value.ToString(CultureInfo.InvariantCulture);
        }

        return "." + text;
    }

    private static string RenderAttachedMods(
        IsaDescriptor descriptor, IsaClass cls, ulong low, ulong high, string text)
    {
        foreach (var mod in descriptor.Mods)
        {
            string type = mod.Type;
            if (type is "RelOpt" or "PSEUDO_OPCODE" or "LOOnly" or "SIGNONLY")
                continue;

            if (type == "REUSE")
            {
                if (GetFieldValue(cls, mod.Field, low, high) != 0)
                    text += ".reuse";
                continue;
            }

            if (type == "ONLY64")
            {
                text += ".64";
                continue;
            }

            if (type == "EXP_DESC")
                continue;

            if (!Table.Enums.TryGetValue(type, out var values))
                continue;

            long value = GetFieldValue(cls, mod.Field, low, high);
            if (type == "B1B0" && mod.Args == "B0" && value == 0)
                continue;
            long? defaultValue = DefaultValue(type, mod.Args);
            if (defaultValue.HasValue && value == defaultValue.Value)
                continue;

            if (values.TryGetValue(value.ToString(CultureInfo.InvariantCulture), out var enumText))
                text += "." + enumText;
            else
                text += ".???" + value.ToString(CultureInfo.InvariantCulture);
        }

        return text;
    }

    private static string RenderMarkerPrefix(
        List<string> markers, IsaClass cls, ulong low, ulong high, string? baseField,
        out string suffix)
    {
        bool Flag(string suffix)
        {
            IsaField? linked = FindLinked(cls, baseField, suffix);
            if (linked != null)
                return GetBits(low, high, linked.Segs) != 0;

            // IADD/ISUB-style classes use PSign instead of per-operand
            // negate fields.  Bit 2 selects -Ra and bit 0 selects -Rb or the
            // constant source.
            if (suffix == "negate" &&
                baseField is "Ra" or "Rb" or "srcConst")
            {
                IsaField? signs = cls.Fields.FirstOrDefault(field => field.Name == "PSign" && field.Width >= 2);
                if (signs != null)
                {
                    long value = GetFieldValue(cls, "PSign", low, high);
                    long mask = baseField == "Ra" ? 2 : 1;
                    if ((value & mask) != 0)
                        return true;
                }
            }

            long? tableValue = GetTableFlag(cls, baseField, suffix, low, high);
            if (tableValue.HasValue)
                return tableValue.Value != 0;

            // PSign-style selectors pack several markers into one field and
            // name the ones they carry, e.g. FFMA's nAB is
            // PSignFFMA(Ra@negate,Rb@negate).  They are plain var fields,
            // so IsaField.Prepare leaves TableName/TableArgs empty and
            // GetTableFlag never sees them.
            long? expressionValue = GetTableExpressionFlag(
                cls, baseField, suffix, low, high);
            return expressionValue.HasValue && expressionValue.Value != 0;
        }

        var builder = new StringBuilder();
        var suffixBuilder = new StringBuilder();
        foreach (string marker in markers)
        {
            if (marker == "!" && Flag("not")) builder.Append('!');
            else if (marker == "-" && Flag("negate")) builder.Append('-');
            else if (marker == "~" && Flag("invert")) builder.Append('~');
            else if (marker == "||" && Flag("absolute"))
            {
                builder.Append('|');
                suffixBuilder.Append('|');
            }
        }
        suffix = suffixBuilder.ToString();
        return builder.ToString();
    }

    private static string NormalizeLegacyRegister(string text)
    {
        if (text.Length > 2 && text[0] == 'R' && text[1] == '0' &&
            text.Skip(2).All(char.IsDigit))
            return "R" + text[2..].TrimStart('0').PadLeft(1, '0');
        return text;
    }

    private static string LegacyRegisterText(
        string type, long value, string? text)
    {
        if (type is "Register" or "NonZeroRegister" or "RegisterFAU" or
            "ZeroRegister")
        {
            if (value == 255)
                return "RZ";
            if (string.IsNullOrEmpty(text) ||
                text.Contains("..", StringComparison.Ordinal) ||
                text.StartsWith("NonZeroRegister", StringComparison.Ordinal) ||
                text.StartsWith("Register", StringComparison.Ordinal))
            {
                return "R" + value.ToString(CultureInfo.InvariantCulture);
            }
            return NormalizeLegacyRegister(text);
        }
        return text ?? "";
    }

    // The CUDA 10.2 LDSInteger enum uses a compact hardware encoding that
    // differs from its declared source enumeration.  Values were verified
    // against nvdisasm --binary SM50.
    private static string? LegacyLdsSizeText(long value) => value switch
    {
        0 => "U8",
        1 => "U.U8",
        2 => "S8",
        3 => "U.S8",
        4 => "U16",
        5 => "U.U16",
        6 => "S16",
        7 => "U.S16",
        8 => null,
        9 => "U.32",
        10 => "64",
        11 => "U.64",
        12 => "128",
        13 => "U.128",
        14 => "INVALIDSIZE7",
        15 => "U.INVALIDSIZE7",
        _ => LegacyLdsSizeText(value & 15),
    };

    private static string FormatImm(long value, bool signed)
    {
        if (signed && value < 0)
            return "-0x" + (-value).ToString("x", CultureInfo.InvariantCulture);
        return "0x" + value.ToString("x", CultureInfo.InvariantCulture);
    }

    private static string FormatFloat(string type, ulong bits)
    {
        return type switch
        {
            "F16Imm" => FormatHalf((ushort)bits),
            "F32Imm" => FormatSingle((uint)bits),
            "F64Imm" => FormatDouble(bits),
            _ => "0x" + bits.ToString("x", CultureInfo.InvariantCulture),
        };
    }

    private static string FormatHalf(ushort bits)
    {
        Half value = BitConverter.UInt16BitsToHalf(bits);
        if (Half.IsNaN(value))
            return FormatNan((bits & 0x8000) != 0, (bits & 0x0200) != 0);
        if (Half.IsPositiveInfinity(value))
            return "+INF ";
        if (Half.IsNegativeInfinity(value))
            return "-INF ";
        if (bits == 0x8000)
            return "-0.0";
        return FormatDecimal((double)value);
    }

    private static string FormatBf16(ushort bits)
    {
        // BF16 is the upper half of an IEEE-754 binary32 value.
        uint floatBits = (uint)bits << 16;
        float value = BitConverter.Int32BitsToSingle(unchecked((int)floatBits));
        if (float.IsNaN(value))
            return FormatNan((bits & 0x8000) != 0, (bits & 0x0040) != 0);
        if (float.IsPositiveInfinity(value))
            return "+INF ";
        if (float.IsNegativeInfinity(value))
            return "-INF ";
        if (bits == 0x8000)
            return "-0.0";
        return FormatDecimal(value);
    }

    private static bool IsBf16Immediate(IsaClass cls, ulong low, ulong high)
    {
        if (cls.Format == null)
            return false;

        foreach (IsaMod mod in cls.Format.Mods)
        {
            if (mod.Type is not ("OFMT" or "DSTFMT" or
                "DSTFMT_SRCFMT_F16F32_E8M7F32_E6M9F32_BF16F32" or
                "__HIR0X1A1" or "SRCFMT_hmma"))
                continue;

            long value = GetFieldValue(cls, mod.Field, low, high);
            string? text = EnumText(mod.Type, value);
            if (text != null && text.Contains("BF16", StringComparison.Ordinal))
                return true;
        }

        return false;
    }

    private static string FormatSingle(uint bits)
    {
        float value = BitConverter.Int32BitsToSingle(unchecked((int)bits));
        if (float.IsNaN(value))
            return FormatNan((bits & 0x80000000u) != 0, (bits & 0x00400000u) != 0);
        if (float.IsPositiveInfinity(value))
            return "+INF ";
        if (float.IsNegativeInfinity(value))
            return "-INF ";
        if (bits == 0x80000000u)
            return "-0.0";
        return FormatDecimal(value);
    }

    private static string FormatDouble(ulong bits)
    {
        double value = BitConverter.Int64BitsToDouble(unchecked((long)bits));
        if (double.IsNaN(value))
            return FormatNan(
                (bits & 0x8000000000000000UL) != 0,
                (bits & 0x0008000000000000UL) != 0);
        if (double.IsPositiveInfinity(value))
            return "+INF ";
        if (double.IsNegativeInfinity(value))
            return "-INF ";
        if (bits == 0x8000000000000000UL)
            return "-0.0";
        return FormatDecimal(value);
    }

    // nvdisasm prints with 20 significant digits.  Values that would be
    // rendered as a large integer switch to scientific notation with 20
    // fractional digits (for example 1.07374182400000000000e+09).
    private static string FormatDecimal(double value)
    {
        double magnitude = Math.Abs(value);
        string text = magnitude >= 1e9
            ? value.ToString("E20", CultureInfo.InvariantCulture)
            : value.ToString("G20", CultureInfo.InvariantCulture);
        text = LowerExponent(text);
        // .NET writes e+009; nvdisasm uses a two-digit exponent.
        int e = text.IndexOf('e');
        if (e > 0)
        {
            string mantissa = text[..e];
            string exponent = text[(e + 1)..];
            char sign = exponent[0];
            string digits = exponent[1..].TrimStart('0');
            if (digits.Length == 0) digits = "0";
            if (digits.Length < 2) digits = digits.PadLeft(2, '0');
            text = mantissa + "e" + sign + digits;
        }
        return text;
    }

    private static string FormatNan(bool negative, bool quiet) =>
        (negative ? "-" : "+") + (quiet ? "QNAN" : "SNAN") + " ";

    private static string LowerExponent(string text) =>
        text.Replace('E', 'e');

    private static string? RenderFieldDescriptor(
        IsaDescriptor descriptor, IsaClass cls, ulong low, ulong high, ulong address)
    {
        string type = descriptor.Type;
        if (type is "RegisterFAU" or "NonZeroRegister" or
            "NonZeroRegisterFAU")
            type = "NonZeroRegister";
        if (type == "Register" &&
            !Table.Enums.ContainsKey("Register") &&
            Table.Enums.ContainsKey("NonZeroRegister"))
            type = "NonZeroRegister";
        string fieldName = descriptor.Field;
        IsaField? field = cls.FindField(fieldName);
        long value = GetFieldValue(cls, fieldName, low, high);

        if (type == "CC" && cls.Mnemonic == "EXIT")
            return null;

        if (type == "RSImm")
        {
            int width = field?.Width ?? ParseWidth(descriptor.Args);
            int scale = field?.Scale ?? 4;
            long immediate = SignExtend((ulong)value, width);
            ulong target = unchecked(address + BranchBase() + (ulong)(immediate * scale));
            return "0x" + target.ToString("x", CultureInfo.InvariantCulture);
        }

        if (type == "BITSET")
        {
            int width = field?.Width ?? ParseWidth(descriptor.Args);
            if (width <= 0 || width > 31)
                width = 6;
            if (value == 0)
                return null;
            var members = new List<int>();
            for (int bit = width - 1; bit >= 0; bit--)
            {
                if ((value & (1L << bit)) != 0)
                    members.Add(bit);
            }
            return "{" + string.Join(",", members) + "}";
        }

        if (type == "NonZeroRegister" && value == 255)
            return RenderAttachedMods(descriptor, cls, low, high, "RZ");

        if (type == "SpecialRegister" && IsPreVoltaTable())
        {
            string special = Table.SpecialRegisters.TryGetValue(
                value.ToString(CultureInfo.InvariantCulture), out var specialText)
                ? specialText
                : $"SR{value}";
            return RenderAttachedMods(descriptor, cls, low, high, special);
        }

        // The default TRUE condition code on control operations is omitted
        // by nvdisasm.
        if ((cls.Mnemonic is "BRK" or "CONT" or "KIL" or "LONGJMP" or
             "NOP") && type == "CC" && value == 15)
            return null;

        // CUDA 10.2 prints the uniform predicate TRUE encoding as UP7;
        // CUDA 13.4 canonicalizes it as UPT.
        if (type == "UniformPredicate" && Table.Toolkit == "102")
            return RenderAttachedMods(
                descriptor, cls, low, high,
                $"UP{value.ToString(CultureInfo.InvariantCulture)}");

        if (Table.Enums.TryGetValue(type, out var values))
        {
            long? defaultValue = DefaultValue(type, descriptor.Args);
            if (defaultValue.HasValue && value == defaultValue.Value)
                return null;

            string text = values.TryGetValue(value.ToString(CultureInfo.InvariantCulture), out var enumText)
                ? enumText
                : $"{type}{value}";
            text = LegacyRegisterText(type, value, text);
            return RenderAttachedMods(descriptor, cls, low, high, text);
        }

        if (type == "SpecialRegister")
        {
            string text = Table.SpecialRegisters.TryGetValue(
                value.ToString(CultureInfo.InvariantCulture), out var special)
                ? special
                : $"SR{value}";
            return RenderAttachedMods(descriptor, cls, low, high, text);
        }

        if (type is "UImm" or "SImm" or "CInteger" or "QInteger" or "AInteger" or
            "F16Imm" or "F32Imm" or "F64Imm")
        {
            int width = field?.Width ?? ParseWidth(descriptor.Args);
            long immediate = type == "SImm"
                ? SignExtend((ulong)value, width)
                : value;
            // Absolute call targets and other scaled immediates store a
            // byte-address field as a scaled word offset.  nvdisasm prints
            // the byte address, so apply the field scale during rendering.
            if (field != null && field.Scale > 1 &&
                field.Expr.Contains("SCALE", StringComparison.OrdinalIgnoreCase) &&
                !field.Expr.Contains("MULTIPLY", StringComparison.OrdinalIgnoreCase) &&
                type is not ("F16Imm" or "F32Imm" or "F64Imm"))
                immediate *= field.Scale;

            var (_, defaultText) = ParseArgs(descriptor.Args);
            if (defaultText != null &&
                TryParseInt(defaultText.TrimEnd('*'), out long defaultImmediate) &&
                immediate == defaultImmediate)
                return null;

            if (type is "F16Imm" or "F32Imm" or "F64Imm")
            {
                if (width == 16)
                {
                    if (IsBf16Immediate(cls, low, high))
                        return FormatBf16((ushort)value);
                    return FormatHalf((ushort)value);
                }

                if (type == "F32Imm")
                {
                    uint floatBits = (uint)value;
                    if (width < 32)
                        floatBits <<= 32 - width;
                    string floatText = FormatSingle(floatBits);
                    if (cls.FindField("UnaryNeg") != null &&
                        GetFieldValue(cls, "UnaryNeg", low, high) != 0)
                        floatText += ".NEG";
                    if (cls.FindField("UnaryAbs") != null &&
                        GetFieldValue(cls, "UnaryAbs", low, high) != 0)
                        floatText += ".ABS";
                    return floatText;
                }

                if (type == "F64Imm")
                {
                    // MUFU encodes convertFloatType(..., F64Imm, F32Imm)
                    // in a 32-bit field.  RCP64H/RSQ64H use the upper half
                    // of a double; every other operation uses a float.
                    if (width == 32 &&
                        field?.Expr.Contains("convertFloatType", StringComparison.Ordinal) == true)
                    {
                        long mufuOperation = GetFieldValue(cls, "mufuop", low, high);
                        if (mufuOperation is not (6 or 7))
                            return FormatSingle((uint)value);
                    }

                    ulong floatBits = (ulong)value;
                    if (width < 64)
                        floatBits <<= 64 - width;
                    return FormatDouble(floatBits);
                }

                return FormatFloat(type, (ulong)value);
            }

            return FormatImm(immediate, type == "SImm");
        }

        return null;
    }

    private static string RenderAddressParts(
        List<IsaDescriptor> parts, IsaClass cls, ulong low, ulong high,
        bool includeMods = true)
    {
        var rendered = new List<(string Sign, string Text)>();
        string? defaultRegisterText = null;

        // memdesc=1 selects the ordinary address form for non-descriptor
        // loads/stores; that form does not carry the uniform-register part.
        // Classes with a real DESC operand keep their other address parts.
        bool hasDescOperand = cls.Format != null && cls.Format.Operands.Any(
            o => o.Kind == "field" && o.Desc?.Type == "DESC");
        bool suppressUniform = !hasDescOperand &&
            cls.FindField("memdesc") != null &&
            GetBits(low, high, cls.FindField("memdesc")!.Segs) == 1;

        foreach (var part in parts)
        {
            bool isRegister = part.Type.Contains("Register", StringComparison.Ordinal);
            if (isRegister && suppressUniform &&
                part.Type.Contains("Uniform", StringComparison.Ordinal))
                continue;

            if (isRegister)
            {
                string partType = part.Type == "NonZeroRegisterFAU"
                    ? "NonZeroRegister"
                    : part.Type;
                long value = GetFieldValue(cls, part.Field, low, high);
                string? enumText = EnumText(partType, value)
                    ?? EnumText("Register", value)
                    ?? EnumText("UniformRegister", value);
                string text = partType is "Register" or "NonZeroRegister"
                    ? LegacyRegisterText(partType, value, enumText)
                    : enumText ?? $"{partType}{value}";
                text = RenderAttachedMods(part, cls, low, high, text);
                long? defaultValue = DefaultValue(partType, part.Args);
                bool defaultRz =
                    (part.Args != null &&
                     part.Args.Trim().Trim('"').Equals(
                         "RZ", StringComparison.OrdinalIgnoreCase) &&
                     value is 63 or 255) ||
                    (partType == "NonZeroRegister" &&
                     value is 63 or 255);
                if (defaultRz ||
                    (defaultValue.HasValue && value == defaultValue.Value))
                {
                    defaultRegisterText ??= text;
                    continue;
                }
                rendered.Add((rendered.Count == 0 ? "" : "+", text));
            }
            else if (part.Type is "SImm" or "UImm")
            {
                IsaField? field = cls.FindField(part.Field);
                if (field == null) continue;

                long raw = (long)GetBits(low, high, field.Segs);
                long value = part.Type == "SImm"
                    ? SignExtend((ulong)raw, field.Width)
                    : raw;
                value *= field.Scale;
                if (cls.Mnemonic == "CCTL" && part.Field == "sImm")
                    value *= 4;
                // ConstBankAddress2 encodes a word address; nvdisasm prints
                // the byte address.  Modern tables already expose a scaled
                // field, so only apply the legacy scale when it is unscaled.
                if (field.Scale == 1 && field.Expr.Contains(
                        "ConstBankAddress2", StringComparison.Ordinal))
                    value *= 4;

                var (_, defaultText) = ParseArgs(part.Args);
                if (defaultText != null &&
                    TryParseInt(defaultText.TrimEnd('*'), out long defaultImmediate) &&
                    value == defaultImmediate)
                    continue;

                if (value < 0)
                {
                    string text = "-0x" + (-value).ToString("x", CultureInfo.InvariantCulture);
                    if (includeMods)
                        text = RenderAttachedMods(part, cls, low, high, text);
                    rendered.Add((rendered.Count == 0 ? "" : "+", text));
                }
                else
                {
                    string text = "0x" + value.ToString("x", CultureInfo.InvariantCulture);
                    if (includeMods)
                        text = RenderAttachedMods(part, cls, low, high, text);
                    rendered.Add((rendered.Count == 0 ? "" : "+", text));
                }
            }
            else if (part.Type == "text")
            {
                rendered.Add((rendered.Count == 0 ? "" : "+", part.Text ?? ""));
            }
        }

        if (rendered.Count == 0)
            return defaultRegisterText ?? "0x0";

        var builder = new StringBuilder();
        foreach (var (sign, text) in rendered)
            builder.Append(sign).Append(text);
        return builder.ToString();
    }

    private static bool HasFieldOrTableArg(IsaClass cls, string name)
    {
        if (cls.FindField(name) != null)
            return true;
        foreach (var tableField in cls.TableFields)
        {
            if (tableField.TableArgs != null &&
                Array.IndexOf(tableField.TableArgs, name) >= 0)
                return true;
        }
        return false;
    }

    private static string RenderDescriptor(IsaClass cls, ulong low, ulong high)
    {
        string? uniformName = null;
        foreach (string candidate in new[] { "Ra_URd", "Ra_URc", "Ra_URb" })
        {
            if (HasFieldOrTableArg(cls, candidate))
            {
                uniformName = candidate;
                break;
            }
        }

        long uniformRegister = uniformName != null
            ? GetFieldValue(cls, uniformName, low, high)
            : 0;
        long addressRegister = GetFieldValue(cls, "Ra", low, high);
        IsaField? offsetField = cls.FindField("Ra_offset");
        long offset = GetFieldValue(cls, "Ra_offset", low, high);
        if (offsetField != null && offsetField.Width < 64)
            offset = SignExtend((ulong)offset, offsetField.Width);

        string uniformText = "UR" + uniformRegister.ToString(CultureInfo.InvariantCulture);
        string addressText = addressRegister == 255
            ? ""
            : "R" + addressRegister.ToString(CultureInfo.InvariantCulture);

        IsaField? size64Field = cls.Fields.FirstOrDefault(
            f => f.Logical.StartsWith(
                "input_reg_sz_64", StringComparison.Ordinal));
        bool is64 = cls.Name.Contains("Ra64", StringComparison.Ordinal) ||
            (size64Field != null &&
                GetBits(low, high, size64Field.Segs) == 1);
        if (addressText.Length > 0 && is64)
            addressText += ".64";

        // e_desc=1 selects the descriptor-capable class.  The memdesc bit
        // then chooses between the real descriptor form and the normal
        // address form; memdesc=1 also suppresses the uniform register.
        if (cls.FindField("memdesc") != null &&
            GetBits(low, high, cls.FindField("memdesc")!.Segs) == 1)
        {
            string descriptorAddress = addressText;
            if (offset != 0)
            {
                descriptorAddress += (descriptorAddress.Length > 0 ? "+" : "") +
                    FormatImm(offset, true);
            }
            if (descriptorAddress.Length == 0)
                descriptorAddress = "0x0";

            return $"desc[{uniformText}][{descriptorAddress}]";
        }

        var addressParts = new List<string>();
        if (addressText.Length > 0)
            addressParts.Add(addressText);
        addressParts.Add(uniformText);
        if (offset != 0)
            addressParts.Add(FormatImm(offset, true));

        return "[" + string.Join("+", addressParts) + "]";
    }

    private static string? RenderOperand(
        IsaOperand operand, IsaClass cls, ulong low, ulong high, ulong address)
    {
        List<string> markers = operand.Markers;

        switch (operand.Kind)
        {
            case "C":
            {
                long bank = operand.Bank != null
                    ? GetFieldValue(cls, operand.Bank.Field, low, high)
                    : 0;

                string prefix = RenderMarkerPrefix(
                    markers, cls, low, high, operand.Label ?? operand.Addr?.Field, out string suffix);

                // Half-precision constants print a separating space before
                // the address and carry their word selector as a suffix:
                //   c[0x0] [0x168].H0_H0
                string separator = operand.Addr?.Mods.Count > 0 ? " " : "";

                if (operand.Parts != null && operand.Parts.Count > 0)
                {
                    string inner = RenderAddressParts(
                        operand.Parts, cls, low, high, includeMods: false);
                    string mods = operand.Addr != null
                        ? RenderAttachedMods(operand.Addr, cls, low, high, "")
                        : "";
                    return $"{prefix}c[0x{bank:x}]{separator}[{inner}]{mods}{suffix}";
                }

                long constantAddress = 0;
                int scale = 4;
                if (operand.Addr != null)
                {
                    constantAddress = GetFieldValue(cls, operand.Addr.Field, low, high);
                    IsaField? field = cls.FindField(operand.Addr.Field);
                    scale = field?.Scale ?? 4;
                }

                return $"{prefix}c[0x{bank:x}]{separator}[0x{(constantAddress * scale):x}]{suffix}";
            }

            case "CX":
            {
                long uniformRegister = operand.Ureg != null
                    ? GetFieldValue(cls, operand.Ureg.Field, low, high)
                    : 0;
                long offset = 0;
                int scale = 4;

                if (operand.Offset != null)
                {
                    offset = GetFieldValue(cls, operand.Offset.Field, low, high);
                    IsaField? field = cls.FindField(operand.Offset.Field);
                    scale = field?.Scale ?? 4;
                }

                offset *= scale;
                string uniformText = EnumText("UniformRegister", uniformRegister)
                    ?? $"UR{uniformRegister}";
                return $"cx[{uniformText}][0x{offset:x}]";
            }

            case "A":
            {
                string inner = RenderAddressParts(
                    operand.Parts ?? new List<IsaDescriptor>(), cls, low, high);
                return "[" + inner + "]";
            }

            case "field":
            {
                IsaDescriptor descriptor = operand.Desc!;
                if (descriptor.Type == "DESC")
                    return RenderDescriptor(cls, low, high);

                string? text = RenderFieldDescriptor(descriptor, cls, low, high, address);
                if (text == null) return null;

                string prefix = RenderMarkerPrefix(
                    markers, cls, low, high, descriptor.Field, out string suffix);

                // For registers the absolute-value bar closes before the
                // attached word selector: |R5|.H0_H0.  Constants keep the
                // selector inside the bars: |c[0x0] [0x168].H0_H0|.
                if (suffix.Length > 0 &&
                    descriptor.Type == "Register")
                {
                    int dot = text.IndexOf('.');
                    if (dot >= 0)
                        return prefix + text[..dot] + suffix + text[dot..];
                }
                return prefix + text + suffix;
            }

            case "text":
                return operand.Text;

            case "unknown":
                // The FORMAT parser leaves an unknown token when it cannot
                // classify a piece of text.  Do not print parser debris.
                return null;

            default:
                return operand.Text;
        }
    }

    private static string ApplyPseudoMnemonic(
        IsaClass cls, string mnemonic, ulong low, ulong high)
    {
        if (!cls.Name.StartsWith("imad", StringComparison.Ordinal))
            return mnemonic;

        // The .X/.WIDE/.HI variants have their own suffixes; nvdisasm does
        // not fold a .MOV/.IADD/.SHL pseudo-op into them.
        if (cls.Name.Contains("_x_", StringComparison.Ordinal) ||
            cls.Name.StartsWith("imad_wide", StringComparison.Ordinal) ||
            cls.Name.StartsWith("imad_hi", StringComparison.Ordinal))
            return mnemonic;

        long ra = GetFieldValue(cls, "Ra", low, high);
        long rb = GetFieldValue(cls, "Rb", low, high);
        long rc = cls.FindField("Rc") != null
            ? GetFieldValue(cls, "Rc", low, high)
            : 255;
        long sb = cls.FindField("Sb") != null
            ? GetFieldValue(cls, "Sb", low, high)
            : GetFieldValue(cls, "Sc", low, high);

        if (cls.Name.Contains("__RRR", StringComparison.Ordinal) ||
            cls.Name.Contains("__RRC", StringComparison.Ordinal))
        {
            if (ra == 255 || rb == 255)
                return mnemonic + ".MOV";
        }
        else if (cls.Name.Contains("__RCR", StringComparison.Ordinal))
        {
            if (ra == 255)
                return mnemonic + ".MOV";
        }
        else if (cls.Name.Contains("__RRsI", StringComparison.Ordinal))
        {
            // Rd, Ra, Rb, immediate.  There is no accumulator register, so
            // only a source-zero form becomes .MOV.
            if (ra == 255 || rb == 255)
                return mnemonic + ".MOV";
        }
        else if (cls.Name.Contains("__RsIR", StringComparison.Ordinal) ||
                 cls.Name.Contains("__RIR", StringComparison.Ordinal))
        {
            if (ra == 255 || sb == 0)
                return mnemonic + ".MOV";
            if (sb == 1)
                return mnemonic + (rc == 255 ? ".MOV" : ".IADD");
            if (sb > 0 && sb < 0x80000000 && sb != 0x10000 &&
                (sb & (sb - 1)) == 0 && rc == 255)
                return mnemonic + ".SHL";
        }

        return mnemonic;
    }

    private static ulong BranchBase() => IsPreVoltaTable() ? 8UL : 16UL;

    private static ulong? GetBranchTarget(
        IsaClass cls, ulong low, ulong high, ulong address)
    {
        if (cls.Format == null) return null;

        foreach (var operand in cls.Format.Operands)
        {
            if (operand.Kind != "field" || operand.Desc?.Type != "RSImm")
                continue;

            IsaField? field = cls.FindField(operand.Desc.Field);
            int width = field?.Width ?? 0;
            int scale = field?.Scale ?? 4;
            long immediate = SignExtend(
                (ulong)GetFieldValue(cls, operand.Desc.Field, low, high), width);
            return unchecked(address + BranchBase() + (ulong)(immediate * scale));
        }

        // RET.REL carries an implicit return address but no RSImm operand.
        if (cls.Format.Mods.Any(m => m.Type == "RELONLY"))
        {
            IsaField? target = cls.FindField("sImm") ?? cls.FindField("Ra_offset");
            if (target != null)
            {
                long immediate = SignExtend(
                    GetBits(low, high, target.Segs), target.Width);
                return unchecked(
                    address + BranchBase() + (ulong)(immediate * target.Scale));
            }
        }

        return null;
    }

    // -- public decode API --------------------------------------------------

    public static DecodedInstruction Decode(
        ulong low, ulong high, ulong address = 0, int reuseCode = 0)
    {
        IsaClass? cls = SelectClass(low, high);
        int opcode = (int)GetBits(low, high, Table.OpcodeSegments);

        // SM70/SM72 encode padding NOPs using the all-RZ SHFL bit pattern.
        // nvdisasm presents that pattern as NOP rather than SHFL.
        if (low == 0x000000fffffff389UL)
        {
            IsaClass? nop = Table.Classes.FirstOrDefault(
                candidate => candidate.Mnemonic == "NOP");
            return new DecodedInstruction
            {
                Instruction = ((UInt128)high << 64) | low,
                Low = low,
                High = high,
                Address = address,
                IsValid = true,
                Text = "NOP",
                Mnemonic = "NOP",
                Opcode = nop?.Opcode ?? opcode,
                ClassName = nop?.Name ?? "nop_",
                Class = nop,
                Guard = 7,
                GuardNot = false,
            };
        }

        if (cls == null)
        {
            return new DecodedInstruction
            {
                Instruction = ((UInt128)high << 64) | low,
                Low = low,
                High = high,
                Address = address,
                IsValid = false,
                Text = "???",
                Opcode = opcode,
            };
        }

        string mnemonic = ApplyPseudoMnemonic(cls, cls.Mnemonic, low, high);
        var builder = new StringBuilder();

        if (cls.Format?.Guard is { } guard)
        {
            long guardValue = GetFieldValue(cls, guard.Field, low, high);
            long? defaultValue = DefaultValue(guard.Type, guard.Default);
            IsaField? notField = FindLinked(cls, guard.Field, "not");
            bool negated = notField != null &&
                GetBits(low, high, notField.Segs) != 0;
            if (negated ||
                !defaultValue.HasValue || guardValue != defaultValue.Value)
            {
                string guardName = EnumText(guard.Type, guardValue)
                    ?? (guard.Type == "UniformPredicate"
                        ? $"UP{guardValue}"
                        : $"P{guardValue}");
                builder.Append('@').Append(negated ? "!" : "").Append(guardName).Append(' ');
            }
        }

        builder.Append(mnemonic);

        if (cls.Format != null)
        {
            foreach (var mod in cls.Format.Mods)
                builder.Append(RenderModifier(mod, cls, low, high));

            // SM35 keeps NOP's /S outside the parsed modifier list; SM30 has
            // a real S modifier, so only add the fallback when it is absent.
            if (cls.Mnemonic == "NOP" &&
                !cls.Format.Mods.Any(mod => mod.Type == "S") &&
                GetFieldValue(cls, "Sync", low, high) != 0)
                builder.Append(".S");

            bool first = true;
            foreach (var operand in cls.Format.Operands)
            {
                string? text = RenderOperand(operand, cls, low, high, address);
                if (text == null) continue;
                if (operand.Kind == "field")
                {
                    string? fieldName = operand.Desc?.Field;
                    if (cls.Mnemonic == "IADD" && fieldName is "Ra" or "Rb")
                    {
                        long signs = GetFieldValue(cls, "PSign", low, high);
                        bool negate = fieldName == "Ra"
                            ? (signs & 2) != 0
                            : (signs & 1) != 0;
                        if (negate && !text.StartsWith("-", StringComparison.Ordinal))
                            text = "-" + text;
                    }

                    int reuseBit = fieldName switch
                    {
                        "Ra" => 1,
                        "Rb" => 2,
                        "Rc" => 4,
                        _ => 0,
                    };
                    if (reuseBit != 0 && (reuseCode & reuseBit) != 0)
                        text += ".reuse";
                }
                builder.Append(first ? " " : ", ").Append(text);
                first = false;
            }

            // BRX carries a signed target offset in sImm.  The CUDA 10.2
            // BRX FORMAT omits that operand from the parsed operand list, so
            // add nvdisasm's trailing annotation explicitly.
            if (cls.Mnemonic == "BRX")
            {
                IsaField? target = cls.FindField("sImm");
                if (target != null)
                {
                    long immediate = SignExtend(
                        GetBits(low, high, target.Segs), target.Width);
                    if (immediate != 0)
                    {
                        long magnitude = immediate < 0 ? -immediate : immediate;
                        builder.Append(immediate < 0 ? " -0x" : " 0x")
                            .Append(magnitude.ToString(
                                "x", CultureInfo.InvariantCulture));
                    }
                }
            }

            // A relative RET prints its implicit return address after the
            // register operand, e.g. RET.REL.NODEC R6 0x0.
            bool relativeRet = cls.Format.Mods.Any(m => m.Type == "RELONLY");
            bool absoluteRet = cls.Format.Mods.Any(m => m.Type == "ABSONLY_ret");
            if (relativeRet || absoluteRet)
            {
                IsaField? target = cls.FindField("sImm") ?? cls.FindField("Ra_offset");
                if (target != null)
                {
                    long immediate = SignExtend(
                        GetBits(low, high, target.Segs), target.Width);
                    if (relativeRet)
                    {
                        ulong value = unchecked(
                            address + BranchBase() + (ulong)(immediate * target.Scale));
                        builder.Append(" 0x").Append(
                            value.ToString("x", CultureInfo.InvariantCulture));
                    }
                    else
                    {
                        long absolute = immediate * target.Scale;
                        ulong magnitude = absolute < 0
                            ? unchecked((ulong)(-(absolute + 1)) + 1)
                            : (ulong)absolute;
                        builder.Append(absolute < 0 ? " -0x" : " 0x").Append(
                            magnitude.ToString("x", CultureInfo.InvariantCulture));
                    }
                }
            }
        }

        // nvdisasm presents a pre-Volta branch with a false condition code as
        // NOP, even though the branch word and target are still present.
        if (IsPreVoltaTable() &&
            (mnemonic.Split('.')[0] is "BRA" or "BRX" or "JMP") &&
            cls.FindField("CCC_1") != null &&
            GetFieldValue(cls, "CCC_1", low, high) == 0)
        {
            mnemonic = "NOP";
            builder.Clear().Append("NOP");
        }

        int guardRegister = (int)GetFieldValue(cls, "Pg", low, high);
        bool guardNot = GetFieldValue(cls, "Pg_not", low, high) != 0;

        return new DecodedInstruction
        {
            Instruction = ((UInt128)high << 64) | low,
            Low = low,
            High = high,
            Address = address,
            IsValid = true,
            Text = builder.ToString(),
            Mnemonic = mnemonic,
            Opcode = opcode,
            ClassName = cls.Name,
            Class = cls,
            BranchTarget = GetBranchTarget(cls, low, high, address),
            Guard = guardRegister,
            GuardNot = guardNot,
            Rd = ExtractRegister(cls, "Rd", low, high),
            Ra = ExtractRegister(cls, "Ra", low, high),
            Rb = ExtractRegister(cls, "Rb", low, high),
            Rc = ExtractRegister(cls, "Rc", low, high),
        };
    }

    private static byte ExtractRegister(
        IsaClass cls, string name, ulong low, ulong high)
    {
        // A class that does not encode this register at all is reported as
        // RZ.  This is important for formats such as IMAD RRC, where the
        // constant operand is not Rc, and for control instructions such as
        // BRA, which have no destination register.
        if (cls.FindField(name) == null)
            return 255;
        return (byte)GetFieldValue(cls, name, low, high);
    }

    public static DecodedInstruction Decode(UInt128 instruction, ulong address = 0)
    {
        ulong low = (ulong)instruction;
        ulong high = (ulong)(instruction >> 64);
        return Decode(low, high, address);
    }
}
