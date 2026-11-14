using System.Globalization;
using System.Text.RegularExpressions;
using System.Text;

namespace rtxdisam;

// ---------------------------------------------------------------------------
// Small SASS assembly IR.
//
// The parser deliberately keeps instructions structural rather than storing
// pre-encoded words: section/label layout, operand parsing and branch fixups
// are resolved by the encoder after the complete program is known.
// ---------------------------------------------------------------------------

public sealed class SassProgram
{
    public string Architecture { get; set; } = "";
    public string Toolkit { get; set; } = "";
    public List<SassSection> Sections { get; } = new();
}

public sealed class SassSection
{
    public string Name { get; set; } = ".text";
    public int Alignment { get; set; } = 128;
    public int? RegisterCount { get; set; }
    // Architecture active when this section was parsed.  A fatbin can contain
    // cubins for several architectures, so this must be kept per section.
    public string? Architecture { get; set; }
    public List<SassNode> Nodes { get; } = new();
}

public abstract class SassNode
{
    public int Address { get; set; }
}

public sealed class SassLabel : SassNode
{
    public string Name { get; init; } = "";
}

public sealed class SassRelocation
{
    public string Symbol { get; init; } = "";
    public string AddendSymbol { get; init; } = "";
    public long Addend { get; init; }
    public int Type { get; init; }
    public bool IsRel { get; init; }
}

public sealed class SassInstruction : SassNode
{
    public string Mnemonic { get; init; } = "";
    public string? Guard { get; init; }
    public List<string> Operands { get; init; } = new();
    public List<string> BranchTargets { get; init; } = new();
    public List<SassRelocation> Relocations { get; init; } = new();
    // Optional explicit KeplerAs/Kepler control byte:
    // T:G:D:S:nn (texture/global/dual/shared/stall).
    public int? ControlByte { get; init; }
    // Optional exact 64-bit control word for the bundle containing this
    // instruction.  This covers both Kepler and Maxwell/Pascal layouts.
    public ulong? ControlWord { get; init; }
    // Optional exact 64-bit instruction word.  This preserves source forms
    // that have several equivalent encodings, such as padding NOPs.
    public ulong? InstructionWord { get; set; }
    public string Source { get; init; } = "";
    public int Line { get; init; }
    // Legacy FADD32I/FMUL32I print .NEG/.ABS after the immediate.
    public string? LegacyFloatModifiers { get; init; }
}

public static class SassParser
{
    public static SassProgram Parse(string path) =>
        ParseLines(File.ReadAllLines(path), path);

    public static SassProgram ParseLines(
        string[] lines, string sourceName = "<text>",
        bool separateSections = false)
    {
        SassProgram program = new();
        SassSection current = new();
        program.Sections.Add(current);

        ulong? pendingControlWord = null;
        SassInstruction? lastInstruction = null;
        bool lastHadEncoding = false;
        for (int index = 0; index < lines.Length; index++)
        {
            string raw = lines[index];
            Match standaloneControl = Regex.Match(raw,
                @"^\s*/\*\s*0x([0-9A-Fa-f]{16})\s*\*/\s*$",
                RegexOptions.IgnoreCase);
            if (standaloneControl.Success)
            {
                ulong standaloneWord = ulong.Parse(
                    standaloneControl.Groups[1].Value,
                    NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                if (lastInstruction != null && !lastHadEncoding &&
                    lastInstruction.InstructionWord == null)
                {
                    lastInstruction.InstructionWord = standaloneWord;
                    lastHadEncoding = true;
                }
                else
                {
                    pendingControlWord = standaloneWord;
                }
                continue;
            }
            Match trailingInstruction = Regex.Match(raw,
                @"/\*\s*0x([0-9A-Fa-f]{16})\s*\*/\s*$",
                RegexOptions.IgnoreCase);
            ulong? nvInstructionWord = trailingInstruction.Success
                ? ulong.Parse(trailingInstruction.Groups[1].Value,
                    NumberStyles.HexNumber, CultureInfo.InvariantCulture)
                : null;
            string line = StripComments(raw).Trim();
            if (line.Length == 0)
                continue;

            Match addressMatch = Regex.Match(raw,
                @"^\s*/\*\s*([0-9A-Fa-f]+)\s*\*/");
            int? instructionAddress = addressMatch.Success
                ? int.Parse(addressMatch.Groups[1].Value,
                    NumberStyles.HexNumber, CultureInfo.InvariantCulture)
                : null;

            if (line.EndsWith(':'))
            {
                string name = line[..^1].Trim();
                if (name.Length == 0)
                    throw Error(sourceName, index, "empty label");
                current.Nodes.Add(new SassLabel { Name = name });
                lastInstruction = null;
                lastHadEncoding = false;
                continue;
            }

            if (line.StartsWith(".", StringComparison.Ordinal))
            {
                SassSection? next = ParseDirective(
                    line, program, current, separateSections);
                if (next != null)
                    current = next;
                lastInstruction = null;
                lastHadEncoding = false;
                continue;
            }

            if (line.StartsWith("//", StringComparison.Ordinal))
                continue;

            string instruction = line.TrimEnd(';').Trim();
            if (instruction.Length == 0)
                continue;

            SassInstruction parsedInstruction = ParseInstruction(
                instruction, index + 1, pendingControlWord, nvInstructionWord);
            if (instructionAddress.HasValue)
                parsedInstruction.Address = instructionAddress.Value;
            current.Nodes.Add(parsedInstruction);
            lastInstruction = parsedInstruction;
            lastHadEncoding = nvInstructionWord != null;
            pendingControlWord = null;
        }

        foreach (SassSection section in program.Sections)
        {
            for (int i = 0; i < section.Nodes.Count; i++)
            {
                if (section.Nodes[i] is not SassLabel label)
                    continue;
                for (int j = i + 1; j < section.Nodes.Count; j++)
                {
                    if (section.Nodes[j] is SassInstruction sassInstruction)
                    {
                        label.Address = sassInstruction.Address;
                        break;
                    }
                }
            }
        }

        return program;
    }

    private static string TrimDualIssueBraces(string text)
    {
        text = text.Trim();
        if (text.StartsWith('{'))
            text = text[1..].TrimStart();
        int opens = text.Count(ch => ch == '{');
        int closes = text.Count(ch => ch == '}');
        if (text.EndsWith('}') && opens < closes)
            text = text[..^1].TrimEnd();
        return text;
    }

    private static SassInstruction ParseInstruction(
        string text, int line, ulong? nvControlWord = null,
        ulong? nvInstructionWord = null)
    {
        text = TrimDualIssueBraces(text);

        int? controlByte = null;
        ulong? controlWord = null;
        ulong? instructionWord = null;
        while (true)
        {
            Match wordControl = Regex.Match(text,
                @"^W:0x([0-9A-Fa-f]{16})\s+(.+)$",
                RegexOptions.IgnoreCase);
            if (wordControl.Success)
            {
                controlWord = ulong.Parse(
                    wordControl.Groups[1].Value,
                    NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                text = TrimDualIssueBraces(wordControl.Groups[2].Value);
                continue;
            }
            Match instWord = Regex.Match(text,
                @"^I:0x([0-9A-Fa-f]{16})\s+(.+)$",
                RegexOptions.IgnoreCase);
            if (instWord.Success)
            {
                instructionWord = ulong.Parse(
                    instWord.Groups[1].Value,
                    NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                text = TrimDualIssueBraces(instWord.Groups[2].Value);
                continue;
            }
            break;
        }
        controlWord ??= nvControlWord;
        instructionWord ??= nvInstructionWord;
        Match control = Regex.Match(text,
            @"^([T-]):([G-]):([D-]):([S-]):([0-9A-Fa-f]{2})\s+(.+)$",
            RegexOptions.IgnoreCase);
        if (control.Success)
        {
            int tex = control.Groups[1].Value.Equals(
                "T", StringComparison.OrdinalIgnoreCase) ? 0x80 : 0;
            int global = control.Groups[2].Value.Equals(
                "G", StringComparison.OrdinalIgnoreCase) ? 0x40 : 0;
            int singleIssue = control.Groups[3].Value.Equals(
                "D", StringComparison.OrdinalIgnoreCase) ? 0 : 0x20;
            int shared = control.Groups[4].Value.Equals(
                "S", StringComparison.OrdinalIgnoreCase) ? 0x10 : 0;
            if (!int.TryParse(control.Groups[5].Value,
                    NumberStyles.Integer, CultureInfo.InvariantCulture,
                    out int stall))
                throw Error("", line, "invalid control stall");
            controlByte = tex | global | singleIssue | shared | (stall & 0xF);
            text = TrimDualIssueBraces(control.Groups[6].Value);
        }

        var branchTargets = new List<string>();
        Match annotation = Regex.Match(text,
            @"\(\*\s*""BRANCH_TARGETS\s+([^""]*)""\s*\*\)",
            RegexOptions.IgnoreCase);
        if (annotation.Success)
        {
            foreach (string target in annotation.Groups[1].Value.Split(','))
            {
                string name = target.Trim();
                if (name.Length > 0)
                    branchTargets.Add(name);
            }
            text = text.Remove(annotation.Index, annotation.Length).Trim();
        }

        // Pre-Volta SYNC carries its reconvergence point in a TARGET=
        // annotation rather than an explicit operand.  Keep the target so
        // the lifter can lower the reconvergence edge instead of dropping
        // it with the other annotations.
        Match targetAnnotation = Regex.Match(text,
            @"\(\*\s*""TARGET=\s*([^""]+)""\s*\*\)",
            RegexOptions.IgnoreCase);
        if (targetAnnotation.Success)
        {
            string name = targetAnnotation.Groups[1].Value.Trim();
            if (name.Length > 0)
                branchTargets.Add(name);
        }

        var relocations = new List<SassRelocation>();
        foreach (Match relocation in Regex.Matches(text,
            @"(?:\d+)@(?<kind>lo|hi)\(\((?<expr>[^()]+)\)\)",
            RegexOptions.IgnoreCase))
        {
            string expression = relocation.Groups["expr"].Value.Trim();
            string symbol = expression;
            string addend = "";
            int plus = expression.IndexOf('+');
            if (plus >= 0)
            {
                symbol = expression[..plus].Trim();
                addend = expression[(plus + 1)..].Trim();
                if (addend.EndsWith("@srel", StringComparison.OrdinalIgnoreCase))
                    addend = addend[..^5].Trim();
            }
            relocations.Add(new SassRelocation
            {
                Symbol = symbol,
                AddendSymbol = addend,
                Type = relocation.Groups["kind"].Value.Equals(
                    "hi", StringComparison.OrdinalIgnoreCase) ? 0x39 : 0x38,
            });
        }
        if (relocations.Count > 0)
            text = Regex.Replace(text,
                @"(?:\d+)@(?:lo|hi)\(\([^()]+\)\)", "0x0",
                RegexOptions.IgnoreCase);

        // Shared-memory window operands use a symbolic offset.  Keep the
        // register addressing form and recover the relocation from it.
        if (Regex.IsMatch(text, @"^(?:@\S+\s+)?(?:LDS|STS)\b",
                RegexOptions.IgnoreCase))
        {
            text = Regex.Replace(text,
                @"\+\s*`\(\(\s*(?<symbol>[A-Za-z_][\w.$]*)\s*\+\s*(?<addend>-?(?:0x[0-9a-fA-F]+|\d+))\s*\)\)(?=\])",
                match =>
                {
                    TryParseInteger(match.Groups["addend"].Value,
                        out long addend);
                    relocations.Add(new SassRelocation
                    {
                        Symbol = match.Groups["symbol"].Value,
                        Addend = addend,
                        Type = 0x4a,
                        IsRel = addend == 0,
                    });
                    return "";
                },
                RegexOptions.IgnoreCase);
            text = Regex.Replace(text,
                @"\+\s*`\(\s*(?<symbol>[A-Za-z_][\w.$]*)\s*\)(?=\])",
                match =>
                {
                    relocations.Add(new SassRelocation
                    {
                        Symbol = match.Groups["symbol"].Value,
                        Type = 0x4a,
                        IsRel = true,
                    });
                    return "";
                },
                RegexOptions.IgnoreCase);
        }

        // Relocations that nvdisasm cannot express as SASS operands are
        // carried in explicit annotations when the decoder is invoked with
        // --not-nvdisasm-compatible.  RELOCREL marks an SHT_REL entry.
        var annotatedRelocations = new List<SassRelocation>();
        foreach (Match preserved in Regex.Matches(text,
            @"\(\*\s*""(?<kind>RELOCREL|RELOC)\s+(?<type>\S+)\s+(?<symbol>[^\s""]+)\s+(?<addend>\S+)""\s*\*\)",
            RegexOptions.IgnoreCase))
        {
            if (!TryParseInteger(preserved.Groups["type"].Value, out long type) ||
                !TryParseInteger(preserved.Groups["addend"].Value, out long addend))
                continue;
            annotatedRelocations.Add(new SassRelocation
            {
                Symbol = preserved.Groups["symbol"].Value,
                Addend = addend,
                Type = (int)type,
                IsRel = preserved.Groups["kind"].Value.Equals(
                    "RELOCREL", StringComparison.OrdinalIgnoreCase),
            });
        }
        text = Regex.Replace(text, @"\(\*.*?\*\)", "").Trim();
        // Constant-bank addresses can contain a symbol expression.
        Match constantRelocation = Regex.Match(text,
            @"c\[(R\d+)\+`\(\(([^()]*)\)\)\]",
            RegexOptions.IgnoreCase);
        if (constantRelocation.Success)
        {
            string expression = constantRelocation.Groups[2].Value.Trim();
            string symbol = expression;
            long addend = 0;
            int plus = expression.IndexOf('+');
            if (plus >= 0)
            {
                symbol = expression[..plus].Trim();
                string addendText = expression[(plus + 1)..].Trim();
                bool negativeAddend = addendText.StartsWith('-');
                if (negativeAddend)
                    addendText = addendText[1..];
                if (addendText.StartsWith("0x", StringComparison.OrdinalIgnoreCase) &&
                    long.TryParse(addendText[2..], NumberStyles.HexNumber,
                        CultureInfo.InvariantCulture, out long hexAddend))
                    addend = hexAddend;
                else
                    long.TryParse(addendText, NumberStyles.Integer,
                        CultureInfo.InvariantCulture, out addend);
                if (negativeAddend)
                    addend = -addend;
            }
            relocations.Add(new SassRelocation
            {
                Symbol = symbol,
                Addend = addend,
                Type = 0x42,
            });
            text = text.Remove(constantRelocation.Index,
                    constantRelocation.Length)
                .Insert(constantRelocation.Index,
                    "c[0x0][" + constantRelocation.Groups[1].Value + "]");
        }

        if (annotatedRelocations.Count > 0)
        {
            // Relocation annotations are authoritative.  Remove symbolic
            // operand placeholders (memory offsets, call targets, UMOV
            // operands) so the instruction can be encoded normally; the
            // annotated relocation carries the symbol.  A bare constant
            // operand is encoded as bank 0, address 0.  A symbolic memory
            // offset is dropped along with its '+', leaving [Rn].
            text = Regex.Replace(text,
                @"c\[\s*`\(\([^()]*\)\)\s*\]", "c[0x0][0x0]");
            text = Regex.Replace(text,
                @"c\[\s*`\([^()]+\)\s*\]", "c[0x0][0x0]");
            text = Regex.Replace(text, @"\+\s*`\(\([^()]*\)\)(?=\])", "");
            text = Regex.Replace(text, @"\+\s*`\([^()]+\)(?=\])", "");
            text = Regex.Replace(text, @"`\(\([^()]*\)\)", "0x0");
            text = Regex.Replace(text, @"`\([^()]+\)", "0x0");
        }

        // nvdisasm marks the halves of a dual-issued pair with SLOT 0/1.
        text = Regex.Replace(text, @"\bSLOT\s*[01]\b", "",
            RegexOptions.IgnoreCase).Trim();

        int space = text.IndexOfAny(new[] { ' ', '\t' });
        string head = space < 0 ? text : text[..space];
        string tail = space < 0 ? "" : text[(space + 1)..].Trim();

        string? guard = null;
        if (head.StartsWith('@'))
        {
            guard = head;
            int next = tail.IndexOfAny(new[] { ' ', '\t' });
            if (next < 0)
            {
                head = tail;
                tail = "";
            }
            else
            {
                head = tail[..next];
                tail = tail[(next + 1)..].Trim();
            }
        }

        var operands = SplitOperands(tail);
        var expanded = new List<string>(operands.Count);
        foreach (string operand in operands)
        {
            // nvdisasm prints a RET target symbol after the register without
            // a comma: RET.REL.NODEC R6 `(more).
            int backtick = operand.IndexOf('`');
            if (backtick > 0)
            {
                expanded.Add(operand[..backtick].Trim());
                expanded.Add(operand[backtick..].Trim());
            }
            else
            {
                expanded.Add(operand);
            }
        }

        // A non-uniform MOV/immediate form can reference a shared-memory
        // window symbol directly.
        if (!head.StartsWith("UMOV", StringComparison.OrdinalIgnoreCase) &&
            !head.StartsWith("CALL.ABS", StringComparison.OrdinalIgnoreCase))
        {
            for (int i = 0; i < expanded.Count; i++)
            {
                Match shared = Regex.Match(expanded[i].Trim(),
                    @"^`\((?<symbol>[A-Za-z_][\w.$]*)\)$",
                    RegexOptions.IgnoreCase);
                if (!shared.Success)
                    continue;
                string symbol = shared.Groups["symbol"].Value;
                int type = symbol.StartsWith("constTbl",
                    StringComparison.OrdinalIgnoreCase)
                    ? 0x3b
                    : symbol.Equals("aShrdMem", StringComparison.OrdinalIgnoreCase) ||
                      symbol.Equals("shrdMem32", StringComparison.OrdinalIgnoreCase) ||
                      symbol.Equals("smem_full", StringComparison.OrdinalIgnoreCase)
                        ? 0x37
                        : -1;
                if (type < 0)
                    continue;
                relocations.Add(new SassRelocation
                {
                    Symbol = symbol,
                    Type = type,
                    IsRel = type == 0x3b,
                });
                expanded[i] = "0x0";
            }
        }

        // Uniform moves can carry an absolute 32-bit symbol reference.
        if (head.StartsWith("UMOV", StringComparison.OrdinalIgnoreCase))
        {
            for (int i = 0; i < expanded.Count; i++)
            {
                string value = expanded[i].Trim();
                if (!value.StartsWith('`'))
                    continue;
                string symbol = value.Trim('`', '(', ')');
                if (symbol.Length == 0)
                    continue;
                relocations.Add(new SassRelocation
                {
                    Symbol = symbol,
                    Type = 0x37,
                });
                expanded[i] = "0x0";
            }
        }

        // Absolute and PC-relative calls carry a relocation for their
        // target symbol.
        if (head.StartsWith("CALL.ABS", StringComparison.OrdinalIgnoreCase) ||
            head.StartsWith("CALL.REL", StringComparison.OrdinalIgnoreCase))
        {
            for (int i = 0; i < expanded.Count; i++)
            {
                string value = expanded[i].Trim().Trim('`', '(', ')');
                if (value.Length == 0 ||
                    int.TryParse(value, NumberStyles.Integer,
                        CultureInfo.InvariantCulture, out _) ||
                    value.StartsWith("0x", StringComparison.OrdinalIgnoreCase) ||
                    Regex.IsMatch(value, @"^(?:U?R\d+|RZ|URZ)$",
                        RegexOptions.IgnoreCase))
                    continue;
                relocations.Add(new SassRelocation
                {
                    Symbol = value,
                    Type = 0x4b,
                });
                expanded[i] = "0x0";
            }
        }

        // An explicit annotation set is authoritative.  The decoder emits it
        // for every relocation on the instruction when non-nvdisasm-compatible
        // mode is enabled, so discard the heuristically reconstructed
        // textual relocations to preserve the original types and REL/RELA
        // section kind.
        if (annotatedRelocations.Count > 0)
        {
            relocations.Clear();
            relocations.AddRange(annotatedRelocations);
        }

        // Some relative-return and indirect-branch forms print a numeric
        // target after the register with a space instead of an annotation.
        if (expanded.Count == 1 &&
            (head.StartsWith("RET", StringComparison.OrdinalIgnoreCase) ||
             head.StartsWith("BRX", StringComparison.OrdinalIgnoreCase)))
        {
            int lastSpace = expanded[0].LastIndexOf(' ');
            if (lastSpace > 0)
            {
                string target = expanded[0][(lastSpace + 1)..].Trim();
                if (target.StartsWith("0x", StringComparison.OrdinalIgnoreCase) ||
                    target.StartsWith("-0x", StringComparison.OrdinalIgnoreCase) ||
                    target.StartsWith('('))
                {
                    expanded[0] = expanded[0][..lastSpace].Trim();
                    expanded.Add(target);
                }
            }
        }

        string headOp = head.Split('.')[0].ToUpperInvariant();
        string? legacyFloatModifiers = null;
        if (headOp == "FADD32I" || headOp == "FMUL32I")
        {
            if (expanded.Count >= 4 &&
                IsLegacyFloatModifier(expanded[^1]))
            {
                legacyFloatModifiers = expanded[^1]
                    .Trim().TrimStart('.').ToUpperInvariant();
                expanded.RemoveAt(expanded.Count - 1);
            }
            if (expanded.Count >= 3)
            {
                string immediate = expanded[^1];
                foreach (string suffix in new[]
                {
                    ".NEG.ABS", ".ABS.NEG", ".NEG", ".ABS",
                })
                {
                    if (!immediate.EndsWith(
                            suffix, StringComparison.OrdinalIgnoreCase))
                        continue;
                    legacyFloatModifiers = suffix
                        .TrimStart('.').ToUpperInvariant();
                    expanded[^1] = immediate[..^suffix.Length];
                    break;
                }
            }
        }

        if (controlByte != null && controlWord != null)
            throw Error("", line, "control byte and control word are exclusive");

        return new SassInstruction
        {
            Mnemonic = head,
            Guard = guard,
            Operands = expanded,
            BranchTargets = branchTargets,
            Relocations = relocations,
            ControlByte = controlByte,
            ControlWord = controlWord,
            InstructionWord = instructionWord,
            Source = text,
            Line = line,
            LegacyFloatModifiers = legacyFloatModifiers,
        };
    }

    private static bool IsLegacyFloatModifier(string operand)
    {
        string value = operand.Trim().ToUpperInvariant();
        return value is ".NEG" or ".ABS" or ".NEG.ABS" or ".ABS.NEG";
    }

    private static SassSection? ParseDirective(
        string line, SassProgram program, SassSection current,
        bool separateSections)
    {
        int space = line.IndexOfAny(new[] { ' ', '\t' });
        string directive = space < 0 ? line : line[..space];
        string value = space < 0 ? "" : line[(space + 1)..].Trim();

        switch (directive)
        {
            case ".target":
                program.Architecture = NormalizeArchitecture(value);
                break;
            case ".toolkit":
                program.Toolkit = value.Trim().Trim('"');
                break;
            case ".section":
            {
                string name = value.Split(',')[0].Trim().Trim('"');
                if (name.Length == 0)
                    break;
                SassSection? section = separateSections
                    ? null
                    : program.Sections.LastOrDefault(section =>
                        section.Name == name &&
                        string.Equals(
                            section.Architecture,
                            program.Architecture,
                            StringComparison.OrdinalIgnoreCase));
                if (section == null)
                {
                    section = new SassSection
                    {
                        Name = name,
                        Architecture = program.Architecture,
                    };
                    program.Sections.Add(section);
                }
                return section;
            }
            case ".sectioninfo":
            {
                Match registerCount = Regex.Match(value,
                    @"SHI_REGISTERS\s*=\s*(\d+)",
                    RegexOptions.IgnoreCase);
                if (registerCount.Success &&
                    int.TryParse(registerCount.Groups[1].Value,
                        NumberStyles.None, CultureInfo.InvariantCulture,
                        out int count))
                    current.RegisterCount = count;
                break;
            }
            case ".align":
                if (int.TryParse(value, NumberStyles.Integer,
                        CultureInfo.InvariantCulture, out int alignment) &&
                    alignment > 0)
                    current.Alignment = alignment;
                break;
        }
        return null;
    }

    private static string NormalizeArchitecture(string text)
    {
        text = text.Trim().Trim('"');
        int dot = text.IndexOf('.');
        if (dot >= 0)
            text = text[..dot];
        return text.StartsWith("sm_", StringComparison.OrdinalIgnoreCase)
            ? "sm_" + text[3..].ToLowerInvariant()
            : "sm_" + text.TrimStart('s', 'm', 'S', 'M');
    }

    private static string StripComments(string line)
    {
        int block = line.IndexOf("/*", StringComparison.Ordinal);
        if (block >= 0)
        {
            int end = line.IndexOf("*/", block + 2, StringComparison.Ordinal);
            line = end >= 0
                ? line[..block] + line[(end + 2)..]
                : line[..block];
        }
        int slash = line.IndexOf("//", StringComparison.Ordinal);
        if (slash >= 0)
            line = line[..slash];
        return line;
    }

    private static List<string> SplitOperands(string text)
    {
        var result = new List<string>();
        if (text.Length == 0)
            return result;

        int start = 0;
        int square = 0;
        int round = 0;
        int curly = 0;
        for (int i = 0; i < text.Length; i++)
        {
            char ch = text[i];
            if (ch == '[') square++;
            else if (ch == ']') square--;
            else if (ch == '(') round++;
            else if (ch == ')') round--;
            else if (ch == '{') curly++;
            else if (ch == '}') curly--;
            else if (ch == ',' && square == 0 && round == 0 && curly == 0)
            {
                result.Add(text[start..i].Trim());
                start = i + 1;
            }
        }
        result.Add(text[start..].Trim());
        return result.Where(x => x.Length > 0).ToList();
    }

    private static bool TryParseInteger(string text, out long value)
    {
        bool negative = text.StartsWith('-');
        if (negative)
            text = text[1..];
        bool parsed = text.StartsWith("0x", StringComparison.OrdinalIgnoreCase)
            ? long.TryParse(text[2..], NumberStyles.HexNumber,
                CultureInfo.InvariantCulture, out value)
            : long.TryParse(text, NumberStyles.Integer,
                CultureInfo.InvariantCulture, out value);
        if (parsed && negative)
            value = -value;
        return parsed;
    }

    private static FormatException Error(string path, int line, string message) =>
        new($"{path}:{line + 1}: {message}");
}
