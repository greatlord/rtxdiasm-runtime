using System.Buffers.Binary;
using System.Globalization;
using System.Text.RegularExpressions;
using nvDismDecode;

namespace rtxdisam;

// ---------------------------------------------------------------------------
// SASS -> IR -> cubin assembler.
//
// The parser produces SassProgram/SassSection/SassInstruction nodes.  This
// class assigns addresses, resolves labels, encodes each instruction through
// the generated ISA class table and finally hands executable sections to
// CubinWriter.  Encoding is table-driven: no opcode masks are hard-coded.
// ---------------------------------------------------------------------------
public static class SassAssembler
{
    public static void Assemble(
        string sourcePath, string? requestedArchitecture, string outputPath,
        string? irOutputPath = null, string? toolkit = null)
    {
        SassProgram program = SassParser.Parse(sourcePath);
        string architecture = requestedArchitecture ??
            (program.Architecture.Length > 0 ? program.Architecture : "");
        if (architecture.Length == 0)
            throw new InvalidDataException(
                "No architecture specified; use .target or -arch.");
        string? effectiveToolkit = InstructionDecoder.NormalizeToolkit(
            toolkit ?? (program.Toolkit.Length > 0 ? program.Toolkit : null));

        if (IsPreVoltaArchitecture(architecture))
        {
            // Validate the ABI/table before writing IR or a cubin.
            PreVoltaDecoder.ResolveTablePath(architecture, effectiveToolkit);
            effectiveToolkit ??= "102";
            PreVoltaAssembler.Assemble(
                program, architecture, outputPath, irOutputPath,
                effectiveToolkit);
            return;
        }

        if (ArchitectureNames.TryCanonicalizeLegacySm101(
                architecture, out string canonical))
        {
            Console.Error.WriteLine(
                "WARNING: SM101 was renamed to SM110 in PTX ISA 9.0; " +
                $"emitting an {canonical} cubin.");
            architecture = canonical;
        }

        // Resolve the table before writing IR or encoding instructions.  This
        // fails when the requested architecture/toolkit ABI does not exist.
        InstructionDecoder.UseArchitecture(architecture, effectiveToolkit);
        effectiveToolkit ??= InstructionDecoder.Table.Toolkit;

        if (irOutputPath != null)
            SassIRSerializer.Write(
                irOutputPath, program, architecture, effectiveToolkit);

        var imageSections = new List<CubinImageSection>();
        var imageSymbols = new List<CubinImageSymbol>();
        foreach (SassSection section in program.Sections)
        {
            var instructions = section.Nodes
                .OfType<SassInstruction>()
                .ToList();
            if (instructions.Count == 0)
                continue;

            var labels = new Dictionary<string, int>(StringComparer.Ordinal);
            int address = 0;
            foreach (SassNode node in section.Nodes)
            {
                if (node is SassLabel label)
                    labels[label.Name] = address;
                else if (node is SassInstruction instruction)
                {
                    instruction.Address = address;
                    address += 16;
                }
            }

            // The 0x4b (absolute-call) relocation is only understood by
            // nvdisasm for SM90 and newer.  Older cubins encode local
            // absolute calls directly; emitting the relocation there makes
            // nvdisasm reject the whole code section.
            if (!SupportsAbsoluteCallRelocation(architecture))
            {
                foreach (SassInstruction instruction in instructions)
                {
                    SassRelocation? absoluteCall = instruction.Relocations
                        .FirstOrDefault(
                            relocation => relocation.Type == 0x4b);
                    if (absoluteCall != null &&
                        labels.ContainsKey(absoluteCall.Symbol))
                    {
                        int operand = instruction.Operands.IndexOf("0x0");
                        if (operand >= 0)
                            instruction.Operands[operand] = absoluteCall.Symbol;
                    }
                    instruction.Relocations.RemoveAll(
                        relocation => relocation.Type == 0x4b);
                }
            }

            byte[] data = new byte[instructions.Count * 16];
            for (int i = 0; i < instructions.Count; i++)
            {
                SassInstruction instruction = instructions[i];
                ulong low;
                ulong high;
                try
                {
                    (low, high) = EncodeInstruction(
                        instruction, labels);
                }
                catch (Exception ex)
                {
                    throw new InvalidDataException(
                        $"{Path.GetFileName(sourcePath)}:{instruction.Line}: " +
                        $"{instruction.Source}: {ex.Message}", ex);
                }

                BitConverter.GetBytes(low).CopyTo(data, i * 16);
                BitConverter.GetBytes(high).CopyTo(data, i * 16 + 8);
            }

            string sectionName = section.Name.Length > 0
                ? section.Name
                : ".text";
            // CUDA metadata is associated by the section suffix.  A bare
            // .text section cannot carry .nv.info.<suffix>; give the
            // implicit section a stable function suffix.
            if (sectionName == ".text")
                sectionName = ".text._start";
            int sectionHeaderIndex = imageSections.Count;
            imageSections.Add(new CubinImageSection
            {
                Name = sectionName,
                Alignment = Math.Max(1, section.Alignment),
                Data = data,
            });

            var relocations = new List<CubinImageRelocation>();
            foreach (SassInstruction instruction in instructions)
            {
                foreach (SassRelocation relocation in instruction.Relocations)
                {
                    long addend = relocation.Addend;
                    if (relocation.AddendSymbol.Length > 0 &&
                        labels.TryGetValue(
                            relocation.AddendSymbol, out int symbolAddend))
                        addend = symbolAddend;
                    relocations.Add(new CubinImageRelocation
                    {
                        Offset = (ulong)instruction.Address,
                        Symbol = relocation.Symbol,
                        Type = relocation.Type,
                        Addend = addend,
                        IsRel = relocation.IsRel,
                    });
                }
            }
            foreach (var group in relocations
                .GroupBy(relocation => relocation.IsRel)
                .OrderByDescending(group => group.Key))
            {
                bool isRel = group.Key;
                imageSections.Add(new CubinImageSection
                {
                    Name = (isRel ? ".rel" : ".rela") + sectionName,
                    Type = isRel ? 9u : 4u,
                    Flags = 0x40,
                    Alignment = 8,
                    Info = (uint)(sectionHeaderIndex + 1),
                    EntrySize = isRel ? 16u : 24u,
                    Relocations = group.ToList(),
                });
            }

            var indirectBranches = instructions
                .Where(i => i.Mnemonic.StartsWith(
                    "BRX", StringComparison.OrdinalIgnoreCase))
                .ToList();
            if (indirectBranches.Count > 0)
            {
                var targets = labels.Values
                    .Distinct()
                    .OrderBy(value => value)
                    .Select(value => (uint)value)
                    .ToList();
                if (targets.Count == 0)
                {
                    targets = Enumerable.Range(0, instructions.Count)
                        .Select(index => (uint)(index * 16))
                        .ToList();
                }

                var info = new List<byte>();
                foreach (SassInstruction branch in indirectBranches)
                {
                    List<uint> branchTargets = targets;
                    if (branch.BranchTargets.Count > 0)
                    {
                        var annotated = branch.BranchTargets
                            .Select(name => labels.TryGetValue(name, out int value)
                                ? (uint)value
                                : (uint?)null)
                            .Where(value => value.HasValue)
                            .Select(value => value!.Value)
                            .Distinct()
                            .OrderBy(value => value)
                            .ToList();
                        if (annotated.Count > 0)
                            branchTargets = annotated;
                    }

                    int payloadWords = 3 + branchTargets.Count;
                    int payloadSize = payloadWords * 4;
                    info.Add(4); // format: byte payload
                    info.Add(52); // EIATTR_BRANCH_TARGETS
                    info.Add((byte)payloadSize);
                    info.Add((byte)(payloadSize >> 8));
                    AddInfoWord(info, (uint)branch.Address); // BRX offset
                    AddInfoWord(info, 0); // reserved
                    AddInfoWord(info, (uint)branchTargets.Count);
                    foreach (uint target in branchTargets)
                        AddInfoWord(info, target);
                }

                string suffix = sectionName.StartsWith(
                        ".text.", StringComparison.Ordinal)
                    ? sectionName[6..]
                    : sectionName.TrimStart('.');
                imageSections.Add(new CubinImageSection
                {
                    Name = ".nv.info." + suffix,
                    Type = 0x70000000,
                    Flags = 0x40,
                    Alignment = 4,
                    Data = info.ToArray(),
                });
            }

            int registerCount = section.RegisterCount ??
                RegisterCount(instructions);
            foreach (var pair in labels)
            {
                if (pair.Key.StartsWith(".L", StringComparison.Ordinal) ||
                    pair.Key.StartsWith(".text.", StringComparison.Ordinal))
                    continue;
                imageSymbols.Add(new CubinImageSymbol
                {
                    Name = pair.Key,
                    Section = sectionHeaderIndex,
                    Value = (ulong)pair.Value,
                    Size = (ulong)(data.Length - pair.Value),
                    RegisterCount = pair.Key.StartsWith(
                            ".text.", StringComparison.Ordinal)
                        ? null
                        : registerCount,
                });
            }
            if (!labels.Keys.Any(k => !k.StartsWith(".L", StringComparison.Ordinal)))
            {
                imageSymbols.Add(new CubinImageSymbol
                {
                    Name = SafeFunctionName(sectionName),
                    Section = sectionHeaderIndex,
                    Value = 0,
                    Size = (ulong)data.Length,
                    RegisterCount = registerCount,
                });
            }

        }

        if (imageSections.Count == 0)
            throw new InvalidDataException("The source contains no instructions.");

        uint architectureNumber = ArchitectureNumber(architecture);
        CubinWriter.Write(
            outputPath, architectureNumber, architecture,
            imageSections, imageSymbols, effectiveToolkit);
    }

    internal static (ulong Low, ulong High) EncodeInstruction(
        SassInstruction instruction, IReadOnlyDictionary<string, int> labels)
    {
        if (TryExpandLegacyRemap(instruction, out SassInstruction? expanded))
            return EncodeInstruction(expanded, labels);

        // KeplerAs's BRK template plus the default PT predicate bits.  The
        // shared CC descriptor does not preserve this raw control encoding.
        if (InstructionDecoder.IsPreVoltaTable() &&
            InstructionDecoder.Table.Architecture is "sm_35" or "sm_37" &&
            instruction.Guard == null && instruction.Operands.Count == 0)
        {
            string op = instruction.Mnemonic.Split('.')[0]
                .ToUpperInvariant();
            if (op == "BRK")
                return (0x1a000000001c003cUL, 0);
            if (op == "CONT")
                return (0x1a800000001c003cUL, 0);
            if (op == "KIL")
                return (0x19800000001c003cUL, 0);
            if (op == "LONGJMP")
                return (0x18800000001c003cUL, 0);
        }

        string mnemonic = instruction.Mnemonic;
        IsaClass? bestClass = null;
        ulong bestLow = 0;
        ulong bestHigh = 0;
        var problems = new List<string>();

        foreach (IsaClass candidate in InstructionDecoder.Table.Classes)
        {
            if (!MnemonicMatches(candidate.Mnemonic, mnemonic, out string suffix))
                continue;
            try
            {
                (ulong low, ulong high) = EncodeCandidate(
                    candidate, suffix, instruction, labels);
                if (!RoundTrips(candidate, low, high, instruction))
                    continue;
                bestClass = candidate;
                bestLow = low;
                bestHigh = high;
                break;
            }
            catch (Exception ex)
            {
                problems.Add($"{candidate.Name}: {ex.Message}");
            }
        }

        if (bestClass == null)
        {
            string detail = problems.Count == 0
                ? ""
                : " (" + string.Join("; ", problems.Take(5)) +
                    (problems.Count > 5 ? "; ..." : "") + ")";
            throw new InvalidDataException(
                $"cannot find an encoding class for '{mnemonic}'" +
                (instruction.Line > 0 ? $" at line {instruction.Line}" : "") +
                detail);
        }
        return (bestLow, bestHigh);
    }

    private static bool TryExpandLegacyRemap(
        SassInstruction instruction, out SassInstruction expanded)
    {
        expanded = null!;
        string op = instruction.Mnemonic.Split('.')[0]
            .ToUpperInvariant();
        if (op != "ISUB" || instruction.Operands.Count != 3)
            return false;

        // CUDA 10.2 defines ISUB as a REMAP to IADD with a negated source.
        // ISUB1 is the form with the immediate/constant listed second.
        string first = instruction.Operands[1];
        string second = instruction.Operands[2];
        bool firstRegister = IsRegisterLike(first);
        bool secondRegister = IsRegisterLike(second);
        var operands = new List<string> { instruction.Operands[0] };
        if (firstRegister || !secondRegister)
        {
            operands.Add(first);
            operands.Add(Negate(second));
        }
        else
        {
            operands.Add(Negate(second));
            operands.Add(first);
        }

        expanded = new SassInstruction
        {
            Mnemonic = "IADD",
            Guard = instruction.Guard,
            Source = instruction.Source,
            Line = instruction.Line,
        };
        expanded.Operands.AddRange(operands);
        expanded.BranchTargets.AddRange(instruction.BranchTargets);
        expanded.Relocations.AddRange(instruction.Relocations);
        return true;
    }

    private static bool IsRegisterLike(string operand)
    {
        string value = operand.Trim().TrimStart('-', '~', '|');
        return value.StartsWith("R", StringComparison.OrdinalIgnoreCase) &&
            (value.Length == 1 || char.IsDigit(value[1]) ||
                value.Equals("RZ", StringComparison.OrdinalIgnoreCase));
    }

    private static string Negate(string operand)
    {
        string value = operand.Trim();
        return value.StartsWith('-') ? value[1..] : "-" + value;
    }

    private static bool MnemonicMatches(
        string classMnemonic, string sourceMnemonic, out string suffix)
    {
        suffix = "";
        if (!sourceMnemonic.StartsWith(
                classMnemonic, StringComparison.OrdinalIgnoreCase))
            return false;
        if (sourceMnemonic.Length == classMnemonic.Length)
            return true;
        if (sourceMnemonic[classMnemonic.Length] != '.')
            return false;
        suffix = sourceMnemonic[(classMnemonic.Length + 1)..];
        return true;
    }

    private static (ulong Low, ulong High) EncodeCandidate(
        IsaClass cls, string suffix, SassInstruction instruction,
        IReadOnlyDictionary<string, int> labels)
    {
        ulong low = 0;
        ulong high = 0;
        var remaining = new List<string>(instruction.Operands);
        var suffixTokens = suffix.Split('.',
            StringSplitOptions.RemoveEmptyEntries).ToList();

        // Fixed/opcode/control fields start from their encoded defaults.
        foreach (IsaField field in cls.Fields)
        {
            // Variable fields are populated by their FORMAT descriptors.
            // Do not initialize them here: sibling classes can contribute
            // duplicate field names/segments, and a late variable field would
            // otherwise overwrite an already encoded opcode or fixed value.
            if (InstructionDecoder.IsPreVoltaTable() &&
                field.Kind == "derived" &&
                TryDerivedValue(field, out long derivedValue))
            {
                SetBits(ref low, ref high, field.Segs, derivedValue);
                continue;
            }
            if (field.Kind is not ("fixed" or "opcode"))
                continue;
            long value = field.Kind == "fixed"
                ? field.Value ?? 0
                : cls.Opcode;
            SetBits(ref low, ref high, field.Segs, value);
        }

        if (InstructionDecoder.IsPreVoltaTable() &&
            (cls.Mnemonic is "BRA" or "BRX" or "BRK" or "CONT" or
                "KIL" or "LONGJMP") &&
            cls.FindField("CCC_1") != null)
        {
            // Pre-Volta branches and control operations default to an
            // unconditional condition code.
            SetField(cls, ref low, ref high, "CCC_1", 15);
        }

        // Guard predicate.
        if (cls.Format?.Guard != null)
        {
            ParseGuard(instruction.Guard, out int predicate, out bool negated,
                out bool hasGuard);
            string guardType = cls.Format.Guard.Type;
            long value = hasGuard
                ? EnumValue(guardType, predicate.ToString(
                    CultureInfo.InvariantCulture))
                : (DefaultEnumValue(guardType, cls.Format.Guard.Default) ?? 7);
            SetField(cls, ref low, ref high, cls.Format.Guard.Field, value);
            IsaField? notField = InstructionDecoder.FindLinked(
                cls, cls.Format.Guard.Field, "not");
            if (notField != null)
                SetBits(ref low, ref high, notField.Segs, negated ? 1 : 0);
        }

        // FORMAT modifiers (LDG.E, .SYS, .128, .SAT, ...).
        foreach (IsaMod mod in cls.Format?.Mods ?? new List<IsaMod>())
        {
            if (mod.Field.Length == 0)
                continue;
            string modField = FindModFieldName(cls, mod);
            string? matchedModifierText = null;
            long? value = null;
            if (InstructionDecoder.Table.Enums.TryGetValue(
                    mod.Type, out var values))
            {
                int bestStart = int.MaxValue;
                string[]? bestTokens = null;
                foreach (var pair in values)
                {
                    string[] valueTokens = pair.Value.Split('.');
                    int start = FindTokenSequence(
                        suffixTokens, valueTokens);
                    if (start >= 0 &&
                        (start < bestStart ||
                         (start == bestStart &&
                          valueTokens.Length > (bestTokens?.Length ?? 0))))
                    {
                        bestStart = start;
                        bestTokens = valueTokens;
                        value = long.Parse(pair.Key,
                            CultureInfo.InvariantCulture);
                    }
                }
                if (bestTokens != null)
                {
                    matchedModifierText = string.Join('.', bestTokens);
                    suffixTokens.RemoveRange(bestStart, bestTokens.Length);
                    if (InstructionDecoder.IsPreVoltaTable() &&
                        mod.Type is "LDInteger" or "CInteger")
                    {
                        long? legacySize = LegacySizeValue(matchedModifierText);
                        if (legacySize.HasValue)
                            value = legacySize;
                        if (mod.Type == "LDInteger" &&
                            matchedModifierText == "U.128")
                            value = 7;
                    }
                }
            }
            if (!value.HasValue &&
                InstructionDecoder.IsPreVoltaTable() &&
                mod.Type is "LDInteger" or "CInteger")
            {
                for (int token = 0; token < suffixTokens.Count; token++)
                {
                    long? legacySize = LegacySizeValue(suffixTokens[token]);
                    if (!legacySize.HasValue)
                        continue;
                    value = legacySize;
                    matchedModifierText = suffixTokens[token];
                    suffixTokens.RemoveAt(token);
                    break;
                }
            }
            if (!value.HasValue)
                value = DefaultValue(mod.Type, mod.Args) ?? 0;
            if (InstructionDecoder.IsPreVoltaTable() &&
                mod.Type == "LDSInteger")
            {
                string sizeText = matchedModifierText ??
                    mod.Args?.Trim().Trim('"') ?? "";
                if (LegacyLdsSizeValue(sizeText) is long ldsSize)
                    value = ldsSize;
            }
            if (InstructionDecoder.IsPreVoltaTable() &&
                mod.Type == "SQInteger")
            {
                string sizeText = matchedModifierText ??
                    mod.Args?.Trim().Trim('"') ?? "";
                if (LegacySqIntegerValue(sizeText) is long sqSize)
                    value = sqSize;
            }
            if (InstructionDecoder.IsPreVoltaTable() &&
                ((cls.Mnemonic == "I2F" && mod.Field == "srcfmt") ||
                 (cls.Mnemonic == "F2I" && mod.Field == "dstfmt")))
            {
                value = LegacyIntSizeRaw(value.Value);
            }
            if (InstructionDecoder.IsPreVoltaTable() &&
                mod.Type == "Integer32" && value.Value is 4 or 5)
            {
                // Legacy tables store signedness as a one-bit IsSigned
                // selector: 0 = U32, 1 = S32.
                long signedRaw = value.Value == 5 ? 1 : 0;
                IsaField? signedField = cls.FindField(mod.Field) ??
                    cls.Fields.FirstOrDefault(field =>
                        field.Width == 1 &&
                        (field.Logical is "us32" or "Integer32"));
                if (signedField != null)
                {
                    SetBits(ref low, ref high, signedField.Segs, signedRaw);
                    continue;
                }
            }
            SetField(cls, ref low, ref high, modField, value.Value);
        }

        // NOP's source FORMAT keeps /S outside the parsed modifier list.
        if (InstructionDecoder.IsPreVoltaTable() && cls.Mnemonic == "NOP")
        {
            if (cls.FindField("CCC_2") != null)
                SetField(cls, ref low, ref high, "CCC_2", 15);
            int sIndex = suffixTokens.FindIndex(token =>
                token.Equals("S", StringComparison.OrdinalIgnoreCase));
            if (sIndex >= 0)
            {
                suffixTokens.RemoveAt(sIndex);
                SetField(cls, ref low, ref high, "Sync", 1);
            }
        }

        // A suffix token left over after all FORMAT modifiers means this
        // class does not actually describe the requested instruction.  This
        // is important for the legacy LDG aliases: LDG.CT and LDG.CG share a
        // mnemonic but encode different hardware operations.
        if (InstructionDecoder.IsPreVoltaTable() && suffixTokens.Count > 0)
            throw new InvalidDataException(
                "unconsumed modifier " + suffixTokens[0]);

        var operands = cls.Format?.Operands ?? new List<IsaOperand>();
        foreach (IsaOperand operand in operands)
        {
            switch (operand.Kind)
            {
                case "field":
                {
                    IsaDescriptor? desc = operand.Desc;
                    if (desc == null)
                        throw new InvalidDataException("missing descriptor");
                    if (desc.Type == "DESC")
                    {
                        if (remaining.Count == 0)
                            throw new InvalidDataException("missing descriptor operand");
                        EncodeDescriptor(cls, ref low, ref high,
                            remaining[0], remaining);
                        continue;
                    }

                    if (remaining.Count > 0 &&
                        TryParseField(cls, desc, remaining[0],
                            ref low, ref high, labels, instruction.Address))
                    {
                        remaining.RemoveAt(0);
                        continue;
                    }

                    // Legacy texture formats mark operands such as
                    // Register(RZ):Rb as optional.  Their FORMAT default is a
                    // register name rather than a value handled by
                    // DefaultValue, so synthesize the omitted register.
                    if (TryDefaultRegister(desc, out long defaultRegister))
                    {
                        SetField(cls, ref low, ref high, desc.Field,
                            defaultRegister);
                        continue;
                    }

                    if (DefaultValue(desc.Type, desc.Args) is long dflt)
                    {
                        SetField(cls, ref low, ref high, desc.Field, dflt);
                        continue;
                    }
                    throw new InvalidDataException(
                        $"missing {desc.Type} operand");
                }
                case "A":
                {
                    if (remaining.Count == 0)
                        throw new InvalidDataException("missing address operand");
                    EncodeAddress(cls, ref low, ref high,
                        operand.Parts ?? new List<IsaDescriptor>(),
                        remaining[0]);
                    remaining.RemoveAt(0);
                    break;
                }
                case "C":
                case "CX":
                {
                    if (remaining.Count == 0)
                        throw new InvalidDataException("missing C operand");
                    string constantText = remaining[0];
                    string markers = ConsumeMarkers(operand.Markers,
                        ref constantText);
                    EncodeConstant(cls, ref low, ref high, operand,
                        constantText);
                    ApplyMarkers(cls,
                        operand.Label ?? operand.Addr?.Field ?? "",
                        markers, ref low, ref high);
                    remaining.RemoveAt(0);
                    break;
                }
            }
        }

        // BRX prints a signed target offset after the register.  That offset
        // is the sImm field even though the legacy FORMAT omits it from the
        // operand list.
        if (remaining.Count > 0 &&
            instruction.Mnemonic.StartsWith(
                "BRX", StringComparison.OrdinalIgnoreCase) &&
            remaining.All(value => value.StartsWith('`') ||
                value.StartsWith("0x", StringComparison.OrdinalIgnoreCase) ||
                value.StartsWith("-0x", StringComparison.OrdinalIgnoreCase) ||
                (value.StartsWith('(') && value.EndsWith(')'))))
        {
            if (remaining.Count == 1 &&
                TryInteger(remaining[0], out long branchOffset) &&
                cls.FindField("sImm") != null)
            {
                SetField(cls, ref low, ref high, "sImm", branchOffset);
            }
            remaining.Clear();
        }

        // RET prints its target symbol as an annotation, but the register is
        // the encoded operand.  Relative returns still encode the target in
        // the sImm field.
        if (remaining.Count > 0 &&
            instruction.Mnemonic.StartsWith("RET", StringComparison.OrdinalIgnoreCase) &&
            remaining.All(value => value.StartsWith('`') ||
                value.StartsWith("0x", StringComparison.OrdinalIgnoreCase) ||
                value.StartsWith("-0x", StringComparison.OrdinalIgnoreCase) ||
                (value.StartsWith('(') && value.EndsWith(')'))))
        {
            if (remaining.Count == 1 &&
                TryAddress(remaining[0], labels, out long returnTarget))
            {
                IsaField? branchField = cls.FindField("sImm");
                if (branchField != null)
                {
                    long scale = Math.Max(1, branchField.Scale);
                    bool absoluteRet = cls.Format?.Mods.Any(
                        mod => mod.Type == "ABSONLY_ret") == true;
                    SetBits(ref low, ref high, branchField.Segs,
                        absoluteRet
                            ? returnTarget / scale
                            : (returnTarget - instruction.Address - 16) / scale);
                }
            }
            remaining.Clear();
        }

        if (remaining.Count > 0)
            throw new InvalidDataException(
                "unconsumed operand " + remaining[0]);

        if (Environment.GetEnvironmentVariable("SASS_DEBUG_DADD") == "1" &&
            cls.Name == "Imm DADD")
            Console.Error.WriteLine($"daddword {low:x16}");
        // The SM50 BRK/CONT/KIL/LONGJMP FORMAT lists CC as an operand whose
        // default is the non-printing CC/1 value.  Restore the unconditional
        // TRUE/15 encoding after the operand loop has run.
        if (InstructionDecoder.IsPreVoltaTable() &&
            (cls.Mnemonic is "BRK" or "CONT" or "KIL" or "LONGJMP") &&
            cls.FindField("CCC_1") != null)
            SetField(cls, ref low, ref high, "CCC_1", 15);

        if (InstructionDecoder.IsPreVoltaTable() && cls.Mnemonic == "NOP" &&
            cls.FindField("CCC_2") != null)
            SetField(cls, ref low, ref high, "CCC_2", 15);

        if (instruction.LegacyFloatModifiers is string legacyModifiers)
        {
            foreach (IsaField field in cls.Fields)
            {
                string? logical = field.Logical;
                if (logical == null)
                    continue;
                if (logical.Equals("UnaryNeg",
                        StringComparison.OrdinalIgnoreCase) &&
                    legacyModifiers.Contains(
                        "NEG", StringComparison.OrdinalIgnoreCase))
                {
                    SetField(cls, ref low, ref high, field.Name, 1);
                }
                else if (logical.Equals("UnaryAbs",
                        StringComparison.OrdinalIgnoreCase) &&
                    legacyModifiers.Contains(
                        "ABS", StringComparison.OrdinalIgnoreCase))
                {
                    SetField(cls, ref low, ref high, field.Name, 1);
                }
            }
        }

        return (low, high);
    }

    private static bool RoundTrips(
        IsaClass cls, ulong low, ulong high, SassInstruction instruction)
    {
        DecodedInstruction decoded = InstructionDecoder.Decode(
            low, InstructionDecoder.IsPreVoltaTable() ? 0 : high,
            (ulong)instruction.Address);
        if (decoded.ClassName != cls.Name)
        {
            // Some CUDA 10.2 legacy classes are source-level aliases whose
            // text differs from the hardware TEX/TLD form they encode.
            if (IsEquivalentLegacyDecode(cls.Name, decoded.ClassName ?? ""))
                return true;
            return false;
        }

        // Branch targets are resolved after layout; the numeric target in the
        // decoded text cannot match the source label text.  Still compare the
        // mnemonic/modifier prefix so CALL.REL is not silently assembled as
        // CALL.ABS (the absolute class appears first in the table).
        if (IsBranchInstruction(instruction))
        {
            string expectedPrefix =
                (instruction.Guard != null ? instruction.Guard + " " : "") +
                instruction.Mnemonic;
            return decoded.Text.StartsWith(
                expectedPrefix, StringComparison.OrdinalIgnoreCase);
        }

        string expected = NormalizeText(instruction.Guard, instruction.Mnemonic,
            instruction.Operands);
        string actual = NormalizeText(null, decoded.Text, null);
        if (instruction.LegacyFloatModifiers != null)
        {
            foreach (string suffix in new[]
            {
                ".NEG.ABS", ".ABS.NEG", ".NEG", ".ABS",
            })
            {
                expected = expected.Replace(
                    suffix, "", StringComparison.OrdinalIgnoreCase);
                actual = actual.Replace(
                    suffix, "", StringComparison.OrdinalIgnoreCase);
            }
        }
        if (cls.Name == "CCTLT")
        {
            expected = expected.Replace(".ivall", "", StringComparison.Ordinal);
            actual = actual.Replace(".ivall", "", StringComparison.Ordinal);
        }
        if (cls.Format?.Mods.Any(mod => mod.Type == "SQInteger") == true)
        {
            expected = NormalizeSqIntegerText(expected);
            actual = NormalizeSqIntegerText(actual);
        }
        return expected == actual;
    }

    private static string NormalizeSqIntegerText(string text) =>
        text.Replace(".u32", "", StringComparison.Ordinal)
            .Replace(".32", "", StringComparison.Ordinal)
            .Replace(".u64", ".64", StringComparison.Ordinal)
            .Replace(".u128", ".128", StringComparison.Ordinal);

    private static bool IsEquivalentLegacyDecode(
        string encodedClass, string decodedClass) =>
        (encodedClass, decodedClass) switch
        {
            ("LDG CT", "TEX") => true,
            ("LDG CG CS", "TLD") => true,
            ("CCTLT", "TEX") => true,
            ("CCTLT", "CCTLT") => true,
            _ => false,
        };

    private static string NormalizeText(
        string? guard, string mnemonic, List<string>? operands)
    {
        string text = guard != null ? guard + " " : "";
        text += mnemonic;
        if (operands != null && operands.Count > 0)
            text += " " + string.Join(",", operands.Select(
                o => InstructionDecoder.IsPreVoltaTable()
                    ? o.Replace(" ", "", StringComparison.Ordinal)
                        .Replace(".reuse", "", StringComparison.OrdinalIgnoreCase)
                    : o.Replace(" ", "", StringComparison.Ordinal)));
        return text.Replace(" ", "", StringComparison.Ordinal)
            .Replace("\t", "", StringComparison.Ordinal)
            .ToLowerInvariant();
    }

    private static string FindModFieldName(IsaClass cls, IsaMod mod)
    {
        IsaField? exact = cls.FindField(mod.Field);
        if (!InstructionDecoder.IsPreVoltaTable())
            return mod.Field;

        // Legacy tables sometimes name the FORMAT selector differently from
        // the ENCODING field.  Prefer the logical name before the legacy
        // texture aliases below.
        if (exact == null)
        {
            IsaField? byLogical = cls.Fields.FirstOrDefault(field =>
                string.Equals(field.Logical, mod.Field,
                    StringComparison.OrdinalIgnoreCase));
            if (byLogical != null)
                return byLogical.Name;
        }
        IsaField? byType = cls.Fields.FirstOrDefault(field =>
            string.Equals(field.Logical, mod.Type,
                StringComparison.OrdinalIgnoreCase));
        if (byType != null &&
            (exact == null || byType.Width < exact.Width))
            return byType.Name;
        if (exact != null)
            return exact.Name;

        // Legacy texture FORMAT descriptors use short field names (cop, e,
        // phase, sync) while the encoding names the logical fields
        // LDGcop2/CL, LDG_E/ParamA, TPhase/Phase and S/Sync.
        string? logicalName = mod.Field switch
        {
            "cop" => "LDGcop2",
            "e" => "LDG_E",
            "phase" => "TPhase",
            "sync" => "S",
            "clamp" => "CL",
            "hib" => "HILO",
            _ => null,
        };
        if (logicalName == null)
            return mod.Field;
        IsaField? logical = cls.Fields.FirstOrDefault(field =>
            string.Equals(field.Logical, logicalName,
                StringComparison.OrdinalIgnoreCase) ||
            (logicalName == "LDGcop2" &&
             field.Logical?.StartsWith("LDGcop",
                 StringComparison.OrdinalIgnoreCase) == true));
        return logical?.Name ?? mod.Field;
    }

    private static int FindTokenSequence(
        List<string> tokens, string[] sequence)
    {
        for (int start = 0; start + sequence.Length <= tokens.Count; start++)
        {
            bool match = true;
            for (int i = 0; i < sequence.Length; i++)
            {
                if (!tokens[start + i].Equals(
                        sequence[i], StringComparison.OrdinalIgnoreCase))
                {
                    match = false;
                    break;
                }
            }
            if (match)
                return start;
        }
        return -1;
    }

    private static bool IsBranchInstruction(SassInstruction instruction)
    {
        string mnemonic = instruction.Mnemonic.Split('.')[0]
            .ToUpperInvariant();
        return mnemonic is "BRA" or "BRX" or "JMP" or "CAL" or "CALL" or
            "RET" or "BSSY" or "WARPSYNC" or "LEPC" or "SSY" or "PBK" or
            "PCNT" or "PLONGJMP";
    }

    private static void ParseGuard(
        string? text, out int predicate, out bool negated, out bool hasGuard)
    {
        predicate = 7; // PT
        negated = false;
        hasGuard = false;
        if (string.IsNullOrEmpty(text) || !text.StartsWith('@'))
            return;
        hasGuard = true;
        string name = text[1..];
        if (name.StartsWith('!'))
        {
            negated = true;
            name = name[1..];
        }
        predicate = name switch
        {
            "PT" or "UPT" => 7,
            _ when name.StartsWith("UP", StringComparison.OrdinalIgnoreCase) &&
                int.TryParse(name[2..], out int up) => up,
            _ when name.StartsWith('P') &&
                int.TryParse(name[1..], out int p) => p,
            _ => 7,
        };
    }

    private static bool TryParseField(
        IsaClass cls, IsaDescriptor desc, string text,
        ref ulong low, ref ulong high,
        IReadOnlyDictionary<string, int> labels, int address)
    {
        string valueText = text.Trim();
        string appliedMarkers = ConsumeMarkers(desc, ref valueText);

        switch (desc.Type)
        {
            case "Register":
            case "RegisterFAU":
            case "NonZeroRegister":
            case "NonZeroRegisterFAU":
            case "ZeroRegister":
            {
                string registerText = valueText;
                string modifiers = "";
                int dot = registerText.IndexOf('.');
                if (dot > 0)
                {
                    modifiers = registerText[(dot + 1)..];
                    registerText = registerText[..dot];
                }
                if (!TryRegister(registerText, out long register))
                    return false;
                SetField(cls, ref low, ref high, desc.Field, register);
                string[] modifierTokens = modifiers.Split('.',
                    StringSplitOptions.RemoveEmptyEntries);
                ApplyDescriptorModifiers(cls, desc, modifierTokens,
                    ref low, ref high);
                ApplyRegisterModifiers(cls, desc, modifiers, ref low, ref high);
                ApplyMarkers(cls, desc.Field, appliedMarkers,
                    ref low, ref high);
                return true;
            }
            case "UniformRegister":
            case "NonZeroUniformRegister":
            case "ZeroUniformRegister":
            {
                string uniformText = valueText;
                string uniformModifiers = "";
                int uniformDot = uniformText.IndexOf('.');
                if (uniformDot > 0)
                {
                    uniformModifiers = uniformText[(uniformDot + 1)..];
                    uniformText = uniformText[..uniformDot];
                }
                if (!TryUniformRegister(uniformText, out long uniform))
                    return false;
                SetField(cls, ref low, ref high, desc.Field, uniform);
                ApplyDescriptorModifiers(cls, desc,
                    uniformModifiers.Split('.',
                        StringSplitOptions.RemoveEmptyEntries),
                    ref low, ref high);
                ApplyMarkers(cls, desc.Field, appliedMarkers,
                    ref low, ref high);
                return true;
            }
            case "Predicate":
                if (!TryPredicate(valueText, false, out long pred))
                    return false;
                SetField(cls, ref low, ref high, desc.Field, pred);
                ApplyMarkers(cls, desc.Field, appliedMarkers,
                    ref low, ref high);
                return true;
            case "UniformPredicate":
                if (!TryPredicate(valueText, true, out long upred))
                    return false;
                SetField(cls, ref low, ref high, desc.Field, upred);
                ApplyMarkers(cls, desc.Field, appliedMarkers,
                    ref low, ref high);
                return true;
            case "PR":
                if (!valueText.Equals("PR", StringComparison.OrdinalIgnoreCase))
                    return false;
                SetField(cls, ref low, ref high, desc.Field, 0);
                return true;
            case "SpecialRegister":
                if (!TrySpecialRegister(valueText, out long sreg))
                    return false;
                SetField(cls, ref low, ref high, desc.Field, sreg);
                return true;
            case "BD":
                if (!TryBarrier(valueText, out long barrier))
                    return false;
                SetField(cls, ref low, ref high, desc.Field, barrier);
                return true;
            case "RSImm":
                if (!TryAddress(valueText, labels, out long target))
                    return false;
                int scale = cls.FindField(desc.Field)?.Scale ?? 4;
                int branchBase = InstructionDecoder.IsPreVoltaTable() ? 8 : 16;
                long relative = (target - address - branchBase) / scale;
                SetField(cls, ref low, ref high, desc.Field, relative);
                return true;
            case "F16Imm":
            case "F32Imm":
            case "F64Imm":
                if (!TryFloatBits(desc.Type, valueText, out long fbits))
                    return false;
                IsaField? immediateField = cls.FindField(desc.Field);
                if (immediateField is { Width: > 0 and < 32 } &&
                    desc.Type == "F32Imm")
                    fbits >>= 32 - immediateField.Width;
                if (desc.Type == "F64Imm" && immediateField != null)
                {
                    double d = BitConverter.Int64BitsToDouble(fbits);
                    if (immediateField.Width == 16)
                    {
                        IsaField? formatField = cls.FindField("ofmt");
                        long format = formatField == null
                            ? 0
                            : GetBits(low, high, formatField.Segs);
                        fbits = format == 2
                            ? (ushort)(BitConverter.SingleToInt32Bits(
                                (float)d) >> 16)
                            : BitConverter.HalfToUInt16Bits((Half)d);
                    }
                    else if (immediateField.Width <= 32)
                    {
                        // MUFU (and similar converted immediates) store an
                        // F32 bit pattern; DSETP/DADD-style F64 immediates
                        // store the high 32 bits of the double.
                        IsaField? mufuField = cls.FindField("mufuop");
                        long mufuOp = mufuField == null
                            ? -1
                            : GetBits(low, high, mufuField.Segs);
                        bool f64Operand = !immediateField.Expr.Contains(
                                "convertFloatType", StringComparison.Ordinal) ||
                            mufuOp is 6 or 7;
                        fbits = f64Operand
                            ? (long)(BitConverter.DoubleToInt64Bits(d) >> 32)
                            : BitConverter.SingleToInt32Bits((float)d);
                    }
                }
                if (desc.Type == "F64Imm" && immediateField != null)
                {
                    if (immediateField.Width is > 32 and < 64)
                        fbits >>= 64 - immediateField.Width;
                    else if (immediateField.Width is > 16 and <= 32)
                        fbits >>= 32 - immediateField.Width;
                }
                SetField(cls, ref low, ref high,
                    immediateField?.Name ?? desc.Field, fbits);
                return true;
            case "BITSET":
            {
                if (!valueText.StartsWith('{') || !valueText.EndsWith('}'))
                    return false;
                long bits = 0;
                foreach (string member in valueText[1..^1].Split(','))
                {
                    if (!int.TryParse(member.Trim(), NumberStyles.None,
                            CultureInfo.InvariantCulture, out int bit) ||
                        bit < 0 || bit >= 31)
                        return false;
                    bits |= 1L << bit;
                }
                SetField(cls, ref low, ref high, desc.Field, bits);
                return true;
            }
            case "UImm":
            case "SImm":
            case "CInteger":
            case "QInteger":
            case "AInteger":
                if (TryInteger(valueText, out long integer))
                {
                    SetField(cls, ref low, ref high, desc.Field, integer);
                    return true;
                }
                // Pre-SM90 absolute calls are encoded directly instead of
                // carrying an 0x4b relocation, so resolve local labels.
                if (TryAddress(valueText, labels, out long targetAddress))
                {
                    long addressScale = Math.Max(1,
                        cls.FindField(desc.Field)?.Scale ?? 1);
                    SetField(cls, ref low, ref high, desc.Field,
                        targetAddress / addressScale);
                    return true;
                }
                // Absolute call targets are section-relative symbols; an
                // ET_EXEC cubin can encode the section-relative target 0.
                if (valueText.StartsWith('`') ||
                    (valueText.StartsWith('(') && valueText.EndsWith(')')))
                {
                    SetField(cls, ref low, ref high, desc.Field, 0);
                    return true;
                }
                return false;
            default:
                if (InstructionDecoder.Table.TextToValue.TryGetValue(
                        desc.Type, out var enumMap) &&
                    enumMap.TryGetValue(valueText, out int enumValue))
                {
                    SetField(cls, ref low, ref high, desc.Field, enumValue);
                    return true;
                }
                return false;
        }
    }

    private static void ApplyDescriptorModifiers(
        IsaClass cls, IsaDescriptor desc, string[] tokens,
        ref ulong low, ref ulong high)
    {
        foreach (string token in tokens)
        {
            foreach (IsaMod mod in desc.Mods)
            {
                if (mod.Field.Length == 0)
                    continue;
                if (InstructionDecoder.Table.TextToValue.TryGetValue(
                        mod.Type, out var map) &&
                    map.TryGetValue(token, out int value))
                {
                    SetField(cls, ref low, ref high, mod.Field, value);
                    break;
                }
            }
        }
    }

    private static void ApplyRegisterModifiers(
        IsaClass cls, IsaDescriptor desc, string modifiers,
        ref ulong low, ref ulong high)
    {
        foreach (string token in modifiers.Split('.',
            StringSplitOptions.RemoveEmptyEntries))
        {
            if (token.Equals("reuse", StringComparison.OrdinalIgnoreCase))
            {
                IsaMod? mod = desc.Mods.FirstOrDefault(
                    m => m.Type == "REUSE");
                if (mod != null)
                    SetField(cls, ref low, ref high, mod.Field, 1);
            }
            else if (int.TryParse(token, out int size) &&
                size is 8 or 16 or 32 or 64 or 128)
            {
                SetSizeModifier(cls, size, ref low, ref high);
                if (size == 64)
                    SetPairSize(cls, ref low, ref high, 64);
            }
            else if (token.StartsWith('H') || token.StartsWith("H1", StringComparison.OrdinalIgnoreCase))
            {
                IsaMod? mod = desc.Mods.FirstOrDefault(m => m.Type == "ISWZ");
                if (mod != null)
                {
                    long? value = EnumValueByText(mod.Type, token);
                    if (value.HasValue)
                        SetField(cls, ref low, ref high, mod.Field, value.Value);
                }
            }
        }
    }


    private static string ConsumeMarkers(IsaDescriptor desc, ref string text)
    {
        string applied = "";
        foreach (string marker in desc.Markers
            .OrderByDescending(m => m.Length))
        {
            if (text.StartsWith(marker, StringComparison.Ordinal))
            {
                applied += marker;
                text = text[marker.Length..];
            }
        }
        int firstBar = text.IndexOf('|');
        if (firstBar >= 0)
        {
            int secondBar = text.IndexOf('|', firstBar + 1);
            if (secondBar > firstBar)
            {
                applied += "||";
                text = text.Remove(secondBar, 1).Remove(firstBar, 1);
            }
        }
        if (applied.Contains('|') && text.EndsWith('|'))
            text = text[..^1].Trim();
        text = text.Trim();
        return applied;
    }

    private static string ConsumeMarkers(List<string> markers, ref string text)
    {
        string applied = "";
        foreach (string marker in markers.OrderByDescending(m => m.Length))
        {
            if (text.StartsWith(marker, StringComparison.Ordinal))
            {
                applied += marker;
                text = text[marker.Length..];
            }
        }
        if (applied.Contains('|') && text.EndsWith('|'))
            text = text[..^1].Trim();
        int firstBar = text.IndexOf('|');
        if (firstBar >= 0)
        {
            int secondBar = text.IndexOf('|', firstBar + 1);
            if (secondBar > firstBar)
            {
                applied += "||";
                text = text.Remove(secondBar, 1).Remove(firstBar, 1);
            }
        }
        if (applied.Contains('|') && text.EndsWith('|'))
            text = text[..^1].Trim();
        text = text.Trim();
        return applied;
    }

    private static void ApplyMarkers(
        IsaClass cls, string baseField, string markers,
        ref ulong low, ref ulong high)
    {
        if (markers.Contains('!'))
        {
            IsaField? field = InstructionDecoder.FindLinked(
                cls, baseField, "not");
            if (field != null)
                SetBits(ref low, ref high, field.Segs, 1);
        }
        if (markers.Contains('~'))
        {
            IsaField? field = InstructionDecoder.FindLinked(
                cls, baseField, "invert");
            if (field != null)
                SetBits(ref low, ref high, field.Segs, 1);
        }
        if (markers.Contains('-'))
        {
            // IADD/ISUB-style classes encode source negation through the
            // two-bit PSign selector instead of individual nA/nB fields.
            // 2 selects -Ra, 1 selects -Rb.  Do not also set the overlapping
            // nAB/nB aliases, or the other source is negated as well.
            IsaField? pSign = cls.Fields.FirstOrDefault(
                field => field.Name == "PSign" && field.Width >= 2);
            bool usedPSign = pSign != null && pSign.Width >= 2 &&
                cls.Mnemonic == "IADD" &&
                (baseField == "Ra" || baseField == "Rb" ||
                    baseField == "srcConst");
            if (usedPSign)
            {
                long current = GetBits(low, high, pSign!.Segs);
                long sign = baseField == "Ra" ? 2 : 1;
                SetBits(ref low, ref high, pSign.Segs, current | sign);
            }
            else
            {
                IsaField? field = InstructionDecoder.FindLinked(
                    cls, baseField, "negate");
                if (field != null)
                    SetBits(ref low, ref high, field.Segs, 1);
            }
        }
        if (markers.Contains('|'))
        {
            IsaField? field = InstructionDecoder.FindLinked(
                cls, baseField, "absolute");
            if (field != null)
                SetBits(ref low, ref high, field.Segs, 1);
        }
    }

    private static void SetSizeModifier(
        IsaClass cls, int bits, ref ulong low, ref ulong high)
    {
        foreach (IsaMod mod in cls.Format?.Mods ?? new List<IsaMod>())
        {
            if (mod.Field.Length == 0)
                continue;
            if (!mod.Type.Contains("SZ", StringComparison.Ordinal) &&
                !mod.Type.Contains("SIZE", StringComparison.Ordinal) &&
                !mod.Field.Equals("sz", StringComparison.OrdinalIgnoreCase))
                continue;
            if (InstructionDecoder.Table.TextToValue.TryGetValue(
                    mod.Type, out var map) &&
                map.TryGetValue(bits.ToString(CultureInfo.InvariantCulture),
                    out int value))
                SetField(cls, ref low, ref high, mod.Field, value);
        }
    }

    private static void SetPairSize(
        IsaClass cls, ref ulong low, ref ulong high, int bits)
    {
        foreach (IsaField field in cls.Fields)
        {
            if (field.Logical.StartsWith(
                    bits == 64 ? "input_reg_sz_64" : "input_reg_sz_32",
                    StringComparison.Ordinal))
                SetBits(ref low, ref high, field.Segs, 1);
        }
    }

    private static void EncodeAddress(
        IsaClass cls, ref ulong low, ref ulong high,
        List<IsaDescriptor> parts, string text)
    {
        string address = text.Trim();
        if (address.StartsWith('[') && address.EndsWith(']'))
            address = address[1..^1];

        var registers = Regex.Matches(address, @"\b(UR\d+|URZ|R\d+|RZ)\b",
            RegexOptions.IgnoreCase);
        var offsetMatch = Regex.Match(address, @"([+-])(0x[0-9a-f]+|\d+)",
            RegexOptions.IgnoreCase);
        if (!offsetMatch.Success)
            offsetMatch = Regex.Match(address, @"^(0x[0-9a-f]+|\d+)$",
                RegexOptions.IgnoreCase);
        bool is64 = address.Contains(".64", StringComparison.OrdinalIgnoreCase);
        bool used64 = false;
        bool hasUniformPart = false;
        bool matchedUniformPart = false;
        bool uniformHasDefault = false;
        string[] addressModifiers = Regex.Matches(
                address, @"\.([A-Za-z][A-Za-z0-9_]*)")
            .Select(match => match.Groups[1].Value).ToArray();

        foreach (IsaDescriptor part in parts)
        {
            if (part.Type.Contains("Register", StringComparison.Ordinal) &&
                addressModifiers.Length > 0)
                ApplyDescriptorModifiers(cls, part, addressModifiers,
                    ref low, ref high);
            if (part.Type.Contains("UniformRegister", StringComparison.Ordinal))
            {
                hasUniformPart = true;
                bool matched = false;
                foreach (Match match in registers)
                {
                    if (match.Value.StartsWith("UR", StringComparison.OrdinalIgnoreCase) &&
                        TryUniformRegister(match.Value, out long uniform))
                    {
                        SetField(cls, ref low, ref high, part.Field, uniform);
                        matched = true;
                        matchedUniformPart = true;
                        break;
                    }
                }
                if (!matched && DefaultValue(part.Type, part.Args) is long defaultUniform)
                {
                    SetField(cls, ref low, ref high, part.Field, defaultUniform);
                    uniformHasDefault = true;
                }
            }
            else if (part.Type.Contains("Register", StringComparison.Ordinal))
            {
                bool matched = false;
                foreach (Match match in registers)
                {
                    if (!match.Value.StartsWith("UR", StringComparison.OrdinalIgnoreCase) &&
                        TryRegister(match.Value, out long register))
                    {
                        SetField(cls, ref low, ref high, part.Field, register);
                        used64 |= is64;
                        matched = true;
                        break;
                    }
                }
                if (!matched && registers.Count == 0 &&
                    part.Type.StartsWith("NonZeroRegister",
                        StringComparison.Ordinal))
                {
                    // An immediate-only shared address is encoded with the
                    // implicit RZ base.  The field width truncates 0xff to the
                    // architecture's RZ code (63 on SM30, 255 elsewhere).
                    SetField(cls, ref low, ref high, part.Field, 255);
                }
                else if (!matched && TryAddressDefaultRegister(
                        part, out long defaultRegister))
                    SetField(cls, ref low, ref high, part.Field, defaultRegister);
                else if (!matched &&
                    DefaultValue(part.Type, part.Args) is long defaultValue)
                    SetField(cls, ref low, ref high, part.Field, defaultValue);
            }
            else if (part.Type is "UImm" or "SImm")
            {
                if (offsetMatch.Success &&
                    TryInteger(
                        offsetMatch.Groups[1].Value + offsetMatch.Groups[2].Value,
                        out long offset))
                {
                    IsaField? field = cls.FindField(part.Field);
                    long scale = Math.Max(1, field?.Scale ?? 1);
                    if (cls.Mnemonic == "CCTL" && part.Field == "sImm")
                        scale *= 4;
                    SetField(cls, ref low, ref high, part.Field, offset / scale);
                }
            }
        }
        if (used64)
            SetPairSize(cls, ref low, ref high, 64);
        if (hasUniformPart && !matchedUniformPart && !uniformHasDefault)
            SetField(cls, ref low, ref high, "memdesc", 1);
    }

    private static void EncodeConstant(
        IsaClass cls, ref ulong low, ref ulong high,
        IsaOperand operand, string text)
    {
        string clean = Regex.Replace(text.Trim(), @"\s+", "");
        bool constantNegated = clean.StartsWith('-');
        if (constantNegated)
        {
            clean = clean[1..];
            IsaField? pSign = cls.Fields.FirstOrDefault(field => field.Name == "PSign" && field.Width >= 2);
            if (pSign != null && pSign.Width >= 2)
            {
                long current = GetBits(low, high, pSign.Segs);
                SetBits(ref low, ref high, pSign.Segs, current | 1);
            }
        }
        Match cxMatch = Regex.Match(clean,
            @"^cx\[(UR\d+|URZ)\]\[(.+)\]$",
            RegexOptions.IgnoreCase);
        if (cxMatch.Success)
        {
            if (!TryUniformRegister(cxMatch.Groups[1].Value,
                    out long uniformRegister))
                throw new InvalidDataException(
                    $"invalid CX uniform register '{text}'");
            if (operand.Ureg != null)
                SetField(cls, ref low, ref high, operand.Ureg.Field,
                    uniformRegister);
            if (operand.Offset != null)
            {
                if (TryInteger(cxMatch.Groups[2].Value, out long cxOffset))
                {
                    IsaField? field = cls.FindField(operand.Offset.Field);
                    long scale = Math.Max(1, field?.Scale ?? 1);
                    SetField(cls, ref low, ref high, operand.Offset.Field,
                        cxOffset / scale);
                }
            }
            return;
        }

        Match match = Regex.Match(clean,
            @"^c\[([^\]]+)\]\[([^\]]+)\](?:\.([A-Za-z0-9_]+))?$",
            RegexOptions.IgnoreCase);
        if (!match.Success)
            throw new InvalidDataException($"invalid constant operand '{text}'");
        if (!TryInteger(match.Groups[1].Value, out long bank))
            throw new InvalidDataException($"invalid constant bank '{text}'");

        if (operand.Bank != null)
            SetField(cls, ref low, ref high, operand.Bank.Field, bank);

        string addressText = match.Groups[2].Value.Trim();
        string constantModifiers = match.Groups[3].Value;
        Match registerMatch = Regex.Match(addressText,
            @"^\s*(UR\d+|URZ|R\d+|RZ)\b", RegexOptions.IgnoreCase);
        if (registerMatch.Success && operand.Addr != null &&
            operand.Addr.Type.Contains("Register", StringComparison.Ordinal))
        {
            if (!TryRegister(registerMatch.Value, out long register) &&
                !TryUniformRegister(registerMatch.Value, out register))
                throw new InvalidDataException(
                    $"invalid constant address register '{text}'");
            SetField(cls, ref low, ref high, operand.Addr.Field, register);

            Match offsetMatch = Regex.Match(addressText,
                @"([+-])(0x[0-9a-f]+|\d+)", RegexOptions.IgnoreCase);
            if (offsetMatch.Success && TryInteger(
                    offsetMatch.Groups[1].Value + offsetMatch.Groups[2].Value,
                    out long offset))
            {
                IsaDescriptor? offsetDesc = operand.Offset ??
                    operand.Parts?.FirstOrDefault(part =>
                        part.Field != operand.Addr.Field &&
                        part.Type is "SImm" or "UImm");
                if (offsetDesc != null)
                {
                    IsaField? field = cls.FindField(offsetDesc.Field);
                    long scale = Math.Max(1, field?.Scale ?? 1);
                    SetField(cls, ref low, ref high, offsetDesc.Field,
                        offset / scale);
                }
            }
            ApplyConstantModifiers(cls, operand, constantModifiers,
                ref low, ref high);
            return;
        }

        if (!TryInteger(addressText, out long address))
            throw new InvalidDataException($"invalid constant address '{text}'");

        if (operand.Addr != null &&
            operand.Addr.Type.Contains("Register", StringComparison.Ordinal))
        {
            // c[bank][imm] with a register-address form uses RZ and puts the
            // immediate in the address-offset field.
            SetField(cls, ref low, ref high, operand.Addr.Field, 255);
            IsaDescriptor? offsetDesc = operand.Offset ??
                operand.Parts?.FirstOrDefault(part =>
                    part.Field != operand.Addr.Field &&
                    part.Type is "SImm" or "UImm");
            if (offsetDesc != null)
            {
                IsaField? field = cls.FindField(offsetDesc.Field);
                long scale = Math.Max(1, field?.Scale ?? 1);
                SetField(cls, ref low, ref high, offsetDesc.Field,
                    address / scale);
            }
        }
        else if (operand.Addr != null)
        {
            IsaField? field = cls.FindField(operand.Addr.Field);
            long scale = ConstantAddressScale(field);
            SetField(cls, ref low, ref high, operand.Addr.Field,
                address / scale);
        }
        if (operand.Ureg != null)
            SetField(cls, ref low, ref high, operand.Ureg.Field, bank);
        if (operand.Offset != null &&
            (operand.Addr == null ||
             !operand.Addr.Type.Contains("Register", StringComparison.Ordinal)))
        {
            IsaField? field = cls.FindField(operand.Offset.Field);
            long scale = ConstantAddressScale(field);
            SetField(cls, ref low, ref high, operand.Offset.Field,
                address / scale);
        }
        ApplyConstantModifiers(cls, operand, constantModifiers,
            ref low, ref high);
    }

    private static long ConstantAddressScale(IsaField? field)
    {
        long scale = Math.Max(1, field?.Scale ?? 1);
        if (field != null && scale == 1 &&
            field.Expr.Contains(
                "ConstBankAddress2", StringComparison.Ordinal))
            scale = 4;
        return scale;
    }

    private static void ApplyConstantModifiers(
        IsaClass cls, IsaOperand operand, string modifiers,
        ref ulong low, ref ulong high)
    {
        if (modifiers.Length == 0)
            return;
        string[] tokens = modifiers.Split('.',
            StringSplitOptions.RemoveEmptyEntries);
        if (operand.Addr != null)
            ApplyDescriptorModifiers(cls, operand.Addr, tokens,
                ref low, ref high);
        if (operand.Parts != null)
            foreach (IsaDescriptor part in operand.Parts)
                ApplyDescriptorModifiers(cls, part, tokens,
                    ref low, ref high);
    }

    private static void EncodeDescriptor(
        IsaClass cls, ref ulong low, ref ulong high,
        string text, List<string> remaining)
    {
        Match match = Regex.Match(text.Trim(),
            @"^desc\[(UR\d+|URZ)\]\[([^\]]+)\]$", RegexOptions.IgnoreCase);
        if (!match.Success)
            throw new InvalidDataException($"invalid descriptor operand '{text}'");
        if (!TryUniformRegister(match.Groups[1].Value, out long uniform))
            throw new InvalidDataException("invalid descriptor register");

        string? uniformField = new[] { "Ra_URd", "Ra_URc", "Ra_URb" }
            .FirstOrDefault(name =>
                cls.FindField(name) != null ||
                cls.TableFields.Any(tableField =>
                    tableField.TableArgs != null &&
                    Array.IndexOf(tableField.TableArgs, name) >= 0));
        if (uniformField != null)
            SetField(cls, ref low, ref high, uniformField, uniform);
        IsaField? descriptorFlag = cls.FindField("e_desc");
        if (descriptorFlag != null)
            SetBits(ref low, ref high, descriptorFlag.Segs, 1);

        string address = match.Groups[2].Value;
        bool is64 = address.Contains(".64", StringComparison.OrdinalIgnoreCase);
        Match reg = Regex.Match(address, @"R(\d+|Z)", RegexOptions.IgnoreCase);
        if (reg.Success && TryRegister(reg.Value, out long register))
            SetField(cls, ref low, ref high, "Ra", register);
        Match off = Regex.Match(address,
            @"([+-])(0x[0-9a-f]+|\d+)", RegexOptions.IgnoreCase);
        if (off.Success && TryInteger(
                off.Groups[1].Value + off.Groups[2].Value, out long offset))
        {
            IsaField? field = cls.FindField("Ra_offset");
            long scale = Math.Max(1, field?.Scale ?? 1);
            SetField(cls, ref low, ref high, "Ra_offset", offset / scale);
        }
        if (is64)
            SetPairSize(cls, ref low, ref high, 64);
        remaining.RemoveAt(0);
    }

    private static bool TryAddressDefaultRegister(
        IsaDescriptor desc, out long value)
    {
        value = 0;
        if (desc.Args is null)
            return false;
        string text = desc.Args.Trim().Trim('"');
        if (InstructionDecoder.Table.EnumTextToValue.TryGetValue(
                desc.Type, out var map) &&
            map.TryGetValue(text, out int enumValue))
        {
            value = enumValue;
            return true;
        }
        return TryRegister(text, out value);
    }

    private static bool TryDefaultRegister(
        IsaDescriptor desc, out long value)
    {
        value = 0;
        if (desc.Args is null || !InstructionDecoder.IsPreVoltaTable())
            return false;
        if (desc.Type is not ("Register" or "RegisterFAU" or
                "NonZeroRegister" or "ZeroRegister"))
            return false;
        string text = desc.Args.Trim().Trim('"');
        return TryRegister(text, out value);
    }

    private static bool TryRegister(string text, out long value)
    {
        text = text.Trim();
        if (text.Equals("RZ", StringComparison.OrdinalIgnoreCase))
        {
            value = 255;
            return true;
        }
        if (text.Length > 1 && (text[0] == 'R' || text[0] == 'r') &&
            int.TryParse(text[1..], NumberStyles.None,
                CultureInfo.InvariantCulture, out int register) &&
            register is >= 0 and < 256)
        {
            value = register;
            return true;
        }
        value = 0;
        return false;
    }

    private static bool TryUniformRegister(string text, out long value)
    {
        text = text.Trim();
        if (text.Equals("URZ", StringComparison.OrdinalIgnoreCase))
        {
            if (InstructionDecoder.Table.TextToValue.TryGetValue(
                    "UniformRegister", out var map) &&
                map.TryGetValue("URZ", out int zero))
            {
                value = zero;
                return true;
            }
            value = 63;
            return true;
        }
        if (text.StartsWith("UR", StringComparison.OrdinalIgnoreCase) &&
            int.TryParse(text[2..], NumberStyles.None,
                CultureInfo.InvariantCulture, out int register) &&
            register is >= 0 and < 64)
        {
            value = register;
            return true;
        }
        value = 0;
        return false;
    }

    private static bool TryPredicate(string text, bool uniform, out long value)
    {
        text = text.Trim();
        string prefix = uniform ? "UP" : "P";
        if (text.Equals(uniform ? "UPT" : "PT",
                StringComparison.OrdinalIgnoreCase))
        {
            value = 7;
            return true;
        }
        if (text.StartsWith(prefix, StringComparison.OrdinalIgnoreCase) &&
            int.TryParse(text[prefix.Length..], NumberStyles.None,
                CultureInfo.InvariantCulture, out int predicate) &&
            predicate is >= 0 and < 7)
        {
            value = predicate;
            return true;
        }
        value = 0;
        return false;
    }

    private static bool TrySpecialRegister(string text, out long value)
    {
        foreach (var pair in InstructionDecoder.Table.SpecialRegisters)
        {
            if (pair.Value.Equals(text, StringComparison.OrdinalIgnoreCase))
            {
                value = long.Parse(pair.Key, CultureInfo.InvariantCulture);
                return true;
            }
        }
        value = 0;
        return false;
    }

    private static bool TryBarrier(string text, out long value)
    {
        text = text.Trim();
        if (text.Length > 1 &&
            (text[0] == 'B' || text[0] == 'b') &&
            int.TryParse(text[1..], NumberStyles.None,
                CultureInfo.InvariantCulture, out int barrier) &&
            barrier is >= 0 and < 16)
        {
            value = barrier;
            return true;
        }
        value = 0;
        return false;
    }

    private static bool TryInteger(string text, out long value)
    {
        value = 0;
        text = text.Trim();
        bool negative = text.StartsWith('-');
        if (negative || text.StartsWith('+'))
            text = text[1..];
        if (text.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
        {
            if (!ulong.TryParse(text[2..], NumberStyles.HexNumber,
                    CultureInfo.InvariantCulture, out ulong unsigned))
                return false;
            value = unchecked((long)unsigned);
        }
        else if (!long.TryParse(text, NumberStyles.Integer,
                CultureInfo.InvariantCulture, out value))
            return false;
        if (negative)
            value = -value;
        return true;
    }

    private static bool TryAddress(
        string text, IReadOnlyDictionary<string, int> labels, out long value)
    {
        text = text.Trim().Trim('`', '(', ')');
        if (labels.TryGetValue(text, out int label))
        {
            value = label;
            return true;
        }
        return TryInteger(text, out value);
    }

    private static bool TryFloatBits(string type, string text, out long value)
    {
        if (text.StartsWith("0x", StringComparison.OrdinalIgnoreCase) &&
            long.TryParse(text[2..], NumberStyles.HexNumber,
                CultureInfo.InvariantCulture, out value))
            return true;

        string special = text.Trim().ToUpperInvariant();
        if (special is "+INF" or "INF" or "-INF" or "NAN" or "+NAN" or "-NAN"
            or "QNAN" or "+QNAN" or "-QNAN")
        {
            bool negativeSpecial = special.StartsWith("-", StringComparison.Ordinal);
            bool nanSpecial = special.Contains("NAN", StringComparison.Ordinal);
            if (nanSpecial)
            {
                value = type switch
                {
                    "F16Imm" => negativeSpecial ? 0xfe00 : 0x7e00,
                    "F32Imm" => negativeSpecial
                        ? unchecked((long)0xffc00000)
                        : 0x7fc00000,
                    _ => negativeSpecial
                        ? unchecked((long)0xfff8000000000000UL)
                        : 0x7ff8000000000000,
                };
                return true;
            }
            double specialValue = special == "-INF"
                ? double.NegativeInfinity
                : double.PositiveInfinity;
            value = type switch
            {
                "F16Imm" => BitConverter.HalfToUInt16Bits((Half)specialValue),
                "F32Imm" => BitConverter.SingleToInt32Bits((float)specialValue),
                _ => BitConverter.DoubleToInt64Bits(specialValue),
            };
            return true;
        }

        if (!double.TryParse(text, NumberStyles.Float,
                CultureInfo.InvariantCulture, out double number))
        {
            value = 0;
            return false;
        }
        value = type switch
        {
            "F16Imm" => BitConverter.HalfToUInt16Bits((Half)number),
            "F32Imm" => BitConverter.SingleToInt32Bits((float)number),
            _ => BitConverter.DoubleToInt64Bits(number),
        };
        return true;
    }

    private static void SetField(
        IsaClass cls, ref ulong low, ref ulong high, string name, long value)
    {
        IsaField? field = cls.FindField(name);
        if (field == null)
        {
            SetTableFlag(cls, ref low, ref high, name, value);
            return;
        }

        if (field.Kind == "table")
        {
            foreach (IsaField tableField in cls.TableFields)
            {
                if (tableField.TableName == null ||
                    tableField.TableArgs == null)
                    continue;
                int index = Array.IndexOf(tableField.TableArgs, name);
                if (index < 0)
                    continue;
                if (InstructionDecoder.Table.Tables.TryGetValue(
                        tableField.TableName, out var table))
                {
                    foreach (var pair in table)
                    {
                        if (pair.Value.Length > index &&
                            pair.Value[index] == value &&
                            int.TryParse(pair.Key, NumberStyles.Integer,
                                CultureInfo.InvariantCulture, out int raw))
                        {
                            SetBits(ref low, ref high, tableField.Segs, raw);
                            return;
                        }
                    }
                }
            }
        }

        SetBits(ref low, ref high, field.Segs, value);
    }

    private static bool SetTableFlag(
        IsaClass cls, ref ulong low, ref ulong high, string name, long value)
    {
        foreach (IsaField tableField in cls.TableFields)
        {
            if (tableField.TableName == null || tableField.TableArgs == null)
                continue;
            int index = Array.IndexOf(tableField.TableArgs, name);
            if (index < 0)
                continue;
            if (!InstructionDecoder.Table.Tables.TryGetValue(
                    tableField.TableName, out var table))
                continue;

            long currentRaw = GetBits(low, high, tableField.Segs);
            int[] desired = table.TryGetValue(
                currentRaw.ToString(CultureInfo.InvariantCulture),
                out int[]? current)
                ? (int[])current.Clone()
                : new int[tableField.TableArgs.Length];
            if (index >= desired.Length)
                continue;
            desired[index] = (int)value;

            if (TryFindTableEntry(table, desired, out int raw))
            {
                SetBits(ref low, ref high, tableField.Segs, raw);
                return true;
            }

            // No exact combination exists.  Preserve the requested flag and
            // prefer zeroes for the remaining arguments.
            foreach (var pair in table)
            {
                if (pair.Value.Length > index &&
                    pair.Value[index] == value &&
                    int.TryParse(pair.Key, NumberStyles.Integer,
                        CultureInfo.InvariantCulture, out raw))
                {
                    SetBits(ref low, ref high, tableField.Segs, raw);
                    return true;
                }
            }
        }
        return false;
    }

    private static bool TryFindTableEntry(
        Dictionary<string, int[]> table, int[] desired, out int raw)
    {
        foreach (var pair in table)
        {
            if (pair.Value.Length != desired.Length)
                continue;
            bool match = true;
            for (int i = 0; i < desired.Length; i++)
            {
                if (pair.Value[i] != desired[i])
                {
                    match = false;
                    break;
                }
            }
            if (match && int.TryParse(pair.Key, NumberStyles.Integer,
                    CultureInfo.InvariantCulture, out raw))
                return true;
        }
        raw = 0;
        return false;
    }

    private static long GetBits(
        ulong low, ulong high, int[][] segs)
    {
        int total = segs.Sum(seg => seg[0] - seg[1] + 1);
        int consumed = 0;
        long value = 0;
        foreach (int[] seg in segs)
        {
            int width = seg[0] - seg[1] + 1;
            int shift = total - consumed - width;
            for (int bit = seg[1]; bit <= seg[0]; bit++)
            {
                ulong source = bit < 64
                    ? (low >> bit) & 1
                    : (high >> (bit - 64)) & 1;
                value |= (long)source << (shift + (bit - seg[1]));
            }
            consumed += width;
        }
        return value;
    }

    private static void SetBits(
        ref ulong low, ref ulong high, int[][] segs, long value)
    {
        ulong raw = unchecked((ulong)value);
        int total = segs.Sum(s => s[0] - s[1] + 1);
        int consumed = 0;
        foreach (int[] seg in segs)
        {
            int msb = seg[0];
            int lsb = seg[1];
            int width = msb - lsb + 1;
            int valueShift = Math.Max(0, total - consumed - width);
            ulong part = width >= 64
                ? raw
                : (raw >> valueShift) & ((1UL << width) - 1);
            consumed += width;

            // A field can cross the 64-bit half boundary (for example a
            // branch sImm spanning bits 81:34), so place it bit by bit.
            for (int bit = 0; bit < width; bit++)
            {
                int absolute = lsb + bit;
                ulong mask = 1UL << (absolute < 64 ? absolute : absolute - 64);
                if (absolute < 64)
                {
                    low = (low & ~mask) | (((part >> bit) & 1) << absolute);
                }
                else
                {
                    high = (high & ~mask) |
                        (((part >> bit) & 1) << (absolute - 64));
                }
            }
        }
    }

    private static long LegacyIntSizeRaw(long value)
    {
        if (InstructionDecoder.Table.Tables.TryGetValue(
                "IntSize", out var table))
        {
            foreach (var pair in table)
            {
                if (pair.Value.Length > 0 && pair.Value[0] == value &&
                    long.TryParse(pair.Key, NumberStyles.Integer,
                        CultureInfo.InvariantCulture, out long raw))
                    return raw;
            }
        }
        return value switch
        {
            0 => 0,
            1 => 4,
            2 => 1,
            3 => 5,
            4 => 2,
            5 => 6,
            6 => 3,
            7 => 7,
            _ => value,
        };
    }

    private static long? LegacySqIntegerValue(string text)
    {
        if (InstructionDecoder.Table.Architecture != "sm_30" &&
            InstructionDecoder.Table.TextToValue.TryGetValue(
                "SQInteger", out var map) &&
            map.TryGetValue(text, out int value))
            return value;
        return text switch
        {
        "8" or "U8" => 0,
        "16" or "U16" => 2,
        "32" or "U32" => 4,
        "64" or "U64" => 5,
        "128" or "U128" => 6,
        "S8" => 12,
        "S16" => 14,
        "S32" => 18,
        "S64" => 19,
        "S128" => 20,
        "F16" => 21,
        "F32" or "F32.FTZ.RN" => 11,
        "F64" => 23,
        _ => null,
        };
    }

    private static long? LegacyLdsSizeValue(string text) => text switch
    {
        "8" or "U8" => 0,
        "U.8" or "U.U8" => 1,
        "S8" => 2,
        "U.S8" => 3,
        "16" or "U16" => 4,
        "U.16" or "U.U16" => 5,
        "S16" => 6,
        "U.S16" => 7,
        "32" => 8,
        "U.32" => 9,
        "64" => 10,
        "U.64" => 11,
        "128" => 12,
        "U.128" => 13,
        "INVALIDSIZE7" => 14,
        "U.INVALIDSIZE7" => 15,
        _ => null,
    };

    private static long? LegacySizeValue(string text) => text switch
    {
        "8" or "U8" => 0,
        "S8" => 1,
        "16" or "U16" => 2,
        "S16" => 3,
        "32" => 4,
        "64" => 5,
        "128" => 6,
        _ => null,
    };

    private static bool TryDerivedValue(IsaField field, out long value)
    {
        value = 0;
        if (string.IsNullOrEmpty(field.Linked) ||
            string.IsNullOrEmpty(field.Mod) ||
            field.Mod is "not")
            return false;

        string mod = field.Mod.Trim().Trim('"');
        if (TryRegister(mod, out value))
            return true;

        string linked = field.Linked.Trim().TrimStart('`');
        if (InstructionDecoder.Table.TextToValue.TryGetValue(
                linked, out var map) &&
            map.TryGetValue(mod, out int enumValue))
        {
            value = enumValue;
            return true;
        }
        return TryInteger(mod, out value);
    }

    private static long? DefaultEnumValue(string type, string text)
    {
        if (InstructionDecoder.Table.TextToValue.TryGetValue(type, out var map) &&
            map.TryGetValue(text.Trim('"'), out int value))
            return value;
        return null;
    }

    private static long? DefaultValue(string type, string? args)
    {
        if (args == null)
            return null;
        string text = args.Trim();
        bool quoted = text.Length >= 2 && text[0] == '"' && text[^1] == '"';
        int slash = text.LastIndexOf('/');
        if (slash >= 0)
        {
            text = text[(slash + 1)..];
        }
        else if (!quoted &&
            !type.StartsWith("Integer", StringComparison.Ordinal) &&
            !type.StartsWith("Float", StringComparison.Ordinal) &&
            type is not ("Predicate" or "UniformPredicate" or "CC"))
        {
            // A non-quoted argument without '/' is usually a width, not a
            // default.  Format selectors such as Integer32 still carry their
            // default enum text directly.
            return null;
        }
        text = text.Trim('"');
        if (InstructionDecoder.IsPreVoltaTable() &&
            type is "LDInteger" or "CInteger")
        {
            long? legacySize = LegacySizeValue(text);
            if (legacySize.HasValue)
                return legacySize;
        }
        if (InstructionDecoder.Table.TextToValue.TryGetValue(type, out var map) &&
            map.TryGetValue(text, out int enumValue))
            return enumValue;
        if (long.TryParse(text, NumberStyles.Integer,
                CultureInfo.InvariantCulture, out long number))
            return number;
        if (text.StartsWith("0x", StringComparison.OrdinalIgnoreCase) &&
            long.TryParse(text[2..], NumberStyles.HexNumber,
                CultureInfo.InvariantCulture, out number))
            return number;
        return null;
    }

    private static long EnumValue(string type, string valueText)
    {
        if (InstructionDecoder.Table.TextToValue.TryGetValue(type, out var map) &&
            map.TryGetValue(valueText, out int value))
            return value;
        return long.Parse(valueText, CultureInfo.InvariantCulture);
    }

    private static long? EnumValueByText(string type, string text)
    {
        if (InstructionDecoder.Table.TextToValue.TryGetValue(type, out var map) &&
            map.TryGetValue(text, out int value))
            return value;
        return null;
    }

    private static void AddInfoWord(List<byte> data, uint value)
    {
        data.Add((byte)value);
        data.Add((byte)(value >> 8));
        data.Add((byte)(value >> 16));
        data.Add((byte)(value >> 24));
    }

    private static int RegisterCount(
        IEnumerable<SassInstruction> instructions)
    {
        int maximum = -1;
        foreach (SassInstruction instruction in instructions)
        {
            foreach (Match match in Regex.Matches(
                instruction.Source, @"\bR(\d+)\b"))
            {
                if (int.TryParse(match.Groups[1].Value,
                        NumberStyles.None, CultureInfo.InvariantCulture,
                        out int register))
                    maximum = Math.Max(maximum, register);
            }
        }
        return maximum < 0 ? 0 : maximum + 1;
    }

    private static bool IsPreVoltaArchitecture(string architecture)
    {
        string value = architecture.Trim().ToLowerInvariant();
        if (value.StartsWith("sm_", StringComparison.Ordinal))
            value = value[3..];
        else if (value.StartsWith("sm", StringComparison.Ordinal))
            value = value[2..];
        return int.TryParse(value, out int number) && number is >= 30 and < 70;
    }

    private static bool SupportsAbsoluteCallRelocation(string architecture)
    {
        try
        {
            return ArchitectureNumber(architecture) >= 90;
        }
        catch (Exception)
        {
            // Preserve the previous behaviour for architectures whose table
            // name does not start with a numeric SM version.
            return true;
        }
    }

    private static uint ArchitectureNumber(string architecture)
    {
        int start = architecture.StartsWith("sm_", StringComparison.OrdinalIgnoreCase)
            ? 3 : 0;
        int end = start;
        while (end < architecture.Length && char.IsDigit(architecture[end]))
            end++;
        return uint.Parse(architecture[start..end], CultureInfo.InvariantCulture);
    }

    private static string SafeFunctionName(string section) =>
        section.StartsWith(".text.", StringComparison.Ordinal)
            ? section[6..]
            : "_start";

    public static void Convert(
        string sourcePath, string sourceArchitecture, string targetArchitecture,
        string outputPath, string? irOutputPath = null, string? toolkit = null)
    {
        SassProgram program = SassParser.Parse(sourcePath);
        ConvertProgram(program, sourceArchitecture, targetArchitecture, outputPath, irOutputPath, toolkit);
    }

    public static void ConvertProgram(
        SassProgram program, string sourceArchitecture, string targetArchitecture,
        string outputPath, string? irOutputPath = null, string? toolkit = null)
    {
        if (ArchitectureNames.TryCanonicalizeLegacySm101(
                targetArchitecture, out string canonical))
        {
            Console.Error.WriteLine(
                "WARNING: SM101 was renamed to SM110 in PTX ISA 9.0; " +
                $"emitting an {canonical} cubin.");
            targetArchitecture = canonical;
        }
        var transformed = TransformForTarget(program, sourceArchitecture, targetArchitecture);
        string? effectiveToolkit = InstructionDecoder.NormalizeToolkit(toolkit);
        if (IsPreVoltaArchitecture(targetArchitecture))
        {
            PreVoltaDecoder.ResolveTablePath(targetArchitecture, effectiveToolkit);
            effectiveToolkit ??= "102";
            PreVoltaAssembler.Assemble(
                transformed, targetArchitecture, outputPath, irOutputPath,
                effectiveToolkit);
            return;
        }

        InstructionDecoder.UseArchitecture(targetArchitecture, effectiveToolkit);
        effectiveToolkit ??= InstructionDecoder.Table.Toolkit;

        if (irOutputPath != null)
            SassIRSerializer.Write(
                irOutputPath, transformed, targetArchitecture, effectiveToolkit);

        AssembleProgram(transformed, targetArchitecture, outputPath, effectiveToolkit);
    }

    private static SassProgram TransformForTarget(SassProgram program, string sourceArch, string targetArch)
    {
        var result = new SassProgram();
        result.Architecture = targetArch;
        result.Toolkit = program.Toolkit;
        foreach (var section in program.Sections)
        {
            var newSection = new SassSection
            {
                Name = section.Name,
                Alignment = section.Alignment,
                RegisterCount = section.RegisterCount
            };
            foreach (var node in section.Nodes)
            {
                if (node is SassLabel label)
                {
                    newSection.Nodes.Add(new SassLabel { Name = label.Name, Address = label.Address });
                }
                else if (node is SassInstruction instr)
                {
                    newSection.Nodes.Add(TransformInstruction(instr, sourceArch, targetArch));
                }
            }
            result.Sections.Add(newSection);
        }
        return result;
    }

    private static SassInstruction TransformInstruction(SassInstruction instr, string sourceArch, string targetArch)
    {
        string mnemonic = instr.Mnemonic;
        string baseMnemonic = mnemonic.Split('.')[0].ToUpperInvariant();

        // Basic emulation for some SM120-specific instructions when targeting SM86
        // Many of these don't have direct equivalents; try best-effort mappings
        switch (baseMnemonic)
        {
            case "OMMA":
            case "QMMA":
            case "HMMA":
                // Tensor core MMA instructions differ between architectures
                // For now, try to pass through - may need adjustment
                break;
            case "UBLKCP":
            case "UBLKPF":
            case "UBLKRED":
                // Asynchronous bulk copy ops - may not exist on older archs
                // Try to replace with NOP or similar if not supported
                break;
            case "FENCE":
            case "SYNCS":
            case "ACQBULK":
            case "ACQSHMINIT":
                // Memory fence variants
                break;
            case "CGAERRBAR":
            case "UCGABARARV":
            case "UCGABARGET":
            case "UCGABARSET":
            case "UCGABARWAIT":
            case "UCGABAR_ARV":
            case "UCGABAR_WAIT":
                // Cluster/group barrier ops - not present on SM86
                // Replace with NOP as best effort
                var nopInstr = new SassInstruction
                {
                    Mnemonic = "NOP",
                    Guard = instr.Guard,
                    Source = instr.Source,
                    Line = instr.Line,
                    Address = instr.Address
                };
                return nopInstr;
            case "REDAS":
            case "REDG":
            case "REDS":
            case "STAS":
            case "STSM":
                // Async/shared variants
                break;
            case "ENDCOLLECTIVE":
            case "PREEXIT":
            case "ELECT":
            case "ULEPC":
            case "UMEMSETS":
            case "USETMAXREG":
            case "USETSHMSZ":
            case "UGETNEXTWORKID":
            case "UVIRTCOUNT":
                // Special/utility ops
                break;
            case "CS2UR":
            case "LDCU":
            case "LDGMC":
            case "MOV64IUR":
                // Uniform/register ops
                break;
            case "FHADD":
            case "FHFMA":
            case "UF2F":
            case "UF2I":
            case "UF2IP":
            case "UFADD":
            case "UFFMA":
            case "UFHADD":
            case "UFHFMA":
            case "UFMNMX":
            case "UFMUL":
            case "UFRND":
            case "UFSEL":
            case "UFSET":
            case "UFSETP":
            case "UI2F":
            case "UI2FP":
            case "UI2I":
            case "UI2IP":
            case "UIABS":
            case "UIMNMX":
            case "UVIADD":
            case "UVIMNMX":
            case "VIADD":
            case "VIMNMX":
                // Various arithmetic ops - try to pass through
                break;
            case "UTMACCTL":
            case "UTMACMDFLUSH":
            case "UTMALDG":
            case "UTMAPF":
            case "UTMAREDG":
            case "UTMASTG":
                // TMEM ops
                break;
        }

        var result = new SassInstruction
        {
            Mnemonic = instr.Mnemonic,
            Guard = instr.Guard,
            Source = instr.Source,
            Line = instr.Line,
            Address = instr.Address
        };
        result.Operands.AddRange(instr.Operands);
        result.BranchTargets.AddRange(instr.BranchTargets);
        result.Relocations.AddRange(instr.Relocations);
        return result;
    }

    private static void AssembleProgram(SassProgram program, string architecture, string outputPath, string? effectiveToolkit)
    {
        var imageSections = new List<CubinImageSection>();
        var imageSymbols = new List<CubinImageSymbol>();
        foreach (SassSection section in program.Sections)
        {
            var instructions = section.Nodes.OfType<SassInstruction>().ToList();
            if (instructions.Count == 0)
                continue;

            var labels = new Dictionary<string, int>(StringComparer.Ordinal);
            int address = 0;
            foreach (SassNode node in section.Nodes)
            {
                if (node is SassLabel label)
                    labels[label.Name] = address;
                else if (node is SassInstruction instruction)
                {
                    instruction.Address = address;
                    address += 16;
                }
            }

            if (!SupportsAbsoluteCallRelocation(architecture))
            {
                foreach (SassInstruction instruction in instructions)
                {
                    SassRelocation? absoluteCall = instruction.Relocations.FirstOrDefault(r => r.Type == 0x4b);
                    if (absoluteCall != null && labels.ContainsKey(absoluteCall.Symbol))
                    {
                        int operand = instruction.Operands.IndexOf("0x0");
                        if (operand >= 0)
                            instruction.Operands[operand] = absoluteCall.Symbol;
                    }
                    instruction.Relocations.RemoveAll(r => r.Type == 0x4b);
                }
            }

            byte[] data = new byte[instructions.Count * 16];
            for (int i = 0; i < instructions.Count; i++)
            {
                SassInstruction instruction = instructions[i];
                var (low, high) = EncodeInstruction(instruction, labels);
                BitConverter.GetBytes(low).CopyTo(data, i * 16);
                BitConverter.GetBytes(high).CopyTo(data, i * 16 + 8);
            }

            string sectionName = section.Name.Length > 0 ? section.Name : ".text";
            if (sectionName == ".text") sectionName = ".text._start";
            int sectionHeaderIndex = imageSections.Count;
            imageSections.Add(new CubinImageSection { Name = sectionName, Alignment = Math.Max(1, section.Alignment), Data = data });

            int registerCount = section.RegisterCount ?? RegisterCount(instructions);
            foreach (var pair in labels)
            {
                if (pair.Key.StartsWith(".L", StringComparison.Ordinal) || pair.Key.StartsWith(".text.", StringComparison.Ordinal))
                    continue;
                imageSymbols.Add(new CubinImageSymbol
                {
                    Name = pair.Key,
                    Section = sectionHeaderIndex,
                    Value = (ulong)pair.Value,
                    Size = (ulong)(data.Length - pair.Value),
                    RegisterCount = pair.Key.StartsWith(".text.", StringComparison.Ordinal) ? null : registerCount,
                });
            }
            if (!labels.Keys.Any(k => !k.StartsWith(".L", StringComparison.Ordinal)))
            {
                imageSymbols.Add(new CubinImageSymbol
                {
                    Name = SafeFunctionName(sectionName),
                    Section = sectionHeaderIndex,
                    Value = 0,
                    Size = (ulong)data.Length,
                    RegisterCount = registerCount,
                });
            }
        }

        if (imageSections.Count == 0)
            throw new InvalidDataException("The source contains no instructions.");

        uint architectureNumber = ArchitectureNumber(architecture);
        CubinWriter.Write(outputPath, architectureNumber, architecture, imageSections, imageSymbols, effectiveToolkit);
    }
}
