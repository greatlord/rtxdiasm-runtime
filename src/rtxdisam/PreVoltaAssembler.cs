using System.Buffers.Binary;
using nvDismDecode;

namespace rtxdisam;

// ---------------------------------------------------------------------------
// Pre-Volta assembler.
//
// Pre-Volta SASS uses 64-bit instructions in control-word bundles.  This
// encoder reuses the table-driven instruction encoder from SassAssembler.
// Maxwell/Pascal control words carry synthesized stalls, reuse flags and
// scoreboard barriers.  SM35/SM37 build seven 8-bit slot sections; SM32 uses
// the older GK104/Fermi-era delay bytes.  SM30 retains its control template.
// ---------------------------------------------------------------------------

public static class PreVoltaAssembler
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
        if (!IsPreVoltaArchitecture(architecture))
            throw new InvalidDataException(
                $"{architecture} is not a pre-Volta architecture.");
        string? effectiveToolkit = InstructionDecoder.NormalizeToolkit(
            toolkit ?? (program.Toolkit.Length > 0 ? program.Toolkit : null));
        PreVoltaDecoder.ResolveTablePath(architecture, effectiveToolkit);
        effectiveToolkit ??= "102";

        Assemble(program, architecture, outputPath, irOutputPath,
            effectiveToolkit);
    }

    internal static void Assemble(
        SassProgram program, string architecture, string outputPath,
        string? irOutputPath, string? toolkit = null)
    {
        if (irOutputPath != null)
            SassIRSerializer.Write(
                irOutputPath, program, architecture, toolkit);

        var imageSections = new List<CubinImageSection>();
        var imageSymbols = new List<CubinImageSymbol>();
        int bundleSize = PreVoltaDecoder.GetBundleSize(architecture);

        foreach (SassSection section in program.Sections)
        {
            if (!section.Nodes.OfType<SassInstruction>().Any())
                continue;

            (byte[] data, Dictionary<string, int> labels) =
                EncodeSectionWithLabels(architecture, section, toolkit: toolkit);
            if (data.Length == 0)
                continue;

            string sectionName = section.Name.Length > 0
                ? section.Name
                : ".text";
            if (sectionName == ".text")
                sectionName = ".text._start";

            int sectionIndex = imageSections.Count;
            imageSections.Add(new CubinImageSection
            {
                Name = sectionName,
                Alignment = Math.Max(1, bundleSize),
                Data = data,
            });

            int registerCount = section.RegisterCount ??
                (int)RegisterCountEstimator.Estimate(
                    section.Nodes
                        .OfType<SassInstruction>()
                        .Select(instruction => instruction.Source),
                    preVolta: true);

            foreach (var pair in labels)
            {
                if (pair.Key.StartsWith(".L", StringComparison.Ordinal) ||
                    pair.Key.StartsWith(".text.", StringComparison.Ordinal))
                    continue;
                imageSymbols.Add(new CubinImageSymbol
                {
                    Name = pair.Key,
                    Section = sectionIndex,
                    Value = (ulong)pair.Value,
                    Size = (ulong)Math.Max(0, data.Length - pair.Value),
                    RegisterCount = registerCount,
                });
            }

            if (!labels.Keys.Any(
                name => !name.StartsWith(".L", StringComparison.Ordinal) &&
                    !name.StartsWith(".text.", StringComparison.Ordinal)))
            {
                imageSymbols.Add(new CubinImageSymbol
                {
                    Name = SafeFunctionName(sectionName),
                    Section = sectionIndex,
                    Value = 0,
                    Size = (ulong)data.Length,
                    RegisterCount = registerCount,
                });
            }
        }

        if (imageSections.Count == 0)
            throw new InvalidDataException("The source contains no instructions.");

        CubinWriter.Write(
            outputPath, ArchitectureNumber(architecture), architecture,
            imageSections, imageSymbols, toolkit ?? "102");
    }

    private static bool IsBranchOperation(string mnemonic)
    {
        string op = mnemonic.Split('.')[0].ToUpperInvariant();
        return op is "BRA" or "BRX" or "JMP" or "CALL" or "RET";
    }

    private static bool IsTerminator(string mnemonic)
    {
        string op = mnemonic.Split('.')[0].ToUpperInvariant();
        return op is "EXIT" or "RET" or "BRK" or "KIL" or "LONGJMP" or
            "PLONGJMP";
    }

    private static bool IsTerminalAfterPadding(
        IReadOnlyList<SassInstruction> instructions, int start)
    {
        for (int index = start; index < instructions.Count; index++)
        {
            if (instructions[index].Mnemonic.StartsWith(
                    "NOP", StringComparison.OrdinalIgnoreCase))
                continue;
            return IsTerminator(instructions[index].Mnemonic);
        }
        return false;
    }

    private static bool IsLoadOperation(string mnemonic)
    {
        string op = mnemonic.Split('.')[0].ToUpperInvariant();
        return op is "LD" or "LDG" or "LDS" or "LDL" or "LDC";
    }

    private static bool IsAluOperation(string mnemonic)
    {
        string op = mnemonic.Split('.')[0].ToUpperInvariant();
        return op is "IADD" or "IADD32I" or "IMAD" or "IMUL" or "SHL" or
            "SHR" or "SHF" or "LOP" or "LOP32I" or "LOP3" or "ISCADD" or
            "LEA" or "IMNMX" or "BFE" or "BFI" or "PRMT";
    }

    private static bool IsIntegerOrConvertOperation(string mnemonic)
    {
        if (IsAluOperation(mnemonic))
            return true;
        string op = mnemonic.Split('.')[0].ToUpperInvariant();
        return op is "F2I" or "I2F";
    }

    private static bool IsStoreOperation(SassInstruction? previous)
    {
        if (previous == null)
            return false;
        string op = previous.Mnemonic.Split('.')[0].ToUpperInvariant();
        return op is "ST" or "STG" or "STS" or "STL" or "RED";
    }

    private static bool NeedsExitDrain(SassInstruction? previous)
    {
        if (previous == null)
            return false;
        string op = previous.Mnemonic.Split('.')[0].ToUpperInvariant();
        return op is "LD" or "ST" or "LDG" or "STG" or "LDS" or "STS" or
            "LDL" or "STL" or "LDC" or "ATOM" or "RED" or "BAR" or
            "MEMBAR" or "NOP" or "BRA" or "BRX" or "JMP" or "CALL" or
            "RET" or "EXIT" or "SSY" or "PBK" or "PCNT";
    }

    private static Dictionary<
        int, (int? Write, int? Read, int Wait)> BuildBarrierPlans(
        IReadOnlyList<SassInstruction> instructions)
    {
        var plans = new Dictionary<
            int, (int? Write, int? Read, int Wait)>();
        var pending = new Dictionary<string, int>(StringComparer.Ordinal);
        int nextBarrier = 1;
        int nextFirstLoadBarrier = 0;

        void AddWait(int address, int waitMask)
        {
            plans.TryGetValue(address, out var plan);
            plans[address] = (plan.Write, plan.Read, plan.Wait | waitMask);
        }

        for (int instructionIndex = 0;
            instructionIndex < instructions.Count; instructionIndex++)
        {
            SassInstruction instruction = instructions[instructionIndex];
            string op = instruction.Mnemonic.Split('.')[0]
                .ToUpperInvariant();
            int waitMask = 0;
            var consumedBarrier1 = new List<string>();
            foreach (string source in PreVoltaScheduling.SourceRegisters(
                instruction.Mnemonic, instruction.Operands))
            {
                if (pending.TryGetValue(source, out int mask))
                {
                    waitMask |= mask;
                    if (mask == 1)
                        consumedBarrier1.Add(source);
                }
            }
            if (waitMask != 0)
                AddWait(instruction.Address, waitMask);

            string? destination = PreVoltaScheduling.DestinationRegister(
                instruction.Mnemonic, instruction.Operands);
            bool load = op is "LD" or "LDG" or "LDS" or "LDL" or "LDC";
            bool store = op is "ST" or "STG" or "STS" or "STL";
            bool specialRegister = op is "S2R" or "CS2R";
            if (specialRegister && destination != null && destination != "RZ")
            {
                bool usedLater = false;
                for (int later = instructionIndex + 1;
                    later < instructions.Count; later++)
                {
                    SassInstruction next = instructions[later];
                    if (PreVoltaScheduling.SourceRegisters(
                            next.Mnemonic, next.Operands)
                        .Contains(destination,
                            StringComparer.OrdinalIgnoreCase))
                    {
                        usedLater = true;
                        break;
                    }
                    if (PreVoltaScheduling.DestinationRegister(
                            next.Mnemonic, next.Operands) == destination)
                        break;
                }
                if (usedLater)
                {
                    // S2R publishes barrier 1 only when a consumer uses it.
                    plans.TryGetValue(instruction.Address, out var existing);
                    plans[instruction.Address] =
                        (1, existing.Read, existing.Wait);
                    pending[destination] = 1;
                    if (nextBarrier == 1)
                        nextBarrier = 2;
                }
            }
            else if (load && destination != null && destination != "RZ")
            {
                // ptxas tracks only the final load of an adjacent load pair;
                // the later load's barrier also covers the earlier one.
                bool followedByLoad = false;
                if (instructionIndex + 1 < instructions.Count &&
                    IsLoadOperation(
                        instructions[instructionIndex + 1].Mnemonic))
                {
                    SassInstruction next = instructions[instructionIndex + 1];
                    bool nextDependsOnThisLoad =
                        PreVoltaScheduling.SourceRegisters(
                            next.Mnemonic, next.Operands)
                        .Contains(destination, StringComparer.OrdinalIgnoreCase);
                    followedByLoad = !nextDependsOnThisLoad;
                }
                bool firstLoadStorePair =
                    op == "LDG" &&
                    instructionIndex == 0 &&
                    instructionIndex + 1 < instructions.Count &&
                    IsStoreOperation(instructions[instructionIndex + 1]) &&
                    PreVoltaScheduling.SourceRegisters(
                        instructions[instructionIndex + 1].Mnemonic,
                        instructions[instructionIndex + 1].Operands)
                    .Contains(destination,
                        StringComparer.OrdinalIgnoreCase);
                if (!followedByLoad && !firstLoadStorePair)
                {
                    // CUDA 10.2 ptxas uses scoreboard barrier 6 for global
                    // loads.  Reusing one load barrier also matches consumers
                    // that wait on several loads in the same bundle.
                    int loadBarrier;
                    if (op == "LDS")
                    {
                        loadBarrier = 1;
                    }
                    else if (instructionIndex == 0)
                    {
                        loadBarrier = 1;
                        nextFirstLoadBarrier = 2;
                    }
                    else if (nextFirstLoadBarrier > 0)
                    {
                        loadBarrier = nextFirstLoadBarrier;
                        nextFirstLoadBarrier = nextFirstLoadBarrier == 6
                            ? 1
                            : nextFirstLoadBarrier + 1;
                    }
                    else
                    {
                        loadBarrier = 6;
                    }
                    plans.TryGetValue(instruction.Address, out var existing);
                    // A load can both wait on an earlier load and publish its
                    // own write barrier.  Preserve the wait mask instead of
                    // replacing the whole plan.
                    plans[instruction.Address] =
                        (loadBarrier, existing.Read, existing.Wait);
                    pending[destination] = 1 << (loadBarrier - 1);
                    if (loadBarrier < 6 && nextBarrier <= loadBarrier)
                        nextBarrier = loadBarrier + 1;
                }
            }
            else if (store)
            {
                bool nextIsNop =
                    instructionIndex + 1 < instructions.Count &&
                    instructions[instructionIndex + 1].Mnemonic.StartsWith(
                        "NOP", StringComparison.OrdinalIgnoreCase);
                bool sourceFromLoad =
                    PreVoltaScheduling.SourceRegisters(
                        instruction.Mnemonic, instruction.Operands)
                    .Any(register => pending.ContainsKey(register));
                if (!nextIsNop && sourceFromLoad)
                {
                    plans.TryGetValue(instruction.Address, out var existing);
                    // A read barrier holds a load result until the store has
                    // consumed it.  Stores fed by ALU/MOV results and stores
                    // followed by padding NOPs do not need one.
                    plans[instruction.Address] =
                        (existing.Write, nextBarrier, existing.Wait);
                    nextBarrier = nextBarrier == 6 ? 1 : nextBarrier + 1;
                }
            }
            else if (destination != null)
            {
                pending.Remove(destination);
            }
            foreach (string source in consumedBarrier1)
                pending.Remove(source);
        }
        return plans;
    }

    private static int ReuseCode(SassInstruction instruction)
    {
        int code = 0;
        int source = 0;
        foreach (string operand in instruction.Operands.Skip(1))
        {
            if (operand.Contains(".reuse", StringComparison.OrdinalIgnoreCase) &&
                source < 3)
            {
                code |= 1 << source;
            }
            source++;
        }
        string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
        if (code != 0 && op is "ISETP" or "FSETP" or "ICMP" or "FCMP")
            return 1;
        return code;
    }

    private static ulong EncodeInstructionWord(
        SassInstruction instruction, IReadOnlyDictionary<string, int> labels,
        ulong exitWord)
    {
        // EXIT has an implicit condition-code operand in the legacy tables;
        // the canonical CUDA 10.2 encodings are stable for each family.
        if (instruction.Mnemonic.Equals("EXIT", StringComparison.OrdinalIgnoreCase))
            return exitWord;

        string[] originalOperands = instruction.Operands.ToArray();
        for (int i = 0; i < instruction.Operands.Count; i++)
        {
            instruction.Operands[i] = instruction.Operands[i]
                .Replace(".reuse", "", StringComparison.OrdinalIgnoreCase);
        }
        try
        {
            (ulong low, _) = SassAssembler.EncodeInstruction(
                instruction, labels);
            return low;
        }
        finally
        {
            instruction.Operands.Clear();
            instruction.Operands.AddRange(originalOperands);
        }
    }

    public static byte[] EncodeSection(
        string architecture, SassSection section, ulong? controlWord = null,
        string? toolkit = null) =>
        EncodeSectionWithLabels(
            architecture, section, controlWord, toolkit).Data;

    public static ulong DefaultControlWord(string architecture)
    {
        int arch = (int)ArchitectureNumber(architecture);
        if (arch is >= 30 and <= 37)
            return arch == 30 ? 0x200000000002f307UL : 0x08a0a0a0fca010c0UL;
        return 0x001fc800fe2007f6UL;
    }

    private static (byte[] Data, Dictionary<string, int> Labels)
        EncodeSectionWithLabels(
            string architecture, SassSection section,
            ulong? controlWord = null, string? toolkit = null)
    {
        PreVoltaDecoder.UseArchitecture(architecture, toolkit);
        int bundleSize = PreVoltaDecoder.GetBundleSize(architecture);
        int slotsPerBundle = bundleSize == 64 ? 7 : 3;
        if (bundleSize == 32 &&
            section.Nodes.OfType<SassInstruction>()
                .Any(instruction => instruction.ControlByte != null))
            throw new InvalidDataException(
                "Kepler control bytes require a 64-byte bundle");

        var labels = new Dictionary<string, int>(StringComparer.Ordinal);
        var instructions = new List<SassInstruction>();
        bool preserveLayout = section.Nodes
            .OfType<SassInstruction>()
            .Any(instruction =>
                instruction.ControlByte != null ||
                instruction.ControlWord != null ||
                instruction.InstructionWord != null);
        SassInstruction? lastNonNopInstruction = section.Nodes
            .OfType<SassInstruction>()
            .LastOrDefault(instruction => !instruction.Mnemonic.StartsWith(
                "NOP", StringComparison.OrdinalIgnoreCase));
        var controllessBundles = new HashSet<int>();
        int controllessDataEnd = 0;
        int bundleBase = 0;
        int slot = 0;
        foreach (SassNode node in section.Nodes)
        {
            int address = bundleBase + (slot + 1) * 8;
            if (node is SassLabel label)
            {
                labels[label.Name] = address;
                continue;
            }
            if (node is not SassInstruction instruction)
                continue;

            // After a slot-6 EXIT, ptxas places a terminal BRA at the next
            // bundle's control offset and leaves the seven slots as padding.
            bool controllessBranch = bundleSize == 64 && slot == 0 &&
                bundleBase > 0 &&
                instruction.Mnemonic.StartsWith(
                    "BRA", StringComparison.OrdinalIgnoreCase) &&
                instruction == lastNonNopInstruction &&
                instructions.Count > 0 &&
                instructions[^1].Mnemonic.StartsWith(
                    "EXIT", StringComparison.OrdinalIgnoreCase) &&
                instructions[^1].Address == bundleBase - 8;
            if (controllessBranch)
            {
                controllessBundles.Add(bundleBase);
                controllessDataEnd = Math.Max(
                    controllessDataEnd, bundleBase + bundleSize);
            }

            instruction.Address = address;
            instructions.Add(instruction);
            slot++;
            if (slot == slotsPerBundle)
            {
                slot = 0;
                bundleBase += bundleSize;
            }
        }

        int dataSize = bundleBase + (slot > 0 ? bundleSize : 0);
        if (controllessDataEnd > 0)
        {
            dataSize = Math.Min(dataSize, controllessDataEnd);
            instructions.RemoveAll(
                instruction => instruction.Address >= controllessDataEnd);
        }
        if (bundleSize == 32 && dataSize > 32)
        {
            // ptxas pads multi-bundle Maxwell text sections to 64 bytes.
            dataSize = (dataSize + 63) & ~63;
        }
        var explicitControlWords = new Dictionary<int, ulong>();
        foreach (SassInstruction instruction in instructions)
        {
            if (instruction.ControlWord is not ulong word)
                continue;
            int controlBundle = instruction.Address -
                instruction.Address % bundleSize;
            if (explicitControlWords.TryGetValue(
                    controlBundle, out ulong existing) &&
                existing != word)
                throw new InvalidDataException(
                    "conflicting control words in one pre-Volta bundle");
            explicitControlWords[controlBundle] = word;
        }
        if (instructions.Any(instruction =>
                instruction.ControlByte != null &&
                explicitControlWords.ContainsKey(
                    instruction.Address - instruction.Address % bundleSize)))
            throw new InvalidDataException(
                "control byte and control word cannot share a bundle");
        byte[] data = new byte[dataSize];
        // A pre-Volta bundle needs a valid control word even when no
        // scheduling information is synthesized.  These are the neutral
        // control words used by the CUDA 10.2 toolchain for the two layout
        // families.
        int archNumber = (int)ArchitectureNumber(architecture);
        bool earlyKepler = archNumber == 30;
        // SM30 needs a different control template: its slot/unit bits
        // distinguish EXIT from JMX even for the same instruction word.
        ulong defaultControl = controlWord ??
            (bundleSize == 64
                ? (earlyKepler
                    ? 0x200000000002f307UL
                    : 0x08a0a0a0fca010c0UL)
                : 0x001fc800fe2007f6UL);
        ulong exitWord = bundleSize == 64
            ? (earlyKepler ? 0x8000000000001de7UL : 0x18000000001c003cUL)
            : 0xe30000000007000fUL;
        for (int controlOffset = 0; controlOffset < dataSize;
            controlOffset += bundleSize)
        {
            BinaryPrimitives.WriteUInt64LittleEndian(
                data.AsSpan(controlOffset), defaultControl);
        }

        var usedOffsets = new HashSet<int>();
        var paddingOffsets = new HashSet<int>();
        var reuseCodes = new Dictionary<int, int>();
        foreach (SassInstruction instruction in instructions)
        {
            reuseCodes[instruction.Address] = ReuseCode(instruction);
        }
        Dictionary<int, (int? Write, int? Read, int Wait)> barrierPlans =
            BuildBarrierPlans(instructions);
        foreach (SassInstruction instruction in instructions)
        {
            ulong low = instruction.InstructionWord ??
                EncodeInstructionWord(instruction, labels, exitWord);
            // Instruction addresses are already byte offsets into the bundle
            // (the control word occupies offset 0).
            BinaryPrimitives.WriteUInt64LittleEndian(
                data.AsSpan(instruction.Address), low);
            usedOffsets.Add(instruction.Address);
        }

        // Every instruction slot in a bundle must contain a valid operation.
        // Fill unused slots with NOPs instead of leaving zero words.
        foreach (int offset in Enumerable.Range(0, dataSize)
            .Where(offset => offset % 8 == 0 && offset % bundleSize != 0 &&
                !usedOffsets.Contains(offset)))
        {
            var nop = new SassInstruction
            {
                Mnemonic = preserveLayout ? "NOP" : "MOV",
                Address = offset,
            };
            if (!preserveLayout)
                nop.Operands.AddRange(new[] { "RZ", "RZ" });
            ulong low = EncodeInstructionWord(nop, labels, exitWord);
            BinaryPrimitives.WriteUInt64LittleEndian(
                data.AsSpan(offset), low);
            paddingOffsets.Add(offset);
        }
        // On Maxwell/Pascal the control word has one 21-bit section per
        // instruction slot.  Preserve the validated template but replace just
        // the per-slot stall counts from the timing model.
        if (bundleSize == 32)
        {
            var instructionByAddress = instructions.ToDictionary(
                instruction => instruction.Address);
            SassInstruction?[] slotInstructions = new SassInstruction?[3];
            SassInstruction? previousInstruction = null;
            for (int controlBundle = 0; controlBundle < dataSize;
                controlBundle += bundleSize)
            {
                for (int controlSlot = 0; controlSlot < 3; controlSlot++)
                {
                    int offset = controlBundle + (controlSlot + 1) * 8;
                    slotInstructions[controlSlot] =
                        instructionByAddress.TryGetValue(
                            offset, out SassInstruction? instruction)
                            ? instruction
                            : null;
                }
                if (controllessBundles.Contains(controlBundle))
                {
                    previousInstruction =
                        slotInstructions.LastOrDefault(item => item != null);
                    continue;
                }
                if (explicitControlWords.TryGetValue(
                        controlBundle, out ulong exactControl))
                {
                    previousInstruction =
                        slotInstructions.LastOrDefault(item => item != null);
                    BinaryPrimitives.WriteUInt64LittleEndian(
                        data.AsSpan(controlBundle), exactControl);
                    continue;
                }
                int[] stalls = new int[3];
                bool[] emptySlots = new bool[3];
                bool[] independentMov = new bool[3];
                bool[] clearStallBit4 = new bool[3];
                int[] terminalStoreStalls = { -1, -1, -1 };
                for (int i = 0; i < 3; i++)
                {
                    SassInstruction? current = slotInstructions[i];
                    if (current == null)
                    {
                        int offset = controlBundle + (i + 1) * 8;
                        bool autoPadding = paddingOffsets.Contains(offset);
                        bool afterBranch = previousInstruction != null &&
                            IsBranchOperation(previousInstruction.Mnemonic);
                        if (!autoPadding || afterBranch)
                            emptySlots[i] = true;
                        continue;
                    }
                    stalls[i] = PreVoltaScheduling.MinimumStall(
                        current.Mnemonic);
                    if (previousInstruction != null)
                    {
                        int dependency = PreVoltaScheduling.DependencyStall(
                            previousInstruction.Mnemonic,
                            previousInstruction.Operands,
                            current.Mnemonic, current.Operands,
                            maxwell: true);
                        stalls[i] = Math.Max(stalls[i], dependency);
                        if (dependency == 0 &&
                            current.Mnemonic.StartsWith(
                                "MOV", StringComparison.OrdinalIgnoreCase))
                        {
                            independentMov[i] = true;
                        }
                        if (IsLoadOperation(previousInstruction.Mnemonic) &&
                            IsAluOperation(current.Mnemonic))
                        {
                            // Maxwell ptxas uses a two-cycle load-to-ALU
                            // dependency, not the generic read latency.
                            stalls[i] = Math.Min(stalls[i], 2);
                        }
                    }
                    if (current.Mnemonic.StartsWith(
                            "EXIT", StringComparison.OrdinalIgnoreCase) &&
                        !NeedsExitDrain(previousInstruction))
                    {
                        // ptxas commonly uses a one-cycle EXIT stall after a
                        // register/ALU instruction; keep the conservative
                        // value only when a memory/control op must drain.
                        stalls[i] = 1;
                    }
                    if (IsBranchOperation(current.Mnemonic))
                    {
                        if (NeedsExitDrain(previousInstruction))
                        {
                            // Branches after memory/control operations need
                            // the full conservative drain in ptxas output.
                            stalls[i] = Math.Max(stalls[i], 15);
                        }
                        else
                        {
                            // A branch after a normal ALU instruction is
                            // commonly two cycles in ptxas output.  Keep any
                            // larger register dependency.
                            int dependency = previousInstruction != null
                                ? PreVoltaScheduling.DependencyStall(
                                    previousInstruction.Mnemonic,
                                    previousInstruction.Operands,
                                    current.Mnemonic, current.Operands,
                                    maxwell: true)
                                : 0;
                            stalls[i] = Math.Max(2, dependency);
                        }
                    }
                    if (current.Mnemonic.StartsWith(
                            "EXIT", StringComparison.OrdinalIgnoreCase) &&
                        IsStoreOperation(previousInstruction))
                    {
                        // A direct store -> EXIT uses the minimum drain.
                        stalls[i] = Math.Max(stalls[i], 5);
                    }
                    if (current.Mnemonic.StartsWith(
                            "EXIT", StringComparison.OrdinalIgnoreCase) &&
                        previousInstruction != null &&
                        previousInstruction.Mnemonic.StartsWith(
                            "NOP", StringComparison.OrdinalIgnoreCase))
                    {
                        // Padding NOPs keep the full drain before EXIT.
                        stalls[i] = Math.Max(stalls[i], 15);
                    }
                    int currentIndex = instructions.IndexOf(current);
                    if (IsStoreOperation(current) &&
                        IsTerminalAfterPadding(
                            instructions, currentIndex + 1))
                    {
                        int dependency = previousInstruction != null
                            ? PreVoltaScheduling.DependencyStall(
                                previousInstruction.Mnemonic,
                                previousInstruction.Operands,
                                current.Mnemonic, current.Operands,
                                maxwell: true)
                            : 0;
                        terminalStoreStalls[i] = Math.Max(1, dependency);
                        if (barrierPlans.TryGetValue(
                                current.Address, out var terminalPlan) &&
                            terminalPlan.Wait is 1 or 2)
                        {
                            terminalStoreStalls[i] = Math.Max(
                                terminalStoreStalls[i], terminalPlan.Wait);
                        }
                    }
                    previousInstruction = current;
                }
                for (int i = 0; i < stalls.Length; i++)
                {
                    SassInstruction? currentSlotInstruction = slotInstructions[i];
                    if (emptySlots[i])
                    {
                        // ptxas leaves unused slots at stall zero.
                        stalls[i] = 0;
                        continue;
                    }
                    int templateStall = (int)((defaultControl >> (i * 21)) & 0xF);
                    stalls[i] = Math.Max(stalls[i], templateStall);
                    if (currentSlotInstruction != null)
                    {
                        string currentOp = currentSlotInstruction.Mnemonic
                            .Split('.')[0].ToUpperInvariant();
                        if (currentOp == "BAR")
                            stalls[i] = 15;
                        if (IsStoreOperation(currentSlotInstruction) &&
                            barrierPlans.TryGetValue(
                                currentSlotInstruction.Address,
                                out var storeBarrierPlan) &&
                            storeBarrierPlan.Wait != 0)
                        {
                            stalls[i] = storeBarrierPlan.Wait is 1 or 2
                                ? storeBarrierPlan.Wait
                                : 1;
                        }
                    }
                    if (independentMov[i] &&
                        currentSlotInstruction != null &&
                        (i == 0 || i == 2))
                    {
                        int currentIndex = instructions.IndexOf(
                            currentSlotInstruction);
                        SassInstruction? nextInstruction =
                            currentIndex + 1 < instructions.Count
                                ? instructions[currentIndex + 1]
                                : null;
                        string nextOp = nextInstruction?.Mnemonic
                            .Split('.')[0].ToUpperInvariant() ?? "";
                        bool nextIsMovOrControl = nextOp is
                            "MOV" or "MOV32I" or "NOP" or "BRA" or "BRX" or
                            "JMP" or "RET" or "EXIT";
                        if (nextIsMovOrControl)
                            stalls[i] = 1;
                        else if (i == 0 && controlBundle > 0 &&
                            nextInstruction != null &&
                            (IsLoadOperation(nextInstruction.Mnemonic) ||
                             IsStoreOperation(nextInstruction)))
                            stalls[i] = 2;
                        else if (i == 0 && controlBundle > 0 &&
                            nextInstruction != null &&
                            IsAluOperation(nextInstruction.Mnemonic))
                            stalls[i] = 4;
                    }
                    if (currentSlotInstruction != null && i < 2 &&
                        IsAluOperation(currentSlotInstruction.Mnemonic) &&
                        slotInstructions[i + 1] is SassInstruction storeSlot &&
                        IsStoreOperation(storeSlot))
                    {
                        string? aluDestination =
                            PreVoltaScheduling.DestinationRegister(
                                currentSlotInstruction.Mnemonic,
                                currentSlotInstruction.Operands);
                        if (aluDestination != null &&
                            PreVoltaScheduling.SourceRegisters(
                                storeSlot.Mnemonic, storeSlot.Operands)
                            .Contains(aluDestination,
                                StringComparer.OrdinalIgnoreCase))
                        {
                            stalls[i] = 2;
                        }
                    }
                    if (currentSlotInstruction != null &&
                        IsLoadOperation(currentSlotInstruction.Mnemonic) &&
                        !(i == 0 && currentSlotInstruction.Mnemonic
                            .StartsWith("LDS", StringComparison.OrdinalIgnoreCase)))
                    {
                        bool followedByLoad = i < 2 &&
                            slotInstructions[i + 1] is SassInstruction nextSlot &&
                            IsLoadOperation(nextSlot.Mnemonic);
                        bool nextDependsOnLoad = false;
                        if (followedByLoad)
                        {
                            string? loadDestination =
                                PreVoltaScheduling.DestinationRegister(
                                    currentSlotInstruction.Mnemonic,
                                    currentSlotInstruction.Operands);
                            nextDependsOnLoad = loadDestination != null &&
                                PreVoltaScheduling.SourceRegisters(
                                    slotInstructions[i + 1]!.Mnemonic,
                                    slotInstructions[i + 1]!.Operands)
                                .Contains(loadDestination,
                                    StringComparer.OrdinalIgnoreCase);
                        }
                        stalls[i] = followedByLoad && !nextDependsOnLoad
                            ? 1
                            : 2;
                    }
                    if (currentSlotInstruction != null &&
                        currentSlotInstruction.Mnemonic.StartsWith(
                            "NOP", StringComparison.OrdinalIgnoreCase))
                    {
                        // Explicit padding NOPs normally carry the maximum
                        // stall.  The exact value for the final NOP varies.
                        stalls[i] = 15;
                    }
                    if (terminalStoreStalls[i] >= 0)
                        stalls[i] = terminalStoreStalls[i];
                    if (currentSlotInstruction != null)
                    {
                        string currentFull =
                            currentSlotInstruction.Mnemonic;
                        string currentOp = currentFull
                            .Split('.')[0].ToUpperInvariant();
                        int currentIndex = instructions.IndexOf(
                            currentSlotInstruction);
                        SassInstruction? previous = currentIndex > 0
                            ? instructions[currentIndex - 1]
                            : null;
                        SassInstruction? next = currentIndex + 1 <
                            instructions.Count
                                ? instructions[currentIndex + 1]
                                : null;
                        string previousFull = previous?.Mnemonic ?? "";
                        string previousOp = previous?.Mnemonic
                            .Split('.')[0].ToUpperInvariant() ?? "";
                        SassInstruction? previousPrevious = currentIndex > 1
                            ? instructions[currentIndex - 2]
                            : null;
                        string previousPreviousOp = previousPrevious?.Mnemonic
                            .Split('.')[0].ToUpperInvariant() ?? "";
                        if (currentOp is "S2R" or "CS2R")
                        {
                            string? destination =
                                PreVoltaScheduling.DestinationRegister(
                                    currentSlotInstruction.Mnemonic,
                                    currentSlotInstruction.Operands);
                            bool dependent = destination != null &&
                                next != null &&
                                PreVoltaScheduling.SourceRegisters(
                                    next.Mnemonic, next.Operands)
                                .Contains(destination,
                                    StringComparer.OrdinalIgnoreCase);
                            if (dependent)
                                stalls[i] = 15;
                            else if (i > 0)
                                stalls[i] = 1;
                        }
                        else if (currentOp is "SHL" or "SHR" or "SHF")
                        {
                            if (previousOp is "S2R" or "CS2R")
                                stalls[i] = 1;
                            else if (previousOp is "SHL" &&
                                (previousPreviousOp is "S2R" or "CS2R"))
                                stalls[i] = 5;
                            else if (previousOp is "SHL" or "SHR" or "SHF")
                                stalls[i] = 1;
                        }
                        else if (currentFull.StartsWith(
                                "IADD.X", StringComparison.OrdinalIgnoreCase))
                        {
                            stalls[i] = 2;
                        }
                        else if (currentOp == "IADD" &&
                            (previousOp is "SHL" or "SHR" or "SHF") &&
                            previousPreviousOp is "SHL")
                        {
                            stalls[i] = 6;
                        }
                        else if (IsLoadOperation(currentOp) &&
                            previousFull.StartsWith(
                                "IADD.X", StringComparison.OrdinalIgnoreCase))
                        {
                            stalls[i] = 1;
                        }
                        else if (currentOp == "MOV" && previous != null &&
                            IsLoadOperation(previous.Mnemonic))
                        {
                            stalls[i] = 4;
                        }
                        else if (currentOp == "MEMBAR")
                        {
                            stalls[i] = 5;
                        }
                        else if (currentOp == "LDS" && i > 0)
                        {
                            stalls[i] = 15;
                        }
                        else if (currentOp == "EXIT")
                        {
                            if (previousOp == "IADD" &&
                                previous!.Operands.Any(operand =>
                                    operand.Contains(
                                        "c[", StringComparison.Ordinal) ||
                                    operand.TrimStart().StartsWith('-')))
                            {
                                stalls[i] = 5;
                            }
                            else if (i == 2 && previousOp == "IADD" &&
                                previousPreviousOp == "IADD")
                            {
                                stalls[i] = 6;
                            }
                        }
                        else if ((currentOp is "F2I" or "F2F" or "I2F" or "I2I") &&
                            (previousOp is "F2I" or "F2F" or "I2F" or "I2I"))
                        {
                            stalls[i] = 1;
                        }
                        else if (currentOp == "SEL" &&
                            previousOp == "MOV32I")
                        {
                            stalls[i] = 2;
                        }
                        else if (currentOp == "NOP")
                        {
                            if (i == 1 && previous != null &&
                                IsStoreOperation(previous) &&
                                next != null &&
                                next.Mnemonic.StartsWith(
                                    "BAR",
                                    StringComparison.OrdinalIgnoreCase))
                            {
                                stalls[i] = 8;
                            }
                            else if (i == 1 && previous != null &&
                                previous.Mnemonic.StartsWith(
                                    "NOP",
                                    StringComparison.OrdinalIgnoreCase) &&
                                next != null &&
                                next.Mnemonic.StartsWith(
                                    "EXIT",
                                    StringComparison.OrdinalIgnoreCase) &&
                                instructions.Any(instruction =>
                                    instruction.Address >= controlBundle - 32 &&
                                    instruction.Address < controlBundle &&
                                    instruction.Mnemonic.Split('.')[0]
                                        .ToUpperInvariant() is
                                            "LDS" or "STS" or "BAR" or
                                            "MEMBAR"))
                            {
                                stalls[i] = 13;
                            }
                            else if (i == 1 && previous != null &&
                                previous.Mnemonic.StartsWith(
                                    "NOP",
                                    StringComparison.OrdinalIgnoreCase) &&
                                next != null &&
                                next.Mnemonic.StartsWith(
                                    "EXIT",
                                    StringComparison.OrdinalIgnoreCase) &&
                                instructions.Any(instruction =>
                                    instruction.Address >= controlBundle - 32 &&
                                    instruction.Address < controlBundle &&
                                    instruction.Mnemonic.Contains(
                                        ".U16",
                                        StringComparison.OrdinalIgnoreCase)))
                            {
                                stalls[i] = 7;
                            }
                            else if (i == 0)
                            {
                                SassInstruction? previousStore =
                                    instructions.LastOrDefault(instruction =>
                                        instruction.Address >=
                                            controlBundle - 32 &&
                                        instruction.Address < controlBundle &&
                                        IsStoreOperation(instruction) &&
                                        !instruction.Mnemonic.Contains(
                                            ".U16",
                                            StringComparison.OrdinalIgnoreCase));
                                if (previousStore != null)
                                {
                                    bool storeFromLoad = false;
                                    var storeSources =
                                        PreVoltaScheduling.SourceRegisters(
                                            previousStore.Mnemonic,
                                            previousStore.Operands);
                                    foreach (SassInstruction producer in
                                        instructions)
                                    {
                                        if (producer.Address <
                                                controlBundle - 32 ||
                                            producer.Address >=
                                                previousStore.Address ||
                                            !IsLoadOperation(
                                                producer.Mnemonic))
                                            continue;
                                        string? destination =
                                            PreVoltaScheduling
                                                .DestinationRegister(
                                                    producer.Mnemonic,
                                                    producer.Operands);
                                        if (destination != null &&
                                            storeSources.Contains(
                                                destination,
                                                StringComparer
                                                    .OrdinalIgnoreCase))
                                        {
                                            storeFromLoad = true;
                                            break;
                                        }
                                    }
                                    if (!storeFromLoad)
                                        stalls[i] = 9;
                                }
                            }
                        }
                        if (currentIndex == 0 &&
                            IsLoadOperation(currentOp))
                        {
                            stalls[i] = currentSlotInstruction.Mnemonic
                                .Contains(".U16",
                                    StringComparison.OrdinalIgnoreCase)
                                ? 2
                                : 6;
                        }
                        if (IsLoadOperation(currentOp) &&
                            currentIndex == 1 &&
                            previousFull.StartsWith(
                                "LDG", StringComparison.OrdinalIgnoreCase))
                        {
                            stalls[i] = 1;
                        }
                        if (currentOp == "IADD" && currentIndex == 1 &&
                            previousFull.StartsWith(
                                "LDG", StringComparison.OrdinalIgnoreCase))
                        {
                            stalls[i] = 1;
                        }
                        if (i == 0 && IsLoadOperation(currentOp) &&
                            slotInstructions[1] is SassInstruction firstStore &&
                            IsStoreOperation(firstStore))
                        {
                            string? loadDestination =
                                PreVoltaScheduling.DestinationRegister(
                                    currentSlotInstruction.Mnemonic,
                                    currentSlotInstruction.Operands);
                            if (loadDestination != null &&
                                PreVoltaScheduling.SourceRegisters(
                                    firstStore.Mnemonic, firstStore.Operands)
                                .Contains(loadDestination,
                                    StringComparer.OrdinalIgnoreCase))
                            {
                                stalls[i] = currentSlotInstruction.Mnemonic
                                    .Contains(".U16",
                                        StringComparison.OrdinalIgnoreCase)
                                    ? 2
                                    : 6;
                            }
                        }
                        if (i == 2 && previous != null &&
                            IsStoreOperation(previous) &&
                            previousPrevious != null &&
                            previousPrevious.Mnemonic.StartsWith(
                                "LDG", StringComparison.OrdinalIgnoreCase))
                        {
                            stalls[i] =
                                previous.Mnemonic.Contains(
                                    ".U16",
                                    StringComparison.OrdinalIgnoreCase) ||
                                previousPrevious.Mnemonic.Contains(
                                    ".U16",
                                    StringComparison.OrdinalIgnoreCase)
                                    ? 15
                                    : 2;
                        }
                        if (previous != null &&
                            previousOp is "ISETP" or "FSETP" or "ICMP" or
                                "FCMP" &&
                            currentOp is not ("ISETP" or "FSETP" or "ICMP" or
                                "FCMP" or "BRA" or "EXIT"))
                        {
                            stalls[i] = Math.Max(stalls[i], 12);
                        }
                        if (i == 1 && previous != null &&
                            IsLoadOperation(previous.Mnemonic) &&
                            currentOp == "IADD" &&
                            next != null &&
                            PreVoltaScheduling.DestinationRegister(
                                currentSlotInstruction.Mnemonic,
                                currentSlotInstruction.Operands) is
                                string iaddDestination &&
                            PreVoltaScheduling.SourceRegisters(
                                next.Mnemonic, next.Operands)
                            .Contains(iaddDestination,
                                StringComparer.OrdinalIgnoreCase))
                        {
                            stalls[i] = Math.Max(stalls[i], 6);
                        }
                        if (currentOp == "XMAD" &&
                            previousOp == "XMAD")
                        {
                            stalls[i] = Math.Max(stalls[i], 6);
                        }
                        if (i == 0 && currentOp == "XMAD" &&
                            previousOp != "XMAD" &&
                            next != null &&
                            next.Mnemonic.StartsWith(
                                "XMAD", StringComparison.OrdinalIgnoreCase))
                        {
                            stalls[i] = 1;
                            clearStallBit4[i] = true;
                        }
                        if (i == 0 && currentOp == "LOP3" &&
                            previousOp == "LOP32I" &&
                            archNumber is not (60 or 62))
                        {
                            clearStallBit4[i] = true;
                        }
                        if (i == 1 && currentOp == "IADD32I" &&
                            previousOp == "ISETP")
                        {
                            clearStallBit4[i] = true;
                        }
                        if (currentSlotInstruction.Guard is string guard &&
                            currentOp == "IADD32I")
                        {
                            string guardPredicate = guard
                                .TrimStart('@', '!')
                                .Trim();
                            for (int j = currentIndex - 1; j >= 0; j--)
                            {
                                SassInstruction producer = instructions[j];
                                string producerOp = producer.Mnemonic
                                    .Split('.')[0].ToUpperInvariant();
                                if (producerOp is not ("ISETP" or "FSETP" or
                                        "ICMP" or "FCMP"))
                                    continue;
                                if (producer.Operands.Count > 0 &&
                                    producer.Operands[0].Trim().Equals(
                                        guardPredicate,
                                        StringComparison.OrdinalIgnoreCase))
                                {
                                    stalls[i] = Math.Max(stalls[i], 13);
                                    break;
                                }
                            }
                        }
                        if (currentOp == "SEL" &&
                            previous != null &&
                            previous.Guard != null &&
                            previousOp == "IADD32I")
                        {
                            stalls[i] = 2;
                        }
                        if (currentOp == "SHR" &&
                            previousOp == "SHL" &&
                            next != null &&
                            next.Mnemonic.StartsWith(
                                "ISETP", StringComparison.OrdinalIgnoreCase))
                        {
                            stalls[i] = Math.Max(stalls[i], 6);
                        }
                        if (currentOp == "ISETP" &&
                            previousOp == "SHR")
                        {
                            stalls[i] = 1;
                        }
                        if (currentOp == "SEL" &&
                            previousOp == "IADD32I" &&
                            (previous == null || previous.Guard == null))
                        {
                            stalls[i] = 1;
                        }
                        if (currentOp == "EXIT" &&
                            previousOp == "STG" &&
                            previous != null)
                        {
                            bool storeFromLoad = false;
                            var storeSources =
                                PreVoltaScheduling.SourceRegisters(
                                    previous.Mnemonic, previous.Operands);
                            for (int j = currentIndex - 2; j >= 0; j--)
                            {
                                SassInstruction producer = instructions[j];
                                string? destination =
                                    PreVoltaScheduling.DestinationRegister(
                                        producer.Mnemonic, producer.Operands);
                                if (destination == null ||
                                    !storeSources.Contains(
                                        destination,
                                        StringComparer.OrdinalIgnoreCase))
                                    continue;
                                storeFromLoad = IsLoadOperation(
                                    producer.Mnemonic);
                                break;
                            }
                            if (!storeFromLoad)
                                stalls[i] = Math.Max(stalls[i], 15);
                        }
                    }
                }
                if (slotInstructions[0] is SassInstruction independentLoad &&
                    IsLoadOperation(independentLoad.Mnemonic) &&
                    slotInstructions[1] is SassInstruction independentMove &&
                    independentMove.Mnemonic.StartsWith(
                        "MOV32I", StringComparison.OrdinalIgnoreCase) &&
                    slotInstructions[2] is SassInstruction dependentAlu &&
                    (IsAluOperation(dependentAlu.Mnemonic) ||
                     dependentAlu.Mnemonic.StartsWith(
                         "XMAD", StringComparison.OrdinalIgnoreCase)))
                {
                    string? loadDestination =
                        PreVoltaScheduling.DestinationRegister(
                            independentLoad.Mnemonic,
                            independentLoad.Operands);
                    if (loadDestination != null &&
                        !PreVoltaScheduling.SourceRegisters(
                            independentMove.Mnemonic,
                            independentMove.Operands)
                        .Contains(loadDestination,
                            StringComparer.OrdinalIgnoreCase) &&
                        PreVoltaScheduling.SourceRegisters(
                            dependentAlu.Mnemonic,
                            dependentAlu.Operands)
                        .Contains(loadDestination,
                            StringComparer.OrdinalIgnoreCase))
                    {
                        stalls[0] = 1;
                        stalls[1] = 6;
                    }
                }
                ulong control = PreVoltaScheduling.SetMaxwellStalls(
                    defaultControl, stalls);
                for (int clearSlot = 0; clearSlot < 3; clearSlot++)
                {
                    if (clearStallBit4[clearSlot])
                        control &= ~(0x10UL << (clearSlot * 21));
                }
                for (int controlSlot = 0; controlSlot < 3; controlSlot++)
                {
                    SassInstruction? controlInstruction =
                        slotInstructions[controlSlot];
                    if (emptySlots[controlSlot])
                    {
                        // ptxas encodes a completely unused slot as 0x7e0.
                        int emptyShift = controlSlot * 21;
                        ulong emptyMask = 0x1FFFFFUL << emptyShift;
                        control = (control & ~emptyMask) |
                            (0x7E0UL << emptyShift);
                    }
                    else if (controlInstruction != null &&
                        controlInstruction.Mnemonic.StartsWith(
                            "NOP", StringComparison.OrdinalIgnoreCase))
                    {
                        // NOP sections clear bit 4 and keep the stall in the
                        // low nibble, giving 0x7ef for a full stall.
                        int nopShift = controlSlot * 21;
                        ulong nopMask = 0x1FFFFFUL << nopShift;
                        control = (control & ~nopMask) |
                            ((0x7E0UL | (uint)(stalls[controlSlot] & 0xF))
                                << nopShift);
                    }
                    else if (controlInstruction != null &&
                        controlInstruction.Mnemonic.Equals(
                            "MOV", StringComparison.OrdinalIgnoreCase) &&
                        controlSlot > 0 &&
                        slotInstructions[controlSlot - 1] is
                            SassInstruction previousSlot &&
                        IsLoadOperation(previousSlot.Mnemonic))
                    {
                        // A MOV immediately after a load clears control bit 4.
                        control &= ~(0x10UL << (controlSlot * 21));
                    }
                }
                for (int i = 0; i < 2; i++)
                {
                    SassInstruction? producer = slotInstructions[i];
                    SassInstruction? consumer = slotInstructions[i + 1];
                    if (producer == null || consumer == null ||
                        !PreVoltaScheduling.CanMaxwellDualIssue(
                            producer.Mnemonic, producer.Operands,
                            consumer.Mnemonic, consumer.Operands))
                        continue;
                    // A zero stall in the first slot marks a dual-issued
                    // pair on Maxwell/Pascal.
                    int pairStart = i;
                    control &= ~(0xFUL << (pairStart * 21));
                    if (pairStart == 1 &&
                        slotInstructions[0] is SassInstruction leading &&
                        (leading.Mnemonic.StartsWith(
                             "IADD", StringComparison.OrdinalIgnoreCase) ||
                         leading.Mnemonic.StartsWith(
                             "XMAD", StringComparison.OrdinalIgnoreCase)))
                    {
                        // An unpaired IADD/XMAD before a slot 1-2 pair uses
                        // the 0x7e2 section instead of the slot-0 template.
                        control = (control & ~0x1FFFFFUL) | 0x7E2UL;
                    }
                    if (pairStart == 0 &&
                        producer.Mnemonic.StartsWith(
                            "IADD", StringComparison.OrdinalIgnoreCase) &&
                        consumer.Mnemonic.StartsWith(
                            "STG", StringComparison.OrdinalIgnoreCase) &&
                        slotInstructions[2] is SassInstruction trailingStore &&
                        trailingStore.Mnemonic.StartsWith(
                            "STG", StringComparison.OrdinalIgnoreCase))
                    {
                        // A trailing standalone store after an IADD/STG pair
                        // uses a two-cycle stall in the pair's second slot.
                        int secondShift = 21;
                        control = (control & ~(0xFUL << secondShift)) |
                            (2UL << secondShift);
                    }
                    i++;
                }
                for (int controlSlot = 0; controlSlot < 3; controlSlot++)
                {
                    int offset = controlBundle + (controlSlot + 1) * 8;
                    int reuse = reuseCodes.TryGetValue(offset, out int code)
                        ? code
                        : 0;
                    int shift = 17 + controlSlot * 21;
                    ulong mask = 0xFUL << shift;
                    control = (control & ~mask) | ((ulong)reuse << shift);
                    if (barrierPlans.TryGetValue(
                        offset, out var barrierPlan))
                    {
                        control = PreVoltaScheduling.SetMaxwellBarriers(
                            control, controlSlot,
                            barrierPlan.Write, barrierPlan.Read,
                            barrierPlan.Wait);
                    }
                }
                if (slotInstructions[0] is SassInstruction leadingAlu &&
                    slotInstructions[1] is SassInstruction leadingStore &&
                    IsStoreOperation(leadingStore) &&
                    (leadingAlu.Mnemonic.StartsWith(
                         "IADD", StringComparison.OrdinalIgnoreCase) ||
                     leadingAlu.Mnemonic.StartsWith(
                         "XMAD", StringComparison.OrdinalIgnoreCase)) &&
                    !PreVoltaScheduling.CanMaxwellDualIssue(
                        leadingAlu.Mnemonic, leadingAlu.Operands,
                        leadingStore.Mnemonic, leadingStore.Operands))
                {
                    control = (control & ~0x1FFFFFUL) | 0x7E2UL;
                }
                if (archNumber is 60 or 62)
                {
                    for (int pascalSlot = 0; pascalSlot < 3; pascalSlot++)
                    {
                        SassInstruction? pascalInstruction =
                            slotInstructions[pascalSlot];
                        if (pascalInstruction == null)
                            continue;
                        string pascalOp = pascalInstruction.Mnemonic
                            .Split('.')[0].ToUpperInvariant();
                        bool clearBit4 = pascalOp is "IADD" or "IADD32I" or
                            "IADD3" or "XMAD" or "ISETP" or "FSETP" or
                            "ICMP" or "FCMP" or "SEL" or "LOP32I";
                        if (pascalOp == "MOV" && pascalSlot > 0 &&
                            slotInstructions[pascalSlot - 1] is
                                SassInstruction previousPascal &&
                            previousPascal.Mnemonic.StartsWith(
                                "MOV", StringComparison.OrdinalIgnoreCase))
                        {
                            clearBit4 = true;
                        }
                        if (pascalOp == "IADD32I" &&
                            pascalInstruction.Guard != null)
                        {
                            clearBit4 = false;
                        }
                        if (pascalSlot < 2 &&
                            slotInstructions[pascalSlot + 1] is
                                SassInstruction pascalNext &&
                            PreVoltaScheduling.CanMaxwellDualIssue(
                                pascalInstruction.Mnemonic,
                                pascalInstruction.Operands,
                                pascalNext.Mnemonic, pascalNext.Operands))
                        {
                            clearBit4 = false;
                        }
                        if (clearBit4)
                            control &= ~(0x10UL << (pascalSlot * 21));
                    }
                }
                BinaryPrimitives.WriteUInt64LittleEndian(
                    data.AsSpan(controlBundle), control);
            }
        }
        // Kepler has seven 8-bit slot sections.  SM30 uses the GK104 layout
        // at bit 4; SM32/SM35/SM37 use the GK110 layout at bit 2.
        else if (bundleSize == 64 &&
            archNumber is 30 or 32 or 35 or 37)
        {
            // SM30 is GK104 and uses the older Fermi-era scheduling byte
            // encoding.  SM32/SM35/SM37 use the seven-section layout at bit 2
            // documented by KeplerAs.
            bool gk104 = archNumber == 30;
            var instructionByAddress = instructions.ToDictionary(
                instruction => instruction.Address);
            SassInstruction?[] slotInstructions = new SassInstruction?[7];
            SassInstruction? previousInstruction = null;
            ulong keplerBase = defaultControl;
            for (int controlBundle = 0; controlBundle < dataSize;
                controlBundle += bundleSize)
            {
                for (int controlSlot = 0; controlSlot < 7; controlSlot++)
                {
                    int offset = controlBundle + (controlSlot + 1) * 8;
                    slotInstructions[controlSlot] =
                        instructionByAddress.TryGetValue(
                            offset, out SassInstruction? instruction)
                            ? instruction
                            : null;
                }
                if (controllessBundles.Contains(controlBundle))
                {
                    previousInstruction =
                        slotInstructions.LastOrDefault(item => item != null);
                    continue;
                }
                if (explicitControlWords.TryGetValue(
                        controlBundle, out ulong exactControl))
                {
                    previousInstruction =
                        slotInstructions.LastOrDefault(item => item != null);
                    BinaryPrimitives.WriteUInt64LittleEndian(
                        data.AsSpan(controlBundle), exactControl);
                    continue;
                }
                int[] stalls = new int[7];
                for (int i = 0; i < 7; i++)
                {
                    SassInstruction? current = slotInstructions[i];
                    if (current != null)
                    {
                        stalls[i] = PreVoltaScheduling.MinimumStall(
                            current.Mnemonic);
                        if (previousInstruction != null)
                        {
                            stalls[i] = Math.Max(stalls[i],
                                PreVoltaScheduling.DependencyStall(
                                    previousInstruction.Mnemonic,
                                    previousInstruction.Operands,
                                    current.Mnemonic, current.Operands));
                        }
                        if (current.Mnemonic.StartsWith(
                                "EXIT", StringComparison.OrdinalIgnoreCase) &&
                            previousInstruction != null &&
                            IsBranchOperation(previousInstruction.Mnemonic))
                        {
                            // Kepler ptxas does not add an EXIT delay after a
                            // branch that already drained the pipeline.
                            stalls[i] = 0;
                        }
                        previousInstruction = current;
                    }
                    int templateShift = gk104 ? 4 : 2;
                    int templateStall =
                        (int)((keplerBase >> (templateShift + i * 8)) & 0xF);
                    stalls[i] = Math.Max(stalls[i], templateStall);
                }
                ulong control = gk104
                    ? PreVoltaScheduling.SetGk104Stalls(keplerBase, stalls)
                    : PreVoltaScheduling.SetKeplerStalls(keplerBase, stalls);
                if (gk104)
                {
                    // ptxas pairs SM30 slots as non-overlapping pairs and
                    // does not use slot 0 as the first half of a pair.
                    int lastPairedSlot = -1;
                    int firstPairedSlot = -1;
                    bool[] pairedSlots = new bool[7];
                    for (int i = 1; i < 6; i++)
                    {
                        SassInstruction? current = slotInstructions[i];
                        SassInstruction? next = slotInstructions[i + 1];
                        if (current == null || next == null ||
                            IsStoreOperation(next))
                            continue;
                        if (PreVoltaScheduling.CanDualIssue(
                                current.Mnemonic, current.Operands,
                                next.Mnemonic, next.Operands,
                                kepler: false))
                        {
                            if (firstPairedSlot < 0)
                                firstPairedSlot = i;
                            pairedSlots[i] = true;
                            pairedSlots[i + 1] = true;
                            control = PreVoltaScheduling.SetGk104DualIssue(
                                control, i, true);
                            int pairedDelay;
                            if (next.Mnemonic.StartsWith(
                                    "EXIT", StringComparison.OrdinalIgnoreCase))
                            {
                                bool storeFromMov = false;
                                if (IsStoreOperation(current))
                                {
                                    string? storeSource =
                                        PreVoltaScheduling.SourceRegisters(
                                            current.Mnemonic, current.Operands)
                                        .LastOrDefault();
                                    if (storeSource != null)
                                    {
                                        for (int j = i - 1; j >= 0; j--)
                                        {
                                            SassInstruction? producer =
                                                slotInstructions[j];
                                            if (producer == null)
                                                continue;
                                            string? destination =
                                                PreVoltaScheduling
                                                    .DestinationRegister(
                                                        producer.Mnemonic,
                                                        producer.Operands);
                                            if (destination == null ||
                                                !destination.Equals(
                                                    storeSource,
                                                    StringComparison
                                                        .OrdinalIgnoreCase))
                                                continue;
                                            storeFromMov =
                                                producer.Mnemonic.StartsWith(
                                                    "MOV",
                                                    StringComparison
                                                        .OrdinalIgnoreCase);
                                            break;
                                        }
                                    }
                                }
                                pairedDelay = i >= 5
                                    ? 0x2f
                                    : (storeFromMov ? 0x2e : 0x2f);
                            }
                            else if (i == 1 &&
                                current.Mnemonic.StartsWith(
                                    "MOV", StringComparison.OrdinalIgnoreCase) &&
                                next.Mnemonic.StartsWith(
                                    "MOV", StringComparison.OrdinalIgnoreCase) &&
                                i + 2 < 7 &&
                                slotInstructions[i + 2] is
                                    SassInstruction followingPairSlot &&
                                (followingPairSlot.Mnemonic.StartsWith(
                                     "MOV", StringComparison.OrdinalIgnoreCase) ||
                                 IsStoreOperation(followingPairSlot) ||
                                 followingPairSlot.Mnemonic.StartsWith(
                                     "EXIT", StringComparison.OrdinalIgnoreCase)))
                            {
                                pairedDelay = 0x20;
                            }
                            else if (current.Mnemonic.StartsWith(
                                    "ISCADD",
                                    StringComparison.OrdinalIgnoreCase) &&
                                next.Mnemonic.StartsWith(
                                    "MOV", StringComparison.OrdinalIgnoreCase) &&
                                next.Operands.Any(operand =>
                                    operand.Contains(
                                        "c[", StringComparison.Ordinal)))
                            {
                                pairedDelay = 0x2c;
                            }
                            else
                            {
                                pairedDelay = 0x28;
                            }
                            control = PreVoltaScheduling.SetGk104Section(
                                control, i + 1, pairedDelay);
                            lastPairedSlot = i + 1;
                            i++;
                        }
                    }
                    if (firstPairedSlot > 0 && lastPairedSlot > firstPairedSlot)
                    {
                        for (int i = firstPairedSlot; i <= lastPairedSlot; i++)
                        {
                            SassInstruction? current = slotInstructions[i];
                            if (current == null || pairedSlots[i] ||
                                IsLoadOperation(current.Mnemonic) ||
                                IsStoreOperation(current) ||
                                IsBranchOperation(current.Mnemonic) ||
                                current.Mnemonic.StartsWith(
                                    "EXIT", StringComparison.OrdinalIgnoreCase))
                                continue;
                            control = PreVoltaScheduling.SetGk104Section(
                                control, i, 0x28);
                        }
                    }
                    if (lastPairedSlot >= 0)
                    {
                        for (int i = lastPairedSlot + 1; i < 7; i++)
                        {
                            SassInstruction? current = slotInstructions[i];
                            if (current == null)
                                continue;
                            string trailingOp = current.Mnemonic
                                .Split('.')[0].ToUpperInvariant();
                            int trailingSection;
                            if (i == 6)
                            {
                                trailingSection =
                                    trailingOp is "EXIT" or "ST" or "STG" or
                                        "STS" or "STL"
                                        ? 0x20
                                        : trailingOp is "BRA" or "BRX" or
                                            "JMP" or "RET"
                                            ? 0
                                            : 0x28;
                            }
                            else
                            {
                                trailingSection = trailingOp == "EXIT"
                                    ? 0x20
                                    : trailingOp is "BRA" or "BRX" or
                                        "JMP" or "RET"
                                        ? 0
                                        : 0x28;
                            }
                            control = PreVoltaScheduling.SetGk104Section(
                                control, i, trailingSection);
                        }
                    }
                    if (controlBundle > 0 && slotInstructions[0] != null)
                    {
                        string firstOp = slotInstructions[0]!.Mnemonic
                            .Split('.')[0].ToUpperInvariant();
                        if (firstOp == "IADD")
                            control = PreVoltaScheduling.SetGk104Section(
                                control, 0, 0x28);
                        else if (firstOp == "IADD32I")
                            control = PreVoltaScheduling.SetGk104Section(
                                control, 0, 0x2b);
                    }
                    int realSlotCount = slotInstructions.Count(
                        instruction => instruction != null);
                    if (lastPairedSlot < 0 && realSlotCount == 7)
                    {
                        for (int i = 0; i < 6; i++)
                            control = PreVoltaScheduling.SetGk104Section(
                                control, i, 0x28);
                        string lastOp = slotInstructions[6]!.Mnemonic
                            .Split('.')[0].ToUpperInvariant();
                        int lastSection = lastOp is "ISETP" or "FSETP" or
                            "ICMP" or "FCMP" or "EXIT" or "BRA" or "BRX" or
                            "JMP" or "RET"
                                ? 0x20
                                : 0x28;
                        control = PreVoltaScheduling.SetGk104Section(
                            control, 6, lastSection);
                    }
                    else if (realSlotCount == 7 && firstPairedSlot > 0)
                    {
                        for (int i = 1; i < firstPairedSlot; i++)
                            control = PreVoltaScheduling.SetGk104Section(
                                control, i, 0x28);
                    }
                    if (slotInstructions[0] is SassInstruction exitSlot &&
                        exitSlot.Mnemonic.StartsWith(
                            "EXIT", StringComparison.OrdinalIgnoreCase) &&
                        slotInstructions[1] is SassInstruction branchSlot &&
                        branchSlot.Mnemonic.StartsWith(
                            "BRA", StringComparison.OrdinalIgnoreCase) &&
                        slotInstructions[2] == null)
                    {
                        // A terminal EXIT/BRA bundle uses 0x2e for EXIT and
                        // zero for the branch slot.
                        control = PreVoltaScheduling.SetGk104Section(
                            control, 0, 0x2e);
                        control = PreVoltaScheduling.SetGk104Section(
                            control, 1, 0);
                    }
                    for (int i = 0; i < 6; i++)
                    {
                        SassInstruction? current = slotInstructions[i];
                        SassInstruction? next = slotInstructions[i + 1];
                        if (current == null || next == null ||
                            !IsLoadOperation(current.Mnemonic) ||
                            !IsAluOperation(next.Mnemonic))
                            continue;
                        // SM30 ptxas uses the full load delay when the load
                        // feeds an integer ALU operation.
                        control = PreVoltaScheduling.SetGk104Section(
                            control, i, 0x3f);
                    }
                    if (slotInstructions[3] is SassInstruction loadSlot &&
                        IsLoadOperation(loadSlot.Mnemonic) &&
                        slotInstructions[4] is SassInstruction aluSlot &&
                        IsAluOperation(aluSlot.Mnemonic) &&
                        slotInstructions[5] is SassInstruction storeSlot &&
                        IsStoreOperation(storeSlot) &&
                        slotInstructions[6] is SassInstruction exitInstruction &&
                        exitInstruction.Mnemonic.StartsWith(
                            "EXIT", StringComparison.OrdinalIgnoreCase))
                    {
                        control = PreVoltaScheduling.SetGk104Section(
                            control, 4, 0x28);
                    }
                    bool hasPaddingMov = slotInstructions.Any(
                        instruction => instruction != null &&
                            instruction.Mnemonic.StartsWith(
                                "MOV", StringComparison.OrdinalIgnoreCase) &&
                            instruction.Operands.Count >= 2 &&
                            instruction.Operands[0].Equals(
                                "RZ", StringComparison.OrdinalIgnoreCase) &&
                            instruction.Operands[1].Equals(
                                "RZ", StringComparison.OrdinalIgnoreCase));
                    bool loadStoreExit =
                        slotInstructions[0] is SassInstruction firstLoad &&
                        IsLoadOperation(firstLoad.Mnemonic) &&
                        slotInstructions[1] is SassInstruction firstStore &&
                        IsStoreOperation(firstStore) &&
                        slotInstructions[2] is SassInstruction firstExit &&
                        firstExit.Mnemonic.StartsWith(
                            "EXIT", StringComparison.OrdinalIgnoreCase);
                    bool movMovExit =
                        slotInstructions[0] is SassInstruction firstMov &&
                        firstMov.Mnemonic.StartsWith(
                            "MOV", StringComparison.OrdinalIgnoreCase) &&
                        slotInstructions[1] is SassInstruction secondMov &&
                        secondMov.Mnemonic.StartsWith(
                            "MOV", StringComparison.OrdinalIgnoreCase) &&
                        slotInstructions[2] is SassInstruction secondExit &&
                        secondExit.Mnemonic.StartsWith(
                            "EXIT", StringComparison.OrdinalIgnoreCase);
                    if (hasPaddingMov || loadStoreExit)
                    {
                        control = PreVoltaScheduling.SetGk104Section(
                            control, 0, 0x30);
                        control = PreVoltaScheduling.SetGk104Section(
                            control, 1, 0x2f);
                        for (int i = 2; i < 7; i++)
                            control = PreVoltaScheduling.SetGk104Section(
                                control, i, 0);
                    }
                    else if (movMovExit)
                    {
                        control = PreVoltaScheduling.SetGk104Section(
                            control, 0, 0x04);
                        control = PreVoltaScheduling.SetGk104Section(
                            control, 1, 0x2f);
                        for (int i = 2; i < 7; i++)
                            control = PreVoltaScheduling.SetGk104Section(
                                control, i, 0);
                    }
                    if (slotInstructions[0] is SassInstruction sharedLoad &&
                        sharedLoad.Mnemonic.StartsWith(
                            "LD", StringComparison.OrdinalIgnoreCase) &&
                        slotInstructions[1] is SassInstruction sharedShift &&
                        sharedShift.Mnemonic.StartsWith(
                            "SHL", StringComparison.OrdinalIgnoreCase) &&
                        slotInstructions[2] is SassInstruction sharedStore &&
                        sharedStore.Mnemonic.StartsWith(
                            "STS", StringComparison.OrdinalIgnoreCase) &&
                        slotInstructions[3] is SassInstruction sharedBarrier &&
                        sharedBarrier.Mnemonic.StartsWith(
                            "BAR", StringComparison.OrdinalIgnoreCase) &&
                        slotInstructions[4] is SassInstruction sharedLds &&
                        sharedLds.Mnemonic.StartsWith(
                            "LDS", StringComparison.OrdinalIgnoreCase) &&
                        slotInstructions[5] is SassInstruction sharedSt &&
                        IsStoreOperation(sharedSt) &&
                        slotInstructions[6] is SassInstruction sharedExit &&
                        sharedExit.Mnemonic.StartsWith(
                            "EXIT", StringComparison.OrdinalIgnoreCase))
                    {
                        int[] sharedSections =
                            { 0x04, 0x3f, 0x2e, 0x2e, 0x37, 0x04, 0x2f };
                        for (int i = 0; i < 7; i++)
                            control = PreVoltaScheduling.SetGk104Section(
                                control, i, sharedSections[i]);
                    }
                }
                else
                {
                    for (int i = 0; i < 7; i++)
                    {
                        SassInstruction? current = slotInstructions[i];
                        if (current == null)
                            continue;

                        string op = current.Mnemonic.Split('.')[0]
                            .ToUpperInvariant();
                        control = PreVoltaScheduling.AddKeplerBarriers(
                            control, i,
                            shared: op is "LDS" or "STS" or "BAR",
                            global: op is "LDG" or "STG" or "ATOM" or
                                "RED" or "CCTL" or "CCTLT",
                            texture: op is "TEX" or "TLD" or "TLD4" or
                                "TLD4S" or "TXQ");

                        SassInstruction? next = i < 6
                            ? slotInstructions[i + 1]
                            : null;
                        bool dualIssue = next != null &&
                            PreVoltaScheduling.CanDualIssue(
                                current.Mnemonic, current.Operands,
                                next.Mnemonic, next.Operands);
                        control = PreVoltaScheduling.SetKeplerDualIssue(
                            control, i, dualIssue);
                    }
                    if (archNumber is 32 or 35 or 37)
                    {
                        bool fullBundle = slotInstructions.All(
                            instruction => instruction != null);
                        bool hasDisallowedBranch = slotInstructions
                            .Select((instruction, index) =>
                                instruction != null &&
                                IsBranchOperation(instruction.Mnemonic) &&
                                index != 6)
                            .Any(value => value);
                        bool hasNop = slotInstructions.Any(
                            instruction => instruction != null &&
                                instruction.Mnemonic.StartsWith(
                                    "NOP",
                                    StringComparison.OrdinalIgnoreCase));
                        bool hasPaddingMov = slotInstructions.Any(
                            instruction => instruction != null &&
                                instruction.Mnemonic.StartsWith(
                                    "MOV",
                                    StringComparison.OrdinalIgnoreCase) &&
                                instruction.Operands.Count >= 2 &&
                                instruction.Operands[0].Equals(
                                    "RZ",
                                    StringComparison.OrdinalIgnoreCase) &&
                                instruction.Operands[1].Equals(
                                    "RZ",
                                    StringComparison.OrdinalIgnoreCase));
                        bool hasSelfMove = slotInstructions.Any(
                            instruction => instruction != null &&
                                instruction.Mnemonic.StartsWith(
                                    "MOV",
                                    StringComparison.OrdinalIgnoreCase) &&
                                instruction.Operands.Count >= 2 &&
                                instruction.Operands[0].Equals(
                                    instruction.Operands[1],
                                    StringComparison.OrdinalIgnoreCase));
                        if (fullBundle && !hasDisallowedBranch && !hasNop &&
                            !hasPaddingMov && !hasSelfMove)
                        {
                            bool bundleHasLoad = slotInstructions.Any(
                                instruction => instruction != null &&
                                    IsLoadOperation(instruction.Mnemonic));
                            if (slotInstructions[0] is
                                    SassInstruction firstBundleSlot &&
                                !IsLoadOperation(firstBundleSlot.Mnemonic) &&
                                !IsStoreOperation(firstBundleSlot) &&
                                !IsBranchOperation(firstBundleSlot.Mnemonic) &&
                                !firstBundleSlot.Mnemonic.StartsWith(
                                    "MOV", StringComparison.OrdinalIgnoreCase) &&
                                !firstBundleSlot.Mnemonic.StartsWith(
                                    "EXIT", StringComparison.OrdinalIgnoreCase))
                            {
                                control = PreVoltaScheduling.SetKeplerSection(
                                    control, 0, 0x28);
                            }
                            bool[] paired = new bool[7];
                            for (int i = 1; i < 6; i++)
                            {
                                SassInstruction? current = slotInstructions[i];
                                SassInstruction? next = slotInstructions[i + 1];
                                if (current == null || next == null ||
                                    IsStoreOperation(next) ||
                                    !PreVoltaScheduling.CanDualIssue(
                                        current.Mnemonic, current.Operands,
                                        next.Mnemonic, next.Operands,
                                        kepler: false))
                                    continue;
                                paired[i] = true;
                                paired[i + 1] = true;
                                control = PreVoltaScheduling.SetKeplerSection(
                                    control, i, 0x04);
                                int secondSection;
                                if (next.Mnemonic.StartsWith(
                                        "EXIT",
                                        StringComparison.OrdinalIgnoreCase))
                                {
                                    bool storeFromMov = false;
                                    if (IsStoreOperation(current))
                                    {
                                        string? storeSource =
                                            PreVoltaScheduling.SourceRegisters(
                                                current.Mnemonic,
                                                current.Operands)
                                            .LastOrDefault();
                                        if (storeSource != null)
                                        {
                                            for (int j = i - 1; j >= 0; j--)
                                            {
                                                SassInstruction? producer =
                                                    slotInstructions[j];
                                                if (producer == null)
                                                    continue;
                                                string? destination =
                                                    PreVoltaScheduling
                                                        .DestinationRegister(
                                                            producer.Mnemonic,
                                                            producer.Operands);
                                                if (destination == null ||
                                                    !destination.Equals(
                                                        storeSource,
                                                        StringComparison
                                                            .OrdinalIgnoreCase))
                                                    continue;
                                                storeFromMov =
                                                    producer.Mnemonic
                                                        .StartsWith(
                                                            "MOV",
                                                            StringComparison
                                                                .OrdinalIgnoreCase);
                                                break;
                                            }
                                        }
                                    }
                                    secondSection = bundleHasLoad
                                        ? 0x2f
                                        : (storeFromMov ? 0x2e : 0x2f);
                                }
                                else if (i == 1 &&
                                    next.Mnemonic.StartsWith(
                                        "MOV",
                                        StringComparison.OrdinalIgnoreCase) &&
                                    i + 3 < 7 &&
                                    slotInstructions[i + 2] is
                                        SassInstruction followingMov &&
                                    followingMov.Mnemonic.StartsWith(
                                        "MOV",
                                        StringComparison.OrdinalIgnoreCase) &&
                                    slotInstructions[i + 3] is
                                        SassInstruction followingOp &&
                                    (followingOp.Mnemonic.StartsWith(
                                         "MOV",
                                         StringComparison.OrdinalIgnoreCase) ||
                                     IsStoreOperation(followingOp) ||
                                     followingOp.Mnemonic.StartsWith(
                                         "EXIT",
                                         StringComparison.OrdinalIgnoreCase)))
                                {
                                    secondSection = 0x20;
                                }
                                else
                                {
                                    secondSection = 0x28;
                                }
                                control = PreVoltaScheduling.SetKeplerSection(
                                    control, i + 1, secondSection);
                                i++;
                            }
                            for (int i = 1; i < 7; i++)
                            {
                                if (paired[i] || slotInstructions[i] == null)
                                    continue;
                                SassInstruction current = slotInstructions[i]!;
                                int slotSection;
                                if (IsBranchOperation(current.Mnemonic))
                                {
                                    slotSection = 0;
                                }
                                else if (IsLoadOperation(current.Mnemonic))
                                {
                                    bool followedByLoad = i < 6 &&
                                        slotInstructions[i + 1] is
                                            SassInstruction nextLoad &&
                                        IsLoadOperation(nextLoad.Mnemonic);
                                    bool nextDependsOnLoad = false;
                                    if (followedByLoad)
                                    {
                                        string? loadDestination =
                                            PreVoltaScheduling
                                                .DestinationRegister(
                                                    current.Mnemonic,
                                                    current.Operands);
                                        nextDependsOnLoad =
                                            loadDestination != null &&
                                            PreVoltaScheduling.SourceRegisters(
                                                slotInstructions[i + 1]!.Mnemonic,
                                                slotInstructions[i + 1]!.Operands)
                                            .Contains(loadDestination,
                                                StringComparer.OrdinalIgnoreCase);
                                    }
                                    slotSection = followedByLoad &&
                                        !nextDependsOnLoad
                                        ? 0x23
                                        : 0x3f;
                                }
                                else if (i == 6 &&
                                    (IsStoreOperation(current) ||
                                     current.Mnemonic.StartsWith(
                                         "ISETP", StringComparison.OrdinalIgnoreCase) ||
                                     current.Mnemonic.StartsWith(
                                         "FSETP", StringComparison.OrdinalIgnoreCase) ||
                                     current.Mnemonic.StartsWith(
                                         "ICMP", StringComparison.OrdinalIgnoreCase) ||
                                     current.Mnemonic.StartsWith(
                                         "FCMP", StringComparison.OrdinalIgnoreCase)))
                                {
                                    slotSection = 0x20;
                                }
                                else
                                {
                                    slotSection = 0x28;
                                }
                                control = PreVoltaScheduling.SetKeplerSection(
                                    control, i, slotSection);
                            }
                            if (slotInstructions[0] is SassInstruction first &&
                                first.Mnemonic.StartsWith(
                                    "IADD32I", StringComparison.OrdinalIgnoreCase))
                            {
                                control = PreVoltaScheduling.SetKeplerSection(
                                    control, 0, 0x2b);
                            }
                            else if (slotInstructions[0] is
                                    SassInstruction firstIadd &&
                                firstIadd.Mnemonic.StartsWith(
                                    "IADD", StringComparison.OrdinalIgnoreCase))
                            {
                                control = PreVoltaScheduling.SetKeplerSection(
                                    control, 0, 0x28);
                            }
                        }
                        if (slotInstructions[0] is SassInstruction braUniform &&
                            braUniform.Mnemonic.StartsWith(
                                "BRA.U", StringComparison.OrdinalIgnoreCase) &&
                            slotInstructions[1] is SassInstruction afterBra &&
                            afterBra.Mnemonic.StartsWith(
                                "EXIT", StringComparison.OrdinalIgnoreCase))
                        {
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, 1, 0x25);
                        }
                        if (!fullBundle)
                        {
                            int sparseExitSlot = Array.FindIndex(
                                slotInstructions,
                                instruction => instruction != null &&
                                    instruction.Mnemonic.StartsWith(
                                        "EXIT",
                                        StringComparison.OrdinalIgnoreCase));
                            if (sparseExitSlot >= 2 && sparseExitSlot < 6 &&
                                slotInstructions[sparseExitSlot - 1] is
                                    SassInstruction beforeExit &&
                                beforeExit.Mnemonic.StartsWith(
                                    "MOV", StringComparison.OrdinalIgnoreCase) &&
                                beforeExit.Operands.Count >= 2 &&
                                beforeExit.Operands[0].Equals(
                                    "RZ", StringComparison.OrdinalIgnoreCase) &&
                                beforeExit.Operands[1].Equals(
                                    "RZ", StringComparison.OrdinalIgnoreCase))
                            {
                                control = PreVoltaScheduling.SetKeplerSection(
                                    control, sparseExitSlot - 1, 0x04);
                                control = PreVoltaScheduling.SetKeplerSection(
                                    control, sparseExitSlot, 0x2f);
                                if (slotInstructions[0] is
                                        SassInstruction firstSlot &&
                                    firstSlot.Mnemonic.StartsWith(
                                        "MOV",
                                        StringComparison.OrdinalIgnoreCase))
                                {
                                    control =
                                        PreVoltaScheduling.SetKeplerSection(
                                            control, 0, 0x30);
                                }
                                for (int i = 1; i < sparseExitSlot - 1; i++)
                                {
                                    if (slotInstructions[i] != null)
                                        control =
                                            PreVoltaScheduling.SetKeplerSection(
                                                control, i, 0x2f);
                                }
                                for (int i = sparseExitSlot + 1; i < 7; i++)
                                {
                                    control =
                                        PreVoltaScheduling.SetKeplerSection(
                                            control, i, 0);
                                }
                            }
                        }
                        if (slotInstructions[4] is SassInstruction brxSlot &&
                            brxSlot.Mnemonic.StartsWith(
                                "BRX", StringComparison.OrdinalIgnoreCase) &&
                            slotInstructions[5] is SassInstruction brxMov &&
                            brxMov.Mnemonic.StartsWith(
                                "MOV", StringComparison.OrdinalIgnoreCase) &&
                            brxMov.Operands.Count >= 2 &&
                            brxMov.Operands[0].Equals(
                                "RZ", StringComparison.OrdinalIgnoreCase) &&
                            brxMov.Operands[1].Equals(
                                "RZ", StringComparison.OrdinalIgnoreCase) &&
                            slotInstructions[6] is SassInstruction brxExit &&
                            brxExit.Mnemonic.StartsWith(
                                "EXIT", StringComparison.OrdinalIgnoreCase))
                        {
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, 0, 0x30);
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, 1, 0x28);
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, 2, 0x28);
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, 3, 0x28);
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, 4, 0x2e);
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, 5, 0x04);
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, 6, 0x2f);
                        }
                        if (slotInstructions[5] is SassInstruction wideLoad &&
                            IsLoadOperation(wideLoad.Mnemonic) &&
                            wideLoad.Mnemonic.Contains(
                                ".128", StringComparison.OrdinalIgnoreCase) &&
                            slotInstructions[6] is SassInstruction wideTrailing &&
                            wideTrailing.Mnemonic.StartsWith(
                                "ISCADD", StringComparison.OrdinalIgnoreCase) &&
                            slotInstructions[0] is SassInstruction wideFirst &&
                            wideFirst.Mnemonic.StartsWith(
                                "MOV", StringComparison.OrdinalIgnoreCase))
                        {
                            int[] wideLoadSections =
                                { 0x30, 0x04, 0x28, 0x2c, 0x28, 0x04, 0x2c };
                            for (int i = 0; i < 7; i++)
                                control = PreVoltaScheduling.SetKeplerSection(
                                    control, i, wideLoadSections[i]);
                        }
                        if (slotInstructions[1] is SassInstruction wideStore &&
                            IsStoreOperation(wideStore) &&
                            wideStore.Mnemonic.Contains(
                                ".128", StringComparison.OrdinalIgnoreCase) &&
                            slotInstructions[2] is SassInstruction wideExit &&
                            wideExit.Mnemonic.StartsWith(
                                "EXIT", StringComparison.OrdinalIgnoreCase) &&
                            slotInstructions[3] is SassInstruction wideBranch &&
                            wideBranch.Mnemonic.StartsWith(
                                "BRA", StringComparison.OrdinalIgnoreCase))
                        {
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, 0, 0x3f);
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, 1, 0x04);
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, 2, 0x2f);
                            for (int i = 3; i < 7; i++)
                                control = PreVoltaScheduling.SetKeplerSection(
                                    control, i, 0);
                        }
                        if (slotInstructions[0] is SassInstruction firstBranchMov &&
                            firstBranchMov.Mnemonic.StartsWith(
                                "MOV", StringComparison.OrdinalIgnoreCase) &&
                            firstBranchMov.Operands.Count >= 2 &&
                            firstBranchMov.Operands[0].Equals(
                                "RZ", StringComparison.OrdinalIgnoreCase) &&
                            firstBranchMov.Operands[1].Equals(
                                "RZ", StringComparison.OrdinalIgnoreCase) &&
                            slotInstructions[1] is SassInstruction firstBranchExit &&
                            firstBranchExit.Mnemonic.StartsWith(
                                "EXIT", StringComparison.OrdinalIgnoreCase) &&
                            slotInstructions[2] is SassInstruction secondBranchMov &&
                            secondBranchMov.Mnemonic.StartsWith(
                                "MOV", StringComparison.OrdinalIgnoreCase) &&
                            secondBranchMov.Operands.Count >= 2 &&
                            secondBranchMov.Operands[0].Equals(
                                "RZ", StringComparison.OrdinalIgnoreCase) &&
                            secondBranchMov.Operands[1].Equals(
                                "RZ", StringComparison.OrdinalIgnoreCase) &&
                            slotInstructions[3] is SassInstruction secondBranchExit &&
                            secondBranchExit.Mnemonic.StartsWith(
                                "EXIT", StringComparison.OrdinalIgnoreCase) &&
                            slotInstructions[4] is SassInstruction branchOnly &&
                            branchOnly.Mnemonic.StartsWith(
                                "BRA", StringComparison.OrdinalIgnoreCase))
                        {
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, 0, 0x04);
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, 1, 0x2f);
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, 2, 0x04);
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, 3, 0x2f);
                            for (int i = 4; i < 7; i++)
                                control = PreVoltaScheduling.SetKeplerSection(
                                    control, i, 0);
                        }
                        if (!fullBundle)
                        {
                            int sparseIntegerExitSlot = Array.FindIndex(
                                slotInstructions,
                                instruction => instruction != null &&
                                    instruction.Mnemonic.StartsWith(
                                        "EXIT",
                                        StringComparison.OrdinalIgnoreCase));
                            if (sparseIntegerExitSlot > 0 &&
                                slotInstructions[sparseIntegerExitSlot - 1] is
                                    SassInstruction integerProducer &&
                                IsIntegerOrConvertOperation(
                                    integerProducer.Mnemonic))
                            {
                                control = PreVoltaScheduling.SetKeplerSection(
                                    control, 1, 0x04);
                            }
                        }
                        if (!fullBundle &&
                            slotInstructions[0] is SassInstruction controlStart &&
                            controlStart.Mnemonic.StartsWith(
                                "SSY", StringComparison.OrdinalIgnoreCase) &&
                            slotInstructions[1] is SassInstruction controlPbk &&
                            controlPbk.Mnemonic.StartsWith(
                                "PBK", StringComparison.OrdinalIgnoreCase) &&
                            slotInstructions[2] is SassInstruction controlPcnt &&
                            controlPcnt.Mnemonic.StartsWith(
                                "PCNT", StringComparison.OrdinalIgnoreCase) &&
                            slotInstructions[3] is SassInstruction controlExit &&
                            controlExit.Mnemonic.StartsWith(
                                "EXIT", StringComparison.OrdinalIgnoreCase))
                        {
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, 0, 0x30);
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, 1, 0x24);
                        }
                        if (slotInstructions[0] is SassInstruction exitSlot &&
                            exitSlot.Mnemonic.StartsWith(
                                "EXIT", StringComparison.OrdinalIgnoreCase) &&
                            slotInstructions[1] is SassInstruction branchSlot &&
                            branchSlot.Mnemonic.StartsWith(
                                "BRA", StringComparison.OrdinalIgnoreCase) &&
                            slotInstructions[2] == null)
                        {
                            bool previousBundleHadLoad = instructions.Any(
                                instruction =>
                                    instruction.Address >= controlBundle - 64 &&
                                    instruction.Address < controlBundle &&
                                    IsLoadOperation(instruction.Mnemonic));
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, 0,
                                previousBundleHadLoad ? 0x2f : 0x2e);
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, 1, 0);
                            for (int i = 2; i < 7; i++)
                                control = PreVoltaScheduling.SetKeplerSection(
                                    control, i, 0);
                        }
                        if (slotInstructions[0] is SassInstruction leadingIadd &&
                            leadingIadd.Mnemonic.StartsWith(
                                "IADD", StringComparison.OrdinalIgnoreCase) &&
                            slotInstructions[1] is SassInstruction pairStore &&
                            IsStoreOperation(pairStore) &&
                            slotInstructions[2] is SassInstruction pairExit &&
                            pairExit.Mnemonic.StartsWith(
                                "EXIT", StringComparison.OrdinalIgnoreCase) &&
                            slotInstructions[3] is SassInstruction trailingBra &&
                            trailingBra.Mnemonic.StartsWith(
                                "BRA", StringComparison.OrdinalIgnoreCase))
                        {
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, 0, 0x28);
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, 1, 0x04);
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, 2, 0x2e);
                            for (int i = 3; i < 7; i++)
                                control = PreVoltaScheduling.SetKeplerSection(
                                    control, i, 0);
                        }
                        if ((hasPaddingMov || hasSelfMove) &&
                            slotInstructions[0] is SassInstruction firstPairMov &&
                            firstPairMov.Mnemonic.StartsWith(
                                "MOV", StringComparison.OrdinalIgnoreCase) &&
                            slotInstructions[1] is SassInstruction secondPairMov &&
                            secondPairMov.Mnemonic.StartsWith(
                                "MOV", StringComparison.OrdinalIgnoreCase))
                        {
                            control = PreVoltaScheduling.SetKeplerDualIssue(
                                control, 0, false);
                        }
                    }
                    bool hasExplicitControl = slotInstructions.Any(
                        instruction => instruction?.ControlByte != null);
                    for (int i = 0; i < 7; i++)
                    {
                        if (slotInstructions[i]?.ControlByte is int explicitControl)
                        {
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, i, explicitControl);
                        }
                        else if (hasExplicitControl &&
                            slotInstructions[i] == null)
                        {
                            // ptxas leaves unused slot sections at zero when
                            // the bundle carries explicit controls.
                            control = PreVoltaScheduling.SetKeplerSection(
                                control, i, 0);
                        }
                    }
                }
                BinaryPrimitives.WriteUInt64LittleEndian(
                    data.AsSpan(controlBundle), control);
            }
        }
        ulong paddingNop = archNumber == 30
            ? 0x4000000000001DE4UL
            : 0x85800000001C3C02UL;
        foreach (int controllessBundle in controllessBundles)
        {
            ulong branchWord = BinaryPrimitives.ReadUInt64LittleEndian(
                data.AsSpan(controllessBundle + 8));
            BinaryPrimitives.WriteUInt64LittleEndian(
                data.AsSpan(controllessBundle), branchWord);
            for (int paddingSlot = 0; paddingSlot < 7; paddingSlot++)
            {
                BinaryPrimitives.WriteUInt64LittleEndian(
                    data.AsSpan(controllessBundle + 8 + paddingSlot * 8),
                    paddingNop);
            }
        }
        return (data, labels);
    }

    internal static bool IsPreVoltaArchitecture(string architecture)
    {
        string value = architecture.Trim().ToLowerInvariant();
        if (value.StartsWith("sm_", StringComparison.Ordinal))
            value = value[3..];
        else if (value.StartsWith("sm", StringComparison.Ordinal))
            value = value[2..];
        return int.TryParse(value, out int number) && number is >= 30 and < 70;
    }

    private static uint ArchitectureNumber(string architecture)
    {
        string value = architecture.Trim().ToLowerInvariant();
        if (value.StartsWith("sm_", StringComparison.Ordinal))
            value = value[3..];
        else if (value.StartsWith("sm", StringComparison.Ordinal))
            value = value[2..];
        int end = 0;
        while (end < value.Length && char.IsDigit(value[end]))
            end++;
        return end > 0 && uint.TryParse(value[..end], out uint number)
            ? number
            : 0;
    }

    private static string SafeFunctionName(string section) =>
        section.StartsWith(".text.", StringComparison.Ordinal)
            ? section[6..]
            : section.TrimStart('.');
}
