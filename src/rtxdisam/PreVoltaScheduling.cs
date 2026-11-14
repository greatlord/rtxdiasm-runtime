namespace rtxdisam;

// ---------------------------------------------------------------------------
// Small pre-Volta timing model.
//
// The values mirror the Maxwell/Kepler scheduling classes used by maxas.  They
// are deliberately conservative: the encoder still uses validated control-word
// templates, but callers can use this data to build a dependency-aware
// scheduler later.
// ---------------------------------------------------------------------------

public readonly record struct PreVoltaTiming(
    int Latency,
    int Throughput,
    bool DualIssue,
    bool SupportsReuse,
    int ReadLatency = 0,
    int BarrierLatency = 0,
    int ReadHold = 0);

public static class PreVoltaScheduling
{
    // Maxwell/Kepler-style timing classes.  Memory latency is variable; the
    // values here are the fixed ALU/queue latency rather than cache latency.
    private static readonly PreVoltaTiming X32 = new(6, 1, false, true);
    private static readonly PreVoltaTiming X64 = new(2, 128, false, true);
    private static readonly PreVoltaTiming Shift = new(6, 2, false, true);
    private static readonly PreVoltaTiming Compare = new(13, 2, false, true);
    private static readonly PreVoltaTiming Quarter = new(8, 1, true, false);
    private static readonly PreVoltaTiming Memory =
        new(2, 1, true, false, ReadLatency: 4,
            BarrierLatency: 200, ReadHold: 20);
    private static readonly PreVoltaTiming S2r =
        new(2, 1, false, false, BarrierLatency: 25);
    private static readonly PreVoltaTiming Rro = new(2, 1, false, false);
    private static readonly PreVoltaTiming Vote = new(2, 1, false, false);
    private static readonly PreVoltaTiming Control = new(0, 1, false, false);

    public static bool TryGet(string mnemonic, out PreVoltaTiming timing)
    {
        string op = mnemonic.Split('.')[0].ToUpperInvariant();
        timing = op switch
        {
            "MOV" or "MOV32I" or "IADD" or "IADD32I" or "IADD3" or
                "IMAD" or "IMUL" or "XMAD" or "FADD" or "FADD32I" or
                "FMUL" or "FMUL32I" or "FFMA" or "LOP" or "LOP32I" or
                "LOP3" or "PRMT" or "SEL" => X32,

            "DADD" or "DDFMA" or "DFMA" or "DMUL" or "DMNMX" => X64,

            "SHL" or "SHR" or "SHF" or "BFE" or "BFI" or "ISCADD" or
                "LEA" or "IMNMX" => Shift,

            "ISETP" or "ISET" or "ICMP" or "FSETP" or "FSET" or
                "FCMP" or "DSETP" or "DSET" => Compare,

            "MUFU" or "RRO" or "F2F" or "F2I" or "I2F" or "I2I" => Quarter,

            "LD" or "ST" or "LDG" or "STG" or "LDS" or "STS" or
                "LDL" or "STL" or "LDC" or "ATOM" or "ATOMS" or
                "RED" or "BAR" or "MEMBAR" => Memory,

            "S2R" or "CS2R" => S2r,
            "VOTE" => Vote,

            "BRA" or "BRX" or "JMP" or "CALL" or "RET" or "EXIT" or
                "NOP" => Control,

            _ => default,
        };
        return timing != default;
    }

    public const ulong KeplerControlTag = 0x0800000000000000UL;

    // GK104/Fermi-era SM30 scheduling bytes are simpler than the
    // SM32/SM35/SM37 sections: 0x00 means a 32-cycle suspend, 0x04 means
    // dual-issue, and 0x20 | n means suspend for n cycles.  The seven sched
    // bytes start at bit 4 because bits 0-3 contain the sched opcode.
    public static ulong SetGk104Stalls(
        ulong controlWord, IReadOnlyList<int> stalls)
    {
        ulong result = controlWord;
        for (int slot = 0; slot < 7 && slot < stalls.Count; slot++)
        {
            int shift = 4 + slot * 8;
            ulong baseSection = (controlWord >> shift) & 0xFF;
            int templateStall = baseSection == 0
                ? 32
                : (baseSection & 0x20) != 0
                    ? (int)(baseSection & 0x1F)
                    : 0;
            int delay = Math.Max(
                Math.Max(0, stalls[slot]),
                templateStall);
            ulong section = delay >= 32
                ? 0UL
                : (ulong)(0x20u | (uint)Math.Min(31, delay));
            section |= baseSection & 0xC0UL;
            ulong mask = 0xFFUL << shift;
            result = (result & ~mask) | (section << shift);
        }
        return result;
    }

    public static ulong SetGk104Section(
        ulong controlWord, int slot, int section)
    {
        if (slot is < 0 or > 6)
            throw new ArgumentOutOfRangeException(nameof(slot));
        int shift = 4 + slot * 8;
        ulong mask = 0xFFUL << shift;
        return (controlWord & ~mask) |
            ((ulong)(section & 0xFF) << shift);
    }

    // GK104/SM30 uses the Fermi-era scheduling byte at bit 4.  A section
    // value of 0x04 marks the current slot as dual-issued with the next slot.
    public static ulong SetGk104DualIssue(
        ulong controlWord, int slot, bool dualIssue)
    {
        if (slot is < 0 or > 6)
            throw new ArgumentOutOfRangeException(nameof(slot));
        if (!dualIssue)
            return controlWord;
        int shift = 4 + slot * 8;
        ulong section = (controlWord >> shift) & 0xFF;
        section = (section & 0xC0UL) | 0x04UL;
        ulong mask = 0xFFUL << shift;
        return (controlWord & ~mask) | (section << shift);
    }

    // Kepler control words contain seven 8-bit sections starting at bit 2.
    // Each section has a 4-bit stall count in bits 0..3; the upper bits are
    // the shared/global/texture barrier flags and the single-issue bit.
    public static ulong SetKeplerStalls(
        ulong controlWord, IReadOnlyList<int> stalls)
    {
        ulong result = controlWord;
        for (int slot = 0; slot < 7 && slot < stalls.Count; slot++)
        {
            int shift = 2 + slot * 8;
            ulong mask = 0xFUL << shift;
            ulong value = (ulong)(Math.Max(0, Math.Min(15, stalls[slot])))
                << shift;
            result = (result & ~mask) | value;
        }
        return result;
    }

    // Set the shared/global/texture barrier flags for one Kepler section.
    // Flags are only ever added; the validated template values remain as
    // conservative minimums.
    public static ulong AddKeplerBarriers(
        ulong controlWord, int slot, bool shared, bool global, bool texture)
    {
        if (slot is < 0 or > 6)
            throw new ArgumentOutOfRangeException(nameof(slot));
        int shift = 2 + slot * 8;
        if (shared) controlWord |= 1UL << (shift + 4);
        if (global) controlWord |= 1UL << (shift + 6);
        if (texture) controlWord |= 1UL << (shift + 7);
        return controlWord;
    }

    public static ulong SetKeplerSection(
        ulong controlWord, int slot, int section)
    {
        if (slot is < 0 or > 6)
            throw new ArgumentOutOfRangeException(nameof(slot));
        int shift = 2 + slot * 8;
        ulong mask = 0xFFUL << shift;
        return (controlWord & ~mask) |
            ((ulong)(section & 0xFF) << shift);
    }

    // Select the Kepler single/dual-issue bit for one slot.  Clearing the
    // bit permits the slot to issue together with the following one.
    public static ulong SetKeplerDualIssue(
        ulong controlWord, int slot, bool dualIssue)
    {
        if (slot is < 0 or > 6)
            throw new ArgumentOutOfRangeException(nameof(slot));
        int bit = 2 + slot * 8 + 5;
        return dualIssue
            ? controlWord & ~(1UL << bit)
            : controlWord | (1UL << bit);
    }

    public static bool CanDualIssue(
        string producerMnemonic, IReadOnlyList<string> producerOperands,
        string consumerMnemonic, IReadOnlyList<string> consumerOperands,
        bool kepler = true)
    {
        string producer = producerMnemonic.Split('.')[0].ToUpperInvariant();
        string consumer = consumerMnemonic.Split('.')[0].ToUpperInvariant();

        bool gk104StoreExit = !kepler && consumer == "EXIT" &&
            (producer is "ST" or "STG" or "STS" or "STL");
        if (!gk104StoreExit &&
            (IsControlOperation(producer) || IsControlOperation(consumer) ||
             IsGlobalMemory(producer) || IsGlobalMemory(consumer) ||
             IsTextureOperation(producer) || IsTextureOperation(consumer)))
            return false;

        // 64-bit memory/FP operations occupy more than one issue slot.
        if (HasWideOperand(producer, producerOperands) ||
            HasWideOperand(consumer, consumerOperands))
            return false;

        if (IsMemoryOperation(producer) && IsMemoryOperation(consumer))
            return false;
        if (kepler &&
            producerOperands.Any(
                operand => operand.Contains("c[", StringComparison.Ordinal)) &&
            consumerOperands.Any(
                operand => operand.Contains("c[", StringComparison.Ordinal)))
            return false;

        // Kepler ptxas dual-issues normal MOV/MOV pairs, but not pairs
        // involving a self-move. Padding MOV RZ, RZ is pairable as the
        // second slot; the assembler keeps slot 0 single-issued when a
        // padding MOV is present.
        bool paddingMov = consumer == "MOV" &&
            consumerOperands.Count >= 2 &&
            consumerOperands[0].Trim().Equals("RZ",
                StringComparison.OrdinalIgnoreCase) &&
            consumerOperands[1].Trim().Equals("RZ",
                StringComparison.OrdinalIgnoreCase);
        bool producerSelfMove = producer == "MOV" &&
            producerOperands.Count >= 2 &&
            producerOperands[0].Trim().Equals(
                producerOperands[1].Trim(),
                StringComparison.OrdinalIgnoreCase);
        bool consumerSelfMove = consumer == "MOV" &&
            consumerOperands.Count >= 2 &&
            consumerOperands[0].Trim().Equals(
                consumerOperands[1].Trim(),
                StringComparison.OrdinalIgnoreCase);
        if (kepler && producer == "MOV" && consumer == "MOV" &&
            !paddingMov && (producerSelfMove || consumerSelfMove))
            return false;

        // GK104 ptxas keeps integer and FP ALU operations single-issued.
        if (!kepler &&
            (producer is "IADD" or "IADD32I" or "IMAD" or "IMUL") &&
            (consumer is "FADD" or "FADD32I" or "FMUL" or "FMUL32I" or
                "FFMA"))
            return false;

        // Specific illegal pairings from the GK104 dual-issue notes.
        if ((IsSetOperation(producer) && IsSetOperation(consumer)) ||
            (IsMultiply(producer) && IsMultiplyLike(consumer)) ||
            (IsMultiply(consumer) && IsMultiplyLike(producer)) ||
            (IsConvert(producer) && IsConvertLike(consumer)) ||
            (IsConvert(consumer) && IsConvertLike(producer)) ||
            (IsPopc(producer) && IsPopc(consumer)) ||
            (IsLogop(producer) && IsSelect(consumer)) ||
            (IsLogop(consumer) && IsSelect(producer)))
            return false;

        string? destination = DestinationRegister(
            producerMnemonic, producerOperands);
        if (destination != null &&
            (consumerOperands.Any(operand =>
                 RegisterNames(operand).Contains(destination)) ||
             DestinationRegister(consumerMnemonic, consumerOperands) ==
                 destination))
            return false;
        return true;
    }

    // Maxwell/Pascal dual issue is controlled by the first slot's stall
    // field: a zero stall marks the slot as issued with the following slot.
    // ptxas pairs adjacent instructions from different execution units.
    public static bool CanMaxwellDualIssue(
        string producerMnemonic, IReadOnlyList<string> producerOperands,
        string consumerMnemonic, IReadOnlyList<string> consumerOperands)
    {
        string producer = producerMnemonic.Split('.')[0].ToUpperInvariant();
        string consumer = consumerMnemonic.Split('.')[0].ToUpperInvariant();
        if (IsControlOperation(producer) || IsControlOperation(consumer))
            return false;
        if (HasWideOperand(producer, producerOperands) ||
            HasWideOperand(consumer, consumerOperands))
            return false;

        string producerUnit = MaxwellUnit(producer);
        string consumerUnit = MaxwellUnit(consumer);
        if (producerUnit == "?" || consumerUnit == "?")
            return false;
        // The observed ptxas pairs are ALU/compare/vote producers with
        // memory, S2R, or quarter-rate consumers.  Keep other combinations
        // single-issued rather than guessing.
        if (producerUnit is not ("ALU" or "Vote"))
            return false;
        if (consumerUnit is not ("Memory" or "S2R" or "Quarter"))
            return false;
        // ptxas does not pair a constant MOV with a following memory
        // operation, although MOV/S2R and IADD/STG pairs are common.
        if (producer == "MOV" && consumerUnit == "Memory")
            return false;
        if (producerUnit == consumerUnit)
            return false;
        if (producerMnemonic.StartsWith(
                "IADD.X", StringComparison.OrdinalIgnoreCase) &&
            consumerUnit == "Memory")
            return false;

        string? destination = DestinationRegister(
            producerMnemonic, producerOperands);
        if (destination != null &&
            (consumerOperands.Any(operand =>
                 RegisterNames(operand).Contains(destination)) ||
             DestinationRegister(consumerMnemonic, consumerOperands) ==
                 destination))
            return false;
        return true;
    }

    private static string MaxwellUnit(string op) => op switch
    {
        "MOV" or "MOV32I" or "IADD" or "IADD32I" or "IADD3" or
            "IMAD" or "IMUL" or "XMAD" or "FADD" or "FADD32I" or
            "FMUL" or "FMUL32I" or "FFMA" or "LOP" or "LOP32I" or
            "LOP3" or "PRMT" or "SEL" or "SHL" or "SHR" or "SHF" or
            "BFE" or "BFI" or "ISCADD" or "LEA" or "IMNMX" or
            "ISETP" or "ISET" or "ICMP" or "FSETP" or "FSET" or
            "FCMP" or "DSETP" or "DSET" => "ALU",

        "MUFU" or "RRO" or "F2F" or "F2I" or "I2F" or "I2I" or
            "FLO" => "Quarter",

        "LD" or "ST" or "LDG" or "STG" or "LDS" or "STS" or
            "LDL" or "STL" or "LDC" or "ATOM" or "RED" => "Memory",

        "S2R" or "CS2R" => "S2R",
        "VOTE" => "Vote",

        "TEX" or "TLD" or "TLD4" or "TLD4S" or "TXQ" or "SULD" or
            "SUST" or "SULDGA" or "SUSTGA" => "Texture",

        _ => "?",
    };

    private static bool IsSetOperation(string op) =>
        op is "ISET" or "ISETP" or "ICMP" or "FSET" or "FSETP" or
            "FCMP" or "DSET" or "DSETP";

    private static bool IsMultiply(string op) =>
        op is "IMAD" or "IMADSP" or "IMUL" or "XMAD";

    private static bool IsMultiplyLike(string op) =>
        IsMultiply(op) || op is "SHL" or "SHR" or "SHF" or "BFE" or
            "BFI" or "ISCADD" or "PRMT" or "POPC" or "FLO" or
            "F2F" or "F2I" or "I2F" or "I2I";

    private static bool IsConvert(string op) =>
        op is "F2F" or "F2I" or "I2F" or "I2I";

    private static bool IsConvertLike(string op) =>
        IsConvert(op) || IsMultiply(op) || op is "SHL" or "SHR" or
            "SHF" or "PRMT" or "POPC" or "FLO";

    private static bool IsPopc(string op) => op is "POPC" or "FLO";

    private static bool IsLogop(string op) => op is "LOP" or "LOP3";

    private static bool IsSelect(string op) => op == "SEL";

    private static bool IsControlOperation(string op) =>
        op is "BRA" or "BRX" or "JMP" or "CALL" or "RET" or "EXIT" or
            "BRK" or "CONT" or "KIL" or "LONGJMP" or "PLONGJMP" or
            "SSY" or "PBK" or "PCNT" or "BAR" or "MEMBAR";

    private static bool IsGlobalMemory(string op) =>
        op is "LDG" or "STG" or "ATOM" or "RED" or "CCTL" or "CCTLT";

    private static bool IsTextureOperation(string op) =>
        op is "TEX" or "TLD" or "TLD4" or "TLD4S" or "TXQ" or
            "SULD" or "SUST" or "SULDGA" or "SUSTGA";

    private static bool IsMemoryOperation(string op) =>
        op is "LD" or "ST" or "LDG" or "STG" or "LDS" or "STS" or
            "LDL" or "STL" or "LDC" or "ATOM" or "RED";

    private static bool HasWideOperand(
        string op, IReadOnlyList<string> operands) =>
        op.StartsWith('D') &&
            op is "DADD" or "DDFMA" or "DFMA" or "DMUL" or "DMNMX" or
                "DSET" or "DSETP" ||
        operands.Any(operand =>
            operand.Contains(".64", StringComparison.OrdinalIgnoreCase) ||
            operand.Contains(".128", StringComparison.OrdinalIgnoreCase));

    // Maxwell/Pascal control words are three 21-bit slot sections.  The low
    // four bits of each section are the stall count for that slot.
    public static ulong SetMaxwellStalls(
        ulong controlWord, IReadOnlyList<int> stalls)
    {
        ulong result = controlWord;
        for (int slot = 0; slot < 3 && slot < stalls.Count; slot++)
        {
            int shift = slot * 21;
            ulong mask = 0xFUL << shift;
            ulong value = (ulong)(Math.Max(0, Math.Min(15, stalls[slot])))
                << shift;
            result = (result & ~mask) | value;
        }
        return result;
    }

    // Set the write/read barrier and wait-mask fields in one Maxwell/Pascal
    // instruction-slot control section.  Barrier IDs are 1..6; null means the
    // "no barrier" encoding (7).
    public static ulong SetMaxwellBarriers(
        ulong controlWord, int slot, int? writeBarrier = null,
        int? readBarrier = null, int waitMask = 0)
    {
        if (slot is < 0 or > 2)
            throw new ArgumentOutOfRangeException(nameof(slot));
        int shift = slot * 21;
        ulong result = controlWord;
        ulong Write(int? id, int bit)
        {
            ulong value = id is >= 1 and <= 6
                ? (ulong)(id.Value - 1)
                : 7UL;
            ulong mask = 0x7UL << (shift + bit);
            return (result & ~mask) | (value << (shift + bit));
        }
        result = Write(writeBarrier, 5);
        result = Write(readBarrier, 8);
        ulong waitMaskBits = (ulong)(waitMask & 0x3F) << (shift + 11);
        ulong waitMaskField = 0x3FUL << (shift + 11);
        return (result & ~waitMaskField) | waitMaskBits;
    }

    // Branch and barrier operations need at least five control-word stall
    // slots even when there is no explicit data dependency.
    public static int MinimumStall(string mnemonic)
    {
        string op = mnemonic.Split('.')[0].ToUpperInvariant();
        return op is "BRA" or "BRX" or "JMP" or "CALL" or "RET" or
            "EXIT" or "BAR" or "MEMBAR" or "BRK" or "CONT" or "KIL" or
            "LONGJMP" or "PLONGJMP" or "SSY" or "PBK" or "PCNT"
            ? 5
            : 0;
    }

    public static int DependencyStall(
        string producerMnemonic, IReadOnlyList<string> producerOperands,
        string consumerMnemonic, IReadOnlyList<string> consumerOperands,
        int instructionsBetween = 0, bool maxwell = false)
    {
        string? destination = DestinationRegister(
            producerMnemonic, producerOperands);
        if (destination == null)
            return 0;
        foreach (string operand in consumerOperands)
        {
            foreach (string source in RegisterNames(operand))
            {
                if (source == destination)
                {
                    return DependencyStall(
                        producerMnemonic, consumerMnemonic,
                        instructionsBetween, maxwell);
                }
            }
        }
        return 0;
    }

    public static string? DestinationRegister(
        string mnemonic, IReadOnlyList<string> operands)
    {
        string op = mnemonic.Split('.')[0].ToUpperInvariant();
        if (op is "ST" or "STG" or "STS" or "STL" or "RED" or
            "BRA" or "BRX" or "JMP" or "CALL" or "RET" or "EXIT" or
            "NOP" or "BAR" or "MEMBAR" or "CCTL")
            return null;
        if (operands.Count == 0)
            return null;
        var match = System.Text.RegularExpressions.Regex.Match(
            operands[0], @"(R\d+|RZ)",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        return match.Success ? match.Value.ToUpperInvariant() : null;
    }

    private static IEnumerable<string> RegisterNames(string operand)
    {
        foreach (System.Text.RegularExpressions.Match match in
            System.Text.RegularExpressions.Regex.Matches(
                operand, @"(R\d+|RZ)",
                System.Text.RegularExpressions.RegexOptions.IgnoreCase))
            yield return match.Value.ToUpperInvariant();
    }

    public static IEnumerable<string> SourceRegisters(
        string mnemonic, IReadOnlyList<string> operands)
    {
        int start = DestinationRegister(mnemonic, operands) == null ? 0 : 1;
        for (int i = start; i < operands.Count; i++)
        {
            foreach (string register in RegisterNames(operands[i]))
                yield return register;
        }
    }

    public static int[] EstimateStalls(
        IReadOnlyList<string> mnemonics, string? beforeFirst = null)
    {
        int[] stalls = new int[mnemonics.Count];
        if (mnemonics.Count == 0)
            return stalls;
        if (!string.IsNullOrEmpty(beforeFirst))
            stalls[0] = DependencyStall(beforeFirst, mnemonics[0]);
        for (int i = 1; i < mnemonics.Count; i++)
            stalls[i] = DependencyStall(mnemonics[i - 1], mnemonics[i]);
        return stalls;
    }

    // Conservative dependency delay: a consumer immediately following a
    // producer must wait for the producer's ALU/source latency.  Throughput
    // limitations are reported separately by same-unit instructions.
    public static int DependencyStall(
        string producerMnemonic, string consumerMnemonic,
        int instructionsBetween = 0, bool maxwell = false)
    {
        if (!TryGet(producerMnemonic, out var producer))
            return 0;
        if (instructionsBetween < 0)
            instructionsBetween = 0;
        string consumerOp = consumerMnemonic.Split('.')[0].ToUpperInvariant();
        if (maxwell &&
            (consumerOp is "ST" or "STG" or "STS" or "STL" or "RED"))
            return Math.Max(0, 1 - instructionsBetween);
        int producerLatency = producer.ReadLatency > 0
            ? producer.ReadLatency
            : producer.Latency;
        int remaining = producerLatency - instructionsBetween;
        if (TryGet(consumerMnemonic, out var consumer) &&
            producer.Throughput > 1 &&
            !consumer.DualIssue)
        {
            remaining = Math.Max(
                remaining,
                producer.Throughput - instructionsBetween);
        }
        return Math.Max(0, remaining);
    }
}
