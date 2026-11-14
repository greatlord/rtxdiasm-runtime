using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace rtxdisam;

public sealed record PtxParameter(string Name, string Type, int Offset, int Size)
{
    // 0 = the first u64 of a by-value struct, 1 = its second half.
    public int Part { get; init; }
}

// A module-scope data object referenced by a relocated constant-bank
// slot (`.rel.nv.constant4`).  The cubin keeps the bytes in
// `.nv.global.init`; ACuda serves a read-only per-thread image from the
// `.global` initializer, so the lifter emits the bytes and materializes
// `mov.u64 %rd, name` as the relocated 64-bit value.
public sealed record PtxGlobalImage(string Name, long Addend, byte[] Data);

// Translates decoded SASS IR into PTX text.  The translator is intentionally
// strict: unsupported instructions fail instead of producing silently wrong
// PTX.  ACuda owns PTX -> SPIR-V and the Vulkan host path.
public static class SassToPtx
{
    public static string Translate(
        SassSection section, string entry,
        IReadOnlyList<PtxParameter> parameters, string target = "sm_50",
        Func<int, uint?>? constantResolver = null, int sharedBytes = 0,
        Func<int, uint?>? constant2Resolver = null, int constant2Bytes = 0,
        bool emitFunctions = false,
        IReadOnlyList<SassSection>? linkedSections = null,
        Func<int, uint?>? constant4Resolver = null, int constant4Bytes = 0,
        IReadOnlyDictionary<int, string>? texrefs = null,
        IReadOnlyDictionary<int, PtxGlobalImage>? globalImages = null)
    {
        var emitter = new Emitter(
            section, entry, parameters, target, constantResolver, sharedBytes,
            constant2Resolver, constant2Bytes, emitFunctions, linkedSections,
            constant4Resolver, constant4Bytes, texrefs, globalImages);
        return emitter.Run();
    }

    private sealed class Emitter
    {
        private readonly SassSection _section;
        private readonly string _entry;
        private readonly string _target;
        private string _ptxVersion;
        private readonly IReadOnlyList<PtxParameter> _parameters;
        private readonly Func<int, uint?>? _constantResolver;
        private readonly int _sharedBytes;
        private readonly Func<int, uint?>? _constant2Resolver;
        private readonly int _constant2Bytes;
        private readonly Func<int, uint?>? _constant4Resolver;
        private readonly int _constant4Bytes;
        private readonly IReadOnlyDictionary<int, PtxGlobalImage> _globalImages;
        private bool _usesTrigSlowpath;
        private readonly bool _emitFunctions;
        private readonly IReadOnlyList<SassSection> _linkedSections;
        private readonly List<PtxFunctionAbi> _functions = new();
        private readonly Dictionary<string, PtxFunctionAbi> _functionsByTarget =
            new(StringComparer.Ordinal);
        private readonly List<string> _functionTexts = new();
        private int _functionCallSerial;
        private bool _dynamicLdc;
        private int _dynamicLdcBound;
        private int _dynamicLdcRangeStart;
        private int _paramBase;
        private int _paramBlockSize;
        private int _paramBlockOffset;
        private readonly List<string> _lines = new();
        private readonly Dictionary<string, string> _regs = new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, string> _pointerRegs = new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, string> _preds = new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, string> _labels = new(StringComparer.Ordinal);
        private readonly Dictionary<int, PtxParameter> _paramsByOffset = new();
        private readonly Dictionary<PtxParameter, string> _paramPointers = new();
        private readonly HashSet<string> _surfaceRefs =
            new(StringComparer.Ordinal);
        // Cubin texture descriptor index (the immediate in TLD/TLDS) to the
        // original ELF texref symbol name, recovered from .rel.nv.constant0.
        private readonly IReadOnlyDictionary<int, string> _texrefs;
        private readonly HashSet<string> _usedTexrefs =
            new(StringComparer.Ordinal);
        private readonly Dictionary<string, string> _linkedShared =
            new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, string> _linkedPredicates =
            new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, int> _sharedAddressConstants =
            new(StringComparer.OrdinalIgnoreCase);
        // Dynamic (`.extern`) shared memory: ptxas places unions/arrays
        // that outlive their scope after the static allocation and folds
        // the static base into the instruction offset or an address
        // register.  These maps describe that split.
        private bool _usesDynamicShared;
        private int _staticSharedBytes;
        private SassInstruction? _currentInstruction;
        private readonly Dictionary<(SassInstruction, string), long>
            _sharedOperandConstants = new();
        private readonly Dictionary<int, string> _jumpTableLabels = new();
        private readonly HashSet<int> _skipNodes = new();
        // Deferred high halves of a folded 64-bit add, keyed by the node
        // index of the .X instruction that completes the pair.
        private readonly Dictionary<int, (string Low, string High, string Pair)>
            _deferredPairHigh = new();
        // Deferred .X high word whose source registers are rewritten
        // between the two halves: the low addends are snapshotted and the
        // high sources are read at the .X site instead.
        private readonly Dictionary<int, List<string>> _deferredPairLines = new();
        private readonly HashSet<string> _usedRegs = new(StringComparer.OrdinalIgnoreCase);
        private readonly HashSet<string> _usedPreds = new(StringComparer.OrdinalIgnoreCase);
        private readonly HashSet<string> _paramValueLoads = new(StringComparer.OrdinalIgnoreCase);
        private bool _usesShared;
        private bool _usesLocalFrame;
        private bool _usesDynamicLocalFrame;
        private int _calleeFrameBytes;
        private bool _useCommonExit;
        private string _commonExitLabel = "L_exit_common";
        private int _nextTemp;
        private int _nextTempPred;
        private int _nextLabel;
        private int _nextSnapPred;
        private int _nextTempFloat;
        private int _nextTempDouble;
        private int _nextTempU32;
        private int _nextTempHalf;
        // Set while emitting a predicated predicate-destination
        // instruction wrapped in an explicit skip branch.
        private bool _suppressGuard;
        private bool _leaCarryValid;
        private readonly Dictionary<string, string> _pairRegs =
            new(StringComparer.OrdinalIgnoreCase);
        private readonly HashSet<string> _materializedPairs =
            new(StringComparer.OrdinalIgnoreCase);
        // SASS predicates are reused; guards must see the value that was
        // live at the instruction's address even if a later block is
        // emitted first.  Each definition gets a unique snapshot.
        private readonly Dictionary<SassInstruction, string> _guardSnapshot =
            new();
        // `@P0 BRA P1, target` carries a second predicate operand: the
        // branch is taken only when both P0 (instruction guard) and P1
        // hold.  Keep the snapshot of that operand alongside the guard.
        private readonly Dictionary<SassInstruction, string> _operandSnapshot =
            new();
        private readonly Dictionary<SassInstruction,
            List<(string Predicate, string Snapshot)>> _predSnapshotDefs = new();
        private string? _extraGuard;
        private Dictionary<string, string>? _inlineLabelOverrides;
        // Out-of-line NVCC f64 division helper: emit its body once and
        // return through a constant-address dispatch instead of
        // duplicating ~150 instructions at every call site (a Welford
        // kernel calls it 30+ times and would exceed the register cap).
        private string? _dispatchHelperLabel;
        private readonly HashSet<SassInstruction> _dispatchReturnInstructions =
            new();
        private readonly Dictionary<uint, string> _helperDispatchReturns = new();
        private int _inlineSerial;
        private readonly HashSet<string> _inlineCallStack =
            new(StringComparer.Ordinal);
        private readonly Dictionary<string, string> _inlineTargetLabels =
            new(StringComparer.Ordinal);
        private int _indirectDispatchDepth;
        private bool _topLevelLocalAccessSeen;
        private bool _topLevelFrameBaseMoved;
        private IReadOnlyList<PtxFunctionArgument> _functionArguments =
            Array.Empty<PtxFunctionArgument>();
        private IReadOnlyList<string> _functionReturns =
            Array.Empty<string>();
        private string _functionName = "";

        private sealed record PtxFunctionArgument(
            string Register, string? HighRegister, bool IsPointer);

        private sealed record PtxFunctionAbi(
            string Name, string Target, int LabelIndex, int BodyStart,
            int BodyEnd, int ReturnIndex,
            IReadOnlyList<PtxFunctionArgument> Arguments,
            IReadOnlyList<string> Returns);

        private static string MinimumPtxVersion(string target)
        {
            // Emit at least the PTX ISA version that first accepted the
            // selected target. PTX 6.0 remains sufficient for older SM
            // targets, preserving the existing listings.
            string t = target.ToLowerInvariant();
            if (t.StartsWith("sm_107", StringComparison.Ordinal))
                return "9.4";
            if (t.StartsWith("sm_110", StringComparison.Ordinal) ||
                t == "sm_88")
                return "9.0";
            if (t.StartsWith("sm_100f", StringComparison.Ordinal) ||
                t.StartsWith("sm_101f", StringComparison.Ordinal) ||
                t.StartsWith("sm_103", StringComparison.Ordinal) ||
                t.StartsWith("sm_120", StringComparison.Ordinal) ||
                t.StartsWith("sm_121", StringComparison.Ordinal))
                return "8.8";
            if (t.StartsWith("sm_100", StringComparison.Ordinal) ||
                t.StartsWith("sm_101", StringComparison.Ordinal))
                return "8.6";
            if (t.StartsWith("sm_90a", StringComparison.Ordinal))
                return "8.0";
            if (t.StartsWith("sm_89", StringComparison.Ordinal) ||
                t.StartsWith("sm_90", StringComparison.Ordinal))
                return "7.8";
            if (t.StartsWith("sm_87", StringComparison.Ordinal))
                return "7.4";
            if (t.StartsWith("sm_86", StringComparison.Ordinal))
                return "7.1";
            if (t.StartsWith("sm_80", StringComparison.Ordinal))
                return "7.0";
            return "6.0";
        }

        private void RequirePtxVersion(string version)
        {
            if (Version.TryParse(_ptxVersion, out Version? current) &&
                Version.TryParse(version, out Version? required) &&
                current < required)
                _ptxVersion = version;
        }

        public Emitter(
            SassSection section, string entry,
            IReadOnlyList<PtxParameter> parameters, string target,
            Func<int, uint?>? constantResolver, int sharedBytes,
            Func<int, uint?>? constant2Resolver, int constant2Bytes,
            bool emitFunctions = false,
            IReadOnlyList<SassSection>? linkedSections = null,
            Func<int, uint?>? constant4Resolver = null, int constant4Bytes = 0,
            IReadOnlyDictionary<int, string>? texrefs = null,
            IReadOnlyDictionary<int, PtxGlobalImage>? globalImages = null)
        {
            _section = section;
            _entry = entry;
            _parameters = parameters;
            _target = ArchitectureNames.CanonicalizeLegacySm101(target);
            _ptxVersion = MinimumPtxVersion(_target);
            _constantResolver = constantResolver;
            _sharedBytes = sharedBytes;
            _constant2Resolver = constant2Resolver;
            _constant2Bytes = constant2Bytes;
            _constant4Resolver = constant4Resolver;
            _constant4Bytes = constant4Bytes;
            _globalImages = globalImages ??
                new Dictionary<int, PtxGlobalImage>();
            _emitFunctions = emitFunctions;
            _linkedSections = linkedSections ?? Array.Empty<SassSection>();
            _texrefs = texrefs ?? new Dictionary<int, string>();
            foreach (PtxParameter parameter in parameters)
            {
                if (WideWords(parameter) > 0)
                {
                    int words = WideWords(parameter);
                    for (int part = 0; part < words; part++)
                    {
                        PtxParameter q = ParamPart(parameter, part);
                        _paramsByOffset[q.Offset] = q;
                        _paramsByOffset[q.Offset + 4] = q;
                    }
                }
                else
                {
                    _paramsByOffset[parameter.Offset] = parameter;
                    if (parameter.Type == "u64")
                        _paramsByOffset[parameter.Offset + 4] = parameter;
                }
            }
        }

        public string Run()
        {
            CollectLabelsAndRegisters();
            CollectParameterValueLoads();
            CollectPredicateSnapshots();
            SelectPredicateGuardsDominance(_section.Nodes);
            CollectDynamicLdc();
            if (_usesLocalFrame)
                _pointerRegs["R1"] = "__callee_frame";
            if (_emitFunctions)
                CollectFunctions();
            DetectDynamicShared();
            _usesTrigSlowpath = DetectTrigSlowpathUse();
            EmitHeader();
            EmitRegisters();
            EmitParameters();
            List<SassNode> nodes = _section.Nodes.ToList();
            SkipDeviceRuntimeHelpers(nodes);
            _useCommonExit = nodes.Count(node =>
                node is SassInstruction exit &&
                (exit.Mnemonic.StartsWith("EXIT", StringComparison.OrdinalIgnoreCase) ||
                 exit.Mnemonic.StartsWith("RET", StringComparison.OrdinalIgnoreCase))) > 1 &&
                nodes.Any(node =>
                    node is SassInstruction brx &&
                    brx.Mnemonic.StartsWith("BRX", StringComparison.OrdinalIgnoreCase));
            if (_useCommonExit)
            {
                while (_labels.Values.Contains(_commonExitLabel))
                    _commonExitLabel += "_";
            }
            for (int index = 0; index < nodes.Count; index++)
            {
                if (_skipNodes.Contains(index))
                    continue;
                if (_deferredPairLines.Remove(index, out var deferredLines))
                {
                    _lines.AddRange(deferredLines);
                    continue;
                }
                if (_deferredPairHigh.Remove(index,
                        out var deferredPair))
                {
                    string highScratch = $"%rt{_nextTemp++}";
                    _lines.Add($"    mov.b64 {{{highScratch}, " +
                        $"{Reg(deferredPair.High)}}}, {deferredPair.Pair};");
                    _pairRegs[deferredPair.Low] = deferredPair.Pair;
                    _materializedPairs.Add(deferredPair.Pair);
                    continue;
                }
                if (nodes[index] is SassLabel label)
                {
                    // A label starts a new control-flow point.  A pair SSA
                    // temp cached in one block may be defined in a block
                    // that this path skipped, so it cannot be reused here.
                    // The physical halves are still live; reads will
                    // re-materialise the pair from them.
                    _pairRegs.Clear();
                    _lines.Add($"{LabelName(label.Name)}:");
                    continue;
                }
                if (nodes[index] is SassInstruction instruction)
                {
                    if (_usesLocalFrame && _topLevelLocalAccessSeen &&
                        !instruction.Mnemonic.StartsWith(
                            "LDL", StringComparison.OrdinalIgnoreCase) &&
                        !instruction.Mnemonic.StartsWith(
                            "STL", StringComparison.OrdinalIgnoreCase) &&
                        WrittenRegisters(instruction).Contains(
                            Reg("R1"), StringComparer.OrdinalIgnoreCase))
                        _topLevelFrameBaseMoved = true;
                    if (nodes[index].Address != 0 &&
                        _jumpTableLabels.TryGetValue(
                            nodes[index].Address, out string? jumpLabel))
                        _lines.Add($"{jumpLabel}:");
                    if (TryEmitStackCheckSkip(nodes, ref index))
                        continue;
                    if (TryEmitRuntimeCallTrap(nodes, ref index))
                        continue;
                    if (TryEmitAssertBlock(nodes, ref index))
                        continue;
                    if (TryEmitDynamicLdc(nodes, ref index))
                        continue;
                    if (TryEmitJumpTable(nodes, ref index))
                        continue;
                    if (TryEmitConstantLdc(nodes, ref index))
                        continue;
                    if (TryEmitUnboundedLdc(nodes, ref index))
                        continue;
                    if (TryEmitLea64(nodes, ref index))
                        continue;
                    if (TryEmitLeaHigh(nodes, ref index))
                        continue;
                    if (TryEmitLeaPointer(nodes, ref index))
                        continue;
                    if (TryEmitGenericAddPair(nodes, ref index))
                        continue;
                    if (TryEmitShift64(nodes, ref index, allowGap: true))
                        continue;
                    if (TryEmitIsubPair(nodes, ref index, allowGap: true))
                        continue;
                    if (TryEmitFastUnsignedDiv(nodes, ref index))
                        continue;
                    if (TryEmitIadd3Pair(nodes, ref index, allowGap: true))
                        continue;
                    if (TryEmitUadd3Pair(nodes, ref index))
                        continue;
                    if (TryEmitLeaCarryPair(nodes, ref index, allowGap: true))
                        continue;
                    if (TryEmitAddressAdd(nodes, ref index))
                        continue;
                    if (TryEmitInterleavedXmad(nodes, ref index))
                        continue;
                    if (TryEmitXmadMultiply(nodes, ref index))
                        continue;
                    if (TryEmitSync(nodes, ref index))
                        continue;
                    if (TryEmitNopSync(nodes, ref index))
                        continue;
                    if (TryEmitEarlyExitRegion(nodes, ref index))
                        continue;
                    if (TryEmitEmptyCall(nodes, ref index))
                        continue;
                    if (TryEmitIndirectCallDispatch(nodes, ref index))
                        continue;
                    if (TryEmitDeviceHelperCall(nodes, ref index))
                        continue;
                    if (TryEmitGuardedExitCall(nodes, ref index))
                        continue;
                    if (TryEmitFuncCall(nodes, ref index))
                        continue;
                    if (TryEmitLeafCallInline(nodes, ref index))
                        continue;
                    if (TryEmitRedundantBranch(nodes, ref index))
                        continue;
                    if (TryEmitSwitchSelect(nodes, ref index))
                        continue;
                    if ((instruction.Mnemonic.Equals("FADD32I", StringComparison.OrdinalIgnoreCase) ||
                         instruction.Mnemonic.Equals("FMUL32I", StringComparison.OrdinalIgnoreCase)) &&
                        instruction.Operands.Count > 0 &&
                        instruction.Operands[0].Contains(
                            ".CC", StringComparison.OrdinalIgnoreCase) &&
                        !IsCarryDead(nodes, index))
                        throw new NotSupportedException(
                            $"SASS instruction '{instruction.Mnemonic}' writes a live carry flag");
                    EmitInstruction(instruction);
                }
            }
            if (_useCommonExit)
            {
                _lines.Add($"{_commonExitLabel}:");
                _lines.Add("ret;");
            }
            _lines.Add("}");
            return ResolveRegisterDeclarations(
                string.Join('\n', _lines) + "\n");
        }

        // Assign a unique predicate snapshot to every SASS predicate
        // definition.  A guard is resolved against the snapshot that was
        // live at that instruction in SASS address order, not against the
        // mutable %pN register at emission time.
        private void CollectPredicateSnapshots()
        {
            Dictionary<string, string> live =
                new(StringComparer.OrdinalIgnoreCase);
            foreach (SassNode node in _section.Nodes)
            {
                if (node is not SassInstruction instruction)
                    continue;
                if (!string.IsNullOrWhiteSpace(instruction.Guard))
                {
                    string guardText = instruction.Guard.Trim();
                    bool negated = guardText.StartsWith(
                        "@!", StringComparison.Ordinal);
                    string predicate = guardText.TrimStart('@', '!');
                    if (!predicate.Equals("PT",
                            StringComparison.OrdinalIgnoreCase))
                    {
                        string name = live.TryGetValue(predicate,
                                                       out string? snap)
                            ? snap : Pred(predicate);
                        _guardSnapshot[instruction] =
                            (negated ? "@!" : "@") + name;
                    }
                }
                string? branchPred = BranchPredicateOperand(instruction);
                if (branchPred != null)
                {
                    bool negated = branchPred.StartsWith('!');
                    string predicate = branchPred.TrimStart('!');
                    string name = live.TryGetValue(predicate, out string? snap)
                        ? snap : Pred(predicate);
                    _operandSnapshot[instruction] =
                        (negated ? "@!" : "@") + name;
                }
                if (instruction.Operands.Count == 0)
                    continue;
                string op = instruction.Mnemonic.Split('.')[0]
                    .ToUpperInvariant();
                if (op == "R2P")
                {
                    // R2P writes the predicates selected by its mask
                    // from a register.  Only those definitions get a
                    // fresh snapshot; claiming the other predicates
                    // would shadow their real (possibly path-dependent)
                    // definitions.
                    uint mask = 0x7f;
                    if (instruction.Operands.Count >= 3)
                    {
                        string maskText = StripReuse(
                            instruction.Operands[2]);
                        if (!Regex.IsMatch(maskText, @"^R\d+$",
                                RegexOptions.IgnoreCase))
                        {
                            if (maskText.StartsWith(
                                    "0x", StringComparison.OrdinalIgnoreCase) &&
                                uint.TryParse(maskText[2..],
                                    NumberStyles.HexNumber,
                                    CultureInfo.InvariantCulture,
                                    out uint immediate))
                                mask = immediate & 0x7f;
                            else if (uint.TryParse(maskText,
                                NumberStyles.Integer,
                                CultureInfo.InvariantCulture,
                                out immediate))
                                mask = immediate & 0x7f;
                        }
                    }
                    var defs = new List<(string, string)>();
                    for (int p = 0; p < 7; p++)
                    {
                        if ((mask & (1u << p)) == 0)
                            continue;
                        string predicate = "P" + p;
                        string definition = $"%ps{_nextSnapPred++}";
                        live[predicate] = definition;
                        defs.Add((predicate, definition));
                    }
                    if (defs.Count > 0)
                        _predSnapshotDefs[instruction] = defs;
                    continue;
                }
                string destination = instruction.Operands[0].Trim();
                if (!Regex.IsMatch(destination, @"^P\d+$",
                        RegexOptions.IgnoreCase))
                    continue;
                string snapshot = $"%ps{_nextSnapPred++}";
                live[destination] = snapshot;
                _predSnapshotDefs[instruction] =
                    new List<(string, string)> { (destination, snapshot) };
            }
        }

        private static bool IsPredicateGuard(string? guard)
        {
            if (string.IsNullOrWhiteSpace(guard))
                return false;
            string g = guard.Trim();
            if (!g.StartsWith('@'))
                return false;
            g = g.TrimStart('@', '!');
            return Regex.IsMatch(g, @"^P\d+$", RegexOptions.IgnoreCase);
        }

        // `BRA`/`JMP` may carry a second predicate operand besides the
        // instruction guard: `@P0 BRA P1, target` branches only when
        // both hold.  Return that operand ("P1" or "!P1"), or null.
        private static string? BranchPredicateOperand(SassInstruction ins)
        {
            string op = ins.Mnemonic.Split('.')[0].ToUpperInvariant();
            if (op is not "BRA" and not "JMP")
                return null;
            foreach (string raw in ins.Operands)
            {
                string t = raw.Trim();
                if (!Regex.IsMatch(t, @"^!?(?:P\d+|PT)$",
                        RegexOptions.IgnoreCase))
                    continue;
                if (t.Equals("PT", StringComparison.OrdinalIgnoreCase) ||
                    t.Equals("!PT", StringComparison.OrdinalIgnoreCase))
                    continue;
                return t;
            }
            return null;
        }

        private static string GuardPredicate(string guard) =>
            guard.Trim().TrimStart('@', '!');

        private static bool IsBlockTerminator(SassInstruction ins)
        {
            string op = ins.Mnemonic.Split('.')[0].ToUpperInvariant();
            return op is "BRA" or "BRX" or "JMP" or "RET" or "EXIT";
        }

        private static string? BranchTargetName(SassInstruction ins)
        {
            string? t = ins.BranchTargets.FirstOrDefault()
                ?? ins.Operands.LastOrDefault();
            if (t == null)
                return null;
            t = t.Trim().Trim('`', '(', ')', ';');
            if (t.StartsWith(".L_", StringComparison.Ordinal))
                t = t[3..];
            return t;
        }

        // Resolve each predicate guard to the snapshot of the nearest
        // *dominating* definition rather than the nearest definition in
        // linear SASS order.  A linear choice can sit in a sibling arm
        // that the branch skips; its snapshot is then never written and
        // reads an uninitialised version variable.  Loop headers keep the
        // linear choice because their predicate is loop-carried.
        private void SelectPredicateGuardsDominance(List<SassNode> nodes)
        {
            int n = nodes.Count;
            if (n == 0)
                return;
            var bounds = new List<int> { 0 };
            for (int i = 0; i < n; i++)
            {
                if (nodes[i] is SassLabel && i > bounds[^1])
                    bounds.Add(i);
                if (nodes[i] is SassInstruction ins && IsBlockTerminator(ins) &&
                    i + 1 < n && i + 1 > bounds[^1])
                    bounds.Add(i + 1);
            }
            if (bounds[^1] != n)
                bounds.Add(n);
            int nb = bounds.Count - 1;
            if (nb < 2)
                return;
            var nodeBlock = new int[n];
            var labelBlock = new Dictionary<string, int>(
                StringComparer.OrdinalIgnoreCase);
            for (int b = 0; b < nb; b++)
                for (int i = bounds[b]; i < bounds[b + 1]; i++)
                {
                    nodeBlock[i] = b;
                    if (nodes[i] is SassLabel lb)
                    {
                        string nm = lb.Name;
                        if (nm.StartsWith(".L_", StringComparison.Ordinal))
                            nm = nm[3..];
                        labelBlock[nm] = b;
                    }
                }
            var succ = new List<int>[nb];
            var preds = new List<int>[nb];
            for (int b = 0; b < nb; b++)
            {
                succ[b] = new List<int>();
                preds[b] = new List<int>();
            }
            for (int b = 0; b < nb; b++)
            {
                bool conditional = false, term = false;
                int last = bounds[b + 1] - 1;
                if (last >= bounds[b] && nodes[last] is SassInstruction li &&
                    IsBlockTerminator(li))
                {
                    term = true;
                    string op = li.Mnemonic.Split('.')[0].ToUpperInvariant();
                    bool isBranch = op is "BRA" or "JMP";
                    conditional = isBranch &&
                                  !string.IsNullOrWhiteSpace(li.Guard);
                    if (isBranch)
                    {
                        string? tn = BranchTargetName(li);
                        if (tn != null && labelBlock.TryGetValue(tn, out int tb))
                            succ[b].Add(tb);
                    }
                }
                if ((!term || conditional) && b + 1 < nb)
                    succ[b].Add(b + 1);
                foreach (int t in succ[b])
                    preds[t].Add(b);
            }
            var dom = new HashSet<int>[nb];
            for (int b = 0; b < nb; b++)
                dom[b] = new HashSet<int>(Enumerable.Range(0, nb));
            dom[0].Clear();
            dom[0].Add(0);
            bool changed = true;
            while (changed)
            {
                changed = false;
                for (int b = 1; b < nb; b++)
                {
                    HashSet<int> nd;
                    if (preds[b].Count == 0)
                    {
                        nd = new HashSet<int> { b };
                    }
                    else
                    {
                        nd = new HashSet<int>(dom[preds[b][0]]);
                        for (int qi = 1; qi < preds[b].Count; qi++)
                            nd.IntersectWith(dom[preds[b][qi]]);
                        nd.Add(b);
                    }
                    if (!nd.SetEquals(dom[b]))
                    {
                        dom[b] = nd;
                        changed = true;
                    }
                }
            }
            var defs = new Dictionary<string,
                List<(int Index, int Block, string Snapshot)>>(
                    StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < n; i++)
            {
                if (nodes[i] is not SassInstruction di)
                    continue;
                if (!_predSnapshotDefs.TryGetValue(di, out var pds))
                    continue;
                foreach (var pd in pds)
                {
                    if (!defs.TryGetValue(pd.Predicate, out var list))
                    {
                        list = new List<(int, int, string)>();
                        defs[pd.Predicate] = list;
                    }
                    list.Add((i, nodeBlock[i], pd.Snapshot));
                }
            }
            // Resolve one predicate use at node `i` to the snapshot of
            // its nearest dominating definition.  A loop header keeps
            // the loop-carried (linear) choice.
            //
            // If the most recent (address-order) definition does not
            // dominate the use, the value at the use is path dependent
            // (e.g. the bitonic sort's NaN-handling comparator: one arm
            // defines P0 from a comparison, the other leaves it at a
            // constant).  SASS keeps the physical predicate register
            // across the join, so the use must read that register rather
            // than any single immutable snapshot.
            string? ResolveUse(string predicate, int i, int blk)
            {
                if (!defs.TryGetValue(predicate, out var list))
                    return null;
                if (preds[blk].Any(p => bounds[p] > bounds[blk]))
                    return null;
                (int Index, int Block, string Snapshot) best =
                    (-1, -1, string.Empty);
                (int Index, int Block, string Snapshot) linear =
                    (-1, -1, string.Empty);
                foreach (var d in list)
                {
                    if (d.Index >= i)
                        continue;
                    if (d.Index > linear.Index)
                        linear = d;
                    if (!dom[blk].Contains(d.Block))
                        continue;
                    if (d.Index > best.Index)
                        best = d;
                }
                if (best.Index < 0 || string.IsNullOrEmpty(best.Snapshot))
                    return null;
                if (linear.Index != best.Index)
                {
                    // Path-dependent value: the SASS physical predicate
                    // holds whichever reaching definition executed, so a
                    // guard on that register has the exact join
                    // semantics.  An immutable snapshot would be stale on
                    // the path that skipped its definition (e.g. the
                    // bitonic sort's valid-flag comparator).
                    return Pred(predicate);
                }
                return best.Snapshot;
            }
            for (int i = 0; i < n; i++)
            {
                if (nodes[i] is not SassInstruction gi)
                    continue;
                int blk = nodeBlock[i];
                if (IsPredicateGuard(gi.Guard))
                {
                    string snap = ResolveUse(
                        GuardPredicate(gi.Guard!), i, blk) ?? "";
                    if (snap.Length > 0)
                    {
                        string neg = gi.Guard!.Trim().StartsWith("@!",
                            StringComparison.Ordinal) ? "@!" : "@";
                        _guardSnapshot[gi] = neg + snap;
                    }
                }
                string? branchPred = BranchPredicateOperand(gi);
                if (branchPred != null)
                {
                    bool negated = branchPred.StartsWith('!');
                    string snap = ResolveUse(
                        branchPred.TrimStart('!'), i, blk) ?? "";
                    if (snap.Length > 0)
                        _operandSnapshot[gi] =
                            (negated ? "@!" : "@") + snap;
                }
            }
        }

        private void CollectParameterValueLoads()
        {
            List<SassNode> nodes = _section.Nodes.ToList();
            for (int i = 0; i < nodes.Count; i++)
            {
                if (nodes[i] is not SassInstruction mov ||
                    !mov.Mnemonic.Equals("MOV", StringComparison.OrdinalIgnoreCase) ||
                    mov.Operands.Count < 2)
                    continue;
                Match constant = Regex.Match(mov.Operands[1],
                    @"^c\[0x0\]\[0x([0-9A-Fa-f]+)\]$");
                if (!constant.Success)
                    continue;
                int offset = int.Parse(constant.Groups[1].Value,
                    NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                if (!_paramsByOffset.TryGetValue(offset, out PtxParameter? parameter) ||
                    parameter.Type != "u64")
                    continue;
                string destination = BaseReg(mov.Operands[0]);
                if (ParameterValueReadBeforeWrite(nodes, i, destination))
                    _paramValueLoads.Add(destination);
            }
        }

        private static bool ParameterValueReadBeforeWrite(
            List<SassNode> nodes, int index, string register)
        {
            for (int i = index + 1; i < nodes.Count; i++)
            {
                if (nodes[i] is SassLabel)
                    continue;
                if (nodes[i] is not SassInstruction instruction)
                    continue;
                int destination = DestinationIndex(instruction);
                for (int operand = 0;
                     operand < instruction.Operands.Count; operand++)
                {
                    if (operand == destination)
                        continue;
                    string text = instruction.Operands[operand];
                    if (text.TrimStart().StartsWith("[", StringComparison.Ordinal))
                        continue;
                    if (ContainsRegister(text, register))
                        return true;
                }
                if (destination >= 0 &&
                    destination < instruction.Operands.Count &&
                    ContainsRegister(instruction.Operands[destination], register))
                    return false;
            }
            return false;
        }

        private static bool TryParseConstant0Address(
            string operand, out string register, out int offset)
        {
            register = "";
            offset = 0;
            Match address = Regex.Match(operand.Trim(),
                @"^c\[0x0\]\[(R\d+)(?:([+-])(0x[0-9a-f]+|\d+))?\]$",
                RegexOptions.IgnoreCase);
            if (!address.Success)
                return false;
            register = BaseReg(address.Groups[1].Value);
            if (!address.Groups[2].Success)
                return true;
            uint magnitude = ParseImmediate(address.Groups[3].Value);
            if (magnitude > int.MaxValue)
                return false;
            offset = address.Groups[2].Value == "-"
                ? -(int)magnitude : (int)magnitude;
            return true;
        }

        private void CollectDynamicLdc()
        {
            _dynamicLdc = false;
            List<SassNode> nodes = _section.Nodes.ToList();
            int maxBound = int.MinValue;
            int minStart = int.MaxValue;
            bool anyDynamic = false;
            bool hasUnbounded = false;
            for (int i = 0; i < nodes.Count; i++)
            {
                if (nodes[i] is not SassInstruction ldc ||
                    ldc.Mnemonic.Contains(".128", StringComparison.OrdinalIgnoreCase) ||
                    ldc.Operands.Count < 2)
                    continue;
                bool uniform = ldc.Mnemonic.StartsWith(
                    "ULDC", StringComparison.OrdinalIgnoreCase);
                if (!uniform &&
                    !ldc.Mnemonic.StartsWith("LDC", StringComparison.OrdinalIgnoreCase))
                    continue;
                if (uniform)
                {
                    // c[0x0][URn+imm]: the uniform index selects a parameter
                    // word at run time ("ld.const" has no such addressing).
                    if (Regex.IsMatch(ldc.Operands[1].Trim(),
                            @"^c\[0x0\]\[UR\d+(?:[+-](?:0x[0-9A-Fa-f]+|\d+))?\]$",
                            RegexOptions.IgnoreCase))
                    {
                        anyDynamic = true;
                        hasUnbounded = true;
                    }
                    continue;
                }
                if (!TryParseConstant0Address(
                        ldc.Operands[1], out string register, out int addressOffset))
                    continue;
                if (IsConstantLdcAddress(nodes, i, register))
                    continue;
                anyDynamic = true;
                if (!TryGetDynamicLdcBound(
                        nodes, i, register, out int bound, out int rangeStart))
                {
                    // The base cannot be bounded statically (for example a
                    // ctaid-derived index).  The shared parameter copy is
                    // still correct because every access is clamped to the
                    // copied block at run time, so fall back to the full
                    // parameter extent instead of emitting ld.const into an
                    // empty constant image.
                    hasUnbounded = true;
                    continue;
                }
                long adjustedBound = (long)bound + addressOffset;
                long adjustedStart = rangeStart >= 0
                    ? (long)rangeStart + addressOffset : -1;
                if (adjustedBound <= 0 || adjustedBound > int.MaxValue ||
                    adjustedStart < int.MinValue ||
                    adjustedStart > int.MaxValue)
                {
                    hasUnbounded = true;
                    continue;
                }
                maxBound = Math.Max(maxBound, (int)adjustedBound);
                if (adjustedStart >= 0)
                    minStart = Math.Min(minStart, (int)adjustedStart);
            }
            if (!anyDynamic)
                return;
            _dynamicLdc = true;
            _dynamicLdcBound = maxBound;
            _dynamicLdcRangeStart = hasUnbounded
                ? -1
                : (minStart == int.MaxValue ? -1 : minStart);
            if (_parameters.Count == 0)
            {
                _dynamicLdc = false;
                return;
            }
            _paramBase = _parameters.Min(parameter => parameter.Offset);
            if (!hasUnbounded &&
                ((_dynamicLdcRangeStart >= 0 &&
                  _dynamicLdcRangeStart < _paramBase) ||
                 _dynamicLdcBound <= _paramBase))
            {
                _dynamicLdc = false;
                return;
            }
            int end = _parameters.Max(parameter => parameter.Offset + parameter.Size);
            if (!hasUnbounded && _dynamicLdcBound > _paramBase &&
                _dynamicLdcBound < end)
                end = _dynamicLdcBound;
            _paramBlockSize = ((end - _paramBase) + 15) & ~15;
            if (_paramBlockSize <= 0)
            {
                _dynamicLdc = false;
                return;
            }
            // The parameter image has its own Workgroup array (see
            // SharedConstant); no offset into __shared is needed.
            _paramBlockOffset = 0;
            _usesShared = true;
        }

        private static bool IsConstantLdcAddress(
            List<SassNode> nodes, int index, string register)
        {
            for (int i = index - 1; i >= 0; i--)
            {
                if (nodes[i] is SassLabel)
                    continue;
                if (nodes[i] is not SassInstruction instruction)
                    continue;
                int destination = DestinationIndex(instruction);
                if (destination >= 0 &&
                    destination < instruction.Operands.Count &&
                    ContainsRegister(instruction.Operands[destination], register))
                {
                    if (!string.IsNullOrWhiteSpace(instruction.Guard))
                        continue;
                    return instruction.Mnemonic.Equals(
                            "MOV32I", StringComparison.OrdinalIgnoreCase) &&
                        instruction.Operands.Count > 1 &&
                        TrySwitchImmediate(instruction.Operands[1], out _);
                }
                if (IsControlFlow(instruction))
                    return false;
            }
            return false;
        }

        private static bool TryGetDynamicLdcBound(
            List<SassNode> nodes, int index, string register, out int bound,
            out int rangeStart)
        {
            bound = 0;
            rangeStart = -1;
            if (TryGetDirectDynamicLdcBound(
                    nodes, index, register, out bound))
                return true;
            int definition = FindDefiningWrite(nodes, index, register);
            if (definition < 0 ||
                nodes[definition] is not SassInstruction instruction ||
                !string.IsNullOrWhiteSpace(instruction.Guard))
                return false;
            if (!TryResolveAffineAddress(
                    nodes, definition, instruction,
                    out int baseValue, out int scale, out string selector,
                    out int selectorDefinition))
                return false;
            if (!TryGetUpperBound(
                    nodes, selectorDefinition, selector, out int limit) ||
                limit <= 0)
                return false;
            long exclusive = (long)baseValue + (long)limit * scale;
            if (exclusive <= 0 || exclusive > int.MaxValue)
                return false;
            bound = (int)exclusive;
            rangeStart = baseValue;
            return true;
        }

        private static bool TryResolveAffineAddress(
            List<SassNode> nodes, int definition,
            SassInstruction instruction, out int baseValue, out int scale,
            out string selector, out int selectorDefinition)
        {
            baseValue = 0;
            scale = 0;
            selector = "";
            selectorDefinition = -1;
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            if (mnemonic.StartsWith("IMAD", StringComparison.Ordinal))
            {
                if (mnemonic.Contains(".HI", StringComparison.Ordinal) ||
                    mnemonic.Contains(".WIDE", StringComparison.Ordinal) ||
                    mnemonic.Contains(".MRG", StringComparison.Ordinal) ||
                    mnemonic.Contains(".CBCC", StringComparison.Ordinal) ||
                    instruction.Operands.Count < 4)
                    return false;
                string a = instruction.Operands[1];
                string b = instruction.Operands[2];
                string c = instruction.Operands[3];
                bool aImmediate = TrySwitchImmediate(a, out int scaleA);
                bool bImmediate = TrySwitchImmediate(b, out int scaleB);
                if (aImmediate == bImmediate)
                    return false;
                selector = BaseReg(aImmediate ? b : a);
                scale = aImmediate ? scaleA : scaleB;
                if (selector.Length == 0 || scale <= 0)
                    return false;
                if (TrySwitchImmediate(c, out int immediateBase))
                    baseValue = immediateBase;
                else if (!TryResolveConstantRegister(
                             nodes, definition, BaseReg(c), out baseValue))
                    return false;
                selectorDefinition = definition;
                return true;
            }

            if (!mnemonic.StartsWith("IADD", StringComparison.Ordinal) ||
                mnemonic.StartsWith("IADD3", StringComparison.Ordinal) ||
                mnemonic.Contains(".CC", StringComparison.Ordinal) ||
                mnemonic.Contains(".X", StringComparison.Ordinal) ||
                instruction.Operands.Count < 3)
                return false;
            string left = instruction.Operands[1];
            string right = instruction.Operands[2];
            bool leftImmediate = TrySwitchImmediate(left, out int leftValue);
            bool rightImmediate = TrySwitchImmediate(right, out int rightValue);
            if (leftImmediate && rightImmediate)
                return false;
            string valueOperand;
            if (leftImmediate)
            {
                baseValue = leftValue;
                valueOperand = right;
            }
            else if (rightImmediate)
            {
                baseValue = rightValue;
                valueOperand = left;
            }
            else if (TryResolveConstantRegister(
                         nodes, definition, BaseReg(left), out baseValue))
            {
                valueOperand = right;
            }
            else if (TryResolveConstantRegister(
                         nodes, definition, BaseReg(right), out baseValue))
            {
                valueOperand = left;
            }
            else
            {
                return false;
            }

            string valueRegister = BaseReg(valueOperand);
            if (valueRegister.Length == 0)
                return false;
            int valueDefinition = FindLastWrite(nodes, definition, valueRegister);
            if (valueDefinition >= 0 &&
                nodes[valueDefinition] is SassInstruction valueInstruction &&
                string.IsNullOrWhiteSpace(valueInstruction.Guard) &&
                valueInstruction.Mnemonic.StartsWith(
                    "SHL", StringComparison.OrdinalIgnoreCase) &&
                valueInstruction.Operands.Count >= 3 &&
                TrySwitchImmediate(
                    valueInstruction.Operands[2], out int shift) &&
                shift >= 0 && shift <= 30)
            {
                selector = BaseReg(valueInstruction.Operands[1]);
                if (selector.Length == 0)
                    return false;
                scale = 1 << shift;
                selectorDefinition = valueDefinition;
                return true;
            }
            selector = valueRegister;
            scale = 1;
            selectorDefinition = definition;
            return true;
        }

        private static int FindLastWrite(
            List<SassNode> nodes, int index, string register)
        {
            for (int i = index - 1; i >= 0; i--)
            {
                if (nodes[i] is SassLabel)
                    continue;
                if (nodes[i] is not SassInstruction instruction)
                    continue;
                int destination = DestinationIndex(instruction);
                if (destination >= 0 &&
                    destination < instruction.Operands.Count &&
                    ContainsRegister(instruction.Operands[destination], register))
                    return i;
                if (IsControlFlow(instruction))
                    return -1;
            }
            return -1;
        }

        private static bool TryGetDirectDynamicLdcBound(
            List<SassNode> nodes, int index, string register, out int bound)
        {
            bound = 0;
            for (int i = index - 1; i >= 0; i--)
            {
                if (nodes[i] is SassLabel)
                    continue;
                if (nodes[i] is not SassInstruction instruction)
                    continue;
                int destination = DestinationIndex(instruction);
                if (destination >= 0 &&
                    destination < instruction.Operands.Count &&
                    ContainsRegister(instruction.Operands[destination], register))
                {
                    if (!string.IsNullOrWhiteSpace(instruction.Guard))
                        continue;
                    return false;
                }
                if (instruction.Mnemonic.StartsWith(
                        "ISETP", StringComparison.OrdinalIgnoreCase) &&
                    instruction.Operands.Count >= 4 &&
                    (instruction.Mnemonic.Contains(
                         ".LT", StringComparison.OrdinalIgnoreCase) ||
                     instruction.Mnemonic.Contains(
                         ".LE", StringComparison.OrdinalIgnoreCase)) &&
                    ContainsRegister(instruction.Operands[2], register) &&
                    TrySwitchImmediate(instruction.Operands[3], out int limit))
                {
                    bound = limit + (instruction.Mnemonic.Contains(
                        ".LE", StringComparison.OrdinalIgnoreCase) ? 1 : 0);
                    return true;
                }
                if (IsControlFlow(instruction))
                {
                    if (instruction.Mnemonic.StartsWith(
                            "BRA", StringComparison.OrdinalIgnoreCase) &&
                        !string.IsNullOrWhiteSpace(instruction.Guard))
                        continue;
                    return false;
                }
            }
            return false;
        }

        private static bool TryGetUpperBound(
            List<SassNode> nodes, int index, string register, out int limit)
        {
            limit = 0;
            for (int i = index - 1; i >= 0; i--)
            {
                if (nodes[i] is SassLabel)
                    continue;
                if (nodes[i] is not SassInstruction instruction)
                    continue;
                int destination = DestinationIndex(instruction);
                if (destination >= 0 &&
                    destination < instruction.Operands.Count &&
                    ContainsRegister(instruction.Operands[destination], register))
                {
                    if (!string.IsNullOrWhiteSpace(instruction.Guard))
                        continue;
                    return false;
                }
                if (instruction.Mnemonic.StartsWith(
                        "ISETP", StringComparison.OrdinalIgnoreCase) &&
                    instruction.Operands.Count >= 4 &&
                    (instruction.Mnemonic.Contains(
                         ".LT", StringComparison.OrdinalIgnoreCase) ||
                     instruction.Mnemonic.Contains(
                         ".LE", StringComparison.OrdinalIgnoreCase)) &&
                    ContainsRegister(instruction.Operands[2], register) &&
                    TrySwitchImmediate(instruction.Operands[3], out int value))
                {
                    limit = value + (instruction.Mnemonic.Contains(
                        ".LE", StringComparison.OrdinalIgnoreCase) ? 1 : 0);
                    return limit > 0;
                }
                if (IsControlFlow(instruction))
                {
                    if (instruction.Mnemonic.StartsWith(
                            "BRA", StringComparison.OrdinalIgnoreCase) &&
                        !string.IsNullOrWhiteSpace(instruction.Guard))
                        continue;
                    return false;
                }
            }
            return false;
        }

        private static int FindDefiningWrite(
            List<SassNode> nodes, int index, string register)
        {
            for (int i = index - 1; i >= 0; i--)
            {
                if (nodes[i] is SassLabel)
                    continue;
                if (nodes[i] is not SassInstruction instruction)
                    continue;
                int destination = DestinationIndex(instruction);
                if (destination < 0 ||
                    destination >= instruction.Operands.Count ||
                    !ContainsRegister(instruction.Operands[destination], register))
                {
                    if (IsControlFlow(instruction))
                        return -1;
                    continue;
                }
                if (!string.IsNullOrWhiteSpace(instruction.Guard))
                    continue;
                return i;
            }
            return -1;
        }

        private static bool TryResolveConstantRegister(
            List<SassNode> nodes, int index, string register, out int value)
        {
            value = 0;
            for (int i = index - 1; i >= 0; i--)
            {
                if (nodes[i] is SassLabel)
                    continue;
                if (nodes[i] is not SassInstruction instruction)
                    continue;
                int destination = DestinationIndex(instruction);
                if (destination < 0 ||
                    destination >= instruction.Operands.Count ||
                    !ContainsRegister(instruction.Operands[destination], register))
                {
                    if (IsControlFlow(instruction))
                        return false;
                    continue;
                }
                if (!string.IsNullOrWhiteSpace(instruction.Guard))
                    return false;
                if ((instruction.Mnemonic.Equals(
                         "MOV32I", StringComparison.OrdinalIgnoreCase) ||
                     instruction.Mnemonic.Equals(
                         "MOV", StringComparison.OrdinalIgnoreCase)) &&
                    instruction.Operands.Count > 1 &&
                    TrySwitchImmediate(instruction.Operands[1], out value))
                    return true;
                return false;
            }
            return false;
        }

        private void CollectLabelsAndRegisters()
        {
            foreach (SassNode node in _section.Nodes)
            {
                if (node is SassLabel label)
                {
                    _labels[label.Name] = $"L_{_nextLabel++}";
                    continue;
                }
                if (node is not SassInstruction instruction)
                    continue;
                CollectInstructionUsage(instruction);
            }
            foreach (SassSection linked in _linkedSections)
            {
                if (ReferenceEquals(linked, _section))
                    continue;
                foreach (SassNode node in linked.Nodes)
                {
                    if (node is SassInstruction instruction)
                        CollectInstructionUsage(instruction);
                }
            }
            foreach (string reg in _usedRegs)
                _regs[reg] = "%r" + reg[1..];
            foreach (string pred in _usedPreds)
                _preds[pred] = "%p" + pred[1..];
        }

        private void CollectSurfaceReference(SassInstruction instruction)
        {
            string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
            if (op is not ("SULD" or "SUST" or "SURED") ||
                instruction.Operands.Count < 2)
                return;
            if (TrySwitchImmediate(instruction.Operands[^2], out int bank) &&
                TrySwitchImmediate(instruction.Operands[^1], out int index))
                _surfaceRefs.Add(SurfaceRefName(bank, index));
        }

        private void CollectTextureReference(SassInstruction instruction)
        {
            string op = instruction.Mnemonic.Split('.')[0]
                .ToUpperInvariant();
            if (op is not ("TLD" or "TLDS"))
                return;
            if (!TryTextureDescriptor(
                    instruction, out int descriptor, out _))
                return;
            if (_texrefs.TryGetValue(descriptor, out string? texref))
                _usedTexrefs.Add(texref);
        }

        private static bool TryTextureDescriptor(
            SassInstruction instruction, out int descriptor,
            out int geometryIndex)
        {
            descriptor = 0;
            geometryIndex = -1;
            for (int i = 0; i < instruction.Operands.Count; i++)
            {
                if (IsTextureGeometry(instruction.Operands[i]))
                {
                    geometryIndex = i;
                    break;
                }
            }
            if (geometryIndex <= 0)
                return false;
            return TrySwitchImmediate(
                instruction.Operands[geometryIndex - 1], out descriptor);
        }

        private static bool IsTextureGeometry(string operand)
        {
            string text = operand.Trim();
            return text.Equals("1D", StringComparison.OrdinalIgnoreCase) ||
                text.Equals("2D", StringComparison.OrdinalIgnoreCase) ||
                text.Equals("3D", StringComparison.OrdinalIgnoreCase) ||
                text.Equals("A1D", StringComparison.OrdinalIgnoreCase) ||
                text.Equals("A2D", StringComparison.OrdinalIgnoreCase) ||
                text.Equals("CUBE", StringComparison.OrdinalIgnoreCase) ||
                text.Equals("ACUBE", StringComparison.OrdinalIgnoreCase) ||
                text.Equals("2DMS", StringComparison.OrdinalIgnoreCase) ||
                text.Equals("A2DMS", StringComparison.OrdinalIgnoreCase);
        }

        private void CollectInstructionUsage(SassInstruction instruction)
        {
            CollectSurfaceReference(instruction);
            CollectTextureReference(instruction);
            if (instruction.Mnemonic.Split('.')[0].Equals(
                    "P2R", StringComparison.OrdinalIgnoreCase) &&
                (instruction.Operands.Count >= 4 ||
                 (instruction.Operands.Count == 2 &&
                  instruction.Operands[1].Trim().Equals(
                      "PR", StringComparison.OrdinalIgnoreCase))))
            {
                for (int i = 0; i < 7; i++)
                    _usedPreds.Add("P" + i);
            }
            bool asyncSharedStore = instruction.Mnemonic.StartsWith(
                    "ST", StringComparison.OrdinalIgnoreCase) &&
                instruction.Mnemonic.Contains(
                    "ASYNC", StringComparison.OrdinalIgnoreCase) &&
                !instruction.Mnemonic.Contains(
                    "GLOBAL", StringComparison.OrdinalIgnoreCase);
            bool bulkStore = instruction.Mnemonic.StartsWith(
                    "ST", StringComparison.OrdinalIgnoreCase) &&
                instruction.Mnemonic.Contains(
                    "BULK", StringComparison.OrdinalIgnoreCase);
            bool cpAsyncCopy = instruction.Mnemonic.StartsWith(
                    "CP.ASYNC", StringComparison.OrdinalIgnoreCase) &&
                !instruction.Mnemonic.Contains(
                    "COMMIT", StringComparison.OrdinalIgnoreCase) &&
                !instruction.Mnemonic.Contains(
                    "WAIT", StringComparison.OrdinalIgnoreCase);
            bool cpReduceBulk = instruction.Mnemonic.StartsWith(
                "CP.REDUCE.ASYNC.BULK", StringComparison.OrdinalIgnoreCase);
            bool multimemBulk = instruction.Mnemonic.StartsWith(
                "MULTIMEM.CP.", StringComparison.OrdinalIgnoreCase);
            bool tensormapShared = instruction.Mnemonic.StartsWith(
                    "TENSORMAP.", StringComparison.OrdinalIgnoreCase) &&
                instruction.Mnemonic.Contains(
                    "SHARED", StringComparison.OrdinalIgnoreCase);
            if (instruction.Mnemonic.StartsWith(
                    "CP.ASYNC.BULK", StringComparison.OrdinalIgnoreCase) ||
                cpReduceBulk)
                RequirePtxVersion("8.0");
            if (instruction.Mnemonic.StartsWith(
                    "CP.ASYNC.BULK.TENSOR", StringComparison.OrdinalIgnoreCase))
                RequirePtxVersion("8.6");
            if (multimemBulk)
                RequirePtxVersion("9.1");
            if (instruction.Mnemonic.StartsWith(
                    "TENSORMAP.", StringComparison.OrdinalIgnoreCase))
                RequirePtxVersion("8.6");
            if (instruction.Mnemonic.StartsWith(
                    "SPCOMPRESS", StringComparison.OrdinalIgnoreCase) ||
                instruction.Mnemonic.StartsWith(
                    "SPDECOMPRESS", StringComparison.OrdinalIgnoreCase))
                RequirePtxVersion("9.4");
            if (instruction.Mnemonic.StartsWith(
                    "SURED", StringComparison.OrdinalIgnoreCase) &&
                (instruction.Mnemonic.Contains(
                     "MIN", StringComparison.OrdinalIgnoreCase) ||
                 instruction.Mnemonic.Contains(
                     "MAX", StringComparison.OrdinalIgnoreCase)) &&
                (instruction.Mnemonic.Contains(
                     "64", StringComparison.OrdinalIgnoreCase) ||
                 instruction.Mnemonic.Contains(
                     "S64", StringComparison.OrdinalIgnoreCase) ||
                 instruction.Mnemonic.Contains(
                     "U64", StringComparison.OrdinalIgnoreCase)))
                RequirePtxVersion("8.1");
            if ((instruction.Mnemonic.StartsWith(
                     "CP.ASYNC", StringComparison.OrdinalIgnoreCase) ||
                 instruction.Mnemonic.StartsWith(
                     "PREFETCH", StringComparison.OrdinalIgnoreCase)) &&
                instruction.Mnemonic.Contains(
                    "L2::", StringComparison.OrdinalIgnoreCase))
                RequirePtxVersion("7.4");
            if (instruction.Mnemonic.StartsWith("LDS",
                    StringComparison.OrdinalIgnoreCase) ||
                instruction.Mnemonic.StartsWith("STS",
                    StringComparison.OrdinalIgnoreCase) ||
                asyncSharedStore || bulkStore || cpAsyncCopy ||
                cpReduceBulk || multimemBulk || tensormapShared)
                _usesShared = true;
            if (instruction.Mnemonic.StartsWith("LDL",
                    StringComparison.OrdinalIgnoreCase) ||
                instruction.Mnemonic.StartsWith("STL",
                    StringComparison.OrdinalIgnoreCase))
            {
                _usesLocalFrame = true;
                foreach (string operand in instruction.Operands)
                {
                    Match frame = Regex.Match(operand,
                        @"^\[R1(?:\+(0x[0-9A-Fa-f]+|\d+))?\]$",
                        RegexOptions.IgnoreCase);
                    if (!frame.Success)
                        continue;
                    int offset = frame.Groups[1].Success
                        ? (frame.Groups[1].Value.StartsWith(
                               "0x", StringComparison.OrdinalIgnoreCase)
                           ? int.Parse(frame.Groups[1].Value[2..],
                               NumberStyles.HexNumber,
                               CultureInfo.InvariantCulture)
                           : int.Parse(frame.Groups[1].Value,
                               CultureInfo.InvariantCulture))
                        : 0;
                    _calleeFrameBytes = Math.Max(
                        _calleeFrameBytes,
                        offset + (instruction.Mnemonic.Contains(
                            ".64", StringComparison.OrdinalIgnoreCase)
                            ? 16 : 8));
                }
            }
            if (HasDynamicFrameAdjustment(instruction))
                _usesDynamicLocalFrame = true;
            foreach (string operand in instruction.Operands)
            {
                foreach (Match match in Regex.Matches(
                    operand, @"\bR(\d+)\b",
                    RegexOptions.IgnoreCase))
                {
                    if (match.Value.Equals("RZ", StringComparison.OrdinalIgnoreCase))
                        continue;
                    _usedRegs.Add("R" + match.Groups[1].Value);
                }
                foreach (Match match in Regex.Matches(
                    operand, @"\bP(\d+)\b",
                    RegexOptions.IgnoreCase))
                    _usedPreds.Add("P" + match.Groups[1].Value);
            }
            if (instruction.Guard != null)
            {
                foreach (Match match in Regex.Matches(
                    instruction.Guard, @"\bP(\d+)\b",
                    RegexOptions.IgnoreCase))
                    _usedPreds.Add("P" + match.Groups[1].Value);
            }
        }

        private bool IsFrameAdjustment(SassInstruction instruction)
        {
            string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
            // IADD3 is the three-operand form ptxas emits for a dynamic
            // stack adjustment (R1, R1, Rc, RZ); missing it left a
            // later-inlined f64 division helper without its
            // __callee_dynamic_frame declaration.
            return op is "IADD" or "IADD32I" or "IADD3" &&
                instruction.Operands.Count >= 3 &&
                BaseReg(instruction.Operands[0]).Equals(
                    "R1", StringComparison.OrdinalIgnoreCase) &&
                ContainsRegister(instruction.Operands[1], "R1") &&
                !ContainsRegister(instruction.Operands[2], "R1");
        }

        private bool HasDynamicFrameAdjustment(SassInstruction instruction)
        {
            if (!IsFrameAdjustment(instruction))
                return false;
            string source = instruction.Operands[2].Trim();
            if (IsRz(source) || TryFrameImmediate(source, out _))
                return false;
            return Regex.IsMatch(source.TrimStart('-'), @"^R\d+$",
                RegexOptions.IgnoreCase);
        }

        private void EmitHeader()
        {
            _lines.Add($".version {_ptxVersion}");
            _lines.Add($".target {_target}");
            _lines.Add(".address_size 64");
            if (_usesShared)
            {
                if (_usesDynamicShared)
                {
                    // Dynamic (`extern __shared__`) accesses live after
                    // the static allocation.  ACuda keeps one Workgroup
                    // array per static variable and places all unsized
                    // extern arrays at the next 16-byte boundary, which
                    // is exactly the static size declared here.
                    if (_sharedBytes > 0)
                        _lines.Add(".shared .align 16 .b8 __shared[" +
                            _staticSharedBytes + "];");
                    _lines.Add(
                        ".extern .shared .align 16 .b8 __shared_dynamic[];");
                }
                else
                {
                    int sharedBytes = _sharedBytes > 0 ? _sharedBytes : 16384;
                    _lines.Add(
                        $".shared .align 16 .b8 __shared[{sharedBytes}];");
                }
            }
            if (_dynamicLdc)
                _lines.Add($".shared .align 16 .b8 __param_block[{_paramBlockSize}];");
            foreach (string texref in _usedTexrefs.OrderBy(
                         name => name, StringComparer.Ordinal))
                _lines.Add($".global .texref {texref};");
            foreach (string surface in _surfaceRefs.OrderBy(
                         name => name, StringComparer.Ordinal))
                _lines.Add($".global .surfref {surface};");
            EmitGlobalImages();
            if (_usesTrigSlowpath)
            {
                // libdevice's Payne-Hanek table (2/pi in 64-bit limbs).
                _lines.Add(".const .align 8 .b64 __rtx_trig_i2opi_d[18] = {");
                for (int i = 0; i < TrigI2opiD.Length; i++)
                {
                    string comma = i + 1 < TrigI2opiD.Length ? "," : "";
                    _lines.Add($"    0x{TrigI2opiD[i]:x16}{comma}");
                }
                _lines.Add("};");
            }
            _lines.AddRange(_functionTexts);
            string parameters = string.Join(", ",
                _parameters.Select(p => $".param .{p.Type} {p.Name}"));
            _lines.Add($".visible .entry {_entry}({parameters})");
            _lines.Add("{");
            if (_usesLocalFrame)
            {
                int frameBytes = _calleeFrameBytes > 0
                    ? (_calleeFrameBytes + 15) & ~15 : 256;
                _lines.Add($"    .local .align 16 .b8 __callee_frame[{frameBytes}];");
            }
            if (_usesDynamicLocalFrame)
                _lines.Add(
                    "    .local .align 16 .b8 __callee_dynamic_frame[65536];");
        }

        private void EmitParameters()
        {
            for (int i = 0; i < _parameters.Count; i++)
            {
                PtxParameter parameter = _parameters[i];
                if (parameter.Type == "u64")
                {
                    _lines.Add($"    .reg .b64 %rdp{i};");
                    _lines.Add($"    .reg .b64 %rpptr{i};");
                    _lines.Add($"    ld.param.u64 %rdp{i}, [{parameter.Name}];");
                    _lines.Add($"    mov.b64 %rpptr{i}, %rdp{i};");
                    _paramPointers[parameter] = $"%rpptr{i}";
                }
                else if (WideWords(parameter) > 0)
                {
                    int words = WideWords(parameter);
                    // Large by-value structs use ACuda's storage-buffer
                    // ABI and are loaded directly at [param+offset].
                    // Declare only word 0: the SASS entry reads the
                    // first pointer/word through %rdp{i}, while other
                    // fields use direct ld.param offsets.
                    if (words > 16)
                    {
                        _lines.Add($"    .reg .b64 %rdp{i};");
                        _lines.Add($"    .reg .b64 %rpptr{i};");
                        _lines.Add($"    ld.param.u64 %rdp{i}, " +
                            $"[{parameter.Name}];");
                        _lines.Add($"    mov.b64 %rpptr{i}, %rdp{i};");
                        _paramPointers[ParamPart(parameter, 0)] =
                            $"%rpptr{i}";
                        continue;
                    }
                    for (int part = 0; part < words; part++)
                    {
                        string suffix = part == 0 ? "" : $"p{part}";
                        _lines.Add($"    .reg .b64 %rdp{i}{suffix};");
                        _lines.Add($"    .reg .b64 %rpptr{i}{suffix};");
                        _lines.Add($"    ld.param.u64 %rdp{i}{suffix}, " +
                            $"[{parameter.Name}+{part * 8}];");
                        _lines.Add($"    mov.b64 %rpptr{i}{suffix}, " +
                            $"%rdp{i}{suffix};");
                        _paramPointers[ParamPart(parameter, part)] =
                            $"%rpptr{i}{suffix}";
                    }
                }
                else if (parameter.Type == "u32" || parameter.Type == "s32")
                {
                    _lines.Add($"    .reg .b32 %rpp{i};");
                    _lines.Add($"    ld.param.u32 %rpp{i}, [{parameter.Name}];");
                }
                else
                {
                    throw new NotSupportedException(
                        $"unsupported PTX parameter type '{parameter.Type}'");
                }
            }
            if (_dynamicLdc)
                EmitParameterBlockCopy();
        }

        private void EmitParameterBlockCopy()
        {
            string zero = $"%rt{_nextTemp++}";
            _lines.Add($"    mov.u32 {zero}, 0;");
            for (int offset = 0; offset < _paramBlockSize; offset += 4)
                _lines.Add(
                    $"    st.shared.u32 " +
                    $"{SharedConstant(_paramBlockOffset + offset)}, {zero};");
            foreach (PtxParameter parameter in _parameters)
            {
                if (parameter.Offset < _paramBase ||
                    parameter.Offset + parameter.Size >
                        _paramBase + _paramBlockSize)
                    continue;
                int offset = parameter.Offset - _paramBase;
                // Copy word-by-word as .u32.  Loading a 64-bit pointer
                // array with ld.param.u64 gives each word pointer lineage
                // and reclassifies the whole argument after the copy is
                // added; the shared image only needs the raw bytes.
                int bytes = parameter.Size > 0
                    ? parameter.Size
                    : (parameter.Type == "u64" ? 8 : 4);
                for (int address = 0; address < bytes; address += 4)
                {
                    string value = $"%rt{_nextTemp++}";
                    string source = address == 0
                        ? $"[{parameter.Name}]"
                        : $"[{parameter.Name}+0x{address:x}]";
                    _lines.Add($"    ld.param.u32 {value}, {source};");
                    _lines.Add(
                        $"    st.shared.u32 {SharedConstant(offset + address)}, {value};");
                }
            }
        }

        // The parameter-word image lives in its own Workgroup array so it
        // cannot overlap static shared data or the launch-sized dynamic
        // shared region, whose size is only known at run time.
        private static string SharedConstant(int offset)
        {
            return offset == 0
                ? "[__param_block]"
                : $"[__param_block+0x{offset:x}]";
        }

        // Load one 32-bit parameter word through the shared-memory copy
        // of the constant bank.  `address` is a byte offset into the SASS
        // constant bank; accesses outside the copied parameter block are
        // clamped to word 0 (they are not parameter reads).
        private void EmitSharedParamWord(string address, string destination,
                                         int addend)
        {
            string indexOperand = address;
            if (addend != 0)
            {
                string adjusted = $"%rt{_nextTemp++}";
                _lines.Add($"    add.s32 {adjusted}, {indexOperand}, {addend};");
                indexOperand = adjusted;
            }
            string offset = $"%rt{_nextTemp++}";
            string basePredicate = $"%pt{_nextTempPred++}";
            _lines.Add($"    setp.ge.u32 {basePredicate}, {indexOperand}, " +
                $"0x{_paramBase:x};");
            _lines.Add($"    @{basePredicate} sub.u32 {offset}, " +
                $"{indexOperand}, 0x{_paramBase:x};");
            _lines.Add($"    @!{basePredicate} mov.u32 {offset}, 0;");
            string boundPredicate = $"%pt{_nextTempPred++}";
            _lines.Add($"    setp.lt.u32 {boundPredicate}, {offset}, " +
                $"{_paramBlockSize};");
            _lines.Add($"    @!{boundPredicate} mov.u32 {offset}, 0;");
            string sharedOffset = $"%rdt{_nextTempDouble++}";
            string sharedBase = $"%rdt{_nextTempDouble++}";
            _lines.Add($"    cvt.u64.u32 {sharedOffset}, {offset};");
            _lines.Add($"    mov.u64 {sharedBase}, __param_block;");
            _lines.Add($"    add.u64 {sharedBase}, {sharedBase}, {sharedOffset};");
            _lines.Add($"    ld.shared.u32 {destination}, [{sharedBase}];");
        }

        // The fixed temporary banks below are only a starting point: a
        // large TensorIterator functor can need more than 384 %rt/%rdt
        // temporaries.  Rewrite the declarations once the body has been
        // emitted so the count always covers the registers that were used.
        private string ResolveRegisterDeclarations(string text)
        {
            // Declare each bank at the highest index the body actually
            // references.  A fixed-size bank (or a full 384-entry temp
            // bank) can push a large TensorIterator kernel over the
            // backend's 2048-register ceiling even though only a few
            // entries are live.
            int Count(string prefix)
            {
                int max = -1;
                foreach (Match m in Regex.Matches(text,
                             "%" + Regex.Escape(prefix) + @"(\d+)"))
                {
                    if (int.TryParse(m.Groups[1].Value, out int n) && n > max)
                        max = n;
                }
                return Math.Max(max + 1, 1);
            }
            text = Regex.Replace(text, @"\.reg \.b32 %r<\d+>;",
                $".reg .b32 %r<{Count("r")}>;");
            text = Regex.Replace(text, @"\.reg \.pred %p<\d+>;",
                $".reg .pred %p<{Count("p")}>;");
            text = Regex.Replace(text, @"\.reg \.pred %pt<\d+>;",
                $".reg .pred %pt<{Count("pt")}>;");
            text = Regex.Replace(text, @"\.reg \.pred %ps<\d+>;",
                $".reg .pred %ps<{Count("ps")}>;");
            text = Regex.Replace(text, @"\.reg \.f32 %ft<\d+>;",
                $".reg .f32 %ft<{Count("ft")}>;");
            text = Regex.Replace(text, @"\.reg \.f64 %dt<\d+>;",
                $".reg .f64 %dt<{Count("dt")}>;");
            text = Regex.Replace(text, @"\.reg \.b32 %rt<\d+>;",
                $".reg .b32 %rt<{Count("rt")}>;");
            text = Regex.Replace(text, @"\.reg \.u32 %ru<\d+>;",
                $".reg .u32 %ru<{Count("ru")}>;");
            text = Regex.Replace(text, @"\.reg \.b16 %rh<\d+>;",
                $".reg .b16 %rh<{Count("rh")}>;");
            text = Regex.Replace(text, @"\.reg \.u32 %ur<\d+>;",
                $".reg .u32 %ur<{Count("ur")}>;");
            text = Regex.Replace(text, @"\.reg \.b64 %rd<\d+>;",
                $".reg .b64 %rd<{Count("rd")}>;");
            text = Regex.Replace(text, @"\.reg \.b64 %rdt<\d+>;",
                $".reg .b64 %rdt<{Count("rdt")}>;");
            return text;
        }

        private void EmitRegisters()
        {
            if (_usedRegs.Count > 0)
                _lines.Add("    .reg .b32 %r<256>;");
            if (_usedPreds.Count > 0)
                _lines.Add("    .reg .pred %p<8>;");
            _lines.Add("    .reg .pred %pt<128>;");
            _lines.Add("    .reg .pred %up<64>;");
            _lines.Add("    .reg .pred %ps<1>;");
            _lines.Add("    .reg .f32 %ft<64>;");
            _lines.Add("    .reg .f64 %dt<64>;");
            _lines.Add("    .reg .b32 %rt<384>;");
            _lines.Add("    .reg .u32 %ru<32>;");
            _lines.Add("    .reg .b16 %rh<64>;");
            _lines.Add("    .reg .u32 %ur<32>;");
            _lines.Add("    .reg .b64 %rd<256>;");
            _lines.Add("    .reg .b64 %rdt<384>;");
        }

        private bool TryEmitLeaPointer(List<SassNode> nodes, ref int index)
        {
            if (index >= nodes.Count)
                return false;
            if (nodes[index] is not SassInstruction low ||
                low.Guard != null ||
                !low.Mnemonic.Equals("LEA", StringComparison.OrdinalIgnoreCase) ||
                low.Operands.Count < 3 ||
                !low.Operands[0].Trim().EndsWith(".CC",
                    StringComparison.OrdinalIgnoreCase) ||
                !IsRz(low.Operands[2]))
                return false;
            string source = BaseReg(low.Operands[1]);
            if (!_pointerRegs.TryGetValue(source, out string? pointer))
                return false;
            string lowDestination = BaseReg(low.Operands[0]);

            if (index + 1 < nodes.Count &&
                nodes[index + 1] is SassInstruction high &&
                high.Guard == null &&
                high.Mnemonic.StartsWith("LEA.HI.X",
                    StringComparison.OrdinalIgnoreCase) &&
                high.Operands.Count >= 5 &&
                IsRz(high.Operands[3]) &&
                source.Equals(BaseReg(high.Operands[2]),
                    StringComparison.OrdinalIgnoreCase))
            {
                string highDestination = BaseReg(high.Operands[1]);
                if (NeedsUnpack(nodes, index + 2,
                        lowDestination, highDestination))
                    return false;
                _pointerRegs[lowDestination] = pointer;
                index += 1;
                return true;
            }

            if (NeedsUnpack(nodes, index + 1,
                    lowDestination, lowDestination))
                return false;
            _pointerRegs[lowDestination] = pointer;
            return true;
        }

        private bool TryEmitLeaHigh(List<SassNode> nodes, ref int index)
        {
            if (index >= nodes.Count)
                return false;
            if (nodes[index] is not SassInstruction high ||
                high.Guard != null ||
                !high.Mnemonic.StartsWith("LEA.HI.X",
                    StringComparison.OrdinalIgnoreCase) ||
                high.Operands.Count < 5)
                return false;
            string source = BaseReg(high.Operands[2]);
            string destination = BaseReg(high.Operands[1]);
            if (!source.Equals(destination, StringComparison.OrdinalIgnoreCase) ||
                !_pointerRegs.ContainsKey(source))
                return false;
            if (NeedsUnpack(nodes, index + 1,
                    destination, destination))
                return false;
            return true;
        }

        private bool TryEmitIaddCarryPair(List<SassNode> nodes, ref int index)
        {
            if (index + 1 >= nodes.Count)
                return false;
            if (nodes[index] is not SassInstruction low ||
                nodes[index + 1] is not SassInstruction high)
                return false;
            if (low.Guard != null || high.Guard != null)
                return false;
            if (!low.Mnemonic.Equals("IADD", StringComparison.OrdinalIgnoreCase) ||
                low.Operands.Count < 3 ||
                !low.Operands[0].EndsWith(
                    ".CC", StringComparison.OrdinalIgnoreCase) ||
                !high.Mnemonic.StartsWith(
                    "IADD.X", StringComparison.OrdinalIgnoreCase) ||
                high.Operands.Count < 3)
                return false;
            string lowDestination = BaseReg(low.Operands[0]);
            string highDestination = BaseReg(high.Operands[0]);
            if (!highDestination.Equals(
                    NextRegister(lowDestination),
                    StringComparison.OrdinalIgnoreCase))
                return false;
            string lowA = BaseReg(low.Operands[1]);
            string lowB = BaseReg(low.Operands[2]);
            string highA = BaseReg(high.Operands[1]);
            string highB = BaseReg(high.Operands[2]);
            if (!Regex.IsMatch(lowA, @"^R\d+$", RegexOptions.IgnoreCase) ||
                !Regex.IsMatch(lowB, @"^R\d+$", RegexOptions.IgnoreCase) ||
                !Regex.IsMatch(highA, @"^R\d+$", RegexOptions.IgnoreCase) ||
                !Regex.IsMatch(highB, @"^R\d+$", RegexOptions.IgnoreCase))
                return false;
            string a = $"%rdt{_nextTempDouble++}";
            string b = $"%rdt{_nextTempDouble++}";
            string sum = $"%rdt{_nextTempDouble++}";
            _lines.Add($"    mov.b64 {a}, {{{Reg(lowA)}, {Reg(highA)}}};");
            _lines.Add($"    mov.b64 {b}, {{{Reg(lowB)}, {Reg(highB)}}};");
            _lines.Add($"    add.u64 {sum}, {a}, {b};");
            _lines.Add(
                $"    mov.b64 {{{Reg(lowDestination)}, " +
                $"{Reg(highDestination)}}}, {sum};");
            string pointer = $"%rdt{_nextTempDouble++}";
            _lines.Add($"    cvta.to.global.u64 {pointer}, {sum};");
            _pointerRegs[lowDestination] = pointer;
            index++;
            return true;
        }

        private bool TryEmitGenericAddPair(List<SassNode> nodes, ref int index)
        {
            if (index + 1 >= nodes.Count)
                return false;
            if (nodes[index] is not SassInstruction low ||
                nodes[index + 1] is not SassInstruction high)
                return false;
            if (low.Guard != null || high.Guard != null)
                return false;
            if (!low.Mnemonic.Equals("IADD32I", StringComparison.OrdinalIgnoreCase) ||
                low.Operands.Count < 3 ||
                !low.Operands[0].EndsWith(".CC", StringComparison.OrdinalIgnoreCase))
                return false;
            if (!high.Mnemonic.StartsWith("IADD.X", StringComparison.OrdinalIgnoreCase) ||
                high.Operands.Count < 3)
                return false;
            string lowSource = BaseReg(low.Operands[1]);
            string highSource;
            if (IsRz(high.Operands[1]))
                highSource = BaseReg(high.Operands[2]);
            else if (IsRz(high.Operands[2]))
                highSource = BaseReg(high.Operands[1]);
            else
                return false;
            if (IsRz(lowSource) || IsRz(highSource) ||
                !Regex.IsMatch(lowSource, @"^R\d+$", RegexOptions.IgnoreCase) ||
                !Regex.IsMatch(highSource, @"^R\d+$", RegexOptions.IgnoreCase))
                return false;

            string lowDestination = BaseReg(low.Operands[0]);
            string highDestination = BaseReg(high.Operands[0]);
            string pointerBase;
            if (_pointerRegs.TryGetValue(lowSource, out string? basePointer) &&
                highSource.Equals(NextRegister(lowSource),
                    StringComparison.OrdinalIgnoreCase))
            {
                pointerBase = basePointer;
            }
            else
            {
                pointerBase = $"%rdt{_nextTemp++}";
                _lines.Add($"    mov.b64 {pointerBase}, " +
                    $"{{{Reg(lowSource)}, {Reg(highSource)}}};");
            }
            string sum = $"%rdt{_nextTemp++}";
            _lines.Add($"    add.u64 {sum}, {pointerBase}, " +
                $"{Immediate(low.Operands[2])};");
            if (RegisterNeedsScalarValue(nodes, index + 2, lowDestination) ||
                RegisterNeedsScalarValue(nodes, index + 2, highDestination))
            {
                _lines.Add($"    mov.b64 " +
                    $"{{{Reg(lowDestination)}, {Reg(highDestination)}}}, {sum};");
            }
            string pointer = $"%rdt{_nextTemp++}";
            _lines.Add($"    cvta.to.global.u64 {pointer}, {sum};");
            _pointerRegs[lowDestination] = pointer;
            index += 1;
            return true;
        }

        private bool TryEmitShift64(
            List<SassNode> nodes, ref int index, bool allowGap)
        {
            if (index >= nodes.Count ||
                nodes[index] is not SassInstruction first ||
                first.Guard != null)
                return false;

            SassInstruction? low = null;
            SassInstruction? high = null;
            string operation = "";
            int highIndex = -1;
            if (index + 1 < nodes.Count &&
                nodes[index + 1] is SassInstruction second &&
                second.Guard == null)
            {
                if (IsShiftLeftHigh(first) && IsShiftLeftLow(second))
                {
                    high = first;
                    low = second;
                    highIndex = index;
                    operation = "shl.b64";
                }
                else if (IsShiftLeftLow(first) && IsShiftLeftHigh(second))
                {
                    low = first;
                    high = second;
                    highIndex = index + 1;
                    operation = "shl.b64";
                }
                else if (IsShiftRightLow(first) && IsShiftRightHigh(second))
                {
                    low = first;
                    high = second;
                    highIndex = index + 1;
                    operation = first.Mnemonic.Contains(
                        ".S64", StringComparison.OrdinalIgnoreCase)
                        ? "shr.s64" : "shr.u64";
                }
                else if (IsShiftRightLow(second) && IsShiftRightHigh(first))
                {
                    low = second;
                    high = first;
                    highIndex = index;
                    operation = low.Mnemonic.Contains(
                        ".S64", StringComparison.OrdinalIgnoreCase)
                        ? "shr.s64" : "shr.u64";
                }
            }

            if ((low == null || high == null) && allowGap &&
                (IsShiftRightLow(first) || IsShiftLeftLow(first)))
            {
                int end = Math.Min(index + 4, nodes.Count - 1);
                for (int candidateIndex = index + 2;
                     candidateIndex <= end; candidateIndex++)
                {
                    if (nodes[candidateIndex] is not SassInstruction candidate ||
                        candidate.Guard != null)
                        continue;
                    if (IsShiftLeftLow(first) && IsShiftLeftHigh(candidate))
                    {
                        low = first;
                        high = candidate;
                        highIndex = candidateIndex;
                        operation = "shl.b64";
                        break;
                    }
                    if (IsShiftRightLow(first) && IsShiftRightHigh(candidate))
                    {
                        low = first;
                        high = candidate;
                        highIndex = candidateIndex;
                        operation = first.Mnemonic.Contains(
                            ".S64", StringComparison.OrdinalIgnoreCase)
                            ? "shr.s64" : "shr.u64";
                        break;
                    }
                }
            }
            if (low == null || high == null)
                return false;

            uint shift;
            string sourceLow;
            string sourceHigh;
            if (operation == "shl.b64")
            {
                if (!TryImadShift(low, out shift) ||
                    low.Operands.Count < 4 ||
                    high.Operands.Count < 4)
                    return false;
                sourceLow = low.Operands[1].Trim();
                sourceHigh = high.Operands[3].Trim();
                if (!TryParseImmediate(high.Operands[2], out uint highShift) ||
                    highShift != shift ||
                    !BaseReg(high.Operands[1]).Equals(
                        BaseReg(sourceLow),
                        StringComparison.OrdinalIgnoreCase))
                    return false;
            }
            else
            {
                if (low.Operands.Count < 4 || high.Operands.Count < 4 ||
                    !IsRz(high.Operands[1]))
                    return false;
                // A register-controlled shift amount is not the paired
                // immediate shift this peephole recognises; fall back to
                // the generic per-instruction lowering.
                if (!TryParseImmediate(low.Operands[2], out shift) ||
                    !TryParseImmediate(high.Operands[2], out uint highShift) ||
                    highShift != shift)
                    return false;
                sourceLow = low.Operands[1].Trim();
                sourceHigh = low.Operands[3].Trim();
                if (!BaseReg(sourceHigh).Equals(
                        BaseReg(high.Operands[3]),
                        StringComparison.OrdinalIgnoreCase))
                    return false;
            }

            string source = PackPair64(sourceLow, sourceHigh);
            string shifted = $"%rdt{_nextTemp++}";
            _lines.Add($"    {operation} {shifted}, {source}, {shift};");
            UnpackPair64(shifted, low.Operands[0].Trim(),
                high.Operands[0].Trim());
            if (highIndex == index)
                index++;
            else if (highIndex > index + 1)
                _skipNodes.Add(highIndex);
            else if (highIndex == index + 1)
                index++;
            return true;
        }

        private static bool IsShiftLeftLow(SassInstruction instruction) =>
            instruction.Mnemonic.StartsWith("IMAD.SHL",
                StringComparison.OrdinalIgnoreCase);

        private static bool IsShiftLeftHigh(SassInstruction instruction)
        {
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            return mnemonic.StartsWith("SHF.L", StringComparison.Ordinal) &&
                mnemonic.Contains(".U64", StringComparison.Ordinal) &&
                mnemonic.Contains(".HI", StringComparison.Ordinal);
        }

        private static bool IsShiftRightLow(SassInstruction instruction)
        {
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            return mnemonic.StartsWith("SHF.R", StringComparison.Ordinal) &&
                (mnemonic.Contains(".U64", StringComparison.Ordinal) ||
                 mnemonic.Contains(".S64", StringComparison.Ordinal));
        }

        private static bool IsShiftRightHigh(SassInstruction instruction)
        {
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            return mnemonic.StartsWith("SHF.R", StringComparison.Ordinal) &&
                (mnemonic.Contains(".U32", StringComparison.Ordinal) ||
                 mnemonic.Contains(".S32", StringComparison.Ordinal)) &&
                mnemonic.Contains(".HI", StringComparison.Ordinal);
        }

        private static bool TryImadShift(SassInstruction instruction, out uint shift)
        {
            shift = 0;
            if (!instruction.Mnemonic.StartsWith("IMAD.SHL",
                    StringComparison.OrdinalIgnoreCase) ||
                instruction.Operands.Count < 4 ||
                !IsRz(instruction.Operands[3]))
                return false;
            if (!TryParseImmediate(instruction.Operands[2], out uint multiplier))
                return false;
            if (multiplier == 0 || (multiplier & (multiplier - 1)) != 0)
                return false;
            while ((1u << (int)shift) < multiplier)
                shift++;
            return true;
        }

        private bool TryEmitIsubPair(
            List<SassNode> nodes, ref int index, bool allowGap)
        {
            if (index >= nodes.Count ||
                nodes[index] is not SassInstruction low ||
                low.Guard != null ||
                !low.Mnemonic.Equals("ISUB", StringComparison.OrdinalIgnoreCase) ||
                low.Operands.Count < 3 ||
                !low.Operands[0].Trim().EndsWith(
                    ".CC", StringComparison.OrdinalIgnoreCase))
                return false;

            SassInstruction? high = null;
            int highIndex = -1;
            int end = Math.Min(index + (allowGap ? 6 : 1), nodes.Count - 1);
            for (int candidateIndex = index + 1;
                 candidateIndex <= end; candidateIndex++)
            {
                if (nodes[candidateIndex] is not SassInstruction candidate ||
                    candidate.Guard != null ||
                    !candidate.Mnemonic.StartsWith(
                        "ISUB.X", StringComparison.OrdinalIgnoreCase) ||
                    candidate.Operands.Count < 3)
                    continue;
                high = candidate;
                highIndex = candidateIndex;
                break;
            }
            if (high == null)
                return false;

            string lowDestination = BaseReg(low.Operands[0]);
            string highDestination = BaseReg(high.Operands[0]);
            if (!Regex.IsMatch(lowDestination, @"^R\d+$",
                    RegexOptions.IgnoreCase) ||
                !Regex.IsMatch(highDestination, @"^R\d+$",
                    RegexOptions.IgnoreCase))
                return false;
            string a = PackPair64(low.Operands[1], high.Operands[1]);
            string b = PackPair64(low.Operands[2], high.Operands[2]);
            string result = $"%rdt{_nextTemp++}";
            _lines.Add($"    sub.u64 {result}, {a}, {b};");
            UnpackPair64(result, lowDestination, highDestination);
            if (highIndex == index)
                index++;
            else if (highIndex > index + 1)
                _skipNodes.Add(highIndex);
            else if (highIndex == index + 1)
                index++;
            return true;
        }

        // nvcc expands a runtime-divisor u32 division as a reciprocal
        // estimate plus a fixed-integer correction chain.  That chain is
        // only exact while the estimate matches the hardware MUFU.RCP
        // bias; replaying it on another FP unit gives a quotient that
        // sticks after the first few multiples (cat's alignedK kernel
        // divides the element index by the input size, exactly this
        // idiom).  Recognise the whole `t = n / d` chain and emit an
        // exact div.u32 instead.
        private bool TryEmitFastUnsignedDiv(List<SassNode> nodes, ref int index)
        {
            if (index >= nodes.Count ||
                nodes[index] is not SassInstruction high ||
                !high.Mnemonic.Equals("IMAD.HI.U32",
                    StringComparison.OrdinalIgnoreCase) ||
                high.Operands.Count < 4 ||
                !string.IsNullOrWhiteSpace(high.Guard) ||
                !IsZeroReg(high.Operands[3]))
                return false;
            string t = BaseReg(high.Operands[0]);
            string magic = BaseReg(high.Operands[1]);
            string n = BaseReg(high.Operands[2]);
            if (IsZeroReg(t) || IsZeroReg(n) ||
                !IsFastDivMagic(nodes, index, magic))
            {
                return false;
            }

            int limit = Math.Min(nodes.Count, index + 64);
            var matched = new List<int>();
            string? negReg = null, rem = null, d = null;
            string? p1 = null, p2 = null;
            bool p1Inverted = false, p2Inverted = false;
            int negIdx = -1, p2Idx = -1, inc2Idx = -1;

            int p = index + 1;
            for (; p < limit; p++)
            {
                if (nodes[p] is SassLabel)
                    return false;
                if (nodes[p] is not SassInstruction ins)
                    continue;
                if (TryNegatedRegister(ins, t, out negReg))
                {
                    negIdx = p;
                    p++;
                    break;
                }
                if (WritesRegister(ins, t))
                    return false;
            }
            if (negIdx < 0)
            {
                return false;
            }
            // Keep the negation in the output: some kernels read -q
            // after the division chain (e.g. the adaptive-pool W
            // address).  It is recomputed from the div result.

            for (; p < limit; p++)
            {
                if (nodes[p] is SassLabel)
                    return false;
                if (nodes[p] is not SassInstruction ins)
                    continue;
                if (ins.Mnemonic.Equals("IMAD",
                        StringComparison.OrdinalIgnoreCase) &&
                    ins.Operands.Count >= 4 &&
                    string.IsNullOrWhiteSpace(ins.Guard) &&
                    BaseReg(ins.Operands[3]).Equals(n,
                        StringComparison.OrdinalIgnoreCase) &&
                    (BaseReg(ins.Operands[2]).Equals(negReg,
                         StringComparison.OrdinalIgnoreCase) ||
                     BaseReg(ins.Operands[1]).Equals(negReg,
                         StringComparison.OrdinalIgnoreCase)))
                {
                    rem = BaseReg(ins.Operands[0]);
                    d = BaseReg(ins.Operands[1]).Equals(negReg,
                            StringComparison.OrdinalIgnoreCase)
                        ? BaseReg(ins.Operands[2])
                        : BaseReg(ins.Operands[1]);
                    matched.Add(p);
                    p++;
                    break;
                }
                if (WritesRegister(ins, negReg!))
                {
                    return false;
                }
            }
            if (rem == null || d == null)
            {
                return false;
            }

            // First remainder check: r >= d?
            for (; p < limit; p++)
            {
                if (nodes[p] is SassLabel)
                    return false;
                if (nodes[p] is not SassInstruction ins)
                    continue;
                if (IsGeU32(ins, rem, d, out p1, out p1Inverted) &&
                    string.IsNullOrWhiteSpace(ins.Guard))
                {
                    matched.Add(p);
                    p++;
                    break;
                }
                if (WritesRegister(ins, rem))
                    return false;
            }
            if (p1 == null)
            {
                return false;
            }

            // First correction (r -= d, t += 1) and the second check
            // (r >= d) are independent instructions and the compiler
            // interleaves them in either order.
            bool gotSub = false, gotInc1 = false, gotP2 = false;
            for (; p < limit && !(gotSub && gotInc1 && gotP2); p++)
            {
                if (nodes[p] is SassLabel)
                    return false;
                if (nodes[p] is not SassInstruction ins)
                    continue;
                if (!gotSub)
                {
                    if (IsSubD(ins, rem, d, p1, p1Inverted))
                    {
                        gotSub = true;
                        matched.Add(p);
                        continue;
                    }
                    if (WritesRegister(ins, rem))
                        return false;
                    continue;
                }
                if (!gotInc1 && IsAddOne(ins, t, p1, p1Inverted))
                {
                    gotInc1 = true;
                    matched.Add(p);
                    continue;
                }
                if (!gotP2 &&
                    IsGeU32(ins, rem, d, out p2, out p2Inverted) &&
                    string.IsNullOrWhiteSpace(ins.Guard))
                {
                    gotP2 = true;
                    p2Idx = p;
                    matched.Add(p);
                    continue;
                }
                if (!gotP2 && WritesRegister(ins, rem))
                    return false;
            }
            if (!gotSub || !gotInc1 || !gotP2 || p2 == null)
            {
                return false;
            }

            for (int j = p2Idx + 1;
                 j < Math.Min(nodes.Count, p2Idx + 24); j++)
            {
                if (nodes[j] is SassLabel)
                    break;
                if (nodes[j] is not SassInstruction ins)
                    continue;
                if (matched.Contains(j))
                    continue;
                if (IsAddOne(ins, t, p2, p2Inverted))
                {
                    inc2Idx = j;
                    matched.Add(j);
                    break;
                }
                if (WritesRegister(ins, t))
                    return false;
            }
            if (inc2Idx < 0)
            {
                return false;
            }

            // Only rewrite the chain when its temporaries are dead after
            // the chain; otherwise leave it untouched.
            if (UsesRegisterOutside(nodes, matched, rem, index))
            {
                return false;
            }
            if (UsesPredicateOutside(nodes, matched, p1, index) ||
                UsesPredicateOutside(nodes, matched, p2, index))
            {
                return false;
            }

            _lines.Add($"    div.u32 {Reg(t)}, {Operand(n)}, {Operand(d)};");
            foreach (int j in matched)
                _skipNodes.Add(j);
            return true;
        }

        private static bool WritesRegister(
            SassInstruction instruction, string register)
        {
            int destination = DestinationIndex(instruction);
            return destination >= 0 &&
                destination < instruction.Operands.Count &&
                BaseReg(instruction.Operands[destination]).Equals(
                    register, StringComparison.OrdinalIgnoreCase);
        }

        private static bool WrittenBetween(
            List<SassNode> nodes, int start, int end, string register)
        {
            for (int i = start; i < end; i++)
            {
                if (nodes[i] is SassInstruction instruction &&
                    WritesRegister(instruction, register))
                    return true;
            }
            return false;
        }

        private static bool TryNegatedRegister(
            SassInstruction instruction, string register, out string? negated)
        {
            negated = null;
            if (instruction.Mnemonic.Equals("IADD3",
                    StringComparison.OrdinalIgnoreCase) &&
                instruction.Operands.Count >= 3)
            {
                if (IsNegated(instruction.Operands[1], register) ||
                    IsNegated(instruction.Operands[2], register))
                {
                    negated = BaseReg(instruction.Operands[0]);
                    return true;
                }
                return false;
            }
            if (instruction.Mnemonic.StartsWith("IMAD.MOV",
                    StringComparison.OrdinalIgnoreCase) &&
                instruction.Operands.Count >= 4 &&
                IsNegated(instruction.Operands[3], register))
            {
                negated = BaseReg(instruction.Operands[0]);
                return true;
            }
            return false;
        }

        private static bool IsNegated(string operand, string register)
        {
            string text = operand.Trim();
            return text.StartsWith("-", StringComparison.Ordinal) &&
                BaseReg(text[1..]).Equals(register,
                    StringComparison.OrdinalIgnoreCase);
        }

        // Two equivalent encodings guard the remainder correction:
        //   ISETP.GE.U32 P, rem, d  -> @P correction
        //   ISETP.GT.U32 P, d, rem  -> @!P correction
        private static bool IsGeU32(
            SassInstruction instruction, string rem, string d,
            out string? pred, out bool inverted)
        {
            pred = null;
            inverted = false;
            if (instruction.Operands.Count < 4)
                return false;
            if (instruction.Mnemonic.StartsWith("ISETP.GE.U32",
                    StringComparison.OrdinalIgnoreCase) &&
                BaseReg(instruction.Operands[2]).Equals(rem,
                    StringComparison.OrdinalIgnoreCase) &&
                BaseReg(instruction.Operands[3]).Equals(d,
                    StringComparison.OrdinalIgnoreCase))
            {
                pred = instruction.Operands[0].Trim();
                return true;
            }
            if (instruction.Mnemonic.StartsWith("ISETP.GT.U32",
                    StringComparison.OrdinalIgnoreCase) &&
                BaseReg(instruction.Operands[2]).Equals(d,
                    StringComparison.OrdinalIgnoreCase) &&
                BaseReg(instruction.Operands[3]).Equals(rem,
                    StringComparison.OrdinalIgnoreCase))
            {
                pred = instruction.Operands[0].Trim();
                inverted = true;
                return true;
            }
            return false;
        }

        private static bool GuardMatches(
            SassInstruction instruction, string pred, bool inverted)
        {
            string expected = inverted ? "@!" + pred : "@" + pred;
            return string.Equals(instruction.Guard, expected,
                StringComparison.OrdinalIgnoreCase);
        }

        private static bool IsSubD(
            SassInstruction instruction, string rem, string d,
            string pred, bool inverted)
        {
            if (!GuardMatches(instruction, pred, inverted))
                return false;
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            if (mnemonic.Equals("IADD3", StringComparison.Ordinal))
            {
                // @P IADD3 rem, -d, rem, RZ  or  @P IADD3 rem, rem, -d, RZ
                return instruction.Operands.Count >= 3 &&
                    BaseReg(instruction.Operands[0]).Equals(rem,
                        StringComparison.OrdinalIgnoreCase) &&
                    ((IsNegated(instruction.Operands[1], d) &&
                      BaseReg(instruction.Operands[2]).Equals(rem,
                          StringComparison.OrdinalIgnoreCase)) ||
                     (BaseReg(instruction.Operands[1]).Equals(rem,
                          StringComparison.OrdinalIgnoreCase) &&
                      IsNegated(instruction.Operands[2], d)));
            }
            if (mnemonic.StartsWith("IMAD", StringComparison.Ordinal))
            {
                // @P IMAD.IADD rem, rem, 1, -d
                return instruction.Operands.Count >= 4 &&
                    BaseReg(instruction.Operands[0]).Equals(rem,
                        StringComparison.OrdinalIgnoreCase) &&
                    BaseReg(instruction.Operands[1]).Equals(rem,
                        StringComparison.OrdinalIgnoreCase) &&
                    IsNegated(instruction.Operands[3], d);
            }
            return false;
        }

        private static bool IsAddOne(
            SassInstruction instruction, string t, string pred,
            bool inverted)
        {
            if (!GuardMatches(instruction, pred, inverted))
                return false;
            if (!instruction.Mnemonic.Equals("IADD3",
                    StringComparison.OrdinalIgnoreCase) ||
                instruction.Operands.Count < 3)
                return false;
            return BaseReg(instruction.Operands[0]).Equals(t,
                    StringComparison.OrdinalIgnoreCase) &&
                BaseReg(instruction.Operands[1]).Equals(t,
                    StringComparison.OrdinalIgnoreCase) &&
                TrySwitchImmediate(instruction.Operands[2], out int addend) &&
                addend == 1;
        }

        private static bool UsesRegisterOutside(
            List<SassNode> nodes, List<int> matched, string register,
            int chainStart)
        {
            var skip = new HashSet<int>(matched);
            for (int i = chainStart + 1; i < nodes.Count; i++)
            {
                if (skip.Contains(i) ||
                    nodes[i] is not SassInstruction ins)
                    continue;
                int destination = DestinationIndex(ins);
                bool read = false;
                for (int op = 0; op < ins.Operands.Count; op++)
                {
                    if (op == destination)
                        continue;
                    string text = ins.Operands[op].Trim();
                    if (text.StartsWith("-", StringComparison.Ordinal))
                        text = text[1..].Trim();
                    if (BaseReg(text).Equals(register,
                            StringComparison.OrdinalIgnoreCase))
                    {
                        read = true;
                        break;
                    }
                }
                if (read)
                    return true;
                if (WritesRegister(ins, register))
                    return false;
            }
            return false;
        }

        private static bool UsesPredicateOutside(
            List<SassNode> nodes, List<int> matched, string pred,
            int chainStart)
        {
            var skip = new HashSet<int>(matched);
            for (int i = chainStart + 1; i < nodes.Count; i++)
            {
                if (skip.Contains(i) ||
                    nodes[i] is not SassInstruction ins)
                    continue;
                if ((ins.Guard ?? "").TrimStart('@', '!').Equals(pred,
                        StringComparison.OrdinalIgnoreCase))
                    return true;
                if (WritesRegister(ins, pred))
                    return false;
            }
            return false;
        }

        private static bool IsFastDivMagic(
            List<SassNode> nodes, int index, string magic)
        {
            int definition = -1;
            for (int i = index - 1; i >= 0 && i >= index - 96; i--)
            {
                if (nodes[i] is not SassInstruction ins)
                    continue;
                if (WritesRegister(ins, magic))
                {
                    definition = i;
                    break;
                }
            }
            if (definition < 0 ||
                nodes[definition] is not SassInstruction prep ||
                !prep.Mnemonic.Equals("IMAD.HI.U32",
                    StringComparison.OrdinalIgnoreCase) ||
                prep.Operands.Count < 3)
                return false;
            for (int i = definition - 1;
                 i >= 0 && i >= definition - 20; i--)
            {
                if (nodes[i] is not SassInstruction conv ||
                    !conv.Mnemonic.StartsWith("F2I",
                        StringComparison.OrdinalIgnoreCase) ||
                    conv.Operands.Count < 1)
                    continue;
                string destination = BaseReg(conv.Operands[0]);
                if (destination.Equals(BaseReg(prep.Operands[1]),
                        StringComparison.OrdinalIgnoreCase) ||
                    destination.Equals(BaseReg(prep.Operands[2]),
                        StringComparison.OrdinalIgnoreCase))
                    return true;
            }
            return false;
        }

        private bool TryEmitIadd3Pair(
            List<SassNode> nodes, ref int index, bool allowGap)
        {
            if (index >= nodes.Count ||
                nodes[index] is not SassInstruction low)
                return false;
            if (!low.Mnemonic.Equals("IADD3", StringComparison.OrdinalIgnoreCase))
                return false;

            int lowFirst = 1;
            string? carry = null;
            if (lowFirst < low.Operands.Count &&
                Regex.IsMatch(low.Operands[lowFirst].Trim(),
                    @"^!?(?:P\d+|PT)$", RegexOptions.IgnoreCase))
            {
                carry = low.Operands[lowFirst].Trim();
                lowFirst++;
            }
            // Six-operand form: IADD3 Rd, Pc, Pin, Ra, Rb, Rc.  The carry-in
            // predicate is not one of the three addends; skipping it here
            // (as well as in EmitIadd3) is what keeps Rc -- the lane offset
            // in cub's radix store -- from disappearing.
            string? carryIn = null;
            if (carry != null && lowFirst < low.Operands.Count &&
                Regex.IsMatch(low.Operands[lowFirst].Trim(),
                    @"^!?(?:P\d+|PT)$", RegexOptions.IgnoreCase))
            {
                carryIn = low.Operands[lowFirst].Trim();
                lowFirst++;
            }
            if (lowFirst + 2 >= low.Operands.Count)
                return false;
            string lowDestination = low.Operands[0].Trim();
            string lowA = low.Operands[lowFirst].Trim();
            string lowB = low.Operands[lowFirst + 1].Trim();
            string lowC = low.Operands[lowFirst + 2].Trim();
            if (!Regex.IsMatch(BaseReg(lowDestination), @"^R\d+$",
                    RegexOptions.IgnoreCase))
                return false;

            SassInstruction? high = null;
            int highIndex = -1;
            int end = Math.Min(index + (allowGap ? 6 : 1), nodes.Count - 1);
            for (int candidateIndex = index + 1;
                 candidateIndex <= end; candidateIndex++)
            {
                if (nodes[candidateIndex] is not SassInstruction candidate ||
                    !string.Equals(candidate.Guard, low.Guard,
                        StringComparison.OrdinalIgnoreCase))
                    continue;
                bool isIaddX = candidate.Mnemonic.StartsWith(
                    "IADD3.X", StringComparison.OrdinalIgnoreCase);
                bool isImadX = candidate.Mnemonic.StartsWith(
                    "IMAD.X", StringComparison.OrdinalIgnoreCase);
                if (!isIaddX && !isImadX)
                    continue;
                int carryIndex = isIaddX ? 4 : candidate.Operands.Count - 1;
                string candidateCarry = "RZ";
                if (carryIndex < candidate.Operands.Count)
                {
                    string candidateCarryText =
                        candidate.Operands[carryIndex].Trim();
                    if (Regex.IsMatch(candidateCarryText,
                            @"^!?(?:P\d+|PT)$", RegexOptions.IgnoreCase))
                        candidateCarry = candidateCarryText;
                }
                if (carry != null)
                {
                    if (!candidateCarry.Equals(carry,
                            StringComparison.OrdinalIgnoreCase))
                        continue;
                }
                else if (!IsRz(candidateCarry))
                {
                    // The low add produced no carry-out, so a following
                    // .X that consumes another instruction's carry is not
                    // part of this pair.
                    continue;
                }
                high = candidate;
                highIndex = candidateIndex;
                break;
            }
            if (high == null)
                return false;

            string highDestination = high.Operands[0].Trim();
            if (high.Operands.Count < 5 ||
                !Regex.IsMatch(BaseReg(highDestination), @"^R\d+$",
                    RegexOptions.IgnoreCase))
                return false;

            string a;
            string b;
            string? c = null;
            bool subtract = false;
            if (high.Mnemonic.StartsWith("IMAD.X",
                    StringComparison.OrdinalIgnoreCase))
            {
                if (high.Operands.Count < 5 || !IsRz(lowC))
                    return false;
                string imadMultiplier = high.Operands[2].Trim();
                string highSource = high.Operands[1].Trim();
                if (IsRz(imadMultiplier))
                {
                    // IMAD.X Rd, RZ, RZ, c[..], P: high = addend + carry.
                    // The zero multiplicand contributes nothing, so the
                    // pair is {lowA, 0} + {lowB, addend}.
                    if (!IsRz(highSource))
                        return false;
                }
                else
                {
                    if (!Regex.IsMatch(imadMultiplier,
                            @"^-?(?:0x[0-9A-Fa-f]+|\d+)$"))
                        return false;
                    if (ParseImmediate(imadMultiplier) != 1)
                        return false;
                }
                a = lowA;
                b = lowB;
                string highAddend = high.Operands[3].Trim();
                if (b.StartsWith("-", StringComparison.Ordinal) &&
                    highAddend.StartsWith("~", StringComparison.Ordinal))
                {
                    subtract = true;
                    b = b[1..].Trim();
                    highAddend = highAddend[1..].Trim();
                }
                else if (b.StartsWith("-", StringComparison.Ordinal) ||
                         highAddend.StartsWith("~", StringComparison.Ordinal))
                {
                    return false;
                }
                if (!subtract &&
                    (WrittenBetween(nodes, index + 1, highIndex,
                         BaseReg(highSource)) ||
                     WrittenBetween(nodes, index + 1, highIndex,
                         BaseReg(highAddend))) &&
                    TryDeferPairHighOperands(nodes, index, highIndex,
                        a, b, "RZ", highSource, highAddend, "RZ", null,
                        lowDestination, highDestination, Guard(low.Guard)))
                {
                    return true;
                }
                string aPair = PackPair64(a, highSource);
                string bPair = PackPair64(b, highAddend);
                string result = $"%rdt{_nextTemp++}";
                _lines.Add($"    {(subtract ? "sub" : "add")}.u64 " +
                    $"{result}, {aPair}, {bPair};");
                if (TryDeferPairHigh(nodes, index, highIndex, result,
                        lowDestination, highDestination))
                    return true;
                UnpackPair64(result, lowDestination, highDestination,
                    Guard(low.Guard));
                AdvancePair(ref index, highIndex);
                return true;
            }

            if (!high.Mnemonic.StartsWith("IADD3.X",
                    StringComparison.OrdinalIgnoreCase))
                return false;
            if (high.Operands.Count < 6)
                return false;
            a = lowA;
            b = lowB;
            c = lowC;
            string highA = high.Operands[1].Trim();
            string highB = high.Operands[2].Trim();
            string highC = high.Operands[3].Trim();
            if ((WrittenBetween(nodes, index + 1, highIndex, BaseReg(highA)) ||
                 WrittenBetween(nodes, index + 1, highIndex, BaseReg(highB)) ||
                 WrittenBetween(nodes, index + 1, highIndex, BaseReg(highC))) &&
                TryDeferPairHighOperands(nodes, index, highIndex,
                    a, b, c!, highA, highB, highC, carryIn,
                    lowDestination, highDestination, Guard(low.Guard)))
            {
                return true;
            }
            string aValue = PackPair64(a, highA);
            string bValue = PackPair64(b, highB);
            string sum = $"%rdt{_nextTemp++}";
            _lines.Add($"    add.u64 {sum}, {aValue}, {bValue};");
            if (!IsRz(c) || !IsRz(highC))
            {
                string cValue = PackPair64(c!, highC);
                _lines.Add($"    add.u64 {sum}, {sum}, {cValue};");
            }
            if (carryIn != null)
            {
                // Carry-in addend (0/1) participates in the low word.  The
                // 64-bit sum's high half still lands in the .X destination,
                // so the natural carry propagation is preserved.
                string pinValue = Operand(carryIn);
                string pinPair = PackPair64(pinValue, "0");
                _lines.Add($"    add.u64 {sum}, {sum}, {pinPair};");
            }
            // IADD3.X may also define a carry-out predicate which later
            // instructions read as a carry-in.  If the high sources are all
            // RZ the high word is just the incoming carry (0/1), so the
            // 65th bit is always clear; define the predicate accordingly
            // instead of leaving it stale.
            if (high.Operands.Count >= 6 &&
                Regex.IsMatch(high.Operands[5].Trim(), @"^P\d+$") &&
                IsRz(highA) && IsRz(highB) && IsRz(highC))
            {
                _lines.Add($"    setp.eq.u32 {Pred(high.Operands[5].Trim())}, " +
                    "0, 1;");
            }
            if (TryDeferPairHigh(nodes, index, highIndex, sum,
                    lowDestination, highDestination))
                return true;
            UnpackPair64(sum, lowDestination, highDestination, Guard(low.Guard));
            AdvancePair(ref index, highIndex);
            return true;
        }

        // When a paired 64-bit add has its .X half several instructions
        // after the low half, the high destination register can still be
        // read in between (and must keep its old value).  Unpacking both
        // halves at the low instruction would clobber it.  Materialise the
        // low half now and defer the high half until the .X instruction,
        // where the emission loop substitutes the deferred copy.
        private bool TryDeferPairHigh(List<SassNode> nodes, int index,
            int highIndex, string pair, string lowDestination,
            string highDestination)
        {
            if (highIndex <= index + 1 || _inlineLabelOverrides != null)
                return false;
            string low = BaseReg(lowDestination);
            string high = BaseReg(highDestination);
            if (!RegisterReadBetween(nodes, index + 1, highIndex, high))
                return false;
            string scratch = $"%rt{_nextTemp++}";
            _lines.Add($"    mov.b64 {{{Reg(low)}, {scratch}}}, {pair};");
            // The 64-bit pair is not complete until the .X half runs, so
            // do not let the pair cache materialise it earlier.
            _pairRegs.Remove(low);
            _deferredPairHigh[highIndex] = (low, high, pair);
            return true;
        }

        // ptxas sometimes reuses a register that held the low word of a
        // 64-bit value for the high word of one of the addends between the
        // IADD3 and its .X half.  The low addends are only valid at the
        // IADD3 and the .X sources only at the .X, so snapshot the low
        // operands, materialise the low word (plus its carry-out as 0/1)
        // now, and compute the high word from the live registers when the
        // .X instruction would have run.  Folding the pair at the IADD3
        // instead captured the pre-overwrite value: argmax/max over the
        // fastest dimension returned {i,i} because the rewritten register
        // was the index's own low word.
        private bool TryDeferPairHighOperands(List<SassNode> nodes, int index,
            int highIndex, string a, string b, string c, string highA,
            string highB, string highC, string? carryIn, string lowDestination,
            string highDestination, string guard)
        {
            if (highIndex <= index + 1 || _inlineLabelOverrides != null)
                return false;
            string prefix = guard.Length == 0 ? "" : guard + " ";
            string sa = $"%rt{_nextTemp++}";
            string sb = $"%rt{_nextTemp++}";
            _lines.Add($"    mov.b32 {sa}, {(IsRz(a) ? "0" : Operand(a))};");
            _lines.Add($"    mov.b32 {sb}, {(IsRz(b) ? "0" : Operand(b))};");
            string? sc = null;
            if (!IsRz(c))
            {
                sc = $"%rt{_nextTemp++}";
                _lines.Add($"    mov.b32 {sc}, {Operand(c)};");
            }
            string low = $"%rt{_nextTemp++}";
            _lines.Add($"    {prefix}add.cc.u32 {low}, {sa}, {sb};");
            if (sc != null)
                _lines.Add($"    {prefix}addc.u32 {low}, {low}, {sc};");
            if (carryIn != null)
                _lines.Add($"    {prefix}addc.u32 {low}, {low}, " +
                    $"{Operand(carryIn)};");
            string carry = $"%rt{_nextTemp++}";
            _lines.Add($"    {prefix}addc.u32 {carry}, 0, 0;");
            _lines.Add($"    {prefix}mov.b32 {Reg(lowDestination)}, {low};");
            _pairRegs.Remove(BaseReg(lowDestination));

            string high = $"%rt{_nextTemp++}";
            var lines = new List<string>
            {
                $"    {prefix}add.u32 {high}, " +
                    $"{(IsRz(highA) ? "0" : Operand(highA))}, " +
                    $"{(IsRz(highB) ? "0" : Operand(highB))};",
            };
            if (!IsRz(highC))
                lines.Add($"    {prefix}add.u32 {high}, {high}, " +
                    $"{Operand(highC)};");
            lines.Add($"    {prefix}add.u32 {high}, {high}, {carry};");
            lines.Add($"    {prefix}mov.b32 {Reg(highDestination)}, {high};");
            _deferredPairLines[highIndex] = lines;
            return true;
        }

        private static bool RegisterReadBetween(List<SassNode> nodes,
            int start, int end, string register)
        {
            for (int i = start; i < end; i++)
            {
                if (nodes[i] is not SassInstruction instruction)
                    continue;
                int destination = DestinationIndex(instruction);
                for (int operandIndex = 0;
                     operandIndex < instruction.Operands.Count; operandIndex++)
                {
                    if (operandIndex == destination)
                        continue;
                    if (ContainsRegister(instruction.Operands[operandIndex],
                            register))
                        return true;
                }
            }
            return false;
        }

        // LEA can feed a following IADD3.X carry, e.g.
        //   LEA R19, P1, R29, R10, 0x1
        //   IADD3.X R18, RZ, R9, RZ, P1, !PT
        // Fold the pair into one 64-bit add.
        private bool TryEmitLeaCarryPair(
            List<SassNode> nodes, ref int index, bool allowGap)
        {
            if (index >= nodes.Count ||
                nodes[index] is not SassInstruction low ||
                low.Guard != null ||
                !low.Mnemonic.Equals("LEA", StringComparison.OrdinalIgnoreCase) ||
                low.Operands.Count < 5)
                return false;
            string lowDestination = StripReuse(low.Operands[0]);
            Match lowReg = Regex.Match(BaseReg(lowDestination), @"^R(\d+)$",
                RegexOptions.IgnoreCase);
            string carry = low.Operands[1].Trim();
            if (!lowReg.Success ||
                !Regex.IsMatch(carry, @"^!?(?:P\d+|PT)$",
                    RegexOptions.IgnoreCase))
                return false;
            SassInstruction? high = null;
            int highIndex = -1;
            int end = Math.Min(index + (allowGap ? 6 : 1), nodes.Count - 1);
            for (int candidateIndex = index + 1;
                 candidateIndex <= end; candidateIndex++)
            {
                if (nodes[candidateIndex] is not SassInstruction candidate ||
                    candidate.Guard != null ||
                    !candidate.Mnemonic.StartsWith(
                        "IADD3.X", StringComparison.OrdinalIgnoreCase) ||
                    candidate.Operands.Count < 6)
                    continue;
                if (!candidate.Operands[4].Trim().Equals(
                        carry, StringComparison.OrdinalIgnoreCase))
                    continue;
                high = candidate;
                highIndex = candidateIndex;
                break;
            }
            if (high == null)
                return false;
            string highDestination = StripReuse(high.Operands[0]);
            Match highReg = Regex.Match(BaseReg(highDestination), @"^R(\d+)$",
                RegexOptions.IgnoreCase);
            if (!highReg.Success)
                return false;
            int lowNumber = int.Parse(lowReg.Groups[1].Value,
                CultureInfo.InvariantCulture);
            int highNumber = int.Parse(highReg.Groups[1].Value,
                CultureInfo.InvariantCulture);
            if (highNumber != lowNumber - 1 && highNumber != lowNumber + 1)
                return false;
            string lowA = low.Operands[2].Trim();
            string lowB = low.Operands[3].Trim();
            string shift = low.Operands[4].Trim();
            string highA = high.Operands[1].Trim();
            string highB = high.Operands[2].Trim();
            string highC = high.Operands[3].Trim();
            string aPair = PackPair64(lowA, highA);
            string lowBValue = IsRz(lowB) ? "0" : Operand(lowB);
            string shifted = $"%rt{_nextTemp++}";
            _lines.Add($"    shl.b32 {shifted}, {lowBValue}, {shift};");
            string highBValue = IsRz(highB) ? "0" : Operand(highB);
            string bPair = PackPair64Values(shifted, highBValue);
            string sum = $"%rdt{_nextTemp++}";
            _lines.Add($"    add.u64 {sum}, {aPair}, {bPair};");
            if (!IsRz(highC))
            {
                string addend = $"%rdt{_nextTemp++}";
                _lines.Add($"    cvt.u64.u32 {addend}, {Operand(highC)};");
                _lines.Add($"    add.u64 {sum}, {sum}, {addend};");
            }
            UnpackPair64(sum, lowDestination, highDestination);
            AdvancePair(ref index, highIndex);
            return true;
        }

        private string PackPair64Values(string lowValue, string highValue)
        {
            string lowReg = MaterializeB32(lowValue);
            string highReg = MaterializeB32(highValue);
            string pair = $"%rdt{_nextTemp++}";
            _lines.Add($"    mov.b64 {pair}, {{{lowReg}, {highReg}}};");
            return pair;
        }

        private void AdvancePair(ref int index, int highIndex)
        {
            if (highIndex == index + 1)
                index++;
            else if (highIndex > index + 1)
                _skipNodes.Add(highIndex);
        }

        // UIADD3/UIADD3.X is the uniform-datapath 64-bit add pair.  The
        // low instruction's operands are the low 32-bit halves, the .X
        // instruction carries the matching high halves; fold them back
        // into one 64-bit add over the uniform register pair.
        private bool TryEmitUadd3Pair(List<SassNode> nodes, ref int index)
        {
            if (index + 1 >= nodes.Count ||
                nodes[index] is not SassInstruction low ||
                nodes[index + 1] is not SassInstruction high ||
                low.Guard != null || high.Guard != null ||
                !low.Mnemonic.Equals("UIADD3", StringComparison.OrdinalIgnoreCase) ||
                !high.Mnemonic.StartsWith(
                    "UIADD3.X", StringComparison.OrdinalIgnoreCase) ||
                high.Operands.Count < 4)
                return false;
            int lowFirst = 1;
            if (lowFirst < low.Operands.Count &&
                Regex.IsMatch(low.Operands[lowFirst].Trim(),
                    @"^!?U?P\d+$|^!?U?PT$", RegexOptions.IgnoreCase))
                lowFirst++;
            if (lowFirst + 2 >= low.Operands.Count ||
                high.Operands.Count < 6)
                return false;
            string lowDestination = Regex.Replace(StripReuse(low.Operands[0]),
                @"\.CC$", "", RegexOptions.IgnoreCase);
            string highDestination = Regex.Replace(StripReuse(high.Operands[0]),
                @"\.CC$", "", RegexOptions.IgnoreCase);
            Match lowReg = Regex.Match(lowDestination, @"^UR(\d+)$",
                RegexOptions.IgnoreCase);
            Match highReg = Regex.Match(highDestination, @"^UR(\d+)$",
                RegexOptions.IgnoreCase);
            if (!lowReg.Success || !highReg.Success ||
                int.Parse(highReg.Groups[1].Value,
                    CultureInfo.InvariantCulture) !=
                int.Parse(lowReg.Groups[1].Value,
                    CultureInfo.InvariantCulture) + 1)
                return false;
            string lowA = low.Operands[lowFirst].Trim();
            string lowB = low.Operands[lowFirst + 1].Trim();
            string lowC = low.Operands[lowFirst + 2].Trim();
            string highA = high.Operands[1].Trim();
            string highB = high.Operands[2].Trim();
            string highC = high.Operands[3].Trim();
            string aValue = PackUniformOperandPair(lowA, highA);
            string bValue = PackUniformOperandPair(lowB, highB);
            string sum = $"%rdt{_nextTemp++}";
            _lines.Add($"    add.u64 {sum}, {aValue}, {bValue};");
            if (!IsUniformZero(lowC) || !IsUniformZero(highC))
            {
                string cValue = PackUniformOperandPair(lowC, highC);
                _lines.Add($"    add.u64 {sum}, {sum}, {cValue};");
            }
            int number = int.Parse(lowReg.Groups[1].Value,
                CultureInfo.InvariantCulture);
            _lines.Add($"    mov.b64 {{%ur{number}, %ur{number + 1}}}, {sum};");
            index++;
            return true;
        }

        private static bool IsUniformZero(string operand) =>
            StripReuse(operand).Equals("URZ", StringComparison.OrdinalIgnoreCase);

        private string PackUniformOperandPair(string low, string high)
        {
            string lowValue = MaterializeB32(UniformOperand(low));
            string highValue = MaterializeB32(UniformOperand(high));
            string pair = $"%rdt{_nextTemp++}";
            _lines.Add($"    mov.b64 {pair}, {{{lowValue}, {highValue}}};");
            return pair;
        }

        private string MaterializeB32(string value)
        {
            string text = value.Trim();
            if (text.StartsWith("%", StringComparison.Ordinal))
                return text;
            string temp = $"%rt{_nextTemp++}";
            _lines.Add($"    mov.b32 {temp}, {text};");
            return temp;
        }

        private string PackPair64(string low, string high)
        {
            low = StripReuse(low);
            high = StripReuse(high);
            string lowValue = $"%rt{_nextTemp++}";
            string highValue = $"%rt{_nextTemp++}";
            _lines.Add($"    mov.b32 {lowValue}, " +
                $"{(IsRz(low) ? "0" : Operand(low))};");
            _lines.Add($"    mov.b32 {highValue}, " +
                $"{(IsRz(high) ? "0" : Operand(high))};");
            string pair = $"%rdt{_nextTemp++}";
            _lines.Add($"    mov.b64 {pair}, {{{lowValue}, {highValue}}};");
            return pair;
        }

        // Cache a 64-bit value under its low register and write both
        // halves immediately.  Lazy materialisation at a later read is
        // wrong across control flow: a pair defined in one arm (e.g. the
        // vector-path address LEA) would be materialised in a different
        // block where the SASS register still holds its other value.
        private void CachePair(string lowKey, string value,
                               string prefix = "")
        {
            _pairRegs[lowKey] = value;
            if (!_materializedPairs.Add(value))
                return;
            string highKey = NextRegister(lowKey);
            _lines.Add($"    {prefix}mov.b64 " +
                $"{{{Reg(lowKey)}, {Reg(highKey)}}}, {value};");
        }

        private void UnpackPair64(string value, string low, string high,
                                  string guard = "")
        {
            string prefix = guard.Length == 0 ? "" : guard + " ";
            _lines.Add($"    {prefix}mov.b64 {{{Reg(low)}, {Reg(high)}}}, {value};");
            // Keep the address-pair cache in sync: a later [Rlow.64] must
            // use this fresh value, not one built before the low/high
            // halves were overwritten.
            _pairRegs[BaseReg(low)] = value;
            _materializedPairs.Add(value);
        }

        private static string StripReuse(string operand) =>
            Regex.Replace(operand.Trim(), @"\.reuse\b", "",
                RegexOptions.IgnoreCase);

        private bool TryEmitAddressAdd(List<SassNode> nodes, ref int index)
        {
            if (index + 1 >= nodes.Count)
                return false;
            if (nodes[index] is not SassInstruction first ||
                nodes[index + 1] is not SassInstruction second)
                return false;
            if (first.Guard != null || second.Guard != null)
                return false;
            string firstMnemonic = first.Mnemonic.ToUpperInvariant();
            if (!firstMnemonic.Equals("IADD", StringComparison.Ordinal) &&
                !firstMnemonic.Equals("IADD32I", StringComparison.Ordinal) &&
                !firstMnemonic.Equals("ISCADD", StringComparison.Ordinal))
                return false;
            bool secondIsIaddX = second.Mnemonic.StartsWith("IADD.X",
                StringComparison.OrdinalIgnoreCase);
            bool secondIsImadHiX = second.Mnemonic.StartsWith("IMAD",
                StringComparison.OrdinalIgnoreCase) &&
                second.Mnemonic.Contains("HI.X",
                    StringComparison.OrdinalIgnoreCase);
            if (!secondIsIaddX && !secondIsImadHiX)
                return false;
            if (!first.Operands[0].Trim().EndsWith(".CC",
                    StringComparison.OrdinalIgnoreCase))
                return false;
            if (first.Operands.Count < (firstMnemonic == "ISCADD" ? 4 : 3) ||
                second.Operands.Count < 3)
                return false;
            int highOperand = secondIsImadHiX ? 3 : 2;
            if (second.Operands.Count <= highOperand ||
                !TryConstantOffset(second.Operands[highOperand],
                    out int highOffset))
                return false;

            if (firstMnemonic == "IADD32I")
            {
                if (!secondIsIaddX)
                    return false;
                if (highOffset < 4 ||
                    !_paramsByOffset.TryGetValue(highOffset - 4,
                        out PtxParameter? immediateParameter) ||
                    immediateParameter.Type != "u64" ||
                    immediateParameter.Offset != highOffset - 4)
                    return false;
                string immediate = Immediate(first.Operands[2]);
                if (_pointerRegs.TryGetValue(
                        BaseReg(first.Operands[1]), out string? pointer))
                {
                    if (!IsRz(second.Operands[1]))
                        return false;
                    bool unpackPointer = NeedsUnpack(nodes, index + 2,
                        BaseReg(first.Operands[0]),
                        BaseReg(second.Operands[0]));
                    string pointerBase = $"%rdt{_nextTemp++}";
                    _lines.Add($"    mov.b64 {pointerBase}, {pointer};");
                    string pointerAddress = $"%rdt{_nextTemp++}";
                    _lines.Add($"    add.u64 {pointerAddress}, {pointerBase}, {immediate};");
                    EmitAddressUnpack(pointerAddress, immediateParameter,
                        first, second, unpackPointer);
                    index += 1;
                    return true;
                }
                // The high half is only safe to consume when the low source
                // is already a known 64-bit pointer.
                return false;
            }

            if (secondIsImadHiX)
            {
                if (firstMnemonic != "ISCADD" || second.Operands.Count < 4)
                    return false;
                if (!BaseReg(first.Operands[1]).Equals(
                        BaseReg(second.Operands[1]),
                        StringComparison.OrdinalIgnoreCase))
                    return false;
                if (!TryConstantOffset(first.Operands[2], out int imadBase) ||
                    imadBase != highOffset - 4)
                    return false;
                if (!_paramsByOffset.TryGetValue(imadBase,
                        out PtxParameter? imadParameter) ||
                    imadParameter.Type != "u64")
                    return false;
                bool unpackProduct = NeedsUnpack(nodes, index + 2,
                    BaseReg(first.Operands[0]),
                    BaseReg(second.Operands[0]));
                string product = $"%rdt{_nextTemp++}";
                _lines.Add($"    mul.wide.u32 {product}, " +
                    $"{PackOperand(first.Operands[1])}, " +
                    $"{PackOperand(second.Operands[2])};");
                string productBase = $"%rdt{_nextTemp++}";
                _lines.Add($"    mov.b64 {productBase}, " +
                    $"{ParamPointer(imadParameter)};");
                _lines.Add($"    add.u64 {product}, {product}, {productBase};");
                EmitAddressUnpack(product, imadParameter, first, second,
                    unpackProduct);
                index += 1;
                return true;
            }

            if (!TryConstantOffset(first.Operands[2], out int baseOffset) ||
                highOffset != baseOffset + 4)
                return false;
            if (!_paramsByOffset.TryGetValue(baseOffset, out PtxParameter? parameter) ||
                parameter.Type != "u64" ||
                !_paramsByOffset.TryGetValue(highOffset, out PtxParameter? highParameter) ||
                highParameter != parameter)
                return false;

            string lowSource;
            if (firstMnemonic == "ISCADD")
            {
                string source = PackOperand(first.Operands[1]);
                uint shift = ParseImmediate(first.Operands[3]);
                if (shift > 31)
                    return false;
                if (shift == 0)
                {
                    lowSource = source;
                }
                else
                {
                    lowSource = $"%rt{_nextTemp++}";
                    _lines.Add($"    shl.b32 {lowSource}, {source}, {shift};");
                }
            }
            else
            {
                lowSource = PackOperand(first.Operands[1]);
            }
            bool unpackValues = NeedsUnpack(nodes, index + 2,
                BaseReg(first.Operands[0]),
                BaseReg(second.Operands[0]));
            EmitAddressPack(lowSource, PackOperand(second.Operands[1]),
                parameter, first, second, unpackValues);
            index += 1;
            return true;
        }

        private void EmitAddressPack(
            string lowSource, string highSource, PtxParameter parameter,
            SassInstruction first, SassInstruction second, bool unpack)
        {
            string address = $"%rdt{_nextTemp++}";
            _lines.Add($"    mov.b64 {address}, {{{lowSource}, {highSource}}};");
            string basePointer = $"%rdt{_nextTemp++}";
            _lines.Add($"    mov.b64 {basePointer}, " +
                $"{ParamPointer(parameter)};");
            _lines.Add($"    add.u64 {address}, {address}, {basePointer};");
            EmitAddressUnpack(address, parameter, first, second, unpack);
        }

        private void EmitAddressUnpack(
            string address, PtxParameter parameter,
            SassInstruction first, SassInstruction second, bool unpack)
        {
            if (unpack)
            {
                string lowDestination = Reg(BaseReg(first.Operands[0]));
                string highDestination = Reg(BaseReg(second.Operands[0]));
                _lines.Add($"    mov.b64 {{{lowDestination}, {highDestination}}}, {address};");
            }
            _pointerRegs[BaseReg(first.Operands[0])] = address;
        }

        private bool NeedsUnpack(
            List<SassNode> nodes, int start, string low, string high)
        {
            bool lowPending = true;
            bool highPending = true;
            for (int i = start; i < nodes.Count; i++)
            {
                if (nodes[i] is not SassInstruction instruction)
                    continue;
                int destination = DestinationIndex(instruction);
                for (int operandIndex = 0;
                     operandIndex < instruction.Operands.Count; operandIndex++)
                {
                    string operand = instruction.Operands[operandIndex];
                    bool hasLow = ContainsRegister(operand, low);
                    bool hasHigh = ContainsRegister(operand, high);
                    if (!hasLow && !hasHigh)
                        continue;
                    if (operandIndex == destination)
                    {
                        if (hasLow)
                            lowPending = false;
                        if (hasHigh)
                            highPending = false;
                    }
                    else if (((hasLow && lowPending) ||
                              (hasHigh && highPending)) &&
                             !operand.TrimStart().StartsWith("[",
                                 StringComparison.Ordinal))
                    {
                        return true;
                    }
                }
                if (!lowPending && !highPending)
                    return false;
            }
            return false;
        }

        private static bool ContainsRegister(string operand, string register)
        {
            return Regex.IsMatch(operand, $@"\b{Regex.Escape(register)}\b",
                RegexOptions.IgnoreCase);
        }

        private static bool RegisterReadAfter(
            List<SassNode> nodes, int start, string register)
        {
            for (int i = start; i < nodes.Count; i++)
            {
                if (nodes[i] is SassLabel)
                    continue;
                if (nodes[i] is not SassInstruction instruction)
                    continue;
                int destination = DestinationIndex(instruction);
                for (int operand = 0;
                     operand < instruction.Operands.Count; operand++)
                {
                    if (operand == destination)
                        continue;
                    if (ContainsRegister(instruction.Operands[operand], register))
                        return true;
                }
                if (destination >= 0 &&
                    destination < instruction.Operands.Count &&
                    ContainsRegister(instruction.Operands[destination], register))
                    return false;
            }
            return false;
        }

        private static bool RegisterNeedsScalarValue(
            List<SassNode> nodes, int start, string register)
        {
            for (int i = start; i < nodes.Count; i++)
            {
                if (nodes[i] is SassLabel)
                    continue;
                if (nodes[i] is not SassInstruction instruction)
                    continue;
                int destination = DestinationIndex(instruction);
                bool scalarRead = false;
                for (int operand = 0;
                     operand < instruction.Operands.Count; operand++)
                {
                    if (operand == destination)
                        continue;
                    string text = instruction.Operands[operand];
                    if (text.TrimStart().StartsWith("[", StringComparison.Ordinal))
                        continue;
                    if (!ContainsRegister(text, register))
                        continue;
                    if (instruction.Mnemonic.Equals(
                            "MOV", StringComparison.OrdinalIgnoreCase) &&
                        destination >= 0 &&
                        destination < instruction.Operands.Count &&
                        BaseReg(instruction.Operands[destination]).Equals(
                            register, StringComparison.OrdinalIgnoreCase) &&
                        BaseReg(text).Equals(
                            register, StringComparison.OrdinalIgnoreCase))
                        continue;
                    scalarRead = true;
                    break;
                }
                if (scalarRead)
                    return true;
                if (destination >= 0 &&
                    destination < instruction.Operands.Count &&
                    ContainsRegister(instruction.Operands[destination], register))
                    return false;
            }
            return false;
        }

        private static int DestinationIndex(SassInstruction instruction)
        {
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            if (mnemonic.StartsWith("ST", StringComparison.Ordinal) ||
                mnemonic.StartsWith("BAR", StringComparison.Ordinal) ||
                mnemonic.StartsWith("MEMBAR", StringComparison.Ordinal) ||
                mnemonic.StartsWith("BRA", StringComparison.Ordinal) ||
                mnemonic.StartsWith("EXIT", StringComparison.Ordinal) ||
                mnemonic.StartsWith("RET", StringComparison.Ordinal) ||
                mnemonic.StartsWith("NOP", StringComparison.Ordinal) ||
                mnemonic.StartsWith("CCTL", StringComparison.Ordinal) ||
                mnemonic.StartsWith("SSY", StringComparison.Ordinal) ||
                mnemonic.StartsWith("PBK", StringComparison.Ordinal) ||
                mnemonic.StartsWith("PCNT", StringComparison.Ordinal) ||
                mnemonic.StartsWith("BSSY", StringComparison.Ordinal) ||
                mnemonic.StartsWith("BSYNC", StringComparison.Ordinal) ||
                mnemonic.StartsWith("DEPBAR", StringComparison.Ordinal) ||
                mnemonic.StartsWith("WARPSYNC", StringComparison.Ordinal))
                return -1;
            if (mnemonic.StartsWith("SHFL", StringComparison.Ordinal))
                return 1;
            return 0;
        }

        private string PackOperand(string operand)
        {
            string value = Operand(operand);
            if (value == "0")
            {
                string zero = $"%rt{_nextTemp++}";
                _lines.Add($"    mov.u32 {zero}, 0;");
                return zero;
            }
            if (value.StartsWith("%rd", StringComparison.Ordinal))
                throw new NotSupportedException(
                    $"cannot pack 64-bit operand '{operand}' into an address pair");
            return value;
        }

        private static bool TryConstantOffset(string operand, out int offset)
        {
            offset = 0;
            Match match = Regex.Match(operand.Trim(),
                @"^c\[0x0\]\[0x([0-9A-Fa-f]+)\]$",
                RegexOptions.IgnoreCase);
            if (!match.Success)
                return false;
            offset = int.Parse(match.Groups[1].Value,
                NumberStyles.HexNumber, CultureInfo.InvariantCulture);
            return true;
        }

        private bool TryEmitInterleavedXmad(List<SassNode> nodes, ref int index)
        {
            if (index + 2 >= nodes.Count)
                return false;
            if (nodes[index] is not SassInstruction mrg ||
                !mrg.Mnemonic.StartsWith("XMAD.MRG",
                    StringComparison.OrdinalIgnoreCase) ||
                mrg.Guard != null || mrg.Operands.Count < 4)
                return false;
            string mrgDestination = BaseReg(mrg.Operands[0]);
            string multiplicand = BaseReg(mrg.Operands[1]);
            string multiplier = BaseReg(mrg.Operands[2]);
            if (!IsRz(mrg.Operands[3]))
                return false;
            for (int xmadIndex = index + 1;
                 xmadIndex <= index + 3 && xmadIndex + 1 < nodes.Count;
                 xmadIndex++)
            {
                if (nodes[xmadIndex] is not SassInstruction xmad ||
                    !xmad.Mnemonic.Equals("XMAD", StringComparison.OrdinalIgnoreCase) ||
                    xmad.Guard != null || xmad.Operands.Count < 4)
                    continue;
                if (!BaseReg(xmad.Operands[1]).Equals(multiplicand,
                        StringComparison.OrdinalIgnoreCase) ||
                    !BaseReg(xmad.Operands[2]).Equals(multiplier,
                        StringComparison.OrdinalIgnoreCase))
                    continue;
                if (nodes[xmadIndex + 1] is not SassInstruction psl ||
                    !psl.Mnemonic.StartsWith("XMAD.PSL",
                        StringComparison.OrdinalIgnoreCase) ||
                    psl.Guard != null || psl.Operands.Count < 4)
                    continue;
                if (!BaseReg(psl.Operands[1]).Equals(multiplicand,
                        StringComparison.OrdinalIgnoreCase) ||
                    !BaseReg(psl.Operands[2]).Equals(mrgDestination,
                        StringComparison.OrdinalIgnoreCase) ||
                    !BaseReg(psl.Operands[3]).Equals(
                        BaseReg(xmad.Operands[0]),
                        StringComparison.OrdinalIgnoreCase))
                    continue;
                bool safe = true;
                for (int middle = index + 1; middle < xmadIndex; middle++)
                {
                    if (nodes[middle] is not SassInstruction intervening)
                        continue;
                    int destination = DestinationIndex(intervening);
                    if (destination < 0)
                        continue;
                    string written = BaseReg(intervening.Operands[destination]);
                    if (written.Equals(multiplicand,
                            StringComparison.OrdinalIgnoreCase) ||
                        written.Equals(multiplier,
                            StringComparison.OrdinalIgnoreCase) ||
                        written.Equals(mrgDestination,
                            StringComparison.OrdinalIgnoreCase))
                    {
                        safe = false;
                        break;
                    }
                }
                if (!safe)
                    continue;
                for (int middle = index + 1; middle < xmadIndex; middle++)
                {
                    if (nodes[middle] is SassInstruction intervening)
                        EmitInstruction(intervening);
                }
                _lines.Add($"mad.lo.s32 {Reg(psl.Operands[0])}, " +
                    $"{Operand(mrg.Operands[1])}, " +
                    $"{Operand(xmad.Operands[2])}, " +
                    $"{Operand(xmad.Operands[3])};");
                index = xmadIndex + 1;
                return true;
            }
            return false;
        }

        private bool TryEmitXmadMultiply(List<SassNode> nodes, ref int index)
        {
            if (index + 2 >= nodes.Count)
                return false;
            if (nodes[index] is not SassInstruction first ||
                first.Guard != null ||
                !first.Mnemonic.Equals("XMAD", StringComparison.OrdinalIgnoreCase) ||
                first.Operands.Count < 4)
                return false;
            string d1 = first.Operands[0];
            string a = first.Operands[1];
            string b = first.Operands[2];

            bool MatchMrg(SassInstruction mrg) =>
                mrg.Guard == null &&
                mrg.Mnemonic.StartsWith("XMAD.MRG",
                    StringComparison.OrdinalIgnoreCase) &&
                mrg.Operands.Count >= 4 &&
                BaseReg(a).Equals(BaseReg(mrg.Operands[1]),
                    StringComparison.OrdinalIgnoreCase) &&
                BaseReg(b).Equals(BaseReg(mrg.Operands[2]),
                    StringComparison.OrdinalIgnoreCase) &&
                IsRz(mrg.Operands[3]);

            SassInstruction? second = null;
            int mrgIndex = -1;
            for (int candidate = index + 1;
                 candidate <= index + 3 && candidate < nodes.Count; candidate++)
            {
                if (nodes[candidate] is SassInstruction mrg &&
                    MatchMrg(mrg))
                {
                    second = mrg;
                    mrgIndex = candidate;
                    break;
                }
            }
            if (second == null)
            {
                // A second multiply chain can interleave its XMAD.MRG
                // before its XMAD.  Reuse that merge operation when the
                // register pair matches.
                for (int candidate = index - 1; candidate >= 0; candidate--)
                {
                    if (nodes[candidate] is SassInstruction mrg &&
                        MatchMrg(mrg))
                    {
                        second = mrg;
                        mrgIndex = candidate;
                        break;
                    }
                }
            }
            if (second == null)
                return false;
            string d2 = second.Operands[0];

            SassInstruction? third = null;
            int pslIndex = -1;
            int pslStart = Math.Max(index, mrgIndex) + 1;
            for (int candidate = pslStart;
                 candidate <= pslStart + 2 && candidate < nodes.Count;
                 candidate++)
            {
                if (nodes[candidate] is SassInstruction psl &&
                    psl.Guard == null &&
                    psl.Mnemonic.StartsWith("XMAD.PSL.CBCC",
                        StringComparison.OrdinalIgnoreCase) &&
                    psl.Operands.Count >= 4 &&
                    BaseReg(a).Equals(BaseReg(psl.Operands[1]),
                        StringComparison.OrdinalIgnoreCase) &&
                    BaseReg(d2).Equals(BaseReg(psl.Operands[2]),
                        StringComparison.OrdinalIgnoreCase) &&
                    BaseReg(d1).Equals(BaseReg(psl.Operands[3]),
                        StringComparison.OrdinalIgnoreCase))
                {
                    third = psl;
                    pslIndex = candidate;
                    break;
                }
            }
            if (third == null)
                return false;

            string result = BaseReg(third.Operands[0]);
            for (int middle = index + 1; middle < pslIndex; middle++)
            {
                if (middle == mrgIndex)
                    continue;
                if (nodes[middle] is not SassInstruction intervening ||
                    intervening.Guard != null)
                    return false;
                if (intervening.Mnemonic.StartsWith(
                        "XMAD.MRG", StringComparison.OrdinalIgnoreCase) ||
                    intervening.Mnemonic.StartsWith(
                        "XMAD.PSL", StringComparison.OrdinalIgnoreCase))
                    continue;
                int destination = DestinationIndex(intervening);
                if (destination >= 0)
                {
                    string written = BaseReg(intervening.Operands[destination]);
                    if (written.Equals(BaseReg(a),
                            StringComparison.OrdinalIgnoreCase) ||
                        written.Equals(BaseReg(d1),
                            StringComparison.OrdinalIgnoreCase) ||
                        written.Equals(BaseReg(d2),
                            StringComparison.OrdinalIgnoreCase))
                        return false;
                }
                for (int operandIndex = 0;
                     operandIndex < intervening.Operands.Count; operandIndex++)
                {
                    if (operandIndex == destination)
                        continue;
                    if (ContainsRegister(intervening.Operands[operandIndex],
                            result) ||
                        ContainsRegister(intervening.Operands[operandIndex],
                            d2))
                        return false;
                }
            }

            _lines.Add($"mad.lo.s32 {Reg(third.Operands[0])}, " +
                $"{Operand(a)}, {Operand(b)}, {Operand(first.Operands[3])};");
            for (int middle = index + 1; middle < pslIndex; middle++)
            {
                if (middle == mrgIndex)
                    continue;
                if (nodes[middle] is SassInstruction intervening &&
                    !intervening.Mnemonic.StartsWith(
                        "XMAD.MRG", StringComparison.OrdinalIgnoreCase) &&
                    !intervening.Mnemonic.StartsWith(
                        "XMAD.PSL", StringComparison.OrdinalIgnoreCase))
                    EmitInstruction(intervening);
            }
            index = pslIndex;
            return true;
        }

        private static bool IsRz(string operand) =>
            operand.Trim().TrimStart('!').Equals(
                "RZ", StringComparison.OrdinalIgnoreCase);

        private static string BaseReg(string operand)
        {
            Match match = Regex.Match(operand,
                @"\b(R\d+)\b", RegexOptions.IgnoreCase);
            return match.Success ? match.Value : operand.Trim();
        }

        private bool TryEmitConstantLdc(List<SassNode> nodes, ref int index)
        {
            if (nodes[index] is not SassInstruction mov ||
                !mov.Mnemonic.Equals("MOV32I", StringComparison.OrdinalIgnoreCase) ||
                mov.Operands.Count < 2 ||
                !string.IsNullOrWhiteSpace(mov.Guard))
                return false;
            string addressRegister = BaseReg(mov.Operands[0]);
            int ldcIndex = index + 1;
            while (ldcIndex < nodes.Count &&
                   nodes[ldcIndex] is SassInstruction between)
            {
                if (between.Mnemonic.StartsWith(
                        "NOP", StringComparison.OrdinalIgnoreCase))
                {
                    ldcIndex++;
                    continue;
                }
                if (between.Mnemonic.Equals("MOV", StringComparison.OrdinalIgnoreCase) &&
                    between.Operands.Count == 2 &&
                    BaseReg(between.Operands[0]).Equals(
                        addressRegister, StringComparison.OrdinalIgnoreCase) &&
                    BaseReg(between.Operands[1]).Equals(
                        addressRegister, StringComparison.OrdinalIgnoreCase))
                {
                    ldcIndex++;
                    continue;
                }
                break;
            }
            if (ldcIndex >= nodes.Count ||
                nodes[ldcIndex] is not SassInstruction ldc ||
                !ldc.Mnemonic.StartsWith("LDC", StringComparison.OrdinalIgnoreCase) ||
                ldc.Mnemonic.Contains(".128", StringComparison.OrdinalIgnoreCase) ||
                ldc.Operands.Count < 2 ||
                !string.IsNullOrWhiteSpace(ldc.Guard))
                return false;
            if (!TryParseConstant0Address(
                    ldc.Operands[1], out string addressBase,
                    out int addressOffset) ||
                !addressBase.Equals(
                    addressRegister, StringComparison.OrdinalIgnoreCase))
                return false;

            string destination = BaseReg(ldc.Operands[0]);
            if (!destination.Equals(addressRegister, StringComparison.OrdinalIgnoreCase))
            {
                for (int i = ldcIndex + 1; i < nodes.Count; i++)
                {
                    if (nodes[i] is SassInstruction later &&
                        later.Operands.Any(operand =>
                            ContainsRegister(operand, addressRegister)))
                        return false;
                }
            }

            int offset = (int)ParseImmediate(mov.Operands[1]) + addressOffset;
            bool wide = ldc.Mnemonic.Contains(
                ".64", StringComparison.OrdinalIgnoreCase);
            if (_paramsByOffset.TryGetValue(offset, out PtxParameter? parameter))
            {
                int parameterIndex = ParameterIndex(parameter);
                if (parameter.Type == "u64")
                {
                    if (wide)
                    {
                        _lines.Add(
                            $"    mov.b64 {{{Reg(destination)}, " +
                            $"{Reg(NextRegister(destination))}}}, " +
                            $"%rdp{parameterIndex};");
                        _pointerRegs[destination] = ParamPointer(parameter);
                    }
                    else
                    {
                        string low = $"%ru{_nextTempU32++}";
                        string high = $"%ru{_nextTempU32++}";
                        _lines.Add(
                            $"    mov.b64 {{{low}, {high}}}, " +
                            $"%rdp{parameterIndex};");
                        _lines.Add(
                            $"    mov.b32 {Reg(destination)}, {low};");
                    }
                }
                else
                {
                    _lines.Add($"    mov.u32 {Reg(destination)}, " +
                        $"%rpp{parameterIndex};");
                    if (wide)
                        _lines.Add($"    mov.u32 " +
                            $"{Reg(NextRegister(destination))}, 0;");
                }
                index = ldcIndex;
                return true;
            }
            if (_constantResolver != null)
            {
                uint? lowValue = _constantResolver(offset);
                uint? highValue = _constantResolver(offset + 4);
                if (!lowValue.HasValue || (wide && !highValue.HasValue))
                    return false;
                _lines.Add($"    mov.u32 {Reg(destination)}, " +
                    $"{lowValue.Value};");
                if (wide)
                    _lines.Add($"    mov.u32 " +
                        $"{Reg(NextRegister(destination))}, " +
                        $"{highValue!.Value};");
                index = ldcIndex;
                return true;
            }
            return false;
        }

        private bool TryEmitStackCheckSkip(List<SassNode> nodes, ref int index)
        {
            if (index + 4 >= nodes.Count)
                return false;
            if (nodes[index] is not SassInstruction s2r ||
                !s2r.Mnemonic.Equals("S2R", StringComparison.OrdinalIgnoreCase) ||
                s2r.Operands.Count < 2 ||
                !s2r.Operands[1].Contains(
                    "SR_LMEMHIOFF", StringComparison.OrdinalIgnoreCase))
                return false;
            if (nodes[index + 1] is not SassInstruction compare ||
                !compare.Mnemonic.StartsWith(
                    "ISETP", StringComparison.OrdinalIgnoreCase) ||
                !(compare.Mnemonic.Contains(
                      ".GE", StringComparison.OrdinalIgnoreCase) ||
                  compare.Mnemonic.Contains(
                      ".GT", StringComparison.OrdinalIgnoreCase)) ||
                compare.Operands.Count < 5)
                return false;
            string special = BaseReg(s2r.Operands[0]);
            string predicate = BaseReg(compare.Operands[0]);
            if (!ContainsRegister(compare.Operands[2], "R1") ||
                !ContainsRegister(compare.Operands[3], special))
                return false;
            if (nodes[index + 2] is not SassInstruction branch ||
                !branch.Mnemonic.Equals("BRA", StringComparison.OrdinalIgnoreCase) ||
                string.IsNullOrWhiteSpace(branch.Guard) ||
                !branch.Guard.Trim().Equals(
                    "@" + predicate, StringComparison.OrdinalIgnoreCase))
                return false;
            if (nodes[index + 3] is not SassInstruction bpt ||
                !bpt.Mnemonic.StartsWith("BPT", StringComparison.OrdinalIgnoreCase))
                return false;
            if (nodes[index + 4] is not SassLabel label)
                return false;
            string target;
            try
            {
                target = BranchLabel(branch);
            }
            catch (NotSupportedException)
            {
                return false;
            }
            if (!LabelName(label.Name).Equals(target, StringComparison.Ordinal))
                return false;
            for (int i = index; i < index + 4; i++)
                _skipNodes.Add(i);
            index += 3;
            return true;
        }

        // CUDA's device-side assertion path loads __assertfail from the
        // constant-bank-4 function table and calls it indirectly.  The
        // table entry is an undefined external symbol.  Emit an in-kernel
        // `trap` rather than `call __assertfail`: a call to an unresolved
        // symbol routes the whole kernel through the host-call state
        // machine, whose per-segment register reloads miscompile the
        // surrounding reduction loops.  `trap` lowers to an invocation
        // exit and is never reached unless an assertion actually fires.
        private bool TryEmitAssertBlock(List<SassNode> nodes, ref int index)
        {
            if (nodes[index] is not SassInstruction ldc ||
                ldc.Guard != null ||
                !ldc.Mnemonic.StartsWith(
                    "LDC.64", StringComparison.OrdinalIgnoreCase) ||
                ldc.Operands.Count < 2)
                return false;
            Match address = Regex.Match(ldc.Operands[1],
                @"^c\[0x4\]\[(R\d+)\]$", RegexOptions.IgnoreCase);
            if (!address.Success)
                return false;
            string loadedRegister = BaseReg(ldc.Operands[0]);
            int callIndex = -1;
            for (int j = index + 1; j < nodes.Count && j <= index + 64; j++)
            {
                if (nodes[j] is SassLabel)
                    break;
                if (nodes[j] is not SassInstruction candidate)
                    continue;
                if (!candidate.Mnemonic.StartsWith(
                        "CALL.ABS", StringComparison.OrdinalIgnoreCase) ||
                    candidate.Operands.Count == 0 ||
                    !BaseReg(candidate.Operands[0]).Equals(
                        loadedRegister, StringComparison.OrdinalIgnoreCase))
                    continue;
                callIndex = j;
                break;
            }
            if (callIndex < 0)
                return false;
            _lines.Add("    trap;");
            for (int j = index; j <= callIndex; j++)
                _skipNodes.Add(j);
            index = callIndex;
            return true;
        }

        // Device-runtime calls (__assertfail and friends) are emitted by
        // ptxas as an explicit-return-address sequence: LEPC captures the
        // PC, a few MOV/IADD3 build the return address, and CALL.ABS
        // dispatches through a function pointer in constant bank 4.  The
        // call is a trap path, so lower the whole sequence to `trap`.
        // Detecting it at LEPC is reliable; the LDC that loads the
        // pointer is often part of a fused pair and never visited.
        private bool TryEmitRuntimeCallTrap(List<SassNode> nodes, ref int index)
        {
            if (nodes[index] is not SassInstruction lepc ||
                !lepc.Mnemonic.Equals("LEPC", StringComparison.OrdinalIgnoreCase))
                return false;
            // The function pointer may be loaded before or after LEPC,
            // depending on how ptxas scheduled the block.
            bool fromTable = false;
            int lo = Math.Max(0, index - 24);
            int hi = Math.Min(nodes.Count - 1, index + 20);
            for (int j = lo; j <= hi; j++)
            {
                if (nodes[j] is not SassInstruction producer)
                    continue;
                if (producer.Mnemonic.StartsWith(
                        "LDC.64", StringComparison.OrdinalIgnoreCase) &&
                    producer.Operands.Count >= 2 &&
                    producer.Operands[1].Contains(
                        "c[0x4]", StringComparison.OrdinalIgnoreCase))
                {
                    fromTable = true;
                    break;
                }
            }
            if (!fromTable)
                return false;
            int callIndex = -1;
            for (int j = index + 1; j < nodes.Count && j <= index + 20; j++)
            {
                if (nodes[j] is SassLabel)
                    break;
                if (nodes[j] is not SassInstruction call)
                    continue;
                if (call.Mnemonic.StartsWith(
                        "CALL.ABS", StringComparison.OrdinalIgnoreCase))
                {
                    callIndex = j;
                    break;
                }
                if (call.Mnemonic.StartsWith(
                        "BRA", StringComparison.OrdinalIgnoreCase))
                    break;
            }
            if (callIndex < 0)
                return false;
            _lines.Add("    trap;");
            for (int j = index; j <= callIndex; j++)
                _skipNodes.Add(j);
            index = callIndex;
            return true;
        }

        private bool TryEmitUnboundedLdc(List<SassNode> nodes, ref int index)
        {
            if (nodes[index] is not SassInstruction ldc ||
                !ldc.Mnemonic.StartsWith("LDC", StringComparison.OrdinalIgnoreCase) ||
                ldc.Mnemonic.Contains(".128", StringComparison.OrdinalIgnoreCase) ||
                ldc.Operands.Count < 2)
                return false;
            if (!TryParseConstant0Address(
                    ldc.Operands[1], out string addressBase,
                    out int addressOffset))
                return false;
            string destination = BaseReg(ldc.Operands[0]);
            if (destination.Equals("RZ", StringComparison.OrdinalIgnoreCase))
                return false;
            string addressRegister = Operand(addressBase);
            string temporary = $"%rt{_nextTemp++}";
            string prefix = Guard(ldc.Guard);
            prefix = prefix.Length == 0 ? "" : prefix + " ";
            _lines.Add($"mov.u32 {temporary}, {addressRegister};");
            if (addressOffset != 0)
                _lines.Add(
                    $"add.s32 {temporary}, {temporary}, {addressOffset};");
            _lines.Add(
                $"{prefix}ld.const.u32 {Reg(destination)}, " +
                $"[{temporary}];");
            if (ldc.Mnemonic.Contains(".64", StringComparison.OrdinalIgnoreCase))
            {
                _lines.Add(
                    $"{prefix}ld.const.u32 " +
                    $"{Reg(NextRegister(destination))}, " +
                    $"[{temporary}+4];");
            }
            return true;
        }

        private bool TryEmitDynamicLdc(List<SassNode> nodes, ref int index)
        {
            if (!_dynamicLdc ||
                nodes[index] is not SassInstruction ldc ||
                !ldc.Mnemonic.StartsWith("LDC", StringComparison.OrdinalIgnoreCase) ||
                ldc.Mnemonic.Contains(".128", StringComparison.OrdinalIgnoreCase) ||
                ldc.Operands.Count < 2 ||
                !string.IsNullOrWhiteSpace(ldc.Guard))
                return false;
            if (!TryParseConstant0Address(
                    ldc.Operands[1], out string addressBase,
                    out int addressOffset))
                return false;
            string destination = BaseReg(ldc.Operands[0]);
            if (destination.Equals("RZ", StringComparison.OrdinalIgnoreCase))
                return false;
            bool wide = ldc.Mnemonic.Contains(".64", StringComparison.OrdinalIgnoreCase);
            string indexOperand = Operand(addressBase);
            if (addressOffset != 0)
            {
                string adjusted = $"%rt{_nextTemp++}";
                _lines.Add(
                    $"    add.s32 {adjusted}, {indexOperand}, {addressOffset};");
                indexOperand = adjusted;
            }
            string offset = $"%rt{_nextTemp++}";
            string basePredicate = $"%pt{_nextTempPred++}";
            _lines.Add(
                $"    setp.ge.u32 {basePredicate}, {indexOperand}, " +
                $"0x{_paramBase:x};");
            _lines.Add(
                $"    @{basePredicate} sub.u32 {offset}, {indexOperand}, " +
                $"0x{_paramBase:x};");
            _lines.Add($"    @!{basePredicate} mov.u32 {offset}, 0;");
            string boundPredicate = $"%pt{_nextTempPred++}";
            _lines.Add(
                $"    setp.lt.u32 {boundPredicate}, {offset}, " +
                $"{_paramBlockSize};");
            _lines.Add($"    @!{boundPredicate} mov.u32 {offset}, 0;");
            string sharedOffset = $"%rdt{_nextTempDouble++}";
            string sharedBase = $"%rdt{_nextTempDouble++}";
            _lines.Add($"    cvt.u64.u32 {sharedOffset}, {offset};");
            // The dynamic-LDC index is parameter-image relative, and the
            // image lives in __param_block, not in the static/dynamic
            // __shared arena.
            _lines.Add($"    mov.u64 {sharedBase}, __param_block;");
            _lines.Add(
                $"    add.u64 {sharedBase}, {sharedBase}, {sharedOffset};");
            string sharedAddress = $"[{sharedBase}+0x{_paramBlockOffset:x}]";
            _lines.Add(
                $"    ld.shared.u32 {Reg(destination)}, {sharedAddress};");
            if (wide)
            {
                string highAddress =
                    $"[{sharedBase}+0x{_paramBlockOffset + 4:x}]";
                _lines.Add(
                    $"    ld.shared.u32 {Reg(NextRegister(destination))}, " +
                    $"{highAddress};");
                string packed = $"%rdt{_nextTempDouble++}";
                _lines.Add(
                    $"    mov.b64 {packed}, " +
                    $"{{{Reg(destination)}, {Reg(NextRegister(destination))}}};");
                _pointerRegs[destination] = packed;
            }
            return true;
        }

        private bool TryEmitJumpTable(List<SassNode> nodes, ref int index)
        {
            if (nodes[index] is not SassInstruction ldc ||
                !ldc.Mnemonic.StartsWith("LDC", StringComparison.OrdinalIgnoreCase) ||
                ldc.Mnemonic.Contains(".64", StringComparison.OrdinalIgnoreCase) ||
                ldc.Mnemonic.Contains(".128", StringComparison.OrdinalIgnoreCase) ||
                ldc.Operands.Count < 2 ||
                !string.IsNullOrWhiteSpace(ldc.Guard) ||
                _constant2Resolver == null || _constant2Bytes < 4)
                return false;
            Match address = Regex.Match(ldc.Operands[1],
                @"^c\[0x2\]\[(R\d+)\]$", RegexOptions.IgnoreCase);
            if (!address.Success)
                return false;

            int brxIndex = index + 1;
            while (brxIndex < nodes.Count &&
                   nodes[brxIndex] is SassInstruction nop &&
                   nop.Mnemonic.StartsWith(
                       "NOP", StringComparison.OrdinalIgnoreCase))
                brxIndex++;
            if (brxIndex >= nodes.Count ||
                nodes[brxIndex] is not SassInstruction brx ||
                !brx.Mnemonic.Equals("BRX", StringComparison.OrdinalIgnoreCase) ||
                brx.Operands.Count < 2 ||
                !string.IsNullOrWhiteSpace(brx.Guard) ||
                !BaseReg(brx.Operands[0]).Equals(
                    BaseReg(ldc.Operands[0]), StringComparison.OrdinalIgnoreCase))
                return false;

            string destination = BaseReg(ldc.Operands[0]);
            if (RegisterReadAfter(nodes, brxIndex + 1, destination))
                return false;

            var targets = new List<string>();
            int entries = _constant2Bytes / 4;
            for (int i = 0; i < entries; i++)
            {
                uint? value = _constant2Resolver(i * 4);
                if (!value.HasValue)
                    return false;
                if (value.Value == 0)
                    break;
                string? target = JumpLabelForAddress(nodes, (int)value.Value);
                if (target == null)
                    return false;
                targets.Add(target);
            }
            if (targets.Count == 0)
                return false;
            string? defaultTarget = JumpLabelForAddress(nodes, brx.Address + 8);
            if (defaultTarget == null)
                return false;

            string indexOperand = Operand(address.Groups[1].Value);
            if (TryEmitJumpTableSelect(
                    nodes, brxIndex, targets, defaultTarget,
                    indexOperand, out int mergeIndex))
            {
                for (int i = index; i < mergeIndex; i++)
                    _skipNodes.Add(i);
                index = mergeIndex - 1;
                return true;
            }
            for (int i = 0; i < targets.Count; i++)
            {
                string predicate = $"%pt{_nextTempPred++}";
                _lines.Add($"setp.eq.u32 {predicate}, {indexOperand}, {i * 4};");
                _lines.Add($"@{predicate} bra {targets[i]};");
            }
            _lines.Add($"bra {defaultTarget};");
            index = brxIndex;
            return true;
        }

        private bool TryEmitJumpTableSelect(
            List<SassNode> nodes, int brxIndex,
            List<string> targets, string defaultTarget, string indexOperand,
            out int mergeIndex)
        {
            mergeIndex = -1;
            if (targets.Count < 4 || !targets.Contains(defaultTarget))
                return false;
            string? mergeLabel = null;
            var fallthroughLabels = new List<string>();
            var constants = new List<string>();
            var targetIndices = new List<int>();
            string? destination = null;
            for (int i = 0; i < targets.Count; i++)
            {
                int labelIndex = FindLabelNodeIndex(nodes, targets[i]);
                if (labelIndex < 0)
                    return false;
                targetIndices.Add(labelIndex);
                string? constant = null;
                string? blockDestination = null;
                string? branchTarget = null;
                string? fallthrough = null;
                int scan = labelIndex;
                while (scan < nodes.Count && nodes[scan] is SassLabel)
                    scan++;
                for (; scan < nodes.Count; scan++)
                {
                    if (nodes[scan] is SassLabel label)
                    {
                        fallthrough = LabelName(label.Name);
                        break;
                    }
                    if (nodes[scan] is not SassInstruction instruction)
                        return false;
                    string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
                    if (op == "NOP")
                        continue;
                    if (op is "MOV" or "MOV32I")
                    {
                        if (instruction.Operands.Count < 2)
                            return false;
                        string blockDest = BaseReg(instruction.Operands[0]);
                        if (op == "MOV" &&
                            BaseReg(instruction.Operands[1]).Equals(
                                blockDest, StringComparison.OrdinalIgnoreCase))
                            continue;
                        if (!TrySwitchImmediate(instruction.Operands[1], out _) ||
                            blockDest.Equals("RZ", StringComparison.OrdinalIgnoreCase))
                            return false;
                        if (constant != null)
                            return false;
                        constant = instruction.Operands[1];
                        blockDestination = blockDest;
                        continue;
                    }
                    if (op == "BRA" &&
                        string.IsNullOrWhiteSpace(instruction.Guard))
                    {
                        try
                        {
                            branchTarget = BranchLabel(instruction);
                        }
                        catch (NotSupportedException)
                        {
                            return false;
                        }
                        break;
                    }
                    return false;
                }
                if (constant == null || blockDestination == null)
                    return false;
                if (destination == null)
                    destination = blockDestination;
                else if (!destination.Equals(
                             blockDestination, StringComparison.OrdinalIgnoreCase))
                    return false;
                constants.Add(Immediate(constant));
                if (branchTarget != null)
                {
                    if (mergeLabel == null)
                        mergeLabel = branchTarget;
                    else if (!mergeLabel.Equals(
                                 branchTarget, StringComparison.Ordinal))
                        return false;
                }
                else if (fallthrough != null)
                {
                    fallthroughLabels.Add(fallthrough);
                }
                else
                {
                    return false;
                }
            }
            if (mergeLabel == null)
            {
                if (fallthroughLabels.Count == 0)
                    return false;
                mergeLabel = fallthroughLabels[0];
            }
            foreach (string fallthrough in fallthroughLabels)
            {
                if (!fallthrough.Equals(mergeLabel, StringComparison.Ordinal))
                    return false;
            }
            mergeIndex = FindLabelNodeIndex(nodes, mergeLabel);
            if (mergeIndex <= brxIndex)
                return false;
            foreach (int targetIndex in targetIndices)
            {
                if (targetIndex >= mergeIndex)
                    return false;
            }

            int defaultIndex = targets.IndexOf(defaultTarget);
            if (defaultIndex < 0 || destination == null)
                return false;
            string result = Reg(destination);
            string defaultConstant = constants[defaultIndex];
            _lines.Add($"mov.u32 {result}, {defaultConstant};");
            for (int i = 0; i < constants.Count; i++)
            {
                if (i == defaultIndex || constants[i] == defaultConstant)
                    continue;
                string predicate = $"%pt{_nextTempPred++}";
                _lines.Add(
                    $"setp.eq.u32 {predicate}, {indexOperand}, {i * 4};");
                _lines.Add(
                    $"@{predicate} mov.u32 {result}, {constants[i]};");
            }
            return true;
        }

        private int FindLabelNodeIndex(List<SassNode> nodes, string labelName)
        {
            for (int i = 0; i < nodes.Count; i++)
            {
                if (nodes[i] is SassLabel label &&
                    LabelName(label.Name).Equals(
                        labelName, StringComparison.Ordinal))
                    return i;
            }
            foreach (KeyValuePair<int, string> entry in _jumpTableLabels)
            {
                if (!entry.Value.Equals(labelName, StringComparison.Ordinal))
                    continue;
                for (int i = 0; i < nodes.Count; i++)
                {
                    if (nodes[i].Address == entry.Key)
                        return i;
                }
            }
            return -1;
        }

        private string? JumpLabelForAddress(List<SassNode> nodes, int address)
        {
            if (address <= 0)
                return null;
            int emissionAddress = address;
            if (!nodes.Any(node => node.Address == address))
            {
                int next = nodes
                    .Where(node => node.Address > address)
                    .Select(node => node.Address)
                    .DefaultIfEmpty(0)
                    .Min();
                if (next == 0 || next - address > 8)
                    return null;
                emissionAddress = next;
            }
            if (_jumpTableLabels.TryGetValue(
                    emissionAddress, out string? existing))
                return existing;
            foreach (SassNode node in nodes)
            {
                if (node is SassLabel label && label.Address == emissionAddress)
                    return LabelName(label.Name);
            }
            string labelName = $"L_jt_{_jumpTableLabels.Count}";
            _jumpTableLabels[emissionAddress] = labelName;
            return labelName;
        }

        private bool TryEmitSync(List<SassNode> nodes, ref int index)
        {
            if (nodes[index] is not SassInstruction instruction)
                return false;
            string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
            if (op is not "SYNC" and not "PBK" and not "PCNT")
                return false;
            if (op == "SYNC")
            {
                // CC15.TRUE/FALSE is a warp reconvergence token, not a
                // branch target; the SIMT branch structure already
                // reconverges at the enclosing SSY point.
                string token = (instruction.BranchTargets.FirstOrDefault()
                        ?? instruction.Operands.LastOrDefault() ?? "")
                    .Trim().Trim('`', '(', ')', ';');
                if (token.StartsWith(
                        "CC", StringComparison.OrdinalIgnoreCase))
                    return true;
            }
            string target;
            try
            {
                target = BranchLabel(instruction);
            }
            catch (NotSupportedException)
            {
                return false;
            }
            int next = index + 1;
            while (next < nodes.Count &&
                   nodes[next] is SassInstruction nop &&
                   nop.Mnemonic.StartsWith(
                       "NOP", StringComparison.OrdinalIgnoreCase))
                next++;
            if (next < nodes.Count && nodes[next] is SassLabel label &&
                LabelName(label.Name).Equals(target, StringComparison.Ordinal))
                return true;
            string guard = Guard(instruction.Guard);
            _lines.Add($"{(guard.Length == 0 ? "" : guard + " ")}bra {target};");
            return true;
        }

        private bool TryEmitNopSync(List<SassNode> nodes, ref int index)
        {
            if (nodes[index] is not SassInstruction instruction ||
                !instruction.Mnemonic.StartsWith(
                    "NOP", StringComparison.OrdinalIgnoreCase) ||
                !instruction.Mnemonic.Contains(
                    ".S", StringComparison.OrdinalIgnoreCase) ||
                instruction.BranchTargets.Count == 0)
                return false;
            string target;
            try
            {
                target = BranchLabel(instruction);
            }
            catch (NotSupportedException)
            {
                return false;
            }
            int next = index + 1;
            while (next < nodes.Count &&
                   nodes[next] is SassInstruction nop &&
                   nop.Mnemonic.StartsWith(
                       "NOP", StringComparison.OrdinalIgnoreCase))
                next++;
            if (next < nodes.Count && nodes[next] is SassLabel label &&
                LabelName(label.Name).Equals(target, StringComparison.Ordinal))
                return true;
            string guard = Guard(instruction.Guard);
            _lines.Add($"{(guard.Length == 0 ? "" : guard + " ")}bra {target};");
            return true;
        }

        private bool TryEmitEarlyExitRegion(List<SassNode> nodes, ref int index)
        {
            if (nodes[index] is not SassInstruction ssy ||
                !ssy.Mnemonic.StartsWith("SSY", StringComparison.OrdinalIgnoreCase))
                return false;

            string target;
            try
            {
                target = BranchLabel(ssy);
            }
            catch (NotSupportedException)
            {
                return false;
            }

            int guardedSyncIndex = -1;
            SassInstruction? guardedSync = null;
            for (int i = index + 1; i < nodes.Count; i++)
            {
                if (nodes[i] is not SassInstruction candidate ||
                    !candidate.Mnemonic.StartsWith("SYNC", StringComparison.OrdinalIgnoreCase) ||
                    string.IsNullOrWhiteSpace(candidate.Guard) ||
                    !TargetMatches(candidate, target))
                    continue;
                guardedSyncIndex = i;
                guardedSync = candidate;
                break;
            }
            if (guardedSync == null)
                return false;

            int endSyncIndex = -1;
            for (int i = guardedSyncIndex + 1; i < nodes.Count; i++)
            {
                if (nodes[i] is not SassInstruction candidate ||
                    !candidate.Mnemonic.StartsWith("SYNC", StringComparison.OrdinalIgnoreCase) ||
                    !string.IsNullOrWhiteSpace(candidate.Guard) ||
                    !TargetMatches(candidate, target))
                    continue;
                endSyncIndex = i;
                break;
            }
            if (endSyncIndex < 0)
                return false;

            int labelIndex = endSyncIndex + 1;
            if (labelIndex >= nodes.Count ||
                nodes[labelIndex] is not SassLabel label ||
                !LabelName(label.Name).Equals(target, StringComparison.Ordinal))
                return false;

            int guardedExitIndex = -1;
            for (int i = labelIndex + 1; i < nodes.Count; i++)
            {
                if (nodes[i] is not SassInstruction candidate)
                    return false;
                if (candidate.Mnemonic.StartsWith("NOP", StringComparison.OrdinalIgnoreCase))
                    continue;
                if ((candidate.Mnemonic.StartsWith("EXIT", StringComparison.OrdinalIgnoreCase) ||
                     candidate.Mnemonic.StartsWith("RET", StringComparison.OrdinalIgnoreCase)) &&
                    string.Equals(candidate.Guard, guardedSync.Guard,
                        StringComparison.OrdinalIgnoreCase))
                {
                    guardedExitIndex = i;
                    break;
                }
                return false;
            }
            if (guardedExitIndex < 0)
                return false;

            int finalExitIndex = -1;
            for (int i = guardedExitIndex + 1; i < nodes.Count; i++)
            {
                if (nodes[i] is SassLabel)
                    return false;
                if (nodes[i] is not SassInstruction candidate)
                    return false;
                if (candidate.Mnemonic.StartsWith("NOP", StringComparison.OrdinalIgnoreCase))
                    continue;
                if (candidate.Mnemonic.StartsWith("EXIT", StringComparison.OrdinalIgnoreCase) ||
                    candidate.Mnemonic.StartsWith("RET", StringComparison.OrdinalIgnoreCase))
                {
                    if (!string.IsNullOrWhiteSpace(candidate.Guard))
                        return false;
                    finalExitIndex = i;
                    break;
                }
                if (IsControlFlow(candidate) || IsBarrier(candidate) ||
                    !IsSimplePredicable(candidate))
                    return false;
            }
            if (finalExitIndex < 0)
                return false;

            for (int i = 0; i < nodes.Count; i++)
            {
                if (i >= index && i <= finalExitIndex)
                    continue;
                if (nodes[i] is SassInstruction outside &&
                    IsControlFlow(outside) &&
                    TargetMatches(outside, target))
                    return false;
            }

            for (int i = index + 1; i < guardedSyncIndex; i++)
            {
                if (nodes[i] is SassLabel)
                    continue;
                if (nodes[i] is not SassInstruction prelude ||
                    IsControlFlow(prelude) || IsBarrier(prelude) ||
                    !IsSimplePredicable(prelude))
                    return false;
            }

            var bodyInstructions = new List<SassInstruction>();
            for (int i = guardedSyncIndex + 1; i < endSyncIndex; i++)
            {
                if (nodes[i] is SassLabel)
                    continue;
                if (nodes[i] is not SassInstruction body ||
                    !string.IsNullOrWhiteSpace(body.Guard) ||
                    IsControlFlow(body) || IsBarrier(body) ||
                    !IsSimplePredicable(body))
                    return false;
                bodyInstructions.Add(body);
            }
            if (bodyInstructions.Count == 0)
                return false;

            var tailInstructions = new List<SassInstruction>();
            for (int i = guardedExitIndex + 1; i < finalExitIndex; i++)
            {
                if (nodes[i] is not SassInstruction tail)
                    return false;
                if (tail.Mnemonic.StartsWith("NOP", StringComparison.OrdinalIgnoreCase))
                    continue;
                if (!string.IsNullOrWhiteSpace(tail.Guard) ||
                    IsControlFlow(tail) || IsBarrier(tail) ||
                    !IsSimplePredicable(tail))
                    return false;
                tailInstructions.Add(tail);
            }

            var bodyWrites = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (SassInstruction body in bodyInstructions)
                foreach (string reg in WrittenRegisters(body))
                    bodyWrites.Add(reg);
            var tailReads = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (SassInstruction tail in tailInstructions)
                foreach (string reg in ReadRegisters(tail))
                    tailReads.Add(reg);

            var initialize = new List<string>();
            foreach (string reg in bodyWrites.Intersect(tailReads))
            {
                bool firstAccessIsWrite = true;
                foreach (SassInstruction body in bodyInstructions)
                {
                    if (ReadRegisters(body).Contains(
                            reg, StringComparer.OrdinalIgnoreCase))
                    {
                        firstAccessIsWrite = false;
                        break;
                    }
                    if (WrittenRegisters(body).Contains(
                            reg, StringComparer.OrdinalIgnoreCase))
                        break;
                }
                if (!firstAccessIsWrite)
                    return false;
                initialize.Add(reg);
            }

            for (int i = index + 1; i < guardedSyncIndex; i++)
                EmitNode(nodes[i]);
            foreach (string reg in initialize)
                _lines.Add($"mov.u32 {reg}, 0;");

            string skipGuard = Guard(guardedSync.Guard);
            _lines.Add($"{skipGuard} bra {target};");
            for (int i = guardedSyncIndex + 1; i < endSyncIndex; i++)
                EmitNode(nodes[i]);
            for (int i = guardedExitIndex + 1; i < finalExitIndex; i++)
                EmitNode(nodes[i]);
            EmitReturn("");
            _lines.Add($"{target}:");
            EmitReturn("");
            index = IsTrailingPadding(nodes, finalExitIndex + 1)
                ? nodes.Count
                : finalExitIndex;
            return true;
        }

        private bool IsTrailingPadding(List<SassNode> nodes, int start)
        {
            var labels = new HashSet<string>(StringComparer.Ordinal);
            for (int i = start; i < nodes.Count; i++)
            {
                if (nodes[i] is SassLabel label)
                    labels.Add(LabelName(label.Name));
            }
            if (labels.Count == 0 && start >= nodes.Count)
                return true;

            // A label in the trailing region that is targeted from reachable
            // code is not padding.
            for (int i = 0; i < start; i++)
            {
                if (nodes[i] is not SassInstruction instruction ||
                    !HasBranchTarget(instruction))
                    continue;
                try
                {
                    if (labels.Contains(BranchLabel(instruction)))
                        return false;
                }
                catch (NotSupportedException)
                {
                    return false;
                }
            }

            for (int i = start; i < nodes.Count; i++)
            {
                if (nodes[i] is SassLabel)
                    continue;
                if (nodes[i] is not SassInstruction instruction)
                    return false;
                if (instruction.Mnemonic.StartsWith(
                        "NOP", StringComparison.OrdinalIgnoreCase))
                    continue;
                if (instruction.Mnemonic.StartsWith(
                        "EXIT", StringComparison.OrdinalIgnoreCase) ||
                    instruction.Mnemonic.StartsWith(
                        "RET", StringComparison.OrdinalIgnoreCase))
                    continue;
                if (instruction.Mnemonic.StartsWith(
                        "BRA", StringComparison.OrdinalIgnoreCase))
                {
                    try
                    {
                        if (labels.Contains(BranchLabel(instruction)))
                            continue;
                    }
                    catch (NotSupportedException)
                    {
                        return false;
                    }
                }
                return false;
            }
            return true;
        }

        private static bool HasBranchTarget(SassInstruction instruction)
        {
            string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
            return op is "BRA" or "SSY" or "SYNC" or "PBK" or "PCNT"
                or "BSSY" or "BSYNC" or "BRX" ||
                (op == "NOP" &&
                 instruction.Mnemonic.Contains(".S", StringComparison.OrdinalIgnoreCase) &&
                 instruction.BranchTargets.Count > 0);
        }

        private void EmitNode(SassNode node)
        {
            if (node is SassLabel label)
                _lines.Add($"{LabelName(label.Name)}:");
            else if (node is SassInstruction instruction)
                EmitInstruction(instruction);
        }

        private bool TargetMatches(SassInstruction instruction, string target)
        {
            try
            {
                return BranchLabel(instruction).Equals(
                    target, StringComparison.Ordinal);
            }
            catch (NotSupportedException)
            {
                return false;
            }
        }

        private static string NegateGuard(string guard)
        {
            string trimmed = guard.Trim();
            if (trimmed.StartsWith("@!", StringComparison.Ordinal))
                return "@" + trimmed[2..];
            if (trimmed.StartsWith("@", StringComparison.Ordinal))
                return "@!" + trimmed[1..];
            return guard;
        }

        private static bool IsControlFlow(SassInstruction instruction)
        {
            string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
            return op is "BRA" or "EXIT" or "RET" or "SSY" or "SYNC" or "PBK"
                or "PCNT" or "BSSY" or "BSYNC" or "WARPSYNC" or "BRX"
                or "CAL" or "JCAL" or "CALL"
                or "BRK" or "CONT" or "LONGJMP" or "KIL";
        }

        private static bool IsBarrier(SassInstruction instruction)
        {
            string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
            return op is "BAR" or "MEMBAR" or "DEPBAR" or "CCTL" or "TEXDEPBAR";
        }

        private static bool IsSimplePredicable(SassInstruction instruction)
        {
            string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
            if (op is "XMAD" or "LEA" ||
                (op is "IADD" or "IADD32I" &&
                 (instruction.Mnemonic.Contains(".X", StringComparison.OrdinalIgnoreCase) ||
                  instruction.Mnemonic.Contains(".CC", StringComparison.OrdinalIgnoreCase))))
                return false;
            return op is "MOV" or "MOV32I" or "LDG" or "LD" or "LDS" or "STG"
                or "ST" or "STS" or "ATOM" or "ATOMS" or "IADD" or "IADD32I"
                or "IMAD" or "ISETP" or "S2R" or "SHFL" or "VOTE" or "POPC"
                or "FLO" or "LOP" or "LOP32I" or "SHL" or "SHR" or "BFE"
                or "BFI" or "PRMT" or "SEL" or "PSETP" or "FSET" or "FSETP"
                or "FMUL" or "FADD" or "FFMA" or "MUFU" or "RRO" or "I2F"
                or "F2I" or "ISUB" or "FADD32I" or "FMUL32I";
        }

        private IEnumerable<string> WrittenRegisters(SassInstruction instruction)
        {
            int destination = DestinationIndex(instruction);
            if (destination < 0 || destination >= instruction.Operands.Count)
                yield break;
            Match match = Regex.Match(instruction.Operands[destination],
                @"\bR(\d+)\b", RegexOptions.IgnoreCase);
            if (!match.Success)
                yield break;
            int first = int.Parse(match.Groups[1].Value,
                CultureInfo.InvariantCulture);
            int count = instruction.Mnemonic.Contains(
                    ".128", StringComparison.OrdinalIgnoreCase) ? 4
                : instruction.Mnemonic.Contains(
                    ".64", StringComparison.OrdinalIgnoreCase) ? 2
                : 1;
            for (int i = 0; i < count; i++)
                yield return "%r" + (first + i);
        }

        private IEnumerable<string> ReadRegisters(SassInstruction instruction)
        {
            int destination = DestinationIndex(instruction);
            for (int i = 0; i < instruction.Operands.Count; i++)
            {
                if (i == destination)
                    continue;
                string operand = instruction.Operands[i];
                bool bareRegister = !operand.Contains('[', StringComparison.Ordinal) &&
                    !operand.Contains('+', StringComparison.Ordinal);
                int count = bareRegister && instruction.Mnemonic.Contains(
                        ".128", StringComparison.OrdinalIgnoreCase) ? 4
                    : bareRegister && instruction.Mnemonic.Contains(
                        ".64", StringComparison.OrdinalIgnoreCase) ? 2
                    : 1;
                foreach (Match match in Regex.Matches(operand,
                    @"\bR(\d+)\b", RegexOptions.IgnoreCase))
                {
                    if (match.Value.Equals("RZ", StringComparison.OrdinalIgnoreCase))
                        continue;
                    int first = int.Parse(match.Groups[1].Value,
                        CultureInfo.InvariantCulture);
                    for (int j = 0; j < count; j++)
                        yield return "%r" + (first + j);
                }
            }
        }

        // NVCC device-runtime helpers are lowered at their call sites
        // (EmitExternalHelperCall); their SASS bodies must not be emitted
        // as part of the entry kernel.
        private void SkipDeviceRuntimeHelpers(List<SassNode> nodes)
        {
            for (int i = 0; i < nodes.Count; i++)
            {
                if (nodes[i] is not SassLabel label ||
                    (!label.Name.Contains(
                         "__cuda_sm20_", StringComparison.Ordinal) &&
                     !label.Name.Contains(
                         "__internal_trig_reduction_slowpathd",
                         StringComparison.Ordinal)))
                    continue;
                int end = nodes.Count;
                for (int j = i + 1; j < nodes.Count; j++)
                {
                    if (nodes[j] is SassLabel boundary &&
                        !boundary.Name.StartsWith(
                            ".L_", StringComparison.Ordinal))
                    {
                        end = j;
                        break;
                    }
                }
                if (label.Name.Contains(
                        "__cuda_sm20_div_rn_f64_full", StringComparison.Ordinal))
                {
                    // Keep this body: calls branch to it and its RET
                    // dispatches on the constant return address.
                    _dispatchHelperLabel = LabelName(label.Name);
                    for (int j = i + 1; j < end; j++)
                        if (nodes[j] is SassInstruction ret &&
                            ret.Mnemonic.StartsWith(
                                "RET", StringComparison.OrdinalIgnoreCase))
                            _dispatchReturnInstructions.Add(ret);
                    continue;
                }
                for (int j = i; j < end; j++)
                    _skipNodes.Add(j);
            }
        }

        private void CollectFunctions()
        {
            List<SassNode> nodes = _section.Nodes.ToList();
            SkipDeviceRuntimeHelpers(nodes);
            var callCounts = new Dictionary<string, int>(
                StringComparer.Ordinal);
            for (int i = 0; i < nodes.Count; i++)
            {
                if (nodes[i] is not SassInstruction call ||
                    !IsCallMnemonic(call.Mnemonic) ||
                    !string.IsNullOrWhiteSpace(call.Guard))
                    continue;
                string target;
                try
                {
                    target = CallTarget(nodes, i, call);
                }
                catch (NotSupportedException)
                {
                    continue;
                }
                callCounts[target] = callCounts.GetValueOrDefault(target) + 1;
            }
            var claimed = new List<(int Start, int End)>();
            for (int i = 0; i < nodes.Count; i++)
            {
                if (claimed.Any(range => i >= range.Start && i < range.End))
                    continue;
                if (nodes[i] is not SassInstruction call ||
                    !IsCallMnemonic(call.Mnemonic) ||
                    !string.IsNullOrWhiteSpace(call.Guard))
                    continue;
                string target;
                try
                {
                    target = CallTarget(nodes, i, call);
                }
                catch (NotSupportedException)
                {
                    continue;
                }
                if (callCounts.GetValueOrDefault(target) != 1 ||
                    _functionsByTarget.ContainsKey(target) ||
                    target.Contains("__cuda_sm20_", StringComparison.Ordinal))
                    continue;
                if (!TryComputeCallLiveOut(
                        nodes, i, out HashSet<string>? liveOut))
                    continue;
                if (!TryAnalyzeFunction(
                        nodes, i, target, liveOut, out PtxFunctionAbi abi))
                    continue;
                _functions.Add(abi);
                _functionsByTarget[target] = abi;
                claimed.Add((abi.LabelIndex, abi.BodyEnd));
            }
            foreach (PtxFunctionAbi function in _functions.ToList())
            {
                string text;
                try
                {
                    text = BuildFunction(nodes, function);
                }
                catch (NotSupportedException ex)
                {
                    Console.Error.WriteLine($"DBG build {function.Name}: {ex.Message}");
                    _functions.Remove(function);
                    _functionsByTarget.Remove(function.Target);
                    continue;
                }
                _functionTexts.AddRange(text.Split('\n'));
                if (_functionTexts.Count > 0 &&
                    _functionTexts[^1].Length != 0)
                    _functionTexts.Add("");
            }
        }

        private bool TryComputeCallLiveOut(
            List<SassNode> nodes, int callIndex, out HashSet<string> liveOut)
        {
            liveOut = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var written = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            for (int i = callIndex + 1; i < nodes.Count; i++)
            {
                if (nodes[i] is SassLabel)
                    return false;
                if (nodes[i] is not SassInstruction instruction)
                    return false;
                string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
                if (op is "EXIT" or "RET")
                    return true;
                if (op is "BRA" or "SSY" or "SYNC" or "PBK" or "PCNT"
                    or "BSSY" or "BSYNC" or "BRX" or "CAL" or "JCAL" or "CALL")
                    return false;
                if (IsSelfMove(instruction))
                    continue;
                foreach (string read in ReadRegisters(instruction))
                {
                    string reg = SassRegister(read);
                    if (reg != "RZ" && !written.Contains(reg))
                        liveOut.Add(reg);
                }
                foreach (string write in WrittenRegisters(instruction))
                {
                    string reg = SassRegister(write);
                    if (reg != "RZ")
                        written.Add(reg);
                }
            }
            return false;
        }

        private bool TryAnalyzeFunction(
            List<SassNode> nodes, int callIndex, string target,
            HashSet<string> liveOut, out PtxFunctionAbi abi)
        {
            abi = null!;
            int labelIndex = -1;
            for (int i = callIndex + 1; i < nodes.Count; i++)
            {
                if (nodes[i] is SassLabel label &&
                    LabelName(label.Name).Equals(
                        target, StringComparison.Ordinal))
                {
                    labelIndex = i;
                    break;
                }
            }
            if (labelIndex < 0)
                return false;
            int bodyEnd = nodes.Count;
            for (int i = labelIndex + 1; i < nodes.Count; i++)
            {
                if (nodes[i] is SassLabel boundary &&
                    !boundary.Name.StartsWith(
                        ".L_", StringComparison.Ordinal))
                {
                    bodyEnd = i;
                    break;
                }
            }
            int returnIndex = -1;
            for (int i = labelIndex + 1; i < bodyEnd; i++)
            {
                if (nodes[i] is SassLabel)
                {
                    if (returnIndex < 0)
                        return false;
                    continue;
                }
                if (nodes[i] is not SassInstruction instruction)
                    return false;
                string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
                if (op == "RET")
                {
                    if (!string.IsNullOrWhiteSpace(instruction.Guard))
                        return false;
                    returnIndex = i;
                    break;
                }
                if (!IsFunctionBodyInstruction(instruction))
                    return false;
            }
            if (returnIndex < 0)
                return false;

            var pointerPairs = new Dictionary<string, string>(
                StringComparer.OrdinalIgnoreCase);
            var pointerRegisters = new HashSet<string>(
                StringComparer.OrdinalIgnoreCase);
            var lastPointerUse = new Dictionary<string, int>(
                StringComparer.OrdinalIgnoreCase);
            for (int i = labelIndex + 1; i < returnIndex; i++)
            {
                if (nodes[i] is not SassInstruction instruction)
                    continue;
                foreach (string operand in instruction.Operands)
                {
                    string op =
                        instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
                    if (op is "LDS" or "STS")
                        continue;
                    if (!TryGetAddressPair(operand, out string? baseReg,
                            out string? highReg))
                        continue;
                    if (baseReg == null || highReg == null ||
                        baseReg.Equals("R1", StringComparison.OrdinalIgnoreCase) ||
                        highReg.Equals("R1", StringComparison.OrdinalIgnoreCase))
                        return false;
                    if (pointerRegisters.Contains(baseReg) ||
                        pointerRegisters.Contains(highReg))
                    {
                        if (!pointerPairs.TryGetValue(
                                baseReg, out string? existingHigh) ||
                            !existingHigh.Equals(
                                highReg, StringComparison.OrdinalIgnoreCase))
                            return false;
                    }
                    else
                    {
                        pointerPairs[baseReg] = highReg;
                        pointerRegisters.Add(baseReg);
                        pointerRegisters.Add(highReg);
                    }
                    lastPointerUse[baseReg] = i;
                    lastPointerUse[highReg] = i;
                }
            }
            var arguments = new List<PtxFunctionArgument>();
            var argumentSet = new HashSet<string>(
                StringComparer.OrdinalIgnoreCase);
            var returns = new List<string>();
            var returnSet = new HashSet<string>(
                StringComparer.OrdinalIgnoreCase);
            var writtenScalar = new HashSet<string>(
                StringComparer.OrdinalIgnoreCase);
            var pointerScratch = new HashSet<string>(
                StringComparer.OrdinalIgnoreCase);
            var writtenPredicates = new HashSet<string>(
                StringComparer.OrdinalIgnoreCase);
            bool carryReady = false;
            for (int i = labelIndex + 1; i < returnIndex; i++)
            {
                if (nodes[i] is not SassInstruction instruction)
                    continue;
                bool selfMove = IsSelfMove(instruction);
                var predicateDestinations =
                    PredicateDestinationIndexes(instruction);
                foreach (string predicate in ReadPredicateNames(
                             instruction, predicateDestinations))
                {
                    if (!writtenPredicates.Contains(predicate))
                        return false;
                }
                foreach (int destination in predicateDestinations)
                {
                    if (destination < instruction.Operands.Count)
                    {
                        string predicate =
                            BaseReg(instruction.Operands[destination]);
                        if (Regex.IsMatch(
                                predicate, @"^P\d+$",
                                RegexOptions.IgnoreCase))
                            writtenPredicates.Add(predicate);
                    }
                }
                bool readsCarry =
                    instruction.Mnemonic.Contains(
                        ".X", StringComparison.OrdinalIgnoreCase);
                bool writesCarry =
                    instruction.Mnemonic.Contains(
                        ".CC", StringComparison.OrdinalIgnoreCase) ||
                    instruction.Operands.Any(operand =>
                        operand.Contains(
                            ".CC", StringComparison.OrdinalIgnoreCase));
                if (readsCarry && !carryReady)
                    return false;
                if (writesCarry)
                    carryReady = true;
                int destinationIndex = DestinationIndex(instruction);
                for (int operandIndex = 0;
                     operandIndex < instruction.Operands.Count; operandIndex++)
                {
                    if (operandIndex == destinationIndex)
                        continue;
                    string operand = instruction.Operands[operandIndex];
                    string operandOp =
                        instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
                    if (operandOp is not ("LDS" or "STS") &&
                        TryGetAddressPair(operand, out _, out _))
                        continue;
                    foreach (Match match in Regex.Matches(
                        operand, @"\bR(\d+)\b", RegexOptions.IgnoreCase))
                    {
                        string reg = "R" + match.Groups[1].Value;
                        if (!pointerRegisters.Contains(reg))
                            continue;
                        if (selfMove &&
                            instruction.Operands.Count >= 2 &&
                            BaseReg(instruction.Operands[0]).Equals(
                                BaseReg(instruction.Operands[1]),
                                StringComparison.OrdinalIgnoreCase))
                            continue;
                        if (i <= lastPointerUse[reg])
                            return false;
                        if (!writtenScalar.Contains(reg) &&
                            !pointerScratch.Contains(reg))
                            return false;
                    }
                }
                foreach (string read in ReadRegisters(instruction))
                {
                    string reg = SassRegister(read);
                    if (reg == "RZ" ||
                        reg.Equals("R1", StringComparison.OrdinalIgnoreCase) ||
                        pointerRegisters.Contains(reg) ||
                        writtenScalar.Contains(reg) ||
                        argumentSet.Contains(reg))
                        continue;
                    argumentSet.Add(reg);
                    arguments.Add(new PtxFunctionArgument(
                        reg, null, false));
                }
                foreach (string write in WrittenRegisters(instruction))
                {
                    string reg = SassRegister(write);
                    if (pointerRegisters.Contains(reg))
                    {
                        if (i < lastPointerUse[reg] && !selfMove)
                            return false;
                        if (i >= lastPointerUse[reg])
                            pointerScratch.Add(reg);
                        continue;
                    }
                    writtenScalar.Add(reg);
                    if (liveOut.Contains(reg) && returnSet.Add(reg))
                        returns.Add(reg);
                }
            }
            if (returns.Any(pointerRegisters.Contains))
                return false;
            foreach (string reg in pointerScratch)
            {
                if (liveOut.Contains(reg))
                    return false;
            }
            foreach (var pair in pointerPairs)
            {
                if (argumentSet.Contains(pair.Key) ||
                    argumentSet.Contains(pair.Value))
                    return false;
                argumentSet.Add(pair.Key);
                arguments.Add(new PtxFunctionArgument(
                    pair.Key, pair.Value, true));
            }
            if (arguments.Count == 0 && returns.Count == 0)
                return false;
            if (arguments.Count > 32 || returns.Count > 4)
                return false;
            string name = "fn_" + Sanitize(target);
            abi = new PtxFunctionAbi(
                name, target, labelIndex, labelIndex + 1, bodyEnd, returnIndex,
                arguments, returns);
            return true;
        }

        private bool IsSelfMove(SassInstruction instruction)
        {
            return instruction.Mnemonic.Equals(
                    "MOV", StringComparison.OrdinalIgnoreCase) &&
                instruction.Operands.Count >= 2 &&
                BaseReg(instruction.Operands[0]).Equals(
                    BaseReg(instruction.Operands[1]),
                    StringComparison.OrdinalIgnoreCase);
        }

        private bool TryGetAddressPair(
            string operand, out string? baseReg, out string? highReg)
        {
            baseReg = null;
            highReg = null;
            Match match = Regex.Match(operand,
                @"^\[(R\d+)(?:\+(0x[0-9A-Fa-f]+|\d+))?\]$",
                RegexOptions.IgnoreCase);
            if (!match.Success)
                return false;
            baseReg = "R" + match.Groups[1].Value[1..];
            if (baseReg.Equals("R1", StringComparison.OrdinalIgnoreCase))
                return false;
            highReg = NextRegister(baseReg);
            return true;
        }

        private bool IsFunctionBodyInstruction(SassInstruction instruction)
        {
            string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
            if (op is "CAL" or "JCAL" or "CALL" or "EXIT" or "RET" or "BRA" or "SSY"
                or "SYNC"
                or "PBK" or "PCNT" or "BSSY" or "BSYNC" or "BRX" or "BPT"
                or "DEPBAR" or "CCTL" or "TEXDEPBAR" or "WARPSYNC"
                or "KIL" or "LONGJMP" or "BRK" or "CONT"
                or "LDSLK" or "STSCUL" or "ATOM" or "ATOMS" or "ATOMG")
                return false;
            foreach (string operand in instruction.Operands)
            {
                if (operand.Contains("c[0x", StringComparison.OrdinalIgnoreCase))
                    return false;
                if (!Regex.IsMatch(
                        operand, @"\bR1\b", RegexOptions.IgnoreCase))
                    continue;
                bool localSlot = op is "LDL" or "STL" &&
                    Regex.IsMatch(operand,
                        @"^\[R1(?:\+(0x[0-9A-Fa-f]+|\d+))?\]$",
                        RegexOptions.IgnoreCase);
                if (!localSlot)
                    return false;
            }
            return true;
        }

        private HashSet<int> PredicateDestinationIndexes(
            SassInstruction instruction)
        {
            var indexes = new HashSet<int>();
            string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
            if (op is "ISETP" or "FSETP" or "DSETP" or "VOTE" or "ICMP")
                indexes.Add(0);
            if (op == "PSETP")
            {
                indexes.Add(0);
                indexes.Add(1);
            }
            indexes.RemoveWhere(index =>
                index >= instruction.Operands.Count ||
                !Regex.IsMatch(
                    BaseReg(instruction.Operands[index]), @"^P\d+$",
                    RegexOptions.IgnoreCase));
            return indexes;
        }

        private IEnumerable<string> ReadPredicateNames(
            SassInstruction instruction, HashSet<int> destinations)
        {
            for (int i = 0; i < instruction.Operands.Count; i++)
            {
                if (destinations.Contains(i))
                    continue;
                foreach (Match match in Regex.Matches(
                    instruction.Operands[i], @"\bP(\d+)\b",
                    RegexOptions.IgnoreCase))
                    yield return "P" + match.Groups[1].Value;
            }
            if (!string.IsNullOrWhiteSpace(instruction.Guard))
            {
                foreach (Match match in Regex.Matches(
                    instruction.Guard, @"\bP(\d+)\b",
                    RegexOptions.IgnoreCase))
                    yield return "P" + match.Groups[1].Value;
            }
        }

        private static string SassRegister(string ptxRegister)
        {
            return ptxRegister.StartsWith(
                    "%r", StringComparison.OrdinalIgnoreCase)
                ? "R" + ptxRegister[2..]
                : ptxRegister;
        }

        private string BuildFunction(
            List<SassNode> nodes, PtxFunctionAbi function)
        {
            var section = new SassSection
            {
                Name = ".text." + function.Name,
            };
            for (int i = function.BodyStart; i <= function.ReturnIndex; i++)
                section.Nodes.Add(nodes[i]);
            var emitter = new Emitter(
                section, function.Name, Array.Empty<PtxParameter>(), _target,
                _constantResolver, _sharedBytes, _constant2Resolver,
                _constant2Bytes);
            emitter._functionName = function.Name;
            emitter._functionArguments = function.Arguments;
            emitter._functionReturns = function.Returns;
            return emitter.RunFunction();
        }

        private string RunFunction()
        {
            CollectLabelsAndRegisters();
            if (_usesLocalFrame)
                _pointerRegs["R1"] = "__callee_frame";
            EmitFunctionHeader();
            int declarationIndex = _lines.Count;
            int pointerIndex = 0;
            for (int i = 0; i < _functionArguments.Count; i++)
            {
                PtxFunctionArgument argument = _functionArguments[i];
                if (argument.IsPointer)
                {
                    string pointer = $"%rdfn{pointerIndex++}";
                    _lines.Add(
                        $"    ld.param.u64 {pointer}, [__a{i}];");
                    _pointerRegs[argument.Register] = pointer;
                }
                else
                {
                    _lines.Add(
                        $"    ld.param.u32 {Reg(argument.Register)}, " +
                        $"[__a{i}];");
                }
            }
            List<SassNode> body = _section.Nodes.ToList();
            for (int bodyIndex = 0; bodyIndex < body.Count; bodyIndex++)
            {
                if (TryEmitIaddCarryPair(body, ref bodyIndex) ||
                    TryEmitLea64(body, ref bodyIndex) ||
                    TryEmitLeaHigh(body, ref bodyIndex) ||
                    TryEmitLeaPointer(body, ref bodyIndex) ||
                    TryEmitGenericAddPair(body, ref bodyIndex) ||
                    TryEmitShift64(body, ref bodyIndex, allowGap: false) ||
                    TryEmitIsubPair(body, ref bodyIndex, allowGap: false) ||
                    TryEmitIadd3Pair(body, ref bodyIndex, allowGap: false) ||
                    TryEmitUadd3Pair(body, ref bodyIndex) ||
                    TryEmitLeaCarryPair(body, ref bodyIndex, allowGap: false) ||
                    TryEmitAddressAdd(body, ref bodyIndex))
                    continue;
                SassNode node = body[bodyIndex];
                if (node is not SassInstruction instruction)
                    throw new NotSupportedException(
                        "labels are not supported inside a lifted function");
                string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
                if (op == "RET")
                {
                    if (_functionReturns.Count > 4)
                        throw new NotSupportedException(
                            "more than four function return registers are not supported");
                    if (_functionReturns.Count == 1)
                    {
                        _lines.Add(
                            $"    st.param.u32 [__ret], " +
                            $"{Reg(_functionReturns[0])};");
                    }
                    else if (_functionReturns.Count == 2)
                    {
                        string packed = $"%rdt{_nextTempDouble++}";
                        _lines.Add(
                            $"    mov.b64 {packed}, " +
                            $"{{{Reg(_functionReturns[0])}, " +
                            $"{Reg(_functionReturns[1])}}};");
                        _lines.Add(
                            $"    st.param.u64 [__ret], {packed};");
                    }
                    // Three/four returns use the PTX .reg-form signature;
                    // the named return registers are already SSA values.
                    _lines.Add("    ret;");
                    _lines.Add("}");
                    _lines.InsertRange(
                        declarationIndex, CompactRegisterDeclarations());
                    return ResolveRegisterDeclarations(
                string.Join('\n', _lines) + "\n");
                }
                if (op is "CAL" or "JCAL" or "CALL" or "EXIT" or "BRA" or "SSY"
                    or "SYNC" or "PBK" or "PCNT" or "BSSY" or "BSYNC" or "BRX")
                    throw new NotSupportedException(
                        $"control flow '{instruction.Mnemonic}' is not " +
                        "supported inside a lifted function");
                if (IsSelfMove(instruction))
                    continue;
                bool constantProducer =
                    TrySetSharedConstant(instruction);
                EmitInstruction(instruction);
                if (!constantProducer)
                    RemoveSharedConstantDestination(instruction);
            }
            throw new NotSupportedException(
                "lifted function has no RET");
        }

        private List<string> CompactRegisterDeclarations()
        {
            var declarations = new List<string>();
            if (_usesLocalFrame)
            {
                int frameBytes = _calleeFrameBytes > 0
                    ? (_calleeFrameBytes + 15) & ~15 : 256;
                declarations.Add(
                    $"    .local .align 16 .b8 __callee_frame[{frameBytes}];");
            }
            if (_usesDynamicLocalFrame)
                declarations.Add(
                    "    .local .align 16 .b8 __callee_dynamic_frame[65536];");
            int maxRegister = 0;
            foreach (string register in _usedRegs)
            {
                if (register.StartsWith(
                        "R", StringComparison.OrdinalIgnoreCase) &&
                    int.TryParse(register[1..], out int index))
                    maxRegister = Math.Max(maxRegister, index + 1);
            }
            foreach (string register in _functionArguments.Select(
                         argument => argument.Register).Concat(
                         _functionReturns))
            {
                if (register.StartsWith(
                        "R", StringComparison.OrdinalIgnoreCase) &&
                    int.TryParse(register[1..], out int index))
                    maxRegister = Math.Max(maxRegister, index + 1);
            }
            if (maxRegister > 0)
                declarations.Add($"    .reg .b32 %r<{maxRegister}>;");
            int maxPredicate = 0;
            foreach (string predicate in _usedPreds)
            {
                if (predicate.StartsWith(
                        "P", StringComparison.OrdinalIgnoreCase) &&
                    int.TryParse(predicate[1..], out int index))
                    maxPredicate = Math.Max(maxPredicate, index + 1);
            }
            if (maxPredicate > 0)
                declarations.Add($"    .reg .pred %p<{maxPredicate}>;");
            declarations.Add($"    .reg .pred %pt<{Math.Max(_nextTempPred, 1)}>;");
            declarations.Add("    .reg .pred %up<64>;");
            declarations.Add($"    .reg .pred %ps<{Math.Max(_nextSnapPred, 1)}>;");
            declarations.Add($"    .reg .f32 %ft<{Math.Max(_nextTempFloat, 1)}>;");
            declarations.Add($"    .reg .f64 %dt<{Math.Max(_nextTempDouble, 1)}>;");
            declarations.Add($"    .reg .b32 %rt<{Math.Max(_nextTemp, 1)}>;");
            declarations.Add(
                $"    .reg .u32 %ru<{Math.Max(_nextTempU32, 1)}>;");
            declarations.Add(
                $"    .reg .b64 %rdt<{Math.Max(_nextTempDouble, 1)}>;");
            int pointerCount = _functionArguments.Count(
                argument => argument.IsPointer);
            if (pointerCount > 0)
                declarations.Add(
                    $"    .reg .b64 %rdfn<{pointerCount}>;");
            return declarations;
        }

        private bool TrySetSharedConstant(SassInstruction instruction)
        {
            string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
            if (op is not ("MOV" or "MOV32I" or "IADD" or "IADD32I"))
                return false;
            if (instruction.Operands.Count < 2)
                return true;
            string destination = BaseReg(instruction.Operands[0]);
            if (destination.Equals("RZ", StringComparison.OrdinalIgnoreCase))
                return true;
            int value;
            if (op is "MOV" or "MOV32I")
            {
                string source = instruction.Operands[1];
                if (IsRz(source))
                    value = 0;
                else if (_sharedAddressConstants.TryGetValue(
                             BaseReg(source), out int copied))
                    value = copied;
                else if (op == "MOV32I" &&
                         TrySwitchImmediate(source, out int immediate))
                    value = immediate;
                else
                {
                    _sharedAddressConstants.Remove(destination);
                    return true;
                }
            }
            else if (instruction.Operands.Count < 3)
            {
                _sharedAddressConstants.Remove(destination);
                return true;
            }
            else
            {
                string left = BaseReg(instruction.Operands[1]);
                string right = BaseReg(instruction.Operands[2]);
                if (_sharedAddressConstants.TryGetValue(left, out int leftValue) &&
                    TrySwitchImmediate(right, out int addend))
                    value = leftValue + addend;
                else if (_sharedAddressConstants.TryGetValue(right, out int rightValue) &&
                         TrySwitchImmediate(left, out addend))
                    value = rightValue + addend;
                else if (_sharedAddressConstants.TryGetValue(left, out leftValue) &&
                         IsRz(right))
                    value = leftValue;
                else if (_sharedAddressConstants.TryGetValue(right, out rightValue) &&
                         IsRz(left))
                    value = rightValue;
                else
                {
                    _sharedAddressConstants.Remove(destination);
                    return true;
                }
            }
            _sharedAddressConstants[destination] = value;
            return true;
        }

        private void RemoveSharedConstantDestination(
            SassInstruction instruction)
        {
            int destination = DestinationIndex(instruction);
            if (destination < 0 || destination >= instruction.Operands.Count)
                return;
            string register = BaseReg(instruction.Operands[destination]);
            _sharedAddressConstants.Remove(register);
        }

        private void EmitFunctionHeader()
        {
            string returns = _functionReturns.Count switch
            {
                1 => "(.param .u32 __ret) ",
                2 => "(.param .u64 __ret) ",
                3 or 4 => "(.reg .u32 " +
                     string.Join(", ", _functionReturns.Select(
                         reg => Reg(reg))) + ") ",
                _ => "",
            };
            string arguments = string.Join(", ",
                _functionArguments.Select(
                    (argument, index) =>
                        $".param .{(argument.IsPointer ? "u64" : "u32")} " +
                        $"__a{index}"));
            _lines.Add($".func {returns}{_functionName}({arguments})");
            _lines.Add("{");
        }

        private string ScalarCallArgument(string register)
        {
            if (_pointerRegs.TryGetValue(register, out string? pointer))
            {
                string low = $"%ru{_nextTempU32++}";
                string high = $"%ru{_nextTempU32++}";
                _lines.Add(
                    $"    mov.b64 {{{low}, {high}}}, {pointer};");
                return low;
            }
            foreach (var pair in _pointerRegs)
            {
                if (NextRegister(pair.Key)?.Equals(
                        register, StringComparison.OrdinalIgnoreCase) != true)
                    continue;
                string low = $"%ru{_nextTempU32++}";
                string high = $"%ru{_nextTempU32++}";
                _lines.Add(
                    $"    mov.b64 {{{low}, {high}}}, {pair.Value};");
                return high;
            }
            return Reg(register);
        }

        // A guarded call to a local EXIT block is the compiler's
        // predicated early-exit idiom.  NVCC also uses the same form for
        // reconvergence: the target is a BSYNC/BSSY label, so lower it as
        // a predicated branch instead of a subroutine call.
        // Device-runtime helpers are shared SASS functions whose register
        // ABI is chosen per module.  Derive the input/output registers
        // from the helper body instead of assuming one module's layout.
        // The sm_20 64-bit signed-division helper is linked with a
        // per-module register ABI.  In the helper the first abs/negate
        // pair is the divisor (it feeds MUFU.RCP) and the second is the
        // dividend; the quotient comes back in the caller-visible
        // register.  Derive all three from the helper body and the
        // caller's post-call reads instead of assuming one layout.
        private bool TryEmitDivS64Helper(
            List<SassNode> nodes, int index, SassInstruction call)
        {
            if (!TryDivS64Abi(nodes, index, call,
                    out string divisorLow, out string divisorHigh,
                    out string dividendLow, out string dividendHigh,
                    out List<string> outputs) ||
                outputs.Count == 0)
                return false;
            string dividend = $"%rdt{_nextTemp++}";
            string divisor = $"%rdt{_nextTemp++}";
            string quotient = $"%rdt{_nextTemp++}";
            _lines.Add($"    mov.b64 {dividend}, " +
                $"{{{RegisterOrUniform(dividendLow)}, " +
                $"{RegisterOrUniform(dividendHigh)}}};");
            _lines.Add($"    mov.b64 {divisor}, " +
                $"{{{RegisterOrUniform(divisorLow)}, " +
                $"{RegisterOrUniform(divisorHigh)}}};");
            _lines.Add($"    div.s64 {quotient}, {dividend}, {divisor};");
            foreach (string output in outputs)
                ForgetRegisterPair(output);
            string low = $"%rt{_nextTemp++}";
            string high = $"%rt{_nextTemp++}";
            _lines.Add($"    mov.b64 {{{low}, {high}}}, {quotient};");
            _lines.Add($"    mov.b32 {RegisterOrUniform(outputs[0])}, {low};");
            if (outputs.Count >= 2 &&
                NextRegister(outputs[0])?.Equals(outputs[1],
                    StringComparison.OrdinalIgnoreCase) == true)
                _lines.Add($"    mov.b32 {RegisterOrUniform(outputs[1])}, " +
                    $"{high};");
            return true;
        }

        private string RegisterOrUniform(string register) =>
            register.StartsWith("UR", StringComparison.OrdinalIgnoreCase)
                ? UniformReg(register) : Reg(register);

        private bool TryDivS64Abi(
            List<SassNode> nodes, int index, SassInstruction call,
            out string divisorLow, out string divisorHigh,
            out string dividendLow, out string dividendHigh,
            out List<string> outputs)
        {
            divisorLow = divisorHigh = dividendLow = dividendHigh = "";
            outputs = new List<string>();
            string? symbol = call.Relocations.FirstOrDefault(
                relocation => !string.IsNullOrWhiteSpace(
                    relocation.Symbol))?.Symbol;
            int start = -1;
            if (symbol != null)
            {
                for (int i = 0; i < nodes.Count; i++)
                {
                    if (nodes[i] is SassLabel label &&
                        (label.Name.Equals(symbol,
                             StringComparison.Ordinal) ||
                         LabelName(label.Name).Equals(symbol,
                             StringComparison.Ordinal)))
                    {
                        start = i;
                        break;
                    }
                }
            }
            if (start < 0)
            {
                string? raw = call.BranchTargets.FirstOrDefault()
                    ?? call.Operands.LastOrDefault();
                if (raw == null ||
                    !TrySwitchImmediate(
                        raw.Trim().Trim('`', '(', ')', ';'),
                        out int target))
                    return false;
                for (int i = 0; i < nodes.Count; i++)
                {
                    if (nodes[i] is SassLabel label &&
                        label.Address == target)
                    {
                        start = i;
                        break;
                    }
                }
            }
            if (start < 0)
                return false;
            int end = start + 1;
            for (; end < nodes.Count; end++)
                if (nodes[end] is SassInstruction ret &&
                    ret.Mnemonic.StartsWith(
                        "RET", StringComparison.OrdinalIgnoreCase))
                    break;
            if (end >= nodes.Count)
                return false;

            var pairs = new List<(string Low, string High)>();
            for (int i = start; i + 1 < end && pairs.Count < 2; i++)
            {
                if (nodes[i] is not SassInstruction low ||
                    !low.Mnemonic.StartsWith(
                        "IADD3", StringComparison.OrdinalIgnoreCase) ||
                    !TryNegatedRegisterName(low, out string lowRegister))
                    continue;
                string? predicate = low.Operands.FirstOrDefault(op =>
                    Regex.IsMatch(op.Trim(), @"^!?P\d+$",
                        RegexOptions.IgnoreCase));
                for (int j = i + 1; j < Math.Min(end, i + 6); j++)
                {
                    if (nodes[j] is not SassInstruction high ||
                        !high.Mnemonic.StartsWith(
                            "IMAD.X", StringComparison.OrdinalIgnoreCase))
                        continue;
                    if (predicate != null &&
                        !high.Operands.Any(op => op.Trim().Equals(
                            predicate.Trim(), StringComparison.OrdinalIgnoreCase)))
                        continue;
                    if (TryComplementedRegisterName(high, out string highRegister) &&
                        IsHelperInput(nodes, start, end, lowRegister) &&
                        IsHelperInput(nodes, start, end, highRegister))
                    {
                        pairs.Add((lowRegister, highRegister));
                        break;
                    }
                }
            }
            if (pairs.Count < 2)
                return false;
            divisorLow = pairs[0].Low;
            divisorHigh = pairs[0].High;
            dividendLow = pairs[1].Low;
            dividendHigh = pairs[1].High;

            var helperWrites = new HashSet<string>(
                StringComparer.OrdinalIgnoreCase);
            for (int i = start; i < end; i++)
            {
                if (nodes[i] is not SassInstruction ins)
                    continue;
                int destination = DestinationIndex(ins);
                if (destination >= 0 && destination < ins.Operands.Count)
                    helperWrites.Add(BaseReg(ins.Operands[destination]));
            }
            var callerWrites = new HashSet<string>(
                StringComparer.OrdinalIgnoreCase);
            // The call is sometimes followed immediately by a branch to
            // the merge point; the quotient is first read there.
            int scan = index + 1;
            int scanned = 0;
            bool followedBranch = false;
            while (scan < nodes.Count && scanned < 24)
            {
                if (nodes[scan] is SassLabel)
                {
                    if (followedBranch)
                    {
                        scan++;
                        scanned++;
                        continue;
                    }
                    break;
                }
                if (nodes[scan] is not SassInstruction ins)
                {
                    scan++;
                    scanned++;
                    continue;
                }
                if (ins.Mnemonic.StartsWith("BRA", StringComparison.OrdinalIgnoreCase) &&
                    string.IsNullOrWhiteSpace(ins.Guard))
                {
                    int target = FindLabelNode(nodes, ins);
                    if (target < 0)
                        break;
                    scan = target + 1;
                    scanned++;
                    followedBranch = true;
                    continue;
                }
                if (ins.Mnemonic.StartsWith("BRA", StringComparison.OrdinalIgnoreCase) ||
                    ins.Mnemonic.StartsWith("EXIT", StringComparison.OrdinalIgnoreCase) ||
                    ins.Mnemonic.StartsWith("RET", StringComparison.OrdinalIgnoreCase))
                    break;
                int destination = DestinationIndex(ins);
                for (int op = 0; op < ins.Operands.Count; op++)
                {
                    if (op == destination)
                        continue;
                    string register = BaseReg(ins.Operands[op]);
                    if (!Regex.IsMatch(register, @"^(?:U)?R\d+$",
                            RegexOptions.IgnoreCase))
                        continue;
                    if (helperWrites.Contains(register) &&
                        !callerWrites.Contains(register) &&
                        !outputs.Contains(register, StringComparer.OrdinalIgnoreCase))
                        outputs.Add(register);
                }
                if (destination >= 0 && destination < ins.Operands.Count)
                    callerWrites.Add(BaseReg(ins.Operands[destination]));
                scan++;
                scanned++;
            }
            return outputs.Count > 0;
        }

        private int FindLabelNode(
            List<SassNode> nodes, SassInstruction branch)
        {
            string? raw = branch.BranchTargets.FirstOrDefault()
                ?? branch.Operands.LastOrDefault();
            if (raw == null)
                return -1;
            string candidate = raw.Trim().Trim('`', '(', ')', ';');
            if (TrySwitchImmediate(candidate, out int address))
            {
                for (int i = 0; i < nodes.Count; i++)
                    if (nodes[i] is SassLabel label &&
                        label.Address == address)
                        return i;
            }
            string name = LabelName(candidate);
            for (int i = 0; i < nodes.Count; i++)
                if (nodes[i] is SassLabel label &&
                    LabelName(label.Name).Equals(name,
                        StringComparison.Ordinal))
                    return i;
            return -1;
        }

        private static bool TryNegatedRegisterName(
            SassInstruction instruction, out string register)
        {
            register = "";
            foreach (string operand in instruction.Operands)
            {
                string text = operand.Trim();
                if (!text.StartsWith("-", StringComparison.Ordinal))
                    continue;
                string candidate = BaseReg(text[1..]);
                if (Regex.IsMatch(candidate, @"^(?:U)?R\d+$",
                        RegexOptions.IgnoreCase))
                {
                    register = candidate;
                    return true;
                }
            }
            return false;
        }

        private static bool TryComplementedRegisterName(
            SassInstruction instruction, out string register)
        {
            register = "";
            foreach (string operand in instruction.Operands)
            {
                string text = operand.Trim();
                if (!text.StartsWith("~", StringComparison.Ordinal))
                    continue;
                string candidate = BaseReg(text[1..]);
                if (Regex.IsMatch(candidate, @"^(?:U)?R\d+$",
                        RegexOptions.IgnoreCase))
                {
                    register = candidate;
                    return true;
                }
            }
            return false;
        }

        private static bool IsHelperInput(
            List<SassNode> nodes, int start, int end, string register)
        {
            for (int i = start; i < end; i++)
            {
                if (nodes[i] is not SassInstruction instruction)
                    continue;
                int destination = DestinationIndex(instruction);
                for (int op = 0; op < instruction.Operands.Count; op++)
                {
                    if (op == destination)
                        continue;
                    string text = instruction.Operands[op].Trim();
                    if (text.StartsWith("-", StringComparison.Ordinal) ||
                        text.StartsWith("~", StringComparison.Ordinal))
                        text = text[1..];
                    if (BaseReg(text).Equals(register,
                            StringComparison.OrdinalIgnoreCase))
                        return true;
                }
                if (destination >= 0 &&
                    destination < instruction.Operands.Count &&
                    BaseReg(instruction.Operands[destination]).Equals(
                        register, StringComparison.OrdinalIgnoreCase))
                    return false;
            }
            return false;
        }

        private bool TryEmitDeviceHelperCall(List<SassNode> nodes, ref int index)
        {
            if (nodes[index] is not SassInstruction call ||
                !IsCallMnemonic(call.Mnemonic) ||
                !string.IsNullOrWhiteSpace(call.Guard))
                return false;
            string? symbol = call.Relocations
                .FirstOrDefault(relocation =>
                    !string.IsNullOrWhiteSpace(relocation.Symbol))
                ?.Symbol;
            if (symbol == null)
                return false;
            if (symbol.Contains("__cuda_sm20_sqrt_rn_f32_slowpath",
                    StringComparison.Ordinal))
                return TryEmitUnaryDeviceHelper(
                    nodes, index, call, "sqrt.rn.f32");
            if (symbol.Contains("__cuda_sm20_rcp_rn_f32_slowpath",
                    StringComparison.Ordinal))
                return TryEmitUnaryDeviceHelper(
                    nodes, index, call, "rcp.rn.f32");
            if (symbol.Contains("__cuda_sm3x_div_rn_noftz_f32_slowpath",
                    StringComparison.Ordinal) ||
                symbol.Contains("__cuda_sm20_div_rn_f32_slowpath",
                    StringComparison.Ordinal))
                return TryEmitBinaryDeviceHelper(
                    nodes, index, call, "div.rn.f32");
            if (symbol.Contains("__cuda_sm20_div_s64",
                    StringComparison.Ordinal))
                return TryEmitDivS64Helper(nodes, index, call);
            if (symbol.Contains("__cuda_sm20_rem_u64",
                    StringComparison.Ordinal))
            {
                EmitWideDivide("rem", "R14", "R15", "R10", "R11",
                    "R12", "R13");
                return true;
            }
            if (symbol.Contains("__cuda_sm20_div_u64",
                    StringComparison.Ordinal))
            {
                EmitWideDivide("div", "R12", "R13", "R10", "R11",
                    "R14", "R15");
                return true;
            }
            return false;
        }

        // 64-bit divide/remainder slow paths share the compiler's internal
        // ABI: two register pairs in, one pair out.  The slow path is only
        // reached for 64-bit divisors; ordinary tensor sizes take the
        // inlined 32-bit fast path and skip the call entirely.
        private void EmitWideDivide(string operation,
            string dividendLow, string dividendHigh,
            string divisorLow, string divisorHigh,
            string resultLow, string resultHigh)
        {
            string dividend = $"%rdt{_nextTemp++}";
            string divisor = $"%rdt{_nextTemp++}";
            string result = $"%rdt{_nextTemp++}";
            ForgetRegisterPair(dividendLow);
            ForgetRegisterPair(divisorLow);
            ForgetRegisterPair(resultLow);
            _lines.Add($"    mov.b64 {dividend}, " +
                $"{{{Reg(dividendLow)}, {Reg(dividendHigh)}}};");
            _lines.Add($"    mov.b64 {divisor}, " +
                $"{{{Reg(divisorLow)}, {Reg(divisorHigh)}}};");
            _lines.Add($"    {operation}.u64 {result}, {dividend}, {divisor};");
            _lines.Add($"    mov.b64 {{{Reg(resultLow)}, {Reg(resultHigh)}}}, " +
                $"{result};");
        }

        // Shared lowering for one-input/one-output device-runtime helpers
        // (sqrt.rn, rcp.rn).  NVCC materialises the input a few
        // instructions before the call and reads the result just after it;
        // the helper body is the fallback when the call site is scheduled
        // differently.
        private bool TryEmitUnaryDeviceHelper(
            List<SassNode> nodes, int index, SassInstruction call,
            string operation)
        {
            string target;
            try
            {
                target = CallTarget(nodes, index, call);
            }
            catch (NotSupportedException)
            {
                return false;
            }
            int labelIndex = -1;
            for (int i = index + 1; i < nodes.Count; i++)
            {
                if (nodes[i] is SassLabel label &&
                    LabelName(label.Name).Equals(
                        target, StringComparison.Ordinal))
                {
                    labelIndex = i;
                    break;
                }
            }
            string input = "", output = "";
            for (int b = index - 1; b >= 0 && b >= index - 4; b--)
            {
                if (nodes[b] is not SassInstruction before)
                    continue;
                if (!before.Mnemonic.Equals(
                        "MOV", StringComparison.OrdinalIgnoreCase) &&
                    !before.Mnemonic.StartsWith(
                        "IMAD.MOV", StringComparison.OrdinalIgnoreCase))
                    continue;
                if (before.Operands.Count < 2)
                    continue;
                string source = BaseReg(before.Operands[^1]);
                if (!Regex.IsMatch(source, @"^R\d+$"))
                    continue;
                int dst = DestinationIndex(before);
                if (dst < 0 || dst >= before.Operands.Count)
                    continue;
                string targetRegister = BaseReg(before.Operands[dst]);
                if (Regex.IsMatch(targetRegister, @"^R\d+$"))
                {
                    input = targetRegister;
                    break;
                }
            }
            for (int a = index + 1; a < nodes.Count && a <= index + 4; a++)
            {
                if (nodes[a] is SassLabel)
                    continue;
                if (nodes[a] is not SassInstruction after)
                    break;
                if (!after.Mnemonic.Equals(
                        "MOV", StringComparison.OrdinalIgnoreCase) &&
                    !after.Mnemonic.StartsWith(
                        "IMAD.MOV", StringComparison.OrdinalIgnoreCase))
                    continue;
                string source = after.Mnemonic.Equals(
                        "MOV", StringComparison.OrdinalIgnoreCase)
                    ? after.Operands[1].Trim()
                    : after.Operands[^1].Trim();
                if (Regex.IsMatch(BaseReg(source), @"^R\d+$"))
                {
                    output = BaseReg(source);
                    break;
                }
            }
            if ((input.Length == 0 || output.Length == 0) &&
                labelIndex >= 0)
                AnalyzeHelperRegisters(nodes, labelIndex,
                    out input, out output);
            if (input.Length == 0 || output.Length == 0)
                return false;
            ForgetRegisterPair(input);
            ForgetRegisterPair(output);
            _lines.Add($"    {operation} {Reg(output)}, {Reg(input)};");
            return true;
        }

        // Two-input device helper (the f32 divide slow path).  The call
        // site materialises dividend/divisor in an FCHK immediately before
        // the call; the result lands in the register the helper writes last
        // (or, when the helper body is in another section, the register the
        // fast path computed just before the call).
        private bool TryEmitBinaryDeviceHelper(
            List<SassNode> nodes, int index, SassInstruction call,
            string operation)
        {
            string dividend = "", divisor = "";
            for (int b = index - 1; b >= 0 && b >= index - 40; b--)
            {
                if (nodes[b] is SassLabel blockLabel &&
                    !blockLabel.Name.StartsWith(
                        ".L_", StringComparison.Ordinal))
                    break;
                if (nodes[b] is not SassInstruction before)
                    continue;
                if (!before.Mnemonic.Equals(
                        "FCHK", StringComparison.OrdinalIgnoreCase))
                    continue;
                if (before.Operands.Count >= 3)
                {
                    string a = SassRegister(BaseReg(before.Operands[1]));
                    string d = SassRegister(BaseReg(before.Operands[2]));
                    if (Regex.IsMatch(a, @"^R\d+$") &&
                        Regex.IsMatch(d, @"^R\d+$"))
                    {
                        dividend = a;
                        divisor = d;
                    }
                }
                break;
            }
            if (dividend.Length == 0 || divisor.Length == 0)
                return false;

            string output = "";
            string? target = null;
            try
            {
                target = CallTarget(nodes, index, call);
            }
            catch (NotSupportedException)
            {
                // Helper body lives in a linked section; use the call-site
                // heuristic below.
            }
            int labelIndex = -1;
            if (target != null)
            {
                for (int i = index + 1; i < nodes.Count; i++)
                {
                    if (nodes[i] is SassLabel label &&
                        LabelName(label.Name).Equals(
                            target, StringComparison.Ordinal))
                    {
                        labelIndex = i;
                        break;
                    }
                }
            }
            if (labelIndex >= 0)
                AnalyzeHelperRegisters(nodes, labelIndex, out _, out output);
            if (output.Length == 0)
            {
                // The helper lives in a linked section.  The fast-path
                // quotient is written by the last arithmetic instruction
                // before the branch that skips the slow path; ABI argument
                // moves sit between that branch and the call itself.
                int scanEnd = index - 1;
                for (int b = index - 1; b >= 0 && b >= index - 40; b--)
                {
                    if (nodes[b] is SassLabel blockLabel &&
                        !blockLabel.Name.StartsWith(
                            ".L_", StringComparison.Ordinal))
                    {
                        scanEnd = b - 1;
                        break;
                    }
                    if (nodes[b] is SassInstruction branch &&
                        (branch.Mnemonic.StartsWith(
                             "BRA", StringComparison.OrdinalIgnoreCase) ||
                         branch.Mnemonic.StartsWith(
                             "BSSY", StringComparison.OrdinalIgnoreCase) ||
                         branch.Mnemonic.StartsWith(
                             "BSYNC", StringComparison.OrdinalIgnoreCase)))
                    {
                        scanEnd = b - 1;
                        break;
                    }
                }
                for (int b = scanEnd; b >= 0 && b >= index - 40; b--)
                {
                    if (nodes[b] is SassLabel blockLabel &&
                        !blockLabel.Name.StartsWith(
                            ".L_", StringComparison.Ordinal))
                        break;
                    if (nodes[b] is not SassInstruction before)
                        continue;
                    int dst = DestinationIndex(before);
                    if (dst < 0 || dst >= before.Operands.Count)
                        continue;
                    string written =
                        SassRegister(BaseReg(before.Operands[dst]));
                    if (Regex.IsMatch(written, @"^R\d+$") &&
                        !written.Equals("RZ",
                            StringComparison.OrdinalIgnoreCase))
                    {
                        output = written;
                        break;
                    }
                }
            }
            if (output.Length == 0)
                return false;
            ForgetRegisterPair(output);
            ForgetRegisterPair(dividend);
            ForgetRegisterPair(divisor);
            _lines.Add($"    {operation} {Reg(output)}, " +
                $"{Reg(dividend)}, {Reg(divisor)};");
            return true;
        }

        private void ForgetRegisterPair(string register)
        {
            string value = BaseReg(register);
            _pairRegs.Remove(value);
            Match match = Regex.Match(value, @"^R(\d+)$",
                RegexOptions.IgnoreCase);
            if (!match.Success)
                return;
            int number = int.Parse(match.Groups[1].Value,
                CultureInfo.InvariantCulture);
            if (number > 0)
                _pairRegs.Remove("R" + (number - 1).ToString(
                    CultureInfo.InvariantCulture));
        }

        private void AnalyzeHelperRegisters(List<SassNode> nodes, int labelIndex,
                                            out string input, out string output)
        {
            input = "";
            output = "";
            // RET.REL names the return-address pair; its writes are ABI
            // plumbing, not the helper's result.
            var excluded = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            bool retStarted = false;
            for (int i = labelIndex + 1; i < nodes.Count; i++)
            {
                if (nodes[i] is SassLabel retLabel)
                {
                    // A function can have several leading labels (one per
                    // symbol/section view); only a label after real code
                    // marks the next function.
                    if (!retStarted)
                        continue;
                    if (!retLabel.Name.StartsWith(
                            ".L_", StringComparison.Ordinal))
                        break;
                    continue;
                }
                if (nodes[i] is not SassInstruction retInstruction)
                    break;
                retStarted = true;
                string retOp = retInstruction.Mnemonic
                    .Split('.')[0].ToUpperInvariant();
                if (retOp is not ("RET" or "EXIT"))
                    continue;
                if (retOp == "RET" && retInstruction.Operands.Count > 0)
                {
                    string low = BaseReg(retInstruction.Operands[^1]);
                    if (Regex.IsMatch(low, @"^R\d+$"))
                    {
                        excluded.Add(low);
                        excluded.Add(NextRegister(low));
                    }
                }
                break;
            }
            bool started = false;
            for (int i = labelIndex + 1; i < nodes.Count; i++)
            {
                if (nodes[i] is SassLabel inner)
                {
                    if (!started)
                        continue;
                    if (!inner.Name.StartsWith(
                            ".L_", StringComparison.Ordinal))
                        break;
                    continue;
                }
                if (nodes[i] is not SassInstruction ins)
                    break;
                started = true;
                string op = ins.Mnemonic.Split('.')[0].ToUpperInvariant();
                if (op is "RET" or "EXIT")
                    break;
                int dst = DestinationIndex(ins);
                if (dst >= 0 && dst < ins.Operands.Count)
                {
                    string written = SassRegister(BaseReg(ins.Operands[dst]));
                    if (Regex.IsMatch(written, @"^R\d+$") &&
                        !written.Equals("RZ", StringComparison.OrdinalIgnoreCase) &&
                        !excluded.Contains(written))
                        output = written;
                }
                if (input.Length == 0)
                {
                    foreach (string read in ReadRegisters(ins))
                    {
                        // ReadRegisters yields PTX-style "%rN" names; do
                        // not run them through BaseReg (which returns the
                        // lowercase "rN" and then fails the R\d+ check).
                        string value = SassRegister(read);
                        if (Regex.IsMatch(value, @"^R\d+$",
                                RegexOptions.IgnoreCase) &&
                            !value.Equals("RZ", StringComparison.OrdinalIgnoreCase))
                        {
                            input = value;
                            break;
                        }
                    }
                }
            }
        }

        private bool TryEmitGuardedExitCall(List<SassNode> nodes, ref int index)
        {
            if (nodes[index] is not SassInstruction call ||
                !IsCallMnemonic(call.Mnemonic) ||
                string.IsNullOrWhiteSpace(call.Guard))
                return false;
            string target;
            try
            {
                target = CallTarget(nodes, index, call);
            }
            catch (NotSupportedException)
            {
                return false;
            }
            int labelIndex = -1;
            for (int i = index + 1; i < nodes.Count; i++)
            {
                if (nodes[i] is SassLabel label &&
                    LabelName(label.Name).Equals(
                        target, StringComparison.Ordinal))
                {
                    labelIndex = i;
                    break;
                }
            }
            if (labelIndex < 0 || _skipNodes.Contains(labelIndex))
                return false;
            int next = labelIndex + 1;
            while (next < nodes.Count && nodes[next] is SassLabel)
                next++;
            if (next >= nodes.Count || nodes[next] is not SassInstruction body)
                return false;
            if (body.Mnemonic.StartsWith(
                    "EXIT", StringComparison.OrdinalIgnoreCase))
            {
                _lines.Add($"{Guard(call.Guard)} exit;");
                return true;
            }
            if (body.Mnemonic.StartsWith(
                    "BSYNC", StringComparison.OrdinalIgnoreCase) ||
                body.Mnemonic.StartsWith(
                    "BSSY", StringComparison.OrdinalIgnoreCase) ||
                body.Mnemonic.StartsWith(
                    "BRA", StringComparison.OrdinalIgnoreCase))
            {
                _lines.Add($"{Guard(call.Guard)} bra {target};");
                return true;
            }
            return false;
        }

        private bool TryEmitFuncCall(List<SassNode> nodes, ref int index)
        {
            if (!_emitFunctions ||
                nodes[index] is not SassInstruction call ||
                !IsCallMnemonic(call.Mnemonic) ||
                !string.IsNullOrWhiteSpace(call.Guard))
                return false;
            string target;
            try
            {
                target = CallTarget(nodes, index, call);
            }
            catch (NotSupportedException)
            {
                return false;
            }
            if (!_functionsByTarget.TryGetValue(
                    target, out PtxFunctionAbi? function) ||
                target.Contains("__cuda_sm20_", StringComparison.Ordinal))
                return false;
            int serial = _functionCallSerial++;
            var arguments = new List<string>();
            for (int i = 0; i < function.Arguments.Count; i++)
            {
                PtxFunctionArgument argument = function.Arguments[i];
                string name = $"__fc{serial}_a{i}";
                arguments.Add(name);
                if (argument.IsPointer)
                {
                    _lines.Add($"    .param .u64 {name};");
                    string value;
                    if (_pointerRegs.TryGetValue(
                            argument.Register, out string? pointer))
                    {
                        value = pointer;
                    }
                    else
                    {
                        value = $"%rdt{_nextTempDouble++}";
                        _lines.Add(
                            $"    mov.b64 {value}, " +
                            $"{{{Reg(argument.Register)}, " +
                            $"{Reg(argument.HighRegister!)}}};");
                    }
                    _lines.Add($"    st.param.u64 [{name}], {value};");
                }
                else
                {
                    _lines.Add($"    .param .u32 {name};");
                    _lines.Add(
                        $"    st.param.u32 [{name}], " +
                        $"{ScalarCallArgument(argument.Register)};");
                }
            }
            string callArguments = string.Join(", ", arguments);
            if (function.Returns.Count == 1)
            {
                string returnName = $"__fc{serial}_r";
                _lines.Add($"    .param .u32 {returnName};");
                _lines.Add(
                    $"    call.uni ({returnName}), {function.Name}, " +
                    $"({callArguments});");
                string reg = function.Returns[0];
                _lines.Add(
                    $"    ld.param.u32 {Reg(reg)}, [{returnName}];");
                _pointerRegs.Remove(reg);
            }
            else if (function.Returns.Count == 2)
            {
                string returnName = $"__fc{serial}_r";
                _lines.Add($"    .param .u64 {returnName};");
                _lines.Add(
                    $"    call.uni ({returnName}), {function.Name}, " +
                    $"({callArguments});");
                string packed = $"%rdt{_nextTempDouble++}";
                string low = $"%ru{_nextTempU32++}";
                string high = $"%ru{_nextTempU32++}";
                _lines.Add(
                    $"    ld.param.u64 {packed}, [{returnName}];");
                _lines.Add(
                    $"    mov.b64 {{{low}, {high}}}, {packed};");
                _lines.Add(
                    $"    mov.b32 {Reg(function.Returns[0])}, {low};");
                _lines.Add(
                    $"    mov.b32 {Reg(function.Returns[1])}, {high};");
                _pointerRegs.Remove(function.Returns[0]);
                _pointerRegs.Remove(function.Returns[1]);
            }
            else if (function.Returns.Count is 3 or 4)
            {
                _lines.Add(
                    $"    call.uni (" +
                    $"{string.Join(", ", function.Returns.Select(
                        reg => Reg(reg)))}), {function.Name}, " +
                    $"({callArguments});");
                foreach (string reg in function.Returns)
                    _pointerRegs.Remove(reg);
            }
            else if (function.Returns.Count == 0)
            {
                _lines.Add(
                    $"    call.uni {function.Name}, ({callArguments});");
            }
            else
            {
                return false;
            }
            for (int i = function.LabelIndex; i < function.BodyEnd; i++)
                _skipNodes.Add(i);
            return true;
        }

        private bool TryEmitLeafCallInline(List<SassNode> nodes, ref int index)
        {
            if (nodes[index] is not SassInstruction call ||
                !IsCallMnemonic(call.Mnemonic))
                return false;
            if (!string.IsNullOrWhiteSpace(call.Guard) && _extraGuard != null)
                return false;
            string target;
            try
            {
                target = CallTarget(nodes, index, call);
            }
            catch (NotSupportedException)
            {
                return false;
            }
            // The f64 sqrt mediumpath is a single-return leaf with a
            // plain register ABI; inline its body instead of modelling
            // the register convention by hand (it differs per arch).
            // The other NVCC runtime helpers keep their call-site
            // lowering.
            bool deviceHelper = target.Contains(
                "__cuda_sm20_", StringComparison.Ordinal);
            bool inlinableHelper = deviceHelper &&
                target.Contains("__cuda_sm20_dsqrt_rn_f64_mediumpath",
                    StringComparison.Ordinal);
            if (deviceHelper && !inlinableHelper)
                return false;
            if (_inlineCallStack.Contains(target))
                return false;
            // NVCC's f64 runtime helpers propagate their own carry
            // chains (IADD3 carry-out followed by IMAD.X), so allow .X
            // inside the body even though callers cannot export carry
            // state.
            bool allowInternalCarry = inlinableHelper;
            // Labels are emitted with synthetic numeric names; a call in
            // one section and its callee in another therefore cannot be
            // matched by emitted name.  Keep the raw call target as a
            // secondary key.
            string rawTarget = call.Relocations
                .FirstOrDefault(relocation =>
                    !string.IsNullOrWhiteSpace(relocation.Symbol))?.Symbol
                ?? (call.BranchTargets.FirstOrDefault()
                    ?? call.Operands.LastOrDefault() ?? "")
                    .Trim().Trim('`', '(', ')', ';');
            int labelIndex = -1;
            for (int i = index + 1; i < nodes.Count; i++)
            {
                if (nodes[i] is SassLabel label &&
                    (LabelName(label.Name).Equals(
                         target, StringComparison.Ordinal) ||
                     label.Name.Equals(
                         rawTarget, StringComparison.Ordinal)))
                {
                    labelIndex = i;
                    break;
                }
            }
            if (labelIndex < 0)
            {
                // A linked cubin may keep callees in a separate .text
                // section.  Give the normal inline machinery a synthetic
                // call followed by that section's nodes, and keep its node
                // indices out of the caller's skip set.
                foreach (SassSection linked in _linkedSections)
                {
                    if (ReferenceEquals(linked, _section))
                        continue;
                    List<SassNode> linkedNodes = linked.Nodes.ToList();
                    SassLabel? linkedTarget = linkedNodes
                        .OfType<SassLabel>()
                        .FirstOrDefault(linkedLabel =>
                            LabelName(linkedLabel.Name).Equals(
                                target, StringComparison.Ordinal) ||
                            linkedLabel.Name.Equals(
                                rawTarget, StringComparison.Ordinal));
                    if (linkedTarget == null)
                        continue;
                    var callNodes = new List<SassNode>(linkedNodes.Count + 1);
                    var synthetic = new SassInstruction { Mnemonic = "CAL" };
                    synthetic.Operands.Add(linkedTarget.Name);
                    callNodes.Add(synthetic);
                    callNodes.AddRange(linkedNodes);
                    var savedSkip = new HashSet<int>(_skipNodes);
                    _skipNodes.Clear();
                    int nestedIndex = 0;
                    bool linkedSuccess;
                    try
                    {
                        linkedSuccess = TryEmitLeafCallInline(
                            callNodes, ref nestedIndex);
                    }
                    finally
                    {
                        _skipNodes.Clear();
                        foreach (int saved in savedSkip)
                            _skipNodes.Add(saved);
                    }
                    return linkedSuccess;
                }
                return false;
            }

            var body = new List<SassNode>();
            int bodyEnd = nodes.Count;
            for (int i = labelIndex + 1; i < nodes.Count; i++)
            {
                if (nodes[i] is SassLabel boundary &&
                    !boundary.Name.StartsWith(".L_", StringComparison.Ordinal))
                {
                    bodyEnd = i;
                    break;
                }
            }

            var labelIndices = new Dictionary<string, int>(StringComparer.Ordinal);
            for (int i = labelIndex; i < bodyEnd; i++)
            {
                if (nodes[i] is SassLabel label &&
                    !AddLabelIndex(labelIndices, label.Name, i))
                    return false;
            }

            var reachable = new SortedSet<int>();
            var returns = new SortedSet<int>();
            var work = new Stack<int>();
            work.Push(labelIndex);
            while (work.Count > 0)
            {
                int pc = work.Pop();
                bool fellThrough = false;
                while (true)
                {
                    if (pc < labelIndex || pc >= bodyEnd)
                    {
                        fellThrough = true;
                        break;
                    }
                    if (!reachable.Add(pc))
                        break;
                    if (nodes[pc] is SassLabel)
                    {
                        pc++;
                        continue;
                    }
                    if (nodes[pc] is not SassInstruction instruction)
                        return false;
                    string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
                    if (op == "RET")
                    {
                        returns.Add(pc);
                        if (!string.IsNullOrWhiteSpace(instruction.Guard))
                        {
                            pc++;
                            continue;
                        }
                        break;
                    }
                    if (IsCallOp(op))
                    {
                        if (!string.IsNullOrWhiteSpace(instruction.Guard))
                            return false;
                        pc++;
                        continue;
                    }
                    if (!IsInlineSafeCallBody(
                            instruction, allowInternalCarry))
                        return false;
                    if (op == "SSY")
                    {
                        string syncTarget;
                        try
                        {
                            syncTarget = RawBranchTarget(instruction);
                        }
                        catch (NotSupportedException)
                        {
                            return false;
                        }
                        if (!TryResolveLabelIndex(
                                labelIndices, syncTarget, out int syncIndex) ||
                            syncIndex < labelIndex || syncIndex >= bodyEnd)
                            return false;
                        pc++;
                        continue;
                    }
                    if (op is "BRA" or "SYNC" or "PBK" or "PCNT")
                    {
                        string branchTarget;
                        try
                        {
                            branchTarget = RawBranchTarget(instruction);
                        }
                        catch (NotSupportedException)
                        {
                            return false;
                        }
                        if (!TryResolveLabelIndex(
                                labelIndices, branchTarget, out int targetIndex))
                            return false;
                        if (targetIndex < labelIndex || targetIndex >= bodyEnd)
                            return false;
                        work.Push(targetIndex);
                        if (!string.IsNullOrWhiteSpace(instruction.Guard))
                        {
                            pc++;
                            continue;
                        }
                        break;
                    }
                    pc++;
                }
                if (fellThrough)
                    return false;
            }
            if (returns.Count == 0 || reachable.Count == 0)
                return false;
            bool singleReturn = returns.Count == 1 &&
                nodes[returns.Min] is SassInstruction singleReturnInstruction &&
                string.IsNullOrWhiteSpace(singleReturnInstruction.Guard);
            bool usesLocalFrame = reachable.Any(index =>
                nodes[index] is SassInstruction localInstruction &&
                (localInstruction.Mnemonic.StartsWith(
                     "LDL", StringComparison.OrdinalIgnoreCase) ||
                 localInstruction.Mnemonic.StartsWith(
                     "STL", StringComparison.OrdinalIgnoreCase)));
            bool hasNestedCall = reachable.Any(index =>
                nodes[index] is SassInstruction nested &&
                IsCallMnemonic(nested.Mnemonic));
            if (hasNestedCall && (!singleReturn || usesLocalFrame))
                return false;
            if (!singleReturn && usesLocalFrame)
                return false;
            if (!singleReturn)
            {
                if (TryEmitIfConvertedGuardedReturn(
                        nodes, reachable, returns) ||
                    TryEmitIfConvertedDiamond(
                        nodes, reachable, labelIndices, bodyEnd) ||
                    TryEmitGeneralMultiReturnInline(nodes, reachable))
                {
                    for (int i = labelIndex; i < bodyEnd; i++)
                        _skipNodes.Add(i);
                    return true;
                }
                return false;
            }
            if (!usesLocalFrame && !hasNestedCall &&
                TryEmitIfConvertedSingleReturnDiamond(
                    nodes, reachable, labelIndices, bodyEnd))
            {
                for (int i = labelIndex; i < bodyEnd; i++)
                    _skipNodes.Add(i);
                return true;
            }
            foreach (int nodeIndex in reachable)
                body.Add(nodes[nodeIndex]);
            var bodyGlobalIndices = reachable.ToList();

            var inlineGuards = new Dictionary<int, string>();
            var redundantSyncs = new HashSet<int>();
            BuildInlineGuardPlan(
                body, inlineGuards, redundantSyncs);
            Dictionary<int, LocalPlanAction>? localPlan = null;
            if (usesLocalFrame &&
                !TryBuildLocalAddressPlan(body, out localPlan))
                localPlan = null;
            bool dynamicLocalFrame = false;
            if (usesLocalFrame && localPlan == null &&
                !TryValidateDynamicLocalFrame(body))
                return false;
            if (usesLocalFrame && localPlan == null)
                dynamicLocalFrame = true;

            var overrides = new Dictionary<string, string>(StringComparer.Ordinal);
            string prefix = $"L_inline_{_inlineSerial++}_";
            string? selfTailLabel = null;
            int inlineLabelSerial = 0;
            bool shortInlineLabels = target.Contains(
                "__cuda_sm20_", StringComparison.Ordinal);
            foreach (SassNode node in body)
            {
                if (node is not SassLabel label)
                    continue;
                // Keep the emitted name short for NVCC runtime helpers:
                // embedding their sanitized symbol produced labels over
                // ACuda's 64-byte identifier limit.
                string unique = shortInlineLabels
                    ? prefix + (inlineLabelSerial++).ToString(
                        CultureInfo.InvariantCulture)
                    : prefix + Sanitize(label.Name);
                while (_labels.Values.Contains(unique) ||
                       overrides.ContainsValue(unique))
                    unique += "_";
                if (!AddLabelOverride(overrides, label.Name, unique))
                    return false;
                if (LabelName(label.Name).Equals(
                        target, StringComparison.Ordinal))
                    selfTailLabel = unique;
            }

            bool hadTargetLabel = _inlineTargetLabels.TryGetValue(
                target, out string? previousTargetLabel);
            if (selfTailLabel != null)
                _inlineTargetLabels[target] = selfTailLabel;
            bool hadR1 = _pointerRegs.TryGetValue("R1", out string? previousR1);
            string? dynamicFramePointer = null;
            if (usesLocalFrame)
            {
                if (dynamicLocalFrame)
                {
                    // The inline body was not visible to the header
                    // pre-scan (its stack adjustment can be an IADD3 or
                    // a non-R1 form), so declare the frame on first use.
                    if (!_usesDynamicLocalFrame)
                    {
                        _lines.Add("    .local .align 16 .b8 " +
                            "__callee_dynamic_frame[65536];");
                        _usesDynamicLocalFrame = true;
                    }
                    dynamicFramePointer = $"%rdt{_nextTemp++}";
                    _lines.Add(
                        $"    mov.u64 {dynamicFramePointer}, " +
                        "__callee_dynamic_frame;");
                    _lines.Add(
                        $"    add.u64 {dynamicFramePointer}, " +
                        $"{dynamicFramePointer}, 0x8000;");
                    _pointerRegs["R1"] = dynamicFramePointer;
                }
                else
                {
                    _pointerRegs["R1"] = "__callee_frame";
                }
            }
            Dictionary<string, string>? previousOverrides =
                _inlineLabelOverrides;
            _inlineLabelOverrides = overrides;
            _inlineCallStack.Add(target);
            string? callSkip = null;
            if (!string.IsNullOrWhiteSpace(call.Guard))
            {
                string callGuard = Guard(call.Guard);
                callSkip = $"L_g{_nextTemp++}";
                string inverted = callGuard.StartsWith("@!", StringComparison.Ordinal)
                    ? "@" + callGuard[2..]
                    : "@!" + callGuard[1..];
                _lines.Add($"    {inverted} bra {callSkip};");
            }
            try
            {
                for (int bodyIndex = 0; bodyIndex < body.Count; bodyIndex++)
                {
                    SassNode node = body[bodyIndex];
                    if (node is SassLabel label)
                    {
                        _lines.Add($"{overrides[label.Name]}:");
                        continue;
                    }
                    if (redundantSyncs.Contains(bodyIndex) ||
                        (node is SassInstruction sync &&
                         IsRedundantInlineSync(body, bodyIndex)))
                        continue;
                    string? previousInlineGuard = _extraGuard;
                    if (inlineGuards.TryGetValue(
                            bodyIndex, out string? inlineGuard))
                        _extraGuard = inlineGuard;
                    try
                    {
                        if (localPlan != null &&
                            localPlan.TryGetValue(
                                bodyIndex, out LocalPlanAction? action))
                        {
                            EmitLocalPlanAction(action);
                            continue;
                        }
                        SassInstruction instruction = (SassInstruction)node;
                        if (dynamicFramePointer != null &&
                            IsFrameAdjustment(instruction))
                        {
                            EmitDynamicFrameAdjustment(
                                instruction, dynamicFramePointer);
                            continue;
                        }
                        if (instruction.Mnemonic.StartsWith(
                                "RET", StringComparison.OrdinalIgnoreCase))
                            continue;
                        if (IsCallMnemonic(instruction.Mnemonic))
                        {
                            if (_extraGuard != null)
                                return false;
                            int nestedIndex = bodyGlobalIndices[bodyIndex];
                            string nestedTarget;
                            try
                            {
                                nestedTarget = CallTarget(
                                    nodes, nestedIndex, instruction);
                            }
                            catch (NotSupportedException)
                            {
                                return false;
                            }
                            string? tailLabel = null;
                            if (selfTailLabel != null &&
                                nestedTarget.Equals(
                                    selfTailLabel, StringComparison.Ordinal))
                                tailLabel = selfTailLabel;
                            else if (_inlineTargetLabels.TryGetValue(
                                         nestedTarget, out string? mapped))
                                tailLabel = mapped;
                            if (tailLabel != null &&
                                string.IsNullOrWhiteSpace(instruction.Guard) &&
                                IsTailReturn(body, bodyIndex))
                            {
                                _lines.Add($"bra {tailLabel};");
                                continue;
                            }
                            if (!TryEmitLeafCallInline(nodes, ref nestedIndex))
                                return false;
                            continue;
                        }
                        EmitInstruction(instruction);
                    }
                    finally
                    {
                        _extraGuard = previousInlineGuard;
                    }
                }
            }
            finally
            {
                _inlineCallStack.Remove(target);
                _inlineLabelOverrides = previousOverrides;
                if (hadTargetLabel && previousTargetLabel != null)
                    _inlineTargetLabels[target] = previousTargetLabel;
                else
                    _inlineTargetLabels.Remove(target);
                if (usesLocalFrame)
                {
                    if (hadR1 && previousR1 != null)
                        _pointerRegs["R1"] = previousR1;
                    else
                        _pointerRegs.Remove("R1");
                }
            }
            if (callSkip != null)
                _lines.Add($"{callSkip}:");
            for (int i = labelIndex; i < bodyEnd; i++)
                _skipNodes.Add(i);
            return true;
        }

        private bool TryEmitIfConvertedDiamond(
            List<SassNode> nodes, SortedSet<int> reachable,
            Dictionary<string, int> labelIndices, int bodyEnd)
        {
            var branches = reachable
                .Where(index =>
                    nodes[index] is SassInstruction instruction &&
                    instruction.Mnemonic.StartsWith(
                        "BRA", StringComparison.OrdinalIgnoreCase))
                .ToList();
            if (branches.Count != 1)
                return false;
            int branchIndex = branches[0];
            if (nodes[branchIndex] is not SassInstruction branch ||
                string.IsNullOrWhiteSpace(branch.Guard))
                return false;
            string branchTarget;
            try
            {
                branchTarget = RawBranchTarget(branch);
            }
            catch (NotSupportedException)
            {
                return false;
            }
            if (!TryResolveLabelIndex(
                    labelIndices, branchTarget, out int targetIndex) ||
                targetIndex <= branchIndex || targetIndex >= bodyEnd)
                return false;

            var prefix = reachable.Where(index => index < branchIndex).ToList();
            var elseArm = reachable
                .Where(index => index > branchIndex && index < targetIndex)
                .ToList();
            var thenArm = reachable
                .Where(index => index >= targetIndex)
                .ToList();
            if (!ValidateIfConvertedArm(nodes, elseArm, out _) ||
                !ValidateIfConvertedArm(nodes, thenArm, out _))
                return false;
            foreach (int nodeIndex in elseArm.Concat(thenArm))
            {
                if (nodes[nodeIndex] is SassInstruction instruction &&
                    !instruction.Mnemonic.StartsWith(
                        "RET", StringComparison.OrdinalIgnoreCase) &&
                    !IsIfConvertibleArmInstruction(instruction))
                    return false;
            }

            foreach (int nodeIndex in prefix)
            {
                if (nodes[nodeIndex] is SassInstruction instruction)
                    EmitInstruction(instruction);
            }
            string thenGuard = branch.Guard?.Trim() ?? "";
            if (!thenGuard.StartsWith("@", StringComparison.Ordinal))
                thenGuard = "@" + thenGuard;
            string elseGuard = NegateGuard(thenGuard);
            _extraGuard = elseGuard;
            try
            {
                foreach (int nodeIndex in elseArm)
                {
                    if (nodes[nodeIndex] is SassInstruction instruction &&
                        !instruction.Mnemonic.StartsWith(
                            "RET", StringComparison.OrdinalIgnoreCase))
                        EmitInstruction(instruction);
                }
            }
            finally
            {
                _extraGuard = null;
            }
            _extraGuard = thenGuard;
            try
            {
                foreach (int nodeIndex in thenArm)
                {
                    if (nodes[nodeIndex] is SassInstruction instruction &&
                        !instruction.Mnemonic.StartsWith(
                            "RET", StringComparison.OrdinalIgnoreCase))
                        EmitInstruction(instruction);
                }
            }
            finally
            {
                _extraGuard = null;
            }
            return true;
        }

        private bool TryEmitGeneralMultiReturnInline(
            List<SassNode> nodes, SortedSet<int> reachable)
        {
            var body = reachable.Select(index => nodes[index]).ToList();
            var overrides = new Dictionary<string, string>(StringComparer.Ordinal);
            string prefix = $"L_inline_{_inlineSerial++}_";
            foreach (SassNode node in body)
            {
                if (node is not SassLabel label)
                    continue;
                string unique = prefix + Sanitize(label.Name);
                while (_labels.Values.Contains(unique) ||
                       overrides.ContainsValue(unique))
                    unique += "_";
                if (!AddLabelOverride(overrides, label.Name, unique))
                    return false;
            }
            string exitLabel = prefix + "exit";
            while (_labels.Values.Contains(exitLabel) ||
                   overrides.ContainsValue(exitLabel))
                exitLabel += "_";
            int lastInstruction = reachable
                .Where(index => nodes[index] is SassInstruction)
                .DefaultIfEmpty(-1)
                .Max();

            Dictionary<string, string>? previousOverrides =
                _inlineLabelOverrides;
            _inlineLabelOverrides = overrides;
            try
            {
                foreach (int nodeIndex in reachable)
                {
                    SassNode node = nodes[nodeIndex];
                    if (node is SassLabel label)
                    {
                        _lines.Add($"{overrides[label.Name]}:");
                        continue;
                    }
                    SassInstruction instruction = (SassInstruction)node;
                    if (instruction.Mnemonic.StartsWith(
                            "RET", StringComparison.OrdinalIgnoreCase))
                    {
                        if (nodeIndex == lastInstruction)
                            continue;
                        string guard = Guard(instruction.Guard);
                        _lines.Add(
                            $"{(guard.Length == 0 ? "" : guard + " ")}" +
                            $"bra {exitLabel};");
                        continue;
                    }
                    EmitInstruction(instruction);
                }
                _lines.Add($"{exitLabel}:");
            }
            finally
            {
                _inlineLabelOverrides = previousOverrides;
            }
            return true;
        }

        private bool TryEmitIfConvertedGuardedReturn(
            List<SassNode> nodes, SortedSet<int> reachable,
            SortedSet<int> returns)
        {
            if (returns.Count != 2)
                return false;
            if (reachable.Any(index =>
                    nodes[index] is SassInstruction instruction &&
                    instruction.Mnemonic.StartsWith(
                        "BRA", StringComparison.OrdinalIgnoreCase)))
                return false;
            int guardedIndex = -1;
            int finalIndex = -1;
            foreach (int returnIndex in returns)
            {
                if (nodes[returnIndex] is not SassInstruction instruction)
                    return false;
                if (string.IsNullOrWhiteSpace(instruction.Guard))
                {
                    if (finalIndex >= 0)
                        return false;
                    finalIndex = returnIndex;
                }
                else
                {
                    if (guardedIndex >= 0)
                        return false;
                    guardedIndex = returnIndex;
                }
            }
            if (guardedIndex < 0 || finalIndex < 0 || guardedIndex >= finalIndex)
                return false;
            var rest = reachable
                .Where(index => index > guardedIndex && index < finalIndex)
                .ToList();
            foreach (int nodeIndex in rest)
            {
                if (nodes[nodeIndex] is SassInstruction instruction &&
                    !IsIfConvertibleArmInstruction(instruction))
                    return false;
            }

            foreach (int nodeIndex in reachable)
            {
                if (nodeIndex >= guardedIndex)
                    break;
                if (nodes[nodeIndex] is SassInstruction instruction)
                    EmitInstruction(instruction);
            }
            string guard =
                ((SassInstruction)nodes[guardedIndex]).Guard?.Trim() ?? "";
            if (!guard.StartsWith("@", StringComparison.Ordinal))
                guard = "@" + guard;
            _extraGuard = NegateGuard(guard);
            try
            {
                foreach (int nodeIndex in rest)
                {
                    if (nodes[nodeIndex] is SassInstruction instruction)
                        EmitInstruction(instruction);
                }
            }
            finally
            {
                _extraGuard = null;
            }
            return true;
        }

        private bool TryEmitIfConvertedSingleReturnDiamond(
            List<SassNode> nodes, SortedSet<int> reachable,
            Dictionary<string, int> labelIndices, int bodyEnd)
        {
            var branches = reachable
                .Where(index =>
                    nodes[index] is SassInstruction instruction &&
                    instruction.Mnemonic.StartsWith(
                        "BRA", StringComparison.OrdinalIgnoreCase))
                .ToList();
            if (branches.Count < 1 || branches.Count > 2)
                return false;
            var conditionalBranches = branches
                .Where(index =>
                    nodes[index] is SassInstruction instruction &&
                    !string.IsNullOrWhiteSpace(instruction.Guard))
                .ToList();
            if (conditionalBranches.Count != 1)
                return false;
            int branchIndex = conditionalBranches[0];
            if (nodes[branchIndex] is not SassInstruction branch)
                return false;
            string branchTarget;
            try
            {
                branchTarget = RawBranchTarget(branch);
            }
            catch (NotSupportedException)
            {
                return false;
            }
            if (!TryResolveLabelIndex(
                    labelIndices, branchTarget, out int targetIndex) ||
                targetIndex <= branchIndex || targetIndex >= bodyEnd)
                return false;

            int mergeIndex = targetIndex;
            var unconditionalBranches = branches
                .Where(index => index != branchIndex)
                .ToList();
            if (unconditionalBranches.Count > 1)
                return false;
            if (unconditionalBranches.Count == 1)
            {
                int unconditionalIndex = unconditionalBranches[0];
                if (nodes[unconditionalIndex] is not SassInstruction unconditional ||
                    !string.IsNullOrWhiteSpace(unconditional.Guard))
                    return false;
                string unconditionalTarget;
                try
                {
                    unconditionalTarget = RawBranchTarget(unconditional);
                }
                catch (NotSupportedException)
                {
                    return false;
                }
                if (!TryResolveLabelIndex(
                        labelIndices, unconditionalTarget, out int merge) ||
                    merge <= unconditionalIndex || merge >= bodyEnd ||
                    merge < targetIndex)
                    return false;
                mergeIndex = merge;
            }

            List<int> elseArm;
            List<int> thenArm;
            if (targetIndex < mergeIndex)
            {
                elseArm = reachable
                    .Where(index => index > branchIndex && index < targetIndex)
                    .ToList();
                thenArm = reachable
                    .Where(index => index >= targetIndex && index < mergeIndex)
                    .ToList();
            }
            else
            {
                elseArm = reachable
                    .Where(index => index > branchIndex && index < mergeIndex)
                    .ToList();
                thenArm = new List<int>();
            }
            var mergeBlock = reachable
                .Where(index => index >= mergeIndex)
                .ToList();
            if (!ValidateSingleReturnArm(
                    nodes, labelIndices, elseArm, mergeIndex,
                    targetIndex < mergeIndex) ||
                (thenArm.Count > 0 &&
                 !ValidateSingleReturnArm(
                     nodes, labelIndices, thenArm, mergeIndex, false)) ||
                !ValidateSingleReturnMerge(nodes, mergeBlock))
                return false;

            foreach (int nodeIndex in reachable)
            {
                if (nodeIndex >= branchIndex)
                    break;
                if (nodes[nodeIndex] is SassInstruction instruction)
                    EmitInstruction(instruction);
            }
            string thenGuard = branch.Guard?.Trim() ?? "";
            if (!thenGuard.StartsWith("@", StringComparison.Ordinal))
                thenGuard = "@" + thenGuard;
            string elseGuard = NegateGuard(thenGuard);
            _extraGuard = elseGuard;
            try
            {
                foreach (int nodeIndex in elseArm)
                {
                    if (nodes[nodeIndex] is SassInstruction instruction &&
                        !instruction.Mnemonic.StartsWith(
                            "BRA", StringComparison.OrdinalIgnoreCase))
                        EmitInstruction(instruction);
                }
            }
            finally
            {
                _extraGuard = null;
            }
            _extraGuard = thenGuard;
            try
            {
                foreach (int nodeIndex in thenArm)
                {
                    if (nodes[nodeIndex] is SassInstruction instruction &&
                        !instruction.Mnemonic.StartsWith(
                            "BRA", StringComparison.OrdinalIgnoreCase))
                        EmitInstruction(instruction);
                }
            }
            finally
            {
                _extraGuard = null;
            }
            foreach (int nodeIndex in mergeBlock)
            {
                if (nodes[nodeIndex] is SassInstruction instruction &&
                    !instruction.Mnemonic.StartsWith(
                        "RET", StringComparison.OrdinalIgnoreCase))
                    EmitInstruction(instruction);
            }
            return true;
        }

        private static bool ValidateSingleReturnArm(
            List<SassNode> nodes, Dictionary<string, int> labelIndices,
            List<int> arm, int mergeIndex, bool requireBranch)
        {
            bool sawBranch = false;
            foreach (int nodeIndex in arm)
            {
                if (nodes[nodeIndex] is not SassInstruction instruction)
                    continue;
                if (instruction.Mnemonic.StartsWith(
                        "BRA", StringComparison.OrdinalIgnoreCase))
                {
                    if (sawBranch ||
                        !string.IsNullOrWhiteSpace(instruction.Guard))
                        return false;
                    string target;
                    try
                    {
                        target = RawBranchTarget(instruction);
                    }
                    catch (NotSupportedException)
                    {
                        return false;
                    }
                    if (!TryResolveLabelIndex(
                            labelIndices, target, out int targetIndex) ||
                        targetIndex != mergeIndex)
                        return false;
                    sawBranch = true;
                    continue;
                }
                if (sawBranch ||
                    instruction.Mnemonic.StartsWith(
                        "RET", StringComparison.OrdinalIgnoreCase) ||
                    !IsIfConvertibleArmInstruction(instruction))
                    return false;
            }
            return !requireBranch || sawBranch;
        }

        private static bool ValidateSingleReturnMerge(
            List<SassNode> nodes, List<int> mergeBlock)
        {
            bool sawReturn = false;
            foreach (int nodeIndex in mergeBlock)
            {
                if (nodes[nodeIndex] is not SassInstruction instruction)
                    continue;
                if (sawReturn ||
                    instruction.Mnemonic.StartsWith(
                        "BRA", StringComparison.OrdinalIgnoreCase))
                    return false;
                if (instruction.Mnemonic.StartsWith(
                        "RET", StringComparison.OrdinalIgnoreCase))
                {
                    if (!string.IsNullOrWhiteSpace(instruction.Guard))
                        return false;
                    sawReturn = true;
                }
            }
            return sawReturn;
        }

        private static bool ValidateIfConvertedArm(
            List<SassNode> nodes, List<int> arm, out int returnIndex)
        {
            returnIndex = -1;
            foreach (int nodeIndex in arm)
            {
                if (nodes[nodeIndex] is not SassInstruction instruction)
                    continue;
                if (instruction.Mnemonic.StartsWith(
                        "RET", StringComparison.OrdinalIgnoreCase))
                {
                    if (returnIndex >= 0 ||
                        !string.IsNullOrWhiteSpace(instruction.Guard))
                        return false;
                    returnIndex = nodeIndex;
                    continue;
                }
                if (returnIndex >= 0)
                    return false;
            }
            return returnIndex >= 0;
        }

        private static bool IsIfConvertibleArmInstruction(
            SassInstruction instruction)
        {
            if (!string.IsNullOrWhiteSpace(instruction.Guard))
                return false;
            string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
            if (op is not ("MOV" or "MOV32I" or "IADD" or "IADD32I" or "IMAD"
                or "IMUL" or "IMUL32I" or "ISUB" or "LOP" or "LOP32I"
                or "SHL" or "SHR" or "BFE" or "BFI" or "PRMT" or "SEL"
                or "FADD" or "FADD32I" or "FMUL" or "FMUL32I" or "FFMA"
                or "MUFU" or "I2F" or "F2I" or "FLO" or "POPC" or "NOP"))
                return false;
            int destination = DestinationIndex(instruction);
            if (destination >= 0 &&
                destination < instruction.Operands.Count &&
                Regex.IsMatch(instruction.Operands[destination], @"^P\d+$",
                    RegexOptions.IgnoreCase))
                return false;
            return true;
        }

        private static bool AddLabelIndex(
            Dictionary<string, int> labels, string name, int index)
        {
            foreach (string alias in LabelAliases(name))
            {
                if (labels.TryGetValue(alias, out int existing) &&
                    existing != index)
                    return false;
                labels[alias] = index;
            }
            return true;
        }

        private static bool TryResolveLabelIndex(
            Dictionary<string, int> labels, string target, out int index)
        {
            if (labels.TryGetValue(target, out index))
                return true;
            string stripped = target.StartsWith(".L_", StringComparison.Ordinal)
                ? target[3..]
                : target;
            return labels.TryGetValue(stripped, out index) ||
                labels.TryGetValue(".L_" + stripped, out index);
        }

        private static IEnumerable<string> LabelAliases(string name)
        {
            yield return name;
            string stripped = name.StartsWith(".L_", StringComparison.Ordinal)
                ? name[3..]
                : name;
            yield return stripped;
            yield return ".L_" + stripped;
        }

        private static bool AddLabelOverride(
            Dictionary<string, string> overrides, string name, string unique)
        {
            string stripped = name.StartsWith(".L_", StringComparison.Ordinal)
                ? name[3..]
                : name;
            foreach (string alias in new[] { name, stripped, ".L_" + stripped })
            {
                if (overrides.TryGetValue(alias, out string? existing) &&
                    !existing.Equals(unique, StringComparison.Ordinal))
                    return false;
                overrides[alias] = unique;
            }
            return true;
        }

        private static string RawBranchTarget(SassInstruction instruction)
        {
            string target = instruction.BranchTargets.FirstOrDefault()
                ?? instruction.Operands.LastOrDefault()
                ?? throw new NotSupportedException("branch without a target");
            return target.Trim().Trim('`', '(', ')', ';');
        }

        private bool IsInlineSafeCallBody(
            SassInstruction instruction, bool allowInternalCarry = false)
        {
            string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
            bool localFrame = op is "LDL" or "STL" or "LD" or "ST"
                or "IADD" or "IADD32I" or "I2I" or "MOV" or "LEA";
            int localAddressOperand = op == "LDL" ? 1 : op == "STL" ? 0 : -1;
            if (localFrame && localAddressOperand >= 0 &&
                (localAddressOperand >= instruction.Operands.Count ||
                 !Regex.IsMatch(instruction.Operands[localAddressOperand],
                     @"^\[R1(?:\+(0x[0-9A-Fa-f]+|\d+))?\]$",
                     RegexOptions.IgnoreCase)))
                return false;
            // BSSY/BSYNC only bookkeep reconvergence; a single-return
            // leaf (e.g. NVCC's f64 sqrt mediumpath) can inline them as
            // no-ops because its branches are remapped to local labels.
            if (op is "CAL" or "JCAL" or "CALL" or "EXIT" or "RET" or "BRX" or "BPT"
                or "BAR" or "MEMBAR" or "DEPBAR" or "CCTL" or "TEXDEPBAR"
                or "KIL" or "LONGJMP" or "BRK" or "CONT")
                return false;
            if (!allowInternalCarry &&
                (instruction.Mnemonic.Contains(".CC", StringComparison.OrdinalIgnoreCase) ||
                 instruction.Mnemonic.Contains(".X", StringComparison.OrdinalIgnoreCase) ||
                 instruction.Operands.Any(operand =>
                     operand.Contains(".CC", StringComparison.OrdinalIgnoreCase))) &&
                (!_usesLocalFrame || op is not ("IADD" or "IADD32I" or "LEA")))
                return false;
            if (instruction.Mnemonic.StartsWith(
                    "IMAD", StringComparison.OrdinalIgnoreCase) &&
                (instruction.Mnemonic.Contains(".HI", StringComparison.OrdinalIgnoreCase) ||
                 instruction.Mnemonic.Contains(".WIDE", StringComparison.OrdinalIgnoreCase)))
                return false;
            foreach (string operand in instruction.Operands)
            {
                if (operand.Contains("c[0x", StringComparison.OrdinalIgnoreCase))
                    return false;
                if (Regex.IsMatch(operand, @"\bR1\b", RegexOptions.IgnoreCase) &&
                    (!localFrame ||
                     (localAddressOperand >= 0 &&
                      !operand.Equals(
                          instruction.Operands[localAddressOperand],
                          StringComparison.OrdinalIgnoreCase))))
                    return false;
            }
            return true;
        }

        private void BuildInlineGuardPlan(
            List<SassNode> body, Dictionary<int, string> guards,
            HashSet<int> redundantSyncs)
        {
            for (int i = 0; i < body.Count; i++)
            {
                if (body[i] is not SassInstruction guardedSync ||
                    string.IsNullOrWhiteSpace(guardedSync.Guard))
                    continue;
                string op = guardedSync.Mnemonic.Split('.')[0].ToUpperInvariant();
                if (op is not "SYNC" and not "PBK" and not "PCNT")
                    continue;
                string target;
                try
                {
                    target = BranchLabel(guardedSync);
                }
                catch (NotSupportedException)
                {
                    continue;
                }
                int endSync = -1;
                for (int j = i + 1; j < body.Count; j++)
                {
                    if (body[j] is not SassInstruction candidate ||
                        !string.IsNullOrWhiteSpace(candidate.Guard) ||
                        !TargetMatches(candidate, target))
                        continue;
                    string candidateOp =
                        candidate.Mnemonic.Split('.')[0].ToUpperInvariant();
                    if (candidateOp is "SYNC" or "PBK" or "PCNT")
                    {
                        endSync = j;
                        break;
                    }
                }
                if (endSync < 0 || endSync + 1 >= body.Count ||
                    body[endSync + 1] is not SassLabel label ||
                    !LabelName(label.Name).Equals(target, StringComparison.Ordinal))
                    continue;
                bool safe = true;
                for (int j = i + 1; j < endSync; j++)
                {
                    if (body[j] is SassLabel)
                    {
                        safe = false;
                        break;
                    }
                    if (body[j] is not SassInstruction instruction)
                        continue;
                    string regionOp =
                        instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
                    if (!string.IsNullOrWhiteSpace(instruction.Guard) ||
                        IsControlFlow(instruction) ||
                        IsBarrier(instruction) ||
                        regionOp is "LDG" or "LD" or "LDS" or "LDL")
                    {
                        safe = false;
                        break;
                    }
                }
                if (!safe)
                    continue;
                string executeGuard = NegateGuard(guardedSync.Guard);
                for (int j = i + 1; j < endSync; j++)
                {
                    if (body[j] is SassInstruction)
                        guards[j] = executeGuard;
                }
                redundantSyncs.Add(i);
                redundantSyncs.Add(endSync);
            }
        }

        private bool IsRedundantInlineSync(List<SassNode> body, int index)
        {
            if (body[index] is not SassInstruction instruction)
                return false;
            string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
            if (op is not "SYNC" and not "PBK" and not "PCNT")
                return false;
            string target;
            try
            {
                target = BranchLabel(instruction);
            }
            catch (NotSupportedException)
            {
                return false;
            }
            for (int i = index + 1; i < body.Count; i++)
            {
                if (body[i] is SassInstruction nop &&
                    nop.Mnemonic.StartsWith(
                        "NOP", StringComparison.OrdinalIgnoreCase))
                    continue;
                return body[i] is SassLabel label &&
                    LabelName(label.Name).Equals(target, StringComparison.Ordinal);
            }
            return false;
        }

        private enum LocalPlanKind
        {
            Skip,
            Load,
            Store,
        }

        private sealed record LocalPlanAction(
            LocalPlanKind Kind, int Offset, bool Wide,
            SassInstruction Instruction);

        private bool TryBuildLocalAddressPlan(
            List<SassNode> body, out Dictionary<int, LocalPlanAction> plan)
        {
            plan = new Dictionary<int, LocalPlanAction>();
            var low = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)
            {
                ["R1"] = 0,
            };
            var zero = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < body.Count; i++)
            {
                if (body[i] is SassLabel)
                    continue;
                if (body[i] is not SassInstruction instruction)
                    return false;
                string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
                if (op == "NOP")
                    continue;
                string mnemonic = instruction.Mnemonic.ToUpperInvariant();
                bool wide = mnemonic.Contains(".64", StringComparison.Ordinal);
                if (op is "LDL" or "STL" or "LD" or "ST")
                {
                    bool store = op is "STL" or "ST";
                    int addressIndex = store ? 0 : 1;
                    if (addressIndex >= instruction.Operands.Count)
                        return false;
                    bool pair = op is "LD" or "ST";
                    if (TryGetLocalAddress(
                            instruction.Operands[addressIndex],
                            low, zero, pair, out int offset))
                    {
                        plan[i] = new LocalPlanAction(
                            store ? LocalPlanKind.Store : LocalPlanKind.Load,
                            offset, wide, instruction);
                        if (!store &&
                            instruction.Operands.Count > 0)
                        {
                            ClearLocalRegister(
                                low, zero, BaseReg(instruction.Operands[0]), wide);
                        }
                        else if (store &&
                                 instruction.Operands.Count > 1)
                        {
                            if (ReadsLocalRegister(
                                    instruction.Operands[1], low, zero))
                                return false;
                        }
                        continue;
                    }
                    if (op is "LDL" or "STL")
                        return false;
                }

                if (op is "MOV")
                {
                    if (instruction.Operands.Count < 2)
                        return false;
                    string destination = BaseReg(instruction.Operands[0]);
                    string source = BaseReg(instruction.Operands[1]);
                    if (source.Equals("RZ", StringComparison.OrdinalIgnoreCase))
                    {
                        low.Remove(destination);
                        zero.Add(destination);
                        plan[i] = new LocalPlanAction(
                            LocalPlanKind.Skip, 0, false, instruction);
                    }
                    else if (low.TryGetValue(source, out int sourceOffset))
                    {
                        low[destination] = sourceOffset;
                        zero.Remove(destination);
                        plan[i] = new LocalPlanAction(
                            LocalPlanKind.Skip, 0, false, instruction);
                    }
                    else if (zero.Contains(source))
                    {
                        zero.Add(destination);
                        low.Remove(destination);
                        plan[i] = new LocalPlanAction(
                            LocalPlanKind.Skip, 0, false, instruction);
                    }
                    else
                    {
                        ClearLocalRegister(low, zero, destination, false);
                    }
                    continue;
                }

                if (op == "I2I")
                {
                    if (!mnemonic.Contains(".U32.U32", StringComparison.Ordinal) ||
                        instruction.Operands.Count < 2)
                        return false;
                    string destination = BaseReg(instruction.Operands[0]);
                    string source = BaseReg(instruction.Operands[1]);
                    if (low.TryGetValue(source, out int sourceOffset))
                    {
                        low[destination] = sourceOffset;
                        zero.Remove(destination);
                        plan[i] = new LocalPlanAction(
                            LocalPlanKind.Skip, 0, false, instruction);
                    }
                    else if (zero.Contains(source))
                    {
                        zero.Add(destination);
                        low.Remove(destination);
                        plan[i] = new LocalPlanAction(
                            LocalPlanKind.Skip, 0, false, instruction);
                    }
                    else
                    {
                        ClearLocalRegister(low, zero, destination, false);
                    }
                    continue;
                }

                if (op is "IADD" or "IADD32I")
                {
                    if (instruction.Operands.Count < 3)
                        return false;
                    if (mnemonic.Contains(".CC", StringComparison.Ordinal) ||
                        mnemonic.Contains(".X", StringComparison.Ordinal) ||
                        instruction.Operands.Any(operand =>
                            operand.Contains(".CC", StringComparison.OrdinalIgnoreCase)))
                    {
                        if (!(mnemonic.Contains(".CC", StringComparison.Ordinal) ||
                              instruction.Operands.Any(operand =>
                                  operand.Contains(
                                      ".CC", StringComparison.OrdinalIgnoreCase))) ||
                            i + 1 >= body.Count ||
                            body[i + 1] is not SassInstruction high ||
                            !high.Mnemonic.StartsWith(
                                "IADD.X", StringComparison.OrdinalIgnoreCase) ||
                            high.Operands.Count < 3 ||
                            instruction.Operands.Count < 3)
                            return false;
                        string lowDestination =
                            BaseReg(instruction.Operands[0]);
                        string lowSource = BaseReg(instruction.Operands[1]);
                        string highDestination = BaseReg(high.Operands[0]);
                        string highSource = BaseReg(high.Operands[1]);
                        if (!low.TryGetValue(lowSource, out int lowOffset) ||
                            !zero.Contains(highSource) ||
                            !highDestination.Equals(
                                NextRegister(lowDestination),
                                StringComparison.OrdinalIgnoreCase) ||
                            !TrySwitchImmediate(
                                instruction.Operands[2], out int addend))
                            return false;
                        low.Clear();
                        zero.Clear();
                        low[lowDestination] = lowOffset + addend;
                        zero.Add(highDestination);
                        plan[i] = new LocalPlanAction(
                            LocalPlanKind.Skip, 0, false, instruction);
                        plan[i + 1] = new LocalPlanAction(
                            LocalPlanKind.Skip, 0, false, high);
                        i++;
                        continue;
                    }
                    string destination = BaseReg(instruction.Operands[0]);
                    string left = BaseReg(instruction.Operands[1]);
                    string right = BaseReg(instruction.Operands[2]);
                    if (op == "IADD32I")
                    {
                        if (low.TryGetValue(left, out int sourceOffset) &&
                            TrySwitchImmediate(right, out int immediate))
                        {
                            low[destination] = sourceOffset + immediate;
                            zero.Remove(destination);
                            plan[i] = new LocalPlanAction(
                                LocalPlanKind.Skip, 0, false, instruction);
                        }
                        else if (ReadsLocalRegister(left, low, zero))
                        {
                            return false;
                        }
                        else
                        {
                            ClearLocalRegister(low, zero, destination, false);
                        }
                        continue;
                    }
                    if (IsRz(left) && low.TryGetValue(right, out int rightOffset))
                    {
                        low[destination] = rightOffset;
                        zero.Remove(destination);
                        plan[i] = new LocalPlanAction(
                            LocalPlanKind.Skip, 0, false, instruction);
                    }
                    else if (IsRz(right) &&
                             low.TryGetValue(left, out int leftOffset))
                    {
                        low[destination] = leftOffset;
                        zero.Remove(destination);
                        plan[i] = new LocalPlanAction(
                            LocalPlanKind.Skip, 0, false, instruction);
                    }
                    else if (ReadsLocalRegister(left, low, zero) ||
                             ReadsLocalRegister(right, low, zero))
                    {
                        return false;
                    }
                    else
                    {
                        ClearLocalRegister(low, zero, destination, false);
                    }
                    continue;
                }

                int destinationIndex = DestinationIndex(instruction);
                for (int operand = 0;
                     operand < instruction.Operands.Count; operand++)
                {
                    if (operand == destinationIndex)
                        continue;
                    if (ReadsLocalRegister(
                            instruction.Operands[operand], low, zero))
                        return false;
                }
                if (destinationIndex >= 0 &&
                    destinationIndex < instruction.Operands.Count)
                {
                    ClearLocalRegister(
                        low, zero, BaseReg(instruction.Operands[destinationIndex]),
                        false);
                }
            }
            return true;
        }

        private static void ClearLocalRegister(
            Dictionary<string, int> low, HashSet<string> zero,
            string register, bool wide)
        {
            low.Remove(register);
            zero.Remove(register);
            if (!wide)
                return;
            string? next = NextRegister(register);
            if (next == null)
                return;
            low.Remove(next);
            zero.Remove(next);
        }

        private static bool ReadsLocalRegister(
            string operand, Dictionary<string, int> low, HashSet<string> zero)
        {
            foreach (Match match in Regex.Matches(
                operand, @"\bR(\d+)\b", RegexOptions.IgnoreCase))
            {
                string register = "R" + match.Groups[1].Value;
                if (low.ContainsKey(register) || zero.Contains(register))
                    return true;
            }
            return false;
        }

        private static bool TryGetLocalAddress(
            string operand, Dictionary<string, int> low, HashSet<string> zero,
            bool requireHighZero, out int offset)
        {
            offset = 0;
            Match match = Regex.Match(operand,
                @"^\[(R\d+)(?:\+(0x[0-9A-Fa-f]+|\d+))?\]$",
                RegexOptions.IgnoreCase);
            if (!match.Success)
                return false;
            string register = match.Groups[1].Value;
            if (!low.TryGetValue(register, out int baseOffset))
                return false;
            if (requireHighZero)
            {
                string? high = NextRegister(register);
                if (high == null || !zero.Contains(high))
                    return false;
            }
            int immediate = 0;
            if (match.Groups[2].Success)
            {
                string text = match.Groups[2].Value;
                if (!(text.StartsWith("0x", StringComparison.OrdinalIgnoreCase)
                        ? int.TryParse(text[2..], NumberStyles.HexNumber,
                            CultureInfo.InvariantCulture, out immediate)
                        : int.TryParse(text, NumberStyles.Integer,
                            CultureInfo.InvariantCulture, out immediate)))
                    return false;
            }
            offset = baseOffset + immediate;
            return true;
        }

        private bool TryValidateDynamicLocalFrame(List<SassNode> body)
        {
            foreach (SassNode node in body)
            {
                if (node is not SassInstruction instruction)
                    continue;
                string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
                if (op is "LDL" or "STL")
                {
                    int addressIndex = op == "LDL" ? 1 : 0;
                    if (addressIndex >= instruction.Operands.Count ||
                        !Regex.IsMatch(instruction.Operands[addressIndex],
                            @"^\[R1(?:\+(0x[0-9A-Fa-f]+|\d+))?\]$",
                            RegexOptions.IgnoreCase))
                        return false;
                    continue;
                }
                bool touchesR1 = instruction.Operands.Any(operand =>
                    Regex.IsMatch(operand, @"\bR1\b",
                        RegexOptions.IgnoreCase));
                if (!touchesR1)
                    continue;
                if (op is not ("IADD" or "IADD32I") ||
                    instruction.Mnemonic.Contains(
                        ".CC", StringComparison.OrdinalIgnoreCase) ||
                    instruction.Mnemonic.Contains(
                        ".X", StringComparison.OrdinalIgnoreCase) ||
                    instruction.Operands.Count < 3 ||
                    !BaseReg(instruction.Operands[0]).Equals(
                        "R1", StringComparison.OrdinalIgnoreCase) ||
                    !ContainsRegister(instruction.Operands[1], "R1") ||
                    ContainsRegister(instruction.Operands[2], "R1"))
                    return false;
                string source = instruction.Operands[2].Trim();
                if (IsRz(source) || TryFrameImmediate(source, out _))
                    continue;
                string register = source.TrimStart('-');
                if (!Regex.IsMatch(register, @"^R\d+$",
                        RegexOptions.IgnoreCase))
                    return false;
            }
            return true;
        }

        private static bool TryFrameImmediate(string operand, out long value)
        {
            string text = operand.Trim();
            bool negative = text.StartsWith('-');
            if (negative)
                text = text[1..];
            bool parsed;
            if (text.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
            {
                parsed = ulong.TryParse(text[2..], NumberStyles.HexNumber,
                    CultureInfo.InvariantCulture, out ulong hex);
                value = hex <= int.MaxValue
                    ? (long)hex
                    : hex <= uint.MaxValue
                        ? (int)(uint)hex
                        : (long)hex;
            }
            else
            {
                parsed = long.TryParse(text, NumberStyles.Integer,
                    CultureInfo.InvariantCulture, out value);
            }
            if (!parsed)
                return false;
            if (negative)
                value = -value;
            return true;
        }

        private void EmitDynamicFrameAdjustment(
            SassInstruction instruction, string pointer)
        {
            string source = instruction.Operands[2].Trim();
            if (IsRz(source))
                return;
            if (TryFrameImmediate(source, out long immediate))
            {
                if (immediate == 0)
                    return;
                string op = immediate < 0 ? "sub" : "add";
                ulong magnitude = immediate < 0
                    ? (ulong)(-(immediate + 1)) + 1
                    : (ulong)immediate;
                _lines.Add(
                    $"{op}.u64 {pointer}, {pointer}, 0x{magnitude:x};");
                return;
            }
            bool negative = source.StartsWith('-');
            string register = BaseReg(source.TrimStart('-'));
            string converted = $"%rdt{_nextTemp++}";
            _lines.Add($"cvt.u64.s32 {converted}, {Reg(register)};");
            _lines.Add(
                $"{(negative ? "sub" : "add")}.u64 {pointer}, {pointer}, " +
                $"{converted};");
        }

        private void EmitLocalPlanAction(LocalPlanAction action)
        {
            if (action.Kind == LocalPlanKind.Skip)
                return;
            SassInstruction instruction = action.Instruction;
            string address = $"[__callee_frame+0x{action.Offset:x}]";
            if (action.Kind == LocalPlanKind.Load)
            {
                string destination = action.Wide
                    ? VectorRegs(instruction.Operands[0], 2)
                    : Reg(instruction.Operands[0]);
                _lines.Add(
                    $"{(action.Wide ? "ld.local.v2.u32" : "ld.local.u32")} " +
                    $"{destination}, {address};");
                return;
            }
            string source = action.Wide
                ? VectorRegs(instruction.Operands[1], 2)
                : Operand(instruction.Operands[1]);
            _lines.Add(
                $"{(action.Wide ? "st.local.v2.u32" : "st.local.u32")} " +
                $"{address}, {source};");
        }

        private bool TryEmitEmptyCall(List<SassNode> nodes, ref int index)
        {
            if (nodes[index] is not SassInstruction call ||
                !IsCallMnemonic(call.Mnemonic))
                return false;
            string target;
            try
            {
                target = CallTarget(nodes, index, call);
            }
            catch (NotSupportedException)
            {
                return false;
            }
            int labelIndex = -1;
            for (int i = index + 1; i < nodes.Count; i++)
            {
                if (nodes[i] is SassLabel label &&
                    LabelName(label.Name).Equals(target, StringComparison.Ordinal))
                {
                    labelIndex = i;
                    break;
                }
            }
            if (labelIndex < 0)
                return false;
            int returnIndex = -1;
            for (int i = labelIndex + 1; i < nodes.Count; i++)
            {
                if (nodes[i] is SassLabel)
                    continue;
                if (nodes[i] is not SassInstruction body)
                    return false;
                if (body.Mnemonic.StartsWith(
                        "NOP", StringComparison.OrdinalIgnoreCase))
                    continue;
                if (body.Mnemonic.StartsWith(
                        "RET", StringComparison.OrdinalIgnoreCase) &&
                    string.IsNullOrWhiteSpace(body.Guard))
                {
                    returnIndex = i;
                    break;
                }
                return false;
            }
            if (returnIndex < 0)
                return false;
            for (int i = labelIndex; i <= returnIndex; i++)
                _skipNodes.Add(i);
            return true;
        }

        private bool TryEmitSwitchSelect(List<SassNode> nodes, ref int index)
        {
            if (nodes[index] is not SassInstruction first ||
                !first.Mnemonic.StartsWith("ISETP", StringComparison.OrdinalIgnoreCase))
                return false;
            int mergeIndex = -1;
            for (int i = index + 1; i < nodes.Count; i++)
            {
                if (nodes[i] is SassInstruction exit &&
                    (exit.Mnemonic.StartsWith("EXIT", StringComparison.OrdinalIgnoreCase) ||
                     exit.Mnemonic.StartsWith("RET", StringComparison.OrdinalIgnoreCase)))
                {
                    mergeIndex = i;
                    break;
                }
            }
            if (mergeIndex < 0)
                return false;

            var labels = new Dictionary<string, int>(StringComparer.Ordinal);
            for (int i = 0; i < nodes.Count; i++)
            {
                if (nodes[i] is SassLabel label)
                    labels[LabelName(label.Name)] = i;
            }

            string? selector = null;
            if (first.Operands.Count >= 4)
            {
                string a = first.Operands[2];
                string b = first.Operands[3];
                if (IsSwitchImmediate(b) &&
                    Regex.IsMatch(BaseReg(a), @"^R\d+$", RegexOptions.IgnoreCase))
                    selector = BaseReg(a);
                else if (IsSwitchImmediate(a) &&
                    Regex.IsMatch(BaseReg(b), @"^R\d+$", RegexOptions.IgnoreCase))
                    selector = BaseReg(b);
            }
            if (selector == null)
                return false;

            int maxValue = -1;
            for (int i = index; i < mergeIndex; i++)
            {
                if (nodes[i] is not SassInstruction instruction ||
                    !instruction.Mnemonic.StartsWith(
                        "ISETP", StringComparison.OrdinalIgnoreCase))
                    continue;
                for (int operand = 2;
                     operand < 4 && operand < instruction.Operands.Count; operand++)
                {
                    if (TrySwitchImmediate(instruction.Operands[operand],
                            out int value) && value > maxValue)
                        maxValue = value;
                }
            }
            if (maxValue < 0 || maxValue > 31)
                return false;

            var results = new string[maxValue + 2];
            string? store = null;
            for (int value = 0; value <= maxValue + 1; value++)
            {
                if (!SimulateSwitch(nodes, labels, index, mergeIndex,
                        selector, value, out string? constant,
                        out string? storeOperand) ||
                    constant == null || storeOperand == null)
                    return false;
                if (store == null)
                    store = storeOperand;
                else if (!store.Equals(storeOperand, StringComparison.OrdinalIgnoreCase))
                    return false;
                results[value] = constant;
            }

            string defaultConstant = results[maxValue + 1];
            string selectorOperand = Reg(selector);
            string saved = $"%rt{_nextTemp++}";
            _lines.Add($"    mov.b32 {saved}, {selectorOperand};");
            string result = $"%rt{_nextTemp++}";
            _lines.Add($"    mov.u32 {result}, {defaultConstant};");
            for (int value = 0; value <= maxValue; value++)
            {
                if (results[value] == defaultConstant)
                    continue;
                string predicate = $"%pt{_nextTempPred++}";
                _lines.Add($"setp.eq.u32 {predicate}, {saved}, {value};");
                _lines.Add($"@{predicate} mov.u32 {result}, {results[value]};");
            }
            string address = Address(store!);
            _lines.Add($"    st.global.u32 {address}, {result};");
            EmitReturn("");
            index = mergeIndex;
            return true;
        }

        private bool SimulateSwitch(
            List<SassNode> nodes, Dictionary<string, int> labels,
            int start, int merge, string selector, int value,
            out string? constant, out string? store)
        {
            constant = null;
            store = null;
            var predicates = new Dictionary<string, bool>(StringComparer.Ordinal);
            int pc = start;
            int steps = 0;
            while (pc < merge && steps++ < 1000)
            {
                if (nodes[pc] is SassLabel)
                {
                    pc++;
                    continue;
                }
                if (nodes[pc] is not SassInstruction instruction)
                    return false;
                string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
                if (op == "ISETP")
                {
                    if (!EvaluateSwitchIsetp(instruction, selector, value, predicates))
                        return false;
                    pc++;
                    continue;
                }
                if (op == "BRA")
                {
                    bool take = true;
                    if (!string.IsNullOrWhiteSpace(instruction.Guard))
                    {
                        string guard = instruction.Guard.Trim();
                        bool negated = guard.StartsWith("@!", StringComparison.Ordinal);
                        string predicate = Pred(guard.TrimStart('@', '!'));
                        if (!predicates.TryGetValue(predicate, out bool predicateValue))
                            return false;
                        take = negated ? !predicateValue : predicateValue;
                    }
                    if (take)
                    {
                        string target = BranchLabel(instruction);
                        if (!labels.TryGetValue(target, out int targetIndex))
                            return false;
                        pc = targetIndex;
                    }
                    else
                    {
                        pc++;
                    }
                    continue;
                }
                if (op is "MOV" or "MOV32I")
                {
                    if (instruction.Operands.Count >= 2 &&
                        BaseReg(instruction.Operands[0]).Equals(
                            selector, StringComparison.OrdinalIgnoreCase))
                    {
                        if (BaseReg(instruction.Operands[1]).Equals(
                                selector, StringComparison.OrdinalIgnoreCase))
                        {
                            pc++;
                            continue;
                        }
                        if (!TrySwitchImmediate(
                                instruction.Operands[1], out int resultValue))
                            return false;
                        if (constant != null)
                            return false;
                        constant = resultValue.ToString(
                            CultureInfo.InvariantCulture);
                    }
                    pc++;
                    continue;
                }
                if (op is "STG" or "ST")
                {
                    if (instruction.Operands.Count < 1)
                        return false;
                    store ??= instruction.Operands[0];
                    pc++;
                    continue;
                }
                if (op is "EXIT" or "RET")
                    break;
                if (op == "NOP")
                {
                    pc++;
                    continue;
                }
                return false;
            }
            return constant != null && store != null;
        }

        private bool EvaluateSwitchIsetp(
            SassInstruction instruction, string selector, int value,
            Dictionary<string, bool> predicates)
        {
            if (instruction.Operands.Count < 5)
                return false;
            string guard = instruction.Guard?.Trim() ?? "";
            if (!string.IsNullOrEmpty(guard) &&
                !guard.Equals("PT", StringComparison.OrdinalIgnoreCase))
                return false;
            string sourcePredicate = instruction.Operands[4].Trim();
            if (!sourcePredicate.Equals("PT", StringComparison.OrdinalIgnoreCase))
                return false;
            string suffix = instruction.Mnemonic.ToUpperInvariant();
            if (!suffix.Contains(".AND", StringComparison.Ordinal))
                return false;
            string a = instruction.Operands[2];
            string b = instruction.Operands[3];
            bool selectorIsA = BaseReg(a).Equals(
                selector, StringComparison.OrdinalIgnoreCase);
            bool selectorIsB = BaseReg(b).Equals(
                selector, StringComparison.OrdinalIgnoreCase);
            int immediate;
            if (selectorIsA)
            {
                if (!TrySwitchImmediate(b, out immediate))
                    return false;
            }
            else if (selectorIsB)
            {
                if (!TrySwitchImmediate(a, out immediate))
                    return false;
            }
            else
            {
                return false;
            }

            int left = selectorIsA ? value : immediate;
            int right = selectorIsA ? immediate : value;
            bool result = suffix.Contains(".GT", StringComparison.Ordinal) ? left > right
                : suffix.Contains(".GE", StringComparison.Ordinal) ? left >= right
                : suffix.Contains(".LT", StringComparison.Ordinal) ? left < right
                : suffix.Contains(".LE", StringComparison.Ordinal) ? left <= right
                : suffix.Contains(".EQ", StringComparison.Ordinal) ? left == right
                : suffix.Contains(".NE", StringComparison.Ordinal) ? left != right
                : false;
            predicates[Pred(instruction.Operands[0])] = result;
            return true;
        }

        private static bool IsSwitchImmediate(string operand) =>
            TrySwitchImmediate(operand, out _);

        private static bool TrySwitchImmediate(string operand, out int value)
        {
            string text = operand.Trim();
            if (text.Equals("RZ", StringComparison.OrdinalIgnoreCase))
            {
                value = 0;
                return true;
            }
            bool negative = text.StartsWith("-", StringComparison.Ordinal);
            string body = negative ? text[1..] : text;
            bool ok;
            if (body.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
                ok = int.TryParse(body[2..], NumberStyles.HexNumber,
                    CultureInfo.InvariantCulture, out value);
            else
                ok = int.TryParse(body, NumberStyles.Integer,
                    CultureInfo.InvariantCulture, out value);
            if (ok && negative)
                value = -value;
            return ok;
        }

        private bool TryEmitRedundantBranch(List<SassNode> nodes, ref int index)
        {
            if (nodes[index] is not SassInstruction branch ||
                !branch.Mnemonic.Equals("BRA", StringComparison.OrdinalIgnoreCase))
                return false;
            string target;
            try
            {
                target = BranchLabel(branch);
            }
            catch (NotSupportedException)
            {
                return false;
            }
            int next = index + 1;
            while (next < nodes.Count &&
                   nodes[next] is SassInstruction nop &&
                   nop.Mnemonic.StartsWith(
                       "NOP", StringComparison.OrdinalIgnoreCase))
                next++;
            return next < nodes.Count &&
                nodes[next] is SassLabel label &&
                LabelName(label.Name).Equals(target, StringComparison.Ordinal);
        }

        private void EmitReturn(string prefix)
        {
            if (_useCommonExit)
                _lines.Add($"{prefix}bra {_commonExitLabel};");
            else
                _lines.Add($"{prefix}ret;");
        }

        // The out-of-line f64 division helper returns through a register
        // holding the caller's constant return address.  Calls branch to
        // the single body and record (address -> label); the body's RET
        // compares and branches back.  This keeps one copy of the ~150
        // instruction helper even when a fully unrolled kernel calls it
        // dozens of times.
        private void EmitHelperDispatchReturn(SassInstruction instruction)
        {
            string returnRegister = instruction.Operands.Count > 0
                ? Reg(instruction.Operands[0])
                : "%r0";
            foreach ((uint address, string label) in
                     _helperDispatchReturns.OrderBy(pair => pair.Key))
            {
                string predicate = $"%pt{_nextTempPred++}";
                _lines.Add($"    setp.eq.u32 {predicate}, " +
                    $"{returnRegister}, 0x{address:x};");
                _lines.Add($"    @{predicate} bra {label};");
            }
            EmitReturn("");
        }

        private void EmitInstruction(SassInstruction instruction)
        {
            SassInstruction? previousInstruction = _currentInstruction;
            _currentInstruction = instruction;
            // Snapshot only the pair-cache entries a write can touch, not
            // the whole cache: copying it per instruction is quadratic in
            // the unrolled reduce kernels (10k instructions x thousands of
            // cached address pairs) and made them take minutes.
            HashSet<string> writtenKeys =
                new(StringComparer.OrdinalIgnoreCase);
            foreach (string raw in WrittenRegisters(instruction))
            {
                string key = BaseReg(raw);
                Match number = Regex.Match(key, @"^R(\d+)$",
                    RegexOptions.IgnoreCase);
                if (!number.Success)
                    continue;
                writtenKeys.Add(key);
                int index = int.Parse(number.Groups[1].Value,
                    CultureInfo.InvariantCulture);
                if (index > 0)
                    writtenKeys.Add("R" + (index - 1).ToString(
                        CultureInfo.InvariantCulture));
            }
            Dictionary<string, string?> before =
                new(StringComparer.OrdinalIgnoreCase);
            foreach (string key in writtenKeys)
                before[key] = _pairRegs.TryGetValue(key, out string? value)
                    ? value : null;
            // A predicated instruction that writes a predicate cannot use
            // ACuda's predicated-value select (the skip arm must keep the
            // physical predicate register, and loop-carried predicates are
            // not visible as a prior definition).  Emit an explicit skip
            // branch instead.
            if (_extraGuard == null && !string.IsNullOrWhiteSpace(instruction.Guard) &&
                instruction.Operands.Count > 0 &&
                Regex.IsMatch(instruction.Operands[0].Trim(),
                    @"^!?U?P\d+$|^!?U?PT$", RegexOptions.IgnoreCase) &&
                instruction.Mnemonic.Split('.')[0].ToUpperInvariant()
                    is "ISETP" or "FSETP" or "DSETP" or "SETP" or
                       "PLOP3" or "LOP3" or "ULOP3")
            {
                string rawGuard = Guard(instruction.Guard);
                string skip = $"L_g{_nextTemp++}";
                string inverted = rawGuard.StartsWith("@!", StringComparison.Ordinal)
                    ? "@" + rawGuard[2..]
                    : "@!" + rawGuard[1..];
                _lines.Add($"    {inverted} bra {skip};");
                _suppressGuard = true;
                EmitInstructionCore(instruction);
                _suppressGuard = false;
                _lines.Add($"{skip}:");
            }
            else
            {
                EmitInstructionCore(instruction);
            }
            foreach (string key in writtenKeys)
            {
                bool changed = before.TryGetValue(key, out string? oldValue) &&
                        oldValue != null
                    ? !_pairRegs.TryGetValue(key, out string? newValue) ||
                      !string.Equals(oldValue, newValue, StringComparison.Ordinal)
                    : _pairRegs.ContainsKey(key);
                if (!changed)
                    _pairRegs.Remove(key);
            }
            if (_predSnapshotDefs.TryGetValue(instruction,
                    out var predicateDefs))
                foreach (var predicateDef in predicateDefs)
                    _lines.Add($"    mov.pred {predicateDef.Snapshot}, " +
                        $"{Pred(predicateDef.Predicate)};");
            _currentInstruction = previousInstruction;
        }

        private void EmitInstructionCore(SassInstruction instruction)
        {
            if (_dispatchReturnInstructions.Contains(instruction))
            {
                EmitHelperDispatchReturn(instruction);
                return;
            }
            string op = instruction.Mnemonic.Split('.')[0].ToUpperInvariant();
            if (_usesLocalFrame && _inlineLabelOverrides == null &&
                _topLevelLocalAccessSeen &&
                op is not "LDL" and not "STL" &&
                WrittenRegisters(instruction).Contains(
                    Reg("R1"), StringComparer.OrdinalIgnoreCase))
                _topLevelFrameBaseMoved = true;
            string resolvedGuard =
                _guardSnapshot.TryGetValue(instruction, out string? snap)
                    ? snap : Guard(instruction.Guard);
            string guard = _suppressGuard
                ? ""
                : _extraGuard == null
                    ? resolvedGuard
                    : CombineGuard(resolvedGuard, _extraGuard);
            string Prefix() => guard.Length == 0 ? "" : guard + " ";
            switch (op)
            {
                case "MOV":
                    EmitMov(instruction, Prefix());
                    return;
                case "MOV32I":
                    _lines.Add($"{Prefix()}mov.u32 {Reg(instruction.Operands[0])}, " +
                        $"{Immediate(instruction.Operands[1])};");
                    return;
                case "I2I":
                    EmitI2I(instruction, Prefix());
                    return;
                case "IADD":
                case "IADD32I":
                    EmitIadd(instruction, Prefix());
                    return;
                case "ISCADD":
                    EmitIscadd(instruction, Prefix());
                    return;
                case "IADD3":
                    EmitIadd3(instruction, Prefix());
                    return;
                case "UIADD3":
                    if (instruction.Mnemonic.Contains(
                            ".64", StringComparison.OrdinalIgnoreCase))
                        EmitUadd3Wide(instruction, Prefix());
                    else
                        EmitUadd3(instruction, Prefix());
                    return;
                case "UIMAD":
                    EmitUimad(instruction, Prefix());
                    return;
                case "USHF":
                    EmitUshf(instruction, Prefix());
                    return;
                case "CALL":
                case "CAL":
                case "JCAL":
                    EmitExternalHelperCall(instruction, Prefix());
                    return;
                case "ISUB":
                case "SUB":
                    EmitSimpleArithmetic(
                        instruction, Prefix(), "sub", signedDefault: true);
                    return;
                case "ADD":
                    EmitSimpleArithmetic(
                        instruction, Prefix(), "add", signedDefault: true);
                    return;
                case "MUL":
                    EmitSimpleArithmetic(
                        instruction, Prefix(), "mul.lo", signedDefault: true);
                    return;
                case "MAD":
                    EmitSimpleArithmetic(
                        instruction, Prefix(), "mad.lo", signedDefault: true);
                    return;
                case "MUL24":
                    EmitSimpleArithmetic(
                        instruction, Prefix(), "mul24.lo", signedDefault: true);
                    return;
                case "MAD24":
                    EmitSimpleArithmetic(
                        instruction, Prefix(), "mad24.lo", signedDefault: true);
                    return;
                case "MIN":
                    EmitSimpleArithmetic(
                        instruction, Prefix(), "min", signedDefault: true);
                    return;
                case "MAX":
                    EmitSimpleArithmetic(
                        instruction, Prefix(), "max", signedDefault: true);
                    return;
                case "CLZ":
                    EmitClz(instruction, Prefix());
                    return;
                case "BFIND":
                    EmitBfind(instruction, Prefix());
                    return;
                case "FNS":
                    EmitFns(instruction, Prefix());
                    return;
                case "BREV":
                    EmitBrev(instruction, Prefix());
                    return;
                case "CLMAD":
                    EmitClmad(instruction, Prefix());
                    return;
                case "FMA":
                    EmitSimpleArithmetic(
                        instruction, Prefix(), "fma.rn", signedDefault: false);
                    return;
                case "RCP":
                case "SQRT":
                case "RSQRT":
                case "SIN":
                case "COS":
                case "LG2":
                case "EX2":
                case "TANH":
                    EmitFloatUnary(instruction, Prefix());
                    return;
                case "TESTP":
                    EmitTestp(instruction, Prefix());
                    return;
                case "COPYSIGN":
                    EmitCopysign(instruction, Prefix());
                    return;
                case "IMAD":
                case "XMAD":
                    EmitMad(instruction, Prefix());
                    return;
                case "IMUL":
                    if (instruction.Mnemonic.Contains(
                            "WIDE", StringComparison.OrdinalIgnoreCase))
                        EmitImulWide(instruction, Prefix());
                    else
                        _lines.Add($"{Prefix()}mul.lo.s32 " +
                            $"{Reg(instruction.Operands[0])}, " +
                            $"{Reg(instruction.Operands[1])}, " +
                            $"{Operand(instruction.Operands[2])};");
                    return;
                case "IMUL32I":
                    EmitImul32i(instruction, Prefix());
                    return;
                case "FADD":
                    _lines.Add($"{Prefix()}add.f32 {Reg(instruction.Operands[0])}, " +
                        $"{FloatOperand(instruction.Operands[1])}, " +
                        $"{FloatOperand(instruction.Operands[2])};");
                    return;
                case "FADD32I":
                case "FMUL32I":
                    EmitFloatImmediate(instruction, Prefix());
                    return;
                case "DADD":
                    EmitDadd(instruction, Prefix());
                    return;
                case "DMUL":
                    EmitDmul(instruction, Prefix());
                    return;
                case "DFMA":
                    EmitDfma(instruction, Prefix());
                    return;
                case "FMUL":
                    _lines.Add($"{Prefix()}mul.f32 {Reg(instruction.Operands[0])}, " +
                        $"{FloatOperand(instruction.Operands[1])}, " +
                        $"{FloatOperand(instruction.Operands[2])};");
                    return;
                case "FMNMX":
                    EmitFmnmx(instruction, Prefix());
                    return;
                case "MUFU":
                    EmitMufu(instruction, Prefix());
                    return;
                case "HMMA":
                case "IMMA":
                    EmitMma(instruction, Prefix());
                    return;
                case "IDP":
                    EmitIdp(instruction, Prefix());
                    return;
                case "DMMA":
                    EmitDmma(instruction, Prefix());
                    return;
                case "RRO":
                    EmitRro(instruction, Prefix());
                    return;
                case "FFMA":
                    EmitFfma(instruction, Prefix());
                    return;
                case "LDG":
                case "LD":
                case "LDS":
                case "LDU":
                    EmitLoad(instruction, Prefix());
                    return;
                case "TLD":
                case "TLDS":
                    EmitTextureLoad(instruction, Prefix());
                    return;
                case "LDL":
                    EmitLocalLoad(instruction, Prefix());
                    return;
                case "LDSLK":
                    EmitLdslk(instruction, Prefix());
                    return;
                case "STSCUL":
                    EmitStscul(instruction, Prefix());
                    return;
                case "STG":
                case "ST":
                case "STS":
                    if (instruction.Mnemonic.Contains(
                            "ASYNC", StringComparison.OrdinalIgnoreCase))
                        EmitAsyncStore(instruction, Prefix());
                    else if (instruction.Mnemonic.Contains(
                                 "BULK", StringComparison.OrdinalIgnoreCase))
                        EmitBulkStore(instruction, Prefix());
                    else
                        EmitStore(instruction, Prefix());
                    return;
                case "STL":
                    EmitLocalStore(instruction, Prefix());
                    return;
                case "SULD":
                    EmitSurfaceLoad(instruction, Prefix());
                    return;
                case "SUST":
                    EmitSurfaceStore(instruction, Prefix());
                    return;
                case "SURED":
                    EmitSurfaceReduce(instruction, Prefix());
                    return;
                case "SUQ":
                    EmitSurfaceQuery(instruction, Prefix());
                    return;
                case "R2P":
                    EmitR2P(instruction, Prefix());
                    return;
                case "P2R":
                    EmitP2R(instruction, Prefix());
                    return;
                case "S2R":
                    EmitS2R(instruction, Prefix());
                    return;
                case "S2UR":
                    EmitS2UR(instruction, Prefix());
                    return;
                case "UISETP":
                    EmitUisetp(instruction, Prefix());
                    return;
                case "SHL":
                case "SHR":
                    EmitShift(instruction, Prefix());
                    return;
                case "SHF":
                    EmitShf(instruction, Prefix());
                    return;
                case "LEA":
                    if (instruction.Mnemonic.StartsWith("LEA.HI.X",
                            StringComparison.OrdinalIgnoreCase))
                        EmitLeaHiX(instruction, Prefix());
                    else if (instruction.Mnemonic.StartsWith("LEA.HI",
                                 StringComparison.OrdinalIgnoreCase))
                        EmitLeaHi(instruction, Prefix());
                    else
                        EmitLea(instruction, Prefix());
                    return;
                case "ULEA":
                    EmitUlea(instruction, Prefix());
                    return;
                case "ULDC":
                    EmitUldc(instruction, Prefix());
                    return;
                case "LDC":
                    EmitLdc(instruction, Prefix());
                    return;
                case "F2FP":
                    EmitF2fp(instruction, Prefix());
                    return;
                case "FSEL":
                    EmitFsel(instruction, Prefix());
                    return;
                case "HFMA2":
                    EmitHfma2(instruction, Prefix());
                    return;
                case "HADD2":
                case "HMUL2":
                    EmitHadd2(instruction, Prefix());
                    return;
                case "CS2R":
                    EmitCs2r(instruction, Prefix());
                    return;
                case "UMOV":
                    _lines.Add($"{Prefix()}mov.u32 " +
                        $"{UniformReg(instruction.Operands[0])}, " +
                        $"{Operand(instruction.Operands[1])};");
                    return;
                case "F2F":
                    EmitF2f(instruction, Prefix());
                    return;
                case "FRND":
                    EmitFrnd(instruction, Prefix());
                    return;
                case "IMNMX":
                    EmitImnmx(instruction, Prefix());
                    return;
                case "BMOV":
                    // Barrier-register bookkeeping (e.g.
                    // BMOV.32.CLEAR RZ, B0); BSSY/BSYNC already carry the
                    // reconvergence information the lift needs.
                    return;
                case "BSSY":
                    _ = BranchLabel(instruction);
                    return;
                case "WARPSYNC":
                    _lines.Add($"{Prefix()}bar.warp.sync 0xffffffff;");
                    return;
                case "BSYNC":
                    return;
                case "POPC":
                    EmitPopc(instruction, Prefix());
                    return;
                case "BFE":
                    EmitBfe(instruction, Prefix());
                    return;
                case "IABS":
                case "ABS":
                    EmitIabs(instruction, Prefix());
                    return;
                case "NEG":
                    EmitNeg(instruction, Prefix());
                    return;
                case "SAD":
                    EmitSad(instruction, Prefix());
                    return;
                case "DIV":
                    EmitDivRem(instruction, Prefix(), remainder: false);
                    return;
                case "REM":
                    EmitDivRem(instruction, Prefix(), remainder: true);
                    return;
                case "VABSDIFF":
                    EmitVabsdiff(instruction, Prefix());
                    return;
                case "VABSDIFF4":
                    EmitVabsdiff4(instruction, Prefix());
                    return;
                case "BMSK":
                    EmitBmsk(instruction, Prefix());
                    return;
                case "UBMSK":
                    EmitUbmsk(instruction, Prefix());
                    return;
                case "SGXT":
                    EmitSgxt(instruction, Prefix());
                    return;
                case "BFI":
                    EmitBfi(instruction, Prefix());
                    return;
                case "PRMT":
                    if (instruction.Mnemonic.Contains(
                            ".B32", StringComparison.OrdinalIgnoreCase))
                        EmitPrmtAlias(instruction, Prefix());
                    else
                        EmitPrmt(instruction, Prefix());
                    return;
                case "FLO":
                    EmitFlo(instruction, Prefix());
                    return;
                case "SHFL":
                    if (instruction.Mnemonic.Contains(
                            "SYNC", StringComparison.OrdinalIgnoreCase))
                        EmitShflSync(instruction, Prefix());
                    else
                        EmitShfl(instruction, Prefix());
                    return;
                case "VOTE":
                    EmitVote(instruction, Prefix());
                    return;
                case "REDUX":
                    EmitRedux(instruction, Prefix());
                    return;
                case "R2UR":
                    EmitR2ur(instruction, Prefix());
                    return;
                case "VOTEU":
                    EmitVoteU(instruction, Prefix());
                    return;
                case "ATOM":
                case "ATOMG":
                case "ATOMS":
                    EmitAtom(instruction, Prefix());
                    return;
                case "AND":
                    EmitBitwise(instruction, Prefix(), "and");
                    return;
                case "OR":
                    EmitBitwise(instruction, Prefix(), "or");
                    return;
                case "XOR":
                    EmitBitwise(instruction, Prefix(), "xor");
                    return;
                case "NOT":
                    EmitNot(instruction, Prefix());
                    return;
                case "CNOT":
                    EmitCnot(instruction, Prefix());
                    return;
                case "LOP":
                case "LOP32I":
                case "LOP3":
                    EmitLop(instruction, Prefix());
                    return;
                case "ULOP3":
                    EmitUlop3(instruction, Prefix());
                    return;
                case "ISETP":
                    EmitIsetp(instruction, Prefix());
                    return;
                case "FCHK":
                    // Divide range check: the fast path (MUFU.RCP plus
                    // FFMA refinement) is exact for ordinary operands.
                    // Clear the predicate so that path is taken; the
                    // slow exceptional path remains unsupported.
                    _lines.Add($"{Prefix()}mov.pred " +
                        $"{Pred(instruction.Operands[0])}, 0;");
                    return;
                case "FSETP":
                    EmitFsetp(instruction, Prefix());
                    return;
                case "DSETP":
                    EmitDsetp(instruction, Prefix());
                    return;
                case "FSET":
                    EmitFset(instruction, Prefix());
                    return;
                case "SET":
                    EmitSet(instruction, Prefix());
                    return;
                case "SETP":
                    EmitSetp(instruction, Prefix());
                    return;
                case "SELP":
                    EmitSelp(instruction, Prefix());
                    return;
                case "SLCT":
                    EmitSlct(instruction, Prefix());
                    return;
                case "ISET":
                    EmitIset(instruction, Prefix());
                    return;
                case "ICMP":
                    EmitIcmp(instruction, Prefix());
                    return;
                case "PSETP":
                    EmitPsetp(instruction, Prefix());
                    return;
                case "PLOP3":
                    EmitPlop3(instruction, Prefix());
                    return;
                case "SEL":
                    EmitSel(instruction, Prefix());
                    return;
                case "CVT":
                    EmitCvtAlias(instruction, Prefix());
                    return;
                case "MULTIMEM":
                    EmitMultimem(instruction, Prefix());
                    return;
                case "PREFETCH":
                case "PREFETCHU":
                    EmitPrefetch(instruction, Prefix());
                    return;
                case "ISSPACEP":
                    EmitIsspacep(instruction, Prefix());
                    return;
                case "CVTA":
                    EmitCvta(instruction, Prefix());
                    return;
                case "CP":
                    EmitCpAsync(instruction, Prefix());
                    return;
                case "TENSORMAP":
                    EmitTensormap(instruction, Prefix());
                    return;
                case "SPCOMPRESS":
                case "SPDECOMPRESS":
                    EmitNativeInstruction(instruction, Prefix());
                    return;
                case "I2F":
                case "I2FP":
                    EmitI2f(instruction, Prefix());
                    return;
                case "F2I":
                    EmitF2i(instruction, Prefix());
                    return;
                case "BAR":
                    // A predicated BAR.SYNC is ptxas's lowering of a
                    // uniformly-controlled barrier: both arms of the
                    // diamond contain a barrier, so at most one executes
                    // per block.  A CTA barrier must stay converged, and
                    // an extra unconditional rendezvous is harmless, so
                    // drop the guard instead of emitting an illegal
                    // predicated barrier.
                    _lines.Add("bar.sync 0;");
                    return;
                case "MEMBAR":
                    _lines.Add($"{Prefix()}membar.gl;");
                    return;
                case "ERRBAR":
                    // Error barrier: order prior shared/CTA accesses.
                    _lines.Add($"{Prefix()}membar.cta;");
                    return;
                case "CCTL":
                    EmitCctl(instruction, Prefix());
                    return;
                case "DEPBAR":
                    _lines.Add($"{Prefix()}membar.cta;");
                    return;
                case "TEXDEPBAR":
                    // Texture dependency barriers have no PTX equivalent.
                    // Texture instructions are outside the lifted subset, so
                    // this is a hint with no observable effect here.
                    return;
                case "BRA":
                {
                    string target = BranchLabel(instruction);
                    if (!_operandSnapshot.TryGetValue(instruction,
                            out string? branchPred) ||
                        string.IsNullOrEmpty(branchPred))
                    {
                        _lines.Add($"{Prefix()}bra {target};");
                        return;
                    }
                    // `@P0 BRA P1, target`: the branch needs both the
                    // instruction guard and the predicate operand.
                    if (guard.Length == 0)
                    {
                        _lines.Add($"{branchPred} bra {target};");
                        return;
                    }
                    string a = PredicateValue(guard);
                    string b = PredicateValue(branchPred);
                    string combined = $"%pt{_nextTempPred++}";
                    _lines.Add($"    and.pred {combined}, {a}, {b};");
                    _lines.Add($"@{combined} bra {target};");
                    return;
                }
                case "SSY":
                    // SSY pushes a reconvergence point.  The following
                    // guarded/unconditional SYNC carries the actual edge, so
                    // validate the target and omit the stack bookkeeping.
                    _ = BranchLabel(instruction);
                    return;
                case "SYNC":
                {
                    // A SYNC carrying a convergence token (CC15.TRUE /
                    // CC15.FALSE) is a warp reconvergence barrier, not a
                    // branch.  A SYNC with a real label is an early-exit
                    // edge from an SSY-protected region.
                    string syncTarget =
                        instruction.BranchTargets.FirstOrDefault()
                        ?? instruction.Operands.LastOrDefault() ?? "";
                    syncTarget = syncTarget.Trim()
                        .Trim('`', '(', ')', ';');
                    if (syncTarget.StartsWith(
                            "CC", StringComparison.OrdinalIgnoreCase))
                        return;
                    _lines.Add($"{Prefix()}bra {BranchLabel(instruction)};");
                    return;
                }
                case "PBK":
                case "PCNT":
                    // Pop-branch/pop-continue transfer control to the target
                    // recorded by the matching loop construct.
                    _lines.Add($"{Prefix()}bra {BranchLabel(instruction)};");
                    return;
                case "EXIT":
                case "RET":
                    // SASS can attach a second predicate to the exit:
                    // `@P0 EXIT P1` only exits when both hold.  Dropping
                    // the operand made every thread exit before the output
                    // write in PyTorch's reduce epilogue.
                    if (op == "EXIT" && instruction.Operands.Count > 0 &&
                        Regex.IsMatch(instruction.Operands[^1].Trim(),
                            @"^!?(?:P\d+|PT)$", RegexOptions.IgnoreCase))
                    {
                        string exitText = instruction.Operands[^1].Trim();
                        string? guardText = string.IsNullOrWhiteSpace(
                            instruction.Guard)
                            ? null
                            : instruction.Guard.Trim().TrimStart('@');
                        if (exitText.Equals("PT",
                                StringComparison.OrdinalIgnoreCase))
                        {
                            EmitReturn(Prefix());
                            return;
                        }
                        string exitPredicate = PredicateOperand(exitText);
                        if (guardText == null)
                        {
                            EmitReturn($"@{exitPredicate} ");
                            return;
                        }
                        string guardPredicate =
                            PredicateOperand(guardText);
                        string combined = $"%pt{_nextTempPred++}";
                        _lines.Add($"    and.pred {combined}, " +
                            $"{guardPredicate}, {exitPredicate};");
                        EmitReturn($"@{combined} ");
                        return;
                    }
                    EmitReturn(Prefix());
                    return;
                case "NOP":
                    // SASS padding NOPs have no PTX equivalent; omit them.
                    return;
                case "YIELD":
                    // Scheduler hint only.
                    return;
                case "BPT":
                case "BREAK":
                case "BRK":
                    // CUDA lowers brkpt/assert traps to BPT/BREAK/BRK.
                    _lines.Add($"{Prefix()}brkpt;");
                    return;
                default:
                    throw new NotSupportedException(
                        $"SASS instruction '{instruction.Mnemonic}' is not supported by the PTX lifter");
            }
        }

        private void EmitMov(SassInstruction instruction, string prefix)
        {
            string destination = instruction.Operands[0];
            string source = instruction.Operands[1];
            if (IsRz(destination))
                return;
            if (instruction.Mnemonic.Contains(
                    "64", StringComparison.OrdinalIgnoreCase))
            {
                string destinationBase = BaseReg(destination);
                string value = $"%rdt{_nextTemp++}";
                _lines.Add($"{prefix}mov.b64 {value}, " +
                    $"{PackRegisterPair(source)};");
                UnpackPair64(value, destinationBase,
                    NextRegister(destinationBase));
                return;
            }
            if (destination.Trim().StartsWith(
                    "P", StringComparison.OrdinalIgnoreCase))
            {
                _lines.Add($"{prefix}mov.pred {Pred(destination)}, " +
                    $"{Pred(source)};");
                return;
            }
            if (TryParamLoad(destination, source, prefix,
                    out string? paramLine))
            {
                if (paramLine != null)
                    _lines.Add(prefix + paramLine);
                return;
            }
            if (!IsRz(destination) &&
                Regex.IsMatch(source.Trim(), @"^R\d+$", RegexOptions.IgnoreCase) &&
                _pointerRegs.TryGetValue(
                    BaseReg(source), out string? pointer))
                _pointerRegs[BaseReg(destination)] = pointer;
            _lines.Add($"{prefix}mov.b32 {Reg(destination)}, {Operand(source)};");
        }

        private bool TryParamLoad(
            string destination, string source, string prefix,
            out string? line)
        {
            line = null;
            Match constant = Regex.Match(source,
                @"^c\[0x0\]\[0x([0-9A-Fa-f]+)\]$");
            if (!constant.Success)
                return false;
            int offset = int.Parse(constant.Groups[1].Value,
                NumberStyles.HexNumber, CultureInfo.InvariantCulture);
            if (!_paramsByOffset.TryGetValue(offset, out PtxParameter? parameter))
                return false;
            string reg = Reg(destination);
            if (parameter.Type == "u64")
            {
                int index = ParameterIndex(parameter);
                bool valueLoad = _paramValueLoads.Contains(BaseReg(destination));
                if (valueLoad)
                {
                    if (offset != parameter.Offset &&
                        offset != parameter.Offset + 4)
                    {
                        throw new NotSupportedException(
                            $"constant bank offset 0x{offset:x} is not a parameter word");
                    }
                    string low = $"%ru{_nextTempU32++}";
                    string high = $"%ru{_nextTempU32++}";
                    // Each qword of a by-value struct has its own
                    // loaded pair (%rdpN, %rdpNp1, ...).  Reading the
                    // base pair for a later word silently returned the
                    // wrong field.
                    bool wideStruct = parameter.Part > 0 ||
                        WideWords(_parameters[index]) > 0;
                    bool largeStruct =
                        WideWords(_parameters[index]) > 16;
                    if (largeStruct && parameter.Part > 0)
                    {
                        // Large structs live in a storage buffer and
                        // only word 0 has a %rdp register; read the
                        // requested word by its byte offset.
                        int pbase = parameter.Part * 8 +
                            (offset - parameter.Offset);
                        string direct = $"%rt{_nextTemp++}";
                        _lines.Add($"{prefix}ld.param.u32 {direct}, " +
                            $"[{parameter.Name}+{pbase}];");
                        line = $"mov.b32 {reg}, {direct};";
                        return true;
                    }
                    string indexText = parameter.Part == 0
                        ? $"%rdp{index}"
                        : $"%rdp{index}p{parameter.Part}";
                    _lines.Add(
                        $"{prefix}mov.b64 {{{low}, {high}}}, " +
                        $"{indexText};");
                    if (offset == parameter.Offset)
                    {
                        // A word of a scalar by-value struct is a
                        // plain value; treating it as a pointer made
                        // integer address arithmetic ambiguous.
                        if (!wideStruct)
                            _pointerRegs[destination] =
                                ParamPointer(parameter);
                        line = $"mov.b32 {reg}, {low};";
                    }
                    else
                    {
                        line = $"mov.b32 {reg}, {high};";
                    }
                }
                else
                {
                    if (offset != parameter.Offset &&
                        offset != parameter.Offset + 4)
                    {
                        throw new NotSupportedException(
                            $"constant bank offset 0x{offset:x} is not a parameter word");
                    }
                    // This word was not classified as a value load
                    // because it is overwritten in another block before
                    // being read on this path.  It may still be read on
                    // a path that skips that overwrite (the reduce
                    // kernel initialises its accumulators from a
                    // constant-bank word before a divergent branch).
                    // Always materialise the 32-bit word; leaving the
                    // register undefined poisons the reduction.
                    bool largeStruct = WideWords(_parameters[index]) > 16;
                    if (largeStruct && parameter.Part > 0)
                    {
                        int pbase = parameter.Part * 8 +
                            (offset - parameter.Offset);
                        string direct = $"%rt{_nextTemp++}";
                        _lines.Add($"{prefix}ld.param.u32 {direct}, " +
                            $"[{parameter.Name}+{pbase}];");
                        line = $"mov.b32 {reg}, {direct};";
                        return true;
                    }
                    string low = $"%ru{_nextTempU32++}";
                    string high = $"%ru{_nextTempU32++}";
                    string indexText = parameter.Part == 0
                        ? $"%rdp{index}"
                        : $"%rdp{index}p{parameter.Part}";
                    _lines.Add(
                        $"{prefix}mov.b64 {{{low}, {high}}}, " +
                        $"{indexText};");
                    if (offset == parameter.Offset)
                    {
                        _pointerRegs[destination] = ParamPointer(parameter);
                        line = $"mov.b32 {reg}, {low};";
                    }
                    else
                    {
                        line = $"mov.b32 {reg}, {high};";
                    }
                }
            }
            else
            {
                line = $"mov.u32 {reg}, %rpp{ParameterIndex(parameter)};";
            }
            return true;
        }

        private void EmitI2I(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            bool saturate = mnemonic.Contains(
                ".SAT", StringComparison.OrdinalIgnoreCase);
            string[] parts = mnemonic.Split('.');
            string? destinationType = null;
            string? sourceType = null;
            foreach (string part in parts)
            {
                if (part is "U8" or "S8" or "U16" or "S16" or "U32" or "S32")
                {
                    if (destinationType == null)
                        destinationType = part;
                    else if (sourceType == null)
                        sourceType = part;
                }
            }
            if (destinationType == null)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            string destination = Reg(instruction.Operands[0]);
            string source = Operand(instruction.Operands[1]);
            if (destinationType is "U32" or "S32")
            {
                if (sourceType != null &&
                    sourceType != "U32" &&
                    sourceType != "S32")
                    throw new NotSupportedException(
                        $"SASS instruction '{instruction.Mnemonic}' is not supported");
                _lines.Add($"{prefix}mov.b32 {destination}, {source};");
                return;
            }
            string sourceTypeName = (sourceType ?? "S32").ToLowerInvariant();
            string conversion = $"cvt.{(saturate ? "sat." : "")}" +
                $"{destinationType.ToLowerInvariant()}.{sourceTypeName}";
            _lines.Add($"{prefix}{conversion} {destination}, {source};");
        }

        private void EmitIadd(SassInstruction instruction, string prefix)
        {
            string destinationText = instruction.Operands[0].Trim();
            bool carryOut = destinationText.EndsWith(".CC",
                StringComparison.OrdinalIgnoreCase);
            string destination = Reg(carryOut
                ? destinationText[..^3] : destinationText);
            string left = Operand(instruction.Operands[1]);
            string right = Operand(instruction.Operands[2]);
            if (instruction.Mnemonic.Contains(".X",
                    StringComparison.OrdinalIgnoreCase))
            {
                // IADD.X Rd, Ra, Rb, carry: add the carry flag left by the
                // preceding IADD.CC.
                _lines.Add($"{prefix}addc.u32 {destination}, {left}, " +
                    $"{right};");
                return;
            }
            if (carryOut)
            {
                _lines.Add($"{prefix}add.cc.u32 {destination}, {left}, " +
                    $"{right};");
                return;
            }
            _lines.Add($"{prefix}add.s32 {destination}, {left}, {right};");
        }

        // Lower SASS TLD/TLDS to a real PTX texture fetch.  The cubin
        // relocation table maps the descriptor immediate (0x67, 0x68, ...)
        // to the original texref symbol; ACuda's module-scope .texref path
        // then binds it through cuModuleGetTexRef at launch.
        //
        // The legacy buffer-backed texture path in ACuda only implements
        // tex.2d.v4, so 1D fetches are expressed as 2D with a zero row.
        private void EmitTextureLoad(SassInstruction instruction,
                                     string prefix)
        {
            string op = instruction.Mnemonic.Split('.')[0]
                .ToUpperInvariant();
            bool isTlds = op == "TLDS";
            bool vector = instruction.Mnemonic.Contains(
                ".128", StringComparison.OrdinalIgnoreCase);
            if (!vector && instruction.Operands.Count > 0)
            {
                string mode = instruction.Operands[^1].Trim();
                if (mode.Equals("RGBA", StringComparison.OrdinalIgnoreCase))
                    vector = true;
                else if (TrySwitchImmediate(mode, out int mask))
                    vector = (mask & 0xf) == 0xf;
            }

            // TLDS carries an auxiliary destination first; its data
            // destination is always the second operand.  TLD's data
            // destination is the first operand.
            int destinationIndex = isTlds ? 1 : 0;
            if (instruction.Operands.Count <= destinationIndex)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' has no " +
                    "texture destination");
            if (!TryTextureDescriptor(
                    instruction, out int descriptor, out int geometryIndex))
                throw new NotSupportedException(
                    $"cannot parse texture descriptor for " +
                    $"'{instruction.Mnemonic}'");
            if (!_texrefs.TryGetValue(descriptor, out string? texref))
                throw new NotSupportedException(
                    $"texture descriptor 0x{descriptor:x} in " +
                    $"'{instruction.Mnemonic}' has no .texref relocation");

            // The pre-Volta decoder drops the TLD component mask, so a
            // four-component float fetch would otherwise be mistaken for
            // a scalar integer load.  PhysX's float4 textures are the
            // sorted positions/velocities.
            if (!vector && !isTlds &&
                (texref.Contains(
                     "Velocities", StringComparison.OrdinalIgnoreCase) ||
                 texref.Contains(
                     "Positions", StringComparison.OrdinalIgnoreCase)))
                vector = true;

            string geometry = instruction.Operands[geometryIndex].Trim()
                .ToLowerInvariant();
            if (geometry != "1d")
                throw new NotSupportedException(
                    $"texture geometry '{geometry}' is not supported");

            int coordinateIndex = isTlds ? 2 : 1;
            if (instruction.Operands.Count <= coordinateIndex)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' has no " +
                    "texture coordinate");
            string coordinate = Operand(instruction.Operands[coordinateIndex]);

            _usedTexrefs.Add(texref);
            string zero = $"%rt{_nextTemp++}";
            _lines.Add($"{prefix}mov.u32 {zero}, 0;");

            string destinationOperand =
                instruction.Operands[destinationIndex];
            if (destinationOperand.Trim().Equals(
                    "RZ", StringComparison.OrdinalIgnoreCase))
            {
                string sink0 = $"%rt{_nextTemp++}";
                string sink1 = $"%rt{_nextTemp++}";
                string sink2 = $"%rt{_nextTemp++}";
                string sink3 = $"%rt{_nextTemp++}";
                _lines.Add($"{prefix}tex.2d.v4.u32.u32 " +
                    $"{{{sink0}, {sink1}, {sink2}, {sink3}}}, " +
                    $"[{texref}, {{{coordinate}, {zero}}}];");
                return;
            }

            if (vector)
            {
                string registers = VectorRegs(destinationOperand, 4);
                _lines.Add($"{prefix}tex.2d.v4.f32.u32 {registers}, " +
                    $"[{texref}, {{{coordinate}, {zero}}}];");
                return;
            }

            string destination = Reg(destinationOperand);
            string extra0 = $"%rt{_nextTemp++}";
            string extra1 = $"%rt{_nextTemp++}";
            string extra2 = $"%rt{_nextTemp++}";
            _lines.Add($"{prefix}tex.2d.v4.u32.u32 " +
                $"{{{destination}, {extra0}, {extra1}, {extra2}}}, " +
                $"[{texref}, {{{coordinate}, {zero}}}];");
        }

        // ISCADD Rd, Ra, Rb, shift  =>  Rd = (Ra << shift) + Rb.
        private void EmitIscadd(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string destination = Reg(instruction.Operands[0]);
            string left = Operand(instruction.Operands[1]);
            string right = Operand(instruction.Operands[2]);
            uint shift = ParseImmediate(instruction.Operands[3]) & 0x1Fu;
            if (shift == 0)
            {
                _lines.Add($"{prefix}add.s32 {destination}, {left}, {right};");
                return;
            }
            string shifted = $"%rt{_nextTemp++}";
            _lines.Add($"{prefix}shl.b32 {shifted}, {left}, {shift};");
            _lines.Add($"{prefix}add.s32 {destination}, {shifted}, {right};");
        }


        private string UniformReg(string operand)
        {
            Match match = Regex.Match(operand.Trim(), @"^UR(\d+)$",
                RegexOptions.IgnoreCase);
            if (!match.Success)
                throw new NotSupportedException(
                    $"expected uniform register, got '{operand}'");
            return "%ur" + match.Groups[1].Value;
        }

        private string UniformOperand(string operand)
        {
            string text = StripReuse(operand);
            bool invert = false;
            if (text.StartsWith("~", StringComparison.Ordinal))
            {
                invert = true;
                text = text[1..].TrimStart();
            }
            bool negate = false;
            if (text.StartsWith("-", StringComparison.Ordinal))
            {
                negate = true;
                text = text[1..].TrimStart();
            }
            else if (text.StartsWith("+", StringComparison.Ordinal))
            {
                text = text[1..].TrimStart();
            }
            if (text.Equals("URZ", StringComparison.OrdinalIgnoreCase))
                return invert ? "0xffffffff" : "0";
            Match reg = Regex.Match(text, @"^UR(\d+)$",
                RegexOptions.IgnoreCase);
            if (reg.Success)
            {
                string uniform = "%ur" + reg.Groups[1].Value;
                if (negate)
                {
                    // UIADD3 can negate a uniform source.  PTX add.u32 does
                    // not accept a unary minus on a register, so materialize
                    // the two's-complement negation into a temporary.
                    string temp = $"%rt{_nextTemp++}";
                    _lines.Add($"    sub.u32 {temp}, 0, {uniform};");
                    uniform = temp;
                }
                if (!invert)
                    return uniform;
                string inverted = $"%rt{_nextTemp++}";
                _lines.Add($"    not.b32 {inverted}, {uniform};");
                return inverted;
            }
            uint magnitude = ParseImmediate(text);
            if (negate)
                magnitude = unchecked(0u - magnitude);
            if (invert)
                magnitude = ~magnitude;
            return $"0x{magnitude:x}";
        }

        private static int LeaShift(string text)
        {
            string trimmed = text.Trim();
            if (trimmed.Equals("RZ", StringComparison.OrdinalIgnoreCase) ||
                trimmed.Equals("URZ", StringComparison.OrdinalIgnoreCase))
                return 0;
            int value;
            if (trimmed.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
                value = int.Parse(trimmed[2..], NumberStyles.HexNumber,
                    CultureInfo.InvariantCulture);
            else if (!int.TryParse(trimmed, out value))
                throw new NotSupportedException(
                    $"unsupported LEA shift '{text}'");
            if (value < 0 || value > 31)
                throw new NotSupportedException(
                    $"unsupported LEA shift '{text}'");
            return value;
        }

        private bool TryEmitLea64(List<SassNode> nodes, ref int index)
        {
            if (index >= nodes.Count ||
                nodes[index] is not SassInstruction low ||
                low.Guard != null ||
                !low.Mnemonic.Equals("LEA", StringComparison.OrdinalIgnoreCase) ||
                low.Operands.Count != 5)
                return false;
            string rd = NoReuse(low.Operands[0]);
            string carry = NoReuse(low.Operands[1]);
            string ra = NoReuse(low.Operands[2]);
            string baseLow = NoReuse(low.Operands[3]);
            string shiftText = NoReuse(low.Operands[4]);
            if (!Regex.IsMatch(rd, @"^R\d+$", RegexOptions.IgnoreCase) ||
                !Regex.IsMatch(carry, @"^P\d+$", RegexOptions.IgnoreCase) ||
                !Regex.IsMatch(ra, @"^R\d+$", RegexOptions.IgnoreCase))
                return false;
            Match constant = Regex.Match(baseLow,
                @"^c\[0x0\]\[0x([0-9A-Fa-f]+)\]$",
                RegexOptions.IgnoreCase);
            if (!constant.Success)
                return false;
            int offset = int.Parse(constant.Groups[1].Value,
                NumberStyles.HexNumber, CultureInfo.InvariantCulture);
            if (!_paramsByOffset.TryGetValue(offset, out PtxParameter? parameter) ||
                parameter.Type != "u64" ||
                !_paramPointers.TryGetValue(parameter, out string? pointer))
                return false;
            // Use the raw ld.param destination, not the mov.b64 alias: the
            // compiler's pointer-home analysis tracks the parameter load.
            if (pointer.StartsWith("%rpptr", StringComparison.Ordinal))
                pointer = "%rdp" + pointer[6..];
            int shift = LeaShift(shiftText);
            int highIndex = -1;
            string? highAddend = null;
            for (int j = index + 1; j < nodes.Count && j <= index + 10; j++)
            {
                if (nodes[j] is not SassInstruction high)
                    continue;
                if (high.Guard != null ||
                    !high.Mnemonic.StartsWith("LEA.HI.X",
                        StringComparison.OrdinalIgnoreCase) ||
                    high.Operands.Count < 6)
                    continue;
                if (!NoReuse(high.Operands[1]).Equals(ra,
                        StringComparison.OrdinalIgnoreCase) ||
                    !NoReuse(high.Operands[4]).Equals(shiftText,
                        StringComparison.OrdinalIgnoreCase) ||
                    !NoReuse(high.Operands[5]).Equals(carry,
                        StringComparison.OrdinalIgnoreCase))
                    continue;
                highAddend = NoReuse(high.Operands[3]);
                highIndex = j;
                break;
            }
            if (highIndex < 0)
                return false;
            // LEA.HI.X adds Rc<<shift to the high word, i.e. Rc is the
            // high half of a 64-bit index.  If the compiler materialises
            // that half between the two instructions, leave the pair to
            // the generic LEA/LEA.HI.X path so the reads stay ordered.
            bool wideIndex = highAddend != null && !IsRz(highAddend);
            if (wideIndex && (!Regex.IsMatch(highAddend!, @"^R\d+$",
                    RegexOptions.IgnoreCase) ||
                WrittenBetween(nodes, index + 1, highIndex, highAddend!)))
                return false;
            // Keep the scalar offset and the pointer result in separate
            // temporaries: ACuda's pointer-home analysis is flow
            // insensitive, and reusing one register for both homes is
            // rejected as a conflict.
            string scaledOffset = $"%rdt{_nextTemp++}";
            string address = $"%rdt{_nextTemp++}";
            if (wideIndex)
            {
                _lines.Add($"    mov.b64 {scaledOffset}, " +
                    $"{{{Reg(ra)}, {Reg(highAddend!)}}};");
                if (shift > 0)
                    _lines.Add($"    shl.b64 {scaledOffset}, " +
                        $"{scaledOffset}, {shift};");
            }
            else
            {
                _lines.Add($"    cvt.u64.u32 {scaledOffset}, {Reg(ra)};");
                if (shift > 0)
                    _lines.Add($"    shl.b64 {scaledOffset}, " +
                        $"{scaledOffset}, {shift};");
            }
            _lines.Add($"    add.u64 {address}, {pointer}, {scaledOffset};");
            string highRd = NextRegister(rd);
            // The compiler sometimes clobbers the high half between the
            // low LEA and the LEA.HI.X (a dead 32-bit add reusing the
            // register).  The .HI.X is what restores it in SASS, so
            // materialise the pair's low half now and defer the high half
            // until that instruction instead of writing both here.
            if (WrittenBetween(nodes, index + 1, highIndex, highRd))
            {
                string scratch = $"%rt{_nextTemp++}";
                _lines.Add($"    mov.b64 {{{Reg(rd)}, {scratch}}}, {address};");
                _pairRegs.Remove(rd);
                _deferredPairHigh[highIndex] = (rd, highRd, address);
            }
            else
            {
                CachePair(rd, address);
                _skipNodes.Add(highIndex);
            }
            return true;
        }

        // ULEA is the uniform-datapath LEA: URd = (URa << shift) + URb.
        // Fold the shift into a uniform multiply so source aliasing is
        // safe and no uniform scratch register is needed.
        private void EmitUlea(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string destination = Regex.Replace(
                StripReuse(instruction.Operands[0]), @"\.CC$", "",
                RegexOptions.IgnoreCase);
            if (!Regex.IsMatch(destination, @"^UR\d+$",
                    RegexOptions.IgnoreCase))
                throw new NotSupportedException(
                    $"unsupported ULEA destination " +
                    $"'{instruction.Operands[0]}'");
            int first = 1;
            if (Regex.IsMatch(StripReuse(instruction.Operands[1]),
                    @"^!?U?P\d+$|^!?U?PT$", RegexOptions.IgnoreCase))
                first++;
            if (first + 2 >= instruction.Operands.Count)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string a = UniformOperand(instruction.Operands[first]);
            string b = UniformOperand(instruction.Operands[first + 1]);
            int shift = LeaShift(instruction.Operands[first + 2]);
            uint scale = 1u << shift;
            _lines.Add($"{prefix}mad.lo.u32 {UniformReg(destination)}, " +
                $"{a}, {scale.ToString(CultureInfo.InvariantCulture)}, {b};");
        }

        private void EmitLea(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string rd = NoReuse(instruction.Operands[0]);
            string ra;
            string baseOperand;
            int shift;
            if (instruction.Operands.Count == 4)
            {
                ra = NoReuse(instruction.Operands[1]);
                baseOperand = NoReuse(instruction.Operands[2]);
                shift = LeaShift(instruction.Operands[3]);
            }
            else if (Regex.IsMatch(NoReuse(instruction.Operands[1]),
                         @"^P\d+$", RegexOptions.IgnoreCase))
            {
                ra = NoReuse(instruction.Operands[2]);
                baseOperand = NoReuse(instruction.Operands[3]);
                shift = LeaShift(instruction.Operands[4]);
            }
            else
            {
                ra = NoReuse(instruction.Operands[1]);
                baseOperand = NoReuse(instruction.Operands[2]);
                shift = LeaShift(instruction.Operands[4]);
            }
            if (!Regex.IsMatch(ra, @"^R\d+$", RegexOptions.IgnoreCase))
                throw new NotSupportedException(
                    $"unsupported LEA source '{ra}'");
            string shifted = $"%rt{_nextTemp++}";
            if (shift == 0)
                _lines.Add($"    mov.b32 {shifted}, {Reg(ra)};");
            else
                _lines.Add($"    shl.b32 {shifted}, {Reg(ra)}, {shift};");
            bool carryOut = instruction.Operands.Count >= 5 &&
                Regex.IsMatch(instruction.Operands[1].Trim(),
                    @"^P\d+$", RegexOptions.IgnoreCase);
            if (carryOut)
            {
                _lines.Add($"{prefix}add.cc.u32 {Reg(rd)}, {shifted}, " +
                    $"{Operand(baseOperand)};");
                _lines.Add($"{prefix}setp.lt.u32 " +
                    $"{Pred(instruction.Operands[1].Trim())}, " +
                    $"{Reg(rd)}, {shifted};");
                _leaCarryValid = true;
            }
            else
            {
                _lines.Add($"{prefix}add.u32 {Reg(rd)}, {shifted}, " +
                    $"{Operand(baseOperand)};");
            }
        }

        // LEA.HI Rd, Pp, Ra, Rb, Rc, shift computes the high word of
        // (Ra << shift) + Rb + (Rc << shift): the right-shifted Ra
        // survives as the low bits of Rd.  EmitLea's plain form would
        // treat it as `Ra + Rb` and silently drop both the shift and Rc
        // (nvcc uses this to fold a /2^shift into a 64-bit address).
        private void EmitLeaHi(SassInstruction instruction, string prefix)
        {
            bool sx32 = instruction.Mnemonic.Contains(
                ".SX32", StringComparison.OrdinalIgnoreCase);
            int first = instruction.Operands.Count > 1 &&
                Regex.IsMatch(NoReuse(instruction.Operands[1]),
                    @"^P\d+$", RegexOptions.IgnoreCase) ? 2 : 1;
            bool hasCarryAddend = instruction.Operands.Count >= first + 4;
            if (instruction.Operands.Count < first + 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs " +
                    $"{first + 3} operands");
            string rd = NoReuse(instruction.Operands[0]);
            string ra = NoReuse(instruction.Operands[first]);
            string rb = NoReuse(instruction.Operands[first + 1]);
            string? rc = hasCarryAddend
                ? NoReuse(instruction.Operands[first + 2]) : null;
            int shift = LeaShift(instruction.Operands[
                hasCarryAddend ? first + 3 : first + 2]);
            string highShifted = $"%rt{_nextTemp++}";
            if (shift == 0)
                _lines.Add($"    mov.b32 {highShifted}, 0;");
            else if (sx32)
                _lines.Add($"    shr.s32 {highShifted}, {Reg(ra)}, " +
                    $"{32 - shift};");
            else
                _lines.Add($"    shr.u32 {highShifted}, {Reg(ra)}, " +
                    $"{32 - shift};");
            string sum;
            if (rc == null)
            {
                sum = highShifted;
            }
            else
            {
                string shiftedHigh = $"%rt{_nextTemp++}";
                if (shift == 0)
                    _lines.Add($"    mov.b32 {shiftedHigh}, {Operand(rc)};");
                else
                    _lines.Add($"    shl.b32 {shiftedHigh}, {Operand(rc)}, " +
                        $"{shift};");
                sum = $"%rt{_nextTemp++}";
                _lines.Add($"    add.u32 {sum}, {shiftedHigh}, {highShifted};");
            }
            _lines.Add($"{prefix}add.cc.u32 {Reg(rd)}, {sum}, " +
                $"{Operand(rb)};");
            if (first == 2)
                _lines.Add($"{prefix}setp.lt.u32 " +
                    $"{Pred(instruction.Operands[1].Trim())}, " +
                    $"{Reg(rd)}, {sum};");
            _leaCarryValid = true;
        }

        private void EmitLeaHiX(SassInstruction instruction, string prefix)
        {
            bool sx32 = instruction.Mnemonic.Contains(
                ".SX32", StringComparison.OrdinalIgnoreCase);
            int needed = sx32 ? 5 : 6;
            if (instruction.Operands.Count < needed)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs {needed} operands");
            string rdHigh = NoReuse(instruction.Operands[0]);
            string ra = NoReuse(instruction.Operands[1]);
            string baseHigh = NoReuse(instruction.Operands[2]);
            string? rc = sx32 ? null : NoReuse(instruction.Operands[3]);
            int shift = LeaShift(instruction.Operands[sx32 ? 3 : 4]);
            string highShifted = $"%rt{_nextTemp++}";
            if (shift == 0)
                _lines.Add($"    mov.b32 {highShifted}, 0;");
            else if (sx32)
                _lines.Add($"    shr.s32 {highShifted}, {Reg(ra)}, " +
                    $"{32 - shift};");
            else
                _lines.Add($"    shr.u32 {highShifted}, {Reg(ra)}, " +
                    $"{32 - shift};");
            string sum;
            if (rc == null)
            {
                sum = highShifted;
            }
            else
            {
                string shiftedHigh = $"%rt{_nextTemp++}";
                if (shift == 0)
                    _lines.Add($"    mov.b32 {shiftedHigh}, {Reg(rc)};");
                else
                    _lines.Add($"    shl.b32 {shiftedHigh}, {Reg(rc)}, {shift};");
                sum = $"%rt{_nextTemp++}";
                _lines.Add($"    add.u32 {sum}, {shiftedHigh}, {highShifted};");
            }
            // LEA.HI.X consumes the carry produced by the matching low
            // LEA/IADD3 through the predicate in the last operand.  Several
            // low halves may be interleaved before their high consumers, so
            // the PTX carry flag cannot stand in for that predicate.
            string carryText = instruction.Operands[
                instruction.Operands.Count - 1].Trim();
            bool hasCarryIn = Regex.IsMatch(carryText,
                @"^!?(?:P\d+|PT)$", RegexOptions.IgnoreCase);
            string one = hasCarryIn ? PredicateValue(carryText, prefix) : "0";
            string sum2 = $"%rt{_nextTemp++}";
            _lines.Add($"{prefix}add.u32 {sum2}, {sum}, " +
                $"{Operand(baseHigh)};");
            if (one != "0")
                _lines.Add($"{prefix}add.u32 {Reg(rdHigh)}, {sum2}, {one};");
            else
                _lines.Add($"{prefix}mov.b32 {Reg(rdHigh)}, {sum2};");
            _leaCarryValid = false;
        }

        private void EmitUldc(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            string destination = instruction.Operands[0].Trim();
            string source = instruction.Operands[1].Trim();
            // Uniform-register-indexed constant load, e.g.
            // c[0x0][UR38+0x160].  The index selects a parameter word at
            // run time, so go through the dynamic const path.
            Match indexed = Regex.Match(source,
                @"^c\[0x0\]\[(UR\d+)(?:([+-])(0x[0-9A-Fa-f]+|\d+))?\]$",
                RegexOptions.IgnoreCase);
            if (indexed.Success &&
                Regex.IsMatch(destination, @"^UR\d+$",
                    RegexOptions.IgnoreCase))
            {
                int index = int.Parse(
                    Regex.Match(destination, @"^UR(\d+)$",
                        RegexOptions.IgnoreCase).Groups[1].Value,
                    CultureInfo.InvariantCulture);
                string address = $"%rt{_nextTemp++}";
                _lines.Add($"{prefix}mov.u32 {address}, " +
                    $"{UniformOperand(indexed.Groups[1].Value)};");
                if (indexed.Groups[3].Success)
                {
                    string deltaText = indexed.Groups[3].Value;
                    int delta = deltaText.StartsWith("0x",
                            StringComparison.OrdinalIgnoreCase)
                        ? int.Parse(deltaText[2..], NumberStyles.HexNumber,
                            CultureInfo.InvariantCulture)
                        : int.Parse(deltaText, NumberStyles.Integer,
                            CultureInfo.InvariantCulture);
                    if (indexed.Groups[2].Value == "-")
                        delta = -delta;
                    if (delta != 0)
                        _lines.Add($"{prefix}add.s32 {address}, " +
                            $"{address}, {delta};");
                }
                string word = $"%rt{_nextTemp++}";
                if (_dynamicLdc)
                    EmitSharedParamWord(address, word, 0);
                else
                    _lines.Add($"{prefix}ld.const.u32 {word}, [{address}];");
                _lines.Add($"{prefix}mov.u32 %ur{index}, {word};");
                if (instruction.Mnemonic.Contains(".64",
                        StringComparison.OrdinalIgnoreCase))
                {
                    string high = $"%rt{_nextTemp++}";
                    if (_dynamicLdc)
                        EmitSharedParamWord(address, high, 4);
                    else
                        _lines.Add($"{prefix}ld.const.u32 {high}, " +
                            $"[{address}+4];");
                    _lines.Add($"{prefix}mov.u32 %ur{index + 1}, {high};");
                }
                return;
            }
            Match constant = Regex.Match(source,
                @"^c\[0x([0-9A-Fa-f]+)\]\[0x([0-9A-Fa-f]+)\]$",
                RegexOptions.IgnoreCase);
            if (!constant.Success)
                throw new NotSupportedException(
                    $"unsupported ULDC source '{source}'");
            int bank = int.Parse(constant.Groups[1].Value,
                NumberStyles.HexNumber, CultureInfo.InvariantCulture);
            int offset = int.Parse(constant.Groups[2].Value,
                NumberStyles.HexNumber, CultureInfo.InvariantCulture);
            Match reg = Regex.Match(destination, @"^UR(\d+)$",
                RegexOptions.IgnoreCase);
            if (!reg.Success)
                throw new NotSupportedException(
                    $"unsupported ULDC destination '{destination}'");
            int number = int.Parse(reg.Groups[1].Value,
                CultureInfo.InvariantCulture);
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            bool byteLoad = mnemonic.Contains(".U8") ||
                mnemonic.Contains(".S8");
            bool halfLoad = mnemonic.Contains(".U16") ||
                mnemonic.Contains(".S16");
            if (byteLoad || halfLoad)
            {
                // ULDC.U8/U16 loads a narrow field from the constant
                // bank.  Reading the whole word returns neighbouring
                // fields too: PyTorch's reduce flags at c[0x570] keep
                // the low byte clear and byte 1 set, which then enabled
                // the "accumulate into the existing output" path.
                // If a parameter starts exactly at the narrow field
                // (e.g. a 2-byte kernel argument packed after an empty
                // struct), use that parameter directly instead of the
                // containing 4-byte word: the neighbour may be a
                // different host argument with unrelated bytes.
                bool exactParameter = bank == 0 &&
                    _paramsByOffset.ContainsKey(offset);
                int wordOffset = offset & ~3;
                int shift = exactParameter ? 0 : (offset - wordOffset) * 8;
                string word = exactParameter
                    ? Operand($"c[0x0][0x{offset:x}]")
                    : BankWord(bank, wordOffset);
                _lines.Add($"{prefix}mov.u32 %ur{number}, {word};");
                if (shift > 0)
                    _lines.Add($"    shr.u32 %ur{number}, " +
                        $"%ur{number}, {shift};");
                bool signedLoad = mnemonic.Contains(".S8") ||
                    mnemonic.Contains(".S16");
                if (signedLoad)
                {
                    int bits = byteLoad ? 24 : 16;
                    _lines.Add($"    shl.b32 %ur{number}, " +
                        $"%ur{number}, {bits};");
                    _lines.Add($"    shr.s32 %ur{number}, " +
                        $"%ur{number}, {bits};");
                }
                else
                {
                    uint mask = byteLoad ? 0xffu : 0xffffu;
                    _lines.Add($"    and.b32 %ur{number}, %ur{number}, " +
                        $"0x{mask:x};");
                }
                return;
            }
            _lines.Add($"{prefix}mov.u32 %ur{number}, " +
                $"{BankWord(bank, offset)};");
            if (mnemonic.Contains(".64",
                    StringComparison.OrdinalIgnoreCase))
                _lines.Add($"{prefix}mov.u32 %ur{number + 1}, " +
                    $"{BankWord(bank, offset + 4)};");
        }

        private string BankWord(int bank, int offset) =>
            Operand($"c[0x{bank:x}][0x{offset:x}]");

        // Plain (non-uniform) constant-bank load with a direct address.
        // Param-backed words go through Operand's param/constant resolver;
        // narrow loads take the byte/half field out of its containing word.
        private void EmitLdc(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            string destination = instruction.Operands[0].Trim();
            string source = instruction.Operands[1].Trim();
            Match constant = Regex.Match(source,
                @"^c\[0x([0-9A-Fa-f]+)\]\[0x([0-9A-Fa-f]+)\]$",
                RegexOptions.IgnoreCase);
            if (!constant.Success)
            {
                // Dynamic constant-bank loads (c[bank][Rindex]) need the
                // bank image materialized as memory.  Zero them for now so
                // legacy PhysX kernels can be lifted and exercised.
                Match dynamic = Regex.Match(source,
                    @"^c\[0x([0-9A-Fa-f]+)\]\[(R\d+)\]$",
                    RegexOptions.IgnoreCase);
                if (dynamic.Success)
                {
                    // 128-bit dynamic constant loads have no safe zero
                    // fallback (it would leave two destination registers
                    // undefined); keep rejecting them.
                    if (instruction.Mnemonic.Contains(
                            ".128", StringComparison.OrdinalIgnoreCase))
                        throw new NotSupportedException(
                            $"unsupported LDC source '{source}'");
                    string zeroDst = Reg(destination);
                    _lines.Add($"{prefix}mov.u32 {zeroDst}, 0;");
                    if (instruction.Mnemonic.Contains(".64",
                            StringComparison.OrdinalIgnoreCase))
                        _lines.Add($"{prefix}mov.u32 " +
                            $"{Reg(NextRegister(BaseReg(destination)))}, 0;");
                    return;
                }
                throw new NotSupportedException(
                    $"unsupported LDC source '{source}'");
            }
            int bank = int.Parse(constant.Groups[1].Value,
                NumberStyles.HexNumber, CultureInfo.InvariantCulture);
            int offset = int.Parse(constant.Groups[2].Value,
                NumberStyles.HexNumber, CultureInfo.InvariantCulture);
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            bool byteLoad = mnemonic.Contains(".U8") ||
                mnemonic.Contains(".S8");
            bool halfLoad = mnemonic.Contains(".U16") ||
                mnemonic.Contains(".S16");
            string register = Reg(destination);
            if (byteLoad || halfLoad)
            {
                bool exactParameter = bank == 0 &&
                    _paramsByOffset.ContainsKey(offset);
                int wordOffset = offset & ~3;
                int shift = exactParameter ? 0 : (offset - wordOffset) * 8;
                string word = exactParameter
                    ? Operand($"c[0x0][0x{offset:x}]")
                    : BankWord(bank, wordOffset);
                _lines.Add($"{prefix}mov.u32 {register}, {word};");
                if (shift > 0)
                    _lines.Add($"{prefix}shr.u32 {register}, " +
                        $"{register}, {shift};");
                bool signedLoad = mnemonic.Contains(".S8") ||
                    mnemonic.Contains(".S16");
                if (signedLoad)
                {
                    int bits = byteLoad ? 24 : 16;
                    _lines.Add($"{prefix}shl.b32 {register}, " +
                        $"{register}, {bits};");
                    _lines.Add($"{prefix}shr.s32 {register}, " +
                        $"{register}, {bits};");
                }
                else
                {
                    uint mask = byteLoad ? 0xffu : 0xffffu;
                    _lines.Add($"{prefix}and.b32 {register}, " +
                        $"{register}, 0x{mask:x};");
                }
                return;
            }
            _lines.Add($"{prefix}mov.u32 {register}, " +
                $"{BankWord(bank, offset)};");
            if (mnemonic.Contains(".64",
                    StringComparison.OrdinalIgnoreCase))
                _lines.Add($"{prefix}mov.u32 " +
                    $"{Reg(NextRegister(BaseReg(destination)))}, " +
                    $"{BankWord(bank, offset + 4)};");
        }

        private void EmitFfma(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            string rounding = mnemonic.Contains(".RM", StringComparison.Ordinal)
                ? "rm"
                : mnemonic.Contains(".RP", StringComparison.Ordinal)
                    ? "rp"
                    : mnemonic.Contains(".RZ", StringComparison.Ordinal)
                        ? "rz" : "rn";
            string operation = "fma." + rounding;
            if (mnemonic.Contains(".FTZ", StringComparison.Ordinal))
                operation += ".ftz";
            if (mnemonic.Contains(".SAT", StringComparison.Ordinal))
                operation += ".sat";
            _lines.Add($"{prefix}{operation}.f32 " +
                $"{Reg(instruction.Operands[0])}, " +
                $"{FloatOperand(instruction.Operands[1])}, " +
                $"{FloatOperand(instruction.Operands[2])}, " +
                $"{FloatOperand(instruction.Operands[3])};");
        }

        private void EmitCvtAlias(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            string[] parts = instruction.Mnemonic.ToUpperInvariant().Split('.');
            if (parts.Length > 1 && parts[1] == "PACK")
            {
                EmitCvtPack(instruction, prefix, parts);
                return;
            }
            int index = 1;
            bool flushToZero = false;
            if (index < parts.Length && parts[index] == "FTZ")
            {
                flushToZero = true;
                index++;
            }
            string rounding = "rn";
            bool explicitRounding = false;
            if (index < parts.Length)
            {
                string token = parts[index];
                if (token is "RN" or "RZI" or "RZ" or "TRUNC" or "RMI"
                    or "RM" or "FLOOR" or "RPI" or "RP" or "CEIL")
                {
                    rounding = token switch
                    {
                        "RZI" or "RZ" or "TRUNC" => "rzi",
                        "RMI" or "RM" or "FLOOR" => "rmi",
                        "RPI" or "RP" or "CEIL" => "rpi",
                        _ => "rn",
                    };
                    explicitRounding = true;
                    index++;
                }
            }
            if (index + 1 >= parts.Length)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            string destinationType = parts[index].ToLowerInvariant();
            string sourceType = parts[index + 1].ToLowerInvariant();
            if (!IsCvtType(destinationType) || !IsCvtType(sourceType))
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            if (flushToZero && explicitRounding)
                throw new NotSupportedException(
                    "cvt.ftz does not take an explicit rounding modifier");
            bool wideSource = IsWideCvtType(sourceType);
            bool wideDestination = IsWideCvtType(destinationType);
            string source = wideSource
                ? PackRegisterPair(instruction.Operands[1])
                : Operand(instruction.Operands[1]);
            string conversion = flushToZero
                ? $"cvt.ftz.{destinationType}.{sourceType}"
                : $"cvt.{rounding}.{destinationType}.{sourceType}";
            if (!wideDestination)
            {
                _lines.Add($"{prefix}{conversion} " +
                    $"{Reg(instruction.Operands[0])}, {source};");
                return;
            }
            string destination = BaseReg(instruction.Operands[0]);
            string value = $"%rdt{_nextTemp++}";
            _lines.Add($"{prefix}{conversion} {value}, {source};");
            UnpackPair64(value, destination, NextRegister(destination));
        }

        private void EmitCvtPack(
            SassInstruction instruction, string prefix, string[] parts)
        {
            if (parts.Length < 5 || parts[2] != "SAT")
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            string conversion = string.Join(".",
                parts.Select(part => part.ToLowerInvariant()));
            string operands = string.Join(", ",
                instruction.Operands.Skip(1).Select(Operand));
            _lines.Add($"{prefix}{conversion} " +
                $"{Reg(instruction.Operands[0])}, {operands};");
        }

        private static bool IsCvtType(string type) =>
            type is "f32" or "f64" or "s32" or "u32" or "s64" or "u64";

        private static bool IsWideCvtType(string type) =>
            type is "f64" or "s64" or "u64";

        private void EmitI2f(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            bool f64 = mnemonic.Contains(".F64", StringComparison.Ordinal);
            string sourceType = mnemonic.Contains(".U64", StringComparison.Ordinal)
                ? "u64"
                : mnemonic.Contains(".S64", StringComparison.Ordinal)
                    ? "s64"
                    : mnemonic.Contains(".U32", StringComparison.Ordinal)
                        ? "u32" : "s32";
            string source = IsWideCvtType(sourceType)
                ? PackRegisterPair(instruction.Operands[1])
                : Operand(instruction.Operands[1]);
            string destinationType = f64 ? "f64" : "f32";
            string rounding = mnemonic.Contains(".RP", StringComparison.Ordinal)
                ? "rp"
                : mnemonic.Contains(".RM", StringComparison.Ordinal)
                    ? "rm"
                    : mnemonic.Contains(".RZ", StringComparison.Ordinal)
                        ? "rz" : "rn";
            if (!f64)
            {
                _lines.Add($"{prefix}cvt.{rounding}.{destinationType}.{sourceType} " +
                    $"{Reg(instruction.Operands[0])}, {source};");
                return;
            }
            string destination = BaseReg(instruction.Operands[0]);
            string value = $"%rdt{_nextTemp++}";
            _lines.Add($"{prefix}cvt.{rounding}.f64.{sourceType} {value}, {source};");
            UnpackPair64(value, destination, NextRegister(destination));
        }

        private void EmitF2i(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            string destinationType = mnemonic.Contains(".U64", StringComparison.Ordinal)
                ? "u64"
                : mnemonic.Contains(".S64", StringComparison.Ordinal)
                    ? "s64"
                    : mnemonic.Contains(".U32", StringComparison.Ordinal)
                        ? "u32" : "s32";
            string sourceType = mnemonic.Contains(".F64", StringComparison.Ordinal)
                ? "f64" : "f32";
            // SASS F2I defaults to Round3(ROUND), i.e. round-to-nearest
            // (matching __float2int_rn), not truncation.  Only an explicit
            // .TRUNC/.RZ truncates; treating the default as rzi made powf's
            // exponent split disagree with its FRND and produced 0/inf.
            string rounding = mnemonic.Contains(".FLOOR", StringComparison.Ordinal)
                ? "rmi"
                : mnemonic.Contains(".CEIL", StringComparison.Ordinal)
                    ? "rpi"
                    : mnemonic.Contains(".TRUNC", StringComparison.Ordinal) ||
                      mnemonic.Contains(".RZ", StringComparison.Ordinal)
                        ? "rzi" : "rni";
            string source = sourceType == "f64"
                ? PackRegisterPair(instruction.Operands[1])
                : Operand(instruction.Operands[1]);
            if (!IsWideCvtType(destinationType))
            {
                _lines.Add($"{prefix}cvt.{rounding}.{destinationType}.{sourceType} " +
                    $"{Reg(instruction.Operands[0])}, {source};");
                return;
            }
            string destination = BaseReg(instruction.Operands[0]);
            string value = $"%rdt{_nextTemp++}";
            _lines.Add($"{prefix}cvt.{rounding}.{destinationType}.{sourceType} " +
                $"{value}, {source};");
            UnpackPair64(value, destination, NextRegister(destination));
        }

        // Predicated 16-bit conversion sequences cannot use ACuda's
        // select form (the destination is a raw bit-pattern holder), so
        // wrap the whole sequence in an explicit skip branch.
        private bool BeginGuardedRegion(string prefix, out string skip)
        {
            skip = "";
            string guard = prefix.Trim();
            if (guard.Length == 0)
                return false;
            skip = $"L_g{_nextTemp++}";
            string inverted = guard.StartsWith("@!", StringComparison.Ordinal)
                ? "@" + guard[2..]
                : "@!" + guard[1..];
            _lines.Add($"    {inverted} bra {skip};");
            return true;
        }

        private void EmitF2fp(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            bool bf16 = instruction.Mnemonic.Contains(
                "BF16", StringComparison.OrdinalIgnoreCase);
            if (!instruction.Mnemonic.Contains("PACK_AB",
                    StringComparison.OrdinalIgnoreCase) ||
                instruction.Mnemonic.Contains("MERGE_C",
                    StringComparison.OrdinalIgnoreCase) ||
                instruction.Mnemonic.Contains("UNPACK",
                    StringComparison.OrdinalIgnoreCase))
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            bool wrapped = BeginGuardedRegion(prefix, out string skip);
            string p = wrapped ? "" : prefix;
            string a = FloatOperand(instruction.Operands[1]);
            string b = FloatOperand(instruction.Operands[2]);
            string halfA = $"%rh{_nextTempHalf++}";
            string halfB = $"%rh{_nextTempHalf++}";
            string type = bf16 ? "bf16" : "f16";
            _lines.Add($"{p}cvt.rn.{type}.f32 {halfA}, {a};");
            _lines.Add($"{p}cvt.rn.{type}.f32 {halfB}, {b};");
            _lines.Add($"{p}mov.b32 {Reg(instruction.Operands[0])}, " +
                $"{{{halfB}, {halfA}}};");
            if (wrapped)
                _lines.Add($"{skip}:");
        }

        // Extract one half of a vector register and widen it to f32.
        private string HalfToFloatSource(string operand)
        {
            string text = StripReuse(operand).Trim();
            bool negate = text.StartsWith("-", StringComparison.Ordinal);
            if (negate)
                text = text[1..].Trim();
            if (text.Equals("RZ", StringComparison.OrdinalIgnoreCase) ||
                text.Equals("RZ.H0_H0", StringComparison.OrdinalIgnoreCase) ||
                text.Equals("RZ.H0", StringComparison.OrdinalIgnoreCase))
                return negate ? "0f80000000" : "0f00000000";
            Match half = Regex.Match(text, @"^R(\d+)\.H([01])_H\2$",
                RegexOptions.IgnoreCase);
            if (!half.Success)
                half = Regex.Match(text, @"^R(\d+)\.H([01])$",
                    RegexOptions.IgnoreCase);
            string bits;
            if (half.Success)
            {
                string source = Reg("R" + half.Groups[1].Value);
                bits = $"%rt{_nextTemp++}";
                if (half.Groups[2].Value == "1")
                    _lines.Add($"    shr.u32 {bits}, {source}, 16;");
                else
                    _lines.Add($"    and.b32 {bits}, {source}, 0xffff;");
            }
            else
            {
                string value = Operand(text);
                bits = $"%rt{_nextTemp++}";
                _lines.Add($"    and.b32 {bits}, {value}, 0xffff;");
            }
            string holder = $"%rh{_nextTempHalf++}";
            _lines.Add($"    cvt.u16.u32 {holder}, {bits};");
            string widened = $"%ft{_nextTempFloat++}";
            _lines.Add($"    cvt.rn.f32.f16 {widened}, {holder};");
            if (negate)
            {
                string negated = $"%ft{_nextTempFloat++}";
                _lines.Add($"    neg.f32 {negated}, {widened};");
                widened = negated;
            }
            return widened;
        }

        private void EmitHadd2(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            string type = instruction.Mnemonic.Contains("BF16",
                StringComparison.OrdinalIgnoreCase) ? "bf16x2" : "f16x2";
            string op = instruction.Mnemonic.StartsWith("HMUL2",
                StringComparison.OrdinalIgnoreCase) ? "mul" : "add";
            if (instruction.Mnemonic.Contains(
                    ".F32", StringComparison.Ordinal) &&
                !instruction.Mnemonic.Contains(
                    "F32.F16", StringComparison.Ordinal))
            {
                // Mixed-precision form: the destination is f32 and each
                // source is one half of a register (HADD2.F32 is the
                // compiler's f16 -> f32 widening idiom).
                string destination = Reg(instruction.Operands[0]);
                string a = HalfToFloatSource(instruction.Operands[1]);
                string b = HalfToFloatSource(instruction.Operands[2]);
                _lines.Add($"{prefix}{op}.f32 {destination}, {a}, {b};");
                return;
            }
            _lines.Add($"{prefix}{op}.rn.{type} {Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])}, " +
                $"{Operand(instruction.Operands[2])};");
        }

        private void EmitCs2r(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            string source = instruction.Operands[1].Trim().ToUpperInvariant();
            string destination = BaseReg(instruction.Operands[0]);
            bool wide = !instruction.Mnemonic.Contains(
                ".32", StringComparison.OrdinalIgnoreCase);
            if (source == "SRZ")
            {
                if (wide)
                    _lines.Add($"{prefix}mov.b64 " +
                        $"{{{Reg(destination)}, {Reg(NextRegister(destination))}}}, 0;");
                else
                    _lines.Add($"{prefix}mov.u32 {Reg(destination)}, 0;");
                return;
            }
            if (source == "SR_CLOCKLO")
            {
                if (wide)
                {
                    string value = $"%rdt{_nextTemp++}";
                    _lines.Add($"{prefix}mov.u64 {value}, %clock64;");
                    UnpackPair64(value, destination, NextRegister(destination));
                }
                else
                {
                    _lines.Add($"{prefix}mov.u32 {Reg(destination)}, %clock;");
                }
                return;
            }
            throw new NotSupportedException(
                $"SASS instruction '{instruction.Mnemonic}' from '{source}' is not supported");
        }

        private void EmitF2f(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            bool ftz = mnemonic.Contains(".FTZ", StringComparison.Ordinal);
            string rnd = mnemonic.Contains(".RM", StringComparison.Ordinal) ? "rm"
                : mnemonic.Contains(".RP", StringComparison.Ordinal) ? "rp"
                : mnemonic.Contains(".RZ", StringComparison.Ordinal) ? "rz"
                : "rn";
            string ftzs = ftz ? ".ftz" : "";
            string conversion;
            bool sourceDouble = false, destinationDouble = false;
            if (mnemonic.Contains(".F32.F16", StringComparison.Ordinal))
                conversion = $"cvt{ftzs}.f32.f16";
            else if (mnemonic.Contains(".F16.F32", StringComparison.Ordinal))
                conversion = $"cvt.{rnd}{ftzs}.f16.f32";
            else if (mnemonic.Contains(".F32.F64", StringComparison.Ordinal))
            { conversion = $"cvt.{rnd}{ftzs}.f32.f64"; sourceDouble = true; }
            else if (mnemonic.Contains(".F64.F32", StringComparison.Ordinal))
            { conversion = $"cvt.{rnd}{ftzs}.f64.f32"; destinationDouble = true; }
            else if (mnemonic.Contains(".F16.F64", StringComparison.Ordinal))
            { conversion = $"cvt.{rnd}{ftzs}.f16.f64"; sourceDouble = true; }
            else if (mnemonic.Contains(".F32.F32", StringComparison.Ordinal))
                conversion = $"cvt.{rnd}{ftzs}.f32.f32";
            else if (mnemonic.Contains(".F64.F64", StringComparison.Ordinal))
            { conversion = $"cvt.{rnd}{ftzs}.f64.f64"; sourceDouble = true; destinationDouble = true; }
            else if (mnemonic.Contains(".F16.F16", StringComparison.Ordinal))
                conversion = $"cvt.{rnd}{ftzs}.f16.f16";
            else
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            // A 64-bit source/destination is a register pair, not one
            // b32 register: `cvt.rn.f32.f64 %r8, %r8` otherwise reads
            // only the low word and fails the type check.
            if (sourceDouble)
            {
                string source = PackRegisterPair(instruction.Operands[1]);
                string destination = destinationDouble
                    ? $"%dt{_nextTempDouble++}"
                    : Reg(instruction.Operands[0]);
                _lines.Add($"{prefix}{conversion} {destination}, {source};");
                if (destinationDouble)
                {
                    string low = BaseReg(instruction.Operands[0]);
                    _lines.Add($"{prefix}mov.b64 " +
                        $"{{{Reg(low)}, {Reg(NextRegister(low))}}}, " +
                        $"{destination};");
                }
                return;
            }
            if (destinationDouble)
            {
                string value = $"%dt{_nextTempDouble++}";
                _lines.Add($"{prefix}{conversion} {value}, " +
                    $"{Operand(instruction.Operands[1])};");
                string low = BaseReg(instruction.Operands[0]);
                _lines.Add($"{prefix}mov.b64 " +
                    $"{{{Reg(low)}, {Reg(NextRegister(low))}}}, {value};");
                return;
            }
            _lines.Add($"{prefix}{conversion} {Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])};");
        }

        private void EmitFrnd(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            string mode = mnemonic.Contains("FLOOR") ? "rmi"
                : mnemonic.Contains("CEIL") ? "rpi"
                : mnemonic.Contains("TRUNC") ? "rzi"
                : "rni";
            string type = mnemonic.Contains(".F64") ? "f64"
                : mnemonic.Contains(".F16") ? "f16"
                : "f32";
            if (type == "f64")
            {
                // A 64-bit value is a register pair.  Writing only the
                // low word (cvt.rni.f32.f32) leaves the high word stale
                // and corrupts every following f64 operation.
                string source = PackRegisterPair(instruction.Operands[1]);
                string value = $"%dt{_nextTempDouble++}";
                _lines.Add($"{prefix}cvt.{mode}.f64.f64 {value}, {source};");
                string low = BaseReg(instruction.Operands[0]);
                _lines.Add($"{prefix}mov.b64 " +
                    $"{{{Reg(low)}, {Reg(NextRegister(low))}}}, {value};");
                return;
            }
            _lines.Add($"{prefix}cvt.{mode}.{type}.{type} " +
                $"{Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])};");
        }

        private void EmitImnmx(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            string type = mnemonic.Contains(".U32") ? "u32" : "s32";
            string operation = "min";
            if (instruction.Operands.Count >= 4)
            {
                string predicate = instruction.Operands[3].Trim();
                if (predicate.StartsWith("!", StringComparison.Ordinal))
                    operation = "max";
            }
            _lines.Add($"{prefix}{operation}.{type} " +
                $"{Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])}, " +
                $"{Operand(instruction.Operands[2])};");
        }

        private void EmitHfma2(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            string type = mnemonic.Contains("BF16",
                StringComparison.OrdinalIgnoreCase) ? "bf16x2" : "f16x2";
            if (mnemonic.Contains(".MMA") && instruction.Operands.Count >= 5)
            {
                // HFMA2.MMA Rd, Ra, Rb, Sc, Sc1: two f16 immediates
                // form the addend (Sc high, Sc1 low).  With zero
                // multiplicands this materializes a 32-bit constant,
                // e.g. ptxas uses it for the element-size stride 4.
                ushort high = F16ImmediateBits(instruction.Operands[3]);
                ushort low = F16ImmediateBits(instruction.Operands[4]);
                uint packed = ((uint)high << 16) | low;
                string destination = Reg(instruction.Operands[0]);
                if (IsZeroReg(instruction.Operands[1]) &&
                    IsZeroReg(instruction.Operands[2]))
                {
                    _lines.Add($"{prefix}mov.u32 {destination}, {packed};");
                    return;
                }
                string imm = $"%rt{_nextTemp++}";
                _lines.Add($"    mov.u32 {imm}, {packed};");
                _lines.Add($"{prefix}fma.rn.{type} {destination}, " +
                    $"{Operand(instruction.Operands[1])}, " +
                    $"{Operand(instruction.Operands[2])}, {imm};");
                return;
            }
            _lines.Add($"{prefix}fma.rn.{type} {Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])}, " +
                $"{Operand(instruction.Operands[2])}, " +
                $"{Operand(instruction.Operands[3])};");
        }

        private void EmitFsel(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string first = FloatBitsOperand(instruction.Operands[1]);
            string second = FloatBitsOperand(instruction.Operands[2]);
            string predicateText = instruction.Operands[3].Trim();
            if (predicateText.StartsWith("!", StringComparison.Ordinal))
                (first, second) = (second, first);
            string firstFloat = MaterializeFloatBits(first);
            string secondFloat = MaterializeFloatBits(second);
            string selected = $"%ft{_nextTempFloat++}";
            _lines.Add($"{prefix}selp.f32 {selected}, {firstFloat}, " +
                $"{secondFloat}, {Pred(predicateText)};");
            // The copy must share the instruction's guard: when the SASS
            // instruction is predicated off the destination has to keep its
            // old value, but an unguarded copy of the (undefined) temporary
            // would clobber it.
            _lines.Add($"{prefix}mov.b32 {Reg(instruction.Operands[0])}, " +
                $"{selected};");
        }

        private string MaterializeFloatBits(string bits)
        {
            string value = $"%ft{_nextTempFloat++}";
            _lines.Add($"    mov.b32 {value}, {bits};");
            return value;
        }

        // FSEL selects raw 32-bit float storage, so immediates stay in
        // their integer bit-pattern form and selp.b32 keeps every operand
        // the same PTX type.
        private string FloatBitsOperand(string operand)
        {
            string text = StripReuse(operand);
            if (text.Equals("RZ", StringComparison.OrdinalIgnoreCase))
                return "0";
            if (text.Equals("-RZ", StringComparison.OrdinalIgnoreCase))
                return "0x80000000";
            if (text.Equals("+INF", StringComparison.OrdinalIgnoreCase) ||
                text.Equals("INF", StringComparison.OrdinalIgnoreCase))
                return "0x7f800000";
            if (text.Equals("-INF", StringComparison.OrdinalIgnoreCase))
                return "0xff800000";
            if (text.Equals("+NAN", StringComparison.OrdinalIgnoreCase) ||
                text.Equals("NAN", StringComparison.OrdinalIgnoreCase) ||
                text.Equals("+QNAN", StringComparison.OrdinalIgnoreCase) ||
                text.Equals("QNAN", StringComparison.OrdinalIgnoreCase))
                return "0x7fc00000";
            if (text.Equals("-NAN", StringComparison.OrdinalIgnoreCase) ||
                text.Equals("-QNAN", StringComparison.OrdinalIgnoreCase))
                return "0xffc00000";
            // FSEL keeps raw storage bits: negating a float register means
            // flipping the sign bit, not two's-complement integer negation.
            Match negatedRegister = Regex.Match(text, @"^-R(\d+)$",
                RegexOptions.IgnoreCase);
            if (negatedRegister.Success)
            {
                string negated = $"%rt{_nextTemp++}";
                _lines.Add($"    xor.b32 {negated}, " +
                    $"{Reg("R" + negatedRegister.Groups[1].Value)}, " +
                    "0x80000000;");
                return negated;
            }
            // FSEL keeps raw storage bits, so decimal float literals must
            // become their 0x........ bit pattern: mov.b32 rejects a
            // float literal source.
            if (Regex.IsMatch(text, @"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$"))
            {
                if (float.TryParse(text, NumberStyles.Float,
                        CultureInfo.InvariantCulture, out float literal))
                    return "0x" + BitConverter.SingleToUInt32Bits(literal)
                        .ToString("x8", CultureInfo.InvariantCulture);
            }
            // FSEL keeps raw storage bits: absolute value clears the sign
            // bit and a negated absolute value sets it.
            Match absRegister = Regex.Match(text, @"^(-?)\|R(\d+)\|$",
                RegexOptions.IgnoreCase);
            if (absRegister.Success)
            {
                string source = Reg("R" + absRegister.Groups[2].Value);
                string bits = $"%rt{_nextTemp++}";
                if (absRegister.Groups[1].Value == "-")
                    _lines.Add($"    or.b32 {bits}, {source}, 0x80000000;");
                else
                    _lines.Add($"    and.b32 {bits}, {source}, 0x7fffffff;");
                return bits;
            }
            return Operand(text);
        }

        private void EmitIadd3(SassInstruction instruction, string prefix)
        {
            if (instruction.Mnemonic.Contains(".X", StringComparison.OrdinalIgnoreCase))
            {
                // IADD3.X Rd, Ra, Rb, Rc, Pin[, Pout]: the high word of a
                // 64-bit add.  The matching low-word IADD3 emitted
                // add.cc and produced Pin in the condition code, so
                // consume it with addc.  All lifted sites use RZ for Rc;
                // keep the generic form for completeness.
                if (instruction.Operands.Count < 4)
                    throw new NotSupportedException(
                        $"SASS instruction '{instruction.Mnemonic}' needs four operands");
                string xdst = Reg(instruction.Operands[0]);
                if (xdst == "0")
                    xdst = $"%rt{_nextTemp++}"; // RZ: discard the result
                string xa = Operand(instruction.Operands[1]);
                string xb = Operand(instruction.Operands[2]);
                string xc = Operand(instruction.Operands[3]);
                bool hasCarryIn = instruction.Operands.Count >= 5 &&
                    Regex.IsMatch(instruction.Operands[4].Trim(),
                        @"^!?(?:P\d+|PT)$", RegexOptions.IgnoreCase);
                string xsum = $"%rt{_nextTemp++}";
                if (hasCarryIn)
                {
                    // The carry-in is an explicit predicate, not necessarily
                    // the PTX carry flag: ptxas interleaves several low adds
                    // (e.g. four LEA.HI.X.SX32) before their .X consumers.
                    string one = PredicateValue(
                        instruction.Operands[4].Trim(), prefix);
                    _lines.Add($"{prefix}add.u32 {xsum}, {xa}, {xb};");
                    if (one != "0")
                        _lines.Add($"{prefix}add.u32 {xsum}, {xsum}, {one};");
                    if (xc == "0")
                    {
                        _lines.Add($"{prefix}mov.b32 {xdst}, {xsum};");
                    }
                    else
                    {
                        _lines.Add($"{prefix}add.u32 {xdst}, {xsum}, {xc};");
                    }
                }
                else if (xc == "0")
                {
                    _lines.Add($"{prefix}add.s32 {xdst}, {xa}, {xb};");
                }
                else
                {
                    _lines.Add($"{prefix}add.s32 {xsum}, {xa}, {xb};");
                    _lines.Add($"{prefix}add.s32 {xdst}, {xsum}, {xc};");
                }
                // Define the carry-out predicate when the high add can
                // only produce the incoming carry (all three sources RZ).
                if (instruction.Operands.Count >= 6 &&
                    IsRz(instruction.Operands[1]) &&
                    IsRz(instruction.Operands[2]) &&
                    IsRz(instruction.Operands[3]) &&
                    Regex.IsMatch(instruction.Operands[5].Trim(),
                        @"^P\d+$", RegexOptions.IgnoreCase))
                    _lines.Add($"{prefix}setp.eq.u32 " +
                        $"{Pred(instruction.Operands[5].Trim())}, 0, 1;");
                return;
            }
            int first = 1;
            bool carry = false;
            if (first < instruction.Operands.Count &&
                Regex.IsMatch(instruction.Operands[first].Trim(),
                    @"^!?(?:P\d+|PT)$", RegexOptions.IgnoreCase))
            {
                carry = true;
                first++;
            }
            // Six-operand form: IADD3 Rd, Pc, Pin, Ra, Rb, Rc.  The second
            // predicate is a carry-in addend, not one of the three sources;
            // treating it as Ra silently drops Rc (ptxas uses this in the
            // radix-sort store address where Rc is the lane offset).
            bool carryIn = false;
            if (carry && first < instruction.Operands.Count &&
                Regex.IsMatch(instruction.Operands[first].Trim(),
                    @"^!?(?:P\d+|PT)$", RegexOptions.IgnoreCase))
            {
                carryIn = true;
                first++;
            }
            if (first + 2 >= instruction.Operands.Count)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string destination = Reg(instruction.Operands[0]);
            if (destination == "0")
                destination = $"%rt{_nextTemp++}"; // RZ: discard the result
            string a = Operand(instruction.Operands[first]);
            string b = Operand(instruction.Operands[first + 1]);
            string c = Operand(instruction.Operands[first + 2]);
            if (carry)
            {
                // Exact three-source (plus optional carry-in) sum.  The
                // carry-out must reach the following .X/LEA.HI.X through
                // the PTX carry flag, so compute the full sum in 64 bits
                // and synthesise the flag with a conditional add.cc.
                // Chaining add.cc/addc on the low word would re-add the
                // intermediate carry when Rc is non-zero.
                string pin = carryIn
                    ? Operand(instruction.Operands[first - 1]) : "0";
                string wide = $"%rdt{_nextTemp++}";
                string term = $"%rdt{_nextTemp++}";
                _lines.Add($"    cvt.u64.u32 {wide}, {a};");
                if (b != "0")
                {
                    _lines.Add($"    cvt.u64.u32 {term}, {b};");
                    _lines.Add($"    add.u64 {wide}, {wide}, {term};");
                }
                if (c != "0")
                {
                    _lines.Add($"    cvt.u64.u32 {term}, {c};");
                    _lines.Add($"    add.u64 {wide}, {wide}, {term};");
                }
                if (pin != "0")
                {
                    _lines.Add($"    cvt.u64.u32 {term}, {pin};");
                    _lines.Add($"    add.u64 {wide}, {wide}, {term};");
                }
                if (destination != "0")
                    _lines.Add($"    cvt.u32.u64 {destination}, {wide};");
                if (carry)
                {
                    string high = $"%rdt{_nextTemp++}";
                    string high32 = $"%rt{_nextTemp++}";
                    string pred = $"%pt{_nextTemp++}";
                    string flag = $"%rt{_nextTemp++}";
                    string scratch = $"%rt{_nextTemp++}";
                    _lines.Add($"    shr.u64 {high}, {wide}, 32;");
                    _lines.Add($"    cvt.u32.u64 {high32}, {high};");
                    _lines.Add($"    setp.ne.u32 {pred}, {high32}, 0;");
                    _lines.Add($"    selp.u32 {flag}, 0xffffffff, 0, {pred};");
                    _lines.Add($"    add.cc.u32 {scratch}, {flag}, 1;");
                    _leaCarryValid = true;
                    string outPred = instruction.Operands[1].Trim();
                    if (Regex.IsMatch(outPred, @"^P\d+$",
                            RegexOptions.IgnoreCase))
                        _lines.Add($"    setp.ne.u32 {Pred(outPred)}, " +
                            $"{high32}, 0;");
                }
                return;
            }
            if (c == "0")
            {
                _lines.Add($"{prefix}add.s32 {destination}, {a}, {b};");
                return;
            }
            string sum = $"%rt{_nextTemp++}";
            _lines.Add($"    add.s32 {sum}, {a}, {b};");
            _lines.Add($"{prefix}add.s32 {destination}, {sum}, {c};");
        }

        private void EmitUadd3(SassInstruction instruction, string prefix)
        {
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            bool wide = mnemonic.Contains(".X");
            if (mnemonic.Contains(".64", StringComparison.Ordinal))
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs a paired uniform add");
            int first = 1;
            string? carryOut = null;
            if (first < instruction.Operands.Count &&
                Regex.IsMatch(instruction.Operands[first].Trim(),
                    @"^!?U?P\d+$|^!?U?PT$", RegexOptions.IgnoreCase))
            {
                string predicate = instruction.Operands[first].Trim();
                if (!predicate.StartsWith("!", StringComparison.Ordinal))
                    carryOut = Pred(predicate);
                first++;
            }
            if (first + 2 >= instruction.Operands.Count)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string destination = UniformOperand(instruction.Operands[0]);
            string a = UniformOperand(instruction.Operands[first]);
            string b = UniformOperand(instruction.Operands[first + 1]);
            string c = UniformOperand(instruction.Operands[first + 2]);
            if (wide)
            {
                // UIADD3.X dest, a, b, c, carry-in, carry-out.  The carry
                // predicate comes from the earlier UIADD3's carry-out.
                if (instruction.Operands.Count < 5)
                    throw new NotSupportedException(
                        $"SASS instruction '{instruction.Mnemonic}' needs a carry-in");
                string carryIn = Pred(instruction.Operands[4].Trim());
                string one = $"%rt{_nextTemp++}";
                string partial = $"%rt{_nextTemp++}";
                string sum = $"%rt{_nextTemp++}";
                _lines.Add($"    selp.u32 {one}, 1, 0, {carryIn};");
                _lines.Add($"    add.u32 {partial}, {a}, {b};");
                _lines.Add($"    add.u32 {sum}, {partial}, {c};");
                _lines.Add($"    add.u32 {sum}, {sum}, {one};");
                _lines.Add($"{prefix}mov.u32 {destination}, {sum};");
                return;
            }
            if (c == "0" && carryOut == null)
            {
                _lines.Add($"{prefix}add.u32 {destination}, {a}, {b};");
                return;
            }
            string low = $"%rt{_nextTemp++}";
            _lines.Add($"    add.u32 {low}, {a}, {b};");
            _lines.Add($"{prefix}add.u32 {destination}, {low}, {c};");
            if (carryOut != null)
            {
                string c1 = $"%pt{_nextTempPred++}";
                string c2 = $"%pt{_nextTempPred++}";
                _lines.Add($"    setp.lt.u32 {c1}, {low}, {a};");
                _lines.Add($"    setp.lt.u32 {c2}, {destination}, {low};");
                _lines.Add($"    or.pred {carryOut}, {c1}, {c2};");
            }
        }

        // UIMAD is the uniform-datapath integer multiply-add.  The plain
        // form is a 32-bit mad.lo; .WIDE/.X forms need the matching pair
        // and are rejected until a producer for them is seen.
        private void EmitUimad(SassInstruction instruction, string prefix)
        {
            if (instruction.Mnemonic.Contains(".WIDE",
                    StringComparison.OrdinalIgnoreCase))
            {
                EmitUimadWide(instruction, prefix);
                return;
            }
            if (!instruction.Mnemonic.Equals("UIMAD",
                    StringComparison.OrdinalIgnoreCase))
                throw new NotSupportedException(
                    $"unsupported UIMAD form '{instruction.Mnemonic}'");
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    "SASS instruction 'UIMAD' needs four operands");
            string d = UniformOperand(instruction.Operands[0]);
            string a = UniformOperand(instruction.Operands[1]);
            string b = UniformOperand(instruction.Operands[2]);
            string c = UniformOperand(instruction.Operands[3]);
            _lines.Add($"{prefix}mad.lo.u32 {d}, {a}, {b}, {c};");
        }

        // UIMAD.WIDE writes a 64-bit product into a uniform register pair;
        // the optional uniform predicate carries the high-half overflow.
        private void EmitUimadWide(SassInstruction instruction, string prefix)
        {
            int first = 1;
            if (first < instruction.Operands.Count &&
                Regex.IsMatch(instruction.Operands[first].Trim(),
                    @"^!?U?P\d+$|^!?U?PT$", RegexOptions.IgnoreCase))
                first++;
            if (first + 2 >= instruction.Operands.Count)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string destination = Regex.Replace(
                StripReuse(instruction.Operands[0]), @"\.CC$", "",
                RegexOptions.IgnoreCase);
            Match reg = Regex.Match(destination, @"^UR(\d+)$",
                RegexOptions.IgnoreCase);
            if (!reg.Success)
                throw new NotSupportedException(
                    $"unsupported UIMAD.WIDE destination " +
                    $"'{instruction.Operands[0]}'");
            string a = UniformOperand(instruction.Operands[first]);
            string b = UniformOperand(instruction.Operands[first + 1]);
            string c = instruction.Operands[first + 2].Trim();
            string product = $"%rdt{_nextTemp++}";
            _lines.Add($"{prefix}mul.wide.u32 {product}, {a}, {b};");
            if (!IsUniformZero(c))
            {
                string addend = $"%rdt{_nextTemp++}";
                _lines.Add($"    cvt.u64.u32 {addend}, {UniformOperand(c)};");
                _lines.Add($"    add.u64 {product}, {product}, {addend};");
            }
            int number = int.Parse(reg.Groups[1].Value,
                CultureInfo.InvariantCulture);
            _lines.Add($"    mov.b64 {{%ur{number}, %ur{number + 1}}}, " +
                $"{product};");
        }

        // USHF is the uniform-datapath funnel shift, with the same
        // operand order as the regular SHF forms.
        private void EmitUshf(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            bool left = mnemonic.Contains(".L", StringComparison.Ordinal);
            bool signed = mnemonic.Contains(".S32", StringComparison.Ordinal);
            string destination = UniformOperand(instruction.Operands[0]);
            string a = UniformOperand(instruction.Operands[1]);
            string shift = UniformOperand(instruction.Operands[2]);
            string c = UniformOperand(instruction.Operands[3]);
            if (!left && mnemonic.Contains(".HI", StringComparison.Ordinal))
            {
                if (signed && IsUniformZero(instruction.Operands[1]))
                {
                    _lines.Add($"{prefix}shr.s32 {destination}, {c}, {shift};");
                    return;
                }
                _lines.Add($"{prefix}shf.r.wrap.b32 {destination}, {c}, {a}, " +
                    $"{shift};");
                return;
            }
            string mode = left ? "l" : "r";
            string clamp = mnemonic.Contains(".C", StringComparison.Ordinal)
                ? "clamp" : "wrap";
            // PTX shf.l returns the high half of {b:a} << n and shf.r the
            // low half.  SASS USHF.L.U32 without .HI writes the low half,
            // so its funnel operands are exchanged; the .HI forms and
            // right shifts already line up.
            if (left && !mnemonic.Contains(".HI", StringComparison.Ordinal))
                (a, c) = (c, a);
            _lines.Add($"{prefix}shf.{mode}.{clamp}.b32 {destination}, {a}, " +
                $"{c}, {shift};");
        }

        // A few NVCC device-runtime helpers are linked as separate SASS
        // functions that are not present in the decoded entry section.
        // Their PTX-level ABI is fixed, so lower the ones the PyTorch RNG
        // kernel uses directly instead of failing the whole kernel.
        // The ptxas dsqrt slow-path helper is compiled with a per-section
        // register allocation, so the input/output/return registers cannot
        // be assumed.  Recover them from the helper body that lives in the
        // same text section as the call.
        private sealed class DsqrtHelperAbi
        {
            public string Input = "R32";
            public string Output = "R20";
            public string? Zero = "R39";
        }

        private static int RegNumber(string operand)
        {
            Match match = Regex.Match(StripReuse(operand),
                @"^R(\d+)$", RegexOptions.IgnoreCase);
            return match.Success
                ? int.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture)
                : -1;
        }

        private DsqrtHelperAbi? FindDsqrtHelperAbi(string symbol)
        {
            List<SassNode> nodes = _section.Nodes.ToList();
            int start = -1;
            for (int i = 0; i < nodes.Count; i++)
            {
                if (nodes[i] is SassLabel label &&
                    label.Name.Contains(symbol, StringComparison.Ordinal))
                {
                    start = i;
                    break;
                }
            }
            if (start < 0)
                return null;
            int end = nodes.Count;
            for (int i = start + 1; i < nodes.Count; i++)
            {
                if (nodes[i] is SassLabel boundary &&
                    !boundary.Name.StartsWith(".L_", StringComparison.Ordinal))
                {
                    end = i;
                    break;
                }
            }
            var body = new List<SassInstruction>();
            for (int i = start + 1; i < end; i++)
                if (nodes[i] is SassInstruction ins)
                    body.Add(ins);
            if (body.Count == 0)
                return null;

            // Input pair: the helper's NaN test compares the working copy
            // of the argument against zero.  The entry may first copy the
            // real input into that working pair.
            int input = -1;
            foreach (SassInstruction ins in body)
            {
                if (!ins.Mnemonic.StartsWith("DSETP", StringComparison.OrdinalIgnoreCase))
                    continue;
                for (int o = 2; o + 1 < ins.Operands.Count; o++)
                {
                    if (!StripReuse(ins.Operands[o + 1])
                            .Equals("RZ", StringComparison.OrdinalIgnoreCase))
                        continue;
                    int candidate = RegNumber(ins.Operands[o]);
                    if (candidate >= 0)
                    {
                        input = candidate;
                        break;
                    }
                }
                if (input >= 0)
                    break;
            }
            if (input < 0)
                return null;
            foreach (SassInstruction ins in body.Take(8))
            {
                if (!(ins.Mnemonic.StartsWith("MOV", StringComparison.OrdinalIgnoreCase) ||
                      ins.Mnemonic.StartsWith("IMAD.MOV", StringComparison.OrdinalIgnoreCase)))
                    continue;
                if (ins.Operands.Count < 2 ||
                    RegNumber(ins.Operands[0]) != input)
                    continue;
                int source = RegNumber(ins.Operands[^1]);
                if (source >= 0)
                {
                    input = source;
                    break;
                }
            }

            // Output pair: prefer the adjacent even/odd pair written in
            // the final block before RET.  Otherwise use the destination
            // of the NaN/infinity path's DADD (the helper's slow path).
            int output = -1;
            var tailWrites = new List<int>();
            for (int i = Math.Max(0, body.Count - 6); i < body.Count; i++)
            {
                SassInstruction ins = body[i];
                if (ins.Mnemonic.StartsWith("RET", StringComparison.OrdinalIgnoreCase))
                    continue;
                int dst = ins.Operands.Count > 0 ? RegNumber(ins.Operands[0]) : -1;
                if (dst >= 0)
                    tailWrites.Add(dst);
            }
            int even = tailWrites.FirstOrDefault(r => (r & 1) == 0, -1);
            if (even >= 0 && tailWrites.Contains(even + 1))
                output = even;
            if (output < 0)
            {
                foreach (SassInstruction ins in body)
                {
                    if (!ins.Mnemonic.StartsWith("DADD", StringComparison.OrdinalIgnoreCase) ||
                        ins.Operands.Count < 4)
                        continue;
                    if (!StripReuse(ins.Operands[2]).Equals(
                            StripReuse(ins.Operands[3]),
                            StringComparison.OrdinalIgnoreCase))
                        continue;
                    int dst = RegNumber(ins.Operands[0]);
                    if (dst >= 0)
                    {
                        output = dst & ~1;
                        break;
                    }
                }
            }
            if (output < 0)
                return null;

            // The helper zeroes a scratch flag for the caller before RET.
            int zero = -1;
            for (int i = body.Count - 1; i >= Math.Max(0, body.Count - 6); i--)
            {
                SassInstruction ins = body[i];
                if (!(ins.Mnemonic.StartsWith("MOV", StringComparison.OrdinalIgnoreCase) ||
                      ins.Mnemonic.StartsWith("IMAD.MOV", StringComparison.OrdinalIgnoreCase)))
                    continue;
                int dst = RegNumber(ins.Operands[0]);
                string last = StripReuse(ins.Operands[^1]);
                if (dst >= 0 && dst != output && dst != output + 1 &&
                    (last == "0" || last == "RZ" ||
                     last.Equals("0x0", StringComparison.OrdinalIgnoreCase)))
                {
                    zero = dst;
                    break;
                }
            }

            return new DsqrtHelperAbi
            {
                Input = "R" + input.ToString(CultureInfo.InvariantCulture),
                Output = "R" + output.ToString(CultureInfo.InvariantCulture),
                Zero = zero >= 0
                    ? "R" + zero.ToString(CultureInfo.InvariantCulture)
                    : null,
            };
        }

        // libdevice's __cudart_i2opi_d: the binary expansion of 2/pi in
        // 64-bit limbs, least significant first.  Used by the f64
        // Payne-Hanek range reduction slow path.
        private static readonly ulong[] TrigI2opiD =
        {
            7780917995555872008UL, 4397547296490951402UL,
            8441921394348257659UL, 5712322887342352941UL,
            7869616827067468215UL,
            unchecked((ulong)-1211730484530615009L),
            2303758334597371919UL,
            unchecked((ulong)-7168499653074671557L),
            4148332274289687028UL,
            unchecked((ulong)-1613291254968254911L),
            unchecked((ulong)-1692731182770600828L),
            unchecked((ulong)-135693905287338178L),
            452944820249399836UL,
            unchecked((ulong)-5249950069107600672L),
            unchecked((ulong)-121206125134887583L),
            unchecked((ulong)-2638381946312093631L),
            unchecked((ulong)-277156292786332224L),
            unchecked((ulong)-6703182060581546711L),
        };

        private sealed record TrigReductionAbi(
            int InputLow, int InputHigh, int OutputLow, int OutputHigh,
            int Quadrant);

        private string U64Temp() => $"%rdt{_nextTemp++}";

        private string U32Temp() => $"%rt{_nextTemp++}";

        // A relocated constant-bank slot holds the address of a module
        // data object (e.g. __cudart_sin_cos_coeffs).  Emit the symbol
        // address so ACuda can track the `.global` image home.
        private bool TryGlobalImageOperand(
            string text, out string? address)
        {
            address = null;
            Match match = Regex.Match(text,
                @"^c\[0x([0-9A-Fa-f]+)\]\[0x([0-9A-Fa-f]+)\]$",
                RegexOptions.IgnoreCase);
            if (!match.Success)
                return false;
            int bank = int.Parse(match.Groups[1].Value,
                NumberStyles.HexNumber, CultureInfo.InvariantCulture);
            // The relocation map is for bank 4 only; bank 2 holds math
            // constants whose offsets would otherwise collide.
            if (bank != 4)
                return false;
            int offset = int.Parse(match.Groups[2].Value,
                NumberStyles.HexNumber, CultureInfo.InvariantCulture);
            if (!_globalImages.TryGetValue(offset, out PtxGlobalImage? image))
                return false;
            string value = U64Temp();
            _lines.Add($"    mov.u64 {value}, {image.Name};");
            if (image.Addend != 0)
            {
                string addend = U64Temp();
                _lines.Add($"    mov.u64 {addend}, " +
                    $"0x{unchecked((ulong)image.Addend):x16};");
                _lines.Add($"    add.u64 {value}, {value}, {addend};");
            }
            address = value;
            return true;
        }

        private string PredTemp() => $"%pt{_nextTempPred++}";

        private bool DetectTrigSlowpathUse()
        {
            foreach (SassSection section in new[] { _section }
                         .Concat(_linkedSections))
            {
                foreach (SassNode node in section.Nodes)
                {
                    if (node is not SassInstruction instruction ||
                        !IsCallMnemonic(instruction.Mnemonic))
                        continue;
                    string? symbol = instruction.Relocations
                        .FirstOrDefault(relocation =>
                            !string.IsNullOrWhiteSpace(relocation.Symbol))
                        ?.Symbol;
                    if (string.IsNullOrEmpty(symbol))
                        symbol = instruction.BranchTargets.FirstOrDefault()
                            ?? instruction.Operands.LastOrDefault();
                    if (symbol != null && symbol.Contains(
                            "__internal_trig_reduction_slowpathd",
                            StringComparison.Ordinal))
                        return true;
                }
            }
            return false;
        }

        private void EmitGlobalImages()
        {
            if (_globalImages.Count == 0)
                return;
            // The cubin's .rel.nv.constant4 covers the whole module, so
            // it also lists unrelated kernels' data (cuRAND tables,
            // strings, ...).  Emit only the images this entry actually
            // dereferences; a stray 100 KiB table would otherwise blow
            // the SPIR-V constant pool.
            var used = new HashSet<int>();
            foreach (SassSection section in new[] { _section }
                         .Concat(_linkedSections))
            {
                foreach (SassNode node in section.Nodes)
                {
                    if (node is not SassInstruction instruction)
                        continue;
                    foreach (string operand in instruction.Operands)
                    {
                        Match match = Regex.Match(operand,
                            @"c\[0x4\]\[0x([0-9A-Fa-f]+)\]",
                            RegexOptions.IgnoreCase);
                        if (match.Success)
                            used.Add(int.Parse(match.Groups[1].Value,
                                NumberStyles.HexNumber,
                                CultureInfo.InvariantCulture));
                    }
                }
            }
            var emitted = new HashSet<string>(StringComparer.Ordinal);
            foreach (KeyValuePair<int, PtxGlobalImage> entry in _globalImages)
            {
                if (!used.Contains(entry.Key))
                    continue;
                PtxGlobalImage image = entry.Value;
                if (!emitted.Add(image.Name))
                    continue;
                int size = image.Data.Length;
                if (size == 0)
                    size = 1;
                _lines.Add($".global .align 8 .b8 {image.Name}[{size}] = {{");
                for (int i = 0; i < image.Data.Length; i++)
                {
                    string comma = i + 1 < image.Data.Length ? "," : "";
                    _lines.Add($"    0x{image.Data[i]:x2}{comma}");
                }
                _lines.Add("};");
            }
        }

        // The slow path is a leaf compiled by ptxas; its register
        // convention differs per kernel.  Recover it from the body:
        //   - the first `SHF.R.U32.HI d, RZ, 0x14, H` names the input
        //     high word H;
        //   - `SHF.L.U64.HI d, L, 0xb, H` names the input low word L
        //     (possibly through a MOV copy);
        //   - the quadrant is initialized to zero and later negated;
        //   - the returned double is the last `IADD3 ..., -UR4` low word
        //     and the last sign-OR (`LOP3 ..., 0xfc`) high word, after
        //     the tail shuffles.
        private TrigReductionAbi? FindTrigReductionAbi(string symbol)
        {
            foreach (SassSection section in new[] { _section }
                         .Concat(_linkedSections))
            {
                List<SassNode> nodes = section.Nodes.ToList();
                int start = -1;
                for (int i = 0; i < nodes.Count; i++)
                {
                    if (nodes[i] is SassLabel label &&
                        (label.Name.Equals(symbol, StringComparison.Ordinal) ||
                         label.Name.EndsWith(
                             "$__internal_trig_reduction_slowpathd",
                             StringComparison.Ordinal)))
                    {
                        start = i;
                        break;
                    }
                }
                if (start < 0)
                    continue;
                int end = nodes.Count;
                for (int i = start + 1; i < nodes.Count; i++)
                {
                    if (nodes[i] is SassLabel boundary &&
                        !boundary.Name.StartsWith(".L_", StringComparison.Ordinal))
                    {
                        end = i;
                        break;
                    }
                }
                var instructions = new List<SassInstruction>();
                foreach (SassNode node in nodes.Skip(start + 1).Take(end - start - 1))
                    if (node is SassInstruction instruction)
                        instructions.Add(instruction);
                if (instructions.Count == 0)
                    continue;

                int retReg = -1;
                for (int i = instructions.Count - 1; i >= 0; i--)
                {
                    if (instructions[i].Mnemonic.StartsWith(
                            "RET", StringComparison.OrdinalIgnoreCase))
                    {
                        foreach (string operand in instructions[i].Operands)
                        {
                            retReg = RegNumber(operand.Trim('`', '(', ')'));
                            if (retReg >= 0)
                                break;
                        }
                        break;
                    }
                }
                if (retReg < 0)
                    continue;

                int inputHigh = -1;
                int lowSourceIndex = -1;
                int lowSourceReg = -1;
                int highSourceReg = -1;
                for (int i = 0; i < instructions.Count; i++)
                {
                    SassInstruction ins = instructions[i];
                    if (inputHigh < 0 &&
                        ins.Mnemonic.StartsWith("SHF.R.U32.HI",
                            StringComparison.OrdinalIgnoreCase) &&
                        ins.Operands.Count >= 4 &&
                        StripReuse(ins.Operands[1]).Equals(
                            "RZ", StringComparison.OrdinalIgnoreCase) &&
                        ins.Operands[2].Trim().Equals(
                            "0x14", StringComparison.OrdinalIgnoreCase))
                    {
                        int candidate = RegNumber(ins.Operands[3]);
                        if (candidate >= 0)
                            inputHigh = candidate;
                    }
                    if (lowSourceIndex < 0 &&
                        ins.Mnemonic.StartsWith("SHF.L.U64.HI",
                            StringComparison.OrdinalIgnoreCase) &&
                        ins.Operands.Count >= 4 &&
                        ins.Operands[2].Trim().Equals(
                            "0xb", StringComparison.OrdinalIgnoreCase))
                    {
                        lowSourceIndex = i;
                        lowSourceReg = RegNumber(ins.Operands[1]);
                        highSourceReg = RegNumber(ins.Operands[3]);
                    }
                }
                if (inputHigh < 0 && highSourceReg >= 0)
                    inputHigh = highSourceReg;
                if (inputHigh < 0 || lowSourceIndex < 0)
                    continue;

                // Trace the low operand through MOV copies to the
                // live-in register the caller filled.
                int inputLow = TraceInputRegister(
                    instructions, lowSourceIndex, lowSourceReg);
                if (inputLow < 0)
                    inputLow = lowSourceReg;
                if (inputHigh == lowSourceReg)
                {
                    int tracedHigh = TraceInputRegister(
                        instructions, lowSourceIndex, highSourceReg);
                    if (tracedHigh >= 0)
                        inputHigh = tracedHigh;
                }

                int quadrantInternal = -1;
                int quadrantNegateIndex = -1;
                for (int i = 0; i < instructions.Count; i++)
                {
                    SassInstruction ins = instructions[i];
                    string mnemonic = ins.Mnemonic.ToUpperInvariant();
                    if (!(mnemonic.StartsWith("IMAD.MOV",
                              StringComparison.Ordinal) ||
                          mnemonic.StartsWith("IADD3",
                              StringComparison.Ordinal)))
                        continue;
                    if (ins.Operands.Count < 4)
                        continue;
                    int dst = RegNumber(ins.Operands[0]);
                    if (dst < 0)
                        continue;
                    bool negated =
                        StripReuse(ins.Operands[^1]).Equals(
                            "-R" + dst, StringComparison.OrdinalIgnoreCase) ||
                        (ins.Operands.Count >= 2 &&
                         StripReuse(ins.Operands[1]).Equals(
                             "-R" + dst,
                             StringComparison.OrdinalIgnoreCase));
                    if (negated)
                    {
                        quadrantInternal = dst;
                        quadrantNegateIndex = i;
                    }
                }
                if (quadrantInternal < 0)
                    continue;
                int quadrant = quadrantInternal;
                for (int i = quadrantNegateIndex + 1;
                     i < instructions.Count; i++)
                {
                    SassInstruction ins = instructions[i];
                    string mnemonic = ins.Mnemonic.ToUpperInvariant();
                    if (!mnemonic.StartsWith("MOV",
                            StringComparison.Ordinal) &&
                        !mnemonic.StartsWith("IMAD.MOV",
                            StringComparison.Ordinal))
                        continue;
                    if (ins.Operands.Count < 2)
                        continue;
                    if (MovSource(ins) == quadrantInternal)
                        quadrant = RegNumber(ins.Operands[0]);
                }

                int lowInternal = -1;
                int highInternal = -1;
                int lowInternalIndex = -1;
                for (int i = 0; i < instructions.Count; i++)
                {
                    SassInstruction ins = instructions[i];
                    string mnemonic = ins.Mnemonic.ToUpperInvariant();
                    if (mnemonic.StartsWith("IADD3", StringComparison.Ordinal) &&
                        ins.Operands.Any(operand =>
                            StripReuse(operand).Equals(
                                "-UR4", StringComparison.OrdinalIgnoreCase)))
                    {
                        lowInternal = RegNumber(ins.Operands[0]);
                        lowInternalIndex = i;
                    }
                    if (mnemonic.StartsWith("LOP3.LUT",
                            StringComparison.Ordinal) &&
                        ins.Operands.Any(operand =>
                            StripReuse(operand).Equals(
                                "0xfc", StringComparison.OrdinalIgnoreCase)))
                    {
                        highInternal = RegNumber(ins.Operands[0]);
                    }
                }
                if (lowInternal < 0 || highInternal < 0)
                    continue;

                // Apply the tail MOV shuffles to both returned words.
                int outputLow = lowInternal;
                int outputHigh = highInternal;
                for (int i = Math.Max(lowInternalIndex, 0);
                     i < instructions.Count; i++)
                {
                    SassInstruction ins = instructions[i];
                    string mnemonic = ins.Mnemonic.ToUpperInvariant();
                    if (!mnemonic.StartsWith("MOV", StringComparison.Ordinal) &&
                        !mnemonic.StartsWith("IMAD.MOV",
                            StringComparison.Ordinal))
                        continue;
                    if (ins.Operands.Count < 2)
                        continue;
                    int dst = RegNumber(ins.Operands[0]);
                    int src = MovSource(ins);
                    if (dst < 0 || src < 0)
                        continue;
                    if (src == outputLow)
                        outputLow = dst;
                    if (src == outputHigh)
                        outputHigh = dst;
                }
                if (outputLow < 0 || outputHigh < 0 ||
                    outputLow == outputHigh)
                    continue;
                return new TrigReductionAbi(
                    inputLow, inputHigh, outputLow, outputHigh, quadrant);
            }
            return null;
        }

        // MOV-family source register: IMAD.MOV.U32 d, RZ, RZ, s keeps
        // its source in the last operand.
        private static int MovSource(SassInstruction instruction)
        {
            if (instruction.Operands.Count < 2)
                return -1;
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            if (mnemonic.StartsWith("IMAD.MOV", StringComparison.Ordinal))
                return RegNumber(instruction.Operands[^1]);
            if (mnemonic.StartsWith("MOV", StringComparison.Ordinal))
                return RegNumber(instruction.Operands[1]);
            return -1;
        }

        private static bool WritesRegister(
            SassInstruction instruction, int register)
        {
            if (instruction.Operands.Count == 0)
                return false;
            return RegNumber(instruction.Operands[0]) == register;
        }

        // Finds the live-in register whose value reaches `register` at
        // `index` through a chain of MOV copies.
        private static int TraceInputRegister(
            List<SassInstruction> instructions, int index, int register)
        {
            int current = register;
            for (int guard = 0; guard < 8 && current >= 0; guard++)
            {
                int writeIndex = -1;
                int source = -1;
                for (int i = index - 1; i >= 0; i--)
                {
                    SassInstruction ins = instructions[i];
                    if (!WritesRegister(ins, current))
                        continue;
                    writeIndex = i;
                    source = MovSource(ins);
                    break;
                }
                if (writeIndex < 0 || source < 0)
                    return current;
                current = source;
                index = writeIndex;
            }
            return current;
        }

        // Emits one step of the fixed-point limb multiply-add:
        // (lo, p) = word[q+k] * mantissa + p.
        private void EmitTrigMulAdd(
            string table, string q, int offset, string mantissa, string p,
            string loOut)
        {
            string index = U32Temp();
            if (offset == 0)
                _lines.Add($"    mov.u32 {index}, {q};");
            else
                _lines.Add($"    add.u32 {index}, {q}, {offset};");
            string address = U64Temp();
            _lines.Add($"    mad.wide.u32 {address}, {index}, 8, {table};");
            string word = U64Temp();
            _lines.Add($"    ld.const.u64 {word}, [{address}];");
            string lo = U64Temp();
            string high = U64Temp();
            _lines.Add($"    mul.lo.u64 {lo}, {word}, {mantissa};");
            _lines.Add($"    mul.hi.u64 {high}, {word}, {mantissa};");
            _lines.Add($"    add.cc.u64 {loOut}, {lo}, {p};");
            _lines.Add($"    addc.u64 {p}, {high}, 0;");
        }

        // Semantic lowering of libdevice's f64 Payne-Hanek range
        // reduction.  The compiled body is a leaf whose register numbers
        // differ per kernel, so the ABI is recovered and the algorithm
        // is re-expressed in PTX (see FindTrigReductionAbi).
        private void EmitTrigReductionSlowpath(string symbol)
        {
            TrigReductionAbi? found = FindTrigReductionAbi(symbol);
            if (found == null)
                throw new NotSupportedException(
                    "cannot recover the register ABI of device helper " +
                    $"'{symbol}'");
            TrigReductionAbi abi = found;
            string inputLow = Reg("R" + abi.InputLow);
            string inputHigh = Reg("R" + abi.InputHigh);
            string outputLow = Reg("R" + abi.OutputLow);
            string outputHigh = Reg("R" + abi.OutputHigh);
            string quadrant = Reg("R" + abi.Quadrant);

            int serial = _nextTemp++;
            string special = $"L_trig_special_{serial}";
            string skipFourth = $"L_trig_skip4_{serial}";
            string noShift = $"L_trig_noshift_{serial}";
            string afterShift = $"L_trig_shift_{serial}";
            string afterNormalize = $"L_trig_norm_{serial}";
            string afterDouble = $"L_trig_double_{serial}";
            string afterNegate = $"L_trig_negate_{serial}";
            string done = $"L_trig_done_{serial}";

            string a = U64Temp();
            _lines.Add($"    mov.b64 {a}, {{{inputLow}, {inputHigh}}};");

            string hiWord = U32Temp();
            _lines.Add($"    shr.u32 {hiWord}, {inputHigh}, 20;");
            string exponent = U32Temp();
            _lines.Add($"    and.b32 {exponent}, {hiWord}, 0x7ff;");
            string specialPredicate = PredTemp();
            _lines.Add(
                $"    setp.eq.u32 {specialPredicate}, {exponent}, 0x7ff;");
            _lines.Add($"    @{specialPredicate} bra {special};");

            string e2 = U32Temp();
            _lines.Add($"    sub.u32 {e2}, {exponent}, 0x400;");
            string mantissa = U64Temp();
            _lines.Add($"    shl.b64 {mantissa}, {a}, 11;");
            _lines.Add(
                $"    or.b64 {mantissa}, {mantissa}, 0x8000000000000000;");
            string shiftIndex = U32Temp();
            _lines.Add($"    shr.u32 {shiftIndex}, {e2}, 6;");
            string baseIndex = U32Temp();
            _lines.Add($"    sub.u32 {baseIndex}, 16, {shiftIndex};");
            string q = U32Temp();
            _lines.Add($"    sub.u32 {q}, {baseIndex}, 1;");
            string hasFourth = PredTemp();
            _lines.Add(
                $"    setp.le.u32 {hasFourth}, {baseIndex}, 15;");

            string table = U64Temp();
            _lines.Add($"    mov.u64 {table}, __rtx_trig_i2opi_d;");
            string p = U64Temp();
            _lines.Add($"    mov.u64 {p}, 0;");
            string[] lows = new string[4];
            for (int k = 0; k < 3; k++)
            {
                lows[k] = U64Temp();
                EmitTrigMulAdd(table, q, k, mantissa, p, lows[k]);
            }
            string result3 = U64Temp();
            _lines.Add($"    mov.b64 {result3}, {p};");
            _lines.Add($"    @!{hasFourth} bra {skipFourth};");
            lows[3] = U64Temp();
            EmitTrigMulAdd(table, q, 3, mantissa, p, lows[3]);
            _lines.Add($"    mov.b64 {result3}, {lows[3]};");
            _lines.Add($"{skipFourth}:");

            string shift = U32Temp();
            _lines.Add($"    and.b32 {shift}, {e2}, 63;");
            string noShiftPredicate = PredTemp();
            _lines.Add($"    setp.eq.u32 {noShiftPredicate}, {shift}, 0;");
            string hiBits = U64Temp();
            string loBits = U64Temp();
            _lines.Add($"    @{noShiftPredicate} bra {noShift};");
            string inverseShift = U32Temp();
            _lines.Add($"    sub.u32 {inverseShift}, 64, {shift};");
            string scratch = U64Temp();
            _lines.Add($"    shl.b64 {hiBits}, {result3}, {shift};");
            _lines.Add($"    shr.u64 {scratch}, {lows[2]}, {inverseShift};");
            _lines.Add($"    or.b64 {hiBits}, {hiBits}, {scratch};");
            _lines.Add($"    shl.b64 {loBits}, {lows[2]}, {shift};");
            _lines.Add($"    shr.u64 {scratch}, {lows[1]}, {inverseShift};");
            _lines.Add($"    or.b64 {loBits}, {loBits}, {scratch};");
            _lines.Add($"    bra {afterShift};");
            _lines.Add($"{noShift}:");
            _lines.Add($"    mov.b64 {hiBits}, {result3};");
            _lines.Add($"    mov.b64 {loBits}, {lows[2]};");
            _lines.Add($"{afterShift}:");

            string top = U32Temp();
            _lines.Add($"    shr.u64 {scratch}, {hiBits}, 62;");
            _lines.Add($"    cvt.u32.u64 {top}, {scratch};");
            string x = U64Temp();
            string y = U64Temp();
            _lines.Add($"    shl.b64 {x}, {hiBits}, 2;");
            _lines.Add($"    shr.u64 {scratch}, {loBits}, 62;");
            _lines.Add($"    or.b64 {x}, {x}, {scratch};");
            _lines.Add($"    shl.b64 {y}, {loBits}, 2;");
            string bit = U32Temp();
            _lines.Add($"    shr.u64 {scratch}, {x}, 63;");
            _lines.Add($"    cvt.u32.u64 {bit}, {scratch};");
            string quadrantValue = U32Temp();
            _lines.Add($"    add.u32 {quadrantValue}, {top}, {bit};");
            string signMask = U32Temp();
            _lines.Add(
                $"    and.b32 {signMask}, {inputHigh}, 0x80000000;");
            string negative = PredTemp();
            _lines.Add($"    setp.ne.u32 {negative}, {signMask}, 0;");
            _lines.Add(
                $"    @{negative} sub.u32 {quadrantValue}, 0, " +
                $"{quadrantValue};");

            string bitPredicate = PredTemp();
            _lines.Add($"    setp.ne.u32 {bitPredicate}, {bit}, 0;");
            _lines.Add($"    @!{bitPredicate} bra {afterNegate};");
            string negX = U64Temp();
            string negY = U64Temp();
            _lines.Add($"    not.b64 {negX}, {x};");
            _lines.Add($"    not.b64 {negY}, {y};");
            _lines.Add($"    add.cc.u64 {negY}, {negY}, 1;");
            _lines.Add($"    addc.u64 {negX}, {negX}, 0;");
            _lines.Add($"    mov.b64 {x}, {negX};");
            _lines.Add($"    mov.b64 {y}, {negY};");
            _lines.Add(
                $"    xor.b32 {signMask}, {signMask}, 0x80000000;");
            _lines.Add($"{afterNegate}:");

            string leadingZeros = U32Temp();
            _lines.Add($"    clz.b64 {leadingZeros}, {x};");
            string normalized = U64Temp();
            _lines.Add($"    mov.b64 {normalized}, {x};");
            string zeroShift = PredTemp();
            _lines.Add(
                $"    setp.eq.u32 {zeroShift}, {leadingZeros}, 0;");
            _lines.Add($"    @{zeroShift} bra {afterNormalize};");
            string inverseZeros = U32Temp();
            _lines.Add(
                $"    sub.u32 {inverseZeros}, 64, {leadingZeros};");
            _lines.Add(
                $"    shl.b64 {normalized}, {x}, {leadingZeros};");
            _lines.Add(
                $"    shr.u64 {scratch}, {y}, {inverseZeros};");
            _lines.Add(
                $"    or.b64 {normalized}, {normalized}, {scratch};");
            _lines.Add($"{afterNormalize}:");

            string multiplier = U64Temp();
            _lines.Add(
                $"    mov.u64 {multiplier}, 0xc90fdaa22168c235;");
            string productLow = U64Temp();
            string productHigh = U64Temp();
            _lines.Add(
                $"    mul.lo.u64 {productLow}, {normalized}, {multiplier};");
            _lines.Add(
                $"    mul.hi.u64 {productHigh}, {normalized}, {multiplier};");
            string productPositive = PredTemp();
            _lines.Add(
                $"    setp.gt.s64 {productPositive}, {productHigh}, 0;");
            string scaled = U64Temp();
            _lines.Add($"    mov.b64 {scaled}, {productHigh};");
            string exponentOrder = U32Temp();
            _lines.Add($"    mov.u32 {exponentOrder}, {leadingZeros};");
            _lines.Add($"    @!{productPositive} bra {afterDouble};");
            _lines.Add($"    add.cc.u64 {scratch}, {productLow}, {productLow};");
            _lines.Add($"    addc.u64 {scaled}, {productHigh}, {productHigh};");
            _lines.Add($"    add.u32 {exponentOrder}, {leadingZeros}, 1;");
            _lines.Add($"{afterDouble}:");

            string resultExponent = U32Temp();
            _lines.Add(
                $"    sub.u32 {resultExponent}, 1022, {exponentOrder};");
            string resultBits = U64Temp();
            _lines.Add($"    cvt.u64.u32 {resultBits}, {resultExponent};");
            _lines.Add($"    shl.b64 {resultBits}, {resultBits}, 52;");
            string significand = U64Temp();
            _lines.Add($"    add.u64 {significand}, {scaled}, 1;");
            _lines.Add($"    shr.u64 {significand}, {significand}, 10;");
            _lines.Add($"    add.u64 {significand}, {significand}, 1;");
            _lines.Add($"    shr.u64 {significand}, {significand}, 1;");
            _lines.Add($"    add.u64 {resultBits}, {resultBits}, {significand};");
            string signBits = U64Temp();
            _lines.Add($"    cvt.u64.u32 {signBits}, {signMask};");
            _lines.Add($"    shl.b64 {signBits}, {signBits}, 32;");
            _lines.Add($"    or.b64 {resultBits}, {resultBits}, {signBits};");
            _lines.Add(
                $"    mov.b64 {{{outputLow}, {outputHigh}}}, {resultBits};");
            _lines.Add($"    mov.u32 {quadrant}, {quadrantValue};");
            _lines.Add($"    bra {done};");

            _lines.Add($"{special}:");
            _lines.Add($"    mov.b64 {{{outputLow}, {outputHigh}}}, {a};");
            _lines.Add($"    mov.u32 {quadrant}, 0;");
            _lines.Add($"{done}:");
        }

        private void EmitExternalHelperCall(SassInstruction instruction, string prefix)
        {
            string? symbol = instruction.Relocations
                .FirstOrDefault(relocation =>
                    !string.IsNullOrWhiteSpace(relocation.Symbol))
                ?.Symbol;
            if (string.IsNullOrEmpty(symbol))
            {
                // Text-mode SASS has no relocation table; the CALL operand
                // still names the target symbol.
                symbol = (instruction.BranchTargets.FirstOrDefault()
                    ?? instruction.Operands.LastOrDefault() ?? "")
                    .Trim().Trim('`', '(', ')', ';', ' ');
                // Local `.L_` blocks still go through the inline lifter;
                // a call whose callee cannot be inlined must fail rather
                // than silently become a branch.
                if (symbol.Length == 0 ||
                    symbol.StartsWith(".L_", StringComparison.Ordinal))
                    symbol = null;
            }
            // Compiler-generated out-of-line block: ptxas places a cold
            // predicate block after the hot path and reaches it with a
            // relative call.  The block rejoins the surrounding control
            // flow (or exits) rather than returning through the call
            // stack, so lower the call as the branch it actually is.
            if (symbol != null &&
                symbol.StartsWith(".L_", StringComparison.Ordinal))
            {
                _lines.Add($"{prefix}bra {LabelName(symbol)};");
                return;
            }
            if (symbol != null &&
                symbol.Contains("__internal_trig_reduction_slowpathd",
                    StringComparison.Ordinal))
            {
                EmitTrigReductionSlowpath(symbol);
                return;
            }
            if (symbol != null &&
                symbol.Contains("__cuda_sm20_div_s64",
                    StringComparison.Ordinal))
            {
                // ABI used by this helper: dividend UR6:UR7, divisor UR4,
                // quotient R16:R17 (the same registers the inlined path
                // uses for the 64-bit result).
                string dividend = $"%rdt{_nextTemp++}";
                string divisor = $"%rdt{_nextTemp++}";
                string quotient = $"%rdt{_nextTemp++}";
                _lines.Add($"    mov.b64 {dividend}, " +
                    $"{{{UniformReg("UR6")}, {UniformReg("UR7")}}};");
                _lines.Add($"    cvt.u64.u32 {divisor}, {UniformReg("UR4")};");
                _lines.Add($"    div.u64 {quotient}, {dividend}, {divisor};");
                _lines.Add($"    mov.b64 {{{Reg("R16")}, {Reg("R17")}}}, " +
                    $"{quotient};");
                return;
            }
            if (symbol != null &&
                symbol.Contains("__cuda_sm20_sqrt_rn_f32_slowpath",
                    StringComparison.Ordinal))
            {
                // Input R33, result R35 for this slow-path helper.
                string value = $"%ft{_nextTempFloat++}";
                _lines.Add($"    mov.b32 {value}, {Reg("R33")};");
                _lines.Add($"    sqrt.rn.f32 {value}, {value};");
                _lines.Add($"    mov.b32 {Reg("R35")}, {value};");
                return;
            }
            if (symbol != null &&
                symbol.Contains("dsqrt_rn_f64_mediumpath",
                    StringComparison.Ordinal))
            {
                DsqrtHelperAbi abi = FindDsqrtHelperAbi(symbol)
                    ?? new DsqrtHelperAbi();
                int input = RegNumber(abi.Input);
                int output = RegNumber(abi.Output);
                string value = $"%rdt{_nextTemp++}";
                _lines.Add($"    mov.b64 {value}, " +
                    $"{{{Reg("R" + input)}, {Reg("R" + (input + 1))}}};");
                _lines.Add($"    sqrt.rn.f64 {value}, {value};");
                _lines.Add(
                    $"    mov.b64 {{{Reg("R" + output)}, " +
                    $"{Reg("R" + (output + 1))}}}, {value};");
                if (abi.Zero != null)
                    _lines.Add($"    mov.b32 {Reg(abi.Zero)}, 0;");
                return;
            }
            if (symbol != null &&
                symbol.Contains("div_rn_f64_full", StringComparison.Ordinal) &&
                _dispatchHelperLabel != null)
            {
                uint returnAddress = (uint)(instruction.Address + 16);
                string returnLabel = $"L_helper_ret_{_nextTemp++}";
                _helperDispatchReturns[returnAddress] = returnLabel;
                _lines.Add($"{prefix}bra {_dispatchHelperLabel};");
                _lines.Add($"{returnLabel}:");
                return;
            }
            throw new NotSupportedException(
                $"SASS instruction '{instruction.Mnemonic}' calls unsupported " +
                $"device helper '{symbol}'");
        }

        private void EmitUadd3Wide(SassInstruction instruction, string prefix)
        {
            if (instruction.Mnemonic.Contains(".X", StringComparison.OrdinalIgnoreCase))
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs a paired uniform carry");
            if (instruction.Operands.Count < 6)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs six operands");
            string destination = Regex.Match(instruction.Operands[0].Trim(),
                @"^UR(\d+)$", RegexOptions.IgnoreCase).Groups[1].Value;
            if (destination.Length == 0)
                throw new NotSupportedException(
                    $"unsupported UIADD3.64 destination '{instruction.Operands[0]}'");
            string a = PackUniformPair(instruction.Operands[3]);
            string b = PackUniformPair(instruction.Operands[4]);
            string c = PackUniformPair(instruction.Operands[5]);
            string sum = $"%rdt{_nextTemp++}";
            _lines.Add($"    add.u64 {sum}, {a}, {b};");
            _lines.Add($"    add.u64 {sum}, {sum}, {c};");
            int number = int.Parse(destination, CultureInfo.InvariantCulture);
            _lines.Add($"    mov.b64 {{%ur{number}, %ur{number + 1}}}, {sum};");
        }

        private string PackUniformPair(string operand)
        {
            string text = StripReuse(operand);
            if (text.Equals("URZ", StringComparison.OrdinalIgnoreCase))
            {
                string zero = $"%rdt{_nextTemp++}";
                _lines.Add($"    mov.b64 {zero}, 0;");
                return zero;
            }
            Match match = Regex.Match(text, @"^UR(\d+)$",
                RegexOptions.IgnoreCase);
            if (!match.Success)
                throw new NotSupportedException(
                    $"expected uniform register, got '{operand}'");
            int number = int.Parse(match.Groups[1].Value,
                CultureInfo.InvariantCulture);
            string pair = $"%rdt{_nextTemp++}";
            _lines.Add($"    mov.b64 {pair}, " +
                $"{{%ur{number}, %ur{number + 1}}};");
            return pair;
        }

        private void EmitImul32i(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            if (instruction.Mnemonic.Contains(
                    ".WIDE", StringComparison.OrdinalIgnoreCase))
            {
                string wide = $"%rdt{_nextTemp++}";
                _lines.Add($"{prefix}mul.wide.s32 {wide}, " +
                    $"{Operand(instruction.Operands[1])}, " +
                    $"{Operand(instruction.Operands[2])};");
                CachePair(BaseReg(instruction.Operands[0]), wide, prefix);
                return;
            }
            string operation = instruction.Mnemonic.Contains(
                ".HI", StringComparison.OrdinalIgnoreCase)
                ? "mul.hi.s32" : "mul.lo.s32";
            _lines.Add($"{prefix}{operation} {Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])}, " +
                $"{Operand(instruction.Operands[2])};");
        }

        private void EmitImulWide(SassInstruction instruction, string prefix)
        {
            int source = 1;
            if (source < instruction.Operands.Count &&
                Regex.IsMatch(instruction.Operands[source].Trim(),
                    @"^!?(?:P\d+|PT)$", RegexOptions.IgnoreCase))
                source++;
            if (source + 1 >= instruction.Operands.Count)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two sources");
            string wide = $"%rdt{_nextTemp++}";
            _lines.Add($"{prefix}mul.wide.s32 {wide}, " +
                $"{Operand(instruction.Operands[source])}, " +
                $"{Operand(instruction.Operands[source + 1])};");
            CachePair(BaseReg(instruction.Operands[0]), wide, prefix);
        }

        private void EmitFmnmx(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string destination = Reg(instruction.Operands[0]);
            string a = Operand(instruction.Operands[1]);
            string b = Operand(instruction.Operands[2]);
            string fa = $"%ft{_nextTempFloat++}";
            string fb = $"%ft{_nextTempFloat++}";
            _lines.Add($"    mov.b32 {fa}, {a};");
            _lines.Add($"    mov.b32 {fb}, {b};");
            string predicateText = instruction.Operands[3].Trim();
            if (predicateText.Equals("PT", StringComparison.OrdinalIgnoreCase) ||
                predicateText.Equals("!PT", StringComparison.OrdinalIgnoreCase))
            {
                string operation = predicateText.Equals(
                    "PT", StringComparison.OrdinalIgnoreCase) ? "min" : "max";
                string value = $"%ft{_nextTempFloat++}";
                _lines.Add($"{prefix}{operation}.f32 {value}, {fa}, {fb};");
                _lines.Add($"    mov.b32 {destination}, {value};");
                return;
            }
            string minimum = $"%ft{_nextTempFloat++}";
            string maximum = $"%ft{_nextTempFloat++}";
            _lines.Add($"    min.f32 {minimum}, {fa}, {fb};");
            _lines.Add($"    max.f32 {maximum}, {fa}, {fb};");
            bool negated = predicateText.StartsWith("!", StringComparison.Ordinal);
            string first = negated ? maximum : minimum;
            string second = negated ? minimum : maximum;
            string selected = $"%ft{_nextTempFloat++}";
            _lines.Add($"{prefix}selp.f32 {selected}, {first}, {second}, " +
                $"{Pred(predicateText)};");
            _lines.Add($"    mov.b32 {destination}, {selected};");
        }

        private void EmitMma(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            string shape;
            string types;
            int destinations;
            int aRegisters;
            int bRegisters;
            int cRegisters;
            bool tensorF16 = false;
            if (mnemonic.StartsWith("HMMA", StringComparison.Ordinal))
            {
                if (mnemonic.Contains(".16816", StringComparison.Ordinal))
                {
                    shape = "m16n8k16";
                    destinations = 4;
                    aRegisters = 4;
                    bRegisters = 2;
                    cRegisters = 4;
                }
                else if (mnemonic.Contains(".1688", StringComparison.Ordinal))
                {
                    shape = "m16n8k8";
                    destinations = 4;
                    aRegisters = 2;
                    bRegisters = 1;
                    cRegisters = 4;
                }
                else if (mnemonic.Contains(".1684", StringComparison.Ordinal))
                {
                    shape = "m16n8k4";
                    destinations = 4;
                    aRegisters = 2;
                    bRegisters = 1;
                    cRegisters = 4;
                }
                else if (mnemonic.Contains(".884", StringComparison.Ordinal))
                {
                    shape = "m8n8k4";
                    destinations = 8;
                    aRegisters = 2;
                    bRegisters = 2;
                    cRegisters = 8;
                }
                else
                {
                    throw new NotSupportedException(
                        $"SASS instruction '{instruction.Mnemonic}' is not supported");
                }
                tensorF16 = mnemonic.Contains(".F16", StringComparison.Ordinal) &&
                    !mnemonic.Contains(".F32", StringComparison.Ordinal);
                if (tensorF16)
                {
                    if (shape is not "m16n8k16" and not "m16n8k8")
                        throw new NotSupportedException(
                            $"SASS instruction '{instruction.Mnemonic}' is not supported");
                    destinations = 2;
                    cRegisters = 2;
                }
                types = mnemonic.Contains(
                    ".TF32", StringComparison.Ordinal)
                    ? "f32.tf32.tf32.f32"
                    : tensorF16
                        ? "f16.f16.f16.f16"
                        : "f32.f16.f16.f32";
            }
            else
            {
                if (mnemonic.Contains(".8816", StringComparison.Ordinal))
                {
                    shape = "m8n8k16";
                    destinations = 2;
                    aRegisters = 1;
                    bRegisters = 1;
                    cRegisters = 2;
                }
                else if (mnemonic.Contains(".16816", StringComparison.Ordinal))
                {
                    shape = "m16n8k16";
                    destinations = 4;
                    aRegisters = 2;
                    bRegisters = 1;
                    cRegisters = 4;
                }
                else if (mnemonic.Contains(".16832", StringComparison.Ordinal))
                {
                    shape = "m16n8k32";
                    destinations = 4;
                    aRegisters = 4;
                    bRegisters = 2;
                    cRegisters = 4;
                }
                else
                {
                    throw new NotSupportedException(
                        $"SASS instruction '{instruction.Mnemonic}' is not supported");
                }
                types = "s32.s8.s8.s32";
            }

            string destination = Fragment(
                instruction.Operands[0], destinations);
            string a = Fragment(instruction.Operands[1], aRegisters);
            string b = Fragment(instruction.Operands[2], bRegisters);
            string c = Fragment(instruction.Operands[3], cRegisters);
            _lines.Add($"{prefix}mma.sync.aligned.{shape}.row.col.{types} " +
                $"{destination}, {a}, {b}, {c};");
        }

        private string Fragment(string operand, int count)
        {
            string text = StripReuse(operand);
            text = Regex.Replace(text, @"\.(?:ROW|COL)$", "",
                RegexOptions.IgnoreCase);
            if (text.Equals("RZ", StringComparison.OrdinalIgnoreCase))
            {
                var zeros = new List<string>();
                for (int i = 0; i < count; i++)
                {
                    string zero = $"%rt{_nextTemp++}";
                    _lines.Add($"    mov.u32 {zero}, 0;");
                    zeros.Add(zero);
                }
                return "{" + string.Join(", ", zeros) + "}";
            }
            Match match = Regex.Match(text, @"^R(\d+)$",
                RegexOptions.IgnoreCase);
            if (!match.Success)
                throw new NotSupportedException(
                    $"expected register fragment, got '{operand}'");
            int first = int.Parse(match.Groups[1].Value,
                CultureInfo.InvariantCulture);
            var registers = new List<string>();
            for (int i = 0; i < count; i++)
                registers.Add(Reg("R" + (first + i)));
            return "{" + string.Join(", ", registers) + "}";
        }

        private void EmitIdp(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            string operation;
            string aType;
            string bType;
            if (mnemonic.Contains(".4A", StringComparison.Ordinal))
            {
                Match formats = Regex.Match(mnemonic,
                    @"\.4A\.([US])8\.([US])8");
                if (!formats.Success)
                    throw new NotSupportedException(
                        $"SASS instruction '{instruction.Mnemonic}' is not supported");
                aType = formats.Groups[1].Value == "S" ? "s32" : "u32";
                bType = formats.Groups[2].Value == "S" ? "s32" : "u32";
                operation = "dp4a";
            }
            else if (mnemonic.Contains(".2A", StringComparison.Ordinal))
            {
                Match formats = Regex.Match(mnemonic,
                    @"\.2A\.(LO|HI)\.([US])16\.([US])8");
                if (!formats.Success)
                    throw new NotSupportedException(
                        $"SASS instruction '{instruction.Mnemonic}' is not supported");
                aType = formats.Groups[2].Value == "S" ? "s32" : "u32";
                bType = formats.Groups[3].Value == "S" ? "s32" : "u32";
                operation = $"dp2a.{formats.Groups[1].Value.ToLowerInvariant()}";
            }
            else
            {
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            }
            _lines.Add($"{prefix}{operation}.{aType}.{bType} " +
                $"{Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])}, " +
                $"{Operand(instruction.Operands[2])}, " +
                $"{Operand(instruction.Operands[3])};");
        }

        private void EmitDmma(SassInstruction instruction, string prefix)
        {
            if (!instruction.Mnemonic.Contains(
                    ".884", StringComparison.OrdinalIgnoreCase) ||
                instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            string destination = instruction.Operands[0];
            string a = Fragment64(instruction.Operands[1], 1);
            string b = Fragment64(instruction.Operands[2], 1);
            string c = Fragment64(instruction.Operands[3], 2);
            string d = DestinationFragment64(2);
            _lines.Add($"{prefix}mma.sync.aligned.m8n8k4.row.col." +
                $"f64.f64.f64.f64 {d}, {a}, {b}, {c};");
            UnpackFragment64(destination, 2, d);
        }

        private string Fragment64(string operand, int count)
        {
            string text = StripReuse(operand);
            var values = new List<string>();
            if (text.Equals("RZ", StringComparison.OrdinalIgnoreCase))
            {
                for (int i = 0; i < count; i++)
                {
                    string zero = $"%rdt{_nextTemp++}";
                    _lines.Add($"    mov.b64 {zero}, 0;");
                    values.Add(zero);
                }
                return "{" + string.Join(", ", values) + "}";
            }
            Match match = Regex.Match(text, @"^R(\d+)$",
                RegexOptions.IgnoreCase);
            if (!match.Success)
                throw new NotSupportedException(
                    $"expected register fragment, got '{operand}'");
            int first = int.Parse(match.Groups[1].Value,
                CultureInfo.InvariantCulture);
            for (int i = 0; i < count; i++)
            {
                string value = $"%rdt{_nextTemp++}";
                _lines.Add($"    mov.b64 {value}, " +
                    $"{{{Reg("R" + (first + i * 2))}, " +
                    $"{Reg("R" + (first + i * 2 + 1))}}};");
                values.Add(value);
            }
            return "{" + string.Join(", ", values) + "}";
        }

        private string DestinationFragment64(int count)
        {
            var values = new List<string>();
            for (int i = 0; i < count; i++)
                values.Add($"%rdt{_nextTemp++}");
            return "{" + string.Join(", ", values) + "}";
        }

        private void UnpackFragment64(
            string operand, int count, string values)
        {
            string text = StripReuse(operand);
            Match match = Regex.Match(text, @"^R(\d+)$",
                RegexOptions.IgnoreCase);
            if (!match.Success)
                throw new NotSupportedException(
                    $"expected register fragment, got '{operand}'");
            int first = int.Parse(match.Groups[1].Value,
                CultureInfo.InvariantCulture);
            var items = values.Trim('{', '}').Split(
                ',', StringSplitOptions.TrimEntries);
            if (items.Length != count)
                throw new NotSupportedException(
                    "fragment width mismatch");
            for (int i = 0; i < count; i++)
                _lines.Add($"    mov.b64 " +
                    $"{{{Reg("R" + (first + i * 2))}, " +
                    $"{Reg("R" + (first + i * 2 + 1))}}}, {items[i]};");
        }

        private void EmitFloatUnary(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            string operation = mnemonic.StartsWith("RCP", StringComparison.Ordinal)
                ? "rcp.approx"
                : mnemonic.StartsWith("SQRT", StringComparison.Ordinal)
                    ? "sqrt.rn"
                    : mnemonic.StartsWith("RSQRT", StringComparison.Ordinal)
                        ? "rsqrt.approx"
                        : mnemonic.StartsWith("SIN", StringComparison.Ordinal)
                            ? "sin.approx"
                            : mnemonic.StartsWith("COS", StringComparison.Ordinal)
                                ? "cos.approx"
                                : mnemonic.StartsWith("LG2", StringComparison.Ordinal)
                                    ? "lg2.approx"
                                : mnemonic.StartsWith("EX2", StringComparison.Ordinal)
                                    ? "ex2.approx"
                                : mnemonic.StartsWith("TANH", StringComparison.Ordinal)
                                    ? "tanh.approx"
                                    : throw new NotSupportedException(
                                        $"SASS instruction '{instruction.Mnemonic}' is not supported");
            bool f64 = mnemonic.Contains(".F64", StringComparison.Ordinal);
            if (f64 &&
                (mnemonic.StartsWith("SIN", StringComparison.Ordinal) ||
                 mnemonic.StartsWith("COS", StringComparison.Ordinal) ||
                 mnemonic.StartsWith("LG2", StringComparison.Ordinal) ||
                 mnemonic.StartsWith("EX2", StringComparison.Ordinal) ||
                 mnemonic.StartsWith("TANH", StringComparison.Ordinal)))
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            if (f64)
            {
                string destination = BaseReg(instruction.Operands[0]);
                string value = $"%rdt{_nextTemp++}";
                _lines.Add($"{prefix}{operation}.f64 {value}, " +
                    $"{PackRegisterPair(instruction.Operands[1])};");
                UnpackPair64(value, destination, NextRegister(destination),
                             prefix.Trim());
                return;
            }
            _lines.Add($"{prefix}{operation}.f32 " +
                $"{Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])};");
        }

        private void EmitTestp(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            string test = mnemonic.Contains(".FINITE", StringComparison.Ordinal) ? "finite"
                : mnemonic.Contains(".INFINITE", StringComparison.Ordinal) ? "infinite"
                : mnemonic.Contains(".NUMBER", StringComparison.Ordinal) ? "number"
                : mnemonic.Contains(".NOTANUMBER", StringComparison.Ordinal) ? "notanumber"
                : mnemonic.Contains(".SUBNORMAL", StringComparison.Ordinal) ? "subnormal"
                : "normal";
            string type = mnemonic.Contains(".F64", StringComparison.Ordinal)
                ? "f64" : "f32";
            string source = type == "f64"
                ? PackRegisterPair(instruction.Operands[1])
                : Operand(instruction.Operands[1]);
            _lines.Add($"{prefix}testp.{test}.{type} " +
                $"{Pred(instruction.Operands[0])}, {source};");
        }

        private void EmitCopysign(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            bool f64 = instruction.Mnemonic.Contains(
                ".F64", StringComparison.Ordinal);
            if (f64)
            {
                string destination = BaseReg(instruction.Operands[0]);
                string value = $"%rdt{_nextTemp++}";
                _lines.Add($"{prefix}copysign.f64 {value}, " +
                    $"{PackRegisterPair(instruction.Operands[1])}, " +
                    $"{PackRegisterPair(instruction.Operands[2])};");
                UnpackPair64(value, destination, NextRegister(destination));
                return;
            }
            _lines.Add($"{prefix}copysign.f32 " +
                $"{Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])}, " +
                $"{Operand(instruction.Operands[2])};");
        }

        private void EmitMufu(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            if (mnemonic.Contains(".RSQ64H", StringComparison.Ordinal) ||
                mnemonic.Contains(".RCP64H", StringComparison.Ordinal))
            {
                // High-word estimates for f64: the H forms ignore the
                // low word, form a double from the high word, and return
                // the high word of an approximate 1/sqrt or 1/x.  NVCC's
                // dsqrt/fdiv mediumpaths then refine the result.
                bool reciprocal = mnemonic.Contains(
                    ".RCP64H", StringComparison.Ordinal);
                string high = Operand(instruction.Operands[1]);
                string bits = $"%rdt{_nextTemp++}";
                _lines.Add($"    cvt.u64.u32 {bits}, {high};");
                _lines.Add($"    shl.b64 {bits}, {bits}, 32;");
                string value = $"%dt{_nextTempDouble++}";
                _lines.Add($"    mov.b64 {value}, {bits};");
                string estimate = $"%dt{_nextTempDouble++}";
                _lines.Add($"{prefix}" +
                    (reciprocal
                        ? $"rcp.rn.f64 {estimate}, {value};"
                        : $"rsqrt.approx.f64 {estimate}, {value};"));
                string packed = $"%rdt{_nextTemp++}";
                _lines.Add($"    mov.b64 {packed}, {estimate};");
                string low = $"%rt{_nextTemp++}";
                string resultHigh = $"%rt{_nextTemp++}";
                _lines.Add($"    mov.b64 {{{low}, {resultHigh}}}, {packed};");
                _lines.Add($"{prefix}mov.b32 " +
                    $"{Reg(instruction.Operands[0])}, {resultHigh};");
                return;
            }
            string operation = mnemonic.Contains(".RCP", StringComparison.Ordinal) ? "rcp"
                : mnemonic.Contains(".RSQ", StringComparison.Ordinal) ? "rsqrt"
                : mnemonic.Contains(".EX2", StringComparison.Ordinal) ? "ex2"
                : mnemonic.Contains(".LG2", StringComparison.Ordinal) ? "lg2"
                : mnemonic.Contains(".SIN", StringComparison.Ordinal) ? "sin"
                : mnemonic.Contains(".COS", StringComparison.Ordinal) ? "cos"
                : mnemonic.Contains(".TANH", StringComparison.Ordinal) ? "tanh"
                : mnemonic.Contains(".SQRT", StringComparison.Ordinal) ? "sqrt"
                : throw new NotSupportedException(
                    $"SASS MUFU mode '{instruction.Mnemonic}' is not supported");
            string source = $"%ft{_nextTempFloat++}";
            string result = $"%ft{_nextTempFloat++}";
            // Float sources may be negated/absolute (`MUFU.EX2 R0, -|R3|`);
            // the generic integer operand would two's-complement the bits.
            _lines.Add($"    mov.b32 {source}, " +
                $"{FloatOperand(instruction.Operands[1])};");
            if (operation is "sin" or "cos")
            {
                // SASS MUFU.SIN/COS take revolutions, PTX takes radians.
                string radians = $"%ft{_nextTempFloat++}";
                _lines.Add($"    mul.f32 {radians}, {source}, 0f40c90fdb;");
                source = radians;
            }
            _lines.Add($"{prefix}{operation}.approx.f32 {result}, {source};");
            // Keep the copy under the same guard so a predicated-off MUFU
            // leaves its destination untouched instead of copying an
            // undefined temporary (e.g. the `@P0 MUFU.EX2` in powf/erf).
            _lines.Add($"{prefix}mov.b32 {Reg(instruction.Operands[0])}, " +
                $"{result};");
        }

        private void EmitRro(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            if (!mnemonic.Contains(".SINCOS", StringComparison.Ordinal) &&
                !mnemonic.Contains(".EX2", StringComparison.Ordinal))
                throw new NotSupportedException(
                    $"SASS RRO mode '{instruction.Mnemonic}' is not supported");
            // PTX sin/cos/ex2 approximations perform their own range
            // reduction, but SASS RRO pre-scales SINCOS by 1/(2*pi) and
            // EX2 by log2(e).
            string scale = mnemonic.Contains(
                ".SINCOS", StringComparison.Ordinal)
                ? "0.15915493667125701904" : "1.4426950408889634";
            _lines.Add($"{prefix}mul.f32 {Reg(instruction.Operands[0])}, " +
                $"{FloatOperand(instruction.Operands[1])}, {scale};");
        }

        private void EmitCctl(SassInstruction instruction, string prefix)
        {
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            if (!mnemonic.Contains(".IV", StringComparison.Ordinal))
                throw new NotSupportedException(
                    $"SASS CCTL mode '{instruction.Mnemonic}' is not supported");
            bool invalidateAll = mnemonic.Contains(
                ".IVALL", StringComparison.Ordinal);
            if (!invalidateAll && instruction.Operands.Count == 0)
                throw new NotSupportedException(
                    $"SASS CCTL instruction '{instruction.Mnemonic}' needs an address");
            // PTX has no safe pre-Volta cache-invalidate operation.  Vulkan
            // global memory is coherent, so a global memory fence preserves
            // ordering without discarding dirty data.
            _lines.Add($"{prefix}membar.gl;");
        }

        private void EmitFloatImmediate(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            string destination = Reg(BaseReg(instruction.Operands[0]));
            string source = FloatSource(instruction.Operands[1]);
            string immediate = FloatImmediate(
                instruction.Operands[2], instruction.LegacyFloatModifiers);
            string operation = mnemonic.StartsWith(
                "FMUL", StringComparison.Ordinal) ? "mul" : "add";
            string flush = mnemonic.Contains(
                ".FTZ", StringComparison.Ordinal) ? ".ftz" : "";
            string result = $"%ft{_nextTempFloat++}";
            _lines.Add($"{prefix}{operation}{flush}.f32 " +
                $"{result}, {source}, {immediate};");
            _lines.Add($"{prefix}mov.b32 {destination}, {result};");
        }

        private string FloatSource(string operand)
        {
            string text = Regex.Replace(operand.Trim(), @"\.reuse\b", "",
                RegexOptions.IgnoreCase);
            bool negate = text.StartsWith("-", StringComparison.Ordinal);
            if (negate)
                text = text[1..].Trim();
            bool absolute = text.StartsWith("|", StringComparison.Ordinal) &&
                text.EndsWith("|", StringComparison.Ordinal);
            if (absolute)
                text = text[1..^1].Trim();

            string value;
            if (text.Equals("RZ", StringComparison.OrdinalIgnoreCase))
                value = "0f00000000";
            else if (Regex.IsMatch(text, @"^R\d+$", RegexOptions.IgnoreCase))
                value = Reg(text);
            else
                value = text;

            string temporary = $"%ft{_nextTempFloat++}";
            if (value.StartsWith("0f", StringComparison.OrdinalIgnoreCase) ||
                value.Contains('.') ||
                Regex.IsMatch(value, @"^-?\d+[eE]", RegexOptions.IgnoreCase))
                _lines.Add($"    mov.f32 {temporary}, {value};");
            else
                _lines.Add($"    mov.b32 {temporary}, {value};");
            if (absolute)
            {
                string absoluteValue = $"%ft{_nextTempFloat++}";
                _lines.Add($"    abs.f32 {absoluteValue}, {temporary};");
                temporary = absoluteValue;
            }
            if (negate)
            {
                string negated = $"%ft{_nextTempFloat++}";
                _lines.Add($"    neg.f32 {negated}, {temporary};");
                temporary = negated;
            }
            return temporary;
        }

        private string FloatImmediate(string operand, string? modifiers)
        {
            string text = operand.Trim();
            uint bits;
            if (text.StartsWith("0x", StringComparison.OrdinalIgnoreCase) ||
                text.StartsWith("-0x", StringComparison.OrdinalIgnoreCase))
            {
                bool negative = text.StartsWith("-", StringComparison.Ordinal);
                string hex = text.TrimStart('-');
                long parsed = Convert.ToInt64(hex[2..], 16);
                bits = (uint)(negative ? -parsed : parsed);
            }
            else if (text.Contains('.') ||
                     text.Contains('e') || text.Contains('E'))
            {
                float value = float.Parse(text, NumberStyles.Float,
                    CultureInfo.InvariantCulture);
                bits = BitConverter.SingleToUInt32Bits(value);
            }
            else
            {
                long parsed = long.Parse(text, NumberStyles.Integer,
                    CultureInfo.InvariantCulture);
                bits = (uint)parsed;
            }

            string mode = (modifiers ?? "").ToUpperInvariant();
            if (mode.Contains("ABS", StringComparison.Ordinal))
                bits &= 0x7FFFFFFF;
            if (mode.Contains("NEG", StringComparison.Ordinal))
                bits ^= 0x80000000;

            string temporary = $"%ft{_nextTempFloat++}";
            _lines.Add($"    mov.b32 {temporary}, 0x{bits:x8};");
            return temporary;
        }

        private static bool IsCarryDead(List<SassNode> nodes, int index)
        {
            for (int i = index + 1; i < nodes.Count; i++)
            {
                if (nodes[i] is SassLabel)
                    return false;
                if (nodes[i] is not SassInstruction next)
                    continue;
                string op = next.Mnemonic.Split('.')[0].ToUpperInvariant();
                if (op == "NOP")
                {
                    if (next.Mnemonic.Contains(
                            ".S", StringComparison.OrdinalIgnoreCase))
                        return false;
                    continue;
                }
                if (IsControlFlow(next) || IsBarrier(next))
                    return false;
                if (next.Mnemonic.Contains(
                        ".CC", StringComparison.OrdinalIgnoreCase) ||
                    (next.Operands.Count > 0 &&
                     next.Operands[0].Contains(
                         ".CC", StringComparison.OrdinalIgnoreCase)))
                    return true;
                if ((op is "IADD" or "IADD32I" or "IMAD" or "ISCADD" or "LEA") &&
                    next.Mnemonic.Contains(
                        ".X", StringComparison.OrdinalIgnoreCase))
                    return false;
            }
            return true;
        }

        private void EmitDadd(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            string destinationBase = BaseReg(instruction.Operands[0]);
            string source = PackRegisterPair(instruction.Operands[1]);
            // The addend may be a register, a negated register
            // (`-R30`), a constant-bank pair, or an f64 immediate such
            // as 1/-1.  PackRegisterPair handles all four; emitting the
            // raw text produced invalid PTX like `add.f64 ..., -R30`.
            string addend = PackRegisterPair(instruction.Operands[2]);
            string result = $"%dt{_nextTempDouble++}";
            _lines.Add($"{prefix}add.f64 {result}, {source}, {addend};");
            _lines.Add($"{prefix}mov.b64 " +
                $"{{{Reg(destinationBase)}, {Reg(NextRegister(destinationBase))}}}, " +
                $"{result};");
        }

        private void EmitDmul(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            string result = $"%dt{_nextTempDouble++}";
            _lines.Add($"{prefix}mul.f64 {result}, " +
                $"{PackRegisterPair(instruction.Operands[1])}, " +
                $"{PackRegisterPair(instruction.Operands[2])};");
            string destination = BaseReg(instruction.Operands[0]);
            _lines.Add($"{prefix}mov.b64 " +
                $"{{{Reg(destination)}, {Reg(NextRegister(destination))}}}, {result};");
        }

        private void EmitDfma(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            string rounding = mnemonic.Contains(".RM", StringComparison.Ordinal)
                ? "rm"
                : mnemonic.Contains(".RP", StringComparison.Ordinal) ? "rp"
                : mnemonic.Contains(".RZ", StringComparison.Ordinal) ? "rz"
                : "rn";
            string result = $"%dt{_nextTempDouble++}";
            _lines.Add($"{prefix}fma.{rounding}.f64 {result}, " +
                $"{PackRegisterPair(instruction.Operands[1])}, " +
                $"{PackRegisterPair(instruction.Operands[2])}, " +
                $"{PackRegisterPair(instruction.Operands[3])};");
            string destination = BaseReg(instruction.Operands[0]);
            _lines.Add($"{prefix}mov.b64 " +
                $"{{{Reg(destination)}, {Reg(NextRegister(destination))}}}, {result};");
        }

        private string PackRegisterPair(string operand)
        {
            string text = Regex.Replace(operand.Trim(), @"\.reuse\b", "",
                RegexOptions.IgnoreCase);
            // SASS prints negated/absolute f64 sources as -R30 or
            // |R30|.  Strip the modifiers, pack the pair, then apply the
            // f64 operation; the raw text is not valid PTX.
            bool negate = false;
            bool absolute = false;
            if (text.StartsWith("-", StringComparison.Ordinal))
            {
                negate = true;
                text = text[1..].TrimStart();
            }
            if (text.Length >= 2 &&
                text.StartsWith("|", StringComparison.Ordinal) &&
                text.EndsWith("|", StringComparison.Ordinal))
            {
                absolute = true;
                text = text[1..^1].Trim();
            }
            string Finish(string packed)
            {
                if (!negate && !absolute)
                    return packed;
                string value = $"%dt{_nextTempDouble++}";
                _lines.Add($"    mov.b64 {value}, {packed};");
                if (absolute)
                    _lines.Add($"    abs.f64 {value}, {value};");
                if (negate)
                    _lines.Add($"    neg.f64 {value}, {value};");
                return value;
            }
            // nvdisasm prints f64 immediates as decimals (e.g. DBL_EPSILON
            // appears in normal-distribution kernels).  Materialise the
            // raw 64 bits: PTX has no decimal f64 immediate form.
            ulong immediateBits = 0;
            bool isImmediate = true;
            string lower = text.ToLowerInvariant();
            if (IsZeroReg(text))
            {
                immediateBits = text.StartsWith(
                    "-", StringComparison.Ordinal)
                    ? 0x8000000000000000UL : 0UL;
            }
            else if (lower is "inf" or "+inf")
            {
                immediateBits = 0x7ff0000000000000UL;
            }
            else if (lower is "-inf")
            {
                immediateBits = 0xfff0000000000000UL;
            }
            else if (lower is "nan" or "+nan")
            {
                immediateBits = 0x7ff8000000000000UL;
            }
            else if (double.TryParse(text, NumberStyles.Float,
                         CultureInfo.InvariantCulture, out double value))
            {
                immediateBits = unchecked(
                    (ulong)BitConverter.DoubleToInt64Bits(value));
            }
            else
            {
                isImmediate = false;
            }
            if (isImmediate)
            {
                string packedImmediate = $"%rdt{_nextTemp++}";
                _lines.Add($"    mov.b64 {packedImmediate}, " +
                    $"0x{immediateBits:x16};");
                return Finish(packedImmediate);
            }
            // Bank 0 holds kernel/launch constants and by-value struct
            // fields; let Operand materialise each word (it knows the
            // parameter image) before packing them.
            Match parameterConstant = Regex.Match(text,
                @"^c\[0x0\]\[0x([0-9A-Fa-f]+)\]$",
                RegexOptions.IgnoreCase);
            if (parameterConstant.Success)
            {
                int offset = int.Parse(parameterConstant.Groups[1].Value,
                    NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                string low = Operand($"c[0x0][0x{offset:x}]");
                string high = Operand($"c[0x0][0x{offset + 4:x}]");
                string packedParam = $"%rdt{_nextTemp++}";
                _lines.Add($"    mov.b64 {packedParam}, " +
                    $"{{{low}, {high}}};");
                return Finish(packedParam);
            }
            if (TryGlobalImageOperand(text, out string? relocatedGlobal))
                return relocatedGlobal;
            Match constant = Regex.Match(text,
                @"^c\[0x([1-9A-Fa-f])\]\[0x([0-9A-Fa-f]+)\]$",
                RegexOptions.IgnoreCase);
            if (constant.Success)
            {
                int bank = int.Parse(constant.Groups[1].Value,
                    NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                int offset = int.Parse(constant.Groups[2].Value,
                    NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                uint? low = ConstantWord(bank, offset);
                uint? high = ConstantWord(bank, offset + 4);
                if (!low.HasValue || !high.HasValue)
                    throw new NotSupportedException(
                        $"constant bank {bank} offset 0x{offset:x} " +
                        "cannot form a 64-bit pair");
                string packed = $"%rdt{_nextTemp++}";
                _lines.Add($"    mov.b64 {packed}, " +
                    $"0x{high.Value:x8}{low.Value:x8};");
                return Finish(packed);
            }
            string baseRegister = BaseReg(text);
            string pair = $"%rdt{_nextTemp++}";
            _lines.Add($"    mov.b64 {pair}, " +
                $"{{{Reg(baseRegister)}, {Reg(NextRegister(baseRegister))}}};");
            return Finish(pair);
        }

        private static string NextRegister(string register)
        {
            Match match = Regex.Match(register, @"^R(\d+)$",
                RegexOptions.IgnoreCase);
            if (!match.Success)
                throw new NotSupportedException(
                    $"cannot form a register pair from '{register}'");
            int number = int.Parse(match.Groups[1].Value,
                CultureInfo.InvariantCulture);
            return "R" + (number + 1);
        }

        private static bool IsZeroReg(string operand)
        {
            string t = operand.Trim();
            return t.Equals("RZ", StringComparison.OrdinalIgnoreCase) ||
                   t.Equals("-RZ", StringComparison.OrdinalIgnoreCase) ||
                   t == "0";
        }

        private static ushort F16ImmediateBits(string text)
        {
            string t = text.Trim();
            if (t == "0" || t.Equals("0f00000000",
                    StringComparison.OrdinalIgnoreCase))
                return 0;
            if (float.TryParse(t, NumberStyles.Float,
                    CultureInfo.InvariantCulture, out float value))
                return BitConverter.HalfToUInt16Bits((Half)value);
            throw new NotSupportedException(
                $"unsupported f16 immediate '{text}'");
        }

        private string WideAddend(string operand)
        {
            // The addend is a 64-bit register pair; a `.reuse` hint must
            // not make it fall through to the 32-bit Operand path.
            string t = StripReuse(operand);
            if (t.Equals("RZ", StringComparison.OrdinalIgnoreCase))
                return "0";
            Match constant = Regex.Match(t,
                @"^c\[0x0\]\[0x([0-9A-Fa-f]+)\]$",
                RegexOptions.IgnoreCase);
            if (constant.Success)
            {
                int offset = int.Parse(constant.Groups[1].Value,
                    NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                if (_paramsByOffset.TryGetValue(offset, out PtxParameter? p) &&
                    p.Type == "u64" &&
                    _paramPointers.TryGetValue(p, out string? pointer))
                {
                    if (pointer.StartsWith("%rpptr", StringComparison.Ordinal))
                        return "%rdp" + pointer[6..];
                    return pointer;
                }
                string lo = Operand($"c[0x0][0x{offset:x}]");
                string hi = Operand($"c[0x0][0x{offset + 4:x}]");
                string pair = $"%rdt{_nextTemp++}";
                _lines.Add($"    mov.b64 {pair}, {{{lo}, {hi}}};");
                return pair;
            }
            if (TryGlobalImageOperand(t, out string? globalAddress))
                return globalAddress;
            // IMAD.WIDE reads a 64-bit addend from the constant bank.
            // Bank 2/4 hold compiler constants; take both words instead
            // of zero-extending the low word (the adaptive-pool kernels
            // store 0xFFFFFFFF:0xFFFFFFFF, i.e. the 64-bit value -1).
            Match bankConstant = Regex.Match(t,
                @"^c\[0x([0-9A-Fa-f]+)\]\[0x([0-9A-Fa-f]+)\]$",
                RegexOptions.IgnoreCase);
            if (bankConstant.Success)
            {
                int constantBank = int.Parse(bankConstant.Groups[1].Value,
                    NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                int constantOffset = int.Parse(bankConstant.Groups[2].Value,
                    NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                uint? constantLow = ConstantWord(constantBank, constantOffset);
                uint? constantHigh = ConstantWord(
                    constantBank, constantOffset + 4);
                if (constantLow.HasValue && constantHigh.HasValue)
                {
                    string pair = $"%rdt{_nextTemp++}";
                    _lines.Add($"    mov.b64 {pair}, " +
                        $"{{0x{constantLow.Value:x}, " +
                        $"0x{constantHigh.Value:x}}};");
                    return pair;
                }
            }
            Match reg = Regex.Match(t, @"^R(\d+)$", RegexOptions.IgnoreCase);
            if (reg.Success)
            {
                if (_pairRegs.TryGetValue(t, out string? pair))
                    return pair;
                string lo = Reg(t);
                string hi = Reg(NextRegister(t));
                string built = $"%rdt{_nextTemp++}";
                _lines.Add($"    mov.b64 {built}, {{{lo}, {hi}}};");
                return built;
            }
            return Operand(t);
        }

        // IMAD.HI reads a 64-bit addend pair {c, c+1} and computes
        // (a*b + (c_high:c_low)) >> 32, not ((a*b) >> 32) + c.  ptxas
        // relies on the high word for idioms such as ATen's IntDivider,
        // `(__umulhi(n, magic) + n) >> shift`, where the low word is a
        // fresh zero register and the high word holds the dividend.
        private (string Low, string High) MadHiAddendPair(string operand)
        {
            string text = StripReuse(operand);
            if (IsZeroReg(text) ||
                text.Equals("URZ", StringComparison.OrdinalIgnoreCase))
                return ("0", "0");
            Match constant = Regex.Match(text,
                @"^c\[0x([0-9A-Fa-f]+)\]\[0x([0-9A-Fa-f]+)\]$",
                RegexOptions.IgnoreCase);
            if (constant.Success)
            {
                int bank = int.Parse(constant.Groups[1].Value,
                    NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                int offset = int.Parse(constant.Groups[2].Value,
                    NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                string highText = $"c[0x{bank:x}][0x{offset + 4:x}]";
                return (Operand(text), Operand(highText));
            }
            Match uniform = Regex.Match(text, @"^UR(\d+)$",
                RegexOptions.IgnoreCase);
            if (uniform.Success)
            {
                int number = int.Parse(uniform.Groups[1].Value,
                    CultureInfo.InvariantCulture);
                return (UniformOperand(text),
                    UniformOperand($"UR{number + 1}"));
            }
            Match register = Regex.Match(text, @"^R(\d+)$",
                RegexOptions.IgnoreCase);
            if (register.Success)
            {
                string lowKey = "R" + register.Groups[1].Value;
                return (Operand(lowKey), Operand(NextRegister(lowKey)));
            }
            return (Operand(text), "0");
        }

        private void EmitMad(SassInstruction instruction, string prefix)
        {
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            if (mnemonic.StartsWith("IMAD.HI",
                    StringComparison.Ordinal))
            {
                if (instruction.Operands.Count < 3)
                    throw new NotSupportedException(
                        $"SASS instruction '{instruction.Mnemonic}' needs three operands");
                string hiType = mnemonic.Contains(
                    ".S32", StringComparison.Ordinal) ? "s32" : "u32";
                string hiLeft = Operand(instruction.Operands[1]);
                string hiRight = Operand(instruction.Operands[2]);
                string hiDst = Reg(instruction.Operands[0]);
                string hiAddend = instruction.Operands.Count >= 4
                    ? instruction.Operands[3] : "RZ";
                // IMAD.HI.X adds the carry from the preceding low-word
                // operation to the high word.  Materialise the predicate
                // as 0/1 instead of assuming the hardware carry is live.
                bool hiCarryIn = mnemonic.Contains(
                    ".X", StringComparison.Ordinal) &&
                    instruction.Operands.Count >= 5;
                string hiPin = hiCarryIn
                    ? Operand(instruction.Operands[4]) : "0";
                if (IsZeroReg(hiAddend) ||
                    hiAddend.Trim().Equals("URZ",
                        StringComparison.OrdinalIgnoreCase))
                {
                    if (hiPin == "0")
                    {
                        _lines.Add($"{prefix}mul.hi.{hiType} {hiDst}, " +
                            $"{hiLeft}, {hiRight};");
                    }
                    else
                    {
                        _lines.Add($"{prefix}mul.hi.{hiType} {hiDst}, " +
                            $"{hiLeft}, {hiRight};");
                        _lines.Add($"{prefix}add.u32 {hiDst}, {hiDst}, " +
                            $"{hiPin};");
                    }
                    return;
                }
                (string addLow, string addHigh) = MadHiAddendPair(hiAddend);
                // The destination may alias either half of the addend
                // pair, in which case the product must go through a
                // scratch register first.
                bool aliases =
                    string.Equals(hiDst, addLow,
                        StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(hiDst, addHigh,
                        StringComparison.OrdinalIgnoreCase);
                string high = aliases ? $"%rt{_nextTemp++}" : hiDst;
                _lines.Add($"{prefix}mul.hi.{hiType} {high}, " +
                    $"{hiLeft}, {hiRight};");
                if (addLow == "0")
                {
                    if (addHigh != "0")
                        _lines.Add($"{prefix}add.u32 {high}, {high}, " +
                            $"{addHigh};");
                }
                else
                {
                    string low = $"%rt{_nextTemp++}";
                    _lines.Add($"{prefix}mul.lo.{hiType} {low}, " +
                        $"{hiLeft}, {hiRight};");
                    _lines.Add($"{prefix}add.cc.u32 {low}, {low}, " +
                        $"{addLow};");
                    _lines.Add($"{prefix}addc.u32 {high}, {high}, " +
                        $"{addHigh};");
                }
                if (hiPin != "0")
                    _lines.Add($"{prefix}add.u32 {high}, {high}, " +
                        $"{hiPin};");
                if (aliases)
                    _lines.Add($"{prefix}mov.u32 {hiDst}, {high};");
                return;
            }
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            if (mnemonic.Contains(".MRG") || mnemonic.Contains(".CBCC"))
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            if (mnemonic.StartsWith("IMAD.X", StringComparison.Ordinal) &&
                instruction.Operands.Count >= 4 &&
                IsZeroReg(instruction.Operands[1]) &&
                IsZeroReg(instruction.Operands[2]))
            {
                // High word of a 64-bit constant/pair load.  With a fifth
                // predicate operand this also carries the previous low-word
                // add: IMAD.X Rd,RZ,RZ,Rc,P is Rd = Rc + carry.  ptxas
                // emits it with Rc = -1 to sign-extend a 32-bit offset, so
                // dropping the carry moved misaligned reduce slices 4 GiB
                // below their buffer.
                string imadAddend = Operand(instruction.Operands[3]);
                bool imadCarryIn = instruction.Operands.Count >= 5 &&
                    Regex.IsMatch(instruction.Operands[4].Trim(),
                        @"^!?(?:P\d+|PT)$", RegexOptions.IgnoreCase);
                if (imadCarryIn)
                    _lines.Add($"{prefix}addc.u32 " +
                        $"{Reg(instruction.Operands[0])}, {imadAddend}, 0;");
                else
                    _lines.Add($"{prefix}mov.u32 " +
                        $"{Reg(instruction.Operands[0])}, {imadAddend};");
                return;
            }
            string destination = Reg(instruction.Operands[0]);
            string left = Operand(instruction.Operands[1]);
            string right = Operand(instruction.Operands[2]);
            string addend = Operand(instruction.Operands[3]);
            if (mnemonic.StartsWith("IMAD.WIDE",
                    StringComparison.Ordinal))
            {
                string wideAddend = WideAddend(instruction.Operands[3]);
                string wide = $"%rdt{_nextTemp++}";
                if (prefix.Trim().Length > 0 && wideAddend != "0")
                    _lines.Add($"    mov.b64 {wide}, {wideAddend};");
                _lines.Add($"{prefix}mad.wide.u32 {wide}, {left}, {right}, " +
                    $"{wideAddend};");
                // IMAD.WIDE.X carries the preceding low-word operation
                // into bit 32 of the 64-bit result.
                if (mnemonic.Contains(".X", StringComparison.Ordinal) &&
                    instruction.Operands.Count >= 5)
                {
                    string pin = Operand(instruction.Operands[4]);
                    if (pin != "0")
                    {
                        string pin64 = $"%rdt{_nextTemp++}";
                        _lines.Add($"    cvt.u64.u32 {pin64}, {pin};");
                        _lines.Add($"    shl.b64 {pin64}, {pin64}, 32;");
                        _lines.Add($"{prefix}add.u64 {wide}, {wide}, " +
                            $"{pin64};");
                    }
                }
                CachePair(BaseReg(instruction.Operands[0]), wide, prefix);
                return;
            }
            if (instruction.Mnemonic.Contains(
                    "PSL", StringComparison.OrdinalIgnoreCase))
            {
                string product = $"%rt{_nextTemp++}";
                _lines.Add($"{prefix}mul.lo.u32 {product}, {left}, {right};");
                _lines.Add($"    shl.b32 {product}, {product}, 16;");
                _lines.Add($"    add.u32 {destination}, {product}, {addend};");
                return;
            }
            if (mnemonic.Contains(".HI", StringComparison.Ordinal))
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs a paired carry instruction");
            if (mnemonic.Contains(".X", StringComparison.Ordinal))
            {
                // IMAD.X Rd, Ra, Rb, Rc, carry: low word of the product
                // plus the carry left by the preceding operation.
                string product = $"%rt{_nextTemp++}";
                _lines.Add($"{prefix}mad.lo.s32 {product}, {left}, " +
                    $"{right}, {addend};");
                _lines.Add($"{prefix}addc.u32 {destination}, {product}, 0;");
                return;
            }
            _lines.Add($"{prefix}mad.lo.s32 {destination}, {left}, {right}, {addend};");
        }

        private static string InvertedGuard(string prefix)
        {
            string guard = prefix.Trim();
            if (guard.Length == 0)
                return "";
            if (guard.StartsWith("@!", StringComparison.Ordinal))
                return "@" + guard[2..];
            if (guard.StartsWith("@", StringComparison.Ordinal))
                return "@!" + guard[1..];
            return "";
        }

        private string? GuardedPrefix(string prefix, out string? label)
        {
            label = null;
            string inverted = InvertedGuard(prefix);
            if (inverted.Length == 0)
                return prefix;
            label = $"L_g{_nextTemp++}";
            _lines.Add($"{inverted} bra {label};");
            return "";
        }

        private void EmitLoad(SassInstruction instruction, string prefix)
        {
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            bool uniform = mnemonic.StartsWith("LDU", StringComparison.Ordinal);
            bool shared = mnemonic.StartsWith("LDS", StringComparison.Ordinal);
            string address = shared
                ? SharedAddress(instruction.Operands[1])
                : Address(instruction.Operands[1]);
            string width = LoadWidth(instruction.Mnemonic);
            int vectorCount = VectorCount(width);
            string destination = vectorCount > 0
                ? VectorRegs(instruction.Operands[0], vectorCount)
                : IsRz(instruction.Operands[0])
                    ? $"%rt{_nextTemp++}"
                    : Reg(instruction.Operands[0]);
            string operation = uniform
                ? "ldu.global"
                : shared ? "ld.shared" : "ld.global";
            if (!string.IsNullOrWhiteSpace(prefix) && shared)
            {
                // A predicated shared load can be kept out of the CFG:
                // perform the load unconditionally and select the old
                // destination value when the predicate is false.  The
                // SASS predicate does not guard the address computation,
                // so the address is valid on both paths.  This preserves
                // "destination unchanged when predicated off" without a
                // divergent skip branch before a following barrier.
                string condition = prefix.Trim();
                bool negated = condition.StartsWith("@!",
                    StringComparison.Ordinal);
                string predicate = negated
                    ? condition[2..]
                    : condition[1..];
                if (vectorCount > 0)
                {
                    Match vec = Regex.Match(
                        instruction.Operands[0].Trim(), @"^R(\d+)$",
                        RegexOptions.IgnoreCase);
                    if (!vec.Success)
                        throw new NotSupportedException(
                            $"expected vector register base, got " +
                            $"'{instruction.Operands[0]}'");
                    int first = int.Parse(vec.Groups[1].Value,
                        CultureInfo.InvariantCulture);
                    var temps = new string[vectorCount];
                    for (int i = 0; i < vectorCount; i++)
                        temps[i] = $"%rt{_nextTemp++}";
                    _lines.Add($"ld.shared.{width} " +
                        $"{{{string.Join(", ", temps)}}}, {address};");
                    for (int i = 0; i < vectorCount; i++)
                    {
                        string dst = Reg($"R{first + i}");
                        _lines.Add(negated
                            ? $"selp.b32 {dst}, {dst}, {temps[i]}, {predicate};"
                            : $"selp.b32 {dst}, {temps[i]}, {dst}, {predicate};");
                    }
                }
                else
                {
                    bool discard = IsRz(instruction.Operands[0]);
                    string temp = $"%rt{_nextTemp++}";
                    _lines.Add($"ld.shared.{width} {temp}, {address};");
                    if (!discard)
                    {
                        _lines.Add(negated
                            ? $"selp.b32 {destination}, {destination}, " +
                              $"{temp}, {predicate};"
                            : $"selp.b32 {destination}, {temp}, " +
                              $"{destination}, {predicate};");
                    }
                }
                return;
            }
            string? guardLabel;
            prefix = GuardedPrefix(prefix, out guardLabel) ?? "";
            _lines.Add($"{prefix}{operation}.{width} {destination}, {address};");
            if (guardLabel != null)
                _lines.Add($"{guardLabel}:");
        }

        private void EmitAsyncStore(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            bool global = instruction.Mnemonic.Contains(
                "GLOBAL", StringComparison.OrdinalIgnoreCase);
            string address = global
                ? Address(instruction.Operands[0])
                : SharedAddress(instruction.Operands[0]);
            string width = LoadWidth(instruction.Mnemonic);
            int vectorCount = VectorCount(width);
            string source = vectorCount > 0
                ? VectorRegs(instruction.Operands[1], vectorCount)
                : Operand(instruction.Operands[1]);
            string? guardLabel;
            prefix = GuardedPrefix(prefix, out guardLabel) ?? "";
            _lines.Add($"{prefix}st.{(global ? "global" : "shared")}.{width} " +
                $"{address}, {source};");
            if (guardLabel != null)
                _lines.Add($"{guardLabel}:");
        }

        private void EmitBulkStore(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            if (!string.IsNullOrWhiteSpace(instruction.Guard))
                throw new NotSupportedException(
                    $"guarded '{instruction.Mnemonic}' is not supported");
            if (!TrySwitchImmediate(instruction.Operands[2], out int initialValue) ||
                initialValue != 0)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' requires a zero fill value");
            string address = SharedAddress(instruction.Operands[0]);
            string expression = address.Trim();
            if (!expression.StartsWith("[", StringComparison.Ordinal) ||
                !expression.EndsWith("]", StringComparison.Ordinal))
                throw new NotSupportedException(
                    $"unsupported address '{instruction.Operands[0]}'");
            expression = expression[1..^1];
            string current = $"%rdt{_nextTemp++}";
            if (expression.Equals("__shared", StringComparison.Ordinal))
            {
                _lines.Add($"{prefix}mov.u64 {current}, __shared;");
            }
            else if (expression.StartsWith("__shared+",
                         StringComparison.Ordinal))
            {
                string offset = expression["__shared+".Length..];
                _lines.Add($"{prefix}mov.u64 {current}, __shared;");
                _lines.Add(
                    $"{prefix}add.u64 {current}, {current}, {offset};");
            }
            else
            {
                _lines.Add($"{prefix}mov.u64 {current}, {expression};");
            }
            string count = $"%ru{_nextTempU32++}";
            _lines.Add($"{prefix}mov.u32 {count}, " +
                $"{Operand(instruction.Operands[1])};");
            string loop = $"L_st_bulk_{_nextTemp++}";
            while (_labels.Values.Contains(loop))
                loop += "_";
            string done = loop + "_done";
            while (_labels.Values.Contains(done))
                done += "_";
            string predicate = $"%pt{_nextTempPred++}";
            _lines.Add($"{loop}:");
            _lines.Add($"setp.eq.u32 {predicate}, {count}, 0;");
            _lines.Add($"@{predicate} bra {done};");
            _lines.Add($"st.shared.u64 [{current}], 0;");
            _lines.Add($"add.u64 {current}, {current}, 8;");
            _lines.Add($"sub.u32 {count}, {count}, 8;");
            _lines.Add($"bra {loop};");
            _lines.Add($"{done}:");
        }

        private void EmitNativeInstruction(
            SassInstruction instruction, string prefix,
            IReadOnlyDictionary<int, bool>? addressSpaces = null,
            IReadOnlyDictionary<int, string>? operandOverrides = null)
        {
            string mnemonic = CanonicalNativeMnemonic(instruction.Mnemonic);
            var operands = new List<string>();
            for (int i = 0; i < instruction.Operands.Count; i++)
            {
                if (operandOverrides != null &&
                    operandOverrides.TryGetValue(i, out string? overridden))
                {
                    operands.Add(overridden);
                    continue;
                }
                bool? shared = addressSpaces != null &&
                    addressSpaces.TryGetValue(i, out bool isShared)
                    ? isShared
                    : null;
                operands.Add(NativeOperand(instruction.Operands[i], shared));
            }
            _lines.Add(operands.Count == 0
                ? $"{prefix}{mnemonic};"
                : $"{prefix}{mnemonic} {string.Join(", ", operands)};");
        }

        private string NativeOperand(string operand, bool? sharedAddress)
        {
            string trimmed = StripReuse(operand);
            if (trimmed.Length == 0)
                return trimmed;
            if (trimmed.StartsWith("[", StringComparison.Ordinal) &&
                trimmed.EndsWith("]", StringComparison.Ordinal))
            {
                string inner = trimmed[1..^1].Trim();
                if (sharedAddress.HasValue &&
                    Regex.IsMatch(inner,
                        @"^(R\d+|RZ)(?:\+(0x[0-9A-Fa-f]+|\d+))?$",
                        RegexOptions.IgnoreCase))
                {
                    if (sharedAddress.Value)
                        return SharedAddress(trimmed);
                    if (inner.StartsWith("RZ",
                            StringComparison.OrdinalIgnoreCase))
                    {
                        Match offset = Regex.Match(inner,
                            @"^RZ(?:\+(0x[0-9A-Fa-f]+|\d+))?$",
                            RegexOptions.IgnoreCase);
                        return offset.Groups[1].Success
                            ? $"[{Immediate(offset.Groups[1].Value)}]"
                            : "[0]";
                    }
                    return Address(trimmed);
                }
                if (Regex.IsMatch(inner,
                        @"^R\d+(?:\+(0x[0-9A-Fa-f]+|\d+))?$",
                        RegexOptions.IgnoreCase))
                    return Address(trimmed);
                return "[" + string.Join(", ", SplitTopLevel(inner)
                    .Select(part => NativeOperand(part, null))) + "]";
            }
            if (trimmed.StartsWith("{", StringComparison.Ordinal) &&
                trimmed.EndsWith("}", StringComparison.Ordinal))
                return "{" + string.Join(", ", SplitTopLevel(trimmed[1..^1])
                    .Select(part => NativeOperand(part, null))) + "}";
            if (Regex.IsMatch(trimmed, @"^R\d+$", RegexOptions.IgnoreCase))
                return Reg(trimmed);
            if (Regex.IsMatch(trimmed, @"^P\d+$", RegexOptions.IgnoreCase))
                return Pred(trimmed);
            return Operand(trimmed);
        }

        private static List<string> SplitTopLevel(string text)
        {
            var result = new List<string>();
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
            return result.Where(part => part.Length > 0).ToList();
        }

        private static string CanonicalNativeMnemonic(string mnemonic)
        {
            string[] parts = mnemonic.ToLowerInvariant().Split('.');
            for (int i = 0; i < parts.Length; i++)
            {
                parts[i] = parts[i].ToUpperInvariant() switch
                {
                    "L1" => "L1",
                    "L2" => "L2",
                    "L1::32B" => "L1::32B",
                    "L2::128B" => "L2::128B",
                    "L2::256B" => "L2::256B",
                    "L2::CACHE_HINT" => "L2::cache_hint",
                    "L2::EVICT_LAST" => "L2::evict_last",
                    "L2::EVICT_NORMAL" => "L2::evict_normal",
                    _ => parts[i],
                };
            }
            return string.Join(".", parts);
        }

        private static int StateTokenIndex(string[] parts, string state)
        {
            for (int i = 0; i < parts.Length; i++)
            {
                if (parts[i].StartsWith(state, StringComparison.Ordinal))
                    return i;
            }
            return -1;
        }

        private void EmitMultimem(SassInstruction instruction, string prefix)
        {
            string[] parts = instruction.Mnemonic.ToUpperInvariant().Split('.');
            if (parts.Length < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            if (parts[1] == "CP")
            {
                EmitMultimemBulk(instruction, prefix);
                return;
            }
            if (parts.Any(part => part is "V2" or "V4" or "V8"))
                throw new NotSupportedException(
                    "multimem vector forms are not supported");
            bool loadReduce = parts[1] == "LD_REDUCE";
            bool store = parts[1] == "ST";
            bool reduce = parts[1] == "RED";
            if (!loadReduce && !store && !reduce)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            string type = parts[^1].ToLowerInvariant();
            bool wide = type is "b64" or "u64" or "s64" or "f64";
            if (type is not ("b32" or "u32" or "s32" or "b64" or "u64"
                or "s64" or "f32" or "f64"))
                throw new NotSupportedException(
                    $"multimem type '{type}' is not supported");
            bool floatType = type is "f32" or "f64";
            string operation = loadReduce || reduce
                ? parts.Skip(2).FirstOrDefault(part =>
                    part is "ADD" or "MIN" or "MAX" or "AND" or "OR" or "XOR")
                    ?.ToLowerInvariant() ?? ""
                : "";
            if (loadReduce || reduce)
            {
                if (operation is not ("add" or "min" or "max" or "and"
                    or "or" or "xor"))
                    throw new NotSupportedException(
                        $"multimem operation '{operation}' is not supported");
                if ((operation is "and" or "or" or "xor") && floatType)
                    throw new NotSupportedException(
                        "bitwise multimem reductions need integer types");
            }
            string semantics = "";
            string scope = "";
            for (int i = 2; i < parts.Length - 1; i++)
            {
                string token = parts[i];
                if (token is "RELAXED" or "ACQUIRE" or "RELEASE" or "WEAK")
                    semantics = token.ToLowerInvariant();
                else if (token is "CTA" or "CLUSTER" or "GPU" or "SYS")
                    scope = token.ToLowerInvariant();
            }
            bool validSemantics = semantics switch
            {
                "" => true,
                "weak" => loadReduce || store,
                "relaxed" => true,
                "acquire" => loadReduce,
                "release" => store || reduce,
                _ => false,
            };
            if (!validSemantics)
                throw new NotSupportedException(
                    $"multimem semantics '{semantics}' are not supported");
            string memory = "";
            if (semantics is not ("" or "weak"))
            {
                memory = "." + semantics;
                if (scope.Length == 0)
                    scope = reduce ? "sys" : "gpu";
                memory += "." + scope;
            }
            if (loadReduce)
            {
                if (instruction.Operands.Count < 2)
                    throw new NotSupportedException(
                        $"SASS instruction '{instruction.Mnemonic}' needs two operands");
                string address = Address(instruction.Operands[1]);
                string destination = BaseReg(instruction.Operands[0]);
                if (wide)
                {
                    string value = $"%rdt{_nextTemp++}";
                    _lines.Add($"{prefix}ld{memory}.global.b64 " +
                        $"{value}, {address};");
                    UnpackPair64(value, destination,
                        NextRegister(destination));
                }
                else
                {
                    _lines.Add($"{prefix}ld{memory}.global.b32 " +
                        $"{Reg(instruction.Operands[0])}, {address};");
                }
                return;
            }
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            string storeAddress = Address(instruction.Operands[0]);
            string source = wide
                ? PackRegisterPair(instruction.Operands[1])
                : Operand(instruction.Operands[1]);
            if (store)
            {
                _lines.Add($"{prefix}st{memory}.global.{(wide ? "b64" : "b32")} " +
                    $"{storeAddress}, {source};");
                return;
            }
            string reductionType = operation is "and" or "or" or "xor"
                ? (wide ? "b64" : "b32")
                : type switch
                {
                    "b32" => "u32",
                    "b64" => "u64",
                    _ => type,
                };
            if (type == "f32")
            {
                string floatSource = $"%ft{_nextTempFloat++}";
                _lines.Add($"mov.b32 {floatSource}, {source};");
                source = floatSource;
            }
            else if (type == "f64")
            {
                string doubleSource = $"%dt{_nextTempDouble++}";
                _lines.Add($"mov.b64 {doubleSource}, {source};");
                source = doubleSource;
            }
            _lines.Add($"{prefix}red{memory}.global.{operation}." +
                $"{reductionType} {storeAddress}, {source};");
        }

        private void EmitMultimemBulk(
            SassInstruction instruction, string prefix)
        {
            string[] parts = instruction.Mnemonic.ToUpperInvariant().Split('.');
            if (parts.Length < 5 || parts[1] != "CP")
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            if (parts.Any(part => part == "TENSOR"))
            {
                EmitNativeInstruction(instruction, prefix);
                return;
            }
            var spaces = new Dictionary<int, bool> { [0] = false, [1] = true };
            if (instruction.Operands.Count > 3 &&
                instruction.Operands[3].TrimStart().StartsWith("["))
                spaces[3] = true;
            EmitNativeInstruction(instruction, prefix, spaces);
        }

        private void EmitTensormap(SassInstruction instruction, string prefix)
        {
            string[] parts = instruction.Mnemonic.ToUpperInvariant().Split('.');
            if (parts.Length < 3 || parts[1] != "REPLACE")
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            var spaces = new Dictionary<int, bool>();
            if (instruction.Operands.Count > 0)
                spaces[0] = parts.Any(part =>
                    part.StartsWith("SHARED", StringComparison.Ordinal));
            var overrides = new Dictionary<int, string>();
            if (parts[^1] == "B64" && instruction.Operands.Count > 1 &&
                Regex.IsMatch(instruction.Operands[^1].Trim(),
                    @"^R\d+$", RegexOptions.IgnoreCase))
                overrides[instruction.Operands.Count - 1] =
                    PackRegisterPair(instruction.Operands[^1]);
            EmitNativeInstruction(instruction, prefix, spaces, overrides);
        }

        private void EmitPrefetch(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 1)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs an address");
            string[] parts = instruction.Mnemonic.Split('.');
            foreach (string part in parts.Skip(1))
            {
                string token = part.ToUpperInvariant();
                if (token is "L1::32B" or "VALID_ADDR" or "TENSORMAP"
                    or "CONST" or "PARAM")
                    throw new NotSupportedException(
                        $"prefetch modifier '{part}' is not supported");
            }
            if (parts[0].Equals("PREFETCHU", StringComparison.OrdinalIgnoreCase) &&
                !parts.Skip(1).All(part =>
                    part.Equals("L1", StringComparison.OrdinalIgnoreCase)))
                throw new NotSupportedException(
                    "prefetchu supports only prefetchu.L1");
            string qualifiers = string.Join(".", parts.Skip(1).Select(part =>
                part.ToUpperInvariant() switch
                {
                    "L1" => "L1",
                    "L2" => "L2",
                    "L2::EVICT_LAST" => "L2::evict_last",
                    "L2::EVICT_NORMAL" => "L2::evict_normal",
                    _ => part.ToLowerInvariant(),
                }));
            string address = Address(instruction.Operands[0]);
            _lines.Add($"{prefix}{parts[0].ToLowerInvariant()}" +
                $"{(qualifiers.Length == 0 ? "" : "." + qualifiers)} " +
                $"{address};");
        }

        private void EmitIsspacep(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            string[] parts = instruction.Mnemonic.ToLowerInvariant().Split('.');
            if (parts.Length < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs a state space");
            string qualifiers = string.Join(".", parts.Skip(1));
            string source = instruction.Operands[1].Contains(
                ".64", StringComparison.OrdinalIgnoreCase)
                ? PackRegisterPair(instruction.Operands[1])
                : Operand(instruction.Operands[1]);
            _lines.Add($"{prefix}isspacep.{qualifiers} " +
                $"{Pred(instruction.Operands[0])}, {source};");
        }

        private void EmitCvta(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            string[] parts = instruction.Mnemonic.ToLowerInvariant().Split('.');
            if (parts.Length < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs a state space");
            string qualifiers = string.Join(".", parts.Skip(1));
            bool wide = parts[^1] is "u64" or "s64";
            if (!wide)
            {
                _lines.Add($"{prefix}cvta.{qualifiers} " +
                    $"{Reg(instruction.Operands[0])}, " +
                    $"{Operand(instruction.Operands[1])};");
                return;
            }
            string destination = BaseReg(instruction.Operands[0]);
            string source = PackRegisterPair(instruction.Operands[1]);
            string value = $"%rdt{_nextTemp++}";
            _lines.Add($"{prefix}cvta.{qualifiers} {value}, {source};");
            UnpackPair64(value, destination, NextRegister(destination));
        }

        private void EmitCpAsync(SassInstruction instruction, string prefix)
        {
            string[] parts = instruction.Mnemonic.ToUpperInvariant().Split('.');
            if (parts.Length < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            if (parts[1] == "REDUCE")
            {
                EmitCpReduceAsyncBulk(instruction, prefix, parts);
                return;
            }
            if (parts[1] != "ASYNC")
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            string form = parts.Length > 2 ? parts[2] : "";
            if (form == "COMMIT_GROUP")
            {
                _lines.Add($"{prefix}cp.async.commit_group;");
                return;
            }
            if (form == "WAIT_ALL")
            {
                _lines.Add($"{prefix}cp.async.wait_all;");
                return;
            }
            if (form == "WAIT_GROUP")
            {
                if (instruction.Operands.Count < 1 ||
                    !TrySwitchImmediate(instruction.Operands[0], out int depth))
                    throw new NotSupportedException(
                        "cp.async.wait_group needs an immediate depth");
                _lines.Add($"{prefix}cp.async.wait_group {depth};");
                return;
            }
            if (form == "BULK")
            {
                EmitCpAsyncBulk(instruction, prefix, parts);
                return;
            }
            if (form == "MBARRIER")
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    "cp.async needs destination, source, and size operands");
            string qualifiers = string.Join(".", parts.Skip(2).Select(part =>
                part.ToUpperInvariant() switch
                {
                    "L2::128B" => "L2::128B",
                    "L2::256B" => "L2::256B",
                    "L2::CACHE_HINT" => "L2::cache_hint",
                    _ => part.ToLowerInvariant(),
                }));
            string destination = SharedAddress(instruction.Operands[0]);
            string source = Address(instruction.Operands[1]);
            string extra = instruction.Operands.Count > 3
                ? ", " + Operand(instruction.Operands[3])
                : "";
            _lines.Add($"{prefix}cp.async.{qualifiers} " +
                $"{destination}, {source}, " +
                $"{Operand(instruction.Operands[2])}{extra};");
        }

        private void EmitCpAsyncBulk(
            SassInstruction instruction, string prefix, string[] parts)
        {
            string form = parts.Length > 3 ? parts[3] : "";
            if (form is "COMMIT_GROUP" or "WAIT_GROUP")
            {
                EmitNativeInstruction(instruction, prefix);
                return;
            }
            if (form == "PREFETCH")
            {
                if (parts.Length > 4 && parts[4] == "TENSOR")
                {
                    EmitNativeInstruction(instruction, prefix);
                    return;
                }
                EmitNativeInstruction(instruction, prefix,
                    new Dictionary<int, bool> { [0] = false });
                return;
            }
            if (form == "TENSOR")
            {
                var tensorSpaces = new Dictionary<int, bool>();
                int tensorShared = StateTokenIndex(parts, "SHARED");
                int tensorGlobal = StateTokenIndex(parts, "GLOBAL");
                if (tensorShared >= 0 && tensorGlobal >= 0)
                {
                    if (tensorShared < tensorGlobal)
                    {
                        tensorSpaces[0] = true;
                        if (instruction.Operands.Count > 2)
                            tensorSpaces[2] = true;
                    }
                    else
                    {
                        tensorSpaces[1] = true;
                    }
                }
                EmitNativeInstruction(instruction, prefix, tensorSpaces);
                return;
            }
            int shared = StateTokenIndex(parts, "SHARED");
            int global = StateTokenIndex(parts, "GLOBAL");
            if (shared < 0 || global < 0)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs "
                    + "shared and global state spaces");
            bool destinationShared = shared < global;
            var spaces = new Dictionary<int, bool>
            {
                [0] = destinationShared,
                [1] = !destinationShared,
            };
            if (destinationShared && instruction.Operands.Count > 3 &&
                instruction.Operands[3].TrimStart().StartsWith("["))
                spaces[3] = true;
            EmitNativeInstruction(instruction, prefix, spaces);
        }

        private void EmitCpReduceAsyncBulk(
            SassInstruction instruction, string prefix, string[] parts)
        {
            if (parts.Length < 4 || parts[2] != "ASYNC" ||
                parts[3] != "BULK")
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            if (parts.Any(part => part == "TENSOR"))
            {
                EmitNativeInstruction(instruction, prefix,
                    new Dictionary<int, bool> { [1] = true });
                return;
            }
            var spaces = new Dictionary<int, bool>
            {
                [0] = false,
                [1] = true,
            };
            EmitNativeInstruction(instruction, prefix, spaces);
        }

        private void EmitStore(SassInstruction instruction, string prefix)
        {
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            bool shared = mnemonic.StartsWith("STS", StringComparison.Ordinal);
            string address = shared
                ? SharedAddress(instruction.Operands[0])
                : Address(instruction.Operands[0]);
            string width = LoadWidth(instruction.Mnemonic);
            int vectorCount = VectorCount(width);
            string source = vectorCount > 0
                ? VectorRegs(instruction.Operands[1], vectorCount)
                : Operand(instruction.Operands[1]);
            string space = shared ? "shared" : "global";
            // A predicated store has no destination to preserve, so keep
            // the PTX predicate on the store itself.  Lowering it as a
            // skip branch would make the following barrier divergent;
            // ACuda if-converts this into a guard diamond that reconverges
            // before any barrier in the remainder.  ACuda's vector
            // load/store path does not accept a predicate, so split a
            // predicated vector store into scalar predicated stores.
            if (!string.IsNullOrEmpty(prefix) && vectorCount > 0)
            {
                Match baseMatch = Regex.Match(instruction.Operands[1].Trim(),
                    @"^R(\d+)$", RegexOptions.IgnoreCase);
                if (!baseMatch.Success)
                    throw new NotSupportedException(
                        $"expected vector register base, got " +
                        $"'{instruction.Operands[1]}'");
                int first = int.Parse(baseMatch.Groups[1].Value,
                    CultureInfo.InvariantCulture);
                // LoadWidth maps .64/.128 to v2/v4.u32, so every
                // component is a 4-byte word.
                const int elementBytes = 4;
                for (int i = 0; i < vectorCount; i++)
                {
                    _lines.Add(
                        $"{prefix}st.{space}.u32 " +
                        $"{AddAddressOffset(address, i * elementBytes)}, " +
                        $"{Reg($"R{first + i}")};");
                }
                return;
            }
            _lines.Add($"{prefix}st.{space}.{width} {address}, {source};");
        }

        // Adds a byte offset to a rendered PTX address such as
        // "[%rdt12]" or "[__shared+0x10]".
        private static string AddAddressOffset(string address, int extra)
        {
            if (extra == 0)
                return address;
            Match match = Regex.Match(address,
                @"^\[(.+?)([+-](?:0x[0-9A-Fa-f]+|\d+))?\]$");
            if (!match.Success)
                return address;
            string inner = match.Groups[1].Value;
            int offset = 0;
            if (match.Groups[2].Success)
            {
                string text = match.Groups[2].Value;
                bool negative = text.StartsWith("-",
                    StringComparison.Ordinal);
                string digits = text[1..];
                int value = digits.StartsWith("0x",
                    StringComparison.OrdinalIgnoreCase)
                    ? Convert.ToInt32(digits, 16)
                    : int.Parse(digits, CultureInfo.InvariantCulture);
                offset = negative ? -value : value;
            }
            offset += extra;
            if (offset == 0)
                return $"[{inner}]";
            return offset > 0
                ? $"[{inner}+0x{offset:x}]"
                : $"[{inner}-0x{-offset:x}]";
        }

        private void EmitLocalLoad(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            if (instruction.Mnemonic.Contains(".128", StringComparison.OrdinalIgnoreCase))
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            bool wide = instruction.Mnemonic.Contains(
                ".64", StringComparison.OrdinalIgnoreCase);
            if (!_pointerRegs.TryGetValue("R1", out string? frame) ||
                (frame != "__callee_frame" &&
                 !frame.StartsWith("%rd", StringComparison.Ordinal)))
                throw new NotSupportedException(
                    "LDL outside an inlined callee frame");
            bool topLevel = _inlineLabelOverrides == null;
            if (topLevel && _topLevelFrameBaseMoved)
                throw new NotSupportedException(
                    "LDL after the entry-point stack pointer moved");
            string address = Address(instruction.Operands[1]);
            string destination = wide
                ? VectorRegs(instruction.Operands[0], 2)
                : Reg(instruction.Operands[0]);
            _lines.Add(
                $"{prefix}ld.local.{(wide ? "v2.u32" : "u32")} " +
                $"{destination}, {address};");
            if (topLevel)
            {
                _topLevelLocalAccessSeen = true;
                if (BaseReg(instruction.Operands[0]).Equals(
                        "R1", StringComparison.OrdinalIgnoreCase))
                    _topLevelFrameBaseMoved = true;
            }
        }

        private void EmitLocalStore(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            if (instruction.Mnemonic.Contains(".128", StringComparison.OrdinalIgnoreCase))
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            bool wide = instruction.Mnemonic.Contains(
                ".64", StringComparison.OrdinalIgnoreCase);
            if (!_pointerRegs.TryGetValue("R1", out string? frame) ||
                (frame != "__callee_frame" &&
                 !frame.StartsWith("%rd", StringComparison.Ordinal)))
                throw new NotSupportedException(
                    "STL outside an inlined callee frame");
            bool topLevel = _inlineLabelOverrides == null;
            if (topLevel && _topLevelFrameBaseMoved)
                throw new NotSupportedException(
                    "STL after the entry-point stack pointer moved");
            string address = Address(instruction.Operands[0]);
            string source = wide
                ? VectorRegs(instruction.Operands[1], 2)
                : Operand(instruction.Operands[1]);
            _lines.Add(
                $"{prefix}st.local.{(wide ? "v2.u32" : "u32")} " +
                $"{address}, {source};");
            if (topLevel)
                _topLevelLocalAccessSeen = true;
        }

        private static string SurfaceRefName(int bank, int index) =>
            $"__surf_{bank}_{index}";

        private static bool TrySurfaceDescriptor(
            SassInstruction instruction, out string reference)
        {
            reference = "";
            if (instruction.Operands.Count < 2 ||
                !TrySwitchImmediate(instruction.Operands[^2], out int bank) ||
                !TrySwitchImmediate(instruction.Operands[^1], out int index))
                return false;
            reference = SurfaceRefName(bank, index);
            return true;
        }

        private static string SurfaceGeometry(string[] parts)
        {
            if (parts.Contains("A1D"))
                return "a1d";
            if (parts.Contains("A2D"))
                return "a2d";
            if (parts.Contains("3D"))
                return "3d";
            if (parts.Contains("2D"))
                return "2d";
            if (parts.Contains("1D"))
                return "1d";
            throw new NotSupportedException("surface geometry is not supported");
        }

        private static (string Width, int Count) SurfaceWidth(string[] parts)
        {
            if (parts.Contains("128"))
                return ("v4.b32", 4);
            if (parts.Contains("64"))
                return ("v2.b32", 2);
            if (parts.Contains("16"))
                return ("b16", 1);
            if (parts.Contains("8"))
                return ("b8", 1);
            return ("b32", 1);
        }

        private static string SurfaceClamp(string[] parts) =>
            parts.Contains("ZERO") ? "zero"
            : parts.Contains("CLAMP") ? "clamp"
            : "trap";

        private string SurfaceCoordinates(string operand, int count)
        {
            string baseRegister = BaseReg(operand);
            if (!baseRegister.StartsWith(
                    "R", StringComparison.OrdinalIgnoreCase) ||
                !int.TryParse(baseRegister[1..], out int baseNumber))
                throw new NotSupportedException(
                    $"unsupported surface coordinate base '{operand}'");
            var coordinates = new List<string>(count);
            for (int i = 0; i < count; i++)
            {
                string coordinate = $"%rt{_nextTemp++}";
                _lines.Add($"    mov.b32 {coordinate}, " +
                    $"{Reg("R" + (baseNumber + i))};");
                coordinates.Add(coordinate);
            }
            return "{" + string.Join(", ", coordinates) + "}";
        }

        private void EmitSurfaceLoad(
            SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4 ||
                !TrySurfaceDescriptor(instruction, out string reference))
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            string[] parts = instruction.Mnemonic.ToUpperInvariant().Split('.');
            string geometry = SurfaceGeometry(parts);
            (string width, int count) = SurfaceWidth(parts);
            int coordinateCount = geometry switch
            {
                "1d" => 1,
                "2d" => 2,
                "3d" => 4,
                "a1d" => 2,
                "a2d" => 4,
                _ => 1,
            };
            string coordinates = SurfaceCoordinates(
                instruction.Operands[1], coordinateCount);
            string destination = count > 1
                ? VectorRegs(instruction.Operands[0], count)
                : Reg(instruction.Operands[0]);
            string clamp = SurfaceClamp(parts);
            _lines.Add($"{prefix}suld.b.{geometry}.{width}.{clamp} " +
                $"{destination}, [{reference}, {coordinates}];");
        }

        private void EmitSurfaceStore(
            SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4 ||
                !TrySurfaceDescriptor(instruction, out string reference))
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            string[] parts = instruction.Mnemonic.ToUpperInvariant().Split('.');
            string geometry = SurfaceGeometry(parts);
            (string width, int count) = SurfaceWidth(parts);
            int coordinateCount = geometry switch
            {
                "1d" => 1,
                "2d" => 2,
                "3d" => 4,
                "a1d" => 2,
                "a2d" => 4,
                _ => 1,
            };
            string coordinates = SurfaceCoordinates(
                instruction.Operands[0], coordinateCount);
            string source = count > 1
                ? VectorRegs(instruction.Operands[1], count)
                : Operand(instruction.Operands[1]);
            string space = parts.Contains("P") ? "p" : "b";
            string clamp = SurfaceClamp(parts);
            _lines.Add($"{prefix}sust.{space}.{geometry}.{width}.{clamp} " +
                $"[{reference}, {coordinates}], {source};");
        }

        private void EmitSurfaceReduce(
            SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4 ||
                !TrySurfaceDescriptor(instruction, out string reference))
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            string[] parts = instruction.Mnemonic.ToUpperInvariant().Split('.');
            string geometry = SurfaceGeometry(parts);
            string operation = parts.FirstOrDefault(part =>
                part is "ADD" or "MIN" or "MAX" or "AND" or "OR")
                ?.ToLowerInvariant() ?? throw new NotSupportedException(
                    "surface reduction operation is not supported");
            string type = parts.Contains("S64") ? "s64"
                : parts.Contains("U64") || parts.Contains("64") ? "u64"
                : parts.Contains("S32") ? "s32"
                : parts.Contains("B32") ? "b32" : "u32";
            int coordinateCount = geometry switch
            {
                "1d" => 1,
                "2d" => 2,
                "3d" => 4,
                "a1d" => 2,
                "a2d" => 4,
                _ => 1,
            };
            string coordinates = SurfaceCoordinates(
                instruction.Operands[0], coordinateCount);
            string source = type is "u64" or "s64"
                ? PackRegisterPair(instruction.Operands[1])
                : Operand(instruction.Operands[1]);
            string space = parts.Contains("P") ? "p" : "b";
            string clamp = SurfaceClamp(parts);
            _lines.Add($"{prefix}sured.{space}.{operation}.{geometry}." +
                $"{type}.{clamp} [{reference}, {coordinates}], {source};");
        }

        private void EmitSurfaceQuery(
            SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2 ||
                !instruction.Mnemonic.Contains('.', StringComparison.Ordinal))
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            string[] parts = instruction.Mnemonic.Split('.');
            string query = parts[1].ToLowerInvariant();
            string reference = NativeOperand(instruction.Operands[1], null);
            if (reference.EndsWith("+0]", StringComparison.Ordinal))
                reference = reference[..^3] + "]";
            _lines.Add($"{prefix}suq.{query}.b32 " +
                $"{Reg(instruction.Operands[0])}, {reference};");
        }

        private void EmitLdslk(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            if (mnemonic.Contains(".64", StringComparison.Ordinal))
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            string predicate = Pred(instruction.Operands[0]);
            string destination = Reg(instruction.Operands[1]);
            string addressText = instruction.Operands[2].Trim();
            if (!string.IsNullOrWhiteSpace(instruction.Guard))
                throw new NotSupportedException(
                    $"guarded LDSLK '{instruction.Mnemonic}' is not supported");
            string address = SharedAddress(addressText);
            _lines.Add($"{prefix}ld.shared.u32 {destination}, {address};");
            // Pre-Volta LDSLK reports a valid link; the retry decision is
            // driven by STSCUL's success predicate.
            _lines.Add($"{prefix}setp.eq.u32 {predicate}, 0, 0;");
            _linkedShared[addressText] = destination;
            _linkedPredicates[addressText] = predicate;
        }

        private void EmitStscul(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            if (mnemonic.Contains(".64", StringComparison.Ordinal))
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            string predicate = Pred(instruction.Operands[0]);
            string addressText = instruction.Operands[1].Trim();
            string newValue = Operand(instruction.Operands[2]);
            if (!_linkedShared.TryGetValue(addressText, out string? expected))
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' has no matching LDSLK");
            string effectivePrefix = prefix;
            if (!string.IsNullOrWhiteSpace(instruction.Guard))
            {
                if (!_linkedPredicates.TryGetValue(
                        addressText, out string? linkedPredicate) ||
                    !Pred(instruction.Guard.TrimStart('@', '!')).Equals(
                        linkedPredicate, StringComparison.Ordinal))
                    throw new NotSupportedException(
                        $"SASS instruction '{instruction.Mnemonic}' guard must " +
                        "match the preceding LDSLK predicate");
                effectivePrefix = "";
            }
            string address = SharedAddress(addressText);
            string oldValue = $"%rt{_nextTemp++}";
            _lines.Add($"{effectivePrefix}atom.shared.cas.b32 {oldValue}, " +
                $"{address}, {expected}, {newValue};");
            _lines.Add($"{effectivePrefix}setp.eq.u32 {predicate}, " +
                $"{oldValue}, {expected};");
            _linkedShared.Remove(addressText);
            _linkedPredicates.Remove(addressText);
        }

        private static int VectorCount(string width) =>
            width.StartsWith("v4", StringComparison.Ordinal) ? 4
            : width.StartsWith("v2", StringComparison.Ordinal) ? 2
            : 0;

        private string VectorRegs(string operand, int count)
        {
            Match match = Regex.Match(operand.Trim(), @"^R(\d+)$",
                RegexOptions.IgnoreCase);
            if (!match.Success)
                throw new NotSupportedException(
                    $"expected vector register base, got '{operand}'");
            int first = int.Parse(match.Groups[1].Value,
                CultureInfo.InvariantCulture);
            var registers = new List<string>(count);
            for (int i = 0; i < count; i++)
                registers.Add(Reg("R" + (first + i)));
            return "{" + string.Join(", ", registers) + "}";
        }

        private string SharedAddress(string operand)
        {
            Match match = Regex.Match(operand,
                @"^\[(R\d+|RZ)(?:\.X(1|2|4|8|16))?([+-]-?(?:0x[0-9A-Fa-f]+|\d+))?\]$",
                RegexOptions.IgnoreCase);
            if (match.Success)
            {
                string baseName = match.Groups[1].Value;
                string register = Reg(baseName);
                int scale = match.Groups[2].Success
                    ? int.Parse(match.Groups[2].Value,
                        CultureInfo.InvariantCulture)
                    : 1;
                string offset = match.Groups[3].Success
                    ? NormalizeOffset(match.Groups[3].Value)
                    : "0";
                bool dynamic = IsDynamicSharedOperand(operand);
                if (dynamic)
                    offset = AdjustOffset(offset, -_staticSharedBytes);
                string sharedBase = dynamic
                    ? "__shared_dynamic" : "__shared";
                if (_sharedAddressConstants.TryGetValue(
                        baseName, out int constant) &&
                    TrySwitchImmediate(
                        NormalizeOffset(match.Groups[3].Success
                            ? match.Groups[3].Value : "0"),
                        out int immediateOffset))
                {
                    long total = (long)constant * scale + immediateOffset;
                    if (dynamic)
                        total -= _staticSharedBytes;
                    if (total == 0)
                        return $"[{sharedBase}]";
                    return total > 0
                        ? $"[{sharedBase}+0x{total:x}]"
                        : $"[{sharedBase}-0x{-total:x}]";
                }
                if (register == "0")
                    return SharedBaseAddr(sharedBase, offset);
                int shift = scale == 16 ? 4 : scale == 8 ? 3
                    : scale == 4 ? 2 : scale == 2 ? 1 : 0;
                string offsetRegister;
                if (register.StartsWith("%rd", StringComparison.Ordinal))
                {
                    if (shift == 0)
                    {
                        offsetRegister = register;
                    }
                    else
                    {
                        offsetRegister = $"%rdt{_nextTemp++}";
                        _lines.Add($"    shl.b64 {offsetRegister}, " +
                            $"{register}, {shift};");
                    }
                }
                else
                {
                    string scaled = register;
                    if (shift != 0)
                    {
                        scaled = $"%rt{_nextTemp++}";
                        _lines.Add($"    shl.b32 {scaled}, {register}, {shift};");
                    }
                    offsetRegister = $"%rdt{_nextTemp++}";
                    _lines.Add($"    cvt.u64.u32 {offsetRegister}, {scaled};");
                }
                string derived = $"%rdt{_nextTemp++}";
                _lines.Add($"    mov.u64 {derived}, {sharedBase};");
                _lines.Add($"    add.u64 {derived}, {derived}, {offsetRegister};");
                // Keep the displacement in the address operand: a
                // reg+imm address preserves the .shared lineage, while
                // folding it into another add from a copied register
                // loses it.
                return SharedBaseAddr(derived, offset);
            }
            Match immediate = Regex.Match(operand,
                @"^\[(0x[0-9A-Fa-f]+|\d+)\]$", RegexOptions.IgnoreCase);
            if (immediate.Success)
            {
                string offset = Immediate(immediate.Groups[1].Value);
                bool dynamic = IsDynamicSharedOperand(operand);
                if (dynamic)
                    offset = AdjustOffset(offset, -_staticSharedBytes);
                return SharedBaseAddr(
                    dynamic ? "__shared_dynamic" : "__shared", offset);
            }
            throw new NotSupportedException(
                $"unsupported shared address '{operand}'");
        }

        private bool IsDynamicSharedOperand(string operand) =>
            _usesDynamicShared && _currentInstruction != null &&
            _sharedOperandConstants.TryGetValue(
                (_currentInstruction, operand), out long absolute) &&
            absolute >= _staticSharedBytes;

        // Shifts a textual address displacement by delta, formatting the
        // result the same way SharedBaseAddr expects.
        private static string AdjustOffset(string offset, long delta)
        {
            if (!TrySwitchImmediate(offset, out int value))
                return offset;
            long adjusted = value + delta;
            if (adjusted == 0)
                return "0";
            return adjusted > 0
                ? "0x" + adjusted.ToString("x", CultureInfo.InvariantCulture)
                : "-0x" + (-adjusted).ToString(
                    "x", CultureInfo.InvariantCulture);
        }

        private static bool IsSharedMemoryInstruction(string mnemonic)
        {
            string op = mnemonic.Split('.')[0].ToUpperInvariant();
            return op is "LDS" or "LDSM" or "STS" or "ATOMS"
                or "LDSLK" or "STSCUL";
        }

        private static bool IsPredicateOperand(string operand)
        {
            string text = operand.Trim().TrimStart('!');
            return Regex.IsMatch(text, @"^(?:U?P\d+|U?PT)$",
                RegexOptions.IgnoreCase);
        }

        private static string CanonicalRegister(string operand)
        {
            Match match = Regex.Match(operand, @"\bR(\d+)\b",
                RegexOptions.IgnoreCase);
            return match.Success ? "R" + match.Groups[1].Value
                : operand.Trim();
        }

        private static long AdditiveValue(
            string operand, Dictionary<string, int> map)
        {
            string text = operand.Trim().TrimStart('!');
            if (TrySwitchImmediate(text, out int immediate))
                return immediate;
            if (map.TryGetValue(CanonicalRegister(text), out int value))
                return value;
            return 0;
        }

        // Tracks the constant part of each 32-bit register.  ptxas folds
        // the static-shared base (e.g. +16) into an address register via
        // LEA/IMAD instead of spelling it as a displacement; knowing the
        // constant part is what lets SharedAddress move such an access
        // onto the dynamic base without double-counting it.
        private void UpdateAdditiveConstants(
            SassInstruction instruction, Dictionary<string, int> map)
        {
            List<string> written = WrittenRegisters(instruction)
                .Select(CanonicalRegister).Distinct().ToList();
            if (written.Count == 0)
                return;
            string primary = instruction.Operands.Count > 0
                ? CanonicalRegister(instruction.Operands[0]) : "";
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            string op = mnemonic.Split('.')[0];
            bool known = true;
            long value = 0;
            if (mnemonic.StartsWith("IMAD.MOV", StringComparison.Ordinal))
            {
                if (instruction.Operands.Count > 0)
                    value = AdditiveValue(instruction.Operands[^1], map);
                else
                    known = false;
            }
            else if (op is "MOV" or "MOV32I")
            {
                value = instruction.Operands.Count > 1
                    ? AdditiveValue(instruction.Operands[1], map) : 0;
            }
            else if (op is "IADD" or "IADD32I" or "IADD3")
            {
                for (int i = 1; i < instruction.Operands.Count; i++)
                {
                    if (IsPredicateOperand(instruction.Operands[i]))
                        continue;
                    value += AdditiveValue(instruction.Operands[i], map);
                }
            }
            else if (op == "ISUB" &&
                     !mnemonic.Contains(".X", StringComparison.Ordinal) &&
                     !mnemonic.Contains(".CC", StringComparison.Ordinal))
            {
                if (instruction.Operands.Count >= 3)
                    value = AdditiveValue(instruction.Operands[1], map) -
                        AdditiveValue(instruction.Operands[2], map);
                else
                    known = false;
            }
            else if (op == "LEA" &&
                     !mnemonic.Contains(".HI", StringComparison.Ordinal) &&
                     !mnemonic.Contains(".X", StringComparison.Ordinal))
            {
                int raIndex;
                int baseIndex;
                int shiftIndex;
                if (instruction.Operands.Count == 4)
                {
                    raIndex = 1;
                    baseIndex = 2;
                    shiftIndex = 3;
                }
                else if (instruction.Operands.Count >= 5 &&
                         IsPredicateOperand(instruction.Operands[1]))
                {
                    raIndex = 2;
                    baseIndex = 3;
                    shiftIndex = 4;
                }
                else if (instruction.Operands.Count >= 5)
                {
                    raIndex = 1;
                    baseIndex = 2;
                    shiftIndex = 4;
                }
                else
                {
                    raIndex = baseIndex = shiftIndex = -1;
                }
                if (raIndex > 0 && TrySwitchImmediate(
                        instruction.Operands[shiftIndex], out int shift) &&
                    shift >= 0 && shift < 64)
                {
                    value = (AdditiveValue(
                        instruction.Operands[raIndex], map) << shift) +
                        AdditiveValue(instruction.Operands[baseIndex], map);
                }
                else
                {
                    known = false;
                }
            }
            else if (op == "SHL" && instruction.Operands.Count >= 3 &&
                     TrySwitchImmediate(
                         instruction.Operands[2], out int shift) &&
                     shift >= 0 && shift < 64)
            {
                value = AdditiveValue(instruction.Operands[1], map) << shift;
            }
            else if (op == "IMAD" &&
                     !mnemonic.Contains(".HI", StringComparison.Ordinal) &&
                     !mnemonic.Contains(".WIDE", StringComparison.Ordinal) &&
                     !mnemonic.Contains(".X", StringComparison.Ordinal) &&
                     instruction.Operands.Count >= 4)
            {
                long addend = AdditiveValue(instruction.Operands[3], map);
                if (TrySwitchImmediate(
                        instruction.Operands[2], out int right))
                {
                    value = AdditiveValue(
                        instruction.Operands[1], map) * right + addend;
                }
                else if (TrySwitchImmediate(
                             instruction.Operands[1], out int left))
                {
                    value = AdditiveValue(
                        instruction.Operands[2], map) * left + addend;
                }
                else
                {
                    // Both factors vary: only the addend's constant
                    // survives the product.
                    value = addend;
                }
            }
            else
            {
                known = false;
            }
            foreach (string register in written)
                map[register] = known &&
                    register.Equals(primary, StringComparison.OrdinalIgnoreCase)
                    ? unchecked((int)value) : 0;
        }

        private static bool TryParseSharedAddress(
            string operand, out string baseName, out int scale,
            out long immediate)
        {
            baseName = "RZ";
            scale = 1;
            immediate = 0;
            Match match = Regex.Match(operand.Trim(),
                @"^\[(R\d+|RZ)(?:\.X(1|2|4|8|16))?([+-]-?(?:0x[0-9A-Fa-f]+|\d+))?\]$",
                RegexOptions.IgnoreCase);
            if (match.Success)
            {
                baseName = match.Groups[1].Value;
                scale = match.Groups[2].Success
                    ? int.Parse(match.Groups[2].Value,
                        CultureInfo.InvariantCulture)
                    : 1;
                if (match.Groups[3].Success && TrySwitchImmediate(
                        NormalizeOffset(match.Groups[3].Value),
                        out int offset))
                    immediate = offset;
                return true;
            }
            Match only = Regex.Match(operand.Trim(),
                @"^\[(0x[0-9A-Fa-f]+|\d+)\]$", RegexOptions.IgnoreCase);
            if (only.Success && TrySwitchImmediate(
                    only.Groups[1].Value, out int value))
            {
                immediate = value;
                return true;
            }
            return false;
        }

        // Decides whether the kernel needs an unsized `.extern .shared`
        // array.  ptxas allocates ordinary static shared variables first
        // and gives the dynamic (`extern __shared__`) region the next
        // 16-byte-aligned address; any access whose constant part reaches
        // that address is a dynamic access.  Recording the constant also
        // lets SharedAddress undo the folded static base.
        private void DetectDynamicShared()
        {
            int staticBytes = _sharedBytes > 0
                ? (_sharedBytes < 16 ? 16 : (_sharedBytes + 15) & ~15)
                : 16384;
            _staticSharedBytes = staticBytes;
            Dictionary<string, int> map =
                new(StringComparer.OrdinalIgnoreCase);
            List<SassNode> nodes = new(_section.Nodes);
            foreach (SassSection linked in _linkedSections)
            {
                if (!ReferenceEquals(linked, _section))
                    nodes.AddRange(linked.Nodes);
            }
            foreach (SassNode node in nodes)
            {
                if (node is not SassInstruction instruction)
                    continue;
                if (IsSharedMemoryInstruction(instruction.Mnemonic))
                {
                    foreach (string operand in instruction.Operands)
                    {
                        if (!operand.TrimStart().StartsWith(
                                "[", StringComparison.Ordinal))
                            continue;
                        if (!TryParseSharedAddress(
                                operand, out string baseName, out int scale,
                                out long immediate))
                            continue;
                        long absolute = AdditiveValue(baseName, map) * scale +
                            immediate;
                        _sharedOperandConstants[(instruction, operand)] =
                            absolute;
                        if (_sharedBytes > 0 && absolute >= staticBytes)
                            _usesDynamicShared = true;
                        break;
                    }
                }
                UpdateAdditiveConstants(instruction, map);
            }
        }

        private void EmitS2R(SassInstruction instruction, string prefix)
        {
            string destination = Reg(instruction.Operands[0]);
            string special = instruction.Operands[1].Trim().ToUpperInvariant();
            if (special == "SR_LEMASK")
            {
                // Lane-enable mask: the set of currently active lanes
                // (PTX activemask).
                _lines.Add($"{prefix}activemask.b32 {destination};");
                return;
            }
            _lines.Add($"{prefix}mov.u32 {destination}, " +
                $"{SpecialToPtx(special, instruction.Operands[1])};");
        }

        private void EmitUisetp(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string suffix = instruction.Mnemonic.ToUpperInvariant();
            string comparison = suffix.Contains(".GT", StringComparison.Ordinal) ? "gt"
                : suffix.Contains(".GE", StringComparison.Ordinal) ? "ge"
                : suffix.Contains(".LT", StringComparison.Ordinal) ? "lt"
                : suffix.Contains(".LE", StringComparison.Ordinal) ? "le"
                : suffix.Contains(".EQ", StringComparison.Ordinal) ? "eq"
                : "ne";
            string type = suffix.Contains(".U32", StringComparison.Ordinal)
                ? "u32" : "s32";
            string destination = Pred(instruction.Operands[0]);
            string a = UniformOperand(instruction.Operands[2]);
            string b = UniformOperand(instruction.Operands[3]);
            // Operand 1 is the second predicate output (usually UPT).
            // Operand 4 is the source predicate joined through the
            // mnemonic boolean op, as with ISETP.
            string chainText = instruction.Operands.Count >= 5
                ? instruction.Operands[4].Trim() : "UPT";
            string? chainOp = suffix.Contains(".OR", StringComparison.Ordinal) ? "or"
                : suffix.Contains(".XOR", StringComparison.Ordinal) ? "xor"
                : suffix.Contains(".AND", StringComparison.Ordinal) ? "and"
                : null;
            bool chainNegated = chainText.StartsWith("!", StringComparison.Ordinal);
            bool chainAlways = chainOp == null ||
                (chainText.Equals("UPT", StringComparison.OrdinalIgnoreCase) &&
                 !chainNegated);
            if (suffix.Contains(".EX") && instruction.Operands.Count >= 6)
            {
                string lowPredicate = Pred(instruction.Operands[5]);
                string exComparison = $"%pt{_nextTempPred++}";
                if (comparison is "eq")
                {
                    string equal = $"%pt{_nextTempPred++}";
                    _lines.Add($"    setp.eq.{type} {equal}, {a}, {b};");
                    _lines.Add($"{prefix}and.pred {exComparison}, {equal}, {lowPredicate};");
                }
                else if (comparison is "ne")
                {
                    string notEqual = $"%pt{_nextTempPred++}";
                    _lines.Add($"    setp.ne.{type} {notEqual}, {a}, {b};");
                    _lines.Add($"{prefix}or.pred {exComparison}, {notEqual}, {lowPredicate};");
                }
                else
                {
                    string highCmp = comparison is "ge" or "gt" ? "gt" : "lt";
                    string ordered = $"%pt{_nextTempPred++}";
                    string sameHigh = $"%pt{_nextTempPred++}";
                    string both = $"%pt{_nextTempPred++}";
                    _lines.Add($"    setp.{highCmp}.{type} {ordered}, {a}, {b};");
                    _lines.Add($"    setp.eq.{type} {sameHigh}, {a}, {b};");
                    _lines.Add($"    and.pred {both}, {sameHigh}, {lowPredicate};");
                    _lines.Add($"{prefix}or.pred {exComparison}, {ordered}, {both};");
                }
                if (chainAlways)
                {
                    _lines.Add($"{prefix}mov.pred {destination}, {exComparison};");
                    return;
                }
                string exChained = Pred(chainText);
                if (chainNegated)
                {
                    string inverted = $"%pt{_nextTempPred++}";
                    _lines.Add($"{prefix}not.pred {inverted}, {exChained};");
                    exChained = inverted;
                }
                _lines.Add($"{prefix}{chainOp}.pred {destination}, " +
                    $"{exComparison}, {exChained};");
                return;
            }
            string result = chainAlways
                ? destination : $"%pt{_nextTempPred++}";
            _lines.Add($"{prefix}setp.{comparison}.{type} {result}, {a}, {b};");
            if (chainAlways)
                return;
            string chained = Pred(chainText);
            if (chainNegated)
            {
                string inverted = $"%pt{_nextTempPred++}";
                _lines.Add($"{prefix}not.pred {inverted}, {chained};");
                chained = inverted;
            }
            _lines.Add($"{prefix}{chainOp}.pred {destination}, {result}, {chained};");
        }

        private void EmitS2UR(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            string destination = UniformOperand(instruction.Operands[0]);
            string special = instruction.Operands[1].Trim().ToUpperInvariant();
            _lines.Add($"{prefix}mov.u32 {destination}, " +
                $"{SpecialToPtx(special, instruction.Operands[1])};");
        }

        private static string SpecialToPtx(string special, string original)
        {
            return special switch
            {
                "SR_TID.X" => "%tid.x",
                "SR_TID.Y" => "%tid.y",
                "SR_TID.Z" => "%tid.z",
                "SR_CTAID.X" => "%ctaid.x",
                "SR_CTAID.Y" => "%ctaid.y",
                "SR_CTAID.Z" => "%ctaid.z",
                "SR_NTID.X" => "%ntid.x",
                "SR_NTID.Y" => "%ntid.y",
                "SR_NTID.Z" => "%ntid.z",
                "SR_NCTAID.X" => "%nctaid.x",
                "SR_NCTAID.Y" => "%nctaid.y",
                "SR_NCTAID.Z" => "%nctaid.z",
                "SR_LANEID" => "%laneid",
                "SR_LTMASK" => "%lanemask_lt",
                "SR_CLOCKLO" => "%clock",
                _ => throw new NotSupportedException(
                    $"S2R special register '{original}' is not supported"),
            };
        }

        private void EmitR2P(SassInstruction instruction, string prefix)
        {
            // R2P PR, Ra, Rb: copy selected bits of Ra into P0..P6.
            // Rb is either an immediate mask or a register holding the
            // mask (bit i selects predicate Pi).
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            string source = Operand(instruction.Operands[1]);
            string maskText = StripReuse(instruction.Operands[2]);
            bool dynamic = Regex.IsMatch(maskText, @"^R\d+$",
                RegexOptions.IgnoreCase);
            if (dynamic)
            {
                string mask = Operand(instruction.Operands[2]);
                for (int i = 0; i < 7; i++)
                {
                    string mbit = $"%rt{_nextTemp++}";
                    string sbit = $"%rt{_nextTemp++}";
                    string selected = $"%pt{_nextTempPred++}";
                    string value = $"%pt{_nextTempPred++}";
                    uint bit = 1u << i;
                    _lines.Add($"{prefix}and.b32 {mbit}, {mask}, {bit};");
                    _lines.Add($"{prefix}setp.ne.u32 {selected}, {mbit}, 0;");
                    _lines.Add($"{prefix}and.b32 {sbit}, {source}, {bit};");
                    _lines.Add($"    setp.ne.u32 {value}, {sbit}, 0;");
                    _lines.Add($"    @{selected} mov.pred {Pred("P" + i)}, {value};");
                }
                return;
            }
            uint maskImm = ParseImmediate(maskText);
            for (int i = 0; i < 7; i++)
            {
                if ((maskImm & (1u << i)) == 0)
                    continue;
                string bit = $"%rt{_nextTemp++}";
                _lines.Add($"{prefix}and.b32 {bit}, {source}, {1u << i};");
                _lines.Add($"{prefix}setp.ne.u32 {Pred("P" + i)}, {bit}, 0;");
            }
        }

        private void EmitP2R(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count == 2)
            {
                if (IsRz(instruction.Operands[0]))
                    return;
                if (instruction.Operands[1].Trim().Equals(
                        "PR", StringComparison.OrdinalIgnoreCase))
                {
                    // p2r_simple_: pack P0-P6 into byte 0; Ra/mask are
                    // pinned to RZ, so the upper bytes are zero.
                    string packed = PackPredicates(prefix);
                    _lines.Add($"{prefix}mov.b32 " +
                        $"{Reg(instruction.Operands[0])}, {packed};");
                    return;
                }
                _lines.Add($"{prefix}selp.b32 " +
                    $"{Reg(instruction.Operands[0])}, 1, 0, " +
                    $"{Pred(instruction.Operands[1])};");
                return;
            }
            if (instruction.Operands.Count < 4 ||
                !instruction.Operands[1].Trim().Equals(
                    "PR", StringComparison.OrdinalIgnoreCase))
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            string[] parts = instruction.Mnemonic.ToUpperInvariant().Split('.');
            int insert = parts.Contains("B3") ? 3
                : parts.Contains("B2") ? 2
                : parts.Contains("B1") ? 1 : 0;
            if (IsRz(instruction.Operands[0]))
                return;
            string predicateBits = PackPredicates(prefix);
            string mask = $"%rt{_nextTemp++}";
            _lines.Add($"{prefix}mov.b32 {mask}, " +
                $"{Operand(instruction.Operands[3])};");
            string selected = $"%rt{_nextTemp++}";
            _lines.Add($"{prefix}and.b32 {selected}, " +
                $"{predicateBits}, {mask};");
            string shiftedMask = mask;
            if (insert != 0)
            {
                string shifted = $"%rt{_nextTemp++}";
                string shiftedSelected = $"%rt{_nextTemp++}";
                _lines.Add($"{prefix}shl.b32 {shifted}, {mask}, {insert * 8};");
                _lines.Add($"{prefix}shl.b32 {shiftedSelected}, " +
                    $"{selected}, {insert * 8};");
                shiftedMask = shifted;
                selected = shiftedSelected;
            }
            string notMask = $"%rt{_nextTemp++}";
            string baseValue = $"%rt{_nextTemp++}";
            _lines.Add($"{prefix}not.b32 {notMask}, {shiftedMask};");
            _lines.Add($"{prefix}and.b32 {baseValue}, " +
                $"{Operand(instruction.Operands[2])}, {notMask};");
            _lines.Add($"{prefix}or.b32 {Reg(instruction.Operands[0])}, " +
                $"{baseValue}, {selected};");
        }

        private string PackPredicates(string prefix)
        {
            string predicateBits = $"%rt{_nextTemp++}";
            _lines.Add($"{prefix}mov.b32 {predicateBits}, 0;");
            for (int predicate = 0; predicate < 7; predicate++)
            {
                string bit = $"%rt{_nextTemp++}";
                _lines.Add($"{prefix}selp.b32 {bit}, " +
                    $"{1 << predicate}, 0, {Pred("P" + predicate)};");
                _lines.Add($"{prefix}or.b32 {predicateBits}, " +
                    $"{predicateBits}, {bit};");
            }
            return predicateBits;
        }

        private void EmitPopc(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count == 2)
            {
                _lines.Add($"{prefix}popc.b32 {Reg(instruction.Operands[0])}, " +
                    $"{Operand(instruction.Operands[1])};");
                return;
            }
            if (instruction.Operands.Count != 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            string source = Operand(instruction.Operands[1]);
            string mask = Operand(instruction.Operands[2]);
            if (source.Equals(mask, StringComparison.Ordinal))
            {
                _lines.Add($"{prefix}popc.b32 {Reg(instruction.Operands[0])}, " +
                    $"{source};");
                return;
            }
            string masked = $"%rt{_nextTemp++}";
            _lines.Add($"{prefix}and.b32 {masked}, {source}, {mask};");
            _lines.Add($"{prefix}popc.b32 {Reg(instruction.Operands[0])}, " +
                $"{masked};");
        }

        private void EmitBfe(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            string type = instruction.Mnemonic.Contains(
                ".S32", StringComparison.OrdinalIgnoreCase) ? "s32" : "u32";
            string packed = instruction.Operands[2].Trim();
            string position;
            string length;
            if (packed.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
            {
                uint value = ParseImmediate(packed);
                position = (value & 0xFF).ToString(CultureInfo.InvariantCulture);
                length = ((value >> 8) & 0xFF).ToString(CultureInfo.InvariantCulture);
            }
            else
            {
                string source = Operand(instruction.Operands[2]);
                position = $"%rt{_nextTemp++}";
                length = $"%rt{_nextTemp++}";
                _lines.Add($"    and.b32 {position}, {source}, 0xff;");
                _lines.Add($"    shr.u32 {length}, {source}, 8;");
            }
            _lines.Add($"{prefix}bfe.{type} {Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])}, {position}, {length};");
        }

        private void EmitIabs(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            if (instruction.Mnemonic.Contains(
                    ".F32", StringComparison.OrdinalIgnoreCase))
            {
                _lines.Add($"{prefix}abs.f32 " +
                    $"{Reg(instruction.Operands[0])}, " +
                    $"{Operand(instruction.Operands[1])};");
                return;
            }
            if (instruction.Mnemonic.Contains(
                    ".F64", StringComparison.OrdinalIgnoreCase))
            {
                string destination = BaseReg(instruction.Operands[0]);
                string value = $"%rdt{_nextTemp++}";
                _lines.Add($"{prefix}abs.f64 {value}, " +
                    $"{PackRegisterPair(instruction.Operands[1])};");
                UnpackPair64(value, destination, NextRegister(destination));
                return;
            }
            if (instruction.Mnemonic.Contains(
                    ".64", StringComparison.OrdinalIgnoreCase))
            {
                string destination = BaseReg(instruction.Operands[0]);
                string value = $"%rdt{_nextTemp++}";
                _lines.Add($"{prefix}abs.s64 {value}, " +
                    $"{PackRegisterPair(instruction.Operands[1])};");
                UnpackPair64(value, destination, NextRegister(destination));
                return;
            }
            _lines.Add($"{prefix}abs.s32 " +
                $"{Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])};");
        }

        private void EmitSimpleArithmetic(
            SassInstruction instruction, string prefix,
            string operation, bool signedDefault)
        {
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            bool mad = mnemonic.StartsWith("MAD", StringComparison.Ordinal) ||
                mnemonic.StartsWith("FMA", StringComparison.Ordinal);
            int sources = mad ? 3 : 2;
            if (instruction.Operands.Count < sources + 1)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs " +
                    $"{sources + 1} operands");
            bool f32 = mnemonic.Contains(".F32", StringComparison.Ordinal);
            bool f64 = mnemonic.Contains(".F64", StringComparison.Ordinal);
            bool wide = mnemonic.Contains(".64", StringComparison.Ordinal) ||
                mnemonic.Contains(".S64", StringComparison.Ordinal) ||
                mnemonic.Contains(".U64", StringComparison.Ordinal) ||
                f64;
            string type = f32 ? "f32"
                : f64 ? "f64"
                : wide
                    ? (mnemonic.Contains(".U64", StringComparison.Ordinal)
                        ? "u64" : "s64")
                    : mnemonic.Contains(".U32", StringComparison.Ordinal)
                        ? "u32" : signedDefault ? "s32" : "u32";
            if (mnemonic.StartsWith("MUL24", StringComparison.Ordinal) ||
                mnemonic.StartsWith("MAD24", StringComparison.Ordinal))
                type = "s32";
            if ((f32 || f64) && mad)
                operation = "fma.rn";
            else if ((f32 || f64) && operation == "mul.lo")
                operation = "mul";
            if (f32 &&
                mnemonic.Contains(".FTZ", StringComparison.Ordinal))
                operation += ".ftz";
            if (f32 &&
                mnemonic.Contains(".SAT", StringComparison.Ordinal))
                operation += ".sat";
            if (wide)
            {
                string destination = BaseReg(instruction.Operands[0]);
                string value = $"%rdt{_nextTemp++}";
                _lines.Add($"{prefix}{operation}.{type} {value}, " +
                    $"{PackRegisterPair(instruction.Operands[1])}, " +
                    $"{PackRegisterPair(instruction.Operands[2])}" +
                    (mad ? $", {PackRegisterPair(instruction.Operands[3])}" : "") +
                    ";");
                UnpackPair64(value, destination, NextRegister(destination));
                return;
            }
            string destinationText = instruction.Operands[0].Trim();
            bool carryOut = destinationText.EndsWith(".CC",
                StringComparison.OrdinalIgnoreCase);
            bool carryIn = mnemonic.Contains(".X",
                StringComparison.Ordinal);
            if (carryOut || carryIn)
            {
                if (f32 || f64 || wide || mad)
                    throw new NotSupportedException(
                        $"SASS instruction '{instruction.Mnemonic}' needs a " +
                        "paired carry");
                string carryDestination = Reg(carryOut
                    ? destinationText[..^3] : destinationText);
                string carryLeft = Operand(instruction.Operands[1]);
                string carryRight = Operand(instruction.Operands[2]);
                if (carryIn)
                {
                    string carryOp = operation == "sub" ? "subc"
                        : operation == "add" ? "addc" : "";
                    if (carryOp.Length == 0)
                        throw new NotSupportedException(
                            $"SASS instruction '{instruction.Mnemonic}' " +
                            "has no carry-in form");
                    _lines.Add($"{prefix}{carryOp}.u32 " +
                        $"{carryDestination}, {carryLeft}, {carryRight};");
                    return;
                }
                string outOp = operation == "sub" ? "sub.cc"
                    : operation == "add" ? "add.cc" : "";
                if (outOp.Length == 0)
                    throw new NotSupportedException(
                        $"SASS instruction '{instruction.Mnemonic}' " +
                        "has no carry-out form");
                _lines.Add($"{prefix}{outOp}.u32 " +
                    $"{carryDestination}, {carryLeft}, {carryRight};");
                return;
            }
            _lines.Add($"{prefix}{operation}.{type} " +
                $"{Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])}, " +
                $"{Operand(instruction.Operands[2])}" +
                (mad ? $", {Operand(instruction.Operands[3])}" : "") +
                ";");
        }

        private void EmitClz(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            bool wide = instruction.Mnemonic.Contains(
                "64", StringComparison.OrdinalIgnoreCase);
            _lines.Add($"{prefix}clz.{(wide ? "b64" : "b32")} " +
                $"{Reg(instruction.Operands[0])}, " +
                $"{(wide ? PackRegisterPair(instruction.Operands[1]) : Operand(instruction.Operands[1]))};");
        }

        private void EmitBfind(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            string type = mnemonic.Contains("64", StringComparison.Ordinal)
                ? (mnemonic.Contains(".S", StringComparison.Ordinal)
                    ? "s64" : "u64")
                : mnemonic.Contains(".S", StringComparison.Ordinal)
                    ? "s32" : "u32";
            string source = type.EndsWith("64", StringComparison.Ordinal)
                ? PackRegisterPair(instruction.Operands[1])
                : Operand(instruction.Operands[1]);
            _lines.Add($"{prefix}bfind.{type} " +
                $"{Reg(instruction.Operands[0])}, {source};");
        }

        private void EmitFns(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            _lines.Add($"{prefix}fns.b32 " +
                $"{Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])}, " +
                $"{Operand(instruction.Operands[2])}, " +
                $"{Operand(instruction.Operands[3])};");
        }

        private void EmitBrev(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            bool wide = instruction.Mnemonic.Contains(
                "64", StringComparison.OrdinalIgnoreCase);
            if (!wide)
            {
                _lines.Add($"{prefix}brev.b32 " +
                    $"{Reg(instruction.Operands[0])}, " +
                    $"{Operand(instruction.Operands[1])};");
                return;
            }
            string destination = BaseReg(instruction.Operands[0]);
            string value = $"%rdt{_nextTemp++}";
            _lines.Add($"{prefix}brev.b64 {value}, " +
                $"{PackRegisterPair(instruction.Operands[1])};");
            UnpackPair64(value, destination, NextRegister(destination));
        }

        private void EmitClmad(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string mode = instruction.Mnemonic.Contains(
                ".HI", StringComparison.OrdinalIgnoreCase) ? "hi" : "lo";
            string destination = BaseReg(instruction.Operands[0]);
            string value = $"%rdt{_nextTemp++}";
            _lines.Add($"{prefix}clmad.{mode}.u64 {value}, " +
                $"{PackRegisterPair(instruction.Operands[1])}, " +
                $"{PackRegisterPair(instruction.Operands[2])}, " +
                $"{PackRegisterPair(instruction.Operands[3])};");
            UnpackPair64(value, destination, NextRegister(destination));
        }

        private void EmitNeg(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            if (instruction.Mnemonic.Contains(
                    ".F32", StringComparison.OrdinalIgnoreCase))
            {
                _lines.Add($"{prefix}neg.f32 " +
                    $"{Reg(instruction.Operands[0])}, " +
                    $"{Operand(instruction.Operands[1])};");
                return;
            }
            if (instruction.Mnemonic.Contains(
                    ".F64", StringComparison.OrdinalIgnoreCase))
            {
                string destination = BaseReg(instruction.Operands[0]);
                string value = $"%rdt{_nextTemp++}";
                _lines.Add($"{prefix}neg.f64 {value}, " +
                    $"{PackRegisterPair(instruction.Operands[1])};");
                UnpackPair64(value, destination, NextRegister(destination));
                return;
            }
            if (instruction.Mnemonic.Contains(
                    ".64", StringComparison.OrdinalIgnoreCase))
            {
                string destination = BaseReg(instruction.Operands[0]);
                string value = $"%rdt{_nextTemp++}";
                _lines.Add($"{prefix}neg.s64 {value}, " +
                    $"{PackRegisterPair(instruction.Operands[1])};");
                UnpackPair64(value, destination, NextRegister(destination));
                return;
            }
            _lines.Add($"{prefix}neg.s32 " +
                $"{Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])};");
        }

        private void EmitSad(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string difference = $"%rt{_nextTemp++}";
            _lines.Add($"    sub.s32 {difference}, " +
                $"{Operand(instruction.Operands[1])}, " +
                $"{Operand(instruction.Operands[2])};");
            _lines.Add($"    abs.s32 {difference}, {difference};");
            _lines.Add($"{prefix}add.s32 " +
                $"{Reg(instruction.Operands[0])}, {difference}, " +
                $"{Operand(instruction.Operands[3])};");
        }

        private void EmitDivRem(
            SassInstruction instruction, string prefix, bool remainder)
        {
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            if (remainder &&
                (mnemonic.Contains(".F32", StringComparison.Ordinal) ||
                 mnemonic.Contains(".F64", StringComparison.Ordinal)))
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            bool f32 = mnemonic.Contains(".F32", StringComparison.Ordinal);
            bool f64 = mnemonic.Contains(".F64", StringComparison.Ordinal);
            if (f32)
            {
                _lines.Add($"{prefix}div.rn.f32 " +
                    $"{Reg(instruction.Operands[0])}, " +
                    $"{Operand(instruction.Operands[1])}, " +
                    $"{Operand(instruction.Operands[2])};");
                return;
            }
            bool wide = mnemonic.Contains(".64", StringComparison.Ordinal) ||
                mnemonic.Contains(".S64", StringComparison.Ordinal) ||
                mnemonic.Contains(".U64", StringComparison.Ordinal) ||
                f64;
            string operation = remainder ? "rem" : "div";
            string type = f64 ? "f64" : wide ? "u64" : "u32";
            if (mnemonic.Contains(".S", StringComparison.Ordinal) && !f64)
                type = wide ? "s64" : "s32";
            if (f32 || f64)
                operation = "div.rn";
            if (wide)
            {
                string destination = BaseReg(instruction.Operands[0]);
                string value = $"%rdt{_nextTemp++}";
                _lines.Add($"{prefix}{operation}.{type} {value}, " +
                    $"{PackRegisterPair(instruction.Operands[1])}, " +
                    $"{PackRegisterPair(instruction.Operands[2])};");
                UnpackPair64(value, destination, NextRegister(destination));
                return;
            }
            _lines.Add($"{prefix}{operation}.{type} " +
                $"{Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])}, " +
                $"{Operand(instruction.Operands[2])};");
        }

        private void EmitVabsdiff(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string difference = $"%rt{_nextTemp++}";
            _lines.Add($"    sub.s32 {difference}, " +
                $"{Operand(instruction.Operands[1])}, " +
                $"{Operand(instruction.Operands[2])};");
            _lines.Add($"    abs.s32 {difference}, {difference};");
            _lines.Add($"{prefix}add.s32 " +
                $"{Reg(instruction.Operands[0])}, {difference}, " +
                $"{Operand(instruction.Operands[3])};");
        }

        private void EmitVabsdiff4(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            bool signed = instruction.Mnemonic.Contains(
                ".S8", StringComparison.OrdinalIgnoreCase);
            string first = Operand(instruction.Operands[1]);
            string second = Operand(instruction.Operands[2]);
            string sum = $"%rt{_nextTemp++}";
            _lines.Add($"    mov.b32 {sum}, " +
                $"{Operand(instruction.Operands[3])};");
            for (int lane = 0; lane < 4; lane++)
            {
                string a = $"%rt{_nextTemp++}";
                string b = $"%rt{_nextTemp++}";
                string difference = $"%rt{_nextTemp++}";
                _lines.Add($"    bfe.{(signed ? "s32" : "u32")} {a}, " +
                    $"{first}, {lane * 8}, 8;");
                _lines.Add($"    bfe.{(signed ? "s32" : "u32")} {b}, " +
                    $"{second}, {lane * 8}, 8;");
                _lines.Add($"    sub.s32 {difference}, {a}, {b};");
                _lines.Add($"    abs.s32 {difference}, {difference};");
                _lines.Add($"    add.u32 {sum}, {sum}, {difference};");
            }
            _lines.Add($"{prefix}mov.b32 " +
                $"{Reg(instruction.Operands[0])}, {sum};");
        }

        private void EmitBmsk(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            string mode = instruction.Mnemonic.Contains(
                ".C", StringComparison.OrdinalIgnoreCase)
                ? "clamp" : "wrap";
            _lines.Add($"{prefix}bmsk.{mode}.b32 " +
                $"{Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])}, " +
                $"{Operand(instruction.Operands[2])};");
        }

        private void EmitUbmsk(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            string mode = instruction.Mnemonic.Contains(
                ".C", StringComparison.OrdinalIgnoreCase)
                ? "clamp" : "wrap";
            _lines.Add($"{prefix}bmsk.{mode}.b32 " +
                $"{UniformOperand(instruction.Operands[0])}, " +
                $"{UniformOperand(instruction.Operands[1])}, " +
                $"{UniformOperand(instruction.Operands[2])};");
        }

        private void EmitSgxt(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            string type = instruction.Mnemonic.Contains(
                ".U32", StringComparison.OrdinalIgnoreCase) ? "u32" : "s32";
            _lines.Add($"{prefix}bfe.{type} " +
                $"{Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])}, 0, " +
                $"{Operand(instruction.Operands[2])};");
        }

        private void EmitBfi(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string packed = instruction.Operands[2].Trim();
            string position;
            string length;
            if (packed.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
            {
                uint value = ParseImmediate(packed);
                position = (value & 0xFF).ToString(CultureInfo.InvariantCulture);
                length = ((value >> 8) & 0xFF).ToString(CultureInfo.InvariantCulture);
            }
            else
            {
                string source = Operand(instruction.Operands[2]);
                position = $"%rt{_nextTemp++}";
                length = $"%rt{_nextTemp++}";
                _lines.Add($"    and.b32 {position}, {source}, 0xff;");
                _lines.Add($"    shr.u32 {length}, {source}, 8;");
            }
            _lines.Add($"{prefix}bfi.b32 {Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])}, " +
                $"{Operand(instruction.Operands[3])}, {position}, {length};");
        }

        private void EmitPrmtAlias(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            _lines.Add($"{prefix}prmt.b32 " +
                $"{Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])}, " +
                $"{Operand(instruction.Operands[2])}, " +
                $"{Operand(instruction.Operands[3])};");
        }

        private void EmitPrmt(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            _lines.Add($"{prefix}prmt.b32 {Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])}, " +
                $"{Operand(instruction.Operands[3])}, " +
                $"{Operand(instruction.Operands[2])};");
        }

        private void EmitFlo(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count != 2 ||
                !instruction.Mnemonic.Contains(
                    ".U32", StringComparison.OrdinalIgnoreCase))
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            // FLO.U32 is the index of the highest set bit, or 0xffffffff
            // for zero.  clz.b32 + 31 - clz produces the same result and is
            // accepted by the current PTX -> SPIR-V backend.
            string destination = Reg(instruction.Operands[0]);
            string source = Operand(instruction.Operands[1]);
            string leading = $"%rt{_nextTemp++}";
            _lines.Add($"    clz.b32 {leading}, {source};");
            _lines.Add($"{prefix}sub.u32 {destination}, 31, {leading};");
        }

        private void EmitShf(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            string mode = mnemonic.Contains(".L", StringComparison.Ordinal) ? "l"
                : mnemonic.Contains(".R", StringComparison.Ordinal) ? "r"
                : throw new NotSupportedException(
                    $"SASS SHF mode '{instruction.Mnemonic}' is not supported");
            // `.W` wraps the SASS count to five bits (0..31), which PTX
            // shf.wrap expresses directly.  A `.C` (or unqualified)
            // shift covers the full 6-bit funnel count 0..63; PTX shf
            // clamps at 32, so dynamic or >32 amounts need a packed
            // 64-bit shift.  (The radix sort's high-digit passes were
            // no-ops because shift 36..60 clamped to 32.)
            bool wrapCount = mnemonic.Contains(".W", StringComparison.Ordinal);
            {
                string shiftText = StripReuse(instruction.Operands[2]);
                bool registerShift = Regex.IsMatch(shiftText,
                    @"^R\d+$", RegexOptions.IgnoreCase);
                bool wideShift = false;
                if (!registerShift)
                {
                    try
                    {
                        wideShift = ParseImmediate(shiftText) > 32;
                    }
                    catch (FormatException)
                    {
                        wideShift = true;
                    }
                    catch (OverflowException)
                    {
                        wideShift = true;
                    }
                }
                if (!wrapCount && (registerShift || wideShift))
                {
                    string wide = $"%rdt{_nextTemp++}";
                    _lines.Add($"    mov.b64 {wide}, " +
                        $"{{{Operand(instruction.Operands[1])}, " +
                        $"{Operand(instruction.Operands[3])}}};");
                    string amount = $"%rt{_nextTemp++}";
                    _lines.Add($"    and.b32 {amount}, " +
                        $"{Operand(instruction.Operands[2])}, 0x3f;");
                    if (mode == "l")
                        _lines.Add($"    shl.b64 {wide}, {wide}, {amount};");
                    else
                        _lines.Add($"    " +
                            $"{(mnemonic.Contains(".S", StringComparison.Ordinal) ? "shr.s64" : "shr.u64")} " +
                            $"{wide}, {wide}, {amount};");
                    string destination = Reg(instruction.Operands[0]);
                    if (mnemonic.Contains(".HI", StringComparison.Ordinal))
                    {
                        string high = $"%rdt{_nextTemp++}";
                        _lines.Add($"    shr.u64 {high}, {wide}, 32;");
                        _lines.Add($"    cvt.u32.u64 {destination}, {high};");
                    }
                    else
                    {
                        _lines.Add($"    cvt.u32.u64 {destination}, {wide};");
                    }
                    return;
                }
            }
            if (mnemonic.Contains(".U64", StringComparison.Ordinal) ||
                mnemonic.Contains(".S64", StringComparison.Ordinal))
            {
                // 64-bit funnel shift by a small immediate: SASS writes
                // one half; PTX shf returns that half directly.
                string wideClamp = wrapCount ? "wrap" : "clamp";
                _lines.Add($"{prefix}shf.{mode}.{wideClamp}.b32 " +
                    $"{Reg(instruction.Operands[0])}, " +
                    $"{Operand(instruction.Operands[1])}, " +
                    $"{Operand(instruction.Operands[3])}, " +
                    $"{Operand(instruction.Operands[2])};");
                return;
            }
            string clamp = mnemonic.Contains(".C", StringComparison.Ordinal)
                ? "clamp" : "wrap";
            // SASS forms the 64-bit source as {Operands[3]:Operands[1]}
            // and .HI selects its high 32 bits after the shift.  PTX shf.r
            // returns the low 32 bits of {b:a} >> n, so the high half is
            // the same funnel with a and b exchanged.  (Using the low-half
            // order here turned every fast-division quotient into
            // n << (32 - shift), which made index % size collapse to
            // index.)
            int funnelFirst = 1;
            int funnelSecond = 3;
            if (mode == "r" &&
                mnemonic.Contains(".HI", StringComparison.Ordinal))
            {
                if (mnemonic.Contains(".S", StringComparison.Ordinal) &&
                    IsRz(instruction.Operands[1]))
                {
                    _lines.Add($"{prefix}shr.s32 " +
                        $"{Reg(instruction.Operands[0])}, " +
                        $"{Operand(instruction.Operands[3])}, " +
                        $"{Operand(instruction.Operands[2])};");
                    return;
                }
                funnelFirst = 3;
                funnelSecond = 1;
            }
            else if (mode == "l" &&
                     !mnemonic.Contains(".HI", StringComparison.Ordinal))
            {
                // PTX shf.l returns the high 32 bits of {b:a} << n, while
                // SASS SHF.L.U32 (without .HI) writes the low half.  The
                // operands must be exchanged: otherwise the block/warp
                // base `SHF.L.U32 R0, R0, 0xa, RZ` became R0 >> 22 and
                // every block recomputed the first 1024 elements.
                funnelFirst = 3;
                funnelSecond = 1;
            }
            _lines.Add($"{prefix}shf.{mode}.{clamp}.b32 " +
                $"{Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[funnelFirst])}, " +
                $"{Operand(instruction.Operands[funnelSecond])}, " +
                $"{Operand(instruction.Operands[2])};");
        }

        private void EmitShflSync(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 5)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs five operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            string mode = mnemonic.Contains(".BFLY", StringComparison.Ordinal)
                ? "bfly"
                : mnemonic.Contains(".DOWN", StringComparison.Ordinal)
                    ? "down"
                    : mnemonic.Contains(".UP", StringComparison.Ordinal)
                        ? "up"
                        : mnemonic.Contains(".IDX", StringComparison.Ordinal)
                            ? "idx"
                            : throw new NotSupportedException(
                                $"SASS SHFL mode '{instruction.Mnemonic}' is not supported");
            _lines.Add($"{prefix}shfl.sync.{mode}.b32 " +
                $"{Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])}, " +
                $"{Operand(instruction.Operands[2])}, " +
                $"{Operand(instruction.Operands[3])}, " +
                $"{Operand(instruction.Operands[4])};");
        }

        private void EmitShfl(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 5)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs five operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            string mode = mnemonic.Contains(".BFLY", StringComparison.Ordinal) ? "bfly"
                : mnemonic.Contains(".DOWN", StringComparison.Ordinal) ? "down"
                : mnemonic.Contains(".UP", StringComparison.Ordinal) ? "up"
                : mnemonic.Contains(".IDX", StringComparison.Ordinal) ? "idx"
                : throw new NotSupportedException(
                    $"SASS SHFL mode '{instruction.Mnemonic}' is not supported");
            string destination = Reg(instruction.Operands[1]);
            string destinationPredicate = instruction.Operands[0].Trim();
            string predicate = destinationPredicate.Equals(
                "PT", StringComparison.OrdinalIgnoreCase)
                ? ""
                : "|" + Pred(destinationPredicate);
            string mask = $"%rt{_nextTemp++}";
            _lines.Add($"    activemask.b32 {mask};");
            _lines.Add($"{prefix}shfl.sync.{mode}.b32 {destination}{predicate}, " +
                $"{Operand(instruction.Operands[2])}, " +
                $"{Operand(instruction.Operands[3])}, " +
                $"{Operand(instruction.Operands[4])}, {mask};");
        }

        private void EmitVote(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            string destination = instruction.Operands[0].Trim();
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            string mask = $"%rt{_nextTemp++}";
            _lines.Add($"    activemask.b32 {mask};");
            if (destination.StartsWith("P", StringComparison.OrdinalIgnoreCase))
            {
                if (destination.Equals("PT", StringComparison.OrdinalIgnoreCase))
                    throw new NotSupportedException(
                        $"SASS VOTE destination '{instruction.Operands[0]}' is not supported");
                string mode = mnemonic.Contains(".ANY", StringComparison.Ordinal) ? "any"
                    : mnemonic.Contains(".UNI", StringComparison.Ordinal) ? "uni"
                    : "all";
                _lines.Add($"{prefix}vote.sync.{mode}.pred {Pred(destination)}, " +
                    $"{Pred(instruction.Operands[1])}, {mask};");
                return;
            }
            if (!destination.StartsWith("R", StringComparison.OrdinalIgnoreCase) ||
                instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS VOTE destination '{instruction.Operands[0]}' is not supported");
            _lines.Add($"{prefix}vote.sync.ballot.b32 {Reg(destination)}, " +
                $"{Pred(instruction.Operands[2])}, {mask};");
        }

        private void EmitVoteU(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            bool any = mnemonic.Contains(".ANY", StringComparison.Ordinal);
            bool all = mnemonic.Contains(".ALL", StringComparison.Ordinal);
            if (!any && !all)
                throw new NotSupportedException(
                    $"SASS VOTEU mode '{instruction.Mnemonic}' is not supported");
            string gate = instruction.Operands[1].Trim();
            if (!gate.Equals("UPT", StringComparison.OrdinalIgnoreCase) &&
                !gate.Equals("!UPT", StringComparison.OrdinalIgnoreCase) &&
                !gate.Equals("PT", StringComparison.OrdinalIgnoreCase))
                throw new NotSupportedException(
                    $"SASS VOTEU uniform predicate '{gate}' is not supported");
            string destination = UniformOperand(instruction.Operands[0]);
            if (gate.StartsWith("!", StringComparison.Ordinal))
            {
                _lines.Add($"{prefix}mov.u32 {destination}, 0;");
                return;
            }
            string mask = $"%rt{_nextTemp++}";
            string result = $"%pt{_nextTempPred++}";
            _lines.Add($"    activemask.b32 {mask};");
            _lines.Add($"    vote.sync.{(any ? "any" : "all")}.pred " +
                $"{result}, {Pred(instruction.Operands[2])}, {mask};");
            _lines.Add($"{prefix}selp.u32 {destination}, 1, 0, {result};");
        }

        private void EmitAtom(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            if (mnemonic.Contains("CAST.SPIN", StringComparison.Ordinal))
            {
                EmitAtomsCastSpin(instruction, prefix);
                return;
            }
            string mode = mnemonic.Contains(".ADD", StringComparison.Ordinal) ? "add"
                : mnemonic.Contains(".INC", StringComparison.Ordinal) ? "inc"
                : mnemonic.Contains(".DEC", StringComparison.Ordinal) ? "dec"
                : mnemonic.Contains(".MIN", StringComparison.Ordinal) ? "min"
                : mnemonic.Contains(".MAX", StringComparison.Ordinal) ? "max"
                : mnemonic.Contains(".AND", StringComparison.Ordinal) ? "and"
                : mnemonic.Contains(".OR", StringComparison.Ordinal) ? "or"
                : mnemonic.Contains(".XOR", StringComparison.Ordinal) ? "xor"
                : mnemonic.Contains(".EXCH", StringComparison.Ordinal) ? "exch"
                : throw new NotSupportedException(
                    $"SASS ATOM mode '{instruction.Mnemonic}' is not supported");
            if (mnemonic.Contains(".64", StringComparison.Ordinal) ||
                mnemonic.Contains(".U64", StringComparison.Ordinal) ||
                mnemonic.Contains(".S64", StringComparison.Ordinal))
            {
                if (instruction.Operands.Count < 4 ||
                    mode != "add" ||
                    !instruction.Operands[0].Trim().Equals(
                        "PT", StringComparison.OrdinalIgnoreCase))
                    throw new NotSupportedException(
                        $"SASS instruction '{instruction.Mnemonic}' is not supported");
                bool wideShared = mnemonic.StartsWith(
                    "ATOMS", StringComparison.Ordinal);
                string wideAddress = wideShared
                    ? SharedAddress(instruction.Operands[2])
                    : Address(instruction.Operands[2]);
                string wideSpace = wideShared ? "shared" : "global";
                string value = PackRegisterPair(instruction.Operands[3]);
                string destination = BaseReg(instruction.Operands[1]);
                if (destination.Equals("RZ", StringComparison.OrdinalIgnoreCase))
                {
                    _lines.Add($"{prefix}red.{wideSpace}.add.u64 " +
                        $"{wideAddress}, {value};");
                    return;
                }
                string old = $"%rdt{_nextTemp++}";
                _lines.Add($"{prefix}atom.{wideSpace}.add.u64 " +
                    $"{old}, {wideAddress}, {value};");
                UnpackPair64(old, destination, NextRegister(destination));
                return;
            }
            string type = mode is "and" or "or" or "xor" or "exch"
                ? "b32"
                : mnemonic.Contains(".S32", StringComparison.Ordinal)
                    ? "s32" : "u32";
            // Some ATOM forms carry a leading return predicate, e.g.
            //   ATOMG.E.ADD.STRONG.GPU PT, R5, [R4.64], R7
            int first = 0;
            if (instruction.Operands.Count >= 4 &&
                Regex.IsMatch(instruction.Operands[0].Trim(),
                    @"^!?U?P\d+$|^!?U?PT$", RegexOptions.IgnoreCase))
                first = 1;
            if (first + 2 >= instruction.Operands.Count)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs an address and value");
            bool shared = mnemonic.StartsWith("ATOMS",
                StringComparison.Ordinal);
            string address = shared
                ? SharedAddress(instruction.Operands[first + 1])
                : Address(instruction.Operands[first + 1]);
            string space = shared ? "shared" : "global";
            string atomValue = Operand(instruction.Operands[first + 2]);
            if (BaseReg(instruction.Operands[first]).Equals(
                    "RZ", StringComparison.OrdinalIgnoreCase))
            {
                // Result discarded: PTX requires `red` (no destination).
                _lines.Add($"{prefix}red.{space}.{mode}.{type} " +
                    $"{address}, {atomValue};");
                return;
            }
            _lines.Add($"{prefix}atom.{space}.{mode}.{type} " +
                $"{Reg(instruction.Operands[first])}, {address}, " +
                $"{atomValue};");
        }

        private void EmitAtomsCastSpin(
            SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string destination = BaseReg(instruction.Operands[0]);
            string address = SharedAddress(instruction.Operands[1]);
            string expected = PackRegisterPair(instruction.Operands[2]);
            string value = PackRegisterPair(instruction.Operands[3]);
            string old = $"%rdt{_nextTemp++}";
            string retry = $"L_cas_spin_{_nextTemp++}";
            _lines.Add($"{retry}:");
            _lines.Add($"    atom.shared.cas.b64 {old}, {address}, " +
                $"{expected}, {value};");
            string mismatch = $"%pt{_nextTempPred++}";
            _lines.Add($"    setp.ne.u64 {mismatch}, {old}, {expected};");
            _lines.Add($"    @{mismatch} bra {retry};");
            _lines.Add($"{prefix}mov.u32 {Reg(destination)}, 1;");
            _lines.Add($"{prefix}mov.u32 " +
                $"{Reg(NextRegister(destination))}, 0;");
        }

        private void EmitBitwise(
            SassInstruction instruction, string prefix, string operation)
        {
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            bool wide = instruction.Mnemonic.Contains(
                "64", StringComparison.OrdinalIgnoreCase);
            if (wide)
            {
                string destination = BaseReg(instruction.Operands[0]);
                string value = $"%rdt{_nextTemp++}";
                _lines.Add($"{prefix}{operation}.b64 {value}, " +
                    $"{PackRegisterPair(instruction.Operands[1])}, " +
                    $"{PackRegisterPair(instruction.Operands[2])};");
                UnpackPair64(value, destination, NextRegister(destination));
                return;
            }
            _lines.Add($"{prefix}{operation}.b32 " +
                $"{Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])}, " +
                $"{Operand(instruction.Operands[2])};");
        }

        private void EmitNot(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            if (instruction.Operands[0].Trim().StartsWith(
                    "P", StringComparison.OrdinalIgnoreCase))
            {
                _lines.Add($"{prefix}not.pred " +
                    $"{Pred(instruction.Operands[0])}, " +
                    $"{Pred(instruction.Operands[1])};");
                return;
            }
            bool wide = instruction.Mnemonic.Contains(
                "64", StringComparison.OrdinalIgnoreCase);
            if (wide)
            {
                string destination = BaseReg(instruction.Operands[0]);
                string value = $"%rdt{_nextTemp++}";
                _lines.Add($"{prefix}not.b64 {value}, " +
                    $"{PackRegisterPair(instruction.Operands[1])};");
                UnpackPair64(value, destination, NextRegister(destination));
                return;
            }
            _lines.Add($"{prefix}not.b32 " +
                $"{Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])};");
        }

        private void EmitCnot(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            bool wide = instruction.Mnemonic.Contains(
                "64", StringComparison.OrdinalIgnoreCase);
            if (wide)
            {
                string destination = BaseReg(instruction.Operands[0]);
                string value = $"%rdt{_nextTemp++}";
                _lines.Add($"{prefix}cnot.b64 {value}, " +
                    $"{PackRegisterPair(instruction.Operands[1])};");
                UnpackPair64(value, destination, NextRegister(destination));
                return;
            }
            _lines.Add($"{prefix}cnot.b32 " +
                $"{Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])};");
        }

        private void EmitShift(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            bool wide = instruction.Mnemonic.Contains(
                "64", StringComparison.OrdinalIgnoreCase);
            string operation = instruction.Mnemonic.StartsWith(
                "SHL", StringComparison.OrdinalIgnoreCase) ? "shl" : "shr";
            if (wide)
            {
                string destination = BaseReg(instruction.Operands[0]);
                string value = $"%rdt{_nextTemp++}";
                _lines.Add($"{prefix}{operation}.b64 {value}, " +
                    $"{PackRegisterPair(instruction.Operands[1])}, " +
                    $"{Operand(instruction.Operands[2])};");
                UnpackPair64(value, destination, NextRegister(destination));
                return;
            }
            _lines.Add($"{prefix}{operation}.b32 " +
                $"{Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])}, " +
                $"{Operand(instruction.Operands[2])};");
        }

        private void EmitLop(SassInstruction instruction, string prefix)
        {
            string suffix = instruction.Mnemonic.ToUpperInvariant();
            if (suffix.StartsWith("LOP3") && instruction.Operands.Count >= 5)
            {
                EmitLop3Lut(instruction, prefix);
                return;
            }
            string operation = suffix.Contains(".AND") ? "and"
                : suffix.Contains(".OR") ? "or"
                : suffix.Contains(".XOR") ? "xor"
                : "and";
            string dst = instruction.Operands[0].Trim();
            string dstT = dst.TrimStart('!');
            bool pdst = dstT.StartsWith("P", StringComparison.OrdinalIgnoreCase) ||
                        dstT.StartsWith("UP", StringComparison.OrdinalIgnoreCase);
            if (pdst)
            {
                // Maxwell/Pascal predicate forms carry both a discarded
                // (or live) register destination and two sources:
                //   LOP.AND.NZ P0, RZ, R9, 0x40000
                string predicate = Pred(dst);
                int sourceStart = instruction.Operands.Count >= 4 ? 2 : 1;
                int registerDestination =
                    instruction.Operands.Count >= 4 ? 1 : -1;
                if (instruction.Operands.Count <= sourceStart + 1)
                    throw new NotSupportedException(
                        $"SASS instruction '{instruction.Mnemonic}' needs " +
                        "two predicate sources");
                string a = Operand(instruction.Operands[sourceStart]);
                string b = Operand(instruction.Operands[sourceStart + 1]);
                string result = $"%rt{_nextTemp++}";
                _lines.Add($"    {operation}.b32 {result}, {a}, {b};");
                _lines.Add($"{prefix}setp.ne.u32 {predicate}, {result}, 0;");
                if (registerDestination >= 0)
                {
                    string regDest =
                        instruction.Operands[registerDestination].Trim();
                    if (!IsRz(regDest))
                        _lines.Add($"{prefix}mov.b32 {Reg(regDest)}, " +
                            $"{result};");
                }
                return;
            }
            _lines.Add($"{prefix}{operation}.b32 {Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])}, {Operand(instruction.Operands[2])};");
        }

        // SASS REDUX.<op> reduces across the warp and writes a uniform
        // register.  PTX redux.sync writes a regular register; copy the
        // result into the uniform destination afterwards.
        // R2UR moves a vector register to the uniform file.  The optional
        // third operand makes it a uniformity test: Pd reports whether all
        // lanes in the warp hold the same value (used to guard BRA.DIV).
        private void EmitR2ur(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count >= 3)
            {
                string predicate = Pred(instruction.Operands[0].Trim());
                string uniformDst = instruction.Operands[1].Trim();
                string source = Operand(instruction.Operands[2]);
                string shuffled = $"%rt{_nextTemp++}";
                string equal = $"%pt{_nextTempPred++}";
                string mask = $"%rt{_nextTemp++}";
                _lines.Add($"    activemask.b32 {mask};");
                _lines.Add($"{prefix}shfl.sync.idx.b32 {shuffled}, {source}, " +
                    $"0, 0x1f, {mask};");
                _lines.Add($"    setp.eq.s32 {equal}, {source}, {shuffled};");
                _lines.Add($"{prefix}vote.sync.all.pred {predicate}, " +
                    $"{equal}, {mask};");
                if (!uniformDst.Equals("URZ",
                        StringComparison.OrdinalIgnoreCase))
                    _lines.Add($"{prefix}mov.u32 " +
                        $"{UniformOperand(uniformDst)}, {source};");
                return;
            }
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            _lines.Add($"{prefix}mov.u32 " +
                $"{UniformOperand(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])};");
        }

        private void EmitRedux(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 2)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs two operands");
            string mnemonic = instruction.Mnemonic.ToUpperInvariant();
            string operation = mnemonic.Contains(".OR", StringComparison.Ordinal)
                ? "or.b32"
                : mnemonic.Contains(".AND", StringComparison.Ordinal)
                    ? "and.b32"
                    : mnemonic.Contains(".XOR", StringComparison.Ordinal)
                        ? "xor.b32"
                        : mnemonic.Contains(".MIN", StringComparison.Ordinal)
                            ? (mnemonic.Contains(".S", StringComparison.Ordinal)
                                ? "min.s32" : "min.u32")
                            : mnemonic.Contains(".MAX", StringComparison.Ordinal)
                                ? (mnemonic.Contains(".S", StringComparison.Ordinal)
                                    ? "max.s32" : "max.u32")
                                : "add.u32";
            string destination = instruction.Operands[0].Trim();
            string source = Operand(instruction.Operands[1]);
            string result = $"%rt{_nextTemp++}";
            _lines.Add($"{prefix}redux.sync.{operation} {result}, " +
                $"{source}, 0xffffffff;");
            if (destination.StartsWith("UR", StringComparison.OrdinalIgnoreCase))
                _lines.Add($"{prefix}mov.u32 {UniformOperand(destination)}, " +
                    $"{result};");
            else
                _lines.Add($"{prefix}mov.b32 {Reg(destination)}, {result};");
        }

        private void EmitUlop3(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 5)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs five operands");
            string destination = Regex.Replace(
                StripReuse(instruction.Operands[0]), @"\.CC$", "",
                RegexOptions.IgnoreCase);
            string dest = UniformOperand(destination);
            string a = UniformOperand(instruction.Operands[1]);
            string b = UniformOperand(instruction.Operands[2]);
            string c = UniformOperand(instruction.Operands[3]);
            int lut = (int)(ParseImmediate(instruction.Operands[4]) & 0xFF);
            if (lut == 0)
            {
                _lines.Add($"{prefix}mov.u32 {dest}, 0;");
                return;
            }
            string notA = $"%rt{_nextTemp++}";
            string notB = $"%rt{_nextTemp++}";
            string notC = $"%rt{_nextTemp++}";
            _lines.Add($"    not.b32 {notA}, {a};");
            _lines.Add($"    not.b32 {notB}, {b};");
            _lines.Add($"    not.b32 {notC}, {c};");
            string? result = null;
            for (int index = 0; index < 8; index++)
            {
                if ((lut & (1 << index)) == 0)
                    continue;
                string termA = ((index >> 2) & 1) != 0 ? a : notA;
                string termB = ((index >> 1) & 1) != 0 ? b : notB;
                string termC = (index & 1) != 0 ? c : notC;
                string term = $"%rt{_nextTemp++}";
                _lines.Add($"    and.b32 {term}, {termA}, {termB};");
                _lines.Add($"    and.b32 {term}, {term}, {termC};");
                if (result == null)
                    result = term;
                else
                    _lines.Add($"    or.b32 {result}, {result}, {term};");
            }
            _lines.Add($"{prefix}mov.u32 {dest}, " +
                $"{(result ?? "0")};");
        }

        private void EmitLop3Lut(SassInstruction instruction, string prefix)
        {
            // LOP3 can write both a predicate and a register.  When the
            // predicate destination is used, the printed form is
            //   LOP3.LUT Pu, Rd, Ra, Rb, Rc, imm8, Pp
            // and the predicate becomes the "result is non-zero" test.
            int first = 0;
            string? predicateDestination = null;
            if (instruction.Operands.Count >= 7 &&
                Regex.IsMatch(instruction.Operands[0].Trim(),
                    @"^!?U?P\d+$|^!?U?PT$", RegexOptions.IgnoreCase))
            {
                predicateDestination = instruction.Operands[0].Trim();
                first = 1;
            }
            string destination = Reg(instruction.Operands[first]);
            string a = Operand(instruction.Operands[first + 1]);
            string b = Operand(instruction.Operands[first + 2]);
            string c = Operand(instruction.Operands[first + 3]);
            int lut = (int)(ParseImmediate(instruction.Operands[first + 4]) & 0xFF);
            if (lut == 0)
            {
                if (destination != "0")
                    _lines.Add($"{prefix}mov.u32 {destination}, 0;");
                if (predicateDestination != null)
                    _lines.Add($"    mov.pred {LopPredicate(predicateDestination)}, 0;");
                return;
            }
            string notA = $"%rt{_nextTemp++}";
            string notB = $"%rt{_nextTemp++}";
            string notC = $"%rt{_nextTemp++}";
            _lines.Add($"    not.b32 {notA}, {a};");
            _lines.Add($"    not.b32 {notB}, {b};");
            _lines.Add($"    not.b32 {notC}, {c};");
            string? result = null;
            for (int index = 0; index < 8; index++)
            {
                if ((lut & (1 << index)) == 0)
                    continue;
                string termA = ((index >> 2) & 1) != 0 ? a : notA;
                string termB = ((index >> 1) & 1) != 0 ? b : notB;
                string termC = (index & 1) != 0 ? c : notC;
                string term = $"%rt{_nextTemp++}";
                _lines.Add($"    and.b32 {term}, {termA}, {termB};");
                _lines.Add($"    and.b32 {term}, {term}, {termC};");
                if (result == null)
                {
                    result = term;
                }
                else
                {
                    _lines.Add($"    or.b32 {result}, {result}, {term};");
                }
            }
            if (result == null)
            {
                if (destination != "0")
                    _lines.Add($"{prefix}mov.u32 {destination}, 0;");
            }
            else if (!result.Equals(destination, StringComparison.Ordinal) &&
                     destination != "0")
            {
                _lines.Add($"{prefix}mov.b32 {destination}, {result};");
            }
            if (predicateDestination != null)
            {
                string predicate = LopPredicate(predicateDestination);
                if (result == null)
                    _lines.Add($"    mov.pred {predicate}, 0;");
                else
                    _lines.Add($"    setp.ne.u32 {predicate}, {result}, 0;");
            }
        }

        private string LopPredicate(string operand)        {
            string text = StripReuse(operand).TrimStart('!');
            if (text.StartsWith("U", StringComparison.OrdinalIgnoreCase))
                text = text[1..];
            return Pred(text);
        }

        private static bool TryParseImmediate(string text, out uint result)
        {
            result = 0;
            string value = text.Trim();
            if (value.StartsWith("R", StringComparison.OrdinalIgnoreCase) ||
                value.StartsWith("UR", StringComparison.OrdinalIgnoreCase) ||
                value.StartsWith("c[", StringComparison.OrdinalIgnoreCase) ||
                value.StartsWith("SR_", StringComparison.OrdinalIgnoreCase))
                return false;
            try
            {
                result = ParseImmediate(value);
                return true;
            }
            catch (FormatException)
            {
                return false;
            }
            catch (OverflowException)
            {
                return false;
            }
        }

        private static uint ParseImmediate(string text)
        {
            string value = text.Trim();
            if (value.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
            {
                return uint.Parse(value[2..], NumberStyles.HexNumber,
                    CultureInfo.InvariantCulture);
            }
            if (value.StartsWith("-0x", StringComparison.OrdinalIgnoreCase))
                return unchecked((uint)(-long.Parse(value[3..],
                    NumberStyles.HexNumber, CultureInfo.InvariantCulture)));
            return uint.Parse(value, CultureInfo.InvariantCulture);
        }

        private void EmitSel(SassInstruction instruction, string prefix)
        {
            string first = Operand(instruction.Operands[1]);
            string second = Operand(instruction.Operands[2]);
            string predicateText = instruction.Operands[3].Trim();
            if (predicateText.StartsWith("!", StringComparison.Ordinal))
            {
                (first, second) = (second, first);
            }
            _lines.Add($"{prefix}selp.b32 {Reg(instruction.Operands[0])}, " +
                $"{first}, {second}, {Pred(predicateText)};");
        }

        private void EmitIcmp(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string suffix = instruction.Mnemonic.ToUpperInvariant();
            string comparison = suffix.Contains(".GT") ? "gt"
                : suffix.Contains(".GE") ? "ge"
                : suffix.Contains(".LT") ? "lt"
                : suffix.Contains(".LE") ? "le"
                : suffix.Contains(".EQ") ? "eq"
                : suffix.Contains(".NE") ? "ne"
                : "eq";
            string type = suffix.Contains(".U32") ? "u32" : "s32";
            string predicate = $"%pt{_nextTempPred++}";
            _lines.Add($"setp.{comparison}.{type} {predicate}, " +
                $"{Operand(instruction.Operands[3])}, 0;");
            _lines.Add($"{prefix}selp.b32 {Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])}, " +
                $"{Operand(instruction.Operands[2])}, {predicate};");
        }

        private static string SetComparison(string mnemonic)
        {
            string upper = mnemonic.ToUpperInvariant();
            if (upper.Contains(".LTU", StringComparison.Ordinal) ||
                upper.Contains(".LO", StringComparison.Ordinal)) return "ltu";
            if (upper.Contains(".LEU", StringComparison.Ordinal) ||
                upper.Contains(".LS", StringComparison.Ordinal)) return "leu";
            if (upper.Contains(".GTU", StringComparison.Ordinal) ||
                upper.Contains(".HI", StringComparison.Ordinal)) return "gtu";
            if (upper.Contains(".GEU", StringComparison.Ordinal) ||
                upper.Contains(".HS", StringComparison.Ordinal)) return "geu";
            if (upper.Contains(".NAN", StringComparison.Ordinal)) return "nan";
            if (upper.Contains(".NUM", StringComparison.Ordinal)) return "num";
            if (upper.Contains(".EQU", StringComparison.Ordinal)) return "equ";
            if (upper.Contains(".NEU", StringComparison.Ordinal)) return "neu";
            if (upper.Contains(".LT", StringComparison.Ordinal)) return "lt";
            if (upper.Contains(".LE", StringComparison.Ordinal)) return "le";
            if (upper.Contains(".GT", StringComparison.Ordinal)) return "gt";
            if (upper.Contains(".GE", StringComparison.Ordinal)) return "ge";
            if (upper.Contains(".NE", StringComparison.Ordinal)) return "ne";
            return "eq";
        }

        private static string SetBoolOperation(string mnemonic)
        {
            string upper = mnemonic.ToUpperInvariant();
            if (upper.Contains(".XOR", StringComparison.Ordinal)) return "xor";
            if (upper.Contains(".OR", StringComparison.Ordinal)) return "or";
            if (upper.Contains(".AND", StringComparison.Ordinal)) return "and";
            return "";
        }

        private static List<string> SetTypes(string mnemonic)
        {
            var types = new List<string>();
            foreach (string part in mnemonic.ToUpperInvariant().Split('.'))
            {
                string type = part.ToLowerInvariant();
                if (type is "u8" or "s8" or "u16" or "s16" or "u32"
                    or "s32" or "u64" or "s64" or "f16" or "f32" or "f64"
                    or "b16" or "b32" or "b64")
                    types.Add(type);
            }
            return types;
        }

        private void EmitSet(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            string comparison = SetComparison(instruction.Mnemonic);
            string boolean = SetBoolOperation(instruction.Mnemonic);
            List<string> types = SetTypes(instruction.Mnemonic);
            string destinationType = types.Count > 0 ? types[0] : "u32";
            string sourceType = types.Count > 1 ? types[1] : "s32";
            string sourceA = sourceType is "f64" or "s64" or "u64"
                ? PackRegisterPair(instruction.Operands[1])
                : Operand(instruction.Operands[1]);
            string sourceB = sourceType is "f64" or "s64" or "u64"
                ? PackRegisterPair(instruction.Operands[2])
                : Operand(instruction.Operands[2]);
            string operation = $"set.{comparison}" +
                (boolean.Length > 0 ? "." + boolean : "") +
                $".{destinationType}.{sourceType}";
            string tail = string.Empty;
            if (instruction.Operands.Count > 3 &&
                !instruction.Operands[3].Trim().Equals(
                    "PT", StringComparison.OrdinalIgnoreCase))
                tail = ", " + PredicateOperand(instruction.Operands[3]);
            _lines.Add($"{prefix}{operation} " +
                $"{Reg(instruction.Operands[0])}, {sourceA}, {sourceB}{tail};");
        }

        private void EmitSetp(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs three operands");
            string comparison = SetComparison(instruction.Mnemonic);
            string boolean = SetBoolOperation(instruction.Mnemonic);
            List<string> types = SetTypes(instruction.Mnemonic);
            string type = types.Count > 0 ? types[0] : "u32";
            string sourceA = type is "f64" or "s64" or "u64"
                ? PackRegisterPair(instruction.Operands[1])
                : Operand(instruction.Operands[1]);
            string sourceB = type is "f64" or "s64" or "u64"
                ? PackRegisterPair(instruction.Operands[2])
                : Operand(instruction.Operands[2]);
            string operation = $"setp.{comparison}" +
                (boolean.Length > 0 ? "." + boolean : "") +
                $".{type}";
            string tail = string.Empty;
            if (instruction.Operands.Count > 3 &&
                !instruction.Operands[3].Trim().Equals(
                    "PT", StringComparison.OrdinalIgnoreCase))
                tail = ", " + PredicateOperand(instruction.Operands[3]);
            _lines.Add($"{prefix}{operation} " +
                $"{Pred(instruction.Operands[0])}, {sourceA}, {sourceB}{tail};");
        }

        private void EmitSelp(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            List<string> types = SetTypes(instruction.Mnemonic);
            string type = types.Count > 0 ? types[0] : "u32";
            string a = type is "f64" or "s64" or "u64"
                ? PackRegisterPair(instruction.Operands[1])
                : Operand(instruction.Operands[1]);
            string b = type is "f64" or "s64" or "u64"
                ? PackRegisterPair(instruction.Operands[2])
                : Operand(instruction.Operands[2]);
            string selectPredicate = PredicateOperand(instruction.Operands[3]);
            if (type is "f64" or "s64" or "u64")
            {
                string value = $"%rdt{_nextTemp++}";
                _lines.Add($"{prefix}selp.{type} {value}, {a}, {b}, " +
                    $"{selectPredicate};");
                string destination = BaseReg(instruction.Operands[0]);
                UnpackPair64(value, destination, NextRegister(destination));
                return;
            }
            _lines.Add($"{prefix}selp.{type} " +
                $"{Reg(instruction.Operands[0])}, {a}, {b}, " +
                $"{selectPredicate};");
        }

        private void EmitSlct(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            List<string> types = SetTypes(instruction.Mnemonic);
            string destinationType = types.Count > 0 ? types[0] : "s32";
            string conditionType = types.Count > 1 ? types[1] : "s32";
            string suffix = instruction.Mnemonic.Contains(
                ".FTZ", StringComparison.OrdinalIgnoreCase)
                ? ".ftz" : "";
            _lines.Add($"{prefix}slct{suffix}.{destinationType}.{conditionType} " +
                $"{Reg(instruction.Operands[0])}, " +
                $"{Operand(instruction.Operands[1])}, " +
                $"{Operand(instruction.Operands[2])}, " +
                $"{Operand(instruction.Operands[3])};");
        }

        private void EmitIset(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string suffix = instruction.Mnemonic.ToUpperInvariant();
            string comparison = suffix.Contains(".GT") ? "gt"
                : suffix.Contains(".GE") ? "ge"
                : suffix.Contains(".LT") ? "lt"
                : suffix.Contains(".LE") ? "le"
                : suffix.Contains(".EQ") ? "eq"
                : suffix.Contains(".NE") ? "ne"
                : "eq";
            string type = suffix.Contains(".U32") ? "u32" : "s32";
            string bitOperation = suffix.Contains(".AND") ? "and.b32"
                : suffix.Contains(".OR") ? "or.b32"
                : suffix.Contains(".XOR") ? "xor.b32"
                : "and.b32";
            string comparisonPredicate = $"%pt{_nextTempPred++}";
            _lines.Add($"setp.{comparison}.{type} " +
                $"{comparisonPredicate}, {Operand(instruction.Operands[1])}, " +
                $"{Operand(instruction.Operands[2])};");
            string sourcePredicate = instruction.Operands[3].Trim();
            string resultPredicate;
            if (sourcePredicate.Equals("PT", StringComparison.OrdinalIgnoreCase))
            {
                resultPredicate = comparisonPredicate;
            }
            else
            {
                string comparisonValue = $"%rt{_nextTemp++}";
                _lines.Add($"selp.b32 {comparisonValue}, 1, 0, " +
                    $"{comparisonPredicate};");
                string predicateValue = PredicateValue(sourcePredicate, "");
                string combined = $"%rt{_nextTemp++}";
                _lines.Add($"{bitOperation} {combined}, " +
                    $"{comparisonValue}, {predicateValue};");
                resultPredicate = $"%pt{_nextTempPred++}";
                _lines.Add($"setp.ne.u32 {resultPredicate}, {combined}, 0;");
            }
            _lines.Add($"{prefix}selp.b32 {Reg(instruction.Operands[0])}, " +
                $"1, 0, {resultPredicate};");
        }

        private void EmitPsetp(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 5)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs five operands");
            string suffix = instruction.Mnemonic.ToUpperInvariant();
            string[] parts = suffix.Split('.');
            if (parts.Length < 3)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' is not supported");
            string firstOperation = PredicateOperation(parts[1]);
            string secondOperation = PredicateOperation(parts[2]);
            string a = PredicateValue(instruction.Operands[2], prefix);
            string b = PredicateValue(instruction.Operands[3], prefix);
            string c = PredicateValue(instruction.Operands[4], prefix);
            string temporary = $"%rt{_nextTemp++}";
            _lines.Add($"{prefix}{firstOperation} {temporary}, {a}, {b};");
            string result = $"%rt{_nextTemp++}";
            _lines.Add($"{prefix}{secondOperation} {result}, {temporary}, {c};");
            _lines.Add($"{prefix}setp.ne.u32 " +
                $"{Pred(instruction.Operands[0])}, {result}, 0;");
        }

        private void EmitPlop3(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 6)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs six operands");
            string destination = instruction.Operands[0].Trim();
            if (destination.Equals("PT", StringComparison.OrdinalIgnoreCase))
                return;
            // Printed form: PLOP3.LUT D, Pp, A, B, C, immLut, immAB.
            // Pp is the predicate-control operand; the truth table is a
            // three-input function of A, B and C.  Treating Pp as a
            // fourth input turned `P0 = UP0` (PLOP3 ... 0x80) into
            // `P0 = 0`, which disabled the bitonic-sort range guard.
            uint lut = ParseImmediate(instruction.Operands[5]) & 0xFF;

            bool?[] constants = new bool?[3];
            string?[] predicates = new string?[3];
            for (int i = 0; i < 3; i++)
            {
                string text = instruction.Operands[i + 2].Trim();
                bool negated = text.StartsWith("!", StringComparison.Ordinal);
                string baseName = text.TrimStart('!');
                if (baseName.Equals("PT", StringComparison.OrdinalIgnoreCase))
                    constants[i] = !negated;
                else
                    predicates[i] = PredicateOperand(text);
            }

            bool allConstant = true;
            int constantIndex = 0;
            for (int i = 0; i < 3; i++)
            {
                if (constants[i] is not bool value)
                {
                    allConstant = false;
                    continue;
                }
                if (value)
                    constantIndex |= 1 << (2 - i);
            }
            if (allConstant)
            {
                bool result = ((lut >> constantIndex) & 1) != 0;
                _lines.Add($"{prefix}mov.pred {Pred(destination)}, " +
                    $"{(result ? 1 : 0)};");
                return;
            }

            string? accumulated = null;
            for (int index = 0; index < 8; index++)
            {
                if ((lut & (1u << index)) == 0)
                    continue;
                var terms = new List<string>();
                bool possible = true;
                for (int i = 0; i < 3; i++)
                {
                    bool bit = (index & (1 << (2 - i))) != 0;
                    if (constants[i] is bool value)
                    {
                        if (value != bit)
                        {
                            possible = false;
                            break;
                        }
                        continue;
                    }
                    string predicate = predicates[i]!;
                    if (!bit)
                    {
                        string inverted = $"%pt{_nextTempPred++}";
                        _lines.Add($"    not.pred {inverted}, {predicate};");
                        predicate = inverted;
                    }
                    terms.Add(predicate);
                }
                if (!possible)
                    continue;
                if (terms.Count == 0)
                {
                    _lines.Add($"{prefix}mov.pred {Pred(destination)}, 1;");
                    return;
                }
                string term = terms[0];
                for (int i = 1; i < terms.Count; i++)
                {
                    string combined = $"%pt{_nextTempPred++}";
                    _lines.Add($"    and.pred {combined}, {term}, {terms[i]};");
                    term = combined;
                }
                if (accumulated == null)
                {
                    accumulated = term;
                }
                else
                {
                    string combined = $"%pt{_nextTempPred++}";
                    _lines.Add($"    or.pred {combined}, {accumulated}, {term};");
                    accumulated = combined;
                }
            }
            if (accumulated == null)
                _lines.Add($"{prefix}mov.pred {Pred(destination)}, 0;");
            else
                _lines.Add($"{prefix}mov.pred {Pred(destination)}, {accumulated};");
        }

        private static string PredicateOperation(string name)
        {
            return name switch
            {
                "AND" => "and.b32",
                "OR" => "or.b32",
                "XOR" => "xor.b32",
                _ => throw new NotSupportedException(
                    $"SASS predicate operation '{name}' is not supported")
            };
        }

        private string PredicateValue(string operand, string prefix)
        {
            string text = operand.Trim();
            bool negated = text.StartsWith("!", StringComparison.Ordinal);
            string baseName = text.TrimStart('!');
            if (baseName.Equals("PT", StringComparison.OrdinalIgnoreCase))
                return negated ? "0" : "1";
            string value = $"%rt{_nextTemp++}";
            _lines.Add($"{prefix}selp.b32 {value}, " +
                $"{(negated ? "0, 1" : "1, 0")}, {Pred(baseName)};");
            return value;
        }

        private string PredicateOperand(string operand)
        {
            string text = operand.Trim();
            bool negated = text.StartsWith("!", StringComparison.Ordinal);
            string baseName = text.TrimStart('!');
            if (baseName.Equals("PT", StringComparison.OrdinalIgnoreCase))
            {
                string constant = $"%pt{_nextTempPred++}";
                _lines.Add($"    mov.pred {constant}, {(negated ? 0 : 1)};");
                return constant;
            }
            string predicate = Pred(text);
            if (!negated)
                return predicate;
            string inverted = $"%pt{_nextTempPred++}";
            _lines.Add($"    not.pred {inverted}, {predicate};");
            return inverted;
        }

        private void EmitDsetp(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 5)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs five operands");
            string suffix = instruction.Mnemonic.ToUpperInvariant();
            bool minimum = suffix.Contains(".MIN", StringComparison.Ordinal);
            bool maximum = suffix.Contains(".MAX", StringComparison.Ordinal);
            if (minimum || maximum)
            {
                if (instruction.Operands.Count < 5 ||
                    !instruction.Operands[4].Trim().Equals(
                        "PT", StringComparison.OrdinalIgnoreCase))
                    throw new NotSupportedException(
                        $"SASS instruction '{instruction.Mnemonic}' is not supported");
                string low = Float64Operand(instruction.Operands[2]);
                string high = Float64Operand(instruction.Operands[3]);
                string less = $"%pt{_nextTempPred++}";
                string lowNan = $"%pt{_nextTempPred++}";
                string highNan = $"%pt{_nextTempPred++}";
                string lowFinite = $"%pt{_nextTempPred++}";
                string highNanLowFinite = $"%pt{_nextTempPred++}";
                string bothNan = $"%pt{_nextTempPred++}";
                _lines.Add($"    setp.{(minimum ? "lt" : "gt")}.f64 " +
                    $"{less}, {low}, {high};");
                _lines.Add($"    setp.nan.f64 {lowNan}, {low}, {low};");
                _lines.Add($"    setp.nan.f64 {highNan}, {high}, {high};");
                _lines.Add($"    not.pred {lowFinite}, {lowNan};");
                _lines.Add($"    and.pred {highNanLowFinite}, " +
                    $"{highNan}, {lowFinite};");
                _lines.Add($"    and.pred {bothNan}, {lowNan}, {highNan};");
                string first = Pred(instruction.Operands[0]);
                string second = Pred(instruction.Operands[1]);
                _lines.Add($"    or.pred {first}, {less}, " +
                    $"{highNanLowFinite};");
                _lines.Add($"    mov.pred {second}, {bothNan};");
                return;
            }
            string comparison = suffix.Contains(".GEU") ? "geu"
                : suffix.Contains(".GTU") ? "gtu"
                : suffix.Contains(".LEU") ? "leu"
                : suffix.Contains(".LTU") ? "ltu"
                : suffix.Contains(".EQU") ? "equ"
                : suffix.Contains(".NEU") ? "neu"
                : suffix.Contains(".GE") ? "ge"
                : suffix.Contains(".GT") ? "gt"
                : suffix.Contains(".LE") ? "le"
                : suffix.Contains(".LT") ? "lt"
                : suffix.Contains(".EQ") ? "eq"
                : suffix.Contains(".NE") ? "ne"
                : suffix.Contains(".NUM") ? "num"
                : suffix.Contains(".NAN") ? "nan"
                : "eq";
            string boolOperation = suffix.Contains(".AND") ? "and"
                : suffix.Contains(".OR") ? "or"
                : suffix.Contains(".XOR") ? "xor"
                : "";
            string a = Float64Operand(instruction.Operands[2]);
            string b = Float64Operand(instruction.Operands[3]);
            string sourcePredicate = instruction.Operands[4].Trim();
            if (sourcePredicate.Equals("PT", StringComparison.OrdinalIgnoreCase))
            {
                _lines.Add($"{prefix}setp.{comparison}.f64 " +
                    $"{Pred(instruction.Operands[0])}, {a}, {b};");
                return;
            }
            string predicate = PredicateOperand(sourcePredicate);
            _lines.Add($"{prefix}setp.{comparison}.{boolOperation}.f64 " +
                $"{Pred(instruction.Operands[0])}, {a}, {b}, {predicate};");
        }

        private string Float64Operand(string operand)
        {
            string text = Regex.Replace(operand.Trim(), @"\.reuse\b", "",
                RegexOptions.IgnoreCase);
            if (text.Equals("RZ", StringComparison.OrdinalIgnoreCase))
                return "0d0000000000000000";
            // Handles plain/negated/absolute register pairs, constant
            // banks, and f64 immediates (e.g. |R14| and +INF, neither of
            // which is valid PTX text).
            return PackRegisterPair(text);
        }

        private void EmitFsetp(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 5)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs five operands");
            string suffix = instruction.Mnemonic.ToUpperInvariant();
            string comparison = suffix.Contains(".GEU") ? "geu"
                : suffix.Contains(".GTU") ? "gtu"
                : suffix.Contains(".LEU") ? "leu"
                : suffix.Contains(".LTU") ? "ltu"
                : suffix.Contains(".EQU") ? "equ"
                : suffix.Contains(".NEU") ? "neu"
                : suffix.Contains(".GE") ? "ge"
                : suffix.Contains(".GT") ? "gt"
                : suffix.Contains(".LE") ? "le"
                : suffix.Contains(".LT") ? "lt"
                : suffix.Contains(".EQ") ? "eq"
                : suffix.Contains(".NE") ? "ne"
                : suffix.Contains(".NUM") ? "num"
                : suffix.Contains(".NAN") ? "nan"
                : "eq";
            string boolOperation = suffix.Contains(".AND") ? "and"
                : suffix.Contains(".OR") ? "or"
                : suffix.Contains(".XOR") ? "xor"
                : "";
            string a = FloatOperand(instruction.Operands[2]);
            string b = FloatOperand(instruction.Operands[3]);
            string sourcePredicate = instruction.Operands[4].Trim();
            if (sourcePredicate.Equals("PT", StringComparison.OrdinalIgnoreCase))
            {
                _lines.Add($"{prefix}setp.{comparison}.f32 " +
                    $"{Pred(instruction.Operands[0])}, {a}, {b};");
                return;
            }
            string predicate = PredicateOperand(sourcePredicate);
            _lines.Add($"{prefix}setp.{comparison}.{boolOperation}.f32 " +
                $"{Pred(instruction.Operands[0])}, {a}, {b}, {predicate};");
        }

        private void EmitFset(SassInstruction instruction, string prefix)
        {
            if (instruction.Operands.Count < 4)
                throw new NotSupportedException(
                    $"SASS instruction '{instruction.Mnemonic}' needs four operands");
            string suffix = instruction.Mnemonic.ToUpperInvariant();
            string comparison = suffix.Contains(".GEU") ? "geu"
                : suffix.Contains(".GTU") ? "gtu"
                : suffix.Contains(".LEU") ? "leu"
                : suffix.Contains(".LTU") ? "ltu"
                : suffix.Contains(".EQU") ? "equ"
                : suffix.Contains(".NEU") ? "neu"
                : suffix.Contains(".GE") ? "ge"
                : suffix.Contains(".GT") ? "gt"
                : suffix.Contains(".LE") ? "le"
                : suffix.Contains(".LT") ? "lt"
                : suffix.Contains(".EQ") ? "eq"
                : suffix.Contains(".NE") ? "ne"
                : suffix.Contains(".NUM") ? "num"
                : suffix.Contains(".NAN") ? "nan"
                : "eq";
            string boolOperation = suffix.Contains(".AND") ? "and"
                : suffix.Contains(".OR") ? "or"
                : suffix.Contains(".XOR") ? "xor"
                : "";
            string a = FloatOperand(instruction.Operands[1]);
            string b = FloatOperand(instruction.Operands[2]);
            string sourcePredicate = instruction.Operands[3].Trim();
            string resultPredicate = $"%pt{_nextTempPred++}";
            if (sourcePredicate.Equals("PT", StringComparison.OrdinalIgnoreCase))
            {
                _lines.Add($"setp.{comparison}.f32 " +
                    $"{resultPredicate}, {a}, {b};");
            }
            else
            {
                string predicate = PredicateOperand(sourcePredicate);
                _lines.Add($"setp.{comparison}.{boolOperation}.f32 " +
                    $"{resultPredicate}, {a}, {b}, {predicate};");
            }
            _lines.Add($"{prefix}selp.b32 {Reg(instruction.Operands[0])}, " +
                $"1, 0, {resultPredicate};");
        }

        private string FloatOperand(string operand)
        {
            string text = operand.Trim();
            // The .reuse hint can be attached to a negated source, as in
            // `FFMA R8, R17, -R0.reuse, c`.  Strip it first, otherwise the
            // negation check below misses and the coefficient is negated
            // as an integer, corrupting log/exp's polynomial.
            text = Regex.Replace(text, @"\.reuse\b", "",
                RegexOptions.IgnoreCase);
            if (text.Equals("RZ", StringComparison.OrdinalIgnoreCase))
                return "0f00000000";
            if (text.Equals("-RZ", StringComparison.OrdinalIgnoreCase))
                return "0f80000000";
            if (text.Equals("+INF", StringComparison.OrdinalIgnoreCase) ||
                text.Equals("INF", StringComparison.OrdinalIgnoreCase))
                return "0f7f800000";
            if (text.Equals("-INF", StringComparison.OrdinalIgnoreCase))
                return "0fff800000";
            if (text.Equals("+NAN", StringComparison.OrdinalIgnoreCase) ||
                text.Equals("NAN", StringComparison.OrdinalIgnoreCase) ||
                text.Equals("+QNAN", StringComparison.OrdinalIgnoreCase) ||
                text.Equals("QNAN", StringComparison.OrdinalIgnoreCase))
                return "0f7fc00000";
            if (text.Equals("-NAN", StringComparison.OrdinalIgnoreCase) ||
                text.Equals("-QNAN", StringComparison.OrdinalIgnoreCase))
                return "0fffc00000";
            if (Regex.IsMatch(text, @"^-R\d+$",
                    RegexOptions.IgnoreCase))
            {
                string negatedValue = Operand(text[1..]);
                string negated = $"%ft{_nextTempFloat++}";
                _lines.Add($"    neg.f32 {negated}, {negatedValue};");
                return negated;
            }
            // General float negation, including `-|R3|` and `-c[...]`.
            // Handling the leading '-' before the |...|/constant logic
            // keeps the sign in the float domain (the generic integer
            // operand would two's-complement the bit pattern).
            if (text.StartsWith("-", StringComparison.Ordinal) &&
                text.Length > 1 && text[1] != '0' &&
                !char.IsDigit(text[1]))
            {
                string positive = FloatOperand(text[1..]);
                string negated = $"%ft{_nextTempFloat++}";
                _lines.Add($"    neg.f32 {negated}, {positive};");
                return negated;
            }
            Match floatConstant = Regex.Match(text,
                @"^c\[0x([1-9A-Fa-f])\]\[0x([0-9A-Fa-f]+)\]$",
                RegexOptions.IgnoreCase);
            if (floatConstant.Success)
            {
                int bank = int.Parse(floatConstant.Groups[1].Value,
                    NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                int offset = int.Parse(floatConstant.Groups[2].Value,
                    NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                uint? bits = ConstantWord(bank, offset);
                if (!bits.HasValue)
                    throw new NotSupportedException(
                        $"constant bank {bank} offset 0x{offset:x} " +
                        "is not available as a float");
                return $"0f{bits.Value:x8}";
            }
            bool absolute = text.Contains("|", StringComparison.Ordinal);
            string cleaned = text.Replace("|", "");
            string value = Operand(cleaned);
            if (Regex.IsMatch(value, @"^-?\d+$"))
                value += ".0";
            if (!absolute)
                return value;
            string temporary = $"%ft{_nextTempFloat++}";
            _lines.Add($"    abs.f32 {temporary}, {value};");
            return temporary;
        }

        private void EmitIsetp(SassInstruction instruction, string prefix)
        {
            string suffix = instruction.Mnemonic.ToUpperInvariant();
            string comparison = suffix.Contains(".GT") ? "gt"
                : suffix.Contains(".GE") ? "ge"
                : suffix.Contains(".LT") ? "lt"
                : suffix.Contains(".LE") ? "le"
                : suffix.Contains(".EQ") ? "eq"
                : suffix.Contains(".NE") ? "ne"
                : "eq";
            string type = suffix.Contains(".U32") ? "u32" : "s32";
            if (suffix.Contains(".EX") && instruction.Operands.Count >= 6)
            {
                // 64-bit compare: the high-word compare is combined with
                // the low-word predicate.  EQ/NE are a plain AND/OR of
                // the two halves; ordered comparisons are
                //     hi <op> hb || (hi == hb && lo <op> lb)
                // where the low half is already in lowPredicate.
                // Treating NE like an ordered compare (e.g. `0 < hi`)
                // broke pointer null checks: a device address with the
                // sign bit set looked null.
                string destination = Pred(instruction.Operands[0]);
                string high = Operand(instruction.Operands[2]);
                string highBound = Operand(instruction.Operands[3]);
                string exChainText = instruction.Operands[4].Trim();
                string lowPredicate = Pred(instruction.Operands[5]);
                string exComparison = $"%pt{_nextTempPred++}";
                if (comparison is "eq")
                {
                    string equal = $"%pt{_nextTempPred++}";
                    _lines.Add($"    setp.eq.{type} {equal}, {high}, {highBound};");
                    _lines.Add($"{prefix}and.pred {exComparison}, {equal}, {lowPredicate};");
                }
                else if (comparison is "ne")
                {
                    string notEqual = $"%pt{_nextTempPred++}";
                    _lines.Add($"    setp.ne.{type} {notEqual}, {high}, {highBound};");
                    _lines.Add($"{prefix}or.pred {exComparison}, {notEqual}, {lowPredicate};");
                }
                else
                {
                    string highCmp = comparison is "ge" or "gt" ? "gt" : "lt";
                    string ordered = $"%pt{_nextTempPred++}";
                    string sameHigh = $"%pt{_nextTempPred++}";
                    string both = $"%pt{_nextTempPred++}";
                    _lines.Add($"    setp.{highCmp}.{type} {ordered}, {high}, {highBound};");
                    _lines.Add($"    setp.eq.{type} {sameHigh}, {high}, {highBound};");
                    _lines.Add($"    and.pred {both}, {sameHigh}, {lowPredicate};");
                    _lines.Add($"{prefix}or.pred {exComparison}, {ordered}, {both};");
                }
                // The predicate in operand 4 is joined with the 64-bit
                // comparison through the mnemonic's boolean op.  nvcc folds
                // an independent bounds predicate into a merge-path compare
                // with `ISETP.LT.OR.EX`; dropping it selected stale entries.
                string? exChainOp = suffix.Contains(".OR", StringComparison.Ordinal)
                    ? "or"
                    : suffix.Contains(".XOR", StringComparison.Ordinal)
                        ? "xor"
                        : suffix.Contains(".AND", StringComparison.Ordinal)
                            ? "and" : null;
                bool exChainNegated = exChainText.StartsWith("!", StringComparison.Ordinal);
                if (exChainOp == null ||
                    (exChainText.Equals("PT", StringComparison.OrdinalIgnoreCase) &&
                     !exChainNegated))
                {
                    _lines.Add($"{prefix}mov.pred {destination}, {exComparison};");
                    return;
                }
                string exChained = Pred(exChainText);
                if (exChainNegated)
                {
                    string inverted = $"%pt{_nextTempPred++}";
                    _lines.Add($"{prefix}not.pred {inverted}, {exChained};");
                    exChained = inverted;
                }
                _lines.Add($"{prefix}{exChainOp}.pred {destination}, " +
                    $"{exComparison}, {exChained};");
                return;
            }
            // ISETP.cmp.bop D, A, B, C, E: the last operand is joined
            // with the comparison result through the mnemonic's boolean
            // op (SASS predicate chaining).  Dropping it inverted the
            // sign fix-up in the integer remainder sequence.
            string? chainOp = suffix.Contains(".OR", StringComparison.Ordinal)
                ? "or"
                : suffix.Contains(".XOR", StringComparison.Ordinal)
                    ? "xor"
                    : suffix.Contains(".AND", StringComparison.Ordinal)
                        ? "and" : null;
            string chainText = instruction.Operands.Count >= 5
                ? instruction.Operands[4].Trim() : "PT";
            bool chainNegated = chainText.StartsWith("!", StringComparison.Ordinal);
            if (chainOp == null ||
                (chainText.Equals("PT", StringComparison.OrdinalIgnoreCase) &&
                 !chainNegated))
            {
                _lines.Add($"{prefix}setp.{comparison}.{type} {Pred(instruction.Operands[0])}, " +
                    $"{Operand(instruction.Operands[2])}, {Operand(instruction.Operands[3])};");
                return;
            }
            string comparisonResult = $"%pt{_nextTempPred++}";
            _lines.Add($"{prefix}setp.{comparison}.{type} {comparisonResult}, " +
                $"{Operand(instruction.Operands[2])}, {Operand(instruction.Operands[3])};");
            string chained = Pred(chainText);
            if (chainNegated)
            {
                string inverted = $"%pt{_nextTempPred++}";
                _lines.Add($"{prefix}not.pred {inverted}, {chained};");
                chained = inverted;
            }
            _lines.Add($"{prefix}{chainOp}.pred {Pred(instruction.Operands[0])}, " +
                $"{comparisonResult}, {chained};");
        }

        private string Reg(string operand)
        {
            string trimmed = operand.Trim();
            if (trimmed.Equals("RZ", StringComparison.OrdinalIgnoreCase))
                return "0";
            Match match = Regex.Match(trimmed, @"^R(\d+)$",
                RegexOptions.IgnoreCase);
            if (!match.Success)
                throw new NotSupportedException($"expected register, got '{operand}'");
            string key = "R" + match.Groups[1].Value;
            if (_regs.TryGetValue(key, out string? mapped))
                return mapped;
            return "%r" + match.Groups[1].Value;
        }

        private string Pred(string operand)
        {
            string trimmed = operand.Trim().TrimStart('!');
            if (trimmed.Equals("PT", StringComparison.OrdinalIgnoreCase))
                return "1";
            Match uniform = Regex.Match(trimmed, @"^UP(\d+)$",
                RegexOptions.IgnoreCase);
            if (uniform.Success)
                return "%up" + uniform.Groups[1].Value;
            Match match = Regex.Match(trimmed, @"^P(\d+)$",
                RegexOptions.IgnoreCase);
            if (!match.Success)
                throw new NotSupportedException($"expected predicate, got '{operand}'");
            return "%p" + match.Groups[1].Value;
        }

        private static string NoReuse(string operand)
        {
            return Regex.Replace(operand.Trim(), @"\.reuse\b", "",
                RegexOptions.IgnoreCase);
        }

        private string Operand(string operand)
        {
            string trimmed = operand.Trim();
            trimmed = Regex.Replace(trimmed, @"\.reuse\b", "",
                RegexOptions.IgnoreCase);
            if (trimmed.Contains('|', StringComparison.Ordinal))
                return FloatOperand(trimmed);
            if (Regex.IsMatch(trimmed, @"^-?UR\d+$",
                    RegexOptions.IgnoreCase) ||
                trimmed.Equals("URZ", StringComparison.OrdinalIgnoreCase))
                return UniformOperand(trimmed);
            // A wide result is cached as a %rdt pair.  Reading its high
            // half as a plain register must first materialise the pair,
            // otherwise the read sees a stale value.  Only do this when
            // the high register is actually used, so we never write a
            // register outside the declared bank.
            Match halfRead = Regex.Match(trimmed, @"^R(\d+)$",
                RegexOptions.IgnoreCase);
            if (halfRead.Success)
            {
                int halfNumber = int.Parse(halfRead.Groups[1].Value,
                    CultureInfo.InvariantCulture);
                string? pair = null;
                string pairLow = "";
                if (_pairRegs.TryGetValue(trimmed, out string? ownPair))
                {
                    pair = ownPair;
                    pairLow = trimmed;
                }
                else if (halfNumber > 0)
                {
                    string lowKey = "R" + (halfNumber - 1).ToString(
                        CultureInfo.InvariantCulture);
                    if (_pairRegs.TryGetValue(lowKey, out string? lowPair))
                    {
                        pair = lowPair;
                        pairLow = lowKey;
                    }
                }
                if (pair != null && _materializedPairs.Add(pair))
                {
                    string highKey = NextRegister(pairLow);
                    _lines.Add($"    mov.b64 {{{Reg(pairLow)}, " +
                        $"{Reg(highKey)}}}, {pair};");
                }
            }
            if (trimmed.Equals("-RZ", StringComparison.OrdinalIgnoreCase))
                return "0";
            // Bitwise complement: PTX has no `~` operand, so
            // materialise the inner operand and emit not.b32.  This
            // covers registers (`~R12`) and constant-bank values
            // (`~c[0x0][0x19c]`).
            if (trimmed.StartsWith("~", StringComparison.Ordinal) &&
                trimmed.Length > 1)
            {
                string inner = Operand(trimmed[1..]);
                string inverted = $"%rt{_nextTemp++}";
                _lines.Add($"    not.b32 {inverted}, {inner};");
                return inverted;
            }
            if (Regex.IsMatch(trimmed, @"^-R\d+$",
                    RegexOptions.IgnoreCase))
            {
                string negated = $"%rt{_nextTemp++}";
                _lines.Add($"    neg.s32 {negated}, " +
                    $"{Reg(trimmed[1..])};");
                return negated;
            }
            // Negated constant-bank operand (e.g. `-c[0x0][0x1a8]`):
            // PTX has no signed memory operand, so materialise and negate.
            if (trimmed.StartsWith("-", StringComparison.Ordinal) &&
                trimmed.Length > 1 && trimmed[1] != '0' &&
                !char.IsDigit(trimmed[1]))
            {
                string inner = Operand(trimmed[1..]);
                string negated = $"%rt{_nextTemp++}";
                _lines.Add($"    neg.s32 {negated}, {inner};");
                return negated;
            }
            Match half = Regex.Match(trimmed, @"^R(\d+)\.H([01])$",
                RegexOptions.IgnoreCase);
            if (half.Success)
            {
                string source = Reg("R" + half.Groups[1].Value);
                string temp = $"%rt{_nextTemp++}";
                if (half.Groups[2].Value == "1")
                    _lines.Add($"    shr.u32 {temp}, {source}, 16;");
                else
                    _lines.Add($"    and.b32 {temp}, {source}, 0xffff;");
                return temp;
            }
            // Replicated half source (HADD2/HFMA2): R4.H0_H0 duplicates
            // the low half into both lanes, R4.H1_H1 the high half.
            Match replicate = Regex.Match(trimmed,
                @"^R(\d+)\.H([01])_H\2$", RegexOptions.IgnoreCase);
            if (replicate.Success)
            {
                string source = Reg("R" + replicate.Groups[1].Value);
                string lane = $"%rt{_nextTemp++}";
                if (replicate.Groups[2].Value == "1")
                    _lines.Add($"    shr.u32 {lane}, {source}, 16;");
                else
                    _lines.Add($"    and.b32 {lane}, {source}, 0xffff;");
                string packed = $"%rt{_nextTemp++}";
                _lines.Add($"    mul.lo.u32 {packed}, {lane}, 0x10001;");
                return packed;
            }
            if (trimmed.StartsWith("c[", StringComparison.OrdinalIgnoreCase))
            {
                Match constant = Regex.Match(trimmed,
                    @"^c\[0x0\]\[0x([0-9A-Fa-f]+)\]$",
                    RegexOptions.IgnoreCase);
                if (constant.Success)
                {
                    int offset = int.Parse(constant.Groups[1].Value,
                        NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                    if (!_paramsByOffset.TryGetValue(offset, out PtxParameter? parameter) &&
                        _paramsByOffset.TryGetValue(offset - 4, out PtxParameter? highParameter) &&
                        highParameter.Type == "u64" &&
                        offset == highParameter.Offset + 4)
                    {
                        string highWord = $"%rt{_nextTemp++}";
                        int hbase = highParameter.Part * 8;
                        _lines.Add($"    ld.param.u32 {highWord}, " +
                            $"[{highParameter.Name}+{hbase + 4}];");
                        return highWord;
                    }
                    if (parameter != null)
                    {
                        int index = ParameterIndex(parameter);
                        if (parameter.Type == "u64")
                        {
                            int pbase = parameter.Part * 8;
                            if (offset == parameter.Offset)
                            {
                                string low = $"%rt{_nextTemp++}";
                                _lines.Add(
                                    $"    ld.param.u32 {low}, " +
                                    $"[{parameter.Name}+{pbase}];");
                                return low;
                            }
                            if (offset == parameter.Offset + 4)
                            {
                                string high = $"%rt{_nextTemp++}";
                                _lines.Add(
                                    $"    ld.param.u32 {high}, " +
                                    $"[{parameter.Name}+{pbase + 4}];");
                                return high;
                            }
                            throw new NotSupportedException(
                                $"constant bank offset 0x{offset:x} is not a parameter word");
                        }
                        return $"%rpp{index}";
                    }
                    if (_constantResolver != null)
                    {
                        uint? value = _constantResolver(offset);
                        if (value.HasValue)
                            return value.Value.ToString(
                                CultureInfo.InvariantCulture);
                        throw new NotSupportedException(
                            $"constant bank offset 0x{offset:x} is not available");
                    }
                    // Constant bank 0 starts with the launch geometry,
                    // not kernel parameters (which begin at 0x160 in the
                    // decoded cubins).
                    switch (offset)
                    {
                    case 0x0: return MaterializeSpecial("%ntid.x");
                    case 0x4: return MaterializeSpecial("%ntid.y");
                    case 0x8: return MaterializeSpecial("%ntid.z");
                    case 0xc: return MaterializeSpecial("%nctaid.x");
                    case 0x10: return MaterializeSpecial("%nctaid.y");
                    case 0x14: return MaterializeSpecial("%nctaid.z");
                    }
                }
                // Compiler literals live in per-kernel constant banks
                // (bank 2 for math constants, bank 4 for launch-size
                // descriptors).  Materialise them as immediates; the
                // decoded SASS alone does not carry the values.
                Match otherBank = Regex.Match(trimmed,
                    @"^c\[0x([1-9A-Fa-f])\]\[0x([0-9A-Fa-f]+)\]$",
                    RegexOptions.IgnoreCase);
                if (otherBank.Success)
                {
                    int bank = int.Parse(otherBank.Groups[1].Value,
                        NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                    int bankOffset = int.Parse(otherBank.Groups[2].Value,
                        NumberStyles.HexNumber, CultureInfo.InvariantCulture);
                    uint? value = ConstantWord(bank, bankOffset);
                    if (value.HasValue)
                        return value.Value.ToString(
                            CultureInfo.InvariantCulture);
                    throw new NotSupportedException(
                        $"constant bank {bank} offset 0x{bankOffset:x} " +
                        "is not available");
                }
                return "0"; // no constant-bank data was supplied
            }
            if (trimmed.Equals("RZ", StringComparison.OrdinalIgnoreCase))
                return "0";
            // A predicate used in a scalar value position (e.g. an IMAD.X
            // carry addend that the pair peephole did not consume) is a
            // 0/1 integer.  Emitting the raw 'P5' name is not valid PTX.
            Match predicate = Regex.Match(trimmed, @"^(!?)(?:P(\d+)|PT)$",
                RegexOptions.IgnoreCase);
            if (predicate.Success)
            {
                if (trimmed.EndsWith("PT", StringComparison.OrdinalIgnoreCase))
                    return predicate.Groups[1].Value.Length > 0 ? "0" : "1";
                string value = $"%rt{_nextTemp++}";
                string pred = Pred("P" + predicate.Groups[2].Value);
                if (predicate.Groups[1].Value.Length > 0)
                    _lines.Add($"    selp.u32 {value}, 0, 1, {pred};");
                else
                    _lines.Add($"    selp.u32 {value}, 1, 0, {pred};");
                return value;
            }
            if (trimmed.StartsWith("R", StringComparison.OrdinalIgnoreCase) &&
                Regex.IsMatch(trimmed, @"^R\d+$", RegexOptions.IgnoreCase))
                return Reg(trimmed);
            if (trimmed.StartsWith("%", StringComparison.Ordinal))
            {
                // Special registers are only valid as mov sources in
                // ACuda's parser; copy them into a plain register.
                return MaterializeSpecial(trimmed);
            }
            return Immediate(trimmed);
        }

        private uint? ConstantWord(int bank, int offset)
        {
            Func<int, uint?>? resolver = bank switch
            {
                2 => _constant2Resolver,
                4 => _constant4Resolver,
                _ => null,
            };
            if (resolver == null || offset < 0)
                return null;
            if (bank == 2 && offset + 4 > _constant2Bytes)
                return null;
            if (bank == 4 && offset + 4 > _constant4Bytes)
                return null;
            return resolver(offset);
        }

        private string MaterializeSpecial(string register)
        {
            string special = $"%rt{_nextTemp++}";
            _lines.Add($"    mov.u32 {special}, {register};");
            return special;
        }

        // SASS prints an add-negative displacement as `+-0x20`; PTX needs
        // the plain signed form (`-0x20`).
        private static string NormalizeOffset(string raw)
        {
            if (raw.StartsWith("+-", StringComparison.Ordinal))
                return raw[1..];
            if (raw.StartsWith("+", StringComparison.Ordinal))
                return raw[1..];
            return raw;
        }

        // Formats a register base+offset operand.  Register-derived
        // addresses keep an explicit `+0` (matches the established output).
        private static string RegAddr(string baseExpr, string offset)
        {
            return offset.StartsWith("-", StringComparison.Ordinal)
                ? $"[{baseExpr}{offset}]"
                : $"[{baseExpr}+{offset}]";
        }

        // Formats an absolute `__shared` base+offset operand; a zero
        // displacement is omitted.
        private static string SharedBaseAddr(string baseExpr, string offset)
        {
            if (string.IsNullOrEmpty(offset) || offset == "0")
                return $"[{baseExpr}]";
            return offset.StartsWith("-", StringComparison.Ordinal)
                ? $"[{baseExpr}{offset}]"
                : $"[{baseExpr}+{offset}]";
        }

        private string Address(string operand)
        {
            Match wide = Regex.Match(operand,
                @"^\[(R\d+)\.64([+-]-?(?:0x[0-9A-Fa-f]+|\d+))?\]$",
                RegexOptions.IgnoreCase);
            if (wide.Success)
            {
                string wideBase = wide.Groups[1].Value;
                string wideOffset = wide.Groups[2].Success
                    ? NormalizeOffset(wide.Groups[2].Value)
                    : "0";
                if (_pairRegs.TryGetValue(wideBase, out string? pair))
                    return RegAddr(pair, wideOffset);
                string low = Reg(wideBase);
                string high = Reg(NextRegister(wideBase));
                string built = $"%rdt{_nextTemp++}";
                _lines.Add($"    mov.b64 {built}, {{{low}, {high}}};");
                return RegAddr(built, wideOffset);
            }
            Match match = Regex.Match(operand,
                @"^\[(R\d+)([+-]-?(?:0x[0-9A-Fa-f]+|\d+))?\]$",
                RegexOptions.IgnoreCase);
            if (!match.Success)
                throw new NotSupportedException($"unsupported address '{operand}'");
            string baseRegister = match.Groups[1].Value;
            string register = _pointerRegs.TryGetValue(
                baseRegister, out string? pointer)
                ? pointer
                : Reg(baseRegister);
            string offset = match.Groups[2].Success
                ? NormalizeOffset(match.Groups[2].Value)
                : "0";
            if (register == "__callee_frame")
                return RegAddr("__callee_frame", offset);
            if (register.StartsWith("%rd", StringComparison.Ordinal) ||
                register.StartsWith("%rpptr", StringComparison.Ordinal))
                return RegAddr(register, offset);
            string temp = $"%rdt{_nextTemp++}";
            _lines.Add($"    cvt.u64.u32 {temp}, {register};");
            return RegAddr(temp, offset);
        }

        private static bool IsCallMnemonic(string mnemonic) =>
            mnemonic.StartsWith("CAL", StringComparison.OrdinalIgnoreCase) ||
            mnemonic.StartsWith("JCAL", StringComparison.OrdinalIgnoreCase);

        private static bool IsCallOp(string op) => op is "CAL" or "JCAL" or "CALL";

        private static bool IsTailReturn(List<SassNode> body, int callIndex)
        {
            for (int i = callIndex + 1; i < body.Count; i++)
            {
                if (body[i] is SassLabel)
                    continue;
                return body[i] is SassInstruction instruction &&
                    instruction.Mnemonic.StartsWith(
                        "RET", StringComparison.OrdinalIgnoreCase) &&
                    string.IsNullOrWhiteSpace(instruction.Guard);
            }
            return false;
        }

        private bool TryEmitIndirectCallDispatch(
            List<SassNode> nodes, ref int index)
        {
            if (nodes[index] is not SassInstruction call ||
                !IsCallMnemonic(call.Mnemonic) ||
                !string.IsNullOrWhiteSpace(call.Guard) ||
                _indirectDispatchDepth > 0)
                return false;
            string? raw = call.BranchTargets.FirstOrDefault()
                ?? call.Operands.LastOrDefault();
            if (raw == null)
                return false;
            string register = raw.Trim().Trim('`', '(', ')', ';');
            if (!Regex.IsMatch(register, @"^R\d+$",
                    RegexOptions.IgnoreCase))
                return false;
            try
            {
                _ = CallTarget(nodes, index, call);
                return false; // a constant-indirect target is handled elsewhere
            }
            catch (NotSupportedException)
            {
                // Fall through to the finite target dispatch.
            }
            var candidates = new List<(
                string Label, int Address, List<SassNode> Nodes)>();
            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (SassSection section in new[] { _section }
                         .Concat(_linkedSections))
            {
                List<SassNode> source = section.Nodes.ToList();
                foreach (SassNode node in source)
                {
                    if (node is not SassLabel label ||
                        label.Name.StartsWith(".L_", StringComparison.Ordinal) ||
                        label.Name.Equals(_entry, StringComparison.Ordinal) ||
                        label.Address == 0 ||
                        !seen.Add(label.Name))
                        continue;
                    candidates.Add((
                        label.Name, label.Address, source));
                }
            }
            if (candidates.Count == 0)
                return false;
            int serial = _inlineSerial++;
            string done = $"L_indirect_done_{serial}";
            string registerName = Reg(register);
            for (int i = 0; i < candidates.Count; i++)
            {
                _lines.Add(
                    $"    setp.eq.u32 %pt0, {registerName}, " +
                    $"0x{candidates[i].Address:x};");
                _lines.Add($"    @%pt0 bra L_indirect_{serial}_{i};");
            }
            _lines.Add("    brkpt;");
            _lines.Add($"    bra {done};");
            _indirectDispatchDepth++;
            try
            {
                for (int i = 0; i < candidates.Count; i++)
                {
                    _lines.Add($"L_indirect_{serial}_{i}:");
                    var callNodes = new List<SassNode>(
                        candidates[i].Nodes.Count + 1);
                    var synthetic = new SassInstruction
                    {
                        Mnemonic = "CAL",
                    };
                    synthetic.Operands.Add(candidates[i].Label);
                    callNodes.Add(synthetic);
                    callNodes.AddRange(candidates[i].Nodes);
                    int nestedIndex = 0;
                    var savedSkip = new HashSet<int>(_skipNodes);
                    _skipNodes.Clear();
                    bool ok;
                    try
                    {
                        ok = TryEmitLeafCallInline(
                            callNodes, ref nestedIndex);
                    }
                    finally
                    {
                        _skipNodes.Clear();
                        foreach (int saved in savedSkip)
                            _skipNodes.Add(saved);
                    }
                    if (!ok)
                        throw new NotSupportedException(
                            "indirect call candidate could not be inlined");
                    // The main-section callee body must not be emitted again.
                    if (nodes.Any(node => node is SassLabel label &&
                            label.Name.Equals(candidates[i].Label,
                                StringComparison.Ordinal)))
                    {
                        int labelIndex = nodes.FindIndex(node =>
                            node is SassLabel label &&
                            label.Name.Equals(candidates[i].Label,
                                StringComparison.Ordinal));
                        if (labelIndex >= 0)
                        {
                            int bodyEnd = nodes.Count;
                            for (int j = labelIndex + 1; j < nodes.Count; j++)
                            {
                                if (nodes[j] is SassLabel boundary &&
                                    !boundary.Name.StartsWith(
                                        ".L_", StringComparison.Ordinal))
                                {
                                    bodyEnd = j;
                                    break;
                                }
                            }
                            for (int j = labelIndex; j < bodyEnd; j++)
                                _skipNodes.Add(j);
                        }
                    }
                    _lines.Add($"    bra {done};");
                }
            }
            finally
            {
                _indirectDispatchDepth--;
            }
            _lines.Add($"{done}:");
            return true;
        }

        private string CallTarget(
            List<SassNode> nodes, int index, SassInstruction call)
        {
            string? relocated = call.Relocations
                .FirstOrDefault(relocation =>
                    !string.IsNullOrWhiteSpace(relocation.Symbol))
                ?.Symbol;
            if (relocated != null)
                return LabelName(relocated);
            string? raw = call.BranchTargets.FirstOrDefault()
                ?? call.Operands.LastOrDefault();
            if (raw == null)
                throw new NotSupportedException("call without a target");
            string candidate = raw.Trim().Trim('`', '(', ')', ';');
            if (!Regex.IsMatch(candidate, @"^R\d+$",
                    RegexOptions.IgnoreCase))
            {
                if (TrySwitchImmediate(candidate, out int callAddress))
                {
                    foreach (SassNode node in nodes)
                    {
                        if (node is SassLabel label &&
                            label.Address == callAddress)
                            return LabelName(label.Name);
                    }
                }
                return BranchLabel(call);
            }
            // Resolve the small constant-indirect subset: a register written
            // by MOV/MOV32I immediately before CAL and holding a label's code
            // address.  A runtime function pointer still needs a real call
            // model.
            for (int i = index - 1; i >= 0; i--)
            {
                if (nodes[i] is SassLabel)
                    break;
                if (nodes[i] is not SassInstruction producer)
                    continue;
                string op = producer.Mnemonic.Split('.')[0].ToUpperInvariant();
                if (op is not ("MOV" or "MOV32I"))
                    continue;
                int destination = DestinationIndex(producer);
                if (destination < 0 ||
                    destination >= producer.Operands.Count ||
                    !BaseReg(producer.Operands[destination]).Equals(
                        candidate, StringComparison.OrdinalIgnoreCase))
                    continue;
                if (producer.Operands.Count < 2 ||
                    !TrySwitchImmediate(producer.Operands[1],
                        out int address))
                    break;
                foreach (SassNode node in nodes)
                {
                    if (node is SassLabel label && label.Address == address)
                        return LabelName(label.Name);
                }
                break;
            }
            throw new NotSupportedException(
                $"indirect CAL through '{candidate}' is not a constant " +
                "function address");
        }

        private string BranchLabel(SassInstruction instruction)
        {
            string target = instruction.BranchTargets.FirstOrDefault()
                ?? instruction.Operands.LastOrDefault()
                ?? throw new NotSupportedException("branch without a target");
            target = target.Trim().Trim('`', '(', ')', ';');
            if (target.StartsWith(".L_", StringComparison.Ordinal))
                target = target[3..];
            if (_inlineLabelOverrides != null)
            {
                if (_inlineLabelOverrides.TryGetValue(target, out string? inline))
                    return inline;
                if (_inlineLabelOverrides.TryGetValue(".L_" + target, out inline))
                    return inline;
            }
            if (_labels.TryGetValue(target, out string? label))
                return label;
            if (_labels.TryGetValue(".L_" + target, out label))
                return label;
            return "L_" + Sanitize(target);
        }

        private static string LoadWidth(string mnemonic)
        {
            string upper = mnemonic.ToUpperInvariant();
            if (upper.Contains(".128"))
                return "v4.u32";
            if (upper.Contains(".64"))
                return "v2.u32";
            if (upper.Contains(".U16"))
                return "u16";
            if (upper.Contains(".U8"))
                return "u8";
            return "u32";
        }

        private static string Immediate(string value)
        {
            string trimmed = value.Trim();
            if (trimmed.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
                return trimmed;
            if (int.TryParse(trimmed, NumberStyles.Integer,
                    CultureInfo.InvariantCulture, out int parsed))
                return parsed.ToString(CultureInfo.InvariantCulture);
            if (trimmed.StartsWith("-0x", StringComparison.OrdinalIgnoreCase))
                return trimmed;
            return trimmed;
        }

        private string Guard(string? guard)
        {
            if (string.IsNullOrWhiteSpace(guard))
                return "";
            string trimmed = guard.Trim();
            bool negated = trimmed.StartsWith("@!", StringComparison.Ordinal);
            string predicate = trimmed.TrimStart('@', '!');
            if (predicate.Equals("PT", StringComparison.OrdinalIgnoreCase))
            {
                // @PT always executes; @!PT never does.  A literal "@1"
                // is not valid PTX, so materialise a constant predicate.
                if (!negated)
                    return "";
                string never = $"%pt{_nextTempPred++}";
                _lines.Add($"    mov.pred {never}, 0;");
                return $"@{never}";
            }
            string mapped = Pred(predicate);
            return negated ? $"@!{mapped}" : $"@{mapped}";
        }

        // Materialize a predicate guard expression ("@%psN"/"@!%psN")
        // as a bare predicate register, emitting a negated temp when the
        // guard is inverted.
        private string PredicateValue(string guardText)
        {
            string t = guardText.Trim();
            bool negated = t.StartsWith("@!", StringComparison.Ordinal);
            t = t.TrimStart('@', '!');
            if (!negated)
                return t;
            string temp = $"%pt{_nextTempPred++}";
            _lines.Add($"    not.pred {temp}, {t};");
            return temp;
        }

        private string CombineGuard(string? guard, string extraGuard)
        {
            if (string.IsNullOrWhiteSpace(guard))
                return Guard(extraGuard);
            throw new NotSupportedException(
                "cannot combine an instruction guard with a reconvergence guard");
        }

        private string ParamPointer(PtxParameter parameter)
        {
            return _paramPointers.TryGetValue(parameter, out string? pointer)
                ? pointer
                : $"%rdp{ParameterIndex(parameter)}";
        }

        private int ParameterIndex(PtxParameter parameter)
        {
            for (int i = 0; i < _parameters.Count; i++)
            {
                if (ReferenceEquals(_parameters[i], parameter) ||
                    _parameters[i] == parameter ||
                    (_parameters[i].Name == parameter.Name &&
                     _parameters[i].Part == 0))
                    return i;
            }
            return -1;
        }

        private static int WideWords(PtxParameter parameter)
        {
            const string prefix = "u64x";
            if (!parameter.Type.StartsWith(prefix, StringComparison.Ordinal))
                return 0;
            return int.TryParse(parameter.Type.AsSpan(prefix.Length),
                out int n) && n >= 2 ? n : 0;
        }

        private static PtxParameter ParamPart(PtxParameter parameter, int part)
        {
            return parameter with
            {
                Type = "u64",
                Offset = parameter.Offset + part * 8,
                Part = part,
            };
        }

        private string LabelName(string name)
        {
            if (_inlineLabelOverrides != null &&
                _inlineLabelOverrides.TryGetValue(name, out string? inline))
                return inline;
            if (_labels.TryGetValue(name, out string? label))
                return label;
            label = $"L_{_nextLabel++}";
            _labels[name] = label;
            return label;
        }

        private static string Sanitize(string value)
        {
            var builder = new StringBuilder();
            foreach (char ch in value)
                builder.Append(char.IsLetterOrDigit(ch) || ch == '_' ? ch : '_');
            return builder.ToString();
        }
    }
}
