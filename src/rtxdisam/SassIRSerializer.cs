namespace rtxdisam;

public static class SassIRSerializer
{
    public static void Write(
        string path, SassProgram program, string architecture,
        string? toolkit = null)
    {
        var model = new
        {
            architecture,
            toolkit,
            sections = program.Sections.Select(section => new
            {
                name = section.Name,
                alignment = section.Alignment,
                registerCount = section.RegisterCount,
                nodes = section.Nodes.Select(node => node switch
                {
                    SassLabel label => (object)new
                    {
                        kind = "label",
                        name = label.Name,
                        address = label.Address,
                    },
                    SassInstruction instruction => new
                    {
                        kind = "instruction",
                        mnemonic = instruction.Mnemonic,
                        guard = instruction.Guard,
                        operands = instruction.Operands,
                        branchTargets = instruction.BranchTargets,
                        address = instruction.Address,
                        line = instruction.Line,
                        source = instruction.Source,
                    },
                    _ => new { kind = "unknown" },
                }),
            }),
        };

        string json = JsonOutput.Serialize(model, indented: true);
        File.WriteAllText(path, json);
    }
}
