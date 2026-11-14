using System.Text.RegularExpressions;

namespace nvDismDecode;

// nvdisasm's SHI_REGISTERS value is derived from the highest referenced
// register slot plus its fixed metadata allowance.  64-bit operands occupy a
// pair and 128-bit vector operands occupy four consecutive registers.
public static class RegisterCountEstimator
{
    public static uint Estimate(
        IEnumerable<string> instructionTexts, bool preVolta = false)
    {
        int maxEnd = 0;
        foreach (string text in instructionTexts)
        {
            int maxToken = 0;
            foreach (Match match in Regex.Matches(
                text, @"(?<![\w.$])R(\d+)(?!\d)"))
            {
                if (int.TryParse(match.Groups[1].Value, out int token))
                {
                    maxToken = Math.Max(maxToken, token);
                    maxEnd = Math.Max(maxEnd, token + 1);
                }
            }

            foreach (Match match in Regex.Matches(
                text, @"(?<![\w.$])R(\d+)\.64\b"))
            {
                if (int.TryParse(match.Groups[1].Value, out int token))
                    maxEnd = Math.Max(maxEnd, token + 2);
            }

            foreach (Match match in Regex.Matches(
                text, @"\.128\s+(?:\[[^\]]*\]\s*,\s*)?R(\d+)"))
            {
                if (int.TryParse(match.Groups[1].Value, out int token))
                    maxEnd = Math.Max(maxEnd, token + 4);
            }

            if (text.Contains("RET.ABS", StringComparison.Ordinal) ||
                text.Contains("RET.REL", StringComparison.Ordinal))
                maxEnd = Math.Max(maxEnd, maxToken + 2);
        }

        return (uint)(maxEnd + (preVolta ? 0 : 2));
    }
}
