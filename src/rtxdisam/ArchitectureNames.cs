namespace rtxdisam;

internal static class ArchitectureNames
{
    // sm_101 was the name used by PTX 8.6-8.8 for what PTX 9.0 calls
    // sm_110.  Keep the alias handling in one place so cubin assembly and PTX
    // emission cannot drift.
    public static bool TryCanonicalizeLegacySm101(
        string architecture, out string canonical)
    {
        string value = architecture.Trim();
        string? suffix = null;
        if (value.Length >= 6 &&
            value[..6].Equals("sm_101", StringComparison.OrdinalIgnoreCase) &&
            (value.Length == 6 || !char.IsDigit(value[6])))
            suffix = value[6..];
        else if (value.Length >= 5 &&
            value[..5].Equals("sm101", StringComparison.OrdinalIgnoreCase) &&
            (value.Length == 5 || !char.IsDigit(value[5])))
            suffix = value[5..];
        canonical = suffix == null
            ? value
            : "sm_110" + suffix.ToLowerInvariant();
        return suffix != null;
    }

    public static string CanonicalizeLegacySm101(string architecture)
    {
        TryCanonicalizeLegacySm101(architecture, out string canonical);
        return canonical;
    }
}
