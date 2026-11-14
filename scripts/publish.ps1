param(
  [string[]]$Rids = @('win-x64','win-x86','linux-x64','linux-x86'),
  [string]$Configuration = 'Release',
  [string]$ArtifactsRoot = "$(Join-Path $PSScriptRoot '..\artifacts')"
)

$ErrorActionPreference = 'Stop'
$repo = Resolve-Path (Join-Path $PSScriptRoot '..')
$art = Resolve-Path (New-Item -ItemType Directory -Force -Path $ArtifactsRoot)

foreach ($rid in $Rids) {
  $out = Join-Path $art $rid
  New-Item -ItemType Directory -Force -Path $out | Out-Null

  Write-Host "== Publishing rtxdisam.native ($rid) =="
  dotnet publish "$repo/src/rtxdisam.native/rtxdisam.native.csproj" -c $Configuration -r $rid -o (Join-Path $out 'native')

  Write-Host "== Publishing rtxdisam.cli ($rid) =="
  dotnet publish "$repo/src/rtxdisam.cli/rtxdisam.cli.csproj" -c $Configuration -r $rid -o (Join-Path $out 'cli')
}

Write-Host "Done. Artifacts: $art"
Write-Host "Note: build rtxdisam.host separately via CMake for target platforms (see src/rtxdisam.host and scripts/build-host.sh)."
