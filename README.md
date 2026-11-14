# rtxdiasm-runtime

Runtime/tooling module providing the C# **SASS→PTX lift** components needed by ACuda (and related tooling).

This module packages:
- `src/rtxdisam` — core lift/disassembly/assembly/fatbin/cubin logic (e.g. `CubinReader`, `RtxDisamApi`, `SassToPtx`)
- `src/rtxdisam.native` — native-exported library consumed by C bridges (e.g. ACuda `rtxdisam_bridge`)
- `src/rtxdisam.host` — lightweight C host/helper process backend
- `src/rtxdisam.cli` — CLI frontend (`rtxdisam`)

## Targets
Intended to build/publish for:
- Windows: `win-x64`, `win-x86`
- Linux: `linux-x64`, `linux-x86`

## ACuda integration
ACuda uses the lift backend when no embedded PTX is available:
- `RTXDISAM_LIBRARY` → `rtxdisam_native` shared library
- `RTXDISAM_HOST` → `rtxdisam_host` executable (fallback/helper path)

## Build (scaffold)
See `scripts/` for publish/build entry points:
- `scripts/publish.ps1` — publish managed/native components for selected RIDs (Windows/PowerShell)
- `scripts/build-host.sh` — build `rtxdisam.host` on Linux

> Note: This module was initially scaffolded by copying lift components from `ConsoleApp2/src` (core/native/host/cli). Other side projects remain in `ConsoleApp2` until migrated explicitly.
