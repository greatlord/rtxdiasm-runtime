# rtxdisam native library

`rtxdisam_native` is a C ABI shared library built with NativeAOT. ACuda can
load/link it at runtime and translate an in-memory cubin or fatbin into SASS,
PTX, or both.

## Build

Run the build script on the target operating system:

```powershell
.\build_windows.ps1
```

```bash
./build_linux.sh
```

Windows emits `out/win-x64/rtxdisam.exe`, `rtxdisam_native.dll`, and
`rtxdisam_native.lib` for link-time use. Linux emits `out/linux-x64/rtxdisam`,
`rtxdisam_native.so`, and a `librtxdisam_native.so` link alias for
`-lrtxdisam_native`.

The native library embeds the zstd-compressed ISA tables, so an embedding host
does not need to deploy the `isa_json` directory beside the library.

## C API

Include [`rtxdisam_native.h`](rtxdisam_native.h). The main memory-to-memory
entry point is:

```c
int rtxdisam_cubin_bytes_to_text(
    const uint8_t* data,
    size_t size,
    const char* entry,
    const char* parameters,
    int output,
    char** sass,
    char** ptx);
```

- `entry` may be `NULL` for SASS; PTX requires it when multiple text sections
  exist.
- `parameters` is optional: `"name:type:offset[:size];..."`, with type
  `u32`, `s32`, or `u64`. If omitted for a cubin/fatbin input, parameter
  metadata is extracted from its `.nv.info` KPARAM records automatically.
- `output` is `RTXDISAM_OUTPUT_SASS`, `RTXDISAM_OUTPUT_PTX`, or
  `RTXDISAM_OUTPUT_BOTH`.
- `rtxdisam_cubin_bytes_to_text_with_target` adds an optional preferred
  architecture such as `"sm_80"`. For a multi-architecture fatbin it selects
  the matching cubin member; the selected member's target and minimum PTX
  version are used for the lifted PTX.
- Release returned strings with `rtxdisam_free`.
- `rtxdisam_cubin_to_text` accepts a file path instead of a buffer.
- `rtxdisam_last_error` returns the calling thread's last error.
- `rtxdisam_version` returns the library version.
- PTX lifted from a legacy `sm_101` cubin is emitted as `.target sm_110`
  with PTX 9.0, matching the PTX ISA rename.

## Example

```c
#include "rtxdisam_native.h"

char* sass = NULL;
char* ptx = NULL;
int rc = rtxdisam_cubin_bytes_to_text(
    data, size, "k", NULL, RTXDISAM_OUTPUT_BOTH, &sass, &ptx);
if (rc != 0) {
    fprintf(stderr, "%s\n", rtxdisam_last_error());
} else {
    puts(sass);
    puts(ptx);
    rtxdisam_free(sass);
    rtxdisam_free(ptx);
}
```
