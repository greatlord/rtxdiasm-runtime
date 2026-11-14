#ifndef RTXDISAM_NATIVE_H
#define RTXDISAM_NATIVE_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define RTXDISAM_API __declspec(dllimport)
#else
#define RTXDISAM_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum rtxdisam_output {
    RTXDISAM_OUTPUT_SASS = 1,
    RTXDISAM_OUTPUT_PTX  = 2,
    RTXDISAM_OUTPUT_BOTH = 3
};

/*
 * Translate a cubin or fatbin file to text.
 *
 * entry: optional entry/function name. Required for PTX when the container
 *        has more than one text section.
 * parameters: optional "name:type:offset[:size];..." list for PTX parameters.
 *             type is u32, s32, or u64. When NULL/empty, cubin/fatbin
 *             KPARAM metadata is extracted automatically.
 * sass/ptx: receive UTF-8, NUL-terminated strings. Release with
 *           rtxdisam_free. Unrequested outputs are set to NULL.
 *
 * Returns 0 on success, 1 on error.
 */
RTXDISAM_API int rtxdisam_cubin_to_text(
    const char* path,
    const char* entry,
    const char* parameters,
    int output,
    char** sass,
    char** ptx);

/*
 * Memory-to-memory variant. The input buffer may be an ELF cubin or a fatbin.
 */
RTXDISAM_API int rtxdisam_cubin_bytes_to_text(
    const uint8_t* data,
    size_t size,
    const char* entry,
    const char* parameters,
    int output,
    char** sass,
    char** ptx);

/*
 * Target-aware memory-to-memory variant. target is an optional preferred
 * architecture such as "sm_80". For a multi-architecture fatbin, the matching
 * cubin member is selected when available; that member's architecture and
 * minimum PTX version drive the emitted PTX .target/.version directives.
 */
RTXDISAM_API int rtxdisam_cubin_bytes_to_text_with_target(
    const uint8_t* data,
    size_t size,
    const char* entry,
    const char* parameters,
    const char* target,
    int output,
    char** sass,
    char** ptx);

/* Release strings returned by this library. */
RTXDISAM_API void rtxdisam_free(void* memory);

/*
 * Last error for the calling thread. The returned storage is owned by the
 * library and is replaced by the next call on the same thread.
 */
RTXDISAM_API const char* rtxdisam_last_error(void);

/* Version string; release with rtxdisam_free. */
RTXDISAM_API const char* rtxdisam_version(void);

#ifdef __cplusplus
}
#endif

#endif
