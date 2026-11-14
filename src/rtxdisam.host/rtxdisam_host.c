// 64-bit command-line host for rtxdisam_native.
//
// ACuda is often a 32-bit DLL, while NativeAOT only supports 64-bit Windows.
// This tiny process bridges that gap: ACuda writes a cubin/fatbin to a temp
// file, runs this helper, and reads back the lifted PTX.
//
// Usage:
//   rtxdisam_host.exe <input.cubin> <output.ptx> [target] [entry]
//
// rtxdisam_native.dll must be next to this executable (or on PATH).

#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef int (*rtxdisam_bytes_target_fn)(
    const uint8_t *data, size_t size, const char *entry,
    const char *parameters, const char *target, int output,
    char **sass, char **ptx);
typedef int (*rtxdisam_bytes_fn)(
    const uint8_t *data, size_t size, const char *entry,
    const char *parameters, int output, char **sass, char **ptx);
typedef void (*rtxdisam_free_fn)(void *memory);
typedef const char *(*rtxdisam_error_fn)(void);

static unsigned char *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long n = ftell(f);
    if (n < 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    unsigned char *data = (unsigned char *)malloc((size_t)n);
    if (!data || fread(data, 1, (size_t)n, f) != (size_t)n) {
        free(data);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *size = (size_t)n;
    return data;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <input.cubin> <output.ptx> [target]\n", argv[0]);
        return 2;
    }

    size_t size = 0;
    unsigned char *data = read_file(argv[1], &size);
    if (!data || size == 0) {
        fprintf(stderr, "rtxdisam_host: cannot read '%s'\n", argv[1]);
        free(data);
        return 3;
    }

    HMODULE library = LoadLibraryA("rtxdisam_native.dll");
    if (!library) {
        fprintf(stderr,
                "rtxdisam_host: cannot load rtxdisam_native.dll "
                "(error %lu)\n", (unsigned long)GetLastError());
        free(data);
        return 4;
    }

    rtxdisam_bytes_target_fn with_target =
        (rtxdisam_bytes_target_fn)GetProcAddress(
            library, "rtxdisam_cubin_bytes_to_text_with_target");
    rtxdisam_bytes_fn bytes =
        (rtxdisam_bytes_fn)GetProcAddress(
            library, "rtxdisam_cubin_bytes_to_text");
    rtxdisam_free_fn release =
        (rtxdisam_free_fn)GetProcAddress(library, "rtxdisam_free");
    rtxdisam_error_fn last_error =
        (rtxdisam_error_fn)GetProcAddress(library, "rtxdisam_last_error");
    if (!release || (!with_target && !bytes)) {
        fprintf(stderr, "rtxdisam_host: rtxdisam_native exports missing\n");
        free(data);
        return 5;
    }

    const char *target = argc > 3 && argv[3][0] ? argv[3] : "sm_86";
    const char *entry = argc > 4 && argv[4][0] &&
        strcmp(argv[4], "-") != 0 ? argv[4] : NULL;
    char *sass = NULL;
    char *ptx = NULL;
    int rc;
    if (with_target) {
        rc = with_target(data, size, entry, NULL, target, 2, &sass, &ptx);
    } else {
        rc = bytes(data, size, entry, NULL, 2, &sass, &ptx);
    }
    free(data);
    if (sass)
        release(sass);
    if (rc != 0 || !ptx) {
        const char *message = last_error ? last_error() : NULL;
        fprintf(stderr, "rtxdisam_host: %s\n",
                message && message[0] ? message : "lift failed");
        if (ptx)
            release(ptx);
        return 6;
    }

    FILE *out = fopen(argv[2], "wb");
    if (!out) {
        fprintf(stderr, "rtxdisam_host: cannot write '%s'\n", argv[2]);
        release(ptx);
        return 7;
    }
    size_t len = strlen(ptx);
    int ok = fwrite(ptx, 1, len, out) == len;
    fclose(out);
    release(ptx);
    return ok ? 0 : 8;
}
