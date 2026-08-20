// Exhaustive (E8M0 scale byte) x (E2M1 code) grid straight out of ggml's own to_float.
// Writes 256*16 little-endian f32 to argv[1], in order e-major then code.
// Also prints the raw bit pattern of every code at e=127 so the sign of the zeros is visible.
//
//   clang -O1 -I/opt/homebrew/include mxfp4_grid.c \
//         -L/opt/homebrew/lib -lggml-base -Wl,-rpath,/opt/homebrew/lib -lm -o mxfp4_grid
//   ./mxfp4_grid grid.f32
//   cargo run -p ferric-gguf --example mxfp4_ref_diff -- --grid grid.f32
//
// No real checkpoint reaches the ends of this grid: a measured MXFP4 tensor used 8 distinct scale
// bytes out of 256 (117..124). e=255 is only reachable synthetically, and e=255 is exactly where a
// dequant written as `E2M1[code] * 2^(e-127)` diverges from ggml — 2^128 is not an f32.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ggml.h"

int main(int argc, char **argv) {
    const struct ggml_type_traits *tr = ggml_get_type_traits(GGML_TYPE_MXFP4);
    unsigned char blk[17];
    float out[32];
    float *grid = malloc(sizeof(float) * 256 * 16);
    for (int e = 0; e < 256; e++) {
        for (int c = 0; c < 16; c++) {
            memset(blk, 0, sizeof blk);
            blk[0] = (unsigned char)e;
            blk[1] = (unsigned char)c;        // low nibble of byte 0 -> element 0
            tr->to_float(blk, out, 32);
            grid[e * 16 + c] = out[0];
        }
    }
    printf("code bits at e=127 (scale 1.0):\n");
    for (int c = 0; c < 16; c++) {
        uint32_t b; float v = grid[127 * 16 + c]; memcpy(&b, &v, 4);
        printf("  code %2d -> 0x%08x  %g\n", c, b, v);
    }
    printf("edge e=0   code1: 0x%08x  code0: 0x%08x\n",
           *(uint32_t*)&grid[0*16+1], *(uint32_t*)&grid[0*16+0]);
    printf("edge e=255 code0: 0x%08x  code2: 0x%08x  code9: 0x%08x\n",
           *(uint32_t*)&grid[255*16+0], *(uint32_t*)&grid[255*16+2], *(uint32_t*)&grid[255*16+9]);
    if (argc > 1) { FILE *f = fopen(argv[1], "wb"); fwrite(grid, 4, 256*16, f); fclose(f); }
    return 0;
}
