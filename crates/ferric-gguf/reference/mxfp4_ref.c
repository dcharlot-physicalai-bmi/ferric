// llama.cpp/ggml REFERENCE dequant for one tensor of a gguf.
//   usage: mxfp4_ref <model.gguf> <tensor-name> <out.f32>
// Reads the tensor with ggml's OWN gguf parser and dequantizes with ggml's OWN to_float,
// then writes little-endian f32 to <out.f32>. Ferric is diffed against this file.
//
//   clang -O1 -I/opt/homebrew/include mxfp4_ref.c \
//         -L/opt/homebrew/lib -lggml-base -Wl,-rpath,/opt/homebrew/lib -lm -o mxfp4_ref
//   ./mxfp4_ref model.gguf blk.0.attn_q.weight ref.f32
//   cargo run -p ferric-gguf --example mxfp4_ref_diff -- model.gguf blk.0.attn_q.weight ref.f32
//
// There is no GPT-OSS checkpoint required to exercise this: any gguf can be given real MXFP4
// tensors with
//   llama-quantize --allow-requantize --tensor-type attn_q=mxfp4 --tensor-type ffn_down=mxfp4 \
//                  src.gguf out.gguf MXFP4_MOE
// Note that plain `MXFP4_MOE` on a DENSE model produces a file with zero MXFP4 tensors — the ftype
// only reaches MoE expert weights — so always check the type histogram before trusting the diff.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ggml.h"
#include "gguf.h"

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <model.gguf> <tensor> <out.f32>\n", argv[0]); return 2; }
    struct gguf_init_params p = { /*no_alloc*/ true, /*ctx*/ NULL };
    struct gguf_context *ctx = gguf_init_from_file(argv[1], p);
    if (!ctx) { fprintf(stderr, "gguf_init_from_file failed\n"); return 1; }
    int64_t id = gguf_find_tensor(ctx, argv[2]);
    if (id < 0) { fprintf(stderr, "no tensor '%s'\n", argv[2]); return 1; }
    enum ggml_type ty = gguf_get_tensor_type(ctx, id);
    size_t nbytes = gguf_get_tensor_size(ctx, id);
    size_t off = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, id);
    const struct ggml_type_traits *tr = ggml_get_type_traits(ty);
    int64_t blck = tr->blck_size;
    size_t  tsz  = tr->type_size;
    int64_t nelem = (int64_t)(nbytes / tsz) * blck;
    printf("tensor      : %s\n", argv[2]);
    printf("ggml_type   : %d (%s)\n", (int)ty, tr->type_name);
    printf("blck_size   : %lld\n", (long long)blck);
    printf("type_size   : %zu\n", tsz);
    printf("nbytes      : %zu\n", nbytes);
    printf("nelem       : %lld\n", (long long)nelem);
    printf("file_offset : %zu\n", off);

    unsigned char *raw = (unsigned char*)malloc(nbytes);
    FILE *f = fopen(argv[1], "rb");
    fseek(f, (long)off, SEEK_SET);
    if (fread(raw, 1, nbytes, f) != nbytes) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);

    float *out = (float*)malloc(sizeof(float) * (size_t)nelem);
    tr->to_float(raw, out, nelem);

    FILE *o = fopen(argv[3], "wb");
    fwrite(out, sizeof(float), (size_t)nelem, o);
    fclose(o);

    double s = 0, amax = 0;
    for (int64_t i = 0; i < nelem; i++) { s += out[i]; if (fabsf(out[i]) > amax) amax = fabsf(out[i]); }
    printf("sum         : %.10g\n", s);
    printf("amax        : %.10g\n", amax);
    printf("first16     :");
    for (int i = 0; i < 16; i++) printf(" %g", out[i]);
    printf("\n");
    // first block's raw bytes, so the layout can be eyeballed against the floats
    printf("raw blk0    :");
    for (size_t i = 0; i < tsz && i < 32; i++) printf(" %02x", raw[i]);
    printf("\n");
    return 0;
}
