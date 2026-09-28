// dumpdeq: dequantize rows [r0, r0+nr) of one GGUF tensor with PrismML's OWN type traits
// (ggml_get_type_traits(type)->to_float) and write the raw block bytes and the float32 result — how the
// committed rows in ferric-gguf/tests/fixtures/bonsai2/ were produced (blk.0.attn_qkv.weight, r0=100,
// nr=2, from each of the three Bonsai 2 files; the fork's f32 came out byte-identical for all three).
// Build inside a checkout of PrismML-Eng/llama.cpp @ adfffbe, linking ggml.
// usage: dumpdeq <model.gguf> <tensor> <r0> <nr> <out.raw> <out.f32>
#include "ggml.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char ** argv) {
    if (argc < 7) { fprintf(stderr, "usage\n"); return 1; }
    ggml_context * ctx = nullptr;
    gguf_init_params ip = { true, &ctx };
    gguf_context * g = gguf_init_from_file(argv[1], ip);
    if (!g) return 2;
    int64_t ti = gguf_find_tensor(g, argv[2]);
    if (ti < 0) { fprintf(stderr, "no tensor\n"); return 3; }
    ggml_type ty = gguf_get_tensor_type(g, ti);
    size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, ti);
    int64_t ne0 = ggml_get_tensor(ctx, argv[2])->ne[0];
    size_t rb = ggml_row_size(ty, ne0);
    long r0 = atol(argv[3]), nr = atol(argv[4]);
    std::vector<uint8_t> raw(rb * nr);
    FILE * f = fopen(argv[1], "rb");
    fseeko(f, off + rb * r0, SEEK_SET);
    if (fread(raw.data(), 1, raw.size(), f) != raw.size()) return 4;
    fclose(f);
    std::vector<float> out(ne0 * nr);
    const ggml_type_traits * tr = ggml_get_type_traits(ty);
    for (long r = 0; r < nr; r++) tr->to_float(raw.data() + r * rb, out.data() + r * ne0, ne0);
    FILE * a = fopen(argv[5], "wb"); fwrite(raw.data(), 1, raw.size(), a); fclose(a);
    FILE * b = fopen(argv[6], "wb"); fwrite(out.data(), 4, out.size(), b); fclose(b);
    printf("type=%d (%s) ne0=%lld row_bytes=%zu rows=%ld\n", (int) ty, ggml_type_name(ty), (long long) ne0, rb, nr);
    gguf_free(g); ggml_free(ctx);
    return 0;
}
