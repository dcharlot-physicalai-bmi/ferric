// The FORK's side of the whole-file decoder check (ferric-gguf/examples/deqhash.rs is Ferric's):
// FNV-1a 64 over the float32 BIT PATTERNS of every quantized tensor's dequantization, using PrismML's
// own ggml_get_type_traits(type)->to_float. One line per tensor: name type n hash. Output for the three
// Bonsai 2 files (identical) is committed as ferric-gguf/tests/fixtures/bonsai2/deqhash_fork_all_packings.txt.
// Build inside a checkout of PrismML-Eng/llama.cpp @ adfffbe (e.g. as examples/dumplogits, linking ggml).
// ⛔ The offset basis is 0xcbf29ce484222325 — a first version typed it in decimal with a digit missing
// and every hash disagreed while every value matched.
// usage: deqhash <model.gguf>      (DEQHASH_ONLY=<tensor>, DEQHASH_ROWS=<n> narrow the run)
#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
int main(int argc, char ** argv) {
    ggml_context * ctx = nullptr;
    gguf_init_params ip = { true, &ctx };
    gguf_context * g = gguf_init_from_file(argv[1], ip);
    if (!g) return 2;
    FILE * f = fopen(argv[1], "rb");
    const size_t base = gguf_get_data_offset(g);
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        const char * name = gguf_get_tensor_name(g, i);
        ggml_type ty = gguf_get_tensor_type(g, i);
        if (!ggml_is_quantized(ty)) continue;
        if (getenv("DEQHASH_ONLY") && std::string(getenv("DEQHASH_ONLY")) != name) continue;
        ggml_tensor * t = ggml_get_tensor(ctx, name);
        const int64_t ne0 = t->ne[0]; int64_t nrows = ggml_nrows(t);
        if (getenv("DEQHASH_ROWS")) nrows = std::min<int64_t>(nrows, atoll(getenv("DEQHASH_ROWS")));
        const size_t rb = ggml_row_size(ty, ne0);
        std::vector<uint8_t> raw(rb);
        std::vector<float> out(ne0);
        const ggml_type_traits * tr = ggml_get_type_traits(ty);
        uint64_t h = 0xcbf29ce484222325ULL;
        fseeko(f, base + gguf_get_tensor_offset(g, i), SEEK_SET);
        for (int64_t r = 0; r < nrows; r++) {
            if (fread(raw.data(), 1, rb, f) != rb) return 4;
            tr->to_float(raw.data(), out.data(), ne0);
            for (int64_t j = 0; j < ne0; j++) { uint32_t b; memcpy(&b, &out[j], 4); for (int k = 0; k < 4; k++) { h ^= (b >> (8*k)) & 0xff; h *= 1099511628211ULL; } }
        }
        printf("%s %s %lld %016llx\n", name, ggml_type_name(ty), (long long) (ne0 * nrows), (unsigned long long) h);
    }
    return 0;
}
