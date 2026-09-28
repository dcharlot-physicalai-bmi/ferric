// dumplogits: run PrismML's fork on EXACT token ids and dump float32 logits for every position,
// then optionally continue greedily. Written for Ferric's Bonsai 2 conformance (reference side).
// usage: dumplogits <model.gguf> <id,id,...> <out.bin> [--gen N] [--kv f32|f16] [--ngl N] [--noflash]
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model ids out.bin [--gen N] [--kv f32|f16] [--ngl N] [--noflash]\n", argv[0]); return 1; }
    const char * mpath = argv[1];
    std::vector<llama_token> ids;
    { std::string s = argv[2]; size_t p = 0; while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); ids.push_back(atoi(s.substr(p, q - p).c_str())); p = q + 1; } }
    const char * out = argv[3];
    int gen = 0, ngl = 99, ub = 0; bool f32kv = true, noflash = false;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--gen")) gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kv")) f32kv = !strcmp(argv[++i], "f32");
        else if (!strcmp(argv[i], "--ngl")) ngl = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--noflash")) noflash = true;
        else if (!strcmp(argv[i], "--ubatch")) ub = atoi(argv[++i]);
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(mpath, mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 2; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    auto cp = llama_context_default_params();
    const int n = (int) ids.size();
    cp.n_ctx = 4096;
    cp.n_batch = cp.n_ubatch = n < 512 ? 512 : n;
    // --ubatch 1: every prompt token goes through the DECODE (mat-vec, f32-activation) kernels,
    // one micro-batch per token — the fork's most precise path, used as the tight reference.
    if (ub > 0) cp.n_ubatch = ub;
    cp.n_seq_max = 1;
    if (f32kv) { cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32; }
    if (noflash) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 3; }
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) { b.token[i] = ids[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1; }
    b.n_tokens = n;
    if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 4; }
    FILE * f = fopen(out, "wb");
    for (int i = 0; i < n; i++) fwrite(llama_get_logits_ith(ctx, i), sizeof(float), nv, f);
    int last = n - 1, pos = n;
    std::vector<llama_token> outids;
    for (int g = 0; g < gen; g++) {
        const float * lg = llama_get_logits_ith(ctx, g == 0 ? last : 0);
        int best = 0; for (int j = 1; j < nv; j++) if (lg[j] > lg[best]) best = j;
        outids.push_back(best);
        if (g + 1 == gen) break;
        llama_batch one = llama_batch_init(1, 0, 1);
        one.token[0] = best; one.pos[0] = pos++; one.n_seq_id[0] = 1; one.seq_id[0][0] = 0; one.logits[0] = 1; one.n_tokens = 1;
        if (llama_decode(ctx, one)) { fprintf(stderr, "decode gen failed\n"); return 5; }
        fwrite(llama_get_logits_ith(ctx, 0), sizeof(float), nv, f);
        llama_batch_free(one);
    }
    fclose(f);
    printf("n_vocab=%d n=%d\n", nv, n);
    if (gen) { printf("gen="); for (size_t i = 0; i < outids.size(); i++) printf("%s%d", i ? "," : "", outids[i]); printf("\n"); }
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
