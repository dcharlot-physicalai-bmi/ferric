// The FORK's side of scripts/bonsai2_decode_joules.sh: PrismML's llama.cpp (llama.h) decoding Bonsai 2
// greedily in the IDLE/RUN/SUMMARY protocol of ferric-llama's bonsai2_decode_joules example — same
// prompt ids, same chunking, same greedy k tokens with a logits read per step, same HASH lines.
// Build against a fork checkout at adfffbe or its release dylibs, e.g.:
//   clang++ -std=c++17 -O2 bonsai2_decode_joules.cpp -I<fork>/include -I<fork>/ggml/include \
//       -L<release dir> -lllama -lggml -lggml-base -Wl,-rpath,<release dir> -o bonsai2_decode_joules
// usage: bonsai2_decode_joules <model.gguf> <id,id,...> [chunks 5] [k 128] [idle_s 4] [--noflash]
//        env BONSAI2_NGL=<gpu layers> BONSAI2_THREADS=<cpu threads> for partial offload
// Linux/CUDA: g++ -std=c++17 -O2 bonsai2_decode_joules.cpp -I<fork>/include -I<fork>/ggml/include \
//       -L<fork>/build/bin -lllama -lggml -lggml-base -Wl,-rpath,<fork>/build/bin -o bonsai2_decode_joules
#include "llama.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
static double now() { return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count(); }
static int argmax(const float * v, int n) { int b = 0; for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i; return b; }
int main(int argc, char ** argv) {
    std::vector<llama_token> ids;
    { std::string s = argv[2]; size_t p = 0; while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); ids.push_back(atoi(s.substr(p, q - p).c_str())); p = q + 1; } }
    int chunks = argc > 3 ? atoi(argv[3]) : 5, k = argc > 4 ? atoi(argv[4]) : 128; double idle = argc > 5 ? atof(argv[5]) : 4.0;
    bool noflash = argc > 6 && !strcmp(argv[6], "--noflash");
    llama_backend_init();
    // BONSAI2_NGL: layers on the GPU (default all) — partial offload for a GPU smaller than the model;
    // BONSAI2_THREADS: CPU threads for the layers left on the host (llama.cpp's -t).
    auto mp = llama_model_default_params(); mp.n_gpu_layers = getenv("BONSAI2_NGL") ? atoi(getenv("BONSAI2_NGL")) : 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto cp = llama_context_default_params(); cp.n_ctx = 4096; cp.n_batch = cp.n_ubatch = 512; cp.n_seq_max = 1;
    if (noflash) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (getenv("BONSAI2_THREADS")) cp.n_threads = cp.n_threads_batch = atoi(getenv("BONSAI2_THREADS"));
    llama_context * ctx = llama_init_from_model(model, cp);
    auto run = [&](int kk) {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch b = llama_batch_init((int) ids.size(), 0, 1);
        for (size_t i = 0; i < ids.size(); i++) { b.token[i] = ids[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = i + 1 == ids.size(); }
        b.n_tokens = ids.size(); llama_decode(ctx, b); llama_batch_free(b);
        return argmax(llama_get_logits_ith(ctx, -1), nv);
    };
    auto decode = [&](int first, int kk, std::vector<int> & out) {
        out.assign(1, first); int pos = ids.size(), next = first;
        llama_batch one = llama_batch_init(1, 0, 1);
        for (int s = 1; s < kk; s++) {
            one.token[0] = next; one.pos[0] = pos++; one.n_seq_id[0] = 1; one.seq_id[0][0] = 0; one.logits[0] = 1; one.n_tokens = 1;
            llama_decode(ctx, one); next = argmax(llama_get_logits_ith(ctx, 0), nv); out.push_back(next);
        }
        llama_batch_free(one);
    };
    std::vector<int> out; decode(run(0), 16, out); // warm-up
    long tok = 0; double secs = 0;
    for (int c = 0; c < chunks; c++) {
        int first = run(0);
        double t0 = now(); std::this_thread::sleep_for(std::chrono::duration<double>(idle)); printf("IDLE %.3f %.3f\n", t0, now());
        t0 = now(); decode(first, k, out); double t1 = now();
        printf("RUN %.3f %.3f %zu 1\n", t0, t1, out.size()); fflush(stdout);
        unsigned long long h = 0xcbf29ce484222325ULL; for (int x : out) { h ^= (unsigned long long) x; h *= 0x100000001b3ULL; }
        fprintf(stderr, "HASH %016llx %zu tokens %.1f ms/token\n", h, out.size(), (t1 - t0) * 1e3 / out.size());
        tok += out.size(); secs += t1 - t0;
    }
    double t0 = now(); std::this_thread::sleep_for(std::chrono::duration<double>(idle)); printf("IDLE %.3f %.3f\n", t0, now());
    printf("SUMMARY tokens %ld seconds %.3f tok_per_s %.2f\n", tok, secs, tok / secs);
    llama_free(ctx); llama_model_free(model);
    return 0;
}
