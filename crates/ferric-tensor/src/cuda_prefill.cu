// Ferric NVIDIA tier 3: PREFILL — T rows at once, resident on the device, on the tensor cores.
//
// The decode kernels (cuda_decode.cu) are GEMVs: one activation row, bandwidth-bound, f32 all the way.
// A prompt is T rows, and there the work is compute: T x (every weight) multiply-adds. This file is the
// matrix half — a tiled quantised GEMM on `mma.sync` tensor cores, a tiled causal attention, and the
// SwiGLU between them. Row-wise norm / rope reuse the decode kernels (they take a row index from
// blockIdx.y), so both paths share one implementation of everything that is not a matmul.
//
// ⚠ NUMERICS. Tensor cores multiply f16 x f16 and accumulate in f32, and NEITHER operand is rounded:
//  • weights enter as their INTEGER codes (exact in f16), with each block's scale — and Q4_K/Q5_K's
//    min, via the activation row-sum — applied per 32-wide K tile in f32 (codes16 / gemm_t);
//  • each activation is split into hi = f16(a) and lo = f16(a - hi), both through the tensor cores,
//    so a·w = hi·w + lo·w to ~2^-22 relative.
//  What is left is f32 accumulation order — the same kind of difference the decode GEMVs have.
//  ⭐ Measured on the way here (Qwen3-0.6B Q5_K_M, 1000-row prompt, max |Δ logit| from the f32 WGSL
//  path): f16 activations and f16-rounded weights 0.219 (argmax 1023/1024); split activations 0.028;
//  the numbers for this final form are in scripts/cuda_conformance.sh. The rounded-weight form also
//  failed a control: with Qwen2.5's biases dropped, Q8_0's `d·q` rounded to f16 put the native prefill
//  1.63 logits from WGSL-under-the-same-control. An activation above f16's 65504 would become inf:
//  every A-tile load checks, and a hit raises `ovf` so the host throws the result away and runs the
//  f32 WGSL prefill instead — a fallback, never a silent inf.
//
// `mma.sync.aligned.m16n8k8` (not k16) keeps this loadable as compute_75 PTX like the decode module:
// m16n8k16 needs sm_80, and the tier's contract is "the driver alone loads it on Turing and newer".
//
// Build: nvcc -O3 -arch=compute_75 -ptx cuda_prefill.cu -o cuda_prefill.ptx

__device__ __forceinline__ float f16_to_f32(unsigned h) {
    const unsigned s = (h >> 15u) & 1u, e = (h >> 10u) & 0x1fu, m = h & 0x3ffu;
    unsigned bits;
    if (e == 0u) {
        if (m == 0u) { bits = s << 31u; }
        else { unsigned mm = m, ee = 127u - 15u + 1u; while ((mm & 0x400u) == 0u) { mm <<= 1u; --ee; }
               mm &= 0x3ffu; bits = (s << 31u) | (ee << 23u) | (mm << 13u); }
    } else if (e == 31u) { bits = (s << 31u) | 0x7f800000u | (m << 13u); }
    else { bits = (s << 31u) | ((e + 127u - 15u) << 23u) | (m << 13u); }
    return __uint_as_float(bits);
}
// f32 -> f16 bits, round-to-nearest-even (the PTX cvt; no header needed).
__device__ __forceinline__ unsigned f2h(float x) {
    unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x)); return (unsigned)h;
}
__device__ __forceinline__ unsigned pack2(float lo, float hi) { return f2h(lo) | (f2h(hi) << 16u); }
// The residual of a packed half2 against the two floats it rounded: the "lo" half of the split.
__device__ __forceinline__ unsigned resid2(float x, float y, unsigned hi) {
    return pack2(x - f16_to_f32(hi & 0xffffu), y - f16_to_f32(hi >> 16u));
}
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (unsigned off = 16u; off > 0u; off >>= 1u) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (unsigned off = 16u; off > 0u; off >>= 1u) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}
__device__ __forceinline__ unsigned scbyte(const unsigned* __restrict__ aux, unsigned ab, unsigned i) {
    return (aux[ab + 1u + (i >> 2u)] >> (8u * (i & 3u))) & 0xffu;
}
// Q4_K / Q5_K 6-bit (scale, min) of sub-block s — ggml `get_scale_min_k4`, as the GEMVs do it.
__device__ __forceinline__ void scmin(const unsigned* __restrict__ aux, unsigned ab, unsigned s, float d, float dmin,
                                      float& ds, float& mm) {
    unsigned sc, mn;
    if (s < 4u) { sc = scbyte(aux, ab, s) & 63u; mn = scbyte(aux, ab, s + 4u) & 63u; }
    else { const unsigned a = scbyte(aux, ab, s + 4u), lo = scbyte(aux, ab, s - 4u), hi = scbyte(aux, ab, s);
           sc = (a & 0x0Fu) | ((lo >> 6u) << 4u); mn = (a >> 4u) | ((hi >> 6u) << 4u); }
    ds = d * (float)sc; mm = dmin * (float)mn;
}

// ── The INTEGER codes of 16 consecutive weights of output row `n` — values [32·kc + 16·h, +16) of
//    the row — as 8 packed half2 words, plus the affine map that turns them into weights:
//        w = scale · q − mn.
//    Every code is a small integer (Q8_0 −128..127 is the widest), so it is EXACT in f16: the tensor
//    cores multiply the true codes, and the scale and min are applied per 32-wide K tile in f32
//    (see gemm_t). This is llama.cpp's MMQ factorisation with the activations kept (split) f16 instead
//    of int8. F: 0 Q4_K, 1 Q5_K, 2 Q6_K, 3 Q8_0, 4 Q5_0 (QFmt order). ──
template <int F>
__device__ __forceinline__ void codes16(const unsigned* __restrict__ codes, const unsigned* __restrict__ aux,
                                        unsigned n, unsigned K, unsigned kc, unsigned h, unsigned (&o)[8],
                                        float& scale, float& mn) {
    float v[16];
    mn = 0.f;
    if (F == 0 || F == 1) {
        const unsigned nblk = K / 256u, b = kc >> 3u, s = kc & 7u, bi = n * nblk + b, ab = bi * 4u;
        const unsigned dd = aux[ab];
        scmin(aux, ab, s, f16_to_f32(dd & 0xffffu), f16_to_f32(dd >> 16u), scale, mn);
        const unsigned c = s >> 1u, sh = 4u * (s & 1u);
        const unsigned cb = bi * (F == 0 ? 32u : 40u);
        const uint4 q = *reinterpret_cast<const uint4*>(codes + cb + 8u * c + 4u * h);   // qs bytes [32c+16h, +16)
        uint4 qh = make_uint4(0u, 0u, 0u, 0u);
        if (F == 1) qh = *reinterpret_cast<const uint4*>(codes + cb + 32u + 4u * h);      // qh bytes [16h, +16)
        const unsigned qw[4] = {q.x, q.y, q.z, q.w}, hw[4] = {qh.x, qh.y, qh.z, qh.w};
        #pragma unroll
        for (unsigned w = 0u; w < 4u; ++w)
            #pragma unroll
            for (unsigned k = 0u; k < 4u; ++k) {
                unsigned qq = (qw[w] >> (8u * k + sh)) & 0xfu;
                if (F == 1) qq |= ((hw[w] >> (8u * k + s)) & 1u) << 4u;
                v[4u * w + k] = (float)qq;
            }
    } else if (F == 2) {
        // ql[128] low nibbles / high nibbles, qh[64] 2-bit pairs, 16 int8 scales, f16 d (Q6_K_BODY).
        // Chunk j of the block: hf = j>>2 (128-value half), qq = j&3 (which of q1..q4); element l of the
        // chunk (0..31) reads ql byte 64hf + 32(qq&1) + l, nibble qq>>1; qh byte 32hf + l, bits 2qq; and
        // scale 8hf + 2qq + l/16 — constant across a 16-value half, which is why halves split cleanly.
        const unsigned nblk = K / 256u, b = kc >> 3u, j = kc & 7u, bi = n * nblk + b;
        const unsigned hf = j >> 2u, qq = j & 3u;
        const unsigned cb = bi * 48u, ab = bi * 5u;
        const uint4 ql = *reinterpret_cast<const uint4*>(codes + cb + (64u * hf + 32u * (qq & 1u) + 16u * h) / 4u);
        const uint4 qh = *reinterpret_cast<const uint4*>(codes + cb + 32u + (32u * hf + 16u * h) / 4u);
        const float d = f16_to_f32(aux[ab] & 0xffffu);
        const unsigned si = 8u * hf + 2u * qq + h;
        scale = d * (float)((int)(((aux[ab + 1u + (si >> 2u)] >> (8u * (si & 3u))) & 0xffu) << 24u) >> 24);
        const unsigned lw[4] = {ql.x, ql.y, ql.z, ql.w}, hw[4] = {qh.x, qh.y, qh.z, qh.w};
        const unsigned nsh = 4u * (qq >> 1u), hsh = 2u * qq;
        #pragma unroll
        for (unsigned w = 0u; w < 4u; ++w)
            #pragma unroll
            for (unsigned k = 0u; k < 4u; ++k)
                v[4u * w + k] = (float)((int)(((lw[w] >> (8u * k + nsh)) & 0xfu) | (((hw[w] >> (8u * k + hsh)) & 3u) << 4u)) - 32);
    } else if (F == 3) {
        const unsigned bi = n * (K / 32u) + kc;
        const unsigned sw = aux[bi >> 1u];
        scale = f16_to_f32((bi & 1u) ? (sw >> 16u) : (sw & 0xffffu));
        const uint4 q = *reinterpret_cast<const uint4*>(codes + bi * 8u + 4u * h);
        const unsigned qw[4] = {q.x, q.y, q.z, q.w};
        #pragma unroll
        for (unsigned w = 0u; w < 4u; ++w)
            #pragma unroll
            for (unsigned k = 0u; k < 4u; ++k) v[4u * w + k] = (float)((int)(qw[w] << (24u - 8u * k)) >> 24);
    } else {
        // Q5_0: 16 code bytes; h = 0 -> low nibbles with qh bits 0..15, h = 1 -> high nibbles, bits 16..31.
        const unsigned bi = n * (K / 32u) + kc;
        const unsigned qh = aux[bi * 2u];
        scale = f16_to_f32(aux[bi * 2u + 1u] & 0xffffu);
        const uint4 q = *reinterpret_cast<const uint4*>(codes + bi * 4u);
        const unsigned qw[4] = {q.x, q.y, q.z, q.w};
        #pragma unroll
        for (unsigned w = 0u; w < 4u; ++w)
            #pragma unroll
            for (unsigned k = 0u; k < 4u; ++k) {
                const unsigned e = 4u * w + k;
                v[e] = (float)((int)(((qw[w] >> (8u * k + 4u * h)) & 0xfu) | (((qh >> (e + 16u * h)) & 1u) << 4u)) - 16);
            }
    }
    #pragma unroll
    for (unsigned i = 0u; i < 8u; ++i) o[i] = pack2(v[2u * i], v[2u * i + 1u]);   // exact: small integers
}

// ── C[M, N] = A[M, K] · W[N, K]ᵀ with W quantised. A f32 row-major (lda), C f32 row-major (ldc).
//    Tile 64x64x32, 128 threads = 2x2 warps of 32x32, each warp 2 (m16) x 4 (n8) mma tiles, k8 steps.
//    Per 32-wide K tile, per output column n, the weights are ONE affine map of integer codes per
//    16-value half: w = s_h[n]·q − mn[n] (s_0 = s_1 except Q6_K, whose 16-value sub-blocks carry their
//    own scales). So the tensor cores accumulate the exact code products into p_h, and the tile folds
//    them in f32:  acc += s_0·p_0 + s_1·p_1 − mn·Σ_k a  (the last term only for Q4_K / Q5_K's mins,
//    with the activation row-sum over the tile). A is held twice, hi and lo of the split, and every
//    fragment gets two mma. Shared rows are padded to 40 halves (80 B = 20 words): the fragment reads
//    of rows g = 0..7 then start on banks 0,20,8,28,16,4,24,12 — conflict-free, where 64 B is 4-way. ──
#define BM 64u
#define BN 64u
#define BK 32u
#define LDS 40u
__device__ __forceinline__ void mma16816(float (&c)[4], unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
}
template <int F>
__device__ __forceinline__ void gemm_t(const float* __restrict__ A, unsigned lda,
                                       const unsigned* __restrict__ codes, const unsigned* __restrict__ aux,
                                       float* __restrict__ C, unsigned ldc, unsigned M, unsigned N, unsigned K,
                                       int* __restrict__ ovf) {
    constexpr bool TWO = F == 2;            // two scales per 32-wide tile (Q6_K's 16-value sub-blocks)
    constexpr bool MIN = F == 0 || F == 1;  // an affine min (Q4_K / Q5_K)
    __shared__ __align__(16) unsigned short As[BM * LDS];      // activation, f16 hi part
    __shared__ __align__(16) unsigned short Al[BM * LDS];      // activation, f16 lo part (a - hi)
    __shared__ __align__(16) unsigned short Ws[BN * LDS];      // integer codes, exact in f16
    __shared__ float Sc[2][BN], Mn[BN], Rs[BM];                // per-tile scales / mins / A row-sums
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const unsigned wm = warp >> 1u, wn = warp & 1u;
    const unsigned m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const unsigned g = lane >> 2u, t4 = lane & 3u;
    float acc[2][4][4], p0[2][4][4], p1[2][4][4];
    #pragma unroll
    for (unsigned i = 0u; i < 2u; ++i)
        #pragma unroll
        for (unsigned j = 0u; j < 4u; ++j)
            #pragma unroll
            for (unsigned c = 0u; c < 4u; ++c) acc[i][j][c] = 0.f;
    bool bad = false;
    const unsigned lr = tid >> 1u, lh = tid & 1u;    // the loaders: row lr of each tile, half lh of its 32 values
    for (unsigned k0 = 0u; k0 < K; k0 += BK) {
        unsigned a[8], al[8], w[8];
        float rsum = 0.f;
        if (m0 + lr < M) {
            const float* src = A + (size_t)(m0 + lr) * lda + k0 + 16u * lh;
            #pragma unroll
            for (unsigned j = 0u; j < 4u; ++j) {
                const float4 v = *reinterpret_cast<const float4*>(src + 4u * j);
                bad |= fabsf(v.x) > 65504.f || fabsf(v.y) > 65504.f || fabsf(v.z) > 65504.f || fabsf(v.w) > 65504.f;
                a[2u * j] = pack2(v.x, v.y); a[2u * j + 1u] = pack2(v.z, v.w);
                al[2u * j] = resid2(v.x, v.y, a[2u * j]); al[2u * j + 1u] = resid2(v.z, v.w, a[2u * j + 1u]);
                if (MIN) rsum += (v.x + v.y) + (v.z + v.w);
            }
        } else {
            #pragma unroll
            for (unsigned j = 0u; j < 8u; ++j) { a[j] = 0u; al[j] = 0u; }
        }
        float sc = 0.f, mn = 0.f;
        if (n0 + lr < N) codes16<F>(codes, aux, n0 + lr, K, k0 / 32u, lh, w, sc, mn);
        else {
            #pragma unroll
            for (unsigned j = 0u; j < 8u; ++j) w[j] = 0u;
        }
        if (MIN) rsum += __shfl_xor_sync(0xffffffffu, rsum, 1);   // the row's other 16 values (lane lr*2+1)
        uint4* as = reinterpret_cast<uint4*>(As + lr * LDS + 16u * lh);
        uint4* als = reinterpret_cast<uint4*>(Al + lr * LDS + 16u * lh);
        uint4* ws = reinterpret_cast<uint4*>(Ws + lr * LDS + 16u * lh);
        as[0] = make_uint4(a[0], a[1], a[2], a[3]); as[1] = make_uint4(a[4], a[5], a[6], a[7]);
        als[0] = make_uint4(al[0], al[1], al[2], al[3]); als[1] = make_uint4(al[4], al[5], al[6], al[7]);
        ws[0] = make_uint4(w[0], w[1], w[2], w[3]); ws[1] = make_uint4(w[4], w[5], w[6], w[7]);
        Sc[lh][lr] = sc;
        if (lh == 0u) { Mn[lr] = mn; Rs[lr] = rsum; }
        __syncthreads();
        #pragma unroll
        for (unsigned i = 0u; i < 2u; ++i)
            #pragma unroll
            for (unsigned j = 0u; j < 4u; ++j)
                #pragma unroll
                for (unsigned c = 0u; c < 4u; ++c) { p0[i][j][c] = 0.f; p1[i][j][c] = 0.f; }
        #pragma unroll
        for (unsigned ks = 0u; ks < BK; ks += 8u) {
            unsigned af[2][2], lf[2][2], bf[4];
            #pragma unroll
            for (unsigned mi = 0u; mi < 2u; ++mi) {
                const unsigned r = wm * 32u + mi * 16u + g;
                af[mi][0] = *reinterpret_cast<const unsigned*>(As + r * LDS + ks + 2u * t4);
                af[mi][1] = *reinterpret_cast<const unsigned*>(As + (r + 8u) * LDS + ks + 2u * t4);
                lf[mi][0] = *reinterpret_cast<const unsigned*>(Al + r * LDS + ks + 2u * t4);
                lf[mi][1] = *reinterpret_cast<const unsigned*>(Al + (r + 8u) * LDS + ks + 2u * t4);
            }
            #pragma unroll
            for (unsigned ni = 0u; ni < 4u; ++ni)
                bf[ni] = *reinterpret_cast<const unsigned*>(Ws + (wn * 32u + ni * 8u + g) * LDS + ks + 2u * t4);
            #pragma unroll
            for (unsigned mi = 0u; mi < 2u; ++mi)
                #pragma unroll
                for (unsigned ni = 0u; ni < 4u; ++ni) {
                    // the 16-value half this k-step is in: its own partial when the format has two scales
                    float (&p)[4] = (TWO && ks >= 16u) ? p1[mi][ni] : p0[mi][ni];
                    // lo first: the small term accumulates before the large one lands on it.
                    mma16816(p, lf[mi][0], lf[mi][1], bf[ni]);
                    mma16816(p, af[mi][0], af[mi][1], bf[ni]);
                }
        }
        // Fold the tile: acc += s0·p0 (+ s1·p1) (− mn·Σa), per column n and row m of the fragment.
        #pragma unroll
        for (unsigned mi = 0u; mi < 2u; ++mi)
            #pragma unroll
            for (unsigned ni = 0u; ni < 4u; ++ni)
                #pragma unroll
                for (unsigned c = 0u; c < 4u; ++c) {
                    const unsigned col = wn * 32u + ni * 8u + 2u * t4 + (c & 1u), row = wm * 32u + mi * 16u + g + 8u * (c >> 1u);
                    float v = Sc[0][col] * p0[mi][ni][c];
                    if (TWO) v += Sc[1][col] * p1[mi][ni][c];
                    if (MIN) v -= Mn[col] * Rs[row];
                    acc[mi][ni][c] += v;
                }
        __syncthreads();
    }
    if (bad) atomicOr(ovf, 1);
    #pragma unroll
    for (unsigned mi = 0u; mi < 2u; ++mi)
        #pragma unroll
        for (unsigned ni = 0u; ni < 4u; ++ni) {
            const unsigned r = m0 + wm * 32u + mi * 16u + g, c = n0 + wn * 32u + ni * 8u + 2u * t4;
            if (r < M) { if (c < N) C[(size_t)r * ldc + c] = acc[mi][ni][0]; if (c + 1u < N) C[(size_t)r * ldc + c + 1u] = acc[mi][ni][1]; }
            if (r + 8u < M) { if (c < N) C[(size_t)(r + 8u) * ldc + c] = acc[mi][ni][2]; if (c + 1u < N) C[(size_t)(r + 8u) * ldc + c + 1u] = acc[mi][ni][3]; }
        }
}
// ⚠ Spelled out, not macro-generated: tests/ptx_artifact.rs parses `extern "C" __global__ void NAME(`.
#define GEMM_ARGS const float* __restrict__ A, unsigned lda, const unsigned* __restrict__ codes, \
                  const unsigned* __restrict__ aux, float* __restrict__ C, unsigned ldc, unsigned M, unsigned N, \
                  unsigned K, int* __restrict__ ovf
extern "C" __global__ void q4k_gemm(GEMM_ARGS)  { gemm_t<0>(A, lda, codes, aux, C, ldc, M, N, K, ovf); }
extern "C" __global__ void q5k_gemm(GEMM_ARGS)  { gemm_t<1>(A, lda, codes, aux, C, ldc, M, N, K, ovf); }
extern "C" __global__ void q6k_gemm(GEMM_ARGS)  { gemm_t<2>(A, lda, codes, aux, C, ldc, M, N, K, ovf); }
extern "C" __global__ void q8_0_gemm(GEMM_ARGS) { gemm_t<3>(A, lda, codes, aux, C, ldc, M, N, K, ovf); }
extern "C" __global__ void q5_0_gemm(GEMM_ARGS) { gemm_t<4>(A, lda, codes, aux, C, ldc, M, N, K, ovf); }
#undef GEMM_ARGS

// ═══════════ GEMM v2: the same arithmetic, fed the way the tensor cores want it ═══════════
// v1 above holds ONE shared tile, loads it, waits, multiplies, waits: the global-load latency and the
// decode/split ALU work sit in series with the tensor cores, every fragment is gathered with 32-bit
// shared loads (12 per k8 step per warp), and the tile fold re-reads its scales from shared memory for
// every accumulator. Measured on the RTX 4050 (examples/cuda_gemm_bench.rs) it ran at the numbers in
// the commit that added this. v2 keeps EVERY rounding decision of v1 (integer codes exact in f16, the
// activation split hi + lo, per-32-K scale and min folded in f32 — see the header) and changes only
// the feeding:
//  • tile 64 (rows of A) x 128 (weight rows) x 32, 256 threads = 2 x 4 warps of 32 x 32 — twice v1's
//    columns per block, so each converted activation feeds twice the multiplies;
//  • TWO shared stages: tile k+1's global loads are issued into registers before tile k's multiplies
//    and converted into the other stage after them — one barrier per K tile, and the load latency
//    hidden behind the tensor work;
//  • fragments by `ldmatrix.x4` (sm_75): A hi, A lo and four B tiles in three instructions per k8 step;
//  • the fold reads its 8 column scales and 4 row sums per thread ONCE per tile, and Q6_K folds after
//    each 16-wide half instead of carrying a second partial — 32 fewer live registers;
//  • f16 -> f32 by the hardware `cvt` (exact, as the bit-twiddling f16_to_f32 is) in the split;
//  • ⭐ each row's 32 activations of a K tile are scaled by a POWER OF TWO that puts their largest near
//    2^14 before the split, and the fold multiplies it back out (exact both ways). ⛔ Without it the lo
//    half of every activation below 2^-3 falls into f16's SUBNORMAL range and keeps only an absolute
//    2^-25: on Qwen2.5-0.5B Q4_K_M under the gate's wrong-rope-pairing control (a model whose hidden
//    states run small) the native prefill landed 1.57e-2 logits from WGSL while WGSL's own prefill and
//    decode schedules agreed to 1.8e-4 and native decode to 8e-5. Scaling by 64 alone took it to
//    3.9e-3; the per-row-tile exponent is the general form. It also makes an f16 OVERFLOW impossible
//    (the flag now only catches inf/NaN inputs).
// Still compute_75 PTX: no cp.async / m16n8k16 (sm_80), so the tier's "the driver alone loads it on
// Turing and newer" holds.
#define B2M 64u
#define B2N 128u
__device__ __forceinline__ float h2f(unsigned h) {
    float f; asm("{ .reg .f16 t; mov.b16 t, %1; cvt.f32.f16 %0, t; }" : "=f"(f) : "h"((unsigned short)h)); return f;
}
__device__ __forceinline__ unsigned resid2h(float x, float y, unsigned hi) {
    return pack2(x - h2f(hi & 0xffffu), y - h2f(hi >> 16u));
}
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
// The raw words of 16 weights (one half of a 32-wide K block of row n), fetched from global memory
// ahead of their decode — the register half of the double buffer.
struct Raw16 { uint4 q, qh, ax; };
template <int F>
__device__ __forceinline__ void fetch16(const unsigned* __restrict__ codes, const unsigned* __restrict__ aux,
                                        unsigned n, unsigned K, unsigned kc, unsigned h, Raw16& r) {
    r.qh = make_uint4(0u, 0u, 0u, 0u); r.ax = make_uint4(0u, 0u, 0u, 0u);
    if (F == 0 || F == 1) {
        const unsigned nblk = K / 256u, b = kc >> 3u, s = kc & 7u, bi = n * nblk + b, c = s >> 1u;
        const unsigned cb = bi * (F == 0 ? 32u : 40u);
        r.q = *reinterpret_cast<const uint4*>(codes + cb + 8u * c + 4u * h);
        if (F == 1) r.qh = *reinterpret_cast<const uint4*>(codes + cb + 32u + 4u * h);
        r.ax = *reinterpret_cast<const uint4*>(aux + bi * 4u);                  // d|dmin, 12 scale bytes
    } else if (F == 2) {
        const unsigned nblk = K / 256u, b = kc >> 3u, j = kc & 7u, bi = n * nblk + b;
        const unsigned hf = j >> 2u, qq = j & 3u, cb = bi * 48u, ab = bi * 5u;
        r.q = *reinterpret_cast<const uint4*>(codes + cb + (64u * hf + 32u * (qq & 1u) + 16u * h) / 4u);
        r.qh = *reinterpret_cast<const uint4*>(codes + cb + 32u + (32u * hf + 16u * h) / 4u);
        const unsigned si = 8u * hf + 2u * qq + h;
        r.ax.x = aux[ab]; r.ax.y = aux[ab + 1u + (si >> 2u)];
    } else if (F == 3) {
        const unsigned bi = n * (K / 32u) + kc;
        r.q = *reinterpret_cast<const uint4*>(codes + bi * 8u + 4u * h);
        r.ax.x = aux[bi >> 1u];
    } else {
        const unsigned bi = n * (K / 32u) + kc;
        r.q = *reinterpret_cast<const uint4*>(codes + bi * 4u);
        r.ax.x = aux[bi * 2u]; r.ax.y = aux[bi * 2u + 1u];
    }
}
// The decode half: codes16's arithmetic on fetched words. Same codes, same scale, same min.
template <int F>
__device__ __forceinline__ void decode16(const Raw16& r, unsigned n, unsigned K, unsigned kc, unsigned h,
                                         unsigned (&o)[8], float& scale, float& mn) {
    float v[16];
    mn = 0.f;
    const unsigned qw[4] = {r.q.x, r.q.y, r.q.z, r.q.w}, hw[4] = {r.qh.x, r.qh.y, r.qh.z, r.qh.w};
    if (F == 0 || F == 1) {
        const unsigned s = kc & 7u, sh = 4u * (s & 1u);
        const unsigned axw[4] = {r.ax.x, r.ax.y, r.ax.z, r.ax.w};
        #define SCB(i) ((axw[1u + ((i) >> 2u)] >> (8u * ((i) & 3u))) & 0xffu)
        unsigned sc, m6;
        if (s < 4u) { sc = SCB(s) & 63u; m6 = SCB(s + 4u) & 63u; }
        else { const unsigned a = SCB(s + 4u), lo = SCB(s - 4u), hi = SCB(s);
               sc = (a & 0x0Fu) | ((lo >> 6u) << 4u); m6 = (a >> 4u) | ((hi >> 6u) << 4u); }
        #undef SCB
        scale = f16_to_f32(axw[0] & 0xffffu) * (float)sc; mn = f16_to_f32(axw[0] >> 16u) * (float)m6;
        #pragma unroll
        for (unsigned w = 0u; w < 4u; ++w)
            #pragma unroll
            for (unsigned k = 0u; k < 4u; ++k) {
                unsigned qq = (qw[w] >> (8u * k + sh)) & 0xfu;
                if (F == 1) qq |= ((hw[w] >> (8u * k + s)) & 1u) << 4u;
                v[4u * w + k] = (float)qq;
            }
    } else if (F == 2) {
        const unsigned j = kc & 7u, hf = j >> 2u, qq = j & 3u, si = 8u * hf + 2u * qq + h;
        scale = f16_to_f32(r.ax.x & 0xffffu) * (float)((int)(((r.ax.y >> (8u * (si & 3u))) & 0xffu) << 24u) >> 24);
        const unsigned nsh = 4u * (qq >> 1u), hsh = 2u * qq;
        #pragma unroll
        for (unsigned w = 0u; w < 4u; ++w)
            #pragma unroll
            for (unsigned k = 0u; k < 4u; ++k)
                v[4u * w + k] = (float)((int)(((qw[w] >> (8u * k + nsh)) & 0xfu) | (((hw[w] >> (8u * k + hsh)) & 3u) << 4u)) - 32);
    } else if (F == 3) {
        const unsigned bi = n * (K / 32u) + kc;
        scale = f16_to_f32((bi & 1u) ? (r.ax.x >> 16u) : (r.ax.x & 0xffffu));
        #pragma unroll
        for (unsigned w = 0u; w < 4u; ++w)
            #pragma unroll
            for (unsigned k = 0u; k < 4u; ++k) v[4u * w + k] = (float)((int)(qw[w] << (24u - 8u * k)) >> 24);
    } else {
        const unsigned qh = r.ax.x;
        scale = f16_to_f32(r.ax.y & 0xffffu);
        #pragma unroll
        for (unsigned w = 0u; w < 4u; ++w)
            #pragma unroll
            for (unsigned k = 0u; k < 4u; ++k) {
                const unsigned e = 4u * w + k;
                v[e] = (float)((int)(((qw[w] >> (8u * k + 4u * h)) & 0xfu) | (((qh >> (e + 16u * h)) & 1u) << 4u)) - 16);
            }
    }
    #pragma unroll
    for (unsigned i = 0u; i < 8u; ++i) o[i] = pack2(v[2u * i], v[2u * i + 1u]);
}
struct __align__(16) Stage2 {
    unsigned short As[B2M * LDS], Al[B2M * LDS], Ws[B2N * LDS];
    float Sc[2][B2N], Mn[B2N], Rs[B2M], Ri[B2M];     // Ri: 2^-e, the row's activation scale undone
};
template <int F>
__device__ __forceinline__ void gemm2_t(const float* __restrict__ A, unsigned lda,
                                        const unsigned* __restrict__ codes, const unsigned* __restrict__ aux,
                                        float* __restrict__ C, unsigned ldc, unsigned M, unsigned N, unsigned K,
                                        int* __restrict__ ovf) {
    constexpr bool TWO = F == 2;
    constexpr bool MIN = F == 0 || F == 1;
    __shared__ Stage2 st[2];
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const unsigned wm = warp >> 2u, wn = warp & 3u;                  // 2 x 4 warps of 32 x 32
    const unsigned m0 = blockIdx.y * B2M, n0 = blockIdx.x * B2N;
    const unsigned g = lane >> 2u, t4 = lane & 3u;
    // loaders: A row ar, eighth-of-32 aq (8 floats); W row wr, half wh (16 codes)
    const unsigned ar = tid >> 2u, aq = tid & 3u, wr = tid >> 1u, wh = tid & 1u;
    float acc[2][4][4], p[2][4][4];
    #pragma unroll
    for (unsigned i = 0u; i < 2u; ++i)
        #pragma unroll
        for (unsigned j = 0u; j < 4u; ++j)
            #pragma unroll
            for (unsigned c = 0u; c < 4u; ++c) acc[i][j][c] = 0.f;
    bool bad = false;
    float4 ra0, ra1; Raw16 rw;
    auto fetch = [&](unsigned k0) {
        if (m0 + ar < M) {
            const float* src = A + (size_t)(m0 + ar) * lda + k0 + 8u * aq;
            ra0 = *reinterpret_cast<const float4*>(src); ra1 = *reinterpret_cast<const float4*>(src + 4u);
        } else { ra0 = make_float4(0.f, 0.f, 0.f, 0.f); ra1 = ra0; }
        if (n0 + wr < N) fetch16<F>(codes, aux, n0 + wr, K, k0 / 32u, wh, rw);
    };
    auto store = [&](Stage2& s, unsigned k0) {
        unsigned a[4], al[4];
        const float x[8] = {ra0.x, ra0.y, ra0.z, ra0.w, ra1.x, ra1.y, ra1.z, ra1.w};
        // The row's power-of-two scale for this tile: its largest |a| over the 32 values (the 4 loader
        // lanes of the row) lands in [2^14, 2^15).
        float amax = 0.f;
        #pragma unroll
        for (unsigned j = 0u; j < 8u; ++j) amax = fmaxf(amax, fabsf(x[j]));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        bad |= !(amax <= 3.0e38f);                                         // inf / NaN
        int e = amax > 0.f ? 14 - ((int)((__float_as_uint(amax) >> 23u) & 0xffu) - 127) : 0;
        e = max(-100, min(100, e));
        const float up = __uint_as_float((unsigned)(127 + e) << 23u), dn = __uint_as_float((unsigned)(127 - e) << 23u);
        float rsum = 0.f;
        #pragma unroll
        for (unsigned j = 0u; j < 4u; ++j) {
            const float u0 = x[2u * j] * up, u1 = x[2u * j + 1u] * up;       // exact: a power of two
            a[j] = pack2(u0, u1);
            al[j] = resid2h(u0, u1, a[j]);
        }
        if (MIN) {
            rsum = ((x[0] + x[1]) + (x[2] + x[3])) + ((x[4] + x[5]) + (x[6] + x[7]));
            rsum += __shfl_xor_sync(0xffffffffu, rsum, 1);
            rsum += __shfl_xor_sync(0xffffffffu, rsum, 2);
        }
        *reinterpret_cast<uint4*>(s.As + ar * LDS + 8u * aq) = make_uint4(a[0], a[1], a[2], a[3]);
        *reinterpret_cast<uint4*>(s.Al + ar * LDS + 8u * aq) = make_uint4(al[0], al[1], al[2], al[3]);
        if (aq == 0u) { s.Rs[ar] = rsum; s.Ri[ar] = dn; }
        unsigned w[8]; float sc = 0.f, mn = 0.f;
        if (n0 + wr < N) decode16<F>(rw, n0 + wr, K, k0 / 32u, wh, w, sc, mn);
        else {
            #pragma unroll
            for (unsigned j = 0u; j < 8u; ++j) w[j] = 0u;
        }
        uint4* ws = reinterpret_cast<uint4*>(s.Ws + wr * LDS + 16u * wh);
        ws[0] = make_uint4(w[0], w[1], w[2], w[3]); ws[1] = make_uint4(w[4], w[5], w[6], w[7]);
        s.Sc[wh][wr] = sc;
        if (wh == 0u) s.Mn[wr] = mn;
    };
    fetch(0u); store(st[0], 0u);
    __syncthreads();
    const unsigned nk = K / BK;
    for (unsigned kt = 0u; kt < nk; ++kt) {
        const Stage2& s = st[kt & 1u];
        if (kt + 1u < nk) fetch((kt + 1u) * BK);
        // this thread's fold operands, read once per tile
        float2 sc0[4], sc1[4], mnv[4]; float rs[2][2], ri[2][2];
        #pragma unroll
        for (unsigned ni = 0u; ni < 4u; ++ni) {
            const unsigned col = wn * 32u + ni * 8u + 2u * t4;
            sc0[ni] = *reinterpret_cast<const float2*>(&s.Sc[0][col]);
            if (TWO) sc1[ni] = *reinterpret_cast<const float2*>(&s.Sc[1][col]);
            if (MIN) mnv[ni] = *reinterpret_cast<const float2*>(&s.Mn[col]);
        }
        #pragma unroll
        for (unsigned mi = 0u; mi < 2u; ++mi) {
            const unsigned row = wm * 32u + mi * 16u + g;
            rs[mi][0] = MIN ? s.Rs[row] : 0.f; rs[mi][1] = MIN ? s.Rs[row + 8u] : 0.f;
            ri[mi][0] = s.Ri[row]; ri[mi][1] = s.Ri[row + 8u];
        }
        auto zero_p = [&]() {
            #pragma unroll
            for (unsigned i = 0u; i < 2u; ++i)
                #pragma unroll
                for (unsigned j = 0u; j < 4u; ++j)
                    #pragma unroll
                    for (unsigned c = 0u; c < 4u; ++c) p[i][j][c] = 0.f;
        };
        auto fold = [&](const float2 (&scv)[4], bool with_min) {
            #pragma unroll
            for (unsigned mi = 0u; mi < 2u; ++mi)
                #pragma unroll
                for (unsigned ni = 0u; ni < 4u; ++ni)
                    #pragma unroll
                    for (unsigned c = 0u; c < 4u; ++c) {
                        float v = ((c & 1u) ? scv[ni].y : scv[ni].x) * p[mi][ni][c] * ri[mi][c >> 1u];
                        if (MIN && with_min) v -= ((c & 1u) ? mnv[ni].y : mnv[ni].x) * rs[mi][c >> 1u];
                        acc[mi][ni][c] += v;
                    }
        };
        zero_p();
        #pragma unroll
        for (unsigned ks = 0u; ks < BK; ks += 8u) {
            unsigned ah[4], alo[4], bf[4];
            // A: matrices (mi 0 rows 0-7, mi 0 rows 8-15, mi 1 rows 0-7, mi 1 rows 8-15); lane -> row lane&7 of matrix lane>>3
            const unsigned arow = wm * 32u + (lane >> 4u) * 16u + ((lane >> 3u) & 1u) * 8u + (lane & 7u);
            ldsm_x4(ah, s.As + arow * LDS + ks);
            ldsm_x4(alo, s.Al + arow * LDS + ks);
            ldsm_x4(bf, s.Ws + (wn * 32u + (lane >> 3u) * 8u + (lane & 7u)) * LDS + ks);
            #pragma unroll
            for (unsigned mi = 0u; mi < 2u; ++mi)
                #pragma unroll
                for (unsigned ni = 0u; ni < 4u; ++ni) {
                    mma16816(p[mi][ni], alo[2u * mi], alo[2u * mi + 1u], bf[ni]);   // lo first, as v1
                    mma16816(p[mi][ni], ah[2u * mi], ah[2u * mi + 1u], bf[ni]);
                }
            if (TWO && ks == 8u) { fold(sc0, false); zero_p(); }
        }
        if (TWO) fold(sc1, false); else fold(sc0, true);
        if (kt + 1u < nk) store(st[(kt + 1u) & 1u], (kt + 1u) * BK);
        __syncthreads();
    }
    if (bad) atomicOr(ovf, 1);
    #pragma unroll
    for (unsigned mi = 0u; mi < 2u; ++mi)
        #pragma unroll
        for (unsigned ni = 0u; ni < 4u; ++ni) {
            const unsigned r = m0 + wm * 32u + mi * 16u + g, c = n0 + wn * 32u + ni * 8u + 2u * t4;
            if (r < M) { if (c < N) C[(size_t)r * ldc + c] = acc[mi][ni][0]; if (c + 1u < N) C[(size_t)r * ldc + c + 1u] = acc[mi][ni][1]; }
            if (r + 8u < M) { if (c < N) C[(size_t)(r + 8u) * ldc + c] = acc[mi][ni][2]; if (c + 1u < N) C[(size_t)(r + 8u) * ldc + c + 1u] = acc[mi][ni][3]; }
        }
}
#define GEMM_ARGS const float* __restrict__ A, unsigned lda, const unsigned* __restrict__ codes, \
                  const unsigned* __restrict__ aux, float* __restrict__ C, unsigned ldc, unsigned M, unsigned N, \
                  unsigned K, int* __restrict__ ovf
extern "C" __global__ void q4k_gemm2(GEMM_ARGS)  { gemm2_t<0>(A, lda, codes, aux, C, ldc, M, N, K, ovf); }
extern "C" __global__ void q5k_gemm2(GEMM_ARGS)  { gemm2_t<1>(A, lda, codes, aux, C, ldc, M, N, K, ovf); }
extern "C" __global__ void q6k_gemm2(GEMM_ARGS)  { gemm2_t<2>(A, lda, codes, aux, C, ldc, M, N, K, ovf); }
extern "C" __global__ void q8_0_gemm2(GEMM_ARGS) { gemm2_t<3>(A, lda, codes, aux, C, ldc, M, N, K, ovf); }
extern "C" __global__ void q5_0_gemm2(GEMM_ARGS) { gemm2_t<4>(A, lda, codes, aux, C, ldc, M, N, K, ovf); }
#undef GEMM_ARGS

// ═══════════ GEMM v3: INTEGER tensor cores — fixed-point activations in three int8 digits ═══════════
// Measured on the RTX 4050 (a 20-line mma loop): f16 x f16 -> f32 `mma` peaks at 16.4 TFLOPS, s8 x s8 -> s32
// at 65.7 TOPS — 4x — and the sm_75 `mma.m8n8k16.s8` reaches it (65.5), so this stays compute_75 PTX.
// v1/v2 spend TWO f16 mma per product (the hi and lo halves) and already run the tensor pipe at 60-85%
// of that 16.4 (examples/cuda_gemm_bench.rs), so no feeding change can buy much; the instruction must.
//
// ⚠ NUMERICS (what differs from v2): each row's 32 activations of a K tile become ONE 22-bit signed
// integer each, q = round(a · 2^e), with e the power of two that puts the row-tile's largest |a| in
// [2^21, 2^22): an absolute error <= 2^-22 of that largest value — the same order as the f32 rounding
// of the sum it is about to join, and 2^-22 RELATIVE for values near the maximum (v2's hi+lo: 2^-22
// relative for every value). q is split EXACTLY into three balanced digits q = 2^16 q2 + 2^8 q1 + q0,
// each in [-128, 127]; the weights enter as their exact integer codes (every format's fit int8). Every
// product and every sum inside a K tile is then an INTEGER — exact, no rounding, no tensor-core
// truncation — and the tile folds into f32 once: acc += scale_n · 2^-e_m · (2^16 P2 + (2^8 P1 + P0))
// (− min_n · Σa for Q4_K/Q5_K, the row sum taken in f32 as before). An f16 overflow cannot happen
// (the exponent absorbs any magnitude); the flag catches inf/NaN only.
// Cost per product: 3 int8 MACs at 4x the rate = 3/8 of v2's two f16 MACs. `__launch_bounds__(256, 2)`:
// two blocks per SM (<= 128 registers), so one block's conversions overlap the other's multiplies.
#define LDB 48u        // bytes per shared row: 32 codes + 16 pad — ldmatrix rows land on 8 distinct bank quads
struct __align__(16) Stage3 {
    unsigned char Aq[3][B2M * LDB], Wq[B2N * LDB];
    float Sc[2][B2N], Mn[B2N], Rs[B2M], Ri[B2M];
};
__device__ __forceinline__ void mma8816(int (&c)[2], unsigned a, unsigned b) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(c[0]), "+r"(c[1]) : "r"(a), "r"(b));
}
// Four small integers (each in [-128, 127]) as the four bytes of a word.
__device__ __forceinline__ unsigned pack4s8(int a, int b, int c, int d) {
    return (unsigned)(a & 0xff) | ((unsigned)(b & 0xff) << 8u) | ((unsigned)(c & 0xff) << 16u) | ((unsigned)(d & 0xff) << 24u);
}
template <int F>
__device__ __forceinline__ void decode16_s8(const Raw16& r, unsigned n, unsigned K, unsigned kc, unsigned h,
                                            unsigned (&o)[4], float& scale, float& mn) {
    // The codes decode16 makes as f16, here as bytes. ⚠ Q5_K's high bit and Q6_K's sign are the same
    // expressions; only the packing differs.
    int v[16];
    mn = 0.f;
    const unsigned qw[4] = {r.q.x, r.q.y, r.q.z, r.q.w}, hw[4] = {r.qh.x, r.qh.y, r.qh.z, r.qh.w};
    if (F == 0 || F == 1) {
        const unsigned s = kc & 7u, sh = 4u * (s & 1u);
        const unsigned axw[4] = {r.ax.x, r.ax.y, r.ax.z, r.ax.w};
        #define SCB(i) ((axw[1u + ((i) >> 2u)] >> (8u * ((i) & 3u))) & 0xffu)
        unsigned sc, m6;
        if (s < 4u) { sc = SCB(s) & 63u; m6 = SCB(s + 4u) & 63u; }
        else { const unsigned a = SCB(s + 4u), lo = SCB(s - 4u), hi = SCB(s);
               sc = (a & 0x0Fu) | ((lo >> 6u) << 4u); m6 = (a >> 4u) | ((hi >> 6u) << 4u); }
        #undef SCB
        scale = h2f(axw[0] & 0xffffu) * (float)sc; mn = h2f(axw[0] >> 16u) * (float)m6;
        #pragma unroll
        for (unsigned w = 0u; w < 4u; ++w)
            #pragma unroll
            for (unsigned k = 0u; k < 4u; ++k) {
                unsigned qq = (qw[w] >> (8u * k + sh)) & 0xfu;
                if (F == 1) qq |= ((hw[w] >> (8u * k + s)) & 1u) << 4u;
                v[4u * w + k] = (int)qq;
            }
    } else if (F == 2) {
        const unsigned j = kc & 7u, hf = j >> 2u, qq = j & 3u, si = 8u * hf + 2u * qq + h;
        scale = h2f(r.ax.x & 0xffffu) * (float)((int)(((r.ax.y >> (8u * (si & 3u))) & 0xffu) << 24u) >> 24);
        const unsigned nsh = 4u * (qq >> 1u), hsh = 2u * qq;
        #pragma unroll
        for (unsigned w = 0u; w < 4u; ++w)
            #pragma unroll
            for (unsigned k = 0u; k < 4u; ++k)
                v[4u * w + k] = (int)(((qw[w] >> (8u * k + nsh)) & 0xfu) | (((hw[w] >> (8u * k + hsh)) & 3u) << 4u)) - 32;
    } else if (F == 3) {
        const unsigned bi = n * (K / 32u) + kc;
        scale = h2f((bi & 1u) ? (r.ax.x >> 16u) : (r.ax.x & 0xffffu));
        #pragma unroll
        for (unsigned w = 0u; w < 4u; ++w)
            #pragma unroll
            for (unsigned k = 0u; k < 4u; ++k) v[4u * w + k] = (int)(qw[w] << (24u - 8u * k)) >> 24;
    } else {
        const unsigned qh = r.ax.x;
        scale = h2f(r.ax.y & 0xffffu);
        #pragma unroll
        for (unsigned w = 0u; w < 4u; ++w)
            #pragma unroll
            for (unsigned k = 0u; k < 4u; ++k) {
                const unsigned e = 4u * w + k;
                v[e] = (int)(((qw[w] >> (8u * k + 4u * h)) & 0xfu) | (((qh >> (e + 16u * h)) & 1u) << 4u)) - 16;
            }
    }
    #pragma unroll
    for (unsigned i = 0u; i < 4u; ++i) o[i] = pack4s8(v[4u * i], v[4u * i + 1u], v[4u * i + 2u], v[4u * i + 3u]);
}
template <int F>
__device__ __forceinline__ void gemm3_t(const float* __restrict__ A, unsigned lda,
                                        const unsigned* __restrict__ codes, const unsigned* __restrict__ aux,
                                        float* __restrict__ C, unsigned ldc, unsigned M, unsigned N, unsigned K,
                                        int* __restrict__ ovf) {
    constexpr bool TWO = F == 2;
    constexpr bool MIN = F == 0 || F == 1;
    __shared__ Stage3 st[2];
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const unsigned wm = warp >> 2u, wn = warp & 3u;                  // 2 x 4 warps of 32 x 32
    const unsigned m0 = blockIdx.y * B2M, n0 = blockIdx.x * B2N;
    const unsigned g = lane >> 2u, t4 = lane & 3u;
    const unsigned ar = tid >> 2u, aq = tid & 3u, wr = tid >> 1u, wh = tid & 1u;
    float acc[4][4][2];                                              // [m8 tile][n8 tile][2]
    #pragma unroll
    for (unsigned i = 0u; i < 4u; ++i)
        #pragma unroll
        for (unsigned j = 0u; j < 4u; ++j) { acc[i][j][0] = 0.f; acc[i][j][1] = 0.f; }
    bool bad = false;
    float4 ra0, ra1; Raw16 rw;
    auto fetch = [&](unsigned k0) {
        if (m0 + ar < M) {
            const float* src = A + (size_t)(m0 + ar) * lda + k0 + 8u * aq;
            ra0 = *reinterpret_cast<const float4*>(src); ra1 = *reinterpret_cast<const float4*>(src + 4u);
        } else { ra0 = make_float4(0.f, 0.f, 0.f, 0.f); ra1 = ra0; }
        if (n0 + wr < N) fetch16<F>(codes, aux, n0 + wr, K, k0 / 32u, wh, rw);
    };
    auto store = [&](Stage3& s, unsigned k0) {
        const float x[8] = {ra0.x, ra0.y, ra0.z, ra0.w, ra1.x, ra1.y, ra1.z, ra1.w};
        float amax = 0.f;
        #pragma unroll
        for (unsigned j = 0u; j < 8u; ++j) amax = fmaxf(amax, fabsf(x[j]));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        bad |= !(amax <= 3.0e38f);                                          // inf / NaN
        // e: largest |a| · 2^e in [2^21, 2^22)
        int e = amax > 0.f ? 21 - ((int)((__float_as_uint(amax) >> 23u) & 0xffu) - 127) : 0;
        e = max(-100, min(100, e));
        const float up = __uint_as_float((unsigned)(127 + e) << 23u), dn = __uint_as_float((unsigned)(127 - e) << 23u);
        int d0[8], d1[8], d2[8];
        #pragma unroll
        for (unsigned j = 0u; j < 8u; ++j) {
            const int q = __float2int_rn(x[j] * up);                       // |q| <= 2^22; the scaling is exact
            const int q0 = ((q + 128) & 255) - 128, r1 = (q - q0) >> 8;      // balanced base-256 digits:
            const int q1 = ((r1 + 128) & 255) - 128, q2 = (r1 - q1) >> 8;    // q = 65536 q2 + 256 q1 + q0
            d0[j] = q0; d1[j] = q1; d2[j] = q2;
        }
        *reinterpret_cast<uint2*>(s.Aq[0] + ar * LDB + 8u * aq) = make_uint2(pack4s8(d0[0], d0[1], d0[2], d0[3]), pack4s8(d0[4], d0[5], d0[6], d0[7]));
        *reinterpret_cast<uint2*>(s.Aq[1] + ar * LDB + 8u * aq) = make_uint2(pack4s8(d1[0], d1[1], d1[2], d1[3]), pack4s8(d1[4], d1[5], d1[6], d1[7]));
        *reinterpret_cast<uint2*>(s.Aq[2] + ar * LDB + 8u * aq) = make_uint2(pack4s8(d2[0], d2[1], d2[2], d2[3]), pack4s8(d2[4], d2[5], d2[6], d2[7]));
        float rsum = 0.f;
        if (MIN) {
            rsum = ((x[0] + x[1]) + (x[2] + x[3])) + ((x[4] + x[5]) + (x[6] + x[7]));
            rsum += __shfl_xor_sync(0xffffffffu, rsum, 1);
            rsum += __shfl_xor_sync(0xffffffffu, rsum, 2);
        }
        if (aq == 0u) { s.Rs[ar] = rsum; s.Ri[ar] = dn; }
        unsigned w[4]; float sc = 0.f, mn = 0.f;
        if (n0 + wr < N) decode16_s8<F>(rw, n0 + wr, K, k0 / 32u, wh, w, sc, mn);
        else { w[0] = w[1] = w[2] = w[3] = 0u; }
        *reinterpret_cast<uint4*>(s.Wq + wr * LDB + 16u * wh) = make_uint4(w[0], w[1], w[2], w[3]);
        s.Sc[wh][wr] = sc;
        if (wh == 0u) s.Mn[wr] = mn;
    };
    fetch(0u); store(st[0], 0u);
    __syncthreads();
    const unsigned nk = K / BK;
    // ldmatrix row addresses: A — matrix mi = lane>>3 is rows [wm·32 + 8·mi, +8); B — matrix ni is rows [wn·32 + 8·ni, +8)
    const unsigned arow = wm * 32u + (lane >> 3u) * 8u + (lane & 7u), brow = wn * 32u + (lane >> 3u) * 8u + (lane & 7u);
    for (unsigned kt = 0u; kt < nk; ++kt) {
        const Stage3& s = st[kt & 1u];
        if (kt + 1u < nk) fetch((kt + 1u) * BK);
        float2 sc[2][4], mnv[4]; float rs[4], ri[4];
        #pragma unroll
        for (unsigned ni = 0u; ni < 4u; ++ni) {
            const unsigned col = wn * 32u + ni * 8u + 2u * t4;
            sc[0][ni] = *reinterpret_cast<const float2*>(&s.Sc[0][col]);
            sc[1][ni] = TWO ? *reinterpret_cast<const float2*>(&s.Sc[1][col]) : sc[0][ni];
            if (MIN) mnv[ni] = *reinterpret_cast<const float2*>(&s.Mn[col]);
        }
        #pragma unroll
        for (unsigned mi = 0u; mi < 4u; ++mi) { const unsigned row = wm * 32u + mi * 8u + g; rs[mi] = MIN ? s.Rs[row] : 0.f; ri[mi] = s.Ri[row]; }
        int chi[4][4][2], clo[4][4][2];                              // 2^16 digit / (2^8 digit, 1 digit)
        auto zero = [&]() {
            #pragma unroll
            for (unsigned i = 0u; i < 4u; ++i)
                #pragma unroll
                for (unsigned j = 0u; j < 4u; ++j) { chi[i][j][0] = chi[i][j][1] = 0; clo[i][j][0] = clo[i][j][1] = 0; }
        };
        auto fold = [&](unsigned half, bool with_min) {
            #pragma unroll
            for (unsigned mi = 0u; mi < 4u; ++mi)
                #pragma unroll
                for (unsigned ni = 0u; ni < 4u; ++ni)
                    #pragma unroll
                    for (unsigned c = 0u; c < 2u; ++c) {
                        const float pv = (float)chi[mi][ni][c] * 65536.f + (float)clo[mi][ni][c];
                        float v = (c ? sc[half][ni].y : sc[half][ni].x) * pv * ri[mi];
                        if (MIN && with_min) v -= (c ? mnv[ni].y : mnv[ni].x) * rs[mi];
                        acc[mi][ni][c] += v;
                    }
        };
        auto mma_all = [&](int (&cc)[4][4][2], const unsigned (&a)[4], const unsigned (&bf)[4]) {
            #pragma unroll
            for (unsigned mi = 0u; mi < 4u; ++mi)
                #pragma unroll
                for (unsigned ni = 0u; ni < 4u; ++ni) mma8816(cc[mi][ni], a[mi], bf[ni]);
        };
        auto times256 = [&]() {
            #pragma unroll
            for (unsigned mi = 0u; mi < 4u; ++mi)
                #pragma unroll
                for (unsigned ni = 0u; ni < 4u; ++ni) { clo[mi][ni][0] *= 256; clo[mi][ni][1] *= 256; }
        };
        zero();
        if (TWO) {
            // Q6_K: each 16-wide half has its own scale — fold per half.
            #pragma unroll
            for (unsigned kh = 0u; kh < 2u; ++kh) {
                unsigned bf[4], ad[4];
                ldsm_x4(bf, s.Wq + brow * LDB + 16u * kh);
                ldsm_x4(ad, s.Aq[2] + arow * LDB + 16u * kh); mma_all(chi, ad, bf);
                ldsm_x4(ad, s.Aq[1] + arow * LDB + 16u * kh); mma_all(clo, ad, bf);
                times256();                                             // clo = 256·Σq1·w, exact (|.| < 2^27)
                ldsm_x4(ad, s.Aq[0] + arow * LDB + 16u * kh); mma_all(clo, ad, bf);
                fold(kh, false); zero();
            }
        } else {
            // One scale per 32: the 2^16 digit into chi, the 2^8 digit into clo over BOTH halves, then
            // clo·256 and the unit digit — every sum an exact integer (|chi| < 2^19, |clo| < 2^28).
            unsigned bf[2][4], ad[4];
            #pragma unroll
            for (unsigned kh = 0u; kh < 2u; ++kh) {
                ldsm_x4(bf[kh], s.Wq + brow * LDB + 16u * kh);
                ldsm_x4(ad, s.Aq[2] + arow * LDB + 16u * kh); mma_all(chi, ad, bf[kh]);
                ldsm_x4(ad, s.Aq[1] + arow * LDB + 16u * kh); mma_all(clo, ad, bf[kh]);
            }
            times256();
            #pragma unroll
            for (unsigned kh = 0u; kh < 2u; ++kh) { ldsm_x4(ad, s.Aq[0] + arow * LDB + 16u * kh); mma_all(clo, ad, bf[kh]); }
            fold(0u, true);
        }
        if (kt + 1u < nk) store(st[(kt + 1u) & 1u], (kt + 1u) * BK);
        __syncthreads();
    }
    if (bad) atomicOr(ovf, 1);
    #pragma unroll
    for (unsigned mi = 0u; mi < 4u; ++mi)
        #pragma unroll
        for (unsigned ni = 0u; ni < 4u; ++ni) {
            const unsigned r = m0 + wm * 32u + mi * 8u + g, c = n0 + wn * 32u + ni * 8u + 2u * t4;
            if (r < M) { if (c < N) C[(size_t)r * ldc + c] = acc[mi][ni][0]; if (c + 1u < N) C[(size_t)r * ldc + c + 1u] = acc[mi][ni][1]; }
        }
}
#define GEMM_ARGS const float* __restrict__ A, unsigned lda, const unsigned* __restrict__ codes, \
                  const unsigned* __restrict__ aux, float* __restrict__ C, unsigned ldc, unsigned M, unsigned N, \
                  unsigned K, int* __restrict__ ovf
extern "C" __global__ void __launch_bounds__(256, 2) q4k_gemm3(GEMM_ARGS)  { gemm3_t<0>(A, lda, codes, aux, C, ldc, M, N, K, ovf); }
extern "C" __global__ void __launch_bounds__(256, 2) q5k_gemm3(GEMM_ARGS)  { gemm3_t<1>(A, lda, codes, aux, C, ldc, M, N, K, ovf); }
extern "C" __global__ void __launch_bounds__(256, 2) q6k_gemm3(GEMM_ARGS)  { gemm3_t<2>(A, lda, codes, aux, C, ldc, M, N, K, ovf); }
extern "C" __global__ void __launch_bounds__(256, 2) q8_0_gemm3(GEMM_ARGS) { gemm3_t<3>(A, lda, codes, aux, C, ldc, M, N, K, ovf); }
extern "C" __global__ void __launch_bounds__(256, 2) q5_0_gemm3(GEMM_ARGS) { gemm3_t<4>(A, lda, codes, aux, C, ldc, M, N, K, ovf); }
#undef GEMM_ARGS

// ── h[t, i] = silu(gu[t, i]) · gu[t, n_ff + i] — the same expression the fused decode kernels use. ──
extern "C" __global__ void swiglu_rows(const float* __restrict__ gu, float* __restrict__ h, unsigned n_ff, unsigned rows) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_ff * rows) return;
    const unsigned t = i / n_ff, c = i - t * n_ff;
    const float g = gu[(size_t)t * 2u * n_ff + c], u = gu[(size_t)t * 2u * n_ff + n_ff + c];
    h[i] = (g / (1.f + expf(-g))) * u;
}

// ── Causal attention for T query rows against the cache: row i (absolute position pos + i) sees
//    keys [0, pos + i]. Block = one q-head x 16 query rows (4 warps x 4 rows); keys stream through
//    shared memory 32 at a time, so each K/V row is read once per 16 queries instead of once per query.
//    Lane j scores key j against the warp's 4 queries (Ks padded to 129 floats per row: lane j reads
//    row j, and an unpadded dh-float stride puts all 32 lanes on one bank). Online softmax per query;
//    lane owns dh/32 output dims for the P·V accumulate. dh <= 128, dh % 32 == 0. ──
extern "C" __global__ void attn_prefill(const float* __restrict__ q, const float* __restrict__ kc,
                                        const float* __restrict__ vc, float* __restrict__ out,
                                        unsigned nh, unsigned nkv, unsigned dh, unsigned T, unsigned pos, float scale) {
    __shared__ float Qs[16][128];
    __shared__ float Ks[32][129];
    __shared__ float Vs[32][128];
    __shared__ float Ps[4][4][32];
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const unsigned head = blockIdx.y, kvh = head / (nh / nkv), q0 = blockIdx.x * 16u;
    const unsigned qw = nh * dh, kvw = nkv * dh, dpl = dh / 32u;
    for (unsigned i = tid; i < 16u * dh; i += 128u) {
        const unsigned r = i / dh, e = i - r * dh;
        Qs[r][e] = (q0 + r < T) ? q[(size_t)(q0 + r) * qw + head * dh + e] : 0.f;
    }
    float m[4], l[4], o[4][4];
    #pragma unroll
    for (unsigned r = 0u; r < 4u; ++r) { m[r] = -3.0e38f; l[r] = 0.f; o[r][0] = o[r][1] = o[r][2] = o[r][3] = 0.f; }
    const unsigned kend = pos + min(q0 + 16u, T);         // keys the tile's last query can see
    for (unsigned k0 = 0u; k0 < kend; k0 += 32u) {
        __syncthreads();                                  // Qs written / previous K,V tile consumed
        for (unsigned i = tid; i < 32u * dh; i += 128u) {
            const unsigned j = i / dh, e = i - j * dh, kr = k0 + j;
            const bool in = kr < kend;
            Ks[j][e] = in ? kc[(size_t)kr * kvw + kvh * dh + e] : 0.f;
            Vs[j][e] = in ? vc[(size_t)kr * kvw + kvh * dh + e] : 0.f;
        }
        __syncthreads();
        const unsigned key = k0 + lane;
        #pragma unroll
        for (unsigned r = 0u; r < 4u; ++r) {
            const unsigned qr = warp * 4u + r, qi = q0 + qr;
            float s = 0.f;
            for (unsigned e = 0u; e < dh; ++e) s += Qs[qr][e] * Ks[lane][e];
            const bool vis = qi < T && key <= pos + qi && key < kend;
            s = vis ? s * scale : -3.0e38f;
            const float mn = fmaxf(m[r], warp_max(s));
            const float corr = expf(m[r] - mn);
            const float p = vis ? expf(s - mn) : 0.f;
            l[r] = l[r] * corr + warp_sum(p);
            #pragma unroll
            for (unsigned e = 0u; e < 4u; ++e) o[r][e] *= corr;
            m[r] = mn;
            Ps[warp][r][lane] = p;
        }
        __syncwarp();
        // ⚠ `e < 4` with a guard, not `e < dpl`: a runtime bound on an index into o[][] would put the
        // accumulators in local memory instead of registers.
        for (unsigned j = 0u; j < 32u; ++j) {
            #pragma unroll
            for (unsigned r = 0u; r < 4u; ++r) {
                const float pj = Ps[warp][r][j];
                #pragma unroll
                for (unsigned e = 0u; e < 4u; ++e) if (e < dpl) o[r][e] += pj * Vs[j][lane * dpl + e];
            }
        }
        __syncwarp();
    }
    #pragma unroll
    for (unsigned r = 0u; r < 4u; ++r) {
        const unsigned qi = q0 + warp * 4u + r;
        #pragma unroll
        for (unsigned e = 0u; e < 4u; ++e)
            if (qi < T && e < dpl) out[(size_t)qi * qw + head * dh + lane * dpl + e] = o[r][e] / l[r];
    }
}


// ── Causal prefill attention v2: REGISTER-TILED. ⛔ attn_prefill above gives each lane one key and has
//    it walk dh one float at a time from shared memory against 4 queries: one shared load per FMA, and
//    the profiler (FERRIC_CUDA_PROFILE) put it at 24% (Llama-3.2-1B), 33% (Qwen2.5-0.5B) and 46%
//    (Qwen3-0.6B, dh 128) of a 512-row prefill's GPU time — ~0.5 TFLOPS of f32 work.
//    Block = 32 query rows x one q-head, 128 threads; thread (tq, tk) = (tid / 8, tid % 8) owns query
//    rows {2tq, 2tq+1} and, of each 32-key tile, keys {tk, tk+8, tk+16, tk+24}: its 2 x 4 scores come
//    from float4 shared loads (two Q rows, four K rows per 4 dh) — 32 FMA per 6 loads — and the 8
//    lanes that share a query row are adjacent, so the row's max and sum are 3 shuffles. P·V: the same
//    thread owns the same two rows x the float4 columns {4tk + 32c}, taking each key's p by shuffle
//    from the lane that scored it. K and V share one shared buffer in turn (V loads after the scores):
//    dh 128 fits in 34 KB. Same f32 math as attn_prefill (online softmax per row, per 32-key tile); only
//    the order of the dh sum differs. dh <= 128, dh % 32 == 0. ──
#define AP_BQ 32u
#define AP_BK 32u
extern "C" __global__ void attn_prefill2(const float* __restrict__ q, const float* __restrict__ kc,
                                         const float* __restrict__ vc, float* __restrict__ out,
                                         unsigned nh, unsigned nkv, unsigned dh, unsigned T, unsigned pos, float scale) {
    __shared__ __align__(16) float Qs[AP_BQ * 132];
    __shared__ __align__(16) float KV[AP_BK * 132];
    const unsigned tid = threadIdx.x, lane = tid & 31u;
    const unsigned tq = tid >> 3u, tk = tid & 7u;
    const unsigned head = blockIdx.y, kvh = head / (nh / nkv), q0 = blockIdx.x * AP_BQ;
    const unsigned qw = nh * dh, kvw = nkv * dh, ld = dh + 4u, d4 = dh / 4u, nc = dh / 32u;
    for (unsigned i = tid; i < AP_BQ * d4; i += 128u) {
        const unsigned r = i / d4, e = (i - r * d4) * 4u;
        const float4 v = (q0 + r < T) ? *reinterpret_cast<const float4*>(q + (size_t)(q0 + r) * qw + head * dh + e) : make_float4(0.f, 0.f, 0.f, 0.f);
        *reinterpret_cast<float4*>(Qs + r * ld + e) = v;
    }
    float m[2] = {-3.0e38f, -3.0e38f}, l[2] = {0.f, 0.f};
    float o[2][4][4];
    #pragma unroll
    for (unsigned r = 0u; r < 2u; ++r)
        #pragma unroll
        for (unsigned c = 0u; c < 4u; ++c) { o[r][c][0] = o[r][c][1] = o[r][c][2] = o[r][c][3] = 0.f; }
    const unsigned qa = 2u * tq, kend = pos + min(q0 + AP_BQ, T);   // keys the tile's last query can see
    for (unsigned k0 = 0u; k0 < kend; k0 += AP_BK) {
        __syncthreads();                                   // Qs written / the previous V consumed
        for (unsigned i = tid; i < AP_BK * d4; i += 128u) {
            const unsigned j = i / d4, e = (i - j * d4) * 4u, kr = k0 + j;
            *reinterpret_cast<float4*>(KV + j * ld + e) = kr < kend ? *reinterpret_cast<const float4*>(kc + (size_t)kr * kvw + kvh * dh + e) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        __syncthreads();
        float sc[2][4];
        #pragma unroll
        for (unsigned r = 0u; r < 2u; ++r)
            #pragma unroll
            for (unsigned j = 0u; j < 4u; ++j) sc[r][j] = 0.f;
        for (unsigned e = 0u; e < dh; e += 4u) {
            const float4 a0 = *reinterpret_cast<const float4*>(Qs + qa * ld + e), a1 = *reinterpret_cast<const float4*>(Qs + (qa + 1u) * ld + e);
            #pragma unroll
            for (unsigned j = 0u; j < 4u; ++j) {
                const float4 b = *reinterpret_cast<const float4*>(KV + (tk + 8u * j) * ld + e);
                sc[0][j] += a0.x * b.x + a0.y * b.y + a0.z * b.z + a0.w * b.w;
                sc[1][j] += a1.x * b.x + a1.y * b.y + a1.z * b.z + a1.w * b.w;
            }
        }
        // mask, then the online softmax per row (its 8 lanes: xor 1, 2, 4)
        float corr[2];
        #pragma unroll
        for (unsigned r = 0u; r < 2u; ++r) {
            const unsigned qi = q0 + qa + r;
            float mx = -3.0e38f;
            #pragma unroll
            for (unsigned j = 0u; j < 4u; ++j) {
                const unsigned key = k0 + tk + 8u * j;
                const bool vis = qi < T && key <= pos + qi && key < kend;
                sc[r][j] = vis ? sc[r][j] * scale : -3.0e38f;
                mx = fmaxf(mx, sc[r][j]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1)); mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
            const float mn = fmaxf(m[r], mx);
            corr[r] = expf(m[r] - mn);
            float ps = 0.f;
            #pragma unroll
            for (unsigned j = 0u; j < 4u; ++j) { const float pv = sc[r][j] > -1.0e38f ? expf(sc[r][j] - mn) : 0.f; sc[r][j] = pv; ps += pv; }
            ps += __shfl_xor_sync(0xffffffffu, ps, 1); ps += __shfl_xor_sync(0xffffffffu, ps, 2); ps += __shfl_xor_sync(0xffffffffu, ps, 4);
            l[r] = l[r] * corr[r] + ps;
            m[r] = mn;
        }
        __syncthreads();                                   // every thread done reading K
        for (unsigned i = tid; i < AP_BK * d4; i += 128u) {
            const unsigned j = i / d4, e = (i - j * d4) * 4u, kr = k0 + j;
            *reinterpret_cast<float4*>(KV + j * ld + e) = kr < kend ? *reinterpret_cast<const float4*>(vc + (size_t)kr * kvw + kvh * dh + e) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned r = 0u; r < 2u; ++r)
            #pragma unroll
            for (unsigned c = 0u; c < 4u; ++c) { o[r][c][0] *= corr[r]; o[r][c][1] *= corr[r]; o[r][c][2] *= corr[r]; o[r][c][3] *= corr[r]; }
        const unsigned base = lane & ~7u;
        #pragma unroll 4
        for (unsigned jj = 0u; jj < AP_BK; ++jj) {
            // key jj was scored by lane (base | jj % 8), slot jj / 8
            const unsigned src = base | (jj & 7u), slot = jj >> 3u;
            const float s0 = slot == 0u ? sc[0][0] : slot == 1u ? sc[0][1] : slot == 2u ? sc[0][2] : sc[0][3];
            const float s1 = slot == 0u ? sc[1][0] : slot == 1u ? sc[1][1] : slot == 2u ? sc[1][2] : sc[1][3];
            const float p0 = __shfl_sync(0xffffffffu, s0, src), p1 = __shfl_sync(0xffffffffu, s1, src);
            #pragma unroll
            for (unsigned c = 0u; c < 4u; ++c) {
                if (c < nc) {
                    const float4 v = *reinterpret_cast<const float4*>(KV + jj * ld + 4u * tk + 32u * c);
                    o[0][c][0] += p0 * v.x; o[0][c][1] += p0 * v.y; o[0][c][2] += p0 * v.z; o[0][c][3] += p0 * v.w;
                    o[1][c][0] += p1 * v.x; o[1][c][1] += p1 * v.y; o[1][c][2] += p1 * v.z; o[1][c][3] += p1 * v.w;
                }
            }
        }
    }
    #pragma unroll
    for (unsigned r = 0u; r < 2u; ++r) {
        const unsigned qi = q0 + qa + r;
        if (qi >= T) continue;
        const float inv = 1.f / l[r];
        #pragma unroll
        for (unsigned c = 0u; c < 4u; ++c)
            if (c < nc) *reinterpret_cast<float4*>(out + (size_t)qi * qw + head * dh + 4u * tk + 32u * c) =
                            make_float4(o[r][c][0] * inv, o[r][c][1] * inv, o[r][c][2] * inv, o[r][c][3] * inv);
    }
}
