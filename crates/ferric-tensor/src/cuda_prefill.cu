// Ferric NVIDIA tier 3: PREFILL — T rows at once, resident on the device, on the tensor cores.
//
// The decode kernels (cuda_decode.cu) are GEMVs: one activation row, bandwidth-bound, f32 all the way.
// A prompt is T rows, and there the work is compute: T x (every weight) multiply-adds. This file is the
// matrix half — a tiled quantised GEMM on `mma.sync` tensor cores, a tiled causal attention, and the
// SwiGLU between them. Row-wise norm / rope reuse the decode kernels (they take a row index from
// blockIdx.y), so both paths share one implementation of everything that is not a matmul.
//
// ⚠ NUMERICS. Tensor cores multiply f16 x f16 and accumulate in f32. Each weight tile is dequantised
// from the SAME repacked words the GEMVs read straight into f16 in shared memory (<= 2^-12 relative
// per weight). Each ACTIVATION is split into two f16 parts, hi = f16(a) and lo = f16(a - hi), and both
// go through the tensor cores (two mma per fragment): a·w = hi·w + lo·w to ~2^-22, so the activation
// is effectively f32. ⭐ Measured before choosing it: with a single f16 activation, Qwen3-0.6B Q5_K_M
// over a 1000-row prompt sat 0.219 max |Δ logit| from the f32 WGSL path (argmax 1023/1024); with the
// split, 0.028 (1024/1024) — activation rounding was 8x the error budget. It costs ~25% of prefill
// throughput on the RTX 4050 (4218 -> 3245 tok/s on that model, 2964 -> 2162 on Llama-3.2-1B), and
// is still a numerics change from the f32 decode path, gated as such by scripts/cuda_conformance.sh.
// An activation above f16's 65504 would become inf: every A-tile load checks, and a hit raises `ovf`
// so the host throws the result away and runs the f32 WGSL prefill instead — a fallback, never an inf.
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

// ── Dequantise 16 consecutive weights of output row `n`: values [32·kc + 16·h, +16) of the row, into
//    8 packed half2 words. Every format's 32-value chunk splits into two such halves, which is what
//    lets one tile loader serve all five. F: 0 Q4_K, 1 Q5_K, 2 Q6_K, 3 Q8_0, 4 Q5_0 (QFmt order). ──
template <int F>
__device__ __forceinline__ void dequant16(const unsigned* __restrict__ codes, const unsigned* __restrict__ aux,
                                          unsigned n, unsigned K, unsigned kc, unsigned h, unsigned (&o)[8]) {
    float v[16];
    if (F == 0 || F == 1) {
        const unsigned nblk = K / 256u, b = kc >> 3u, s = kc & 7u, bi = n * nblk + b, ab = bi * 4u;
        const unsigned dd = aux[ab];
        float ds, mm; scmin(aux, ab, s, f16_to_f32(dd & 0xffffu), f16_to_f32(dd >> 16u), ds, mm);
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
                float nib = (float)((qw[w] >> (8u * k + sh)) & 0xfu);
                if (F == 1) nib += (float)((hw[w] >> (8u * k + s)) & 1u) * 16.f;
                v[4u * w + k] = ds * nib - mm;
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
        const float sc = (float)((int)(((aux[ab + 1u + (si >> 2u)] >> (8u * (si & 3u))) & 0xffu) << 24u) >> 24);
        const float dsc = d * sc;
        const unsigned lw[4] = {ql.x, ql.y, ql.z, ql.w}, hw[4] = {qh.x, qh.y, qh.z, qh.w};
        const unsigned nsh = 4u * (qq >> 1u), hsh = 2u * qq;
        #pragma unroll
        for (unsigned w = 0u; w < 4u; ++w)
            #pragma unroll
            for (unsigned k = 0u; k < 4u; ++k) {
                const int q = (int)(((lw[w] >> (8u * k + nsh)) & 0xfu) | (((hw[w] >> (8u * k + hsh)) & 3u) << 4u)) - 32;
                v[4u * w + k] = dsc * (float)q;
            }
    } else if (F == 3) {
        const unsigned bi = n * (K / 32u) + kc;
        const unsigned sw = aux[bi >> 1u];
        const float d = f16_to_f32((bi & 1u) ? (sw >> 16u) : (sw & 0xffffu));
        const uint4 q = *reinterpret_cast<const uint4*>(codes + bi * 8u + 4u * h);
        const unsigned qw[4] = {q.x, q.y, q.z, q.w};
        #pragma unroll
        for (unsigned w = 0u; w < 4u; ++w)
            #pragma unroll
            for (unsigned k = 0u; k < 4u; ++k) v[4u * w + k] = d * (float)((int)(qw[w] << (24u - 8u * k)) >> 24);
    } else {
        // Q5_0: 16 code bytes; h = 0 -> low nibbles with qh bits 0..15, h = 1 -> high nibbles, bits 16..31.
        const unsigned bi = n * (K / 32u) + kc;
        const unsigned qh = aux[bi * 2u];
        const float d = f16_to_f32(aux[bi * 2u + 1u] & 0xffffu);
        const uint4 q = *reinterpret_cast<const uint4*>(codes + bi * 4u);
        const unsigned qw[4] = {q.x, q.y, q.z, q.w};
        #pragma unroll
        for (unsigned w = 0u; w < 4u; ++w)
            #pragma unroll
            for (unsigned k = 0u; k < 4u; ++k) {
                const unsigned e = 4u * w + k;
                const int q5 = (int)(((qw[w] >> (8u * k + 4u * h)) & 0xfu) | (((qh >> (e + 16u * h)) & 1u) << 4u)) - 16;
                v[e] = (float)q5 * d;
            }
    }
    #pragma unroll
    for (unsigned i = 0u; i < 8u; ++i) o[i] = pack2(v[2u * i], v[2u * i + 1u]);
}

// ── C[M, N] (+)= A[M, K] · W[N, K]ᵀ with W quantised. A f32 row-major (lda), C f32 row-major (ldc).
//    Tile 64x64x32, 128 threads = 2x2 warps of 32x32, each warp 2 (m16) x 4 (n8) mma tiles, k8 steps;
//    A is held twice (hi and lo halves of the split, see NUMERICS) and every fragment gets two mma.
//    Shared rows are padded to 40 halves (80 B = 20 words): the fragment reads of rows g = 0..7 then
//    start on banks 0,20,8,28,16,4,24,12 — conflict-free, where an unpadded 64 B stride is 4-way. ──
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
    __shared__ __align__(16) unsigned short As[BM * LDS];      // activation, f16 hi part
    __shared__ __align__(16) unsigned short Al[BM * LDS];      // activation, f16 lo part (a - hi)
    __shared__ __align__(16) unsigned short Ws[BN * LDS];
    const unsigned tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const unsigned wm = warp >> 1u, wn = warp & 1u;
    const unsigned m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const unsigned g = lane >> 2u, t4 = lane & 3u;
    float acc[2][4][4];
    #pragma unroll
    for (unsigned i = 0u; i < 2u; ++i)
        #pragma unroll
        for (unsigned j = 0u; j < 4u; ++j) { acc[i][j][0] = 0.f; acc[i][j][1] = 0.f; acc[i][j][2] = 0.f; acc[i][j][3] = 0.f; }
    bool bad = false;
    const unsigned lr = tid >> 1u, lh = tid & 1u;    // the loaders: row lr of each tile, half lh of its 32 values
    for (unsigned k0 = 0u; k0 < K; k0 += BK) {
        unsigned a[8], al[8], w[8];
        if (m0 + lr < M) {
            const float* src = A + (size_t)(m0 + lr) * lda + k0 + 16u * lh;
            #pragma unroll
            for (unsigned j = 0u; j < 4u; ++j) {
                const float4 v = *reinterpret_cast<const float4*>(src + 4u * j);
                bad |= fabsf(v.x) > 65504.f || fabsf(v.y) > 65504.f || fabsf(v.z) > 65504.f || fabsf(v.w) > 65504.f;
                a[2u * j] = pack2(v.x, v.y); a[2u * j + 1u] = pack2(v.z, v.w);
                al[2u * j] = resid2(v.x, v.y, a[2u * j]); al[2u * j + 1u] = resid2(v.z, v.w, a[2u * j + 1u]);
            }
        } else {
            #pragma unroll
            for (unsigned j = 0u; j < 8u; ++j) { a[j] = 0u; al[j] = 0u; }
        }
        if (n0 + lr < N) dequant16<F>(codes, aux, n0 + lr, K, k0 / 32u, lh, w);
        else {
            #pragma unroll
            for (unsigned j = 0u; j < 8u; ++j) w[j] = 0u;
        }
        uint4* as = reinterpret_cast<uint4*>(As + lr * LDS + 16u * lh);
        uint4* ws = reinterpret_cast<uint4*>(Ws + lr * LDS + 16u * lh);
        as[0] = make_uint4(a[0], a[1], a[2], a[3]); as[1] = make_uint4(a[4], a[5], a[6], a[7]);
        uint4* als = reinterpret_cast<uint4*>(Al + lr * LDS + 16u * lh);
        als[0] = make_uint4(al[0], al[1], al[2], al[3]); als[1] = make_uint4(al[4], al[5], al[6], al[7]);
        ws[0] = make_uint4(w[0], w[1], w[2], w[3]); ws[1] = make_uint4(w[4], w[5], w[6], w[7]);
        __syncthreads();
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
                    // lo first: the small term accumulates before the large one lands on it.
                    mma16816(acc[mi][ni], lf[mi][0], lf[mi][1], bf[ni]);
                    mma16816(acc[mi][ni], af[mi][0], af[mi][1], bf[ni]);
                }
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
