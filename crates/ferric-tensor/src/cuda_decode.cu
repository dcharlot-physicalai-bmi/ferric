// Ferric NVIDIA tier 2: every op of one dense decode step (rows == 1), resident on the device.
// Same repacked weight layouts as the WGSL kernels (see dtype.rs from_bytes for Q5_K / Q6_K), so the
// host repack is shared and ONLY the kernels differ across fabrics. No cuBLAS, no CUTLASS, no headers.
// Build: nvcc -O3 -arch=compute_75 -ptx cuda_decode.cu -o cuda_decode.ptx

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
// Block-wide sum for blockDim.x == 256 (8 warps). `red` must hold 8 floats.
__device__ __forceinline__ float block_sum256(float v, float* red) {
    v = warp_sum(v);
    if ((threadIdx.x & 31u) == 0u) red[threadIdx.x >> 5u] = v;
    __syncthreads();
    float t = (threadIdx.x < 8u) ? red[threadIdx.x] : 0.f;
    if (threadIdx.x < 32u) t = warp_sum(t);
    if (threadIdx.x == 0u) red[0] = t;
    __syncthreads();
    return red[0];
}

// ── rmsnorm: out = x * rsqrt(mean(x^2) + eps) * w.  One block of 256 threads per ROW (blockIdx.x),
//    any d. Decode launches one block (row 0: the same arithmetic as before rows existed); prefill
//    launches one per prompt row. ──
extern "C" __global__ void rmsnorm(const float* __restrict__ x, const float* __restrict__ w,
                                   float* __restrict__ out, unsigned d, float eps) {
    __shared__ float red[8];
    x += (size_t)blockIdx.x * d; out += (size_t)blockIdx.x * d;
    float ms = 0.f;
    for (unsigned j = threadIdx.x; j < d; j += blockDim.x) { const float v = x[j]; ms += v * v; }
    ms = block_sum256(ms, red);
    const float inv = 1.f / sqrtf(ms / (float)d + eps);
    for (unsigned j = threadIdx.x; j < d; j += blockDim.x) out[j] = x[j] * inv * w[j];
}
// ── add_rmsnorm: sum = x + y (the next residual); norm = rmsnorm(sum) * w. One block per row. ──
extern "C" __global__ void add_rmsnorm(const float* __restrict__ x, const float* __restrict__ y,
                                       const float* __restrict__ w, float* __restrict__ sum,
                                       float* __restrict__ norm, unsigned d, float eps) {
    __shared__ float red[8];
    const size_t ro = (size_t)blockIdx.x * d;
    x += ro; y += ro; sum += ro; norm += ro;
    float ms = 0.f;
    for (unsigned j = threadIdx.x; j < d; j += blockDim.x) { const float v = x[j] + y[j]; sum[j] = v; ms += v * v; }
    ms = block_sum256(ms, red);
    const float inv = 1.f / sqrtf(ms / (float)d + eps);
    for (unsigned j = threadIdx.x; j < d; j += blockDim.x) norm[j] = sum[j] * inv * w[j];
}
// ── add: out = a + b ──
extern "C" __global__ void vadd(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ out, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// ── Q6_K GEMV. codes: 48 u32/block = ql[32 words] qh[16 words]; aux: 5 u32/block = [d f16][16 i8 scales].
//    Mirrors WGSL Q6_K_BODY (dtype.rs) exactly: element order hf x l, four products per l.
//    The per-lane dot is `q6k_dot_lane` below so the fused SwiGLU kernel can share it; `q6k_gemv`'s
//    arithmetic is unchanged by that move (same expressions, same order). ──
__device__ __forceinline__ unsigned q6_qlb(const unsigned* __restrict__ codes, unsigned cb, unsigned i) {
    return (codes[cb + (i >> 2u)] >> (8u * (i & 3u))) & 0xffu;
}
__device__ __forceinline__ unsigned q6_qhb(const unsigned* __restrict__ codes, unsigned cb, unsigned i) {
    return (codes[cb + 32u + (i >> 2u)] >> (8u * (i & 3u))) & 0xffu;
}
__device__ __forceinline__ float q6_scb(const unsigned* __restrict__ aux, unsigned ab, unsigned i) {
    const unsigned b = (aux[ab + 1u + (i >> 2u)] >> (8u * (i & 3u))) & 0xffu;
    return (float)((int)(b << 24u) >> 24);          // i8 sign-extend, as WGSL `i32(b << 24u) >> 24u`
}
// ⛔ SAME OVER-FETCH AS Q5_K HAD, same fix. The first lane plan gave each of 8 sub-lanes one (hf,
// 8-l) slice and read `ql`/`qh` byte-wise through 4-byte words; every Q6_K shape sat at 60-69 GB/s.
// Layout (WGSL Q6_K_BODY): for hf in {0,1}, ql bytes [64hf, +64): byte b<32 -> l=b: low nibble q1
// (elem 128hf+l), high nibble q3 (elem 128hf+64+l); byte b>=32 -> l=b-32: low q2 (elem 128hf+32+l),
// high q4 (elem 128hf+96+l). qh bytes [32hf, +32): byte l holds 2-bit high parts for q1..q4 of that l.
// Lane j (0..8): hf = j>>2, sub = j&3; ql uint4 = bytes [64hf + 16sub, +16) -> 16 consecutive l with
// is = l>>4 constant; q1&q3 if sub<2 (l0 = 16sub) else q2&q4 (l0 = 16(sub-2)); qh uint4 = bytes
// [32hf + l0, +16). One uint4 of ql + one of qh + 32 x floats per lane per block; 8 lanes cover the
// block's 128 B of ql contiguously.
__device__ __forceinline__ float q6k_dot_lane(const float* __restrict__ x, const unsigned* __restrict__ codes,
                                              const unsigned* __restrict__ aux, unsigned o, unsigned nblk,
                                              unsigned lane) {
    const unsigned sl = lane & 7u, bl = lane >> 3u;
    const unsigned hf = sl >> 2u, sub = sl & 3u;
    const bool second = sub >= 2u;                   // q2&q4 instead of q1&q3
    const unsigned l0 = 16u * (second ? sub - 2u : sub);
    const unsigned is = l0 >> 4u;                    // constant over the lane's 16 l's
    float acc = 0.f;
    for (unsigned blk = bl; blk < nblk; blk += 4u) {
        const unsigned bi = o * nblk + blk, cb = bi * 48u, ab = bi * 5u;
        const float d = f16_to_f32(aux[ab] & 0xffffu);
        const unsigned sco = 8u * hf;
        // the two scales this lane uses (q1,q3) or (q2,q4)
        const float sA = d * q6_scb(aux, ab, sco + is + (second ? 2u : 0u));
        const float sB = d * q6_scb(aux, ab, sco + is + (second ? 6u : 4u));
        const uint4 ql = *reinterpret_cast<const uint4*>(codes + cb + (64u * hf + 16u * sub) / 4u);
        const uint4 qh = *reinterpret_cast<const uint4*>(codes + cb + 32u + (32u * hf + l0) / 4u);
        const float* xa = x + blk * 256u + 128u * hf + (second ? 32u : 0u) + l0;   // elem of q1 (or q2)
        const float* xb = xa + 64u;                                               // elem of q3 (or q4)
        const unsigned qlw[4] = {ql.x, ql.y, ql.z, ql.w}, qhw[4] = {qh.x, qh.y, qh.z, qh.w};
        const unsigned shA = second ? 2u : 0u, shB = second ? 6u : 4u;   // qh bit-pair positions
        float accA = 0.f, accB = 0.f;
        #pragma unroll
        for (unsigned w = 0u; w < 4u; ++w) {
            const float4 xA = *reinterpret_cast<const float4*>(xa + 4u * w);
            const float4 xB = *reinterpret_cast<const float4*>(xb + 4u * w);
            const unsigned lw = qlw[w], hw = qhw[w];
            #define Q6(k) \
                { const unsigned lb = (lw >> (8u * (k))) & 0xffu, hb = (hw >> (8u * (k))) & 0xffu; \
                  const int qA = (int)((lb & 0xFu) | (((hb >> shA) & 3u) << 4u)) - 32; \
                  const int qB = (int)((lb >> 4u)  | (((hb >> shB) & 3u) << 4u)) - 32; \
                  const float xv1 = (k)==0u?xA.x:(k)==1u?xA.y:(k)==2u?xA.z:xA.w; \
                  const float xv2 = (k)==0u?xB.x:(k)==1u?xB.y:(k)==2u?xB.z:xB.w; \
                  accA += xv1 * (float)qA; accB += xv2 * (float)qB; }
            Q6(0u) Q6(1u) Q6(2u) Q6(3u)
            #undef Q6
        }
        acc += sA * accA + sB * accB;
    }
    return acc;
}
extern "C" __global__ void q6k_gemv(const float* __restrict__ x, const unsigned* __restrict__ codes,
                                    const unsigned* __restrict__ aux, float* __restrict__ out,
                                    unsigned o_dim, unsigned in_dim) {
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    const unsigned o = blockIdx.x * 4u + warp;
    if (o >= o_dim) return;                          // warp-uniform
    float acc = q6k_dot_lane(x, codes, aux, o, in_dim / 256u, lane);
    acc = warp_sum(acc);
    if (lane == 0u) out[o] = acc;
}

// ── Q5_K dot for one output row, lane-partitioned for COALESCED 16-byte loads. ──
//
// ⛔ THE FIRST VERSION OVER-FETCHED 3.2x. It gave each of 8 lanes one 32-value sub-block: lanes 2j
// and 2j+1 then read the SAME 32 B `qs` chunk (low vs high nibbles), and all eight read the SAME 32 B
// of `qh`, all through 4-byte loads — every shape sat at 59–68 GB/s, ~34% of the floor, including the
// 121 MiB lm_head, which rules out occupancy. Same math, re-partitioned: lane j owns positions
// [16·(j&1), +16) of the TWO sub-blocks 2c and 2c+1 (c = j>>1) — one uint4 of qs (low nibbles -> 2c,
// high -> 2c+1), one uint4 of qh (bits 2c / 2c+1), eight lanes covering a block's 128 B contiguously.
__device__ __forceinline__ unsigned q5_scbyte(const unsigned* __restrict__ aux, unsigned ab, unsigned i) {
    return (aux[ab + 1u + (i >> 2u)] >> (8u * (i & 3u))) & 0xffu;
}
__device__ __forceinline__ void q5_scmin(const unsigned* __restrict__ aux, unsigned ab, unsigned s, float d, float dmin,
                                         float& ds, float& mm) {
    unsigned sc, mn;
    if (s < 4u) { sc = q5_scbyte(aux, ab, s) & 63u; mn = q5_scbyte(aux, ab, s + 4u) & 63u; }
    else { const unsigned a = q5_scbyte(aux, ab, s + 4u), lo = q5_scbyte(aux, ab, s - 4u), hi = q5_scbyte(aux, ab, s);
           sc = (a & 0x0Fu) | ((lo >> 6u) << 4u); mn = (a >> 4u) | ((hi >> 6u) << 4u); }
    ds = d * (float)sc; mm = dmin * (float)mn;
}
__device__ __forceinline__ float q5k_dot_lane(const float* __restrict__ x, const unsigned* __restrict__ codes,
                                              const unsigned* __restrict__ aux, unsigned o, unsigned nblk,
                                              unsigned bl, unsigned j) {
    const unsigned c = j >> 1u, half = j & 1u, s0 = 2u * c, s1 = s0 + 1u;
    float acc = 0.f;
    for (unsigned blk = bl; blk < nblk; blk += 4u) {
        const unsigned bi = o * nblk + blk, ab = bi * 4u, cb40 = bi * 40u;
        const unsigned dd = aux[ab];
        const float d = f16_to_f32(dd & 0xffffu), dmin = f16_to_f32(dd >> 16u);
        float ds0, mm0, ds1, mm1;
        q5_scmin(aux, ab, s0, d, dmin, ds0, mm0);
        q5_scmin(aux, ab, s1, d, dmin, ds1, mm1);
        const uint4 q = *reinterpret_cast<const uint4*>(codes + cb40 + 8u * c + 4u * half);   // 16 B of qs
        const uint4 h = *reinterpret_cast<const uint4*>(codes + cb40 + 32u + 4u * half);      // 16 B of qh
        const float* x0 = x + blk * 256u + 32u * s0 + 16u * half;
        const float* x1 = x + blk * 256u + 32u * s1 + 16u * half;
        const unsigned qw[4] = {q.x, q.y, q.z, q.w}, hw[4] = {h.x, h.y, h.z, h.w};
        float a0 = 0.f, sx0 = 0.f, a1 = 0.f, sx1 = 0.f;
        #pragma unroll
        for (unsigned w = 0u; w < 4u; ++w) {
            const float4 xa = *reinterpret_cast<const float4*>(x0 + 4u * w);
            const float4 xb = *reinterpret_cast<const float4*>(x1 + 4u * w);
            const unsigned word = qw[w], qhw = hw[w];
            const float l0 = (float)(word & 0xfu)         + (float)((qhw >> s0) & 1u)         * 16.f;
            const float l1 = (float)((word >> 8u) & 0xfu) + (float)((qhw >> (8u + s0)) & 1u)  * 16.f;
            const float l2 = (float)((word >> 16u) & 0xfu)+ (float)((qhw >> (16u + s0)) & 1u) * 16.f;
            const float l3 = (float)((word >> 24u) & 0xfu)+ (float)((qhw >> (24u + s0)) & 1u) * 16.f;
            const float u0 = (float)((word >> 4u) & 0xfu) + (float)((qhw >> s1) & 1u)         * 16.f;
            const float u1 = (float)((word >> 12u) & 0xfu)+ (float)((qhw >> (8u + s1)) & 1u)  * 16.f;
            const float u2 = (float)((word >> 20u) & 0xfu)+ (float)((qhw >> (16u + s1)) & 1u) * 16.f;
            const float u3 = (float)((word >> 28u) & 0xfu)+ (float)((qhw >> (24u + s1)) & 1u) * 16.f;
            a0 += xa.x * l0 + xa.y * l1 + xa.z * l2 + xa.w * l3;  sx0 += xa.x + xa.y + xa.z + xa.w;
            a1 += xb.x * u0 + xb.y * u1 + xb.z * u2 + xb.w * u3;  sx1 += xb.x + xb.y + xb.z + xb.w;
        }
        acc += ds0 * a0 - mm0 * sx0 + ds1 * a1 - mm1 * sx1;
    }
    return acc;
}
// ── Q5_K GEMV with the coalesced dot: 4 outputs per 128-thread block, warp-shuffle reduce. ──
extern "C" __global__ void q5k_gemv(const float* __restrict__ x, const unsigned* __restrict__ codes,
                                    const unsigned* __restrict__ aux, float* __restrict__ out,
                                    unsigned o_dim, unsigned in_dim) {
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    const unsigned o = blockIdx.x * 4u + warp;
    if (o >= o_dim) return;
    const unsigned nblk = in_dim / 256u;
    float acc = q5k_dot_lane(x, codes, aux, o, nblk, lane >> 3u, lane & 7u);
    acc = warp_sum(acc);
    if (lane == 0u) out[o] = acc;
}
// ── fused gate|up GEMV + SwiGLU: out[o] = silu(gate_o) * up_o; weight has 2*n_ff rows. ──
extern "C" __global__ void q5k_swiglu_gemv(const float* __restrict__ x, const unsigned* __restrict__ codes,
                                           const unsigned* __restrict__ aux, float* __restrict__ out,
                                           unsigned n_ff, unsigned in_dim) {
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    const unsigned o = blockIdx.x * 4u + warp;
    if (o >= n_ff) return;
    const unsigned nblk = in_dim / 256u, sl = lane & 7u, bl = lane >> 3u;
    float g = q5k_dot_lane(x, codes, aux, o, nblk, bl, sl);
    float u = q5k_dot_lane(x, codes, aux, o + n_ff, nblk, bl, sl);
    g = warp_sum(g); u = warp_sum(u);
    if (lane == 0u) out[o] = (g / (1.f + expf(-g))) * u;
}

// ═══════════ Q4_K, Q8_0, Q5_0 — the formats a Q4_K_M file is actually made of ═══════════
// A "Q4_K_M" file is not all Q4_K: llama.cpp bumps attn_v / ffn_down / the head to Q6_K, and any row
// whose width is not a multiple of 256 falls back to a 32-wide format — Qwen2.5-0.5B (d = 896) ships
// Q5_0 for q/k/v/o/gate/up and Q8_0 for half its attn_v and the output head. Each kernel reads the
// SAME repacked words its WGSL twin reads (dtype.rs `from_bytes`), and keeps the tier's contract:
// one warp per output row, 4 rows per 128-thread block, 16-byte coalesced weight loads, shuffle reduce.

// Q4_K: aux = [d|dmin f16x2][12 scale bytes] (4 u32/block, identical to Q5_K); codes = 32 u32 of qs.
// Lane plan is q5k_dot_lane's minus the qh plane: lane j owns positions [16(j&1), +16) of sub-blocks
// 2c and 2c+1 (c = j>>1) — one uint4 of qs, low nibbles -> 2c, high -> 2c+1; 8 lanes read a block's
// 128 B contiguously, 4 block-lanes stride the blocks.
__device__ __forceinline__ float q4k_dot_lane(const float* __restrict__ x, const unsigned* __restrict__ codes,
                                              const unsigned* __restrict__ aux, unsigned o, unsigned nblk,
                                              unsigned lane) {
    const unsigned bl = lane >> 3u, j = lane & 7u;
    const unsigned c = j >> 1u, half = j & 1u, s0 = 2u * c, s1 = s0 + 1u;
    float acc = 0.f;
    for (unsigned blk = bl; blk < nblk; blk += 4u) {
        const unsigned bi = o * nblk + blk, ab = bi * 4u, cb32 = bi * 32u;
        const unsigned dd = aux[ab];
        const float d = f16_to_f32(dd & 0xffffu), dmin = f16_to_f32(dd >> 16u);
        float ds0, mm0, ds1, mm1;
        q5_scmin(aux, ab, s0, d, dmin, ds0, mm0);
        q5_scmin(aux, ab, s1, d, dmin, ds1, mm1);
        const uint4 q = *reinterpret_cast<const uint4*>(codes + cb32 + 8u * c + 4u * half);
        const float* x0 = x + blk * 256u + 32u * s0 + 16u * half;
        const float* x1 = x + blk * 256u + 32u * s1 + 16u * half;
        const unsigned qw[4] = {q.x, q.y, q.z, q.w};
        float a0 = 0.f, sx0 = 0.f, a1 = 0.f, sx1 = 0.f;
        #pragma unroll
        for (unsigned w = 0u; w < 4u; ++w) {
            const float4 xa = *reinterpret_cast<const float4*>(x0 + 4u * w);
            const float4 xb = *reinterpret_cast<const float4*>(x1 + 4u * w);
            const unsigned word = qw[w];
            a0 += xa.x * (float)(word & 0xfu) + xa.y * (float)((word >> 8u) & 0xfu)
                + xa.z * (float)((word >> 16u) & 0xfu) + xa.w * (float)((word >> 24u) & 0xfu);
            a1 += xb.x * (float)((word >> 4u) & 0xfu) + xb.y * (float)((word >> 12u) & 0xfu)
                + xb.z * (float)((word >> 20u) & 0xfu) + xb.w * (float)((word >> 28u) & 0xfu);
            sx0 += xa.x + xa.y + xa.z + xa.w;
            sx1 += xb.x + xb.y + xb.z + xb.w;
        }
        acc += ds0 * a0 - mm0 * sx0 + ds1 * a1 - mm1 * sx1;
    }
    return acc;
}

// Q8_0: codes = 8 u32/block (32 int8); scales = one f16 per block, TWO PACKED PER u32
// (`scales[b/2] >> 16(b%2)`, dtype.rs). Two lanes per block, each one uint4 = 16 int8 (a block's 32 B
// read by a lane pair; 16 blocks = 512 B contiguous per warp step).
__device__ __forceinline__ float q8_0_dot_lane(const float* __restrict__ x, const unsigned* __restrict__ codes,
                                               const unsigned* __restrict__ scales, unsigned o, unsigned nblk,
                                               unsigned lane) {
    const unsigned half = lane & 1u;
    float acc = 0.f;
    for (unsigned blk = lane >> 1u; blk < nblk; blk += 16u) {
        const unsigned bi = o * nblk + blk;
        const unsigned sw = scales[bi >> 1u];
        const float d = f16_to_f32((bi & 1u) ? (sw >> 16u) : (sw & 0xffffu));
        const uint4 q = *reinterpret_cast<const uint4*>(codes + bi * 8u + 4u * half);
        const float* xp = x + blk * 32u + 16u * half;
        const unsigned qw[4] = {q.x, q.y, q.z, q.w};
        float s = 0.f;
        #pragma unroll
        for (unsigned w = 0u; w < 4u; ++w) {
            const float4 xv = *reinterpret_cast<const float4*>(xp + 4u * w);
            const unsigned word = qw[w];
            s += xv.x * (float)((int)(word << 24u) >> 24) + xv.y * (float)((int)(word << 16u) >> 24)
               + xv.z * (float)((int)(word << 8u) >> 24)  + xv.w * (float)((int)word >> 24);
        }
        acc += d * s;
    }
    return acc;
}

// Q5_0: codes = 4 u32/block (16 bytes: byte i low nibble -> element i, high nibble -> element i+16);
// scales = [qh u32, d f16] per block (2 u32). value = ((nibble | qh_bit << 4) - 16) * d.
// Two lanes per block; lane half h owns bytes [8h, 8h+8): elements [8h, +8) and [16+8h, +8).
__device__ __forceinline__ float q5_0_dot_lane(const float* __restrict__ x, const unsigned* __restrict__ codes,
                                               const unsigned* __restrict__ scales, unsigned o, unsigned nblk,
                                               unsigned lane) {
    const unsigned h = lane & 1u;
    float acc = 0.f;
    for (unsigned blk = lane >> 1u; blk < nblk; blk += 16u) {
        const unsigned bi = o * nblk + blk;
        const uint2 sc = *reinterpret_cast<const uint2*>(scales + bi * 2u);
        const unsigned qh = sc.x;
        const float d = f16_to_f32(sc.y & 0xffffu);
        const uint2 q = *reinterpret_cast<const uint2*>(codes + bi * 4u + 2u * h);
        const float* xl = x + blk * 32u + 8u * h;          // low-nibble elements
        const float* xh = xl + 16u;                        // high-nibble elements
        const unsigned qw[2] = {q.x, q.y};
        float s = 0.f;
        #pragma unroll
        for (unsigned w = 0u; w < 2u; ++w) {
            const float4 a = *reinterpret_cast<const float4*>(xl + 4u * w);
            const float4 b = *reinterpret_cast<const float4*>(xh + 4u * w);
            const unsigned word = qw[w], e = 8u * h + 4u * w;   // element index of this word's byte 0
            #define LO(k) (float)((int)(((word >> (8u * (k))) & 0xfu) | (((qh >> (e + (k))) & 1u) << 4u)) - 16)
            #define HI(k) (float)((int)(((word >> (8u * (k) + 4u)) & 0xfu) | (((qh >> (e + 16u + (k))) & 1u) << 4u)) - 16)
            s += a.x * LO(0u) + a.y * LO(1u) + a.z * LO(2u) + a.w * LO(3u)
               + b.x * HI(0u) + b.y * HI(1u) + b.z * HI(2u) + b.w * HI(3u);
            #undef LO
            #undef HI
        }
        acc += d * s;
    }
    return acc;
}

// One GEMV and one fused gate|up + SwiGLU entry point per format, all through `dot_lane<F>`.
// F: 0 = Q4_K, 1 = Q5_K, 2 = Q6_K, 3 = Q8_0, 4 = Q5_0 — the order of `QFmt` in cuda.rs.
// (Q5_K and Q6_K already have their own q5k_gemv / q6k_gemv / q5k_swiglu_gemv entries above, which the
// host keeps using so their recorded numerics cannot move; q6k_swiglu_gemv is new.)
template <int F>
__device__ __forceinline__ float dot_lane(const float* __restrict__ x, const unsigned* __restrict__ codes,
                                          const unsigned* __restrict__ aux, unsigned o, unsigned in_dim,
                                          unsigned lane) {
    if (F == 0) return q4k_dot_lane(x, codes, aux, o, in_dim / 256u, lane);
    if (F == 1) return q5k_dot_lane(x, codes, aux, o, in_dim / 256u, lane >> 3u, lane & 7u);
    if (F == 2) return q6k_dot_lane(x, codes, aux, o, in_dim / 256u, lane);
    if (F == 3) return q8_0_dot_lane(x, codes, aux, o, in_dim / 32u, lane);
    return q5_0_dot_lane(x, codes, aux, o, in_dim / 32u, lane);
}
template <int F>
__device__ __forceinline__ void gemv_t(const float* __restrict__ x, const unsigned* __restrict__ codes,
                                       const unsigned* __restrict__ aux, float* __restrict__ out,
                                       unsigned o_dim, unsigned in_dim) {
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    const unsigned o = blockIdx.x * 4u + warp;
    if (o >= o_dim) return;                          // warp-uniform
    float acc = dot_lane<F>(x, codes, aux, o, in_dim, lane);
    acc = warp_sum(acc);
    if (lane == 0u) out[o] = acc;
}
template <int F>
__device__ __forceinline__ void swiglu_t(const float* __restrict__ x, const unsigned* __restrict__ codes,
                                         const unsigned* __restrict__ aux, float* __restrict__ out,
                                         unsigned n_ff, unsigned in_dim) {
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    const unsigned o = blockIdx.x * 4u + warp;
    if (o >= n_ff) return;
    float g = dot_lane<F>(x, codes, aux, o, in_dim, lane);
    float u = dot_lane<F>(x, codes, aux, o + n_ff, in_dim, lane);
    g = warp_sum(g); u = warp_sum(u);
    if (lane == 0u) out[o] = (g / (1.f + expf(-g))) * u;
}
// ⚠ Spelled out, not macro-generated: tests/ptx_artifact.rs finds the kernel list by parsing
// `extern "C" __global__ void NAME(` lines, and a macro would hide these from that stale-PTX gate.
#define GEMV_ARGS const float* __restrict__ x, const unsigned* __restrict__ codes, \
                  const unsigned* __restrict__ aux, float* __restrict__ out, unsigned n_out, unsigned in_dim
extern "C" __global__ void q4k_gemv(GEMV_ARGS)         { gemv_t<0>(x, codes, aux, out, n_out, in_dim); }
extern "C" __global__ void q8_0_gemv(GEMV_ARGS)        { gemv_t<3>(x, codes, aux, out, n_out, in_dim); }
extern "C" __global__ void q5_0_gemv(GEMV_ARGS)        { gemv_t<4>(x, codes, aux, out, n_out, in_dim); }
extern "C" __global__ void q4k_swiglu_gemv(GEMV_ARGS)  { swiglu_t<0>(x, codes, aux, out, n_out, in_dim); }
extern "C" __global__ void q6k_swiglu_gemv(GEMV_ARGS)  { swiglu_t<2>(x, codes, aux, out, n_out, in_dim); }
extern "C" __global__ void q8_0_swiglu_gemv(GEMV_ARGS) { swiglu_t<3>(x, codes, aux, out, n_out, in_dim); }
extern "C" __global__ void q5_0_swiglu_gemv(GEMV_ARGS) { swiglu_t<4>(x, codes, aux, out, n_out, in_dim); }
#undef GEMV_ARGS

// ── QK-norm (optional) + NEOX RoPE for ONE row (decode): ONE BLOCK PER HEAD, dh threads.
//    ⛔ The first version was one THREAD per head — 24 threads doing 128 serial sinf/cosf/expf — and the
//    profiler put it at 0.71 ms/tok, 13% of the step, for trivial math. Now: block-reduced sum of
//    squares, then thread c handles the pair (c, c+half). K heads also write their roped row straight
//    into the K cache at `kc_row` and copy this head's V slice into the V cache at `vc_row`, which
//    removes the two cuMemcpyDtoD per layer (56 per token). Pass kc_row = vc_row = 0 to skip that.
//    Mirrors QK_NORM_ROPE_WGSL's math; blocks: [0, nh) are q heads, [nh, nh+nkv) are k heads.
//
//    Three optional inputs widen it past Qwen3 without touching Qwen3's arithmetic (every one of them,
//    when absent, removes an operation rather than adding a neutral one, so the NEOX/no-bias path is
//    the SAME float ops in the same order as before):
//      bias       [q_out + 2 kv_out] added to q, k AND v before anything else — Qwen2's q/k/v biases
//                 (WGSL: `qkv.add(bias)` right after the projection). Null = no bias.
//      ff         [dh/2] per-frequency MULTIPLIER on the inverse frequency — Llama-3 `rope_freqs`
//                 (already inverted at load; see qwen3.rs) or a linear factor. Mirrors ROPE_SCALED_WGSL:
//                 `exp(-2c/dh * ln base) * scale[c]`. Null = plain rope.
//      norm_pairs 1 = ggml NORM pairing, partners (2c, 2c+1) — `llama`'s GGUF rows are permuted for it;
//                 0 = NEOX split-half, partners (c, c + dh/2). The frequency index is c in both.
//                 ⛔ The wrong pairing is the classic silent RoPE failure: finite logits, fluent text.
//    ROWS: blockIdx.y is the row (prefill); row i sits at position pos + i, reads qkv row i (width
//    row_w) and writes q/k row i and cache row i. Decode launches gridDim.y = 1 (row 0, same math).
//
//    ⭐ THE ANGLES. `tab` [rows, dh] = cos of each pair's angle in [0, dh/2), sin in [dh/2, dh), built on
//    the HOST exactly as the WGSL path builds its rope table (qwen3.rs `rope_table`: the authors' float32
//    inverse frequencies from `rope_inv_freq`, angle = f32(position) · inv, cos/sin in f64 then rounded).
//    ⛔ The device formula below (`tab` null) is what this kernel did before: `expf(-2c/dh · logf(base))`
//    misses the authors' float32 inverse frequency by a few ulp, and the position multiplies that — at
//    position 30,000 the WGSL path's copy of it sat 18-109x the authors' float32-vs-float64 distance.
//    It stays reachable (FERRIC_CUDA_ROPE_DEVICE / FERRIC_ROPE_DEVICE) as the gates' negative control. ──
__device__ __forceinline__ void qk_norm_rope_row(const float* __restrict__ qkv, const float* __restrict__ qw,
                                                 const float* __restrict__ kw, float* __restrict__ qo,
                                                 float* __restrict__ ko, unsigned nh, unsigned nkv, unsigned dh,
                                                 float base, unsigned pos, float eps, unsigned q_off, unsigned k_off,
                                                 unsigned has_norm, float* __restrict__ kc_row, float* __restrict__ vc_row,
                                                 unsigned v_off, const float* __restrict__ bias,
                                                 const float* __restrict__ ff, unsigned norm_pairs,
                                                 const float* __restrict__ tab, float* red, float& s_inv) {
    const unsigned id = blockIdx.x, t = threadIdx.x;
    const bool is_k = id >= nh;
    const unsigned head = is_k ? id - nh : id;
    const unsigned so = (is_k ? k_off : q_off) + head * dh;
    const float* src = qkv + so;
    const float* bsrc = bias != 0 ? bias + so : 0;
    float* dst = (is_k ? ko : qo) + head * dh;
    const float* w = is_k ? kw : qw;
    float inv = 1.f;
    if (has_norm) {
        float v = (t < dh) ? (bsrc != 0 ? src[t] + bsrc[t] : src[t]) : 0.f;
        float ms = warp_sum(v * v);
        if ((t & 31u) == 0u) red[t >> 5u] = ms;
        __syncthreads();
        if (t == 0u) { float m = 0.f; for (unsigned i = 0; i < (blockDim.x + 31u) / 32u; ++i) m += red[i];
                       s_inv = 1.f / sqrtf(m / (float)dh + eps); }
        __syncthreads();
        inv = s_inv;
    }
    const unsigned half = dh / 2u;
    if (t < half) {
        const unsigned c = t;
        const unsigned p0 = norm_pairs ? 2u * c : c, p1 = norm_pairs ? 2u * c + 1u : c + half;
        float cs, sn;
        if (tab != 0) { cs = tab[c]; sn = tab[half + c]; }
        else {
            float fr = expf(-2.f * (float)c / (float)dh * logf(base));
            if (ff != 0) fr = fr * ff[c];
            const float ang = (float)pos * fr;
            cs = cosf(ang); sn = sinf(ang);
        }
        const float w1 = has_norm ? w[p0] : 1.f, w2 = has_norm ? w[p1] : 1.f;
        const float a = bsrc != 0 ? src[p0] + bsrc[p0] : src[p0];
        const float b = bsrc != 0 ? src[p1] + bsrc[p1] : src[p1];
        const float x1 = a * inv * w1, x2 = b * inv * w2;
        const float r1 = x1 * cs - x2 * sn, r2 = x2 * cs + x1 * sn;
        dst[p0] = r1; dst[p1] = r2;
        if (is_k && kc_row != 0) { kc_row[head * dh + p0] = r1; kc_row[head * dh + p1] = r2; }
    }
    if (is_k && vc_row != 0 && t < dh) {
        const unsigned vi = v_off + head * dh + t;
        vc_row[head * dh + t] = bias != 0 ? qkv[vi] + bias[vi] : qkv[vi];
    }
}
extern "C" __global__ void qk_norm_rope(const float* __restrict__ qkv, const float* __restrict__ qw,
                                        const float* __restrict__ kw, float* __restrict__ qo,
                                        float* __restrict__ ko, unsigned nh, unsigned nkv, unsigned dh,
                                        float base, unsigned pos, float eps, unsigned q_off, unsigned k_off,
                                        unsigned has_norm, float* __restrict__ kc_row, float* __restrict__ vc_row,
                                        unsigned v_off, const float* __restrict__ bias,
                                        const float* __restrict__ ff, unsigned norm_pairs, unsigned row_w,
                                        const float* __restrict__ tab) {
    __shared__ float red[4]; __shared__ float s_inv;
    const unsigned row = blockIdx.y;
    qkv += (size_t)row * row_w; qo += (size_t)row * nh * dh; ko += (size_t)row * nkv * dh; pos += row;
    if (kc_row != 0) kc_row += (size_t)row * nkv * dh;
    if (vc_row != 0) vc_row += (size_t)row * nkv * dh;
    if (tab != 0) tab += (size_t)row * dh;
    qk_norm_rope_row(qkv, qw, kw, qo, ko, nh, nkv, dh, base, pos, eps, q_off, k_off, has_norm, kc_row, vc_row,
                     v_off, bias, ff, norm_pairs, tab, red, s_inv);
}

// ── THE STEP BLOCK: what changes from one decode token to the next, read from DEVICE memory so a
//    CUDA graph captured once replays every later token unchanged (cuda.rs `DecodeGraph::step`).
//    u32 words: [0] the row this token writes (= rows already cached), [1] its absolute position,
//    [2..4) pad, then 2·n_layer u64 K/V base pointers (K of layer l at [l], V at [n_layer + l]) — the
//    cache is per SEQUENCE and regrows by reallocation, so its addresses cannot be graph constants.
//    The rope row and the input x row follow; the host fills the whole block and ONE copy moves it. ──
#define STEP_KVP(step) (reinterpret_cast<const unsigned long long*>((step) + 4))
extern "C" __global__ void qk_norm_rope_step(const float* __restrict__ qkv, const float* __restrict__ qw,
                                             const float* __restrict__ kw, float* __restrict__ qo,
                                             float* __restrict__ ko, unsigned nh, unsigned nkv, unsigned dh,
                                             float base, float eps, unsigned q_off, unsigned k_off,
                                             unsigned has_norm, unsigned v_off, const float* __restrict__ bias,
                                             const float* __restrict__ ff, unsigned norm_pairs,
                                             const float* __restrict__ tab, const unsigned* __restrict__ step,
                                             unsigned li, unsigned nl) {
    __shared__ float red[4]; __shared__ float s_inv;
    const unsigned row = step[0], pos = step[1];
    const size_t ro = (size_t)row * nkv * dh;
    float* kc_row = reinterpret_cast<float*>(STEP_KVP(step)[li]) + ro;
    float* vc_row = reinterpret_cast<float*>(STEP_KVP(step)[nl + li]) + ro;
    qk_norm_rope_row(qkv, qw, kw, qo, ko, nh, nkv, dh, base, pos, eps, q_off, k_off, has_norm, kc_row, vc_row,
                     v_off, bias, ff, norm_pairs, tab, red, s_inv);
}

// ── Fused single-query attention over an [S, nkv*dh] K/V cache: one block (128 threads = 4 warps)
//    per q-head, GQA head -> kv head, chunked online softmax. dh <= 128.
//    ⛔ The first version scored keys one PER THREAD (each thread a serial 128-wide dot), reduced max
//    and sum with 7-barrier trees, and accumulated V with every thread walking ALL keys serially;
//    the profiler put it at 0.41 ms/tok, the largest non-GEMV class. Now a WARP scores a key (32
//    lanes x 4 dh elements, shuffle-reduced), warps reduce max/sum with one shuffle tree each, and V is
//    accumulated per warp over a strided key subset (lane covers 4 consecutive dh elements: coalesced
//    128 B per key per warp), then combined across the 4 warps once. Mirrors FUSED_ATTN_WGSL's math. ──
__device__ __forceinline__ void attn_decode_body(const float* __restrict__ q, const float* __restrict__ k,
                                                 const float* __restrict__ v, float* __restrict__ out,
                                                 unsigned nh, unsigned nkv, unsigned dh, unsigned s, float scale) {
    __shared__ float sc[2048];
    __shared__ float red[4];
    __shared__ float vacc[4][128];
    const unsigned head = blockIdx.x, t = threadIdx.x, lane = t & 31u, warp = t >> 5u;
    const unsigned g = nh / nkv, kvh = head / g, qbase = head * dh, kvbase = kvh * dh;
    const unsigned dpl = dh / 32u;                       // dh elements per lane (dh <= 128 -> <= 4)
    // this lane's slice of q, in registers
    float qr[4] = {0.f, 0.f, 0.f, 0.f};
    for (unsigned e = 0u; e < dpl; ++e) qr[e] = q[qbase + lane * dpl + e];
    float m_run = -3.0e38f, l_run = 0.f;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};                 // per-warp partial V accumulator (lane's slice)
    for (unsigned c0 = 0u; c0 < s; c0 += 2048u) {
        const unsigned clen = min(2048u, s - c0);
        // scores: warp w takes keys w, w+4, ...
        for (unsigned i = warp; i < clen; i += 4u) {
            const float* kr = k + (c0 + i) * nkv * dh + kvbase + lane * dpl;
            float d = 0.f;
            for (unsigned e = 0u; e < dpl; ++e) d += qr[e] * kr[e];
            d = warp_sum(d);
            if (lane == 0u) sc[i] = d * scale;
        }
        __syncthreads();
        // chunk max
        float cm = -3.0e38f;
        for (unsigned i = t; i < clen; i += 128u) cm = fmaxf(cm, sc[i]);
        cm = warp_max(cm);
        if (lane == 0u) red[warp] = cm;
        __syncthreads();
        const float m_new = fmaxf(m_run, fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3])));
        const float corr = expf(m_run - m_new);
        __syncthreads();
        // exponentiate + chunk sum
        float cs = 0.f;
        for (unsigned i = t; i < clen; i += 128u) { const float e = expf(sc[i] - m_new); sc[i] = e; cs += e; }
        cs = warp_sum(cs);
        if (lane == 0u) red[warp] = cs;
        __syncthreads();
        l_run = l_run * corr + red[0] + red[1] + red[2] + red[3];
        // V: warp w accumulates keys w, w+4, ... over its lane's dh slice
        for (unsigned e = 0u; e < dpl; ++e) acc[e] *= corr;
        for (unsigned i = warp; i < clen; i += 4u) {
            const float p = sc[i];
            const float* vr = v + (c0 + i) * nkv * dh + kvbase + lane * dpl;
            for (unsigned e = 0u; e < dpl; ++e) acc[e] += p * vr[e];
        }
        m_run = m_new;
        __syncthreads();
    }
    for (unsigned e = 0u; e < dpl; ++e) vacc[warp][lane * dpl + e] = acc[e];
    __syncthreads();
    if (t < dh) out[head * dh + t] = (vacc[0][t] + vacc[1][t] + vacc[2][t] + vacc[3][t]) / l_run;
}
extern "C" __global__ void attn_decode(const float* __restrict__ q, const float* __restrict__ k,
                                       const float* __restrict__ v, float* __restrict__ out,
                                       unsigned nh, unsigned nkv, unsigned dh, unsigned s, float scale) {
    attn_decode_body(q, k, v, out, nh, nkv, dh, s, scale);
}
// The graph's attention: this layer's K/V base and the key count come from the step block.
extern "C" __global__ void attn_decode_step(const float* __restrict__ q, float* __restrict__ out,
                                            unsigned nh, unsigned nkv, unsigned dh, float scale,
                                            const unsigned* __restrict__ step, unsigned li, unsigned nl) {
    attn_decode_body(q, reinterpret_cast<const float*>(STEP_KVP(step)[li]),
                     reinterpret_cast<const float*>(STEP_KVP(step)[nl + li]), out, nh, nkv, dh, step[0] + 1u, scale);
}


// ═══════════ OPT-IN (FERRIC_CUDA_Q8X): int8 activations + dp4a integer dots for the Q5_K GEMVs ═══════════
// This is llama.cpp's mul_mat_vec_q technique. ⚠ Unlike every other kernel here it CHANGES THE NUMBERS,
// not just their order: the activation row is quantised to int8 per 32-value block (~0.4% per element).
// It therefore stays behind an explicit switch with its own fingerprint; the harness's identical-ids
// gate and the tolerance test below are what say whether the trade is acceptable.

// x[n] -> xq (4 int8 per u32, 8 words per 32-value block), xs[n/32] scales. One warp per block.
extern "C" __global__ void quant_x_q8(const float* __restrict__ x, int* __restrict__ xq,
                                      float* __restrict__ xs, unsigned n) {
    const unsigned b = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u), lane = threadIdx.x & 31u;
    if (b * 32u >= n) return;
    const float v = x[b * 32u + lane];
    const float am = warp_max(fabsf(v));
    const float inv = am > 0.f ? 127.f / am : 0.f;
    const int q = __float2int_rn(v * inv);
    const int q1 = __shfl_down_sync(0xffffffffu, q, 1), q2 = __shfl_down_sync(0xffffffffu, q, 2), q3 = __shfl_down_sync(0xffffffffu, q, 3);
    if ((lane & 3u) == 0u) xq[b * 8u + (lane >> 2u)] = (q & 0xff) | ((q1 & 0xff) << 8) | ((q2 & 0xff) << 16) | ((q3 & 0xff) << 24);
    if (lane == 0u) xs[b] = am / 127.f;
}
// Same lane partition as q5k_dot_lane; the x side is int8 and the dots are dp4a.
__device__ __forceinline__ float q5k_dot_lane_q8(const int* __restrict__ xq, const float* __restrict__ xs,
                                                 const unsigned* __restrict__ codes, const unsigned* __restrict__ aux,
                                                 unsigned o, unsigned nblk, unsigned bl, unsigned j) {
    const unsigned c = j >> 1u, half = j & 1u, s0 = 2u * c, s1 = s0 + 1u;
    float acc = 0.f;
    for (unsigned blk = bl; blk < nblk; blk += 4u) {
        const unsigned bi = o * nblk + blk, ab = bi * 4u, cb40 = bi * 40u;
        const unsigned dd = aux[ab];
        const float d = f16_to_f32(dd & 0xffffu), dmin = f16_to_f32(dd >> 16u);
        float ds0, mm0, ds1, mm1;
        q5_scmin(aux, ab, s0, d, dmin, ds0, mm0);
        q5_scmin(aux, ab, s1, d, dmin, ds1, mm1);
        const uint4 q = *reinterpret_cast<const uint4*>(codes + cb40 + 8u * c + 4u * half);
        const uint4 h = *reinterpret_cast<const uint4*>(codes + cb40 + 32u + 4u * half);
        const uint4 xa = *reinterpret_cast<const uint4*>(xq + (blk * 8u + s0) * 8u + 4u * half);
        const uint4 xb = *reinterpret_cast<const uint4*>(xq + (blk * 8u + s1) * 8u + 4u * half);
        const float sx0 = xs[blk * 8u + s0], sx1 = xs[blk * 8u + s1];
        const unsigned qw[4] = {q.x, q.y, q.z, q.w}, hw[4] = {h.x, h.y, h.z, h.w};
        const int xaw[4] = {(int)xa.x, (int)xa.y, (int)xa.z, (int)xa.w}, xbw[4] = {(int)xb.x, (int)xb.y, (int)xb.z, (int)xb.w};
        int d0 = 0, n0 = 0, d1 = 0, n1 = 0;
        #pragma unroll
        for (unsigned w = 0u; w < 4u; ++w) {
            const unsigned word = qw[w], qhw = hw[w];
            const int qlo = (int)((word & 0x0f0f0f0fu)         | (((qhw >> s0) & 0x01010101u) << 4u));
            const int qhi = (int)(((word >> 4u) & 0x0f0f0f0fu) | (((qhw >> s1) & 0x01010101u) << 4u));
            d0 = __dp4a(xaw[w], qlo, d0); n0 = __dp4a(xaw[w], 0x01010101, n0);
            d1 = __dp4a(xbw[w], qhi, d1); n1 = __dp4a(xbw[w], 0x01010101, n1);
        }
        acc += sx0 * (ds0 * (float)d0 - mm0 * (float)n0) + sx1 * (ds1 * (float)d1 - mm1 * (float)n1);
    }
    return acc;
}
extern "C" __global__ void q5k_gemv_q8(const int* __restrict__ xq, const float* __restrict__ xs,
                                       const unsigned* __restrict__ codes, const unsigned* __restrict__ aux,
                                       float* __restrict__ out, unsigned o_dim, unsigned in_dim) {
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    const unsigned o = blockIdx.x * 4u + warp;
    if (o >= o_dim) return;
    float acc = q5k_dot_lane_q8(xq, xs, codes, aux, o, in_dim / 256u, lane >> 3u, lane & 7u);
    acc = warp_sum(acc);
    if (lane == 0u) out[o] = acc;
}
extern "C" __global__ void q5k_swiglu_gemv_q8(const int* __restrict__ xq, const float* __restrict__ xs,
                                              const unsigned* __restrict__ codes, const unsigned* __restrict__ aux,
                                              float* __restrict__ out, unsigned n_ff, unsigned in_dim) {
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    const unsigned o = blockIdx.x * 4u + warp;
    if (o >= n_ff) return;
    const unsigned nblk = in_dim / 256u, sl = lane & 7u, bl = lane >> 3u;
    float g = q5k_dot_lane_q8(xq, xs, codes, aux, o, nblk, bl, sl);
    float u = q5k_dot_lane_q8(xq, xs, codes, aux, o + n_ff, nblk, bl, sl);
    g = warp_sum(g); u = warp_sum(u);
    if (lane == 0u) out[o] = (g / (1.f + expf(-g))) * u;
}
