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

// ── rmsnorm: out = x * rsqrt(mean(x^2) + eps) * w.  One block of 256 threads, any d. ──
extern "C" __global__ void rmsnorm(const float* __restrict__ x, const float* __restrict__ w,
                                   float* __restrict__ out, unsigned d, float eps) {
    __shared__ float red[8];
    float ms = 0.f;
    for (unsigned j = threadIdx.x; j < d; j += blockDim.x) { const float v = x[j]; ms += v * v; }
    ms = block_sum256(ms, red);
    const float inv = 1.f / sqrtf(ms / (float)d + eps);
    for (unsigned j = threadIdx.x; j < d; j += blockDim.x) out[j] = x[j] * inv * w[j];
}
// ── add_rmsnorm: sum = x + y (the next residual); norm = rmsnorm(sum) * w. One block. ──
extern "C" __global__ void add_rmsnorm(const float* __restrict__ x, const float* __restrict__ y,
                                       const float* __restrict__ w, float* __restrict__ sum,
                                       float* __restrict__ norm, unsigned d, float eps) {
    __shared__ float red[8];
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
//    Mirrors WGSL Q6_K_BODY (dtype.rs) exactly: element order hf x l, four products per l. ──
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
extern "C" __global__ void q6k_gemv(const float* __restrict__ x, const unsigned* __restrict__ codes,
                                    const unsigned* __restrict__ aux, float* __restrict__ out,
                                    unsigned o_dim, unsigned in_dim) {
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    const unsigned o = blockIdx.x * 4u + warp;
    if (o >= o_dim) return;                          // warp-uniform
    const unsigned nblk = in_dim / 256u;
    // 4 block-lanes x 8 sub-lanes; sub-lane owns hf = sl>>2 and l in [(sl&3)*8, +8): 2*32 (hf,l) pairs / 8.
    const unsigned sl = lane & 7u, bl = lane >> 3u;
    const unsigned hf = sl >> 2u, l0 = (sl & 3u) * 8u;
    float acc = 0.f;
    for (unsigned blk = bl; blk < nblk; blk += 4u) {
        const unsigned bi = o * nblk + blk, cb = bi * 48u, ab = bi * 5u;
        const float d = f16_to_f32(aux[ab] & 0xffffu);
        const float* xb = x + blk * 256u;
        const unsigned qlo = 64u * hf, qho = 32u * hf, sco = 8u * hf, xh = 128u * hf;
        #pragma unroll
        for (unsigned l = l0; l < l0 + 8u; ++l) {
            const unsigned is = l >> 4u, h = q6_qhb(codes, cb, qho + l);
            const int q1 = (int)((q6_qlb(codes, cb, qlo + l) & 0xFu)        | ((h & 3u) << 4u))         - 32;
            const int q2 = (int)((q6_qlb(codes, cb, qlo + l + 32u) & 0xFu)  | (((h >> 2u) & 3u) << 4u))  - 32;
            const int q3 = (int)((q6_qlb(codes, cb, qlo + l) >> 4u)         | (((h >> 4u) & 3u) << 4u))  - 32;
            const int q4 = (int)((q6_qlb(codes, cb, qlo + l + 32u) >> 4u)   | (((h >> 6u) & 3u) << 4u))  - 32;
            acc += xb[xh + l]        * d * q6_scb(aux, ab, sco + is)      * (float)q1;
            acc += xb[xh + 32u + l]  * d * q6_scb(aux, ab, sco + is + 2u) * (float)q2;
            acc += xb[xh + 64u + l]  * d * q6_scb(aux, ab, sco + is + 4u) * (float)q3;
            acc += xb[xh + 96u + l]  * d * q6_scb(aux, ab, sco + is + 6u) * (float)q4;
        }
    }
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

// ── QK-norm (optional) + NEOX RoPE for ONE row (decode). One thread per head: nh q-heads then nkv
//    k-heads, reading q/k in place from the fused qkv buffer at q_off / k_off. Mirrors QK_NORM_ROPE_WGSL. ──
extern "C" __global__ void qk_norm_rope(const float* __restrict__ qkv, const float* __restrict__ qw,
                                        const float* __restrict__ kw, float* __restrict__ qo,
                                        float* __restrict__ ko, unsigned nh, unsigned nkv, unsigned dh,
                                        float base, unsigned pos, float eps, unsigned q_off, unsigned k_off,
                                        unsigned has_norm) {
    const unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nh + nkv) return;
    const bool is_k = id >= nh;
    const unsigned head = is_k ? id - nh : id;
    const float* src = qkv + (is_k ? k_off : q_off) + head * dh;
    float* dst = (is_k ? ko : qo) + head * dh;
    const float* w = is_k ? kw : qw;
    float inv = 1.f;
    if (has_norm) { float ms = 0.f; for (unsigned j = 0; j < dh; ++j) ms += src[j] * src[j];
                    inv = 1.f / sqrtf(ms / (float)dh + eps); }
    const unsigned half = dh / 2u;
    const float lb = logf(base);
    for (unsigned c = 0; c < half; ++c) {
        const float fr = expf(-2.f * (float)c / (float)dh * lb);
        const float ang = (float)pos * fr, cs = cosf(ang), sn = sinf(ang);
        const float w1 = has_norm ? w[c] : 1.f, w2 = has_norm ? w[c + half] : 1.f;
        const float x1 = src[c] * inv * w1, x2 = src[c + half] * inv * w2;
        dst[c] = x1 * cs - x2 * sn; dst[c + half] = x2 * cs + x1 * sn;
    }
}

// ── Fused single-query attention over an [S, nkv*dh] K/V cache. One block (128 threads) per q-head,
//    GQA head -> kv head, chunked online softmax. Mirrors FUSED_ATTN_WGSL; dh <= 128. ──
extern "C" __global__ void attn_decode(const float* __restrict__ q, const float* __restrict__ k,
                                       const float* __restrict__ v, float* __restrict__ out,
                                       unsigned nh, unsigned nkv, unsigned dh, unsigned s, float scale) {
    __shared__ float qs[128]; __shared__ float sc[2048]; __shared__ float red[128];
    const unsigned head = blockIdx.x, t = threadIdx.x;
    const unsigned g = nh / nkv, kvh = head / g, qbase = head * dh, kvbase = kvh * dh;
    if (t < dh) qs[t] = q[qbase + t];
    __syncthreads();
    float m_run = -3.0e38f, l_run = 0.f, accd = 0.f;
    for (unsigned c0 = 0; c0 < s; c0 += 2048u) {
        const unsigned clen = min(2048u, s - c0);
        for (unsigned i = t; i < clen; i += 128u) {
            float dot = 0.f; const unsigned kb = (c0 + i) * nkv * dh + kvbase;
            for (unsigned d = 0; d < dh; ++d) dot += qs[d] * k[kb + d];
            sc[i] = dot * scale;
        }
        __syncthreads();
        float cm = -3.0e38f;
        for (unsigned i = t; i < clen; i += 128u) cm = fmaxf(cm, sc[i]);
        red[t] = cm; __syncthreads();
        for (unsigned st = 64u; st > 0u; st >>= 1u) { if (t < st) red[t] = fmaxf(red[t], red[t + st]); __syncthreads(); }
        const float m_new = fmaxf(m_run, red[0]); const float corr = expf(m_run - m_new); __syncthreads();
        float cs = 0.f;
        for (unsigned i = t; i < clen; i += 128u) { const float e = expf(sc[i] - m_new); sc[i] = e; cs += e; }
        red[t] = cs; __syncthreads();
        for (unsigned st = 64u; st > 0u; st >>= 1u) { if (t < st) red[t] += red[t + st]; __syncthreads(); }
        l_run = l_run * corr + red[0];
        if (t < dh) { float a = 0.f; for (unsigned i = 0; i < clen; ++i) a += sc[i] * v[(c0 + i) * nkv * dh + kvbase + t];
                      accd = accd * corr + a; }
        m_run = m_new;
        __syncthreads();
    }
    if (t < dh) out[head * dh + t] = accd / l_run;
}
