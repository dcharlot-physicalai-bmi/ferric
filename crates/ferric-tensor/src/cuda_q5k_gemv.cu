// Ferric NVIDIA tier, kernel 1: warp-cooperative Q5_K GEMV (decode, rows == 1).
//
// Reads the SAME repacked buffers the WGSL kernel reads (see Q5_KWeights::from_bytes):
//   aux  : 4 u32/block  = [d|dmin as two f16] [12 scale bytes as 3 words]
//   codes: 40 u32/block = [qs: 32 words (128 B)] [qh: 8 words (32 B)]
// so the host repack is shared and ONLY the kernel differs across fabrics.
//
// Lane plan mirrors the WGSL two-level split: 32 lanes per output = 4 block-lanes x 8 sub-block
// lanes; 4 outputs per 128-thread block; warp-shuffle reduction (no shared memory, no barrier).
// No cuBLAS, no CUTLASS, no headers: f16 decode is done by hand so the file is self-contained and
// compiles with plain `nvcc -arch=compute_70 -ptx`.
//
// Build (on any machine with the CUDA toolkit; the DRIVER alone loads the resulting PTX):
//   nvcc -O3 -arch=compute_70 -ptx cuda_q5k_gemv.cu -o cuda_q5k_gemv.ptx

__device__ __forceinline__ float f16_to_f32(unsigned h) {
    const unsigned s = (h >> 15u) & 1u, e = (h >> 10u) & 0x1fu, m = h & 0x3ffu;
    unsigned bits;
    if (e == 0u) {
        if (m == 0u) { bits = s << 31u; }
        else {                                   // subnormal: normalise
            unsigned mm = m, ee = 127u - 15u + 1u;
            while ((mm & 0x400u) == 0u) { mm <<= 1u; --ee; }
            mm &= 0x3ffu;
            bits = (s << 31u) | (ee << 23u) | (mm << 13u);
        }
    } else if (e == 31u) { bits = (s << 31u) | 0x7f800000u | (m << 13u); }
    else { bits = (s << 31u) | ((e + 127u - 15u) << 23u) | (m << 13u); }
    return __uint_as_float(bits);
}

__device__ __forceinline__ unsigned scbyte(const unsigned* __restrict__ aux, unsigned ab, unsigned i) {
    return (aux[ab + 1u + (i >> 2u)] >> (8u * (i & 3u))) & 0xffu;
}

extern "C" __global__ void q5k_gemv(const float*    __restrict__ x,
                                    const unsigned* __restrict__ codes,
                                    const unsigned* __restrict__ aux,
                                    float*          __restrict__ out,
                                    unsigned o_dim, unsigned in_dim) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5u;
    const unsigned o = blockIdx.x * 4u + warp;
    // Warp-uniform (every lane of a warp shares `o`), so the early return keeps the full mask
    // valid for the shuffles below.
    if (o >= o_dim) return;
    const unsigned nblk = in_dim / 256u;
    const unsigned sl = lane & 7u;       // sub-block lane: which of the 8 32-value sub-blocks
    const unsigned bl = lane >> 3u;      // block lane: 4 of them stride the blocks
    float acc = 0.f;
    for (unsigned blk = bl; blk < nblk; blk += 4u) {
        const unsigned bi = o * nblk + blk, ab = bi * 4u, cb40 = bi * 40u;
        const unsigned dd = aux[ab];
        const float d = f16_to_f32(dd & 0xffffu), dmin = f16_to_f32(dd >> 16u);
        const unsigned s = sl;
        unsigned sc, mn;                 // == WGSL scmin(ab, s)
        if (s < 4u) { sc = scbyte(aux, ab, s) & 63u; mn = scbyte(aux, ab, s + 4u) & 63u; }
        else {
            const unsigned a = scbyte(aux, ab, s + 4u), lo = scbyte(aux, ab, s - 4u), hi = scbyte(aux, ab, s);
            sc = (a & 0x0Fu) | ((lo >> 6u) << 4u); mn = (a >> 4u) | ((hi >> 6u) << 4u);
        }
        const float ds = d * (float)sc, mm = dmin * (float)mn;
        const unsigned cw = cb40 + 8u * (s >> 1u), hi = s & 1u;
        const float* xs = x + blk * 256u + 32u * s;                 // 16-B aligned: multiples of 4 floats
        #pragma unroll
        for (unsigned w = 0u; w < 8u; ++w) {
            const unsigned word = codes[cw + w], qhw = codes[cb40 + 32u + w];
            const float4 xw = *reinterpret_cast<const float4*>(xs + 4u * w);
            float n0, n1, n2, n3;
            if (hi == 0u) { n0 = (float)(word & 0xfu);         n1 = (float)((word >> 8u) & 0xfu);
                            n2 = (float)((word >> 16u) & 0xfu); n3 = (float)((word >> 24u) & 0xfu); }
            else          { n0 = (float)((word >> 4u) & 0xfu);  n1 = (float)((word >> 12u) & 0xfu);
                            n2 = (float)((word >> 20u) & 0xfu); n3 = (float)((word >> 28u) & 0xfu); }
            const float b0 = (float)((qhw >> s) & 1u) * 16.f,         b1 = (float)((qhw >> (8u + s)) & 1u) * 16.f;
            const float b2 = (float)((qhw >> (16u + s)) & 1u) * 16.f, b3 = (float)((qhw >> (24u + s)) & 1u) * 16.f;
            acc += ds * (xw.x * (n0 + b0) + xw.y * (n1 + b1) + xw.z * (n2 + b2) + xw.w * (n3 + b3))
                 - mm * (xw.x + xw.y + xw.z + xw.w);
        }
    }
    #pragma unroll
    for (unsigned off = 16u; off > 0u; off >>= 1u) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0u) out[o] = acc;
}
