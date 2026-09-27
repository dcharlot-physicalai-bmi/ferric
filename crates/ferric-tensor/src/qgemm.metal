// Metal-4 tensor-unit GEMM over Ferric's packed quantized weights (Q8_0, Q5_0, Q4_K, Q6_K):
// y[M,N] = x[M,K] · W[N,K]ᵀ.
//
// Dispatched through wgpu as a PASSTHROUGH compute pipeline (see `native_qgemm.rs`), so it binds the
// same wgpu buffers every WGSL kernel binds and records into the same batched compute pass — no
// second queue, no host round trip, no copy of the weights. Binding order = buffer index:
//   0 x (f32, row-major [M,K])   1 codes (u32)   2 aux/scales (u32)   3 y (f32, [M,N])   4 dims
//
// Per threadgroup: a MT×NT output tile, K walked in KT=32 slices. Each slice the 128 threads
//   (a) stage x[MT,32] into threadgroup memory as half,
//   (b) dequantize W[NT,32] from the packed blocks into threadgroup memory as half,
//   (c) hand both to `mpp::tensor_ops::matmul2d` (fp16 inputs, fp32 accumulate in a cooperative
//       tensor that lives across the whole K walk),
// then the cooperative tensor is stored once, bounds-checked by the destination's extents.
//
// `QGEMM_FAULT` (set by FERRIC_QGEMM_FAULT, see native_qgemm.rs) plants one plausible decoding error
// per format — the whole-model gate's negative control, which must be seen to fail.
//
// ⚠ PRECISION CONTRACT: both operands enter the matrix units as fp16. The weight is exactly its
// quantized value rounded ONCE to fp16 (q·d is formed in f32 first); the activation is its f32 value
// rounded once to fp16. Accumulation is fp32. `native_qgemm.rs` measures what that costs.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#ifndef MT
#define MT 64
#endif
#ifndef NT
#define NT 64
#endif
#define KT 32
#define NTHR 128

struct Dims { uint M; uint N; uint K; uint act; };

// ---- per-format tile dequant: 16 consecutive weights of row `n`, starting at column `k` (a multiple
// of 16), written as half to `dst`. Layouts are the repacked GPU buffers built in dtype.rs. ----

// Q8_0: codes = 8 words (32 int8) per 32-block, scales = two f16 per word, blocks row-major.
static inline void dq16_q8_0(device const uint* codes, device const uint* aux, uint n, uint k, uint K,
                             threadgroup half* dst) {
    uint blk = n * (K / 32) + k / 32;
#ifdef QGEMM_FAULT
    uint sb = blk ^ 1; // NEGATIVE CONTROL ONLY: the neighbouring block's scale
#else
    uint sb = blk;
#endif
    float dsc = float(as_type<half2>(aux[sb >> 1])[sb & 1]);
    uint w0 = blk * 8 + ((k & 31) >> 2);
    for (uint w = 0; w < 4; ++w) {
        char4 q = as_type<char4>(codes[w0 + w]);
        *((threadgroup half4*)dst + w) = half4(float4(q) * dsc);
    }
}

// Q5_0: codes = 4 words (16 bytes of nibble pairs) per 32-block, aux = [qh, d] per block. Value j
// (0..15) is the low nibble of byte j with bit j of qh as its 5th bit; value 16+j the high nibble with
// bit 16+j; both minus 16, times d.
static inline void dq16_q5_0(device const uint* codes, device const uint* aux, uint n, uint k, uint K,
                             threadgroup half* dst) {
    uint blk = n * (K / 32) + k / 32;
    uint hh = (k & 31) >> 4;
    uint qh = aux[2 * blk] >> (16 * hh);
#ifdef QGEMM_FAULT
    qh = 0; // NEGATIVE CONTROL ONLY: the fifth bit dropped
#endif
    float d = float(as_type<half2>(aux[2 * blk + 1]).x);
    for (uint w = 0; w < 4; ++w) {
        uint word = codes[blk * 4 + w] >> (4 * hh), hb = qh >> (4 * w);
        int4 q = int4((word & 0xF) | ((hb & 1) << 4), ((word >> 8) & 0xF) | (((hb >> 1) & 1) << 4),
                      ((word >> 16) & 0xF) | (((hb >> 2) & 1) << 4), ((word >> 24) & 0xF) | (((hb >> 3) & 1) << 4)) - 16;
        *((threadgroup half4*)dst + w) = half4(float4(q) * d);
    }
}

// Q4_K: codes = 32 words (128 nibble bytes) per 256-super-block, aux = 4 words [d|dmin<<16, 12 scale
// bytes]. Sub-block s (32 values) lives in words 8*(s/2).., low nibble for even s, high for odd.
static inline uint sbyte(device const uint* aux, uint ab, uint i) { return (aux[ab + 1 + (i >> 2)] >> (8 * (i & 3))) & 0xff; }
static inline void dq16_q4_k(device const uint* codes, device const uint* aux, uint n, uint k, uint K,
                             threadgroup half* dst) {
    uint bi = n * (K / 256) + k / 256;
    uint s = (k & 255) >> 5, l0 = k & 31;
    uint ab = bi * 4;
    float2 dm = float2(as_type<half2>(aux[ab]));
    uint sc, mn;
    if (s < 4) { sc = sbyte(aux, ab, s) & 63; mn = sbyte(aux, ab, s + 4) & 63; }
    else {
        uint a = sbyte(aux, ab, s + 4), lo = sbyte(aux, ab, s - 4), hi = sbyte(aux, ab, s);
        sc = (a & 0x0F) | ((lo >> 6) << 4); mn = (a >> 4) | ((hi >> 6) << 4);
    }
    float ds = dm.x * float(sc), mm = dm.y * float(mn);
#ifdef QGEMM_FAULT
    mm = 0.0f; // NEGATIVE CONTROL ONLY: the sub-block minimum dropped
#endif
    uint cw = bi * 32 + 8 * (s >> 1) + (l0 >> 2);
    uint sh = (s & 1) * 4;
    for (uint w = 0; w < 4; ++w) {
        uint word = codes[cw + w] >> sh;
        float4 q = float4(word & 0xF, (word >> 8) & 0xF, (word >> 16) & 0xF, (word >> 24) & 0xF);
        *((threadgroup half4*)dst + w) = half4(ds * q - mm);
    }
}

// Q6_K: codes = 48 words per 256-super-block (32 words ql, 16 words qh), aux = 5 words [d, 16 int8
// scales]. Half hf of 128 values: ql bytes 64·hf.., qh bytes 32·hf.., scales 8·hf..; quadrant qd
// (32 values) takes ql byte l+32·(qd&1) (nibble qd>>1), qh bits 2·qd, scale l/16 + 2·qd.
static inline void dq16_q6_k(device const uint* codes, device const uint* aux, uint n, uint k, uint K,
                             threadgroup half* dst) {
    uint bi = n * (K / 256) + k / 256;
    uint p = k & 255, hf = p >> 7, qd = (p & 127) >> 5, l0 = p & 31;
    uint cb = bi * 48, ab = bi * 5;
    float d = float(as_type<half2>(aux[ab]).x);
    uint si = 8 * hf + (l0 >> 4) + 2 * qd;
    int scb = int(as_type<char4>(aux[ab + 1 + (si >> 2)])[si & 3]);
    float ds = d * float(scb);
    uint qlw = cb + ((64 * hf + 32 * (qd & 1) + l0) >> 2);
    uint qhw = cb + 32 + ((32 * hf + l0) >> 2);
    uint lsh = (qd >> 1) * 4, hsh = 2 * qd;
#ifdef QGEMM_FAULT
    hsh = (hsh + 2) & 7; // NEGATIVE CONTROL ONLY: the high bits taken from the wrong position
#endif
    for (uint w = 0; w < 4; ++w) {
        uint lw = codes[qlw + w] >> lsh, hw = codes[qhw + w] >> hsh;
        int4 q = int4((lw & 0xF) | ((hw & 3) << 4), ((lw >> 8) & 0xF) | (((hw >> 8) & 3) << 4),
                      ((lw >> 16) & 0xF) | (((hw >> 16) & 3) << 4), ((lw >> 24) & 0xF) | (((hw >> 24) & 3) << 4)) - 32;
        *((threadgroup half4*)dst + w) = half4(float4(q) * ds);
    }
}

// SWI = 0: y = x·Wᵀ, [M, N].
// SWI = 1: fused FFN gate|up + SwiGLU for a gate_up weight [2·n_ff, K] (n_ff = d.act): each tile's 64
//          weight rows are 32 gate rows j0.. and the SAME 32 up rows n_ff+j0.., so the epilogue has
//          both halves in hand and writes silu(gate)·up straight to y[M, n_ff] — the [M, 2·n_ff]
//          intermediate is never written and the separate swiglu dispatch is gone.
#define QMM_KERNEL(NAME, DQ, SWI)                                                                      \
kernel void NAME(device const float* x     [[buffer(0)]],                                              \
                 device const uint*  codes [[buffer(1)]],                                              \
                 device const uint*  aux   [[buffer(2)]],                                              \
                 device float*       out   [[buffer(3)]],                                              \
                 constant Dims&      d     [[buffer(4)]],                                              \
                 uint2  tg [[threadgroup_position_in_grid]],                                           \
                 ushort t  [[thread_index_in_threadgroup]])                                            \
{                                                                                                      \
    threadgroup half sx[MT * KT];                                                                      \
    threadgroup half sw[NT * KT];                                                                      \
    threadgroup float sc[SWI ? MT * NT : 1]; /* the SwiGLU epilogue's gate|up tile (unused if !SWI) */  \
    const uint m0 = tg.y * MT;                                                                         \
    const uint n0 = SWI ? tg.x * (NT / 2) : tg.x * NT;                                                 \
    auto tX = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sx, dextents<int32_t, 2>(KT, MT)); \
    auto tW = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sw, dextents<int32_t, 2>(KT, NT)); \
    constexpr auto desc = matmul2d_descriptor(MT, NT, KT, false, true, false,                         \
                                              matmul2d_descriptor::mode::multiply_accumulate);          \
    matmul2d<desc, execution_simdgroups<4>> mm;                                                        \
    auto cT = mm.get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();             \
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) { if (cT.is_valid_element(i)) cT[i] = 0; }        \
    for (uint k0 = 0; k0 < d.K; k0 += KT) {                                                            \
        for (uint e = t; e < MT * KT / 4; e += NTHR) {                                                 \
            uint r = e / (KT / 4), c4 = e % (KT / 4);                                                  \
            uint gm = m0 + r;                                                                          \
            float4 v = gm < d.M ? *((device const float4*)(x + gm * d.K + k0) + c4) : float4(0);       \
            *((threadgroup half4*)(sx + r * KT) + c4) = half4(v);                                      \
        }                                                                                              \
        for (uint e = t; e < NT * 2; e += NTHR) {                                                      \
            uint r = e >> 1, h = e & 1;                                                                \
            uint wr = SWI ? (r < NT / 2 ? min(n0 + r, d.act - 1) : d.act + min(n0 + r - NT / 2, d.act - 1)) \
                          : min(n0 + r, d.N - 1);                                                      \
            DQ(codes, aux, wr, k0 + 16 * h, d.K, sw + r * KT + 16 * h);                                \
        }                                                                                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                               \
        mm.run(tX, tW, cT);                                                                            \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                               \
    }                                                                                                  \
    if (!SWI) {                                                                                        \
        auto tO = tensor<device float, dextents<int32_t, 2>, tensor_inline>(out, dextents<int32_t, 2>(d.N, d.M)); \
        auto tOs = tO.slice(n0, m0);                                                                   \
        cT.store(tOs);                                                                                 \
    } else {                                                                                           \
        auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc, dextents<int32_t, 2>(NT, MT)); \
        cT.store(tC);                                                                                  \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                               \
        for (uint e = t; e < MT * (NT / 2); e += NTHR) {                                               \
            uint r = e / (NT / 2), c = e % (NT / 2);                                                   \
            if (m0 + r < d.M && n0 + c < d.act) {                                                      \
                float g = sc[r * NT + c], u = sc[r * NT + NT / 2 + c];                                 \
                out[(m0 + r) * d.act + n0 + c] = (g / (1.0f + exp(-g))) * u;                           \
            }                                                                                          \
        }                                                                                              \
    }                                                                                                  \
}

QMM_KERNEL(qmm_q8_0, dq16_q8_0, 0)
QMM_KERNEL(qmm_q4_k, dq16_q4_k, 0)
QMM_KERNEL(qmm_q6_k, dq16_q6_k, 0)
QMM_KERNEL(qmm_q5_0, dq16_q5_0, 0)
QMM_KERNEL(qmm_swiglu_q8_0, dq16_q8_0, 1)
QMM_KERNEL(qmm_swiglu_q5_0, dq16_q5_0, 1)
QMM_KERNEL(qmm_swiglu_q4_k, dq16_q4_k, 1)
QMM_KERNEL(qmm_swiglu_q6_k, dq16_q6_k, 1)
