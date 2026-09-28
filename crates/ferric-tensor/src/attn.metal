// Causal prefill attention on the Metal-4 matrix units: S = Q·Kᵀ and O += P·V as
// `mpp::tensor_ops::matmul2d` 16x32x16 fragments, one simdgroup per 16 query rows, online softmax in fp32.
//
// Dispatched through wgpu as a PASSTHROUGH compute pipeline (see `native_attn.rs`), binding the same
// buffers the WGSL flash kernels bind:  0 q [T, nh·dh]  1 k [off+T, nkv·dh]  2 v [off+T, nkv·dh]
// 3 out [T, nh·dh] (all f32)  4 dims.
//
// Schedule (FlashAttention-2, as in MLX's steel attention_nax — a spec, not an oracle):
//   * rows are flattened per KV head as r = i·g + h (query position i, head h of the g = nh/nkv heads
//     that share a K/V head), 64 rows per threadgroup (4 simdgroups x 16), so each K/V fragment read
//     serves the whole GQA group and a block spans ~64/g positions of the causal diagonal;
//   * each simdgroup walks the keys in 32-key steps up to ITS OWN last row's causal limit (no barrier
//     couples the simdgroups, so none waits for the diagonal of another);
//   * per step: S (16x32) = Q (16xDH) · Kᵀ in DH/16 matmuls; scale, causal mask, row max / rescale /
//     exp2 / row sum on the fragment registers (4 lanes share a row: shuffle-xor 1 and 8); O (16xDH) +=
//     P (16x32) · V (32xDH).
//
// ⚠ PRECISION CONTRACT: Q, K and V enter the matrix units as fp16 (each f32 value rounded once); S, the
// softmax statistics, P and O accumulate in fp32 (P enters the P·V matmul as f32 — MPP's float x half
// combination). `native_attn.rs` measures what that costs against float64 and against the authors.
//
// The fragment layout (lane -> rows fm and fm+8, columns fn..fn+3 of a 16x16 tile) is the one MLX's
// steel/attn/nax.h (BaseNAXFrag) relies on for matmul2d<16,32,16> on one simdgroup. It is
// implementation-defined; the float64 test in native_attn.rs is what shows it holds on this device.
//
// ATTN_FAULT (set by FERRIC_ATTN_FAULT / the tests) plants one plausible defect — the negative control:
//   1 the online-softmax rescale dropped   2 the causal limit ignores the cache offset
//   3 the ragged last key step dropped
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

#ifndef DH
#define DH 64
#endif
#ifndef ATTN_FAULT
#define ATTN_FAULT 0
#endif
#ifndef P_T
#define P_T half
#endif
#ifndef RELAXED
#define RELAXED false
#endif
#define NSG 4
#define BQ (16 * NSG)
#define BK 32
#define TD (DH / 16)

struct Dims { uint nh; uint nkv; uint T; uint off; uint scale_bits; uint nblk; uint window; uint softcap_bits; };

// One score, in the log2 domain the softmax runs in: scaled, Gemma-2's softcap (cap·tanh(x/cap), the
// composed path's order) when cap > 0, and -FLT_MAX where the key is masked — causal (key c visible to
// the query at position lim iff c <= lim) and, with a sliding window, c + win > lim.
static inline float score(float s, uint c, uint lim, float scale, float cap, uint win) {
    if (c > lim || (win > 0 && c + win <= lim)) { return -FLT_MAX; }
    float x = s * scale;
    if (cap > 0.0f) { x = cap * precise::tanh(x / cap); }
    return x * M_LOG2E_F;
}

typedef vec<float, 8> ffrag;
typedef vec<half, 8> hfrag;

// C(16x32) += A(16x16) · B(16x32) on one simdgroup, operands in the fragment layout above. With TB the
// right operand is given as two 16x16 fragments of Bᵀ (rows = the 32 output columns).
template <typename CT, typename AT, typename BT, bool TB>
static inline void mma(thread vec<CT, 8>& c0, thread vec<CT, 8>& c1, thread const vec<AT, 8>& a,
                       thread const vec<BT, 8>& b0, thread const vec<BT, 8>& b1) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, TB, RELAXED, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroup> op;
    auto ca = op.template get_left_input_cooperative_tensor<AT, BT, CT>();
    auto cb = op.template get_right_input_cooperative_tensor<AT, BT, CT>();
    auto cc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ca)>,
                                                              metal::remove_addrspace_t<decltype(cb)>, CT>();
    for (short i = 0; i < 8; i++) { ca[i] = a[i]; }
    for (short i = 0; i < 8; i++) { cb[i] = b0[i]; cb[8 + i] = b1[i]; }
    for (short i = 0; i < 8; i++) { cc[i] = c0[i]; cc[8 + i] = c1[i]; }
    op.run(ca, cb, cc);
    for (short i = 0; i < 8; i++) { c0[i] = cc[i]; c1[i] = cc[8 + i]; }
}

// One 16x16 fragment of a row-major f32 matrix with row stride `ld`, rows ra (elements 0..3) and rb
// (4..7), 4 columns from `col`; a row at or past `nrow` reads as zero (never out of the buffer).
static inline hfrag load_frag(device const float* p, uint ra, uint rb, uint ld, uint col, uint nrow) {
    float4 a = ra < nrow ? *(device const float4*)(p + ra * ld + col) : float4(0);
    float4 b = rb < nrow ? *(device const float4*)(p + rb * ld + col) : float4(0);
    hfrag f;
    f[0] = half(a.x); f[1] = half(a.y); f[2] = half(a.z); f[3] = half(a.w);
    f[4] = half(b.x); f[5] = half(b.y); f[6] = half(b.z); f[7] = half(b.w);
    return f;
}

kernel void flash_attn_mu(device const float* q    [[buffer(0)]],
                          device const float* k    [[buffer(1)]],
                          device const float* v    [[buffer(2)]],
                          device float*       out  [[buffer(3)]],
                          constant Dims&      d    [[buffer(4)]],
                          uint2  tg   [[threadgroup_position_in_grid]],
                          ushort sg   [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]])
{
    const uint nh = d.nh, nkv = d.nkv, T = d.T, off = d.off;
    const uint g = nh / nkv, kvh = tg.y, nrows = T * g, S = off + T, ld = nkv * DH;
    const uint r0 = (d.nblk - 1 - tg.x) * BQ + sg * 16;          // heaviest blocks first
    if (r0 >= nrows) { return; }                                 // a whole simdgroup past the end
    const float scale = as_type<float>(d.scale_bits), cap = as_type<float>(d.softcap_bits);
    const uint win = d.window;
    const short qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    // This lane's two rows (fm, fm+8); rows past the end compute on the last row and are not stored.
    const uint ra = r0 + fm, rb = ra + 8;
    const uint ca = min(ra, nrows - 1), cb = min(rb, nrows - 1);
    const uint ia = ca / g, ib = cb / g;
    const uint qoa = (ia * nh + kvh * g + ca % g) * DH, qob = (ib * nh + kvh * g + cb % g) * DH;
#if ATTN_FAULT == 2
    const uint lim_a = ia, lim_b = ib;                           // NEGATIVE CONTROL ONLY
#else
    const uint lim_a = off + ia, lim_b = off + ib;
#endif
    // Keys this simdgroup needs: up to its last row's causal limit.
    const uint nkeys = off + (min(r0 + 15, nrows - 1) / g) + 1;
#if ATTN_FAULT == 3
    const uint nkb = nkeys / BK;                                 // NEGATIVE CONTROL ONLY
#else
    const uint nkb = (nkeys + BK - 1) / BK;
#endif

    // Sliding window: no row of this simdgroup sees a key at or below off + i_lo - win; skip those steps.
    const uint i_lo = min(r0, nrows - 1) / g;
    const uint kb0 = (win > 0 && off + i_lo + 1 > win) ? (off + i_lo + 1 - win) / BK : 0;

    hfrag Q[TD];
    for (short id = 0; id < TD; id++) { Q[id] = load_frag(q, qoa / DH, qob / DH, DH, id * 16 + fn, 0xffffffffu); }
    ffrag O[TD];
    for (short id = 0; id < TD; id++) { O[id] = ffrag(0); }
    float m_a = -FLT_MAX, m_b = -FLT_MAX, l_a = 0, l_b = 0;

    for (uint kb = kb0; kb < nkb; kb++) {
        const uint k0 = kb * BK;
        const uint kcol = kvh * DH + fn;
        ffrag S0 = ffrag(0), S1 = ffrag(0);
        for (short id = 0; id < TD; id++) {
            hfrag K0 = load_frag(k, k0 + fm, k0 + fm + 8, ld, kcol + id * 16, S);
            hfrag K1 = load_frag(k, k0 + 16 + fm, k0 + 24 + fm, ld, kcol + id * 16, S);
            mma<float, half, half, true>(S0, S1, Q[id], K0, K1);
        }
        float mx_a = -FLT_MAX, mx_b = -FLT_MAX;
        for (short j = 0; j < 4; j++) {
            const uint c0 = k0 + fn + j, c1 = c0 + 16;
            S0[j]     = score(S0[j], c0, lim_a, scale, cap, win);
            S0[4 + j] = score(S0[4 + j], c0, lim_b, scale, cap, win);
            S1[j]     = score(S1[j], c1, lim_a, scale, cap, win);
            S1[4 + j] = score(S1[4 + j], c1, lim_b, scale, cap, win);
            mx_a = max(mx_a, max(S0[j], S1[j]));
            mx_b = max(mx_b, max(S0[4 + j], S1[4 + j]));
        }
        mx_a = max(mx_a, simd_shuffle_xor(mx_a, ushort(1))); mx_a = max(mx_a, simd_shuffle_xor(mx_a, ushort(8)));
        mx_b = max(mx_b, simd_shuffle_xor(mx_b, ushort(1))); mx_b = max(mx_b, simd_shuffle_xor(mx_b, ushort(8)));
        const float mn_a = max(m_a, mx_a), mn_b = max(m_b, mx_b);
#if ATTN_FAULT == 1
        const float fa = 1.0f, fb = 1.0f;                        // NEGATIVE CONTROL ONLY
#else
        const float fa = exp2(m_a - mn_a), fb = exp2(m_b - mn_b);
#endif
        m_a = mn_a; m_b = mn_b;
        float sa = 0, sb = 0;
        for (short j = 0; j < 4; j++) {
            S0[j] = exp2(S0[j] - mn_a); S1[j] = exp2(S1[j] - mn_a);
            S0[4 + j] = exp2(S0[4 + j] - mn_b); S1[4 + j] = exp2(S1[4 + j] - mn_b);
            sa += S0[j] + S1[j]; sb += S0[4 + j] + S1[4 + j];
        }
        sa += simd_shuffle_xor(sa, ushort(1)); sa += simd_shuffle_xor(sa, ushort(8));
        sb += simd_shuffle_xor(sb, ushort(1)); sb += simd_shuffle_xor(sb, ushort(8));
        l_a = l_a * fa + sa; l_b = l_b * fb + sb;
        for (short id = 0; id < TD; id++) {
            for (short j = 0; j < 4; j++) { O[id][j] *= fa; O[id][4 + j] *= fb; }
        }
        vec<P_T, 8> P0, P1;
        for (short i = 0; i < 8; i++) { P0[i] = P_T(S0[i]); P1[i] = P_T(S1[i]); }
        for (short id = 0; id < TD; id += 2) {
            hfrag V0 = load_frag(v, k0 + fm, k0 + fm + 8, ld, kcol + id * 16, S);
            hfrag V1 = load_frag(v, k0 + fm, k0 + fm + 8, ld, kcol + id * 16 + 16, S);
            mma<float, P_T, half, false>(O[id], O[id + 1], P0, V0, V1);
            hfrag V2 = load_frag(v, k0 + 16 + fm, k0 + 24 + fm, ld, kcol + id * 16, S);
            hfrag V3 = load_frag(v, k0 + 16 + fm, k0 + 24 + fm, ld, kcol + id * 16 + 16, S);
            mma<float, P_T, half, false>(O[id], O[id + 1], P1, V2, V3);
        }
    }

    const float inv_a = 1.0f / l_a, inv_b = 1.0f / l_b;
    for (short id = 0; id < TD; id++) {
        if (ra < nrows) { *(device float4*)(out + qoa + id * 16 + fn) = float4(O[id][0], O[id][1], O[id][2], O[id][3]) * inv_a; }
        if (rb < nrows) { *(device float4*)(out + qob + id * 16 + fn) = float4(O[id][4], O[id][5], O[id][6], O[id][7]) * inv_b; }
    }
}
