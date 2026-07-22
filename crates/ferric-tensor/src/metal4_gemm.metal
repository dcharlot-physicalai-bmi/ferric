#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void matMul(tensor<device half,  dextents<int32_t, 2>> A,
                   tensor<device half,  dextents<int32_t, 2>> B,
                   tensor<device float, dextents<int32_t, 2>> C,
                   uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr auto desc = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent));
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tA = A.slice(0, tgid.y*64);
    auto tB = B.slice(tgid.x*32, 0);
    auto tC = C.slice(tgid.x*32, tgid.y*64);
    op.run(tA, tB, tC);
}

// NT variant: y = x·Wᵀ with W in the HF [out,in] layout, consumed directly via transpose_right —
// no transpose materialization. B extents are [k, np] (k innermost = W's row layout), tile slice
// shifts the N dimension (extent 1).
kernel void matMulBT(tensor<device half,  dextents<int32_t, 2>> A,
                     tensor<device half,  dextents<int32_t, 2>> B,
                     tensor<device float, dextents<int32_t, 2>> C,
                     uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr auto desc = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent),
                                              false, /* transpose_left  (N) */
                                              true); /* transpose_right (T) */
    matmul2d<desc, execution_simdgroups<4>> op;
    auto tA = A.slice(0, tgid.y*64);
    auto tB = B.slice(0, tgid.x*32);
    auto tC = C.slice(tgid.x*32, tgid.y*64);
    op.run(tA, tB, tC);
}

// GPU-side f32→f16 pad-convert for the resident path: copy `count` source elements into a padded
// destination whose blocks are stretched srcBlock→dstBlock (A: m·k→mp·k per batch; B: n→np per row;
// the NT weight copy is the identity map srcBlock == dstBlock == count, pads being the tail rows).
// Pad elements are zeroed once at cache build and never touched here.
kernel void padConvert(device const float* src [[buffer(0)]],
                       device half*        dst [[buffer(1)]],
                       constant uint*      p   [[buffer(2)]], // count, srcBlock, dstBlock
                       uint gid [[thread_position_in_grid]])
{
    if (gid >= p[0]) return;
    dst[(gid / p[1]) * p[2] + gid % p[1]] = half(src[gid]);
}

// Activation epilogue codes match the WGSL matmul_bt_act contract (0 id, 1 relu, 2 silu, 3 gelu,
// 4 sigmoid); gelu uses the same Abramowitz-Stegun erf approximation for cross-path parity.
static float applyAct(float v, uint a)
{
    switch (a) {
        case 1: return max(v, 0.0f);
        case 2: return v / (1.0f + exp(-v));
        case 3: {
            float x = v * 0.7071067811865476f;
            float t = 1.0f / (1.0f + 0.3275911f * abs(x));
            float e = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-x * x);
            return 0.5f * v * (1.0f + (v >= 0.0f ? e : -e));
        }
        case 4: return 1.0f / (1.0f + exp(-v));
        default: return v;
    }
}

// NHWC spatial pad-convert for the resident conv path: f32 [n,h,w,c] → f16 [n,hp,wp,c] with the
// data region placed at (ph,pw). Pad elements are zeroed once at cache build.
kernel void padConvertNHWC(device const float* src [[buffer(0)]],
                           device half*        dst [[buffer(1)]],
                           constant uint*      p   [[buffer(2)]], // count,h,w,c,hp,wp,ph,pw
                           uint gid [[thread_position_in_grid]])
{
    if (gid >= p[0]) return;
    uint ci = gid % p[3]; uint r = gid / p[3];
    uint x = r % p[2]; r = r / p[2];
    uint y = r % p[1]; uint b = r / p[1];
    dst[((b * p[4] + y + p[6]) * p[5] + x + p[7]) * p[3] + ci] = half(src[gid]);
}

// Inverse for C: gather the [m,n] data region out of the padded [mp,np] result, per batch, applying
// the fused activation epilogue on the way out.
kernel void unpad(device const float* src [[buffer(0)]],
                  device float*       dst [[buffer(1)]],
                  constant uint*      p   [[buffer(2)]], // count, n, np, m, mp, act
                  uint gid [[thread_position_in_grid]])
{
    if (gid >= p[0]) return;
    uint row = gid / p[1], col = gid % p[1];
    uint srow = (row / p[3]) * p[4] + row % p[3];
    dst[gid] = applyAct(src[srow * p[2] + col], p[5]);
}
