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

// GPU-side f32→f16 pad-convert for the resident path: copy `count` source elements into a padded
// destination whose blocks are stretched srcBlock→dstBlock (A: m·k→mp·k per batch; B: n→np per row).
// Pad elements are zeroed once at cache build and never touched here.
kernel void padConvert(device const float* src [[buffer(0)]],
                       device half*        dst [[buffer(1)]],
                       constant uint*      p   [[buffer(2)]], // count, srcBlock, dstBlock
                       uint gid [[thread_position_in_grid]])
{
    if (gid >= p[0]) return;
    dst[(gid / p[1]) * p[2] + gid % p[1]] = half(src[gid]);
}

// Inverse for C: gather the [m,n] data region out of the padded [mp,np] result, per batch.
kernel void unpad(device const float* src [[buffer(0)]],
                  device float*       dst [[buffer(1)]],
                  constant uint*      p   [[buffer(2)]], // count, n, np, m, mp
                  uint gid [[thread_position_in_grid]])
{
    if (gid >= p[0]) return;
    uint row = gid / p[1], col = gid % p[1];
    uint srow = (row / p[3]) * p[4] + row % p[3];
    dst[gid] = src[srow * p[2] + col];
}
