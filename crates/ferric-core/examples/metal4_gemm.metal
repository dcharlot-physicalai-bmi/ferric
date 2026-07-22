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
