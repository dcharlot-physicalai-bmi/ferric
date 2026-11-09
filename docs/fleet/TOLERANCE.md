# SmolVLA f32 forward-error bound

This bound is the pairwise difference between two f32 evaluations of the same graph. It is not fitted to RADV, llvmpipe, or the CPU run.

`u = 2^-23`. A length-K matmul whose terms are size `1/sqrt(K)` (so the output is order 1) has an accumulation-order difference of `u * sqrt(K)`. An unfused multiply adds another `u * sqrt(K)`. The per-element bound is therefore `2 * u * sqrt(K)`. Softmax and RMSNorm use the same form on their reduction length. GELU contributes `4u`.

Inside a layer the terms add in root-sum-square. Across residual layers they add in root-sum-square too, with Jacobian 1. Ten Euler steps of `dt = 0.1` add in root-sum-square after scaling by `0.1`. A max-abs over N values multiplies by `sqrt(2 ln N)`.

The machine-readable result is `docs/fleet/smolvla-forward-error-bounds.json`. `ferric_smolvla::within_bound` grades a measured max-abs against it.

Recorded max-abs, not used to choose the bound:

| stage | CPU vs PyTorch | llvmpipe vs CPU | RADV vs PyTorch | bound | result |
| --- | ---: | ---: | ---: | ---: | --- |
| raw actions | 3.70e-6 | 4.29e-6 | 9.36e-6 | 1.95e-4 | all three pass |

The full per-stage table is produced by the tolerance unit test. RADV does not fail this bound. It does exceed the smaller CPU-achieved envelope on `vision_post_token0`, `text_l0_k_token0`, and `raw_actions`. That envelope is not the tolerance.
