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

## Fault injection

`docs/fleet/smolvla-fault-injection.json` injects a fixed catalogue, seed `0x5a17`, into the CPU kernels and scores each output against this raw-action bound and against the mapped per-stage bound. A low-mantissa flip stays under both. That is the same limit arXiv:2609.00363 measured for a one-spacing tolerance: the check does not fire on a one-spacing fault. That paper is not about SmolVLA. The added checks are a per-stage FNV digest of the f32 bits, which sees those flips, and a matmul row/column checksum, which sees a dropped product term. There is no quantized GEMM on this path.

## RoPE invariant

A RoPE pair is `y0 = x0 cos θ − x1 sin θ`, `y1 = x1 cos θ + x0 sin θ`, with `θ = pos / base^(2c/dh)` and `base = 10000`. Two properties follow from that, not from a measured run.

The dataflow of one component is two multiplies and one add, and sin and cos each add one unit roundoff on those multiplies. That is five roundings, `γ_5 = 5u / (1 − 5u)`. The Euclidean norm of the two component errors is at most `√2 γ_5 (|x0| + |x1|)`. A faithful rotation stays inside that distance of the input pair's norm.

Recovering the angle is another five roundings, `γ_5` radians, and the f32 power and division that form `θ` admit a relative error `γ_5`, which is an absolute error `γ_5 |θ|`. The angle check allows `γ_5 (1 + |θ|)`. A pair whose squared radius is at most `u` does not determine an angle, so that check is skipped there.

`rope` in the CPU forward calls this check and refuses the forward if it fails. The seeded catalogue's dropped and duplicated cross terms fail it. The low-mantissa flip does not: one ulp of a normal value is inside `γ_5`, which is the legal gap between two f32 evaluations. Tightening the bound below one ulp would also reject a clean `sin` that is off by one ulp, so 9/9 is not available from this invariant. The stage digest is what sees that flip.

No other kernel in the catalogue has a lower end-to-end rate than its per-stage rate, so the same kind of check was not added elsewhere.

Measured on clean rotations, not used to choose the bound: llvmpipe `rope_positions` on the seeded fixture had norm residual 1.222e-7 against a bound of 2.141e-6, and angle residual 4.891e-7 against 2.831e-6. The reference `rope` in `scripts/smolvla_f32_reference.py`, on 25 tensors of shape `[1, 4, 2, 64]`, had max norm residual 4.272e-6 and max angle residual 1.196e-5, both inside the per-pair bounds. The full pinned-weight forward was not re-executed here; `model.safetensors` is not on this machine.
