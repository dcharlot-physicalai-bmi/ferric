# Scientific ML on the Ferric fabric — a pure-Rust PINN & neural-operator stack

Physics-informed neural networks and neural operators, trained **GPU-native in pure Rust** on the Ferric
fabric — no Python, no autodiff library. This fills a real gap: every serious PINN/operator stack today is
Python (DeepXDE, NVIDIA PhysicsNeMo, jax-pi) or Julia (NeuralPDE.jl); there was **no pure-Rust / WebGPU
one**. The enabling primitive is Ferric's differentiable second-order `grad()` (each op carries a
Var-valued VJP), which supplies the derivatives PINNs need — including the input second derivatives that
make PINNs the hardest autodiff case.

## The library — `ferric_tensor::sciml`

- **`Siren`** — a sine-activation MLP (SIREN). Sine represents a function *and* its derivatives well, so it
  is the natural PINN network; `sin` is a native second-order Var op here. `Siren::new(ctx, dims, seed)` /
  `.vars()` / `Siren::forward(pv, x)`.
- **`deriv(y, x)`** — the JAX-style differentiable derivative: `u' = deriv(u, x)`, `u'' = deriv(u', x)`.
  Built on `grad()`, so it composes to any order and stays differentiable wrt the parameters (train through it).
- Test: `cargo test -p ferric-tensor sciml` — trains a PINN from the residual and asserts convergence.

## Examples (each verified against exact or held-out ground truth, on Metal)

| Example | What it is | Verified result |
|---|---|---|
| `pinn_siren` | 1-D ODE PINN, `u''+ω²u=0` from the residual alone; **self-certifying** (computes an a-posteriori error bound from its own residual) | max err vs `cos ωt` = **0.0002**; certificate `‖e‖ ≤ 0.024`, **sound** |
| `pinn_poisson2d` | 2-D **PDE** PINN, Poisson `∇²u=f` on the unit square (Laplacian via per-axis grad-of-grad) | rel-L2 vs exact = **0.43 %** |
| `deeponet` | **DeepONet** neural operator, learns the antiderivative operator `G[f]=∫₀ˣf` | held-out rel-L2 = **4.3 %** |
| `fno` | **Fourier Neural Operator**, learns the 1-D Poisson solution operator; spectral conv via DFT-as-matmul | held-out rel-L2 = **0.002 %**; recovers Green's function `Rr_k = 1/k²` exactly |

Run any with `cargo run --release --example <name>`.

## The training recipe — what separates a PINN that works from one that returns a flat line

Architectures were never the gap between this stack and DeepXDE / PhysicsNeMo / jaxpi / PINA; the
**training recipe** was. Each piece below is implemented from its paper, and each ships with an oracle in
`sciml/oracles.rs` of the same shape: the vanilla loss on a problem where it measurably **fails**, then the
technique on the same problem, same net, same optimiser, same steps. A technique that only matches its
baseline is decoration; a fixture on which the baseline already succeeds proves nothing — several fixtures
had their difficulty raised until the control broke, and the numbers below are what was measured.

| piece | source | oracle | control → technique |
|---|---|---|---|
| `FourierNet` — random Fourier features, multi-scale | Tancik 2020 (2006.10739); Wang–Wang–Perdikaris 2021 (2012.10047) | `u''+ω²u=0` at ω = 10, tanh MLP vs Fourier net, same width and steps | rel-L2 **0.958 → 0.0017** |
| `Lbfgs` — strong-Wolfe L-BFGS | Nocedal & Wright Alg. 7.4/7.5 + 3.5/3.6 | 1-D Poisson: Adam to its plateau, then L-BFGS from the same point | loss **4.9e-3 → 1.2e-4** in 300 iterations |
| inverse problems — a PDE coefficient as a trained parameter | Raissi et al. 2019 | heat equation, `D = 0.7` unknown, 30 measurements at 1 % noise, `D` started at 0.2 | **D = 0.6985 (0.21 % off)** |
| physics-informed DeepONet | Wang–Wang–Perdikaris 2021 (2103.10974) | antiderivative operator from `∂ₓG[f] = f`, `G[f](0) = 0` — no solution data | held-out rel-L2 **0.035** (the data-trained example: 0.035) |
| `LossBalancer` — gradient-norm balancing | Wang–Teng–Perdikaris 2021 (2001.04536) | 1-D Poisson `u'' = −(6π)² sin 6πx`, tanh MLP: the residual's gradients drown the boundary's | rel-L2 **0.464 → 0.0044**, λ_bc → 9.9e3 |
| `Causal` — causal training in time | Wang–Sankaran–Perdikaris 2022 (2203.07404) | reaction equation `u_t = 10u(1−u)`, where a vanilla PINN lands on a wrong branch (Krishnapriyan 2021) | rel-L2 **0.937 → 0.094 and 0.117** on two runs of the same configuration (GPU run-to-run variance; asserted at 0.15) |
| `rar_select` — residual-based adaptive refinement | Lu et al. 2021 (1907.04502) | selects the layer of `tanh(k(x−½))` (≥12 of 16 picks inside it) | ⚠ primitive verified; **benefit not demonstrated** — see below |
| `Rba` — residual-based attention (per-point weights) | Anagnostopoulos et al. 2023 (2307.00379) | bookkeeping test; no fix-oracle yet | — |

Also: `Act::{Tanh, Sin, Relu}` on `Mlp::forward_act`, `mse`, `scalar`, and `flatten`/`unflatten` for moving
parameters through an L-BFGS closure.

**Where refinement stands.** A sweep over `k ∈ {20, 30}` and budgets `{64, 128}`, uniform against 75 % uniform +
25 % refined at equal budget, found refinement never lower than uniform (0.170 vs 0.177, 0.536 vs 0.613, …) and the
error tracking `k` rather than the budget: with Adam alone the regime is optimisation-limited, and no placement
of points helps until it is sampling-limited. DeepXDE's demonstration uses thousands of base points plus an
L-BFGS stage on Burgers. That fixture is the open item; the sweep is kept as `fixture_sweep_for_rar_and_causal`.

**Running the oracles.** The six training oracles train two nets each on the GPU and take about 40
minutes together, so they are `#[ignore]`d out of the default lane:

```
cargo test --release -p ferric-tensor sciml -- --ignored
```

The default lane keeps the cheap proofs: the harmonic-oscillator PINN, the Rosenbrock and ill-conditioned
quadratic L-BFGS checks, the causal/RAR/RBA bookkeeping, and the activation gradient check below.

**One safeguard the balancing paper does not have.** Implemented as printed, `λ̂ = max|∇L_r| / mean|∇L_bc|`
has no floor: once the boundary term is satisfied to float precision its gradient is ~0 and the weight ran
to **1.4 × 10¹²**, after which any boundary deviation of 1e-8 produced an O(1e4) gradient — the balanced
run came out at rel-L2 0.93 against 0.03 unweighted. The denominator is now floored at `1e-4 × max|∇L_r|`,
capping the weight at 1e4; on the oracle above the weight settles at 9.9e3, so the cap was doing work.

### ⛔ Two defects the recipe exposed in the fabric itself

**`tanh` had no differentiable VJP, and `grad()` was silent about it.** `Var::tanh` was built with
`Var::node` (first-order only) while `sin`/`exp`/`sqrt` use `node_d`. The functional `grad()` — which
`deriv` is built on — *dropped* the gradient at any node without a VJP, so `deriv(tanh(x), x)` returned
**zero at every point** and every tanh-activated physics-informed net trained to nothing while reporting a
finite loss. Five oracles failed at once (the unknown coefficient sat at exactly its initial value); the
one thing they shared was the activation. A finite-difference check gave the sentence:
`tanh'(−1.3) = 0.2574 but the VJP gave 0`. `tanh` is now `node_d`, `grad()` **panics** when a gradient
reaches a first-order-only op instead of returning zeros, and
`every_activation_differentiates_correctly_to_second_order` checks `tanh`/`sin`/`exp`/`sqrt` against finite
differences to second order, including the parameter gradient of a loss built on the derivative. The
whole crate's suite (92 tests) passes with the loud `grad()` — nothing relied on the silence.

Ops safe under `deriv` (they carry a differentiable VJP): `add sub mul div matmul relu neg transpose reshape
sum sum_all exp log sin cos tanh sqrt`. Not safe: `cat narrow broadcast_to silu rmsnorm conv2d rope
selective_scan contiguous` — which is why `FourierNet` feeds `sin(xB)` and `cos(xB)` through two weight
matrices rather than concatenating them.

**L-BFGS needs a Wolfe line search, not Armijo.** With backtracking alone, Rosenbrock stalled at
`f = 3.47` for 200 iterations: from iteration 3 every accepted step had `sᵀy ≈ −5e-7`, so every curvature
pair was rejected, the history froze, and its direction collapsed to `|d| ≈ 1.8e-3` — accepted at unit
step every time. The Wolfe curvature condition makes `sᵀy > 0` by construction; with it, 36 iterations,
49 evaluations, `f = 0` at `(1, 1)`.

## Design notes

- **PINN loss = physics, no data.** Minimize the PDE residual at collocation points plus the boundary/initial
  conditions. The residual needs the network's own input derivatives (`u'`, `u''`, `∇²u`) — obtained with
  `deriv`, then the whole residual loss is differentiated wrt the parameters (`loss.backward()` runs through
  the `deriv` computation: training-through-differentiation).
- **Self-certification.** For a well-posed linear problem the solution error `e` obeys the same equation
  forced by the trained net's own residual `r`, giving a computable a-posteriori bound
  `‖e‖∞ ≤ |e₀| + |e₁|/ω + (1/ω)∫|r|` (Grönwall / Mishra–Molinaro form) — the fabric trains a PINN *and*
  certifies it. Sound in-distribution / for the well-posed regime; state the domain of validity.
- **Operators vs PINNs.** A PINN solves one instance; an operator learns the whole solution *map* (one
  forward pass per new input, no re-solving) — the primitive for real-time / parametric / many-query use.
- **FNO without FFT.** Ferric has no FFT/complex, but the DFT is a matmul, so the spectral conv is
  DFT-as-matmul with a real/imag split. `O(n²)` vs `O(n log n)` — FFT is the asymptotic speedup only, and at
  these grid sizes it does not matter; the learned per-mode weights are identical either way.

## Honest scope & what's not here

- **Nano / demo scale, single-seed, Metal-verified.** These are correctness demonstrations of the mechanism,
  not benchmarks. On tiny problems a learned surrogate is not faster than the true solver — the large
  operator/surrogate speedups (≈466× soft-robot MPC, ≈44,000× Cosserat-rod) are real only for stiff / soft /
  high-DOF plants, and are cited, not claimed here.
- **Joules-per-solve is not measured** — Apple Silicon exposes no RAPL; an honest per-solve energy number
  needs the external-meter / Jetson path (see `FABRIC.md`). Not fabricated.
- **Remaining:** an FFT primitive (to make the FNO asymptotically fast), a separable PINN (SPINN) for
  higher dimensions, and a WebGPU **in-browser** build — the same fabric runs in-browser (Bonsai does), but
  in-browser *training* + second-order `grad()` on WebGPU is unproven and needs a de-risking spike first.
