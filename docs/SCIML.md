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
