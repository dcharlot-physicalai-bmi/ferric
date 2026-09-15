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

## The benchmark harness — the field's problems, scored against independent references

`sciml::harness` solves the standard problems by a named recipe and scores them against a reference the
network never sees, so a number from this stack is comparable to one from PINNacle or jinns. The references
(`sciml::bench`) are themselves tested: each closed form is checked against its PDE by finite differences,
and Burgers' Cole–Hopf integral — the one that is not a closed form — is checked the same way plus against
its initial condition and boundaries (worst FD residual 1.75e-7, shock formed by `t = 1`).

| problem | equation | reference |
|---|---|---|
| `Heat` | `u_t = u_xx` on `[0,1]`, Dirichlet | `e^{−π²t} sin πx` |
| `Helmholtz` | `Δu + k²u = q` on `[−1,1]²`, `u = 0` on `∂Ω` | manufactured `sin(a₁πx) sin(a₂πy)` (Wang 2021's `a = (1, 4)`) |
| `Burgers` | `u_t + u u_x = ν u_xx`, `ν = 0.01/π` | Cole–Hopf, Hermite–Gauss form (Basdevant 1986) |
| `Advection` | `u_t + β u_x = 0`, periodic, `β = 30` | `sin(x − βt)` (Krishnapriyan 2021's hard case) |

`Recipe::vanilla()` is a tanh MLP with Adam and a fixed boundary weight; `Recipe::full()` is Fourier
features, gradient-norm balancing, Adam, then strong-Wolfe L-BFGS. `the_full_recipe_beats_vanilla_on_every_benchmark`
prints the table and asserts the ordering; `refinement_beats_uniform_sampling_on_the_burgers_shock_at_equal_budget`
is the RAR fixture in the regime where it can be shown to help — DeepXDE's own numbers for this problem,
2540 uniform against 2000 + 540 refined, both arms with the L-BFGS stage.

## Domains, boundaries and boundary conditions

`sciml::geometry` is the ergonomic layer every PINN user meets first. `BoxDomain` and `Disk` are the
primitives; `Union`, `Difference` and `Intersection` compose them (constructive solid geometry), with the
composite's boundary taken from the parts' boundaries filtered by membership in the other part — an annulus
is a disk minus a disk, and its boundary is the outer circle plus the inner circle with the normal flipped.
`interior` samples a shape by rejection; `boundary_f32` returns boundary points with outward unit normals;
`dirichlet`, `neumann`, `robin` and `periodic` turn the conditions into loss terms, with `normal_derivative`
underneath the flux ones.

⭐ Every sampler is checked against its own geometry: a boundary point stepped a little along its normal
must leave the shape and stepped against it must stay inside — on the primitives and on every composite
(`primitives_and_csg_composites_have_outward_normals_and_consistent_membership`). That check is what gives a
Neumann condition its sign; a normal pointing the wrong way is a flux condition of the wrong sign, and no
loss curve would show it. Two ignored GPU oracles close the loop with closed forms: Laplace on the annulus
(`u = ln(r/½)/ln 2`) through CSG + Dirichlet, and the 1-D Neumann sign test (`u'' = 0`, `u(0) = 0`,
`u'(1) = 1` ⇒ `u = x`; a flipped normal would return `−x`).

## The certificate, extended to PDEs — the piece the incumbents do not ship

`sciml::certify` turns a trained network's residual into a bound on the error it cannot see. For
`−Δe − k²e = r` in `Ω`, `e = g` on `∂Ω` — the error of an approximation to Poisson (`k = 0`) or Helmholtz —
split `e = e₁ + e₂` with `e₁` carrying the residual and zero boundary data and `e₂` the boundary correction
with zero residual; then `‖e₁‖ ≤ ‖r‖ / (λ₁ − k²)` by the spectral gap and `‖e₂‖ ≤ |Ω|^{1/2} ‖g‖_∞` by the
maximum principle, so

```text
‖e‖_L²(Ω) ≤ ‖r‖_L²(Ω) / (λ₁ − k²) + |Ω|^{1/2} · ‖g‖_L∞(∂Ω)        (λ₁ the first Dirichlet eigenvalue; k² < λ₁)
```

Both inputs come from the network alone. The bound is **sound** and, on the first eigenfunction, **tight**:
`the_bound_is_sharp_on_the_first_eigenfunction` checks that on `ε sin πx sin πy` the bound equals the error
(0.150000 against 0.150000). Scope, stated: elliptic, linear, coercive — the domain of the Mishra–Molinaro
error-versus-residual results (2006.16144). Hyperbolic and nonlinear problems have no bound of this form
here and none is claimed. `the_certificate_bounds_a_trained_poisson_pinn_from_its_residual_alone` runs it on a
trained net: a Fourier-feature PINN on the unit-square Poisson problem, 4000 Adam steps with balancing, gives
`‖r‖ = 1.44`, boundary max `2.5e-3`, hence **‖e‖ ≤ 7.56e-2**, against a true error of **8.25e-3** — sound, and
9.2× the truth rather than a vacuous number. The residual norm dominates the bound, which is the honest
reading: the certificate says what the network still owes the equation, and a tighter bound is earned by
training the residual down, not by argument.

## A 2-D spectral operator

`sciml::operators::SpectralConv2d` is the Fourier-neural-operator layer on a periodic `n × n` grid. The fabric
has no complex dtype and no FFT, so the transform is the DFT matrix applied by matmul, as the 1-D `fno`
example does, lifted to 2-D by the Kronecker product `F₂ = F₁ ⊗ F₁` (an `n² × n²` matrix; 256 × 256 at
`n = 16`). The layer keeps the `modes × modes` lowest frequencies and multiplies each by a learned complex
weight — channel-diagonal, the form in which a constant-coefficient linear solution operator is exactly
representable, which is what lets `a_2d_spectral_operator_recovers_the_poisson_greens_function` check the
learned weights against `1/(4π²|k|²)`. Trained on random band-limited forcings of the periodic Poisson
problem: held-out rel-L2 **0.0026**, and the learned multiplier matches the Green's function on every
retained mode to within **0.82 %** (k = (0,1): 0.02533 against 0.02533; (0,2): 0.00633 against 0.00633).
`the_kronecker_dft_inverts_itself` checks `F⁻¹F = I` on random fields (rel-L2 1.9e-7).

⛔ **The weights live on a half-spectrum, and they have to.** A real field's DFT is conjugate-symmetric, so a
layer with independent weights at `k` and `−k` can only identify `R(k) + conj(R(−k))`: the first version
learned the operator to 1.4 % and its individual weights matched nothing (k = (1,1): 0.0433 against 0.0127).
Parameters are now one complex weight per half-spectrum mode, expanded with the symmetry built in — half
the parameters, and each one means one thing. A second training stage at a decayed rate is what brings the
smallest multipliers (~20× below the largest) from 12 % to under 1 %; they sit at Adam's noise floor otherwise.

⚠ The multi-channel FNO — a `w × w` complex matrix per mode plus pointwise channel mixing — is the follow-on,
not this layer.

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
