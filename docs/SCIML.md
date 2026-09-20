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
features, gradient-norm balancing, Adam, then strong-Wolfe L-BFGS. Measured (2000 collocation points,
4000 Adam steps, + 500 L-BFGS for the full recipe, scored on a 51² grid):

| problem | vanilla | full (isotropic σ) | full (per-axis σ) |
|---|---|---|---|
| heat | 0.0448 | 0.0074 | **0.0082** |
| burgers (ν = 0.01/π) | **0.2079** | 0.5294 | 0.2434 |
| advection (β = 30) | 0.9144 | 1.0815 | 0.9712 |
| helmholtz (a = (1, 4), k = 1) | 0.4766 | 0.4781 | **0.3066** |

⛔ **The helmholtz row measured the formulation, not the PDE.** Every number in it uses a *soft* boundary
penalty, so the recipe has to balance a residual term against a boundary term — which is what the Fourier
features, the gradient-norm and NTK balancers and the L-BFGS polish are all working on. Hard-constrain the
boundary instead, `u = (1−x²)(1−y²)·M(x,y)`, exactly zero on all four edges by construction, and a plain
`[2,20,20,1]` tanh MLP — **501 parameters, no Fourier features, no balancer, no L-BFGS** — reaches
**0.0059 / 0.0064** at two seeds on 900 points with Adam alone. That is ~50× better than the best recipe in
the row above. ⚠ It is not a controlled recipe comparison (different net, different point count, different
formulation); it is evidence that the binding constraint here was the loss balancing that a soft penalty
forces on you.

⛔⛔ **And it does not generalise — the advection row was tried next and hard constraints made it worse.**
`u_t + β u_x = 0` at `β = 30`: same net body, same 4096 points, same steps, same seed, only the treatment of
the initial and periodic conditions differing. Soft penalties **0.8098**; hard constraints
(`u = sin x + t·M(sin x, cos x, t)`, exactly `sin x` at `t = 0` and exactly `2π`-periodic for every `M`)
**0.9781**. The companion fixture supplies the diagnosis: *that same ansatz*, fitted to `sin(x − βt)` by
plain regression, reaches **0.0142**. So the ansatz can represent the answer to 1.4 % and residual training
from it lands at 98 % — the limitation is neither the network nor the boundary treatment, but that
minimising this residual does not lead to this solution (the large-`β` advection failure of Krishnapriyan
et al., 2109.01050, which is what *time marching* addresses).

**So: hard constraints fix the failure mode they address.** Helmholtz's binding constraint was loss
balancing, which they remove by construction; advection's is the residual landscape, which they do not
touch. "Re-run the table with hard constraints" is not a general prescription — it is worth trying per row,
and the fixtures record both the win and the counterexample.

⭐⭐ **And then the two compose, and the advection row falls: 0.0251.** The diagnosis says advection needs
the *horizon* shortened, which is what time marching does; hard constraints then remove the one thing time
marching adds back, namely an initial-condition penalty per window. Eight windows, each carrying its
initial condition structurally as `u = g(x) + τ·M(sin x, cos x, τ)` where `g` is the previous window's
solution at its own end, `[3,48,48,1]` per window, 3000 Adam steps each:

| method | rel-L2 |
|---|---|
| vanilla recipe (table) | 0.9144 |
| full recipe (table) | 0.9712 |
| one global fit, soft conditions | 0.8098 |
| one global fit, hard conditions | 0.9781 |
| time marching alone, 8 windows | 0.3252 |
| **time marching × hard conditions, 8 windows** | **0.0251** |

13× better than marching alone and 36× better than the table's best. The per-window errors are 0.0051,
0.0095, 0.0140, 0.0205, 0.0261, 0.0321, 0.0352, 0.0365 — mild monotone accumulation, which is how a
marching scheme is supposed to behave and also its limit: each window inherits the last one's error, so
more windows is not monotonically better. ⚠ Only the two global-fit rows and this one share a
configuration; the table rows and the marching-alone number come from other nets and point counts and are
context, not a controlled comparison.

⛔⛔ **The helmholtz "~50×" decomposed, on matched arms — and most of it was not the constraint.** That row
was the one comparison here that was never controlled: it put a hard-constrained fixture against the *recipe
table*, which differs in net, point count, schedule and formulation all at once. Two measurements settle it.

First, `Problem` now carries a `hard_constraint` hook, so the harness itself can impose conditions
structurally. Turning it on for helmholtz under the harness's own recipes is **worse**: vanilla **6.3903**
against 0.4766, full recipe **0.8154** against 0.3066.

Second, the matched pair the row should have had from the start — same `[2,20,20,1]` tanh net, same 900
cell-centred interior points, same 10,000 Adam steps and schedule, same seed, only the boundary treatment
differing:

| | rel-L2 |
|---|---|
| recipe table's best (different net, 2000 random points, soft penalty) | 0.3066 |
| matched arm, **soft** boundary penalty | **0.0244** |
| matched arm, **hard** boundary constraint | **0.0091** |

So the original ~50× splits as **12.6× from the configuration** and **2.7× from the constraint**. Hard
constraints do help this row — the diagnosis was directionally right, and 2.7× on matched arms is a real
effect — but the headline number was mostly the cell-centred sampling, the small net and the longer schedule,
none of which is what I attributed it to. ⚠ The advection and burgers attributions are unaffected: those
arms were matched from the start.

The hook ships; helmholtz does **not** enable it by default, because enabling it under the existing recipes
is a retuning job and not a switch.

### The four rows, diagnosed

The method that came out of this: **rule representation in or out first**, with a cheap regression fit of
the hard-constrained ansatz onto the reference and no PDE residual in the loop. Then hard constraints
separate the remaining two failure modes, because they remove loss balancing entirely and do nothing about
the residual landscape.

| row | regression fit | soft | hard | diagnosis |
|---|---|---|---|---|
| heat | — | 0.0448 | — | already solved by the recipe (0.0082) |
| helmholtz | — | **0.0244** matched | **0.0091** matched | **loss balancing**, worth 2.7× on matched arms |
| advection | 0.0142 | 0.8098 | 0.9781 | **residual landscape** → marching × hard ICs, **0.0251** |
| burgers | **0.0050** | 0.1980 | 0.3957 | representation ruled out; balancing ruled out → **residual landscape** → marching × hard, **0.0142** |

**All four rows now have a diagnosis, and the two the recipes could not reach are solved:**

| row | recipe table's best | after diagnosis | |
|---|---|---|---|
| heat | **0.0082** | — | the recipe already reaches it |
| helmholtz | 0.3066 | **0.0091** matched (0.0059 at 15k steps) | configuration 12.6× **×** hard constraints 2.7× — decomposed below |
| advection | 0.9712 | **0.0251** | marching × hard conditions, 36× |
| burgers | 0.2079 | **0.0142** | marching × hard conditions, 14× |

Burgers is the row where the obvious suspect was wrong twice over. The shock at `ν = 0.01/π` is genuinely
steep — the reference jumps **0.615** between neighbouring `x` at `t = 1` on a `Δx = 0.0078` grid — so
spectral bias looked like the limit, and it is not: the ansatz regresses onto the reference at **0.0050**,
and a much larger net does no better (0.0056). Nor is it loss balancing: hard constraints make it *worse*
(0.3957 against 0.1980 soft), exactly as on advection. By elimination it is the residual landscape, which
predicts that marching is its fix too — **and it does: 0.0142** over eight windows, 14× the table's best
and within 3× of the 0.0050 the ansatz can express at all. Per-window errors 0.0006, 0.0014, 0.0093, 0.0127,
0.0102, 0.0157, 0.0304, 0.0283.

Burgers marching has to carry `g″` from window to window, because the residual needs `u_xx` and the ansatz's
second derivative is `g″ + τ(−2M − 4x·M_x + b·M_xx)`. A second derivative accumulated across windows was the
obvious place for this to fall apart, and it did not — but that is the thing to watch if the window count
goes up. The boundary stays exact across every window for free: `u(±1) = g(±1)` because `b(±1) = 0`, and `g`
starts at `−sin(πx)`, which is already zero there.

`g` and `g′` are evaluated once per window and carried as constants, with the residual written out
(`u_τ = M + τM_τ`, `u_x = g′ + τM_x`) rather than differentiated through a growing stack of frozen networks
— so the cost per window stays flat instead of growing with the window index.

⭐⭐ **The middle column was mostly one bad assumption, not four bad rows.** `FourierNet` drew frequencies
from an isotropic `N(0, σ²)` — one `σ` for every input axis — and the solutions here are anisotropic:
advection's `sin(x − 30t)` has a wavevector of `(1, 30)` rad/unit, so no single `σ` can serve both. Giving
each axis its own scale, set a priori from the solution's known frequency content and never searched, moved
Burgers **0.5294 → 0.2434** (from 2.5× worse than plain Adam to comparable) and Helmholtz
**0.4781 → 0.3066** (from no difference to 1.6× better than vanilla). Heat is unchanged and advection is
still unsolved.

⚠ So the earlier reading — *the bundled recipe is not a free lunch* — stands, but its diagnosis was too
generous to me: two of the four rows were a configuration flaw in my own Fourier layer, and the harness is
what made that visible. The remaining honest statement is narrower: the recipe wins on heat (5.5×) and
Helmholtz (1.6×), ties on Burgers, and neither arm solves advection.

⭐ It is still one `Recipe` applied to four problems: the pieces each have their own fail-then-fix oracle
above, but bundling them and pointing the bundle at an arbitrary PDE is a different, weaker claim. The rows
are measurements; only `rel_l2.is_finite()` is asserted, and each row prints solved / partial / unsolved
rather than passing or failing a bar chosen after the fact.

### Weighting: what the table named, and what measuring it said

`Weighting::{Fixed, GradNorm, Ntk}` and `causal_slabs` are recipe options, so a row that identifies a
missing piece gets a head-to-head test rather than a note.

**NTK-trace weighting** (Wang, Yu & Perdikaris 2022, arXiv 2007.14527) — what jaxpi and PirateNets run.
Gradient-norm balancing equalises how hard each term *pulls*; the NTK view asks how fast each term's error
*decays*, and weights by the inverse kernel trace `λ_i = (Σ_j tr K_jj) / tr K_ii`. ⛔ The trace is
**estimated, not assembled**: `tr K_ii = ‖J_i‖_F²` needs one backward pass per collocation point (jaxpi gets
them from `jacrev` + `vmap`; this fabric has neither), so Hutchinson's identity is used — for a Rademacher
`v`, `E‖∇_θ(vᵀr)‖² = tr(J Jᵀ)`. Checked against a closed form on a residual linear in the parameters, where
`tr K = ‖A‖_F²` exactly: against 127.817, one probe is +6.6 %, four +10.6 %, sixteen +4.7 %, sixty-four
**+1.7 %** — the test watches it converge rather than asserting one draw. Weights for two terms whose traces
differ 10⁴× come out 9371× apart, and a term whose trace has gone to zero is floored, the same guard
`LossBalancer` needed.

**Causal weighting** reuses the `Causal` machinery (which has its own fail-then-fix oracle above) against
the problem's `time_axis`. `Recipe::ntk()` and `Recipe::causal(k)` are the named recipes.

⚠ **Neither head-to-head comparison discriminates, before or after the frequency fix.** Helmholtz,
gradient-norm against NTK: **0.4781 / 0.5183** with isotropic scales, **0.3066 / 0.3014** with per-axis ones
— NTK flipped from marginally worse to marginally better, 1.7 % apart, with both still unsolved. Advection:
**1.0815 / 1.0935** isotropic, **0.9712 / 0.9706** per-axis. In every case *both* arms fail, so the ordering
is noise: two failing arms cannot rank two techniques. That is the rule that sent the balancing and causal
fixtures back to be re-sized earlier on this page, applied to my own new work — and `compare()` prints
`⚠ every arm failed — this comparison does not rank the recipes` itself rather than leaving it to the reader.
Both sets of numbers are kept: a technique that changes sign when an unrelated layer is fixed is worth
showing twice.

⛔ `Problem::constraints` returns residual **vectors**, not a summed loss: an NTK trace is a property of the
per-point Jacobian and summing first destroys it. Every boundary, initial and periodic condition is its own
term — which is also the paper's formulation.

### Time marching

`run_time_marched` is the piece the advection row named: a training **strategy**, not a loss term
(Krishnapriyan et al. 2021 §5). The horizon is split into `k` windows solved in order; window `w` sees only
`t ∈ [t_w, t_{w+1}]`, keeps the problem's spatial boundary conditions, and takes its initial condition from
window `w−1`'s trained network, evaluated on the slice `t = t_w` and frozen. That required splitting
`Problem`'s constraints into `boundary_constraints` (true at every time) and `initial_constraint` (true only
at `t = 0`) — a single list cannot express "same walls, different start".

⛔ **Each window keeps its own network, and that is not an optimisation.** Warm-starting one network through
the windows ends with a network that fits only the last one; scoring therefore evaluates each point with the
network that owns its time slab.

⭐⭐ **Measured, and it is the first thing to move advection.** 4000 collocation points, same network, same
per-window budget: one global fit **0.7745**, eight marched windows **0.3252** — 2.4× lower. Every weighting
scheme on this page left that row between 0.91 and 1.08; the literature says the fix is a training strategy
rather than a loss term, and on this stack it is.

⛔ **Advection at β = 30 is not solved by any *single-fit* recipe here** (0.9144 / 0.9712 / 0.9706 with
causal — all barely better than predicting zero). Per-axis scales moved it 1.08 → 0.97 and no weighting
moves it further. **Time marching does**: 0.7745 → 0.3252 at eight windows (see above). That is consistent
with Krishnapriyan et al., who solve this case the same way — and it is the clearest result on this page
that *how you train* can outrank every term you put in the loss.
One `#[ignore]`d test per problem (`benchmark_heat`, `benchmark_burgers`, …) prints its row.
`refinement_beats_uniform_sampling_on_the_burgers_shock_at_equal_budget` is the RAR fixture in the regime
where it can be shown to help — DeepXDE's own numbers for this problem,
2540 uniform against 2000 + 540 refined, both arms with the L-BFGS stage. Measured at **exactly equal
budget**: uniform(2540) **0.5008**, refined(2540) **0.3833** — 23 % lower from where the points are, not
how many. Contrast the earlier `k`-sweep on a 1-D layer, where refinement never beat uniform because Adam
alone left that regime optimisation-limited: the difference is that both arms here end on L-BFGS, so the
smooth part is solved and the shock is what remains.

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
loss curve would show it. Two GPU oracles close the loop with closed forms, both measured: Laplace on the annulus through
CSG + Dirichlet reaches **rel-L2 0.0082** against `u = ln(r/½)/ln 2`, and the 1-D Neumann sign test
(`u'' = 0`, `u(0) = 0`, `u'(1) = 1` ⇒ `u = x`) returns **u(½) = 0.5002** where a flipped outward normal
would have given −0.5.

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

**Every one of the safe ops is now checked against central differences**, on broadcast and mixed-rank shapes,
in three ways at once: `backward()`, the functional `grad()`, and — for the smooth ones — the gradient of
`Σ(∂L/∂x)²`, which is the only check that exercises a VJP's *own* VJP. 23 cases; worst first-order gap
3.0e-3. This is the battery that would have caught the `matmul` defect on the day it was written.

⛔ **A finite-difference battery cannot see a defect in the forward pass** — it checks the derivative of
whatever function is implemented, so a wrong constant changes the value and its gradient consistently and
passes. Mutating `mean`'s divisor to 1, making it `sum`, left the whole battery green. A companion test now
pins 26 forward behaviours to closed forms computed in Rust from the inputs (`mean` against its own sum over
its own count, `softmax` rows summing to 1 with ratios equal to `exp` of the difference, `matmul` against a
product written out by hand, the elementwise maps against `std`). Mutations that the battery missed and the
companion catches: `mean` divides by 1; `softmax` normalises over the wrong axis; `sum` drops keepdim;
`mean_all` divides by the rank.

**L-BFGS needs a Wolfe line search, not Armijo.** With backtracking alone, Rosenbrock stalled at
`f = 3.47` for 200 iterations: from iteration 3 every accepted step had `sᵀy ≈ −5e-7`, so every curvature
pair was rejected, the history froze, and its direction collapsed to `|d| ≈ 1.8e-3` — accepted at unit
step every time. The Wolfe curvature condition makes `sᵀy > 0` by construction; with it, 36 iterations,
49 evaluations, `f = 0` at `(1, 1)`.

## The multi-channel FNO — `sciml::fno`

The channel-diagonal layer above is one complex *scalar* per mode. The FNO of Li et al. (2010.08895) is the
general case: at each retained wavevector a learned complex **matrix** `R(k) ∈ ℂ^{w×w}` applied to the channel
vector of the spectrum. `SpectralConvMulti` is that layer and `Fno2d` the surrounding network — lift `d_in → w`
pointwise, `L` Fourier layers `h ← σ(W h + b + SpectralConvMulti(h))`, project `w → d_out`.

The per-mode matrices are **not** a loop over modes: the retained spectrum is compressed to `[B, w, M]`,
transposed to mode-major `[M, B, w]`, and multiplied by `R` of shape `[M, w, w]` as a single batched GEMM.

### The transform: two exact plans, chosen by measurement

`F₂ = F₁ ⊗ F₁` as one `n² × n²` matmul costs `n⁴` per field and needs an `n² × n²` matrix — **4.29 GB** of
DFT matrices at `n = 128`, which is the wall the layer actually hits. The transform is separable, so it can
instead be two `n × n` matmuls with a transpose between: `2n³` per field from a **65.5 kB** matrix, exact,
and built only from `matmul`/`transpose`/`reshape`, so the layer stays second-order differentiable.
`Dft2` carries both plans and they agree to **1.4e-7**.

⚠ **The flop ratio is not the speed ratio, and measuring it changed the design.** A forward-plus-inverse
pair is 12 matmuls of `n³` against 4 of `n⁴` — `n/3` the arithmetic, but **three times the dispatches**.
Measured at batch 48 on Metal:

| n | Kronecker | separable | speedup | matrices |
|---|---|---|---|---|
| 16 | 1.21 ms | 2.06 ms | **0.59×** | 1.0 MB vs 4.1 kB |
| 32 | 1.23 ms | 1.68 ms | **0.73×** | 16.8 MB vs 16.4 kB |
| 64 | 11.64 ms | 2.40 ms | **4.84×** | 268.4 MB vs 65.5 kB |

The separable plan *loses* at `n = 16` — dispatch-bound, not compute-bound — and wins by ~5× at `n = 64`.
(⚠ The absolute milliseconds move a lot with machine load — contended runs of this same test gave
1.61/4.51/3.46/6.49/15.14/4.18 and 0.39/1.23/1.01/1.37/11.90/2.36 ms. Here the crossover held across all
three runs, so the threshold is safe; it did **not** hold for the separable-PINN measurement below, where
contention moved the crossover an octave. Measure on an idle machine.) So `Dft2::new` picks by grid size (`SEPARABLE_ABOVE = 32`, the crossover as measured) rather than
replacing one with the other, and the memory ratio (`n²`, always) is what unlocks the resolutions the
Kronecker form could never hold: the layer is exact to **1.9e-7** at `n = 64` with 113 retained modes.

⛔ **Making the choice automatic nearly made three tests vacuous.** Once `Dft2::new` picked by `n`, the
separable-vs-Kronecker agreement test and the wall-clock benchmark were both constructing the *Kronecker*
plan for their "separable" arm at `n ≤ 32` — each comparing a plan to itself and passing. Every test that
depends on which plan is running now names it with `with_plan` and asserts `dft.plan`; mutating
`SEPARABLE_ABOVE` to 0 or to a huge value is caught in both directions.

⚠ `SpectralConv2d`, the channel-diagonal layer, is deliberately left on its own Kronecker matrices. It is
the independent reference the separable plan is checked against, and porting it would collapse that check
into a tautology. This is also not the FFT: `n² log n` needs a butterfly primitive built from gather/scatter,
and those carry no differentiable VJP here. The separable plan is the whole of the asymptotic gain available
from differentiable ops today.

| oracle | result |
|---|---|
| `SpectralConvMulti` on a channel-**coupling** operator `M(k) = g(k)A + i s(k)B` | held-out rel-L2 **0.0001** |
| the same layer with its off-diagonal blocks masked (channel-diagonal ablation) | rel-L2 **0.6007** — it cannot represent the coupling |
| the learned per-mode matrices against the closed-form `M(k)`, entry by entry | worst entry off by **1.24e-4**, against entries spanning ±1.0 |
| `Fno2d` (width 8, 2 layers, tanh) on the nonlinear operator `f ↦ s + s²`, `s` = `f` smoothed by `4/|k|²` | held-out rel-L2 **0.0248** |
| the layer at `n = 64`, 113 retained modes, reference weights installed | rel-L2 vs the closed form **1.9e-7** |
| the same network with its activations removed (exactly linear, same parameter count) | rel-L2 **0.5999** |

(`s` is a Poisson-type smoothing of `f` — the multiplier `4/|k|²`, scaled so `s` is O(1) rather than the
Green's function's `1/(4π²|k|²)`.) The nonlinear fixture asserts its own premise in numbers before it compares
anything: `‖s²‖/‖u‖ = 0.734`, so
the quadratic term genuinely dominates and the linear arm is *forced* to fail. Both reference operators are
built in real space as explicit sums of sinusoids with the multiplier applied in closed form per mode — they
touch neither the network nor its DFT matrices.

⛔ **Every fixture was symmetric under the transformation that was wrong.** The batched GEMM contracts the
channel axis against `R`'s last axis, so the stored layout is `[in, out]`; the oracle wrote its reference
matrix `[out, in]`, the way an operator matrix is always written. The layer was learning the **transpose of the
truth**, and three tests passed over it: `R = I` is the identity and `Iᵀ = I`; width 1 matches the
channel-diagonal layer and a 1×1 matrix is its own transpose; and held-out error after training was **0.0000**,
because the transpose is exactly as learnable as the truth. Only the trained weight read wrong — `0` where the
truth said `1.0` — and the first explanation that came to mind (an untrainable DC mode) was wrong.

What found it in one run, and is now a permanent test: **install the reference weights and compare the forward
map to the closed form with no training in the loop** — if that fails the conventions differ and no amount of
training reconciles them — then repeat it **one mode at a time** on a field carrying that mode alone, because a
whole-field norm hides one bad mode under the modes that dominate it. `R` is now stored and read `[out, in]`
and transposed at the GEMM. Same family as the conjugate-symmetry defect above: a good loss is not evidence
that a weight means anything.

⚠ **Two weight groups are masked to zero rather than left to drift**, for the same reason: the imaginary part
at a self-mirrored mode (`k = −k mod n`), whose spectrum is real for a real field and whose imaginary output
the symmetry expansion discards, and the off-diagonal blocks of the channel-diagonal ablation. Unmasked they
sit at their random initialisation with zero gradient and read like learned values.

### ⛔ A third defect the operator exposed in the fabric

**`Var::matmul` had a broken backward pass for a batched operand against an unbatched one.** The VJP
transposed *both* operands by the rank of the first, so `[B, m, k] × [k, n]` — the shape every FNO layer and
every pointwise channel projection makes — panicked with `permute rank mismatch` the moment a gradient flowed
through it. The forward pass was always correct, which is why the shape reads as supported; nothing in the
stack had mixed ranks before. Each operand now transposes by its own rank, and the unbatched operand's
gradient is summed back over the broadcast batch (which `accumulate` and `grad()` already did centrally — a
belt-and-braces `unbroadcast_var` inside the VJP was written and then removed, because no mutation could
distinguish it from its absence). Covered by a finite-difference check on both operands plus an assertion on
the shape `grad()` returns for the unbatched one.

## The separable PINN — `sciml::spinn`

Cho et al., *Separable Physics-Informed Neural Networks* (2306.15969). A dense PINN on a `d`-dimensional
tensor grid of `N` points per axis evaluates its network at `N^d` collocation points. A separable one
factors the field into a rank-`r` sum of products of **one-dimensional** networks,
`u(x₁,…,x_d) = Σ_{j≤r} Π_i f_i(x_i)_j`, so the same `N^d` grid costs `d·N` network evaluations. The residual
still sees every grid point, because the product is formed as a tensor contraction *after* the networks run.

**`jvp_1d` is what makes it affordable here.** The residual needs `∂f_i/∂x_i` for a `[N,1] → [N,r]` map, and
reverse mode gives one output column per pass — `r` passes. The original uses forward mode (`jax.jvp`),
which this fabric does not have. Instead: differentiate `uᵀy` with respect to `x`, giving `Jᵀu` as a graph
linear in the cotangent `u`, then differentiate *that* with respect to `u`. Two passes, whatever `r` is.
Checked against central differences to second order (worst 5.9e-5 / 8.0e-5) **and** against `r` seeded
reverse passes (1.2e-7) — two independent routes to the same quantity. The contraction is checked against
the same sum written as explicit loops in Rust (exact).

| oracle | result |
|---|---|
| separable PINN on 3-D Poisson, `16³ = 4096` points from **48 network rows/step** | rel-L2 vs the exact solution **0.0003** |
| residual cost per step, separable vs dense, `n = 8 / 16 / 32` | **0.99× / 2.81× / 11.4×** |
| fit vs rank on a separation-rank-5 field, ranks 1 / 2 / 5 / 12 | 0.862 / 0.703 / 0.348 / 0.289 |

⚠ **The saving appears with scale, and it is worth ~1.0× until it does.** Separable 61.6 / 93.5 / 114.3 ms
against dense 61.2 / 262.5 / **1299.0** ms: the separable arm's cost grows slowly with the grid, the dense
arm's tracks it. At `n = 8` both sit on the dispatch floor and the 21× fewer network rows buy nothing. The
test asserts a large-grid phenomenon rather than a blanket win.

⛔ **These numbers moved by a factor of three when the machine went idle, and one of them changed sign.** An
earlier run of this same test with other cargo jobs sharing the GPU read **0.97× / 0.90× / 2.82×** — which
says the saving does not appear until `n = 32`, a full octave too high, and says it is *negative* at `n = 16`
where it is really 2.8×. Contention did not add noise, it moved the conclusion. Every wall-clock number in
this document is from an otherwise idle machine, and the transform crossover above was re-measured idle for
the same reason (0.59× / 0.73× / 4.84× at n = 16 / 32 / 64 — the `SEPARABLE_ABOVE = 32` threshold stands).

⛔ **What is *not* claimed, and why.** A head-to-head "separable beats dense" was built first and then
deleted: the dense arm did not solve the problem. At `n = 16` on a rank-3 target it reached rel-L2 0.198
after 3500 steps, and on an easier rank-2 target it plateaued at 0.26 — a plain tanh MLP on a 3-D input with
a hard `Π x_k(1−x_k)` constraint. The fixture's own assertion caught it (*"the dense arm must solve the
problem, or this comparison ranks nothing"*), and rather than tune the baseline until it lost politely, the
comparison was split: cost is measured on its own with no accuracy claim, and accuracy is measured against
the closed-form solution with no dense arm. One fixture, one claim.

⛔ **The rank sweep shows less than it was built to show, and says so.** A clean rank limit would fall to
near zero at `r = R` and flatten; it does not — 0.348 at rank 5 and still falling to 0.289 at rank 12. So
rank is not cleanly separated from optimisation difficulty (fitting a sum of products is a nonconvex tensor
factorisation, plainly not solved to optimality here). Two earlier targets failed differently and are
recorded in the test: `sin(mπ·)` up to `m = 5` stalled at 0.185 because the 1-D nets could not fit
`sin(5πx)` — spectral bias, not rank — and Legendre terms weighted `1/(q+1)` left ranks 1 and 2 scoring
*identically* 0.0990, because the `q = 0` term carried nearly all the energy and the sweep measured nothing.
What survives is the monotone dependence on rank, and the claim is kept to that.

## Natural gradients — `sciml::natgrad`

Adam and L-BFGS descend in the *parameter* metric. For a least-squares PINN loss `½‖r(θ)‖²` the natural
direction in the metric induced by the residual map is the Gauss-Newton step `δθ = (JᵀJ)⁺Jᵀr`, with
`J = ∂r/∂θ` at the collocation points. This is the mechanism behind the accuracy gap reported by Müller &
Zeinhofer, *Achieving High Accuracy with PINNs via Energy Natural Gradient Descent* (2302.13163).

⚠ **Named precisely.** This is the **Gauss-Newton** natural gradient — the Gramian of the *residual* map in
`L²`. The paper's headline variant uses the PDE's **energy** (`H¹`-type) inner product, a different Gramian.
⛔ **That variant was attempted and is not shipped** — see below.

1-D Poisson `−u'' = f`, `u* = sin(πx) + ½sin(3πx)`, boundary conditions hard-constrained by `x(1−x)` so both
arms optimise exactly the same objective on the same `[1,12,12,1]` tanh net (193 parameters) and the same
300 collocation points. The only difference is the metric the step is taken in:

| arm | rel-L2 vs the exact solution |
|---|---|
| Adam, 20,000 steps | 1.376e-4 |
| Adam, 1,500 steps | 2.339e-4 |
| + **30 Gauss-Newton steps** (118 s) | **1.153e-6** — 119× better |

⛔ **The premise is not "Adam is bad".** 1.4e-4 is a respectable PINN accuracy. The premise is that Adam has
**plateaued**: 13× more steps (1,500 → 20,000) bought 1.7×, so the remaining error is not a step-count
problem and cannot be optimised away by running the same method longer. That is what makes 119× from 30
Gauss-Newton steps a statement about the metric rather than about budget, and it is what the fixture asserts.
A first version asserted `rel_adam > 1e-3` and failed *because Adam did well* — the wrong thing to require.

⛔ **And it is not a universal upgrade.** On 2-D Helmholtz, with the same hard-constrained formulation, 25
Gauss-Newton steps over 818 s moved a 0.4439 warm-up to only **0.4121** — where plain Adam on that same
configuration reaches 0.0059. The 119× on 1-D Poisson is a real result about that problem, not a property of
the method; the Helmholtz fixture is where the counterexample is on the record, and it deliberately asserts
no Gauss-Newton win.

**The cost is real and structural.** Reverse mode yields one *row* of `J` per pass, so a step costs `N`
backward passes over a graph that already contains the residual's own second derivatives — 300 passes here,
about 4 s per step. It pays only because the step count collapses from tens of thousands to tens. `J` must
also have at least as many rows as parameters or the Gramian is singular; the Tikhonov term is **relative**
(`λ·tr(JᵀJ)/p`), so it carries no units and survives a rescaling of the residual.

⛔ **The energy variant was built, measured three ways, and removed.** The `H¹` Gramian
`G_ij = ∫∂ₓ(∂u/∂θ_i)·∂ₓ(∂u/∂θ_j)` assembles from the θ-Jacobian of `u_x` with exactly the machinery above,
and is the natural gradient of the variational functional `E(u) = ½∫|∇u|² − ∫fu`. On the same net the
Gauss-Newton arm takes to **1.153e-6**:

| configuration | rel-L2 |
|---|---|
| `H¹` metric + residual loss `½‖Δu+f‖²` | 7.045e-3 — 30× *worse* than its own warm-up |
| `H¹` metric + Ritz energy, mean quadrature | 7.5e-4 from a residual warm-up; 6.344e-3 from its own |
| `H¹` metric + Ritz energy, sum quadrature | 1.129e-2 |

Three diagnoses were tried and **measurement refuted each**: a mismatched metric/loss pairing (fixing it
helped, did not converge); a quadrature-limited objective (refuted — refining 300 → 1200 points moved the
error 1.3×, where an `O(h²)` limit predicts 16×); and a quadrature-weight mismatch between the raw `JᵀJ`
Gramian and a `mean`-normalised gradient (making them consistent made it *worse*). What remains unexplained
is that the energy steps reliably **decrease** the discrete Ritz energy while moving the solution **away**
from `u*`, on an ansatz that demonstrably represents `u*` to 1e-6. The public entry point was deleted rather
than shipped with that behaviour; `gramian_solve` already accepts several Jacobians, so the metric is one
argument away whenever this is understood. This remains an open gap, not a completed feature.

Both primitives are checked against something that shares nothing with them: the `f64` Cholesky against a
system whose answer is known by construction *and* against its own residual recomputed from a kept copy of
the matrix (1.4e-16 / 4.4e-16; an indefinite matrix is refused, not silently mis-solved), and the residual
Jacobian against central differences in **every** parameter (worst 1.9e-4) — a wrong `J` is a
plausible-looking step that descends the wrong direction, and nothing downstream would say so.

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
- **FNO without FFT.** Ferric has no FFT/complex, so the spectral conv is DFT-as-matmul with a real/imag
  split. ⚠ Corrected: on a 2-D `n × n` grid the Kronecker form is `O(n⁴)` per field with an `n² × n²`
  matrix, not `O(n²)` — it is the **memory**, not the asymptotics, that bites first (4.29 GB at `n = 128`).
  The separable plan brings that to `2n³` from an `n × n` matrix and is what makes `n = 64`+ reachable; the
  remaining gap to a true FFT is `n/log n`. The learned per-mode weights are identical under every plan.

## Honest scope & what's not here

- **Nano / demo scale, single-seed, Metal-verified.** These are correctness demonstrations of the mechanism,
  not benchmarks. On tiny problems a learned surrogate is not faster than the true solver — the large
  operator/surrogate speedups (≈466× soft-robot MPC, ≈44,000× Cosserat-rod) are real only for stiff / soft /
  high-DOF plants, and are cited, not claimed here.
- **Joules-per-solve is not measured** — Apple Silicon exposes no RAPL; an honest per-solve energy number
  needs the external-meter / Jetson path (see `FABRIC.md`). Not fabricated.
- **Remaining:** an FFT primitive (the separable plan took the 2-D transform from `n⁴` to `2n³`; an FFT
  would take it to `n² log n`, and needs a butterfly built from gather/scatter, neither of which carries a
  differentiable VJP here), the **energy** natural gradient (attempted, three configurations measured, none
  convergent — see above), and a WebGPU **in-browser** build. On the last: `cargo check -p ferric-tensor --target
  wasm32-unknown-unknown` is **clean**, so nothing in the tensor crate — `sciml` included — is host-only at
  the type level. That is a compile, not a run: it says nothing about whether the WebGPU compute path, the
  tape, or second-order `grad()` behave in a browser, and in-browser *training* remains unproven and still
  needs a de-risking spike.
