//! **Mapping work across a heterogeneous fabric** — which missing experts cross the bus, and which
//! execute where they already are.
//!
//! `ferric-tier`'s other modules answer *where do bytes live*. This one answers the question that
//! comes next and that placement alone cannot: when `m` experts are missing from device memory,
//! some can be shipped over the bus and computed on the accelerator while the rest are computed
//! **in place on the host**, and those two paths run at the same time. The split is not a
//! preference. It is arithmetic over two bandwidths that differ per machine by more than an order of
//! magnitude, so a constant here is wrong everywhere except the desk it was written on.
//!
//! ## The latency split
//!
//! Let `S` be bytes per expert, `BP` the measured host→device bandwidth and `BH` the measured
//! host-side expert-processing bandwidth. Shipping consumes host memory bandwidth too, so the host
//! computes with only the **residual** `BR = BH − BP`. Sending `q` of `m` experts across:
//!
//! ```text
//!     bus path      q·S / BP        host path   (m−q)·S / BR       (concurrent)
//! ```
//!
//! Wall clock is the slower of the two, minimised where they are equal:
//!
//! ```text
//!     q/BP = (m−q)/BR   →   q(BR + BP) = m·BP   →   q* = m·BP / BH
//! ```
//!
//! ⚠ **`BR` can be zero or negative**, and that is a real machine rather than a degenerate one: when
//! the bus can absorb everything the host can produce, the host has nothing left to compute with and
//! the in-place path is worth exactly nothing. [`FabricProfile::split_for_latency`] returns `q = m`
//! there instead of dividing by it.
//!
//! ## ⭐ The energy split is a CORNER, and that is the interesting part
//!
//! Every engine that does this optimises wall clock. Energy gives a different answer, and not by a
//! little — by kind. Total joules over the two paths is
//!
//! ```text
//!     E(q) = q·S·e_dev + (m−q)·S·e_host
//! ```
//!
//! which is **linear in q**, so its minimum is always an endpoint: all-bus or all-host, never the
//! balance point. The latency optimum is interior and the energy optimum is a corner, so on any
//! machine where the two subsystems differ in joules-per-byte they disagree by construction. Running
//! `q*` and calling it efficient is picking one objective without noticing there was a choice.
//!
//! The useful operating point is neither: **minimise joules subject to a latency budget**
//! ([`FabricProfile::split_within_budget`]), which slides along the segment between the corner and
//! `q*` and degenerates to each of them at the ends of the budget range.
//!
//! ## Nothing here is allowed to guess
//!
//! [`FabricProfile`] has no `Default` and no constant fallbacks. A profile is built from
//! measurements or not at all, because a bandwidth model that silently substitutes a plausible
//! number produces a plan that is confidently wrong and reports no error — the failure mode this
//! crate exists to remove from placement.

/// Measured bandwidths of one machine, in bytes per second. **Every field is measured.**
///
/// There is deliberately no `Default`: a guessed profile yields a split that looks reasonable and is
/// wrong, with nothing downstream able to tell. See `examples/fabric_profile.rs` for the measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct FabricProfile {
    /// `BP` — host → device transfer, as achieved by real uploads, not the link's nameplate rate.
    pub pcie: f64,
    /// `BH` — host-side expert processing: bytes of weight the CPU pushes through per second.
    pub host: f64,
    /// `BD` — backing-store read. Governs whether an expert can reach the host in time at all.
    pub disk: f64,
}

/// How joules are spent on each path, in joules per byte of expert weight.
///
/// ⚠ These are **not** power draws. A watt figure cannot be compared across paths that take
/// different amounts of time; joules per byte can, and is what the split arithmetic needs.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct EnergyModel {
    /// Transfer across the bus plus the device-side compute, per byte.
    pub dev_j_per_byte: f64,
    /// Host-side compute in place, per byte.
    pub host_j_per_byte: f64,
}

impl EnergyModel {
    /// Build from a measured joules-and-bytes pair per path — the shape `ferric_joule::Reading`
    /// gives you, so a caller cannot assemble one from watts alone and lose the time term.
    pub fn from_measurements(dev_joules: f64, dev_bytes: u64, host_joules: f64, host_bytes: u64)
        -> Result<EnergyModel, String>
    {
        if dev_bytes == 0 || host_bytes == 0 {
            return Err("an energy model needs a non-zero byte count on BOTH paths; a path that \
                        moved no bytes has no joules-per-byte and cannot be compared".into());
        }
        Ok(EnergyModel {
            dev_j_per_byte: dev_joules / dev_bytes as f64,
            host_j_per_byte: host_joules / host_bytes as f64,
        })
    }
}

/// How `m` missing experts divide between the two paths. `to_device + on_host == m`, always.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Split {
    /// Shipped across the bus and computed on the accelerator.
    pub to_device: u64,
    /// Computed in place on the host, never crossing the bus.
    pub on_host: u64,
}

impl Split {
    pub fn total(&self) -> u64 { self.to_device + self.on_host }
}

impl FabricProfile {
    /// A profile from three measured rates. Refuses non-positive or non-finite values rather than
    /// producing a plan from them.
    pub fn measured(pcie: f64, host: f64, disk: f64) -> Result<FabricProfile, String> {
        for (v, n) in [(pcie, "pcie"), (host, "host"), (disk, "disk")] {
            if !(v > 0.0) || !v.is_finite() {
                return Err(format!("{n} bandwidth measured as {v}, which is not a rate a plan can \
                                    be built from"));
            }
        }
        Ok(FabricProfile { pcie, host, disk })
    }

    /// Residual host bandwidth once the bus transfer has taken its share: `BR = BH − BP`.
    ///
    /// Zero or negative means the host cannot feed the bus AND compute — see the module note.
    pub fn residual_host(&self) -> f64 { self.host - self.pcie }

    /// **`q* = m·BP/BH`** — the split that minimises wall clock.
    pub fn split_for_latency(&self, m: u64) -> Split {
        if m == 0 { return Split { to_device: 0, on_host: 0 } }
        if self.residual_host() <= 0.0 {
            // The host has nothing left over; every expert goes across.
            return Split { to_device: m, on_host: 0 };
        }
        // ⛔ ROUNDING TO NEAREST IS WRONG HERE, and it was wrong in this file first. `q*` is the
        // continuous optimum of `max(q·S/BP, (m−q)·S/BR)`, and that max is a V with DIFFERENT slopes
        // on its two arms: descending at `S/BR` on the left, ascending at `S/BP` on the right. When
        // `BR < BP` — which is every machine where the bus is a decent fraction of host bandwidth —
        // overshooting costs less than undershooting, so the nearest integer is not the best one.
        //
        // Measured here at BP 0.29 / BH 0.43 GB/s: q* = 5.40, nearest is 5 at 44.5 ms, and 6 is
        // 42.4 ms. The rounded answer was 5% slower than the split it claimed to be the minimum of,
        // and it was `split_within_budget` — which brute-forces its own interval — printing a FASTER
        // plan on the next line that exposed it.
        //
        // Both neighbours, keep the better. Exact by construction, and cheaper than arguing.
        let qf = m as f64 * self.pcie / self.host;
        let lo = qf.floor().clamp(0.0, m as f64) as u64;
        let hi = qf.ceil().clamp(0.0, m as f64) as u64;
        let (a, b) = (Split { to_device: lo, on_host: m - lo }, Split { to_device: hi, on_host: m - hi });
        // `bytes_per_expert` cancels out of the comparison, so any positive probe orders them.
        if self.latency_s(&a, 1) <= self.latency_s(&b, 1) { a } else { b }
    }

    /// The energy-optimal split, which is always all-bus or all-host — see the module note on why
    /// `E(q)` being linear in `q` forces a corner.
    pub fn split_for_energy(&self, m: u64, e: &EnergyModel) -> Split {
        if e.dev_j_per_byte <= e.host_j_per_byte { Split { to_device: m, on_host: 0 } }
        else { Split { to_device: 0, on_host: m } }
    }

    /// Wall clock for a split: the two paths run concurrently, so it is the slower of them.
    pub fn latency_s(&self, s: &Split, bytes_per_expert: u64) -> f64 {
        let b = bytes_per_expert as f64;
        let bus = s.to_device as f64 * b / self.pcie;
        let br = self.residual_host();
        let host = if s.on_host == 0 { 0.0 }
                   else if br <= 0.0 { f64::INFINITY }   // no residual: this path cannot finish
                   else { s.on_host as f64 * b / br };
        bus.max(host)
    }

    /// Joules for a split. Linear in `to_device` — the property that makes the energy optimum a
    /// corner, asserted in the tests rather than left as a claim.
    pub fn joules(&self, s: &Split, bytes_per_expert: u64, e: &EnergyModel) -> f64 {
        let b = bytes_per_expert as f64;
        s.to_device as f64 * b * e.dev_j_per_byte + s.on_host as f64 * b * e.host_j_per_byte
    }

    /// ⭐ **Minimise joules subject to `latency ≤ budget_s`** — the operating point neither of the
    /// two pure objectives gives you.
    ///
    /// The latency constraint bounds `q` from both sides:
    ///
    /// ```text
    ///     q·S/BP  ≤ T   →   q ≤ T·BP/S
    ///     (m−q)·S/BR ≤ T   →   q ≥ m − T·BR/S
    /// ```
    ///
    /// so the feasible set is an interval containing `q*`, and since `E` is monotone in `q` the
    /// answer is whichever endpoint of that interval the cheaper path lies toward. Returns `None`
    /// when the budget is below what `q*` itself achieves — an honest "this machine cannot" rather
    /// than a plan that will miss.
    pub fn split_within_budget(&self, m: u64, bytes_per_expert: u64, e: &EnergyModel, budget_s: f64)
        -> Option<Split>
    {
        if m == 0 { return Some(Split { to_device: 0, on_host: 0 }) }
        if !(budget_s > 0.0) { return None }
        let best = self.split_for_latency(m);
        if self.latency_s(&best, bytes_per_expert) > budget_s { return None }

        let b = bytes_per_expert as f64;
        let hi_f = budget_s * self.pcie / b;
        let br = self.residual_host();
        // With no residual host bandwidth the host path can never finish, so q is pinned at m.
        let lo_f = if br <= 0.0 { m as f64 } else { m as f64 - budget_s * br / b };
        let lo = lo_f.ceil().max(0.0).min(m as f64) as u64;
        let hi = hi_f.floor().max(0.0).min(m as f64) as u64;
        if lo > hi { return None }

        let q = if e.dev_j_per_byte <= e.host_j_per_byte { hi } else { lo };
        let s = Split { to_device: q, on_host: m - q };
        // The interval arithmetic used floors and ceils on a continuous relaxation; experts are
        // whole. Verify the integer answer against the constraint it was derived from rather than
        // trusting the derivation — an off-by-one here silently misses the budget every step.
        if self.latency_s(&s, bytes_per_expert) > budget_s { return None }
        Some(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prof() -> FabricProfile { FabricProfile::measured(8.0e9, 20.0e9, 3.0e9).unwrap() }
    const S: u64 = 64 << 20; // 64 MiB per expert

    #[test]
    fn a_profile_cannot_be_built_from_a_number_that_is_not_a_rate() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(FabricProfile::measured(bad, 1.0e9, 1.0e9).is_err(), "accepted pcie={bad}");
            assert!(FabricProfile::measured(1.0e9, bad, 1.0e9).is_err(), "accepted host={bad}");
            assert!(FabricProfile::measured(1.0e9, 1.0e9, bad).is_err(), "accepted disk={bad}");
        }
    }

    /// ⛔ THE TEST THAT SHOULD HAVE EXISTED FIRST. The earlier version asserted the two paths
    /// finish within a rounding-sized skew of each other, which is a property of the DERIVATION and
    /// not of the answer — and a `round()` that picks the wrong neighbour satisfies it comfortably.
    /// It shipped a split 5% slower than the true integer minimum, and what caught that was an
    /// example printing a faster plan two lines below the one labelled "min latency".
    ///
    /// The property worth asserting is the one the function's name claims: it is the ARGMIN over
    /// every integer split. Brute force has nowhere to hide an off-by-one.
    #[test]
    fn the_latency_split_is_the_argmin_over_every_integer_split() {
        for (bp, bh) in [(8.0e9, 20.0e9), (0.29e9, 0.43e9), (1.0e9, 1.05e9), (3.0e9, 12.0e9),
                         (5.0e9, 6.0e9), (0.5e9, 9.0e9)] {
            let p = FabricProfile::measured(bp, bh, 3.0e9).unwrap();
            for m in [1u64, 2, 3, 5, 8, 16, 47, 64, 129] {
                let got = p.split_for_latency(m);
                assert_eq!(got.total(), m, "the split lost or invented an expert at m={m}");
                let want = (0..=m).map(|q| Split { to_device: q, on_host: m - q })
                    .min_by(|a, b| p.latency_s(a, S).partial_cmp(&p.latency_s(b, S)).unwrap())
                    .unwrap();
                assert!((p.latency_s(&got, S) - p.latency_s(&want, S)).abs() < 1e-12,
                        "BP {bp:.2e} BH {bh:.2e} m={m}: returned {got:?} at {:.6}s, but {want:?} \
                         takes {:.6}s", p.latency_s(&got, S), p.latency_s(&want, S));
            }
        }
    }

    /// ⚠ The V is ASYMMETRIC — that is why nearest-rounding failed. Left arm falls at S/BR, right
    /// arm rises at S/BP, so whenever BR < BP the optimum sits at the CEILING of q*, not the
    /// nearest. Pinned directly, on the exact machine measurement that exposed it.
    #[test]
    fn overshooting_the_continuous_optimum_costs_less_than_undershooting() {
        let p = FabricProfile::measured(0.29e9, 0.43e9, 2.4e9).unwrap();
        assert!(p.residual_host() < p.pcie, "fixture must have BR < BP for the asymmetry to bite");
        let m = 8u64;
        let qf = m as f64 * p.pcie / p.host;
        assert!((qf - 5.4).abs() < 0.1, "q* should be ~5.4 here, got {qf}");
        let under = Split { to_device: 5, on_host: 3 };
        let over = Split { to_device: 6, on_host: 2 };
        assert!(p.latency_s(&over, S) < p.latency_s(&under, S),
                "over {:.6}s should beat under {:.6}s", p.latency_s(&over, S), p.latency_s(&under, S));
        assert_eq!(p.split_for_latency(m), over, "and split_for_latency must pick it");
    }

    /// ⚠ A real machine, not a degenerate one: the bus absorbs everything the host can produce, so
    /// the in-place path has no bandwidth to run in. Dividing by BR here would return a plan.
    #[test]
    fn when_the_bus_is_as_fast_as_the_host_the_in_place_path_is_worth_nothing() {
        let p = FabricProfile::measured(20.0e9, 20.0e9, 3.0e9).unwrap();
        assert_eq!(p.residual_host(), 0.0);
        let s = p.split_for_latency(32);
        assert_eq!(s, Split { to_device: 32, on_host: 0 });
        assert!(p.latency_s(&s, S).is_finite(), "the all-bus plan must be finite");
        // And a split that DOES put work on the host is correctly reported as never finishing,
        // rather than as fast.
        let bad = Split { to_device: 0, on_host: 32 };
        assert!(p.latency_s(&bad, S).is_infinite(),
                "with no residual bandwidth, host-side work cannot complete — reporting a finite \
                 time for it would make the worst plan look like the best");
    }

    /// ⭐ THE LOAD-BEARING CLAIM: energy is linear in q, so its optimum is a corner and can never be
    /// the interior latency optimum unless the two paths cost exactly the same per byte.
    #[test]
    fn the_energy_optimum_is_a_corner_and_the_latency_optimum_is_not() {
        let p = prof();
        let m = 64u64;
        let e = EnergyModel::from_measurements(120.0, 1 << 30, 400.0, 1 << 30).unwrap();
        let best_e = p.split_for_energy(m, &e);
        assert!(best_e.to_device == 0 || best_e.to_device == m, "energy optimum was interior");

        // Exhaustive: no q beats the corner on joules.
        let ej = p.joules(&best_e, S, &e);
        for q in 0..=m {
            let j = p.joules(&Split { to_device: q, on_host: m - q }, S, &e);
            assert!(j >= ej - 1e-9, "q={q} costs {j:.3} J, less than the claimed optimum {ej:.3}");
        }
        // And the two objectives genuinely disagree here.
        let best_t = p.split_for_latency(m);
        assert_ne!(best_t, best_e,
                   "latency and energy optima coincided; this fixture is supposed to separate them");
        assert!(p.joules(&best_t, S, &e) > ej, "the latency plan should cost MORE joules");
        assert!(p.latency_s(&best_e, S) > p.latency_s(&best_t, S), "and be SLOWER");
    }

    /// The constrained optimum must degenerate to each pure objective at the ends of the budget
    /// range — otherwise it is a third thing rather than the interpolation it claims to be.
    #[test]
    fn the_budgeted_split_becomes_each_pure_objective_at_the_ends_of_its_range() {
        let p = prof();
        let m = 64u64;
        let e = EnergyModel::from_measurements(120.0, 1 << 30, 400.0, 1 << 30).unwrap();
        let fastest = p.latency_s(&p.split_for_latency(m), S);

        // ⚠ THE LATENCY OPTIMUM IS NOT UNIQUE, and asserting split equality here was wrong. At this
        // fixture q=25 and q=26 both take exactly max(25/8, 39/12) = 3.25 units — the V's vertex
        // falls between two integers that tie. So the tight-budget plan cannot be required to EQUAL
        // `split_for_latency`; what it must do is achieve the same latency, and among the tied plans
        // pick the one that is cheaper in joules. That is strictly stronger, and it is the behaviour
        // that makes the budgeted call worth preferring over the pure one.
        let tight = p.split_within_budget(m, S, &e, fastest * 1.000001).expect("q* is feasible");
        assert!((p.latency_s(&tight, S) - fastest).abs() < 1e-9,
                "tight budget gave {:.6}s against the achievable {fastest:.6}s", p.latency_s(&tight, S));
        assert!(p.joules(&tight, S, &e) <= p.joules(&p.split_for_latency(m), S, &e) + 1e-9,
                "at equal latency the budgeted plan should never cost MORE joules: {:.3} vs {:.3}",
                p.joules(&tight, S, &e), p.joules(&p.split_for_latency(m), S, &e));

        // Generous: the energy corner fits, so take it.
        let loose = p.split_within_budget(m, S, &e, fastest * 100.0).expect("feasible");
        assert_eq!(loose, p.split_for_energy(m, &e));

        // Below the achievable minimum: say so, do not return a plan that will miss.
        assert!(p.split_within_budget(m, S, &e, fastest * 0.5).is_none(),
                "returned a plan for a budget below what the machine can do");
    }

    /// Every budget in a fine sweep must produce a plan that ACTUALLY meets it and is the cheapest
    /// such plan. Checked against brute force over all m+1 splits, because the closed form is where
    /// an off-by-one hides and brute force has nowhere to hide one.
    #[test]
    fn the_budgeted_split_matches_brute_force_at_every_budget() {
        let p = prof();
        let m = 48u64;
        for (dj, hj) in [(120.0, 400.0), (400.0, 120.0), (250.0, 250.0)] {
            let e = EnergyModel::from_measurements(dj, 1 << 30, hj, 1 << 30).unwrap();
            let fastest = p.latency_s(&p.split_for_latency(m), S);
            for step in 0..40 {
                let budget = fastest * (0.8 + 0.1 * step as f64);
                let got = p.split_within_budget(m, S, &e, budget);
                let want = (0..=m)
                    .map(|q| Split { to_device: q, on_host: m - q })
                    .filter(|s| p.latency_s(s, S) <= budget)
                    .min_by(|a, b| p.joules(a, S, &e).partial_cmp(&p.joules(b, S, &e)).unwrap());
                match (got, want) {
                    (None, None) => {}
                    (Some(g), Some(w)) => assert!(
                        (p.joules(&g, S, &e) - p.joules(&w, S, &e)).abs() < 1e-6,
                        "budget {budget:.4}s: closed form gave {g:?} ({:.2} J), brute force found \
                         {w:?} ({:.2} J)", p.joules(&g, S, &e), p.joules(&w, S, &e)),
                    (g, w) => panic!("budget {budget:.4}s: feasibility disagrees — {g:?} vs {w:?}"),
                }
            }
        }
    }

    #[test]
    fn an_energy_model_needs_bytes_on_both_paths() {
        assert!(EnergyModel::from_measurements(1.0, 0, 1.0, 100).is_err());
        assert!(EnergyModel::from_measurements(1.0, 100, 1.0, 0).is_err());
        assert!(EnergyModel::from_measurements(1.0, 100, 1.0, 100).is_ok());
    }
}
