//! **Heterogeneous compute: use the whole machine, not one unit of it.**
//!
//! Every runtime in the 2026 survey picks a backend. CUDA *or* Metal *or* CPU *or* an NPU. The choice is
//! presented as a portability question, and it is really an accounting failure: a machine with an idle
//! CPU while its GPU saturates is a machine running at a fraction of the hardware someone paid for.
//!
//! The barrier between compute modules is a software convention, not a property of the silicon.
//!
//! ## Why this is not merely tidy, on the evidence
//!
//! Decode is **bandwidth-bound**, not FLOPs-bound. Measured in this workspace: a decode step reads
//! ~525 MB of weights per token and that read *is* the cost, which is why cutting 29% of GPU dispatches
//! changed wall time by 0.00 ms. Independently confirmed at the edge: MoE used **2.1x MORE** energy per
//! token than a dense model at matched active parameters, because *"on bandwidth-bound hardware,
//! inference cost tracks total parameters, not active ones"* (arXiv:2606.21428).
//!
//! That has a direct consequence people skip. When the bound is bandwidth, the question is not which
//! unit computes fastest. It is **how many independent paths to memory the machine has, and whether you
//! are using all of them.** On unified-memory parts the CPU and GPU share a controller but issue
//! independently and do not saturate it alone; on discrete parts they have genuinely separate pools. In
//! both cases a saturated GPU next to an idle CPU is leaving bandwidth on the floor.
//!
//! ## The rule this module enforces
//!
//! **A split ratio must be measured, never assumed.** Handing 30% of a matmul to the CPU because 30%
//! sounds right produces a slower system than not splitting at all, and the failure is silent: total
//! time becomes `max(gpu, cpu)` and the slow arm hides inside it. So [`Fabric::calibrate`] measures each
//! unit's actual throughput on the actual work, and [`Fabric::split`] apportions from those measurements.
//!
//! And the win is only real if the units run **concurrently**. Sequential dispatch across two units is
//! strictly worse than one unit. [`Split::wall_seconds`] therefore reports `max`, not `sum`, and
//! [`Split::speedup_vs_best_single`] compares against the best single unit rather than against the
//! slowest, because beating the worse option is not an achievement.

use crate::{Class, Meter, Reading};

/// A compute unit on this machine.
///
/// Deliberately not an enum of vendors. What matters for scheduling is the *shape* of the unit: how it
/// reaches memory and what it is good at, not whose logo is on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Unit {
    /// Programmable GPU cores. wgpu reaches these on Metal, Vulkan, DX12 and WebGPU.
    Gpu,
    /// CPU vector units: NEON, AVX-512, VNNI. An independent issue path to memory, and on quantised
    /// integer work with dot-product instructions it is not the poor relation people assume.
    CpuSimd,
    /// A fixed-function neural accelerator: Apple's ANE, a Qualcomm HTP, an Intel NPU. Very efficient on
    /// the shapes it supports and useless outside them, which is why it is a unit and not a backend.
    ///
    /// ⚠ **Declared, not implemented, and it will not arrive the way `CpuSimd` did.** Checked on this
    /// hardware: CoreML is present and the ANE is visible in IOKit, but it does not accept a dispatched
    /// kernel. You compile a model *graph* to `.mlmodelc` and the framework decides what lands on ANE
    /// versus GPU versus CPU, in fp16, on the op shapes it supports. There is no path to handing it a
    /// Q8_0 matvec and getting a span of rows back, which is the contract [`Fabric::split`] assumes.
    ///
    /// So filling this variant is not the next increment after `CpuSimd`. It is a second execution model
    /// living beside the kernel-dispatch one, and pretending otherwise would put a unit in the fabric
    /// that silently never receives work.
    Npu,
}

impl Unit {
    pub fn label(self) -> &'static str {
        match self {
            Unit::Gpu => "gpu",
            Unit::CpuSimd => "cpu-simd",
            Unit::Npu => "npu",
        }
    }
}

/// What one unit can actually do, measured rather than declared.
#[derive(Debug, Clone, Copy)]
pub struct Capability {
    pub unit: Unit,
    /// Work items per second, measured by [`Fabric::calibrate`] on the real workload.
    pub throughput: f64,
    /// Joules per work item, when a meter was available. `None` is honest and common.
    pub joules_per_item: Option<f64>,
}

impl Capability {
    /// Items per joule. The scheduling objective when the goal is energy rather than latency, and
    /// `None` when nothing was measurable, which must not silently become zero.
    pub fn items_per_joule(&self) -> Option<f64> {
        self.joules_per_item.filter(|j| *j > 0.0).map(|j| 1.0 / j)
    }
}

/// The compute units present on this machine, with their measured capabilities.
#[derive(Debug, Clone, Default)]
pub struct Fabric {
    caps: Vec<Capability>,
}

impl Fabric {
    pub fn new() -> Self { Self::default() }

    /// Measure one unit by running `work(n)` and timing it.
    ///
    /// The closure takes an item count so calibration uses the *real* workload rather than a synthetic
    /// proxy. A unit calibrated on a microbenchmark and deployed on a matmul has been calibrated on the
    /// wrong thing, and the split ratio will be wrong in a way nothing downstream can detect.
    pub fn calibrate<M: Meter>(
        &mut self,
        unit: Unit,
        meter: Option<&M>,
        items: u64,
        mut work: impl FnMut(u64),
    ) -> Capability {
        let e0 = meter.and_then(|m| m.read_joules());
        let t0 = std::time::Instant::now();
        work(items);
        let secs = t0.elapsed().as_secs_f64();
        let e1 = meter.and_then(|m| m.read_joules());

        let joules_per_item = match (e0, e1) {
            (Some(a), Some(b)) if b >= a && items > 0 => Some((b - a) / items as f64),
            _ => None,
        };
        let cap = Capability {
            unit,
            throughput: if secs > 0.0 { items as f64 / secs } else { 0.0 },
            joules_per_item,
        };
        self.caps.retain(|c| c.unit != unit);
        self.caps.push(cap);
        cap
    }

    pub fn units(&self) -> &[Capability] { &self.caps }
    pub fn is_empty(&self) -> bool { self.caps.is_empty() }

    /// The single fastest unit. What a conventional runtime would pick, and the bar a split must clear.
    pub fn best_single(&self) -> Option<Capability> {
        self.caps.iter().copied().max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())
    }

    /// The single most energy-efficient unit, when energy was measurable.
    pub fn most_efficient(&self) -> Option<Capability> {
        self.caps.iter().copied()
            .filter(|c| c.items_per_joule().is_some())
            .max_by(|a, b| a.items_per_joule().unwrap().partial_cmp(&b.items_per_joule().unwrap()).unwrap())
    }

    /// Apportion `items` across every unit in proportion to measured throughput.
    ///
    /// Proportional-to-throughput is the allocation that makes all units finish at the same moment,
    /// which is the allocation that minimises `max` over the units. Any other split leaves someone idle
    /// while someone else is still working.
    pub fn split(&self, items: u64) -> Option<Split> {
        if self.caps.is_empty() { return None; }
        let total: f64 = self.caps.iter().map(|c| c.throughput).sum();
        if total <= 0.0 { return None; }

        let mut shares: Vec<(Capability, u64)> = Vec::with_capacity(self.caps.len());
        let mut assigned = 0u64;
        for (i, c) in self.caps.iter().enumerate() {
            let n = if i + 1 == self.caps.len() {
                items.saturating_sub(assigned)   // last unit absorbs the rounding, so nothing is lost
            } else {
                ((items as f64) * c.throughput / total).round() as u64
            };
            assigned += n;
            shares.push((*c, n));
        }
        Some(Split { shares, items })
    }
}

/// A work apportionment across units, and what it is predicted to cost.
#[derive(Debug, Clone)]
pub struct Split {
    pub shares: Vec<(Capability, u64)>,
    pub items: u64,
}

impl Split {
    /// Wall time if the units run **concurrently**: the slowest arm, not the sum.
    ///
    /// This is `max` and that is the whole point. If a caller dispatches sequentially, this number is a
    /// lie and the split is strictly worse than using one unit. Concurrency is not an optimisation on
    /// top of splitting; it is the thing that makes splitting worth doing.
    pub fn wall_seconds(&self) -> f64 {
        self.shares.iter()
            .map(|(c, n)| if c.throughput > 0.0 { *n as f64 / c.throughput } else { f64::INFINITY })
            .fold(0.0f64, f64::max)
    }

    /// Total joules, which unlike time DOES sum: every unit draws power while it works.
    ///
    /// Splitting can therefore be faster and less efficient at the same time, and reporting only the
    /// speedup would hide that. `None` when any participating unit had no meter.
    pub fn joules(&self) -> Option<f64> {
        self.shares.iter()
            .map(|(c, n)| c.joules_per_item.map(|j| j * *n as f64))
            .sum::<Option<f64>>()
    }

    /// Speedup against the **best** single unit, not the worst.
    ///
    /// Beating the slower option is not an achievement, and a split that loses to the best single unit
    /// returns a value below 1.0 rather than being quietly reported as a win.
    pub fn speedup_vs_best_single(&self) -> f64 {
        let best = self.shares.iter()
            .map(|(c, _)| c.throughput)
            .fold(0.0f64, f64::max);
        if best <= 0.0 || self.wall_seconds() <= 0.0 { return f64::NAN; }
        (self.items as f64 / best) / self.wall_seconds()
    }

    /// Whether splitting **looks** worth it, from calibration alone.
    ///
    /// ⚠ This is a PREDICTION and it does not include coordination cost: thread spawn and join, a second
    /// dispatch, and the buffer setup each unit needs. Measured on a 4.5 MB matmul on an 18-core machine,
    /// this returned a predicted 1.61x while the split actually ran 1.6x SLOWER than the best single
    /// unit, because the whole job took 0.5 ms and spawning threads cost more than the work.
    ///
    /// So treat an `Ok` here as permission to run the A/B, never as the result of one. Use
    /// [`measured_speedup`] to settle it, which is the only thing that does.
    pub fn worthwhile(&self) -> Result<(), String> {
        if self.shares.len() < 2 {
            return Err("only one unit: nothing to split across".into());
        }
        let s = self.speedup_vs_best_single();
        if !(s > 1.0) {
            return Err(format!("split is {s:.2}x against the best single unit, so it is a regression"));
        }
        if s < 1.05 {
            return Err(format!("split gains only {s:.2}x, which will not survive the coordination cost \
                                of dispatching to two units and joining them"));
        }
        Ok(())
    }

    /// The only honest verdict: run both arms and compare wall clock.
    ///
    /// `split_secs` is the measured concurrent wall time of the split; `single_secs` the measured time
    /// of the best single unit on the same work. Returns the real ratio, which is frequently below 1.0
    /// on small workloads however good the prediction looked.
    ///
    /// The gap between this and [`speedup_vs_best_single`] is exactly the coordination cost, and naming
    /// it is more useful than pretending calibration captured it.
    pub fn measured_speedup(&self, split_secs: f64, single_secs: f64) -> f64 {
        if split_secs <= 0.0 { return f64::NAN; }
        single_secs / split_secs
    }

    /// Coordination cost implied by a measurement: the time the split spent on something other than the
    /// work itself. Positive means overhead, which is the normal case.
    pub fn coordination_seconds(&self, split_secs: f64) -> f64 {
        split_secs - self.wall_seconds()
    }

    /// Energy per item under this split, for comparison against a single unit's figure.
    pub fn joules_per_item(&self) -> Option<f64> {
        self.joules().filter(|_| self.items > 0).map(|j| j / self.items as f64)
    }
}

impl std::fmt::Display for Split {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (c, n) in &self.shares {
            let pct = if self.items > 0 { 100.0 * *n as f64 / self.items as f64 } else { 0.0 };
            writeln!(f, "  {:<9} {:>7} items ({:>5.1}%)  {:>10.1} items/s", c.unit.label(), n, pct, c.throughput)?;
        }
        write!(f, "  wall {:.4} s concurrent, {:.2}x vs best single unit{}",
               self.wall_seconds(), self.speedup_vs_best_single(),
               match self.joules_per_item() { Some(j) => format!(", {j:.4} J/item"), None => String::new() })
    }
}

/// A reading tagged with which unit produced it, so a fabric-wide total is attributable.
pub fn attribute(unit: Unit, r: Reading) -> (Unit, Reading) { (unit, r) }

/// Sum readings across units into one figure, keeping the weakest class.
///
/// Total energy across concurrently-running units is a sum; the class is the worst of the parts, since a
/// total is only as sound as its least sound term.
pub fn fabric_total(readings: &[(Unit, Reading)]) -> Option<Reading> {
    let first = readings.first()?.1;
    let joules = readings.iter().map(|(_, r)| r.joules).sum();
    let seconds = readings.iter().map(|(_, r)| r.seconds).fold(0.0f64, f64::max); // concurrent
    let class = readings.iter().map(|(_, r)| r.class).max().unwrap_or(Class::Estimated);
    Some(Reading { joules, seconds, class, source: "fabric:sum", boundary: first.boundary })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Boundary, Nameplate};

    fn cap(unit: Unit, tput: f64, jpi: Option<f64>) -> Capability {
        Capability { unit, throughput: tput, joules_per_item: jpi }
    }

    #[test]
    fn a_split_apportions_in_proportion_to_measured_throughput() {
        // 3:1 throughput means 75/25, which is the allocation where both finish at the same instant.
        let f = Fabric { caps: vec![cap(Unit::Gpu, 300.0, None), cap(Unit::CpuSimd, 100.0, None)] };
        let s = f.split(400).unwrap();
        assert_eq!(s.shares[0].1, 300);
        assert_eq!(s.shares[1].1, 100);
        // Both arms take 1.0 s, so wall time is 1.0 s and not 2.0 s.
        assert!((s.wall_seconds() - 1.0).abs() < 1e-9, "wall {}", s.wall_seconds());
    }

    #[test]
    fn wall_time_is_max_not_sum_because_the_units_run_concurrently() {
        // If this were a sum, splitting would always look worse than one unit and nobody would do it.
        // If a caller dispatches sequentially, this number is a lie, which the docs say plainly.
        let f = Fabric { caps: vec![cap(Unit::Gpu, 100.0, None), cap(Unit::CpuSimd, 100.0, None)] };
        let s = f.split(200).unwrap();
        assert!((s.wall_seconds() - 1.0).abs() < 1e-9);
        assert!((s.speedup_vs_best_single() - 2.0).abs() < 1e-9, "two equal units must give 2x");
    }

    #[test]
    fn energy_sums_even_though_time_does_not() {
        // The trap: a split can be FASTER and LESS EFFICIENT at once, because every unit draws power
        // while it works. Reporting only the speedup hides that.
        let f = Fabric { caps: vec![cap(Unit::Gpu, 100.0, Some(1.0)), cap(Unit::CpuSimd, 100.0, Some(4.0))] };
        let s = f.split(200).unwrap();
        assert!((s.speedup_vs_best_single() - 2.0).abs() < 1e-9, "2x faster");
        // 100 items at 1 J + 100 at 4 J = 500 J, i.e. 2.5 J/item against the GPU's 1.0 J/item alone.
        assert!((s.joules().unwrap() - 500.0).abs() < 1e-9);
        assert!(s.joules_per_item().unwrap() > 1.0, "the split must be reported as less efficient per item");
    }

    #[test]
    fn a_prediction_is_not_a_measurement_and_the_api_says_so() {
        // Measured on a 4.5 MB matmul, 18 cores: predicted 1.61x, actual 1.6x SLOWER, because the job
        // took 0.5 ms and spawning threads cost more than the work. This is the guard against reading
        // worthwhile() as a verdict.
        let f = Fabric { caps: vec![cap(Unit::Gpu, 4_831_613.0, None), cap(Unit::CpuSimd, 7_864_944.0, None)] };
        let s = f.split(4096).unwrap();
        assert!(s.worthwhile().is_ok(), "calibration predicts a win");
        assert!(s.speedup_vs_best_single() > 1.5, "predicted speedup");
        // The A/B that settles it: 0.8 ms split against 0.5 ms single.
        let real = s.measured_speedup(0.0008, 0.0005);
        assert!(real < 1.0, "measured {real:.2}x should be a regression");
        assert!(s.coordination_seconds(0.0008) > 0.0, "coordination cost must be visible");
    }

    #[test]
    fn a_split_that_loses_to_the_best_single_unit_is_refused() {
        // A unit 100x slower contributes almost nothing and is not worth coordinating with.
        let f = Fabric { caps: vec![cap(Unit::Gpu, 1000.0, None), cap(Unit::Npu, 1.0, None)] };
        let s = f.split(1000).unwrap();
        assert!(s.worthwhile().is_err(), "a negligible second unit was accepted as worthwhile");
    }

    #[test]
    fn one_unit_is_not_a_split() {
        let f = Fabric { caps: vec![cap(Unit::Gpu, 100.0, None)] };
        assert!(f.split(10).unwrap().worthwhile().unwrap_err().contains("only one unit"));
    }

    #[test]
    fn no_items_are_lost_to_rounding() {
        // The last unit absorbs the remainder, because a split that silently drops work would report a
        // speedup for doing less.
        let f = Fabric { caps: vec![cap(Unit::Gpu, 7.0, None), cap(Unit::CpuSimd, 3.0, None), cap(Unit::Npu, 1.0, None)] };
        for n in [1u64, 7, 99, 1000, 12345] {
            let s = f.split(n).unwrap();
            assert_eq!(s.shares.iter().map(|(_, k)| k).sum::<u64>(), n, "lost work at n={n}");
        }
    }

    #[test]
    fn calibration_uses_the_real_workload_and_records_no_energy_when_unmeasurable() {
        let mut f = Fabric::new();
        let c = f.calibrate::<Nameplate>(Unit::Gpu, None, 1000, |n| {
            for _ in 0..n { std::hint::black_box(1u64 + 1); }
        });
        assert!(c.throughput > 0.0);
        assert!(c.joules_per_item.is_none(), "no meter must mean no energy figure, not a zero");
        assert!(c.items_per_joule().is_none());
    }

    #[test]
    fn the_most_efficient_unit_is_not_always_the_fastest() {
        // The reason both accessors exist. Scheduling for latency and scheduling for joules are
        // different objectives and can pick different units.
        let f = Fabric { caps: vec![
            cap(Unit::Gpu, 1000.0, Some(10.0)),     // fast, thirsty
            cap(Unit::Npu, 200.0, Some(0.5)),       // slow, frugal
        ] };
        assert_eq!(f.best_single().unwrap().unit, Unit::Gpu);
        assert_eq!(f.most_efficient().unwrap().unit, Unit::Npu);
    }

    #[test]
    fn a_fabric_total_takes_max_time_and_summed_energy_and_the_worst_class() {
        let r = |j: f64, s: f64, c: Class| Reading { joules: j, seconds: s, class: c, source: "u", boundary: Boundary::DEVICE };
        let t = fabric_total(&[(Unit::Gpu, r(10.0, 2.0, Class::Measured)),
                               (Unit::CpuSimd, r(4.0, 1.0, Class::Derived))]).unwrap();
        assert_eq!(t.joules, 14.0);
        assert_eq!(t.seconds, 2.0, "concurrent units: time is max");
        assert_eq!(t.class, Class::Derived, "a total is only as sound as its weakest term");
    }
}
