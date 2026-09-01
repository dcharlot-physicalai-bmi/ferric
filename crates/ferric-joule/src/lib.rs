//! # ferric-joule — joules per task, measured, or not reported at all.
//!
//! Every efficiency claim in this field is a ratio, and almost none of them say what the denominator
//! was. A training framework advertises 83% less energy; its own technical report shows the baseline
//! running at 3.4% model-FLOPs utilisation, where competent practice is 40-50%. The number is real and
//! the claim is not, because the baseline was broken rather than the method good.
//!
//! That failure is not caught by being careful. It is caught by making it impossible to state a saving
//! without stating what it was measured against. So that is this crate's entire contract:
//!
//! 1. **A [`Reading`] carries its measurement class.** A joule figure that came from a power sensor and
//!    one that came from multiplying a nameplate TDP by a utilisation guess are different kinds of
//!    object, and they do not silently mix. See [`Class`].
//! 2. **A [`Saving`] cannot be constructed by hand.** The only ways to obtain one are [`compare`] and
//!    [`compare_tasks`], which take two closures, run both, and keep both readings. There is no
//!    `Saving::new`. If you have a saving, you have the baseline, because the type system would not
//!    let you have it otherwise — enforced by `#[non_exhaustive]` and a `compile_fail` doc-test on
//!    [`Saving`], which is an external crate at test time and so is the only place this is checkable.
//!    Until 2026-08-21 this paragraph was prose and every field was `pub`; the property it asserted
//!    was simply absent.
//! 4. **A success count is observed, not accepted.** [`compare_tasks`] grades each task itself and
//!    tallies per arm, because joules-per-completed-task turns entirely on that denominator and a
//!    denominator supplied by the claimant is the failure mode in the paragraph above this list.
//! 3. **An unavailable meter is an error, not a zero.** When no sensor can be read, [`Meter::read`]
//!    returns `None` and every downstream figure is `None`. It never falls back to an estimate wearing
//!    a measurement's clothes.
//!
//! ## What can actually be measured, honestly
//!
//! Less than the field implies. On this hardware right now:
//!
//! | platform | source | class | catch |
//! |---|---|---|---|
//! | NVIDIA | NVML `power.draw` | measured | GPU board only, excludes host and DRAM |
//! | Intel/AMD | RAPL `energy_uj` | measured | package domain, excludes GPU |
//! | Apple | `powermetrics` | measured | needs root, so unusable unattended |
//! | Apple | battery `InstantAmperage` × `Voltage` | measured | only while discharging, and it is whole-system |
//! | anything | TDP × utilisation | estimated | not a measurement, and labelled so |
//!
//! Refusing to report is a feature. A framework that always produces a number teaches its users that a
//! number is always available, and it is not.

#![forbid(unsafe_code)]

pub mod budget;
pub mod compaction;
pub mod fabric;
pub mod ladder;
pub mod recursion;
pub mod router;
pub use budget::{Budget, Budgeted, Outcome, Stop};
pub use compaction::{Decision, FitError, Policy, StepCost};
pub use fabric::{Capability, Fabric, Split, Unit};
pub use ladder::{Ladder, Routed, Trail};
pub use recursion::{Model as RecursionModel, Shape};
pub use router::{prompt_len_bucket, Calibration, OnlineRate, Plan, PlanError, Predictor, Profile, Router, RungProfile, Uniform};

use std::time::{Duration, Instant};

/// How a joule figure was arrived at. This rides with every reading and never gets dropped.
///
/// The ordering matters and is deliberate: `Measured < Derived < Estimated` by trustworthiness, and a
/// comparison between two readings takes the *worse* of the two, because a difference is only as sound
/// as its weaker half.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    /// A power or energy sensor was read. The number came from the hardware.
    Measured,
    /// Computed from a model that was itself validated against measurement on this hardware.
    Derived,
    /// Nameplate arithmetic. Useful for sizing, never for a claim.
    Estimated,
}

impl Class {
    pub fn label(self) -> &'static str {
        match self {
            Class::Measured => "measured",
            Class::Derived => "derived",
            Class::Estimated => "estimated",
        }
    }
    /// Whether a figure of this class may back a public efficiency claim.
    ///
    /// Estimates may not. This is the crate having an opinion, and it is the opinion the field lacks.
    pub fn claimable(self) -> bool {
        matches!(self, Class::Measured | Class::Derived)
    }
}

/// What the meter's number actually encloses.
///
/// This moves the answer more than the technology does. Google measured the SAME Gemini prompt on the
/// SAME fleet at 0.10 Wh accelerator-only and 0.24 Wh counting host CPU, DRAM, provisioned-idle machines
/// and facility overhead: **2.4x from accounting alone** (arXiv:2508.15734). Across published methods one
/// model on one task spans 6.2x. A joules figure without its boundary is not a measurement, so this rides
/// with every reading and `Saving::claimable` refuses a comparison whose two arms enclose different things.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Boundary {
    /// The accelerator itself.
    pub accelerator: bool,
    /// Host CPU and DRAM. Excluded by most published figures.
    pub host: bool,
    /// Provisioned-but-idle capacity. The term that makes a fleet number honest.
    pub idle: bool,
    /// Facility overhead (PUE). Datacenter figures that omit it understate by 10-30%.
    pub facility: bool,
}

impl Boundary {
    /// What a hardware counter on a single device sees: the silicon, nothing around it.
    pub const DEVICE: Boundary = Boundary { accelerator: true, host: false, idle: false, facility: false };
    /// Whole system at the wall, which is what a battery or a plug meter reads.
    pub const SYSTEM: Boundary = Boundary { accelerator: true, host: true, idle: true, facility: false };
    pub fn label(&self) -> String {
        let mut v = vec![];
        if self.accelerator { v.push("accel"); }
        if self.host { v.push("host"); }
        if self.idle { v.push("idle"); }
        if self.facility { v.push("pue"); }
        if v.is_empty() { "unspecified".into() } else { v.join("+") }
    }
}

/// Energy consumed over an interval, with provenance attached.
#[derive(Debug, Clone, Copy)]
pub struct Reading {
    pub joules: f64,
    pub seconds: f64,
    pub class: Class,
    /// Which sensor or model produced this, named so a reader can go and check it.
    pub source: &'static str,
    /// What the number encloses. See [`Boundary`].
    pub boundary: Boundary,
}

impl Reading {
    pub fn watts(&self) -> f64 {
        if self.seconds <= 0.0 { 0.0 } else { self.joules / self.seconds }
    }
    /// Joules per task, the Institute's standing metric. `n` is the number of tasks in the interval.
    pub fn per_task(&self, n: u64) -> f64 {
        if n == 0 { f64::NAN } else { self.joules / n as f64 }
    }
}

impl std::fmt::Display for Reading {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.3} J over {:.3} s ({:.1} W) [{} via {}]",
               self.joules, self.seconds, self.watts(), self.class.label(),
               format!("{} :: {}", self.source, self.boundary.label()))
    }
}

/// A source of energy readings.
///
/// Implementors sample a counter; the crate handles differencing and timing. `read_joules` returns a
/// monotonically increasing total, or `None` when the sensor is unavailable, which is not the same as
/// zero and must never be conflated with it.
pub trait Meter {
    /// Cumulative joules since some fixed origin, or `None` if unreadable right now.
    fn read_joules(&self) -> Option<f64>;
    fn class(&self) -> Class;
    fn source(&self) -> &'static str;
    /// What this meter's number encloses. See [`Boundary`].
    fn boundary(&self) -> Boundary;

    /// Whether this meter can currently produce readings. Checked before a run rather than after, so a
    /// benchmark fails at the start instead of producing hours of nothing.
    fn available(&self) -> bool {
        self.read_joules().is_some()
    }
}

/// Run `f` and measure the energy it consumed.
///
/// Returns `None` when the meter is unavailable, which propagates all the way out. That is deliberate:
/// the alternative is a zero, and a zero is indistinguishable from a very efficient run.
pub fn measure<M: Meter, T>(meter: &M, f: impl FnOnce() -> T) -> (T, Option<Reading>) {
    let e0 = meter.read_joules();
    let t0 = Instant::now();
    let out = f();
    let dt = t0.elapsed().as_secs_f64();
    let e1 = meter.read_joules();
    let reading = match (e0, e1) {
        (Some(a), Some(b)) if b >= a => Some(Reading {
            joules: b - a, seconds: dt, class: meter.class(), source: meter.source(),
            boundary: meter.boundary(),
        }),
        // A counter that went backwards wrapped or was reset. Report nothing rather than a wrong number.
        _ => None,
    };
    (out, reading)
}

/// The result of measuring two arms of the same task. **The only way to obtain a saving.**
///
/// There is deliberately no constructor. `Saving` is produced solely by [`compare`], so possessing one
/// is proof that both arms were actually run, on the same meter, in the same process. A percentage
/// pulled from a slide cannot be turned into this type.
/// Constructing one outside this crate is a compile error, and that is load-bearing rather than
/// stylistic:
///
/// ```compile_fail
/// use ferric_joule::{Saving, Reading, Class, Boundary};
/// let r = Reading { joules: 1.0, seconds: 1.0, class: Class::Measured, source: "m", boundary: Boundary::DEVICE };
/// // `Saving` is #[non_exhaustive]: no struct literal, so no saving without two measured arms.
/// let _ = Saving { baseline: r, candidate: r, tasks: 1, successes: (1, 1) };
/// ```
///
/// That doc-test is the enforcement. The module header claimed this property from the beginning and
/// the type did not have it until 2026-08-21 — every field was `pub`, so any crate could write the
/// literal and hand-assemble a "saving" with no baseline behind it. A contract stated in prose and
/// not in the type is a comment. Note the remaining gap, stated rather than hidden: [`Reading`] IS
/// still constructible by hand, so a fabricated *reading* is possible; what is not possible is
/// assembling two of them into the object that backs a claim.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Saving {
    pub baseline: Reading,
    pub candidate: Reading,
    /// Tasks ATTEMPTED by each arm.
    pub tasks: u64,
    /// Tasks that SUCCEEDED, **per arm**. This is the denominator that matters, and it is separate
    /// from `tasks` because unattended automation pays full price for its failures.
    ///
    /// It is a PAIR, and that is the whole point. This field was a single `u64` shared by both arms
    /// until 2026-08-21, which quietly made it decorative: dividing both arms by the same number
    /// changes the absolute figures and leaves every ratio — [`Saving::fraction`], [`Saving::percent`],
    /// the ranking — bit-identical to [`Saving::per_attempt`]. A field that cannot change any
    /// comparison cannot correct one, so the crate's central argument was unstatable in the crate's own
    /// type. Two arms that succeed at the same rate is a *measurement*, never an assumption.
    ///
    /// Measured: in one agentic run, 2,256 J of 3,614 J (62.4%) went to a failed attempt before the
    /// successful retry (arXiv:2605.22883). On GAIA, the model burning 7.31 kJ per query scored 16.4%
    /// and the one burning 1.18 kJ scored 5.5% (arXiv:2511.07885) — per *successful* goal that is
    /// 44.6 kJ against 21.5 kJ. Those numbers **narrow** the gap from 6.19x to 2.08x; they do not
    /// reverse it, and this doc said "reverses" until the arithmetic was checked. The reversal is
    /// real but needs a wider success gap than GAIA's, and `energy_per_success_can_reverse_the_ranking`
    /// constructs the smallest one that does it.
    pub successes: (u64, u64),
}

impl Saving {
    /// Fraction of energy removed, in `[0, 1]`. Negative when the candidate is worse, which happens and
    /// should be reportable rather than hidden.
    pub fn fraction(&self) -> f64 {
        if self.baseline.joules <= 0.0 { return f64::NAN; }
        (self.baseline.joules - self.candidate.joules) / self.baseline.joules
    }

    pub fn percent(&self) -> f64 { self.fraction() * 100.0 }

    /// Speed change, separate from energy. A candidate can be faster and use more energy, and reporting
    /// only one of the two is how a regression gets shipped as a win.
    pub fn speedup(&self) -> f64 {
        if self.candidate.seconds <= 0.0 { return f64::NAN; }
        self.baseline.seconds / self.candidate.seconds
    }

    /// Joules per SUCCESSFUL task, **each arm against its own success count**. The metric that
    /// actually matters.
    ///
    /// Energy per query flatters anything that fails cheaply; energy per token flatters anything
    /// terse. Neither is a unit of useful work.
    pub fn per_success(&self) -> (f64, f64) {
        (self.baseline.per_task(self.successes.0), self.candidate.per_task(self.successes.1))
    }

    /// The saving computed on the unit that matters, which can differ in SIGN from [`percent`].
    ///
    /// [`percent`] compares total energy, so an arm that fails more looks cheaper for failing. This
    /// charges each arm for the work it actually completed. When the two disagree, this one is the
    /// answer and the other is the artefact.
    ///
    /// [`percent`]: Saving::percent
    pub fn percent_per_success(&self) -> f64 {
        let (b, c) = self.per_success();
        if !(b > 0.0) { return f64::NAN; }
        (b - c) / b * 100.0
    }

    /// Fraction of attempts each arm completed, `(baseline, candidate)`.
    pub fn success_rate(&self) -> (f64, f64) {
        if self.tasks == 0 { return (f64::NAN, f64::NAN); }
        (self.successes.0 as f64 / self.tasks as f64, self.successes.1 as f64 / self.tasks as f64)
    }

    /// Joules per attempt, which is what most published figures actually report.
    pub fn per_attempt(&self) -> (f64, f64) {
        (self.baseline.per_task(self.tasks), self.candidate.per_task(self.tasks))
    }

    /// The weaker of the two classes. A comparison is only as sound as its worse half.
    pub fn class(&self) -> Class {
        self.baseline.class.max(self.candidate.class)
    }

    /// Whether this may back a public claim, and why not when it may not.
    ///
    /// The bar is deliberately awkward to clear, because every rejected reason here corresponds to a
    /// published claim that should not have been made.
    pub fn claimable(&self) -> Result<(), &'static str> {
        if !self.class().claimable() {
            return Err("at least one arm is an estimate, not a measurement");
        }
        if self.baseline.source != self.candidate.source {
            return Err("the two arms were measured by different meters, so the difference is not attributable");
        }
        if self.baseline.boundary != self.candidate.boundary {
            return Err("the two arms enclose different things: accounting alone moves a figure 2.4x, so this difference is not attributable to the method");
        }
        if self.tasks == 0 {
            return Err("no task count was recorded, so joules-per-task is undefined");
        }
        if self.successes.0 == 0 || self.successes.1 == 0 {
            return Err("an arm recorded no successes: energy per successful task is the unit, and zero successes at any energy is not an efficiency result");
        }
        if self.successes.0 > self.tasks || self.successes.1 > self.tasks {
            return Err("an arm recorded more successes than attempts, so the accounting is wrong before the energy is");
        }
        if self.baseline.seconds < 1.0 || self.candidate.seconds < 1.0 {
            return Err("an arm ran for under a second, which is inside the noise of every sensor listed in this crate");
        }
        Ok(())
    }
}

impl std::fmt::Display for Saving {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (b, c) = self.per_success();
        let (rb, rc) = self.success_rate();
        writeln!(f, "  baseline   {}", self.baseline)?;
        writeln!(f, "  candidate  {}", self.candidate)?;
        writeln!(f, "  per success {b:.4} J -> {c:.4} J  ({}/{} vs {}/{} succeeded, {:.0}% vs {:.0}%)",
                 self.successes.0, self.tasks, self.successes.1, self.tasks, rb * 100.0, rc * 100.0)?;
        // Both percentages, always. Printing only the first is how an arm that fails more gets
        // reported as an efficiency win for failing more cheaply.
        writeln!(f, "  per success saving {:.1}%{}", self.percent_per_success(),
                 if self.percent_per_success() * self.percent() < 0.0 {
                     "  ⚠ OPPOSITE SIGN to the energy saving below: the arms did not do equal work"
                 } else { "" })?;
        write!(f, "  saving     {:.1}% energy, {:.2}x speed [{}]  {}",
               self.percent(), self.speedup(), self.class().label(),
               match self.claimable() {
                   Ok(()) => "claimable".to_string(),
                   Err(why) => format!("NOT CLAIMABLE: {why}"),
               })
    }
}

/// Measure two arms of the same task and return their comparison.
///
/// Both arms must do the *same work*, and `tasks` records how much. The crate cannot check that for
/// you, but it records the number so a reader can, and [`Saving::claimable`] refuses a comparison with
/// no task count at all.
///
/// The arms are run baseline-first then candidate-first on alternate repetitions when `reps > 1`, so a
/// machine that warms up or throttles during the run does not systematically favour whichever arm went
/// second. That ordering bias is worth more than it sounds: it is the difference between measuring an
/// optimisation and measuring a thermal ramp.
pub fn compare<M: Meter>(
    meter: &M,
    tasks: u64,
    successes: (u64, u64),
    reps: usize,
    mut baseline: impl FnMut(),
    mut candidate: impl FnMut(),
) -> Option<Saving> {
    if !meter.available() { return None; }
    let reps = reps.max(1);
    let (mut bj, mut bs, mut cj, mut cs) = (0.0, 0.0, 0.0, 0.0);

    for i in 0..reps {
        // Alternate which arm runs first, so warm-up and thermal drift hit both equally.
        if i % 2 == 0 {
            let (_, b) = measure(meter, &mut baseline);
            let (_, c) = measure(meter, &mut candidate);
            let (b, c) = (b?, c?);
            bj += b.joules; bs += b.seconds; cj += c.joules; cs += c.seconds;
        } else {
            let (_, c) = measure(meter, &mut candidate);
            let (_, b) = measure(meter, &mut baseline);
            let (b, c) = (b?, c?);
            bj += b.joules; bs += b.seconds; cj += c.joules; cs += c.seconds;
        }
    }

    let n = reps as f64;
    Some(Saving {
        baseline: Reading { joules: bj / n, seconds: bs / n, class: meter.class(), source: meter.source(), boundary: meter.boundary() },
        candidate: Reading { joules: cj / n, seconds: cs / n, class: meter.class(), source: meter.source(), boundary: meter.boundary() },
        tasks,
        successes,
    })
}

/// Measure two arms over a **graded task set**, counting each arm's successes rather than believing them.
///
/// [`compare`] takes the success counts as arguments, which means the number that decides the whole
/// result — joules per completed task — arrives from outside the measurement, exactly like the
/// baselines this crate exists to distrust. Here the closures return `bool` per task and the crate
/// tallies them, so a `Saving` from this path cannot report a success rate no arm demonstrated.
///
/// Both arms see the SAME task slice in the same order. Energy is measured around the whole arm, not
/// per task, because every sensor listed in this crate is far too coarse for a single short task; the
/// per-task figure is division afterwards, and [`Saving::claimable`] refuses arms under a second.
///
/// Reps alternate which arm runs first, so thermal drift is not attributed to whichever went second.
pub fn compare_tasks<M: Meter, T>(
    meter: &M,
    tasks: &[T],
    reps: usize,
    mut baseline: impl FnMut(&T) -> bool,
    mut candidate: impl FnMut(&T) -> bool,
) -> Option<(Saving, Vec<bool>, Vec<bool>)> {
    if !meter.available() { return None; }
    if tasks.is_empty() { return None; }
    let reps = reps.max(1);
    let (mut bj, mut bs, mut cj, mut cs) = (0.0, 0.0, 0.0, 0.0);
    let (mut bok, mut cok) = (vec![false; tasks.len()], vec![false; tasks.len()]);

    for i in 0..reps {
        let mut run_b = |bok: &mut Vec<bool>| measure(meter, || {
            for (n, t) in tasks.iter().enumerate() { bok[n] = baseline(t); }
        });
        let mut run_c = |cok: &mut Vec<bool>| measure(meter, || {
            for (n, t) in tasks.iter().enumerate() { cok[n] = candidate(t); }
        });
        let (b, c) = if i % 2 == 0 {
            let (_, b) = run_b(&mut bok);
            let (_, c) = run_c(&mut cok);
            (b?, c?)
        } else {
            let (_, c) = run_c(&mut cok);
            let (_, b) = run_b(&mut bok);
            (b?, c?)
        };
        bj += b.joules; bs += b.seconds; cj += c.joules; cs += c.seconds;
    }

    let n = reps as f64;
    let mk = |j: f64, sec: f64| Reading {
        joules: j / n, seconds: sec / n, class: meter.class(), source: meter.source(), boundary: meter.boundary(),
    };
    let saving = Saving {
        baseline: mk(bj, bs),
        candidate: mk(cj, cs),
        tasks: tasks.len() as u64,
        successes: (bok.iter().filter(|x| **x).count() as u64,
                    cok.iter().filter(|x| **x).count() as u64),
    };
    Some((saving, bok, cok))
}

/// The task-set half of [`compare_tasks`] when no meter is available.
///
/// On a laptop on AC power there is no readable energy sensor, and this crate refuses to invent one.
/// That must not also mean the *success* half of the comparison cannot be measured — success rates
/// are what the energy figure would be divided BY, and they are measurable on any machine. This runs
/// both arms and returns the graded outcomes with wall-clock, and deliberately returns no [`Saving`],
/// because a saving without a meter is the thing this crate was written to prevent.
pub fn grade_tasks<T>(
    tasks: &[T],
    mut baseline: impl FnMut(&T) -> bool,
    mut candidate: impl FnMut(&T) -> bool,
) -> (Vec<bool>, Vec<bool>, (f64, f64)) {
    let t0 = Instant::now();
    let bok: Vec<bool> = tasks.iter().map(&mut baseline).collect();
    let bsec = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let cok: Vec<bool> = tasks.iter().map(&mut candidate).collect();
    (bok, cok, (bsec, t1.elapsed().as_secs_f64()))
}

// ---- meters ----

/// Linux RAPL, the package energy domain. Real joules from a hardware counter.
///
/// Covers the CPU package and, on most parts, its DRAM controller. It does **not** see a discrete GPU,
/// so on a machine doing GPU inference this measures the host only, and `source` says so.
pub struct Rapl {
    path: std::path::PathBuf,
}

impl Rapl {
    pub fn new() -> Option<Self> {
        let p = std::path::Path::new("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj");
        p.exists().then(|| Self { path: p.to_path_buf() })
    }
}

impl Meter for Rapl {
    fn read_joules(&self) -> Option<f64> {
        std::fs::read_to_string(&self.path).ok()?.trim().parse::<f64>().ok().map(|uj| uj / 1e6)
    }
    fn class(&self) -> Class { Class::Measured }
    fn source(&self) -> &'static str { "rapl:package" }
    fn boundary(&self) -> Boundary { Boundary { accelerator: false, host: true, idle: true, facility: false } }
}

/// NVIDIA board power via `nvidia-smi`, integrated over the interval.
///
/// Shelling out rather than linking NVML keeps this dependency-free and portable, at the cost of ~10 ms
/// per sample. That is fine for task-level measurement and useless for kernel-level, which is stated
/// here so nobody discovers it by confusion.
pub struct NvidiaSmi {
    last: std::cell::Cell<Option<(Instant, f64)>>,
}

impl NvidiaSmi {
    pub fn new() -> Option<Self> {
        let m = Self { last: std::cell::Cell::new(None) };
        m.watts().map(|_| m)
    }
    fn watts(&self) -> Option<f64> {
        let out = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=power.draw", "--format=csv,noheader,nounits"])
            .output().ok()?;
        String::from_utf8_lossy(&out.stdout).trim().lines().next()?.trim().parse().ok()
    }
}

impl Meter for NvidiaSmi {
    /// Trapezoidal integration of instantaneous power. Sampling, not a counter, so this is `Derived`:
    /// power between samples is interpolated rather than observed.
    fn read_joules(&self) -> Option<f64> {
        let w = self.watts()?;
        let now = Instant::now();
        let acc = match self.last.get() {
            Some((t0, prev_j)) => prev_j + w * now.duration_since(t0).as_secs_f64(),
            None => 0.0,
        };
        self.last.set(Some((now, acc)));
        Some(acc)
    }
    fn class(&self) -> Class { Class::Derived }
    fn source(&self) -> &'static str { "nvidia-smi:board" }
    fn boundary(&self) -> Boundary { Boundary::DEVICE }
}

/// Apple battery discharge: whole-system power, readable without root.
///
/// `InstantAmperage` is negative while discharging. This only works **on battery**, which is a real
/// restriction and the reason [`Meter::available`] is checked up front: plugged in, amperage reads zero
/// and this meter correctly reports that it cannot measure rather than reporting 0 J.
pub struct MacBattery {
    last: std::cell::Cell<Option<(Instant, f64)>>,
}

impl MacBattery {
    pub fn new() -> Option<Self> {
        let m = Self { last: std::cell::Cell::new(None) };
        m.watts().map(|_| m)
    }
    fn watts(&self) -> Option<f64> {
        let out = std::process::Command::new("ioreg")
            .args(["-c", "AppleSmartBattery", "-r"]).output().ok()?;
        let s = String::from_utf8_lossy(&out.stdout);
        let field = |k: &str| -> Option<f64> {
            s.lines().find(|l| l.contains(&format!("\"{k}\"")))?
                .split('=').nth(1)?.trim().parse().ok()
        };
        let amps_ma: f64 = field("InstantAmperage")?;
        let volts_mv: f64 = field("Voltage")?;
        // Discharging only. Zero or positive means mains powered, and there is nothing to measure.
        let draw = -amps_ma;
        if draw <= 0.0 { return None; }
        Some(draw / 1000.0 * volts_mv / 1000.0)
    }
}

impl Meter for MacBattery {
    fn read_joules(&self) -> Option<f64> {
        let w = self.watts()?;
        let now = Instant::now();
        let acc = match self.last.get() {
            Some((t0, prev)) => prev + w * now.duration_since(t0).as_secs_f64(),
            None => 0.0,
        };
        self.last.set(Some((now, acc)));
        Some(acc)
    }
    fn class(&self) -> Class { Class::Derived }
    fn source(&self) -> &'static str { "macos:battery-discharge" }
    fn boundary(&self) -> Boundary { Boundary::SYSTEM }
}

/// What a [`Macmon`] reading encloses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacmonScope {
    /// `cpu + gpu + ane + ram`: the SoC and its memory, which is what the computation actually
    /// draws. Excludes display, peripherals and anything charging a battery.
    Soc,
    /// `gpu + ram`: the accelerator and the memory it streams, and NOTHING host-side.
    ///
    /// A narrower boundary than [`MacmonScope::Soc`], and it exists because on a shared laptop the
    /// SoC figure is often unusable: `cpu_power` swings tens of watts with whatever else is running,
    /// while a GPU kernel's own draw sits at a fraction of a watt. Metering the accelerator rail is
    /// then the difference between a measurement and a noise report.
    ///
    /// ⚠ It genuinely EXCLUDES the host cost of driving the kernel — command encoding, submission,
    /// the driver. That is real work and it is not counted here, so an accelerator-scoped figure is
    /// never a per-token or per-task figure. [`Boundary`] carries the exclusion so a `Saving` cannot
    /// silently compare it against an SoC-scoped one.
    Accelerator,
    /// `sys_power`: the whole machine at the wall. On a laptop this includes the display and, while
    /// plugged in, whatever current is going into the battery — which is not the computation and
    /// does not belong in a per-token figure.
    System,
}

/// Apple Silicon SoC power, read WITHOUT root through `macmon`.
///
/// This is the meter that made an energy figure possible on this hardware at all. The battery route
/// ([`MacBattery`]) only works on battery, and `powermetrics` needs root; `macmon` reads the same
/// counters sudoless, so a plugged-in machine can be measured.
///
/// ## Why this is `Derived` and not `Measured`
///
/// `macmon` reports instantaneous POWER, so joules come from integrating samples over time. This
/// crate's existing line is exactly that: [`Rapl`] reads a cumulative ENERGY counter and is
/// `Measured`; anything that samples watts and integrates is `Derived`, as `nvidia-smi` and the
/// battery route already are. The number still comes from hardware sensors — what it inherits is
/// sampling error, not modelling error.
///
/// ## What it does NOT isolate
///
/// **These counters are system-wide, not per-process.** Everything else running on the machine is
/// inside the number. A figure attributed to one workload therefore needs an idle baseline measured
/// on the same machine in the same thermal state, and even then it is an attribution rather than an
/// isolation. [`Session`] and the caller must say so.
pub struct Macmon {
    scope: MacmonScope,
    state: std::sync::Arc<std::sync::Mutex<MacmonState>>,
    child: std::sync::Arc<std::sync::Mutex<Option<std::process::Child>>>,
}

/// One `macmon` line, kept whole. The context fields are not decoration: a power figure is only
/// reproducible if a reader can tell what state the machine was in, and the frequency columns are
/// the ones that actually move when the machine is throttling.
#[derive(Debug, Clone, Copy)]
pub struct MacmonSample {
    /// Seconds since the meter started, from the local clock at receipt.
    pub t: f64,
    pub cpu: f64,
    pub gpu: f64,
    pub ane: f64,
    pub ram: f64,
    pub sys: f64,
    pub cpu_temp: f64,
    pub gpu_temp: f64,
    pub pcpu_mhz: f64,
    pub ecpu_mhz: f64,
    pub gpu_mhz: f64,
    pub cpu_active: f64,
}

impl MacmonSample {
    /// `cpu + gpu + ane + ram`.
    ///
    /// **Not `all_power`**, which is `cpu + gpu + ane` and excludes DRAM entirely. RAM is 7–9% of
    /// the SoC total and rises for a bandwidth-bound kernel — and this crate's own arithmetic puts
    /// the tokenizer at ~8 FLOP per weight byte against the 100–300 a modern part needs to be
    /// compute-bound, so DRAM is a first-order term for exactly this workload.
    pub fn soc(&self) -> f64 {
        self.cpu + self.gpu + self.ane + self.ram
    }
}

#[derive(Default)]
struct MacmonState {
    joules: f64,
    watt_sum: f64,
    last: Option<Instant>,
    first: Option<Instant>,
    all: Vec<MacmonSample>,
}

impl Macmon {
    /// Start sampling. Returns `None` if `macmon` is not installed or produced no usable line
    /// within a short window — never a meter that silently reports zero.
    pub fn start(scope: MacmonScope, interval_ms: u64) -> Option<Self> {
        let mut child = std::process::Command::new("macmon")
            .args(["pipe", "-i", &interval_ms.to_string(), "-s", "0"])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .ok()?;
        let stdout = child.stdout.take()?;
        let state = std::sync::Arc::new(std::sync::Mutex::new(MacmonState::default()));
        let worker = std::sync::Arc::clone(&state);
        std::thread::spawn(move || {
            use std::io::BufRead;
            for line in std::io::BufReader::new(stdout).lines().map_while(Result::ok) {
                let Some(mut sample) = parse_sample(&line) else { continue };
                let w = match scope {
                    MacmonScope::Soc => sample.soc(),
                    MacmonScope::Accelerator => sample.gpu + sample.ram,
                    MacmonScope::System => sample.sys,
                };
                let now = Instant::now();
                let Ok(mut st) = worker.lock() else { return };
                // TRAPEZOID over the ACTUAL interval between samples, not the nominal one. The
                // requested `-i` is not the delivered interval — a loaded machine runs late, and
                // multiplying by the flag would count time at a power never observed. Trapezoid
                // rather than a left sum because a rectangle systematically misses the ramp on
                // every edge, and every edge biases the same direction so it never averages out.
                if let Some(prev) = st.last {
                    let dt = now.duration_since(prev).as_secs_f64();
                    let prev_w = st.all.last().map(|s| match scope {
                        MacmonScope::Soc => s.soc(),
            MacmonScope::Accelerator => s.gpu + s.ram,
                        MacmonScope::System => s.sys,
                    });
                    st.joules += match prev_w {
                        Some(p) => 0.5 * (p + w) * dt,
                        None => w * dt,
                    };
                } else {
                    st.first = Some(now);
                }
                sample.t = st.first.map(|f| now.duration_since(f).as_secs_f64()).unwrap_or(0.0);
                st.last = Some(now);
                st.watt_sum += w;
                st.all.push(sample);
            }
        });
        let me = Self {
            scope,
            state,
            child: std::sync::Arc::new(std::sync::Mutex::new(Some(child))),
        };
        // Wait for the meter to actually produce something. Checked up front so a benchmark fails
        // at the start rather than producing an hour of nothing.
        for _ in 0..40 {
            if me.samples() > 0 {
                return Some(me);
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        None
    }

    /// Samples accumulated so far. **A joules figure over one or two samples is not a measurement**,
    /// and the caller is expected to check this before reporting.
    pub fn samples(&self) -> usize {
        self.state.lock().map(|s| s.all.len()).unwrap_or(0)
    }

    /// Every sample so far, whole. Callers integrating a sub-window need these rather than the
    /// running total, because a running total cannot be windowed after the fact.
    pub fn trace(&self) -> Vec<MacmonSample> {
        self.state.lock().map(|s| s.all.clone()).unwrap_or_default()
    }

    /// Energy over `[t0, t1]` seconds since start, trapezoid with INTERPOLATED EDGES.
    ///
    /// Returns `None` unless a sample exists strictly on each side of the window — truncating to
    /// the interior samples instead loses a systematic few percent, in the same direction for every
    /// arm, so it never averages out. Callers must therefore pad each measured block with a few
    /// seconds of captured samples on both sides.
    pub fn energy_over(&self, t0: f64, t1: f64) -> Option<f64> {
        if t1 <= t0 {
            return None;
        }
        let all = self.trace();
        let w = |s: &MacmonSample| match self.scope {
            MacmonScope::Soc => s.soc(),
            MacmonScope::Accelerator => s.gpu + s.ram,
            MacmonScope::System => s.sys,
        };
        let before = all.iter().rposition(|s| s.t <= t0)?;
        let after = all.iter().position(|s| s.t >= t1)?;
        if after <= before {
            return None;
        }
        let at = |t: f64| -> f64 {
            let i = all.iter().rposition(|s| s.t <= t).unwrap_or(0);
            let j = (i + 1).min(all.len() - 1);
            if j == i || all[j].t <= all[i].t {
                return w(&all[i]);
            }
            let f = (t - all[i].t) / (all[j].t - all[i].t);
            w(&all[i]) + f * (w(&all[j]) - w(&all[i]))
        };
        let mut pts: Vec<(f64, f64)> = vec![(t0, at(t0))];
        for s in &all[before + 1..after] {
            pts.push((s.t, w(s)));
        }
        pts.push((t1, at(t1)));
        Some(pts.windows(2).map(|p| 0.5 * (p[0].1 + p[1].1) * (p[1].0 - p[0].0)).sum())
    }

    /// Mean power over every sample seen, which is what makes a figure comparable across runs of
    /// different length.
    pub fn mean_watts(&self) -> Option<f64> {
        let s = self.state.lock().ok()?;
        (!s.all.is_empty()).then(|| s.watt_sum / s.all.len() as f64)
    }

    /// Seconds spanned by the integration window — the time actually accounted for, which is one
    /// sample interval shorter than the process has been alive.
    pub fn window_secs(&self) -> f64 {
        let Ok(s) = self.state.lock() else { return 0.0 };
        match (s.first, s.last) {
            (Some(a), Some(b)) => b.duration_since(a).as_secs_f64(),
            _ => 0.0,
        }
    }

    pub fn scope(&self) -> MacmonScope {
        self.scope
    }

    /// Stop the sampler. Also runs on drop, so a forgotten meter does not outlive its program.
    pub fn stop(&self) {
        if let Ok(mut c) = self.child.lock() {
            if let Some(mut child) = c.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

impl Drop for Macmon {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Pull one sample out of a `macmon pipe` JSON line.
///
/// Parsed by field name rather than with a JSON library because this crate carries no dependencies.
/// **Each lookup includes the opening quote** — `"ram_power":` — which is what keeps it from
/// matching inside `"gpu_ram_power":`, a collision that would silently add the wrong term and that
/// nothing downstream would report.
fn parse_sample(line: &str) -> Option<MacmonSample> {
    let field = |k: &str| -> Option<f64> {
        let pat = format!("\"{k}\":");
        let i = line.find(&pat)? + pat.len();
        let rest = &line[i..];
        let end = rest.find([',', '}', ']']).unwrap_or(rest.len());
        rest[..end].trim().parse().ok()
    };
    Some(MacmonSample {
        t: 0.0,
        cpu: field("cpu_power")?,
        gpu: field("gpu_power")?,
        ane: field("ane_power").unwrap_or(0.0),
        ram: field("ram_power").unwrap_or(0.0),
        sys: field("sys_power").unwrap_or(f64::NAN),
        cpu_temp: field("cpu_temp_avg").unwrap_or(f64::NAN),
        gpu_temp: field("gpu_temp_avg").unwrap_or(f64::NAN),
        pcpu_mhz: field("pcpu_freq_mhz").unwrap_or(f64::NAN),
        ecpu_mhz: field("ecpu_freq_mhz").unwrap_or(f64::NAN),
        gpu_mhz: field("gpu_freq_mhz").unwrap_or(f64::NAN),
        cpu_active: field("cpu_active_ratio").unwrap_or(f64::NAN),
    })
}

impl Meter for Macmon {
    fn read_joules(&self) -> Option<f64> {
        let s = self.state.lock().ok()?;
        (!s.all.is_empty()).then_some(s.joules)
    }
    fn class(&self) -> Class {
        Class::Derived
    }
    fn source(&self) -> &'static str {
        match self.scope {
            MacmonScope::Soc => "macmon:soc",
            MacmonScope::Accelerator => "macmon:gpu+ram",
            MacmonScope::System => "macmon:system",
        }
    }
    fn boundary(&self) -> Boundary {
        match self.scope {
            // The SoC is accelerator and host on the same die, with its memory. No facility term on
            // a laptop, and no provisioned-idle term because there is no fleet.
            MacmonScope::Soc => Boundary { accelerator: true, host: true, idle: false, facility: false },
            MacmonScope::Accelerator => Boundary { accelerator: true, host: false, idle: false, facility: false },
            MacmonScope::System => Boundary::SYSTEM,
        }
    }
}

/// One precondition on the machine, and whether it currently holds.
#[derive(Debug, Clone)]
pub struct Gate {
    pub name: &'static str,
    pub ok: bool,
    /// What was actually read, so a failure names a number rather than a mood.
    pub detail: String,
    /// Why this gate exists — printed on failure, because a refused measurement is only useful if
    /// the reader learns what to change.
    pub because: &'static str,
}

/// Swap traffic, sampled from the kernel's cumulative page counters.
///
/// **This replaces an occupancy gate that was wrong in both directions.** The earlier gate refused
/// a measurement when `vm.swapusage` reported more than half the swap file in use. Two things are
/// wrong with that. macOS grows the swap file on demand and sizes it to what is resident, so
/// `used/total` sits near 1 on any machine that has ever swapped and near 0 on one that has not —
/// it is close to a constant, not a discriminator. And the quantity it reports is *history*: pages
/// belonging to idle applications, sitting on disk, moving nothing. A machine reading 18241M of
/// 19456M with `Swapins` and `Swapouts` frozen to the digit across five seconds is doing no paging
/// at all, and was being refused. Meanwhile a machine with a small, violently thrashing swap file
/// would have passed.
///
/// The quantity the gate is actually about is DRAM traffic that belongs to no workload, and traffic
/// is a rate. `Swapins + Swapouts` are cumulative page counts; their difference over a known
/// interval is the paging bandwidth, which is what lands in `ram_power`.
#[derive(Debug, Clone, Copy)]
pub struct Paging {
    /// Cumulative pages read back from swap since boot.
    pub swapins: u64,
    /// Cumulative pages written to swap since boot.
    pub swapouts: u64,
    /// Bytes per page, read from the `vm_stat` header rather than assumed.
    pub page_bytes: u64,
    at: Instant,
}

/// Paging traffic the gate treats as negligible, in bytes per second.
///
/// One mebibyte a second. Against a memory system moving tens of gigabytes a second this is under
/// a hundredth of a percent of DRAM traffic and cannot move `ram_power` by anything a joules figure
/// would resolve. The gate is not "no paging" because a cumulative counter can tick for reasons
/// that have nothing to do with the block; it is "paging too small to appear in the answer".
pub const PAGING_BUDGET_BPS: f64 = 1024.0 * 1024.0;

impl Paging {
    /// Sample the counters now. Fields are `NaN`-free but the sample is `None`-shaped on failure:
    /// unreadable counters produce zeros, and [`Self::gate_since`] fails the gate rather than
    /// passing it, because an unmeasurable machine is not a quiet one.
    pub fn now() -> Self {
        let out = std::process::Command::new("vm_stat")
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default();
        let count = |k: &str| -> u64 {
            out.lines()
                .find(|l| l.trim_start().starts_with(k))
                .and_then(|l| l.split(':').nth(1))
                .map(|v| v.trim().trim_end_matches('.'))
                .and_then(|v| v.parse().ok())
                .unwrap_or(0)
        };
        // "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
        let page_bytes = out
            .lines()
            .next()
            .and_then(|l| l.split("page size of").nth(1))
            .and_then(|r| r.split_whitespace().next())
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        Self { swapins: count("Swapins"), swapouts: count("Swapouts"), page_bytes, at: Instant::now() }
    }

    /// Pages moved to or from swap since `start`.
    pub fn pages_since(&self, start: &Paging) -> u64 {
        self.swapins.saturating_sub(start.swapins) + self.swapouts.saturating_sub(start.swapouts)
    }

    /// Paging bandwidth since `start`, in bytes per second.
    pub fn bytes_per_second_since(&self, start: &Paging) -> f64 {
        let secs = self.at.duration_since(start.at).as_secs_f64();
        if secs <= 0.0 { return f64::INFINITY; }
        self.pages_since(start) as f64 * self.page_bytes as f64 / secs
    }

    /// The gate, as a rate over the interval that just elapsed.
    ///
    /// Hold a [`Paging`] from before a measured block and call this after it, and the gate covers
    /// the block itself rather than a moment beside it.
    pub fn gate_since(&self, start: &Paging) -> Gate {
        let readable = self.page_bytes > 0 && start.page_bytes > 0;
        let bps = self.bytes_per_second_since(start);
        let pages = self.pages_since(start);
        let secs = self.at.duration_since(start.at).as_secs_f64();
        Gate {
            name: "paging traffic",
            ok: readable && bps < PAGING_BUDGET_BPS,
            detail: if readable {
                format!(
                    "{pages} pages in {secs:.2}s = {:.2} MiB/s, gate < {:.2} MiB/s",
                    bps / (1024.0 * 1024.0),
                    PAGING_BUDGET_BPS / (1024.0 * 1024.0)
                )
            } else {
                "vm_stat counters unreadable".to_string()
            },
            because: "paging is DRAM traffic that belongs to no workload and lands in ram_power; \
                      swap OCCUPANCY is history and does not measure it",
        }
    }
}

/// Whether this machine is in a state where a power difference means anything.
///
/// **These are not caveats, they are admission criteria.** A joules-per-workload figure is a
/// difference between two system-wide windows, and on a machine that is loaded, swapping, or
/// running off its battery because the adapter cannot keep up, that difference is dominated by
/// things that have nothing to do with the workload — routinely including a NEGATIVE marginal
/// power, which is not a small number but a meaningless one.
///
/// Checked before and after every measured block. A failure makes the block void rather than
/// caveated: [`Meter::read_joules`] returning `None` is this crate's whole position on the matter.
pub fn power_gates() -> Vec<Gate> {
    let read = |cmd: &str, args: &[&str]| -> String {
        std::process::Command::new(cmd)
            .args(args)
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default()
    };
    let mut gates = Vec::new();

    // Load average. A saturated machine schedules the workload against everything else, and the
    // difference measures the contention rather than the work.
    let la = read("sysctl", &["-n", "vm.loadavg"]);
    let load1: f64 = la
        .trim()
        .trim_start_matches('{')
        .split_whitespace()
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(f64::NAN);
    gates.push(Gate {
        name: "load average (1 min)",
        ok: load1 < 2.0,
        detail: format!("{load1:.2}, gate < 2.0"),
        because: "a saturated machine measures its own contention, not the workload",
    });

    // Paging. Paging is memory traffic that is not the workload's, and it moves DRAM power
    // directly. What matters is the TRAFFIC during the block, not how much has ever been swapped:
    // see `Paging` for why the occupancy form of this gate was wrong in both directions.
    let before = Paging::now();
    std::thread::sleep(Duration::from_millis(250));
    gates.push(Paging::now().gate_since(&before));

    // Battery current. Plugged in is not the same as mains-powered: an adapter smaller than the
    // load leaves the battery financing the difference, and that term inverts sign the moment
    // demand drops between blocks.
    let io = read("ioreg", &["-rn", "AppleSmartBattery"]);
    let amps: f64 = io
        .lines()
        .find(|l| l.contains("\"InstantAmperage\""))
        .and_then(|l| l.split('=').nth(1))
        .map(str::trim)
        .and_then(|v| v.parse::<u64>().ok())
        .map(|v| if v > u64::MAX / 2 { v as i128 - (1i128 << 64) } else { v as i128 } as f64)
        .unwrap_or(f64::NAN);
    gates.push(Gate {
        name: "battery current",
        ok: amps.abs() < 100.0,
        detail: format!("{amps:.0} mA, gate |I| < 100 mA"),
        because: "an undersized adapter makes the battery finance the run, and that term changes sign between blocks",
    });

    gates
}

/// Whether every gate holds.
pub fn machine_is_measurable() -> bool {
    power_gates().iter().all(|g| g.ok)
}

/// Nameplate arithmetic. Present so that sizing work has something to use, and classed `Estimated` so
/// it can never back a claim.
///
/// This is what most published AI energy figures actually are.
pub struct Nameplate {
    pub watts: f64,
    start: Instant,
}

impl Nameplate {
    pub fn new(watts: f64) -> Self { Self { watts, start: Instant::now() } }
}

impl Meter for Nameplate {
    fn read_joules(&self) -> Option<f64> { Some(self.watts * self.start.elapsed().as_secs_f64()) }
    fn class(&self) -> Class { Class::Estimated }
    fn source(&self) -> &'static str { "nameplate:tdp" }
    fn boundary(&self) -> Boundary { Boundary::SYSTEM }
}

/// Pick the best meter this machine can actually provide, best first.
///
/// Returns `None` when nothing real is available, rather than silently handing back a [`Nameplate`].
/// A caller that wants an estimate has to ask for one by name.
pub fn best() -> Option<Box<dyn Meter>> {
    if let Some(m) = Rapl::new() { return Some(Box::new(m)); }
    if let Some(m) = NvidiaSmi::new() { return Some(Box::new(m)); }
    if let Some(m) = MacBattery::new() { return Some(Box::new(m)); }
    None
}

/// Human-readable account of what this machine can and cannot measure, and why.
pub fn capability_report() -> String {
    let mut out = String::from("energy measurement on this machine:\n");
    let rows: [(&str, bool, &str); 3] = [
        ("rapl:package", Rapl::new().is_some(), "Linux only, CPU package, excludes discrete GPU"),
        ("nvidia-smi:board", NvidiaSmi::new().is_some(), "GPU board only, sampled at ~10 ms so task-level not kernel-level"),
        ("macos:battery-discharge", MacBattery::new().is_some(), "whole system, but ONLY while on battery"),
    ];
    for (name, ok, note) in rows {
        out.push_str(&format!("  [{}] {:<26} {}\n", if ok { "available" } else { "  --     " }, name, note));
    }
    if best().is_none() {
        out.push_str("\n  Nothing measurable here. `Nameplate` exists for sizing, is classed Estimated,\n");
        out.push_str("  and Saving::claimable() will refuse to let it back a claim. That refusal is the point.\n");
    }
    out
}

/// A stopwatch that reports joules-per-task alongside wall time, for the common case of one arm.
///
/// Deliberately cannot produce a [`Saving`]. If you want to claim an improvement you have to run both
/// arms, and that is not an oversight.
pub struct Session<'a, M: Meter> {
    meter: &'a M,
    start_j: Option<f64>,
    start_t: Instant,
    tasks: u64,
}

impl<'a, M: Meter> Session<'a, M> {
    pub fn start(meter: &'a M) -> Self {
        Self { meter, start_j: meter.read_joules(), start_t: Instant::now(), tasks: 0 }
    }
    pub fn task(&mut self) { self.tasks += 1; }
    pub fn tasks(&mut self, n: u64) { self.tasks += n; }
    pub fn finish(self) -> Option<(Reading, u64)> {
        let j1 = self.meter.read_joules()?;
        let j0 = self.start_j?;
        if j1 < j0 { return None; }
        Some((Reading {
            joules: j1 - j0,
            seconds: self.start_t.elapsed().as_secs_f64(),
            class: self.meter.class(),
            source: self.meter.source(),
            boundary: self.meter.boundary(),
        }, self.tasks))
    }
}

/// Wall-clock duration helper for callers that want a quick shape check without a meter.
pub fn time_it<T>(f: impl FnOnce() -> T) -> (T, Duration) {
    let t = Instant::now();
    let out = f();
    (out, t.elapsed())
}

/// **A median that refuses when its samples are too unstable to be a measurement.**
///
/// ⛔ This exists because I wrote the same guard three times and forgot it twice. `FabricProfile`
/// has one for bandwidth samples; the mirror-striping measurement did not, published a direction,
/// and two runs minutes apart disagreed on the SIGN; the readback-cost gate did not, and produced a
/// 265x spread on a loaded machine. **A guard protects the code path it is wired into, never the
/// class of mistake** — so the guard has to be the easy thing to reach for, not a thing to remember.
///
/// `max_spread` is max/min, not a standard deviation: timing distributions on a contended machine
/// are not normal, and one 200x outlier is exactly the event that matters. A ratio sees it; a
/// standard deviation buries it under n.
///
/// ⚠ Refuses on fewer than 3 samples too. A spread cannot be judged from two.
pub fn stable_median(samples: &[f64], max_spread: f64, what: &str) -> Result<f64, String> {
    if samples.len() < 3 {
        return Err(format!("{what}: {} sample(s) — a spread cannot be judged from fewer than 3",
                           samples.len()));
    }
    // ⛔ EVERY sample, checked BEFORE sorting. Checking only the extremes afterwards is what the
    // first version did, and it let `[1.0, NaN, 1.0]` through: `partial_cmp` has no ordering for
    // NaN, so `unwrap_or(Equal)` leaves it wherever it started — the MIDDLE — where `v[0]` and
    // `v[len-1]` are both 1.0, the spread reads 1.00, and the returned median is NaN. A guard
    // against bad measurements that emits NaN is worse than no guard, because it emits it with
    // authority. Its own test caught this.
    if let Some(bad) = samples.iter().find(|x| !x.is_finite()) {
        return Err(format!("{what}: a sample is {bad}, which is not finite. NaN has no ordering, so \
                            it sorts arbitrarily and hides in the middle where checking the extremes \
                            cannot see it"));
    }
    let mut v: Vec<f64> = samples.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("all samples were checked finite above"));
    let (lo, hi) = (v[0], v[v.len() - 1]);
    if !(lo > 0.0) {
        return Err(format!("{what}: minimum sample is {lo}, which is not a rate or a duration"));
    }
    let spread = hi / lo;
    if spread > max_spread {
        return Err(format!(
            "{what}: spread {spread:.2}x (min {lo:.4}, max {hi:.4}) exceeds the {max_spread:.2}x \
             this figure may be reported from. The median of samples that unstable is whatever else \
             the machine was doing — re-measure when it is idle rather than widening the tolerance"));
    }
    Ok(v[v.len() / 2])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake counter, so the crate's own logic is testable on a machine with no sensors.
    struct Fake { j: std::cell::Cell<f64>, step: f64, class: Class }
    impl Fake {
        fn new(step: f64, class: Class) -> Self { Self { j: std::cell::Cell::new(0.0), step, class } }
    }
    impl Meter for Fake {
        fn read_joules(&self) -> Option<f64> {
            let v = self.j.get() + self.step;
            self.j.set(v);
            Some(v)
        }
        fn class(&self) -> Class { self.class }
        fn source(&self) -> &'static str { "fake" }
        fn boundary(&self) -> Boundary { Boundary::DEVICE }
    }

    struct Dead;
    impl Meter for Dead {
        fn read_joules(&self) -> Option<f64> { None }
        fn class(&self) -> Class { Class::Measured }
        fn source(&self) -> &'static str { "dead" }
        fn boundary(&self) -> Boundary { Boundary::DEVICE }
    }

    #[test]
    fn an_unavailable_meter_yields_no_reading_rather_than_zero() {
        // THE failure this crate exists to prevent. A zero is indistinguishable from a perfectly
        // efficient run, so a broken sensor must not be able to look like a win.
        let (_, r) = measure(&Dead, || std::hint::black_box(1 + 1));
        assert!(r.is_none(), "a dead meter produced a reading");
        assert!(compare(&Dead, 1, (1, 1), 1, || {}, || {}).is_none(), "a dead meter produced a saving");
    }

    #[test]
    fn a_saving_cannot_be_built_without_running_both_arms() {
        // Enforced by there being no constructor. This test documents the intent so that adding one
        // later is a visible decision rather than a convenience someone slipped in.
        let m = Fake::new(1.0, Class::Measured);
        let s = compare(&m, 10, (10, 9), 1, || {}, || {}).expect("meter is available");
        assert_eq!(s.tasks, 10);
        assert_eq!(s.successes, (10, 9));
        // The only public path to Saving is compare(); the struct's fields are readable but there is no
        // way to fabricate the readings without a Meter having produced them.
        let _ = s.baseline;
    }

    #[test]
    fn an_estimate_may_not_back_a_claim() {
        let est = Nameplate::new(50.0);
        std::thread::sleep(Duration::from_millis(1100));
        let s = compare(&est, 100, (100, 100), 1, || std::thread::sleep(Duration::from_millis(1100)),
                                       || std::thread::sleep(Duration::from_millis(1100)))
            .expect("nameplate is always available");
        assert_eq!(s.class(), Class::Estimated);
        assert_eq!(s.claimable(), Err("at least one arm is an estimate, not a measurement"));
    }

    #[test]
    fn a_sub_second_arm_is_refused_because_it_is_inside_sensor_noise() {
        let m = Fake::new(1.0, Class::Measured);
        let s = compare(&m, 5, (5, 5), 1, || {}, || {}).unwrap();
        assert!(s.claimable().is_err(), "a run of microseconds was accepted as claimable");
    }

    #[test]
    fn a_candidate_that_uses_more_energy_reports_a_negative_saving() {
        // Regressions must be reportable. A framework that can only express wins is a marketing tool.
        struct Ramp { n: std::cell::Cell<u32> }
        impl Meter for Ramp {
            fn read_joules(&self) -> Option<f64> {
                let i = self.n.get(); self.n.set(i + 1);
                // 0, 10 (baseline = 10 J), then 10, 40 (candidate = 30 J)
                Some(match i { 0 => 0.0, 1 => 10.0, 2 => 10.0, _ => 40.0 })
            }
            fn class(&self) -> Class { Class::Measured }
            fn source(&self) -> &'static str { "ramp" }
            fn boundary(&self) -> Boundary { Boundary::DEVICE }
        }
        // `available()` consumes a sample, so start the ramp accounting after it.
        let r = Ramp { n: std::cell::Cell::new(0) };
        let (_, b) = measure(&r, || {});
        let (_, c) = measure(&r, || {});
        let s = Saving { baseline: b.unwrap(), candidate: c.unwrap(), tasks: 1, successes: (1, 1) };
        assert!(s.fraction() < 0.0, "a worse candidate was not reported as negative");
        assert!((s.percent() + 200.0).abs() < 1e-6, "expected -200%, got {}", s.percent());
    }

    #[test]
    fn the_weaker_class_wins_a_comparison() {
        let a = Reading { joules: 10.0, seconds: 2.0, class: Class::Measured, source: "x", boundary: Boundary::DEVICE };
        let b = Reading { joules: 5.0, seconds: 2.0, class: Class::Estimated, source: "x", boundary: Boundary::DEVICE };
        let s = Saving { baseline: a, candidate: b, tasks: 1, successes: (1, 1) };
        assert_eq!(s.class(), Class::Estimated, "a comparison claimed to be stronger than its weaker arm");
    }

    #[test]
    fn mismatched_meters_are_not_comparable() {
        let a = Reading { joules: 10.0, seconds: 2.0, class: Class::Measured, source: "rapl:package", boundary: Boundary::DEVICE };
        let b = Reading { joules: 5.0, seconds: 2.0, class: Class::Measured, source: "nvidia-smi:board", boundary: Boundary::DEVICE };
        let s = Saving { baseline: a, candidate: b, tasks: 1, successes: (1, 1) };
        assert!(s.claimable().is_err(), "energy from two different sensors was accepted as a difference");
    }

    #[test]
    fn joules_per_task_is_the_reported_metric() {
        let r = Reading { joules: 100.0, seconds: 10.0, class: Class::Measured, source: "x", boundary: Boundary::DEVICE };
        assert_eq!(r.per_task(50), 2.0);
        assert_eq!(r.watts(), 10.0);
        assert!(r.per_task(0).is_nan(), "zero tasks must not silently divide");
    }

    #[test]
    fn capability_report_names_what_is_missing() {
        let s = capability_report();
        assert!(s.contains("rapl") && s.contains("battery"), "report omits a backend");
    }
}

#[cfg(test)]
mod boundary_tests {
    use super::*;

    struct M(Boundary);
    impl Meter for M {
        fn read_joules(&self) -> Option<f64> { Some(0.0) }
        fn class(&self) -> Class { Class::Measured }
        fn source(&self) -> &'static str { "m" }
        fn boundary(&self) -> Boundary { self.0 }
    }

    #[test]
    fn arms_that_enclose_different_things_are_not_comparable() {
        // The failure this exists for: Google measured the SAME prompt on the SAME fleet at 0.10 Wh
        // accelerator-only and 0.24 Wh counting host, idle and facility. 2.4x from accounting alone.
        // Comparing across that boundary attributes an accounting choice to a method.
        let dev = Reading { joules: 100.0, seconds: 2.0, class: Class::Measured, source: "m", boundary: Boundary::DEVICE };
        let sys = Reading { joules: 42.0, seconds: 2.0, class: Class::Measured, source: "m", boundary: Boundary::SYSTEM };
        let s = Saving { baseline: sys, candidate: dev, tasks: 10, successes: (10, 10) };
        assert!(s.claimable().is_err(), "a cross-boundary comparison was accepted");
        assert!(s.claimable().unwrap_err().contains("enclose different things"));
    }

    #[test]
    fn zero_successes_is_never_an_efficiency_result() {
        // On GAIA the model burning 7.31 kJ/query scored 16.4% and the one burning 1.18 kJ scored 5.5%.
        // Per query the cheap one wins; per SUCCESS it is 21.5 kJ against 44.6 kJ and the ranking holds
        // only because both succeeded sometimes. At zero successes there is no efficiency at any energy.
        let r = Reading { joules: 10.0, seconds: 2.0, class: Class::Measured, source: "m", boundary: Boundary::DEVICE };
        let s = Saving { baseline: r, candidate: r, tasks: 100, successes: (100, 0) };
        assert!(s.claimable().unwrap_err().contains("no successes"));
    }

    #[test]
    fn energy_per_success_can_reverse_the_ranking() {
        // The previous version of this test was VACUOUS, and it is worth saying how, because the
        // mechanism is new: it compared `s.per_success().1 < t.per_success().1` across two DIFFERENT
        // `Saving`s that shared a Reading, which reduces to 1180/164 < 1180/55 — the monotonicity of
        // division, true for every input, including inputs where the ranking does not reverse at all.
        // The name claimed a property BETWEEN THE TWO ARMS; the assertion tested arithmetic inside one
        // accessor. It could not have failed.
        let r = |j: f64| Reading { joules: j, seconds: 2.0, class: Class::Measured, source: "m", boundary: Boundary::DEVICE };
        let (dear, cheap) = (r(7310.0), r(1180.0));

        // 1. The real GAIA figures NARROW the gap; they do not reverse it. This is the case the field
        //    quotes, and the honest reading of it is 6.19x -> 2.08x, still in the cheap arm's favour.
        let gaia = Saving { baseline: dear, candidate: cheap, tasks: 1000, successes: (164, 55) };
        let (b, c) = gaia.per_attempt();
        assert!((b / c - 6.19).abs() < 0.01, "per attempt the dear arm costs 6.19x, got {:.3}x", b / c);
        let (b, c) = gaia.per_success();
        assert!((b / c - 2.08).abs() < 0.01, "per success that falls to 2.08x, got {:.3}x", b / c);
        assert!(gaia.percent() > 0.0 && gaia.percent_per_success() > 0.0,
                "GAIA narrows the gap without crossing zero, so both savings stay positive");

        // 2. A reversal needs a wider success gap, and then BOTH SIGNS FLIP — which is the property
        //    the old assertion could not express, because a shared denominator cancels out of every
        //    ratio this type computes.
        let rev = Saving { baseline: dear, candidate: cheap, tasks: 1000, successes: (500, 55) };
        assert!(rev.percent() > 0.0, "the cheap arm still burns less total energy: {:.1}%", rev.percent());
        assert!(rev.percent_per_success() < 0.0,
                "and yet it costs MORE per completed task; that reversal is the whole point of the \
                 field, and it must show up as a sign disagreement: {:.1}% vs {:.1}%",
                rev.percent(), rev.percent_per_success());

        // 3. THE REGRESSION GUARD for the defect this replaced: with the denominators shared, no
        //    reversal is representable, because per_success and per_attempt differ by a constant
        //    factor that divides out. Equal success counts must therefore agree in sign with
        //    per_attempt for every possible pair of readings.
        let equal = Saving { baseline: dear, candidate: cheap, tasks: 1000, successes: (164, 164) };
        assert!((equal.percent() - equal.percent_per_success()).abs() < 1e-9,
                "with equal successes the two savings are the SAME number — if they ever differ, the \
                 per-arm denominators stopped being applied per arm");
    }

    #[test]
    fn compare_tasks_counts_successes_instead_of_believing_them() {
        // The point of this path: the success counts are OBSERVED from the closures, so a caller
        // cannot report a rate no arm demonstrated. Here the baseline solves the even tasks and the
        // candidate solves the first three — chosen so the two counts differ AND neither equals the
        // task count, which is what makes a wrong denominator visible.
        let m = Nameplate::new(10.0);
        let tasks: Vec<u32> = (0..10).collect();
        let (s, bok, cok) = compare_tasks(&m, &tasks, 1,
            |t| t % 2 == 0,
            |t| *t < 3,
        ).expect("nameplate is always available");
        assert_eq!(s.successes, (5, 3), "counted from the closures, not from an argument");
        assert_eq!(bok.iter().filter(|x| **x).count(), 5);
        assert_eq!(cok.iter().filter(|x| **x).count(), 3);
        assert_eq!(s.tasks, 10);
        // And the graded outcomes come back per task, so a caller can show WHICH tasks each arm lost
        // rather than only how many. A bench that reports 5/10 without saying which five cannot be
        // audited by anyone, including its author.
        assert!(bok[0] && !bok[1], "per-task outcomes must survive, not just the tally");
        assert!(cok[2] && !cok[3]);
    }

    #[test]
    fn compare_tasks_refuses_an_empty_task_set() {
        // Zero tasks is not a comparison of zero cost; it is an absent comparison. Returning a Saving
        // with tasks: 0 would put a NaN into every downstream figure and let `claimable` be the only
        // thing standing between that and a published number.
        let m = Nameplate::new(10.0);
        let empty: [u32; 0] = [];
        assert!(compare_tasks(&m, &empty, 1, |_| true, |_| true).is_none());
    }

    #[test]
    fn grade_tasks_measures_the_denominator_when_no_meter_exists() {
        // The case this machine is actually in: on AC power there is no readable sensor, and the
        // success halves of the comparison are still measurable. This path returns no Saving BY
        // CONSTRUCTION — there is no meter, so there is no energy claim to be made — while still
        // producing the per-task outcomes that any later energy figure would be divided by.
        let tasks: Vec<u32> = (0..8).collect();
        let (bok, cok, (bs, cs)) = grade_tasks(&tasks, |t| t % 4 == 0, |t| *t < 6);
        assert_eq!((bok.iter().filter(|x| **x).count(), cok.iter().filter(|x| **x).count()), (2, 6));
        assert!(bs >= 0.0 && cs >= 0.0, "wall-clock is reported even when joules are not");
    }

    #[test]
    fn every_meter_declares_a_boundary() {
        assert_eq!(M(Boundary::DEVICE).boundary().label(), "accel");
        assert_eq!(M(Boundary::SYSTEM).boundary().label(), "accel+host+idle");
        assert_eq!(Nameplate::new(1.0).boundary(), Boundary::SYSTEM);
    }

    /// The guard has to FIRE on real instability and NOT fire on ordinary jitter, or it is either
    /// decorative or useless. Both directions, with the boundary pinned from each side.
    ///
    /// ⛔ Written first with the arithmetic wrong: `[1.00, 1.02, 0.99, 1.01, 1.00]` spreads
    /// 1.02/0.99 = **1.030**, and the test asserted it should PASS at a 1.02 tolerance. The guard
    /// refused, correctly, and the test was the thing that was broken. Spread is a RATIO of the
    /// extremes — eyeballing the values is how you get this wrong.
    #[test]
    fn stable_median_refuses_exactly_the_samples_that_fooled_me() {
        let steady = [1.00, 1.02, 0.99, 1.01, 1.00]; // spread 1.030
        assert_eq!(stable_median(&steady, 1.5, "x").unwrap(), 1.00);
        // The boundary, from both sides — a guard that only ever fires is not a guard.
        assert!(stable_median(&steady, 1.05, "x").is_ok(), "1.030 refused at a 1.05 tolerance");
        assert!(stable_median(&steady, 1.02, "x").is_err(), "1.030 accepted at a 1.02 tolerance");

        // The real sample sets from this session, each with its ACTUAL spread computed rather than
        // assumed, and a tolerance chosen to sit below it.
        for (name, s, tol) in [
            // ⚠ Striping's failure was BETWEEN runs (2.46 vs 7.58 GB/s minutes apart), which a
            // within-run guard cannot see. These are its within-run samples, spread 3.08 — included
            // to keep that distinction visible rather than to claim the guard would have caught it.
            ("striping", vec![2.46, 7.58, 3.1, 6.9, 2.5], 2.0),
            ("pcie", vec![0.1585, 1.224, 0.4, 0.9, 0.15], 4.0),        // 8.16
            ("readback-warm", vec![0.485, 128.7, 0.5, 0.49, 0.6], 4.0), // 265.4
        ] {
            let e = stable_median(&s, tol, name).unwrap_err();
            assert!(e.contains(name) && e.contains("spread"), "unhelpful refusal for {name}: {e}");
        }
        assert!(stable_median(&[1.0, 1.0], 1.5, "x").is_err(), "two samples cannot show a spread");
        assert!(stable_median(&[0.0, 1.0, 1.0], 1.5, "x").is_err(), "a zero is not a rate");
        // ⛔ THIS ONE FOUND A REAL BUG. NaN in the MIDDLE survived a check on the extremes: it
        // sorts arbitrarily, both ends read 1.0, spread reads 1.00, and the median came back NaN.
        // Every position, because the position is the whole point.
        for n in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for pos in 0..3 {
                let mut v = [1.0, 1.0, 1.0];
                v[pos] = n;
                let r = stable_median(&v, 1.5, "x");
                assert!(r.is_err(), "{n} at position {pos} passed the guard: {r:?}");
            }
        }
    }


    // The paging gate is a RATE gate. These pin the two directions the occupancy form got wrong.

    #[test]
    fn paging_gate_passes_a_quiet_machine_however_much_is_already_swapped() {
        // The counters are cumulative, so swap that happened before `start` is invisible to the
        // delta by construction. A machine with 18 GiB resident in swap and no traffic passes.
        let start = Paging::now();
        std::thread::sleep(Duration::from_millis(120));
        let end = Paging::now();
        if end.page_bytes == 0 {
            return; // no vm_stat on this platform; the gate fails closed, tested below.
        }
        let quiet = Paging { swapins: start.swapins, swapouts: start.swapouts, ..end };
        assert!(quiet.gate_since(&start).ok, "zero delta must pass: {:?}", quiet.gate_since(&start).detail);
    }

    #[test]
    fn paging_gate_fails_a_thrashing_machine_however_little_is_swapped() {
        let start = Paging::now();
        std::thread::sleep(Duration::from_millis(120));
        let mut end = Paging::now();
        if end.page_bytes == 0 { return; }
        // 4096 pages in ~0.12 s is ~64 pages/ms: far over budget at any page size we would see.
        end.swapouts = start.swapouts + 4096;
        let g = end.gate_since(&start);
        assert!(!g.ok, "heavy paging must fail: {}", g.detail);
        assert!(g.detail.contains("MiB/s"), "the gate reports a rate, not an occupancy: {}", g.detail);
    }

    #[test]
    fn paging_gate_fails_closed_when_the_counters_are_unreadable() {
        let now = Instant::now();
        let a = Paging { swapins: 0, swapouts: 0, page_bytes: 0, at: now };
        let b = Paging { swapins: 0, swapouts: 0, page_bytes: 0, at: now + Duration::from_secs(1) };
        assert!(!b.gate_since(&a).ok, "an unmeasurable machine is not a quiet one");
    }

    #[test]
    fn paging_gate_fails_a_zero_width_interval() {
        // No elapsed time is no evidence of quiet. Rate is infinite, not zero.
        let now = Instant::now();
        let a = Paging { swapins: 0, swapouts: 0, page_bytes: 16384, at: now };
        let b = Paging { swapins: 0, swapouts: 0, page_bytes: 16384, at: now };
        assert!(!b.gate_since(&a).ok, "a zero-width window cannot certify anything");
    }

    #[test]
    fn paging_counters_survive_a_reset() {
        let now = Instant::now();
        let a = Paging { swapins: 900, swapouts: 900, page_bytes: 16384, at: now };
        let b = Paging { swapins: 1, swapouts: 1, page_bytes: 16384, at: now + Duration::from_secs(1) };
        assert_eq!(b.pages_since(&a), 0, "a counter that went backwards is not negative traffic");
    }

    #[test]
    fn power_gates_no_longer_name_swap_occupancy() {
        // The gate list is the crate's published admission criteria; this pins the replacement so a
        // revert cannot happen quietly.
        let names: Vec<&str> = power_gates().iter().map(|g| g.name).collect();
        assert!(names.contains(&"paging traffic"), "{names:?}");
        assert!(!names.contains(&"swap in use"), "occupancy gate is retracted: {names:?}");
    }
}
