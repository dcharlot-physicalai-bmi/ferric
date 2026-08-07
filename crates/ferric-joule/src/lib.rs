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
//! 2. **A [`Saving`] cannot be constructed by hand.** The only way to obtain one is [`compare`], which
//!    takes two closures, runs both, and keeps both readings. There is no `Saving::new`. If you have a
//!    saving, you have the baseline, because the type system would not let you have it otherwise.
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

pub mod ladder;
pub use ladder::{Ladder, Routed, Trail};

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
#[derive(Debug, Clone, Copy)]
pub struct Saving {
    pub baseline: Reading,
    pub candidate: Reading,
    /// Tasks ATTEMPTED by each arm.
    pub tasks: u64,
    /// Tasks that SUCCEEDED. This is the denominator that matters, and it is separate from `tasks`
    /// because unattended automation pays full price for its failures.
    ///
    /// Measured: in one agentic run, 2,256 J of 3,614 J (62.4%) went to a failed attempt before the
    /// successful retry (arXiv:2605.22883). On GAIA, the model burning 7.31 kJ per query scored 16.4%
    /// and the one burning 1.18 kJ scored 5.5% (arXiv:2511.07885) — per *successful* goal that is
    /// 44.6 kJ against 21.5 kJ, which reverses the ranking energy-per-query gives you.
    pub successes: u64,
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

    /// Joules per SUCCESSFUL task, both arms. The metric that actually matters.
    ///
    /// Energy per query flatters anything that fails cheaply; energy per token flatters anything
    /// terse. Neither is a unit of useful work.
    pub fn per_success(&self) -> (f64, f64) {
        (self.baseline.per_task(self.successes), self.candidate.per_task(self.successes))
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
        if self.successes == 0 {
            return Err("no successes recorded: energy per successful task is the unit, and zero successes at any energy is not an efficiency result");
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
        writeln!(f, "  baseline   {}", self.baseline)?;
        writeln!(f, "  candidate  {}", self.candidate)?;
        writeln!(f, "  per success {b:.4} J -> {c:.4} J  ({}/{} succeeded)", self.successes, self.tasks)?;
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
    successes: u64,
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
        assert!(compare(&Dead, 1, 1, 1, || {}, || {}).is_none(), "a dead meter produced a saving");
    }

    #[test]
    fn a_saving_cannot_be_built_without_running_both_arms() {
        // Enforced by there being no constructor. This test documents the intent so that adding one
        // later is a visible decision rather than a convenience someone slipped in.
        let m = Fake::new(1.0, Class::Measured);
        let s = compare(&m, 10, 10, 1, || {}, || {}).expect("meter is available");
        assert_eq!(s.tasks, 10);
        assert_eq!(s.successes, 10);
        // The only public path to Saving is compare(); the struct's fields are readable but there is no
        // way to fabricate the readings without a Meter having produced them.
        let _ = s.baseline;
    }

    #[test]
    fn an_estimate_may_not_back_a_claim() {
        let est = Nameplate::new(50.0);
        std::thread::sleep(Duration::from_millis(1100));
        let s = compare(&est, 100, 100, 1, || std::thread::sleep(Duration::from_millis(1100)),
                                       || std::thread::sleep(Duration::from_millis(1100)))
            .expect("nameplate is always available");
        assert_eq!(s.class(), Class::Estimated);
        assert_eq!(s.claimable(), Err("at least one arm is an estimate, not a measurement"));
    }

    #[test]
    fn a_sub_second_arm_is_refused_because_it_is_inside_sensor_noise() {
        let m = Fake::new(1.0, Class::Measured);
        let s = compare(&m, 5, 5, 1, || {}, || {}).unwrap();
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
        let s = Saving { baseline: b.unwrap(), candidate: c.unwrap(), tasks: 1, successes: 1 };
        assert!(s.fraction() < 0.0, "a worse candidate was not reported as negative");
        assert!((s.percent() + 200.0).abs() < 1e-6, "expected -200%, got {}", s.percent());
    }

    #[test]
    fn the_weaker_class_wins_a_comparison() {
        let a = Reading { joules: 10.0, seconds: 2.0, class: Class::Measured, source: "x", boundary: Boundary::DEVICE };
        let b = Reading { joules: 5.0, seconds: 2.0, class: Class::Estimated, source: "x", boundary: Boundary::DEVICE };
        let s = Saving { baseline: a, candidate: b, tasks: 1, successes: 1 };
        assert_eq!(s.class(), Class::Estimated, "a comparison claimed to be stronger than its weaker arm");
    }

    #[test]
    fn mismatched_meters_are_not_comparable() {
        let a = Reading { joules: 10.0, seconds: 2.0, class: Class::Measured, source: "rapl:package", boundary: Boundary::DEVICE };
        let b = Reading { joules: 5.0, seconds: 2.0, class: Class::Measured, source: "nvidia-smi:board", boundary: Boundary::DEVICE };
        let s = Saving { baseline: a, candidate: b, tasks: 1, successes: 1 };
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
        let s = Saving { baseline: sys, candidate: dev, tasks: 10, successes: 10 };
        assert!(s.claimable().is_err(), "a cross-boundary comparison was accepted");
        assert!(s.claimable().unwrap_err().contains("enclose different things"));
    }

    #[test]
    fn zero_successes_is_never_an_efficiency_result() {
        // On GAIA the model burning 7.31 kJ/query scored 16.4% and the one burning 1.18 kJ scored 5.5%.
        // Per query the cheap one wins; per SUCCESS it is 21.5 kJ against 44.6 kJ and the ranking holds
        // only because both succeeded sometimes. At zero successes there is no efficiency at any energy.
        let r = Reading { joules: 10.0, seconds: 2.0, class: Class::Measured, source: "m", boundary: Boundary::DEVICE };
        let s = Saving { baseline: r, candidate: r, tasks: 100, successes: 0 };
        assert!(s.claimable().unwrap_err().contains("no successes"));
    }

    #[test]
    fn energy_per_success_can_reverse_energy_per_attempt() {
        // The whole reason `successes` is a separate field.
        let cheap = Reading { joules: 1180.0, seconds: 2.0, class: Class::Measured, source: "m", boundary: Boundary::DEVICE };
        let dear  = Reading { joules: 7310.0, seconds: 2.0, class: Class::Measured, source: "m", boundary: Boundary::DEVICE };
        // 1000 attempts each; the expensive arm succeeds 3x as often.
        let s = Saving { baseline: dear, candidate: cheap, tasks: 1000, successes: 164 };
        let t = Saving { baseline: dear, candidate: cheap, tasks: 1000, successes: 55 };
        // Per attempt the cheap arm always looks better; per success depends on which denominator is real.
        let (_, cheap_per_attempt) = s.per_attempt();
        assert!(cheap_per_attempt < 7.31, "per-attempt arithmetic broke");
        assert!(s.per_success().1 < t.per_success().1, "more successes must lower joules per success");
    }

    #[test]
    fn every_meter_declares_a_boundary() {
        assert_eq!(M(Boundary::DEVICE).boundary().label(), "accel");
        assert_eq!(M(Boundary::SYSTEM).boundary().label(), "accel+host+idle");
        assert_eq!(Nameplate::new(1.0).boundary(), Boundary::SYSTEM);
    }
}
