//! Quiet-host energy driver.
//!
//! An exclusive `flock` on `/tmp/bmi-energy.lock` is held, without a gap, from
//! preflight through the post-trial idle bracket. If the lock is already taken
//! the attempt writes a refusal and retries later. It does not wait while
//! counters are running, and a refusal never carries a joule figure.
//!
//! RAPL package, RAPL core, and amdgpu PPT stay separate meters.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use fs4::fs_std::FileExt;

use crate::meter::delta_uj;

pub const LOCK_PATH: &str = "/tmp/bmi-energy.lock";
pub const SETTLE_S: f64 = 60.0;
pub const LOAD1_MAX: f64 = 2.0;
/// Framework package/core counter modulus, from `max_energy_range_uj`.
pub const FRAMEWORK_RANGE_UJ: u64 = 65_532_610_987;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grade {
    Quiet,
    Refused,
    MeasuredAttributable,
}

impl Grade {
    pub fn as_str(self) -> &'static str {
        match self {
            Grade::Quiet => "quiet",
            Grade::Refused => "refused",
            Grade::MeasuredAttributable => "measured_attributable",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CpuTicks {
    pub cpu: usize,
    pub busy: u64,
    pub total: u64,
}

#[derive(Debug, Clone)]
pub struct EnergySeries {
    pub load1_pre: Vec<f64>,
    pub load1_post: Vec<f64>,
    /// `(t_ns, energy_uj)` inside the pre-trial idle bracket.
    pub pre_package: Vec<(u64, u64)>,
    pub post_package: Vec<(u64, u64)>,
    pub pre_core: Vec<(u64, u64)>,
    pub post_core: Vec<(u64, u64)>,
    /// Instantaneous watts, one sample per interval, not an energy counter.
    pub pre_ppt_w: Vec<f64>,
    pub post_ppt_w: Vec<f64>,
    pub trial_package: (u64, u64),
    pub trial_core: (u64, u64),
    pub trial_ppt_j: Option<f64>,
    pub cpu_before: Vec<CpuTicks>,
    pub cpu_after: Vec<CpuTicks>,
    pub pinned: Vec<usize>,
    /// Fixed in the receipt before the workload starts.
    pub busy_fraction_max: f64,
    pub load1_max: f64,
    pub range_uj: u64,
    pub cpu_tctl_c: Option<f64>,
    pub gpu_edge_c: Option<f64>,
    /// Busy fraction of non-pinned cores on each settle interval.
    pub settle_busy: Vec<f64>,
    /// Busy fraction of non-pinned cores on each workload sample.
    pub workload_busy: Vec<f64>,
    pub tctl_pre: Option<f64>,
    pub tctl_workload: Option<f64>,
    pub tctl_post: Option<f64>,
    pub edge_pre: Option<f64>,
    pub edge_workload: Option<f64>,
    pub edge_post: Option<f64>,
    pub workload_reached: bool,
    pub post_reached: bool,
    pub pre_reached: bool,
    pub settle_load1_pre: Vec<f64>,
    pub settle_load1_post: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct LockHold {
    pub path: String,
    pub pid: u32,
    pub acquire_unix_ns: Option<u64>,
    pub release_unix_ns: Option<u64>,
    pub first_attempt_contended: bool,
    pub contended: bool,
    pub wait_s: f64,
}

#[derive(Debug, Clone)]
pub struct DriveReceipt {
    pub attributable: bool,
    pub reason: String,
    pub grades: Grades,
    pub lock: LockHold,
    pub busy_fraction_max: f64,
    pub busy_fraction: Option<f64>,
    pub cpu_tctl_c: Option<f64>,
    pub gpu_edge_c: Option<f64>,
    pub rapl_package_j: Option<f64>,
    pub rapl_core_j: Option<f64>,
    pub amdgpu_ppt_j: Option<f64>,
    /// Samples collected up to the point the attempt stopped.
    pub series: EnergySeries,
}

#[derive(Debug, Clone)]
pub struct Grades {
    pub pre_idle: Grade,
    pub workload: Grade,
    pub post_idle: Grade,
    pub rapl_package: Grade,
    pub rapl_core: Grade,
    pub amdgpu_ppt: Grade,
}

#[derive(Debug, Clone, Copy)]
pub struct BusyFraction {
    pub fraction: f64,
}

pub fn busy_fraction(before: &[CpuTicks], after: &[CpuTicks], pinned: &[usize]) -> Result<BusyFraction, String> {
    let mut busy = 0u64;
    let mut total = 0u64;
    for end in after {
        if pinned.contains(&end.cpu) {
            continue;
        }
        let Some(start) = before.iter().find(|s| s.cpu == end.cpu) else {
            return Err(format!("cpu {} missing from the opening sample", end.cpu));
        };
        if end.total < start.total || end.busy < start.busy {
            return Err(format!("cpu {} tick counter went backwards", end.cpu));
        }
        busy += end.busy - start.busy;
        total += end.total - start.total;
    }
    if total == 0 {
        return Err("non-pinned cores recorded no tick delta".into());
    }
    Ok(BusyFraction { fraction: busy as f64 / total as f64 })
}

pub fn powers_from_counter(samples: &[(u64, u64)], range_uj: u64) -> Result<Vec<f64>, String> {
    let mut out = Vec::new();
    for w in samples.windows(2) {
        let dt = w[1].0.saturating_sub(w[0].0);
        if dt == 0 {
            return Err("energy samples share a timestamp".into());
        }
        let (du, _) = delta_uj(w[0].1, w[1].1, range_uj)?;
        out.push(du as f64 * 1e-6 / (dt as f64 * 1e-9));
    }
    Ok(out)
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn sample_var(v: &[f64]) -> Result<f64, String> {
    if v.len() < 2 {
        return Err("baseline series has fewer than two intervals".into());
    }
    let m = mean(v);
    Ok(v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64)
}

/// `|mean_before - mean_after| > 3 σ` refuses. `σ` is the pooled sample
/// standard deviation of the two baseline power series.
pub fn drift_within_3_sigma(before: &[f64], after: &[f64]) -> Result<(), String> {
    let v1 = sample_var(before)?;
    let v2 = sample_var(after)?;
    let n1 = before.len();
    let n2 = after.len();
    let denom = n1 + n2 - 2;
    if denom == 0 {
        return Err("baseline series has fewer than two intervals".into());
    }
    let pooled = ((n1 - 1) as f64 * v1 + (n2 - 1) as f64 * v2) / denom as f64;
    let sigma = pooled.sqrt();
    let delta = (mean(before) - mean(after)).abs();
    if delta > 3.0 * sigma {
        return Err(format!("baseline drift {delta} W exceeds 3σ ({sigma} W)"));
    }
    Ok(())
}

pub fn load_is_quiet(samples: &[f64], load1_max: f64) -> Result<(), String> {
    if samples.is_empty() {
        return Err("idle bracket has no load samples".into());
    }
    if let Some(v) = samples.iter().find(|v| **v > load1_max) {
        return Err(format!("load1 {v} exceeds {load1_max}"));
    }
    Ok(())
}

/// Gate used after the settle and before the workload is spawned.
/// A failing sample must not be followed by the trial load.
pub fn pre_idle_gate(load1: &[f64], settle_busy: &[f64], load1_max: f64, busy_max: f64) -> Result<(), String> {
    load_is_quiet(load1, load1_max)?;
    if let Some(v) = settle_busy.iter().copied().find(|v| *v > busy_max) {
        return Err(format!("settle busy fraction {v} exceeds {busy_max}"));
    }
    Ok(())
}

/// SMT siblings that were left outside the pin set. `siblings_of(cpu)` includes `cpu`.
pub fn unpinned_siblings(pinned: &[usize], siblings_of: impl Fn(usize) -> Vec<usize>) -> Vec<(usize, usize)> {
    let mut missing = Vec::new();
    for &cpu in pinned {
        for sib in siblings_of(cpu) {
            if sib != cpu && !pinned.contains(&sib) && !missing.contains(&(cpu, sib)) {
                missing.push((cpu, sib));
            }
        }
    }
    missing
}

pub fn parse_cpu_list(text: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for part in text.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            let start: usize = a.parse().map_err(|e| format!("{part}: {e}"))?;
            let end: usize = b.parse().map_err(|e| format!("{part}: {e}"))?;
            if end < start {
                return Err(format!("{part} is reversed"));
            }
            out.extend(start..=end);
        } else {
            out.push(part.parse().map_err(|e| format!("{part}: {e}"))?);
        }
    }
    if out.is_empty() {
        return Err("cpu list is empty".into());
    }
    Ok(out)
}

pub fn thread_siblings_list(cpu: usize) -> Result<Vec<usize>, String> {
    let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
    parse_cpu_list(text.trim())
}

pub fn require_smt_siblings(pinned: &[usize]) -> Result<(), String> {
    let mut known = Vec::new();
    for &cpu in pinned {
        let sibs = thread_siblings_list(cpu)?;
        known.push((cpu, sibs));
    }
    let missing = unpinned_siblings(pinned, |cpu| {
        known.iter().find(|(c, _)| *c == cpu).map(|(_, s)| s.clone()).unwrap_or_else(|| vec![cpu])
    });
    if missing.is_empty() {
        Ok(())
    } else {
        let pairs: Vec<String> = missing.iter().map(|(c, s)| format!("{c}->{s}")).collect();
        Err(format!("pinned core leaves its SMT sibling unpinned: {}", pairs.join(", ")))
    }
}

fn phase_temp(vals: &[Option<f64>]) -> Option<f64> {
    vals.iter().copied().flatten().reduce(f64::max)
}

fn refusal_grades(series: &EnergySeries, busy: Option<f64>) -> Grades {
    let pre_ok = series.pre_reached
        && pre_idle_gate(&series.load1_pre, &series.settle_busy, series.load1_max, series.busy_fraction_max).is_ok();
    let workload_over = busy.is_some_and(|b| b > series.busy_fraction_max)
        || series.workload_busy.iter().any(|v| *v > series.busy_fraction_max);
    let workload_ok = series.workload_reached && !workload_over;
    let post_ok = series.post_reached && load_is_quiet(&series.load1_post, series.load1_max).is_ok();
    let refused = Grade::Refused;
    Grades {
        pre_idle: if pre_ok { Grade::Quiet } else { refused },
        workload: if workload_ok { Grade::Quiet } else { refused },
        post_idle: if post_ok { Grade::Quiet } else { refused },
        rapl_package: refused,
        rapl_core: refused,
        amdgpu_ppt: refused,
    }
}

fn refused(reason: String, lock: LockHold, series: &EnergySeries, busy: Option<f64>) -> DriveReceipt {
    DriveReceipt {
        attributable: false,
        reason,
        grades: refusal_grades(series, busy),
        lock,
        busy_fraction_max: series.busy_fraction_max,
        busy_fraction: busy,
        cpu_tctl_c: phase_temp(&[series.tctl_pre, series.tctl_workload, series.tctl_post]),
        gpu_edge_c: phase_temp(&[series.edge_pre, series.edge_workload, series.edge_post]),
        rapl_package_j: None,
        rapl_core_j: None,
        amdgpu_ppt_j: None,
        series: series.clone(),
    }
}

/// Grade one captured attempt. Joules are present only when every gate passes.
pub fn judge(series: &EnergySeries, lock: LockHold) -> DriveReceipt {
    if lock.contended || lock.acquire_unix_ns.is_none() {
                return refused(crate::fleet_gate::HOST_LOCK_REFUSAL.into(), lock, series, None);
    }
    if let Err(e) = load_is_quiet(&series.load1_pre, series.load1_max) {
        return refused(format!("pre-trial idle: {e}"), lock, series, None);
    }
    let busy = match busy_fraction(&series.cpu_before, &series.cpu_after, &series.pinned) {
        Ok(b) => b.fraction,
        Err(e) => return refused(format!("workload cores: {e}"), lock, series, None),
    };
    if busy > series.busy_fraction_max {
        return refused(
            format!("non-pinned busy fraction {busy} exceeds the predeclared {}", series.busy_fraction_max),
            lock,
            series,
            Some(busy),
        );
    }
    if let Err(e) = load_is_quiet(&series.load1_post, series.load1_max) {
        return refused(format!("post-trial idle: {e}"), lock, series, Some(busy));
    }
    let pre_pkg = match powers_from_counter(&series.pre_package, series.range_uj) {
        Ok(v) => v,
        Err(e) => return refused(format!("package baseline: {e}"), lock, series, Some(busy)),
    };
    let post_pkg = match powers_from_counter(&series.post_package, series.range_uj) {
        Ok(v) => v,
        Err(e) => return refused(format!("package baseline: {e}"), lock, series, Some(busy)),
    };
    if let Err(e) = drift_within_3_sigma(&pre_pkg, &post_pkg) {
        return refused(e, lock, series, Some(busy));
    }
    let (pkg_du, _) = match delta_uj(series.trial_package.0, series.trial_package.1, series.range_uj) {
        Ok(v) => v,
        Err(e) => return refused(e, lock, series, Some(busy)),
    };
    let (core_du, _) = match delta_uj(series.trial_core.0, series.trial_core.1, series.range_uj) {
        Ok(v) => v,
        Err(e) => return refused(e, lock, series, Some(busy)),
    };
    DriveReceipt {
        attributable: true,
        reason: "quiet brackets, busy fraction, and 3σ drift all passed".into(),
        grades: Grades {
            pre_idle: Grade::Quiet,
            workload: Grade::Quiet,
            post_idle: Grade::Quiet,
            rapl_package: Grade::MeasuredAttributable,
            rapl_core: Grade::MeasuredAttributable,
            amdgpu_ppt: if series.trial_ppt_j.is_some() { Grade::MeasuredAttributable } else { Grade::Refused },
        },
        lock,
        busy_fraction_max: series.busy_fraction_max,
        busy_fraction: Some(busy),
        cpu_tctl_c: phase_temp(&[series.tctl_pre, series.tctl_workload, series.tctl_post]).or(series.cpu_tctl_c),
        gpu_edge_c: phase_temp(&[series.edge_pre, series.edge_workload, series.edge_post]).or(series.gpu_edge_c),
        rapl_package_j: Some(pkg_du as f64 * 1e-6),
        rapl_core_j: Some(core_du as f64 * 1e-6),
        amdgpu_ppt_j: series.trial_ppt_j,
        series: series.clone(),
    }
}

fn finite_max(v: &[f64]) -> Option<f64> {
    v.iter().copied().filter(|x| x.is_finite()).reduce(f64::max)
}

fn drift_json(series: &EnergySeries) -> serde_json::Value {
    if !series.pre_reached {
        return serde_json::json!({"status": "not_reached", "phase": "pre_idle"});
    }
    if !series.workload_reached {
        return serde_json::json!({"status": "not_reached", "phase": "workload"});
    }
    if !series.post_reached {
        return serde_json::json!({"status": "not_reached", "phase": "post_idle"});
    }
    let pre = powers_from_counter(&series.pre_package, series.range_uj).ok();
    let post = powers_from_counter(&series.post_package, series.range_uj).ok();
    match (pre, post) {
        (Some(pre), Some(post)) => match drift_within_3_sigma(&pre, &post) {
            Ok(()) => {
                let delta = (mean(&pre) - mean(&post)).abs();
                let sigma = {
                    let v1 = sample_var(&pre).unwrap_or(0.0);
                    let v2 = sample_var(&post).unwrap_or(0.0);
                    let n1 = pre.len();
                    let n2 = post.len();
                    let pooled = ((n1 - 1) as f64 * v1 + (n2 - 1) as f64 * v2) / (n1 + n2 - 2) as f64;
                    pooled.sqrt()
                };
                serde_json::json!({"status": "computed", "delta_w": delta, "sigma_w": sigma, "within_3_sigma": true})
            }
            Err(reason) => serde_json::json!({"status": "computed", "within_3_sigma": false, "reason": reason}),
        },
        _ => serde_json::json!({"status": "not_reached", "phase": "post_idle"}),
    }
}

fn phase_json(reached: bool, grade: Grade, load1: &[f64], busy: &[f64], package_w: &[f64], core_w: &[f64], ppt_w: &[f64], tctl: Option<f64>, edge: Option<f64>) -> serde_json::Value {
    serde_json::json!({
        "reached": reached,
        "grade": grade.as_str(),
        "load1": load1,
        "load1_max": finite_max(load1),
        "busy_fraction": busy,
        "baseline_power_w": {
            "rapl_package": package_w,
            "rapl_core": core_w,
            "amdgpu_ppt": ppt_w,
        },
        "cpu_tctl_max_c": tctl,
        "gpu_edge_max_c": edge,
    })
}

pub fn receipt_json(r: &DriveReceipt) -> serde_json::Value {
    let series = &r.series;
    let pre_pkg = powers_from_counter(&series.pre_package, series.range_uj).unwrap_or_default();
    let post_pkg = if series.post_reached { powers_from_counter(&series.post_package, series.range_uj).unwrap_or_default() } else { Vec::new() };
    let pre_core = powers_from_counter(&series.pre_core, series.range_uj).unwrap_or_default();
    let post_core = if series.post_reached { powers_from_counter(&series.post_core, series.range_uj).unwrap_or_default() } else { Vec::new() };
    let joules = if r.attributable {
        serde_json::json!({
            "rapl_package": r.rapl_package_j,
            "rapl_core": r.rapl_core_j,
            "amdgpu_ppt": r.amdgpu_ppt_j,
        })
    } else {
        serde_json::Value::Null
    };
    serde_json::json!({
        "attributable": r.attributable,
        "reason": r.reason,
        "evidence_grade": {
            "pre_idle": r.grades.pre_idle.as_str(),
            "workload": r.grades.workload.as_str(),
            "post_idle": r.grades.post_idle.as_str(),
            "rapl_package": r.grades.rapl_package.as_str(),
            "rapl_core": r.grades.rapl_core.as_str(),
            "amdgpu_ppt": r.grades.amdgpu_ppt.as_str(),
            "meters_pooled": false,
        },
        "phases": {
            "pre_idle": phase_json(series.pre_reached, r.grades.pre_idle, &series.load1_pre, &series.settle_busy, &pre_pkg, &pre_core, &series.pre_ppt_w, series.tctl_pre, series.edge_pre),
            "workload": phase_json(series.workload_reached, r.grades.workload, &[], &series.workload_busy, &[], &[], &[], series.tctl_workload, series.edge_workload),
            "post_idle": phase_json(series.post_reached, r.grades.post_idle, &series.load1_post, &[], &post_pkg, &post_core, &series.post_ppt_w, series.tctl_post, series.edge_post),
        },
        "drift": drift_json(series),
        "lock": {
            "path": r.lock.path,
            "pid": r.lock.pid,
            "acquire_unix_ns": r.lock.acquire_unix_ns,
            "release_unix_ns": r.lock.release_unix_ns,
            "first_attempt_contended": r.lock.first_attempt_contended,
            "contended": r.lock.contended,
        },
        "busy_fraction_max": r.busy_fraction_max,
        "busy_fraction": r.busy_fraction,
        "cpu_tctl_c": r.cpu_tctl_c,
        "gpu_edge_c": r.gpu_edge_c,
        "joules": joules,
        "evidence_grade": if !r.lock.acquire_unix_ns.is_some() || r.lock.contended {
            crate::fleet_gate::HOST_LOCK_REFUSAL
        } else if r.attributable {
            "quiet"
        } else {
            "load_gated"
        },
        "host_lock": {
            "path": r.lock.path,
            "acquired_utc": r.lock.acquire_unix_ns.map(|ns| crate::fleet_gate::format_unix_utc(ns as f64 / 1e9)),
            "released_utc": r.lock.release_unix_ns.map(|ns| crate::fleet_gate::format_unix_utc(ns as f64 / 1e9)),
            "wait_s": r.lock.wait_s,
            "holder_pid": r.lock.pid,
            "held": r.lock.acquire_unix_ns.is_some() && !r.lock.contended,
            "reason": if r.lock.acquire_unix_ns.is_some() && !r.lock.contended { serde_json::Value::Null } else { serde_json::json!(crate::fleet_gate::HOST_LOCK_REFUSAL) },
        },
        "trial_window_rule": {
            "enforce": true,
            "other_core_mean_max": r.busy_fraction_max,
            "other_core_sample_max": crate::fleet_gate::OTHER_CORE_SAMPLE_MAX,
            "why_load1_is_not_a_trial_criterion": "load1 counts runnable threads, so the workload under test raises it. Brackets use load1. Trial windows do not.",
        },
        "baseline_settle_load1": r.series.settle_load1_pre,
        "rapl_package_j": r.rapl_package_j,
        "rapl_core_j": r.rapl_core_j,
        "amdgpu_ppt_j": r.amdgpu_ppt_j,
    })
}

pub fn pinned_argv(cores: &[usize], workload: &[String]) -> Vec<String> {
    let spec = cores.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",");
    let mut argv = vec!["taskset".into(), "-c".into(), spec];
    argv.extend(workload.iter().cloned());
    argv
}

pub struct LockGuard {
    file: File,
    pub path: String,
    pub pid: u32,
    pub acquire_unix_ns: u64,
    pub first_attempt_contended: bool,
}

impl LockGuard {
    pub fn hold(&self) -> LockHold {
        LockHold {
            path: self.path.clone(),
            pid: self.pid,
            acquire_unix_ns: Some(self.acquire_unix_ns),
            release_unix_ns: None,
            first_attempt_contended: self.first_attempt_contended,
            contended: false,
            wait_s: 0.0,
        }
    }
    pub fn release_stamp(&self) -> LockHold {
        let mut h = self.hold();
        h.release_unix_ns = Some(unix_ns());
        h
    }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

pub fn try_lock_exclusive(path: &Path) -> Result<Option<LockGuard>, String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }
    let file = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    match file.try_lock_exclusive() {
        Ok(()) =>         Ok(Some(LockGuard {
            file: {
                use std::io::Write;
                let mut owned = file;
                let _ = owned.write_all(format!("{}\n", std::process::id()).as_bytes());
                let _ = owned.flush();
                owned
            },
            path: path.display().to_string(),
            pid: std::process::id(),
            acquire_unix_ns: unix_ns(),
            first_attempt_contended: false,
        })),
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
        Err(e) => Err(format!("flock {}: {e}", path.display())),
    }
}

fn unix_ns() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
}

pub fn write_receipt(path: &Path, receipt: &DriveReceipt) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut f = File::create(path).map_err(|e| e.to_string())?;
    let text = serde_json::to_string_pretty(&receipt_json(receipt)).map_err(|e| e.to_string())?;
    f.write_all(text.as_bytes()).map_err(|e| e.to_string())?;
    Ok(())
}

/// Retry schedule. Sleeps only between attempts, never while a counter window is open.
pub fn poll_host_lock(path: &Path, wait_s: f64) -> Result<Option<LockGuard>, String> {
    let started = std::time::Instant::now();
    let wait = if wait_s.is_finite() && wait_s > 0.0 { wait_s } else { 0.0 };
    loop {
        if let Some(guard) = try_lock_exclusive(path)? {
            return Ok(Some(guard));
        }
        let waited = started.elapsed().as_secs_f64();
        if waited >= wait {
            return Ok(None);
        }
        sleep(Duration::from_millis(200));
    }
}

pub fn run_until_quiet<F>(
    lock_path: &Path,
    out_dir: &Path,
    deadline: Duration,
    retry_wait: Duration,
    lock_wait_s: f64,
    mut attempt: F,
) -> Result<DriveReceipt, String>
where
    F: FnMut(&LockGuard) -> DriveReceipt,
{
    let started = std::time::Instant::now();
    let mut first_contended = false;
    let mut n = 0u32;
    loop {
        if started.elapsed() > deadline {
            let series = empty_series();
            let lock = LockHold {
                path: lock_path.display().to_string(),
                pid: std::process::id(),
                acquire_unix_ns: None,
                release_unix_ns: None,
                first_attempt_contended: first_contended,
                contended: false,
                wait_s: 0.0,
            };
            let receipt = refused("deadline reached before a quiet attributable run".into(), lock, &series, None);
            write_receipt(&out_dir.join(format!("refusal-deadline.json")), &receipt)?;
            return Ok(receipt);
        }
        match poll_host_lock(lock_path, lock_wait_s)? {
            None => {
                first_contended = true;
                let series = empty_series();
                let lock = LockHold {
                    path: lock_path.display().to_string(),
                    pid: std::process::id(),
                    acquire_unix_ns: None,
                    release_unix_ns: None,
                    first_attempt_contended: true,
                    contended: true,
                    wait_s: 0.0,
                };
                let receipt = refused(crate::fleet_gate::HOST_LOCK_REFUSAL.into(), lock, &series, None);
                write_receipt(&out_dir.join(format!("refusal-lock-{n}.json")), &receipt)?;
                n += 1;
                sleep(retry_wait);
            }
            Some(mut guard) => {
                guard.first_attempt_contended = first_contended;
                let mut receipt = attempt(&guard);
                receipt.lock.first_attempt_contended = first_contended;
                receipt.lock.release_unix_ns = Some(unix_ns());
                drop(guard);
                if receipt.attributable {
                    write_receipt(&out_dir.join("attributable.json"), &receipt)?;
                    return Ok(receipt);
                }
                write_receipt(&out_dir.join(format!("refusal-{n}.json")), &receipt)?;
                n += 1;
                sleep(retry_wait);
            }
        }
    }
}

fn empty_series() -> EnergySeries {
    EnergySeries {
        load1_pre: vec![],
        load1_post: vec![],
        pre_package: vec![],
        post_package: vec![],
        pre_core: vec![],
        post_core: vec![],
        pre_ppt_w: vec![],
        post_ppt_w: vec![],
        trial_package: (0, 0),
        trial_core: (0, 0),
        trial_ppt_j: None,
        cpu_before: vec![],
        cpu_after: vec![],
        pinned: vec![],
        busy_fraction_max: 0.0,
        load1_max: LOAD1_MAX,
        range_uj: FRAMEWORK_RANGE_UJ,
        cpu_tctl_c: None,
        gpu_edge_c: None,
        settle_busy: vec![],
        workload_busy: vec![],
        tctl_pre: None,
        tctl_workload: None,
        tctl_post: None,
        edge_pre: None,
        edge_workload: None,
        edge_post: None,
        workload_reached: false,
        post_reached: false,
        pre_reached: false,
        settle_load1_pre: vec![],
        settle_load1_post: vec![],
    }
}

pub fn read_load1() -> Result<f64, String> {
    if let Ok(text) = std::fs::read_to_string("/proc/loadavg") {
        return crate::macmon::parse_proc_load1(&text);
    }
    ferric_joule::macos_load1().ok_or_else(|| "getloadavg returned no load1".into())
}

pub fn read_cpu_ticks() -> Result<Vec<CpuTicks>, String> {
    let text = std::fs::read_to_string("/proc/stat").map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let Some(name) = parts.next() else { continue };
        if !name.starts_with("cpu") || name == "cpu" {
            continue;
        }
        let cpu: usize = name[3..].parse().map_err(|e| format!("{name}: {e}"))?;
        let nums: Vec<u64> = parts.map(|p| p.parse::<u64>().unwrap_or(0)).collect();
        if nums.len() < 5 {
            continue;
        }
        let user = nums[0];
        let nice = nums[1];
        let system = nums[2];
        let idle = nums[3];
        let iowait = nums[4];
        let irq = *nums.get(5).unwrap_or(&0);
        let soft = *nums.get(6).unwrap_or(&0);
        let steal = *nums.get(7).unwrap_or(&0);
        let busy = user + nice + system + irq + soft + steal;
        let total = busy + idle + iowait;
        out.push(CpuTicks { cpu, busy, total });
    }
    Ok(out)
}

pub fn spawn_pinned(cores: &[usize], workload: &[String]) -> Result<std::process::Child, String> {
    let argv = pinned_argv(cores, workload);
    if argv.len() < 4 {
        return Err("workload command is empty".into());
    }
    std::process::Command::new(&argv[0])
        .args(&argv[1..])
        .spawn()
        .map_err(|e| format!("spawn {}: {e}", argv[0]))
}

pub struct LiveConfig {
    pub rapl_package: PathBuf,
    pub rapl_core: PathBuf,
    pub ppt: Option<PathBuf>,
    pub cpu_tctl: Option<PathBuf>,
    pub gpu_edge: Option<PathBuf>,
    pub settle_s: f64,
    pub load1_max: f64,
    pub busy_fraction_max: f64,
    pub pin_cores: Vec<usize>,
    pub workload: Vec<String>,
}

fn read_uj(path: &Path) -> Result<u64, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.trim().parse::<u64>().map_err(|e| format!("{}: {e}", path.display()))
}

fn range_uj(energy_path: &Path) -> Result<u64, String> {
    let name = energy_path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if name != "energy_uj" {
        return Err(format!("{} is not an energy_uj file", energy_path.display()));
    }
    let range = energy_path.with_file_name("max_energy_range_uj");
    read_uj(&range)
}

fn millic_to_c(path: &Option<PathBuf>) -> Option<f64> {
    let path = path.as_ref()?;
    read_uj(path).ok().map(|v| v as f64 / 1000.0)
}

struct Bracket {
    load1: Vec<f64>,
    package: Vec<(u64, u64)>,
    core: Vec<(u64, u64)>,
    ppt_w: Vec<f64>,
    cpu_tctl_c: Option<f64>,
    gpu_edge_c: Option<f64>,
    busy: Vec<f64>,
    blocked: Option<String>,
}

fn note_temp(slot: &mut Option<f64>, sample: Option<f64>) {
    if let Some(c) = sample {
        *slot = Some(slot.map(|m| m.max(c)).unwrap_or(c));
    }
}

fn take_bracket(cfg: &LiveConfig, origin: std::time::Instant) -> Result<Bracket, String> {
    let steps = cfg.settle_s.round().max(1.0) as u32;
    let mut bracket = Bracket {
        load1: vec![],
        package: vec![],
        core: vec![],
        ppt_w: vec![],
        cpu_tctl_c: None,
        gpu_edge_c: None,
        busy: vec![],
        blocked: None,
    };
    let mut prev_ticks: Option<Vec<CpuTicks>> = None;
    for i in 0..steps {
        match read_load1() {
            Ok(v) => bracket.load1.push(v),
            Err(e) if bracket.load1.is_empty() => return Err(e),
            Err(e) => {
                bracket.blocked = Some(e);
                return Ok(bracket);
            }
        }
        let t_ns = origin.elapsed().as_nanos() as u64;
        match (read_uj(&cfg.rapl_package), read_uj(&cfg.rapl_core)) {
            (Ok(pkg), Ok(core)) => {
                bracket.package.push((t_ns, pkg));
                bracket.core.push((t_ns, core));
            }
            (Err(e), _) | (_, Err(e)) => {
                bracket.blocked = Some(e);
                return Ok(bracket);
            }
        }
        if let Some(path) = &cfg.ppt {
            if let Ok(uw) = read_uj(path) {
                bracket.ppt_w.push(uw as f64 * 1e-6);
            }
        }
        note_temp(&mut bracket.cpu_tctl_c, millic_to_c(&cfg.cpu_tctl));
        note_temp(&mut bracket.gpu_edge_c, millic_to_c(&cfg.gpu_edge));
        if let Ok(now) = read_cpu_ticks() {
            if let Some(prev) = &prev_ticks {
                if let Ok(b) = busy_fraction(prev, &now, &cfg.pin_cores) {
                    bracket.busy.push(b.fraction);
                }
            }
            prev_ticks = Some(now);
        }
        if crate::fleet_gate::bracket_load_offender(&bracket.load1.iter().copied().map(Some).collect::<Vec<_>>()).is_some() {
            bracket.blocked = load_is_quiet(&bracket.load1, cfg.load1_max).err();
            return Ok(bracket);
        }
        if i + 1 < steps {
            sleep(Duration::from_secs(1));
        }
    }
    Ok(bracket)
}

fn series_from_pre(cfg: &LiveConfig, range: u64, pre: &Bracket) -> EnergySeries {
    let mut series = empty_series();
    series.load1_pre = pre.load1.clone();
    series.pre_package = pre.package.clone();
    series.pre_core = pre.core.clone();
    series.pre_ppt_w = pre.ppt_w.clone();
    series.settle_busy = pre.busy.clone();
    series.tctl_pre = pre.cpu_tctl_c;
    series.edge_pre = pre.gpu_edge_c;
    series.pre_reached = true;
    series.busy_fraction_max = cfg.busy_fraction_max;
    series.load1_max = cfg.load1_max;
    series.pinned = cfg.pin_cores.clone();
    series.range_uj = range;
    series
}

fn refuse_pre(lock: &LockGuard, cfg: &LiveConfig, range: u64, pre: &Bracket, reason: String) -> DriveReceipt {
    refused(reason, lock.hold(), &series_from_pre(cfg, range, pre), None)
}

/// One locked attempt: preflight, both idle brackets, and the pinned workload.
/// The caller holds the flock for the whole call.
pub fn run_locked_attempt(cfg: &LiveConfig, lock: &LockGuard) -> DriveReceipt {
    let range = match (range_uj(&cfg.rapl_package), range_uj(&cfg.rapl_core)) {
        (Ok(a), Ok(b)) if a == b => a,
        (Ok(a), Ok(b)) => {
            return refusal_now(lock, cfg, format!("package range {a} and core range {b} differ"));
        }
        (Err(e), _) | (_, Err(e)) => return refusal_now(lock, cfg, e),
    };
    if let Err(e) = require_smt_siblings(&cfg.pin_cores) {
        return refusal_now(lock, cfg, e);
    }
    if let Err(e) = wait_for_load_streak() {
        return refusal_now(lock, cfg, format!("baseline settle: {e}"));
    }
    let origin = std::time::Instant::now();
    let pre = match take_bracket(cfg, origin) {
        Ok(b) => b,
        Err(e) => return refusal_now(lock, cfg, e),
    };
    if let Some(reason) = pre.blocked.clone() {
        return refuse_pre(lock, cfg, range, &pre, format!("pre-trial idle: {reason}"));
    }
    if let Err(e) = load_is_quiet(&pre.load1, cfg.load1_max) {
        return refuse_pre(lock, cfg, range, &pre, format!("pre-trial idle: {e}"));
    }
    let ticks0 = match read_cpu_ticks() {
        Ok(v) => v,
        Err(e) => return refuse_pre(lock, cfg, range, &pre, e),
    };
    let pkg0 = match read_uj(&cfg.rapl_package) {
        Ok(v) => v,
        Err(e) => return refuse_pre(lock, cfg, range, &pre, e),
    };
    let core0 = match read_uj(&cfg.rapl_core) {
        Ok(v) => v,
        Err(e) => return refuse_pre(lock, cfg, range, &pre, e),
    };
    let mut child = match spawn_pinned(&cfg.pin_cores, &cfg.workload) {
        Ok(c) => c,
        Err(e) => return refuse_pre(lock, cfg, range, &pre, e),
    };
    let mut ppt_series: Vec<(u64, f64)> = Vec::new();
    let mut workload_busy = Vec::new();
    let mut tctl_workload = None;
    let mut edge_workload = None;
    let mut prev_ticks = ticks0.clone();
    let abort_workload = |reason: String, busy: Option<f64>, workload_busy: Vec<f64>, tctl_workload: Option<f64>, edge_workload: Option<f64>| {
        let mut series = series_from_pre(cfg, range, &pre);
        series.workload_reached = true;
        series.workload_busy = workload_busy;
        series.tctl_workload = tctl_workload;
        series.edge_workload = edge_workload;
        refused(reason, lock.hold(), &series, busy)
    };
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    return abort_workload(format!("workload exit {status}"), None, workload_busy, tctl_workload, edge_workload);
                }
                break;
            }
            Ok(None) => {
                sleep(Duration::from_secs(1));
                let t_ns = origin.elapsed().as_nanos() as u64;
                note_temp(&mut tctl_workload, millic_to_c(&cfg.cpu_tctl));
                note_temp(&mut edge_workload, millic_to_c(&cfg.gpu_edge));
                if let Some(path) = &cfg.ppt {
                    if let Ok(uw) = read_uj(path) {
                        ppt_series.push((t_ns, uw as f64 * 1e-6));
                    }
                }
                if let Ok(now_ticks) = read_cpu_ticks() {
                    if let Ok(interval) = busy_fraction(&prev_ticks, &now_ticks, &cfg.pin_cores) {
                        workload_busy.push(interval.fraction);
                    }
                    if let Ok(busy) = busy_fraction(&ticks0, &now_ticks, &cfg.pin_cores) {
                        let interval_over = workload_busy.last().copied().is_some_and(|v| v > crate::fleet_gate::OTHER_CORE_SAMPLE_MAX);
                        let mean_over = crate::fleet_gate::fraction_refusal(&workload_busy, cfg.busy_fraction_max, crate::fleet_gate::OTHER_CORE_SAMPLE_MAX)
                            == Some("the mean exceeded the maximum");
                        if busy.fraction > cfg.busy_fraction_max || interval_over || mean_over {
                            let _ = child.kill();
                            let _ = child.wait();
                            return abort_workload(
                                format!("non-pinned busy fraction {} exceeds the predeclared {}", busy.fraction, cfg.busy_fraction_max),
                                Some(busy.fraction),
                                workload_busy,
                                tctl_workload,
                                edge_workload,
                            );
                        }
                    }
                    prev_ticks = now_ticks;
                }
            }
            Err(e) => return abort_workload(e.to_string(), None, workload_busy, tctl_workload, edge_workload),
        }
    }
    let pkg1 = match read_uj(&cfg.rapl_package) {
        Ok(v) => v,
        Err(e) => return abort_workload(e, None, workload_busy, tctl_workload, edge_workload),
    };
    let core1 = match read_uj(&cfg.rapl_core) {
        Ok(v) => v,
        Err(e) => return abort_workload(e, None, workload_busy, tctl_workload, edge_workload),
    };
    let ticks1 = match read_cpu_ticks() {
        Ok(v) => v,
        Err(e) => return abort_workload(e, None, workload_busy, tctl_workload, edge_workload),
    };
    if let Err(e) = wait_for_load_streak() {
        return abort_workload(format!("post-trial settle: {e}"), None, workload_busy, tctl_workload, edge_workload);
    }
    let post = match take_bracket(cfg, origin) {
        Ok(b) => b,
        Err(e) => return abort_workload(e, None, workload_busy, tctl_workload, edge_workload),
    };
    let mut series = series_from_pre(cfg, range, &pre);
    series.load1_post = post.load1;
    series.post_package = post.package;
    series.post_core = post.core;
    series.post_ppt_w = post.ppt_w;
    series.tctl_post = post.cpu_tctl_c;
    series.edge_post = post.gpu_edge_c;
    series.trial_package = (pkg0, pkg1);
    series.trial_core = (core0, core1);
    series.trial_ppt_j = crate::trapezoid_joules(&ppt_series);
    series.cpu_before = ticks0;
    series.cpu_after = ticks1;
    series.workload_busy = workload_busy;
    series.tctl_workload = tctl_workload;
    series.edge_workload = edge_workload;
    series.workload_reached = true;
    series.post_reached = true;
    series.cpu_tctl_c = phase_temp(&[series.tctl_pre, series.tctl_workload, series.tctl_post]);
    series.gpu_edge_c = phase_temp(&[series.edge_pre, series.edge_workload, series.edge_post]);
    if let Some(reason) = post.blocked {
        return refused(format!("post-trial idle: {reason}"), lock.hold(), &series, None);
    }
    judge(&series, lock.hold())
}

fn wait_for_load_streak() -> Result<(), String> {
    let mut raw = Vec::new();
    while raw.len() < crate::fleet_gate::SETTLE_BOUND {
        raw.push(Some(read_load1()?));
        let outcome = crate::fleet_gate::settle_outcome(&raw);
        if outcome.settled {
            return Ok(());
        }
        if raw.len() == crate::fleet_gate::SETTLE_BOUND {
            return Err(outcome.reason.into());
        }
        sleep(Duration::from_secs(1));
    }
    Err("settle bound exhausted".into())
}

fn refusal_now(lock: &LockGuard, cfg: &LiveConfig, reason: String) -> DriveReceipt {
    let mut series = empty_series();
    series.busy_fraction_max = cfg.busy_fraction_max;
    series.load1_max = cfg.load1_max;
    series.pinned = cfg.pin_cores.clone();
    refused(reason, lock.hold(), &series, None)
}

pub struct MacmonConfig {
    pub settle_s: f64,
    pub load1_max: f64,
    pub sample_ms: u64,
    pub macmon_bin: String,
    pub workload: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct MacmonAttempt {
    pub load1_pre: Vec<f64>,
    pub load1_post: Vec<f64>,
    pub pre: Vec<(u64, crate::macmon::Reading)>,
    pub trial: Vec<(u64, crate::macmon::Reading)>,
    pub post: Vec<(u64, crate::macmon::Reading)>,
    pub load1_max: f64,
    pub pre_reached: bool,
    pub workload_reached: bool,
    pub post_reached: bool,
    pub failure: Option<String>,
}

pub fn macmon_receipt(attempt: &MacmonAttempt, lock: &LockHold) -> serde_json::Value {
    let sys = |rows: &[(u64, crate::macmon::Reading)]| -> Vec<f64> {
        rows.iter().filter_map(|(_, r)| r.sys_w).collect()
    };
    let temp_max = |rows: &[(u64, crate::macmon::Reading)], cpu: bool| -> Option<f64> {
        rows.iter()
            .filter_map(|(_, r)| if cpu { r.cpu_temp_c } else { r.gpu_temp_c })
            .reduce(f64::max)
    };
    let mut reason = String::new();
    let mut ok = attempt.failure.is_none();
    if let Some(e) = &attempt.failure {
        ok = false;
        reason = e.clone();
    }
    if ok {
        if let Err(e) = load_is_quiet(&attempt.load1_pre, attempt.load1_max) {
            ok = false;
            reason = format!("pre-trial idle: {e}");
        }
    }
    if ok && attempt.post_reached {
        if let Err(e) = load_is_quiet(&attempt.load1_post, attempt.load1_max) {
            ok = false;
            reason = format!("post-trial idle: {e}");
        }
    }
    let pre_sys = sys(&attempt.pre);
    let post_sys = sys(&attempt.post);
    let drift = if !attempt.pre_reached {
        serde_json::json!({"status": "not_reached", "phase": "pre_idle"})
    } else if !attempt.workload_reached {
        serde_json::json!({"status": "not_reached", "phase": "workload"})
    } else if !attempt.post_reached {
        serde_json::json!({"status": "not_reached", "phase": "post_idle"})
    } else {
        match drift_within_3_sigma(&pre_sys, &post_sys) {
            Ok(()) => serde_json::json!({"status": "computed", "within_3_sigma": true}),
            Err(e) => {
                if ok {
                    ok = false;
                    reason = e.clone();
                }
                serde_json::json!({"status": "computed", "within_3_sigma": false, "reason": e})
            }
        }
    };
    let joules = if ok { Some(crate::macmon::integrate(&attempt.trial)) } else { None };
    if ok && joules.as_ref().is_some_and(|j| j.sys.is_none() || j.cpu.is_none() || j.gpu.is_none() || j.ane.is_none()) {
        ok = false;
        reason = "a macmon meter has fewer than two samples in the workload window".into();
    }
    let joules = if ok { joules } else { None };
    if reason.is_empty() {
        reason = "quiet load1 and idle sys_power baseline passed; busy fraction was not available".into();
    }
    let phase = |reached: bool, load1: &[f64], rows: &[(u64, crate::macmon::Reading)]| {
        serde_json::json!({
            "reached": reached,
            "load1": load1,
            "load1_max": load1.iter().copied().reduce(f64::max),
            "sys_power_w": sys(rows),
            "cpu_temp_avg_max_c": temp_max(rows, true),
            "gpu_temp_avg_max_c": temp_max(rows, false),
        })
    };
    serde_json::json!({
        "meter": "macmon",
        "attributable": ok,
        "reason": reason,
        "joules": joules.map(|j| serde_json::json!({
            "gpu_power": j.gpu,
            "cpu_power": j.cpu,
            "ane_power": j.ane,
            "ram_power": j.ram,
            "gpu_ram_power": j.gpu_ram,
            "sys_power": j.sys,
            "soc": j.soc,
            "soc_label": j.soc_label,
            "accelerator": j.accelerator,
            "accelerator_label": j.accelerator_label,
            "dram_scope": j.dram_scope,
        })),
        "context": {
            "pre_idle": crate::macmon::context_means(&attempt.pre),
            "workload": crate::macmon::context_means(&attempt.trial),
            "post_idle": crate::macmon::context_means(&attempt.post),
        },
        "power_sample": "macmon pipe -i <ms> -s 0, 100-250 ms, trapezoid over the actual gap between samples",
        "gate_sample_hz": 1,
        "summed": false,
        "soc_telemetry": crate::fleet_gate::SOC_TELEMETRY_LIMIT,
        "macmon_version_note": "ferrotherm c82c1e96 does not name a macmon release. macmon 0.11.0 can report ram_power and gpu_ram_power as exactly 0 while sys_power is nonzero. An exact 0 is unavailable, not a measured zero, and SoC and accelerator totals are refused rather than published without DRAM.",
        "overlaps": crate::macmon::overlaps(),
        "trial_window_rule": {
            "enforce": true,
            "foreign_cpu_mean_max": crate::fleet_gate::FOREIGN_CPU_MEAN_MAX,
            "foreign_cpu_sample_max": crate::fleet_gate::FOREIGN_CPU_SAMPLE_MAX,
            "why_load1_is_not_a_trial_criterion": "load1 counts runnable threads, so the workload under test raises it. Brackets use load1. Trial windows do not.",
        },
        "pinning": crate::macmon::PINNING,
        "pinning_reason": crate::macmon::PINNING_REASON,
        "busy_fraction": null,
        "busy_fraction_status": crate::macmon::BUSY_FRACTION_STATUS,
        "busy_fraction_reason": crate::macmon::BUSY_FRACTION_REASON,
        "declared_before_run": true,
        "phases": {
            "pre_idle": phase(attempt.pre_reached, &attempt.load1_pre, &attempt.pre),
            "workload": phase(attempt.workload_reached, &[], &attempt.trial),
            "post_idle": phase(attempt.post_reached, &attempt.load1_post, &attempt.post),
        },
        "drift": drift,
        "evidence_grade": if lock.acquire_unix_ns.is_none() || lock.contended {
            crate::fleet_gate::HOST_LOCK_REFUSAL
        } else if ok {
            "quiet"
        } else {
            "load_gated"
        },
        "host_lock": {
            "path": lock.path,
            "acquired_utc": lock.acquire_unix_ns.map(|ns| crate::fleet_gate::format_unix_utc(ns as f64 / 1e9)),
            "released_utc": lock.release_unix_ns.map(|ns| crate::fleet_gate::format_unix_utc(ns as f64 / 1e9)),
            "wait_s": lock.wait_s,
            "holder_pid": lock.pid,
            "held": lock.acquire_unix_ns.is_some() && !lock.contended,
            "reason": if lock.acquire_unix_ns.is_some() && !lock.contended { serde_json::Value::Null } else { serde_json::json!(crate::fleet_gate::HOST_LOCK_REFUSAL) },
        },
    })
}

fn slice_new(before: &[(u64, crate::macmon::Reading)], after: &[(u64, crate::macmon::Reading)]) -> Vec<(u64, crate::macmon::Reading)> {
    if after.len() >= before.len() { after[before.len()..].to_vec() } else { Vec::new() }
}

fn macmon_fail(reason: String, load1_pre: Vec<f64>, pre: Vec<(u64, crate::macmon::Reading)>, workload: bool, lock: &LockHold) -> serde_json::Value {
    macmon_receipt(&MacmonAttempt {
        load1_pre,
        load1_post: vec![],
        pre,
        trial: vec![],
        post: vec![],
        load1_max: LOAD1_MAX,
        pre_reached: true,
        workload_reached: workload,
        post_reached: false,
        failure: Some(reason),
    }, lock)
}

pub fn run_locked_macmon(cfg: &MacmonConfig, lock: &LockGuard) -> serde_json::Value {
    let hold = lock.hold();
    let mut sampler = match crate::macmon::Sampler::spawn(std::path::Path::new(&cfg.macmon_bin), cfg.sample_ms) {
        Ok(s) => s,
        Err(e) => {
            return macmon_receipt(&MacmonAttempt {
                load1_pre: vec![],
                load1_post: vec![],
                pre: vec![],
                trial: vec![],
                post: vec![],
                load1_max: cfg.load1_max,
                pre_reached: false,
                workload_reached: false,
                post_reached: false,
                failure: Some(e),
            }, &hold);
        }
    };
    let settle = || -> Result<Vec<f64>, (Vec<f64>, String)> {
        let mut raw: Vec<Option<f64>> = Vec::new();
        while raw.len() < crate::fleet_gate::SETTLE_BOUND {
            match read_load1() {
                Ok(v) => raw.push(Some(v)),
                Err(e) => return Err((raw.into_iter().flatten().collect(), e)),
            }
            let outcome = crate::fleet_gate::settle_outcome(&raw);
            if outcome.settled {
                return Ok(outcome.samples);
            }
            if raw.len() == crate::fleet_gate::SETTLE_BOUND {
                return Err((outcome.samples, outcome.reason.into()));
            }
            sleep(Duration::from_secs(1));
        }
        Err((raw.into_iter().flatten().collect(), "settle bound exhausted".into()))
    };
    let load1_pre = match settle() {
        Ok(v) => v,
        Err((load1, e)) => {
            let pre = sampler.snapshot();
            sampler.stop();
            return macmon_receipt(&MacmonAttempt {
                load1_pre: load1,
                load1_post: vec![],
                pre,
                trial: vec![],
                post: vec![],
                load1_max: cfg.load1_max,
                pre_reached: true,
                workload_reached: false,
                post_reached: false,
                failure: Some(format!("pre-trial idle: {e}")),
            }, &hold);
        }
    };
    let pre = sampler.snapshot();
    if cfg.workload.is_empty() {
        sampler.stop();
        return macmon_fail("workload command is empty".into(), load1_pre, pre, false, &hold);
    }
    let mut child = match std::process::Command::new(&cfg.workload[0]).args(&cfg.workload[1..]).spawn() {
        Ok(c) => c,
        Err(e) => {
            sampler.stop();
            return macmon_fail(format!("spawn workload: {e}"), load1_pre, pre, false, &hold);
        }
    };
    let mut foreign = Vec::new();
    let mut prev_ps = crate::fleet_gate::read_ps_cpu().ok();
    let workload_end = loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => break sampler.snapshot(),
            Ok(Some(status)) => {
                sampler.stop();
                return macmon_fail(format!("workload exit {status}"), load1_pre, pre, true, &hold);
            }
            Ok(None) => {
                sleep(Duration::from_secs(1));
                if let Ok(now) = crate::fleet_gate::read_ps_cpu() {
                    if let Some(prev) = &prev_ps {
                        if let Some(frac) = crate::fleet_gate::foreign_cpu_fraction(prev, &now, std::process::id(), 1.0) {
                            foreign.push(frac);
                        }
                    }
                    prev_ps = Some(now);
                }
            }
            Err(e) => {
                sampler.stop();
                return macmon_fail(e.to_string(), load1_pre, pre, true, &hold);
            }
        }
    };
    if let Some(why) = crate::fleet_gate::fraction_refusal(&foreign, crate::fleet_gate::FOREIGN_CPU_MEAN_MAX, crate::fleet_gate::FOREIGN_CPU_SAMPLE_MAX) {
        sampler.stop();
        return macmon_fail(format!("non-harness CPU: {why}"), load1_pre, pre, true, &hold);
    }
    let trial = slice_new(&pre, &workload_end);
    let load1_post = match settle() {
        Ok(v) => v,
        Err((load1, e)) => {
            let post_all = sampler.snapshot();
            sampler.stop();
            return macmon_receipt(&MacmonAttempt {
                load1_pre,
                load1_post: load1,
                pre,
                trial,
                post: slice_new(&workload_end, &post_all),
                load1_max: cfg.load1_max,
                pre_reached: true,
                workload_reached: true,
                post_reached: true,
                failure: Some(format!("post-trial idle: {e}")),
            }, &hold);
        }
    };
    let post_all = sampler.snapshot();
    sampler.stop();
    macmon_receipt(&MacmonAttempt {
        load1_pre,
        load1_post,
        pre,
        trial,
        post: slice_new(&workload_end, &post_all),
        load1_max: cfg.load1_max,
        pre_reached: true,
        workload_reached: true,
        post_reached: true,
        failure: None,
    }, &hold)
}

pub fn run_until_quiet_macmon(
    lock_path: &Path,
    out_dir: &Path,
    deadline: Duration,
    retry_wait: Duration,
    lock_wait_s: f64,
    cfg: &MacmonConfig,
) -> Result<serde_json::Value, String> {
    let started = std::time::Instant::now();
    let mut first_contended = false;
    let mut n = 0u32;
    let write = |name: &str, body: &serde_json::Value| -> Result<(), String> {
        write_receipt_value(&out_dir.join(name), body)
    };
    loop {
        if started.elapsed() > deadline {
            let body = macmon_receipt(&MacmonAttempt {
                load1_pre: vec![],
                load1_post: vec![],
                pre: vec![],
                trial: vec![],
                post: vec![],
                load1_max: cfg.load1_max,
                pre_reached: false,
                workload_reached: false,
                post_reached: false,
                failure: Some("deadline reached before a quiet attributable run".into()),
            }, &LockHold {
                path: lock_path.display().to_string(),
                pid: std::process::id(),
                acquire_unix_ns: None,
                release_unix_ns: None,
                first_attempt_contended: first_contended,
                contended: false,
                wait_s: 0.0,
            });
            write("refusal-deadline.json", &body)?;
            return Ok(body);
        }
        match poll_host_lock(lock_path, lock_wait_s)? {
            None => {
                first_contended = true;
                let body = macmon_receipt(&MacmonAttempt {
                    load1_pre: vec![],
                    load1_post: vec![],
                    pre: vec![],
                    trial: vec![],
                    post: vec![],
                    load1_max: cfg.load1_max,
                    pre_reached: false,
                    workload_reached: false,
                    post_reached: false,
                    failure: Some(crate::fleet_gate::HOST_LOCK_REFUSAL.into()),
                }, &LockHold {
                    path: lock_path.display().to_string(),
                    pid: std::process::id(),
                    acquire_unix_ns: None,
                    release_unix_ns: None,
                    first_attempt_contended: true,
                    contended: true,
                    wait_s: 0.0,
                });
                write(&format!("refusal-lock-{n}.json"), &body)?;
                n += 1;
                sleep(retry_wait);
            }
            Some(mut guard) => {
                guard.first_attempt_contended = first_contended;
                let mut body = run_locked_macmon(cfg, &guard);
                if let Some(lock) = body.get_mut("lock") {
                    lock["first_attempt_contended"] = serde_json::json!(first_contended);
                    lock["release_unix_ns"] = serde_json::json!(unix_ns());
                }
                drop(guard);
                if body["attributable"].as_bool().unwrap_or(false) {
                    write("attributable.json", &body)?;
                    return Ok(body);
                }
                write(&format!("refusal-{n}.json"), &body)?;
                n += 1;
                sleep(retry_wait);
            }
        }
    }
}

fn write_receipt_value(path: &Path, body: &serde_json::Value) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let text = serde_json::to_string_pretty(body).map_err(|e| e.to_string())?;
    std::fs::write(path, text).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quiet_series() -> EnergySeries {
        let pkg_pre = (0..5).map(|i| (i * 1_000_000_000, 1_000 + i * 100)).collect();
        let pkg_post = (0..5).map(|i| (i * 1_000_000_000, 5_000 + i * 100)).collect();
        EnergySeries {
            load1_pre: vec![0.5, 1.0, 1.5],
            load1_post: vec![0.4, 0.8, 1.2],
            pre_package: pkg_pre,
            post_package: pkg_post,
            pre_core: (0..5).map(|i| (i * 1_000_000_000, 10 + i)).collect(),
            post_core: (0..5).map(|i| (i * 1_000_000_000, 40 + i)).collect(),
            pre_ppt_w: vec![10.0, 10.2, 9.8, 10.1],
            post_ppt_w: vec![10.0, 10.1, 9.9, 10.2],
            trial_package: (1000, 1000 + 5_000_000),
            trial_core: (1000, 1000 + 1_000_000),
            trial_ppt_j: Some(12.5),
            cpu_before: vec![
                CpuTicks { cpu: 0, busy: 0, total: 0 },
                CpuTicks { cpu: 1, busy: 10, total: 100 },
            ],
            cpu_after: vec![
                CpuTicks { cpu: 0, busy: 90, total: 100 },
                CpuTicks { cpu: 1, busy: 12, total: 200 },
            ],
            pinned: vec![0],
            busy_fraction_max: 0.10,
            load1_max: 2.0,
            range_uj: FRAMEWORK_RANGE_UJ,
            cpu_tctl_c: Some(70.0),
            gpu_edge_c: Some(55.0),
            settle_busy: vec![0.01, 0.02],
            workload_busy: vec![0.01],
            tctl_pre: Some(68.0),
            tctl_workload: Some(70.0),
            tctl_post: Some(69.0),
            edge_pre: Some(50.0),
            edge_workload: Some(55.0),
            edge_post: Some(52.0),
            workload_reached: true,
            post_reached: true,
            pre_reached: true,
            settle_load1_pre: vec![],
            settle_load1_post: vec![],
        }
    }

    fn held() -> LockHold {
        LockHold {
            path: LOCK_PATH.into(),
            pid: 7,
            acquire_unix_ns: Some(10),
            release_unix_ns: Some(20),
            first_attempt_contended: false,
            contended: false,
            wait_s: 0.0,
        }
    }

    #[test]
    fn a_quiet_synthetic_run_keeps_meters_separate() {
        let r = judge(&quiet_series(), held());
        assert!(r.attributable, "{}", r.reason);
        assert_eq!(r.rapl_package_j, Some(5.0));
        assert_eq!(r.rapl_core_j, Some(1.0));
        assert_eq!(r.amdgpu_ppt_j, Some(12.5));
        assert_eq!(r.grades.rapl_package, Grade::MeasuredAttributable);
        assert_eq!(r.grades.rapl_core, Grade::MeasuredAttributable);
        assert_eq!(r.grades.amdgpu_ppt, Grade::MeasuredAttributable);
        assert_eq!(r.cpu_tctl_c, Some(70.0));
        assert_eq!(r.gpu_edge_c, Some(55.0));
        let text = receipt_json(&r).to_string();
        assert!(!text.contains("\"meters_pooled\":true"));
    }

    #[test]
    fn load_above_two_refuses_without_joules() {
        let mut s = quiet_series();
        s.load1_pre = vec![0.5, 2.5];
        let r = judge(&s, held());
        assert!(!r.attributable);
        assert!(r.rapl_package_j.is_none());
        assert!(r.rapl_core_j.is_none());
        assert!(r.amdgpu_ppt_j.is_none());
        assert!(r.reason.contains("load1"));
        assert_eq!(r.grades.pre_idle, Grade::Refused);
    }

    #[test]
    fn a_busy_unpinned_core_refuses_without_joules() {
        let mut s = quiet_series();
        s.cpu_after[1].busy = 150;
        let r = judge(&s, held());
        assert!(!r.attributable);
        assert!(r.rapl_package_j.is_none());
        assert!(r.reason.contains("busy fraction"));
        assert!(r.busy_fraction.unwrap() > s.busy_fraction_max);
    }

    #[test]
    fn baseline_drift_beyond_3_sigma_refuses_without_joules() {
        let mut s = quiet_series();
        s.post_package = (0..5).map(|i| (i * 1_000_000_000, 50_000 + i)).collect();
        let r = judge(&s, held());
        assert!(!r.attributable, "{}", r.reason);
        assert!(r.reason.contains("3σ") || r.reason.contains("drift"));
        assert!(r.rapl_package_j.is_none());
    }

    #[test]
    fn the_framework_counter_wraps_by_its_published_modulus() {
        let range = FRAMEWORK_RANGE_UJ;
        let start = range - 100;
        let end = 50;
        let (d, wrapped) = delta_uj(start, end, range).unwrap();
        assert!(wrapped);
        assert_eq!(d, 150);
        let mut s = quiet_series();
        s.trial_package = (start, end);
        s.range_uj = range;
        let r = judge(&s, held());
        assert!(r.attributable, "{}", r.reason);
        assert_eq!(r.rapl_package_j, Some(150.0 * 1e-6));
    }

    #[test]
    fn a_contended_lock_writes_no_energy() {
        let r = judge(
            &quiet_series(),
            LockHold {
                path: LOCK_PATH.into(),
                pid: 1,
                acquire_unix_ns: None,
                release_unix_ns: None,
                first_attempt_contended: true,
                contended: true,
                wait_s: 0.0,
            },
        );
        assert!(!r.attributable);
        assert!(r.rapl_package_j.is_none());
        assert!(r.lock.contended);
        assert!(r.lock.acquire_unix_ns.is_none());
    }

    #[test]
    fn two_processes_cannot_hold_the_energy_lock() {
        let path = std::env::temp_dir().join(format!("bmi-energy-test-{}-{}.lock", std::process::id(), line!()));
        let first = try_lock_exclusive(&path).unwrap().expect("first acquire");
        let second = try_lock_exclusive(&path).unwrap();
        assert!(second.is_none());
        drop(first);
        let third = try_lock_exclusive(&path).unwrap();
        assert!(third.is_some());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn pinned_argv_puts_taskset_in_front_of_the_workload() {
        let argv = pinned_argv(&[0, 2], &["ferric-smolvla".into(), "gpu".into()]);
        assert_eq!(argv, vec!["taskset", "-c", "0,2", "ferric-smolvla", "gpu"]);
    }

    #[test]
    fn hwmon_matches_the_label_not_the_index() {
        let root = std::env::temp_dir().join(format!("hwmon-test-{}", std::process::id()));
        let chip = root.join("hwmon3");
        std::fs::create_dir_all(&chip).unwrap();
        std::fs::write(chip.join("name"), "amdgpu\n").unwrap();
        std::fs::write(chip.join("power1_label"), "PPT\n").unwrap();
        std::fs::write(chip.join("power1_input"), "1\n").unwrap();
        std::fs::write(chip.join("power2_label"), "other\n").unwrap();
        std::fs::write(chip.join("power2_input"), "2\n").unwrap();
        std::fs::write(chip.join("temp2_label"), "edge\n").unwrap();
        std::fs::write(chip.join("temp2_input"), "3\n").unwrap();
        let ppt = crate::hwmon_input(&root, "amdgpu", "power", "PPT").unwrap();
        assert!(ppt.ends_with("power1_input"));
        let edge = crate::hwmon_input(&root, "amdgpu", "temp", "edge").unwrap();
        assert!(edge.ends_with("temp2_input"));
        assert!(crate::hwmon_input(&root, "amdgpu", "temp", "temp1").is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn settle_above_the_limit_blocks_the_workload() {
        assert!(pre_idle_gate(&[0.4, 2.5], &[0.01], 2.0, 0.10).unwrap_err().contains("load1"));
        assert!(pre_idle_gate(&[0.4, 1.0], &[0.01, 0.40], 2.0, 0.10).unwrap_err().contains("busy fraction"));
        assert!(pre_idle_gate(&[0.4, 1.0], &[0.01, 0.05], 2.0, 0.10).is_ok());
    }

    #[test]
    fn a_workload_abort_keeps_settle_evidence_and_nulls_joules() {
        let mut s = quiet_series();
        s.post_reached = false;
        s.load1_post.clear();
        s.post_package.clear();
        s.post_core.clear();
        s.post_ppt_w.clear();
        s.tctl_post = None;
        s.edge_post = None;
        s.workload_reached = true;
        s.workload_busy = vec![0.02, 0.40];
        s.tctl_workload = Some(72.5);
        s.edge_workload = Some(61.0);
        s.trial_ppt_j = None;
        let r = refused(
            "non-pinned busy fraction 0.4 exceeds the predeclared 0.1".into(),
            held(),
            &s,
            Some(0.4),
        );
        assert!(!r.attributable);
        assert!(r.rapl_package_j.is_none());
        assert!(r.rapl_core_j.is_none());
        assert!(r.amdgpu_ppt_j.is_none());
        let j = receipt_json(&r);
        assert!(j["joules"].is_null());
        assert_eq!(j["phases"]["pre_idle"]["reached"], true);
        assert_eq!(j["phases"]["pre_idle"]["load1"], serde_json::json!([0.5, 1.0, 1.5]));
        assert_eq!(j["phases"]["pre_idle"]["load1_max"], serde_json::json!(1.5));
        assert_eq!(j["phases"]["pre_idle"]["busy_fraction"], serde_json::json!([0.01, 0.02]));
        assert_eq!(j["phases"]["pre_idle"]["cpu_tctl_max_c"], serde_json::json!(68.0));
        assert_eq!(j["phases"]["pre_idle"]["gpu_edge_max_c"], serde_json::json!(50.0));
        let pkg = j["phases"]["pre_idle"]["baseline_power_w"]["rapl_package"].as_array().unwrap();
        assert!(!pkg.is_empty());
        assert_eq!(j["phases"]["workload"]["reached"], true);
        assert_eq!(j["phases"]["workload"]["busy_fraction"], serde_json::json!([0.02, 0.40]));
        assert_eq!(j["phases"]["workload"]["cpu_tctl_max_c"], serde_json::json!(72.5));
        assert_eq!(j["phases"]["workload"]["gpu_edge_max_c"], serde_json::json!(61.0));
        assert_eq!(j["phases"]["post_idle"]["reached"], false);
        assert_eq!(j["phases"]["post_idle"]["cpu_tctl_max_c"], serde_json::Value::Null);
        assert_eq!(j["drift"]["status"], "not_reached");
        assert_eq!(j["drift"]["phase"], "post_idle");
    }

    #[test]
    fn a_settle_refusal_names_the_phase_that_was_not_reached() {
        let mut s = quiet_series();
        s.load1_pre = vec![0.2, 3.0];
        s.settle_busy = vec![0.04];
        s.workload_reached = false;
        s.post_reached = false;
        s.workload_busy.clear();
        s.tctl_workload = None;
        s.edge_workload = None;
        s.tctl_post = None;
        s.edge_post = None;
        let r = refused("pre-trial idle: load1 3 exceeds 2".into(), held(), &s, None);
        let j = receipt_json(&r);
        assert!(j["joules"].is_null());
        assert_eq!(j["phases"]["pre_idle"]["load1_max"], serde_json::json!(3.0));
        assert_eq!(j["phases"]["workload"]["reached"], false);
        assert_eq!(j["drift"]["status"], "not_reached");
        assert_eq!(j["drift"]["phase"], "workload");
        assert_eq!(r.grades.pre_idle, Grade::Refused);
    }

    #[test]
    fn both_brackets_compute_drift_on_a_refusal() {
        let mut s = quiet_series();
        s.load1_post = vec![4.0];
        let r = judge(&s, held());
        assert!(!r.attributable);
        assert!(r.rapl_package_j.is_none());
        let j = receipt_json(&r);
        assert_eq!(j["drift"]["status"], "computed");
        assert!(j["joules"].is_null());
    }

    #[test]
    fn an_unpinned_smt_sibling_is_refused() {
        let siblings = |cpu: usize| {
            if cpu < 16 { vec![cpu, cpu + 16] } else { vec![cpu - 16, cpu] }
        };
        let missing = unpinned_siblings(&[0, 1, 2, 3], siblings);
        assert_eq!(missing, vec![(0, 16), (1, 17), (2, 18), (3, 19)]);
        assert!(unpinned_siblings(&[0, 1, 2, 3, 16, 17, 18, 19], siblings).is_empty());
        assert_eq!(parse_cpu_list("0-3,16-19").unwrap(), vec![0, 1, 2, 3, 16, 17, 18, 19]);
    }

    fn mac_reading() -> crate::macmon::Reading {
        crate::macmon::parse_line(
            r#"{"cpu_power":0.20486385,"gpu_power":0.017451683,"ane_power":0.0,"sys_power":5.876533,"chip_power":1.4130547,"temp":{"cpu_temp_avg":43.73614,"gpu_temp_avg":36.95167}}"#,
        )
        .unwrap()
    }

    #[test]
    fn a_quiet_macmon_attempt_keeps_four_meters_and_nulls_joules_when_load_is_high() {
        let r = mac_reading();
        let pre = vec![(0, r.clone()), (1_000_000_000, r.clone())];
        let trial = vec![(2_000_000_000, r.clone()), (3_000_000_000, r.clone())];
        let post = vec![(4_000_000_000, r.clone()), (5_000_000_000, r)];
        let quiet = macmon_receipt(&MacmonAttempt {
            load1_pre: vec![0.4, 0.5],
            load1_post: vec![0.3, 0.6],
            pre,
            trial,
            post,
            load1_max: 2.0,
            pre_reached: true,
            workload_reached: true,
            post_reached: true,
            failure: None,
        }, &held());
        assert_eq!(quiet["attributable"], true);
        assert_eq!(quiet["summed"], false);
        assert_eq!(quiet["pinning"], "no core pinning was applied");
        assert!(quiet["joules"]["ram_power"].is_null());
        assert!(quiet["joules"]["soc"].is_null());
        assert_eq!(quiet["joules"]["dram_scope"], "unavailable");
        assert_eq!(quiet["gate_sample_hz"], 1);
        assert_eq!(quiet["busy_fraction_status"], "unavailable");
        assert!(quiet["busy_fraction"].is_null());
        assert!(quiet["declared_before_run"] == true);
        assert!((quiet["joules"]["sys_power"].as_f64().unwrap() - 5.876533).abs() < 1e-6);
        assert!((quiet["joules"]["cpu_power"].as_f64().unwrap() - 0.20486385).abs() < 1e-6);
        let summed = quiet["joules"]["sys_power"].as_f64().unwrap()
            + quiet["joules"]["cpu_power"].as_f64().unwrap()
            + quiet["joules"]["gpu_power"].as_f64().unwrap()
            + quiet["joules"]["ane_power"].as_f64().unwrap();
        assert!(summed > quiet["joules"]["sys_power"].as_f64().unwrap());
        let mut busy = quiet.clone();
        busy["load1_pre"] = serde_json::json!([3.0]);
        let refused = macmon_receipt(&MacmonAttempt {
            load1_pre: vec![0.2, 3.0],
            load1_post: vec![],
            pre: vec![],
            trial: vec![],
            post: vec![],
            load1_max: 2.0,
            pre_reached: true,
            workload_reached: false,
            post_reached: false,
            failure: Some("pre-trial idle: load1 3 exceeds 2".into()),
        }, &held());
        assert_eq!(refused["attributable"], false);
        assert!(refused["joules"].is_null());
        assert_eq!(refused["phases"]["pre_idle"]["load1"], serde_json::json!([0.2, 3.0]));
        assert_eq!(refused["pinning"], "no core pinning was applied");
        assert_eq!(refused["busy_fraction_status"], "unavailable");
        assert_eq!(refused["drift"]["phase"], "workload");
    }
}
