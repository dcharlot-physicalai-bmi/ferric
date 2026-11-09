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
}

#[derive(Debug, Clone)]
pub struct LockHold {
    pub path: String,
    pub pid: u32,
    pub acquire_unix_ns: Option<u64>,
    pub release_unix_ns: Option<u64>,
    pub first_attempt_contended: bool,
    pub contended: bool,
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

fn refused(reason: String, lock: LockHold, series: &EnergySeries, busy: Option<f64>) -> DriveReceipt {
    let g = Grade::Refused;
    DriveReceipt {
        attributable: false,
        reason,
        grades: Grades {
            pre_idle: g,
            workload: g,
            post_idle: g,
            rapl_package: g,
            rapl_core: g,
            amdgpu_ppt: g,
        },
        lock,
        busy_fraction_max: series.busy_fraction_max,
        busy_fraction: busy,
        cpu_tctl_c: series.cpu_tctl_c,
        gpu_edge_c: series.gpu_edge_c,
        rapl_package_j: None,
        rapl_core_j: None,
        amdgpu_ppt_j: None,
    }
}

/// Grade one captured attempt. Joules are present only when every gate passes.
pub fn judge(series: &EnergySeries, lock: LockHold) -> DriveReceipt {
    if lock.contended || lock.acquire_unix_ns.is_none() {
        return refused("lock contended; counters were not read".into(), lock, series, None);
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
        cpu_tctl_c: series.cpu_tctl_c,
        gpu_edge_c: series.gpu_edge_c,
        rapl_package_j: Some(pkg_du as f64 * 1e-6),
        rapl_core_j: Some(core_du as f64 * 1e-6),
        amdgpu_ppt_j: series.trial_ppt_j,
    }
}

pub fn receipt_json(r: &DriveReceipt) -> serde_json::Value {
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
        Ok(()) => Ok(Some(LockGuard {
            file,
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
pub fn run_until_quiet<F>(
    lock_path: &Path,
    out_dir: &Path,
    deadline: Duration,
    retry_wait: Duration,
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
            };
            let receipt = refused("deadline reached before a quiet attributable run".into(), lock, &series, None);
            write_receipt(&out_dir.join(format!("refusal-deadline.json")), &receipt)?;
            return Ok(receipt);
        }
        match try_lock_exclusive(lock_path)? {
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
                };
                let receipt = refused("lock contended; counters were not read".into(), lock, &series, None);
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
    }
}

pub fn read_load1() -> Result<f64, String> {
    let text = std::fs::read_to_string("/proc/loadavg").map_err(|e| e.to_string())?;
    text.split_whitespace().next().ok_or("loadavg empty")?.parse::<f64>().map_err(|e| e.to_string())
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
}

fn take_bracket(cfg: &LiveConfig, origin: std::time::Instant) -> Result<Bracket, String> {
    let steps = cfg.settle_s.round().max(1.0) as u32;
    let mut bracket = Bracket { load1: vec![], package: vec![], core: vec![], ppt_w: vec![], cpu_tctl_c: None, gpu_edge_c: None };
    for i in 0..steps {
        bracket.load1.push(read_load1()?);
        let t_ns = origin.elapsed().as_nanos() as u64;
        bracket.package.push((t_ns, read_uj(&cfg.rapl_package)?));
        bracket.core.push((t_ns, read_uj(&cfg.rapl_core)?));
        if let Some(path) = &cfg.ppt {
            if let Ok(uw) = read_uj(path) {
                bracket.ppt_w.push(uw as f64 * 1e-6);
            }
        }
        if let Some(c) = millic_to_c(&cfg.cpu_tctl) {
            bracket.cpu_tctl_c = Some(bracket.cpu_tctl_c.map(|m| m.max(c)).unwrap_or(c));
        }
        if let Some(c) = millic_to_c(&cfg.gpu_edge) {
            bracket.gpu_edge_c = Some(bracket.gpu_edge_c.map(|m| m.max(c)).unwrap_or(c));
        }
        if i + 1 < steps {
            sleep(Duration::from_secs(1));
        }
    }
    Ok(bracket)
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
    let origin = std::time::Instant::now();
    let pre = match take_bracket(cfg, origin) {
        Ok(b) => b,
        Err(e) => return refusal_now(lock, cfg, e),
    };
    let ticks0 = match read_cpu_ticks() {
        Ok(v) => v,
        Err(e) => return refusal_now(lock, cfg, e),
    };
    let pkg0 = match read_uj(&cfg.rapl_package) {
        Ok(v) => v,
        Err(e) => return refusal_now(lock, cfg, e),
    };
    let core0 = match read_uj(&cfg.rapl_core) {
        Ok(v) => v,
        Err(e) => return refusal_now(lock, cfg, e),
    };
    let mut child = match spawn_pinned(&cfg.pin_cores, &cfg.workload) {
        Ok(c) => c,
        Err(e) => return refusal_now(lock, cfg, e),
    };
    let mut ppt_series: Vec<(u64, f64)> = Vec::new();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    return refusal_now(lock, cfg, format!("workload exit {status}"));
                }
                break;
            }
            Ok(None) => {
                sleep(Duration::from_secs(1));
                let t_ns = origin.elapsed().as_nanos() as u64;
                if let Some(path) = &cfg.ppt {
                    if let Ok(uw) = read_uj(path) {
                        ppt_series.push((t_ns, uw as f64 * 1e-6));
                    }
                }
                if let Ok(now_ticks) = read_cpu_ticks() {
                    if let Ok(busy) = busy_fraction(&ticks0, &now_ticks, &cfg.pin_cores) {
                        if busy.fraction > cfg.busy_fraction_max {
                            let _ = child.kill();
                            let _ = child.wait();
                            let mut series = empty_series();
                            series.busy_fraction_max = cfg.busy_fraction_max;
                            series.load1_max = cfg.load1_max;
                            series.cpu_tctl_c = millic_to_c(&cfg.cpu_tctl);
                            series.gpu_edge_c = millic_to_c(&cfg.gpu_edge);
                            series.pinned = cfg.pin_cores.clone();
                            return refused(
                                format!("non-pinned busy fraction {} exceeds the predeclared {}", busy.fraction, cfg.busy_fraction_max),
                                lock.hold(),
                                &series,
                                Some(busy.fraction),
                            );
                        }
                    }
                }
            }
            Err(e) => return refusal_now(lock, cfg, e.to_string()),
        }
    }
    let pkg1 = match read_uj(&cfg.rapl_package) {
        Ok(v) => v,
        Err(e) => return refusal_now(lock, cfg, e),
    };
    let core1 = match read_uj(&cfg.rapl_core) {
        Ok(v) => v,
        Err(e) => return refusal_now(lock, cfg, e),
    };
    let ticks1 = match read_cpu_ticks() {
        Ok(v) => v,
        Err(e) => return refusal_now(lock, cfg, e),
    };
    let post = match take_bracket(cfg, origin) {
        Ok(b) => b,
        Err(e) => return refusal_now(lock, cfg, e),
    };
    let series = EnergySeries {
        load1_pre: pre.load1,
        load1_post: post.load1,
        pre_package: pre.package,
        post_package: post.package,
        pre_core: pre.core,
        post_core: post.core,
        pre_ppt_w: pre.ppt_w,
        post_ppt_w: post.ppt_w,
        trial_package: (pkg0, pkg1),
        trial_core: (core0, core1),
        trial_ppt_j: crate::trapezoid_joules(&ppt_series),
        cpu_before: ticks0,
        cpu_after: ticks1,
        pinned: cfg.pin_cores.clone(),
        busy_fraction_max: cfg.busy_fraction_max,
        load1_max: cfg.load1_max,
        range_uj: range,
        cpu_tctl_c: max_temp(pre.cpu_tctl_c, post.cpu_tctl_c),
        gpu_edge_c: max_temp(pre.gpu_edge_c, post.gpu_edge_c),
    };
    judge(&series, lock.hold())
}

fn max_temp(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

fn refusal_now(lock: &LockGuard, cfg: &LiveConfig, reason: String) -> DriveReceipt {
    let mut series = empty_series();
    series.busy_fraction_max = cfg.busy_fraction_max;
    series.load1_max = cfg.load1_max;
    series.cpu_tctl_c = millic_to_c(&cfg.cpu_tctl);
    series.gpu_edge_c = millic_to_c(&cfg.gpu_edge);
    series.pinned = cfg.pin_cores.clone();
    refused(reason, lock.hold(), &series, None)
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
}
