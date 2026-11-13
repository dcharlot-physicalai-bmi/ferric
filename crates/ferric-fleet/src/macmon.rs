//! macmon `pipe` samples.
//!
//! The field names are the ones `macmon` serializes from `Metrics` on current
//! main: `cpu_power`, `gpu_power`, `ane_power`, `sys_power`, `chip_power`,
//! `ram_power`, and `temp.cpu_temp_avg` / `temp.gpu_temp_avg`. A line that
//! only has the grouped `power` object from the unreleased schema is accepted
//! as a fallback (`power.cpu`, `power.gpu`, `power.ane`, `power.board`).
//!
//! `chip_power` is at least CPU + GPU + ANE + DRAM, and `sys_power` is at
//! least `chip_power`. Those domains overlap. This module never adds them.

use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use crate::meter::trapezoid_joules;

pub const PIPE_SAMPLES_FOREVER: &str = "0";

pub const PINNING: &str = "no core pinning was applied";
pub const PINNING_REASON: &str = "no core pinning was applied";

pub const BUSY_FRACTION_STATUS: &str = "unavailable";
pub const BUSY_FRACTION_REASON: &str = "per-core busy fraction via host_processor_info is not collected; this build does not call mach APIs. The quiet gate is load1, from sysctl vm.loadavg or /proc/loadavg, plus the idle sys_power baseline. This is fixed before the run.";

#[derive(Debug, Clone, PartialEq)]
pub struct Reading {
    pub cpu_w: f64,
    pub gpu_w: f64,
    pub ane_w: f64,
    pub sys_w: Option<f64>,
    pub chip_w: Option<f64>,
    pub ram_w: Option<f64>,
    pub gpu_ram_w: Option<f64>,
    pub dram_unavailable: bool,
    pub cpu_temp_c: Option<f64>,
    pub gpu_temp_c: Option<f64>,
    pub pcpu_mhz: Option<f64>,
    pub ecpu_mhz: Option<f64>,
    pub gpu_mhz: Option<f64>,
    pub cpu_active: Option<f64>,
    pub gpu_active: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Joules {
    pub cpu: Option<f64>,
    pub gpu: Option<f64>,
    pub ane: Option<f64>,
    pub sys: Option<f64>,
    pub ram: Option<f64>,
    pub gpu_ram: Option<f64>,
    pub soc: Option<f64>,
    pub accelerator: Option<f64>,
    pub dram_scope: &'static str,
    pub soc_label: &'static str,
    pub accelerator_label: &'static str,
}

pub fn pipe_args(interval_ms: u64) -> Vec<String> {
    vec![
        "pipe".into(),
        "-i".into(),
        interval_ms.to_string(),
        "-s".into(),
        PIPE_SAMPLES_FOREVER.into(),
    ]
}

fn finite(v: f64) -> Option<f64> {
    v.is_finite().then_some(v)
}

/// One `macmon pipe` line, parsed by [`ferric_joule::parse_macmon_line`].
pub fn parse_line(line: &str) -> Result<Reading, String> {
    let s = ferric_joule::parse_macmon_line(line).ok_or("macmon line has no cpu_power/gpu_power")?;
    Ok(Reading {
        cpu_w: s.cpu,
        gpu_w: s.gpu,
        ane_w: s.ane,
        sys_w: finite(s.sys),
        chip_w: None,
        ram_w: (s.ram != 0.0).then_some(s.ram),
        gpu_ram_w: (s.gpu_ram.is_finite() && s.gpu_ram != 0.0).then_some(s.gpu_ram),
        dram_unavailable: s.dram_unavailable,
        cpu_temp_c: finite(s.cpu_temp),
        gpu_temp_c: finite(s.gpu_temp),
        pcpu_mhz: finite(s.pcpu_mhz),
        ecpu_mhz: finite(s.ecpu_mhz),
        gpu_mhz: finite(s.gpu_mhz),
        cpu_active: finite(s.cpu_active),
        gpu_active: finite(s.gpu_active),
    })
}

fn all_zero(samples: &[(u64, Reading)], f: fn(&Reading) -> Option<f64>) -> bool {
    samples.iter().all(|(_, r)| f(r).unwrap_or(0.0) == 0.0)
}

pub fn integrate(samples: &[(u64, Reading)]) -> Joules {
    let series = |f: fn(&Reading) -> Option<f64>| {
        let pts: Vec<(u64, f64)> = samples.iter().filter_map(|(t, r)| f(r).map(|w| (*t, w))).collect();
        trapezoid_joules(&pts)
    };
    let ram_missing = all_zero(samples, |r| r.ram_w);
    let gpu_ram_missing = all_zero(samples, |r| r.gpu_ram_w);
    let cpu = series(|r| Some(r.cpu_w));
    let gpu = series(|r| Some(r.gpu_w));
    let ane = series(|r| Some(r.ane_w));
    let ram = if ram_missing { None } else { series(|r| r.ram_w) };
    let (soc, soc_label) = if ram_missing {
        (None, "refused: DRAM read exactly 0, so a SoC total would exclude DRAM")
    } else {
        (Some(cpu.unwrap_or(0.0) + gpu.unwrap_or(0.0) + ane.unwrap_or(0.0) + ram.unwrap_or(0.0)), "cpu+gpu+ane+ram")
    };
    let (accelerator, accelerator_label) = if ram_missing {
        (None, "refused: DRAM read exactly 0, so an accelerator total would exclude DRAM")
    } else {
        (Some(gpu.unwrap_or(0.0) + ram.unwrap_or(0.0)), "gpu+ram")
    };
    Joules {
        cpu,
        gpu,
        ane,
        sys: series(|r| r.sys_w),
        ram,
        gpu_ram: if gpu_ram_missing { None } else { series(|r| r.gpu_ram_w) },
        soc,
        accelerator,
        dram_scope: if ram_missing && gpu_ram_missing { "unavailable" } else { "reported" },
        soc_label,
        accelerator_label,
    }
}

pub fn context_means(samples: &[(u64, Reading)]) -> serde_json::Value {
    let mean = |f: fn(&Reading) -> Option<f64>| {
        let v: Vec<f64> = samples.iter().filter_map(|(_, r)| f(r)).collect();
        if v.is_empty() { None } else { Some(v.iter().sum::<f64>() / v.len() as f64) }
    };
    serde_json::json!({
        "cpu_temp_avg_c": mean(|r| r.cpu_temp_c),
        "gpu_temp_avg_c": mean(|r| r.gpu_temp_c),
        "pcpu_freq_mhz": mean(|r| r.pcpu_mhz),
        "ecpu_freq_mhz": mean(|r| r.ecpu_mhz),
        "gpu_freq_mhz": mean(|r| r.gpu_mhz),
        "cpu_active_ratio": mean(|r| r.cpu_active),
        "gpu_active_ratio": mean(|r| r.gpu_active),
    })
}

pub fn mean_power(joules: Option<f64>, wall_s: f64) -> Option<f64> {
    if wall_s <= 0.0 {
        return None;
    }
    joules.map(|j| j / wall_s)
}

/// Trial joules minus the idle baseline power times the trial duration.
pub fn incremental_joules(trial_j: f64, trial_wall_s: f64, baseline_w: f64) -> f64 {
    trial_j - baseline_w * trial_wall_s
}

/// Overlaps taken from macmon's own accounting, not measured on a machine.
pub fn overlaps() -> serde_json::Value {
    serde_json::json!([
        {
            "domains": ["sys_power", "chip_power", "cpu_power", "gpu_power", "ane_power", "ram_power"],
            "reason": "sys_power is an SMC system estimate and is at least chip_power"
        },
        {
            "domains": ["chip_power", "cpu_power", "gpu_power", "ane_power", "ram_power", "dcs_power", "media_power"],
            "reason": "chip_power is the chip total and is at least the sum of CPU, GPU, ANE, DRAM, the memory controller, and media when those exist"
        },
        {
            "domains": ["all_power", "chip_power"],
            "reason": "all_power is the deprecated name of chip_power"
        },
        {
            "domains": ["gpu_power", "gpu_ram_power"],
            "reason": "gpu_ram_power is deprecated; gpu_power includes GPU SRAM on some chips"
        }
    ])
}

pub fn parse_proc_load1(text: &str) -> Result<f64, String> {
    text.split_whitespace()
        .next()
        .ok_or("loadavg empty")?
        .parse::<f64>()
        .map_err(|e| e.to_string())
}

/// `sysctl -n vm.loadavg` prints `{ 1.23 0.45 0.20 }`.
pub fn parse_sysctl_loadavg(text: &str) -> Result<f64, String> {
    ferric_joule::parse_sysctl_loadavg(text).ok_or_else(|| "sysctl vm.loadavg empty".into())
}

pub struct Sampler {
    child: std::process::Child,
    samples: Arc<Mutex<Vec<(u64, Reading)>>>,
    stop: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl Sampler {
    pub fn spawn(bin: &Path, interval_ms: u64) -> Result<Self, String> {
        let args = pipe_args(interval_ms);
        let mut child = Command::new(bin)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("spawn {} pipe: {e}", bin.display()))?;
        let stdout = child.stdout.take().ok_or("macmon stdout was not piped")?;
        let samples = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (samples2, stop2) = (samples.clone(), stop.clone());
        let origin = Instant::now();
        let join = std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                if stop2.load(Ordering::Relaxed) {
                    break;
                }
                let Ok(line) = line else { break };
                let t_ns = origin.elapsed().as_nanos() as u64;
                if let Ok(reading) = parse_line(&line) {
                    samples2.lock().unwrap().push((t_ns, reading));
                }
            }
        });
        Ok(Self { child, samples, stop, join: Some(join) })
    }

    pub fn snapshot(&self) -> Vec<(u64, Reading)> {
        self.samples.lock().unwrap().clone()
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const README_LINE: &str = r#"{"timestamp":"2025-02-24T20:38:15.427569+00:00","temp":{"cpu_temp_avg":43.73614,"gpu_temp_avg":36.95167},"cpu_power":0.20486385,"gpu_power":0.017451683,"ane_power":0.0,"chip_power":1.4130547,"all_power":1.4130547,"sys_power":5.876533,"ram_power":0.11635789,"gpu_ram_power":0.0009615385}"#;

    const PR59_LINE: &str = r#"{"all_power":1.1913399696350098,"ane_power":0.0,"cpu_power":0.8916881680488586,"gpu_power":0.2996518015861511,"gpu_ram_power":0.0,"ram_power":0.25657692551612854,"sys_power":16.489110946655273,"temp":{"cpu_temp_avg":48.95563888549805,"gpu_temp_avg":44.01784133911133},"timestamp":"2026-03-29T20:59:59.337500+00:00"}"#;

    const GROUPED_LINE: &str = r#"{"power":{"package":1.399669,"cpu":0.7379913,"gpu":0.37346822,"ram":0.2882096,"gpu_ram":0.0,"ane":0.0,"board":10.063049},"temp":{"cpu_temp_avg":40.0,"gpu_temp_avg":38.0}}"#;

    #[test]
    fn the_readme_line_parses_the_four_meters_and_temperatures() {
        let r = parse_line(README_LINE).unwrap();
        assert!((r.cpu_w - 0.20486385).abs() < 1e-9);
        assert!((r.gpu_w - 0.017451683).abs() < 1e-9);
        assert_eq!(r.ane_w, 0.0);
        assert!((r.sys_w.unwrap() - 5.876533).abs() < 1e-9);
        assert!((r.cpu_temp_c.unwrap() - 43.73614).abs() < 1e-5);
        assert!((r.gpu_temp_c.unwrap() - 36.95167).abs() < 1e-5);
    }

    #[test]
    fn the_pr59_line_parses_and_a_grouped_object_is_not_a_second_parser() {
        let r = parse_line(PR59_LINE).unwrap();
        assert!((r.cpu_w - 0.8916881680488586).abs() < 1e-12);
        assert!((r.sys_w.unwrap() - 16.489110946655273).abs() < 1e-12);
        assert!((r.ram_w.unwrap() - 0.25657692551612854).abs() < 1e-12);
        assert!(r.gpu_ram_w.is_none(), "an exact 0 gpu_ram_power is unavailable");
        assert!(parse_line(GROUPED_LINE).is_err());
    }

    #[test]
    fn exact_zero_dram_is_unavailable_and_soc_is_refused() {
        let line = r#"{"cpu_power":12.0,"gpu_power":30.0,"ane_power":0.5,"ram_power":0.0,"gpu_ram_power":0.0,"sys_power":65.99,"all_power":65.99,"pcpu_freq_mhz":4000.0,"ecpu_freq_mhz":1100.0,"gpu_freq_mhz":900.0,"cpu_active_ratio":0.4,"gpu_active_ratio":0.8,"temp":{"cpu_temp_avg":70.0,"gpu_temp_avg":68.0}}"#;
        let r = parse_line(line).unwrap();
        assert!(r.dram_unavailable);
        assert!(r.ram_w.is_none());
        assert!((r.pcpu_mhz.unwrap() - 4000.0).abs() < 1e-9);
        let j = integrate(&[(0, r.clone()), (200_000_000, r)]);
        assert!(j.ram.is_none());
        assert!(j.gpu_ram.is_none());
        assert!(j.soc.is_none());
        assert!(j.accelerator.is_none());
        assert_eq!(j.dram_scope, "unavailable");
        assert!((j.sys.unwrap() - 65.99 * 0.2).abs() < 1e-6);
    }

    #[test]
    fn a_constant_readme_sample_integrates_to_its_watts_over_one_second() {
        let r = parse_line(README_LINE).unwrap();
        let j = integrate(&[(0, r.clone()), (1_000_000_000, r.clone())]);
        assert!((j.cpu.unwrap() - r.cpu_w).abs() < 1e-9);
        assert!((j.gpu.unwrap() - r.gpu_w).abs() < 1e-9);
        assert!((j.ane.unwrap() - r.ane_w).abs() < 1e-12);
        assert!((j.sys.unwrap() - r.sys_w.unwrap()).abs() < 1e-9);
        let sum = j.cpu.unwrap() + j.gpu.unwrap() + j.ane.unwrap() + j.sys.unwrap();
        assert!(sum > j.sys.unwrap(), "the four meters must not be treated as one total");
    }

    #[test]
    fn overlaps_name_sys_and_chip_and_the_pipe_flags_are_fixed() {
        let text = overlaps().to_string();
        assert!(text.contains("sys_power"));
        assert!(text.contains("chip_power"));
        assert!(text.contains("gpu_ram_power"));
        assert_eq!(pipe_args(1000), vec!["pipe", "-i", "1000", "-s", "0"]);
    }

    #[test]
    fn sysctl_and_proc_load1_parsers_take_the_first_number() {
        assert!((parse_proc_load1("1.25 0.80 0.40 1/200 9\n").unwrap() - 1.25).abs() < 1e-12);
        assert!((parse_sysctl_loadavg("{ 2.50 1.10 0.90 }\n").unwrap() - 2.50).abs() < 1e-12);
        assert!((parse_sysctl_loadavg("{ 3.84 3.09 3.44 }").unwrap() - 3.84).abs() < 1e-12);
    }
}
