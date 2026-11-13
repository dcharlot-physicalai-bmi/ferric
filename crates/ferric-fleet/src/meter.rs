//! Meter rules from the fleet notes. Counters stay in their own domain.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    RaplPackage,
    RaplCore,
    RocmPpt,
    MacmonGpu,
    MacmonDram,
    MacmonSys,
    MacmonCpu,
    MacmonAne,
    MacmonGpuRam,
}

impl Domain {
    pub fn label(self) -> &'static str {
        match self {
            Domain::RaplPackage => "rapl-package",
            Domain::RaplCore => "rapl-core",
            Domain::RocmPpt => "rocm-ppt",
            Domain::MacmonGpu => "macmon-gpu",
            Domain::MacmonDram => "macmon-dram",
            Domain::MacmonSys => "macmon-sys",
            Domain::MacmonCpu => "macmon-cpu",
            Domain::MacmonAne => "macmon-ane",
            Domain::MacmonGpuRam => "macmon-gpu-ram",
        }
    }
    pub fn boundary(self) -> &'static str {
        match self {
            Domain::RaplPackage => "RAPL package energy_uj, not the GPU and not the core domain",
            Domain::RaplCore => "RAPL core energy_uj, kept separate from package",
            Domain::RocmPpt => "ROCm-SMI PPT socket figure; overlap with the APU package is not ruled out",
            Domain::MacmonGpu => "macmon gpu_power",
            Domain::MacmonSys => "macmon sys_power, SoC telemetry, not wall power",
            Domain::MacmonDram => "macmon ram_power; an exact 0 is unavailable, not a measured zero",
            Domain::MacmonCpu => "macmon cpu_power",
            Domain::MacmonAne => "macmon ane_power",
            Domain::MacmonGpuRam => "macmon gpu_ram_power; an exact 0 is unavailable, not a measured zero",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sample {
    pub domain: Domain,
    /// Monotonic host time, nanoseconds. Not a wall clock.
    pub t_monotonic_ns: u64,
    /// Raw counter in microjoules. `None` for a power sample.
    pub counter_uj: Option<u64>,
    /// Wrap modulus in microjoules. Required when the counter goes backwards.
    pub range_uj: u64,
    /// Instantaneous watts. Used for macmon. Empty means this row is a counter.
    pub watts: Option<f64>,
}

/// Trapezoid integral of a power series `(t_ns, watts)` in joules.
/// Fewer than two samples is not an integral.
pub fn trapezoid_joules(series: &[(u64, f64)]) -> Option<f64> {
    if series.len() < 2 {
        return None;
    }
    Some(series.windows(2).map(|p| (p[1].0 - p[0].0) as f64 * 1e-9 * (p[0].1 + p[1].1) / 2.0).sum())
}

/// Find `{prefix}{n}_input` whose `{prefix}{n}_label` equals `label`, under a hwmon
/// chip whose `name` file equals `chip`. The channel index is never assumed.
pub fn hwmon_input(sys_class_hwmon: &std::path::Path, chip: &str, prefix: &str, label: &str) -> Option<std::path::PathBuf> {
    let mut dirs: Vec<_> = std::fs::read_dir(sys_class_hwmon).ok()?.flatten().map(|e| e.path()).collect();
    dirs.sort();
    for dir in dirs {
        let Ok(name) = std::fs::read_to_string(dir.join("name")) else { continue };
        if name.trim() != chip {
            continue;
        }
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for entry in rd.flatten() {
            let path = entry.path();
            let Some(file) = path.file_name().map(|s| s.to_string_lossy().into_owned()) else { continue };
            if !file.starts_with(prefix) || !file.ends_with("_label") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            if text.trim() != label {
                continue;
            }
            let input = dir.join(file.replace("_label", "_input"));
            if input.is_file() {
                return Some(input);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trapezoid_matches_a_two_point_rectangle() {
        let j = trapezoid_joules(&[(0, 10.0), (1_000_000_000, 10.0)]).unwrap();
        assert!((j - 10.0).abs() < 1e-9, "{j}");
        assert!(trapezoid_joules(&[(0, 1.0)]).is_none());
    }
}

/// `(delta_uj, wrapped)`.
pub fn delta_uj(start: u64, end: u64, range_uj: u64) -> Result<(u64, bool), String> {
    if end >= start {
        return Ok((end - start, false));
    }
    if range_uj == 0 || start > range_uj {
        return Err("counter decreased and range_uj does not cover the start".into());
    }
    Ok((range_uj - start + end, true))
}

pub fn dram_joules(samples: &[Sample]) -> Result<Option<f64>, String> {
    let rows: Vec<&Sample> = samples.iter().filter(|s| s.domain == Domain::MacmonDram).collect();
    if rows.is_empty() {
        return Ok(None);
    }
    if rows.iter().all(|s| s.watts.unwrap_or(0.0) == 0.0 && s.counter_uj.unwrap_or(0) == 0) {
        return Ok(None);
    }
    let series: Vec<(u64, f64)> = rows.iter().filter_map(|s| s.watts.map(|w| (s.t_monotonic_ns, w))).collect();
    Ok(trapezoid_joules(&series))
}

/// Refuse the combinations the fleet notes forbid. Does not invent a system total.
pub fn evaluate_domains(samples: &[Sample], sum_rapl: bool, gpu_only: bool, system_wall: bool) -> Result<(), String> {
    if system_wall {
        return Err("macmon and RAPL are not a wall measurement".into());
    }
    if gpu_only {
        return Err("isolated AMD GPU energy is not verified; a GPU-only claim is refused".into());
    }
    let pkg = samples.iter().any(|s| s.domain == Domain::RaplPackage);
    let core = samples.iter().any(|s| s.domain == Domain::RaplCore);
    let ppt = samples.iter().any(|s| s.domain == Domain::RocmPpt);
    if sum_rapl && pkg && core {
        return Err("RAPL package and core are not additive".into());
    }
    if pkg && ppt && sum_rapl {
        return Err("ROCm PPT may overlap the package domain; they are not summed".into());
    }
    let _ = dram_joules(samples)?;
    Ok(())
}
