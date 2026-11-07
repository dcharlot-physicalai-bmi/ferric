//! Meter rules from the fleet notes. Counters stay in their own domain.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    RaplPackage,
    RaplCore,
    RocmPpt,
    MacmonGpu,
    MacmonDram,
    MacmonSys,
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
        }
    }
    pub fn boundary(self) -> &'static str {
        match self {
            Domain::RaplPackage => "RAPL package energy_uj, not the GPU and not the core domain",
            Domain::RaplCore => "RAPL core energy_uj, kept separate from package",
            Domain::RocmPpt => "ROCm-SMI PPT socket figure; overlap with the APU package is not ruled out",
            Domain::MacmonGpu | Domain::MacmonSys => "macmon subsystem telemetry, not wall energy",
            Domain::MacmonDram => "macmon DRAM field",
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
    Err("a non-zero DRAM series is not integrated here; this build does not claim DRAM joules".into())
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
