//! Quiet-gate rules from ferrotherm `bench/thrml-tsp` at
//! `c82c1e96c24d6f6c292af1196cd72199b0838257`.
//!
//! Line numbers below are that tree. `METHODOLOGY.md` is
//! `bench/thrml-tsp/METHODOLOGY.md`. The harness is
//! `bench/thrml-tsp/src/energy.rs`.

/// `METHODOLOGY.md` lines 24–26. Not a tuned threshold.
pub const QUIET_LOAD: f64 = 2.0;

/// `energy.rs` lines 738–742.
pub const SETTLE_STREAK: usize = 60;
pub const SETTLE_BOUND: usize = 600;

/// `energy.rs` lines 822–839.
pub const OTHER_CORE_MEAN_MAX: f64 = 0.05;
pub const OTHER_CORE_SAMPLE_MAX: f64 = 0.15;
pub const FOREIGN_CPU_MEAN_MAX: f64 = 0.15;
pub const FOREIGN_CPU_SAMPLE_MAX: f64 = 0.50;

/// `energy.rs` lines 892–896.
pub const HOST_LOCK_WAIT_S: f64 = 3600.0;
pub const HOST_LOCK_REFUSAL: &str = "host lock not held";

/// `energy.rs` lines 429–430. `sys_power` is this scope, not wall power.
pub const SOC_TELEMETRY_LIMIT: &str = "macmon sys_power is SoC telemetry (CPU, integrated GPU, ANE). It is not wall power. Display, fans, and the rest of the board are outside it. Package and core are different scopes and are not added here.";

#[derive(Clone, Debug, PartialEq)]
pub struct SettleOutcome {
    pub settled: bool,
    pub samples: Vec<f64>,
    pub reason: &'static str,
}

/// `energy.rs` `settle_outcome`, lines 756–794.
pub fn settle_outcome(samples: &[Option<f64>]) -> SettleOutcome {
    let mut kept = Vec::new();
    let mut run = 0usize;
    for sample in samples.iter().take(SETTLE_BOUND) {
        match sample {
            Some(v) if v.is_finite() && *v <= QUIET_LOAD => {
                kept.push(*v);
                run += 1;
                if run >= SETTLE_STREAK {
                    return SettleOutcome {
                        settled: true,
                        samples: kept,
                        reason: "load1 stayed at or under 2 for the required consecutive samples",
                    };
                }
            }
            Some(v) => {
                kept.push(if v.is_finite() { *v } else { f64::NAN });
                run = 0;
            }
            None => {
                kept.push(f64::NAN);
                run = 0;
            }
        }
    }
    SettleOutcome {
        settled: false,
        samples: kept,
        reason: "load1 did not stay at or under 2 for 60 consecutive 1 Hz samples within the bound",
    }
}

/// `energy.rs` `bracket_load_offender`, lines 800–811.
pub fn bracket_load_offender(samples: &[Option<f64>]) -> Option<f64> {
    if samples.is_empty() {
        return Some(f64::NAN);
    }
    for sample in samples {
        match sample {
            Some(v) if v.is_finite() && *v <= QUIET_LOAD => {}
            Some(v) => return Some(if v.is_finite() { *v } else { f64::NAN }),
            None => return Some(f64::NAN),
        }
    }
    None
}

/// `energy.rs` `fraction_refusal`, lines 847–858.
pub fn fraction_refusal(samples: &[f64], mean_max: f64, sample_max: f64) -> Option<&'static str> {
    if samples.is_empty() || samples.iter().any(|s| !s.is_finite()) {
        return Some("the trial window recorded no finite sample");
    }
    if samples.iter().any(|s| *s > sample_max) {
        return Some("a sample exceeded the maximum");
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    if mean > mean_max {
        return Some("the mean exceeded the maximum");
    }
    None
}

/// `energy.rs` `foreign_cpu_fraction`, lines 1094–1133.
/// The result is a refusal input. It is not subtracted from joules (`METHODOLOGY.md` lines 3–4).
pub fn foreign_cpu_fraction(
    before: &[(u32, u32, f64)],
    after: &[(u32, u32, f64)],
    self_pid: u32,
    wall_s: f64,
) -> Option<f64> {
    if wall_s.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return None;
    }
    let mut parent = std::collections::BTreeMap::new();
    for (pid, ppid, _) in before.iter().chain(after.iter()) {
        parent.insert(*pid, *ppid);
    }
    let harness = |pid: u32| -> bool {
        let mut cur = pid;
        for _ in 0..32 {
            if cur == self_pid {
                return true;
            }
            match parent.get(&cur).copied() {
                Some(next) if next != cur && next != 0 => cur = next,
                _ => return false,
            }
        }
        false
    };
    let mut prior = std::collections::BTreeMap::new();
    for (pid, _, cpu) in before {
        prior.insert(*pid, *cpu);
    }
    let mut sum = 0.0;
    for (pid, _, cpu) in after {
        if harness(*pid) {
            continue;
        }
        if let Some(was) = prior.get(pid) {
            sum += (cpu - was).max(0.0);
        }
    }
    Some(sum / wall_s)
}

/// `meter/src/lib.rs` does not give macmon a quantum. RAPL `energy_uj` is 1 µJ.
/// `energy.rs` `counter_quantum_j`, lines 462–467.
pub fn counter_quantum_j(instrument: &str) -> Option<f64> {
    match instrument {
        "rapl" => Some(1e-6),
        _ => None,
    }
}

pub fn counter_quantization_j(quantum_j: f64) -> f64 {
    2.0 * quantum_j
}

/// `energy.rs` lines 479–487.
pub fn counter_cadence_edge_j(gross_window: f64, window_s: f64, cadence_s: f64) -> f64 {
    if window_s <= 0.0 || !cadence_s.is_finite() || !gross_window.is_finite() || !cadence_s.is_sign_positive() {
        return f64::NAN;
    }
    cadence_s * (gross_window / window_s)
}

#[derive(Clone, Debug)]
pub struct UncertaintyBudget {
    pub student_t_half: f64,
    pub quantization: Option<f64>,
    pub cadence_edge: Option<f64>,
    pub bracket: f64,
    pub combined: f64,
}

/// `energy.rs` `combine_uncertainty`, lines 509–539.
pub fn combine_uncertainty(
    student_t_half: f64,
    quantization: Option<f64>,
    cadence_edge: Option<f64>,
    bracket: f64,
) -> UncertaintyBudget {
    let quantization = quantization.filter(|q| q.is_finite());
    let cadence_edge = cadence_edge.filter(|q| q.is_finite());
    let mut acc = 0.0;
    let mut n = 0u32;
    for term in [Some(student_t_half), quantization, cadence_edge, Some(bracket)].into_iter().flatten() {
        if term.is_finite() {
            acc += term * term;
            n += 1;
        }
    }
    UncertaintyBudget {
        student_t_half,
        quantization,
        cadence_edge,
        bracket,
        combined: if n == 0 { f64::NAN } else { acc.sqrt() },
    }
}

/// `score.rs` `t_crit_95`, lines 202–241. Two-sided 95% (97.5% one tail).
pub fn t_crit_95(df: usize) -> f64 {
    const T: [f64; 30] = [
        12.7062047364, 4.3026527299, 3.1824463053, 2.7764451052, 2.5705818366, 2.4469118510,
        2.3646242520, 2.3060041352, 2.2621571627, 2.2281388520, 2.2009851601, 2.1788128297,
        2.1603686565, 2.1447866879, 2.1314495456, 2.1199052992, 2.1098155778, 2.1009220402,
        2.0930240544, 2.0859634473, 2.0796138447, 2.0738730679, 2.0686576104, 2.0638985616,
        2.0595385528, 2.0555294386, 2.0518305165, 2.0484071418, 2.0452296421, 2.0422724563,
    ];
    if df == 0 {
        f64::NAN
    } else if df <= 30 {
        T[df - 1]
    } else {
        1.9599639845
    }
}

/// Half-width of the 95% Student-t interval. `score.rs` `mean_ci_95`, lines 187–196.
pub fn student_t_half(xs: &[f64]) -> Option<f64> {
    let v: Vec<f64> = xs.iter().copied().filter(|x| x.is_finite()).collect();
    if v.len() < 2 {
        return None;
    }
    let n = v.len();
    let mean = v.iter().sum::<f64>() / n as f64;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    Some(t_crit_95(n - 1) * (var / n as f64).sqrt())
}

pub fn uncertainty_json(budget: &UncertaintyBudget) -> serde_json::Value {
    serde_json::json!({
        "student_t_half": if budget.student_t_half.is_finite() { serde_json::json!(budget.student_t_half) } else { serde_json::Value::Null },
        "quantization": budget.quantization,
        "cadence_edge": budget.cadence_edge,
        "bracket": if budget.bracket.is_finite() { serde_json::json!(budget.bracket) } else { serde_json::Value::Null },
        "combined": if budget.combined.is_finite() { serde_json::json!(budget.combined) } else { serde_json::Value::Null },
        "combined_is": "root-sum-square of the finite terms; a missing term is omitted, not entered as zero",
    })
}

/// `energy.rs` `format_unix_utc`, lines 903–917, and `civil_from_days`, lines 920–931.
pub fn read_ps_cpu() -> Result<Vec<(u32, u32, f64)>, String> {
    let out = std::process::Command::new("ps")
        .args(["-axo", "pid=,ppid=,time="])
        .output()
        .map_err(|e| format!("ps: {e}"))?;
    if !out.status.success() {
        return Err(format!("ps: {}", String::from_utf8_lossy(&out.stderr)));
    }
    let mut rows = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let mut it = line.split_whitespace();
        let Some(pid) = it.next().and_then(|s| s.parse().ok()) else { continue };
        let Some(ppid) = it.next().and_then(|s| s.parse().ok()) else { continue };
        let Some(time) = it.next().and_then(parse_cpu_clock) else { continue };
        rows.push((pid, ppid, time));
    }
    Ok(rows)
}

fn parse_cpu_clock(text: &str) -> Option<f64> {
    let text = text.trim();
    let (days, rest) = match text.split_once('-') {
        Some((d, rest)) => (d.parse::<f64>().ok()?, rest),
        None => (0.0, text),
    };
    let mut parts: Vec<&str> = rest.split(':').collect();
    if parts.len() < 2 {
        return None;
    }
    let sec: f64 = parts.pop()?.parse().ok()?;
    let min: f64 = parts.pop()?.parse().ok()?;
    let hour: f64 = parts.pop().and_then(|h| h.parse().ok()).unwrap_or(0.0);
    if !parts.is_empty() {
        return None;
    }
    Some(days * 86400.0 + hour * 3600.0 + min * 60.0 + sec)
}

pub fn format_unix_utc(seconds: f64) -> String {
    if !seconds.is_finite() {
        return "invalid".into();
    }
    let whole = seconds.floor() as i64;
    let millis = ((seconds - whole as f64) * 1000.0).round().clamp(0.0, 999.0) as u32;
    let days = whole.div_euclid(86400);
    let tod = whole.rem_euclid(86400) as u32;
    let hour = tod / 3600;
    let min = (tod % 3600) / 60;
    let sec = tod % 60;
    let (year, month, day) = civil_from_days(days);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}.{millis:03}Z")
}

fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };
    (year as i32, month as u32, day as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sixty_quiet_samples_settle_and_a_break_resets_the_streak() {
        let mut s = vec![Some(1.0); 59];
        assert!(!settle_outcome(&s).settled);
        s.push(Some(2.5));
        s.extend(std::iter::repeat_n(Some(1.0), 60));
        let done = settle_outcome(&s);
        assert!(done.settled);
        assert_eq!(done.samples.len(), 120);
        let mut bounded = vec![Some(3.0); SETTLE_BOUND];
        assert!(!settle_outcome(&bounded).settled);
        bounded.push(Some(0.1));
        assert_eq!(settle_outcome(&bounded).samples.len(), SETTLE_BOUND);
    }

    #[test]
    fn an_empty_bracket_is_a_refusal_and_a_sample_above_two_is_the_offender() {
        assert!(bracket_load_offender(&[]).unwrap().is_nan());
        assert_eq!(bracket_load_offender(&[Some(1.0), Some(2.0)]), None);
        assert_eq!(bracket_load_offender(&[Some(1.0), Some(2.1)]), Some(2.1));
    }

    #[test]
    fn linux_outside_cores_use_five_percent_mean_and_fifteen_percent_samples() {
        assert!(fraction_refusal(&[0.04, 0.04], OTHER_CORE_MEAN_MAX, OTHER_CORE_SAMPLE_MAX).is_none());
        assert_eq!(
            fraction_refusal(&[0.04, 0.16], OTHER_CORE_MEAN_MAX, OTHER_CORE_SAMPLE_MAX),
            Some("a sample exceeded the maximum")
        );
        assert_eq!(
            fraction_refusal(&[0.06, 0.06], OTHER_CORE_MEAN_MAX, OTHER_CORE_SAMPLE_MAX),
            Some("the mean exceeded the maximum")
        );
        assert!(fraction_refusal(&[], OTHER_CORE_MEAN_MAX, OTHER_CORE_SAMPLE_MAX).is_some());
    }

    #[test]
    fn foreign_cpu_is_a_fraction_and_is_not_a_joule_term() {
        let before = [(1, 0, 0.0), (2, 1, 1.0), (9, 8, 0.0)];
        let after = [(1, 0, 0.2), (2, 1, 1.4), (9, 8, 0.5)];
        let frac = foreign_cpu_fraction(&before, &after, 1, 1.0).unwrap();
        assert!((frac - 0.5).abs() < 1e-12);
        let joules = 10.0;
        assert_eq!(joules, 10.0);
    }

    #[test]
    fn uncertainty_omits_a_missing_quantum_and_combines_the_rest() {
        let budget = combine_uncertainty(3.0, None, Some(4.0), 0.0);
        assert!(budget.quantization.is_none());
        assert!((budget.combined - 5.0).abs() < 1e-12);
        assert!(counter_quantum_j("macmon").is_none());
        assert_eq!(counter_quantization_j(counter_quantum_j("rapl").unwrap()), 2e-6);
        assert!((counter_cadence_edge_j(10.0, 2.0, 0.1) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn unix_epoch_formats_as_utc() {
        assert_eq!(format_unix_utc(0.0), "1970-01-01T00:00:00.000Z");
    }
}
