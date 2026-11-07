//! Clock snapshot checked on every bind. Nothing here is cached.

/// `dt` bits must equal the forward product `(period_ns as f64) * 1e-9`.
/// `t0_ns` must be `origin_ns + n * period_ns` for some `n`.
/// `step_duration_ns` must equal `period_ns`.
#[derive(Debug, Clone, Copy)]
pub struct ClockSnapshot {
    pub origin_ns: u64,
    pub period_ns: u64,
    pub t0_ns: u64,
    pub step_duration_ns: u64,
    pub dt_bits: u64,
}

pub fn bind_clock(s: &ClockSnapshot) -> Result<u64, String> {
    if s.period_ns == 0 || s.step_duration_ns == 0 {
        return Err("period_ns and step_duration_ns must be non-zero".into());
    }
    if s.step_duration_ns != s.period_ns {
        return Err("step_duration_ns is not period_ns".into());
    }
    let forward = (s.period_ns as f64) * 1e-9;
    if !forward.is_finite() || forward <= 0.0 || forward.to_bits() != s.dt_bits {
        return Err("dt bits are not the forward product of period_ns".into());
    }
    if s.t0_ns < s.origin_ns {
        return Err("t0_ns is before origin_ns".into());
    }
    let delta = s.t0_ns - s.origin_ns;
    if delta % s.period_ns != 0 {
        return Err("t0_ns is not origin_ns plus an integer number of periods".into());
    }
    Ok(delta / s.period_ns)
}
