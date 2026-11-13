//! Fleet handoff for a SmolVLA run that has not happened.
//!
//! Model receipts are zero. `joint_step_executed` stays false. Energy helpers
//! name the meter boundary and refuse sums the fleet notes already ruled out.

#![forbid(unsafe_code)]

mod acquire;
mod bind;
mod energy_drive;
mod fleet_gate;
mod infer;
mod macmon;
mod meter;
mod window;

pub use acquire::{Artifact, artifacts, url, verify_bytes};
pub use bind::{ClockSnapshot, bind_clock};
pub use infer::{ACTION_CHUNK_BLOCKER, digest_f32, load_tensor, state_proj};
pub use energy_drive::{
    judge, macmon_receipt, parse_cpu_list, pinned_argv, receipt_json, run_locked_attempt, run_locked_macmon,
    run_until_quiet, run_until_quiet_macmon, try_lock_exclusive, BusyFraction, CpuTicks, DriveReceipt,
    EnergySeries, Grade, LiveConfig, LockHold, MacmonAttempt, MacmonConfig, LOAD1_MAX, LOCK_PATH, SETTLE_S,
};
pub use macmon::{
    incremental_joules, integrate, mean_power, overlaps, parse_line, parse_proc_load1, parse_sysctl_loadavg, pipe_args,
    Reading, Sampler, BUSY_FRACTION_REASON, PINNING, PINNING_REASON,
};
pub use meter::{
    hwmon_input, trapezoid_joules, Domain, Sample, delta_uj, dram_joules, evaluate_domains,
};
pub use fleet_gate::{
    bracket_load_offender, combine_uncertainty, counter_cadence_edge_j, counter_quantization_j,
    counter_quantum_j, foreign_cpu_fraction, fraction_refusal, settle_outcome, student_t_half,
    format_unix_utc, uncertainty_json, FOREIGN_CPU_MEAN_MAX, FOREIGN_CPU_SAMPLE_MAX,
    HOST_LOCK_REFUSAL, HOST_LOCK_WAIT_S, OTHER_CORE_MEAN_MAX, OTHER_CORE_SAMPLE_MAX, QUIET_LOAD,
    SETTLE_BOUND, SETTLE_STREAK, SOC_TELEMETRY_LIMIT,
};
pub use window::{
    Drain, Phase, attributable, duration_is_explicit, gpu_receipt_allowed, require_drains,
};

/// No model forward has been attached to this crate.
pub const MODEL_RECEIPTS: usize = 0;

/// A joint step has not been run.
pub const JOINT_STEP_EXECUTED: bool = false;

/// wgpu drain used by Ferric, matching `Context::flush` in `crates/ferric-core/src/lib.rs`.
pub const WGPU_DRAIN: &[&str] = &["queue.submit(empty)", "device.poll(wait_indefinitely)"];

pub const METHODOLOGY: &[&str] = &[
    "https://ml.energy/zeus/measure/",
    "https://ml.energy/blog/energy/measurement/thermally-stable-profiling-for-accurate-gpu-energy-measurement/",
    "https://github.com/ml-energy/benchmark",
];
