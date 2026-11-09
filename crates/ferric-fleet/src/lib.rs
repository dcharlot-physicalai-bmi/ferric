//! Fleet handoff for a SmolVLA run that has not happened.
//!
//! Model receipts are zero. `joint_step_executed` stays false. Energy helpers
//! name the meter boundary and refuse sums the fleet notes already ruled out.

#![forbid(unsafe_code)]

mod acquire;
mod bind;
mod energy_drive;
mod infer;
mod meter;
mod window;

pub use acquire::{Artifact, artifacts, url, verify_bytes};
pub use bind::{ClockSnapshot, bind_clock};
pub use infer::{ACTION_CHUNK_BLOCKER, digest_f32, load_tensor, state_proj};
pub use energy_drive::{
    judge, pinned_argv, receipt_json, run_locked_attempt, run_until_quiet, try_lock_exclusive, BusyFraction,
    CpuTicks, DriveReceipt, EnergySeries, Grade, LiveConfig, LockHold, LOAD1_MAX, LOCK_PATH, SETTLE_S,
};
pub use meter::{
    hwmon_input, trapezoid_joules, Domain, Sample, delta_uj, dram_joules, evaluate_domains,
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
