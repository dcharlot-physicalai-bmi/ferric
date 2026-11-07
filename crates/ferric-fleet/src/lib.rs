//! Fleet handoff for a SmolVLA run that has not happened.
//!
//! Model receipts are zero. `joint_step_executed` stays false. Energy helpers
//! name the meter boundary and refuse sums the fleet notes already ruled out.

#![forbid(unsafe_code)]

mod acquire;
mod bind;
mod meter;
mod window;

pub use acquire::{artifacts, url, verify_bytes, Artifact};
pub use bind::{bind_clock, ClockSnapshot};
pub use meter::{delta_uj, dram_joules, evaluate_domains, Domain, Sample};
pub use window::{attributable, duration_is_explicit, gpu_receipt_allowed, require_drains, Drain, Phase};

/// No model forward has been attached to this crate.
pub const MODEL_RECEIPTS: usize = 0;

/// A joint step has not been run.
pub const JOINT_STEP_EXECUTED: bool = false;

/// wgpu drain used by Ferric, matching `Context::flush` in `crates/ferric-core/src/lib.rs`.
pub const WGPU_DRAIN: &[&str] = &[
    "queue.submit(empty)",
    "device.poll(wait_indefinitely)",
];

pub const METHODOLOGY: &[&str] = &[
    "https://ml.energy/zeus/measure/",
    "https://ml.energy/blog/energy/measurement/thermally-stable-profiling-for-accurate-gpu-energy-measurement/",
    "https://github.com/ml-energy/benchmark",
];
