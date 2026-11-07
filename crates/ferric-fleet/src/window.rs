//! One measured window. Load, compile, warmup, steady inference, and the joint step stay apart.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Load,
    Compile,
    Warmup,
    SteadyInfer,
    JointStep,
}

impl Phase {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "load" => Phase::Load,
            "compile" => Phase::Compile,
            "warmup" => Phase::Warmup,
            "steady-infer" => Phase::SteadyInfer,
            "joint-step" => Phase::JointStep,
            _ => return None,
        })
    }
    pub fn needs_inference(self) -> bool {
        matches!(self, Phase::SteadyInfer | Phase::JointStep)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Drain {
    pub submit_empty: bool,
    pub poll_wait: bool,
    pub readback: bool,
}

pub fn gpu_receipt_allowed(adapter_name: &str) -> Result<(), String> {
    let n = adapter_name.to_ascii_lowercase();
    for bad in ["llvmpipe", "lavapipe", "swiftshader"] {
        if n.contains(bad) {
            return Err(format!("software adapter `{adapter_name}` cannot back a GPU receipt"));
        }
    }
    if adapter_name.trim().is_empty() {
        return Err("adapter name is empty".into());
    }
    Ok(())
}

pub fn require_drains(phase: Phase, before: Drain, after: Drain) -> Result<(), String> {
    if !before.submit_empty || !before.poll_wait {
        return Err("the window must drain wgpu before the measurement: submit empty, then poll".into());
    }
    if !after.submit_empty || !after.poll_wait {
        return Err("the window must drain wgpu after the measurement: submit empty, then poll".into());
    }
    if phase.needs_inference() && !after.readback {
        return Err("steady inference and the joint step must read results back after the second drain".into());
    }
    Ok(())
}

/// A 5 second window is not installed as a default. The caller records the duration they used.
pub fn duration_is_explicit(duration_s: Option<f64>) -> Result<f64, String> {
    match duration_s {
        Some(s) if s.is_finite() && s > 0.0 => Ok(s),
        _ => Err("measurement duration was not supplied; the A100 5 s figure is not a default".into()),
    }
}

pub fn attributable(background: &str, trials: u32) -> Result<(), String> {
    if background != "quiet" {
        return Err("background activity is not a controlled quiet baseline".into());
    }
    if trials < 2 {
        return Err("fewer than two trials; uncertainty is not established".into());
    }
    Ok(())
}
