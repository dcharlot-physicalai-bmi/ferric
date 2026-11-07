//! SmolVLA action chunks from the pinned `lerobot/smolvla_base` weights.
//!
//! The forward runs on `ferric_core::cpu` (matmul-equivalent GEMM, RMSNorm,
//! LayerNorm, SiLU, softmax). A separate wgpu check compares `state_proj`
//! on the VM adapter and labels a software adapter as software. No path
//! here emits a GPU energy figure.
//!
//! `ferric_accept::smolvla::authorize_rollout` stays `ProposedEntryInvoked`.
//! This crate does not call it. Encoding a chunk is the v2 codec. A joint
//! command read from that chunk is not a robot actuation.

mod gemm;
mod model;
pub mod weights;

pub use model::{Forward, Sample, forward, resize_with_pad, sinusoidal};

use ferric_accept::smolvla::{ActionChunk, ChunkIdentity, JointMeta, encode_action_chunk};
use ferric_accept::{hex, sha256};
use ferric_fleet::{ClockSnapshot, bind_clock};

pub const CAMERAS: usize = 3;
pub const STATE_DIM: usize = 6;
pub const CHUNK_STEPS: usize = 50;
pub const ACTION_DIM: usize = 6;
pub const NOISE_LEN: usize = 50 * 32;

pub fn project_state(weight: &[f32], bias: &[f32], x: &[f32]) -> Vec<f32> {
    gemm::linear(x, weight, Some(bias), 1, 32, 960)
}

pub fn synthetic_image(cam: usize, h: usize, w: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; 3 * h * w];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                let p = ((y * 17 + x * 3 + cam * 5 + c * 13) % 251) as f32 / 250.0;
                v[c * h * w + y * w + x] = p;
            }
        }
    }
    v
}

pub fn noise_from_seed(seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(NOISE_LEN);
    while out.len() < NOISE_LEN {
        let u1 = next_unit(&mut state);
        let u2 = next_unit(&mut state);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push((r * theta.cos()) as f32);
        if out.len() < NOISE_LEN {
            out.push((r * theta.sin()) as f32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_proj_column_matches_the_extracted_weights() {
        let bytes = include_bytes!("../../ferric-fleet/tests/fixtures/state_proj.safetensors");
        let store = weights::Store::open_bytes(bytes).unwrap();
        let w = store.tensor("model.state_proj.weight").unwrap();
        let b = store.tensor("model.state_proj.bias").unwrap();
        let mut x = [0.0f32; 32];
        x[0] = 1.0;
        let y = project_state(&w, &b, &x);
        assert!((y[0] - -0.13992482).abs() < 1e-6, "{}", y[0]);
        assert!((y[1] - -0.08349697).abs() < 1e-6, "{}", y[1]);
    }
}

fn next_unit(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x >> 11) as f64 / ((1u64 << 53) as f64)
}

pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&sha256(bytes))
}

pub fn ensure_newline(task: &str) -> String {
    if task.ends_with('\n') {
        task.to_string()
    } else {
        format!("{task}\n")
    }
}

pub fn tokenize(tokenizer_path: &str, task: &str) -> Result<(Vec<u32>, Vec<bool>), String> {
    use tokenizers::{
        PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationParams,
    };
    let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())?;
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(48),
        direction: PaddingDirection::Right,
        pad_to_multiple_of: None,
        pad_id: 2,
        pad_type_id: 0,
        pad_token: "<|im_end|>".into(),
    }));
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: 48,
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
    let enc = tokenizer.encode(task, true).map_err(|e| e.to_string())?;
    let ids = enc.get_ids().to_vec();
    let mask = enc.get_attention_mask().iter().map(|v| *v != 0).collect();
    if ids.len() != 48 {
        return Err(format!("tokenizer returned {} ids", ids.len()));
    }
    Ok((ids, mask))
}

pub fn load_stats(path: &str, key: &str) -> Result<(Vec<f32>, Vec<f32>), String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let store = weights::Store::open_bytes(&bytes)?;
    let mean = store.tensor(&format!("{key}.mean"))?;
    let std = store.tensor(&format!("{key}.std"))?;
    Ok((mean, std))
}

pub struct EncodedChunk {
    pub digest: String,
    pub bytes: Vec<u8>,
    pub command: Vec<f32>,
    pub timestamp_ns: u64,
    pub robot_actuated: bool,
}

pub fn encode_and_read_command(
    actions: &[f32],
    identity: ChunkIdentity,
    origin_ns: u64,
) -> Result<EncodedChunk, String> {
    let chunk = ActionChunk::try_new(CHUNK_STEPS as u32, ACTION_DIM as u32, actions.to_vec())
        .map_err(|e| format!("{e:?}"))?;
    let (bytes, digest) = encode_action_chunk(&chunk, &identity).map_err(|e| format!("{e:?}"))?;
    let period = identity.step_duration_ns;
    let snapshot = ClockSnapshot {
        origin_ns,
        period_ns: period,
        t0_ns: identity.t0_ns,
        step_duration_ns: period,
        dt_bits: ((period as f64) * 1e-9).to_bits(),
    };
    bind_clock(&snapshot).map_err(|e| format!("joint clock: {e}"))?;
    let timestamp_ns =
        ferric_accept::smolvla::action_timestamp_ns(identity.t0_ns, period, 0, CHUNK_STEPS as u32)
            .map_err(|e| format!("{e:?}"))?;
    Ok(EncodedChunk {
        digest,
        bytes,
        command: actions[..ACTION_DIM].to_vec(),
        timestamp_ns,
        robot_actuated: false,
    })
}

pub fn parse_joints(bytes: &[u8]) -> Result<Vec<JointMeta>, String> {
    let v: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    let arr = v.as_array().ok_or("joints file must be an array")?;
    arr.iter()
        .map(|j| {
            Ok(JointMeta {
                name: j["name"].as_str().unwrap_or("").to_string(),
                unit: j["unit"].as_str().unwrap_or("").to_string(),
                frame: j["frame"].as_str().unwrap_or("").to_string(),
                min: j["min"].as_f64().ok_or("joint min")? as f32,
                max: j["max"].as_f64().ok_or("joint max")? as f32,
            })
        })
        .collect()
}

pub async fn wgpu_state_proj(weight: &[f32], bias: &[f32], x: &[f32]) -> Result<WgpuCheck, String> {
    let adapters = ferric_core::Context::enumerate().await;
    let listed: Vec<String> = adapters
        .iter()
        .map(|(n, b, t)| format!("{n} backend={b:?} type={t:?}"))
        .collect();
    let ctx = ferric_core::Context::new().await?;
    let mut bt = vec![0.0f32; 32 * 960];
    for o in 0..960 {
        for i in 0..32 {
            bt[i * 960 + o] = weight[o * 32 + i];
        }
    }
    let y = ctx.matmul(x, &bt, 1, 32, 960).await?;
    let mut with_bias = y;
    for o in 0..960 {
        with_bias[o] += bias[o];
    }
    let cpu = crate::gemm::linear(x, weight, Some(bias), 1, 32, 960);
    let mut max_abs = 0.0f32;
    for (a, b) in with_bias.iter().zip(&cpu) {
        max_abs = max_abs.max((a - b).abs());
    }
    let name_l = ctx.adapter_name.to_lowercase();
    let software = name_l.contains("llvmpipe")
        || name_l.contains("lavapipe")
        || name_l.contains("swiftshader")
        || listed
            .iter()
            .any(|s| s.contains(&ctx.adapter_name) && s.contains("type=Cpu"));
    Ok(WgpuCheck {
        adapter: ctx.adapter_name,
        backend: format!("{:?}", ctx.backend),
        software,
        present: true,
        max_abs: Some(max_abs),
        listed,
        error: None,
    })
}

pub fn wgpu_absent(error: String) -> WgpuCheck {
    WgpuCheck {
        adapter: "none".into(),
        backend: "none".into(),
        software: false,
        present: false,
        max_abs: None,
        listed: Vec::new(),
        error: Some(error),
    }
}

pub struct WgpuCheck {
    pub adapter: String,
    pub backend: String,
    pub software: bool,
    pub present: bool,
    pub max_abs: Option<f32>,
    pub listed: Vec<String>,
    pub error: Option<String>,
}
