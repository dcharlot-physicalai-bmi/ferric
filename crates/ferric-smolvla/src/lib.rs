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

mod faults;
mod gemm;
mod gpu;
mod model;
mod noise_math;
mod tolerance;
pub mod weights;

pub use tolerance::{
    bounds_json, gamma, max_abs_bound, rope_angle_bound, rope_invariant_holds, rope_norm_bound, rope_residual,
    within_bound, F32_U, RopeResidual,
};

pub use gpu::{forward_gpu, GpuForward, GpuSession};

pub use faults::{cpu_fault_report, gemm_abft_residual, stage_digest, FAULT_SEED};
pub use model::{
    ActivationTrace, Forward, Sample, forward, prepare_uint8_v070, resize_with_pad, sinusoidal,
};

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

pub fn noise_from_seed(seed: u64) -> Result<Vec<f32>, String> {
    // `seed | 1` would make an even seed and the next odd seed the same payload.
    // Version 3 refuses the even caller seed instead of emitting that payload.
    if seed & 1 == 0 {
        return Err(format!("even noise seed {seed:#x} is refused"));
    }
    let mut state = seed | 1;
    let mut out = Vec::with_capacity(NOISE_LEN);
    while out.len() < NOISE_LEN {
        let u1 = next_unit(&mut state);
        let u2 = next_unit(&mut state);
        let r = (-2.0 * noise_math::ln(u1)).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push((r * noise_math::cos(theta)) as f32);
        if out.len() < NOISE_LEN {
            out.push((r * noise_math::sin(theta)) as f32);
        }
    }
    if out.iter().any(|v| !v.is_finite()) {
        return Err(format!("noise seed {seed:#x} produced a non-finite sample"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_0x5a17_noise_payload_is_the_receipt_hash() {
        let noise = noise_from_seed(0x5a17).expect("odd seed");
        assert_eq!(noise.len(), NOISE_LEN);
        assert_eq!(f32_bytes(&noise).len(), 6400);
        assert!(noise.iter().all(|v| v.is_finite()));
        assert_eq!(
            sha256_hex(&f32_bytes(&noise)),
            "09adba264f855351fd2a0911ce66497cba8e54f9ca73d6b8d905c22918e0e4c2"
        );
        let canon = canonical_noise_bytes(0x5a17, &noise);
        assert_eq!(canon.len(), 6477);
        assert_eq!(
            sha256_hex(&canon),
            "8db2ebc813c83673d6b61b10b2c2a2352018371cbfe9c661f3b405ac6370b61c"
        );
        let doc = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../docs/acceptance/contracts/fixtures/smolvla-noise-v3.json"
        ))
        .unwrap();
        let doc: serde_json::Value = serde_json::from_str(&doc).unwrap();
        assert_eq!(doc["payload_bytes"], 6400);
        assert_eq!(doc["canonical_bytes"], 6477);
        assert_eq!(doc["payload_sha256"], sha256_hex(&f32_bytes(&noise)));
        assert_eq!(doc["noise_sha256"], sha256_hex(&canon));
        let v3 = ferric_accept::smolvla_v3::ActionChunkV3 {
            lerobot_commit: ferric_accept::smolvla::LEROBOT_COMMIT.into(),
            model_id: ferric_accept::smolvla::MODEL_ID.into(),
            weight_sha256: ferric_accept::smolvla_v3::WEIGHT_SHA256.into(),
            stats_sha256: ferric_accept::smolvla_v3::STATS_FILE_SHA256.into(),
            stats_key: "so100".into(),
            action_mean_sha256: ferric_accept::smolvla_v3::SO100_ACTION_MEAN.into(),
            action_std_sha256: ferric_accept::smolvla_v3::SO100_ACTION_STD.into(),
            state_mean_sha256: ferric_accept::smolvla_v3::STATE_MEAN_SHA256.into(),
            state_std_sha256: ferric_accept::smolvla_v3::STATE_STD_SHA256.into(),
            noise_sha256: sha256_hex(&canon),
            seed: 0x5a17,
            embodiment: "so100".into(),
            t0_num_ns: 100_000_000,
            t0_den: 3,
            period_num_ns: 100_000_000,
            period_den: 3,
            n_action_steps: 1,
            action_dim: 6,
            joints: ferric_accept::smolvla_v3::so100_joints(),
            values: vec![0.0, 0.0, 0.0, 50.0, 90.0, 10.0],
        };
        let (bytes, digest) =
            ferric_accept::smolvla_v3::encode_action_chunk_v3(&v3, &noise, 0, 5_000_000, 3).unwrap();
        assert_eq!(digest, ferric_accept::hex(&ferric_accept::sha256(&bytes)));
        let back = ferric_accept::smolvla_v3::decode_action_chunk_v3(&bytes).unwrap();
        assert_eq!(back.seed, 0x5a17);
        assert_eq!(back.noise_sha256, sha256_hex(&canon));
        assert!(noise_from_seed(2).is_err());
        assert!(noise_from_seed(3).is_ok());
    }

    #[test]
    fn real_noise_golden_matches_seed_0x5a17() {
        let noise = noise_from_seed(0x5a17).unwrap();
        let noise_hash = sha256_hex(&canonical_noise_bytes(0x5a17, &noise));
        assert_eq!(
            noise_hash,
            "8db2ebc813c83673d6b61b10b2c2a2352018371cbfe9c661f3b405ac6370b61c"
        );
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/acceptance/contracts/fixtures");
        let row = [0.0f32, 0.0, 0.0, 50.0, 90.0, 10.0];
        let mut values50 = Vec::new();
        for _ in 0..50 {
            values50.extend_from_slice(&row);
        }
        let chunk = |steps: u32, values: Vec<f32>, t0_num: u64, t0_den: u64| {
            ferric_accept::smolvla_v3::ActionChunkV3 {
                lerobot_commit: ferric_accept::smolvla::LEROBOT_COMMIT.into(),
                model_id: ferric_accept::smolvla::MODEL_ID.into(),
                weight_sha256: ferric_accept::smolvla_v3::WEIGHT_SHA256.into(),
                stats_sha256: ferric_accept::smolvla_v3::STATS_FILE_SHA256.into(),
                stats_key: "so100".into(),
                action_mean_sha256: ferric_accept::smolvla_v3::SO100_ACTION_MEAN.into(),
                action_std_sha256: ferric_accept::smolvla_v3::SO100_ACTION_STD.into(),
                state_mean_sha256: ferric_accept::smolvla_v3::STATE_MEAN_SHA256.into(),
                state_std_sha256: ferric_accept::smolvla_v3::STATE_STD_SHA256.into(),
                noise_sha256: noise_hash.clone(),
                seed: 0x5a17,
                embodiment: "so100".into(),
                t0_num_ns: t0_num,
                t0_den: t0_den,
                period_num_ns: 100_000_000,
                period_den: 3,
                n_action_steps: steps,
                action_dim: 6,
                joints: ferric_accept::smolvla_v3::so100_joints(),
                values,
            }
        };
        let encode = |c: &ferric_accept::smolvla_v3::ActionChunkV3| {
            ferric_accept::smolvla_v3::encode_action_chunk_v3(c, &noise, 0, 5_000_000, 3).unwrap()
        };
        let real: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("smolvla-chunk-v3-real-noise.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(real["action_values"], "fixture values, not model output");
        assert_eq!(real["noise_sha256"], noise_hash);
        assert_eq!(real["n_action_steps"], 50);
        let (bytes, digest) = encode(&chunk(50, values50, 100_000_000, 3));
        assert_eq!(bytes.len(), 2119);
        assert_eq!(digest, "13e82f106898b5a2b1c6fb48f487cf7d6cbe81cb2f8410a71e8f626a2f0c329b");
        assert_eq!(digest, real["digest_sha256"]);
        assert_eq!(hex_bytes(&bytes), real["canonical_hex"].as_str().unwrap());
        let seq: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("smolvla-chunk-v3-sequence.json")).unwrap(),
        )
        .unwrap();
        let (b0, d0) = encode(&chunk(1, row.to_vec(), 0, 1));
        let (b1, d1) = encode(&chunk(1, row.to_vec(), 100_000_000, 3));
        let chunks = seq["chunks"].as_array().unwrap();
        assert_eq!(chunks[0]["chunk_index"], 0);
        assert_eq!(chunks[1]["chunk_index"], 1);
        assert!(chunks[0]["k"].as_u64().unwrap() < chunks[1]["k"].as_u64().unwrap());
        assert_eq!(d0, chunks[0]["digest_sha256"]);
        assert_eq!(d1, chunks[1]["digest_sha256"]);
        assert_eq!(hex_bytes(&b0), chunks[0]["canonical_hex"].as_str().unwrap());
        assert_eq!(hex_bytes(&b1), chunks[1]["canonical_hex"].as_str().unwrap());
    }

    fn hex_bytes(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            s.push(HEX[(b >> 4) as usize] as char);
            s.push(HEX[(b & 0xf) as usize] as char);
        }
        s
    }

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

    #[test]
    fn mean_std_refuses_a_stats_file_without_observation_state() {
        let bytes = include_bytes!("../../ferric-fleet/tests/fixtures/policy_preprocessor_step_5_normalizer_processor.safetensors");
        let err = require_mean_std(bytes, "observation.state", "action").unwrap_err();
        assert!(err.contains("MEAN_STD"));
        assert!(err.contains("observation.state"));
        assert!(!err.contains("IDENTITY is applied"));
        let mapped = require_mean_std(bytes, "observation.state", "so100.buffer.action").unwrap_err();
        assert!(mapped.contains("observation.state"));
    }

    #[test]
    fn simulation_gate_keeps_the_rollout_refused() {
        use ferric_accept::smolvla::{ChunkIdentity, JointMeta};
        let id = ChunkIdentity {
            model_id: "lerobot/smolvla_base".into(),
            weight_sha256: "ab".repeat(32),
            stats_sha256: "cd".repeat(32),
            noise_sha256: "ef".repeat(32),
            embodiment: "simulation-embodiment".into(),
            t0_ns: 1_000_000_000,
            step_duration_ns: 20_000_000,
            joints: vec![JointMeta {
                name: "action_0".into(),
                unit: "dataset_native".into(),
                frame: "model_action_index".into(),
                min: -10.0,
                max: 10.0,
            }],
        };
        let report = simulation_validation(&[0.25, -0.5], id, 1_000_000_000).unwrap();
        assert_eq!(report.gate, "simulation-validation");
        assert!(!report.authorized_rollout);
        assert_eq!(report.authorize_error, "ProposedEntryInvoked");
        assert!(!report.robot_actuated);
        assert_eq!(report.steps.len(), 2);
        assert_eq!(report.steps[0].1, 1_000_000_000);
        assert_eq!(report.steps[1].1, 1_020_000_000);
        assert_eq!(report.steps[1].2, vec![-0.5]);
    }
}

fn canonical_noise_bytes(seed: u64, payload: &[f32]) -> Vec<u8> {
    fn put_str(out: &mut Vec<u8>, s: &str) {
        let b = s.as_bytes();
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
        out.extend_from_slice(b);
    }
    let mut out = Vec::new();
    put_str(&mut out, "f32");
    out.extend_from_slice(&3u32.to_le_bytes());
    for extent in [1u64, 50, 32] {
        out.extend_from_slice(&extent.to_le_bytes());
    }
    put_str(&mut out, "le");
    out.extend_from_slice(&f32_bytes(payload));
    out.extend_from_slice(&seed.to_le_bytes());
    put_str(&mut out, "xorshift64-box-muller-v1");
    out
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

pub struct EmbodimentMap {
    pub embodiment: String,
    pub state_key: String,
    pub action_key: String,
}

pub fn parse_embodiment_map(bytes: &[u8]) -> Result<EmbodimentMap, String> {
    let v: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    Ok(EmbodimentMap {
        embodiment: v["embodiment"].as_str().unwrap_or("").to_string(),
        state_key: v["observation.state"].as_str().unwrap_or("").to_string(),
        action_key: v["action"].as_str().unwrap_or("").to_string(),
    })
}

/// MEAN_STD execution. A missing feature key is a refusal. It is not IDENTITY.
pub fn require_mean_std(stats: &[u8], state_key: &str, action_key: &str) -> Result<(), String> {
    if state_key.is_empty() || action_key.is_empty() {
        return Err("MEAN_STD execution requires a pinned embodiment map with observation.state and action keys".into());
    }
    let store = weights::Store::open_bytes(stats)?;
    let names = store.names();
    let mut missing = Vec::new();
    for key in [state_key, action_key] {
        let has_mean = names.iter().any(|n| n == &format!("{key}.mean"));
        let has_std = names.iter().any(|n| n == &format!("{key}.std"));
        if !has_mean || !has_std {
            missing.push(key.to_string());
        }
    }
    if missing.is_empty() {
        return Ok(());
    }
    let prefixes: Vec<&str> = names.iter().filter_map(|n| n.strip_suffix(".mean")).collect();
    Err(format!(
        "MEAN_STD execution refused. Missing feature keys: {}. Stats prefixes present: {}. The normalization schema is MEAN_STD, not IDENTITY.",
        missing.join(", "),
        prefixes.join(", ")
    ))
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

pub struct SimulationReport {
    pub gate: &'static str,
    pub authorized_rollout: bool,
    pub authorize_error: String,
    pub robot_actuated: bool,
    pub steps: Vec<(u32, u64, Vec<f32>)>,
}

/// Clock-step a codec-valid chunk in simulation. `authorize_rollout` must
/// return `ProposedEntryInvoked`. This function does not turn that refusal
/// into an authorized rollout or a robot write.
pub fn simulation_validation(actions: &[f32], identity: ChunkIdentity, origin_ns: u64) -> Result<SimulationReport, String> {
    let action_dim = identity.joints.len() as u32;
    if action_dim == 0 || actions.len() % action_dim as usize != 0 {
        return Err("simulation actions are not a multiple of the joint count".into());
    }
    let n = (actions.len() / action_dim as usize) as u32;
    let chunk = ActionChunk::try_new(n, action_dim, actions.to_vec()).map_err(|e| format!("{e:?}"))?;
    let authorize_error = match ferric_accept::smolvla::authorize_rollout(&chunk, &identity) {
        Err(ferric_accept::smolvla::ContractError::ProposedEntryInvoked) => "ProposedEntryInvoked".to_string(),
        Err(e) => return Err(format!("chunk is not simulation-eligible: {e:?}")),
        Ok(()) => return Err("authorize_rollout returned Ok".into()),
    };
    let period = identity.step_duration_ns;
    let mut steps = Vec::new();
    for k in 0..n {
        let snapshot = ClockSnapshot {
            origin_ns,
            period_ns: period,
            t0_ns: identity.t0_ns,
            step_duration_ns: period,
            dt_bits: ((period as f64) * 1e-9).to_bits(),
        };
        bind_clock(&snapshot).map_err(|e| format!("joint clock: {e}"))?;
        let timestamp_ns = ferric_accept::smolvla::action_timestamp_ns(identity.t0_ns, period, k, n).map_err(|e| format!("{e:?}"))?;
        let start = k as usize * action_dim as usize;
        steps.push((k, timestamp_ns, actions[start..start + action_dim as usize].to_vec()));
    }
    Ok(SimulationReport {
        gate: "simulation-validation",
        authorized_rollout: false,
        authorize_error,
        robot_actuated: false,
        steps,
    })
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
