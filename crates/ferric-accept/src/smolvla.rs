//! Ferric ↔ Ferromotion contract for a SmolVLA action chunk.
//!
//! This module checks the pins and the byte layout. It does not load
//! `lerobot/smolvla_base`, does not run flow matching, and does not emit an action
//! from pixels. The inference entry point is a name with status [`Support::Proposed`].
//!
//! Pins are the LeRobot sources at commit `a0eb860d1e4d566157dd0eba11b5c86f9ea7d07a`
//! (2026-07-23), files `configuration_smolvla.py`, `processor_smolvla.py`,
//! `modeling_smolvla.py`, `normalize_processor.py`, and `policies/common/vla_utils.py`.
//! Where a docstring in those files disagrees with the code, the code is the pin
//! and the disagreement is recorded on [`DOCSTRING_DISAGREEMENTS`].

use crate::sha256::{hex, sha256};

/// LeRobot commit the pins were read from.
pub const LEROBOT_COMMIT: &str = "a0eb860d1e4d566157dd0eba11b5c86f9ea7d07a";
/// Commit date from the GitHub commit object, UTC.
pub const LEROBOT_COMMIT_DATE: &str = "2026-07-23";
/// Model id named by the LeRobot model card and the tutorial. Not a weight hash.
pub const MODEL_ID: &str = "lerobot/smolvla_base";
/// Proposed stable name. Nothing in this repository implements it.
pub const INFERENCE_ENTRY_POINT: &str = "ferric.smolvla.predict_action_chunk.v1";
/// Bytes this codec writes. Version 1.
pub const CHUNK_CODEC_VERSION: u32 = 1;
pub const CHUNK_DOMAIN: &str = "ferric.smolvla.action-chunk";

pub const N_OBS_STEPS: u32 = 1;
pub const CHUNK_SIZE: u32 = 50;
pub const N_ACTION_STEPS: u32 = 50;
pub const MAX_STATE_DIM: u32 = 32;
pub const MAX_ACTION_DIM: u32 = 32;
/// `resize_imgs_with_padding`, stored as (width, height). The default is square.
pub const RESIZE_WH: (u32, u32) = (512, 512);
pub const TOKENIZER_MAX_LENGTH: u32 = 48;
/// Euler integration count inside `sample_actions`. Not the action horizon.
pub const FLOW_NUM_STEPS: u32 = 10;
pub const VLM_MODEL_NAME: &str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct";
pub const NORM_EPS: f32 = 1e-8;
pub const MIN_PERIOD: f32 = 4e-3;
pub const MAX_PERIOD: f32 = 4.0;
pub const PAD_VALUE: f32 = 0.0;

pub const DOCSTRING_DISAGREEMENTS: &[&str] = &[
    "processor_smolvla.make_smolvla_pre_post_processors documents normalize before add_batch_dim; the input_steps list normalizes last, after tokenize and to_device",
    "prepare_images documents a 224x224 resize; the call uses resize_imgs_with_padding, default (512, 512)",
    "sample_actions documents shape (batch, num_steps, num_motors); the allocated noise shape is (batch, chunk_size, max_action_dim), and num_steps is the Euler count",
];

/// What Ferric actually does with this surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Support {
    /// A function in this crate checks or encodes the value.
    CheckedHere,
    /// The rule is copied from the pinned LeRobot source and is not executed on a model.
    SpecifiedNotRun,
    /// A name reserved for a later runtime. Calling it as if it ran is a contract error.
    Proposed,
}

/// Why a request was refused. These are contract refusals, not model outputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractError {
    ActionStepsExceedChunk { n_action_steps: u32, chunk_size: u32 },
    DeltaAlohaNotPorted,
    MissingNormStats,
    /// LeRobot's `pad_vector` leaves a longer vector unchanged. The state projection is
    /// `Linear(max_state_dim, ...)`. This contract refuses the longer vector so the
    /// width stays the projection width.
    StateDimExceedsMax { got: u32, max: u32 },
    ActionDimExceedsMax { got: u32, max: u32 },
    ImageRank { got: u8 },
    ImageNotUnitInterval,
    TokenLength { got: u32, max: u32 },
    NoiseShape,
    EmptyImages,
    ChunkShape { time: u32, action_dim: u32, values: usize },
    StochasticAgreement,
    ProposedEntryInvoked,
}

/// Dataset-native numbers after unnormalize. Not degrees and not radians.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionUnit {
    DatasetNative,
}

/// The caller names the embodiment. The base model does not convert frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionFrame {
    EmbodimentNamed,
}

/// `action_delta_indices` is `range(chunk_size)`. Index `k` is a step, not a second.
/// Dataset fps is not a field of `SmolVLAConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionTime {
    ChunkStep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Determinism {
    /// `noise` was supplied and is inside the digest.
    PinnedNoise,
    /// `sample_actions` draws noise when `noise` is `None`. Not an agreement receipt.
    Stochastic,
}

/// Stats for MEAN_STD. Visual features are IDENTITY and do not use these.
#[derive(Debug, Clone, Copy)]
pub struct MeanStd {
    pub mean: f32,
    pub std: f32,
}

/// One scalar, the map in `normalize_processor.py` at the pin.
///
/// Forward: `(x - mean) / (std + eps)` with `eps = 1e-8`.
/// Inverse: `y * std + mean`. The inverse multiplies by `std`, not by `std + eps`,
/// so it is not an exact algebraic inverse when `std` is near zero.
pub fn mean_std_forward(x: f32, s: MeanStd) -> Result<f32, ContractError> {
    if !s.mean.is_finite() || !s.std.is_finite() {
        return Err(ContractError::MissingNormStats);
    }
    Ok((x - s.mean) / (s.std + NORM_EPS))
}

pub fn mean_std_inverse(y: f32, s: MeanStd) -> Result<f32, ContractError> {
    if !s.mean.is_finite() || !s.std.is_finite() {
        return Err(ContractError::MissingNormStats);
    }
    Ok(y * s.std + s.mean)
}

/// Right-pad with zeros up to `new_dim`. A longer vector is refused here.
/// `pad_vector(..., truncate=False)` would return it unchanged.
pub fn pad_state(values: &[f32]) -> Result<Vec<f32>, ContractError> {
    if values.len() > MAX_STATE_DIM as usize {
        return Err(ContractError::StateDimExceedsMax {
            got: values.len() as u32,
            max: MAX_STATE_DIM,
        });
    }
    let mut out = vec![0.0; MAX_STATE_DIM as usize];
    out[..values.len()].copy_from_slice(values);
    Ok(out)
}

/// The sentence in `make_smolvla_pre_post_processors`: the task ends with a newline.
pub fn ensure_task_newline(task: &str) -> String {
    if task.ends_with('\n') {
        task.to_string()
    } else {
        format!("{task}\n")
    }
}

pub fn check_horizon(n_action_steps: u32, chunk_size: u32) -> Result<(), ContractError> {
    if n_action_steps > chunk_size {
        Err(ContractError::ActionStepsExceedChunk { n_action_steps, chunk_size })
    } else {
        Ok(())
    }
}

/// `use_delta_joint_actions_aloha` raises `NotImplementedError` in `__post_init__`.
pub fn check_aloha_delta(enabled: bool) -> Result<(), ContractError> {
    if enabled {
        Err(ContractError::DeltaAlohaNotPorted)
    } else {
        Ok(())
    }
}

/// Map a pixel that is already in `[0, 1]` with `img * 2 - 1`, after pad value 0.
/// Values outside `[0, 1]` are refused. `prepare_images` does not check the range;
/// its comment says SigLIP expects the map from `[0, 1]`.
pub fn siglip_range(pixel: f32) -> Result<f32, ContractError> {
    if !(0.0..=1.0).contains(&pixel) {
        return Err(ContractError::ImageNotUnitInterval);
    }
    Ok(pixel * 2.0 - 1.0)
}

/// SmolVLA `resize_with_pad`: bilinear, `align_corners` false, pad on the left and the top.
///
/// `height` and `width` are the helper's arguments. The config tuple is (width, height),
/// and `prepare_images` passes `resize_imgs_with_padding[1]` as height and `[0]` as width.
/// Channel order in is `(batch, channel, height, width)`.
pub fn resize_with_pad(
    image: &[f32],
    batch: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Result<Vec<f32>, ContractError> {
    if channels == 0 || in_h == 0 || in_w == 0 || out_h == 0 || out_w == 0 {
        return Err(ContractError::ImageRank { got: 0 });
    }
    if image.len() != batch * channels * in_h * in_w {
        return Err(ContractError::ImageRank { got: image.len().min(255) as u8 });
    }
    if in_h == out_h && in_w == out_w {
        return Ok(image.to_vec());
    }
    let ratio = (in_w as f32 / out_w as f32).max(in_h as f32 / out_h as f32);
    let resized_h = (in_h as f32 / ratio) as usize;
    let resized_w = (in_w as f32 / ratio) as usize;
    if resized_h == 0 || resized_w == 0 {
        return Err(ContractError::ImageRank { got: 0 });
    }
    let mut resized = vec![0.0; batch * channels * resized_h * resized_w];
    for b in 0..batch {
        for c in 0..channels {
            for y in 0..resized_h {
                for x in 0..resized_w {
                    let src_y = ((y as f32 + 0.5) * ratio - 0.5).max(0.0);
                    let src_x = ((x as f32 + 0.5) * ratio - 0.5).max(0.0);
                    let y0 = (src_y.floor() as usize).min(in_h - 1);
                    let x0 = (src_x.floor() as usize).min(in_w - 1);
                    let y1 = (y0 + 1).min(in_h - 1);
                    let x1 = (x0 + 1).min(in_w - 1);
                    let wy = src_y - y0 as f32;
                    let wx = src_x - x0 as f32;
                    let at = |yy: usize, xx: usize| {
                        image[(((b * channels + c) * in_h + yy) * in_w) + xx]
                    };
                    let v = at(y0, x0) * (1.0 - wy) * (1.0 - wx)
                        + at(y0, x1) * (1.0 - wy) * wx
                        + at(y1, x0) * wy * (1.0 - wx)
                        + at(y1, x1) * wy * wx;
                    resized[(((b * channels + c) * resized_h + y) * resized_w) + x] = v;
                }
            }
        }
    }
    let pad_h = out_h.saturating_sub(resized_h);
    let pad_w = out_w.saturating_sub(resized_w);
    let mut out = vec![PAD_VALUE; batch * channels * out_h * out_w];
    for b in 0..batch {
        for c in 0..channels {
            for y in 0..resized_h {
                for x in 0..resized_w {
                    let dy = y + pad_h;
                    let dx = x + pad_w;
                    if dy < out_h && dx < out_w {
                        let src = (((b * channels + c) * resized_h + y) * resized_w) + x;
                        let dst = (((b * channels + c) * out_h + dy) * out_w) + dx;
                        out[dst] = resized[src];
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Code order of `input_steps`, not the docstring order.
pub const PREPROCESS_CODE_ORDER: &[&str] = &[
    "rename_observations",
    "add_batch_dim",
    "newline_task",
    "tokenize",
    "to_device",
    "normalize",
];

/// What `prepare_images` and `prepare_state` do after the processor, still before the expert.
pub const MODEL_PREPARE_ORDER: &[&str] = &[
    "image_last_time_if_rank5",
    "resize_with_pad_left_top",
    "siglip_times_2_minus_1",
    "state_last_time_if_rank3",
    "pad_state_zeros_to_max_state_dim",
];

pub const POSTPROCESS_CODE_ORDER: &[&str] = &["unnormalize_action", "to_cpu"];

/// One robot, one chunk, f32, time-major then action dimension.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionChunk {
    pub n_action_steps: u32,
    pub action_dim: u32,
    /// Length `n_action_steps * action_dim`. Row `t` starts at `t * action_dim`.
    pub values: Vec<f32>,
    pub unit: ActionUnit,
    pub frame: ActionFrame,
    pub time: ActionTime,
}

impl ActionChunk {
    pub fn try_new(n_action_steps: u32, action_dim: u32, values: Vec<f32>) -> Result<Self, ContractError> {
        check_horizon(n_action_steps, CHUNK_SIZE)?;
        if action_dim == 0 || action_dim > MAX_ACTION_DIM {
            return Err(ContractError::ActionDimExceedsMax { got: action_dim, max: MAX_ACTION_DIM });
        }
        if values.len() != n_action_steps as usize * action_dim as usize {
            return Err(ContractError::ChunkShape {
                time: n_action_steps,
                action_dim,
                values: values.len(),
            });
        }
        Ok(Self {
            n_action_steps,
            action_dim,
            values,
            unit: ActionUnit::DatasetNative,
            frame: ActionFrame::EmbodimentNamed,
            time: ActionTime::ChunkStep,
        })
    }
}

/// Fields the digest covers. Wall-clock, adapter name, and energy are outside it.
#[derive(Debug, Clone)]
pub struct ChunkIdentity {
    pub model_id: String,
    /// 64 lowercase hex characters, or empty when the weight file was not hashed.
    pub weight_sha256: String,
    /// 64 lowercase hex of the dataset mean/std tensors the unnormalize used.
    pub stats_sha256: String,
    /// 64 lowercase hex of the caller noise. Empty means [`Determinism::Stochastic`].
    pub noise_sha256: String,
    pub embodiment: String,
}

/// Canonical bytes, then their sha256.
pub fn encode_action_chunk(chunk: &ActionChunk, id: &ChunkIdentity) -> Result<(Vec<u8>, String), ContractError> {
    if id.weight_sha256.is_empty() || id.stats_sha256.is_empty() {
        return Err(ContractError::MissingNormStats);
    }
    if !id.noise_sha256.is_empty() && !is_sha256(&id.noise_sha256) {
        return Err(ContractError::NoiseShape);
    }
    if !is_sha256(&id.weight_sha256) || !is_sha256(&id.stats_sha256) {
        return Err(ContractError::MissingNormStats);
    }
    let mut buf = Vec::new();
    buf.extend_from_slice(b"FERRIC-SMOLVLA-CHUNK\0");
    buf.extend_from_slice(&CHUNK_CODEC_VERSION.to_le_bytes());
    push_str(&mut buf, LEROBOT_COMMIT);
    push_str(&mut buf, &id.model_id);
    push_str(&mut buf, &id.weight_sha256);
    push_str(&mut buf, &id.stats_sha256);
    push_str(&mut buf, &id.noise_sha256);
    push_str(&mut buf, &id.embodiment);
    buf.extend_from_slice(&chunk.n_action_steps.to_le_bytes());
    buf.extend_from_slice(&chunk.action_dim.to_le_bytes());
    buf.push(1); // f32
    buf.push(1); // time-major
    buf.push(0); // dataset-native unit
    buf.push(0); // embodiment frame
    buf.push(0); // chunk step, not seconds
    for v in &chunk.values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    let digest = hex(&sha256(&buf));
    Ok((buf, digest))
}

pub fn determinism_of(id: &ChunkIdentity) -> Determinism {
    if id.noise_sha256.is_empty() {
        Determinism::Stochastic
    } else {
        Determinism::PinnedNoise
    }
}

/// An agreement receipt needs pinned noise. A stochastic sample is not one.
pub fn require_reference_agreement(id: &ChunkIdentity) -> Result<(), ContractError> {
    match determinism_of(id) {
        Determinism::PinnedNoise => Ok(()),
        Determinism::Stochastic => Err(ContractError::StochasticAgreement),
    }
}

/// The proposed entry. It refuses, because this crate does not run the network.
pub fn predict_action_chunk() -> Result<ActionChunk, ContractError> {
    Err(ContractError::ProposedEntryInvoked)
}

pub fn support_of_entry() -> Support {
    Support::Proposed
}

fn push_str(buf: &mut Vec<u8>, s: &str) {
    let n = s.len() as u32;
    buf.extend_from_slice(&n.to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha() -> String {
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into()
    }

    #[test]
    fn unnormalize_multiplies_by_std_not_by_std_plus_eps() {
        let s = MeanStd { mean: 0.0, std: 0.0 };
        let y = mean_std_forward(1.0, s).unwrap();
        assert!((y - 1.0 / NORM_EPS).abs() < 1.0);
        let back = mean_std_inverse(y, s).unwrap();
        assert_eq!(back, 0.0, "inverse uses std, which is 0, not std+eps");
    }

    #[test]
    fn horizon_and_aloha_match_the_config_post_init() {
        assert!(check_horizon(51, 50).is_err());
        assert!(check_horizon(50, 50).is_ok());
        assert_eq!(check_aloha_delta(true), Err(ContractError::DeltaAlohaNotPorted));
    }

    #[test]
    fn state_pads_with_zeros_and_refuses_past_the_projection_width() {
        let padded = pad_state(&[1.0, 2.0]).unwrap();
        assert_eq!(padded.len(), 32);
        assert_eq!(&padded[..2], &[1.0, 2.0]);
        assert!(padded[2..].iter().all(|v| *v == 0.0));
        assert!(pad_state(&vec![0.0; 33]).is_err());
    }

    #[test]
    fn siglip_map_refuses_outside_the_unit_interval() {
        assert_eq!(siglip_range(0.0).unwrap(), -1.0);
        assert_eq!(siglip_range(1.0).unwrap(), 1.0);
        assert_eq!(siglip_range(0.25).unwrap(), -0.5);
        assert!(siglip_range(1.5).is_err());
        assert!(siglip_range(-0.01).is_err());
    }

    #[test]
    fn resize_pads_on_the_top_and_the_left() {
        // 1x1 -> height 2, width 1. ratio = max(1/1, 1/2) = 1. resized stays 1x1.
        // pad_h = 1, pad_w = 0. The value sits on the bottom row.
        let out = resize_with_pad(&[0.5], 1, 1, 1, 1, 2, 1).unwrap();
        assert_eq!(out, vec![0.0, 0.5]);
    }

    #[test]
    fn newline_is_added_once() {
        assert_eq!(ensure_task_newline("pick"), "pick\n");
        assert_eq!(ensure_task_newline("pick\n"), "pick\n");
    }

    #[test]
    fn digest_covers_the_chunk_and_ignores_nothing_inside_it() {
        let chunk = ActionChunk::try_new(2, 1, vec![0.25, -0.5]).unwrap();
        let id = ChunkIdentity {
            model_id: MODEL_ID.into(),
            weight_sha256: sha(),
            stats_sha256: sha(),
            noise_sha256: sha(),
            embodiment: "so100_follower".into(),
        };
        let (bytes, digest) = encode_action_chunk(&chunk, &id).unwrap();
        let again = encode_action_chunk(&chunk, &id).unwrap();
        assert_eq!(bytes, again.0);
        assert_eq!(digest, again.1);
        let mut other = chunk.clone();
        other.values[0] = 0.26;
        let (_, d2) = encode_action_chunk(&other, &id).unwrap();
        assert_ne!(digest, d2);
        assert_eq!(determinism_of(&id), Determinism::PinnedNoise);
        assert!(require_reference_agreement(&id).is_ok());
    }

    #[test]
    fn a_missing_noise_pin_cannot_be_reference_agreement() {
        let id = ChunkIdentity {
            model_id: MODEL_ID.into(),
            weight_sha256: sha(),
            stats_sha256: sha(),
            noise_sha256: String::new(),
            embodiment: "so100_follower".into(),
        };
        assert_eq!(determinism_of(&id), Determinism::Stochastic);
        assert_eq!(require_reference_agreement(&id), Err(ContractError::StochasticAgreement));
    }

    #[test]
    fn the_inference_entry_is_proposed_and_refuses() {
        assert_eq!(support_of_entry(), Support::Proposed);
        assert_eq!(predict_action_chunk(), Err(ContractError::ProposedEntryInvoked));
        assert_eq!(INFERENCE_ENTRY_POINT, "ferric.smolvla.predict_action_chunk.v1");
    }

    #[test]
    fn flow_steps_are_not_the_action_horizon() {
        assert_eq!(FLOW_NUM_STEPS, 10);
        assert_eq!(CHUNK_SIZE, 50);
        assert_ne!(FLOW_NUM_STEPS, CHUNK_SIZE);
    }
}
