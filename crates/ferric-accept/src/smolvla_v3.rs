//! Version 3 of the SmolVLA action-chunk codec.
//!
//! The byte layout is `docs/acceptance/contracts/smolvla-ferromotion-v3.md`.
//! Version 2 [`super::smolvla::encode_action_chunk`] is a different function and
//! is not used here. The noise payload is not a byte of the chunk. The encoder
//! hashes it into `noise_sha256` and then drops it.

use crate::sha256::{hex, sha256};

pub const CHUNK_MAGIC: &[u8] = b"FERRIC-SMOLVLA-CHUNK\0";
pub const CHUNK_VERSION_V3: u32 = 3;
pub const SO100_PERIOD_NUM: u64 = 100_000_000;
pub const SO100_PERIOD_DEN: u64 = 3;
pub const SO100_DT_NUM: u64 = 5_000_000;
pub const SO100_DT_DEN: u64 = 3;
pub const NOISE_LEN: usize = 50 * 32;

pub const WEIGHT_SHA256: &str = "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb";
pub const STATS_FILE_SHA256: &str =
    "490ab239d96e263687c0b2e386a0afbc235a2eceb9857c36ed32f2f162a3e7c8";
pub const SO100_ACTION_MEAN: &str =
    "e22c3af117ece699b816d50f60df8a1e7e4728e1babf5297634b74b8483ad924";
pub const SO100_ACTION_STD: &str =
    "a09262820c5532e2fffb6da68c7ee6d9670d0da7080f70aadd0b34b7af8755b9";
pub const SO100_BLUE_ACTION_MEAN: &str =
    "8f178ffad9bca313223b2ebf56102779a017dbb0409c557259fa8888e02155fb";
pub const SO100_BLUE_ACTION_STD: &str =
    "a4762d32295adf3bf7d652ea86cb3e7f2cc575b496e7e30a220a572e927a6820";
pub const SO100_RED_ACTION_MEAN: &str =
    "f85d5119f589c058249515c89ed780b4bba4fdaa30b81277bb738e513b7e6cc8";
pub const SO100_RED_ACTION_STD: &str =
    "4deb2fcd2af0aa4667f6b09748b5e527e543ca36c13e204b75273e7cc62c8bf6";
pub const STATE_MEAN_SHA256: &str =
    "c0087077ccbf5362ee8dab2ec5cae2ee915123b176ade6c0be1fad3feb7a7eca";
pub const STATE_STD_SHA256: &str =
    "09d52284cad4193fdaf1fe35c89dab6aa2711514b3043296b7811c13cb8f13ef";
pub const NOISE_PAYLOAD_SHA256: &str =
    "09adba264f855351fd2a0911ce66497cba8e54f9ca73d6b8d905c22918e0e4c2";
pub const NOISE_CANONICAL_SHA256: &str =
    "8db2ebc813c83673d6b61b10b2c2a2352018371cbfe9c661f3b405ac6370b61c";

const JOINTS: &[(&str, &str, f32, f32)] = &[
    ("main_shoulder_pan", "deg", -37.177734375, 72.7734375),
    ("main_shoulder_lift", "deg", -89.47265625, 41.1328125),
    ("main_elbow_flex", "deg", -49.04296875, 74.619140625),
    ("main_wrist_flex", "deg", 9.66796875, 96.591796875),
    ("main_wrist_roll", "deg", 56.25, 123.57421875),
    ("main_gripper", "percent_range", 0.0, 100.0),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum V3Error {
    EvenSeed,
    NonFinitePayload,
    NoiseLength,
    NoiseHashMismatch,
    MissingHash,
    BadHash,
    BadStatsKey,
    StatsKeyMismatch,
    Embodiment,
    Unreduced,
    ZeroDenominator,
    Period,
    Dt,
    BeforeOrigin,
    OffGrid,
    Overflow,
    ActionSteps,
    ActionDim,
    JointCount,
    JointName,
    JointUnit,
    JointFrame,
    JointLimit,
    ValueOutsideLimit,
    NonFiniteValue,
    BadMagic,
    BadVersion,
    Truncated,
    TrailingBytes,
    BadUtf8,
    BadDtype,
    BadOrder,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JointV3 {
    pub name: String,
    pub unit: String,
    pub frame: String,
    pub min: f32,
    pub max: f32,
}

/// One version 3 chunk. The noise payload is not stored here.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionChunkV3 {
    pub lerobot_commit: String,
    pub model_id: String,
    pub weight_sha256: String,
    pub stats_sha256: String,
    pub stats_key: String,
    pub action_mean_sha256: String,
    pub action_std_sha256: String,
    pub state_mean_sha256: String,
    pub state_std_sha256: String,
    pub noise_sha256: String,
    pub seed: u64,
    pub embodiment: String,
    pub t0_num_ns: u64,
    pub t0_den: u64,
    pub period_num_ns: u64,
    pub period_den: u64,
    pub n_action_steps: u32,
    pub action_dim: u32,
    pub joints: Vec<JointV3>,
    pub values: Vec<f32>,
}

pub fn so100_joints() -> Vec<JointV3> {
    JOINTS
        .iter()
        .map(|(name, unit, min, max)| JointV3 {
            name: (*name).into(),
            unit: (*unit).into(),
            frame: "so100_joint".into(),
            min: *min,
            max: *max,
        })
        .collect()
}

/// SHA-256 of the seven-item noise string. Refuses an even seed, a payload
/// whose length is not 1600, and any non-finite sample.
pub fn noise_sha256(seed: u64, payload: &[f32]) -> Result<String, V3Error> {
    if seed & 1 == 0 {
        return Err(V3Error::EvenSeed);
    }
    if payload.len() != NOISE_LEN {
        return Err(V3Error::NoiseLength);
    }
    if payload.iter().any(|v| !v.is_finite()) {
        return Err(V3Error::NonFinitePayload);
    }
    Ok(hex(&sha256(&noise_canonical_bytes(seed, payload))))
}

pub fn noise_canonical_bytes(seed: u64, payload: &[f32]) -> Vec<u8> {
    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }
    let mut out = Vec::new();
    put_str(&mut out, "f32");
    out.extend_from_slice(&3u32.to_le_bytes());
    for extent in [1u64, 50, 32] {
        out.extend_from_slice(&extent.to_le_bytes());
    }
    put_str(&mut out, "le");
    for v in payload {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&seed.to_le_bytes());
    put_str(&mut out, "xorshift64-box-muller-v1");
    out
}

/// The ferric SO-100 profile. The envelope may carry any other reduced pair.
pub fn check_so100_clock(
    period_num: u64,
    period_den: u64,
    dt_num: u64,
    dt_den: u64,
) -> Result<(), V3Error> {
    check_reduced(period_num, period_den)?;
    check_reduced(dt_num, dt_den)?;
    if period_num != SO100_PERIOD_NUM || period_den != SO100_PERIOD_DEN {
        return Err(V3Error::Period);
    }
    if dt_num != SO100_DT_NUM || dt_den != SO100_DT_DEN {
        return Err(V3Error::Dt);
    }
    Ok(())
}

/// `k` such that `t0 = origin_ns + k · period`, or an error.
pub fn t0_grid_k(
    t0_num: u64,
    t0_den: u64,
    period_num: u64,
    period_den: u64,
    origin_ns: u64,
) -> Result<u64, V3Error> {
    check_reduced(t0_num, t0_den)?;
    check_reduced(period_num, period_den)?;
    let left = u128::from(t0_num)
        .checked_mul(u128::from(period_den))
        .ok_or(V3Error::Overflow)?;
    let origin_term = u128::from(origin_ns)
        .checked_mul(u128::from(t0_den))
        .ok_or(V3Error::Overflow)?
        .checked_mul(u128::from(period_den))
        .ok_or(V3Error::Overflow)?;
    if left < origin_term {
        return Err(V3Error::BeforeOrigin);
    }
    let d = left - origin_term;
    let divisor = u128::from(period_num)
        .checked_mul(u128::from(t0_den))
        .ok_or(V3Error::Overflow)?;
    if divisor == 0 || d % divisor != 0 {
        return Err(V3Error::OffGrid);
    }
    let k = d / divisor;
    u64::try_from(k).map_err(|_| V3Error::Overflow)
}

/// Canonical bytes and `chunk_sha256`. `noise` is hashed and not written.
/// `origin_ns` and `dt_*` are the episode clock beside the chunk. They are not bytes of it.
pub fn encode_action_chunk_v3(
    chunk: &ActionChunkV3,
    noise: &[f32],
    origin_ns: u64,
    dt_num_ns: u64,
    dt_den: u64,
) -> Result<(Vec<u8>, String), V3Error> {
    check_chunk(chunk, noise, origin_ns, dt_num_ns, dt_den)?;
    let mut buf = Vec::new();
    write_chunk(&mut buf, chunk);
    let digest = hex(&sha256(&buf));
    Ok((buf, digest))
}

pub fn decode_action_chunk_v3(bytes: &[u8]) -> Result<ActionChunkV3, V3Error> {
    if bytes.len() < CHUNK_MAGIC.len() || &bytes[..CHUNK_MAGIC.len()] != CHUNK_MAGIC {
        return Err(V3Error::BadMagic);
    }
    let mut i = CHUNK_MAGIC.len();
    let version = read_u32(bytes, &mut i)?;
    if version != CHUNK_VERSION_V3 {
        return Err(V3Error::BadVersion);
    }
    let lerobot_commit = read_str(bytes, &mut i)?;
    let model_id = read_str(bytes, &mut i)?;
    let weight_sha256 = read_hash(bytes, &mut i)?;
    let stats_sha256 = read_hash(bytes, &mut i)?;
    let stats_key = read_str(bytes, &mut i)?;
    let action_mean_sha256 = read_hash(bytes, &mut i)?;
    let action_std_sha256 = read_hash(bytes, &mut i)?;
    let state_mean_sha256 = read_hash(bytes, &mut i)?;
    let state_std_sha256 = read_hash(bytes, &mut i)?;
    let noise_sha256 = read_hash(bytes, &mut i)?;
    let seed = read_u64(bytes, &mut i)?;
    let embodiment = read_str(bytes, &mut i)?;
    let t0_num_ns = read_u64(bytes, &mut i)?;
    let t0_den = read_u64(bytes, &mut i)?;
    let period_num_ns = read_u64(bytes, &mut i)?;
    let period_den = read_u64(bytes, &mut i)?;
    let n_action_steps = read_u32(bytes, &mut i)?;
    let action_dim = read_u32(bytes, &mut i)?;
    // Header counts are untrusted. Check them before any Vec is sized from them.
    if !(1..=50).contains(&n_action_steps) {
        return Err(V3Error::ActionSteps);
    }
    if action_dim != 6 {
        return Err(V3Error::ActionDim);
    }
    let joint_count = read_u32(bytes, &mut i)?;
    if joint_count != 6 || joint_count != action_dim {
        return Err(V3Error::JointCount);
    }
    let value_count = n_action_steps as usize * action_dim as usize;
    // Six joints are at least 20 bytes each (three length prefixes and two f32s),
    // then the dtype byte, the order byte, and the f32 values.
    let min_tail = 6 * 20 + 2 + value_count * 4;
    if bytes.len().saturating_sub(i) < min_tail {
        return Err(V3Error::Truncated);
    }
    let mut joints = Vec::with_capacity(6);
    for _ in 0..6 {
        joints.push(JointV3 {
            name: read_str(bytes, &mut i)?,
            unit: read_str(bytes, &mut i)?,
            frame: read_str(bytes, &mut i)?,
            min: read_f32(bytes, &mut i)?,
            max: read_f32(bytes, &mut i)?,
        });
    }
    let dtype = read_u8(bytes, &mut i)?;
    if dtype != 1 {
        return Err(V3Error::BadDtype);
    }
    let order = read_u8(bytes, &mut i)?;
    if order != 1 {
        return Err(V3Error::BadOrder);
    }
    if bytes.len().saturating_sub(i) < value_count * 4 {
        return Err(V3Error::Truncated);
    }
    let mut values = Vec::with_capacity(value_count);
    for _ in 0..value_count {
        values.push(read_f32(bytes, &mut i)?);
    }
    if i != bytes.len() {
        return Err(V3Error::TrailingBytes);
    }
    let chunk = ActionChunkV3 {
        lerobot_commit,
        model_id,
        weight_sha256,
        stats_sha256,
        stats_key,
        action_mean_sha256,
        action_std_sha256,
        state_mean_sha256,
        state_std_sha256,
        noise_sha256,
        seed,
        embodiment,
        t0_num_ns,
        t0_den,
        period_num_ns,
        period_den,
        n_action_steps,
        action_dim,
        joints,
        values,
    };
    // Decoder checks. A zero-length noise payload is not in the chunk; finiteness of
    // the action values is. The seed refusal is.
    validate_decoded(&chunk)?;
    Ok(chunk)
}

/// Decoded version 3 chunk and SHA-256 of `bytes`.
///
/// Binding 1 calls that digest `chunk_sha256`. It is not a suffix of the bytes.
pub fn decode_action_chunk_v3_with_sha256(bytes: &[u8]) -> Result<(ActionChunkV3, String), V3Error> {
    let chunk = decode_action_chunk_v3(bytes)?;
    Ok((chunk, hex(&sha256(bytes))))
}

fn write_chunk(buf: &mut Vec<u8>, chunk: &ActionChunkV3) {
    buf.extend_from_slice(CHUNK_MAGIC);
    buf.extend_from_slice(&CHUNK_VERSION_V3.to_le_bytes());
    push_str(buf, &chunk.lerobot_commit);
    push_str(buf, &chunk.model_id);
    buf.extend_from_slice(chunk.weight_sha256.as_bytes());
    buf.extend_from_slice(chunk.stats_sha256.as_bytes());
    push_str(buf, &chunk.stats_key);
    buf.extend_from_slice(chunk.action_mean_sha256.as_bytes());
    buf.extend_from_slice(chunk.action_std_sha256.as_bytes());
    buf.extend_from_slice(chunk.state_mean_sha256.as_bytes());
    buf.extend_from_slice(chunk.state_std_sha256.as_bytes());
    buf.extend_from_slice(chunk.noise_sha256.as_bytes());
    buf.extend_from_slice(&chunk.seed.to_le_bytes());
    push_str(buf, &chunk.embodiment);
    buf.extend_from_slice(&chunk.t0_num_ns.to_le_bytes());
    buf.extend_from_slice(&chunk.t0_den.to_le_bytes());
    buf.extend_from_slice(&chunk.period_num_ns.to_le_bytes());
    buf.extend_from_slice(&chunk.period_den.to_le_bytes());
    buf.extend_from_slice(&chunk.n_action_steps.to_le_bytes());
    buf.extend_from_slice(&chunk.action_dim.to_le_bytes());
    buf.extend_from_slice(&(chunk.joints.len() as u32).to_le_bytes());
    for joint in &chunk.joints {
        push_str(buf, &joint.name);
        push_str(buf, &joint.unit);
        push_str(buf, &joint.frame);
        buf.extend_from_slice(&joint.min.to_le_bytes());
        buf.extend_from_slice(&joint.max.to_le_bytes());
    }
    buf.push(1);
    buf.push(1);
    for v in &chunk.values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

fn check_chunk(
    chunk: &ActionChunkV3,
    noise: &[f32],
    origin_ns: u64,
    dt_num_ns: u64,
    dt_den: u64,
) -> Result<(), V3Error> {
    if chunk.seed & 1 == 0 {
        return Err(V3Error::EvenSeed);
    }
    refuse_placeholder_hashes(chunk)?;
    let expect = noise_sha256(chunk.seed, noise)?;
    if chunk.noise_sha256 != expect {
        return Err(V3Error::NoiseHashMismatch);
    }
    validate_decoded(chunk)?;
    check_so100_clock(chunk.period_num_ns, chunk.period_den, dt_num_ns, dt_den)?;
    let _k = t0_grid_k(
        chunk.t0_num_ns,
        chunk.t0_den,
        chunk.period_num_ns,
        chunk.period_den,
        origin_ns,
    )?;
    for step in 0..chunk.n_action_steps {
        step_numerator(chunk, step)?;
    }
    Ok(())
}

fn validate_decoded(chunk: &ActionChunkV3) -> Result<(), V3Error> {
    if chunk.seed & 1 == 0 {
        return Err(V3Error::EvenSeed);
    }
    if chunk.lerobot_commit != super::smolvla::LEROBOT_COMMIT
        || chunk.model_id != super::smolvla::MODEL_ID
    {
        return Err(V3Error::Embodiment);
    }
    refuse_placeholder_hashes(chunk)?;
    for hash in [
        chunk.weight_sha256.as_str(),
        chunk.stats_sha256.as_str(),
        chunk.action_mean_sha256.as_str(),
        chunk.action_std_sha256.as_str(),
        chunk.state_mean_sha256.as_str(),
        chunk.state_std_sha256.as_str(),
        chunk.noise_sha256.as_str(),
    ] {
        if !is_sha256(hash) {
            return Err(V3Error::BadHash);
        }
    }
    if chunk.weight_sha256 != WEIGHT_SHA256 || chunk.stats_sha256 != STATS_FILE_SHA256 {
        return Err(V3Error::BadHash);
    }
    check_stats(chunk)?;
    if chunk.embodiment != chunk.stats_key {
        return Err(V3Error::Embodiment);
    }
    check_reduced(chunk.t0_num_ns, chunk.t0_den)?;
    check_so100_clock(
        chunk.period_num_ns,
        chunk.period_den,
        SO100_DT_NUM,
        SO100_DT_DEN,
    )?;
    if !(1..=50).contains(&chunk.n_action_steps) {
        return Err(V3Error::ActionSteps);
    }
    if chunk.action_dim != 6 {
        return Err(V3Error::ActionDim);
    }
    if chunk.joints.len() != chunk.action_dim as usize {
        return Err(V3Error::JointCount);
    }
    if chunk.values.len() != chunk.n_action_steps as usize * chunk.action_dim as usize {
        return Err(V3Error::ActionDim);
    }
    let expect = so100_joints();
    for (i, joint) in chunk.joints.iter().enumerate() {
        let want = &expect[i];
        if joint.name != want.name {
            return Err(V3Error::JointName);
        }
        if joint.unit != want.unit {
            return Err(V3Error::JointUnit);
        }
        if joint.frame != want.frame {
            return Err(V3Error::JointFrame);
        }
        if joint.min.to_bits() != want.min.to_bits() || joint.max.to_bits() != want.max.to_bits() {
            return Err(V3Error::JointLimit);
        }
    }
    for t in 0..chunk.n_action_steps {
        for d in 0..chunk.action_dim {
            let v = chunk.values[(t * chunk.action_dim + d) as usize];
            if !v.is_finite() {
                return Err(V3Error::NonFiniteValue);
            }
            let joint = &chunk.joints[d as usize];
            if v < joint.min || v > joint.max {
                return Err(V3Error::ValueOutsideLimit);
            }
        }
        step_numerator(chunk, t)?;
    }
    Ok(())
}

fn step_numerator(chunk: &ActionChunkV3, step: u32) -> Result<(), V3Error> {
    let k = u128::from(step);
    let _num = u128::from(chunk.t0_num_ns)
        .checked_mul(u128::from(chunk.period_den))
        .ok_or(V3Error::Overflow)?
        .checked_add(
            k.checked_mul(u128::from(chunk.period_num_ns))
                .ok_or(V3Error::Overflow)?
                .checked_mul(u128::from(chunk.t0_den))
                .ok_or(V3Error::Overflow)?,
        )
        .ok_or(V3Error::Overflow)?;
    let _den = u128::from(chunk.t0_den)
        .checked_mul(u128::from(chunk.period_den))
        .ok_or(V3Error::Overflow)?;
    Ok(())
}

fn check_stats(chunk: &ActionChunkV3) -> Result<(), V3Error> {
    let (mean, std) = match chunk.stats_key.as_str() {
        "so100" => (SO100_ACTION_MEAN, SO100_ACTION_STD),
        "so100-blue" => (SO100_BLUE_ACTION_MEAN, SO100_BLUE_ACTION_STD),
        "so100-red" => (SO100_RED_ACTION_MEAN, SO100_RED_ACTION_STD),
        _ => return Err(V3Error::BadStatsKey),
    };
    if chunk.action_mean_sha256 != mean || chunk.action_std_sha256 != std {
        return Err(V3Error::StatsKeyMismatch);
    }
    if chunk.state_mean_sha256 != STATE_MEAN_SHA256 || chunk.state_std_sha256 != STATE_STD_SHA256 {
        return Err(V3Error::StatsKeyMismatch);
    }
    Ok(())
}

fn check_reduced(num: u64, den: u64) -> Result<(), V3Error> {
    if den == 0 {
        return Err(V3Error::ZeroDenominator);
    }
    if gcd(num, den) != 1 {
        return Err(V3Error::Unreduced);
    }
    Ok(())
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn refuse_placeholder_hashes(chunk: &ActionChunkV3) -> Result<(), V3Error> {
    for hash in [
        chunk.weight_sha256.as_str(),
        chunk.stats_sha256.as_str(),
        chunk.action_mean_sha256.as_str(),
        chunk.action_std_sha256.as_str(),
        chunk.state_mean_sha256.as_str(),
        chunk.state_std_sha256.as_str(),
        chunk.noise_sha256.as_str(),
    ] {
        if is_placeholder_hash(hash) {
            return Err(V3Error::MissingHash);
        }
    }
    Ok(())
}

/// `"11" * 32` and `"22" * 32`. Those strings are not values of any header hash.
fn is_placeholder_hash(s: &str) -> bool {
    s.len() == 64 && (s.bytes().all(|c| c == b'1') || s.bytes().all(|c| c == b'2'))
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

fn push_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn read_exact<'a>(bytes: &'a [u8], i: &mut usize, n: usize) -> Result<&'a [u8], V3Error> {
    let end = i.checked_add(n).ok_or(V3Error::Truncated)?;
    if end > bytes.len() {
        return Err(V3Error::Truncated);
    }
    let s = &bytes[*i..end];
    *i = end;
    Ok(s)
}

fn read_u8(bytes: &[u8], i: &mut usize) -> Result<u8, V3Error> {
    Ok(read_exact(bytes, i, 1)?[0])
}

fn read_u32(bytes: &[u8], i: &mut usize) -> Result<u32, V3Error> {
    let s = read_exact(bytes, i, 4)?;
    Ok(u32::from_le_bytes(s.try_into().unwrap()))
}

fn read_u64(bytes: &[u8], i: &mut usize) -> Result<u64, V3Error> {
    let s = read_exact(bytes, i, 8)?;
    Ok(u64::from_le_bytes(s.try_into().unwrap()))
}

fn read_f32(bytes: &[u8], i: &mut usize) -> Result<f32, V3Error> {
    let s = read_exact(bytes, i, 4)?;
    Ok(f32::from_le_bytes(s.try_into().unwrap()))
}

fn read_hash(bytes: &[u8], i: &mut usize) -> Result<String, V3Error> {
    let s = read_exact(bytes, i, 64)?;
    let text = std::str::from_utf8(s).map_err(|_| V3Error::BadUtf8)?;
    if is_placeholder_hash(text) {
        return Err(V3Error::MissingHash);
    }
    if !is_sha256(text) {
        return Err(V3Error::BadHash);
    }
    Ok(text.to_string())
}

fn read_str(bytes: &[u8], i: &mut usize) -> Result<String, V3Error> {
    let n = read_u32(bytes, i)? as usize;
    let rest = bytes.len().checked_sub(*i).ok_or(V3Error::Truncated)?;
    if n > rest {
        return Err(V3Error::Truncated);
    }
    let s = read_exact(bytes, i, n)?;
    std::str::from_utf8(s)
        .map(|t| t.to_string())
        .map_err(|_| V3Error::BadUtf8)
}
