//! Version 3 protocol tests. They are not a model output.

use ferric_accept::smolvla::{LEROBOT_COMMIT, MODEL_ID};
use ferric_accept::smolvla_v3::{
    ActionChunkV3, NOISE_CANONICAL_SHA256, NOISE_LEN, NOISE_PAYLOAD_SHA256, SO100_ACTION_MEAN,
    SO100_ACTION_STD, SO100_DT_DEN, SO100_DT_NUM, SO100_PERIOD_DEN, SO100_PERIOD_NUM,
    STATE_MEAN_SHA256, STATE_STD_SHA256, STATS_FILE_SHA256, V3Error, WEIGHT_SHA256,
    decode_action_chunk_v3, encode_action_chunk_v3, noise_canonical_bytes, noise_sha256,
    so100_joints, t0_grid_k,
};
use ferric_accept::{repo_root, sha256};

fn finite_noise() -> Vec<f32> {
    vec![0.0; NOISE_LEN]
}

fn chunk_for(seed: u64, noise: &[f32]) -> ActionChunkV3 {
    ActionChunkV3 {
        lerobot_commit: LEROBOT_COMMIT.into(),
        model_id: MODEL_ID.into(),
        weight_sha256: WEIGHT_SHA256.into(),
        stats_sha256: STATS_FILE_SHA256.into(),
        stats_key: "so100".into(),
        action_mean_sha256: SO100_ACTION_MEAN.into(),
        action_std_sha256: SO100_ACTION_STD.into(),
        state_mean_sha256: STATE_MEAN_SHA256.into(),
        state_std_sha256: STATE_STD_SHA256.into(),
        noise_sha256: noise_sha256(seed, noise).unwrap(),
        seed,
        embodiment: "so100".into(),
        t0_num_ns: SO100_PERIOD_NUM,
        t0_den: SO100_PERIOD_DEN,
        period_num_ns: SO100_PERIOD_NUM,
        period_den: SO100_PERIOD_DEN,
        n_action_steps: 1,
        action_dim: 6,
        joints: so100_joints(),
        values: vec![0.0, 0.0, 0.0, 50.0, 90.0, 10.0],
    }
}

fn encode(seed: u64, noise: &[f32]) -> (Vec<u8>, String) {
    let chunk = chunk_for(seed, noise);
    encode_action_chunk_v3(&chunk, noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap()
}

#[test]
fn v3_encoder_refuses_an_even_seed_and_accepts_the_next_odd_seed() {
    let noise = finite_noise();
    assert_eq!(noise_sha256(2, &noise), Err(V3Error::EvenSeed));
    let even = chunk_for(3, &noise);
    let mut even = even;
    even.seed = 2;
    assert_eq!(
        encode_action_chunk_v3(&even, &noise, 0, SO100_DT_NUM, SO100_DT_DEN),
        Err(V3Error::EvenSeed)
    );
    let (bytes, digest) = encode(3, &noise);
    assert_eq!(digest, hex_of(&bytes));
    let back = decode_action_chunk_v3(&bytes).unwrap();
    assert_eq!(back.seed, 3);
    assert_eq!(back.stats_key, "so100");
    assert_eq!(
        t0_grid_k(
            back.t0_num_ns,
            back.t0_den,
            back.period_num_ns,
            back.period_den,
            0
        )
        .unwrap(),
        1
    );
}

#[test]
fn v3_decoder_refuses_an_even_seed() {
    let noise = finite_noise();
    let (mut bytes, _) = encode(3, &noise);
    // The odd seed is the u64 immediately after the 64-byte noise sha256.
    let marker = noise_sha256(3, &noise).unwrap();
    let at = bytes
        .windows(64)
        .position(|w| w == marker.as_bytes())
        .unwrap();
    bytes[at + 64..at + 72].copy_from_slice(&2u64.to_le_bytes());
    assert_eq!(decode_action_chunk_v3(&bytes), Err(V3Error::EvenSeed));
}

#[test]
fn v3_round_trip_keeps_the_canonical_digest() {
    let noise = finite_noise();
    let chunk = chunk_for(3, &noise);
    let (bytes, digest) =
        encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap();
    assert_eq!(digest, hex_of(&bytes));
    let back = decode_action_chunk_v3(&bytes).unwrap();
    let (again, digest2) =
        encode_action_chunk_v3(&back, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap();
    assert_eq!(bytes, again);
    assert_eq!(digest, digest2);
}

#[test]
fn v3_refuses_every_malformed_field() {
    let noise = finite_noise();
    let mut chunk = chunk_for(3, &noise);
    chunk.weight_sha256 = "A".repeat(64);
    assert_eq!(
        encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap_err(),
        V3Error::BadHash
    );
    chunk = chunk_for(3, &noise);
    chunk.stats_key = "so100-blue".into();
    assert_eq!(
        encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap_err(),
        V3Error::StatsKeyMismatch
    );
    chunk = chunk_for(3, &noise);
    chunk.embodiment = "so100-blue".into();
    chunk.stats_key = "so100-blue".into();
    chunk.action_mean_sha256 = ferric_accept::smolvla_v3::SO100_BLUE_ACTION_MEAN.into();
    chunk.action_std_sha256 = ferric_accept::smolvla_v3::SO100_BLUE_ACTION_STD.into();
    // embodiment so100-blue is a stats key, not this joint table's dataset embodiment.
    // The contract refuses a so100 chunk that cites so100-blue. A blue key with a blue
    // embodiment still has to carry the SO-100 joint names. This one does, so it encodes.
    assert!(encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).is_ok());
    chunk = chunk_for(3, &noise);
    chunk.t0_num_ns = 2;
    chunk.t0_den = 4;
    assert_eq!(
        encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap_err(),
        V3Error::Unreduced
    );
    chunk = chunk_for(3, &noise);
    chunk.t0_den = 0;
    assert_eq!(
        encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap_err(),
        V3Error::ZeroDenominator
    );
    chunk = chunk_for(3, &noise);
    chunk.period_num_ns = 20_000_000;
    chunk.period_den = 1;
    assert_eq!(
        encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap_err(),
        V3Error::Period
    );
    assert_eq!(
        ferric_accept::smolvla_v3::check_so100_clock(SO100_PERIOD_NUM, SO100_PERIOD_DEN, 1, 1),
        Err(V3Error::Dt)
    );
    chunk = chunk_for(3, &noise);
    chunk.n_action_steps = 0;
    chunk.values.clear();
    assert_eq!(
        encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap_err(),
        V3Error::ActionSteps
    );
    chunk = chunk_for(3, &noise);
    chunk.n_action_steps = 51;
    chunk.values = vec![0.0; 51 * 6];
    chunk.values[4] = 90.0;
    chunk.values[3] = 50.0;
    chunk.values[5] = 10.0;
    // later rows too
    for t in 0..51 {
        chunk.values[t * 6 + 3] = 50.0;
        chunk.values[t * 6 + 4] = 90.0;
        chunk.values[t * 6 + 5] = 10.0;
    }
    assert_eq!(
        encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap_err(),
        V3Error::ActionSteps
    );
    chunk = chunk_for(3, &noise);
    chunk.joints[5].unit = "deg".into();
    assert_eq!(
        encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap_err(),
        V3Error::JointUnit
    );
    chunk = chunk_for(3, &noise);
    chunk.joints[0].name = "shoulder_pan.pos".into();
    assert_eq!(
        encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap_err(),
        V3Error::JointName
    );
    chunk = chunk_for(3, &noise);
    chunk.values[5] = 101.0;
    assert_eq!(
        encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap_err(),
        V3Error::ValueOutsideLimit
    );
    chunk = chunk_for(3, &noise);
    chunk.values[0] = f32::NAN;
    assert_eq!(
        encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap_err(),
        V3Error::NonFiniteValue
    );
    let mut bad_noise = finite_noise();
    bad_noise[0] = f32::INFINITY;
    assert_eq!(noise_sha256(3, &bad_noise), Err(V3Error::NonFinitePayload));
    chunk = chunk_for(3, &noise);
    chunk.t0_num_ns = 1;
    chunk.t0_den = 1;
    assert_eq!(
        encode_action_chunk_v3(&chunk, &noise, 1_000, SO100_DT_NUM, SO100_DT_DEN).unwrap_err(),
        V3Error::BeforeOrigin
    );

    let (mut bytes, _) = encode(3, &noise);
    bytes[0] ^= 1;
    assert_eq!(decode_action_chunk_v3(&bytes), Err(V3Error::BadMagic));
    let (mut bytes, _) = encode(3, &noise);
    bytes.push(0);
    assert_eq!(decode_action_chunk_v3(&bytes), Err(V3Error::TrailingBytes));
    let (mut bytes, _) = encode(3, &noise);
    bytes.pop();
    assert_eq!(decode_action_chunk_v3(&bytes), Err(V3Error::Truncated));
    let (bytes, digest) = encode(3, &noise);
    let mut version = bytes.clone();
    version[CHUNK_OFF..CHUNK_OFF + 4].copy_from_slice(&2u32.to_le_bytes());
    assert_eq!(decode_action_chunk_v3(&version), Err(V3Error::BadVersion));
    assert_eq!(digest, hex_of(&bytes));
}

const CHUNK_OFF: usize = 21;

#[test]
fn v3_noise_string_for_a_finite_payload_is_the_seven_item_layout() {
    let noise = finite_noise();
    let bytes = noise_canonical_bytes(3, &noise);
    assert_eq!(&bytes[..7], &[3, 0, 0, 0, b'f', b'3', b'2']);
    assert!(noise_sha256(3, &noise).is_ok());
    assert_ne!(NOISE_PAYLOAD_SHA256, NOISE_CANONICAL_SHA256);
}

fn hex_of(bytes: &[u8]) -> String {
    ferric_accept::hex(&sha256(bytes))
}

#[test]
fn v3_golden_on_the_30hz_grid_matches_the_documented_digest() {
    let path = repo_root().join("docs/acceptance/contracts/fixtures/smolvla-chunk-v3.json");
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(v["clock"], "100000000/3");
    assert_eq!(v["origin_ns"], 0);
    assert_eq!(v["t0_num_ns"], SO100_PERIOD_NUM);
    assert_eq!(v["t0_den"], SO100_PERIOD_DEN);
    let noise = finite_noise();
    // The documented golden uses seed 3 and a finite zero payload so the file
    // does not carry the 6400-byte x_T. Seed 0x5a17 is checked against the
    // generator in ferric-smolvla.
    assert_eq!(v["seed"], 3);
    let chunk = chunk_for(3, &noise);
    let (bytes, digest) =
        encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap();
    assert_eq!(digest, v["digest_sha256"].as_str().unwrap());
    assert_eq!(to_hex(&bytes), v["canonical_hex"].as_str().unwrap());
    let back = decode_action_chunk_v3(&bytes).unwrap();
    assert_eq!(back.period_num_ns, 100_000_000);
    assert_eq!(back.period_den, 3);
}

fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}
