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

struct V3Spans {
    hashes: [usize; 7],
    n_action_steps: usize,
    action_dim: usize,
    joint_count: usize,
    first_value: usize,
}

fn spans(bytes: &[u8]) -> V3Spans {
    let mut i = 25;
    let skip_str = |bytes: &[u8], i: &mut usize| {
        let n = u32::from_le_bytes(bytes[*i..*i + 4].try_into().unwrap()) as usize;
        *i += 4 + n;
    };
    skip_str(bytes, &mut i);
    skip_str(bytes, &mut i);
    let mut hashes = [0usize; 7];
    hashes[0] = i;
    i += 64;
    hashes[1] = i;
    i += 64;
    skip_str(bytes, &mut i);
    for slot in 2..7 {
        hashes[slot] = i;
        i += 64;
    }
    i += 8;
    skip_str(bytes, &mut i);
    i += 32;
    let n_action_steps = i;
    i += 4;
    let action_dim = i;
    i += 4;
    let joint_count = i;
    let n = u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
    i += 4;
    for _ in 0..n {
        skip_str(bytes, &mut i);
        skip_str(bytes, &mut i);
        skip_str(bytes, &mut i);
        i += 8;
    }
    i += 2;
    V3Spans {
        hashes,
        n_action_steps,
        action_dim,
        joint_count,
        first_value: i,
    }
}

fn put_u32(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

#[test]
fn v3_decoder_refuses_unbounded_header_counts() {
    let noise = finite_noise();
    let (bytes, _) = encode(3, &noise);
    let at = spans(&bytes);

    let mut joint = bytes.clone();
    put_u32(&mut joint, at.joint_count, u32::MAX);
    assert_eq!(decode_action_chunk_v3(&joint), Err(V3Error::JointCount));

    for steps in [0u32, 51, u32::MAX] {
        let mut patched = bytes.clone();
        put_u32(&mut patched, at.n_action_steps, steps);
        assert_eq!(
            decode_action_chunk_v3(&patched),
            Err(V3Error::ActionSteps),
            "n_action_steps {steps}"
        );
    }

    let mut dim = bytes.clone();
    put_u32(&mut dim, at.action_dim, u32::MAX);
    assert_eq!(decode_action_chunk_v3(&dim), Err(V3Error::ActionDim));
}

#[test]
fn v3_string_length_is_checked_against_the_remaining_bytes() {
    let noise = finite_noise();
    let (mut bytes, _) = encode(3, &noise);
    // Commit length sits immediately after magic and version.
    put_u32(&mut bytes, 25, u32::MAX);
    assert_eq!(decode_action_chunk_v3(&bytes), Err(V3Error::Truncated));
}

#[test]
fn v3_encoder_and_decoder_refuse_placeholder_hashes() {
    let noise = finite_noise();
    let placeholders = ["1".repeat(64), "2".repeat(64)];
    let fields: &[fn(&mut ActionChunkV3) -> &mut String] = &[
        |c| &mut c.weight_sha256,
        |c| &mut c.stats_sha256,
        |c| &mut c.action_mean_sha256,
        |c| &mut c.action_std_sha256,
        |c| &mut c.state_mean_sha256,
        |c| &mut c.state_std_sha256,
        |c| &mut c.noise_sha256,
    ];
    for placeholder in &placeholders {
        for field in fields {
            let mut chunk = chunk_for(3, &noise);
            *field(&mut chunk) = placeholder.clone();
            assert_eq!(
                encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN),
                Err(V3Error::MissingHash)
            );
        }
        let (bytes, _) = encode(3, &noise);
        let at = spans(&bytes);
        for hash_at in at.hashes {
            let mut patched = bytes.clone();
            patched[hash_at..hash_at + 64].copy_from_slice(placeholder.as_bytes());
            assert_eq!(decode_action_chunk_v3(&patched), Err(V3Error::MissingHash));
        }
    }
}

#[test]
fn v3_refuses_wrong_frame_swapped_order_and_pos_names() {
    let noise = finite_noise();
    let mut chunk = chunk_for(3, &noise);
    chunk.joints[0].frame = "world".into();
    assert_eq!(
        encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap_err(),
        V3Error::JointFrame
    );
    let (mut bytes, _) = encode(3, &noise);
    let frame = bytes.windows(11).position(|w| w == b"so100_joint").unwrap();
    bytes[frame + 10] = b'X';
    assert_eq!(decode_action_chunk_v3(&bytes), Err(V3Error::JointFrame));

    let mut chunk = chunk_for(3, &noise);
    chunk.joints.swap(0, 1);
    assert_eq!(
        encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap_err(),
        V3Error::JointName
    );
    let mut chunk = chunk_for(3, &noise);
    chunk.joints[0].name = "main_shoulder_pan.pos".into();
    assert_eq!(
        encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap_err(),
        V3Error::JointName
    );
}

#[test]
fn v3_refuses_stats_key_embodiment_and_per_tensor_mismatch() {
    let noise = finite_noise();
    let mut chunk = chunk_for(3, &noise);
    chunk.state_mean_sha256 = SO100_ACTION_MEAN.into();
    assert_eq!(
        encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap_err(),
        V3Error::StatsKeyMismatch
    );
    let mut chunk = chunk_for(3, &noise);
    chunk.state_std_sha256 = SO100_ACTION_STD.into();
    assert_eq!(
        encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap_err(),
        V3Error::StatsKeyMismatch
    );
    let mut chunk = chunk_for(3, &noise);
    chunk.action_mean_sha256 = ferric_accept::smolvla_v3::SO100_BLUE_ACTION_MEAN.into();
    assert_eq!(
        encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap_err(),
        V3Error::StatsKeyMismatch
    );
    let mut chunk = chunk_for(3, &noise);
    chunk.embodiment = "so100-blue".into();
    assert_eq!(
        encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap_err(),
        V3Error::Embodiment
    );
    let (mut bytes, _) = encode(3, &noise);
    let marker = b"\x05\x00\x00\x00so100";
    let stats_key = bytes.windows(marker.len()).position(|w| w == marker).unwrap();
    let embodiment = bytes[stats_key + 1..]
        .windows(marker.len())
        .position(|w| w == marker)
        .unwrap()
        + stats_key
        + 1;
    bytes[embodiment + 4 + 4] = b'1';
    assert_eq!(decode_action_chunk_v3(&bytes), Err(V3Error::Embodiment));
}

#[test]
fn v3_refuses_u128_overflow() {
    let noise = finite_noise();
    let mut chunk = chunk_for(3, &noise);
    chunk.t0_num_ns = 1;
    chunk.t0_den = u64::MAX;
    assert_eq!(
        t0_grid_k(1, u64::MAX, SO100_PERIOD_NUM, SO100_PERIOD_DEN, u64::MAX),
        Err(V3Error::Overflow)
    );
    assert_eq!(
        encode_action_chunk_v3(&chunk, &noise, u64::MAX, SO100_DT_NUM, SO100_DT_DEN).unwrap_err(),
        V3Error::Overflow
    );
}

#[test]
fn v3_decoder_refuses_limit_nan_and_inf() {
    let noise = finite_noise();
    let (bytes, _) = encode(3, &noise);
    let at = spans(&bytes);
    let mut over = bytes.clone();
    over[at.first_value + 20..at.first_value + 24].copy_from_slice(&101.0f32.to_le_bytes());
    assert_eq!(decode_action_chunk_v3(&over), Err(V3Error::ValueOutsideLimit));
    let mut nan = bytes.clone();
    nan[at.first_value..at.first_value + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert_eq!(decode_action_chunk_v3(&nan), Err(V3Error::NonFiniteValue));
    let mut inf = bytes.clone();
    inf[at.first_value..at.first_value + 4].copy_from_slice(&f32::INFINITY.to_le_bytes());
    assert_eq!(decode_action_chunk_v3(&inf), Err(V3Error::NonFiniteValue));
}

#[test]
fn v3_decoder_refuses_every_truncated_prefix_and_trailing_bytes() {
    let noise = finite_noise();
    let (bytes, _) = encode(3, &noise);
    for n in 0..bytes.len() {
        assert!(
            decode_action_chunk_v3(&bytes[..n]).is_err(),
            "prefix {n} decoded"
        );
    }
    let mut extra = bytes.clone();
    extra.push(0);
    assert_eq!(decode_action_chunk_v3(&extra), Err(V3Error::TrailingBytes));
}

#[test]
fn v3_profile_refuses_every_clock_except_the_reduced_30hz_pair() {
    use ferric_accept::smolvla_v3::check_so100_clock;
    assert!(check_so100_clock(SO100_PERIOD_NUM, SO100_PERIOD_DEN, SO100_DT_NUM, SO100_DT_DEN).is_ok());
    assert_eq!(
        check_so100_clock(200_000_000, 6, SO100_DT_NUM, SO100_DT_DEN),
        Err(V3Error::Unreduced)
    );
    assert_eq!(
        check_so100_clock(SO100_PERIOD_NUM, SO100_PERIOD_DEN, 10_000_000, 6),
        Err(V3Error::Unreduced)
    );
    assert_eq!(
        check_so100_clock(20_000_000, 1, SO100_DT_NUM, SO100_DT_DEN),
        Err(V3Error::Period)
    );
    assert_eq!(
        check_so100_clock(SO100_PERIOD_NUM, SO100_PERIOD_DEN, 1, 1),
        Err(V3Error::Dt)
    );
    let noise = finite_noise();
    let mut chunk = chunk_for(3, &noise);
    chunk.period_num_ns = 200_000_000;
    chunk.period_den = 6;
    assert_eq!(
        encode_action_chunk_v3(&chunk, &noise, 0, SO100_DT_NUM, SO100_DT_DEN).unwrap_err(),
        V3Error::Unreduced
    );
    assert_eq!(
        encode_action_chunk_v3(&chunk_for(3, &noise), &noise, 0, 10_000_000, 6).unwrap_err(),
        V3Error::Unreduced
    );
}

fn from_hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn v3_fixture_record_locks_lengths_and_hashes() {
    let root = repo_root();
    let record_path = root.join("docs/acceptance/contracts/fixtures/smolvla-v3-fixture-record.json");
    let record_bytes = std::fs::read(&record_path).unwrap();
    assert_eq!(
        hex_of(&record_bytes),
        "e89abc688a40e2702049e6175f316691bbb646e64aa4f7dff3670c96e7777dd3"
    );
    let record: serde_json::Value = serde_json::from_slice(&record_bytes).unwrap();
    let protocol = &record["fixtures"][0];
    assert_eq!(protocol["canonical_len"], 943);
    assert_eq!(
        protocol["digest_sha256"],
        "4e1722a629708d81221e5c75c1946dcd28af40ab639239bde1ea3893dbb643e5"
    );
    let real = &record["fixtures"][1];
    assert_eq!(real["canonical_len"], 2119);
    assert_eq!(
        real["digest_sha256"],
        "13e82f106898b5a2b1c6fb48f487cf7d6cbe81cb2f8410a71e8f626a2f0c329b"
    );
    assert_eq!(
        real["noise_sha256"],
        "8db2ebc813c83673d6b61b10b2c2a2352018371cbfe9c661f3b405ac6370b61c"
    );
    for entry in record["fixtures"].as_array().unwrap() {
        let path = root.join(entry["path"].as_str().unwrap());
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(hex_of(&bytes), entry["file_sha256"].as_str().unwrap());
        let doc: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        if let Some(chunks) = doc.get("chunks").and_then(|v| v.as_array()) {
            let recorded = entry["chunks"].as_array().unwrap();
            assert_eq!(chunks.len(), 2);
            assert!(chunks[0]["k"].as_u64().unwrap() < chunks[1]["k"].as_u64().unwrap());
            assert_eq!(chunks[0]["chunk_index"], 0);
            assert_eq!(chunks[1]["chunk_index"], 1);
            assert_eq!(doc["action_values"], "fixture values, not model output");
            for (got, want) in chunks.iter().zip(recorded) {
                let raw = from_hex(got["canonical_hex"].as_str().unwrap());
                assert_eq!(raw.len(), want["canonical_len"].as_u64().unwrap() as usize);
                let (chunk, sha) =
                    ferric_accept::smolvla_v3::decode_action_chunk_v3_with_sha256(&raw).unwrap();
                assert_eq!(sha, want["digest_sha256"].as_str().unwrap());
                assert_eq!(sha, got["canonical_sha256"].as_str().unwrap());
                assert_eq!(chunk.noise_sha256, NOISE_CANONICAL_SHA256);
                assert_eq!(chunk.stats_key, "so100");
                assert!(chunk.values.iter().all(|v| v.is_finite()));
            }
        } else {
            let raw = from_hex(doc["canonical_hex"].as_str().unwrap());
            assert_eq!(raw.len() as u64, entry["canonical_len"].as_u64().unwrap());
            let (chunk, sha) =
                ferric_accept::smolvla_v3::decode_action_chunk_v3_with_sha256(&raw).unwrap();
            assert_eq!(sha, entry["digest_sha256"].as_str().unwrap());
            assert_eq!(sha, doc["digest_sha256"].as_str().unwrap());
            assert!(chunk.values.iter().all(|v| v.is_finite()));
        }
    }
    let real_doc: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join(
            "docs/acceptance/contracts/fixtures/smolvla-chunk-v3-real-noise.json",
        ))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(real_doc["action_values"], "fixture values, not model output");
    assert_eq!(real_doc["n_action_steps"], 50);
    assert_eq!(real_doc["noise_sha256"], NOISE_CANONICAL_SHA256);
    assert_eq!(real_doc["k"], 1);
}
