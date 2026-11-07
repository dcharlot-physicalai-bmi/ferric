//! Protocol regression for the published v2 empty horizon. Not a model output.

use ferric_accept::smolvla::{
    action_timestamp_ns, authorize_rollout, check_horizon, encode_action_chunk, ActionChunk, ChunkIdentity,
    ContractError, JointMeta, CHUNK_CODEC_VERSION, LEROBOT_COMMIT,
};
use ferric_accept::{repo_root, sha256};

#[test]
fn published_nonempty_golden_is_unchanged() {
    let v = load("smolvla-chunk-v2.json");
    assert_eq!(
        v["golden"]["digest_sha256"],
        "6d668d7e25c89249c890d895ef15b8fdcbb97dab7a1339a0a97605c6e1172d86"
    );
    assert_eq!(v["golden"]["n_action_steps"], 2);
    assert_eq!(v["golden"]["values_time_major"].as_array().unwrap().len(), 4);
}

#[test]
fn empty_horizon_matches_the_recomputed_fixture() {
    let v = load("smolvla-empty-chunk-v2.json");
    assert_eq!(v["kind"], "protocol_test");
    assert_eq!(v["not_model_evidence"], true);
    assert_eq!(v["not_rollout_evidence"], true);
    assert_eq!(v["not_energy_evidence"], true);
    assert_eq!(v["model_status"], "proposed");
    assert_eq!(v["codec_version"], CHUNK_CODEC_VERSION);
    assert_eq!(v["lerobot_commit"], LEROBOT_COMMIT);
    for step in v["horizon_checks"].as_array().unwrap() {
        let got = check_horizon(step["n_action_steps"].as_u64().unwrap() as u32, step["chunk_size"].as_u64().unwrap() as u32);
        assert_eq!(got.is_ok(), step["ok"].as_bool().unwrap());
    }
    let (chunk, id) = chunk_from(&v["golden"]);
    assert_eq!(chunk.n_action_steps, 0);
    assert_eq!(chunk.action_dim, 2);
    assert!(chunk.values.is_empty());
    let (bytes, digest) = encode_action_chunk(&chunk, &id).unwrap();
    assert_eq!(digest, v["golden"]["digest_sha256"].as_str().unwrap());
    assert_eq!(hex_encode(&bytes), v["golden"]["canonical_hex"].as_str().unwrap());
    assert_eq!(hex_encode(&sha256(&bytes)), digest);
    assert_eq!(
        action_timestamp_ns(id.t0_ns, id.step_duration_ns, 0, 0).unwrap_err(),
        ContractError::TimestampIndex { index: 0, steps: 0 }
    );
    assert_eq!(authorize_rollout(&chunk, &id).unwrap_err(), ContractError::ProposedEntryInvoked);
}

#[test]
fn empty_horizon_negative_cases_refuse() {
    let v = load("smolvla-empty-chunk-v2.json");
    let (chunk, id) = chunk_from(&v["golden"]);
    for case in v["negative_cases"].as_array().unwrap() {
        match case["case"].as_str().unwrap() {
            "missing_weight_provenance" => {
                let mut bad = id.clone();
                bad.weight_sha256.clear();
                assert_eq!(encode_action_chunk(&chunk, &bad).unwrap_err(), ContractError::MissingWeightProvenance);
                assert_eq!(authorize_rollout(&chunk, &bad).unwrap_err(), ContractError::MissingWeightProvenance);
            }
            "missing_stats_provenance" => {
                let mut bad = id.clone();
                bad.stats_sha256.clear();
                assert_eq!(encode_action_chunk(&chunk, &bad).unwrap_err(), ContractError::MissingStatsProvenance);
            }
            "missing_noise_provenance" => {
                let mut bad = id.clone();
                bad.noise_sha256.clear();
                assert_eq!(encode_action_chunk(&chunk, &bad).unwrap_err(), ContractError::MissingNoiseProvenance);
                assert_eq!(authorize_rollout(&chunk, &bad).unwrap_err(), ContractError::MissingNoiseProvenance);
            }
            "nonempty_values_at_zero_horizon" => {
                let values: Vec<f32> = case["values_time_major"].as_array().unwrap().iter().map(|n| n.as_f64().unwrap() as f32).collect();
                assert!(matches!(
                    ActionChunk::try_new(0, chunk.action_dim, values),
                    Err(ContractError::ChunkShape { time: 0, .. })
                ));
            }
            "zero_step_duration" => {
                let mut bad = id.clone();
                bad.step_duration_ns = 0;
                assert_eq!(encode_action_chunk(&chunk, &bad).unwrap_err(), ContractError::StepDurationAbsent);
            }
            other => panic!("unpinned negative case {other}"),
        }
    }
}

fn load(name: &str) -> serde_json::Value {
    let path = repo_root().join("docs/acceptance/contracts/fixtures").join(name);
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn chunk_from(g: &serde_json::Value) -> (ActionChunk, ChunkIdentity) {
    let values: Vec<f32> = g["values_time_major"].as_array().unwrap().iter().map(|n| n.as_f64().unwrap() as f32).collect();
    let chunk = ActionChunk::try_new(g["n_action_steps"].as_u64().unwrap() as u32, g["action_dim"].as_u64().unwrap() as u32, values).unwrap();
    let joints = g["joints"].as_array().unwrap().iter().map(|j| JointMeta {
        name: j["name"].as_str().unwrap().into(),
        unit: j["unit"].as_str().unwrap().into(),
        frame: j["frame"].as_str().unwrap().into(),
        min: j["min"].as_f64().unwrap() as f32,
        max: j["max"].as_f64().unwrap() as f32,
    }).collect();
    let id = ChunkIdentity {
        model_id: g["model_id"].as_str().unwrap().into(),
        weight_sha256: g["weight_sha256"].as_str().unwrap().into(),
        stats_sha256: g["stats_sha256"].as_str().unwrap().into(),
        noise_sha256: g["noise_sha256"].as_str().unwrap().into(),
        embodiment: g["embodiment"].as_str().unwrap().into(),
        t0_ns: g["t0_ns"].as_u64().unwrap(),
        step_duration_ns: g["step_duration_ns"].as_u64().unwrap(),
        joints,
    };
    (chunk, id)
}

fn hex_encode(bytes: &[u8]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(H[(b >> 4) as usize] as char);
        s.push(H[(b & 0xf) as usize] as char);
    }
    s
}
