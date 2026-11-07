//! The handoff fixture is a protocol test. It is not a model output.

use ferric_accept::smolvla::{
    action_timestamp_ns, authorize_rollout, encode_action_chunk, ActionChunk, ChunkIdentity, ContractError,
    JointMeta, CHUNK_CODEC_VERSION, LEROBOT_COMMIT, MODEL_ID,
};
use ferric_accept::{repo_root, sha256};

#[test]
fn the_golden_fixture_matches_the_canonical_bytes() {
    let path = repo_root().join("docs/acceptance/contracts/fixtures/smolvla-chunk-v2.json");
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(v["kind"], "protocol_test");
    assert_eq!(v["not_model_evidence"], true);
    assert_eq!(v["not_rollout_evidence"], true);
    assert_eq!(v["not_energy_evidence"], true);
    assert_eq!(v["model_status"], "proposed");
    assert_eq!(v["codec_version"], CHUNK_CODEC_VERSION);
    assert_eq!(v["lerobot_commit"], LEROBOT_COMMIT);

    let g = &v["golden"];
    let values: Vec<f32> = g["values_time_major"].as_array().unwrap().iter().map(|n| n.as_f64().unwrap() as f32).collect();
    let chunk = ActionChunk::try_new(
        g["n_action_steps"].as_u64().unwrap() as u32,
        g["action_dim"].as_u64().unwrap() as u32,
        values,
    )
    .unwrap();
    let joints = g["joints"]
        .as_array()
        .unwrap()
        .iter()
        .map(|j| JointMeta {
            name: j["name"].as_str().unwrap().into(),
            unit: j["unit"].as_str().unwrap().into(),
            frame: j["frame"].as_str().unwrap().into(),
            min: j["min"].as_f64().unwrap() as f32,
            max: j["max"].as_f64().unwrap() as f32,
        })
        .collect();
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
    assert_eq!(id.model_id, MODEL_ID);
    let (bytes, digest) = encode_action_chunk(&chunk, &id).unwrap();
    assert_eq!(digest, g["digest_sha256"].as_str().unwrap());
    assert_eq!(hex_encode(&bytes), g["canonical_hex"].as_str().unwrap());
    assert_eq!(hex_encode(&sha256(&bytes)), digest);
    let stamps: Vec<u64> = g["timestamps_ns"].as_array().unwrap().iter().map(|n| n.as_u64().unwrap()).collect();
    for (k, stamp) in stamps.iter().enumerate() {
        assert_eq!(
            action_timestamp_ns(id.t0_ns, id.step_duration_ns, k as u32, chunk.n_action_steps).unwrap(),
            *stamp
        );
    }
    assert_eq!(authorize_rollout(&chunk, &id), Err(ContractError::ProposedEntryInvoked));
}

#[test]
fn the_refusal_fixture_blocks_a_rollout() {
    let path = repo_root().join("docs/acceptance/contracts/fixtures/smolvla-chunk-v2.json");
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let g = &v["golden"];
    let r = &v["refusal"];
    assert_eq!(r["case"], "missing_noise_provenance");
    assert_eq!(r["rollout_authorized"], false);
    let values: Vec<f32> = g["values_time_major"].as_array().unwrap().iter().map(|n| n.as_f64().unwrap() as f32).collect();
    let chunk = ActionChunk::try_new(2, 2, values).unwrap();
    let joints = g["joints"]
        .as_array()
        .unwrap()
        .iter()
        .map(|j| JointMeta {
            name: j["name"].as_str().unwrap().into(),
            unit: j["unit"].as_str().unwrap().into(),
            frame: j["frame"].as_str().unwrap().into(),
            min: j["min"].as_f64().unwrap() as f32,
            max: j["max"].as_f64().unwrap() as f32,
        })
        .collect();
    let id = ChunkIdentity {
        model_id: g["model_id"].as_str().unwrap().into(),
        weight_sha256: g["weight_sha256"].as_str().unwrap().into(),
        stats_sha256: g["stats_sha256"].as_str().unwrap().into(),
        noise_sha256: r["noise_sha256"].as_str().unwrap().into(),
        embodiment: g["embodiment"].as_str().unwrap().into(),
        t0_ns: g["t0_ns"].as_u64().unwrap(),
        step_duration_ns: g["step_duration_ns"].as_u64().unwrap(),
        joints,
    };
    assert_eq!(encode_action_chunk(&chunk, &id).unwrap_err(), ContractError::MissingNoiseProvenance);
    assert_eq!(authorize_rollout(&chunk, &id), Err(ContractError::MissingNoiseProvenance));
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
