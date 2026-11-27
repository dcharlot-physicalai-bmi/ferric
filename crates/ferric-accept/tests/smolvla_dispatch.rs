//! Version dispatch. Version 2 goldens stay byte-identical. Version 3 is a different layout.

use ferric_accept::smolvla::{
    decode_action_chunk, decode_action_chunk_v2, encode_action_chunk, ChunkDecodeError, DecodedChunk,
};
use ferric_accept::smolvla_v3::decode_action_chunk_v3_with_sha256;
use ferric_accept::{hex, repo_root, sha256};

fn load_hex(name: &str) -> (Vec<u8>, String) {
    let path = repo_root().join("docs/acceptance/contracts/fixtures").join(name);
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let g = v.get("golden").unwrap_or(&v);
    let digest = g["digest_sha256"].as_str().unwrap().to_string();
    let bytes = from_hex(g["canonical_hex"].as_str().unwrap());
    (bytes, digest)
}

fn from_hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn v2_decoder_round_trips_both_goldens() {
    for (name, digest_pin) in [
        (
            "smolvla-chunk-v2.json",
            "6d668d7e25c89249c890d895ef15b8fdcbb97dab7a1339a0a97605c6e1172d86",
        ),
        (
            "smolvla-empty-chunk-v2.json",
            "6295818fa11b85b37ad8054c75e0a1935b0c0cbb5d7e419090c2520a17b1744e",
        ),
    ] {
        let (bytes, digest) = load_hex(name);
        assert_eq!(digest, digest_pin);
        assert_eq!(hex(&sha256(&bytes)), digest);
        let (chunk, id) = decode_action_chunk_v2(&bytes).unwrap();
        let (again, digest2) = encode_action_chunk(&chunk, &id).unwrap();
        assert_eq!(again, bytes, "{name}");
        assert_eq!(digest2, digest);
    }
}

#[test]
fn decode_action_chunk_dispatches_on_the_version() {
    let (v2, v2_digest) = load_hex("smolvla-chunk-v2.json");
    match decode_action_chunk(&v2).unwrap() {
        DecodedChunk::V2 { chunk, identity } => {
            let (again, digest) = encode_action_chunk(&chunk, &identity).unwrap();
            assert_eq!(again, v2);
            assert_eq!(digest, v2_digest);
        }
        DecodedChunk::V3(_) => panic!("version 2 decoded as version 3"),
    }

    let (v3, v3_digest) = load_hex("smolvla-chunk-v3.json");
    match decode_action_chunk(&v3).unwrap() {
        DecodedChunk::V3(chunk) => {
            assert_eq!(chunk.n_action_steps, 1);
            assert_eq!(chunk.seed, 3);
        }
        DecodedChunk::V2 { .. } => panic!("version 3 decoded as version 2"),
    }
    let (chunk, sha) = decode_action_chunk_v3_with_sha256(&v3).unwrap();
    assert_eq!(sha, v3_digest);
    assert_eq!(sha, hex(&sha256(&v3)));
    assert_eq!(chunk.seed, 3);

    let mut unknown = v2.clone();
    unknown[21..25].copy_from_slice(&4u32.to_le_bytes());
    assert!(matches!(decode_action_chunk(&unknown), Err(ChunkDecodeError::BadVersion)));
    let mut zero = v2;
    zero[21..25].copy_from_slice(&0u32.to_le_bytes());
    assert!(matches!(decode_action_chunk(&zero), Err(ChunkDecodeError::BadVersion)));
    assert!(matches!(decode_action_chunk(&[]), Err(ChunkDecodeError::BadMagic)));
    assert!(matches!(decode_action_chunk_v2(&v3), Err(ChunkDecodeError::BadVersion)));
}
