//! Ferric weight loading — a pure-Rust `safetensors` reader. Parses the HF safetensors container
//! (8-byte little-endian header length, a JSON header of name → {dtype, shape, data_offsets}, then a
//! flat data blob) and dequantizes F32/F16/BF16/F64 tensors to f32, ready to upload to a Ferric
//! `Context`. No Python, no C++ — this is how real Llama/SmolVLA checkpoints enter the ecosystem.

use half::{bf16, f16};
use std::collections::HashMap;

/// One tensor decoded to f32 plus its shape.
pub struct STensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// Parse a safetensors byte buffer into name → f32 tensor. `__metadata__` is skipped.
pub fn safetensors(bytes: &[u8]) -> Result<HashMap<String, STensor>, String> {
    if bytes.len() < 8 {
        return Err("safetensors: too short".into());
    }
    let hlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let base = 8 + hlen;
    if bytes.len() < base {
        return Err("safetensors: header exceeds buffer".into());
    }
    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..base]).map_err(|e| format!("safetensors header json: {e}"))?;
    let obj = header.as_object().ok_or("safetensors: header not an object")?;

    let mut out = HashMap::new();
    for (name, v) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = v["dtype"].as_str().ok_or("missing dtype")?;
        let shape: Vec<usize> = v["shape"]
            .as_array()
            .ok_or("missing shape")?
            .iter()
            .map(|d| d.as_u64().unwrap() as usize)
            .collect();
        let off = v["data_offsets"].as_array().ok_or("missing data_offsets")?;
        let (s, e) = (off[0].as_u64().unwrap() as usize, off[1].as_u64().unwrap() as usize);
        let raw = &bytes[base + s..base + e];
        let data = dequant(dtype, raw)?;
        let n: usize = shape.iter().product();
        if data.len() != n {
            return Err(format!("{name}: {} elems for shape {shape:?} ({n})", data.len()));
        }
        out.insert(name.clone(), STensor { data, shape });
    }
    Ok(out)
}

/// Dequantize a raw dtype slice to f32.
fn dequant(dtype: &str, raw: &[u8]) -> Result<Vec<f32>, String> {
    Ok(match dtype {
        "F32" => raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        "F16" => raw.chunks_exact(2).map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
        "BF16" => raw.chunks_exact(2).map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
        "F64" => raw.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap()) as f32).collect(),
        other => return Err(format!("unsupported safetensors dtype '{other}'")),
    })
}
