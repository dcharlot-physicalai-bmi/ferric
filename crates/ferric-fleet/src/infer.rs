//! Load SmolVLA tensors and run the projections that do not need the VLM prefix.
//!
//! A full action chunk needs `vision_model`, the connector, the 16-layer text
//! prefix, expert cross-attention, and the 10-step Euler loop. Those are not
//! executed here. `infer_projections` runs `state_proj` on the real matrix.

use crate::acquire::sha256_hex;

pub struct Tensor {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub fn load_tensor(file: &[u8], name: &str) -> Result<Tensor, String> {
    if file.len() < 8 {
        return Err("safetensors shorter than a header".into());
    }
    let n = u64::from_le_bytes(file[0..8].try_into().unwrap()) as usize;
    if file.len() < 8 + n {
        return Err("safetensors header truncated".into());
    }
    let hdr: serde_json::Value =
        serde_json::from_slice(&file[8..8 + n]).map_err(|e| e.to_string())?;
    let meta = hdr
        .get(name)
        .ok_or_else(|| format!("tensor `{name}` is not in this file"))?;
    let dtype = meta["dtype"].as_str().unwrap_or("").to_string();
    let shape: Vec<usize> = meta["shape"]
        .as_array()
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|v| v.as_u64().map(|x| x as usize))
        .collect();
    let offsets = meta["data_offsets"]
        .as_array()
        .ok_or("missing data_offsets")?;
    let start = offsets[0].as_u64().ok_or("bad offset")? as usize;
    let end = offsets[1].as_u64().ok_or("bad offset")? as usize;
    let base = 8 + n;
    if base + end > file.len() {
        return Err(format!("tensor `{name}` extends past the file"));
    }
    let raw = &file[base + start..base + end];
    let data: Vec<f32> = match dtype.as_str() {
        "F32" => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        "BF16" => raw
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect(),
        other => return Err(format!("unsupported dtype {other}")),
    };
    let expect: usize = shape.iter().product();
    if data.len() != expect {
        return Err(format!(
            "{name}: decoded {} values, shape product {expect}",
            data.len()
        ));
    }
    Ok(Tensor { dtype, shape, data })
}

/// `y = W x + b` with `W` stored `[out, in]`.
pub fn linear(
    weight: &[f32],
    bias: &[f32],
    x: &[f32],
    out: usize,
    inn: usize,
) -> Result<Vec<f32>, String> {
    if weight.len() != out * inn || bias.len() != out || x.len() != inn {
        return Err(format!(
            "linear shape weight {} bias {} x {} out {out} in {inn}",
            weight.len(),
            bias.len(),
            x.len()
        ));
    }
    let mut y = vec![0.0; out];
    for o in 0..out {
        let mut acc = bias[o];
        let row = &weight[o * inn..(o + 1) * inn];
        for i in 0..inn {
            acc += row[i] * x[i];
        }
        y[o] = acc;
    }
    Ok(y)
}

/// State is padded to 32 by the caller. Index 0 of a one-hot input is column 0.
pub fn state_proj(file: &[u8], state32: &[f32]) -> Result<Vec<f32>, String> {
    if state32.len() != 32 {
        return Err(format!("state must be length 32, got {}", state32.len()));
    }
    let w = load_tensor(file, "model.state_proj.weight")?;
    let b = load_tensor(file, "model.state_proj.bias")?;
    if w.shape != [960, 32] || b.shape != [960] {
        return Err(format!(
            "unexpected state_proj shapes {:?} {:?}",
            w.shape, b.shape
        ));
    }
    linear(&w.data, &b.data, state32, 960, 32)
}

pub fn digest_f32(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    sha256_hex(&bytes)
}

/// Why this binary does not return a policy action chunk.
pub const ACTION_CHUNK_BLOCKER: &str = "this binary only multiplies state_proj; ferric-smolvla infer runs vision, text, the expert, and the Euler loop";
