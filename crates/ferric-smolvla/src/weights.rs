//! Safetensors reader for the pinned `model.safetensors`. Tensors are decoded
//! to f32 when asked for. BF16 is the stored format for the VLM and the expert.

use std::collections::HashMap;

pub struct Store {
    file: Vec<u8>,
    header_len: usize,
    index: HashMap<String, (usize, usize, String, Vec<usize>)>,
}

impl Store {
    pub fn open(path: &str) -> Result<Self, String> {
        let file = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        Self::open_bytes(&file)
    }

    pub fn open_bytes(file: &[u8]) -> Result<Self, String> {
        let file = file.to_vec();
        if file.len() < 8 {
            return Err("safetensors shorter than a header".into());
        }
        let n = u64::from_le_bytes(file[0..8].try_into().unwrap()) as usize;
        if file.len() < 8 + n {
            return Err("safetensors header truncated".into());
        }
        let hdr: serde_json::Value =
            serde_json::from_slice(&file[8..8 + n]).map_err(|e| e.to_string())?;
        let obj = hdr
            .as_object()
            .ok_or("safetensors header is not an object")?;
        let mut index = HashMap::new();
        for (name, meta) in obj {
            if name == "__metadata__" {
                continue;
            }
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
            index.insert(name.clone(), (start, end, dtype, shape));
        }
        Ok(Self {
            file,
            header_len: n,
            index,
        })
    }

    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.index.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn shape(&self, name: &str) -> Result<&[usize], String> {
        self.index
            .get(name)
            .map(|t| t.3.as_slice())
            .ok_or_else(|| format!("missing tensor {name}"))
    }

    pub fn tensor(&self, name: &str) -> Result<Vec<f32>, String> {
        let (start, end, dtype, shape) = self
            .index
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        let base = 8 + self.header_len;
        if base + end > self.file.len() {
            return Err(format!("{name} extends past the file"));
        }
        let raw = &self.file[base + start..base + end];
        let data: Vec<f32> = match dtype.as_str() {
            "F32" => raw
                .as_chunks::<4>().0.iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            "BF16" => raw
                .as_chunks::<2>().0.iter()
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                .collect(),
            other => return Err(format!("{name}: unsupported dtype {other}")),
        };
        let expect: usize = shape.iter().product();
        if data.len() != expect {
            return Err(format!(
                "{name}: decoded {} values, shape product {expect}",
                data.len()
            ));
        }
        Ok(data)
    }
}
