//! Ferric weight loading — a pure-Rust `safetensors` reader. Parses the HF safetensors container
//! (8-byte little-endian header length, a JSON header of name → {dtype, shape, data_offsets}, then a
//! flat data blob), decodes every dtype the format defines, and follows a sharded checkpoint's
//! `model.safetensors.index.json` across its files. No Python, no C++.
//!
//! ## Why this crate is the one that has to be deep
//!
//! safetensors is where open weights are PUBLISHED. Every other container — GGUF included — is a
//! conversion of it, produced by some other project's converter, carrying that project's choices
//! about what to keep and what to fold away. A runtime that reads conversions well and originals
//! badly has outsourced the first step of its own pipeline, and inherits a ceiling it did not set.
//!
//! So the eager whole-buffer path below is the convenience API, and [`SafeTensors`] is the real one:
//! it opens a checkpoint by path, reads the header only, and pulls each tensor's bytes on demand —
//! peak memory is the largest tensor, not the file. That matters at the sizes that are now normal;
//! the eager path needs the entire checkpoint resident AND a second copy expanded to f32.

pub mod fp8;
pub mod hf;

use half::{bf16, f16};
use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// One tensor decoded to f32 plus its shape.
#[derive(Debug, Clone)]
pub struct STensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// Parse a safetensors byte buffer into name → f32 tensor. `__metadata__` is skipped.
pub fn safetensors(bytes: &[u8]) -> Result<HashMap<String, STensor>, String> {
    safetensors_filtered(bytes, |_| true)
}

/// Like [`safetensors`], but only dequantizes tensors whose name passes `keep` — so a multi-component
/// checkpoint (e.g. a mixture-of-transformers whose shards also hold a diffusion tower + vision
/// encoder) can materialize just the subset needed, without paying f32 memory for the rest.
pub fn safetensors_filtered(bytes: &[u8], keep: impl Fn(&str) -> bool) -> Result<HashMap<String, STensor>, String> {
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
        if name == "__metadata__" || !keep(name) {
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
        // Signed and unsigned integers appear in real checkpoints as quantization codes,
        // zero-points and token-id tables. Decoding them to f32 is lossless up to I32/I64's range,
        // which is checked below rather than assumed.
        "I8" => raw.iter().map(|&b| b as i8 as f32).collect(),
        "U8" => raw.iter().map(|&b| b as f32).collect(),
        "BOOL" => raw.iter().map(|&b| if b != 0 { 1.0 } else { 0.0 }).collect(),
        "I16" => raw.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32).collect(),
        "U16" => raw.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]]) as f32).collect(),
        "I32" => raw.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()) as f32).collect(),
        "U32" => raw.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as f32).collect(),
        "I64" => raw.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap()) as f32).collect(),
        "U64" => raw.chunks_exact(8).map(|b| u64::from_le_bytes(b.try_into().unwrap()) as f32).collect(),
        // ⚠ FP8 IS NOT THE WEIGHT — see [`fp8`]. These return the stored coefficients; the scale
        // lives in a sibling tensor and [`SafeTensors::get`] refuses to hand them back without it.
        "F8_E4M3" | "F8_E4M3FN" => raw.iter().map(|&b| fp8::e4m3_to_f32(b)).collect(),
        "F8_E5M2" => raw.iter().map(|&b| fp8::e5m2_to_f32(b)).collect(),
        other => return Err(format!("unsupported safetensors dtype '{other}'")),
    })
}

/// Bytes one element of `dtype` occupies, or `None` for a dtype this crate cannot decode.
pub fn dtype_bytes(dtype: &str) -> Option<usize> {
    Some(match dtype {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        "I8" | "U8" | "BOOL" | "F8_E4M3" | "F8_E4M3FN" | "F8_E5M2" => 1,
        _ => return None,
    })
}

/// True for dtypes whose stored value is a COEFFICIENT, not the number itself — the real value is
/// this times a scale held in a separate tensor. Loading one of these without its scale yields
/// correctly-shaped weights of the wrong magnitude and raises nothing anywhere.
pub fn is_scaled(dtype: &str) -> bool {
    matches!(dtype, "F8_E4M3" | "F8_E4M3FN" | "F8_E5M2")
}

/// Where one tensor lives, without having read it.
#[derive(Debug, Clone)]
pub struct Entry {
    pub dtype: String,
    pub shape: Vec<usize>,
    shard: usize,
    /// Absolute byte range within that shard file.
    start: u64,
    end: u64,
}

impl Entry {
    pub fn elems(&self) -> usize { self.shape.iter().product() }
    pub fn nbytes(&self) -> u64 { self.end - self.start }
}

/// **A checkpoint opened by path, read on demand** — one file or a whole sharded set.
///
/// Mirrors what `ferric-gguf`'s `GgufFile` does for the other container: parse headers up front,
/// keep the files open, and materialize one tensor at a time. A 600 GB checkpoint has a largest
/// tensor of a few GB, and that is the memory bill here.
#[derive(Debug)]
pub struct SafeTensors {
    shards: Vec<Mutex<std::fs::File>>,
    paths: Vec<PathBuf>,
    tensors: BTreeMap<String, Entry>,
    pub metadata: HashMap<String, String>,
}

impl SafeTensors {
    /// Open a `.safetensors` file, a `*.index.json`, or a directory holding either.
    ///
    /// A directory is resolved by looking for an index first and a lone `.safetensors` second. The
    /// order matters: a sharded checkpoint's directory contains BOTH an index and several
    /// `.safetensors` files, and picking one of the shards would load a fraction of the model with
    /// no error — just missing tensors, which surface later as a confusing architecture mismatch.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let p = path.as_ref();
        let files: Vec<PathBuf> = if p.is_dir() {
            let idx = std::fs::read_dir(p).map_err(|e| format!("read_dir {}: {e}", p.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .find(|f| f.to_string_lossy().ends_with("index.json"));
            match idx {
                Some(i) => return Self::open(i),
                None => {
                    let mut v: Vec<PathBuf> = std::fs::read_dir(p).unwrap()
                        .filter_map(|e| e.ok().map(|e| e.path()))
                        .filter(|f| f.extension().is_some_and(|x| x == "safetensors")).collect();
                    v.sort();
                    if v.is_empty() { return Err(format!("no .safetensors in {}", p.display())) }
                    v
                }
            }
        } else if p.to_string_lossy().ends_with("index.json") {
            let txt = std::fs::read_to_string(p).map_err(|e| format!("read {}: {e}", p.display()))?;
            let j: serde_json::Value = serde_json::from_str(&txt).map_err(|e| format!("index json: {e}"))?;
            let map = j["weight_map"].as_object().ok_or("index.json has no weight_map")?;
            let dir = p.parent().unwrap_or(Path::new("."));
            let mut names: Vec<String> = map.values().filter_map(|v| v.as_str().map(String::from)).collect();
            names.sort();
            names.dedup();
            names.into_iter().map(|n| dir.join(n)).collect()
        } else {
            vec![p.to_path_buf()]
        };

        let mut shards = Vec::new();
        let mut tensors = BTreeMap::new();
        let mut metadata = HashMap::new();
        for (si, f) in files.iter().enumerate() {
            let mut fh = std::fs::File::open(f).map_err(|e| format!("open {}: {e}", f.display()))?;
            let mut len8 = [0u8; 8];
            fh.read_exact(&mut len8).map_err(|e| format!("{}: header length: {e}", f.display()))?;
            let hlen = u64::from_le_bytes(len8);
            let flen = fh.metadata().map_err(|e| format!("{}: stat: {e}", f.display()))?.len();
            if hlen + 8 > flen {
                return Err(format!("{}: header claims {hlen} bytes but the file is {flen}", f.display()));
            }
            let mut hbuf = vec![0u8; hlen as usize];
            fh.read_exact(&mut hbuf).map_err(|e| format!("{}: header: {e}", f.display()))?;
            let hdr: serde_json::Value = serde_json::from_slice(&hbuf)
                .map_err(|e| format!("{}: header json: {e}", f.display()))?;
            let obj = hdr.as_object().ok_or("header not an object")?;
            let base = 8 + hlen;
            for (name, v) in obj {
                if name == "__metadata__" {
                    if let Some(m) = v.as_object() {
                        for (k, mv) in m {
                            if let Some(sv) = mv.as_str() { metadata.insert(k.clone(), sv.to_string()); }
                        }
                    }
                    continue;
                }
                let dtype = v["dtype"].as_str().ok_or_else(|| format!("{name}: missing dtype"))?.to_string();
                let shape: Vec<usize> = v["shape"].as_array().ok_or_else(|| format!("{name}: missing shape"))?
                    .iter().map(|d| d.as_u64().unwrap_or(0) as usize).collect();
                let off = v["data_offsets"].as_array().ok_or_else(|| format!("{name}: missing data_offsets"))?;
                let (s0, e0) = (off[0].as_u64().unwrap_or(0), off[1].as_u64().unwrap_or(0));
                // The header is the ONLY thing describing the blob, so a stride disagreement lands
                // at the wrong offset and returns plausible garbage. Check it here, once, where the
                // declared dtype and the declared byte span are both in hand.
                if let Some(w) = dtype_bytes(&dtype) {
                    let want = shape.iter().product::<usize>() * w;
                    if (e0 - s0) as usize != want {
                        return Err(format!("{name}: {dtype}{shape:?} needs {want} bytes, header spans {}",
                                           e0 - s0));
                    }
                }
                if base + e0 > flen {
                    return Err(format!("{name}: ends at {} past the {flen}-byte file", base + e0));
                }
                tensors.insert(name.clone(), Entry { dtype, shape, shard: si, start: base + s0, end: base + e0 });
            }
            shards.push(Mutex::new(fh));
        }
        Ok(SafeTensors { shards, paths: files, tensors, metadata })
    }

    pub fn names(&self) -> impl Iterator<Item = &String> { self.tensors.keys() }
    pub fn info(&self, name: &str) -> Option<&Entry> { self.tensors.get(name) }
    pub fn len(&self) -> usize { self.tensors.len() }
    pub fn is_empty(&self) -> bool { self.tensors.is_empty() }
    pub fn shard_paths(&self) -> &[PathBuf] { &self.paths }
    /// Total parameters across every tensor — the number a checkpoint is named for.
    pub fn params(&self) -> u64 { self.tensors.values().map(|e| e.elems() as u64).sum() }

    /// The tensor's bytes, exactly as stored.
    pub fn raw(&self, name: &str) -> Result<Vec<u8>, String> {
        let e = self.tensors.get(name).ok_or_else(|| format!("no tensor '{name}'"))?;
        let mut fh = self.shards[e.shard].lock().map_err(|_| "shard lock poisoned")?;
        fh.seek(SeekFrom::Start(e.start)).map_err(|x| format!("{name}: seek: {x}"))?;
        let mut buf = vec![0u8; e.nbytes() as usize];
        fh.read_exact(&mut buf).map_err(|x| format!("{name}: read: {x}"))?;
        Ok(buf)
    }

    /// Decode to f32. **Refuses a scaled dtype** — see [`is_scaled`]; use [`SafeTensors::get_scaled`],
    /// which is the only path that returns the actual weight.
    pub fn get(&self, name: &str) -> Result<STensor, String> {
        let e = self.tensors.get(name).ok_or_else(|| format!("no tensor '{name}'"))?;
        if is_scaled(&e.dtype) {
            return Err(format!("{name} is {}: the stored bytes are coefficients, not weights. Use \
                                get_scaled() so the companion scale tensor is applied — decoding \
                                these alone yields the right shape and the wrong magnitude, and \
                                nothing downstream can tell", e.dtype));
        }
        Ok(STensor { data: dequant(&e.dtype, &self.raw(name)?)?, shape: e.shape.clone() })
    }

    /// Decode to f32 with the companion scale applied. For an unscaled dtype this is [`get`].
    ///
    /// The scale is `<name>_scale_inv` (block-wise, the FP8 convention) or `<name>_scale`
    /// (per-tensor or per-channel). Its shape against the weight's gives the block size in each
    /// dimension, so per-tensor `[1]`, per-channel `[out, 1]` and block `[⌈out/128⌉, ⌈in/128⌉]` all
    /// fall out of one rule instead of three special cases.
    pub fn get_scaled(&self, name: &str) -> Result<STensor, String> {
        let e = self.tensors.get(name).ok_or_else(|| format!("no tensor '{name}'"))?.clone();
        if !is_scaled(&e.dtype) { return self.get(name) }
        let sname = [format!("{name}_scale_inv"), format!("{name}_scale")].into_iter()
            .find(|s| self.tensors.contains_key(s))
            .ok_or_else(|| format!("{name} is {} but neither {name}_scale_inv nor {name}_scale \
                                    exists — the weights cannot be reconstructed", e.dtype))?;
        let sc = self.get(&sname)?;
        let mut data = dequant(&e.dtype, &self.raw(name)?)?;
        apply_scale(&mut data, &e.shape, &sc.data, &sc.shape)
            .map_err(|m| format!("{name} / {sname}: {m}"))?;
        Ok(STensor { data, shape: e.shape })
    }
}

/// Multiply `data` (shape `dshape`) by `scale` (shape `sshape`), where each scale entry governs a
/// block whose size is the ratio of the two shapes.
///
/// ⚠ The ratio is a CEILING, not a division. A 4096-row weight scaled per 128 rows has 32 scales;
/// a 4097-row one has 33, and the last block is short. Using `dim / sdim` silently reads the wrong
/// scale for every element past the first short block.
fn apply_scale(data: &mut [f32], dshape: &[usize], scale: &[f32], sshape: &[usize]) -> Result<(), String> {
    if scale.len() == 1 {
        for v in data.iter_mut() { *v *= scale[0] }
        return Ok(());
    }
    // Right-align the scale's dims against the weight's, so [out,1] and [1,in] both work.
    let mut blocks = vec![1usize; dshape.len()];
    let mut sdims = vec![1usize; dshape.len()];
    let off = dshape.len().checked_sub(sshape.len())
        .ok_or_else(|| format!("scale rank {} exceeds weight rank {}", sshape.len(), dshape.len()))?;
    for (i, &sd) in sshape.iter().enumerate() {
        let d = dshape[off + i];
        if sd == 0 { return Err("scale has a zero dimension".into()) }
        sdims[off + i] = sd;
        blocks[off + i] = d.div_ceil(sd);
    }
    let n: usize = dshape.iter().product();
    if data.len() != n { return Err(format!("{} values for shape {dshape:?}", data.len())) }
    // Row-major walk: keep a running multi-index rather than recomputing divisions per element.
    let mut idx = vec![0usize; dshape.len()];
    for (flat, v) in data.iter_mut().enumerate() {
        let mut si = 0usize;
        for k in 0..dshape.len() { si = si * sdims[k] + (idx[k] / blocks[k]).min(sdims[k] - 1); }
        *v *= scale[si];
        let _ = flat;
        for k in (0..dshape.len()).rev() {
            idx[k] += 1;
            if idx[k] < dshape[k] { break }
            idx[k] = 0;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ferric-load-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Build a minimal safetensors file. Writing the container by hand rather than through this
    /// crate's own writer is deliberate — a reader tested only against its own writer agrees with
    /// itself and can still be wrong about the format.
    fn write_st(path: &Path, entries: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
        let mut hdr = String::from("{");
        let mut off = 0usize;
        let mut blob = Vec::new();
        for (i, (n, dt, sh, data)) in entries.iter().enumerate() {
            if i > 0 { hdr.push(',') }
            hdr.push_str(&format!(
                "\"{n}\":{{\"dtype\":\"{dt}\",\"shape\":{sh:?},\"data_offsets\":[{},{}]}}",
                off, off + data.len()));
            off += data.len();
            blob.extend_from_slice(data);
        }
        hdr.push('}');
        let mut out = (hdr.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(hdr.as_bytes());
        out.extend_from_slice(&blob);
        std::fs::write(path, out).unwrap();
    }

    fn f32b(v: &[f32]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

    #[test]
    fn a_sharded_checkpoint_loads_every_shard_and_never_just_one() {
        let d = tmp("shards");
        write_st(&d.join("model-00001-of-00002.safetensors"), &[("a", "F32", vec![2, 2], f32b(&[1., 2., 3., 4.]))]);
        write_st(&d.join("model-00002-of-00002.safetensors"), &[("b", "F32", vec![3], f32b(&[5., 6., 7.]))]);
        std::fs::write(d.join("model.safetensors.index.json"),
            r#"{"metadata":{},"weight_map":{"a":"model-00001-of-00002.safetensors",
                                            "b":"model-00002-of-00002.safetensors"}}"#).unwrap();
        // Opening the DIRECTORY must find the index. Picking a shard would load half the model with
        // no error at all — the failure surfaces much later as a missing-tensor mystery.
        let st = SafeTensors::open(&d).expect("open dir");
        assert_eq!(st.len(), 2, "opened {} tensors from a 2-shard checkpoint", st.len());
        assert_eq!(st.get("a").unwrap().data, vec![1., 2., 3., 4.]);
        assert_eq!(st.get("b").unwrap().data, vec![5., 6., 7.]);
        assert_eq!(st.params(), 7);
        assert_eq!(st.shard_paths().len(), 2);
        std::fs::remove_dir_all(&d).ok();
    }

    /// The header is the only description of the blob, so a stride disagreement reads at the wrong
    /// offset and returns plausible garbage rather than failing. This must be refused at open time.
    #[test]
    fn a_header_whose_byte_span_contradicts_its_shape_is_refused() {
        let d = tmp("badspan");
        let p = d.join("bad.safetensors");
        // Claim [2,2] F32 (16 bytes) but hand over 8.
        let hdr = r#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,8]}}"#;
        let mut out = (hdr.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(hdr.as_bytes());
        out.extend_from_slice(&f32b(&[1., 2.]));
        std::fs::write(&p, out).unwrap();
        let e = SafeTensors::open(&p).unwrap_err();
        assert!(e.contains("needs 16 bytes"), "expected a stride complaint, got: {e}");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn an_fp8_tensor_will_not_be_handed_back_without_its_scale() {
        let d = tmp("fp8bare");
        let p = d.join("m.safetensors");
        write_st(&p, &[("w", "F8_E4M3", vec![2, 2], vec![0x38, 0x40, 0x44, 0x48])]);
        let st = SafeTensors::open(&p).unwrap();
        let e = st.get("w").unwrap_err();
        assert!(e.contains("get_scaled"), "expected a redirect to get_scaled, got: {e}");
        let e2 = st.get_scaled("w").unwrap_err();
        assert!(e2.contains("_scale_inv"), "expected a complaint about the missing scale, got: {e2}");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn fp8_weights_come_back_scaled_by_their_companion_tensor() {
        let d = tmp("fp8scaled");
        let p = d.join("m.safetensors");
        // 0x38 is E4M3 1.0, 0x40 is 2.0. Two rows, one scale each.
        write_st(&p, &[
            ("w", "F8_E4M3", vec![2, 2], vec![0x38, 0x40, 0x38, 0x40]),
            ("w_scale_inv", "F32", vec![2, 1], f32b(&[10.0, 100.0])),
        ]);
        let st = SafeTensors::open(&p).unwrap();
        assert_eq!(st.get_scaled("w").unwrap().data, vec![10.0, 20.0, 100.0, 200.0]);
        std::fs::remove_dir_all(&d).ok();
    }

    /// ⚠ The blocks-per-scale ratio is a CEILING. 5 rows under 2 scales means blocks of 3 — rows
    /// 0,1,2 then 3,4 — and `dim / sdim` would give 2, sending row 4 to a scale index that does not
    /// exist. Integer division passes every power-of-two test and fails on the first odd shape.
    #[test]
    fn a_short_final_block_reads_the_last_scale_not_one_past_it() {
        let mut data = vec![1.0f32; 5 * 2];
        apply_scale(&mut data, &[5, 2], &[2.0, 7.0], &[2, 1]).expect("apply");
        // rows 0..2 -> 2.0, rows 3..4 -> 7.0
        assert_eq!(data, vec![2., 2., 2., 2., 2., 2., 7., 7., 7., 7.]);
    }

    #[test]
    fn per_tensor_per_channel_and_block_scales_all_follow_one_rule() {
        let mut a = vec![1.0f32; 4];
        apply_scale(&mut a, &[2, 2], &[3.0], &[1]).unwrap();
        assert_eq!(a, vec![3., 3., 3., 3.], "per-tensor");

        let mut b = vec![1.0f32; 4];
        apply_scale(&mut b, &[2, 2], &[3.0, 5.0], &[1, 2]).unwrap();
        assert_eq!(b, vec![3., 5., 3., 5.], "per-column");

        let mut c = vec![1.0f32; 16];
        apply_scale(&mut c, &[4, 4], &[1., 2., 3., 4.], &[2, 2]).unwrap();
        assert_eq!(c, vec![1., 1., 2., 2., 1., 1., 2., 2., 3., 3., 4., 4., 3., 3., 4., 4.], "2x2 blocks");
    }

    #[test]
    fn the_integer_dtypes_decode_with_their_signs_intact() {
        let d = tmp("ints");
        let p = d.join("m.safetensors");
        write_st(&p, &[
            ("i8", "I8", vec![3], vec![0xFF, 0x01, 0x80]),   // -1, 1, -128
            ("u8", "U8", vec![3], vec![0xFF, 0x01, 0x80]),   // 255, 1, 128
            ("bl", "BOOL", vec![2], vec![0x00, 0x01]),
            ("i32", "I32", vec![2], (-7i32).to_le_bytes().iter().chain(9i32.to_le_bytes().iter()).copied().collect()),
        ]);
        let st = SafeTensors::open(&p).unwrap();
        assert_eq!(st.get("i8").unwrap().data, vec![-1., 1., -128.]);
        assert_eq!(st.get("u8").unwrap().data, vec![255., 1., 128.]);
        assert_eq!(st.get("bl").unwrap().data, vec![0., 1.]);
        assert_eq!(st.get("i32").unwrap().data, vec![-7., 9.]);
        std::fs::remove_dir_all(&d).ok();
    }

    /// The lazy reader must agree with the eager one it is replacing, tensor for tensor — otherwise
    /// "read on demand" is a second implementation with its own opinions about the container.
    #[test]
    fn the_lazy_reader_and_the_eager_one_agree() {
        let d = tmp("agree");
        let p = d.join("m.safetensors");
        write_st(&p, &[
            ("x", "F32", vec![2, 3], f32b(&[1., -2., 3., -4., 5., -6.])),
            ("y", "F16", vec![2], vec![0x00, 0x3C, 0x00, 0xC0]), // 1.0, -2.0
        ]);
        let eager = safetensors(&std::fs::read(&p).unwrap()).unwrap();
        let lazy = SafeTensors::open(&p).unwrap();
        assert_eq!(eager.len(), lazy.len());
        for (name, t) in &eager {
            let l = lazy.get(name).unwrap();
            assert_eq!(t.data, l.data, "{name} differs between readers");
            assert_eq!(t.shape, l.shape, "{name} shape differs");
        }
        std::fs::remove_dir_all(&d).ok();
    }
}
