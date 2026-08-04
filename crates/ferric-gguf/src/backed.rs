//! **One GGUF reader for every embodiment**: header in memory, tensor bytes from a [`Backing`].
//!
//! `Gguf` needs the whole file in memory and `GgufFile` needs a filesystem. Neither works in a browser
//! streaming a model it cannot hold — and inventing a third reader for wasm would mean three places for a
//! format detail to drift.
//!
//! [`GgufBacked`] is the unification. It parses the header from a prefix and fetches tensor bytes through
//! a `Backing`, so the *same* reader serves:
//!
//! | embodiment | backing |
//! |---|---|
//! | native, model on disk | `FileBacking` (positional reads) |
//! | anywhere, model in memory | `SliceBacking` |
//! | **browser** | `StagedBacking`, fed by `fetch` with Range headers |
//! | a test | any of the above, or a mock |
//!
//! The header must be parsed before anything else can happen — it carries the tensor table — but its size
//! is not recorded anywhere, so a reader has to grow a prefix until it parses. [`header_probe`] does that
//! against a backing and reports how many bytes it needed, which a browser wants to know before it
//! commits to a download.

use crate::{deq_raw, type_size, GgufSource, Meta, TensorInfo};
use ferric_tier::{Backing, TierError};
use std::collections::HashMap;
use std::sync::Arc;

/// A GGUF whose tensor bytes come from a backing rather than from memory or a file handle.
pub struct GgufBacked {
    pub metadata: HashMap<String, Meta>,
    pub tensors: Vec<TensorInfo>,
    data_start: u64,
    backing: Arc<dyn Backing + Send + Sync>,
}

impl GgufBacked {
    /// Parse the header out of `header_bytes` and serve tensors from `backing`.
    ///
    /// `header_bytes` must cover at least the metadata and tensor table; [`header_probe`] finds that
    /// length. Passing more is harmless — only the header is read from it.
    pub fn new(
        header_bytes: Vec<u8>,
        backing: Arc<dyn Backing + Send + Sync>,
    ) -> Result<Self, String> {
        let g = crate::parse(header_bytes)?;
        Ok(Self {
            metadata: g.metadata,
            tensors: g.tensors,
            data_start: g.data_start as u64,
            backing,
        })
    }

    pub fn data_start(&self) -> u64 { self.data_start }
    pub fn backing(&self) -> &Arc<dyn Backing + Send + Sync> { &self.backing }

    /// Absolute byte range of `name` in the checkpoint.
    pub fn extent(&self, name: &str) -> Option<(u64, usize)> {
        let t = self.tensor(name)?;
        let n: usize = t.dims.iter().product::<u64>() as usize;
        Some((self.data_start + t.offset, type_size(t.ggml_type, n).ok()?))
    }
}

impl GgufSource for GgufBacked {
    fn metadata(&self) -> &HashMap<String, Meta> { &self.metadata }
    fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
    fn raw(&self, name: &str) -> Result<Vec<u8>, String> {
        let (off, sz) = self.extent(name).ok_or_else(|| format!("no tensor '{name}'"))?;
        let mut buf = vec![0u8; sz];
        self.backing.read_at(off, &mut buf).map_err(|e| e.to_string())?;
        Ok(buf)
    }
    fn dequant(&self, name: &str) -> Result<Vec<f32>, String> {
        let t = self.tensor(name).ok_or_else(|| format!("no tensor '{name}'"))?;
        let n: usize = t.dims.iter().product::<u64>() as usize;
        let ty = t.ggml_type;
        deq_raw(&self.raw(name)?, n, ty)
    }
}

/// Smallest prefix of a checkpoint that parses as a GGUF header.
///
/// The header's length is not recorded in the format — the tensor table is variable and the tokenizer
/// vocabulary alone is often megabytes — so a reader must grow a prefix until it parses. Doubling from
/// `start` keeps that to a handful of reads, and the returned length is what a browser needs in order to
/// issue exactly one more range request rather than guessing.
///
/// Bounded by `max` so a corrupt or non-GGUF file fails after a few reads instead of pulling the whole
/// thing over the network to discover it was never a checkpoint.
pub fn header_probe(
    backing: &dyn Backing,
    total: u64,
    start: usize,
    max: usize,
) -> Result<(Vec<u8>, usize), String> {
    let mut n = start.max(4096).min(total as usize);
    loop {
        let mut buf = vec![0u8; n];
        match backing.read_at(0, &mut buf) {
            Ok(()) => {}
            Err(TierError::ShortRead { got, .. }) if got > 0 => {
                buf.truncate(got);
                n = got;
            }
            Err(e) => return Err(e.to_string()),
        }
        if crate::parse(buf.clone()).is_ok() {
            return Ok((buf, n));
        }
        if n >= total as usize {
            return Err("not a GGUF file (whole file read without a valid header)".into());
        }
        if n >= max {
            return Err(format!("no GGUF header within the first {max} bytes"));
        }
        n = (n * 4).min(max).min(total as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_tier::SliceBacking;

    /// Every embodiment must read a checkpoint identically. This uses the real Qwen GGUF when present,
    /// because a synthetic header would not exercise the tokenizer-sized metadata that makes header
    /// probing necessary in the first place.
    fn model_path() -> Option<String> {
        let p = format!(
            "{}/.cache/ferric/hub/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf",
            std::env::var("HOME").ok()?
        );
        std::path::Path::new(&p).exists().then_some(p)
    }

    #[test]
    fn a_backed_reader_returns_the_same_bytes_as_the_file_reader() {
        let Some(path) = model_path() else { eprintln!("model absent — skipping"); return };
        let file = crate::GgufFile::open(&path).unwrap();
        let backing: Arc<dyn Backing + Send + Sync> =
            Arc::new(ferric_tier::FileBacking::open(&path).unwrap());
        let total = std::fs::metadata(&path).unwrap().len();
        let (header, hlen) = header_probe(&*backing, total, 1 << 20, 64 << 20).unwrap();
        assert!(hlen >= 1 << 20, "probe returned an implausibly small header");
        let backed = GgufBacked::new(header, Arc::clone(&backing)).unwrap();

        assert_eq!(backed.tensors.len(), file.tensors.len(), "tensor table differs");
        assert_eq!(backed.data_start(), file.data_start());
        // Spot-check across the file: identical bytes AND identical dequantized values.
        for name in ["token_embd.weight", "blk.0.attn_q.weight", "blk.23.ffn_down.weight", "output_norm.weight"] {
            if file.tensor(name).is_none() { continue; }
            assert_eq!(backed.raw(name).unwrap(), file.raw(name).unwrap(), "{name}: raw bytes differ");
            assert_eq!(backed.dequant(name).unwrap(), file.dequant(name).unwrap(), "{name}: dequant differs");
        }
    }

    #[test]
    fn an_in_memory_backing_reads_identically_to_a_file_backing() {
        // The browser fallback path (whole model in memory) must not be a different reader.
        let Some(path) = model_path() else { eprintln!("model absent — skipping"); return };
        let file = crate::GgufFile::open(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let total = bytes.len() as u64;
        let backing: Arc<dyn Backing + Send + Sync> = Arc::new(SliceBacking::new(bytes));
        let (header, _) = header_probe(&*backing, total, 1 << 20, 64 << 20).unwrap();
        let backed = GgufBacked::new(header, backing).unwrap();
        for name in ["blk.0.ffn_gate.weight", "blk.5.attn_output.weight"] {
            assert_eq!(backed.raw(name).unwrap(), file.raw(name).unwrap(), "{name} differs");
        }
    }

    #[test]
    fn header_probe_grows_until_it_parses_and_refuses_a_non_checkpoint() {
        let Some(path) = model_path() else { eprintln!("model absent — skipping"); return };
        let bytes = std::fs::read(&path).unwrap();
        let total = bytes.len() as u64;
        let b: Arc<dyn Backing + Send + Sync> = Arc::new(SliceBacking::new(bytes));
        // Starting far too small must still converge, which is the whole point of doubling.
        let (_, n) = header_probe(&*b, total, 4096, 64 << 20).unwrap();
        let (_, n2) = header_probe(&*b, total, 1 << 20, 64 << 20).unwrap();
        assert!(n > 0 && n2 > 0);

        // Garbage must fail after a bounded number of reads, not by downloading everything.
        let junk: Arc<dyn Backing + Send + Sync> = Arc::new(SliceBacking::new(vec![0u8; 1 << 20]));
        assert!(header_probe(&*junk, 1 << 20, 4096, 1 << 20).is_err());
    }
}
