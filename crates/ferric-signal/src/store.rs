//! Saving and loading weights, so a trained model can leave the process it was trained in.
//!
//! Until this existed, every model in the crate was generated from a seed. That is fine for testing
//! shape and determinism and useless for anything else: a trained tokenizer that cannot be written
//! to a file cannot be published, checked by anyone else, or loaded on a sensor node. It also made
//! the receipt's `weights_digest` a digest of the *seed* rather than of the weights.
//!
//! ## Format
//!
//! Deliberately small and self-describing, because the alternative — reaching for a general tensor
//! container — buys features this needs none of and a dependency it would then carry forever.
//!
//! ```text
//!   magic     "FSIG"                      4 bytes
//!   version   u32 LE                      4
//!   n_tensors u32 LE                      4
//!   per tensor:
//!     name_len u32, name utf-8
//!     rank u32, dims [u64; rank]
//!     data f32 LE, prod(dims) values
//! ```
//!
//! Every integer is little-endian and every float is written by bit pattern, so a file written on
//! one machine loads identically on another. That is not decoration: the determinism receipt claims
//! a token stream is recomputable, and a weight file that round-trips differently across machines
//! would quietly break that claim.

use crate::sha256::{hex, Sha256};
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;

const MAGIC: &[u8; 4] = b"FSIG";
const VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    NotOurFile,
    UnsupportedVersion { found: u32, expected: u32 },
    /// The file ended mid-record. Reported rather than zero-filled: a short read that silently
    /// pads produces a model that loads, runs, and is wrong.
    Truncated { at: &'static str },
    /// A tensor's declared size does not match the bytes present.
    SizeMismatch { name: String, declared: usize, present: usize },
    /// The caller asked for a tensor the file does not carry.
    Missing { name: String },
    /// A tensor loaded with a shape the model did not expect.
    ShapeMismatch { name: String, want: Vec<usize>, got: Vec<usize> },
}

impl core::fmt::Display for StoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StoreError::NotOurFile => write!(f, "store: not a ferric-signal weight file"),
            StoreError::UnsupportedVersion { found, expected } => {
                write!(f, "store: file version {found}, this build reads {expected}")
            }
            StoreError::Truncated { at } => write!(f, "store: file ends inside {at}"),
            StoreError::SizeMismatch { name, declared, present } => {
                write!(f, "store: {name} declares {declared} floats, {present} present")
            }
            StoreError::Missing { name } => write!(f, "store: {name} is not in this file"),
            StoreError::ShapeMismatch { name, want, got } => {
                write!(f, "store: {name} has shape {got:?}, expected {want:?}")
            }
        }
    }
}

/// Named tensors, in file order.
#[derive(Clone)]
pub struct Weights {
    pub tensors: Vec<(String, Vec<usize>, Vec<f32>)>,
}

impl Weights {
    pub fn new() -> Self {
        Self { tensors: Vec::new() }
    }

    pub fn push(&mut self, name: impl Into<String>, shape: &[usize], data: Vec<f32>) {
        self.tensors.push((name.into(), shape.to_vec(), data));
    }

    /// Collect from live tensors, reading each back from the device.
    pub fn from_tensors(named: &[(&str, &Tensor)]) -> Self {
        let mut w = Self::new();
        for (n, t) in named {
            w.push(*n, &t.shape, pollster::block_on(t.to_vec()));
        }
        w
    }

    pub fn get(&self, name: &str) -> Result<&(String, Vec<usize>, Vec<f32>), StoreError> {
        self.tensors
            .iter()
            .find(|(n, _, _)| n == name)
            .ok_or_else(|| StoreError::Missing { name: name.into() })
    }

    /// Load one tensor onto a device, checking its shape against what the model expects.
    ///
    /// The shape check is the point. A file whose tensors are the right total size but the wrong
    /// shape loads into a model that runs and produces confident nonsense.
    pub fn tensor(&self, ctx: &Arc<Context>, name: &str, want: &[usize]) -> Result<Tensor, StoreError> {
        let (_, shape, data) = self.get(name)?;
        if shape != want {
            return Err(StoreError::ShapeMismatch {
                name: name.into(),
                want: want.to_vec(),
                got: shape.clone(),
            });
        }
        Ok(Tensor::from_vec(ctx, data, shape))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&VERSION.to_le_bytes());
        b.extend_from_slice(&(self.tensors.len() as u32).to_le_bytes());
        for (name, shape, data) in &self.tensors {
            b.extend_from_slice(&(name.len() as u32).to_le_bytes());
            b.extend_from_slice(name.as_bytes());
            b.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for d in shape {
                b.extend_from_slice(&(*d as u64).to_le_bytes());
            }
            for v in data {
                b.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        b
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, StoreError> {
        let mut p = 0usize;
        let take = |p: &mut usize, n: usize, at: &'static str| -> Result<&[u8], StoreError> {
            if *p + n > bytes.len() {
                return Err(StoreError::Truncated { at });
            }
            let s = &bytes[*p..*p + n];
            *p += n;
            Ok(s)
        };
        if take(&mut p, 4, "magic")? != MAGIC {
            return Err(StoreError::NotOurFile);
        }
        let v = u32::from_le_bytes(take(&mut p, 4, "version")?.try_into().unwrap());
        if v != VERSION {
            return Err(StoreError::UnsupportedVersion { found: v, expected: VERSION });
        }
        let n = u32::from_le_bytes(take(&mut p, 4, "tensor count")?.try_into().unwrap()) as usize;
        let mut out = Self::new();
        for _ in 0..n {
            let nl = u32::from_le_bytes(take(&mut p, 4, "name length")?.try_into().unwrap()) as usize;
            let name = String::from_utf8_lossy(take(&mut p, nl, "name")?).into_owned();
            let rank = u32::from_le_bytes(take(&mut p, 4, "rank")?.try_into().unwrap()) as usize;
            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                shape.push(u64::from_le_bytes(take(&mut p, 8, "dimension")?.try_into().unwrap()) as usize);
            }
            let count: usize = shape.iter().product();
            let raw = take(&mut p, count * 4, "tensor data").map_err(|_| StoreError::SizeMismatch {
                name: name.clone(),
                declared: count,
                present: (bytes.len().saturating_sub(p)) / 4,
            })?;
            let data = raw
                .chunks_exact(4)
                .map(|c| f32::from_bits(u32::from_le_bytes(c.try_into().unwrap())))
                .collect();
            out.push(name, &shape, data);
        }
        Ok(out)
    }

    /// Digest of the file bytes. This is what belongs in a receipt's `weights_digest`.
    pub fn digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(&self.to_bytes());
        hex(&h.finish())
    }
}

/// Deliberately compact. A derived `Debug` would print every float, so a panic in a test against a
/// real model would emit millions of numbers and bury the assertion that fired.
impl core::fmt::Debug for Weights {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Weights[{} tensors: ", self.tensors.len())?;
        for (i, (n, s, d)) in self.tensors.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{n}{s:?}={}f32", d.len())?;
        }
        write!(f, "]")
    }
}

impl Default for Weights {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Weights {
        let mut w = Weights::new();
        w.push("embed", &[3, 2], vec![1.0, -2.5, 3.25, 0.0, -0.0, 7.5]);
        w.push("norm", &[4], vec![1.0, 1.0, 1.0, 1.0]);
        w.push("block.0.wq", &[2, 2], vec![0.1, 0.2, 0.3, 0.4]);
        w
    }

    #[test]
    fn weights_round_trip_exactly() {
        let a = sample();
        let b = Weights::from_bytes(&a.to_bytes()).unwrap();
        assert_eq!(a.tensors.len(), b.tensors.len());
        for ((n1, s1, d1), (n2, s2, d2)) in a.tensors.iter().zip(&b.tensors) {
            assert_eq!(n1, n2);
            assert_eq!(s1, s2);
            // Bit-exact, not approximately: a weight file that rounds is a different model.
            assert!(d1.iter().zip(d2).all(|(x, y)| x.to_bits() == y.to_bits()), "{n1} changed");
        }
        assert_eq!(a.digest(), b.digest());
    }

    /// Signed zero survives, because the file stores bit patterns rather than formatted numbers.
    #[test]
    fn signed_zero_and_special_values_survive() {
        let mut w = Weights::new();
        w.push("t", &[4], vec![0.0, -0.0, f32::MIN_POSITIVE, -1.0]);
        let b = Weights::from_bytes(&w.to_bytes()).unwrap();
        let d = &b.get("t").unwrap().2;
        assert_eq!(d[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(d[1].to_bits(), (-0.0f32).to_bits());
        assert_ne!(d[0].to_bits(), d[1].to_bits(), "0.0 and -0.0 were conflated");
    }

    /// A CORRUPT FILE MUST BE REFUSED, not partially loaded. A short read that zero-fills produces
    /// a model that loads, runs, and is wrong — the worst of the three outcomes.
    #[test]
    fn corrupt_and_foreign_files_are_refused() {
        let good = sample().to_bytes();
        assert_eq!(Weights::from_bytes(b"nope").unwrap_err(), StoreError::NotOurFile);
        assert!(matches!(Weights::from_bytes(&good[..3]), Err(StoreError::Truncated { .. })));

        let mut bad_ver = good.clone();
        bad_ver[4..8].copy_from_slice(&99u32.to_le_bytes());
        assert_eq!(
            Weights::from_bytes(&bad_ver).unwrap_err(),
            StoreError::UnsupportedVersion { found: 99, expected: VERSION }
        );

        // Every truncation point must be refused, not just the convenient ones.
        for cut in 1..good.len() {
            assert!(Weights::from_bytes(&good[..cut]).is_err(), "a file cut at {cut} was accepted");
        }
    }

    /// A shape mismatch is caught even when the element count is right. This is the failure that
    /// otherwise loads cleanly and produces confident nonsense.
    #[test]
    fn a_transposed_tensor_is_refused_even_though_it_is_the_right_size() {
        let ctx = match pollster::block_on(Context::new()) {
            Ok(c) => Arc::new(c),
            Err(_) if std::env::var("FERRIC_NO_GPU").is_ok() => return,
            Err(e) => panic!("no GPU context ({e:?}); set FERRIC_NO_GPU=1 to waive"),
        };
        let w = sample();
        assert!(w.tensor(&ctx, "embed", &[3, 2]).is_ok());
        match w.tensor(&ctx, "embed", &[2, 3]) {
            Err(StoreError::ShapeMismatch { got, want, .. }) => {
                assert_eq!(got, vec![3, 2]);
                assert_eq!(want, vec![2, 3]);
            }
            Err(e) => panic!("wrong error for a transposed shape: {e}"),
            Ok(_) => panic!("a transposed shape was accepted"),
        }
        assert!(w.tensor(&ctx, "absent", &[1]).is_err());
    }

    /// The digest must change when any weight changes, or it cannot back a receipt.
    #[test]
    fn the_digest_moves_with_the_weights() {
        let base = sample().digest();
        let mut a = sample();
        a.tensors[0].2[0] = 1.0000001;
        assert_ne!(a.digest(), base, "a changed weight left the digest alone");
        let mut b = sample();
        b.tensors[0].0 = "renamed".into();
        assert_ne!(b.digest(), base, "a renamed tensor left the digest alone");
        let mut c = sample();
        c.tensors[0].1 = vec![2, 3];
        assert_ne!(c.digest(), base, "a reshaped tensor left the digest alone");
    }
}
