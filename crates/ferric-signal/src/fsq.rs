//! Finite Scalar Quantization.
//!
//! A discrete bottleneck with **no codebook**. Each latent dimension is squashed to a bounded
//! range and rounded to one of `L` levels, so the code is just a mixed-radix number and the
//! "codebook" is implied by the level list. There is nothing to initialise, nothing to keep
//! alive with an EMA, and no commitment loss, which is why codebook collapse — the failure that
//! makes VQ-VAE training fragile — cannot occur here: every code is reachable by construction.
//!
//! Reference: Mentzer, Minnen, Agustsson, Tschannen, *Finite Scalar Quantization: VQ-VAE Made
//! Simple* (arXiv:2309.15505). The bounding function below follows the paper's formulation,
//! including the half-level shift for even `L`, which is the part that is easy to get subtly
//! wrong: without it, an even number of levels is not symmetric about zero and the top level is
//! unreachable.
//!
//! Verification status of this file: the round-trip and bijection properties are **checked
//! exhaustively** over the whole code space in the tests below, not sampled. Nothing here is
//! measured against a trained model yet; this is the quantizer alone.

/// A finite scalar quantizer defined by its per-dimension level counts.
///
/// `levels = [8, 8, 8, 8, 8]` gives 8^5 = 32,768 codes in five dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fsq {
    levels: Vec<u32>,
}

/// Why a level list was rejected. Each of these is a silent-wrong-answer if allowed through.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FsqError {
    /// No dimensions: the code space would be a single empty code.
    Empty,
    /// A dimension with fewer than two levels carries no information and breaks the radix.
    DegenerateLevel { dim: usize, level: u32 },
    /// The product of levels exceeded `u32`, so indices would wrap silently.
    CodebookTooLarge,
}

impl core::fmt::Display for FsqError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FsqError::Empty => write!(f, "fsq: level list is empty"),
            FsqError::DegenerateLevel { dim, level } => {
                write!(f, "fsq: dimension {dim} has {level} level(s); at least 2 are required")
            }
            FsqError::CodebookTooLarge => {
                write!(f, "fsq: codebook size overflows u32")
            }
        }
    }
}

impl Fsq {
    /// Build a quantizer, rejecting level lists that would quantize wrongly rather than loudly.
    pub fn new(levels: impl Into<Vec<u32>>) -> Result<Self, FsqError> {
        let levels = levels.into();
        if levels.is_empty() {
            return Err(FsqError::Empty);
        }
        for (dim, &l) in levels.iter().enumerate() {
            if l < 2 {
                return Err(FsqError::DegenerateLevel { dim, level: l });
            }
        }
        let mut size: u64 = 1;
        for &l in &levels {
            size = size.saturating_mul(l as u64);
            if size > u32::MAX as u64 {
                return Err(FsqError::CodebookTooLarge);
            }
        }
        Ok(Self { levels })
    }

    /// The 32,768-code vocabulary used by the sensor-language tokenizer: 8^5.
    ///
    /// The size is chosen to sit alongside a text vocabulary in one embedding table, so a decoder
    /// reads a signal token and a word token through the same lookup.
    pub fn signal_15bit() -> Self {
        Self::new(vec![8; 5]).expect("8^5 is a valid level list")
    }

    #[inline]
    pub fn levels(&self) -> &[u32] {
        &self.levels
    }

    /// Number of latent dimensions, which is the number of floats per code.
    #[inline]
    pub fn dim(&self) -> usize {
        self.levels.len()
    }

    /// Total number of distinct codes: the product of the level counts.
    #[inline]
    pub fn codebook_size(&self) -> u32 {
        self.levels.iter().fold(1u32, |a, &l| a * l)
    }

    /// Half-width and offset for one dimension, following the paper's bounding function.
    ///
    /// For odd `L` the levels straddle zero symmetrically. For even `L` the paper shifts by half a
    /// level so the set stays symmetric; the `shift` below is applied *inside* `tanh` so that an
    /// input of zero lands between the two central levels rather than on one of them.
    #[inline]
    fn bound_params(l: u32) -> (f32, f32, f32) {
        const EPS: f32 = 1e-3;
        let half_l = (l as f32 - 1.0) * (1.0 - EPS) / 2.0;
        let offset = if l % 2 == 0 { 0.5f32 } else { 0.0f32 };
        // `tan`, NOT `atanh`. Reading "shift the tanh so the levels stay symmetric" as an inverse
        // hyperbolic tangent is the intuitive move and it is wrong: at L=2 the argument is
        // 0.5/0.4995 > 1, atanh is undefined there, and the shift comes out NaN. The tokenizer
        // then never reaches level 0 — it still runs, still round-trips, and quietly halves the
        // resolution of every two-level dimension. The paper's own implementation uses tan.
        let shift = (offset / half_l).tan();
        (half_l, offset, shift)
    }

    /// Squash one latent value into the quantizer's bounded range, before rounding.
    #[inline]
    pub fn bound(&self, dim: usize, z: f32) -> f32 {
        let (half_l, offset, shift) = Self::bound_params(self.levels[dim]);
        (z + shift).tanh() * half_l - offset
    }


    /// Per-dimension `(half_width, offset, shift)` for the bounding function.
    ///
    /// Exposed so a differentiable graph can apply the SAME bound the quantizer uses, rather than
    /// reimplementing it and drifting.
    pub fn bound_terms(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut half = Vec::with_capacity(self.dim());
        let mut off = Vec::with_capacity(self.dim());
        let mut sh = Vec::with_capacity(self.dim());
        for &l in &self.levels {
            let (h, o, s) = Self::bound_params(l);
            half.push(h);
            off.push(o);
            sh.push(s);
        }
        (half, off, sh)
    }

    /// Round an ALREADY-BOUNDED value to its level index. The counterpart to `bound_terms`, for a
    /// caller that applied the bound itself inside an autograd graph.
    pub fn round_bounded(&self, dim: usize, b: f32) -> u32 {
        let l = self.levels[dim];
        let half = (l / 2) as i64;
        let r = if b >= 0.0 { (b + 0.5).floor() } else { (b - 0.5).ceil() } as i64;
        (r.clamp(-half, -half + l as i64 - 1) + half) as u32
    }

    /// Quantize a latent vector to the nearest code, returned as integer level indices in
    /// `0..levels[d]`.
    ///
    /// Input shorter than `dim()` is an error rather than a zero-fill: silently padding a sensor
    /// latent produces a valid-looking token for a signal that was never measured.
    pub fn quantize(&self, z: &[f32]) -> Result<Vec<u32>, FsqError> {
        if z.len() != self.dim() {
            return Err(FsqError::DegenerateLevel { dim: z.len(), level: self.dim() as u32 });
        }
        Ok(z
            .iter()
            .enumerate()
            .map(|(d, &v)| {
                let l = self.levels[d];
                let half = (l / 2) as i64;
                // Round-half-away-from-zero, then clamp. The clamp is load-bearing: tanh can reach
                // the bound in f32 and round can then land one past the top level.
                let b = self.bound(d, v);
                let r = if b >= 0.0 { (b + 0.5).floor() } else { (b - 0.5).ceil() } as i64;
                let lo = -half;
                let hi = lo + l as i64 - 1;
                (r.clamp(lo, hi) - lo) as u32
            })
            .collect())
    }

    /// The centre value of each level, in the same bounded units `bound` produces. This is what a
    /// decoder consumes: the code as a vector, not as an index.
    pub fn dequantize(&self, code: &[u32]) -> Result<Vec<f32>, FsqError> {
        if code.len() != self.dim() {
            return Err(FsqError::DegenerateLevel { dim: code.len(), level: self.dim() as u32 });
        }
        let mut out = Vec::with_capacity(code.len());
        for (d, &c) in code.iter().enumerate() {
            let l = self.levels[d];
            if c >= l {
                return Err(FsqError::DegenerateLevel { dim: d, level: c });
            }
            let half = (l / 2) as i64;
            out.push((c as i64 - half) as f32);
        }
        Ok(out)
    }

    /// Centre value of one level in one dimension, in bounded units.
    pub fn dequantize_dim(&self, dim: usize, code: u32) -> f32 {
        (code as i64 - (self.levels[dim] / 2) as i64) as f32
    }

    /// Pack a code into a single token id, mixed-radix, dimension 0 least significant.
    pub fn to_index(&self, code: &[u32]) -> Result<u32, FsqError> {
        if code.len() != self.dim() {
            return Err(FsqError::DegenerateLevel { dim: code.len(), level: self.dim() as u32 });
        }
        let mut idx: u32 = 0;
        let mut radix: u32 = 1;
        for (d, &c) in code.iter().enumerate() {
            let l = self.levels[d];
            if c >= l {
                return Err(FsqError::DegenerateLevel { dim: d, level: c });
            }
            idx += c * radix;
            radix *= l;
        }
        Ok(idx)
    }

    /// Unpack a token id back into a code. Inverse of [`Fsq::to_index`] over the whole space.
    pub fn from_index(&self, mut idx: u32) -> Result<Vec<u32>, FsqError> {
        if idx >= self.codebook_size() {
            return Err(FsqError::DegenerateLevel { dim: usize::MAX, level: idx });
        }
        let mut code = Vec::with_capacity(self.dim());
        for &l in &self.levels {
            code.push(idx % l);
            idx /= l;
        }
        Ok(code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_size_is_the_product_of_levels() {
        assert_eq!(Fsq::new(vec![8; 5]).unwrap().codebook_size(), 32_768);
        assert_eq!(Fsq::new(vec![8, 5, 5, 5]).unwrap().codebook_size(), 1_000);
        assert_eq!(Fsq::signal_15bit().codebook_size(), 1 << 15);
    }

    /// EXHAUSTIVE, not sampled. Every one of the 32,768 ids must survive the round trip, because a
    /// tokenizer that is a bijection on 99.99% of its space still corrupts one signal in ten
    /// thousand and nothing downstream would report it.
    #[test]
    fn index_and_code_are_a_bijection_over_the_entire_space() {
        let q = Fsq::signal_15bit();
        let n = q.codebook_size();
        let mut seen = vec![false; n as usize];
        for idx in 0..n {
            let code = q.from_index(idx).unwrap();
            assert_eq!(code.len(), q.dim());
            let back = q.to_index(&code).unwrap();
            assert_eq!(back, idx, "round trip failed at {idx}");
            assert!(!seen[idx as usize], "index {idx} produced twice");
            seen[idx as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "some code was unreachable");
    }

    #[test]
    fn bijection_holds_for_a_mixed_radix_too() {
        let q = Fsq::new(vec![3, 4, 5]).unwrap();
        for idx in 0..q.codebook_size() {
            assert_eq!(q.to_index(&q.from_index(idx).unwrap()).unwrap(), idx);
        }
    }

    /// The clamp in `quantize` exists for this: tanh saturates, and rounding a saturated bound can
    /// otherwise land one level past the top. Sweep well past saturation in both directions.
    #[test]
    fn quantize_never_leaves_the_valid_level_range() {
        for l in [2u32, 3, 4, 5, 8, 16] {
            let q = Fsq::new(vec![l]).unwrap();
            let mut z = -60.0f32;
            while z <= 60.0 {
                let c = q.quantize(&[z]).unwrap();
                assert!(c[0] < l, "level {} out of range for L={l} at z={z}", c[0]);
                // And it must be packable, which is the property the decoder depends on.
                assert!(q.to_index(&c).unwrap() < q.codebook_size());
                z += 0.25;
            }
        }
    }

    /// Both extremes must be reachable. An even-L quantizer without the half-level shift silently
    /// loses its top level, which looks like a slightly lossy tokenizer rather than a bug.
    #[test]
    fn saturating_inputs_reach_the_lowest_and_highest_levels() {
        for l in [2u32, 3, 4, 5, 8, 16] {
            let q = Fsq::new(vec![l]).unwrap();
            assert_eq!(q.quantize(&[-40.0]).unwrap()[0], 0, "L={l} never reaches level 0");
            assert_eq!(q.quantize(&[40.0]).unwrap()[0], l - 1, "L={l} never reaches the top level");
        }
    }

    #[test]
    fn quantization_is_monotone_in_the_input() {
        let q = Fsq::new(vec![8]).unwrap();
        let mut prev = 0u32;
        let mut z = -20.0f32;
        while z <= 20.0 {
            let c = q.quantize(&[z]).unwrap()[0];
            assert!(c >= prev, "quantizer went backwards at z={z}: {prev} -> {c}");
            prev = c;
            z += 0.05;
        }
    }

    #[test]
    fn dequantize_is_centred_and_ordered() {
        let q = Fsq::new(vec![8]).unwrap();
        let mut last = f32::NEG_INFINITY;
        for c in 0..8u32 {
            let v = q.dequantize(&[c]).unwrap()[0];
            assert!(v > last, "levels not ordered at {c}");
            last = v;
        }
    }

    #[test]
    fn bad_level_lists_are_refused_rather_than_quantizing_wrongly() {
        assert_eq!(Fsq::new(vec![]), Err(FsqError::Empty));
        assert_eq!(Fsq::new(vec![8, 1]), Err(FsqError::DegenerateLevel { dim: 1, level: 1 }));
        assert_eq!(Fsq::new(vec![65_536, 65_536, 4]), Err(FsqError::CodebookTooLarge));
    }

    #[test]
    fn wrong_width_input_is_an_error_not_a_zero_fill() {
        let q = Fsq::signal_15bit();
        assert!(q.quantize(&[0.0, 0.0]).is_err());
        assert!(q.dequantize(&[0, 0]).is_err());
        assert!(q.to_index(&[0, 0]).is_err());
        assert!(q.from_index(q.codebook_size()).is_err());
    }

    #[test]
    fn quantization_is_deterministic() {
        let q = Fsq::signal_15bit();
        let z = [0.31, -1.7, 4.2, -0.03, 0.98];
        let a = q.quantize(&z).unwrap();
        for _ in 0..64 {
            assert_eq!(q.quantize(&z).unwrap(), a);
        }
    }
}
