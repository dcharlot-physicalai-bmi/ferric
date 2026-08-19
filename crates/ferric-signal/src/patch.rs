//! The signal front end: patching and reversible instance normalization.
//!
//! Before a transformer sees a sensor stream, two things happen to it, and both are exactly
//! invertible on purpose. A patcher cuts the stream into fixed windows that become the sequence
//! positions. A reversible normalizer removes each channel's own level and scale, so the model
//! learns shape rather than units, and puts them back afterwards so a reconstruction comes out in
//! volts rather than in standard deviations.
//!
//! Both are pure arithmetic with no learned parameters, which means both can be checked against
//! their own inverse rather than against a reference model. That is the whole reason to build them
//! first: they are the part of the pipeline where correctness is decidable today.
//!
//! ## The trap this module is built around
//!
//! **A flatlined channel has zero variance, and dividing by it produces NaN.** A stuck sensor is
//! not an edge case in this domain — it is one of the specific conditions an anomaly detector
//! exists to report. A normalizer that returns NaN on the exact input you most need to classify
//! will poison every token in the window, and the failure surfaces far downstream as a model that
//! "cannot detect stuck sensors". [`RevIn`] floors the scale and the behaviour is tested directly.

/// How a signal is cut into transformer positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Patcher {
    patch_len: usize,
    stride: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatchError {
    /// A zero-length patch or stride would loop forever or produce no positions.
    Degenerate { patch_len: usize, stride: usize },
    /// The signal is shorter than one patch, so there is nothing to encode.
    TooShort { len: usize, patch_len: usize },
    /// Channel count does not divide the buffer.
    Ragged { len: usize, channels: usize },
}

impl core::fmt::Display for PatchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PatchError::Degenerate { patch_len, stride } => {
                write!(f, "patch: patch_len={patch_len} stride={stride}; both must be >= 1")
            }
            PatchError::TooShort { len, patch_len } => {
                write!(f, "patch: signal of {len} samples is shorter than one {patch_len}-sample patch")
            }
            PatchError::Ragged { len, channels } => {
                write!(f, "patch: {len} samples do not divide into {channels} channels")
            }
        }
    }
}

impl Patcher {
    pub fn new(patch_len: usize, stride: usize) -> Result<Self, PatchError> {
        if patch_len == 0 || stride == 0 {
            return Err(PatchError::Degenerate { patch_len, stride });
        }
        Ok(Self { patch_len, stride })
    }

    /// Non-overlapping patches, the configuration whose inverse is exact.
    pub fn contiguous(patch_len: usize) -> Result<Self, PatchError> {
        Self::new(patch_len, patch_len)
    }

    #[inline]
    pub fn patch_len(&self) -> usize {
        self.patch_len
    }
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// How many patches a signal of `len` samples yields.
    ///
    /// Samples in a trailing remainder shorter than one patch are **dropped**, and
    /// [`Patcher::covered`] reports exactly how many were kept so a caller can never mistake a
    /// truncated encode for a complete one.
    pub fn count(&self, len: usize) -> usize {
        if len < self.patch_len {
            0
        } else {
            (len - self.patch_len) / self.stride + 1
        }
    }

    /// Number of leading samples the patches actually span. `len - covered` is the tail that will
    /// not survive a round trip.
    pub fn covered(&self, len: usize) -> usize {
        match self.count(len) {
            0 => 0,
            n => (n - 1) * self.stride + self.patch_len,
        }
    }

    /// Cut one channel into patches, row-major: `out[i * patch_len + j]`.
    pub fn patchify(&self, signal: &[f32]) -> Result<Vec<f32>, PatchError> {
        let n = self.count(signal.len());
        if n == 0 {
            return Err(PatchError::TooShort { len: signal.len(), patch_len: self.patch_len });
        }
        let mut out = Vec::with_capacity(n * self.patch_len);
        for i in 0..n {
            let s = i * self.stride;
            out.extend_from_slice(&signal[s..s + self.patch_len]);
        }
        Ok(out)
    }

    /// Rebuild a signal from patches. Overlapping positions are averaged, which is the inverse of
    /// `patchify` wherever the patches actually covered a sample.
    ///
    /// Returns a buffer of `covered(original_len)` samples; the dropped tail is not invented.
    pub fn unpatchify(&self, patches: &[f32]) -> Result<Vec<f32>, PatchError> {
        if patches.len() % self.patch_len != 0 {
            return Err(PatchError::Ragged { len: patches.len(), channels: self.patch_len });
        }
        let n = patches.len() / self.patch_len;
        if n == 0 {
            return Err(PatchError::TooShort { len: 0, patch_len: self.patch_len });
        }
        let out_len = (n - 1) * self.stride + self.patch_len;
        let mut acc = vec![0.0f32; out_len];
        let mut hits = vec![0u32; out_len];
        for i in 0..n {
            let s = i * self.stride;
            for j in 0..self.patch_len {
                acc[s + j] += patches[i * self.patch_len + j];
                hits[s + j] += 1;
            }
        }
        for (v, h) in acc.iter_mut().zip(hits) {
            // `hits` is >= 1 everywhere by construction: position s+j is written by patch i.
            *v /= h as f32;
        }
        Ok(acc)
    }
}

/// Per-channel level and scale, removed before encoding and restored after decoding.
///
/// This is the reversible instance normalization used across time-series transformers: each
/// channel of each window is standardised by its own statistics, so a model sees shape rather than
/// units and a 3-volt vibration and a 3-millivolt one tokenize the same way.
#[derive(Debug, Clone, PartialEq)]
pub struct RevIn {
    /// Per-channel mean, in the original units.
    pub mean: Vec<f32>,
    /// Per-channel scale actually used, already floored. Never zero.
    pub scale: Vec<f32>,
}

impl RevIn {
    /// The smallest scale that will ever be divided by.
    ///
    /// Chosen well above f32 subnormals so that `x / scale` cannot overflow to infinity for any
    /// residual a real sensor produces.
    pub const MIN_SCALE: f32 = 1e-6;

    /// Measure per-channel statistics from an interleaved multi-channel buffer
    /// (`sample0_ch0, sample0_ch1, …`).
    pub fn fit(signal: &[f32], channels: usize) -> Result<Self, PatchError> {
        if channels == 0 || signal.len() % channels != 0 {
            return Err(PatchError::Ragged { len: signal.len(), channels });
        }
        let n = signal.len() / channels;
        // ACCUMULATE IN f64. Summing thousands of f32 samples in f32 leaves the mean off by a few
        // ulp, and for a CONSTANT channel that residue is the entire signal: the measured standard
        // deviation stops being zero and becomes pure accumulation dust. Measured on a channel
        // stuck at 3.3 V over 100 samples, f32 accumulation gives sd = 1.19e-6 — just above the
        // floor below, so the floor never binds — and every sample of a flatlined sensor then
        // normalises to ±1.0. A stuck channel arrives at the model looking like a vigorous one,
        // which is silent, unlike the NaN this floor was originally written to prevent.
        let mut mean64 = vec![0.0f64; channels];
        for (i, &v) in signal.iter().enumerate() {
            mean64[i % channels] += v as f64;
        }
        for m in mean64.iter_mut() {
            *m /= n as f64;
        }
        let mean: Vec<f32> = mean64.iter().map(|&m| m as f32).collect();
        let mut var = vec![0.0f64; channels];
        for (i, &v) in signal.iter().enumerate() {
            let c = i % channels;
            let d = v as f64 - mean64[c];
            var[c] += d * d;
        }
        let scale = var
            .iter()
            .map(|&s| {
                let s = s as f32;
                // THE FLATLINE FLOOR. A stuck channel has variance exactly 0, and a stuck channel
                // is a thing this domain must be able to report rather than crash on. Without the
                // floor every sample of that window becomes NaN, every token derived from it is
                // garbage, and the symptom appears much later as "the model cannot see stuck
                // sensors" — which is true, and has nothing to do with the model.
                let sd = (s / n as f32).sqrt();
                if sd.is_finite() && sd > Self::MIN_SCALE { sd } else { Self::MIN_SCALE }
            })
            .collect();
        Ok(Self { mean, scale })
    }

    #[inline]
    pub fn channels(&self) -> usize {
        self.mean.len()
    }

    /// Remove level and scale, in place semantics on a copy.
    pub fn apply(&self, signal: &[f32]) -> Result<Vec<f32>, PatchError> {
        let c = self.channels();
        if c == 0 || signal.len() % c != 0 {
            return Err(PatchError::Ragged { len: signal.len(), channels: c });
        }
        Ok(signal
            .iter()
            .enumerate()
            .map(|(i, &v)| (v - self.mean[i % c]) / self.scale[i % c])
            .collect())
    }

    /// Put level and scale back. Exact inverse of [`RevIn::apply`] up to float rounding.
    pub fn invert(&self, norm: &[f32]) -> Result<Vec<f32>, PatchError> {
        let c = self.channels();
        if c == 0 || norm.len() % c != 0 {
            return Err(PatchError::Ragged { len: norm.len(), channels: c });
        }
        Ok(norm
            .iter()
            .enumerate()
            .map(|(i, &v)| v * self.scale[i % c] + self.mean[i % c])
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32 * 0.37 - 5.0).collect()
    }

    #[test]
    fn contiguous_patching_round_trips_exactly() {
        let p = Patcher::contiguous(16).unwrap();
        let s = ramp(16 * 7);
        let back = p.unpatchify(&p.patchify(&s).unwrap()).unwrap();
        assert_eq!(back.len(), s.len());
        for (a, b) in s.iter().zip(&back) {
            assert_eq!(a, b, "contiguous patching must be bit-exact");
        }
    }

    /// Overlap-add divides by the hit count, so it must reconstruct the covered span. Any
    /// off-by-one in `stride` shows up here as a visible error rather than a plausible signal.
    #[test]
    fn overlapping_patching_reconstructs_the_covered_span() {
        for (pl, st) in [(16usize, 8usize), (16, 4), (10, 3), (8, 7)] {
            let p = Patcher::new(pl, st).unwrap();
            let s = ramp(200);
            let back = p.unpatchify(&p.patchify(&s).unwrap()).unwrap();
            assert_eq!(back.len(), p.covered(s.len()), "pl={pl} st={st}");
            for (i, (a, b)) in s.iter().zip(&back).enumerate() {
                assert!((a - b).abs() < 1e-4, "pl={pl} st={st} sample {i}: {a} vs {b}");
            }
        }
    }

    /// The dropped tail is REPORTED, not silently absorbed. A caller that encodes 1000 samples and
    /// gets 992 back must be able to learn that from the API rather than from a diff.
    #[test]
    fn the_uncovered_tail_is_countable() {
        let p = Patcher::new(16, 8).unwrap();
        assert_eq!(p.count(100), 11);
        assert_eq!(p.covered(100), 96);
        assert_eq!(100 - p.covered(100), 4);
        let p2 = Patcher::contiguous(16).unwrap();
        assert_eq!(p2.count(100), 6);
        assert_eq!(p2.covered(100), 96);
    }

    #[test]
    fn degenerate_and_short_inputs_are_refused() {
        assert!(Patcher::new(0, 1).is_err());
        assert!(Patcher::new(8, 0).is_err());
        let p = Patcher::contiguous(16).unwrap();
        assert_eq!(p.count(15), 0);
        assert!(p.patchify(&ramp(15)).is_err());
    }

    #[test]
    fn revin_round_trips_a_multichannel_signal() {
        let c = 3;
        let s: Vec<f32> = (0..300)
            .map(|i| {
                let ch = i % c;
                (i as f32 * 0.1).sin() * (ch as f32 + 1.0) * 12.0 + ch as f32 * 100.0
            })
            .collect();
        let r = RevIn::fit(&s, c).unwrap();
        let back = r.invert(&r.apply(&s).unwrap()).unwrap();
        for (i, (a, b)) in s.iter().zip(&back).enumerate() {
            assert!((a - b).abs() < 1e-3, "sample {i}: {a} vs {b}");
        }
    }

    #[test]
    fn revin_actually_standardises_each_channel_independently() {
        let c = 2;
        // Channel 0 is tiny and offset; channel 1 is large. After normalization both should look
        // the same to the model, which is the entire point.
        let s: Vec<f32> = (0..400)
            .map(|i| if i % 2 == 0 { (i as f32 * 0.05).sin() * 0.002 + 7.0 }
                     else { (i as f32 * 0.05).sin() * 900.0 - 4000.0 })
            .collect();
        let r = RevIn::fit(&s, c).unwrap();
        let n = r.apply(&s).unwrap();
        for ch in 0..c {
            let vals: Vec<f32> = n.iter().skip(ch).step_by(c).copied().collect();
            let m = vals.iter().sum::<f32>() / vals.len() as f32;
            let sd = (vals.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / vals.len() as f32).sqrt();
            assert!(m.abs() < 1e-2, "channel {ch} mean {m} not centred");
            assert!((sd - 1.0).abs() < 1e-2, "channel {ch} sd {sd} not unit");
        }
    }

    /// THE FLATLINE. A stuck channel is a condition this domain exists to detect, and a naive
    /// standardiser turns it into NaN — poisoning every token in the window and hiding the very
    /// event that mattered.
    #[test]
    fn a_flatlined_channel_normalises_without_nan_and_still_inverts() {
        let c = 2;
        // Channel 0 is stuck at 3.3 V. Channel 1 is alive.
        let s: Vec<f32> = (0..200)
            .map(|i| if i % 2 == 0 { 3.3 } else { (i as f32 * 0.07).cos() * 2.0 })
            .collect();
        let r = RevIn::fit(&s, c).unwrap();
        assert_eq!(r.scale[0], RevIn::MIN_SCALE, "flat channel scale was not floored");
        let n = r.apply(&s).unwrap();
        assert!(n.iter().all(|v| v.is_finite()), "normalization produced a non-finite value");
        // THE ASSERTION THAT MATTERS, and the one this test lacked at first. Finiteness alone is
        // satisfied by the failure mode: with an f32-accumulated mean the stuck channel had
        // sd = 1.19e-6, cleared the floor, and normalised to ±1.0 — a flatlined sensor arriving as
        // a vigorous one. A flat channel must come out FLAT.
        for (k, v) in n.iter().step_by(2).enumerate() {
            assert!(v.abs() < 1e-3, "stuck channel normalised to {v} at sample {k}, not flat");
        }
        let back = r.invert(&n).unwrap();
        for (i, (a, b)) in s.iter().zip(&back).enumerate() {
            assert!((a - b).abs() < 1e-3, "sample {i}: {a} vs {b}");
        }
        // And the stuck channel must come back at its stuck value, not at zero.
        for v in back.iter().step_by(2) {
            assert!((v - 3.3).abs() < 1e-3, "stuck channel did not survive: {v}");
        }
    }

    #[test]
    fn an_all_zero_channel_is_also_safe() {
        let r = RevIn::fit(&vec![0.0f32; 128], 1).unwrap();
        let n = r.apply(&vec![0.0f32; 128]).unwrap();
        assert!(n.iter().all(|v| v.is_finite() && *v == 0.0));
        assert!(r.invert(&n).unwrap().iter().all(|v| *v == 0.0));
    }

    #[test]
    fn ragged_channel_counts_are_refused() {
        assert!(RevIn::fit(&ramp(101), 2).is_err());
        assert!(RevIn::fit(&ramp(100), 0).is_err());
        let r = RevIn::fit(&ramp(100), 2).unwrap();
        assert!(r.apply(&ramp(101)).is_err());
        assert!(r.invert(&ramp(101)).is_err());
    }

    /// The composition is what the pipeline actually runs: normalize, patch, unpatch, denormalize.
    #[test]
    fn the_whole_front_end_round_trips() {
        let c = 1;
        let s = ramp(16 * 12);
        let r = RevIn::fit(&s, c).unwrap();
        let p = Patcher::contiguous(16).unwrap();
        let out = r.invert(&p.unpatchify(&p.patchify(&r.apply(&s).unwrap()).unwrap()).unwrap()).unwrap();
        assert_eq!(out.len(), s.len());
        for (i, (a, b)) in s.iter().zip(&out).enumerate() {
            assert!((a - b).abs() < 1e-3, "sample {i}: {a} vs {b}");
        }
    }
}
