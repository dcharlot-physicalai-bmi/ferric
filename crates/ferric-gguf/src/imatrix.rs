//! **Importance-matrix calibration** — quantize where it matters, using real activations.
//!
//! A quantizer that minimises `Σ (w − ŵ)²` treats every weight as equally important. It is not: a weight
//! multiplying an input channel that is usually near zero contributes almost nothing to the output, while
//! a weight on a high-energy channel contributes a lot. An *importance matrix* is one number per input
//! channel, measured by running real text through the model, and it lets the quantizer minimise the
//! error that reaches the **output** rather than the error in the weights.
//!
//! This closes a capability gap — Ferric had no imatrix at all, while every production 2-bit engine
//! depends on one. ds4/DwarfStar goes further and marks `IQ2_XXS` as *requiring* an importance vector,
//! falling back to a weight-energy heuristic only with a documented warning that it is inferior.
//!
//! ## What is measured, and where
//!
//! Per input channel of each quantized projection, accumulate `Σ x²` over the calibration corpus — where
//! `x` is **the activation that projection actually consumes**:
//!
//! | projection | its input |
//! |---|---|
//! | `ffn_gate` / `ffn_up` | the FFN-normalised hidden state |
//! | `ffn_down` | the SwiGLU product, *after* gating |
//! | `wqkv` | the attention-normalised hidden state |
//! | `wo` | the attention output |
//!
//! The distinction matters: `ffn_down` sees a gated, one-sided distribution and `ffn_gate` sees a
//! near-symmetric one, so a single shared statistic would misweight both.
//!
//! ## Honest expectations
//!
//! ds4 publishes the only measured delta for its imatrix pipeline: **−1.95% NLL** (0.1774 → 0.1739) over
//! 100 continuations. Real, and modest. Treat a larger claim with suspicion.

use std::collections::HashMap;

/// Per-tensor importance vectors, indexed by tensor name.
#[derive(Debug, Clone, Default)]
pub struct Imatrix {
    /// `name -> (values, ncall)`. `values.len()` is the tensor's input width (or `n_expert × width` for
    /// a packed MoE entry).
    entries: HashMap<String, (Vec<f32>, u32)>,
    pub dataset: String,
    pub chunks: u32,
}

impl Imatrix {
    pub fn new() -> Self { Self::default() }

    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    pub fn names(&self) -> impl Iterator<Item = &String> { self.entries.keys() }

    /// Accumulate `Σ x²` for one observation. `act` is `[rows, cols]` row-major; each row is one token's
    /// activation, and `cols` must equal the projection's input width.
    pub fn accumulate(&mut self, name: &str, act: &[f32], cols: usize) {
        debug_assert!(cols > 0 && act.len() % cols == 0, "activation is not [rows, {cols}]");
        let e = self.entries.entry(name.to_string()).or_insert_with(|| (vec![0.0; cols], 0));
        if e.0.len() != cols { return; } // shape changed: ignore rather than corrupt a partial vector
        for row in act.chunks_exact(cols) {
            for (acc, &x) in e.0.iter_mut().zip(row) {
                *acc += x * x;
            }
        }
        e.1 += (act.len() / cols) as u32;
    }

    /// Importance for `name`, normalised by its own observation count.
    ///
    /// Returns `None` when nothing was recorded — which the caller must handle explicitly rather than
    /// silently substituting zeros. Zero importance would tell the quantizer that *no* weight in the
    /// tensor matters, which is the opposite of the intended default.
    pub fn get(&self, name: &str) -> Option<Vec<f32>> {
        let (v, n) = self.entries.get(name)?;
        if *n == 0 { return None; }
        let inv = 1.0 / *n as f32;
        Some(v.iter().map(|x| x * inv).collect())
    }

    /// Importance for `name`, or an all-ones vector of length `cols`.
    ///
    /// All-ones is exactly "no opinion" — it reduces the weighted quantizer to the unweighted one — which
    /// is why ds4 writes an all-1.0 vector for experts its calibration corpus never selected, rather than
    /// leaving them absent and letting a downstream tool guess.
    pub fn get_or_uniform(&self, name: &str, cols: usize) -> Vec<f32> {
        self.get(name).filter(|v| v.len() == cols).unwrap_or_else(|| vec![1.0; cols])
    }

    /// Serialise to the llama.cpp legacy `.dat` format, so the file is usable outside Ferric.
    ///
    /// ```text
    /// i32 n_entries
    /// per entry:  i32 name_len | name bytes | i32 ncall | i32 nval | f32 values[nval]
    /// i32 chunks
    /// i32 dataset_len | dataset bytes
    /// ```
    /// All little-endian. Entries are written in sorted name order so the output is reproducible.
    pub fn to_dat(&self) -> Vec<u8> {
        let mut o = Vec::new();
        let mut names: Vec<&String> = self.entries.keys().collect();
        names.sort();
        o.extend_from_slice(&(names.len() as i32).to_le_bytes());
        for n in names {
            let (v, ncall) = &self.entries[n];
            o.extend_from_slice(&(n.len() as i32).to_le_bytes());
            o.extend_from_slice(n.as_bytes());
            o.extend_from_slice(&(*ncall as i32).to_le_bytes());
            o.extend_from_slice(&(v.len() as i32).to_le_bytes());
            for x in v { o.extend_from_slice(&x.to_le_bytes()); }
        }
        o.extend_from_slice(&(self.chunks as i32).to_le_bytes());
        o.extend_from_slice(&(self.dataset.len() as i32).to_le_bytes());
        o.extend_from_slice(self.dataset.as_bytes());
        o
    }

    pub fn from_dat(b: &[u8]) -> Result<Self, String> {
        let mut p = 0usize;
        let i32_at = |b: &[u8], p: &mut usize| -> Result<i32, String> {
            if *p + 4 > b.len() { return Err("truncated .dat".into()); }
            let v = i32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
            *p += 4;
            Ok(v)
        };
        let n = i32_at(b, &mut p)?;
        if !(0..=1_000_000).contains(&n) { return Err(format!("implausible entry count {n}")); }
        let mut me = Imatrix::new();
        for _ in 0..n {
            let nl = i32_at(b, &mut p)? as usize;
            if p + nl > b.len() { return Err("truncated name".into()); }
            let name = String::from_utf8_lossy(&b[p..p + nl]).into_owned();
            p += nl;
            let ncall = i32_at(b, &mut p)? as u32;
            let nval = i32_at(b, &mut p)? as usize;
            if p + nval * 4 > b.len() { return Err(format!("truncated values for '{name}'")); }
            let mut v = Vec::with_capacity(nval);
            for k in 0..nval {
                v.push(f32::from_le_bytes(b[p + k * 4..p + k * 4 + 4].try_into().unwrap()));
            }
            p += nval * 4;
            me.entries.insert(name, (v, ncall));
        }
        if p + 4 <= b.len() { me.chunks = i32_at(b, &mut p)? as u32; }
        if p + 4 <= b.len() {
            let dl = i32_at(b, &mut p)? as usize;
            if p + dl <= b.len() { me.dataset = String::from_utf8_lossy(&b[p..p + dl]).into_owned(); }
        }
        Ok(me)
    }
}

/// Threshold candidates as multiples of the group's mean magnitude, swept per group.
///
/// A single fixed threshold (the usual `0.7 · mean|w|`) is optimal only for a particular weight
/// distribution. Sweeping and scoring under the importance-weighted objective is what lets importance
/// actually change the result — with one fixed threshold, importance can only move the scale.
const THRESHOLDS: [f32; 9] = [0.35, 0.45, 0.55, 0.65, 0.70, 0.75, 0.85, 0.95, 1.05];

/// Importance-weighted ternary quantization, returning the dequantized reconstruction.
///
/// `w` is `[rows, cols]` row-major; `imp` is one value per **input channel** (length `cols`), shared by
/// every row — importance is a property of the activation, not of the weight.
///
/// # The objective
///
/// Minimise `Σ_i weight_i · (w_i − a·t_i)²` per group, with `t_i ∈ {−1,0,+1}` and
///
/// ```text
/// weight_i = imp[col_i] · sqrt(sigma2 + w_i²),   sigma2 = Σ_j w_j² / n
/// ```
///
/// which is ds4's form: importance from the activation, times a local-magnitude term so that a large
/// weight on a modest channel is not dismissed. Given the assignment, the optimal scale is the
/// importance-weighted mean magnitude of the non-zero entries — that closed form is why the sweep is
/// cheap.
///
/// Passing `None` for `imp` gives plain unweighted ternary over the same sweep, which is the correct
/// control: it isolates the effect of importance rather than confounding it with the sweep itself.
pub fn quantize_ternary_weighted(w: &[f32], cols: usize, group: usize, imp: Option<&[f32]>) -> Vec<f32> {
    assert!(cols > 0 && w.len() % cols == 0, "weights are not [rows, {cols}]");
    if let Some(v) = imp {
        assert_eq!(v.len(), cols, "importance must have one entry per input channel");
    }
    let mut out = vec![0f32; w.len()];

    for (r, row) in w.chunks_exact(cols).enumerate() {
        let orow = &mut out[r * cols..(r + 1) * cols];
        for g0 in (0..cols).step_by(group) {
            let g1 = (g0 + group).min(cols);
            let gw = &row[g0..g1];
            let n = gw.len() as f32;

            let sumsq: f32 = gw.iter().map(|x| x * x).sum();
            let sigma2 = sumsq / n;
            let mean_abs: f32 = gw.iter().map(|x| x.abs()).sum::<f32>() / n;
            if mean_abs == 0.0 {
                continue; // an all-zero group reconstructs exactly as zero
            }

            // Per-element objective weight.
            let wt = |i: usize| -> f32 {
                let local = (sigma2 + gw[i] * gw[i]).sqrt();
                match imp {
                    Some(v) => v[g0 + i] * local,
                    None => local,
                }
            };

            let (mut best_err, mut best_a, mut best_d) = (f32::INFINITY, 0f32, 0f32);
            for &tf in &THRESHOLDS {
                let d = tf * mean_abs;
                // Closed-form optimal scale for this assignment.
                let (mut num, mut den) = (0f32, 0f32);
                for i in 0..gw.len() {
                    if gw[i].abs() > d {
                        let k = wt(i);
                        num += k * gw[i].abs();
                        den += k;
                    }
                }
                if den == 0.0 { continue; } // this threshold zeroes the whole group
                let a = num / den;
                let mut err = 0f32;
                for i in 0..gw.len() {
                    let q = if gw[i].abs() > d { a * gw[i].signum() } else { 0.0 };
                    let e = gw[i] - q;
                    err += wt(i) * e * e;
                }
                if err < best_err {
                    best_err = err;
                    best_a = a;
                    best_d = d;
                }
            }
            if best_err.is_finite() {
                for i in 0..gw.len() {
                    orow[g0 + i] = if gw[i].abs() > best_d { best_a * gw[i].signum() } else { 0.0 };
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    #[test]
    fn dat_round_trips() {
        let mut m = Imatrix::new();
        m.accumulate("blk.0.ffn_down.weight", &[1.0, 2.0, 3.0, 4.0], 2);
        m.accumulate("blk.0.ffn_gate.weight", &[0.5, -0.5], 2);
        m.dataset = "ferric-test".into();
        m.chunks = 7;
        let back = Imatrix::from_dat(&m.to_dat()).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back.chunks, 7);
        assert_eq!(back.dataset, "ferric-test");
        // 2 rows of [1,2] and [3,4] -> sums of squares [1+9, 4+16] = [10, 20], ncall 2 -> [5, 10]
        assert_eq!(back.get("blk.0.ffn_down.weight").unwrap(), vec![5.0, 10.0]);
    }

    #[test]
    fn truncation_inside_the_entries_is_an_error() {
        // Cutting into the entries loses weights and must be refused — a partially-loaded importance
        // vector would quantize part of a tensor against zeros.
        let mut m = Imatrix::new();
        m.accumulate("t", &[1.0, 2.0], 2);
        let d = m.to_dat();
        // entries section = 4 (count) + 4 + 1 (name) + 4 (ncall) + 4 (nval) + 8 (values) = 25 bytes
        for cut in [1usize, 5, 9, 17, 24] {
            assert!(Imatrix::from_dat(&d[..cut]).is_err(), "accepted a .dat truncated to {cut} bytes");
        }
    }

    #[test]
    fn a_file_without_the_optional_trailer_still_loads_its_entries() {
        // `chunks` and `dataset` are trailing provenance fields; legacy writers omit them. Refusing such
        // a file would reject valid llama.cpp imatrices, so the parser treats them as optional — and this
        // pins that as intended behaviour rather than an accident of the bounds checks.
        let mut m = Imatrix::new();
        m.accumulate("t", &[1.0, 2.0], 2);
        m.dataset = "x".into();
        m.chunks = 3;
        let d = m.to_dat();
        let entries_only = &d[..25];
        let back = Imatrix::from_dat(entries_only).expect("entries-only .dat should load");
        assert_eq!(back.get("t").unwrap(), vec![1.0, 4.0]);
        assert_eq!(back.chunks, 0, "absent trailer must not invent provenance");
        assert!(back.dataset.is_empty());
    }

    #[test]
    fn missing_entry_yields_uniform_not_zero() {
        // Zero importance would tell the quantizer that nothing in the tensor matters. All-ones is the
        // correct "no opinion", and reduces the weighted quantizer to the unweighted one.
        let m = Imatrix::new();
        assert!(m.get("absent").is_none());
        assert_eq!(m.get_or_uniform("absent", 4), vec![1.0; 4]);
    }

    #[test]
    fn importance_moves_error_onto_the_channels_that_do_not_matter() {
        // The property that makes an imatrix worth collecting: with a skewed importance profile, the
        // weighted quantizer must beat the unweighted one on IMPORTANCE-WEIGHTED error, even though it
        // will generally be worse on plain unweighted error — that trade is the entire point.
        let (rows, cols, group) = (48usize, 128usize, 64usize);
        let mut seed = 0xC0FFEEu64;
        let w: Vec<f32> = (0..rows * cols).map(|_| lcg(&mut seed) * 0.05).collect();
        // A realistic profile: a few high-energy channels, a long quiet tail.
        let imp: Vec<f32> = (0..cols).map(|c| if c % 16 == 0 { 40.0 } else { 0.05 }).collect();

        let plain = quantize_ternary_weighted(&w, cols, group, None);
        let weighted = quantize_ternary_weighted(&w, cols, group, Some(&imp));

        let werr = |q: &[f32]| -> f64 {
            w.iter().zip(q).enumerate()
                .map(|(i, (a, b))| imp[i % cols] as f64 * ((a - b) as f64).powi(2))
                .sum()
        };
        let (ep, ew) = (werr(&plain), werr(&weighted));
        assert!(ew < ep, "importance-weighted error {ew:.6e} did not beat unweighted {ep:.6e}");
        // And it should be a real margin, not a rounding difference.
        assert!(ew < ep * 0.999, "improvement {:.4}% is within noise", 100.0 * (1.0 - ew / ep));
    }

    #[test]
    fn uniform_importance_is_identical_to_no_importance() {
        // An all-ones vector must be exactly the unweighted case — otherwise `get_or_uniform`'s fallback
        // would silently perturb every tensor the corpus happened not to cover.
        let (rows, cols, group) = (16usize, 64usize, 32usize);
        let mut seed = 42u64;
        let w: Vec<f32> = (0..rows * cols).map(|_| lcg(&mut seed) * 0.1).collect();
        let a = quantize_ternary_weighted(&w, cols, group, None);
        let b = quantize_ternary_weighted(&w, cols, group, Some(&vec![1.0; cols]));
        assert_eq!(a, b, "uniform importance changed the result");
    }

    #[test]
    fn reconstruction_is_ternary_within_each_group() {
        // Structural check: at most one magnitude per group (plus zero), and signs must follow the input.
        let (rows, cols, group) = (4usize, 32usize, 16usize);
        let mut seed = 7u64;
        let w: Vec<f32> = (0..rows * cols).map(|_| lcg(&mut seed)).collect();
        let q = quantize_ternary_weighted(&w, cols, group, None);
        for r in 0..rows {
            for g0 in (0..cols).step_by(group) {
                let mut mags: Vec<f32> = (g0..g0 + group)
                    .map(|c| q[r * cols + c].abs())
                    .filter(|m| *m > 0.0)
                    .collect();
                mags.sort_by(f32::total_cmp);
                mags.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
                assert!(mags.len() <= 1, "group has {} distinct magnitudes: {mags:?}", mags.len());
                for c in g0..g0 + group {
                    let (a, b) = (w[r * cols + c], q[r * cols + c]);
                    if b != 0.0 { assert_eq!(a.signum(), b.signum(), "sign flipped at ({r},{c})"); }
                }
            }
        }
    }
}
