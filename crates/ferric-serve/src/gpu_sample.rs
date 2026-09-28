//! **The host half of device token selection** (`ferric_tensor::sample`): turn a request's `Sampling`
//! into the device's row request, and turn what the device returns into the token the REFERENCE sampler
//! (`genopts::sample`) picks — by running that sampler itself on the reduced row.
//!
//! The device returns, per row, the greedy argmax (exact), a reduced row, or `Fallback`. For a reduced
//! row this file (1) applies the penalties with `genopts::sample`'s own arithmetic, (2) checks, with the
//! host's own `exp`, that every token the sorted prefix can reach is present — the device's filter is
//! conservative by a stated bound, and this is where that bound is not trusted, only verified — and
//! (3) calls `genopts::sample` on the reduced row, which consumes the RNG exactly as the full row would.
//! Anything it cannot verify returns `None`, and the caller samples the full row as before.
//!
//! `FERRIC_GPU_SAMPLE=0` turns device selection off (the A/B arm and the escape hatch).
use crate::genopts::{self, Sampling};
use ferric_tensor::sample::{Patch, RowOut, RowReq};
use ferric_tensor::Tensor;

/// Device selection on unless `FERRIC_GPU_SAMPLE=0` — and not while the NVIDIA native tier is live
/// (`FERRIC_CUDA`): there the dense model's logits already land in HOST memory
/// (`Qwen3::forward_cached_last_host`), and selecting on the wgpu device would first upload the row the
/// host-logits path exists to not move. That tier samples on the host until selection runs in CUDA.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FERRIC_GPU_SAMPLE").map(|v| v != "0").unwrap_or(true) && !native_host_logits())
}

#[cfg(all(any(target_os = "linux", target_os = "windows"), not(target_arch = "wasm32")))]
fn native_host_logits() -> bool { ferric_tensor::cuda::driver().is_some() }
#[cfg(not(all(any(target_os = "linux", target_os = "windows"), not(target_arch = "wasm32"))))]
fn native_host_logits() -> bool { false }

/// Every per-token logit adjustment `genopts::sample` makes before it samples, in its order:
/// `logit_bias` (each entry, in list order), presence/frequency, repeat, DRY. Returned per token with
/// the exact operands, so the host can replay them bit for bit on a reduced row; `patch` is the device's
/// approximate copy for its filter.
#[derive(Default)]
struct Adjust { bias: Vec<f32>, sub: Option<f32>, rep: Option<f32>, dry: Option<f32> }

fn adjustments(s: &Sampling, prompt: &[u32], generated: &[u32], n_vocab: usize) -> std::collections::BTreeMap<u32, Adjust> {
    let mut m: std::collections::BTreeMap<u32, Adjust> = std::collections::BTreeMap::new();
    for &(t, b) in &s.logit_bias { if (t as usize) < n_vocab { m.entry(t).or_default().bias.push(b); } }
    if s.presence_penalty != 0.0 || s.frequency_penalty != 0.0 {
        let mut counts = std::collections::HashMap::<u32, u32>::new();
        for &t in generated { *counts.entry(t).or_default() += 1; }
        for (&t, &c) in &counts {
            if (t as usize) < n_vocab { m.entry(t).or_default().sub = Some(s.presence_penalty + s.frequency_penalty * c as f32); }
        }
    }
    if s.repeat_penalty != 1.0 && s.repeat_last_n > 0 {
        let tail = &prompt[prompt.len().saturating_sub(s.repeat_last_n)..];
        for &t in tail.iter().chain(generated.iter()).rev().take(s.repeat_last_n) {
            if (t as usize) < n_vocab { m.entry(t).or_default().rep = Some(s.repeat_penalty); }
        }
    }
    if s.dry_multiplier > 0.0 {
        // DRY's penalty depends on the context, not on the row: run the reference on a zero row and read
        // it back (`0 - pen` is `-pen` exactly), so the replay below subtracts the very same operand.
        let mut z = vec![0f32; n_vocab];
        let ctx: Vec<u32> = prompt.iter().chain(generated.iter()).copied().collect();
        genopts::apply_dry(&mut z, &ctx, s.dry_multiplier, s.dry_base, s.dry_allowed_length, &s.dry_breakers, s.dry_range);
        for (t, &x) in z.iter().enumerate() { if x != 0.0 { m.entry(t as u32).or_default().dry = Some(-x); } }
    }
    m
}

/// The device request for row `row`, or `None` when these settings are not served on the device:
/// top-nσ takes the standard deviation of the WHOLE row, which a reduced row does not carry.
pub(crate) fn request(row: usize, s: &Sampling, prompt: &[u32], generated: &[u32], n_vocab: usize) -> Option<RowReq> {
    if !(s.temperature.is_finite() && s.top_p.is_finite() && s.min_p.is_finite()) || s.top_n_sigma > 0.0 { return None; }
    let patches = adjustments(s, prompt, generated, n_vocab).into_iter().map(|(t, a)| Patch {
        token: t,
        add: (!a.bias.is_empty()).then(|| a.bias.iter().sum()),
        sub: a.sub, rep: a.rep, sub2: a.dry,
    }).collect();
    Some(RowReq { row, temperature: s.temperature, top_p: s.top_p, top_k: s.top_k, min_p: s.min_p, patches })
}

/// The token `genopts::sample` picks from the full row, from what the device returned — or `None` if
/// the device's answer cannot be verified, in which case the caller samples the full row. On `Some`,
/// `rng` (and Mirostat's running `mu`) have advanced exactly as `genopts::sample` on the full row
/// advances them.
pub(crate) fn finish(out: &RowOut, s: &Sampling, prompt: &[u32], generated: &[u32], rng: &mut u64) -> Option<u32> {
    let (ids, raw, tau_hi) = match out {
        RowOut::Id(t) => return Some(*t),
        RowOut::Fallback => return None,
        RowOut::Reduced { ids, raw, tau_hi, .. } => (ids, raw, *tau_hi),
    };
    if s.top_n_sigma > 0.0 { return None; }
    // (1) The adjustments, with genopts::sample's arithmetic and order, on the tokens present.
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_by_key(|&k| ids[k]);
    let (ids, mut row): (Vec<u32>, Vec<f32>) = order.iter().map(|&k| (ids[k], raw[k])).unzip();
    if ids.is_empty() || ids.windows(2).any(|w| w[0] == w[1]) { return None; }
    // Only tokens present are adjusted, so the adjustment table need not reach past the largest id here.
    let adj = adjustments(s, prompt, generated, ids[ids.len() - 1] as usize + 1);
    for (&t, a) in &adj {
        let Ok(k) = ids.binary_search(&t) else { continue };
        let x = &mut row[k];
        for &b in &a.bias { *x += b; }
        if let Some(sub) = a.sub { *x -= sub; }
        if let Some(rep) = a.rep { *x = if *x > 0.0 { *x / rep } else { *x * rep }; }
        if let Some(d) = a.dry { *x -= d; }
    }
    if row.iter().any(|x| x.is_nan()) { return None; }
    // (2) Coverage, with the host's own exp: everything the sorted prefix reaches must be here. Every
    // absent token's host probability is below tau_hi.
    if s.temperature > 0.0 {
        let maxl = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if !maxl.is_finite() { return None; }
        let probs: Vec<f32> = row.iter().map(|&l| ((l - maxl) / s.temperature).exp()).collect();
        let above = probs.iter().filter(|&&p| p > tau_hi).count();
        if s.top_k > 0 {
            if above < s.top_k { return None; }
        } else if s.min_p > 0.0 {
            if !(tau_hi < s.min_p) { return None; }
        } else {
            let sum: f32 = probs.iter().sum();
            let mut idx: Vec<usize> = (0..probs.len()).filter(|&i| probs[i] > tau_hi).collect();
            idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
            let mut cum = 0.0f32;
            if !idx.iter().any(|&i| { cum += probs[i] / sum; cum >= s.top_p }) { return None; }
        }
    }
    // (3) The reference, on the reduced row. The adjustments are already in it; XTC's special tokens
    // are named by position in THIS row (the reference compares candidate indices against them).
    let xtc_specials = s.xtc_specials.iter().filter_map(|t| ids.binary_search(t).ok().map(|k| k as u32)).collect();
    let s2 = Sampling { presence_penalty: 0.0, frequency_penalty: 0.0, repeat_penalty: 1.0, logit_bias: Vec::new(),
                        dry_multiplier: 0.0, xtc_specials, ..s.clone() };
    let pos = genopts::sample(&row, &s2, &[], &[], rng) as usize;
    s.mirostat_mu.set(s2.mirostat_mu.get());
    Some(ids[pos])
}

/// One forward's logits, left on the device, with every row's device answer: the whole integration a
/// generation loop needs. `rows[i]` = (the request's sampling, its prompt, what it had generated when
/// this row is sampled). A row whose answer the host cannot verify is read back and handed to the
/// loop's own host sampler, so a loop's output is the host path's output in every case.
pub(crate) struct Picker { lg: Tensor, outs: Vec<Option<RowOut>> }

impl Picker {
    pub(crate) fn new(lg: Tensor, rows: &[(&Sampling, &[u32], &[u32])]) -> Picker {
        let n_vocab = *lg.shape.last().expect("logits");
        let reqs: Vec<Option<RowReq>> = rows.iter().enumerate().map(|(r, (s, p, g))| request(r, s, p, g, n_vocab)).collect();
        let dev: Vec<RowReq> = reqs.iter().flatten().cloned().collect();
        let mut got = ferric_tensor::sample::select_rows(&lg, &dev).into_iter();
        let outs = reqs.iter().map(|r| r.as_ref().and_then(|_| got.next())).collect();
        Picker { lg, outs }
    }
    /// The token for row `row`. `host` is the loop's own sampler, called with the full row only when
    /// the device's answer does not verify.
    pub(crate) fn pick(&self, row: usize, s: &Sampling, prompt: &[u32], generated: &[u32], rng: &mut u64,
                       host: impl FnOnce(&[f32], &mut u64) -> Option<u32>) -> Option<u32> {
        if let Some(t) = self.outs[row].as_ref().and_then(|o| finish(o, s, prompt, generated, rng)) { return Some(t); }
        let r = pollster::block_on(self.lg.narrow(0, row, 1).to_vec());
        host(&r, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genopts::{sample, Sampling, DEFAULT_RNG};
    use std::sync::Arc;

    fn ctx() -> Option<Arc<ferric_core::Context>> { pollster::block_on(ferric_core::Context::new()).ok().map(Arc::new) }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
        fn unit(&mut self) -> f32 { (self.next() >> 40) as f32 / (1u64 << 24) as f32 }
        fn gauss(&mut self) -> f32 { let (a, b) = (self.unit().max(1e-7), self.unit()); (-2.0 * a.ln()).sqrt() * (6.2831855 * b).cos() }
    }

    /// Logit rows shaped like Qwen2.5-0.5B's decode rows (measured: max 16-24, the 10th token 1-4 below
    /// it, the 100th ~7 below, the 1000th ~11 below, a bulk around -3.5 with sd ~3): a Gaussian bulk and
    /// a 1000-token head falling off as rank^0.3, with jitter, ties from rounding to 1/64, and `gap`
    /// scaling how far the head stands above the bulk.
    fn lm_row(rng: &mut Rng, v: usize, gap: f32) -> Vec<f32> {
        let mut row: Vec<f32> = (0..v).map(|_| -3.5 + 3.0 * rng.gauss()).collect();
        let top = 14.0 + 10.0 * rng.unit();
        for k in 0..1000 {
            let i = (rng.next() as usize) % v;
            let x = top - gap * (k as f32 / 1000.0).powf(0.3) + 0.3 * rng.gauss();
            row[i] = (x * 64.0).round() / 64.0;
        }
        row
    }

    /// `hot`: a token the rows make likely, so DRY's penalty and XTC's special-token rule change picks.
    fn settings(hot: u32) -> Vec<Sampling> {
        let d = Sampling::default();
        let mut v = vec![
            Sampling { temperature: 0.0, ..d.clone() },                                   // greedy
            Sampling { temperature: 0.8, ..d.clone() },                                   // the pinned default path
            Sampling { temperature: 1.0, ..d.clone() },
            Sampling { temperature: 0.3, top_p: 0.5, ..d.clone() },
            Sampling { temperature: 1.0, top_k: 40, ..d.clone() },
            Sampling { temperature: 1.2, top_k: 1, ..d.clone() },
            Sampling { temperature: 0.9, min_p: 0.05, ..d.clone() },
            Sampling { temperature: 0.7, top_k: 20, min_p: 0.1, top_p: 0.9, ..d.clone() },
            Sampling { temperature: 0.8, presence_penalty: 0.5, frequency_penalty: 0.3, ..d.clone() },
            Sampling { temperature: 0.8, repeat_penalty: 1.3, ..d.clone() },
            Sampling { temperature: 0.0, repeat_penalty: 1.3, presence_penalty: 0.2, ..d.clone() },
            Sampling { temperature: 1.5, ..d.clone() },                                   // flat: exercises Fallback
            Sampling { temperature: 0.8, top_p: 1.0, ..d.clone() },                       // needs the whole row
        ];
        v.push(Sampling { temperature: 0.8, repeat_penalty: 0.8, repeat_last_n: 8, ..d.clone() });
        // main's extended samplers (S16): each changes what the device must carry or the host must replay
        v.push(Sampling { temperature: 0.8, logit_bias: vec![(1, 5.0), (2, f32::NEG_INFINITY), (1, -0.25), (70_000, 3.5)], ..d.clone() });
        v.push(Sampling { temperature: 0.0, logit_bias: vec![(3, 40.0), (4, f32::NEG_INFINITY)], ..d.clone() });
        v.push(Sampling { temperature: 0.9, dry_multiplier: 0.8, dry_allowed_length: 1, ..d.clone() });
        v.push(Sampling { temperature: 0.9, dry_multiplier: 6.0, dry_allowed_length: 2, ..d.clone() });
        v.push(Sampling { temperature: 0.0, dry_multiplier: 6.0, dry_allowed_length: 2, ..d.clone() });
        v.push(Sampling { temperature: 1.0, xtc_probability: 1.0, xtc_threshold: 0.02, xtc_specials: vec![hot], ..d.clone() });
        v.push(Sampling { temperature: 1.0, typical_p: 0.7, ..d.clone() });
        v.push(Sampling { temperature: 1.0, mirostat: 2, mirostat_tau: 3.0, ..d.clone() });
        v.push(Sampling { temperature: 1.0, xtc_probability: 0.9, xtc_threshold: 0.05, xtc_specials: vec![5, 6], ..d.clone() });
        v.push(Sampling { temperature: 1.0, top_n_sigma: 1.5, ..d });
        v
    }

    /// ⭐ THE CONTRACT: for the same logits and the same RNG state, device selection + host finish picks
    /// the token `genopts::sample` picks from the full row, and leaves the RNG where it leaves it — over
    /// every setting above (greedy, the pinned default T=0.8/top_p .95 path, top-k, min-p, penalties,
    /// flat distributions that must fall back), many seeds, many rows, several rows per device call.
    /// A `None` (fallback) is followed exactly as the serve loops follow it: the full row on the host.
    #[test]
    fn device_selection_picks_the_reference_token_and_advances_the_rng_identically() {
        let Some(ctx) = ctx() else { return };
        let v = 151_936;
        let mut rng = Rng(0xfeed_5eed);
        let (mut same, mut fell_back, mut device, mut sums, mut declined) = (0usize, 0usize, 0usize, 0usize, 0usize);
        for round in 0..6 {
            let mut rows: Vec<Vec<f32>> = (0..4).map(|k| lm_row(&mut rng, v, [11.0, 9.0, 13.0, 7.0][k])).collect();
            let mut prompt: Vec<u32> = (0..50).map(|_| (rng.next() as u32) % v as u32).collect();
            let mut generated: Vec<u32> = (0..12).map(|i| if i % 3 == 0 { prompt[i] } else { (rng.next() as u32) % v as u32 }).collect();
            // A repeat DRY must see: the prompt holds [x, y, hot] and the generation ends [x, y], so DRY
            // penalises `hot` — which every row makes the argmax, so the penalty moves picks. XTC's special
            // token is `hot` too, so its keep-the-special rule decides.
            let hot = (rng.next() as u32) % v as u32;
            let (x, y) = ((rng.next() as u32) % v as u32, (rng.next() as u32) % v as u32);
            prompt[20..23].copy_from_slice(&[x, y, hot]);
            generated[10] = x; generated[11] = y;
            for row in rows.iter_mut() { let m = row.iter().cloned().fold(f32::MIN, f32::max); row[hot as usize] = m + 0.25; }
            let flat: Vec<f32> = rows.concat();
            let t = ferric_tensor::Tensor::from_vec(&ctx, &flat, &[rows.len(), v]);
            for s in settings(hot) {
                let Some(reqs) = (0..rows.len()).map(|r| request(r, &s, &prompt, &generated, v)).collect::<Option<Vec<RowReq>>>() else {
                    declined += 1; continue };
                let outs = ferric_tensor::sample::select_rows(&t, &reqs);
                if std::env::var("GS_DEBUG").is_ok() { eprintln!("{s:?} -> {:?}", outs.iter().map(|o| match o { RowOut::Reduced { ids, .. } => format!("R{}", ids.len()), o => format!("{o:?}") }).collect::<Vec<_>>()); }
                // The invariant the reduced row rests on, checked directly: over a plain-nucleus row the
                // host's sequential Σp over the reduced row is the full row's Σp, to the bit. (Token
                // equality alone would miss most wrong sums: a sum off by an ulp rarely moves a pick.)
                if s.temperature > 0.0 && s.top_k == 0 && s.min_p == 0.0 && s.presence_penalty == 0.0 && s.frequency_penalty == 0.0 && s.repeat_penalty == 1.0
                    && s.logit_bias.is_empty() && s.dry_multiplier == 0.0 {
                    for (r, out) in outs.iter().enumerate() {
                        let RowOut::Reduced { ids, .. } = out else { continue };
                        let row = &rows[r];
                        let maxl = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let full: f32 = row.iter().map(|&l| ((l - maxl) / s.temperature).exp()).sum();
                        let part: f32 = ids.iter().map(|&i| ((row[i as usize] - maxl) / s.temperature).exp()).sum();
                        assert_eq!(part.to_bits(), full.to_bits(), "row {r} {s:?}: reduced-row sum {part:e} vs full {full:e} ({} of {v} kept)", ids.len());
                        sums += 1;
                    }
                }
                for (r, out) in outs.iter().enumerate() {
                    for seed in [DEFAULT_RNG, 1, 0x9E37_79B9_7F4A_7C15 ^ round as u64] {
                        let (mut a, mut b) = (seed, seed);
                        // Two copies of the settings: Mirostat's running mu must evolve identically on both paths.
                        let (sa, sb) = (s.clone(), s.clone());
                        let want = sample(&rows[r], &sa, &prompt, &generated, &mut a);
                        let got = match finish(out, &sb, &prompt, &generated, &mut b) {
                            Some(tok) => { device += 1; tok }
                            None => { fell_back += 1; sample(&rows[r], &sb, &prompt, &generated, &mut b) }
                        };
                        assert_eq!(sa.mirostat_mu.get().to_bits(), sb.mirostat_mu.get().to_bits(), "{s:?}: Mirostat's mu diverged");
                        assert_eq!(got, want, "row {r} {s:?} seed {seed:#x}: device {got} vs reference {want} (out {:?})",
                                   match out { RowOut::Reduced { ids, .. } => format!("reduced {}", ids.len()), o => format!("{o:?}") });
                        assert_eq!(a, b, "row {r} {s:?}: the RNG must advance exactly as the reference advances it");
                        same += 1;
                    }
                }
            }
        }
        eprintln!("device selection: {same} picks identical to genopts::sample ({device} finished from the device's answer, {fell_back} fell back); {sums} reduced-row sums equal to the full row's; {declined} (setting, batch) pairs declined (top-nσ)");
        assert!(declined > 0, "the top-nσ setting must be declined, not served from a reduced row");
        assert!(device >= 2 * fell_back, "the device path must carry most picks, or this compares the reference to itself");
    }

    /// Rows built to break a careless reduction: ties at the maximum (the LAST index wins), -0.0 against
    /// +0.0, -inf entries, a NaN (must fall back), a one-hot row, an all-equal row.
    #[test]
    fn device_selection_handles_ties_signed_zero_inf_and_nan_like_the_reference() {
        let Some(ctx) = ctx() else { return };
        let v = 5000;
        let mut rows: Vec<Vec<f32>> = Vec::new();
        let mut r0 = vec![-5.0f32; v]; r0[10] = 3.0; r0[4000] = 3.0; r0[2500] = 3.0; rows.push(r0);        // tie: 4000
        let mut r1 = vec![-1.0f32; v]; r1[7] = -0.0; r1[3000] = 0.0; r1[20] = 0.0; rows.push(r1);        // ±0 tie: 3000
        let mut r2 = vec![f32::NEG_INFINITY; v]; r2[123] = 1.0; r2[124] = 0.5; rows.push(r2);
        let mut r3 = vec![0.25f32; v]; r3[77] = f32::NAN; rows.push(r3);
        rows.push(vec![1.0f32; v]);
        let mut r5 = vec![f32::NEG_INFINITY; v]; r5[4999] = 7.0; rows.push(r5);
        let flat: Vec<f32> = rows.concat();
        let t = ferric_tensor::Tensor::from_vec(&ctx, &flat, &[rows.len(), v]);
        for s in settings(4000) {
            let Some(reqs) = (0..rows.len()).map(|r| request(r, &s, &[1, 2, 3, 4000], &[20, 3000], v)).collect::<Option<Vec<RowReq>>>() else { continue };
            let outs = ferric_tensor::sample::select_rows(&t, &reqs);
            for (r, out) in outs.iter().enumerate() {
                let (mut a, mut b) = (DEFAULT_RNG, DEFAULT_RNG);
                let (sa, sb) = (s.clone(), s.clone());
                let want = sample(&rows[r], &sa, &[1, 2, 3, 4000], &[20, 3000], &mut a);
                let got = finish(out, &sb, &[1, 2, 3, 4000], &[20, 3000], &mut b).unwrap_or_else(|| sample(&rows[r], &sb, &[1, 2, 3, 4000], &[20, 3000], &mut b));
                assert_eq!((got, b), (want, a), "row {r} {s:?}: {out:?}");
                if r == 3 { assert_eq!(out, &RowOut::Fallback, "a NaN row must come back as Fallback"); }
            }
        }
    }

    /// The same contract on a REAL model's rows (ignored by default: it needs the checkpoint; the gate
    /// `scripts/gpu_sample_conformance.sh` runs it). Qwen2.5-0.5B decodes 3 prompts greedily; at every
    /// step the one-row logits, and a 6-row verify-shaped forward, are device-selected under every
    /// setting and 3 seeds and compared with `genopts::sample` on the read-back rows.
    #[test]
    #[ignore]
    fn real_model_rows_select_identically() {
        let Some(ctx) = ctx() else { return };
        let home = std::env::var("HOME").unwrap_or_default();
        let path = std::env::var("FERRIC_SAMPLE_MODEL").unwrap_or_else(|_| format!("{home}/.cache/ferric/hub/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf"));
        let Ok(g) = ferric_gguf::GgufFile::open(&path) else { eprintln!("no model at {path} — skipped"); return };
        let m = ferric_llama::qwen3::Qwen3::load(&ctx, &g).expect("load");
        let v = m.cfg.n_vocab;
        let (mut same, mut device, mut fell_back) = (0usize, 0usize, 0usize);
        // Real text through the checkpoint's own tokenizer: random ids make a confused, flat model whose
        // rows are not what serving sees.
        let tokens: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") { Some(ferric_gguf::Meta::Arr(a)) => a.iter().map(|m| if let ferric_gguf::Meta::Str(s) = m { s.clone() } else { String::new() }).collect(), _ => panic!("no vocab") };
        let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") { Some(ferric_gguf::Meta::Arr(a)) => a.iter().filter_map(|m| if let ferric_gguf::Meta::Str(s) = m { s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string())) } else { None }).collect(), _ => panic!("no merges") };
        let bpe = ferric_tokenizer::Bpe::new(tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect(), &merges);
        let texts = ["<|im_start|>user\nWrite a short poem about rain on a tin roof.<|im_end|>\n<|im_start|>assistant\n",
                     "<|im_start|>user\nExplain why the sky is blue in two sentences.<|im_end|>\n<|im_start|>assistant\n",
                     "The lighthouse at Point Reyes was built in 1870 and"];
        for (p, text) in texts.iter().enumerate() {
            let prompt: Vec<u32> = bpe.encode(text);
            let mut c = ferric_llama::qwen3::Cache::new(&m.cfg);
            let mut lg = m.forward_cached_last(&prompt, &mut c);
            let mut generated: Vec<u32> = Vec::new();
            for step in 0..24 {
                // every 6th step: a 6-row forward (a verify step's shape), then rewind
                let multi = step % 6 == 5;
                let (t, rows_n) = if multi {
                    let base = c.pos;
                    let feed: Vec<u32> = (0..6).map(|k| generated.last().copied().unwrap_or(1) + k).collect();
                    let t = m.forward_cached(&feed, &mut c);
                    c.truncate(base);
                    (t, 6)
                } else { (lg.clone(), 1) };
                let host = pollster::block_on(t.to_vec());
                for s in settings(generated.last().copied().unwrap_or(0)) {
                    let Some(reqs) = (0..rows_n).map(|k| request(k, &s, &prompt, &generated, v)).collect::<Option<Vec<RowReq>>>() else { continue };
                    let outs = ferric_tensor::sample::select_rows(&t, &reqs);
                    if std::env::var("GS_DEBUG").is_ok() { eprintln!("T={} p={} k={} mp={} -> {:?}", s.temperature, s.top_p, s.top_k, s.min_p, outs.iter().map(|o| match o { RowOut::Reduced { ids, .. } => format!("R{}", ids.len()), o => format!("{o:?}") }).collect::<Vec<_>>()); }
                    for (k, out) in outs.iter().enumerate() {
                        let row = &host[k * v..(k + 1) * v];
                        for seed in [DEFAULT_RNG, 7, 0xABCDEF ^ step as u64] {
                            let (mut a, mut b) = (seed, seed);
                            let (sa, sb) = (s.clone(), s.clone());
                            let want = sample(row, &sa, &prompt, &generated, &mut a);
                            let got = match finish(out, &sb, &prompt, &generated, &mut b) {
                                Some(x) => { device += 1; x }
                                None => { fell_back += 1; sample(row, &sb, &prompt, &generated, &mut b) }
                            };
                            assert_eq!(sa.mirostat_mu.get().to_bits(), sb.mirostat_mu.get().to_bits());
                            assert_eq!((got, b), (want, a), "prompt {p} step {step} row {k} {s:?} seed {seed:#x}");
                            same += 1;
                        }
                    }
                }
                let last = &host[(rows_n - 1) * v..];
                let next = if multi { generated.last().copied().unwrap_or(1) } else {
                    last.iter().enumerate().fold((0usize, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0 as u32 };
                if !multi {
                    generated.push(next);
                    lg = m.forward_cached_last(&[next], &mut c);
                }
            }
        }
        eprintln!("real rows: {same} picks identical to genopts::sample ({device} from the device's answer, {fell_back} fell back)");
        assert!(device >= 2 * fell_back);
    }
}
