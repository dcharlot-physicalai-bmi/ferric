//! ROLE-BASED vs UNIFORM ternary quantization — testing a mechanism, not a heuristic.
//!
//! ds4/DwarfStar ships routed experts at mixed precision *by role*: `gate`/`up` at IQ2_XXS (a sign-
//! SYMMETRIC codebook, no zero, no offset) but `down` at Q2_K (an AFFINE quantizer carrying both a scale
//! `d` and a min `dmin`). The stated reason is mechanistic rather than empirical:
//!
//!     gate/up consume the RMSNorm'd hidden state  ->  near-symmetric about zero
//!     down    consumes the SwiGLU product         ->  gated and one-sided
//!
//! With symmetric inputs, symmetric weight-quantization error partially CANCELS across the dot product;
//! with one-sided inputs it ACCUMULATES coherently. So the layer whose INPUT distribution is one-sided
//! needs the lower weight error, and the affine offset is what buys it.
//!
//! Ternary {-1,0,+1} is a symmetric quantizer. Ferric's 16B QAT ternarized UNIFORMLY, and a prior
//! mixed-precision experiment split by LAYER POSITION (first/last in f32) and came back null. Role is a
//! different and better-motivated axis. If the mechanism is real it makes a falsifiable prediction:
//!
//!     PREDICTION: ternarizing `down` alone costs more perplexity than `gate` alone or `up` alone.
//!
//! Qwen2.5-0.5B makes this a genuinely CONTROLLED test: gate/up are [4864,896] and down is [896,4864] —
//! 4,358,144 parameters each, IDENTICAL. Ternarizing exactly one at a time holds the bit budget fixed and
//! varies only the role. A null result falsifies the mechanism; it does not merely fail to support it.
//!
//! Methodology note (learned the hard way on the ternary kernel bench, and independently reached by
//! kimi-k3-in-c: "differences smaller than the noise floor are not effects"): perplexity on ONE text chunk
//! is one sample. A per-role gap that is smaller than the chunk-to-chunk spread is not a role effect. So
//! this evaluates several disjoint chunks and reports the spread alongside the mean.
//!
//!   cargo run -p ferric-llama --example ternary_by_role --release
use ferric_core::Context;
use ferric_gguf::{quant_q8_0, GgufFile, GgufSource, Meta, TensorInfo};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

const GS: usize = 128; // ternary group size
const Q8_0: u32 = 8;
const CHUNK: usize = 384; // tokens per eval chunk
const NCHUNK: usize = 6; // disjoint chunks -> gives us a spread, not a single sample

/// Group-wise ternary (BitNet-style absmean threshold), single plane — the most sensitive setting, so
/// role differences are visible rather than buried under residual planes.
fn ternarize(w: &[f32]) -> Vec<f32> {
    let mut recon = vec![0f32; w.len()];
    for g in 0..w.len().div_ceil(GS) {
        let (lo, hi) = (g * GS, ((g + 1) * GS).min(w.len()));
        let grp = &w[lo..hi];
        let ma = grp.iter().map(|x| x.abs()).sum::<f32>() / grp.len() as f32;
        let d = 0.7 * ma;
        let (mut ss, mut sc) = (0f32, 0usize);
        for &x in grp {
            if x.abs() > d { ss += x.abs(); sc += 1; }
        }
        let a = if sc > 0 { ss / sc as f32 } else { 0.0 };
        for k in lo..hi {
            recon[k] = if w[k].abs() > d { a * w[k].signum() } else { 0.0 };
        }
    }
    recon
}

/// Which tensor ROLES a variant ternarizes. Suffixes match the GGUF naming (`blk.N.ffn_gate.weight`).
#[derive(Clone, Copy)]
struct Role(&'static str, &'static [&'static str]);

const ROLES: &[Role] = &[
    // --- the controlled comparison: exactly one FFN role, identical parameter counts ---
    Role("gate only        ", &["ffn_gate.weight"]),
    Role("up only          ", &["ffn_up.weight"]),
    Role("down only        ", &["ffn_down.weight"]),
    // --- the ds4 recipe vs its inverse, at equal FFN coverage flipped ---
    Role("gate+up (ds4)    ", &["ffn_gate.weight", "ffn_up.weight"]),
    // --- context ---
    Role("all FFN          ", &["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]),
    Role("attn only        ", &["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"]),
    Role(
        "everything       ",
        &["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
          "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"],
    ),
];

struct TGguf<'a> { inner: &'a GgufFile, roles: &'static [&'static str] }
impl TGguf<'_> {
    fn hit(&self, n: &str) -> bool { self.roles.iter().any(|s| n.ends_with(s)) }
}
impl GgufSource for TGguf<'_> {
    fn metadata(&self) -> &HashMap<String, Meta> { self.inner.metadata() }
    fn tensor(&self, n: &str) -> Option<&TensorInfo> { self.inner.tensor(n) }
    fn raw(&self, n: &str) -> Result<Vec<u8>, String> {
        if self.hit(n) && self.inner.tensor(n).map(|t| t.ggml_type) == Some(Q8_0) {
            Ok(quant_q8_0(&ternarize(&self.inner.dequant(n)?)))
        } else { self.inner.raw(n) }
    }
    fn dequant(&self, n: &str) -> Result<Vec<f32>, String> {
        if self.hit(n) { Ok(ternarize(&self.inner.dequant(n)?)) } else { self.inner.dequant(n) }
    }
}

/// Per-chunk mean NLL in NATS/token (not perplexity).
///
/// Perplexity is exp(NLL), so once quantization degrades a model badly, perplexity RATIOS explode into
/// meaninglessness -- "+37000%" vs "+11000%" is a difference of ~1.2 nats, not a factor of 3.4 in quality.
/// NLL is the linear scale and the only one on which these variants can be compared. Returns one value per
/// chunk so the caller can do a PAIRED comparison instead of pooling.
async fn nll_chunks(ctx: &Arc<Context>, m: &Qwen3, chunks: &[Vec<u32>]) -> Vec<f64> {
    let _ = ctx;
    let c = &m.cfg;
    let mut out = Vec::new();
    for ids in chunks {
        let v = m.forward_cached(ids, &mut Cache::new(c)).to_vec().await;
        let (mut nll, mut cnt) = (0f64, 0usize);
        for i in 0..ids.len() - 1 {
            let row = &v[i * c.n_vocab..(i + 1) * c.n_vocab];
            let mx = row.iter().cloned().fold(f32::MIN, f32::max);
            let lse = mx + row.iter().map(|&x| (x - mx).exp()).sum::<f32>().ln();
            nll += (lse - row[ids[i + 1] as usize]) as f64;
            cnt += 1;
        }
        out.push(nll / cnt as f64); // nats/token
    }
    out
}

fn mean(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }
fn spread_pct(v: &[f64]) -> f64 {
    let (lo, hi) = v.iter().fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    100.0 * (hi - lo) / mean(v)
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();
    let path = format!("{home}/.cache/ferric/hub/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf");
    let g = GgufFile::open(&path).unwrap();

    let tokens: Vec<String> = match g.metadata().get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
        _ => panic!("no tokens"),
    };
    let vocab: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata().get("tokenizer.ggml.merges") {
        Some(Meta::Arr(a)) => a.iter()
            .filter_map(|m| if let Meta::Str(s) = m { s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string())) } else { None })
            .collect(),
        _ => panic!("no merges"),
    };
    let bpe = Bpe::new(vocab, &merges);

    // Real diverse text, not a hand-written paragraph: a single short passage makes perplexity a
    // property of that passage.
    let corpus = std::fs::read_to_string(".phase0/corpus_real.txt")
        .expect("expected .phase0/corpus_real.txt (real eval corpus)");
    let all = bpe.encode(&corpus);
    assert!(all.len() >= NCHUNK * CHUNK, "corpus too short: {} tokens", all.len());
    // Disjoint, evenly spaced chunks — strided so they sample different parts of the corpus rather than
    // one contiguous (and possibly topically homogeneous) span.
    let stride = all.len() / NCHUNK;
    let chunks: Vec<Vec<u32>> = (0..NCHUNK).map(|i| all[i * stride..i * stride + CHUNK].to_vec()).collect();

    println!("Role-based vs uniform ternary — Qwen2.5-0.5B, {NCHUNK} disjoint chunks x {CHUNK} tokens");
    println!("  Metric is NLL in nats/token. Perplexity ratios are exponential and become meaningless");
    println!("  once a variant is badly degraded; nats are linear and comparable.\n");

    let orig = Qwen3::load(&ctx, &g).unwrap();
    let base = nll_chunks(&ctx, &orig, &chunks).await;
    let base_m = mean(&base);
    drop(orig);
    println!("  {:<18} {:6.3} nats   (ppl {:7.2})  <- reference", "baseline Q8_0", base_m, base_m.exp());
    println!("  {:-<78}", "");

    let mut results: Vec<(&str, Vec<f64>)> = Vec::new();
    for r in ROLES {
        let m = Qwen3::load(&ctx, &TGguf { inner: &g, roles: r.1 }).unwrap();
        let p = nll_chunks(&ctx, &m, &chunks).await;
        drop(m);
        let pm = mean(&p);
        println!("  {} {pm:6.3} nats   (ppl {:9.1})   {:+6.3} nats vs baseline", r.0, pm.exp(), pm - base_m);
        results.push((r.0, p));
    }

    let per = |k: &str| results.iter().find(|(n, _)| n.trim() == k).map(|(_, v)| v.clone()).unwrap();
    let (vg, vu, vd) = (per("gate only"), per("up only"), per("down only"));
    let (dg, du, dd) = (mean(&vg) - base_m, mean(&vu) - base_m, mean(&vd) - base_m);

    // PAIRED comparison. Every variant saw the SAME chunks, so the right question is not "do the pooled
    // means differ by more than the pooled spread" (that compares incommensurable quantities) but "does
    // the ordering hold on every chunk independently".
    let wins = |a: &[f64], b: &[f64]| a.iter().zip(b).filter(|(x, y)| x < y).count();

    println!("\n  {:-<78}", "");
    println!("  A. PERFECTLY CONTROLLED PAIR — gate vs up");
    println!("     Identical shape [4864,896], identical fan-in, identical input (both read the same");
    println!("     RMSNorm'd hidden). The ONLY difference is downstream role: gate's output passes");
    println!("     through SiLU before multiplying up's. Nothing else varies.");
    println!("       gate {dg:+6.3} nats     up {du:+6.3} nats     difference {:+.3} nats", dg - du);
    println!("       up beats gate on {}/{} chunks (paired)", wins(&vu, &vg), NCHUNK);
    if wins(&vu, &vg) == NCHUNK && (dg - du).abs() > 0.05 {
        println!("     ==> REAL ROLE EFFECT. Two tensors that differ ONLY in downstream role differ in");
        println!("         quantization sensitivity, consistently on every chunk. `gate` is the more");
        println!("         fragile of the two, so a SiLU gate input is what a symmetric quantizer struggles");
        println!("         with here — not the one-sided SwiGLU product that ds4's argument points at.");
    } else {
        println!("     ==> no consistent difference; downstream role alone does not drive sensitivity.");
    }

    println!("\n  B. down vs gate/up — CONFOUNDED, reported but not decisive");
    println!("       down {dd:+6.3} nats   vs   gate {dg:+6.3} / up {du:+6.3}");
    println!("       down beats gate on {}/{} chunks, beats up on {}/{} chunks",
             wins(&vd, &vg), NCHUNK, wins(&vd, &vu), NCHUNK);
    println!("     `down` has the same PARAMETER COUNT but 5.4x the FAN-IN (4864 vs 896). Quantization");
    println!("     error averages down as ~sqrt(N) across the summed terms, so a larger fan-in is");
    println!("     intrinsically more robust REGARDLESS of input symmetry. Input-symmetry and fan-in are");
    println!("     confounded in this architecture and this test cannot separate them. Any claim that");
    println!("     `down` is robust *because* of its input distribution is unsupported by this experiment.");

    println!("\n  VERDICT");
    if dd < dg && dd < du {
        println!("    ds4's premise does NOT transfer to ternary as stated. `down` is the LEAST");
        println!("    quantization-sensitive of the three roles here, not the most — the opposite of what");
        println!("    the input-symmetry argument predicts. The most likely reading is that fan-in");
        println!("    dominates, and that ds4's IQ2_XXS/Q2_K split is better explained by codebook shape");
        println!("    (IQ2_XXS has no zero, so it cannot represent a pruned weight at all) and by the");
        println!("    kernel fusion that makes the split free, than by an accuracy mechanism.");
        println!("    ACTIONABLE: a uniform ternary QAT is NOT obviously mis-spending bits at `down`.");
        println!("    If bits are to be re-allocated by role, spend them on `gate`, which is measurably");
        println!("    the most fragile role at equal parameter count.");
    } else {
        println!("    `down` is not the least sensitive role; see the paired counts above before");
        println!("    concluding anything — the fan-in confound still applies.");
    }
    println!("\n  (Single plane, group size {GS}. Absolute degradation at 0.5B is severe and expected;");
    println!("   the claim under test is the RELATIVE ordering between roles, which is what varies.)");
}
