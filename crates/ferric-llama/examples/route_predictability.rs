//! **Is MoE routing predictable one layer ahead? Measured, not quoted.**
//!
//! Colibri (26k stars, pure C, 2026) justifies speculative expert prefetch with "routing is
//! measurably 71.6% predictable one layer ahead". That number decides whether an expert prefetcher
//! is worth building, and it is not checkable from the outside — so this checks it on a real local
//! checkpoint, with every content-free baseline that could produce the same number for no reason.
//!
//! ## ⭐ THE RESULT: Colibri prefetches along the wrong axis
//!
//! Measured on Qwen3.6-35B-A3B (256 experts, top-8), 12 MoE layers, 6 prompts, 506 (L → L+1) pairs
//! with popularity fitted on a held-out split:
//!
//! ```text
//!   identity  (L's set -> L+1)               2.7%   <- AT CHANCE
//!   per-layer popularity [held-out]         10.0%
//!   global popularity   [held-out]           6.5%
//!   previous token, same layer              41.0%   <- the real structure
//!   random top-k (the floor)                 3.1%
//!   Colibri's published figure               71.6%
//! ```
//!
//! **Cross-layer prediction is worth nothing here.** Identity sits at the random floor — consecutive
//! layers choose essentially independent expert sets, uniformly across every layer measured (0–5%,
//! so it is not an early-layer artifact). Even *static per-layer popularity*, which requires no
//! prediction at all, beats it nearly 4×.
//!
//! **The exploitable correlation is TEMPORAL.** The same layer's choice for the *previous token*
//! predicts 41.0% — 15.2× identity and 13.1× the floor. Routing follows the hidden state, and the
//! hidden state changes slowly from token to token and completely from layer to layer. That is the
//! mechanism, and it points at a different axis than one-layer-ahead prefetch.
//!
//! ⭐ **Which is the axis `ferric_tier::ExpertCache` already exploits** — hotness-LFU with an LRU
//! tiebreak is precisely a bet on temporal reuse. So this measurement CONFIRMS the cache Ferric
//! shipped and REFUTES the prefetcher it was tempting to add. The prefetcher does not get built.
//!
//! ⚠ **Scope, honestly:** 12 of 41 layers (`FERRIC_MAX_LAYERS`), one checkpoint, 506 pairs. Colibri
//! measured GLM-5.2, a different router — this does not show their number is wrong *for their
//! model*, it shows the premise does not transfer to this one, which is the only thing a single
//! checkpoint can show. Anyone shipping expert speculation should run this on their own weights.
//!
//! ## What "predictable" has to mean, and the trap in it
//!
//! The only predictor available while layer L runs is **layer L's own chosen set**. So "predictable
//! one layer ahead" IS the identity overlap `|S_L ∩ S_{L+1}| / k`, and the interesting question is
//! not whether that number is large — it is whether it beats **per-layer popularity**, which is a
//! STATIC property of the checkpoint requiring no prediction, no per-token work, and no prefetcher.
//!
//! ⛔ **The registered decision rule, before any number exists:** if per-layer popularity matches or
//! beats the identity predictor, speculative prefetch adds nothing over the hotness table already in
//! `ferric_tier::ExpertCache`, and it does not get built. `expert.rs:1` is already frequency-primary
//! with an LRU tiebreak — beating a plain LRU would be a result about LRU, not about speculation.
//!
//! ## Why the baselines are sized, not just listed
//!
//! Top-k of `n_expert` at random overlaps by `k/n_expert` in expectation — for top-8 of 256 that is
//! 0.25 experts, **3.1%**. Anything below that is noise; anything near it is nothing. Stating the
//! floor is what makes the other numbers mean something.
//!
//! ⚠ **Popularity is fitted on a HELD-OUT split.** Fitting it on the tokens it is scored against
//! would make the strongest baseline stronger still — and since the baseline is what the predictor
//! must beat, leakage there biases toward the conclusion "do not build it". The split removes that,
//! in whichever direction it would have cut.
//!
//! ⚠ Contention does not corrupt this measurement the way it corrupts `fabric_profile` — routing
//! indices are load-independent, so a busy machine makes this slow, not wrong. Load is printed
//! anyway, because a reader cannot tell which kind of measurement they are looking at.
//!
//!   FERRIC_ROUTE_TRACE=1 cargo run -p ferric-llama --example route_predictability --release -- <model.gguf>
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::qwen35::{route_trace, Ffn, Qwen35};
use std::collections::HashMap;
use std::sync::Arc;

/// Prompts spread across domains: routing follows the hidden state, so one prompt measures one
/// trajectory and a number averaged over it is about that sentence.
const PROMPTS: &[&str] = &[
    "The capital of France is Paris, and the river that runs through it",
    "def quicksort(a):\n    if len(a) <= 1:\n        return a\n    pivot =",
    "In 1687 Newton published the Principia, which set out three laws of",
    "SELECT customer_id, SUM(total) FROM orders WHERE created_at >",
    "Die Katze sitzt auf dem Tisch und schaut aus dem Fenster, weil",
    "The mitochondrion generates most of the cell's supply of adenosine",
];

fn load_avg() -> Option<f64> {
    let out = std::process::Command::new("uptime").output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout).to_string();
    let tail = s.split("load average").nth(1)?.to_string();
    tail.trim_start_matches(|c: char| !c.is_ascii_digit())
        .split(|c: char| c == ',' || c == ' ')
        .find(|p| !p.is_empty())?.parse().ok()
}

/// |a ∩ b| as a fraction of k.
fn overlap(a: &[u32], b: &[u32]) -> f64 {
    let hit = a.iter().filter(|x| b.contains(x)).count();
    hit as f64 / a.len().max(1) as f64
}

fn top_n(counts: &HashMap<u32, u64>, n: usize) -> Vec<u32> {
    let mut v: Vec<(u32, u64)> = counts.iter().map(|(&e, &c)| (e, c)).collect();
    // Tie-break by expert id so the baseline is deterministic — a popularity list that reorders
    // between runs would make its own score irreproducible.
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v.into_iter().take(n).map(|(e, _)| e).collect()
}

fn main() { pollster::block_on(run()); }

async fn run() {
    assert!(route_trace::enabled(),
            "set FERRIC_ROUTE_TRACE=1 — without it the runtime records nothing and every number \
             below would be computed over an empty trace");

    let path = std::env::args().nth(1)
        .unwrap_or_else(|| format!("{}/.cache/ferric/apodex-1.1-mini-q4km.gguf",
                                   std::env::var("HOME").unwrap()));
    let g = GgufFile::open(&path).expect("open gguf");
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));

    let tokens: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(v)) => v.iter().map(|x| if let Meta::Str(s) = x { s.clone() } else { String::new() }).collect(),
        _ => panic!("no tokenizer in {path}"),
    };
    let vocab: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") {
        Some(Meta::Arr(v)) => v.iter().filter_map(|x| if let Meta::Str(s) = x {
            s.split_once(' ').map(|(p, q)| (p.to_string(), q.to_string())) } else { None }).collect(),
        _ => Vec::new(),
    };
    let bpe = ferric_tokenizer::Bpe::new(vocab, &merges);

    let t0 = std::time::Instant::now();
    let m = Qwen35::load(&ctx, &g).expect("load");
    let moe_layers: Vec<usize> = (0..m.layers.len())
        .filter(|&i| matches!(m.layers[i].ffn, Ffn::Moe(_))).collect();
    let k = m.cfg.n_expert_used;
    let ne = m.cfg.n_expert;
    println!("MoE routing predictability — {path}");
    println!("  loaded in {:.1}s · load avg {:?}", t0.elapsed().as_secs_f64(), load_avg());
    println!("  {} blocks, {} of them MoE, {ne} experts, top-{k}\n", m.layers.len(), moe_layers.len());
    assert!(moe_layers.len() >= 2, "need at least two MoE layers to ask a one-layer-ahead question");

    // ---- capture -------------------------------------------------------------------------
    // sel[prompt][step] = Vec over MoE layers of the k chosen ids for the LAST token of that step.
    let mut sel: Vec<Vec<Vec<Vec<u32>>>> = Vec::new();
    for (pi, p) in PROMPTS.iter().enumerate() {
        let ids: Vec<u32> = bpe.encode(p);
        assert!(ids.len() >= 4, "prompt {pi} tokenized to {} tokens", ids.len());
        let _ = route_trace::take(); // drop anything from a previous prompt
        let mut cache = ferric_llama::qwen35::Cache::new(&m.cfg);
        let n = m.layers.len();
        let _ = m.forward_cached(&ids, &mut cache, n);

        // Read back AFTER the forward: safe, and off the hot path. See route_trace's docs.
        let rec = route_trace::take();
        assert_eq!(rec.len(), moe_layers.len(),
                   "prompt {pi}: recorded {} MoE invocations for {} MoE layers — the trace and the \
                    model disagree about the architecture", rec.len(), moe_layers.len());
        let mut per_token: Vec<Vec<Vec<u32>>> = vec![Vec::new(); ids.len()];
        for t in rec.iter() {
            let v = t.to_vec().await;
            let rows = v.len() / (2 * k);
            assert_eq!(rows, ids.len(), "selection tensor has {rows} rows for {} tokens", ids.len());
            for tok in 0..rows {
                // moe_topk row layout: [w_0..w_{k-1} | idx_0..idx_{k-1}] — ids are the SECOND half.
                let mut e: Vec<u32> = (0..k).map(|j| v[tok * 2 * k + k + j] as u32).collect();
                assert!(e.iter().all(|&x| (x as usize) < ne),
                        "expert id {:?} out of range for {ne} experts — the row layout is being \
                         read wrong (weights mistaken for indices)", e.iter().max());
                e.sort_unstable();
                e.dedup();
                assert_eq!(e.len(), k, "top-{k} returned {} distinct experts", e.len());
                per_token[tok].push(e);
            }
        }
        sel.push(per_token);
        println!("  prompt {pi}: {} tokens x {} MoE layers captured", ids.len(), moe_layers.len());
    }

    // ---- baselines, fitted on a held-out split -------------------------------------------
    let half = sel.len() / 2;
    let (fit, eval) = sel.split_at(half.max(1));
    let mut global: HashMap<u32, u64> = HashMap::new();
    let mut per_layer: Vec<HashMap<u32, u64>> = vec![HashMap::new(); moe_layers.len()];
    for pr in fit { for tok in pr { for (li, e) in tok.iter().enumerate() {
        for &x in e { *global.entry(x).or_default() += 1; *per_layer[li].entry(x).or_default() += 1; }
    }}}
    let g_top = top_n(&global, k);
    let l_top: Vec<Vec<u32>> = per_layer.iter().map(|c| top_n(c, k)).collect();

    // ---- score ----------------------------------------------------------------------------
    let (mut identity, mut pop_layer, mut pop_global, mut prev_tok, mut n) = (0.0, 0.0, 0.0, 0.0, 0u64);
    // ⛔ Its own denominator. `prev_tok` is only defined for ti > 0, and dividing it by `n` — which
    // counts every pair including the first token's — understated it by (T-1)/T. A predictor scored
    // against a larger population than it was evaluated on is penalised for pairs it never saw.
    let mut n_prev = 0u64;
    let mut by_layer: Vec<(f64, f64, u64)> = vec![(0.0, 0.0, 0); moe_layers.len()];
    for pr in eval {
        for (ti, tok) in pr.iter().enumerate() {
            for li in 0..tok.len().saturating_sub(1) {
                let (cur, next) = (&tok[li], &tok[li + 1]);
                identity += overlap(next, cur);
                pop_layer += overlap(next, &l_top[li + 1]);
                pop_global += overlap(next, &g_top);
                if ti > 0 { prev_tok += overlap(next, &pr[ti - 1][li + 1]); n_prev += 1; }
                let b = &mut by_layer[li + 1];
                b.0 += overlap(next, cur); b.1 += overlap(next, &l_top[li + 1]); b.2 += 1;
                n += 1;
            }
        }
    }
    assert!(n > 0, "nothing was scored — the evaluation split was empty");
    let f = n as f64;
    let random_floor = k as f64 / ne as f64;

    println!("\n  scored {n} (layer L -> L+1) pairs on {} held-out prompt(s)\n", eval.len());
    println!("  {:<34} {:>10}", "predictor", "overlap");
    println!("  {:-<46}", "");
    println!("  {:<34} {:>9.1}%", "identity  (L's set -> L+1)", 100.0 * identity / f);
    println!("  {:<34} {:>9.1}%", "per-layer popularity [held-out]", 100.0 * pop_layer / f);
    println!("  {:<34} {:>9.1}%", "global popularity   [held-out]", 100.0 * pop_global / f);
    println!("  {:<34} {:>9.1}%", "previous token, same layer",
             if n_prev == 0 { f64::NAN } else { 100.0 * prev_tok / n_prev as f64 });
    println!("  {:<34} {:>9.1}%", "random top-k (the floor)", 100.0 * random_floor);

    let (id_s, pl_s) = (identity / f, pop_layer / f);
    println!("\n  Colibri's published figure for this quantity: 71.6%");
    println!("  ⚠ SCOPE: {} of the checkpoint's MoE layers, {} prompts, {n} pairs ({n_prev} for\n  \
              prev-token). One checkpoint. Colibri measured GLM-5.2, a different router.",
             moe_layers.len(), PROMPTS.len());
    println!("\n  {:<34} {:>10}", "per-layer detail (first 12)", "id / pop");
    for (li, b) in by_layer.iter().enumerate().take(13).skip(1) {
        if b.2 == 0 { continue }
        println!("  {:<34} {:>9}", format!("  MoE layer {}", moe_layers[li]),
                 format!("{:.0}% / {:.0}%", 100.0 * b.0 / b.2 as f64, 100.0 * b.1 / b.2 as f64));
    }

    let pt = if n_prev == 0 { 0.0 } else { prev_tok / n_prev as f64 };
    println!("\n  ---- what the structure actually is ----");
    println!("  previous-token beats identity {:.1}x and the floor {:.1}x, so the exploitable",
             pt / id_s.max(1e-9), pt / random_floor);
    println!("  correlation is TEMPORAL, not cross-layer. That is what a hotness/recency cache");
    println!("  already exploits, and it is the axis Colibri's one-layer-ahead prefetch does not.");

    println!("\n  ---- the registered decision ----");
    if id_s > pl_s + 0.05 {
        println!("  ✅ identity beats per-layer popularity by {:.1} points. Speculative prefetch has",
                 100.0 * (id_s - pl_s));
        println!("     something to exploit that a static popularity table does not.");
    } else {
        println!("  ⛔ identity does NOT clear per-layer popularity by 5 points ({:.1}% vs {:.1}%).",
                 100.0 * id_s, 100.0 * pl_s);
        println!("     Speculative prefetch would be re-deriving, per token, a table that could be");
        println!("     computed once at load. ferric_tier::ExpertCache's hotness-LFU already holds it.");
        println!("     REFUTED on this checkpoint — the prefetcher does not get built.");
    }
}
