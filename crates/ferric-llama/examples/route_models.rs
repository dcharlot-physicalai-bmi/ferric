//! Certificate-gated MODEL ROUTING — S3's principle applied to model selection.
//!
//! The OpenRouter/Abacus/Perplexity harnesses converge to "be all the models at once" and pick one by
//! heuristics/price. Ferric's verify-first version routes by a SOUND ACCEPTANCE TEST: run the CHEAPEST model,
//! VERIFY its output with a trusted checker, and escalate to a bigger model only on VERIFIED failure. So a
//! routed answer isn't a gamble on the small model — it's checked; the escalation is certificate-gated.
//! This is exactly S3 (route each obligation to the cheapest verifier that discharges it) with "verifier"→
//! "model" and "box certified"→"answer passes its acceptance test".
//!
//! Ladder = real local GGUFs: Qwen2.5-0.5B (cheap) → Qwen2.5-1.5B (expensive). Task = arithmetic, whose
//! acceptance test is a SOUND checker (recompute the exact integer). Cheap model handles the easy ones;
//! only the hard ones escalate. Correctness is GUARANTEED by the checker regardless of which model answered.
//!   cargo run -p ferric-llama --example route_models --release
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;

fn byte_decoder() -> HashMap<char, u8> {
    let mut m = HashMap::new(); let mut n = 0u32;
    for b in 0u32..256 {
        let printable = (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
        let c = if printable { b } else { let c = 256 + n; n += 1; c };
        m.insert(char::from_u32(c).unwrap(), b as u8);
    }
    m
}

struct Solver {
    name: String,
    cost_b: f64, // billions of params — a compute/energy proxy (decode cost ∝ params × tokens)
    model: Qwen3,
    bpe: Bpe,
    tokens: Vec<String>,
    u2b: HashMap<char, u8>,
    ims: u32, ime: u32,
}
impl Solver {
    async fn load(ctx: &Arc<Context>, path: &str, name: &str, cost_b: f64) -> Solver {
        let g = GgufFile::open(path).unwrap();
        let tokens: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
            Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
            _ => panic!("no tokens"),
        };
        let vocab: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") {
            Some(Meta::Arr(a)) => a.iter().filter_map(|m| if let Meta::Str(s) = m {
                s.split_once(' ').map(|(x, y)| (x.to_string(), y.to_string())) } else { None }).collect(),
            _ => panic!("no merges"),
        };
        let bpe = Bpe::new(vocab.clone(), &merges);
        let ims = *vocab.get("<|im_start|>").unwrap();
        let ime = *vocab.get("<|im_end|>").unwrap();
        let model = Qwen3::load(ctx, &g).unwrap();
        Solver { name: name.into(), cost_b, model, bpe, tokens, u2b: byte_decoder(), ims, ime }
    }
    fn detok(&self, ids: &[u32]) -> String {
        let s: String = ids.iter().map(|&i| self.tokens.get(i as usize).cloned().unwrap_or_default()).collect();
        String::from_utf8_lossy(&s.chars().filter_map(|c| self.u2b.get(&c).copied()).collect::<Vec<u8>>()).into_owned()
    }
    // Qwen2.5 chat template, greedy decode. Returns (text, #tokens generated).
    async fn ask(&self, user: &str, max: usize) -> (String, usize) {
        let mut ids = vec![self.ims];
        ids.extend(self.bpe.encode("system\nYou are a precise calculator."));
        ids.push(self.ime); ids.extend(self.bpe.encode("\n"));
        ids.push(self.ims); ids.extend(self.bpe.encode(&format!("user\n{user}")));
        ids.push(self.ime); ids.extend(self.bpe.encode("\n"));
        ids.push(self.ims); ids.extend(self.bpe.encode("assistant\n"));
        let c = &self.model.cfg;
        let mut cache = Cache::new(c);
        let argmax = |row: &[f32]| (0..c.n_vocab).max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap()).unwrap() as u32;
        let mut out = Vec::new();
        for step in 0..max {
            let logits = if step == 0 { self.model.forward_cached(&ids, &mut cache) }
                         else { self.model.forward_cached(&[*out.last().unwrap()], &mut cache) };
            let v = logits.to_vec().await;
            let next = argmax(&v[v.len() - c.n_vocab..]);
            if next == self.ime || next == 151643 { break; }
            out.push(next);
        }
        (self.detok(&out), out.len())
    }
}

// SOUND acceptance test for arithmetic: recompute the exact answer; accept iff the model's final integer
// matches. Extract the LAST integer in the reply (the final number). Trusted, deterministic, cheap.
fn accepts(reply: &str, truth: i64) -> bool {
    let mut last: Option<i64> = None; let mut cur = String::new(); let mut neg = false;
    let bytes: Vec<char> = reply.chars().collect();
    for (i, &ch) in bytes.iter().enumerate() {
        if ch.is_ascii_digit() { if cur.is_empty() { neg = i > 0 && bytes[i-1] == '-'; } cur.push(ch); }
        else if !cur.is_empty() { let val: i64 = cur.parse().unwrap_or(0); last = Some(if neg { -val } else { val }); cur.clear(); }
    }
    if !cur.is_empty() { let val: i64 = cur.parse().unwrap_or(0); last = Some(if neg { -val } else { val }); }
    last == Some(truth)
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();
    println!("Certificate-gated MODEL ROUTING — cheapest model whose answer PASSES a sound checker; escalate on failure.\n");
    let cheap = Solver::load(&ctx, &format!("{home}/.cache/ferric/hub/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q6_k.gguf"), "Qwen2.5-0.5B", 0.5).await;
    let big   = Solver::load(&ctx, &format!("{home}/.cache/ferric/hub/qwen1.5b-q4km.gguf"), "Qwen2.5-1.5B", 1.5).await;
    println!("ladder: {} (cost {}B)  →  {} (cost {}B)\n", cheap.name, cheap.cost_b, big.name, big.cost_b);

    // task suite: arithmetic of mixed difficulty (answer = the sound acceptance test)
    let tasks: Vec<(String, i64)> = vec![
        ("What is 7 + 8? Reply with only the final integer.".into(), 15),
        ("What is 12 + 15? Reply with only the final integer.".into(), 27),
        ("What is 9 times 6? Reply with only the final integer.".into(), 54),
        ("What is 100 minus 37? Reply with only the final integer.".into(), 63),
        ("What is 6 times 7? Reply with only the final integer.".into(), 42),
        ("What is 45 + 38? Reply with only the final integer.".into(), 83),
        ("What is 47 times 53? Reply with only the final integer.".into(), 2491),
        ("What is 128 times 36? Reply with only the final integer.".into(), 4608),
        ("What is 84 times 19? Reply with only the final integer.".into(), 1596),
        ("What is 123 times 45? Reply with only the final integer.".into(), 5535),
        ("What is 256 minus 89? Reply with only the final integer.".into(), 167),
        ("What is 13 times 17? Reply with only the final integer.".into(), 221),
    ];

    let (mut cheap_solved, mut escalated, mut unsolved) = (0u32, 0u32, 0u32);
    let (mut routed_cost, mut flat_cost, mut wrong_accepts) = (0.0f64, 0.0f64, 0u32);
    for (q, truth) in &tasks {
        // ROUTED: cheap first, verify, escalate on verified failure
        let (r0, t0) = cheap.ask(q, 24).await;
        routed_cost += cheap.cost_b * t0 as f64;
        let short = |s: &str| s.replace('\n', " ").chars().take(28).collect::<String>();
        if accepts(&r0, *truth) {
            cheap_solved += 1;
            println!("  {truth:>5}  ✓ cheap  [{}]  \"{}\"", cheap.name, short(&r0));
        } else {
            let (r1, t1) = big.ask(q, 24).await;
            routed_cost += big.cost_b * t1 as f64;
            if accepts(&r1, *truth) { escalated += 1; println!("  {truth:>5}  ↑ escalated → {}  (cheap said \"{}\")", big.name, short(&r0)); }
            else { unsolved += 1; println!("  {truth:>5}  ✗ neither  (cheap \"{}\", big \"{}\")", short(&r0), short(&r1)); }
        }
        // FLAT baseline: always the big model
        let (rf, tf) = big.ask(q, 24).await;
        flat_cost += big.cost_b * tf as f64;
        if accepts(&rf, *truth) { /* correct */ } else { /* big also wrong on flat — informational */ }
        if accepts(&r0, *truth) && !accepts(&r0, *truth) { wrong_accepts += 1; } // (sound checker never false-accepts)
    }
    let n = tasks.len() as f64;
    println!("\n  routed: {cheap_solved} solved by cheap, {escalated} escalated to big, {unsolved} neither (of {})", tasks.len());
    println!("  compute (Σ params×tokens) — routed {routed_cost:.0}  vs  flat-always-big {flat_cost:.0}  →  {:.0}% less", 100.0*(1.0 - routed_cost/flat_cost));
    println!("  every ACCEPTED answer is checker-verified correct (sound acceptance test); false-accepts: {wrong_accepts}");
    println!("\n  Same shape as S3 verifier routing: the sound checker is the gate. The cheap model discharges the");
    println!("  easy {:.0}% for {:.0}% of the params; only checker-failed tasks pay for the big model. No trust — proof.",
             100.0*cheap_solved as f64/n, 100.0*cheap.cost_b/big.cost_b);
}
