//! `ferric bench <model>` — prompt-processing and generation speed, and what each token costs in
//! joules, measured through the server exactly as a user's request is.
//!
//! Two tests, llama-bench's split: **pp** is a long prompt with one generated token (the prefill), **tg**
//! a short prompt with many (the decode). They are separate requests because the server attributes
//! joules per REQUEST — a single request cannot say how its energy divides between the two phases.
//!
//! Every request starts with a unique nonce and is sent `raw`, so no chat template puts a shared system
//! prompt in front of it: ferric-serve caches prompt prefixes across requests, and a repeated prompt
//! would measure the cache, not the model. The machine is shared (load average 20-40 is normal here),
//! so every figure is reported as the range over repetitions, never as one number.
use crate::http::{self, Host};
use serde_json::{json, Value};
use std::time::Duration;

pub struct BenchOpts { pub reps: usize, pub prompt_tokens: usize, pub gen_tokens: usize, pub gap: Duration }

/// Prose of roughly `n` tokens (≈ one per word for BPE vocabularies); the server's own count is
/// what gets reported.
fn filler(n: usize) -> String {
    const WORDS: &[&str] = &["the", "river", "carried", "cold", "water", "past", "an", "old", "stone", "mill", "where",
        "a", "miller", "counted", "sacks", "of", "grain", "each", "morning", "before", "sunrise", "and", "wrote",
        "numbers", "in", "small", "book", "that", "his", "daughter", "later", "read", "aloud", "to", "village"];
    let mut s = String::new();
    for i in 0..n {
        if i > 0 { s.push(' '); }
        s.push_str(WORDS[(i * 7 + i / WORDS.len()) % WORDS.len()]);
        if i % 17 == 16 { s.push('.'); }
    }
    s
}

fn nonce(i: usize) -> String {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    format!("Record {i}-{:x}:", t % 0xffff_ffff)
}

fn generate(host: &Host, model: &str, prompt: &str, n: usize) -> Result<Value, String> {
    http::json(host, "POST", "/api/generate", Some(&json!({
        "model": model, "prompt": prompt, "raw": true, "stream": false,
        "options": {"num_predict": n, "temperature": 0, "seed": 42},
    })))
}

struct Sample { tokens: f64, tok_s: Option<f64>, joules: Option<f64>, why: Option<String> }

fn sample(v: &Value, prefill: bool) -> Sample {
    let (n, d) = if prefill { ("prompt_eval_count", "prompt_eval_duration") } else { ("eval_count", "eval_duration") };
    let tokens = v[n].as_f64().unwrap_or(0.0);
    let dur = v[d].as_f64().unwrap_or(0.0) / 1e9;
    let e = &v["energy"];
    Sample {
        tokens,
        tok_s: (dur > 0.0 && tokens > 0.0).then(|| tokens / dur),
        joules: e["joules"].as_f64(),
        why: if e.is_null() { Some("this server reports no energy".into()) } else { e["why"].as_str().filter(|_| e["joules"].is_null()).map(str::to_string) },
    }
}

fn range(xs: &[f64], prec: usize) -> String {
    if xs.is_empty() { return "—".into(); }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = if v.len() % 2 == 1 { v[v.len() / 2] } else { 0.5 * (v[v.len() / 2 - 1] + v[v.len() / 2]) };
    if v.len() == 1 || v[0] == v[v.len() - 1] { format!("{:.p$}", v[0], p = prec) }
    else { format!("{:.p$}–{:.p$} (median {:.p$})", v[0], v[v.len() - 1], med, p = prec) }
}

pub fn bench(host: &Host, model: &str, label: &str, o: &BenchOpts) -> Result<(), String> {
    eprintln!("warming up {label} …");
    generate(host, model, &format!("{} Hello", nonce(0)), 8)?;
    std::thread::sleep(o.gap);
    let (mut pp, mut tg) = (Vec::new(), Vec::new());
    for r in 1..=o.reps {
        eprintln!("run {r}/{}: pp …", o.reps);
        pp.push(sample(&generate(host, model, &format!("{} {}", nonce(r), filler(o.prompt_tokens)), 1)?, true));
        std::thread::sleep(o.gap);
        eprintln!("run {r}/{}: tg …", o.reps);
        tg.push(sample(&generate(host, model, &format!("{} Once upon a time, in a village by a river,", nonce(r)), o.gen_tokens)?, false));
        std::thread::sleep(o.gap);
    }
    let row = |name: String, s: &[Sample]| {
        let toks: Vec<f64> = s.iter().map(|x| x.tokens).collect();
        let rate: Vec<f64> = s.iter().filter_map(|x| x.tok_s).collect();
        let jpt: Vec<f64> = s.iter().filter_map(|x| x.joules.filter(|_| x.tokens > 0.0).map(|j| j / x.tokens)).collect();
        let j: Vec<f64> = s.iter().filter_map(|x| x.joules).collect();
        vec![name, range(&toks, 0), range(&rate, 1), range(&jpt, 4), range(&j, 3), format!("{}/{}", j.len(), s.len())]
    };
    // Named by the tokens the SERVER counted (llama-bench's pp512 is exactly 512); the filler is
    // words, and a tokenizer turns 512 of them into ~560.
    let med = |s: &[Sample]| { let mut t: Vec<f64> = s.iter().map(|x| x.tokens).collect(); t.sort_by(|a, b| a.partial_cmp(b).unwrap()); t.get(t.len() / 2).copied().unwrap_or(0.0) };
    let rows = vec![
        ["test", "tokens", "tok/s", "J/token", "joules", "metered"].map(String::from).to_vec(),
        row(format!("pp{}", med(&pp)), &pp),
        row(format!("tg{}", med(&tg)), &tg),
    ];
    println!("model  {label}");
    println!("host   {}   {} run(s) per test; ranges are min–max over runs (a shared machine is noisy)", host.addr(), o.reps);
    println!();
    let w: Vec<usize> = (0..rows[0].len()).map(|i| rows.iter().map(|r| r[i].chars().count()).max().unwrap_or(0)).collect();
    for r in &rows {
        println!("{}", r.iter().enumerate().map(|(i, c)| format!("{c:<w$}", w = w[i])).collect::<Vec<_>>().join("  ").trim_end());
    }
    println!();
    println!("pp: prompt tokens ÷ prompt-eval time; J/token = the request's joules ÷ prompt tokens (1 token generated).");
    println!("tg: generated tokens ÷ eval time; J/token = the request's joules ÷ generated tokens (short prompt).");
    let mut whys: Vec<String> = pp.iter().chain(&tg).filter_map(|s| s.why.clone()).collect();
    whys.sort(); whys.dedup();
    for w in whys { println!("energy not measured on some runs: {w}"); }
    // The server subtracts an idle baseline taken from the gaps between requests. On a shared machine
    // the background draw can move by more than a light request adds, and the difference goes
    // negative — which is not a measurement of the request, and is said so rather than averaged away.
    if pp.iter().chain(&tg).any(|s| s.joules.is_some_and(|j| j < 0.0)) {
        println!("a negative figure means the machine's background draw moved more than the request added:\n  that run is below what the meter resolves here; a quieter machine or longer runs (--gen-tokens) narrow it.");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_and_filler() {
        assert_eq!(range(&[3.0, 1.0, 2.0], 1), "1.0–3.0 (median 2.0)");
        assert_eq!(range(&[5.0], 2), "5.00");
        assert_eq!(range(&[8.0, 8.0], 0), "8", "no spread, no range");
        assert_eq!(range(&[], 2), "—");
        assert_eq!(filler(64).split_whitespace().count(), 64);
        let s = sample(&json!({"prompt_eval_count": 500, "prompt_eval_duration": 2.5e8, "energy": {"joules": 1.0}}), true);
        assert_eq!((s.tokens, s.tok_s, s.joules), (500.0, Some(2000.0), Some(1.0)));
        assert!(sample(&json!({"eval_count": 5, "eval_duration": 1e9}), false).why.unwrap().contains("no energy"));
    }
}
