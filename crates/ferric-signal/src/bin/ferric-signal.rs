//! `ferric-signal` — tokenize a physical signal, and check that someone else got the same tokens.
//!
//!   ferric-signal tokenize <file.csv> [--channels N] [--patch N] [--seed N] [--receipt out.kv]
//!   ferric-signal verify   <a.kv> <b.kv>
//!   ferric-signal cost     [--window N]
//!
//! Input is one sample per line, comma-separated across channels. No CSV library: a sensor dump is
//! numbers and newlines, and a parse failure names the line rather than guessing at a value.

use ferric_core::Context;
use ferric_signal::sha256::{hex, sha256};
use ferric_signal::{
    agreement, Agreement, EncoderConfig, EncoderWeights, Fsq, HybridVocab, Patcher, RevIn,
    TokenReceipt, TokenSpec, Weights,
};
use ferric_tensor::Tensor;
use std::sync::Arc;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("tokenize") => tokenize(&args[1..]),
        Some("verify") => verify(&args[1..]),
        Some("cost") => cost(&args[1..]),
        _ => {
            eprintln!("{}", USAGE);
            2
        }
    };
    std::process::exit(code);
}

const USAGE: &str = "\
ferric-signal — open sensor-language tokenization

  tokenize <file.csv> [--channels N] [--patch N] [--seed N] [--receipt out.kv]
           [--weights model.fsig] [--save-weights out.fsig]
      Normalize, patch, encode and quantize a signal. Prints the token ids and a
      determinism receipt. Without --weights the encoder is generated from --seed,
      which is reproducible but untrained.

  verify <a.kv> <b.kv>
      Compare two receipts. Reports whether two runs were asked the same question,
      and whether they gave the same answer.

  cost [--window N]
      Operations per token at a given window. Energy needs a meter; see the
      token_cost example, which reports NOT MEASURED rather than inventing a zero.";

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}
fn num(args: &[String], name: &str, default: usize) -> usize {
    flag(args, name).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn read_signal(path: &str, channels: usize) -> Result<Vec<f32>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut cols = 0;
        for field in line.split(',') {
            let f = field.trim();
            // A blank or unparseable field is an ERROR, not a zero. A zero here is a real sample
            // value and would be indistinguishable from a gap in the recording.
            let v: f32 = f.parse().map_err(|_| format!("{path}:{}: cannot parse {f:?}", n + 1))?;
            out.push(v);
            cols += 1;
        }
        if cols != channels {
            return Err(format!("{path}:{}: {cols} columns, expected {channels}", n + 1));
        }
    }
    if out.is_empty() {
        return Err(format!("{path}: no samples"));
    }
    Ok(out)
}

fn tokenize(args: &[String]) -> i32 {
    let Some(path) = args.first().filter(|a| !a.starts_with("--")) else {
        eprintln!("{USAGE}");
        return 2;
    };
    let channels = num(args, "--channels", 1);
    let patch_len = num(args, "--patch", 16);
    let seed = num(args, "--seed", 1) as u64;

    let raw = match read_signal(path, channels) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            return 1;
        }
    };

    let cfg = EncoderConfig { patch_len, ..EncoderConfig::signal_4m() };
    let Ok(ctx) = pollster::block_on(Context::new()) else {
        eprintln!("error: no GPU context. The encoder needs one; nothing is guessed without it.");
        return 1;
    };
    let ctx = Arc::new(ctx);

    let rev = RevIn::fit(&raw, channels).expect("channel count already validated");
    let norm = rev.apply(&raw).unwrap();
    let patcher = Patcher::contiguous(patch_len).unwrap();
    let patches = match patcher.patchify(&norm) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return 1;
        }
    };
    let t = patches.len() / patch_len;

    // Weights come from a file when one is given, and from the seed otherwise. The distinction
    // matters to the receipt: a seed is provenance for GENERATED weights, whereas a real model's
    // provenance is the digest of its bytes.
    let (enc, wdig) = match flag(args, "--weights") {
        Some(path) => {
            let bytes = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => { eprintln!("error: {path}: {e}"); return 1; }
            };
            let w = match Weights::from_bytes(&bytes) {
                Ok(w) => w,
                Err(e) => { eprintln!("error: {path}: {e}"); return 1; }
            };
            let dig = w.digest();
            match EncoderWeights::from_weights(&ctx, cfg, &w) {
                Ok(e) => (e, dig),
                Err(e) => { eprintln!("error: {path}: {e}"); return 1; }
            }
        }
        None => {
            let e = EncoderWeights::deterministic(&ctx, cfg, seed).unwrap();
            let dig = e.to_weights().digest();
            (e, dig)
        }
    };
    if let Some(out) = flag(args, "--save-weights") {
        if let Err(e) = std::fs::write(&out, enc.to_weights().to_bytes()) {
            eprintln!("error: writing {out}: {e}");
            return 1;
        }
        println!("weights written to {out}  digest {}", &wdig[..16]);
    }
    let latents = pollster::block_on(
        enc.forward(&ctx, &Tensor::from_vec(&ctx, &patches, &[t, patch_len]))
            .unwrap()
            .to_vec(),
    );

    let q = Fsq::signal_15bit();
    let vocab = HybridVocab::new(32_000, q.clone()).unwrap();
    let mut ids = Vec::with_capacity(t);
    let mut non_finite = 0u64;
    for i in 0..t {
        let z = &latents[i * cfg.latent_dim..(i + 1) * cfg.latent_dim];
        if z.iter().any(|v| !v.is_finite()) {
            non_finite += 1;
        }
        ids.push(q.to_index(&q.quantize(z).unwrap()).unwrap());
    }

    let spec = TokenSpec::from_parts(
        hex(&sha256(bytemuck_le(&raw).as_slice())),
        channels,
        (&rev.mean, &rev.scale),
        patch_len,
        patcher.stride(),
        &cfg,
        wdig,
        q.levels(),
        vocab.text_len(),
    );
    let receipt = TokenReceipt::new(spec, &ids, non_finite, platform());

    println!("samples {}  channels {}  patches {}  dropped tail {}",
             raw.len() / channels, channels, t, raw.len() / channels - patcher.covered(raw.len() / channels));
    println!("tokens  {}", ids.len());
    let show = ids.len().min(24);
    println!("first {show}: {}", ids[..show].iter().map(|i| i.to_string()).collect::<Vec<_>>().join(" "));
    if non_finite > 0 {
        println!("WARNING {non_finite} window(s) produced a non-finite latent; FSQ clamps, so those \
                  ids look legal and are not");
    }
    println!("\nreceipt");
    let pairs = receipt.to_pairs();
    for (k, v) in &pairs {
        println!("  {k}={v}");
    }
    if let Some(out) = flag(args, "--receipt") {
        let body: String = pairs.iter().map(|(k, v)| format!("{k}={v}\n")).collect();
        if let Err(e) = std::fs::write(&out, body) {
            eprintln!("error: writing {out}: {e}");
            return 1;
        }
        println!("\nwritten to {out}");
    }
    0
}

/// Little-endian bytes of the raw samples, so the signal digest does not depend on host endianness.
fn bytemuck_le(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    b
}

fn platform() -> String {
    format!("{} / {}", std::env::consts::ARCH, std::env::consts::OS)
}

fn read_pairs(path: &str) -> Result<Vec<(String, String)>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut out = Vec::new();
    for line in text.lines() {
        if let Some((k, v)) = line.split_once('=') {
            out.push((k.trim().to_string(), v.trim().to_string()));
        }
    }
    Ok(out)
}

fn verify(args: &[String]) -> i32 {
    let (Some(a), Some(b)) = (args.first(), args.get(1)) else {
        eprintln!("{USAGE}");
        return 2;
    };
    let (pa, pb) = match (read_pairs(a), read_pairs(b)) {
        (Ok(x), Ok(y)) => (x, y),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("error: {e}");
            return 1;
        }
    };
    let get = |p: &[(String, String)], k: &str| p.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone());
    let (Some(sa), Some(ta), Some(sb), Some(tb)) = (
        get(&pa, "spec_digest"), get(&pa, "token_digest"),
        get(&pb, "spec_digest"), get(&pb, "token_digest"),
    ) else {
        eprintln!("error: both files must carry spec_digest and token_digest");
        return 1;
    };

    println!("  a  spec {}  tokens {}", &sa[..16], get(&pa, "tokens").unwrap_or_default());
    println!("  b  spec {}  tokens {}", &sb[..16], get(&pb, "tokens").unwrap_or_default());
    println!("  a  platform {}", get(&pa, "platform").unwrap_or_default());
    println!("  b  platform {}", get(&pb, "platform").unwrap_or_default());
    println!();
    match agreement(&sa, &ta, &sb, &tb) {
        Agreement::Identical => {
            println!("IDENTICAL. Same question, same answer.");
            0
        }
        Agreement::ComputationDiverged => {
            println!("COMPUTATION DIVERGED. The same question produced different tokens.");
            println!("That is a question, not a verdict: a different kernel, reduction order or");
            println!("device will do this. The spec digests match, so the inputs did not.");
            println!("  a token digest {ta}");
            println!("  b token digest {tb}");
            1
        }
        Agreement::DifferentSpec => {
            println!("DIFFERENT SPEC. These runs were not asked the same question, so comparing");
            println!("their tokens would mean nothing. Reported before any token comparison so");
            println!("nobody goes looking for numerical causes that do not exist.");
            for k in ["signal_digest", "patch_len", "stride", "weights_digest", "text_vocab", "fsq_levels", "channels", "build"] {
                let (x, y) = (get(&pa, k).unwrap_or_default(), get(&pb, k).unwrap_or_default());
                if x != y {
                    println!("  differs: {k}  {x}  vs  {y}");
                }
            }
            1
        }
    }
}

fn cost(args: &[String]) -> i32 {
    let cfg = EncoderConfig::signal_4m();
    let w = num(args, "--window", 256);
    let c = cfg.cost(w);
    println!("window {w}  {:.1} MFLOP total  {:.3} MFLOP/token  attention {:.1}% of it",
             c.flops() as f64 / 1e6, c.flops_per_token() / 1e6, c.quadratic_share() * 100.0);
    println!("arithmetic intensity {:.1} FLOP per weight byte", c.flops_per_weight_byte());
    println!("\nA per-token cost is a property of the WINDOW, not of the token: attention is");
    println!("quadratic, so quoting one without its window length is underspecified.");
    0
}
