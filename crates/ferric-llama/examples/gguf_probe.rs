//! **What would it take to run this checkpoint?** A GGUF header reader that answers that directly.
//!
//! A GGUF's metadata and tensor table sit at the front of the file, so the first few tens of MB of a
//! 17 GB checkpoint are enough to learn its architecture, its layer layout and every quant format it
//! uses. That is the difference between a 30-second range request and an overnight download when the
//! question is "can we load this at all".
//!
//! Prints, in the order you need them to plan work:
//!   * `general.architecture` and the shape metadata, so the loader's arch dispatch can be checked
//!   * whether the file needs an SSM/hybrid path, an MoE path, or plain attention
//!   * every quant type present, and **whether this runtime has a packed kernel for it** — the
//!     distinction that let a correct IQ4_XS kernel sit unreachable for months
//!   * tensor names collapsed to `blk.*.` patterns, which is what a port is actually written against
//!
//!   cargo run -p ferric-llama --example gguf_probe --release -- <file.gguf>
use ferric_gguf::{parse, Meta};

fn short(m: &Meta) -> String {
    match m {
        Meta::U(v) => format!("{v} (u)"),
        Meta::I(v) => v.to_string(),
        Meta::F(v) => format!("{v} (f)"),
        Meta::Bool(v) => v.to_string(),
        Meta::Str(s) if s.len() <= 60 => format!("{s:?}"),
        Meta::Str(s) => format!("{:?}… ({} chars)", &s[..57.min(s.len())], s.len()),
        // Small numeric arrays are printed in full: per-layer arrays (sliding_window_pattern,
        // head_count_kv, feed_forward_length) ARE the layer schedule, and a port is written against
        // their contents, not their length. A loader reading them with a scalar accessor gets Err and
        // silently falls back to a default, which is how a hybrid model quietly becomes all-global.
        Meta::Arr(a) if a.len() <= 64 && !a.iter().any(|m| matches!(m, Meta::Str(_))) => {
            let v: Vec<String> = a.iter().map(|m| match m {
                Meta::U(x) => x.to_string(), Meta::I(x) => x.to_string(),
                Meta::F(x) => format!("{x}"), Meta::Bool(x) => (*x as u8).to_string(), _ => "?".into(),
            }).collect();
            format!("[{}] {}", a.len(), v.join(","))
        }
        Meta::Arr(a) => format!("[{} items]", a.len()),
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: gguf_probe <file.gguf>");
    let bytes = std::fs::read(&path).expect("read");
    println!("Probing {path}  ({:.1} MB of header read)\n", bytes.len() as f64 / 1e6);

    let g = match parse(bytes) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("header did not parse: {e}");
            eprintln!("If this is a truncated download, fetch more bytes — the tokenizer alone can be");
            eprintln!("tens of MB on a 200k-vocab model, and the tensor table follows it.");
            std::process::exit(1);
        }
    };

    // `--tensor NAME` dequantizes one tensor and prints statistics. Verifying a claim about weight
    // VALUES (a norm that is a constant vector, an embedding table that is already row-normalised)
    // needs the numbers, not the header.
    if std::env::args().nth(2).as_deref() == Some("--tensor") {
        let name = std::env::args().nth(3).expect("--tensor NAME");
        let t = g.tensor(&name).unwrap_or_else(|| panic!("no tensor {name}"));
        let dims = t.dims.clone();
        let v = ferric_gguf::GgufSource::dequant(&g, &name).expect("dequant");
        let (mut mn, mut mx, mut sum) = (f32::MAX, f32::MIN, 0f64);
        for &x in &v { mn = mn.min(x); mx = mx.max(x); sum += x as f64; }
        let mean = sum / v.len() as f64;
        let var: f64 = v.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / v.len() as f64;
        println!("{name} dims={dims:?} n={}", v.len());
        println!("  min {mn:.6}  max {mx:.6}  mean {mean:.6}  sd {:.6}", var.sqrt());
        println!("  first 8: {:?}", &v[..8.min(v.len())]);
        // Per-row RMS over the LAST dim, which is what "already row-normalised" would mean.
        let d = *dims.first().unwrap_or(&1) as usize;
        if d > 1 && v.len() % d == 0 && v.len() / d > 1 {
            let rows = v.len() / d;
            let step = (rows / 16).max(1);
            let mut rms: Vec<f32> = Vec::new();
            for r in (0..rows).step_by(step).take(16) {
                let s: f32 = v[r * d..(r + 1) * d].iter().map(|x| x * x).sum();
                rms.push((s / d as f32).sqrt());
            }
            let (lo, hi) = (rms.iter().cloned().fold(f32::MAX, f32::min), rms.iter().cloned().fold(0f32, f32::max));
            println!("  per-row RMS over {d} (16 samples): min {lo:.6} max {hi:.6} ratio {:.4}", hi / lo);
        }
        return;
    }

    // Optional second arg: dump one metadata value in full. Chat templates and tokenizer regexes are
    // the values you actually need verbatim, and they are exactly the ones a summary line truncates.
    if let Some(key) = std::env::args().nth(2) {
        match g.metadata.get(&key) {
            Some(Meta::Str(v)) => { println!("{v}"); return; }
            Some(other) => { println!("{}", short(other)); return; }
            None => { eprintln!("no such key: {key}"); std::process::exit(1); }
        }
    }

    let arch = match g.metadata.get("general.architecture") {
        Some(Meta::Str(s)) => s.clone(),
        _ => "(none)".into(),
    };
    println!("  architecture: {arch}");

    // ---- shape metadata, arch-prefixed keys first ----
    let mut keys: Vec<&String> = g.metadata.keys().collect();
    keys.sort();
    println!("\n  metadata ({} keys; tokenizer.* and arrays elided):", keys.len());
    for k in &keys {
        if k.starts_with("tokenizer.") { continue; }
        println!("    {k:<44} {}", short(&g.metadata[*k]));
    }
    for k in keys.iter().filter(|k| k.starts_with("tokenizer.")) {
        println!("    {k:<44} {}", short(&g.metadata[*k]));
    }

    // ---- what kind of model is this ----
    let names: Vec<&str> = g.tensors.iter().map(|t| t.name.as_str()).collect();
    let has = |sub: &str| names.iter().any(|n| n.contains(sub));
    println!("\n  shape of the port:");
    println!("    SSM / linear-attention layers: {}", if has("ssm_") || has(".conv1d") { "YES — needs a hybrid mixer path" } else { "no" });
    println!("    MoE experts:                   {}", if has("ffn_gate_exps") || has("_exps") { "YES — needs the MoE path" } else { "no" });
    println!("    multi-token prediction:        {}", if has("mtp") || has("nextn") { "YES" } else { "no" });
    println!("    vision tower in this file:     {}", if has("v.enc") || has("vision") || has("mm.") { "YES" } else { "no (separate mmproj if multimodal)" });
    println!("    QK norm:                       {}", if has("attn_q_norm") { "YES" } else { "no" });

    // ---- quant formats present, and whether we can RUN them packed ----
    let mut by: std::collections::BTreeMap<u32, (usize, u64)> = Default::default();
    for t in &g.tensors {
        let e = by.entry(t.ggml_type).or_default();
        e.0 += 1;
        e.1 += t.dims.iter().product::<u64>();
    }
    println!("\n  quant formats present:");
    println!("    {:>5} {:>9} {:>8} {:>12}   {}", "type", "tensors", "M params", "packed?", "name");
    let mut missing: Vec<u32> = Vec::new();
    for (ty, (n, p)) in &by {
        let native = ferric_tensor::QMatrix::block_bytes(*ty).is_some();
        // f32/f16/bf16 are not block quants; the dense path is correct for them, not a fallback.
        let plain = matches!(*ty, 0 | 1 | 30);
        if !native && !plain { missing.push(*ty); }
        let name = match *ty {
            0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1", 8 => "Q8_0",
            10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K", 14 => "Q6_K",
            20 => "IQ4_NL", 23 => "IQ4_XS", 30 => "BF16", 35 => "TQ2_0", 39 => "MXFP4",
            40 => "NVFP4", 41 => "Q1_0", 42 => "Q2_0", _ => "?",
        };
        let status = if native { "packed" } else if plain { "dense (ok)" } else { "NO KERNEL" };
        println!("    {ty:>5} {n:>9} {:>8.1} {status:>12}   {name}", *p as f64 / 1e6);
    }

    // ---- the layer template a port is written against ----
    let mut pat: std::collections::BTreeMap<String, usize> = Default::default();
    for n in &names {
        let p = n.split('.').map(|s| if s.chars().all(|c| c.is_ascii_digit()) { "*" } else { s })
            .collect::<Vec<_>>().join(".");
        *pat.entry(p).or_default() += 1;
    }
    println!("\n  tensor patterns ({} distinct, {} tensors):", pat.len(), g.tensors.len());
    for (p, n) in &pat { println!("    {n:>4}x  {p}"); }

    // ---- blk.0 shapes: a port is written against these, and a wrong assumption here produces
    // finite logits and fluent garbage rather than an error ----
    println!("\n  blk.0 tensor shapes:");
    let mut b0: Vec<&ferric_gguf::TensorInfo> = g.tensors.iter()
        .filter(|t| t.name.starts_with("blk.0.") || !t.name.starts_with("blk.")).collect();
    b0.sort_by(|a, b| a.name.cmp(&b.name));
    for t in b0 {
        let ty = match t.ggml_type { 0=>"F32",1=>"F16",2=>"Q4_0",6=>"Q5_0",8=>"Q8_0",12=>"Q4_K",13=>"Q5_K",14=>"Q6_K",20=>"IQ4_NL",23=>"IQ4_XS",_=>"?" };
        println!("    {:<34} {:<16} {ty}", t.name, format!("{:?}", t.dims));
    }

    println!("\n  verdict:");
    if missing.is_empty() {
        println!("    Every quant format in this file has a packed kernel in this runtime.");
    } else {
        println!("    ⚠ NO PACKED KERNEL for ggml type(s) {missing:?} — the loader would dequantize");
        println!("      these to f32 (correct, 4-8x the memory) or fail. Write those first.");
    }
}
