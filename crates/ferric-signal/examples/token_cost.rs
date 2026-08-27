//! What one sensor token costs, in operations now and in joules where a meter exists.
//!
//!   cargo run -p ferric-signal --example token_cost --release
//!
//! Two halves, deliberately separated. The operation count is exact arithmetic over the encoder
//! configuration and is reported unconditionally. The energy figure needs a hardware counter, and
//! where none is readable this prints that it was not measured rather than printing a zero — a zero
//! is indistinguishable from a very efficient run, which is how most published efficiency claims
//! come to be wrong.

use ferric_joule::{measure, MacBattery, Meter, Nameplate, Rapl};
use ferric_signal::{embed_cost, EncoderConfig, Fsq, Patcher, RevIn};

fn main() {
    let cfg = EncoderConfig::signal_4m();
    let p = cfg.params();

    println!("\nENCODER  patch {}  d_model {}  layers {}  heads {}  d_ff {}  latent {}",
             cfg.patch_len, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff, cfg.latent_dim);
    println!("  parameters {:>12}   ({:.2} MB at f32)", p.total(), p.total() as f64 * 4.0 / 1e6);
    println!("  vocabulary {:>12}   FSQ codes\n", Fsq::signal_15bit().codebook_size());

    println!("  {:>7}  {:>14}  {:>14}  {:>10}  {:>12}", "window", "MFLOP/window", "MFLOP/token", "attn share", "FLOP/wt byte");
    println!("  {:->7}  {:->14}  {:->14}  {:->10}  {:->12}", "", "", "", "", "");
    for w in [16usize, 64, 256, 1024, 4096, 8192] {
        let c = cfg.cost(w);
        println!("  {:>7}  {:>14.1}  {:>14.3}  {:>9.1}%  {:>12.1}",
                 w, c.flops() as f64 / 1e6, c.flops_per_token() / 1e6,
                 c.quadratic_share() * 100.0, c.flops_per_weight_byte());
    }
    println!("\n  A per-token cost is a property of the WINDOW, not of the token: attention is");
    println!("  quadratic, so the same token costs {:.1}x more at 8192 patches than at 16.",
             cfg.cost(8192).flops_per_token() / cfg.cost(16).flops_per_token());

    // ---- the energy half ----
    println!("\nENERGY");
    // Concrete meters tried in order rather than boxed: `measure` takes a sized `M: Meter`, and
    // adding a blanket `impl Meter for Box<dyn Meter>` would mean editing a shared crate for the
    // convenience of one example.
    // The work being measured: normalize, patch, quantize a window. The encoder is EXCLUDED on
    // purpose — it runs on the GPU, and folding a GPU-resident pass into a host-only meter's
    // boundary would produce a number whose boundary nobody could state.
    let n = cfg.patch_len * 4096;
    let raw: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin() * 2.0 + 3.3).collect();
    let q = Fsq::signal_15bit();
    let patcher = Patcher::contiguous(cfg.patch_len).unwrap();
    let rounds = 200u64;
    let run = || {
        let mut tokens = 0u64;
        for _ in 0..rounds {
            let rev = RevIn::fit(&raw, 1).unwrap();
            let norm = rev.apply(&raw).unwrap();
            let patches = patcher.patchify(&norm).unwrap();
            let t = patches.len() / cfg.patch_len;
            for i in 0..t {
                let sl = &patches[i * cfg.patch_len..(i + 1) * cfg.patch_len];
                let z: Vec<f32> = (0..cfg.latent_dim)
                    .map(|d| sl.iter().skip(d).step_by(cfg.latent_dim).sum::<f32>())
                    .collect();
                let _ = q.to_index(&q.quantize(&z).unwrap()).unwrap();
                tokens += 1;
            }
        }
        tokens
    };

    let rapl = Rapl::new().filter(|m| m.available());
    let batt = MacBattery::new().filter(|m| m.available());
    println!("  rapl (linux, cpu package) : {}", if rapl.is_some() { "available" } else { "unavailable" });
    println!("  macos battery discharge   : {}", if batt.is_some() { "available" } else { "unavailable" });

    let report = |tokens: u64, reading: Option<ferric_joule::Reading>| match reading {
        Some(r) => {
            println!("  {r}");
            println!("  {:.3} uJ per token over {tokens} tokens  [front end only, no encoder]",
                     r.per_task(tokens) * 1e6);
        }
        None => println!("  the meter stopped being readable mid-run; NOT MEASURED"),
    };

    if let Some(m) = rapl {
        let (tokens, r) = measure(&m, run);
        report(tokens, r);
    } else if let Some(m) = batt {
        let (tokens, r) = measure(&m, run);
        report(tokens, r);
    } else {
        let tokens = run();
        println!("  NOT MEASURED. No hardware energy counter is readable on this machine.");
        println!("  On macOS the battery meter reads discharge only, so a plugged-in machine");
        println!("  cannot be measured; that is a real restriction, not a missing feature.");
        println!("  {tokens} tokens were produced and deliberately left unpriced.");
        let (_t, r) = measure(&Nameplate::new(20.0), || {});
        if let Some(r) = r {
            println!("  For contrast, nameplate arithmetic would happily invent one, classed {}.",
                     r.class.label());
            println!("  That is what most published AI energy figures actually are.");
        }
    }
    println!();

    // ---- what a trainable embedding costs without a row gather ----
    //
    // Reported here rather than left as a remark in the docs, because it is the largest single
    // term in a caption training run and it is exactly computable. The configurations below are
    // the two this crate has actually trained.
    println!("TRAINABLE EMBEDDING, ONE-HOT AGAINST A NATIVE GATHER");
    println!("  `Var` has no row-gather backward, so a lookup is a one-hot [t, rows] times the");
    println!("  table. Correct, and this is what it moves per optimizer step:\n");
    println!("  {:<28} {:>6} {:>8} {:>12} {:>12} {:>9}",
             "run", "t", "rows", "one-hot", "gather", "ratio");
    println!("  {:-<28} {:->6} {:->8} {:->12} {:->12} {:->9}", "", "", "", "", "", "");
    for (name, t, rows, d) in [
        ("hydraulic, full codebook", 594usize, 32_788u32, 64usize),
        ("hydraulic, compacted", 594, 3_046, 64),
        ("rotating, compacted", 405, 7_883, 64),
    ] {
        let c = embed_cost(t, rows, d);
        println!("  {name:<28} {t:>6} {rows:>8} {:>9.2} MB {:>9.2} MB {:>8.0}x",
                 c.onehot_bytes() as f64 / 1e6,
                 c.gather_bytes() as f64 / 1e6,
                 c.traffic_ratio());
    }
    println!("\n  The ratio is rows / d_model exactly, so it grows with the VOCABULARY and not");
    println!("  with the sequence — which is why compacting the vocabulary to the codes a corpus");
    println!("  actually uses is a traffic result as much as a modelling one. The backward pass");
    println!("  touches the same matrix again.");
    println!();
}
