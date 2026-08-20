//! **MXFP4 dequant, diffed against llama.cpp's own.**
//!
//! MXFP4 (ggml type 39) is how GPT-OSS ships, so getting it wrong is not a quality question — the
//! weights simply are not the weights. A dequant is *exact-representable* (every output is a table
//! value times a power of two), so the bar here is not "small error", it is **bit-identical**.
//!
//! The reference is ggml itself: `ggml_get_type_traits(GGML_TYPE_MXFP4)->to_float`, the same function
//! `llama-eval-callback` prints through. `reference/mxfp4_ref.c` and `reference/mxfp4_grid.c` in this
//! crate produce its output against the installed `libggml-base` — they carry their own build lines
//! and the `llama-quantize` invocation that makes an MXFP4 file out of any gguf.
//!
//! ```text
//! cargo run -p ferric-gguf --example mxfp4_ref_diff -- <model.gguf> [tensor] [reference.f32]
//! cargo run -p ferric-gguf --example mxfp4_ref_diff -- --grid <grid.f32>
//! ```
//!
//! With no tensor name it prints the file's type composition instead, which is the check that says
//! whether a file advertised as MXFP4 actually contains any (a `MXFP4_MOE` requantize of a *dense*
//! model contains none at all — that is how this example's fixture was first produced by mistake).
//! Every path comes from argv: a hardcoded model path once hid a live divergence in this tree.

use ferric_gguf::GgufFile;

fn type_name(t: u32) -> &'static str {
    match t {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1", 8 => "Q8_0",
        12 => "Q4_K", 13 => "Q5_K", 14 => "Q6_K", 20 => "IQ4_NL", 23 => "IQ4_XS", 30 => "BF16",
        35 => "TQ2_0", 39 => "MXFP4", 41 => "Q1_0", 42 => "Q2_0", _ => "?",
    }
}

/// Compare the **whole** (E8M0 scale byte) x (E2M1 code) space — all 256 x 16 of it — against a
/// reference blob ggml wrote in the same order. The in-crate test pins ten exponents chosen for their
/// edges; this pins every one, which is what says the closed form is right rather than tuned to the
/// cases someone thought to look at.
fn grid_mode(refpath: &str) {
    let rb = std::fs::read(refpath).expect("read grid reference");
    assert_eq!(rb.len(), 256 * 16 * 4, "grid reference should be 256*16 f32");
    let want: Vec<f32> = rb.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let mut ndiff = 0usize;
    let mut first = None;
    for e in 0..256usize {
        for c in 0..16usize {
            let mut blk = [0u8; 17];
            blk[0] = e as u8;
            blk[1] = c as u8; // low nibble -> element 0
            let got = ferric_gguf::deq_raw(&blk, 32, 39).expect("deq")[0];
            let w = want[e * 16 + c];
            if got.to_bits() != w.to_bits() {
                ndiff += 1;
                if first.is_none() { first = Some((e, c, got, w)); }
            }
        }
    }
    println!("full E8M0 x E2M1 grid: {} / {} differ", ndiff, 256 * 16);
    match first {
        None => println!("BIT-IDENTICAL across all 4096 (scale, code) pairs"),
        Some((e, c, got, w)) => {
            println!("first difference at e={e} code={c}: ferric {got} (0x{:08x}) vs ggml {w} (0x{:08x})",
                got.to_bits(), w.to_bits());
            std::process::exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <model.gguf> [tensor] [reference.f32]", args[0]);
        eprintln!("       {} --grid <grid.f32>", args[0]);
        std::process::exit(2);
    }
    if args[1] == "--grid" {
        grid_mode(args.get(2).expect("--grid needs a reference blob"));
        return;
    }
    let g = GgufFile::open(&args[1]).expect("open gguf");

    if args.len() < 3 {
        let mut by_type: std::collections::BTreeMap<u32, (usize, u64)> = Default::default();
        for t in &g.tensors {
            let n: u64 = t.dims.iter().product();
            let e = by_type.entry(t.ggml_type).or_default();
            e.0 += 1;
            e.1 += n;
        }
        println!("composition of {}", args[1]);
        for (ty, (cnt, n)) in by_type {
            println!("  type {ty:3} {:8}  {cnt:4} tensors  {:>12} params", type_name(ty), n);
        }
        return;
    }

    let name = &args[2];
    let t = g.tensor(name).unwrap_or_else(|| panic!("no tensor '{name}'"));
    let n: usize = t.dims.iter().product::<u64>() as usize;
    let nbytes = ferric_gguf::type_size(t.ggml_type, n).expect("type_size");
    println!("tensor     : {name}");
    println!("ggml_type  : {} ({})", t.ggml_type, type_name(t.ggml_type));
    println!("dims       : {:?}  ({n} elements)", t.dims);
    println!("bytes      : {nbytes}  ({:.4} bytes/elem)", nbytes as f64 / n as f64);

    let got = g.dequant(name).expect("dequant");
    assert_eq!(got.len(), n, "dequant returned {} values for {n} elements", got.len());
    let sum: f64 = got.iter().map(|&v| v as f64).sum();
    let amax = got.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    println!("sum        : {sum:.10}");
    println!("amax       : {amax}");
    print!("first16    :");
    for v in got.iter().take(16) { print!(" {v}"); }
    println!();

    // Which of the 16 codes and which scale bytes a REAL file actually uses. This is why the
    // in-crate golden is not only real bytes: a quantizer emits code 0 for zero and never code 8
    // (−0), so the negative-zero slot — the one place ggml and HF transformers disagree — is
    // unreachable from any file and only a synthetic block can pin it.
    if t.ggml_type == 39 {
        let raw = g.raw(name).expect("raw");
        let (mut codes, mut scales) = ([0u64; 16], std::collections::BTreeSet::new());
        for blk in raw.chunks_exact(17) {
            scales.insert(blk[0]);
            for &b in &blk[1..17] { codes[(b & 0x0F) as usize] += 1; codes[(b >> 4) as usize] += 1; }
        }
        println!("code histogram (E2M1 index -> count):");
        for (c, n) in codes.iter().enumerate() {
            println!("   {c:2} {:>6} {:>12}{}", if c == 0 || c == 8 { "zero" } else { "" }, n,
                if *n == 0 { "   <- never occurs in this file" } else { "" });
        }
        println!("distinct E8M0 scale bytes: {} (min {:?}, max {:?})",
            scales.len(), scales.iter().next(), scales.iter().next_back());
    }

    let Some(refpath) = args.get(3) else {
        println!("\n(no reference file given — nothing was diffed)");
        return;
    };
    let rb = std::fs::read(refpath).expect("read reference");
    assert_eq!(rb.len(), n * 4, "reference has {} bytes, expected {}", rb.len(), n * 4);
    let want: Vec<f32> = rb.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();

    let (mut maxabs, mut at, mut nbitdiff) = (0.0f32, usize::MAX, 0usize);
    for i in 0..n {
        if got[i].to_bits() != want[i].to_bits() {
            nbitdiff += 1;
            let d = (got[i] - want[i]).abs();
            if d > maxabs || at == usize::MAX { maxabs = d; at = i; }
        }
    }
    println!("\nreference  : {refpath}");
    println!("bit-differing elements : {nbitdiff} / {n}");
    if nbitdiff == 0 {
        println!("max abs error          : 0 (BIT-IDENTICAL)");
    } else {
        let (blk, lane) = (at / 32, at % 32);
        println!("max abs error          : {maxabs:e} at element {at} (block {blk}, lane {lane})");
        println!("  ferric = {} (0x{:08x})", got[at], got[at].to_bits());
        println!("  ggml   = {} (0x{:08x})", want[at], want[at].to_bits());
        std::process::exit(1);
    }
}
