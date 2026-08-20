//! Reconstruction error of every KV-quantization scheme, on **real captured K and V**.
//!
//! Synthetic gaussians cannot answer the question this example exists for. K and V come out of a real
//! forward with different distributions, and K carries outlier channels — which is precisely what
//! breaks a naive per-tensor or per-token scale. So the input is a capture:
//!
//!   cargo run -p ferric-llama --example kv_capture --release -- <model.gguf> <out.fkvc>
//!   cargo run -p ferric-tensor --example kv_quant_error --release -- <out.fkvc> [more.fkvc ...]
//!
//! Reports, per model and split by K vs V:
//!   1. the distribution evidence (per-channel outlier ratio, kurtosis) that motivates the question
//!   2. relative RMSE for five scaling granularities x {8-bit, 4-bit}, each labelled with what it
//!      costs to APPEND one token — because a scheme that must requantize the cache per token is not
//!      a KV-cache scheme however good its error is
//!   3. the shipped block formats (q8_0 / q4_0 / q4_1) as they actually store, f16 scales included
//!   4. GPU kernel vs CPU reference agreement on this same real data
//!   5. bytes per token per layer, against f32
use ferric_tensor::kvquant::{
    append_cost, reference, roundtrip_err, shipped_err, AppendCost, GranKind, KvqFmt, QKvCache,
};
use ferric_tensor::Tensor;
use std::sync::Arc;

struct Capture {
    name: String,
    n_layer: usize,
    rows: usize,
    width: usize,
    n_kv_head: usize,
    head_dim: usize,
    /// `[layer][0]` = K, `[layer][1]` = V, each `rows*width` f32 row-major.
    kv: Vec<[Vec<f32>; 2]>,
}

fn load(path: &str) -> Capture {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert_eq!(&b[0..4], b"FKVC", "{path} is not a capture file");
    let u = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]) as usize;
    assert_eq!(u(4), 1, "unknown capture version");
    let (n_layer, rows, width, n_kv_head, head_dim) = (u(8), u(12), u(16), u(20), u(24));
    let mut off = 28;
    let n = rows * width;
    let mut kv = Vec::with_capacity(n_layer);
    for _ in 0..n_layer {
        let mut pair = [Vec::new(), Vec::new()];
        for slot in pair.iter_mut() {
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let p = off + i * 4;
                v.push(f32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]]));
            }
            off += n * 4;
            *slot = v;
        }
        kv.push(pair);
    }
    assert_eq!(off, b.len(), "{path}: trailing bytes");
    Capture {
        name: std::path::Path::new(path).file_stem().unwrap().to_string_lossy().into_owned(),
        n_layer, rows, width, n_kv_head, head_dim, kv,
    }
}

/// Excess kurtosis — how heavy the tail is. Gaussian is 0.
fn kurtosis(x: &[f32]) -> f32 {
    let n = x.len() as f64;
    let m = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mut m2 = 0.0;
    let mut m4 = 0.0;
    for &v in x {
        let d = v as f64 - m;
        m2 += d * d;
        m4 += d * d * d * d;
    }
    ((m4 / n) / (m2 / n).powi(2) - 3.0) as f32
}

/// The concrete "outlier channel" measurement: per-channel amax, then the ratio of the largest
/// channel to the median channel. A tensor whose channels all have similar range gives ~1-2; a
/// tensor with a few dominant channels gives a large number, and that is exactly the case a
/// per-token scale handles badly.
fn channel_outlier_ratio(x: &[f32], rows: usize, width: usize) -> (f32, f32) {
    let mut amax = vec![0f32; width];
    for r in 0..rows {
        for c in 0..width {
            amax[c] = amax[c].max(x[r * width + c].abs());
        }
    }
    let mut s = amax.clone();
    s.sort_by(|a, b| a.total_cmp(b));
    let med = s[width / 2];
    let top = s[width - 1];
    // Also the row-wise version: within a row, how far does the max channel sit above the median?
    let mut row_ratio = 0f32;
    for r in 0..rows {
        let mut v: Vec<f32> = (0..width).map(|c| x[r * width + c].abs()).collect();
        v.sort_by(|a, b| a.total_cmp(b));
        let m = v[width / 2];
        if m > 0.0 { row_ratio += v[width - 1] / m; }
    }
    (if med > 0.0 { top / med } else { f32::INFINITY }, row_ratio / rows as f32)
}

fn cost_tag(c: AppendCost) -> &'static str {
    match c {
        AppendCost::InPlace => "in-place",
        AppendCost::GroupFlush(_) => "group-flush",
        AppendCost::FullRequant => "FULL REQUANT",
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: kv_quant_error <capture.fkvc> [more.fkvc ...]");
        eprintln!("  produce a capture with: cargo run -p ferric-llama --example kv_capture --release -- <model.gguf> <out.fkvc>");
        std::process::exit(2);
    }
    let ctx = match pollster::block_on(ferric_core::Context::new()) {
        Ok(c) => Some(Arc::new(c)),
        Err(e) => {
            eprintln!("NOTE: no GPU context ({e:?}). The CPU study still runs; the GPU-vs-CPU kernel check does NOT.");
            None
        }
    };

    for path in &args {
        let cap = load(path);
        println!("\n================================================================================");
        println!(
            "{} · {} layers · {} kv-heads x {} head-dim = width {} · {} tokens",
            cap.name, cap.n_layer, cap.n_kv_head, cap.head_dim, cap.width, cap.rows
        );
        println!("================================================================================");

        // ---- 1. why the question is a question ------------------------------------------------
        println!("\n-- distribution of the captured tensors (per layer) --");
        println!("            {:>10} {:>10} {:>10} {:>10}", "kurtosis", "chan-max/", "row-max/", "amax");
        println!("            {:>10} {:>10} {:>10} {:>10}", "(excess)", "chan-med", "row-med", "");
        for il in [0usize, cap.n_layer / 2, cap.n_layer - 1] {
            for (si, tag) in ["K", "V"].iter().enumerate() {
                let x = &cap.kv[il][si];
                let (cr, rr) = channel_outlier_ratio(x, cap.rows, cap.width);
                let amax = x.iter().fold(0f32, |a, &v| a.max(v.abs()));
                println!("  L{il:<3} {tag}   {:>10.2} {:>10.2} {:>10.2} {:>10.3}", kurtosis(x), cr, rr, amax);
            }
        }

        // ---- 2. the granularity ladder --------------------------------------------------------
        // Averaged over all layers, because a single layer is not the claim.
        let grans: [(GranKind, &str); 5] = [
            (GranKind::Tensor, "per-tensor"),
            (GranKind::PerToken, "per-token"),
            (GranKind::PerBlock(32), "per-block(32)"),
            (GranKind::PerChannelGroup(32), "per-chan x 32tok"),
            (GranKind::PerChannel, "per-channel"),
        ];
        for bits in [8u32, 4] {
            println!("\n-- {bits}-bit, relative RMSE (rmse / rms(x)), mean over {} layers, f32 scales --", cap.n_layer);
            println!(
                "  {:<18} {:>6} {:>12} {:>12} {:>12} {:>12}",
                "granularity", "b/val", "K sym", "K asym", "V sym", "V asym"
            );
            for (g, label) in grans {
                let mut acc = [0f64; 4];
                let mut bpv = 0f32;
                for il in 0..cap.n_layer {
                    for (si, base) in [(0usize, 0usize), (1usize, 2usize)] {
                        let x = &cap.kv[il][si];
                        let s = roundtrip_err(x, cap.rows, cap.width, g, bits, false);
                        let a = roundtrip_err(x, cap.rows, cap.width, g, bits, true);
                        acc[base] += s.rel_rmse as f64;
                        acc[base + 1] += a.rel_rmse as f64;
                        bpv = s.bits_per_value;
                    }
                }
                let n = cap.n_layer as f64;
                println!(
                    "  {:<18} {:>6.2} {:>12.5} {:>12.5} {:>12.5} {:>12.5}   [{}]",
                    label,
                    bpv,
                    acc[0] / n, acc[1] / n, acc[2] / n, acc[3] / n,
                    cost_tag(append_cost(g))
                );
            }
        }

        // ---- 3. the formats that actually ship -------------------------------------------------
        println!("\n-- shipped block formats, exactly as stored (f16 scales, per-block(32) along the row) --");
        println!(
            "  {:<8} {:>6} {:>10} {:>12} {:>12} {:>12} {:>12}",
            "format", "b/val", "vs f32", "K rel-rmse", "K cos", "V rel-rmse", "V cos"
        );
        for fmt in KvqFmt::ALL {
            let mut acc = [0f64; 4];
            for il in 0..cap.n_layer {
                let k = shipped_err(&cap.kv[il][0], cap.rows, cap.width, fmt);
                let v = shipped_err(&cap.kv[il][1], cap.rows, cap.width, fmt);
                acc[0] += k.rel_rmse as f64;
                acc[1] += 1.0 - k.cos as f64;
                acc[2] += v.rel_rmse as f64;
                acc[3] += 1.0 - v.cos as f64;
            }
            let n = cap.n_layer as f64;
            println!(
                "  {:<8} {:>6.2} {:>9.2}x {:>12.5} {:>12.2e} {:>12.5} {:>12.2e}",
                fmt.name(),
                fmt.bits_per_value(),
                32.0 / fmt.bits_per_value(),
                acc[0] / n,
                acc[1] / n,
                acc[2] / n,
                acc[3] / n
            );
        }
        println!("  (cos columns are 1 - cosine similarity: attention is a dot product, so the angle is what degrades)");

        // ---- 4. GPU kernels vs the CPU reference on this same real data ------------------------
        if let Some(ctx) = &ctx {
            println!("\n-- GPU kernel vs CPU reference, on the captured tensors --");
            for fmt in KvqFmt::ALL {
                let mut code_words = 0usize;
                let mut code_diff = 0usize;
                let mut deq_bit_diff = 0usize;
                let mut deq_max = 0f32;
                for il in 0..cap.n_layer {
                    for si in 0..2 {
                        let x = &cap.kv[il][si];
                        let t = Tensor::from_vec(ctx, x, &[cap.rows, cap.width]);
                        let q = QKvCache::from_tensor(ctx, &t, fmt);
                        let (gc, gs) = pollster::block_on(q.to_host(ctx));
                        let (rc, rs) = reference::quantize(x, cap.rows, cap.width, fmt);
                        code_words += gc.len() + gs.len();
                        code_diff += gc.iter().zip(&rc).filter(|(a, b)| a != b).count();
                        code_diff += gs.iter().zip(&rs).filter(|(a, b)| a != b).count();
                        let gd = pollster::block_on(q.dequantize(ctx).to_vec());
                        let cd = reference::dequantize_rows(&rc, &rs, 0, cap.rows, cap.width, fmt);
                        for i in 0..gd.len() {
                            if gd[i].to_bits() != cd[i].to_bits() {
                                deq_bit_diff += 1;
                                deq_max = deq_max.max((gd[i] - cd[i]).abs());
                            }
                        }
                    }
                }
                println!(
                    "  {:<8} packed words differing: {}/{}   dequantized values differing: {} (max |delta| {:.3e})",
                    fmt.name(), code_diff, code_words, deq_bit_diff, deq_max
                );
            }
        }

        // ---- 5. what it costs ------------------------------------------------------------------
        println!("\n-- cache cost, per token, all {} layers, K and V --", cap.n_layer);
        let per_tok_f32 = cap.n_layer * 2 * cap.width * 4;
        println!("  f32 (today)   {:>8} B/token   {:>8.2} MB @ 8k ctx", per_tok_f32, (per_tok_f32 * 8192) as f64 / 1e6);
        for fmt in KvqFmt::ALL {
            let b = (cap.n_layer * 2 * cap.width) as f32 * fmt.bits_per_value() / 8.0;
            println!(
                "  {:<8}      {:>8.0} B/token   {:>8.2} MB @ 8k ctx   ({:.2}x more context per GB)",
                fmt.name(), b, (b as f64 * 8192.0) / 1e6, per_tok_f32 as f32 / b
            );
        }
    }
}
