// Validates half-precision storage + on-device dequant against the `half` crate: f32→GPU-pack→GPU-
// dequant must match the CPU half round-trip exactly, and raw fp16/bf16 bits (as from a safetensors
// file) uploaded to the GPU and dequantized on-device must match too. Shows the memory win.
use ferric_core::Context;
use ferric_tensor::{DType, Half, Tensor};
use half::{bf16, f16};
use std::sync::Arc;

fn maxdiff(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) }
fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.31 + s).sin()) * 3.0).collect() }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let mut ok = true;
    let mut check = |name: &str, g: &[f32], c: &[f32]| {
        let d = maxdiff(g, c); let pass = d < 1e-6; ok &= pass;
        println!("  {} {:<40} max|gpu-cpu| = {:.2e}", if pass { "✅" } else { "❌" }, name, d);
    };

    let x = seq(101, 1.0); // odd length → exercises the packing tail
    let shape = [101];

    // f32 -> GPU f16 pack -> GPU dequant  vs  CPU half::f16 round-trip
    let hf16: Half = Tensor::from_vec(&ctx, &x, &shape).to_half(DType::F16);
    let ref16: Vec<f32> = x.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    check("f16  pack→dequant vs half crate", &hf16.dequant().to_vec().await, &ref16);

    // f32 -> GPU bf16 pack -> GPU dequant  vs  CPU half::bf16 round-trip
    let hbf: Half = Tensor::from_vec(&ctx, &x, &shape).to_half(DType::BF16);
    let refbf: Vec<f32> = x.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();
    check("bf16 pack→dequant vs half crate", &hbf.dequant().to_vec().await, &refbf);

    // raw fp16 bits (as a safetensors slice) -> GPU -> dequant  vs  CPU
    let bits16: Vec<u16> = x.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    let loaded = Half::from_bits(&ctx, &bits16, &shape, DType::F16).dequant();
    check("f16  from_bits (safetensors path)", &loaded.to_vec().await, &ref16);

    // int8 quantized matmul vs f32 matmul (within quantization error)
    let (m, k, n) = (8usize, 16usize, 6usize);
    let am = seq(m * k, 2.0); let bm = seq(k * n, 3.0);
    let ta = Tensor::from_vec(&ctx, &am, &[m, k]);
    let tb = Tensor::from_vec(&ctx, &bm, &[k, n]);
    let (qa, qb) = (ta.quantize_i8().await, tb.quantize_i8().await);
    let q = qa.matmul(&qb).to_vec().await;
    let f = ta.matmul(&tb).to_vec().await;
    let rel = q.iter().zip(&f).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / f.iter().map(|v| v.abs()).fold(0.0, f32::max);
    let qok = rel < 0.03; ok &= qok;
    println!("  {} int8 quantized matmul vs f32          rel err = {:.2e}", if qok { "✅" } else { "❌" }, rel);

    // per-row int8 / int4 quantization (per-row scales; int4 = 1/8 the memory)
    let (mr, mc) = (12usize, 20usize);
    let w = seq(mr * mc, 42.0);
    let tw = Tensor::from_vec(&ctx, &w, &[mr, mc]);
    let denom = w.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    for bits in [8u32, 4] {
        let q = tw.quantize_rowwise(bits);
        let deq = q.dequant().to_vec().await;
        let rel = deq.iter().zip(&w).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / denom;
        let tol = if bits == 8 { 0.02 } else { 0.2 };
        let pass = rel < tol; ok &= pass;
        println!("  {} int{bits} per-row quant round-trip        rel err = {:.2e}  ({} B → {} B)", if pass { "✅" } else { "❌" }, rel, mr * mc * 4, q.nbytes());
    }

    // weight-only quantized matmul (W stays int4/int8 in memory, x is f32) vs full f32
    let (rows, inn, outf) = (8usize, 20usize, 10usize);
    let xa = Tensor::from_vec(&ctx, &seq(rows * inn, 50.0), &[rows, inn]);
    let wf = Tensor::from_vec(&ctx, &seq(outf * inn, 51.0), &[outf, inn]); // [out,in] HF layout
    let f_ref = xa.matmul(&wf.transpose(0, 1)).to_vec().await; // x·Wᵀ in f32
    let fden = f_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    for bits in [8u32, 4] {
        let qw = wf.quantize_rowwise(bits);
        let y = xa.matmul_qweight(&qw).to_vec().await;
        let rel = y.iter().zip(&f_ref).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max) / fden;
        let pass = rel < if bits == 8 { 0.02 } else { 0.15 }; ok &= pass;
        println!("  {} int{bits} weight-only matmul (W4A16-style)   rel err = {:.2e}  (W: {} B → {} B)", if pass { "✅" } else { "❌" }, rel, outf * inn * 4, qw.nbytes());
    }

    println!("  memory: {} f32 bytes → {} half bytes ({}% )", x.len() * 4, hf16.nbytes(), hf16.nbytes() * 100 / (x.len() * 4));
    println!("{}", if ok { "✅ Half-precision storage + on-device dequant is exact — real fp16/bf16 weights can live on the GPU" } else { "❌ dtype mismatch" });
    assert!(ok);
}
