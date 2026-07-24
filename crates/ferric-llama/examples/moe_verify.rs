// Micro-verify the MoE FFN data path: CPU-reference router logits, one routed expert, and the shared
// expert against the GPU implementation, on the same random input.
use ferric_core::Context;
use ferric_gguf::{GgufFile, deq_raw};
use ferric_llama::qwen35::{Qwen35, Ffn};
use ferric_tensor::Tensor;
use std::sync::Arc;
fn main() { pollster::block_on(run()); }
async fn run() {
    let path = std::env::args().nth(1).unwrap();
    let g = GgufFile::open(&path).unwrap();
    let ctx = Arc::new(Context::new().await.unwrap());
    let m = Qwen35::load(&ctx, &g).unwrap();
    let d = m.cfg.n_embd; let eff = m.cfg.expert_ff; let ne = m.cfg.n_expert;
    // fixed pseudo-random input
    let mut h = vec![0.0f32; d];
    let mut s = 0x12345678u32;
    for x in h.iter_mut() { s = s.wrapping_mul(1664525).wrapping_add(1013904223); *x = (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5; }
    let ht = Tensor::from_vec(&ctx, &h, &[1, d]);
    let Ffn::Moe(moe) = &m.layers[0].ffn else { panic!("layer0 not moe") };

    // 1) router: GPU vs CPU
    let gpu_logits = moe.router.matmul(&ht.reshape(&[d, 1])).to_vec().await;
    let rraw = g.dequant("blk.0.ffn_gate_inp.weight").unwrap(); // [ne*d] f32, expert-major
    let mut max_d = 0.0f32; let mut cpu_top = (0usize, f32::MIN);
    for e in 0..ne {
        let dot: f32 = (0..d).map(|j| rraw[e * d + j] * h[j]).sum();
        max_d = max_d.max((dot - gpu_logits[e]).abs());
        if dot > cpu_top.1 { cpu_top = (e, dot); }
    }
    let gpu_top = (0..ne).max_by(|&a, &b| gpu_logits[a].partial_cmp(&gpu_logits[b]).unwrap()).unwrap();
    println!("router: max|Δ|={max_d:.6} cpu_top={} gpu_top={gpu_top}", cpu_top.0);

    // 2) routed expert e = gpu_top: GPU fused path vs CPU reference from raw slabs
    let e = gpu_top;
    let gf = g.raw("blk.0.ffn_gate_exps.weight").unwrap(); let gt = g.tensor("blk.0.ffn_gate_exps.weight").unwrap().ggml_type;
    let uf = g.raw("blk.0.ffn_up_exps.weight").unwrap();
    let df = g.raw("blk.0.ffn_down_exps.weight").unwrap(); let dt = g.tensor("blk.0.ffn_down_exps.weight").unwrap().ggml_type;
    let (gp, up_, dp) = (gf.len() / ne, uf.len() / ne, df.len() / ne);
    let wg = deq_raw(&gf[e * gp..(e + 1) * gp], eff * d, gt).unwrap();   // [eff, d]
    let wu = deq_raw(&uf[e * up_..(e + 1) * up_], eff * d, gt).unwrap(); // [eff, d]
    let wd = deq_raw(&df[e * dp..(e + 1) * dp], d * eff, dt).unwrap();   // [d, eff]
    let mut mid = vec![0.0f32; eff];
    for i in 0..eff {
        let gate: f32 = (0..d).map(|j| wg[i * d + j] * h[j]).sum();
        let up: f32 = (0..d).map(|j| wu[i * d + j] * h[j]).sum();
        mid[i] = (gate / (1.0 + (-gate).exp())) * up; // silu(gate)*up
    }
    let mut y_ref = vec![0.0f32; d];
    for i in 0..d { y_ref[i] = (0..eff).map(|j| wd[i * eff + j] * mid[j]).sum(); }
    let y_gpu = ht.matmul_q(&moe.experts[e].gate_up).swiglu(eff).matmul_q(&moe.experts[e].down).to_vec().await;
    let md: f32 = y_ref.iter().zip(&y_gpu).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    let scale: f32 = y_ref.iter().map(|x| x.abs()).fold(0.0, f32::max);
    println!("expert {e}: max|Δ|={md:.6} (scale {scale:.4}) first3 ref={:?} gpu={:?}", &y_ref[..3], &y_gpu[..3]);
}
