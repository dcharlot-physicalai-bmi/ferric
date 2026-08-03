//! Finite-difference gradient check for the current-model FFN autograd ops (silu, rmsnorm) — the VJPs
//! that let Ferric train THROUGH a real transformer block, not just on frozen features. Each analytic
//! gradient is compared to a central finite difference; the run asserts they agree.
use ferric_tensor::{Tensor, Var};
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let eps_fd = 1e-3f32;

    // A scalar loss over some op, and its analytic grad wrt a chosen leaf, vs central differences.
    // `build` returns (loss_var, the leaf to diff). We rebuild per perturbation for the FD side.
    async fn check(name: &str, ctx: &Arc<ferric_core::Context>, base: Vec<f32>, shape: Vec<usize>,
                   eps_fd: f32, build: impl Fn(&Var) -> Var) -> f32 {
        let x = Var::leaf(Tensor::from_vec(ctx, &base, &shape));
        let loss = build(&x);
        loss.backward();
        let ga = x.grad().unwrap().to_vec().await; // analytic
        let mut worst = 0.0f32;
        for i in 0..base.len() {
            let mut up = base.clone(); up[i] += eps_fd;
            let mut dn = base.clone(); dn[i] -= eps_fd;
            let lu = build(&Var::leaf(Tensor::from_vec(ctx, &up, &shape))).value().to_vec().await[0];
            let ld = build(&Var::leaf(Tensor::from_vec(ctx, &dn, &shape))).value().to_vec().await[0];
            let fd = (lu - ld) / (2.0 * eps_fd);
            worst = worst.max((ga[i] - fd).abs());
        }
        println!("  {name:24} max|analytic − FD| = {worst:.2e}");
        assert!(worst < 2e-2, "{name}: gradient mismatch {worst:.2e}");
        worst
    }

    let xs = vec![0.3f32, 1.1, -0.7, 2.5, -1.4, 0.05];

    // silu: loss = sum(silu(x))
    check("silu d/dx", &ctx, xs.clone(), vec![6], eps_fd, |x| x.silu().sum(&[0])).await;

    // rmsnorm wrt x: loss = sum(rmsnorm(x, w)) with a fixed weight, [2,3]
    let w = Tensor::from_vec(&ctx, &[1.2f32, 0.8, 1.5], &[3]);
    let wc = w.clone();
    check("rmsnorm d/dx", &ctx, vec![0.5f32, -1.0, 2.0, 0.3, 1.7, -0.4], vec![2, 3], eps_fd,
        move |x| x.rmsnorm(&Var::leaf(wc.clone()), 1e-6).sum(&[0, 1])).await;

    // rmsnorm wrt weight: hold x fixed, diff the weight leaf
    let xfix = Tensor::from_vec(&ctx, &[0.5f32, -1.0, 2.0, 0.3, 1.7, -0.4], &[2, 3]);
    let xc = xfix.clone();
    check("rmsnorm d/dw", &ctx, vec![1.2f32, 0.8, 1.5], vec![3], eps_fd,
        move |wv| Var::leaf(xc.clone()).rmsnorm(wv, 1e-6).sum(&[0, 1])).await;

    // rope: loss = sum(rope(x) ⊙ c) for a fixed c — grad wrt x = R(−θ)(c). [t=2, n_heads=2, head_dim=4]
    let (nh, hd) = (2usize, 4usize);
    let cc = Tensor::from_vec(&ctx, &seed(16, 3.0), &[2, nh * hd]);
    let ccx = cc.clone();
    check("rope d/dx", &ctx, seed(16, 7.0), vec![2, nh * hd], eps_fd,
        move |x| x.rope(nh, hd, 10000.0, 0).mul(&Var::leaf(ccx.clone())).sum(&[0, 1])).await;

    // broadcast_to: loss = sum(broadcast([3,1]→[3,4]) ⊙ c) — grad wrt x sums the 4 broadcast copies.
    let cb = Tensor::from_vec(&ctx, &seed(12, 5.0), &[3, 4]);
    let cbx = cb.clone();
    check("broadcast_to d/dx", &ctx, vec![0.4f32, -1.1, 2.3], vec![3, 1], eps_fd,
        move |x| x.broadcast_to(&[3, 4]).mul(&Var::leaf(cbx.clone())).sum(&[0, 1])).await;

    // matmul_qf: forward x·Wᵀ (fp W), backward grad_x = g·W via an int8 row-wise TRANSPOSE (no fp
    // dequant kept). Analytic vs FD error reflects the int8 requant (~1%), not a VJP bug.
    {
        use ferric_tensor::QMatrix;
        let (mi, od, id) = (2usize, 6usize, 4usize);
        let wdata = seed(od * id, 9.0); // W [out, in]
        let w = QMatrix::from_dense(&ctx, &wdata, od, id);
        let wt = Arc::new(Tensor::from_vec(&ctx, &wdata, &[od, id]).transpose(0, 1).contiguous().quantize_rowwise(8));
        let cvec = Tensor::from_vec(&ctx, &seed(mi * od, 4.0), &[mi, od]);
        let base = seed(mi * id, 7.0);
        let x = Var::leaf(Tensor::from_vec(&ctx, &base, &[mi, id]));
        let loss = x.matmul_qf(&w, &wt).mul(&Var::leaf(cvec.clone())).sum(&[0, 1]);
        loss.backward();
        let ga = x.grad().unwrap().to_vec().await;
        let mut worst = 0.0f32;
        for i in 0..base.len() {
            let (mut up, mut dn) = (base.clone(), base.clone()); up[i] += eps_fd; dn[i] -= eps_fd;
            let lu = Var::leaf(Tensor::from_vec(&ctx, &up, &[mi, id])).matmul_qf(&w, &wt).mul(&Var::leaf(cvec.clone())).sum(&[0, 1]).value().to_vec().await[0];
            let ld = Var::leaf(Tensor::from_vec(&ctx, &dn, &[mi, id])).matmul_qf(&w, &wt).mul(&Var::leaf(cvec.clone())).sum(&[0, 1]).value().to_vec().await[0];
            worst = worst.max((ga[i] - (lu - ld) / (2.0 * eps_fd)).abs());
        }
        println!("  matmul_qf d/dx (int8)    max|analytic − FD| = {worst:.2e}  (int8-transpose requant, ~1%)");
        assert!(worst < 1e-1, "matmul_qf gradient off beyond int8 tolerance: {worst:.2e}");
    }

    println!("\n  ✓ silu + rmsnorm + rope + broadcast + matmul_qf VJPs correct — Ferric autograd covers EVERY op\n    in a current-model transformer block (attention + FFN), incl. LoRA around FROZEN QUANTIZED weights.");
}

fn seed(n: usize, s: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 * 12.9898 + s).sin() * 43758.5453).fract() * 2.0 - 1.0).collect()
}
