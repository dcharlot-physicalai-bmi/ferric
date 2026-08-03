//! Finite-difference gradcheck for the new Var ops used by the Instella differentiable forward:
//! narrow, cat, apply_rope_costable. Loss = Σ op(x)²; analytic grad (backward) vs central-difference.
//!   cargo run -p ferric-tensor --example gradcheck_ops --release
use ferric_tensor::{Tensor, Var};
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let sq_sum = |v: &[f32]| v.iter().map(|a| a * a).sum::<f32>();

    // generic checker: `build` maps an input Var → output Var; loss = Σ out². Compares analytic vs numeric grad on x.
    async fn check(name: &str, ctx: &Arc<ferric_core::Context>, x0: &[f32], shape: &[usize], build: impl Fn(&Var) -> Var) {
        let x = Tensor::from_vec(ctx, x0, shape);
        let xv = Var::leaf(x.clone());
        let y = build(&xv);
        let loss = y.mul(&y).sum_all();
        loss.backward();
        let ga = xv.grad().unwrap().to_vec().await; // analytic
        let eps = 1e-3f32;
        let mut maxrel = 0f32;
        let sq = |v: &[f32]| v.iter().map(|a| a * a).sum::<f32>();
        for i in 0..x0.len() {
            let (mut xp, mut xm) = (x0.to_vec(), x0.to_vec());
            xp[i] += eps; xm[i] -= eps;
            let fp = sq(&build(&Var::leaf(Tensor::from_vec(ctx, &xp, shape))).value().to_vec().await);
            let fm = sq(&build(&Var::leaf(Tensor::from_vec(ctx, &xm, shape))).value().to_vec().await);
            let gn = (fp - fm) / (2.0 * eps);
            let denom = gn.abs().max(ga[i].abs()).max(1e-3);
            maxrel = maxrel.max((gn - ga[i]).abs() / denom);
        }
        println!("  {name:28} max rel err {maxrel:.2e}   {}", if maxrel < 2e-2 { "✓" } else { "✗ FAIL" });
        // ASSERT, don't just print. Printing "✗ FAIL" and exiting 0 means a broken VJP passes CI silently —
        // which is exactly the state `selective_scan` was in while being described as "gradchecked".
        assert!(maxrel < 2e-2, "{name}: gradient check FAILED (max rel err {maxrel:.2e} >= 2e-2)");
    }

    println!("gradcheck new Var ops (narrow, cat, apply_rope_costable):");
    let x12: Vec<f32> = (0..24).map(|i| ((i as f32 * 0.7).sin())).collect();

    // narrow: [4,6] → narrow(dim1, start2, len3)
    check("narrow(1,2,3)", &ctx, &x12, &[4, 6], |xv| xv.narrow(1, 2, 3)).await;
    // narrow along dim0 (rows) — like per-token MoE
    check("narrow(0,1,2)", &ctx, &x12, &[4, 6], |xv| xv.narrow(0, 1, 2)).await;
    // cat: split x into two halves along dim1 then re-cat (identity-ish, exercises cat VJP on both inputs)
    check("cat(halves,dim1)", &ctx, &x12, &[4, 6], |xv| xv.narrow(1, 0, 3).cat(&xv.narrow(1, 3, 3), 1)).await;
    // cat two different slices
    check("cat(slices,dim1)", &ctx, &x12, &[4, 6], |xv| xv.narrow(1, 1, 2).cat(&xv.narrow(1, 3, 2), 1)).await;

    // apply_rope_costable: x [s=2, heads=2, hd=4] flattened [2, 8]; cos/sin doubled tables [2,4]
    let s = 2; let heads = 2; let hd = 4;
    let xr: Vec<f32> = (0..s * heads * hd).map(|i| ((i as f32 * 0.37 + 0.1).cos())).collect();
    // doubled layout: cos[t] = [c0,c1,c0,c1], sin[t] = [s0,s1,s0,s1]
    let mut cosv = vec![0f32; s * hd]; let mut sinv = vec![0f32; s * hd];
    for t in 0..s { for j in 0..hd / 2 { let ang = 0.3 * (t as f32 + 1.0) * (j as f32 + 1.0);
        cosv[t * hd + j] = ang.cos(); cosv[t * hd + hd / 2 + j] = ang.cos();
        sinv[t * hd + j] = ang.sin(); sinv[t * hd + hd / 2 + j] = ang.sin(); } }
    let cos = Tensor::from_vec(&ctx, &cosv, &[s, hd]);
    let sin = Tensor::from_vec(&ctx, &sinv, &[s, hd]);
    let (c2, s2) = (cos.clone(), sin.clone());
    check("apply_rope_costable", &ctx, &xr, &[s, heads * hd], move |xv| xv.apply_rope_costable(&c2, &s2, heads, hd)).await;

    // selective_scan (SSM recurrence h_t = a_t⊙h_{t-1} + b_t): check grad wrt BOTH a and b.
    // Scan is the super-learner's backbone — its analytic reverse-time VJP must be exact.
    let (ts, ds) = (5usize, 3usize);
    let av: Vec<f32> = (0..ts * ds).map(|i| 0.5 + 0.3 * ((i as f32 * 0.9).sin())).collect(); // decays in (0.2,0.8)
    let bv: Vec<f32> = (0..ts * ds).map(|i| ((i as f32 * 0.6 + 0.2).cos()) * 0.7).collect();
    let b_fixed = Tensor::from_vec(&ctx, &bv, &[ts, ds]);
    let a_fixed = Tensor::from_vec(&ctx, &av, &[ts, ds]);
    {   // d/da
        let bf = b_fixed.clone();
        check("selective_scan d/da", &ctx, &av, &[ts, ds], move |x| Var::selective_scan(x, &Var::leaf(bf.clone()))).await;
    }
    {   // d/db
        let af = a_fixed.clone();
        check("selective_scan d/db", &ctx, &bv, &[ts, ds], move |x| Var::selective_scan(&Var::leaf(af.clone()), x)).await;
    }

    let _ = sq_sum;
    println!("\n(all ✓ ⇒ the MLA/MoE + SSM-scan Var ops are safe to build the differentiable forwards on)");
}
