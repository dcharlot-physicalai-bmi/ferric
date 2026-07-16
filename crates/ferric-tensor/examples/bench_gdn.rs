//! Cost of the gated-delta-rule kernel at Bonsai-27B's real shape, at decode width (T=1).
//! 48 of 64 layers are GDN, so this runs 48x per token.
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let (h, dk, dv) = (48usize, 128usize, 128usize);
    for t in [1usize, 5] {
        let q = Tensor::from_vec(&ctx, &vec![0.01f32; t * h * dk], &[t, h, dk]);
        let k = Tensor::from_vec(&ctx, &vec![0.02f32; t * h * dk], &[t, h, dk]);
        let v = Tensor::from_vec(&ctx, &vec![0.03f32; t * h * dv], &[t, h, dv]);
        let gb = Tensor::from_vec(&ctx, &vec![-0.1f32; t * h * 2], &[t, h, 2]);
        let st = Tensor::zeros(&ctx, &[h, dv, dk]);
        let _ = q.gated_delta_rule_stateful(&k, &v, &gb, h, dk, dv, Some(&st)).0.to_vec().await;
        let reps = 20;
        let t0 = std::time::Instant::now();
        let mut last = None;
        for _ in 0..reps { last = Some(q.gated_delta_rule_stateful(&k, &v, &gb, h, dk, dv, Some(&st)).0); }
        let _ = last.unwrap().to_vec().await;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
        println!("  T={t}: {ms:.3} ms/call → x48 GDN layers = {:.1} ms/token", ms * 48.0);
    }
}
