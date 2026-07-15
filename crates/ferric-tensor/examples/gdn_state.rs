//! The gated delta rule is a *recurrence*, so carrying its state across calls must be
//! indistinguishable from running the whole sequence at once. That property is what lets a model
//! decode one token at a time instead of re-running its entire prefix per token, and it's checkable
//! without any reference fixtures: split the sequence, carry the state, compare to the whole.
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;

fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.37 + s).sin())).collect() }
fn maxdiff(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let (t, h, dk, dv) = (12usize, 4usize, 16usize, 16usize);
    let q = Tensor::from_vec(&ctx, &seq(t * h * dk, 1.0), &[t, h, dk]);
    let k = Tensor::from_vec(&ctx, &seq(t * h * dk, 2.0), &[t, h, dk]);
    let v = Tensor::from_vec(&ctx, &seq(t * h * dv, 3.0), &[t, h, dv]);
    // gb = (g, beta): g<0 so exp(g) is a decay in (0,1); beta in (0,1)
    let gbv: Vec<f32> = (0..t * h).flat_map(|i| [-0.1 - 0.05 * ((i % 3) as f32), 0.3 + 0.1 * ((i % 4) as f32)]).collect();
    let gb = Tensor::from_vec(&ctx, &gbv, &[t, h, 2]);

    let whole = q.gated_delta_rule(&k, &v, &gb, h, dk, dv).to_vec().await;
    let mut ok = true;

    // Split at every boundary: prefix in one call, the rest resuming from the carried state.
    for cut in [1usize, 5, 11] {
        let nar = |x: &Tensor, a: usize, b: usize| x.narrow(0, a, b - a).contiguous();
        let (o1, st) = nar(&q, 0, cut).gated_delta_rule_stateful(&nar(&k, 0, cut), &nar(&v, 0, cut), &nar(&gb, 0, cut), h, dk, dv, None);
        let (o2, _) = nar(&q, cut, t).gated_delta_rule_stateful(&nar(&k, cut, t), &nar(&v, cut, t), &nar(&gb, cut, t), h, dk, dv, Some(&st));
        let mut got = o1.to_vec().await;
        got.extend(o2.to_vec().await);
        let e = maxdiff(&got, &whole);
        let p = e < 1e-5; ok &= p;
        println!("  {} split {cut}+{} == whole {t} · max|Δ| = {e:.1e}", if p { "✅" } else { "❌" }, t - cut);
    }

    // The real decode shape: prefill, then one token at a time.
    let nar = |x: &Tensor, a: usize, b: usize| x.narrow(0, a, b - a).contiguous();
    let pre = 4usize;
    let (o, mut st) = nar(&q, 0, pre).gated_delta_rule_stateful(&nar(&k, 0, pre), &nar(&v, 0, pre), &nar(&gb, 0, pre), h, dk, dv, None);
    let mut got = o.to_vec().await;
    for i in pre..t {
        let (o, s) = nar(&q, i, i + 1).gated_delta_rule_stateful(&nar(&k, i, i + 1), &nar(&v, i, i + 1), &nar(&gb, i, i + 1), h, dk, dv, Some(&st));
        got.extend(o.to_vec().await);
        st = s;
    }
    let e = maxdiff(&got, &whole);
    let p = e < 1e-5; ok &= p;
    println!("  {} prefill {pre} then {} single steps == whole · max|Δ| = {e:.1e}", if p { "✅" } else { "❌" }, t - pre);

    // Guard the default: no state in means start from zero, not from whatever was there.
    let again = q.gated_delta_rule(&k, &v, &gb, h, dk, dv).to_vec().await;
    let e0 = maxdiff(&again, &whole); ok &= e0 == 0.0;
    println!("  {} stateless call is repeatable (starts from zero) · max|Δ| = {e0:.1e}", if e0 == 0.0 { "✅" } else { "❌" });

    println!("{}", if ok { "✅ gated delta rule carries state exactly — incremental decode == full re-run" } else { "❌ state carry diverges" });
    assert!(ok);
}
