//! Flash-attention prefill vs the composed causal_attention (which materializes [nh,T,T]).
use ferric_core::Context;
use ferric_tensor::{nn, Tensor};
use std::sync::Arc; use std::time::Instant;
fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.02 + s).sin()) * 0.2).collect() }
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let mut ok = true;
    // T=3000 and 5000 cross the 2048-key chunk boundary — validates the online-softmax combination.
    for (t, nh, nkv, dh) in [(64usize,16usize,8usize,128usize),(512,16,8,128),(1024,16,16,64),(3000,8,8,64),(5000,4,4,64)] {
        let d = nh*dh; let kd = nkv*dh;
        let q = Tensor::from_vec(&ctx, &seq(t*d, 1.0), &[t, d]);
        let k = Tensor::from_vec(&ctx, &seq(t*kd, 2.0), &[t, kd]);
        let v = Tensor::from_vec(&ctx, &seq(t*kd, 3.0), &[t, kd]);
        let flash = q.flash_attention_prefill(&k, &v, nh, nkv, dh).to_vec().await;
        let comp = nn::causal_attention(&q, &k, &v, nh, nkv, 0.0).to_vec().await;
        let e = flash.iter().zip(&comp).map(|(a,b)|(a-b).abs()).fold(0f32,f32::max);
        let p = e < 1e-4; ok &= p;
        let bench = |f: &dyn Fn()->Tensor| { let mut l=None; let t0=Instant::now(); for _ in 0..20 { l=Some(f()); } let _=pollster::block_on(l.unwrap().to_vec()); t0.elapsed().as_secs_f64()/20.0 };
        let ft = bench(&|| q.flash_attention_prefill(&k,&v,nh,nkv,dh));
        let ct = bench(&|| { let x = nn::causal_attention(&q,&k,&v,nh,nkv, 0.0); x });
        let scores_mb = (nh*t*t*4) as f64/1e6;
        println!("{} T={t:<4} nh={nh}: flash {:.2}ms  composed {:.2}ms ({:.1}×, saves {:.0}MB scores)  max|Δ|={e:.1e}",
            if p {"✅"} else {"❌"}, ft*1e3, ct*1e3, ct/ft, scores_mb);
    }
    println!("{}", if ok {"✅ flash prefill == causal_attention, O(T) memory (no [nh,T,T] materialization)"} else {"❌"});
    assert!(ok);
}
