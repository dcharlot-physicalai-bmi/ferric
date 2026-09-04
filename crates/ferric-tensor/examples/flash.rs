//! Flash-attention prefill vs the composed causal_attention (which materializes [nh,T,T]).
use ferric_core::Context;
use ferric_tensor::{nn, Tensor};
use std::sync::Arc; use std::time::Instant;
fn seq(n: usize, s: f32) -> Vec<f32> { (0..n).map(|i| ((i as f32 * 0.02 + s).sin()) * 0.2).collect() }
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    // ⭐ TEST HOOK, and it is not optional discipline: the constrained branch below is taken only on
    // devices whose `max_storage_buffer_binding_size` is small (lavapipe and the WebGPU baseline
    // report 128 MiB; Metal reports far more). Without a way to force it, that branch would ship
    // having never executed anywhere its author could see — which is how the panic it replaces got
    // shipped in the first place. Run `FERRIC_MAX_BINDING=134217728 cargo run --example flash` to
    // take the small-device path on any machine.
    let max_binding = match std::env::var("FERRIC_MAX_BINDING").ok().and_then(|v| v.parse::<u64>().ok()) {
        Some(v) => { println!("(FERRIC_MAX_BINDING={v} — forcing the constrained path)"); v }
        None => ctx.max_binding,
    };
    let mut ok = true;
    // T=3000 and 5000 cross the 2048-key chunk boundary — validates the online-softmax combination.
    for (t, nh, nkv, dh) in [(64usize,16usize,8usize,128usize),(512,16,8,128),(1024,16,16,64),(3000,8,8,64),(5000,4,4,64)] {
        let d = nh*dh; let kd = nkv*dh;
        let q = Tensor::from_vec(&ctx, &seq(t*d, 1.0), &[t, d]);
        let k = Tensor::from_vec(&ctx, &seq(t*kd, 2.0), &[t, kd]);
        let v = Tensor::from_vec(&ctx, &seq(t*kd, 3.0), &[t, kd]);
        // ⛔ THE BASELINE IS THE THING THAT DOES NOT SCALE, AND CI FOUND IT THE HARD WAY.
        // `causal_attention` materialises [nh,T,T]. At T=3000, nh=8 that is 288,000,000 bytes, and
        // `Context::new` asks for `adapter.limits()` — Metal grants far more, lavapipe and the
        // WebGPU baseline report 134,217,728 (128 MiB). So this case only ever ran because of the
        // hardware it ran on, and the first time CI executed this example on Linux it PANICKED in
        // `Device::create_bind_group`. Skipping the comparison outright would duck the check, so
        // instead the equality is run at the largest head count that FITS — T is unchanged, so the
        // 2048-key chunk boundary is still crossed — and flash is then run at the full head count
        // where the baseline cannot allocate at all. That case is the example's whole thesis.
        let g = nh / nkv;                       // GQA group size; nh_fit must stay a multiple of it
        let per_head = (t * t * 4) as u64;
        let nh_fit = (1..=nh).rev().find(|c| c % g == 0 && nh % c == 0
                                             && (*c as u64) * per_head <= max_binding);
        let (qc, kc, vc, nhc, nkvc) = match nh_fit {
            Some(c) if c < nh => {
                let (d2, kd2) = (c * dh, (c / g) * dh);
                println!("   ↳ T={t} nh={nh}: composed needs {:.0}MB of scores > this device's {:.0}MB \
                          binding limit; comparing at nh={c} instead, then running flash at nh={nh}",
                         (nh as u64 * per_head) as f64 / 1e6, max_binding as f64 / 1e6);
                (Tensor::from_vec(&ctx, &seq(t * d2, 1.0), &[t, d2]),
                 Tensor::from_vec(&ctx, &seq(t * kd2, 2.0), &[t, kd2]),
                 Tensor::from_vec(&ctx, &seq(t * kd2, 3.0), &[t, kd2]), c, c / g)
            }
            Some(_) => (q.clone(), k.clone(), v.clone(), nh, nkv),
            None => {
                println!("❌ T={t} nh={nh}: not even ONE head's scores ({:.0}MB) fit the {:.0}MB limit", 
                         per_head as f64 / 1e6, max_binding as f64 / 1e6);
                ok = false; continue;
            }
        };
        let flash = qc.flash_attention_prefill(&kc, &vc, nhc, nkvc, dh).to_vec().await;
        let comp = nn::causal_attention(&qc, &kc, &vc, nhc, nkvc, 0.0).to_vec().await;
        if nhc < nh {
            // The baseline cannot run here at all; flash can. Finiteness is a weaker claim than the
            // equality above and is labelled as such — there is no oracle at this size on this device.
            let full = q.flash_attention_prefill(&k, &v, nh, nkv, dh).to_vec().await;
            let fin = full.len() == t * nh * dh && full.iter().all(|x| x.is_finite());
            println!("   {} T={t} nh={nh}: flash ran where composed CANNOT allocate (finite output, \
                      no equality oracle at this size on this device)", if fin { "✅" } else { "❌" });
            ok &= fin;
        }
        let e = flash.iter().zip(&comp).map(|(a,b)|(a-b).abs()).fold(0f32,f32::max);
        let p = e < 1e-4; ok &= p;
        let bench = |f: &dyn Fn()->Tensor| { let mut l=None; let t0=Instant::now(); for _ in 0..20 { l=Some(f()); } let _=pollster::block_on(l.unwrap().to_vec()); t0.elapsed().as_secs_f64()/20.0 };
        let ft = bench(&|| qc.flash_attention_prefill(&kc,&vc,nhc,nkvc,dh));
        let ct = bench(&|| { let x = nn::causal_attention(&qc,&kc,&vc,nhc,nkvc, 0.0); x });
        let scores_mb = (nhc*t*t*4) as f64/1e6;
        println!("{} T={t:<4} nh={nhc}: flash {:.2}ms  composed {:.2}ms ({:.1}×, saves {:.0}MB scores)  max|Δ|={e:.1e}",
            if p {"✅"} else {"❌"}, ft*1e3, ct*1e3, ct/ft, scores_mb);
    }
    println!("{}", if ok {"✅ flash prefill == causal_attention, O(T) memory (no [nh,T,T] materialization)"} else {"❌"});
    assert!(ok);
}
