//! Q4_K tensor-core prefill matmul vs the scalar Q4_K kernel (the default llama.cpp format).
use ferric_core::Context;
use ferric_gguf::deq_raw;
use ferric_tensor::{Q4_KWeights, Tensor};
use std::sync::Arc; use std::time::Instant; use half::f16;
fn q4k_block(seed: u32) -> Vec<u8> {
    let mut b = Vec::with_capacity(144);
    b.extend_from_slice(&f16::from_f32(0.05 + 0.01*((seed%7) as f32)).to_le_bytes());
    b.extend_from_slice(&f16::from_f32(0.02 + 0.005*((seed%5) as f32)).to_le_bytes());
    let sc=|j:u32| ((seed.wrapping_mul(2654435761).wrapping_add(j*40503))%64) as u8;
    let mn=|j:u32| ((seed.wrapping_mul(40503).wrapping_add(j*2654435761))%64) as u8;
    let mut s=[0u8;12];
    for j in 0..8u32 { if j<4 { s[j as usize]|=sc(j)&63; s[(j+4)as usize]|=mn(j)&63; }
        else { let(a,m)=(sc(j),mn(j)); s[(j+4)as usize]|=(a&0x0F)|((m&0x0F)<<4); s[(j-4)as usize]|=(a>>4)<<6; s[j as usize]|=(m>>4)<<6; } }
    b.extend_from_slice(&s);
    for i in 0..128u32 { b.push((((seed.wrapping_add(i*2246822519))%256) as u8)&0xff); }
    b
}
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    println!("{:?} · {} · coop_shared={}", ctx.backend, ctx.adapter_name, ctx.coop_shared_ok());
    if !ctx.coop_shared_ok() { println!("⏭  coop-from-shared Metal-only"); return; }
    for (m,k,n) in [(64usize,2048usize,2048usize),(256,4096,4096)] {
        let nblk=k/256; let mut packed=Vec::new();
        for r in 0..n { for b in 0..nblk { packed.extend(q4k_block((r*nblk+b) as u32+1)); } }
        let qw=Q4_KWeights::from_bytes(&ctx,&packed,n,k);
        let x=Tensor::from_vec(&ctx,&(0..m*k).map(|i|(i as f32*0.01).sin()).collect::<Vec<_>>(),&[m,k]);
        let coop=x.matmul_q4_k_coop(&qw).to_vec().await;
        let scalar=x.matmul_q4_k(&qw).to_vec().await;
        let e=coop.iter().zip(&scalar).map(|(a,b)|(a-b).abs()).fold(0f32,f32::max);
        let scl=scalar.iter().map(|v|v.abs()).fold(1e-3,f32::max);
        let bench=|f:&dyn Fn()->Tensor|{let mut l=None;let t=Instant::now();for _ in 0..30{l=Some(f());}let _=pollster::block_on(l.unwrap().to_vec());t.elapsed().as_secs_f64()/30.0};
        let _=x.matmul_q4_k_coop(&qw).to_vec().await;
        let ct=bench(&||x.matmul_q4_k_coop(&qw)); let st=bench(&||x.matmul_q4_k(&qw));
        let flop=2.0*(m as f64)*(k as f64)*(n as f64);
        println!("  [{m}×{k}]·[{k}×{n}]: coop {:.0} GFLOP/s  scalar {:.0}  {:.1}×  rel|Δ|={:.1e}", flop/ct/1e9, flop/st/1e9, st/ct, e/scl);
        assert!(e/scl < 6e-2, "q4_k coop diverged");
    }
    println!("✅ Q4_K tensor-core prefill matmul validated — the default GGUF format on the matrix unit");
}
