//! Q5_K tensor-core prefill matmul vs scalar (Q5_K_M models).
use ferric_core::Context; use ferric_gguf::deq_raw; use ferric_tensor::{Q5_KWeights, Tensor};
use std::sync::Arc; use std::time::Instant; use half::f16;
fn q5k_block(seed:u32)->Vec<u8>{let mut b=vec![0u8;176];
 b[0..2].copy_from_slice(&f16::from_f32(0.05+0.01*((seed%7)as f32)).to_le_bytes());
 b[2..4].copy_from_slice(&f16::from_f32(0.02+0.005*((seed%5)as f32)).to_le_bytes());
 let sc=|j:u32|((seed.wrapping_mul(2654435761).wrapping_add(j*40503))%64)as u8;
 let mn=|j:u32|((seed.wrapping_mul(40503).wrapping_add(j*2654435761))%64)as u8;
 for j in 0..8u32{if j<4{b[4+j as usize]|=sc(j)&63;b[4+(j+4)as usize]|=mn(j)&63;}
  else{let(a,m)=(sc(j),mn(j));b[4+(j+4)as usize]|=(a&0x0F)|((m&0x0F)<<4);b[4+(j-4)as usize]|=(a>>4)<<6;b[4+j as usize]|=(m>>4)<<6;}}
 for i in 0..32u32{b[16+i as usize]=((seed.wrapping_add(i*2246822519))%256)as u8;}
 for i in 0..128u32{b[48+i as usize]=((seed.wrapping_mul(97).wrapping_add(i*40503))%256)as u8;} b}
fn main(){pollster::block_on(run());}
async fn run(){let ctx=Arc::new(Context::new().await.unwrap());
 println!("{:?} · {} · coop_shared={}",ctx.backend,ctx.adapter_name,ctx.coop_shared_ok());
 if !ctx.coop_shared_ok(){println!("⏭  Metal-only");return;}
 for (m,k,n) in [(64usize,2048usize,2048usize),(256,4096,4096)]{
  let nblk=k/256;let mut packed=Vec::new();for r in 0..n{for b in 0..nblk{packed.extend(q5k_block((r*nblk+b)as u32+1));}}
  let qw=Q5_KWeights::from_bytes(&ctx,&packed,n,k);
  let x=Tensor::from_vec(&ctx,&(0..m*k).map(|i|(i as f32*0.01).sin()).collect::<Vec<_>>(),&[m,k]);
  let coop=x.matmul_q5_k_coop(&qw).to_vec().await;let scalar=x.matmul_q5_k(&qw).to_vec().await;
  let e=coop.iter().zip(&scalar).map(|(a,b)|(a-b).abs()).fold(0f32,f32::max);
  let scl=scalar.iter().map(|v|v.abs()).fold(1e-3,f32::max);
  let bench=|f:&dyn Fn()->Tensor|{let mut l=None;let t=Instant::now();for _ in 0..30{l=Some(f());}let _=pollster::block_on(l.unwrap().to_vec());t.elapsed().as_secs_f64()/30.0};
  let _=x.matmul_q5_k_coop(&qw).to_vec().await;let ct=bench(&||x.matmul_q5_k_coop(&qw));let st=bench(&||x.matmul_q5_k(&qw));
  let fl=2.0*(m as f64)*(k as f64)*(n as f64);
  println!("  [{m}×{k}]·[{k}×{n}]: coop {:.0} GFLOP/s  scalar {:.0}  {:.1}×  rel|Δ|={:.1e}",fl/ct/1e9,fl/st/1e9,st/ct,e/scl);
  assert!(e/scl<6e-2);}
 println!("✅ Q5_K tensor-core prefill validated");}
