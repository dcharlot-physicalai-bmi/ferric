//! Q8_0 tensor-core prefill matmul vs scalar.
use ferric_core::Context; use ferric_gguf::{deq_raw, quant_q8_0}; use ferric_tensor::{Q8_0Weights, Tensor};
use std::sync::Arc; use std::time::Instant;
fn main(){pollster::block_on(run());}
async fn run(){let ctx=Arc::new(Context::new().await.unwrap());
 println!("{:?} · {} · coop_shared={}",ctx.backend,ctx.adapter_name,ctx.coop_shared_ok());
 if !ctx.coop_shared_ok(){println!("⏭  Metal-only");return;}
 for (m,k,n) in [(64usize,2048usize,2048usize),(256,4096,4096)]{
  let wf:Vec<f32>=(0..n*k).map(|i|((i%97)as f32-48.0)*0.02).collect();
  let mut packed=Vec::new();for r in 0..n{packed.extend(quant_q8_0(&wf[r*k..(r+1)*k]));}
  let qw=Q8_0Weights::from_bytes(&ctx,&packed,n,k);
  let x=Tensor::from_vec(&ctx,&(0..m*k).map(|i|(i as f32*0.01).sin()).collect::<Vec<_>>(),&[m,k]);
  let coop=x.matmul_q8_0_coop(&qw).to_vec().await;let scalar=x.matmul_q8_0(&qw).to_vec().await;
  let e=coop.iter().zip(&scalar).map(|(a,b)|(a-b).abs()).fold(0f32,f32::max);
  let scl=scalar.iter().map(|v|v.abs()).fold(1e-3,f32::max);
  let bench=|f:&dyn Fn()->Tensor|{let mut l=None;let t=Instant::now();for _ in 0..30{l=Some(f());}let _=pollster::block_on(l.unwrap().to_vec());t.elapsed().as_secs_f64()/30.0};
  let _=x.matmul_q8_0_coop(&qw).to_vec().await;let ct=bench(&||x.matmul_q8_0_coop(&qw));let st=bench(&||x.matmul_q8_0(&qw));
  let fl=2.0*(m as f64)*(k as f64)*(n as f64);
  println!("  [{m}×{k}]·[{k}×{n}]: coop {:.0} GFLOP/s  scalar {:.0}  {:.1}×  rel|Δ|={:.1e}",fl/ct/1e9,fl/st/1e9,st/ct,e/scl);
  assert!(e/scl<6e-2);}
 println!("✅ Q8_0 tensor-core prefill validated");}
