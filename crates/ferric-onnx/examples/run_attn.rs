use ferric_core::{max_abs_diff, Context};
use std::collections::HashMap;
fn main() { pollster::block_on(run()); }
fn f32s(b:&[u8])->Vec<f32>{ b.chunks_exact(4).map(|c|f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect() }
fn ref_y(j:&str)->Vec<f32>{ let s=j.split("\"y\"").nth(1).unwrap(); let i=s.split('[').nth(1).unwrap().split(']').next().unwrap(); i.split(',').filter_map(|x|x.trim().parse().ok()).collect() }
async fn run(){
    let ctx=Context::new().await.unwrap();
    let dir=env!("CARGO_MANIFEST_DIR");
    let model=ferric_onnx::load(&std::fs::read(format!("{dir}/testdata/attn.onnx")).unwrap()).unwrap();
    println!("Ferric ONNX attention · {:?} · ops {:?}", ctx.backend, model.ops());
    let x=f32s(&std::fs::read(format!("{dir}/testdata/attn_x.bin")).unwrap());
    let mut inp=HashMap::new(); inp.insert("X".into(),(x,vec![6usize,16]));
    let (_,y)=model.run(&ctx,&inp).await.unwrap();
    let refy=ref_y(&std::fs::read_to_string(format!("{dir}/testdata/attn_ref.json")).unwrap());
    let d=max_abs_diff(&y,&refy);
    println!("y[:5]     = {:?}", y[..5].iter().map(|v|(v*1e4).round()/1e4).collect::<Vec<_>>());
    println!("ort ref   = {:?}", refy[..5].iter().map(|v|(v*1e4).round()/1e4).collect::<Vec<_>>());
    println!("max|ferric-ort| = {:.3e}", d);
    assert!(d<1e-4,"diverged {d}");
    println!("✅ REAL SELF-ATTENTION ONNX RAN IN FERRIC — matches onnxruntime on {:?}", ctx.backend);
}
