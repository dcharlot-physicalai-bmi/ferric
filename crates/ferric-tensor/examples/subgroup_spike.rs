//! Spike: confirm the `subgroups` WGSL feature works on this fabric — subgroupAdd must sum a
//! workgroup's values correctly. This is the primitive that replaces the split-K barrier tree.
use ferric_core::Context;
use std::sync::Arc;
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    println!("backend {:?} · adapter {} · subgroups={}", ctx.backend, ctx.adapter_name, ctx.subgroups);
    if !ctx.subgroups { println!("⏭  subgroups not available on this adapter"); return; }
    // 64 threads each contribute their index; a subgroup-then-shared reduction must total sum(0..64)=2016.
    let out = ferric_tensor::run_subgroup_sum_test(&ctx).await;
    let expect = (0..64u32).sum::<u32>() as f32;
    let ok = (out - expect).abs() < 1e-3;
    println!("{} subgroupAdd reduction: got {out}, expect {expect}", if ok { "✅" } else { "❌" });
    assert!(ok);
}
