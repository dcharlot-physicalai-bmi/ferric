// Validate the L1 kernel set (add, silu, layernorm, softmax) against CPU references on the native GPU.
use ferric_core::{cpu, max_abs_diff, Context};

fn main() { pollster::block_on(run()); }

async fn run() {
    let ctx = Context::new().await.expect("ctx");
    println!("Ferric kernels · {:?} · {}", ctx.backend, ctx.adapter_name);
    let mut ok = true;
    let mut check = |name: &str, gpu: &[f32], cpu: &[f32]| {
        let d = max_abs_diff(gpu, cpu);
        let pass = d < 1e-4;
        ok &= pass;
        println!("  {} {:<10} max|gpu-cpu| = {:.3e}", if pass { "✅" } else { "❌" }, name, d);
    };

    let n = 4096usize;
    let a: Vec<f32> = (0..n).map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.13).collect();
    let b: Vec<f32> = (0..n).map(|i| ((i * 5 % 17) as f32 - 8.0) * 0.09).collect();
    check("add", &ctx.add(&a, &b).await.unwrap(), &cpu::add(&a, &b));
    check("silu", &ctx.silu(&a).await.unwrap(), &cpu::silu(&a));

    let (rows, d) = (32usize, 128usize);
    let x: Vec<f32> = (0..rows * d).map(|i| ((i * 3 % 29) as f32 - 14.0) * 0.2).collect();
    let w: Vec<f32> = (0..d).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let bias: Vec<f32> = (0..d).map(|i| (i as f32) * 0.002 - 0.1).collect();
    check("layernorm", &ctx.layernorm(&x, &w, &bias, rows as u32, d as u32, 1e-5).await.unwrap(),
          &cpu::layernorm(&x, &w, &bias, rows, d, 1e-5));
    check("softmax", &ctx.softmax(&x, rows as u32, d as u32).await.unwrap(),
          &cpu::softmax(&x, rows, d));

    // matmul_bt (A·Bᵀ, scaled) + full scaled-dot-product attention
    let (rq, rk, dd, dv) = (24usize, 40usize, 64usize, 48usize);
    let scale = 1.0f32 / (dd as f32).sqrt();
    let q: Vec<f32> = (0..rq*dd).map(|i| ((i*11%31) as f32 - 15.0)*0.05).collect();
    let kk: Vec<f32> = (0..rk*dd).map(|i| ((i*13%37) as f32 - 18.0)*0.04).collect();
    let vv: Vec<f32> = (0..rk*dv).map(|i| ((i*7%19) as f32 - 9.0)*0.06).collect();
    check("matmul_bt", &ctx.matmul_bt(&q,&kk,rq as u32,rk as u32,dd as u32,scale).await.unwrap(),
          &cpu::matmul_bt(&q,&kk,rq,rk,dd,scale));
    check("attention", &ctx.attention(&q,&kk,&vv,rq as u32,rk as u32,dd as u32,dv as u32,scale).await.unwrap(),
          &cpu::attention(&q,&kk,&vv,rq,rk,dd,dv,scale));

    println!("{}", if ok { "✅ ALL KERNELS VALIDATED" } else { "❌ a kernel diverged" });
    assert!(ok);
}
