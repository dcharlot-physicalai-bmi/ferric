//! Would fusing FFN gate+up (same input, two [5120→17408] matmuls) into one [5120→34816] matmul
//! actually help at decode width? Measure before building it into the model.
use ferric_core::Context;
use ferric_gguf::quant_q2_0;
use ferric_tensor::{Q2_0Weights, Tensor};
use std::sync::Arc;
fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let inn = 5120usize;
    let mk = |out: usize| {
        let w: Vec<f32> = (0..out * inn).map(|i| ((i % 3) as f32 - 1.0) * 0.02).collect();
        let mut packed = Vec::new();
        for r in 0..out { packed.extend(quant_q2_0(&w[r * inn..(r + 1) * inn])); }
        Q2_0Weights::from_bytes(&ctx, &packed, out, inn)
    };
    let (g, u, fused) = (mk(17408), mk(17408), mk(34816));
    for toks in [1usize, 5] {
        let x = Tensor::from_vec(&ctx, &(0..toks * inn).map(|i| (i as f32 * 0.01).sin()).collect::<Vec<_>>(), &[toks, inn]);
        let bench = |label: &str, f: &dyn Fn() -> Tensor| {
            let _ = pollster::block_on(f().to_vec()); // warm
            let reps = 30;
            let t0 = std::time::Instant::now();
            let mut last = None;
            for _ in 0..reps { last = Some(f()); }
            let _ = pollster::block_on(last.unwrap().to_vec());
            let ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
            println!("    {label:<28} {ms:.3} ms", );
            ms
        };
        println!("  toks={toks}:");
        let sep = bench("2 separate matmuls", &|| { let a = x.matmul_q2_0(&g); let b = x.matmul_q2_0(&u); a.add(&b) });
        let one = bench("1 fused matmul [.,34816]", &|| x.matmul_q2_0(&fused));
        println!("    → fused is {:.2}× the separate pair\n", sep / one);
    }
}
