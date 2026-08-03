//! PHASE 1 — TERNARY-NATIVE SSM: the super-learner's backbone, and the regime post-hoc QAT can't reach.
//!
//! Phase 0 falsified FOUR levers for squeezing a pretrained f32 model into ternary (data scale, bit-width,
//! optimizer, data realism) — the wall is the pretrained-f32→ternary REGIME itself: f32 weights encode
//! precision ternary cannot represent, so compressing them destroys the long tail (Hooker). The way past it
//! is not a better compressor: it's to train TERNARY-NATIVE from step 1, so the network only ever learns
//! ternary-expressible representations.
//!
//! This trains a selective-SSM (Mamba-style `h_t = a_t ⊙ h_{t-1} + b_t`, built on the gradchecked
//! `Var::selective_scan`) on a copy/recall task that REQUIRES the recurrent state to carry information
//! across time — i.e. the state must hold the "tail" that compression forgets. Three arms:
//!   f32 (reference) · PTQ (train f32 → ternarize after = Phase 0's failed regime) · ternary-native (STE from scratch)
//! The claim under test: ternary-NATIVE ≈ f32, while PTQ collapses — the inverse of Phase 0's result.
//!   cargo run -p ferric-tensor --example ternary_ssm --release
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

const GS: usize = 32; // grouped NON-learnable scales (learnable = zero-ratio collapse, per Ternary Mamba)

// per-group absmean ternary {−1,0,+1}·γ — the BitNet b1.58 quantizer.
fn ternarize(w: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; w.len()];
    for g in 0..w.len().div_ceil(GS) {
        let (lo, hi) = (g * GS, ((g + 1) * GS).min(w.len()));
        let gamma = (w[lo..hi].iter().map(|x| x.abs()).sum::<f32>() / (hi - lo) as f32).max(1e-8);
        for k in lo..hi { out[k] = (w[k] / gamma).round().clamp(-1.0, 1.0) * gamma; }
    }
    out
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let (t, d, hid) = (16usize, 1usize, 24usize); // T steps, 1-d signal, hidden width
    let nseq = 24usize;

    // ── task: SELECTIVE COPY. A value appears at a random early step, then the channel is noise;
    // at the final step the model must output that value. Solvable ONLY by writing the value into the
    // recurrent state and holding it — the exact "carry the tail through time" capability we need.
    let mut seed = 0x51F0u64;
    let mut u = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((seed >> 40) as f32 + 0.5) / (1u32 << 24) as f32 };
    let mut xs: Vec<Vec<f32>> = Vec::new(); // [nseq][T*2] : (value_channel, flag_channel)
    let mut ys: Vec<f32> = Vec::new();
    for _ in 0..nseq {
        let pos = (u() * 5.0) as usize;              // value arrives in the first 5 steps
        let val = u() * 2.0 - 1.0;                   // value in [-1,1]
        let mut row = vec![0f32; t * 2];
        for i in 0..t { row[i * 2] = (u() * 2.0 - 1.0) * 0.3; } // noise on the value channel
        row[pos * 2] = val;                          // the real value
        row[pos * 2 + 1] = 1.0;                      // flag marks "remember this one"
        xs.push(row); ys.push(val);
    }

    // ── params: input proj (2→hid) → SSM gates → readout (hid→1)
    let mut gs = |u: &mut dyn FnMut() -> f32| (-2.0 * u().max(1e-7).ln()).sqrt() * (std::f32::consts::TAU * u()).cos();
    let mk = |v: Vec<f32>, s: &[usize]| Tensor::from_vec(&ctx, &v, s);
    let init = |n: usize, sc: f32, u: &mut dyn FnMut() -> f32, gs: &mut dyn FnMut(&mut dyn FnMut() -> f32) -> f32| -> Vec<f32> {
        (0..n).map(|_| gs(u) * sc).collect()
    };
    // w_in [2,hid] → b_t ; w_a [2,hid] → per-step decay a_t (sigmoid) ; w_out [hid,1]
    let mut p0 = || vec![
        mk(init(2 * hid, 0.8, &mut u, &mut gs), &[2, hid]),
        mk(init(2 * hid, 0.8, &mut u, &mut gs), &[2, hid]),
        mk(init(hid, 0.8, &mut u, &mut gs), &[hid, 1]),
        mk(vec![1.0f32; hid], &[hid]),   // 3 post-SSM RMSNorm (Phase-3 finding); kept F32 — BitNet does not quantize norms
    ];
    let xv: Vec<Var> = xs.iter().map(|r| Var::leaf(mk(r.clone(), &[t, 2]))).collect();
    let yv: Vec<Var> = ys.iter().map(|&y| Var::leaf(mk(vec![y], &[1, 1]))).collect();
    let one = Var::leaf(mk(vec![1.0], &[1]));

    // forward: b = x·w_in ; a = sigmoid(x·w_a + BIAS) ; h = scan(a,b) ; ŷ = h_T · w_out
    // BIAS=+3 → a≈0.95 at init, so the state RETAINS by default (standard S4/Mamba practice). Without it
    // a≈0.5 halves the state every step and nothing survives 16 steps — the task becomes unlearnable.
    let abias = Var::leaf(mk(vec![3.0], &[1]));
    let fwd = |x: &Var, p: &[Var]| -> Var {
        let b = x.matmul(&p[0]);                                   // [T,hid]
        let a = one.div(&one.add(&x.matmul(&p[1]).add(&abias).neg().exp())); // sigmoid(·+3) → decay ≈0.95 at init
        // POST-SSM RMSNORM — the Phase-3 finding: the scan ACCUMULATES and swamps the readout's scale.
        // Phase 1 was originally measured WITHOUT this, so its ternary number was likely understated.
        let h = Var::selective_scan(&a, &b).rmsnorm(&p[3], 1e-5);                       // [T,hid] — the gradchecked SSM op
        h.narrow(0, t - 1, 1).matmul(&p[2])                        // last state → readout [1,1]
    };
    let mse = |p: &[Var]| -> Var {
        let mut loss: Option<Var> = None;
        for i in 0..nseq {
            let dlt = fwd(&xv[i], p).sub(&yv[i]);
            let l = dlt.mul(&dlt);
            loss = Some(match loss { Some(a) => a.add(&l), None => l });
        }
        loss.unwrap()
    };
    let eval = |ws: &[Tensor]| -> f32 {
        let p: Vec<Var> = ws.iter().map(|w| Var::leaf(w.clone())).collect();
        let mut se = 0f32;
        for i in 0..nseq {
            let pred = pollster::block_on(fwd(&xv[i], &p).value().to_vec())[0];
            se += (pred - ys[i]).powi(2);
        }
        se / nseq as f32
    };

    let steps = 600;
    // ── ARM 1: f32 reference ───────────────────────────────────────────────────────────────────
    let mut w = p0();
    let mut opt = Adam::new(&w, 0.02);
    for _ in 0..steps {
        let p: Vec<Var> = w.iter().map(|x| Var::leaf(x.clone())).collect();
        let l = mse(&p); l.backward();
        let g: Vec<Tensor> = p.iter().map(|v| v.grad().unwrap()).collect();
        opt.step(&mut w, &g);
    }
    let mse_f32 = eval(&w);

    // ── ARM 2: PTQ — ternarize the trained f32 weights (Phase 0's failed regime, in miniature) ──
    // keep the norm f32 here too, so PTQ and ternary-native are compared on identical conventions
    let w_ptq: Vec<Tensor> = w.iter().enumerate().map(|(i, x)| if i == 3 { x.clone() }
        else { mk(ternarize(&pollster::block_on(x.to_vec())), &x.shape) }).collect();
    let mse_ptq = eval(&w_ptq);

    // ── ARM 3: TERNARY-NATIVE — STE from scratch, never sees f32-precision structure ────────────
    let mut s = p0(); // fresh init (shadow weights)
    let mut sopt = Adam::new(&s, 0.02);
    for _ in 0..steps {
        // forward uses TERNARIZED weights; STE routes their grad straight to the f32 shadows
        let p: Vec<Var> = s.iter().enumerate().map(|(i, x)| if i == 3 { Var::leaf(x.clone()) }
            else { Var::leaf(mk(ternarize(&pollster::block_on(x.to_vec())), &x.shape)) }).collect();
        let l = mse(&p); l.backward();
        let g: Vec<Tensor> = p.iter().map(|v| v.grad().unwrap()).collect();
        sopt.step(&mut s, &g);
    }
    let w_tern: Vec<Tensor> = s.iter().enumerate().map(|(i, x)| if i == 3 { x.clone() }
        else { mk(ternarize(&pollster::block_on(x.to_vec())), &x.shape) }).collect();
    let mse_native = eval(&w_tern);

    let var_y = { let m = ys.iter().sum::<f32>() / nseq as f32; ys.iter().map(|y| (y - m).powi(2)).sum::<f32>() / nseq as f32 };
    let r2 = |m: f32| 1.0 - m / var_y;
    println!("TERNARY-NATIVE SSM — selective-copy over {t} steps (state must carry the value):\n");
    println!("  f32 reference        MSE {:.4}   R² {:+.3}", mse_f32, r2(mse_f32));
    println!("  PTQ  (f32→ternary)   MSE {:.4}   R² {:+.3}   ← Phase 0's regime", mse_ptq, r2(mse_ptq));
    println!("  TERNARY-NATIVE (STE) MSE {:.4}   R² {:+.3}   ← trained ternary from step 1", mse_native, r2(mse_native));
    let verdict = mse_native < mse_ptq * 0.5;
    println!("\n{}  ternary-native is {:.1}× {} than PTQ on the same architecture.",
        if verdict { "✅" } else { "⚠" }, (mse_ptq / mse_native.max(1e-9)).max(mse_native / mse_ptq.max(1e-9)),
        if verdict { "BETTER" } else { "not clearly better" });
    println!("   Built on the gradchecked Var::selective_scan — Ferric's first SSM op (no Rust runtime has one).");
}
