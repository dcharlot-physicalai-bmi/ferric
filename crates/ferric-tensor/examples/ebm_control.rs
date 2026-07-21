//! EFA energy-first #30 — ARCHITECTURAL MATCH: energy-based control of a continuous body (pendulum swing-up).
//!
//! The edge thesis is NOT toy capability (a feedforward solves discrete puzzles; see ebm_edge.rs). It is that an
//! ENERGY architecture is the right SHAPE for the physical world. The sharpest embodied test is UNDERACTUATED
//! swing-up: torque is too weak to lift the pendulum directly, so you must PUMP ENERGY until the system reaches
//! the upright energy level, then stabilize. Energy is the natural control object — a model that IS an energy has
//! it natively; a discrete-token model has nothing to grab.
//!   • We LEARN the pendulum's energy Ê(θ,ω) HNN-style on Ferric (2nd-order autograd: match the symplectic gradient
//!     field of the free dynamics — never told the closed form).
//!   • Energy-shaping control uses Ê: pump u = clamp(k·(Ê_top − Ê)·ω), switch to a local stabilizer near the top.
//!   • Baseline: a naive position controller (no energy) that just drives θ→π — it CANNOT swing up (underactuated).
//! Success = reaches upright and stabilizes. If learned-energy control swings up where position control fails, the
//! energy STRUCTURE is load-bearing for embodied control — architecture matching physical reality.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_control --release`
use ferric_tensor::{grad, Adam, Tensor, Var};
use std::sync::Arc;

const HW: usize = 64;
const UMAX: f32 = 0.35;  // torque limit — underactuated (max gravity torque = 1.0 ≫ UMAX)
const C: f32 = 0.02;     // light damping

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn wrap(x: f32) -> f32 { let mut a = x; while a > std::f32::consts::PI { a -= 2.0 * std::f32::consts::PI; } while a < -std::f32::consts::PI { a += 2.0 * std::f32::consts::PI; } a }

// ---- learned energy Ê(θ,ω): softplus MLP, evaluated CPU-side from read-out weights ----
struct Ehat { w1: Vec<f32>, b1: Vec<f32>, w2: Vec<f32>, b2: f32 } // w1[2*HW], b1[HW], w2[HW], b2
impl Ehat {
    fn eval(&self, th: f32, om: f32) -> f32 {
        let mut e = self.b2;
        for j in 0..HW {
            let pre = th * self.w1[j] + om * self.w1[HW + j] + self.b1[j];
            let sp = (pre.exp() + 1.0).ln(); // softplus
            e += sp * self.w2[j];
        }
        e
    }
}

// simulate the pendulum under a control law; return (min angular distance to upright, stabilized?)
fn simulate<F: Fn(f32, f32) -> f32>(ctrl: F, steps: usize, dt: f32) -> (f32, bool) {
    let (mut th, mut om) = (0.02f32, 0.0f32); // start hanging (θ=0 down), tiny nudge
    let mut best = wrap(th - std::f32::consts::PI).abs();
    let mut upcount = 0;
    for _ in 0..steps {
        let uu = ctrl(th, om).clamp(-UMAX, UMAX);
        let acc = -th.sin() - C * om + uu; // θ̈ = −sinθ − c·ω + u
        om += dt * acc; th += dt * om;      // semi-implicit Euler
        let d = wrap(th - std::f32::consts::PI).abs();
        if d < best { best = d; }
        if d < 0.2 && om.abs() < 0.6 { upcount += 1; } else { upcount = 0; }
    }
    (best, upcount > 200) // stabilized = held near upright for the last stretch
}

// energy-shaping controller around a given energy function e(θ,ω) with top value etop
fn energy_ctrl(th: f32, om: f32, e: f32, etop: f32) -> f32 {
    let d = wrap(th - std::f32::consts::PI);
    if d.abs() < 0.5 { -6.0 * d - 2.0 * om }               // local stabilizer near the top
    else { 1.5 * (etop - e) * om }                          // pump energy toward the top
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — ARCHITECTURAL MATCH: energy-based control of a pendulum (underactuated swing-up)");
    println!("  torque limit UMAX={} ≪ 1.0 (max gravity torque) → cannot lift directly; must PUMP ENERGY\n", UMAX);

    // ---- learn Ê(θ,ω) HNN-style: match the symplectic gradient field  ∂H/∂ω = ω,  ∂H/∂θ = sinθ  ----
    let mut p = vec![
        Tensor::from_vec(&ctx, &(0..2 * HW).map(|i| (u(i as u32, 7) - 0.5) * 0.6).collect::<Vec<_>>(), &[2, HW]), Tensor::zeros(&ctx, &[HW]),
        Tensor::from_vec(&ctx, &(0..HW).map(|i| (u(i as u32, 9) - 0.5) * (1.0 / (HW as f32).sqrt())).collect::<Vec<_>>(), &[HW, 1]), Tensor::zeros(&ctx, &[1]),
    ];
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]);
    let mut adam = Adam::new(&p, 0.003); let bs = 256usize;
    for step in 0..3000 {
        let mut sf = vec![0.0f32; bs * 2]; let mut gth = vec![0.0f32; bs]; let mut gom = vec![0.0f32; bs];
        for i in 0..bs { let th = (u(step as u32 * 3 + i as u32, 1) * 2.0 - 1.0) * 3.2; let om = (u(step as u32 * 3 + i as u32, 2) * 2.0 - 1.0) * 3.0;
            sf[i * 2] = th; sf[i * 2 + 1] = om; gth[i] = th.sin(); gom[i] = om; }
        let sl = Var::leaf(Tensor::from_vec(&ctx, &sf, &[bs, 2]));
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
        let sp = |z: Var| z.exp().add(&ov).log();
        let hh = sp(sl.matmul(&pv[0]).add(&pv[1])).matmul(&pv[2]).add(&pv[3]);
        let g = grad(&hh.sum_all(), &[sl.clone()], None).remove(0); // [bs,2] = (∂H/∂θ, ∂H/∂ω)
        let selq = Var::leaf(Tensor::from_vec(&ctx, &[1.0, 0.0], &[2, 1]));
        let selp = Var::leaf(Tensor::from_vec(&ctx, &[0.0, 1.0], &[2, 1]));
        let eth = g.matmul(&selq).sub(&Var::leaf(Tensor::from_vec(&ctx, &gth, &[bs, 1]))); // ∂H/∂θ − sinθ
        let eom = g.matmul(&selp).sub(&Var::leaf(Tensor::from_vec(&ctx, &gom, &[bs, 1]))); // ∂H/∂ω − ω
        let loss = eth.mul(&eth).mean_all().add(&eom.mul(&eom).mean_all());
        loss.backward();
        let gr: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &gr);
    }
    // read out weights → CPU evaluator
    let eh = Ehat { w1: p[0].to_vec().await, b1: p[1].to_vec().await, w2: p[2].to_vec().await, b2: p[3].to_vec().await[0] };
    // learned-energy fit vs true E=½ω²−cosθ (up to a constant): report the gradient-field match by a spot RMSE
    let mut rmse = 0.0f32; let mut n = 0.0; let c0 = eh.eval(0.0, 0.0) - (-1.0);
    for gi in 0..21 { for gj in 0..21 { let th = (gi as f32 / 20.0 * 2.0 - 1.0) * 3.0; let om = (gj as f32 / 20.0 * 2.0 - 1.0) * 3.0;
        let true_e = 0.5 * om * om - th.cos(); let pred = eh.eval(th, om) - c0; rmse += (pred - true_e).powi(2); n += 1.0; } }
    let rmse = (rmse / n).sqrt();
    let etop_hat = eh.eval(std::f32::consts::PI, 0.0);
    let etop_true = 1.0f32;

    // ---- control experiments (CPU sim, dt=0.02, 30 s) ----
    let (steps, dt) = (1500usize, 0.02f32);
    let (b_true, s_true) = simulate(|th, om| energy_ctrl(th, om, 0.5 * om * om - th.cos(), etop_true), steps, dt);
    let (b_learn, s_learn) = simulate(|th, om| energy_ctrl(th, om, eh.eval(th, om), etop_hat), steps, dt);
    // naive position controller: no energy concept, just drive θ→π (PD), clamped
    let (b_pd, s_pd) = simulate(|th, om| { let d = wrap(th - std::f32::consts::PI); -1.2 * d - 0.5 * om }, steps, dt);

    println!("  learned energy Ê (HNN, never told the formula): gradient-field RMSE vs ½ω²−cosθ = {:.3};  Ê_top={:.2} (true 1.00)\n", rmse, etop_hat);
    println!("  SWING-UP RESULT — closest approach to upright (0 = straight up) and whether it STABILIZES there:");
    println!("     energy-shaping, TRUE energy      min|θ−π|={:.2} rad   stabilized: {}", b_true, s_true);
    println!("     energy-shaping, LEARNED energy   min|θ−π|={:.2} rad   stabilized: {}   ← the EFA architecture", b_learn, s_learn);
    println!("     naive position control (no E)    min|θ−π|={:.2} rad   stabilized: {}   ← no energy to shape", b_pd, s_pd);
    println!("\n  If learned-energy control swings up + stabilizes where position control cannot, the energy STRUCTURE");
    println!("  is load-bearing for embodied control — an energy architecture matches physical reality, on-device.");
}
