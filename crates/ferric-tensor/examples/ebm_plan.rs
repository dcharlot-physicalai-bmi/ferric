//! EFA energy-first #31 — CONTROL BY DESCENT: plan an underactuated swing-up through a LEARNED model, no hand law.
//!
//! ebm_control.rs used a hand-coded energy-shaping law + PD stabilizer. This removes the hand law entirely: the
//! architecture PLANS the control by descending a cost through a LEARNED dynamics model (MPPI — sample action
//! sequences, roll them out through the learned model, weight by exp(−cost), descend). Energy-pumping is never
//! programmed — it EMERGES from planning, because rollouts that pump energy reach the upright goal. And capability
//! scales with the PLANNING HORIZON H = on-device thinking = the compute/energy axis (NOT tokens): a greedy short
//! horizon can't discover the pump and fails; a longer horizon plans it and swings up. This is "act by descending
//! energy" (the EFA planner) on a continuous body — data-center-grade capability from cheap on-device thinking.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_plan --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

const HW: usize = 64;
const UMAX: f32 = 0.35; const C: f32 = 0.02;
use std::f32::consts::PI;

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn nrm(seed: u32) -> f32 { let a = u(seed, 1); let b = u(seed, 2); (-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos() }
fn wrap(x: f32) -> f32 { let mut a = x; while a > PI { a -= 2.0 * PI; } while a < -PI { a += 2.0 * PI; } a }

// learned dynamics: dω = g(sinθ, cosθ, ω, u); dθ = ω (kinematics). CPU eval from read-out weights.
struct Dyn { w1: Vec<f32>, b1: Vec<f32>, w2: Vec<f32>, b2: f32 }
impl Dyn {
    fn domega(&self, th: f32, om: f32, uu: f32) -> f32 {
        let inp = [th.sin(), th.cos(), om, uu]; let mut o = self.b2;
        for j in 0..HW { let mut pre = self.b1[j]; for k in 0..4 { pre += inp[k] * self.w1[k * HW + j]; } o += (pre.exp() + 1.0).ln() * self.w2[j]; }
        o
    }
    fn step(&self, th: f32, om: f32, uu: f32, dt: f32) -> (f32, f32) { let no = om + dt * self.domega(th, om, uu); (th + dt * no, no) }
}

// MPPI: plan H-step action sequences through the LEARNED model, descend the cost, execute on TRUE dynamics.
fn mppi(mdl: &Dyn, horizon: usize, steps: usize, dt: f32) -> (f32, bool) {
    let (n, lambda, sigma) = (300usize, 0.3f32, 0.5f32);
    let (mut th, mut om) = (0.02f32, 0.0f32);
    let mut nom = vec![0.0f32; horizon];
    let mut best = wrap(th - PI).abs(); let mut upc = 0;
    for t in 0..steps {
        let mut costs = vec![0.0f32; n]; let mut eps = vec![vec![0.0f32; horizon]; n];
        for i in 0..n {
            let (mut sh, mut so) = (th, om); let mut cst = 0.0f32;
            for h in 0..horizon {
                let e = sigma * nrm(h32(t as u32 * 2654435761 ^ (i as u32) << 8 ^ h as u32));
                eps[i][h] = e; let a = (nom[h] + e).clamp(-UMAX, UMAX);
                let (nth, nom_) = mdl.step(sh, so, a, dt); sh = nth; so = nom_;
                cst += wrap(sh - PI).powi(2) + 0.05 * so * so + 0.02 * a * a;
            }
            cst += 5.0 * (wrap(sh - PI).powi(2) + 0.3 * so * so); // terminal cost: end upright AND at rest
            costs[i] = cst;
        }
        let cmin = costs.iter().cloned().fold(f32::MAX, f32::min);
        let mut wsum = 0.0f32; let mut w = vec![0.0f32; n];
        for i in 0..n { w[i] = (-(costs[i] - cmin) / lambda).exp(); wsum += w[i]; }
        for h in 0..horizon { let mut du = 0.0f32; for i in 0..n { du += w[i] * eps[i][h]; } nom[h] += du / wsum.max(1e-6); }
        // execute first action on TRUE dynamics
        let a0 = nom[0].clamp(-UMAX, UMAX);
        let acc = -th.sin() - C * om + a0; om += dt * acc; th += dt * om;
        let d = wrap(th - PI).abs(); if d < best { best = d; }
        if d < 0.2 && om.abs() < 0.7 { upc += 1; } else { upc = 0; }
        for h in 0..horizon - 1 { nom[h] = nom[h + 1]; } nom[horizon - 1] = 0.0; // receding horizon shift
    }
    (best, upc > 60)
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — CONTROL BY DESCENT: plan an underactuated swing-up through a LEARNED model (no hand law)");
    println!("  torque UMAX={} ≪ 1.0 → must pump energy; the pump is NOT programmed — it emerges from planning\n", UMAX);

    // ---- learn dynamics dω = g(sinθ,cosθ,ω,u) by regression to the true force law ----
    let mut p = vec![
        Tensor::from_vec(&ctx, &(0..4 * HW).map(|i| (u(i as u32, 7) - 0.5) * 0.5).collect::<Vec<_>>(), &[4, HW]), Tensor::zeros(&ctx, &[HW]),
        Tensor::from_vec(&ctx, &(0..HW).map(|i| (u(i as u32, 9) - 0.5) * (1.0 / (HW as f32).sqrt())).collect::<Vec<_>>(), &[HW, 1]), Tensor::zeros(&ctx, &[1]),
    ];
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]);
    let mut adam = Adam::new(&p, 0.003); let bs = 256usize;
    for step in 0..2500 {
        let mut inp = vec![0.0f32; bs * 4]; let mut tgt = vec![0.0f32; bs];
        for i in 0..bs { let th = (u(step as u32 * 3 + i as u32, 1) * 2.0 - 1.0) * 3.2; let om = (u(step as u32 * 3 + i as u32, 2) * 2.0 - 1.0) * 4.0; let uu = (u(step as u32 * 3 + i as u32, 4) * 2.0 - 1.0) * UMAX;
            inp[i * 4] = th.sin(); inp[i * 4 + 1] = th.cos(); inp[i * 4 + 2] = om; inp[i * 4 + 3] = uu; tgt[i] = -th.sin() - C * om + uu; }
        let xv = Var::leaf(Tensor::from_vec(&ctx, &inp, &[bs, 4]));
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
        let h1 = xv.matmul(&pv[0]).add(&pv[1]).exp().add(&ov).log();
        let pred = h1.matmul(&pv[2]).add(&pv[3]);
        let e = pred.sub(&Var::leaf(Tensor::from_vec(&ctx, &tgt, &[bs, 1])));
        let loss = e.mul(&e).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &g);
    }
    let mdl = Dyn { w1: p[0].to_vec().await, b1: p[1].to_vec().await, w2: p[2].to_vec().await, b2: p[3].to_vec().await[0] };
    // sanity: model force-law RMSE on a grid
    let mut rmse = 0.0f32; let mut nn = 0.0; for gi in 0..15 { for gj in 0..15 { let th = (gi as f32 / 14.0 * 2.0 - 1.0) * 3.0; let om = (gj as f32 / 14.0 * 2.0 - 1.0) * 3.5;
        rmse += (mdl.domega(th, om, 0.1) - (-th.sin() - C * om + 0.1)).powi(2); nn += 1.0; } }
    println!("  learned dynamics: force-law RMSE = {:.3}\n", (rmse / nn).sqrt());

    println!("  SWING-UP vs PLANNING HORIZON H (= on-device thinking = the compute axis):");
    println!("     H (steps)    closest to upright min|θ−π|    stabilized");
    for &hh in &[10usize, 40, 80, 120] {
        let (b, s) = mppi(&mdl, hh, 260, 0.05);
        println!("     {:>3}          {:>6.2} rad                    {}", hh, b, s);
    }
    println!("\n  If short horizons fail and long horizons swing up + stabilize, capability comes from PLANNING COMPUTE");
    println!("  (thinking), not a hand-coded law and not parameters — data-center-grade capability at the edge.");
}
