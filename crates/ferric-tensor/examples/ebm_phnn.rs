//! EFA energy-first #23 — PORT-HAMILTONIAN energy: the honest physics bridge to embodied control.
//!
//! The HNN (`ebm_hamiltonian.rs`) learns a CONSERVED energy — but a real body is neither closed nor undriven:
//! a joint has FRICTION (dissipation) and a MOTOR (an energy PORT). Port-Hamiltonian systems capture exactly
//! this:   ẋ = (J − R)∇H + G·u,   with J skew (lossless interconnection), R⪰0 (dissipation), G the input port.
//! Power balance:  dH/dt = −∇Hᵀ R ∇H + yᵀu   (energy lost to friction, plus energy injected through the port).
//!
//! Test on a damped, driven oscillator (mass–spring–damper + forcing):  q̇ = p/m,  ṗ = −k·q − c·p + u(t).
//! We train three models and ask which one MODELS THE ENERGY BUDGET of an actuated, dissipative body:
//!   • naive  — black-box ẋ = f(q,p,u)                 (no energy notion)
//!   • HNN    — conservation-only ẋ = J∇H + G·u        (structurally CANNOT dissipate — must fail on a damped body)
//!   • PHNN   — learns H AND the dissipation r ≥ 0      (energy-accounted: separates friction-loss from port-injection)
//! Learning H requires ∇H inside the loss → second-order autograd (Ferric has it).
//!
//! Run: `cargo run -p ferric-tensor --example ebm_phnn --release`
use ferric_tensor::{grad, Adam, Tensor, Var};
use std::sync::Arc;

const HW: usize = 64;
const M: f32 = 1.0; const K: f32 = 1.0; const C: f32 = 0.3; // true mass, stiffness, damping
fn uforce(t: f32) -> f32 { 0.5 * (0.7 * t).sin() }           // the "motor": a driving input port

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }

fn energy(s: &Var, p: &[Var], one: &Var) -> Var {
    let sp = |z: Var| z.exp().add(one).log();
    let h1 = sp(s.matmul(&p[0]).add(&p[1]));
    h1.matmul(&p[2]).add(&p[3])
}
// ∇H w.r.t. state; returns (∂H/∂q, ∂H/∂p) as [B,1] each
fn gradH(ctx: &Arc<ferric_core::Context>, sl: &Var, p: &[Var], one: &Var) -> (Var, Var) {
    let gh = grad(&energy(sl, p, one).sum_all(), &[sl.clone()], None).remove(0); // [B,2]
    let selq = Var::leaf(Tensor::from_vec(ctx, &[1.0, 0.0], &[2, 1]));
    let selp = Var::leaf(Tensor::from_vec(ctx, &[0.0, 1.0], &[2, 1]));
    (gh.matmul(&selq), gh.matmul(&selp))
}

// train an energy model; dissipative=true learns r≥0 (PHNN), false forces r=0 (HNN)
async fn train_h(ctx: &Arc<ferric_core::Context>, dissipative: bool) -> (Vec<Tensor>, f32) {
    let one = Tensor::from_vec(ctx, &[1.0], &[1]);
    let mut p = vec![
        Tensor::from_vec(ctx, &randn(2 * HW, 10, 0.7), &[2, HW]), Tensor::zeros(ctx, &[HW]),
        Tensor::from_vec(ctx, &randn(HW, 11, 1.0 / (HW as f32).sqrt()), &[HW, 1]), Tensor::zeros(ctx, &[1]),
    ];
    let mut rp = vec![Tensor::from_vec(ctx, &[-1.0], &[1])]; // softplus(-1)≈0.31 init
    let mut adam = Adam::new(&p, 0.003); let mut adamr = Adam::new(&rp, 0.01);
    let bs = 256usize;
    for step in 0..4000 {
        // random states + input; true derivatives from the known ODE
        let q = randn(bs, step as u32 * 3 + 1, 1.3); let pp = randn(bs, step as u32 * 3 + 2, 1.3);
        let tt: Vec<f32> = (0..bs).map(|i| u(step as u32 * 7 + i as u32, 5) * 18.0).collect();
        let mut sf = vec![0.0f32; bs * 2]; let mut uu = vec![0.0f32; bs]; let mut dq = vec![0.0f32; bs]; let mut dp = vec![0.0f32; bs];
        for i in 0..bs { sf[i * 2] = q[i]; sf[i * 2 + 1] = pp[i]; uu[i] = uforce(tt[i]);
            dq[i] = pp[i] / M; dp[i] = -K * q[i] - C * pp[i] + uu[i]; }
        let sl = Var::leaf(Tensor::from_vec(ctx, &sf, &[bs, 2]));
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
        let (gq, gp) = gradH(ctx, &sl, &pv, &ov);
        let uv = Var::leaf(Tensor::from_vec(ctx, &uu, &[bs, 1]));
        let dq_pred = gp.clone();                                   // q̇ = ∂H/∂p
        let mut dp_pred = gq.neg().add(&uv);                        // ṗ = −∂H/∂q + u  (+ dissipation below)
        let rv = Var::leaf(rp[0].clone());
        let r_soft = rv.exp().add(&ov).log();                       // softplus(r_param) ≥ 0
        if dissipative { dp_pred = dp_pred.sub(&gp.mul(&r_soft)); } // −r·∂H/∂p
        let eq = dq_pred.sub(&Var::leaf(Tensor::from_vec(ctx, &dq, &[bs, 1])));
        let ep = dp_pred.sub(&Var::leaf(Tensor::from_vec(ctx, &dp, &[bs, 1])));
        let loss = eq.mul(&eq).mean_all().add(&ep.mul(&ep).mean_all());
        loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &g);
        if dissipative { let gr = rv.grad().unwrap_or_else(|| Tensor::from_vec(ctx, &[0.0], &[1])); adamr.step(&mut rp, &[gr]); }
    }
    let r_final = { let x = rp[0].to_vec().await[0]; (x.exp()).ln_1p() };
    (p, if dissipative { r_final } else { 0.0 })
}

// roll the learned dynamics forward under the same u(t); return trajectory MSE vs truth
async fn rollout_mse(ctx: &Arc<ferric_core::Context>, p: &[Tensor], r: f32, naive: bool, pn: &[Tensor]) -> f32 {
    let one = Tensor::from_vec(ctx, &[1.0], &[1]);
    let nb = 64usize; let dt = 0.05f32; let steps = 200usize;
    // batch of test initial conditions
    let mut q: Vec<f32> = randn(nb, 424242, 1.2); let mut pp: Vec<f32> = randn(nb, 434343, 1.2);
    let (mut tq, mut tp) = (q.clone(), pp.clone()); // truth
    let mut se = 0.0f64; let mut cnt = 0.0f64;
    for k in 0..steps {
        let t = k as f32 * dt; let uf = uforce(t);
        // --- model step (semi-implicit Euler) ---
        if naive {
            let mut xf = vec![0.0f32; nb * 3]; for i in 0..nb { xf[i * 3] = q[i]; xf[i * 3 + 1] = pp[i]; xf[i * 3 + 2] = uf; }
            let xv = Var::leaf(Tensor::from_vec(ctx, &xf, &[nb, 3]));
            let pv: Vec<Var> = pn.iter().map(|t| Var::leaf(t.clone())).collect();
            let h = xv.matmul(&pv[0]).add(&pv[1]).exp().add(&Var::leaf(one.clone())).log();
            let d = h.matmul(&pv[2]).add(&pv[3]).value().to_vec().await; // [nb,2] = (dq,dp)
            for i in 0..nb { pp[i] += dt * d[i * 2 + 1]; q[i] += dt * d[i * 2]; }
        } else {
            let mut sf = vec![0.0f32; nb * 2]; for i in 0..nb { sf[i * 2] = q[i]; sf[i * 2 + 1] = pp[i]; }
            let sl = Var::leaf(Tensor::from_vec(ctx, &sf, &[nb, 2]));
            let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
            let (gq, gp) = gradH(ctx, &sl, &pv, &ov);
            let gqv = gq.value().to_vec().await; let gpv = gp.value().to_vec().await;
            for i in 0..nb { let dpi = -gqv[i] - r * gpv[i] + uf; pp[i] += dt * dpi; q[i] += dt * gpv[i]; }
        }
        // --- true step ---
        for i in 0..nb { let dpi = -K * tq[i] - C * tp[i] + uf; tp[i] += dt * dpi; tq[i] += dt * tp[i] / M; }
        for i in 0..nb { se += ((q[i] - tq[i]).powi(2) + (pp[i] - tp[i]).powi(2)) as f64; cnt += 2.0; }
    }
    (se / cnt) as f32
}

async fn train_naive(ctx: &Arc<ferric_core::Context>) -> Vec<Tensor> {
    let one = Tensor::from_vec(ctx, &[1.0], &[1]);
    let mut p = vec![
        Tensor::from_vec(ctx, &randn(3 * HW, 20, 0.6), &[3, HW]), Tensor::zeros(ctx, &[HW]),
        Tensor::from_vec(ctx, &randn(HW * 2, 21, 1.0 / (HW as f32).sqrt()), &[HW, 2]), Tensor::zeros(ctx, &[2]),
    ];
    let mut adam = Adam::new(&p, 0.003); let bs = 256usize;
    for step in 0..4000 {
        let q = randn(bs, step as u32 * 3 + 1, 1.3); let pp = randn(bs, step as u32 * 3 + 2, 1.3);
        let tt: Vec<f32> = (0..bs).map(|i| u(step as u32 * 7 + i as u32, 5) * 18.0).collect();
        let mut xf = vec![0.0f32; bs * 3]; let mut tgt = vec![0.0f32; bs * 2];
        for i in 0..bs { let uf = uforce(tt[i]); xf[i * 3] = q[i]; xf[i * 3 + 1] = pp[i]; xf[i * 3 + 2] = uf;
            tgt[i * 2] = pp[i] / M; tgt[i * 2 + 1] = -K * q[i] - C * pp[i] + uf; }
        let xv = Var::leaf(Tensor::from_vec(ctx, &xf, &[bs, 3]));
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
        let h = xv.matmul(&pv[0]).add(&pv[1]).exp().add(&ov).log();
        let d = h.matmul(&pv[2]).add(&pv[3]);
        let e = d.sub(&Var::leaf(Tensor::from_vec(ctx, &tgt, &[bs, 2])));
        let loss = e.mul(&e).mean_all();
        loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &g);
    }
    p
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — PORT-HAMILTONIAN: energy-accounted model of an actuated, dissipative body");
    println!("  true damped-driven oscillator: m={}, k={}, c={} (damping), motor u(t)=0.5·sin(0.7t)\n", M, K, C);

    let pn = train_naive(&ctx).await;
    let (ph, _) = train_h(&ctx, false).await;          // HNN (conservation-only)
    let (pp_, r) = train_h(&ctx, true).await;          // PHNN (learns dissipation r)

    let mse_naive = rollout_mse(&ctx, &[], 0.0, true, &pn).await;
    let mse_hnn = rollout_mse(&ctx, &ph, 0.0, false, &[]).await;
    let mse_phnn = rollout_mse(&ctx, &pp_, r, false, &[]).await;

    println!("  200-step rollout MSE under the same motor input (lower = better tracks the real body):");
    println!("     naive black-box net      {:.4}", mse_naive);
    println!("     HNN (conservation-only)  {:.4}   ← structurally cannot dissipate", mse_hnn);
    println!("     PHNN (learns friction)   {:.4}", mse_phnn);
    println!("\n  recovered dissipation:  PHNN r = {:.3}   vs   true c/m = {:.3}   (physical parameter recovery)", r, C / M);
    println!("\n  A port-Hamiltonian net models the body's ENERGY BUDGET — friction-loss AND motor-injection —");
    println!("  where a conservation-only HNN can't lose or inject energy. The honest bridge to embodied control.");
}
