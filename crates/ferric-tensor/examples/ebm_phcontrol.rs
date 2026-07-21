//! EFA energy-first #51 — INGEST port-Hamiltonian / IDA-PBC, Step 3: earn the certificate STRUCTURALLY (by construction).
//!
//! Step 2 (learned neural-Lyapunov) FAILED to earn a certificate (0% certified region). This earns it a different way:
//! constrain the energy to the mechanical port-Hamiltonian form E(θ,ω;g)=V_φ(θ;g)+½ω² (learned potential + kinetic),
//! fix canonical J=[[0,1],[-1,0]] and PSD damping R=diag(0,r). The energy-shaping (IDA-PBC) controller that makes the
//! closed loop ẋ=(J−R)∂E is, for this collocated pendulum, u = sinθ − dV/dθ + (0.05−r)ω (dV/dθ via autograd on the
//! learned potential). Then dE/dt = ∂Eᵀẋ = −r·ω² ≤ 0 BY CONSTRUCTION — a certificate, not a verification. We measure the
//! monotone-descent fraction (should be ~100%, vs Step-2's learned 50%) and reach. HONEST: model-based (uses the known
//! sinθ), collocated/mechanical only, and dE/dt is negative-SEMI-definite (needs LaSalle for asymptotic convergence).
//!
//! Run: `cargo run -p ferric-tensor --example ebm_phcontrol --release`
use ferric_tensor::{grad, Adam, Tensor, Var};
use std::sync::Arc;
const H: usize = 32; const DT: f32 = 0.05; const UMAX: f32 = 6.0; const R: f32 = 0.6;
const TESTG: [f32; 4] = [-1.5, -0.5, 0.5, 1.5];
use std::f32::consts::PI;
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }
fn wrap(x: f32) -> f32 { let mut a = x; while a > PI { a -= 2.0 * PI; } while a < -PI { a += 2.0 * PI; } a }
fn stepu(th: f32, om: f32, uu: f32) -> (f32, f32) { let no = om + DT * (-th.sin() - 0.05 * om + uu.clamp(-UMAX, UMAX)); (wrap(th + DT * no), no) }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — INGEST port-Hamiltonian/IDA-PBC: earn the certificate STRUCTURALLY (dE/dt=−r·ω² by construction)\n");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let bs = 256usize;
    // learned potential V_φ(θ;g) = softplus-MLP over [cos(θ−g), sin(θ−g)]  (min shaped at θ=g)
    let vnet = |rc: &Var, rs: &Var, pv: &[Var], ov: &Var| {
        let sp = |z: Var| z.exp().add(ov).log();
        let pre = rc.matmul(&pv[0]).add(&rs.matmul(&pv[1])).add(&pv[2]);
        sp(sp(pre).matmul(&pv[3]).add(&pv[4])).matmul(&pv[5]).add(&pv[6])
    };
    let mut p = vec![
        Tensor::from_vec(&ctx, &randn(H, 41, 0.6), &[1, H]), Tensor::from_vec(&ctx, &randn(H, 42, 0.6), &[1, H]), Tensor::zeros(&ctx, &[H]),
        Tensor::from_vec(&ctx, &randn(H * H, 43, 1.0 / (H as f32).sqrt()), &[H, H]), Tensor::zeros(&ctx, &[H]),
        Tensor::from_vec(&ctx, &randn(H, 44, 1.0 / (H as f32).sqrt()), &[H, 1]), Tensor::zeros(&ctx, &[1]),
    ];
    let mut adam = Adam::new(&p, 0.003);
    // train V_φ toward a valid potential with min at g: target P(θ;g)=1−cos(θ−g)
    for it in 0..8000 {
        let (mut rc, mut rs, mut tb) = (vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs]);
        for i in 0..bs { let sd = it as u32 * 7 + i as u32; let th = (u(sd, 1) * 2.0 - 1.0) * PI; let g = (u(sd, 3) * 2.0 - 1.0) * 2.0; let d = th - g; rc[i] = d.cos(); rs[i] = d.sin(); tb[i] = 1.0 - d.cos(); }
        let l = |v: &[f32]| Var::leaf(Tensor::from_vec(&ctx, v, &[bs, 1])); let ov = Var::leaf(one.clone());
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
        let e = vnet(&l(&rc), &l(&rs), &pv, &ov);
        let diff = e.sub(&Var::leaf(Tensor::from_vec(&ctx, &tb, &[bs, 1]))); let loss = diff.mul(&diff).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &g);
    }
    let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());

    // CPU potential eval (for the energy E=V_φ+½ω² monotonicity check)
    let p0: Vec<f32> = p[0].to_vec().await; let p1: Vec<f32> = p[1].to_vec().await; let pb1: Vec<f32> = p[2].to_vec().await;
    let w2: Vec<f32> = p[3].to_vec().await; let pb2: Vec<f32> = p[4].to_vec().await; let w3: Vec<f32> = p[5].to_vec().await; let pb3 = p[6].to_vec().await[0];
    let vphi = |th: f32, g: f32| -> f32 { let d = th - g; let (rc, rs) = (d.cos(), d.sin());
        let mut h1 = [0.0f32; H]; for j in 0..H { let z = pb1[j] + rc * p0[j] + rs * p1[j]; h1[j] = (z.exp() + 1.0).ln(); }
        let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = pb2[j]; for k in 0..H { z += h1[k] * w2[k * H + j]; } h2[j] = (z.exp() + 1.0).ln(); }
        let mut o = pb3; for j in 0..H { o += h2[j] * w3[j]; } o };
    let energy = |th: f32, om: f32, g: f32| vphi(th, g) + 0.5 * om * om;

    // eval: reach% + CERTIFY (fraction of steps with E non-increasing) across test goals
    let nep = 60usize; let ng = TESTG.len(); let n = nep * ng;
    let mut th = vec![0.0f32; n]; let mut om = vec![0.0f32; n]; let mut gg = vec![0.0f32; n];
    for gi in 0..ng { for e in 0..nep { let idx = gi * nep + e; th[idx] = (u(900 + idx as u32, 7) * 2.0 - 1.0) * PI; gg[idx] = TESTG[gi]; } }
    let (mut reach, mut desc, mut dtot) = (vec![true; n], 0.0f32, 0.0f32);
    for t in 0..260 {
        // energy-shaping control: dV/dθ via autograd on the learned potential; u = sinθ − dV/dθ + (0.05−R)ω
        let thv = Var::leaf(Tensor::from_vec(&ctx, &th, &[n, 1])); let gv = Var::leaf(Tensor::from_vec(&ctx, &gg, &[n, 1]));
        let d = thv.sub(&gv); let vphi_v = vnet(&d.cos(), &d.sin(), &pv, &ov);
        let gd = grad(&vphi_v.sum_all(), &[thv.clone()], None); let dvdth = gd[0].value().to_vec().await;
        let uu: Vec<f32> = (0..n).map(|i| (th[i].sin() - dvdth[i] + (0.05 - R) * om[i]).clamp(-UMAX, UMAX)).collect();
        for i in 0..n { let e0 = energy(th[i], om[i], gg[i]); let (nt, no) = stepu(th[i], om[i], uu[i]); let e1 = energy(nt, no, gg[i]);
            if wrap(th[i] - gg[i]).abs() > 0.05 || om[i].abs() > 0.05 { dtot += 1.0; if e1 <= e0 + 1e-6 { desc += 1.0; } }   // monotone-descent fraction away from equilibrium
            th[i] = nt; om[i] = no; if t >= 220 && !(wrap(th[i] - gg[i]).abs() < 0.3 && om[i].abs() < 0.6) { reach[i] = false; } }
    }
    let r = reach.iter().filter(|&&b| b).count() as f32 / n as f32 * 100.0; let cert = desc / dtot * 100.0;

    println!("  E(θ,ω;g) = V_φ(θ;g) + ½ω²  (learned potential + kinetic); controller = IDA-PBC energy shaping, r={:.1}.", R);
    println!("  eval: {} episodes × {} goals.\n", nep, ng);
    println!("     control-reach (within 0.3 rad / 0.6 rad·s):   {:>5.1}%", r);
    println!("     STRUCTURAL CERTIFICATE — energy monotone-descent fraction (dE ≤ 0 away from g):   {:>5.1}%", cert);
    println!("     (compare Step-2 learned neural-Lyapunov: 0% certified region, 50% on-policy monitor)\n");
    println!("  Reading: dE/dt = −r·ω² ≤ 0 holds BY CONSTRUCTION, so the descent fraction should be ~100% (small gaps = Euler");
    println!("  discretization only) — the certificate STRUCTURE earns what the learned Lyapunov (Step 2) could not.");
    println!("  HONEST: model-based (uses known sinθ), collocated/mechanical body only, dE/dt is negative-SEMI-definite");
    println!("  (LaSalle needed for asymptotic convergence); potential learned by regression to 1−cos(θ−g) here.");
}
