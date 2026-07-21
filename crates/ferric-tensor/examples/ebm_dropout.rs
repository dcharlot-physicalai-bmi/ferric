//! EFA energy-first #36 — RICHER PERCEPTION: cart-pole balance through NOISE + OBSERVATION DROPOUT (occlusions).
//!
//! Real sensing isn't just noisy — it drops out (occlusion, missed frames, comms). Here a fraction of timesteps
//! deliver NO observation. This is where perception-as-inference earns its keep: the energy observer COASTS on its
//! learned-dynamics prior through the gaps (predict-only when blind, correct when it sees), while naive finite-diff
//! has nothing to difference (it holds a stale, wrong velocity). We compare, at fixed noise, across dropout rates.
//! If the energy estimate keeps balancing as dropout rises where naive collapses, perception-as-energy-inference is
//! load-bearing under partial observability — the harder, more real perception problem.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_dropout --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;
const HW: usize = 64; const G: f32 = 9.8; const MC: f32 = 1.0; const MP: f32 = 0.1; const L: f32 = 0.5; const DT: f32 = 0.02;
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn nz(seed: u32, sc: f32) -> f32 { let a = u(seed, 1); let b = u(seed, 2); (-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos() * sc }
fn accel(th: f32, thd: f32, f: f32) -> (f32, f32) {
    let (s, c) = (th.sin(), th.cos());
    let temp = (f + MP * L * thd * thd * s) / (MC + MP);
    let thdd = (G * s - c * temp) / (L * (4.0 / 3.0 - MP * c * c / (MC + MP)));
    (temp - MP * L * thdd * c / (MC + MP), thdd)
}
fn control(s: [f32; 4]) -> f32 { 30.0 * s[2] + 6.0 * s[3] } // θ-only (balance; cart may drift)
struct Dyn { w1: Vec<f32>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32> }
impl Dyn {
    fn acc(&self, x: f32, xd: f32, th: f32, thd: f32, f: f32) -> (f32, f32) {
        let inp = [x, xd, th.sin(), th.cos(), thd, f]; let mut hid = [0.0f32; HW];
        for j in 0..HW { let mut pre = self.b1[j]; for k in 0..6 { pre += inp[k] * self.w1[k * HW + j]; } hid[j] = (pre.exp() + 1.0).ln(); }
        let mut o = [self.b2[0], self.b2[1]]; for j in 0..HW { o[0] += hid[j] * self.w2[j * 2]; o[1] += hid[j] * self.w2[j * 2 + 1]; } (o[0], o[1])
    }
    fn step(&self, s: [f32; 4], f: f32) -> [f32; 4] { let (a, b) = self.acc(s[0], s[1], s[2], s[3], f); [s[0] + DT * s[1], s[1] + DT * a, s[2] + DT * s[3], s[3] + DT * b] }
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — RICHER PERCEPTION: cart-pole through NOISE + OBSERVATION DROPOUT\n");
    let mut p = vec![
        Tensor::from_vec(&ctx, &(0..6 * HW).map(|i| (u(i as u32, 7) - 0.5) * 0.4).collect::<Vec<_>>(), &[6, HW]), Tensor::zeros(&ctx, &[HW]),
        Tensor::from_vec(&ctx, &(0..HW * 2).map(|i| (u(i as u32, 9) - 0.5) * (1.0 / (HW as f32).sqrt())).collect::<Vec<_>>(), &[HW, 2]), Tensor::zeros(&ctx, &[2]),
    ];
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let mut adam = Adam::new(&p, 0.003); let bs = 256usize;
    for step in 0..3000 {
        let mut inp = vec![0.0f32; bs * 6]; let mut tgt = vec![0.0f32; bs * 2];
        for i in 0..bs { let sd = step as u32 * 3 + i as u32;
            let x = (u(sd, 1) * 2.0 - 1.0) * 2.0; let xd = (u(sd, 2) * 2.0 - 1.0) * 2.0; let th = (u(sd, 3) * 2.0 - 1.0) * 0.7; let thd = (u(sd, 4) * 2.0 - 1.0) * 2.0; let f = (u(sd, 5) * 2.0 - 1.0) * 15.0;
            let (a, b) = accel(th, thd, f);
            inp[i * 6] = x; inp[i * 6 + 1] = xd; inp[i * 6 + 2] = th.sin(); inp[i * 6 + 3] = th.cos(); inp[i * 6 + 4] = thd; inp[i * 6 + 5] = f; tgt[i * 2] = a; tgt[i * 2 + 1] = b; }
        let xv = Var::leaf(Tensor::from_vec(&ctx, &inp, &[bs, 6]));
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
        let h1 = xv.matmul(&pv[0]).add(&pv[1]).exp().add(&ov).log();
        let e = h1.matmul(&pv[2]).add(&pv[3]).sub(&Var::leaf(Tensor::from_vec(&ctx, &tgt, &[bs, 2])));
        let loss = e.mul(&e).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &g);
    }
    let m = Dyn { w1: p[0].to_vec().await, b1: p[1].to_vec().await, w2: p[2].to_vec().await, b2: p[3].to_vec().await };

    // energy-observer episode with dropout: on a dropped step, PREDICT ONLY (no correction). Returns (RMS, survived).
    let ep_energy = |drop: f32, steps: usize, noise: f32| -> (f32, bool) {
        let mut s = [0.0f32, 0.0, 0.08, 0.0]; let mut est = s; let mut lastf = 0.0f32; let (mut sq, mut alive) = (0.0f32, true);
        for t in 0..steps {
            let pred = m.step(est, lastf);
            let seen = u(t as u32 * 7 + 5, 3) > drop; // observation present?
            est = if seen { let o = [s[0] + nz(t as u32 * 2 + 1, noise), s[2] + nz(t as u32 * 2 + 2, noise)];
                            let (ix, ith) = (o[0] - pred[0], o[1] - pred[2]); [pred[0] + 0.5 * ix, pred[1] + 1.2 * ix, pred[2] + 0.5 * ith, pred[3] + 1.2 * ith] }
                     else { pred }; // COAST on the dynamics prior when blind
            lastf = control(est).clamp(-15.0, 15.0);
            let (a, b) = accel(s[2], s[3], lastf); s = [s[0] + DT * s[1], s[1] + DT * a, s[2] + DT * s[3], s[3] + DT * b];
            sq += s[2] * s[2]; if s[2].abs() > 0.8 { alive = false; }
        }
        ((sq / steps as f32).sqrt(), alive)
    };
    // naive finite-diff with dropout: on a dropped step, hold the last observation (stale) → velocity from stale diff
    let ep_naive = |drop: f32, steps: usize, noise: f32| -> (f32, bool) {
        let mut s = [0.0f32, 0.0, 0.08, 0.0]; let mut prev = [0.0f32, 0.08f32]; let mut lasto = prev; let (mut sq, mut alive) = (0.0f32, true);
        for t in 0..steps {
            let seen = u(t as u32 * 7 + 5, 3) > drop;
            let o = if seen { [s[0] + nz(t as u32 * 2 + 1, noise), s[2] + nz(t as u32 * 2 + 2, noise)] } else { lasto }; // hold stale obs
            let est = [o[0], (o[0] - prev[0]) / DT, o[1], (o[1] - prev[1]) / DT]; prev = o; lasto = o;
            let f = control(est).clamp(-15.0, 15.0);
            let (a, b) = accel(s[2], s[3], f); s = [s[0] + DT * s[1], s[1] + DT * a, s[2] + DT * s[3], s[3] + DT * b];
            sq += s[2] * s[2]; if s[2].abs() > 0.8 { alive = false; }
        }
        ((sq / steps as f32).sqrt(), alive)
    };

    println!("  cart-pole BALANCE, σ=0.08, vs OBSERVATION DROPOUT rate — RMS pole angle & survived:");
    println!("     dropout      ENERGY (coast on dynamics prior)     NAIVE (stale finite-diff)");
    for &dr in &[0.0f32, 0.3, 0.5, 0.7] {
        let (re, se) = ep_energy(dr, 800, 0.08); let (rn, sn) = ep_naive(dr, 800, 0.08);
        println!("     {:>3.0}%         RMS={:>6.3}  survived:{:<5}         RMS={:>6.3}  survived:{}", dr * 100.0, re, se, rn, sn);
    }
    println!("\n  If the energy observer keeps balancing as dropout rises where naive collapses, perception-as-energy-");
    println!("  inference is load-bearing under partial observability — it fills the gaps with the dynamics prior.");
}
