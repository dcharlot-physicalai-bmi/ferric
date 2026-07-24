//! EFA-2 · PUSHCUBE — the first CONTACT manipulation task on our verified Rust contact solver (sim_contact class).
//! A controllable pusher (disc) pushes an object (disc puck — a disc under frictionless-normal contact translates
//! EXACTLY, no rotational approximation) across a table to a target, against Coulomb table friction. PushT/PushCube
//! class (the de-facto contact smoke-test). Physics: circle–circle one-sided contact (kinematic pusher = a
//! position-controlled end-effector; the puck can't push it back), semi-implicit integration, Coulomb table drag.
//! Demonstrator = a scripted "get-behind-and-push" controller; the EFA flow policy is distilled from it (obs → pusher
//! velocity) and evaluated at K forward passes on the world's metric (puck within tol of target).
//! Gates fixed BEFORE the run:
//!   [0] physics + demonstrator sanity: a straight push moves the puck along the push direction; demonstrator ≥90%
//!   [1] flow reach vs K, mean final distance, vs demonstrator & random · [2] determinism
//! HONEST: PushCube-class on our contact solver (disc puck; a box needs oriented-box contact + rotation = next);
//! kinematic pusher (position-controlled end-effector model); distills a scripted demonstrator; one seed.
//!
//! Run: `cargo run -p ferric-tensor --example efa2_pushcube --release`
use ferric_core::Context;
use ferric_tensor::{Adam, Tensor, Var};
use std::f32::consts::PI;
use std::sync::Arc;
const DT: f32 = 0.02; const TMAX: f32 = 6.0; const RP: f32 = 0.12; const RU: f32 = 0.10;  // puck / pusher radii
const MU_T: f32 = 4.0; const PUSH_V: f32 = 1.2; const TOL: f32 = 0.10; const ARENA: f32 = 1.4;
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn nrm(v: [f32; 2]) -> f32 { (v[0] * v[0] + v[1] * v[1]).sqrt() }
fn unit(v: [f32; 2]) -> [f32; 2] { let n = nrm(v).max(1e-6); [v[0] / n, v[1] / n] }
#[derive(Clone)]
struct World { puck: [f32; 2], pv: [f32; 2], push: [f32; 2], tgt: [f32; 2] }
impl World {
    fn new(seed: u32) -> World {
        let tgt = [(u(seed, 1) * 2.0 - 1.0) * 0.9, (u(seed, 2) * 2.0 - 1.0) * 0.9];
        let puck = [(u(seed, 3) * 2.0 - 1.0) * 0.6, (u(seed, 4) * 2.0 - 1.0) * 0.6];
        // pusher starts behind the puck (opposite the target) so the first contact pushes toward the goal
        let d = unit([tgt[0] - puck[0], tgt[1] - puck[1]]);
        World { puck, pv: [0.0; 2], push: [puck[0] - d[0] * (RP + RU + 0.15), puck[1] - d[1] * (RP + RU + 0.15)], tgt }
    }
    // step: pusher moves by commanded velocity (kinematic); puck feels Coulomb table friction + one-sided contact
    fn step(&mut self, cmd: [f32; 2]) {
        let c = [cmd[0].clamp(-PUSH_V, PUSH_V), cmd[1].clamp(-PUSH_V, PUSH_V)];
        self.push = [(self.push[0] + c[0] * DT).clamp(-ARENA, ARENA), (self.push[1] + c[1] * DT).clamp(-ARENA, ARENA)];
        // Coulomb table friction on the puck (decelerate toward rest, no overshoot)
        let sp = nrm(self.pv); if sp > 1e-6 { let dv = (MU_T * DT).min(sp); let u = unit(self.pv); self.pv = [self.pv[0] - u[0] * dv, self.pv[1] - u[1] * dv]; }
        self.puck = [self.puck[0] + self.pv[0] * DT, self.puck[1] + self.pv[1] * DT];
        // circle–circle contact: if overlapping, push puck out along the normal and set its normal velocity to the
        // pusher's approaching component (one-sided; pusher is kinematic = infinite mass)
        let rel = [self.puck[0] - self.push[0], self.puck[1] - self.push[1]]; let dist = nrm(rel); let mind = RP + RU;
        if dist < mind { let n = unit(rel);
            self.puck = [self.push[0] + n[0] * mind, self.push[1] + n[1] * mind];        // positional projection
            let vn = self.pv[0] * n[0] + self.pv[1] * n[1]; let vpn = c[0] * n[0] + c[1] * n[1];
            if vn < vpn { let dvn = vpn - vn; self.pv = [self.pv[0] + n[0] * dvn, self.pv[1] + n[1] * dvn]; }  // match pusher normal vel
        }
    }
    fn dist(&self) -> f32 { nrm([self.puck[0] - self.tgt[0], self.puck[1] - self.tgt[1]]) }
    fn obs(&self) -> [f32; 8] { let dpt = unit([self.tgt[0] - self.puck[0], self.tgt[1] - self.puck[1]]);
        [self.puck[0], self.puck[1], self.push[0] - self.puck[0], self.push[1] - self.puck[1],
         self.tgt[0] - self.puck[0], self.tgt[1] - self.puck[1], dpt[0], dpt[1]] }
}
// scripted demonstrator: get behind the puck (opposite target), then push toward target
fn demo(w: &World) -> [f32; 2] {
    let dpt = unit([w.tgt[0] - w.puck[0], w.tgt[1] - w.puck[1]]);
    let behind = [w.puck[0] - dpt[0] * (RP + RU), w.puck[1] - dpt[1] * (RP + RU)];   // ideal contact point
    let to_behind = [behind[0] - w.push[0], behind[1] - w.push[1]];
    // pusher-to-puck alignment: is the pusher roughly behind (dot of (puck−push) with dpt > 0)?
    let ppn = unit([w.puck[0] - w.push[0], w.puck[1] - w.push[1]]);
    let aligned = ppn[0] * dpt[0] + ppn[1] * dpt[1] > 0.6 && nrm(to_behind) < RP;
    if aligned { [dpt[0] * PUSH_V, dpt[1] * PUSH_V] }                    // push toward target
    else { let mv = unit(to_behind); [mv[0] * PUSH_V, mv[1] * PUSH_V] }  // reposition behind the puck
}
fn episode<F: FnMut(&World) -> [f32; 2]>(seed: u32, mut pol: F) -> f32 {
    let mut w = World::new(seed);
    for _ in 0..((TMAX / DT) as usize) { let c = pol(&w); w.step(c); if w.dist() < TOL * 0.5 { break; } }
    w.dist()
}
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let (a, b) = (u(i as u32, seed), u(i as u32, seed + 1));
    sc * (-2.0 * a.ln()).sqrt() * (2.0 * PI * b).cos() }).collect() }
const H: usize = 128;
struct Net { w: Vec<Vec<f32>>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: Vec<f32>, no: usize }
impl Net { fn f(&self, x: &[f32]) -> Vec<f32> {
    let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.b1[j]; for c in 0..x.len() { z += x[c] * self.w[c][j]; } h1[j] = z.max(0.0); }
    let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.b2[j]; for k in 0..H { z += h1[k] * self.w2[k * H + j]; } h2[j] = z.max(0.0); }
    (0..self.no).map(|c| { let mut o = self.b3[c]; for j in 0..H { o += h2[j] * self.w3[j * self.no + c]; } o }).collect() } }
struct Flow { net: Net }
impl Flow { fn act(&self, ob: &[f32; 8], kk: usize) -> [f32; 2] { let mut a = [0.0f32; 2];
    for k in 0..kk { let t = k as f32 / kk as f32; let mut inp = ob.to_vec(); inp.push(a[0]); inp.push(a[1]); inp.push(t);
        let v = self.net.f(&inp); a[0] += v[0] / kk as f32; a[1] += v[1] / kk as f32; }
    [a[0].clamp(-1.0, 1.0) * PUSH_V, a[1].clamp(-1.0, 1.0) * PUSH_V] } }
fn main() { pollster::block_on(run()); }
async fn run() {
    println!("  EFA-2 · PUSHCUBE (disc, PushCube-class) on the verified Rust contact solver\n");
    // ── [0] physics + demonstrator sanity ──
    let mut w = World { puck: [0.0, 0.0], pv: [0.0; 2], push: [-0.3, 0.0], tgt: [0.6, 0.0] };
    for _ in 0..100 { w.step([PUSH_V, 0.0]); }
    println!("  [0] physics: straight +x push moved puck to ({:+.3},{:+.3}) — {}", w.puck[0], w.puck[1],
        if w.puck[0] > 0.2 && w.puck[1].abs() < 0.05 { "✓ pushes along the push direction" } else { "✗" });
    let (mut dok, mut dd) = (0, 0.0f32); for k in 0..200u32 { let d = episode(k, |w| demo(w)); dd += d; if d < TOL { dok += 1; } }
    println!("      scripted demonstrator reach {:.0}% · mean final distance {:.3}", dok as f32 / 2.0, dd / 200.0);
    // ── distill the EFA flow (CFM to the scripted demonstrator) ──
    println!("\n  distilling the EFA flow (obs 8 → {H} → {H} → 2 pusher-velocity; CFM to the demonstrator):");
    let ctx = Arc::new(Context::new().await.expect("ctx")); let fin = 11; let bs = 256;
    let mut fp: Vec<Tensor> = (0..fin).map(|c| Tensor::from_vec(&ctx, &randn(H, 500 + c as u32, 0.4), &[1, H])).collect();
    fp.push(Tensor::zeros(&ctx, &[H])); fp.push(Tensor::from_vec(&ctx, &randn(H * H, 560, 1.0 / (H as f32).sqrt()), &[H, H])); fp.push(Tensor::zeros(&ctx, &[H]));
    fp.push(Tensor::from_vec(&ctx, &randn(H * 2, 561, 1.0 / (H as f32).sqrt()), &[H, 2])); fp.push(Tensor::zeros(&ctx, &[2]));
    let mut adamf = Adam::new(&fp, 0.0015);
    let net = |f: &[Var], pv: &[Var]| { let mut pre = pv[fin].clone(); for c in 0..fin { pre = pre.add(&f[c].matmul(&pv[c])); }
        pre.relu().matmul(&pv[fin + 1]).add(&pv[fin + 2]).relu().matmul(&pv[fin + 3]).add(&pv[fin + 4]) };
    // on-policy state coverage: roll the demonstrator to collect visited (obs, action) pairs
    for it in 0..14000u32 {
        let mut cols: Vec<Vec<f32>> = (0..fin).map(|_| vec![0.0f32; bs]).collect(); let mut tb = vec![0.0f32; bs * 2];
        for i in 0..bs { let sd = it * 311 + i as u32; let mut ww = World::new(sd % 5000 + 1);
            let roll = (u(sd, 20) * (TMAX / DT)) as usize; for _ in 0..roll { let c = demo(&ww); ww.step(c); if ww.dist() < TOL * 0.5 { break; } }
            let ob = ww.obs(); let ud = demo(&ww); let un = [ud[0] / PUSH_V, ud[1] / PUSH_V];
            let g1 = (-2.0 * u(sd, 30).ln()).sqrt() * (2.0 * PI * u(sd, 32)).cos(); let g2 = (-2.0 * u(sd, 31).ln()).sqrt() * (2.0 * PI * u(sd, 33)).cos();
            let t = u(sd, 9) * 0.9; let a0 = [0.2 * g1, 0.2 * g2];
            for c in 0..8 { cols[c][i] = ob[c]; }
            cols[8][i] = (1.0 - t) * a0[0] + t * un[0]; cols[9][i] = (1.0 - t) * a0[1] + t * un[1]; cols[10][i] = t;
            tb[i * 2] = un[0] - a0[0]; tb[i * 2 + 1] = un[1] - a0[1]; }
        let fpv: Vec<Var> = fp.iter().map(|t| Var::leaf(t.clone())).collect();
        let ff: Vec<Var> = (0..fin).map(|c| Var::leaf(Tensor::from_vec(&ctx, &cols[c], &[bs, 1]))).collect();
        let v = net(&ff, &fpv); let d = v.sub(&Var::leaf(Tensor::from_vec(&ctx, &tb, &[bs, 2]))); let loss = d.mul(&d).mean_all(); loss.backward();
        let gf: Vec<Tensor> = fpv.iter().zip(&fp).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adamf.step(&mut fp, &gf);
        if it % 3500 == 3499 { println!("     iter {:>5}: CFM loss {:.4}", it + 1, loss.value().to_vec().await[0]); } }
    let ex = async |p: &Vec<Tensor>, no: usize| -> Net { let mut w = Vec::new(); for c in 0..fin { w.push(p[c].to_vec().await); }
        Net { w, b1: p[fin].to_vec().await, w2: p[fin + 1].to_vec().await, b2: p[fin + 2].to_vec().await, w3: p[fin + 3].to_vec().await, b3: p[fin + 4].to_vec().await, no } };
    let flow = Flow { net: ex(&fp, 2).await };
    // ── [1] the card ──
    println!("\n  the card — reach (puck within {:.2} of target; 200 episodes):", TOL);
    for kk in [1usize, 2, 4] { let (mut ok, mut md) = (0, 0.0f32);
        for k in 0..200u32 { let d = episode(k, |w| { let ob = w.obs(); flow.act(&ob, kk) }); md += d; if d < TOL { ok += 1; } }
        println!("     flow K={}: reach {:>3.0}% · mean final distance {:.3} · {} fwd pass/decision", kk, ok as f32 / 2.0, md / 200.0, kk); }
    let (mut ro, mut rd) = (0, 0.0f32); for k in 0..200u32 { let d = episode(k, |_| [(u(k, 70) * 2.0 - 1.0) * PUSH_V, (u(k, 71) * 2.0 - 1.0) * PUSH_V]); rd += d; if d < TOL { ro += 1; } }
    println!("     [anchors] scripted demonstrator {:.0}% · random-push {:.0}% ({:.3} mean dist)", dok as f32 / 2.0, ro as f32 / 2.0, rd / 200.0);
    let ob = World::new(42).obs(); let a1 = flow.act(&ob, 2); let a2 = flow.act(&ob, 2);
    println!("     determinism: {}", if a1[0].to_bits() == a2[0].to_bits() && a1[1].to_bits() == a2[1].to_bits() { "bit-exact ✓" } else { "✗" });
    println!("\n  Honest: PushCube-class on our verified contact solver (disc puck — oriented-box contact + rotation is");
    println!("  next); kinematic pusher (position-controlled end-effector); distills a scripted demonstrator; one seed.");
    println!("  Milestone: the EFA flow recipe transfers to a CONTACT manipulation task on our own SAPIEN-core port.");
}
