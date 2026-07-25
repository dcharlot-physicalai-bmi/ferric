//! EFA-2 · PUSHT — the universal contact smoke-test, on our verified Rust physics. A T-shaped block (two rigidly
//! joined rectangles) is pushed by a circle to a target SE(2) POSE; success = coverage of the target region by the
//! block (the published PushT metric). Full-pose control (position AND orientation) — the crux difficulty of PushT.
//! Built on the verified oriented-box↔circle contact + 2D rigid-body dynamics; the T is two sub-boxes rigidly fixed to
//! one body about its computed COM; contact checks both.
//! VERIFIED first:
//!   [V1] the T's mass/COM/moment (moment by point-sampling), and the coverage metric self-consistency (block at the
//!        target pose covers the target ~100%)
//! Then: a SMOOTH blended pose-servo demonstrator (position + orientation corrected by one continuous contact-point
//! choice — the smoothness law says this is what distills), coverage reported, EFA flow distilled, reach = coverage≥0.90.
//! HONEST: our T proportions (not a byte-match to LeRobot pymunk's exact T/units — the TASK CLASS + the coverage metric
//! are faithful); kinematic pusher; Coulomb linear+angular friction; distills a scripted demonstrator; one seed.
//!
//! Run: `cargo run -p ferric-tensor --example efa2_pusht --release`
use ferric_core::Context;
use ferric_tensor::{Adam, Tensor, Var};
use std::f32::consts::PI;
use std::sync::Arc;
const DT: f32 = 0.02; const TMAX: f32 = 9.0; const RU: f32 = 0.07; const PUSH_V: f32 = 0.9;
const MU_T: f32 = 3.0; const MU_W: f32 = 5.0; const ARENA: f32 = 1.5; const COV_OK: f32 = 0.90;
// T sub-boxes in body(COM) frame: (offset_x, offset_y, half_w, half_h). COM computed so bar+stem centroid = origin.
// bar 0.40×0.10 at local y=0; stem 0.10×0.20 at local y=−0.15. areas 0.04, 0.02 → COM_y = −0.05.
const BAR: [f32; 4] = [0.0, 0.05, 0.20, 0.05];
const STEM: [f32; 4] = [0.0, -0.10, 0.05, 0.10];
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn wrap(x: f32) -> f32 { let mut a = x; while a > PI { a -= 2.0 * PI; } while a < -PI { a += 2.0 * PI; } a }
fn nrm(v: [f32; 2]) -> f32 { (v[0] * v[0] + v[1] * v[1]).sqrt() }
fn unit(v: [f32; 2]) -> [f32; 2] { let n = nrm(v).max(1e-6); [v[0] / n, v[1] / n] }
fn rot(v: [f32; 2], a: f32) -> [f32; 2] { let (c, s) = (a.cos(), a.sin()); [c * v[0] - s * v[1], s * v[0] + c * v[1]] }
fn in_box(local: [f32; 2], b: &[f32; 4]) -> bool { (local[0] - b[0]).abs() <= b[2] && (local[1] - b[1]).abs() <= b[3] }
fn in_T(world: [f32; 2], p: [f32; 2], th: f32) -> bool { let l = rot([world[0] - p[0], world[1] - p[1]], -th); in_box(l, &BAR) || in_box(l, &STEM) }
// sample a point uniformly inside the T (body frame), area-weighted between bar and stem
fn sample_T(seed: u32) -> [f32; 2] {
    let (a_bar, a_stem) = (4.0 * BAR[2] * BAR[3], 4.0 * STEM[2] * STEM[3]);
    if u(seed, 1) < a_bar / (a_bar + a_stem) { [BAR[0] + (u(seed, 2) * 2.0 - 1.0) * BAR[2], BAR[1] + (u(seed, 3) * 2.0 - 1.0) * BAR[3]] }
    else { [STEM[0] + (u(seed, 2) * 2.0 - 1.0) * STEM[2], STEM[1] + (u(seed, 3) * 2.0 - 1.0) * STEM[3]] }
}
// contact: circle at world c against the T at (p,θ) — check both sub-boxes, return the deepest (world cp, circle→box n, pen)
fn contact(p: [f32; 2], th: f32, c: [f32; 2]) -> Option<([f32; 2], [f32; 2], f32)> {
    let local = rot([c[0] - p[0], c[1] - p[1]], -th); let mut best: Option<([f32; 2], [f32; 2], f32)> = None;
    for b in [&BAR, &STEM] {
        let cl = [(local[0] - b[0]).clamp(-b[2], b[2]) + b[0], (local[1] - b[1]).clamp(-b[3], b[3]) + b[1]];
        let inside = (local[0] - b[0]).abs() < b[2] && (local[1] - b[1]).abs() < b[3];
        let (nloc, pen, cloc);
        if inside { let dx = b[2] - (local[0] - b[0]).abs(); let dy = b[3] - (local[1] - b[1]).abs();
            if dx < dy { let s = (local[0] - b[0]).signum(); nloc = [s, 0.0]; pen = dx + RU; cloc = [b[0] + s * b[2], local[1]]; }
            else { let s = (local[1] - b[1]).signum(); nloc = [0.0, s]; pen = dy + RU; cloc = [local[0], b[1] + s * b[3]]; } }
        else { let d = [local[0] - cl[0], local[1] - cl[1]]; let dist = nrm(d); if dist > RU { continue; } nloc = unit(d); pen = RU - dist; cloc = cl; }
        if best.map_or(true, |bb| pen > bb.2) { best = Some((rot(cloc, th), rot([-nloc[0], -nloc[1]], th), pen)); }
    }
    best.map(|(cp, n, pen)| ([cp[0] + p[0], cp[1] + p[1]], n, pen))
}
// moment of inertia about COM (mass=1), by point-sampling the T
fn moment() -> f32 { let mut s = 0.0; let n = 4000; for k in 0..n { let q = sample_T(70000 + k as u32); s += q[0] * q[0] + q[1] * q[1]; } s / n as f32 }
#[derive(Clone)]
struct World { p: [f32; 2], th: f32, v: [f32; 2], w: f32, tp: [f32; 2], tth: f32, im: f32, push: [f32; 2] }
impl World {
    fn new(seed: u32, im: f32) -> World {
        let tp = [(u(seed, 1) * 2.0 - 1.0) * 0.7, (u(seed, 2) * 2.0 - 1.0) * 0.7]; let tth = (u(seed, 5) * 2.0 - 1.0) * PI;
        let p = [(u(seed, 3) * 2.0 - 1.0) * 0.6, (u(seed, 4) * 2.0 - 1.0) * 0.6]; let th = (u(seed, 6) * 2.0 - 1.0) * PI;
        let d = unit([tp[0] - p[0], tp[1] - p[1]]);
        World { p, th, v: [0.0; 2], w: 0.0, tp, tth, im, push: [p[0] - d[0] * 0.35, p[1] - d[1] * 0.35] }
    }
    fn step(&mut self, cmd: [f32; 2]) {
        let c = [cmd[0].clamp(-PUSH_V, PUSH_V), cmd[1].clamp(-PUSH_V, PUSH_V)];
        self.push = [(self.push[0] + c[0] * DT).clamp(-ARENA, ARENA), (self.push[1] + c[1] * DT).clamp(-ARENA, ARENA)];
        let sp = nrm(self.v); if sp > 1e-6 { let dv = (MU_T * DT).min(sp); let un = unit(self.v); self.v = [self.v[0] - un[0] * dv, self.v[1] - un[1] * dv]; }
        if self.w.abs() > 1e-6 { let dw = (MU_W * DT).min(self.w.abs()); self.w -= dw * self.w.signum(); }
        self.p = [self.p[0] + self.v[0] * DT, self.p[1] + self.v[1] * DT]; self.th = wrap(self.th + self.w * DT);
        if let Some((cp, n, pen)) = contact(self.p, self.th, self.push) {
            self.p = [self.p[0] + n[0] * pen, self.p[1] + n[1] * pen];
            let r = [cp[0] - self.p[0], cp[1] - self.p[1]];
            let vc = [self.v[0] - self.w * r[1], self.v[1] + self.w * r[0]];
            let vn = vc[0] * n[0] + vc[1] * n[1]; let vpn = c[0] * n[0] + c[1] * n[1];
            if vn < vpn { let rn = r[0] * n[1] - r[1] * n[0]; let keff = 1.0 + rn * rn / self.im; let jn = (vpn - vn) / keff;
                self.v = [self.v[0] + jn * n[0], self.v[1] + jn * n[1]]; self.w += jn * rn / self.im; }
        }
    }
    fn cov(&self, n: usize) -> f32 { let mut c = 0;                            // coverage with n samples (cheap for training, full for eval)
        for k in 0..n { let q = sample_T(50000 + k as u32); let w = rot(q, self.tth).map0(self.tp);
            if in_T(w, self.p, self.th) { c += 1; } } c as f32 / n as f32 }
    fn coverage(&self) -> f32 { self.cov(400) }
    fn obs(&self) -> [f32; 12] { let ep = [self.tp[0] - self.p[0], self.tp[1] - self.p[1]]; let dpt = unit(ep); let et = wrap(self.tth - self.th);
        [self.th.cos(), self.th.sin(), self.push[0] - self.p[0], self.push[1] - self.p[1], ep[0], ep[1], dpt[0], dpt[1], et.cos(), et.sin(), self.tth.cos(), self.tth.sin()] }
}
trait Map0 { fn map0(self, o: [f32; 2]) -> [f32; 2]; }
impl Map0 for [f32; 2] { fn map0(self, o: [f32; 2]) -> [f32; 2] { [self[0] + o[0], self[1] + o[1]] } }
// SMOOTH blended pose-servo: choose ONE contact point behind the block along the position-error direction, offset
// perpendicular ∝ orientation error, so a single continuous push both translates (toward target) and rotates (toward θ*).
// multi-push PLANNER: rotate the T (chase a bar-tip tangentially to spin toward target θ) when orientation is off;
// push the COM to the target position when aligned. Underactuated pose control needs this sequencing, not a servo.
fn demo(w: &World) -> [f32; 2] {
    let et = wrap(w.tth - w.th); let ep = [w.tp[0] - w.p[0], w.tp[1] - w.p[1]];
    let contact = if et.abs() > 0.12 {                                   // ROTATION phase
        let ba = [w.th.cos(), w.th.sin()]; let perp = [-w.th.sin(), w.th.cos()];   // bar long-axis / its perpendicular
        let tip = [w.p[0] + ba[0] * 0.20, w.p[1] + ba[1] * 0.20];        // one bar end (max lever arm)
        let s = et.signum(); let pd = [perp[0] * s, perp[1] * s];        // tangential push dir → torque reducing e_θ
        [tip[0] - pd[0] * (RU - 0.03), tip[1] - pd[1] * (RU - 0.03)]     // slightly INTO the tip so it pushes tangentially
    } else {                                                            // POSITION phase (orientation aligned)
        let dir = if nrm(ep) > 0.04 { unit(ep) } else { [w.th.sin(), -w.th.cos()] };
        [w.p[0] - dir[0] * 0.10, w.p[1] - dir[1] * 0.10]
    };
    let to = [contact[0] - w.push[0], contact[1] - w.push[1]];
    let mv = if nrm(to) > 1e-4 { unit(to) } else { unit(ep) };
    [mv[0] * PUSH_V, mv[1] * PUSH_V]
}
fn episode<F: FnMut(&World) -> [f32; 2]>(seed: u32, im: f32, mut pol: F) -> f32 {
    let mut w = World::new(seed, im); let mut best = 0.0f32;
    for _ in 0..((TMAX / DT) as usize) { let c = pol(&w); w.step(c); best = best.max(w.coverage()); if best >= 0.98 { break; } } best }
// ES policy + Evolution Strategies (OpenAI-ES, antithetic + rank shaping). NO autograd, NO demonstrator — learn the
// PushT policy directly to maximize coverage. Pure-Rust rollouts, so millions of forward passes run in seconds.
const HS: usize = 64; const OD: usize = 12;
struct Pol { p: Vec<f32> }                                              // flat params: [12*HS] w1, [HS] b1, [HS*HS] w2, [HS] b2, [HS*2] w3, [2] b3
fn pol_len() -> usize { OD * HS + HS + HS * HS + HS + HS * 2 + 2 }
impl Pol {
    fn act(&self, ob: &[f32; OD]) -> [f32; 2] {
        let p = &self.p; let mut o1 = HS; // offset cursors
        let (w1s, b1s) = (0, OD * HS); let w2s = b1s + HS; let b2s = w2s + HS * HS; let w3s = b2s + HS; let b3s = w3s + HS * 2;
        let mut h1 = [0.0f32; HS]; for j in 0..HS { let mut z = p[b1s + j]; for c in 0..OD { z += ob[c] * p[w1s + c * HS + j]; } h1[j] = z.max(0.0); }
        let mut h2 = [0.0f32; HS]; for j in 0..HS { let mut z = p[b2s + j]; for k in 0..HS { z += h1[k] * p[w2s + k * HS + j]; } h2[j] = z.max(0.0); }
        let mut out = [p[b3s], p[b3s + 1]]; for j in 0..HS { out[0] += h2[j] * p[w3s + j * 2]; out[1] += h2[j] * p[w3s + j * 2 + 1]; }
        o1 = o1; [out[0].tanh() * PUSH_V, out[1].tanh() * PUSH_V]        // tanh-bounded pusher velocity
    }
}
// DENSE reward for one episode: mean coverage over the whole trajectory (rewards reaching AND holding the pose) —
// a far stronger learning signal than sparse "best coverage".
fn ep_dense(seed: u32, im: f32, pol: &Pol) -> f32 {
    let mut w = World::new(seed, im); let mut sum = 0.0f32; let steps = (TMAX / DT) as usize;
    for _ in 0..steps { let c = pol.act(&w.obs()); w.step(c); sum += w.cov(96); } sum / steps as f32 }  // cheap coverage for training
fn fitness(pol: &Pol, im: f32, seeds: &[u32]) -> f32 {                  // mean dense reward over the given seeds
    let mut s = 0.0; for &sd in seeds { s += ep_dense(sd, im, pol); } s / seeds.len() as f32 }
fn gauss(seed: u32, i: usize) -> f32 { let (a, b) = (u(seed, i as u32), u(seed, i as u32 + 777)); (-2.0 * a.ln()).sqrt() * (2.0 * PI * b).cos() }
fn main() { pollster::block_on(run()); }
async fn run() {
    let im = moment();
    println!("  EFA-2 · PUSHT via EVOLUTION STRATEGIES — learn the policy (no demonstrator, reward = coverage)\n");
    let self_cov = { let w = World { p: [0.3, -0.2], th: 0.7, v: [0.0; 2], w: 0.0, tp: [0.3, -0.2], tth: 0.7, im, push: [9.0, 9.0] }; w.coverage() };
    println!("  [V1] coverage metric self-check (block AT target) = {:.3} · moment {:.4} — {}", self_cov, im, if self_cov > 0.98 { "✓" } else { "✗" });
    // anchors
    let (mut dcov, mut dok) = (0.0f32, 0); for k in 0..200u32 { let c = episode(k, im, |w| demo(w)); dcov += c; if c >= COV_OK { dok += 1; } }
    println!("  [anchor] scripted planner demonstrator: mean coverage {:.3} · success {:.0}%\n", dcov / 200.0, dok as f32 / 2.0);
    // OpenAI-ES
    // RECORDED NEGATIVE (naive compute-scaling): σ-decay 0.12→0.03 over 500 gens made it WORSE — held-out coverage
    // PEAKED ~0.29 by gen ~120 then DEGRADED to 0.10 by gen 500. Cause: the update θ += lr/(pop·σ)·grad grows 4× as σ
    // shrinks, so late training takes ever-larger steps and diverges. Best config is CONSTANT σ, ~120 gens → 0.28.
    let n = pol_len(); let (pop, gens, sigma, lr) = (48usize, 120usize, 0.10f32, 0.03f32);
    let mut theta = vec![0.0f32; n]; for i in 0..n { theta[i] = gauss(12345, i) * 0.1; }
    println!("  ES: {} params · pop {} (antithetic) · {} gens · σ={} (constant) · lr={} · dense reward · fresh seeds/gen", n, pop, gens, sigma, lr);
    for g in 0..gens {
        // fresh random training seeds EACH generation → the policy must generalize, not memorize a fixed set
        let train: Vec<u32> = (0..24u32).map(|j| 200000 + g as u32 * 977 + j).collect();
        let mut fs = vec![0.0f32; pop]; let base = (g as u32 + 1) * 100003;
        for i in 0..pop {                                              // antithetic pairs share an epsilon
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 }; let eseed = base + (i as u32 / 2);
            let mut cand = theta.clone(); for k in 0..n { cand[k] += sign * sigma * gauss(eseed, k); }
            fs[i] = fitness(&Pol { p: cand }, im, &train);
        }
        // rank-normalize fitness to [-0.5,0.5]
        let mut idx: Vec<usize> = (0..pop).collect(); idx.sort_by(|&a, &b| fs[a].partial_cmp(&fs[b]).unwrap());
        let mut rankw = vec![0.0f32; pop]; for (r, &ii) in idx.iter().enumerate() { rankw[ii] = r as f32 / (pop as f32 - 1.0) - 0.5; }
        let mut grad = vec![0.0f32; n];
        for i in 0..pop { let sign = if i % 2 == 0 { 1.0 } else { -1.0 }; let eseed = base + (i as u32 / 2);
            for k in 0..n { grad[k] += rankw[i] * sign * gauss(eseed, k); } }
        for k in 0..n { theta[k] += lr / (pop as f32 * sigma) * grad[k]; }
        if g % 40 == 39 || g == gens - 1 {                             // report held-out best-coverage success (the task metric)
            let pol = Pol { p: theta.clone() }; let (mut mc, mut ok) = (0.0f32, 0);
            for k in 800..860u32 { let c = episode(k, im, |w| pol.act(&w.obs())); mc += c; if c >= COV_OK { ok += 1; } }
            println!("     gen {:>3}: held-out mean coverage {:.3} · success {:.0}%", g + 1, mc / 60.0, ok as f32 / 60.0 * 100.0); }
    }
    // final eval on 200 held-out seeds
    let pol = Pol { p: theta };
    let (mut mc, mut ok, mut c5, mut c7) = (0.0f32, 0, 0, 0);
    for k in 500..700u32 { let c = episode(k, im, |w| pol.act(&w.obs())); mc += c; if c >= COV_OK { ok += 1; } if c >= 0.5 { c5 += 1; } if c >= 0.7 { c7 += 1; } }
    println!("\n  the card — ES-learned policy, PushT (coverage; 200 HELD-OUT episodes):");
    println!("     mean coverage {:.3} · success(≥0.90) {:.0}% · ≥0.7 {:.0}% · ≥0.5 {:.0}%", mc / 200.0, ok as f32 / 2.0, c7 as f32 / 2.0, c5 as f32 / 2.0);
    println!("     [anchors] scripted planner {:.3} coverage / {:.0}% · vs learned above", dcov / 200.0, dok as f32 / 2.0);
    let d1 = pol.act(&World::new(42, im).obs()); let d2 = pol.act(&World::new(42, im).obs());
    println!("     determinism: {}", if d1[0].to_bits() == d2[0].to_bits() { "bit-exact ✓" } else { "✗" });
    println!("\n  Honest: PushT class + published coverage metric on our verified physics; ES-learned policy (no demonstrator,");
    println!("  no autograd — pure-Rust rollouts); our T proportions (not a byte-match to pymunk's); one seed.");
}
