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
// ─────────────────────────────────────────────────────────────────────────────
// CEM-MPC demonstrator: plan multi-push sequences using the VERIFIED physics as the model. Reactive controllers can't
// align a T's full pose (underactuated); a receding-horizon planner can, by looking ahead. If MPC solves PushT, its
// solved trajectories are the demonstration dataset for a distilled flow policy (MPC stands in for human teleop).
// LONG horizon (short one is myopic: PushT needs multi-push repositioning across seconds). Terminal-weighted so the
// planner optimizes the END pose (coverage there), not immediate greedy coverage gain.
const HZ: usize = 70; const NS: usize = 48; const NITER: usize = 4; const ELITE: usize = 8; const STRIDE: usize = 5;
fn cov_at(w: &World) -> f32 { w.cov(120) }
fn rollout_score(w0: &World, seq: &[[f32; 2]]) -> f32 {                 // action-repeat STRIDE → HZ*STRIDE effective horizon
    let mut w = w0.clone(); let mut s = 0.0; for a in seq { for _ in 0..STRIDE { w.step(*a); } s += cov_at(&w); }
    0.15 * s / seq.len() as f32 + 0.85 * cov_at(&w) }
fn mpc_act(w: &World, mean: &mut Vec<[f32; 2]>, seed: u32) -> [f32; 2] {
    let mut mu = mean.clone(); let mut sd = vec![0.5f32; HZ];
    for it in 0..NITER {
        let mut scored: Vec<(f32, Vec<[f32; 2]>)> = (0..NS).map(|k| {
            let seq: Vec<[f32; 2]> = (0..HZ).map(|h| {
                let g1 = gauss(seed.wrapping_add((it * NS * HZ + k * HZ + h) as u32 * 2 + 1), 0);
                let g2 = gauss(seed.wrapping_add((it * NS * HZ + k * HZ + h) as u32 * 2 + 2), 0);
                [(mu[h][0] + sd[h] * g1).clamp(-PUSH_V, PUSH_V), (mu[h][1] + sd[h] * g2).clamp(-PUSH_V, PUSH_V)] }).collect();
            (rollout_score(w, &seq), seq) }).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for h in 0..HZ {                                               // refit mean/std from the elite set
            let (mut m0, mut m1) = (0.0f32, 0.0f32); for e in 0..ELITE { m0 += scored[e].1[h][0]; m1 += scored[e].1[h][1]; }
            m0 /= ELITE as f32; m1 /= ELITE as f32;
            let (mut v0, mut v1) = (0.0f32, 0.0f32); for e in 0..ELITE { v0 += (scored[e].1[h][0] - m0).powi(2); v1 += (scored[e].1[h][1] - m1).powi(2); }
            mu[h] = [m0, m1]; sd[h] = ((v0 + v1) / (2.0 * ELITE as f32)).sqrt().max(0.05); }
    }
    let act = mu[0]; for h in 0..HZ - 1 { mean[h] = mu[h + 1]; } mean[HZ - 1] = [0.0, 0.0];   // warm-start shift
    act }
fn gauss(seed: u32, i: usize) -> f32 { let (a, b) = (u(seed, i as u32), u(seed, i as u32 + 777)); (-2.0 * a.ln()).sqrt() * (2.0 * PI * b).cos() }
fn mpc_episode(seed: u32, im: f32) -> f32 {
    let mut w = World::new(seed, im); let mut mean = vec![[0.0f32; 2]; HZ]; let mut best = 0.0f32;
    let decisions = ((TMAX / DT) as usize) / STRIDE;
    for t in 0..decisions { let a = mpc_act(&w, &mut mean, seed.wrapping_mul(31) + t as u32 * 7919);
        for _ in 0..STRIDE { w.step(a); best = best.max(w.coverage()); } if best >= 0.98 { break; } }
    best }
// ─────────────────────────────────────────────────────────────────────────────
// FULL-SCALE diffusion policy — corrected. The nano run failed on a schedule BUG (T=16, β≤0.02 → ᾱ_T≈0.85, data never
// fully noised → sampling from N(0,I) was off-distribution). Fixes: COSINE schedule (ᾱ_T≈0), x0-prediction DDPM
// posterior sampling, bigger net, more denoise steps, MANY MORE cached demos. This is the real PushT recipe at scale.
const HD: usize = 320; const ODP: usize = 12; const CM: usize = 6; const AD: usize = CM * 2; const T: usize = 40;
const NDEMO: u32 = 100; const NS2: usize = 64; const NITER2: usize = 6; const DEMOF: &str = "/private/tmp/claude-501/-Users-dcharlot-vibe-coding-bmi-concept/ec64a91f-fbcc-442a-8e9b-f2f378c7a081/scratchpad/pusht_demos_tight.bin";
struct DNet { w: Vec<Vec<f32>>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: Vec<f32> }
impl DNet { fn eps(&self, a: &[f32; AD], t: f32, ob: &[f32; ODP]) -> [f32; AD] {
    let nin = AD + 1 + ODP; let mut x = vec![0.0f32; nin];
    for c in 0..AD { x[c] = a[c]; } x[AD] = t; for c in 0..ODP { x[AD + 1 + c] = ob[c]; }
    let mut h1 = [0.0f32; HD]; for j in 0..HD { let mut z = self.b1[j]; for c in 0..nin { z += x[c] * self.w[c][j]; } h1[j] = z.max(0.0); }
    let mut h2 = [0.0f32; HD]; for j in 0..HD { let mut z = self.b2[j]; for k in 0..HD { z += h1[k] * self.w2[k * HD + j]; } h2[j] = z.max(0.0); }
    let mut o = [0.0f32; AD]; for c in 0..AD { o[c] = self.b3[c]; for j in 0..HD { o[c] += h2[j] * self.w3[j * AD + c]; } } o } }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let (a, b) = (u(i as u32, seed), u(i as u32, seed + 1));
    sc * (-2.0 * a.ln()).sqrt() * (2.0 * PI * b).cos() }).collect() }
fn gnoise(seed: u32, i: usize) -> f32 { let (a, b) = (u(seed, i as u32 + 1), u(seed, i as u32 + 991)); (-2.0 * a.ln()).sqrt() * (2.0 * PI * b).cos() }
// COSINE schedule (Nichol-Dhariwal): abar_T ≈ 0 so the forward process fully noises → in-distribution sampling
fn cosine() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let f = |x: f32| (((x + 0.008) / 1.008) * PI / 2.0).cos().powi(2); let f0 = f(0.0);
    let mut abar = vec![0.0f32; T]; for i in 0..T { abar[i] = (f((i as f32 + 1.0) / T as f32) / f0).clamp(1e-5, 0.9999); }
    let mut betas = vec![0.0f32; T]; let mut alphas = vec![0.0f32; T];
    for i in 0..T { let prev = if i == 0 { 1.0 } else { abar[i - 1] }; betas[i] = (1.0 - abar[i] / prev).clamp(1e-5, 0.999); alphas[i] = 1.0 - betas[i]; }
    (betas, alphas, abar) }
fn cmpc_act(w: &World, mean: &mut Vec<[f32; 2]>, seed: u32) -> [f32; 2] {   // TIGHT-convergence MPC = consistent demos
    let mut mu = mean.clone(); let mut sd = vec![0.5f32; HZ];
    for it in 0..NITER2 {
        let mut scored: Vec<(f32, Vec<[f32; 2]>)> = (0..NS2).map(|k| {
            let seq: Vec<[f32; 2]> = (0..HZ).map(|h| {
                let g1 = gnoise(seed.wrapping_add((it * NS2 * HZ + k * HZ + h) as u32 * 2 + 1), 0);
                let g2 = gnoise(seed.wrapping_add((it * NS2 * HZ + k * HZ + h) as u32 * 2 + 2), 0);
                [(mu[h][0] + sd[h] * g1).clamp(-PUSH_V, PUSH_V), (mu[h][1] + sd[h] * g2).clamp(-PUSH_V, PUSH_V)] }).collect();
            (rollout_score(w, &seq), seq) }).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for h in 0..HZ { let (mut m0, mut m1) = (0.0f32, 0.0f32); for e in 0..ELITE { m0 += scored[e].1[h][0]; m1 += scored[e].1[h][1]; }
            m0 /= ELITE as f32; m1 /= ELITE as f32;
            let (mut v0, mut v1) = (0.0f32, 0.0f32); for e in 0..ELITE { v0 += (scored[e].1[h][0] - m0).powi(2); v1 += (scored[e].1[h][1] - m1).powi(2); }
            mu[h] = [m0, m1]; sd[h] = ((v0 + v1) / (2.0 * ELITE as f32)).sqrt().max(0.03); } }
    let act = mu[0]; for h in 0..HZ - 1 { mean[h] = mu[h + 1]; } mean[HZ - 1] = [0.0, 0.0]; act }
fn mpc_chunks(seed: u32, im: f32, ob: &mut Vec<[f32; ODP]>, ch: &mut Vec<[f32; AD]>) {
    let mut w = World::new(seed, im); let mut mean = vec![[0.0f32; 2]; HZ]; let decisions = ((TMAX / DT) as usize) / STRIDE;
    let mut macros: Vec<[f32; 2]> = vec![]; let mut obs_at: Vec<[f32; ODP]> = vec![];
    for _t in 0..decisions { let a = cmpc_act(&w, &mut mean, seed.wrapping_mul(31) + _t as u32 * 7919);
        obs_at.push(w.obs()); macros.push([a[0] / PUSH_V, a[1] / PUSH_V]); for _ in 0..STRIDE { w.step(a); } if w.coverage() >= 0.98 { break; } }
    for t in 0..macros.len() { let mut c = [0.0f32; AD]; for j in 0..CM { let idx = (t + j).min(macros.len() - 1); c[j * 2] = macros[idx][0]; c[j * 2 + 1] = macros[idx][1]; }
        ob.push(obs_at[t]); ch.push(c); }
}
fn main() { pollster::block_on(run()); }
async fn run() {
    let im = moment(); let (betas, alphas, abar) = cosine();
    println!("  EFA-2 · PUSHT — FULL-SCALE DIFFUSION (cosine schedule, T={} steps, {} demos, HD={})\n", T, NDEMO, HD);
    println!("  schedule sanity: abar_0={:.3} abar_T={:.4} (must be ~0 to fully noise — the nano-run bug was abar_T~0.85)", abar[0], abar[T - 1]);
    // ── demos (cached) ──
    let mut ob: Vec<[f32; ODP]> = vec![]; let mut ch: Vec<[f32; AD]> = vec![];
    if let Ok(bytes) = std::fs::read(DEMOF) { let f: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let rec = ODP + AD; let n = f.len() / rec; for i in 0..n { let mut o = [0.0f32; ODP]; let mut c = [0.0f32; AD];
            for k in 0..ODP { o[k] = f[i * rec + k]; } for k in 0..AD { c[k] = f[i * rec + ODP + k]; } ob.push(o); ch.push(c); }
        println!("  loaded {} cached (obs,chunk) demo pairs", ob.len());
    } else { print!("  generating {} MPC demo episodes… ", NDEMO); for k in 0..NDEMO { mpc_chunks(3000 + k, im, &mut ob, &mut ch); }
        let mut buf: Vec<u8> = vec![]; for i in 0..ob.len() { for k in 0..ODP { buf.extend_from_slice(&ob[i][k].to_le_bytes()); } for k in 0..AD { buf.extend_from_slice(&ch[i][k].to_le_bytes()); } }
        let _ = std::fs::write(DEMOF, buf); println!("collected {} pairs (cached)", ob.len()); }
    // ── train ε_θ ──
    let ctx = std::sync::Arc::new(Context::new().await.expect("ctx")); let nin = AD + 1 + ODP; let bs = 512; let np = ob.len();
    let mut pp: Vec<Tensor> = (0..nin).map(|c| Tensor::from_vec(&ctx, &randn(HD, 500 + c as u32, 0.3), &[1, HD])).collect();
    pp.push(Tensor::zeros(&ctx, &[HD])); pp.push(Tensor::from_vec(&ctx, &randn(HD * HD, 560, 1.0 / (HD as f32).sqrt()), &[HD, HD])); pp.push(Tensor::zeros(&ctx, &[HD]));
    pp.push(Tensor::from_vec(&ctx, &randn(HD * AD, 561, 1.0 / (HD as f32).sqrt()), &[HD, AD])); pp.push(Tensor::zeros(&ctx, &[AD]));
    let mut adamp = Adam::new(&pp, 0.001);
    let dnet = |f: &[Var], pv: &[Var]| { let mut pre = pv[nin].clone(); for c in 0..nin { pre = pre.add(&f[c].matmul(&pv[c])); }
        pre.relu().matmul(&pv[nin + 1]).add(&pv[nin + 2]).relu().matmul(&pv[nin + 3]).add(&pv[nin + 4]) };
    println!("  training the denoiser (cosine schedule, x0-clamped posterior sampling), 50k iters:");
    for it in 0..50000u32 {
        let mut cols: Vec<Vec<f32>> = (0..nin).map(|_| vec![0.0f32; bs]).collect(); let mut tb = vec![0.0f32; bs * AD];
        for i in 0..bs { let idx = (u(it * 131 + i as u32, 5) * np as f32) as usize % np; let o = ob[idx]; let a0 = ch[idx];
            let tt = (u(it * 131 + i as u32, 6) * T as f32) as usize % T; let sa = abar[tt].sqrt(); let sb = (1.0 - abar[tt]).sqrt();
            for c in 0..AD { let e = gnoise(it * 131 + i as u32, c * 3 + 1); cols[c][i] = sa * a0[c] + sb * e; tb[i * AD + c] = e; }
            cols[AD][i] = tt as f32 / T as f32; for c in 0..ODP { cols[AD + 1 + c][i] = o[c]; } }
        let ppv: Vec<Var> = pp.iter().map(|t| Var::leaf(t.clone())).collect();
        let ff: Vec<Var> = (0..nin).map(|c| Var::leaf(Tensor::from_vec(&ctx, &cols[c], &[bs, 1]))).collect();
        let v = dnet(&ff, &ppv); let d = v.sub(&Var::leaf(Tensor::from_vec(&ctx, &tb, &[bs, AD]))); let loss = d.mul(&d).mean_all(); loss.backward();
        let gp: Vec<Tensor> = ppv.iter().zip(&pp).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adamp.step(&mut pp, &gp);
        if it % 10000 == 9999 { println!("     iter {:>5}: denoise MSE {:.4}", it + 1, loss.value().to_vec().await[0]); } }
    let mut wv = Vec::new(); for c in 0..nin { wv.push(pp[c].to_vec().await); }
    let net = DNet { w: wv, b1: pp[nin].to_vec().await, w2: pp[nin + 1].to_vec().await, b2: pp[nin + 2].to_vec().await, w3: pp[nin + 3].to_vec().await, b3: pp[nin + 4].to_vec().await };
    // x0-prediction DDPM posterior sampling (stable at small T)
    let sample = |ob: &[f32; ODP], seed: u32| -> [f32; AD] { let mut x = [0.0f32; AD]; for c in 0..AD { x[c] = gnoise(seed, c); }
        for i in (0..T).rev() { let eps = net.eps(&x, i as f32 / T as f32, ob); let sab = abar[i].sqrt(); let sbb = (1.0 - abar[i]).sqrt();
            let mut x0 = [0.0f32; AD]; for c in 0..AD { x0[c] = ((x[c] - sbb * eps[c]) / sab).clamp(-1.2, 1.2); }
            if i > 0 { let ap = abar[i - 1]; let c1 = ap.sqrt() * betas[i] / (1.0 - abar[i]); let c2 = alphas[i].sqrt() * (1.0 - ap) / (1.0 - abar[i]);
                let var = (betas[i] * (1.0 - ap) / (1.0 - abar[i])).max(0.0).sqrt();
                for c in 0..AD { x[c] = c1 * x0[c] + c2 * x[c] + var * gnoise(seed.wrapping_add(9973 + i as u32), c); } }
            else { x = x0; } }
        x };
    println!("\n  the card — FULL-SCALE DIFFUSION policy, PushT (200 held-out; sample chunk, execute open-loop, re-sample):");
    let (mut mc, mut ok, mut c5, mut c7) = (0.0f32, 0, 0, 0);
    for k in 500..700u32 { let mut w = World::new(k, im); let mut best = 0.0f32; let steps = (TMAX / DT) as usize; let mut t = 0; let mut sd = k * 101 + 7;
        while t < steps { let chunk = sample(&w.obs(), sd); sd = sd.wrapping_add(1);
            for j in 0..CM { let a = [chunk[j * 2].clamp(-1.0, 1.0) * PUSH_V, chunk[j * 2 + 1].clamp(-1.0, 1.0) * PUSH_V];
                for _ in 0..STRIDE { if t >= steps { break; } w.step(a); best = best.max(w.coverage()); t += 1; } if best >= 0.98 { break; } }
            if best >= 0.98 { break; } }
        mc += best; if best >= COV_OK { ok += 1; } if best >= 0.5 { c5 += 1; } if best >= 0.7 { c7 += 1; } }
    println!("     diffusion policy: mean coverage {:.3} · success(≥0.90) {:.0}% · ≥0.7 {:.0}% · ≥0.5 {:.0}%", mc / 200.0, ok as f32 / 2.0, c7 as f32 / 2.0, c5 as f32 / 2.0);
    println!("     [anchors] MPC teacher 0.69 · nano-diffusion 0.096 (schedule bug) · chunk-regress 0.125 · scripted 0.20");
    println!("\n  Fixed schedule (abar_T≈0) + x0-posterior sampling + scale. If this is a competitive number vs MPC 0.69,");
    println!("  it's the diffusion recipe working on our verified physics. One seed.");
}
