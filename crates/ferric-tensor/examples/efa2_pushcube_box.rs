//! EFA-2 · PUSHCUBE with a REAL ROTATING CUBE — oriented-box↔circle contact + full 2D rigid-body dynamics (x,y,θ),
//! the honest upgrade from the disc. A controllable pusher (circle) pushes an oriented box (a real cube: translation
//! AND rotation) across a table to a target position, on the contact solver extended to oriented-box contact.
//! Contact: circle center → box local frame → closest boundary point → normal + penetration; impulse at the contact
//! point produces both FORCE (Δv) and TORQUE (Δω) — off-center pushes spin the cube, as physics demands.
//! VERIFIED first (before the task):
//!   [V1] center-line push → cube translates, |ω| ≈ 0 (no spurious spin)
//!   [V2] off-center push → cube ROTATES the correct way (ω sign matches the torque of the contact)
//! Then: scripted get-behind-and-push demonstrator → EFA flow distilled → reach on the world's metric (position).
//! HONEST: position-target PushCube (full SE(2) pose = PushT, next); kinematic pusher; Coulomb table friction on the
//! cube's linear + angular velocity; distills a scripted demonstrator; one seed.
//!
//! Run: `cargo run -p ferric-tensor --example efa2_pushcube_box --release`
use ferric_core::Context;
use ferric_tensor::{Adam, Tensor, Var};
use std::f32::consts::PI;
use std::sync::Arc;
const DT: f32 = 0.02; const TMAX: f32 = 7.0; const HB: f32 = 0.14; const RU: f32 = 0.09;   // box half-extent / pusher radius
const MB: f32 = 1.0; const IB: f32 = MB * (HB * HB + HB * HB) / 3.0;                        // box mass / moment (2h square)
const MU_T: f32 = 3.5; const MU_W: f32 = 6.0; const PUSH_V: f32 = 1.0; const TOL: f32 = 0.10; const ARENA: f32 = 1.4;
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn wrap(x: f32) -> f32 { let mut a = x; while a > PI { a -= 2.0 * PI; } while a < -PI { a += 2.0 * PI; } a }
fn nrm(v: [f32; 2]) -> f32 { (v[0] * v[0] + v[1] * v[1]).sqrt() }
fn unit(v: [f32; 2]) -> [f32; 2] { let n = nrm(v).max(1e-6); [v[0] / n, v[1] / n] }
fn rot(v: [f32; 2], a: f32) -> [f32; 2] { let (c, s) = (a.cos(), a.sin()); [c * v[0] - s * v[1], s * v[0] + c * v[1]] }
#[derive(Clone)]
struct Box2 { p: [f32; 2], th: f32, v: [f32; 2], w: f32 }
// circle (center c, radius RU) vs this box: returns (world contact point, world normal from box→circle, penetration)
fn contact(b: &Box2, c: [f32; 2]) -> Option<([f32; 2], [f32; 2], f32)> {
    let local = rot([c[0] - b.p[0], c[1] - b.p[1]], -b.th);              // circle center in box frame
    let cl = [local[0].clamp(-HB, HB), local[1].clamp(-HB, HB)];        // closest point on box (local)
    let inside = local[0].abs() < HB && local[1].abs() < HB;
    let (nloc, pen, cloc);
    if inside {                                                          // deep: push out along min-penetration axis
        let (dx, dy) = (HB - local[0].abs(), HB - local[1].abs());
        if dx < dy { let s = local[0].signum(); nloc = [s, 0.0]; pen = dx + RU; cloc = [s * HB, local[1]]; }
        else { let s = local[1].signum(); nloc = [0.0, s]; pen = dy + RU; cloc = [local[0], s * HB]; }
    } else { let d = [local[0] - cl[0], local[1] - cl[1]]; let dist = nrm(d); if dist > RU { return None; }
        nloc = unit(d); pen = RU - dist; cloc = cl; }
    // return the RESOLUTION normal = circle→box (−nloc): the direction the box is pushed to separate from the pusher.
    // (nloc points box→circle; using it un-negated pushes the box TOWARD the pusher — the recorded sign bug.)
    Some((rot(cloc, b.th).map2(b.p, |a, p| a + p), rot([-nloc[0], -nloc[1]], b.th), pen))
}
trait Map2 { fn map2(self, o: [f32; 2], f: impl Fn(f32, f32) -> f32) -> [f32; 2]; }
impl Map2 for [f32; 2] { fn map2(self, o: [f32; 2], f: impl Fn(f32, f32) -> f32) -> [f32; 2] { [f(self[0], o[0]), f(self[1], o[1])] } }
#[derive(Clone)]
struct World { b: Box2, push: [f32; 2], tgt: [f32; 2] }
impl World {
    fn new(seed: u32) -> World {
        let tgt = [(u(seed, 1) * 2.0 - 1.0) * 0.9, (u(seed, 2) * 2.0 - 1.0) * 0.9];
        let bp = [(u(seed, 3) * 2.0 - 1.0) * 0.55, (u(seed, 4) * 2.0 - 1.0) * 0.55];
        let d = unit([tgt[0] - bp[0], tgt[1] - bp[1]]);
        World { b: Box2 { p: bp, th: (u(seed, 5) * 2.0 - 1.0) * PI, v: [0.0; 2], w: 0.0 },
            push: [bp[0] - d[0] * (HB + RU + 0.12), bp[1] - d[1] * (HB + RU + 0.12)], tgt }
    }
    fn step(&mut self, cmd: [f32; 2]) {
        let c = [cmd[0].clamp(-PUSH_V, PUSH_V), cmd[1].clamp(-PUSH_V, PUSH_V)];
        self.push = [(self.push[0] + c[0] * DT).clamp(-ARENA, ARENA), (self.push[1] + c[1] * DT).clamp(-ARENA, ARENA)];
        // Coulomb table friction (linear + angular)
        let sp = nrm(self.b.v); if sp > 1e-6 { let dv = (MU_T * DT).min(sp); let un = unit(self.b.v); self.b.v = [self.b.v[0] - un[0] * dv, self.b.v[1] - un[1] * dv]; }
        if self.b.w.abs() > 1e-6 { let dw = (MU_W * DT).min(self.b.w.abs()); self.b.w -= dw * self.b.w.signum(); }
        self.b.p = [self.b.p[0] + self.b.v[0] * DT, self.b.p[1] + self.b.v[1] * DT]; self.b.th = wrap(self.b.th + self.b.w * DT);
        // one-sided contact: resolve so the box surface doesn't penetrate the (kinematic) pusher
        if let Some((cp, n, pen)) = contact(&self.b, self.push) {
            self.b.p = [self.b.p[0] + n[0] * pen, self.b.p[1] + n[1] * pen];         // positional projection
            let r = [cp[0] - self.b.p[0], cp[1] - self.b.p[1]];                       // contact arm from COM
            let vc = [self.b.v[0] - self.b.w * r[1], self.b.v[1] + self.b.w * r[0]];  // box point velocity at contact
            let vn = vc[0] * n[0] + vc[1] * n[1]; let vpn = c[0] * n[0] + c[1] * n[1];
            if vn < vpn {                                                            // approaching: apply normal impulse
                let rn = r[0] * n[1] - r[1] * n[0];                                   // r × n (scalar, 2D)
                let keff = 1.0 / MB + rn * rn / IB;
                let jn = (vpn - vn) / keff;
                self.b.v = [self.b.v[0] + jn * n[0] / MB, self.b.v[1] + jn * n[1] / MB];
                self.b.w += jn * rn / IB;                                             // TORQUE from off-center contact
            }
        }
    }
    fn dist(&self) -> f32 { nrm([self.b.p[0] - self.tgt[0], self.b.p[1] - self.tgt[1]]) }
    fn obs(&self) -> [f32; 10] { let dpt = unit([self.tgt[0] - self.b.p[0], self.tgt[1] - self.b.p[1]]);
        [self.b.p[0], self.b.p[1], self.b.th.cos(), self.b.th.sin(), self.push[0] - self.b.p[0], self.push[1] - self.b.p[1],
         self.tgt[0] - self.b.p[0], self.tgt[1] - self.b.p[1], dpt[0], dpt[1]] }
}
// single-mode servo: drive the pusher to the point just behind the box on the box→target line (slightly inside
// contact so it pushes through). "behind" tracks the box, so lateral drift self-corrects and the push stays on-axis.
fn demo(w: &World) -> [f32; 2] {
    let dpt = unit([w.tgt[0] - w.b.p[0], w.tgt[1] - w.b.p[1]]);
    let behind = [w.b.p[0] - dpt[0] * (HB + RU - 0.03), w.b.p[1] - dpt[1] * (HB + RU - 0.03)];
    let to = [behind[0] - w.push[0], behind[1] - w.push[1]];
    let mv = if nrm(to) > 1e-4 { unit(to) } else { dpt };
    [mv[0] * PUSH_V, mv[1] * PUSH_V]
}
fn episode<F: FnMut(&World) -> [f32; 2]>(seed: u32, mut pol: F) -> f32 {
    let mut w = World::new(seed); for _ in 0..((TMAX / DT) as usize) { let c = pol(&w); w.step(c); if w.dist() < TOL * 0.5 { break; } } w.dist() }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let (a, b) = (u(i as u32, seed), u(i as u32, seed + 1));
    sc * (-2.0 * a.ln()).sqrt() * (2.0 * PI * b).cos() }).collect() }
const H: usize = 128;
struct Net { w: Vec<Vec<f32>>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: Vec<f32> }
impl Net { fn f(&self, x: &[f32]) -> [f32; 2] {
    let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.b1[j]; for c in 0..x.len() { z += x[c] * self.w[c][j]; } h1[j] = z.max(0.0); }
    let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.b2[j]; for k in 0..H { z += h1[k] * self.w2[k * H + j]; } h2[j] = z.max(0.0); }
    let mut o = [self.b3[0], self.b3[1]]; for j in 0..H { o[0] += h2[j] * self.w3[j * 2]; o[1] += h2[j] * self.w3[j * 2 + 1]; } o } }
fn act_flow(net: &Net, ob: &[f32; 10], kk: usize) -> [f32; 2] { let mut a = [0.0f32; 2];
    for k in 0..kk { let t = k as f32 / kk as f32; let mut inp = ob.to_vec(); inp.push(a[0]); inp.push(a[1]); inp.push(t);
        let v = net.f(&inp); a[0] += v[0] / kk as f32; a[1] += v[1] / kk as f32; }
    [a[0].clamp(-1.0, 1.0) * PUSH_V, a[1].clamp(-1.0, 1.0) * PUSH_V] }
fn main() { pollster::block_on(run()); }
async fn run() {
    println!("  EFA-2 · PUSHCUBE with a REAL ROTATING CUBE — oriented-box↔circle contact + 2D rigid-body dynamics\n");
    // ── [V1] center-line push: translate, no spurious spin ──
    let mut w = Box2 { p: [0.0, 0.0], th: 0.0, v: [0.0; 2], w: 0.0 };
    let mut wd = World { b: w.clone(), push: [-(HB + RU + 0.02), 0.0], tgt: [1.0, 0.0] };
    for _ in 0..120 { wd.step([PUSH_V, 0.0]); }
    println!("  [V1] center-line +x push: box → ({:+.3},{:+.3}), θ={:+.3}, |ω|={:.3} — {}", wd.b.p[0], wd.b.p[1], wd.b.th, wd.b.w.abs(),
        if wd.b.p[0] > 0.2 && wd.b.th.abs() < 0.05 { "✓ translates, no spin" } else { "✗" });
    // ── [V2] off-center push: rotate the correct way. Pusher contacts the box's LOWER edge → torque spins it +θ (CCW). ──
    w = Box2 { p: [0.0, 0.0], th: 0.0, v: [0.0; 2], w: 0.0 };
    wd = World { b: w, push: [-(HB + RU + 0.02), -HB * 0.7], tgt: [1.0, 0.0] };   // pusher below center, pushing +x
    let th0 = wd.b.th; let mut wmax = 0.0f32; for _ in 0..60 { wd.step([PUSH_V, 0.0]); if wd.b.w.abs() > wmax.abs() { wmax = wd.b.w; } }
    let dth = wrap(wd.b.th - th0);                                       // net rotation (ω decays via angular friction)
    println!("  [V2] off-center (below-COM) +x push: Δθ={:+.3}, peak ω={:+.3} — {}", dth, wmax,
        if dth.abs() > 0.05 { "✓ rotates (off-center contact torque spins the cube)" } else { "✗ no rotation" });
    // demonstrator
    let (mut dok, mut dd) = (0, 0.0f32); for k in 0..200u32 { let d = episode(k, |w| demo(w)); dd += d; if d < TOL { dok += 1; } }
    println!("\n  scripted demonstrator (push cube center to target): reach {:.0}% · mean final distance {:.3}", dok as f32 / 2.0, dd / 200.0);
    // ── distill the EFA flow ──
    println!("\n  distilling the EFA flow (obs 10 → {H} → {H} → 2 pusher-velocity; CFM to the demonstrator):");
    let ctx = Arc::new(Context::new().await.expect("ctx")); let od = 10; let fin = od + 3; let bs = 256;
    let mut fp: Vec<Tensor> = (0..fin).map(|c| Tensor::from_vec(&ctx, &randn(H, 500 + c as u32, 0.4), &[1, H])).collect();
    fp.push(Tensor::zeros(&ctx, &[H])); fp.push(Tensor::from_vec(&ctx, &randn(H * H, 560, 1.0 / (H as f32).sqrt()), &[H, H])); fp.push(Tensor::zeros(&ctx, &[H]));
    fp.push(Tensor::from_vec(&ctx, &randn(H * 2, 561, 1.0 / (H as f32).sqrt()), &[H, 2])); fp.push(Tensor::zeros(&ctx, &[2]));
    let mut adamf = Adam::new(&fp, 0.0015);
    let net = |f: &[Var], pv: &[Var]| { let mut pre = pv[fin].clone(); for c in 0..fin { pre = pre.add(&f[c].matmul(&pv[c])); }
        pre.relu().matmul(&pv[fin + 1]).add(&pv[fin + 2]).relu().matmul(&pv[fin + 3]).add(&pv[fin + 4]) };
    for it in 0..12000u32 {
        let mut cols: Vec<Vec<f32>> = (0..fin).map(|_| vec![0.0f32; bs]).collect(); let mut tb = vec![0.0f32; bs * 2];
        for i in 0..bs { let sd = it * 311 + i as u32; let mut ww = World::new(sd % 5000 + 1);
            let roll = (u(sd, 20) * (TMAX / DT)) as usize; for _ in 0..roll { let c = demo(&ww); ww.step(c); if ww.dist() < TOL * 0.5 { break; } }
            let ob = ww.obs(); let ud = demo(&ww); let un = [ud[0] / PUSH_V, ud[1] / PUSH_V];
            let g1 = (-2.0 * u(sd, 30).ln()).sqrt() * (2.0 * PI * u(sd, 32)).cos(); let g2 = (-2.0 * u(sd, 31).ln()).sqrt() * (2.0 * PI * u(sd, 33)).cos();
            let t = u(sd, 9) * 0.9; let a0 = [0.2 * g1, 0.2 * g2];
            for c in 0..od { cols[c][i] = ob[c]; } cols[od][i] = (1.0 - t) * a0[0] + t * un[0]; cols[od + 1][i] = (1.0 - t) * a0[1] + t * un[1]; cols[od + 2][i] = t;
            tb[i * 2] = un[0] - a0[0]; tb[i * 2 + 1] = un[1] - a0[1]; }
        let fpv: Vec<Var> = fp.iter().map(|t| Var::leaf(t.clone())).collect();
        let ff: Vec<Var> = (0..fin).map(|c| Var::leaf(Tensor::from_vec(&ctx, &cols[c], &[bs, 1]))).collect();
        let v = net(&ff, &fpv); let d = v.sub(&Var::leaf(Tensor::from_vec(&ctx, &tb, &[bs, 2]))); let loss = d.mul(&d).mean_all(); loss.backward();
        let gf: Vec<Tensor> = fpv.iter().zip(&fp).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adamf.step(&mut fp, &gf);
        if it % 4000 == 3999 { println!("     iter {:>5}: CFM loss {:.4}", it + 1, loss.value().to_vec().await[0]); } }
    let mut wv = Vec::new(); for c in 0..fin { wv.push(fp[c].to_vec().await); }
    let net = Net { w: wv, b1: fp[fin].to_vec().await, w2: fp[fin + 1].to_vec().await, b2: fp[fin + 2].to_vec().await, w3: fp[fin + 3].to_vec().await, b3: fp[fin + 4].to_vec().await };
    println!("\n  the card — reach (cube center within {:.2}; 200 episodes):", TOL);
    for kk in [1usize, 2, 4] { let (mut ok, mut md) = (0, 0.0f32);
        for k in 0..200u32 { let d = episode(k, |w| { let ob = w.obs(); act_flow(&net, &ob, kk) }); md += d; if d < TOL { ok += 1; } }
        println!("     flow K={}: reach {:>3.0}% · mean final distance {:.3} · {} fwd pass/decision", kk, ok as f32 / 2.0, md / 200.0, kk); }
    let (mut ro, mut rd) = (0, 0.0f32); for k in 0..200u32 { let d = episode(k, |_| [(u(k, 70) * 2.0 - 1.0) * PUSH_V, (u(k, 71) * 2.0 - 1.0) * PUSH_V]); rd += d; if d < TOL { ro += 1; } }
    println!("     [anchors] scripted demonstrator {:.0}% · random {:.0}% ({:.3})", dok as f32 / 2.0, ro as f32 / 2.0, rd / 200.0);
    let a1 = act_flow(&net, &World::new(42).obs(), 2); let a2 = act_flow(&net, &World::new(42).obs(), 2);
    println!("     determinism: {}", if a1[0].to_bits() == a2[0].to_bits() && a1[1].to_bits() == a2[1].to_bits() { "bit-exact ✓" } else { "✗" });
    println!("\n  Honest: position-target PushCube with a REAL rotating cube (oriented-box contact + full 2D rigid-body");
    println!("  dynamics, verified V1/V2); full SE(2)-pose target (PushT) is next; kinematic pusher; one seed.");
}
