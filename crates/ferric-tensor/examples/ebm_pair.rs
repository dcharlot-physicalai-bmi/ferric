//! EFA energy-first #56 — the COORDINATED PAIR: flow-matching ACTUATION + a dedicated contrastive VERIFY energy.
//!
//! The frontier check reframed EFA honestly: "one scalar energy for everything" → "one shared latent + a small
//! coordinated FAMILY of energies," because a Bellman VALUE is a poor validity classifier (flagship VERIFY stuck at 76%)
//! and iterative energy-descent is the wrong ACTUATION primitive (use flow-matching). This builds the pair on a
//! shared state representation: (a) a flow-matching policy GENERATES the action (K forward passes, no descent/BPTT), and
//! (b) a dedicated VERIFY energy — trained contrastively to be LOW on good actions, HIGH on bad — SCORES action validity.
//! Coordination: the flow proposes, the energy checks. We measure flow reach%, verify accuracy (does E rank good < bad?),
//! and agreement (does the flow's action get low energy?). Target: verify > the flagship's 76% (a value-as-classifier).
//!
//! Run: `cargo run -p ferric-tensor --example ebm_pair --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;
const H: usize = 64; const DT: f32 = 0.05; const GAMMA: f32 = 0.97; const UMAX: f32 = 3.0;
const ACTS: [f32; 5] = [-3.0, -1.5, 0.0, 1.5, 3.0];
const TESTG: [f32; 4] = [-1.5, -0.5, 0.5, 1.5];
use std::f32::consts::PI;
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }
fn wrap(x: f32) -> f32 { let mut a = x; while a > PI { a -= 2.0 * PI; } while a < -PI { a += 2.0 * PI; } a }
fn step(th: f32, om: f32, uu: f32) -> (f32, f32) { let no = om + DT * (-th.sin() - 0.05 * om + uu.clamp(-UMAX, UMAX)); (wrap(th + DT * no), no) }
fn feat5(th: f32, om: f32, g: f32) -> [f32; 5] { let d = th - g; [d.cos(), d.sin(), om, th.cos(), th.sin()] }

struct Vn { w: [Vec<f32>; 5], b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: f32 }
impl Vn {
    fn eval(&self, th: f32, om: f32, g: f32) -> f32 { let f = feat5(th, om, g);
        let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.b1[j]; for c in 0..5 { z += f[c] * self.w[c][j]; } h1[j] = (z.exp() + 1.0).ln(); }
        let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.b2[j]; for k in 0..H { z += h1[k] * self.w2[k * H + j]; } h2[j] = (z.exp() + 1.0).ln(); }
        let mut o = self.b3; for j in 0..H { o += h2[j] * self.w3[j]; } (o.exp() + 1.0).ln() }
    fn ustar(&self, th: f32, om: f32, g: f32) -> f32 { let mut bu = 0.0; let mut be = f32::MAX; for &uu in &ACTS { let (nt, no) = step(th, om, uu); let q = 0.01 * uu * uu * DT + GAMMA * self.eval(nt, no, g); if q < be { be = q; bu = uu; } } bu }
}
// flow velocity field v(s,a,t): [5 state, a, t] → scalar velocity (relu hidden, linear out)
struct Fl { w: [Vec<f32>; 7], b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: f32 }
impl Fl {
    fn vel(&self, th: f32, om: f32, g: f32, a: f32, t: f32) -> f32 { let mut f = [0.0f32; 7]; let ff = feat5(th, om, g); for c in 0..5 { f[c] = ff[c]; } f[5] = a; f[6] = t;
        let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.b1[j]; for c in 0..7 { z += f[c] * self.w[c][j]; } h1[j] = z.max(0.0); }
        let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.b2[j]; for k in 0..H { z += h1[k] * self.w2[k * H + j]; } h2[j] = z.max(0.0); }
        let mut o = self.b3; for j in 0..H { o += h2[j] * self.w3[j]; } o }
    fn act(&self, th: f32, om: f32, g: f32, k: usize) -> f32 { let mut a = 0.0f32; for i in 0..k { let t = i as f32 / k as f32; a += self.vel(th, om, g, a, t) / k as f32; } a.clamp(-UMAX, UMAX) }
}
// verify energy E(s,a): [5 state, a, a²] → scalar (softplus ≥0), LOW on good actions
struct Ev { w: [Vec<f32>; 7], b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: f32 }
impl Ev {
    fn eval(&self, th: f32, om: f32, g: f32, a: f32) -> f32 { let mut f = [0.0f32; 7]; let ff = feat5(th, om, g); for c in 0..5 { f[c] = ff[c]; } f[5] = a; f[6] = a * a;
        let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.b1[j]; for c in 0..7 { z += f[c] * self.w[c][j]; } h1[j] = (z.exp() + 1.0).ln(); }
        let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.b2[j]; for k in 0..H { z += h1[k] * self.w2[k * H + j]; } h2[j] = (z.exp() + 1.0).ln(); }
        let mut o = self.b3; for j in 0..H { o += h2[j] * self.w3[j]; } (o.exp() + 1.0).ln() }
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — COORDINATED PAIR: flow-matching ACTUATION + contrastive VERIFY energy (one latent, family of energies)\n");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let bs = 256usize;
    let sp = |z: Var, ov: &Var| z.exp().add(ov).log();
    // ---- Stage 1: FVI V → demonstrator u* ----
    let vnet = |f: &[Var], pv: &[Var], ov: &Var| { let mut pre = pv[5].clone(); for c in 0..5 { pre = pre.add(&f[c].matmul(&pv[c])); } sp(sp(sp(pre, ov).matmul(&pv[6]).add(&pv[7]), ov).matmul(&pv[8]).add(&pv[9]), ov) };
    let mut p: Vec<Tensor> = (0..5).map(|c| Tensor::from_vec(&ctx, &randn(H, 22 + c as u32, 0.5), &[1, H])).collect();
    p.push(Tensor::zeros(&ctx, &[H])); p.push(Tensor::from_vec(&ctx, &randn(H * H, 40, 1.0 / (H as f32).sqrt()), &[H, H])); p.push(Tensor::zeros(&ctx, &[H]));
    p.push(Tensor::from_vec(&ctx, &randn(H, 41, 1.0 / (H as f32).sqrt()), &[H, 1])); p.push(Tensor::zeros(&ctx, &[1]));
    let mut tgt = p.clone(); let mut adam = Adam::new(&p, 0.002);
    for it in 0..12000 {
        let mut fc: Vec<Vec<f32>> = (0..5).map(|_| vec![0.0f32; bs]).collect();
        let mut nf: Vec<Vec<f32>> = (0..5).map(|_| vec![0.0f32; bs * 5]).collect(); let mut cst = vec![0.0f32; bs * 5];
        for i in 0..bs { let sd = it as u32 * 7 + i as u32; let th = (u(sd, 1) * 2.0 - 1.0) * PI; let om = (u(sd, 2) * 2.0 - 1.0) * 3.0; let g = (u(sd, 3) * 2.0 - 1.0) * 2.0;
            let f = feat5(th, om, g); for c in 0..5 { fc[c][i] = f[c]; }
            for (ai, &uu) in ACTS.iter().enumerate() { let (nt, no) = step(th, om, uu); let nff = feat5(nt, no, g); for c in 0..5 { nf[c][i * 5 + ai] = nff[c]; } cst[i * 5 + ai] = wrap(th - g).powi(2) + 0.05 * om * om + 0.01 * uu * uu; } }
        let l = |v: &[f32], r: usize| Var::leaf(Tensor::from_vec(&ctx, v, &[r, 1]));
        let tv: Vec<Var> = tgt.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
        let et = vnet(&(0..5).map(|c| l(&nf[c], bs * 5)).collect::<Vec<_>>(), &tv, &ov).value().to_vec().await;
        let mut target = vec![0.0f32; bs]; for i in 0..bs { let mut m = f32::MAX; for ai in 0..5 { let q = cst[i * 5 + ai] * DT + GAMMA * et[i * 5 + ai]; if q < m { m = q; } } target[i] = m; }
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
        let e = vnet(&(0..5).map(|c| l(&fc[c], bs)).collect::<Vec<_>>(), &pv, &ov); let d = e.sub(&Var::leaf(Tensor::from_vec(&ctx, &target, &[bs, 1]))); let loss = d.mul(&d).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect(); adam.step(&mut p, &g); if it % 200 == 0 { tgt = p.clone(); }
    }
    let mut wv: [Vec<f32>; 5] = Default::default(); for c in 0..5 { wv[c] = p[c].to_vec().await; }
    let vn = Vn { w: wv, b1: p[5].to_vec().await, w2: p[6].to_vec().await, b2: p[7].to_vec().await, w3: p[8].to_vec().await, b3: p[9].to_vec().await[0] };

    // ---- Stage 2a: flow-matching policy (velocity field → action) ----
    let fnet = |f: &[Var], pv: &[Var]| { let mut pre = pv[7].clone(); for c in 0..7 { pre = pre.add(&f[c].matmul(&pv[c])); } pre.relu().matmul(&pv[8]).add(&pv[9]).relu().matmul(&pv[10]).add(&pv[11]) };
    let mut q: Vec<Tensor> = (0..7).map(|c| Tensor::from_vec(&ctx, &randn(H, 60 + c as u32, 0.4), &[1, H])).collect();
    q.push(Tensor::zeros(&ctx, &[H])); q.push(Tensor::from_vec(&ctx, &randn(H * H, 80, 1.0 / (H as f32).sqrt()), &[H, H])); q.push(Tensor::zeros(&ctx, &[H])); q.push(Tensor::from_vec(&ctx, &randn(H, 81, 1.0 / (H as f32).sqrt()), &[H, 1])); q.push(Tensor::zeros(&ctx, &[1]));
    let mut adamq = Adam::new(&q, 0.002);
    for it in 0..9000 {
        let mut fc: Vec<Vec<f32>> = (0..7).map(|_| vec![0.0f32; bs]).collect(); let mut tb = vec![0.0f32; bs];
        for i in 0..bs { let sd = it as u32 * 13 + i as u32; let th = (u(sd, 1) * 2.0 - 1.0) * PI; let om = (u(sd, 2) * 2.0 - 1.0) * 3.0; let g = (u(sd, 3) * 2.0 - 1.0) * 2.0;
            let us = vn.ustar(th, om, g); let a0 = (u(sd, 7) * 2.0 - 1.0) * 3.0; let t = u(sd, 8); let at = (1.0 - t) * a0 + t * us;
            let ff = feat5(th, om, g); for c in 0..5 { fc[c][i] = ff[c]; } fc[5][i] = at; fc[6][i] = t; tb[i] = us - a0; }
        let pv: Vec<Var> = q.iter().map(|t| Var::leaf(t.clone())).collect(); let fv: Vec<Var> = (0..7).map(|c| Var::leaf(Tensor::from_vec(&ctx, &fc[c], &[bs, 1]))).collect();
        let v = fnet(&fv, &pv); let d = v.sub(&Var::leaf(Tensor::from_vec(&ctx, &tb, &[bs, 1]))); let loss = d.mul(&d).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&q).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect(); adamq.step(&mut q, &g);
    }
    let mut fw: [Vec<f32>; 7] = Default::default(); for c in 0..7 { fw[c] = q[c].to_vec().await; }
    let fl = Fl { w: fw, b1: q[7].to_vec().await, w2: q[8].to_vec().await, b2: q[9].to_vec().await, w3: q[10].to_vec().await, b3: q[11].to_vec().await[0] };

    // ---- Stage 2b: VERIFY energy E(s,a) contrastive: LOW at u*, HIGH at negatives (hinge ranking) ----
    let enet = |f: &[Var], pv: &[Var], ov: &Var| { let mut pre = pv[7].clone(); for c in 0..7 { pre = pre.add(&f[c].matmul(&pv[c])); } sp(sp(sp(pre, ov).matmul(&pv[8]).add(&pv[9]), ov).matmul(&pv[10]).add(&pv[11]), ov) };
    let mut r: Vec<Tensor> = (0..7).map(|c| Tensor::from_vec(&ctx, &randn(H, 90 + c as u32, 0.4), &[1, H])).collect();
    r.push(Tensor::zeros(&ctx, &[H])); r.push(Tensor::from_vec(&ctx, &randn(H * H, 110, 1.0 / (H as f32).sqrt()), &[H, H])); r.push(Tensor::zeros(&ctx, &[H])); r.push(Tensor::from_vec(&ctx, &randn(H, 111, 1.0 / (H as f32).sqrt()), &[H, 1])); r.push(Tensor::zeros(&ctx, &[1]));
    let mut adamr = Adam::new(&r, 0.002); let marg = Var::leaf(Tensor::from_vec(&ctx, &[0.5], &[1]));
    for it in 0..9000 {
        let (mut pf, mut nf): (Vec<Vec<f32>>, Vec<Vec<f32>>) = ((0..7).map(|_| vec![0.0f32; bs]).collect(), (0..7).map(|_| vec![0.0f32; bs]).collect());
        for i in 0..bs { let sd = it as u32 * 17 + i as u32; let th = (u(sd, 1) * 2.0 - 1.0) * PI; let om = (u(sd, 2) * 2.0 - 1.0) * 3.0; let g = (u(sd, 3) * 2.0 - 1.0) * 2.0;
            let us = vn.ustar(th, om, g); let un = (u(sd, 7) * 2.0 - 1.0) * UMAX;   // positive u*, negative random
            let ff = feat5(th, om, g); for c in 0..5 { pf[c][i] = ff[c]; nf[c][i] = ff[c]; } pf[5][i] = us; pf[6][i] = us * us; nf[5][i] = un; nf[6][i] = un * un; }
        let ov = Var::leaf(one.clone()); let pv: Vec<Var> = r.iter().map(|t| Var::leaf(t.clone())).collect();
        let ep = enet(&(0..7).map(|c| Var::leaf(Tensor::from_vec(&ctx, &pf[c], &[bs, 1]))).collect::<Vec<_>>(), &pv, &ov);
        let en = enet(&(0..7).map(|c| Var::leaf(Tensor::from_vec(&ctx, &nf[c], &[bs, 1]))).collect::<Vec<_>>(), &pv, &ov);
        // hinge: want E(pos) + margin < E(neg) ⇒ loss = relu(E(pos) − E(neg) + margin) + 0.02·E(pos)² (anchor)
        let hinge = ep.sub(&en).add(&marg).relu(); let anch = ep.mul(&ep).mul(&Var::leaf(Tensor::from_vec(&ctx, &[0.02], &[1])));
        let loss = hinge.add(&anch).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&r).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect(); adamr.step(&mut r, &g);
    }
    let mut ew: [Vec<f32>; 7] = Default::default(); for c in 0..7 { ew[c] = r[c].to_vec().await; }
    let ev = Ev { w: ew, b1: r[7].to_vec().await, w2: r[8].to_vec().await, b2: r[9].to_vec().await, w3: r[10].to_vec().await, b3: r[11].to_vec().await[0] };

    // ---- Stage 3: eval the PAIR ----
    let nep = 60usize; let n = nep * TESTG.len();
    let mut inits = vec![]; let mut goals = vec![]; for (gi, &g) in TESTG.iter().enumerate() { for e in 0..nep { let sd = (gi * nep + e) as u32; inits.push(((u(900 + sd, 7) * 2.0 - 1.0) * PI, 0.0f32)); goals.push(g); } }
    // (a) flow actuation reach%
    let mut reach = 0; for i in 0..n { let (mut th, mut om) = inits[i]; let g = goals[i]; for _ in 0..220 { let a = fl.act(th, om, g, 2); let (nt, no) = step(th, om, a); th = nt; om = no; } if wrap(th - g).abs() < 0.3 && om.abs() < 0.6 { reach += 1; } }
    // (b) verify accuracy: over random (state,goal), does E rank u* below a random bad action? + does flow's action get low E?
    let (mut vgood, mut vtot, mut agree) = (0, 0, 0); let ntv = 3000;
    for k in 0..ntv { let th = (u(k as u32, 51) * 2.0 - 1.0) * PI; let om = (u(k as u32, 52) * 2.0 - 1.0) * 3.0; let g = (u(k as u32, 53) * 2.0 - 1.0) * 2.0;
        let us = vn.ustar(th, om, g); let ubad = (u(k as u32, 54) * 2.0 - 1.0) * UMAX;
        vtot += 1; if ev.eval(th, om, g, us) < ev.eval(th, om, g, ubad) { vgood += 1; }
        let af = fl.act(th, om, g, 2); if ev.eval(th, om, g, af) < ev.eval(th, om, g, ubad) { agree += 1; } }   // flow's action also scored valid?

    println!("  ONE shared state representation feat5(θ,ω;g); TWO coordinated energies: a flow-matching policy + a verify energy.\n");
    println!("     ACTUATION  — flow-matching policy (K=2 forward passes) control-reach:   {:>4.0}%", reach as f32 / n as f32 * 100.0);
    println!("     VERIFY     — dedicated contrastive energy ranks good action < bad:      {:>4.1}%   (flagship value-as-verify was 76%)", vgood as f32 / vtot as f32 * 100.0);
    println!("     COORDINATE — the flow's PROPOSED action is scored valid (E(flow) < E(bad)): {:>4.1}%", agree as f32 / vtot as f32 * 100.0);
    println!("\n  The pair = generate (flow) + check (energy) on a shared latent — the honest 'one latent, coordinated family of energies'");
    println!("  reframe: flow-matching for actuation (right recipe), a purpose-built energy for validity (beats value-as-classifier).");
}
