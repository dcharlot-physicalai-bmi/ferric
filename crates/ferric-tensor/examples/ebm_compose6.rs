//! EFA energy-first #45 — DECISIVE SPLITTER: is compositional CONTROL blocked by factor PLACEMENT or by ENERGY SHAPE?
//!
//! compose5's marriage left held-out control at 26%. My diagnosis (relative feature too tolerant → factors drift)
//! silently conflates TWO mechanisms with OPPOSITE fixes: (i) factor under-identification/placement, and (ii) the
//! residual energy's control shape — the ABSOLUTE cosθ/sinθ inputs can memorize goal-specific tilt at the 4 train
//! angles. This freezes the factors at their TRUE values (a=[-1,1], b=[0.5,1.5,2.5]) and re-runs FVI. If held-out
//! control transfers with PERFECT factors → placement was the whole problem (a decode-pinning fix suffices). If
//! control stays weak even with perfect factors → the energy shape is co-responsible and only a structural reshape
//! can fix it. Arm B freezes at compose5's DRIFTED factors and compares final FVI MSE (tests the tolerance claim).
//!
//! Run: `cargo run -p ferric-tensor --example ebm_compose6 --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;
const H: usize = 96; const DT: f32 = 0.05; const GAMMA: f32 = 0.97; const UMAX: f32 = 3.0;
const ACTS: [f32; 5] = [-3.0, -1.5, 0.0, 1.5, 3.0];
const DIST: [f32; 3] = [0.5, 1.5, 2.5];
const TRAIN: [(usize, usize); 4] = [(0, 0), (0, 2), (1, 0), (1, 1)];
const HELD: [(usize, usize); 2] = [(0, 1), (1, 2)];
use std::f32::consts::PI;
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }
fn wrap(x: f32) -> f32 { let mut a = x; while a > PI { a -= 2.0 * PI; } while a < -PI { a += 2.0 * PI; } a }
fn step(th: f32, om: f32, uu: f32) -> (f32, f32) { let no = om + DT * (-th.sin() - 0.05 * om + uu.clamp(-UMAX, UMAX)); (wrap(th + DT * no), no) }
fn goal(dir: usize, dist: usize) -> f32 { (if dir == 0 { -1.0 } else { 1.0 }) * DIST[dist] }

struct En { da: Vec<f32>, db: Vec<f32>, wrc: Vec<f32>, wrs: Vec<f32>, wo: Vec<f32>, wc: Vec<f32>, ws: Vec<f32>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: f32 }
impl En {
    fn eval(&self, th: f32, om: f32, dir: usize, dist: usize) -> f32 {
        let gth = self.da[dir] * self.db[dist];
        let (c, s) = (th.cos(), th.sin()); let (cg, sg) = (gth.cos(), gth.sin());
        let rc = c * cg + s * sg; let rs = s * cg - c * sg;
        let mut h1 = [0.0f32; H];
        for j in 0..H { let p = self.b1[j] + rc * self.wrc[j] + rs * self.wrs[j] + om * self.wo[j] + c * self.wc[j] + s * self.ws[j]; h1[j] = (p.exp() + 1.0).ln(); }
        let mut h2 = [0.0f32; H];
        for j in 0..H { let mut p = self.b2[j]; for k in 0..H { p += h1[k] * self.w2[k * H + j]; } h2[j] = (p.exp() + 1.0).ln(); }
        let mut o = self.b3; for j in 0..H { o += h2[j] * self.w3[j]; } (o.exp() + 1.0).ln()
    }
    fn greedy(&self, th: f32, om: f32, dir: usize, dist: usize) -> f32 { let mut bu = 0.0; let mut be = f32::MAX; for &uu in &ACTS { let (nt, no) = step(th, om, uu); let e = self.eval(nt, no, dir, dist); if e < be { be = e; bu = uu; } } bu }
    // per-command: (energy-min decode err in rad, control-reach %, settled-window reach %)
    fn assess(&self, dir: usize, dist: usize) -> (f32, f32) {
        let g = goal(dir, dist);
        let mut bth = 0.0; let mut be = f32::MAX; for gi in 0..361 { let th = -PI + gi as f32 / 360.0 * 2.0 * PI; let en = self.eval(th, 0.0, dir, dist); if en < be { be = en; bth = th; } }
        let mut reach = 0; let nn = 60; for i in 0..nn { let mut th = (u(700 + (dir * 200 + dist * 60 + i) as u32, 1) * 2.0 - 1.0) * PI; let mut om = 0.0f32;
            // settle: run 200 steps, then check reach over a last-K window (anti-timing-luck)
            let mut ok_window = true;
            for t in 0..260 { let uu = self.greedy(th, om, dir, dist); let (nt, no) = step(th, om, uu); th = nt; om = no;
                if t >= 200 { if !(wrap(th - g).abs() < 0.3 && om.abs() < 0.6) { ok_window = false; } } }
            if ok_window { reach += 1; } }
        (wrap(bth - g).abs(), reach as f32 / nn as f32 * 100.0)
    }
}

// FVI with factors FROZEN at (da_init, db_init). Returns (En, mean final-window MSE).
async fn train_frozen(ctx: &Arc<ferric_core::Context>, da_init: [f32; 2], db_init: [f32; 3]) -> (En, f32) {
    let mut p = vec![
        Tensor::from_vec(ctx, &da_init, &[2, 1]), Tensor::from_vec(ctx, &db_init, &[3, 1]),
        Tensor::from_vec(ctx, &randn(H, 22, 0.6), &[1, H]), Tensor::from_vec(ctx, &randn(H, 23, 0.6), &[1, H]),
        Tensor::from_vec(ctx, &randn(H, 24, 0.6), &[1, H]), Tensor::from_vec(ctx, &randn(H, 25, 0.6), &[1, H]), Tensor::from_vec(ctx, &randn(H, 26, 0.6), &[1, H]), Tensor::zeros(ctx, &[H]),
        Tensor::from_vec(ctx, &randn(H * H, 27, 1.0 / (H as f32).sqrt()), &[H, H]), Tensor::zeros(ctx, &[H]),
        Tensor::from_vec(ctx, &randn(H, 28, 1.0 / (H as f32).sqrt()), &[H, 1]), Tensor::zeros(ctx, &[1]),
    ];
    let mut tgt = p.clone();
    let one = Tensor::from_vec(ctx, &[1.0], &[1]); let mut adam = Adam::new(&p, 0.002); let bs = 256usize;
    let enet = |cth: &Var, sth: &Var, om: &Var, ohd: &Var, ohm: &Var, pv: &[Var], ov: &Var| {
        let sp = |z: Var| z.exp().add(ov).log();
        let a = ohd.matmul(&pv[0]); let b = ohm.matmul(&pv[1]); let gth = a.mul(&b);
        let cg = gth.cos(); let sg = gth.sin();
        let rc = cth.mul(&cg).add(&sth.mul(&sg)); let rs = sth.mul(&cg).sub(&cth.mul(&sg));
        let pre = rc.matmul(&pv[2]).add(&rs.matmul(&pv[3])).add(&om.matmul(&pv[4])).add(&cth.matmul(&pv[5])).add(&sth.matmul(&pv[6])).add(&pv[7]);
        sp(sp(sp(pre).matmul(&pv[8]).add(&pv[9])).matmul(&pv[10]).add(&pv[11]))
    };
    let mut mse_acc = 0.0f32; let mut mse_n = 0.0f32;
    for it in 0..14000 {
        let (mut c_, mut s_, mut om_) = (vec![0.0f32; bs], vec![0.0f32; bs], vec![0.0f32; bs]); let mut ohd = vec![0.0f32; bs * 2]; let mut ohm = vec![0.0f32; bs * 3];
        let (mut nc, mut ns, mut nom) = (vec![0.0f32; bs * 5], vec![0.0f32; bs * 5], vec![0.0f32; bs * 5]); let mut nohd = vec![0.0f32; bs * 5 * 2]; let mut nohm = vec![0.0f32; bs * 5 * 3]; let mut cst = vec![0.0f32; bs * 5];
        for i in 0..bs { let sd = it as u32 * 7 + i as u32; let (dir, dist) = TRAIN[(u(sd, 4) * 4.0) as usize % 4]; let g = goal(dir, dist);
            let th = (u(sd, 1) * 2.0 - 1.0) * PI; let om = (u(sd, 2) * 2.0 - 1.0) * 3.0;
            c_[i] = th.cos(); s_[i] = th.sin(); om_[i] = om; ohd[i * 2 + dir] = 1.0; ohm[i * 3 + dist] = 1.0;
            for (ai, &uu) in ACTS.iter().enumerate() { let (nt, no) = step(th, om, uu); nc[i * 5 + ai] = nt.cos(); ns[i * 5 + ai] = nt.sin(); nom[i * 5 + ai] = no;
                nohd[(i * 5 + ai) * 2 + dir] = 1.0; nohm[(i * 5 + ai) * 3 + dist] = 1.0; cst[i * 5 + ai] = wrap(th - g).powi(2) + 0.05 * om * om + 0.01 * uu * uu; } }
        let l = |v: &[f32], r: usize| Var::leaf(Tensor::from_vec(ctx, v, &[r, 1])); let l2 = |v: &[f32], r: usize, c: usize| Var::leaf(Tensor::from_vec(ctx, v, &[r, c]));
        let tv: Vec<Var> = tgt.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
        let et = enet(&l(&nc, bs * 5), &l(&ns, bs * 5), &l(&nom, bs * 5), &l2(&nohd, bs * 5, 2), &l2(&nohm, bs * 5, 3), &tv, &ov).value().to_vec().await;
        let mut target = vec![0.0f32; bs]; for i in 0..bs { let mut m = f32::MAX; for ai in 0..5 { let q = cst[i * 5 + ai] * DT + GAMMA * et[i * 5 + ai]; if q < m { m = q; } } target[i] = m; }
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
        let e = enet(&l(&c_, bs), &l(&s_, bs), &l(&om_, bs), &l2(&ohd, bs, 2), &l2(&ohm, bs, 3), &pv, &ov);
        let diff = e.sub(&Var::leaf(Tensor::from_vec(ctx, &target, &[bs, 1])));
        let loss = diff.mul(&diff).mean_all(); loss.backward();
        let mut g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        // FREEZE the factor tables: zero their grads so Adam never moves them.
        g[0] = Tensor::from_vec(ctx, &vec![0.0; p[0].numel()], &p[0].shape);
        g[1] = Tensor::from_vec(ctx, &vec![0.0; p[1].numel()], &p[1].shape);
        adam.step(&mut p, &g);
        if it % 200 == 0 { tgt = p.clone(); }
        if it >= 13500 { mse_acc += loss.value().to_vec().await[0]; mse_n += 1.0; }
    }
    let e = En { da: p[0].to_vec().await, db: p[1].to_vec().await, wrc: p[2].to_vec().await, wrs: p[3].to_vec().await, wo: p[4].to_vec().await, wc: p[5].to_vec().await, ws: p[6].to_vec().await, b1: p[7].to_vec().await, w2: p[8].to_vec().await, b2: p[9].to_vec().await, w3: p[10].to_vec().await, b3: p[11].to_vec().await[0] };
    (e, mse_acc / mse_n)
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — SPLITTER: does compositional CONTROL transfer with PERFECT (frozen-at-truth) factors?\n");
    let names = |d: usize, m: usize| format!("{}·{}", if d == 0 { "left " } else { "right" }, ["near", "mid ", "far "][m]);
    let report = |tag: &str, e: &En| {
        println!("  [{}]  factors da=({:+.2},{:+.2}) db=({:.2},{:.2},{:.2})", tag, e.da[0], e.da[1], e.db[0], e.db[1], e.db[2]);
        println!("     command       goal    decode-err   control   split");
        for dir in 0..2 { for dist in 0..3 { let (rr, cc) = e.assess(dir, dist); let held = HELD.contains(&(dir, dist));
            let pass = if held { if rr < 0.25 && cc >= 60.0 { "  PASS" } else { "  FAIL" } } else { "" };
            println!("     {}  {:>5.1}   {:>6.3} rad   {:>4.0}%    {}{}", names(dir, dist), goal(dir, dist), rr, cc, if held { "HELD-OUT" } else { "train" }, pass); } }
    };

    // ARM A — freeze at TRUTH. The decisive run.
    let (e_truth, mse_truth) = train_frozen(&ctx, [-1.0, 1.0], [0.5, 1.5, 2.5]).await;
    report("FROZEN-AT-TRUTH", &e_truth);
    println!("     final-window MSE = {:.5}\n", mse_truth);

    // ARM B — freeze at compose5's DRIFTED factors, to test the TOLERANCE claim via final MSE.
    let (e_drift, mse_drift) = train_frozen(&ctx, [-1.05, 0.71], [0.57, 1.93, 2.53]).await;
    report("FROZEN-AT-DRIFTED (compose5)", &e_drift);
    println!("     final-window MSE = {:.5}\n", mse_drift);

    // VERDICT
    let hd: Vec<(f32, f32)> = HELD.iter().map(|&(d, m)| e_truth.assess(d, m)).collect();
    let both_control = hd.iter().all(|&(_, c)| c >= 60.0);
    let both_decode = hd.iter().all(|&(r, _)| r < 0.25);
    println!("  ── VERDICT ──────────────────────────────────────────────────────────────");
    println!("  With TRUE factors, held-out: {}", hd.iter().zip(HELD.iter()).map(|(&(r, c), &(d, m))| format!("{} {:.2}rad/{:.0}%", names(d, m), r, c)).collect::<Vec<_>>().join("   "));
    if both_control && both_decode {
        println!("  → CONTROL TRANSFERS with perfect factors ⇒ PLACEMENT was the whole problem.");
        println!("    A decode-pinning fix suffices (two-stage grounding, or aux factor-pin loss). Energy shape is fine.");
    } else if both_decode && !both_control {
        println!("  → decode fine but CONTROL still weak WITH PERFECT FACTORS ⇒ the ENERGY SHAPE (absolute cosθ/sinθ leakage)");
        println!("    is co-responsible. No factor-pinning fix can rescue control — need the STRUCTURAL WELL (λ(1−cos(θ−gθ))).");
    } else {
        println!("  → mixed/other: read the per-command rows above; the frozen-truth control shows the achievable ceiling.");
    }
    println!("  TOLERANCE probe: MSE(truth)={:.5} vs MSE(drifted)={:.5} → {}", mse_truth, mse_drift,
        if mse_drift <= mse_truth + 1e-5 { "drifted ≤ truth: FVI has NO incentive to pin (tolerance CONFIRMED)" } else { "drifted > truth: factors identifiable, tolerance NOT the blocker" });
}
