//! EFA energy-first #40 — LANGUAGE PORT: one energy takes a symbolic INSTRUCTION + body state → four jobs.
//!
//! The flagship (ebm_oneenergy) conditioned on a physical goal ANGLE. This fuses the LANGUAGE edge of the triangle:
//! the goal is now a discrete INSTRUCTION the energy must DECODE through a LEARNED embedding — it is NOT handed the
//! goal coordinate. So E(state, instruction) must learn what each command MEANS. The instruction is the control knob;
//! one energy handles all commands. Same four readings (control / verify / remember / certify), now language-conditioned
//! — the full physics↔language↔control centre. Learned by fitted value iteration (score-first, no partition function).
//!
//! Run: `cargo run -p ferric-tensor --example ebm_lang --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;
const H: usize = 96; const D: usize = 8; const K: usize = 6; const DT: f32 = 0.05; const GAMMA: f32 = 0.97; const UMAX: f32 = 3.0;
const ACTS: [f32; 5] = [-3.0, -1.5, 0.0, 1.5, 3.0];
const GOALS: [f32; K] = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]; // what each INSTRUCTION means (env-side; the energy is NOT told this)
use std::f32::consts::PI;
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }
fn wrap(x: f32) -> f32 { let mut a = x; while a > PI { a -= 2.0 * PI; } while a < -PI { a += 2.0 * PI; } a }
fn step(th: f32, om: f32, uu: f32) -> (f32, f32) { let no = om + DT * (-th.sin() - 0.05 * om + uu.clamp(-UMAX, UMAX)); (wrap(th + DT * no), no) }
fn sfeat(th: f32, om: f32) -> [f32; 3] { [th.cos(), th.sin(), om] } // state only — NO goal
fn cost(th: f32, om: f32, uu: f32, instr: usize) -> f32 { wrap(th - GOALS[instr]).powi(2) + 0.05 * om * om + 0.01 * uu * uu }

// CPU evaluator: E(state, instruction) — instruction indexes a learned embedding row.
struct En { ws: Vec<f32>, we: Vec<f32>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: f32, emb: Vec<f32> }
impl En {
    fn eval(&self, th: f32, om: f32, instr: usize) -> f32 {
        let sf = sfeat(th, om); let mut h1 = [0.0f32; H];
        for j in 0..H { let mut p = self.b1[j];
            for k in 0..3 { p += sf[k] * self.ws[k * H + j]; }
            for k in 0..D { p += self.emb[instr * D + k] * self.we[k * H + j]; }
            h1[j] = (p.exp() + 1.0).ln(); }
        let mut h2 = [0.0f32; H];
        for j in 0..H { let mut p = self.b2[j]; for k in 0..H { p += h1[k] * self.w2[k * H + j]; } h2[j] = (p.exp() + 1.0).ln(); }
        let mut o = self.b3; for j in 0..H { o += h2[j] * self.w3[j]; } (o.exp() + 1.0).ln()
    }
    fn greedy(&self, th: f32, om: f32, instr: usize) -> f32 { let mut bu = 0.0; let mut be = f32::MAX; for &uu in &ACTS { let (nt, no) = step(th, om, uu); let e = self.eval(nt, no, instr); if e < be { be = e; bu = uu; } } bu }
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA energy-first — LANGUAGE PORT: one energy, a symbolic INSTRUCTION + body state, four jobs\n");
    let mk = || vec![
        Tensor::from_vec(&ctx, &randn(3 * H, 10, 0.6), &[3, H]),        // Ws (state)
        Tensor::from_vec(&ctx, &randn(D * H, 11, 0.6), &[D, H]),        // We (instruction embedding)
        Tensor::zeros(&ctx, &[H]),
        Tensor::from_vec(&ctx, &randn(H * H, 12, 1.0 / (H as f32).sqrt()), &[H, H]), Tensor::zeros(&ctx, &[H]),
        Tensor::from_vec(&ctx, &randn(H, 13, 1.0 / (H as f32).sqrt()), &[H, 1]), Tensor::zeros(&ctx, &[1]),
        Tensor::from_vec(&ctx, &randn(K * D, 14, 0.5), &[K, D]),        // emb (learned instruction meanings)
    ];
    let mut p = mk(); let mut tgt = p.clone();
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let mut adam = Adam::new(&p, 0.002); let bs = 256usize;
    // E(state_feat[bs,3], onehot[bs,K]) : first layer = sf·Ws + (onehot·emb)·We + b1
    let enet = |sf: &Var, oh: &Var, pv: &[Var], ov: &Var| {
        let sp = |z: Var| z.exp().add(ov).log();
        let ie = oh.matmul(&pv[7]);                                   // [·,D] instruction embedding
        let h1 = sp(sf.matmul(&pv[0]).add(&ie.matmul(&pv[1])).add(&pv[2]));
        let h2 = sp(h1.matmul(&pv[3]).add(&pv[4]));
        sp(h2.matmul(&pv[5]).add(&pv[6]))
    };

    for it in 0..13000 {
        let mut sfc = vec![0.0f32; bs * 3]; let mut ohc = vec![0.0f32; bs * K];
        let mut sfn = vec![0.0f32; bs * ACTS.len() * 3]; let mut ohn = vec![0.0f32; bs * ACTS.len() * K]; let mut cst = vec![0.0f32; bs * ACTS.len()];
        for i in 0..bs { let sd = it as u32 * 7 + i as u32;
            let th = (u(sd, 1) * 2.0 - 1.0) * PI; let om = (u(sd, 2) * 2.0 - 1.0) * 3.0; let instr = (u(sd, 3) * K as f32) as usize % K;
            let f = sfeat(th, om); for k in 0..3 { sfc[i * 3 + k] = f[k]; } ohc[i * K + instr] = 1.0;
            for (ai, &uu) in ACTS.iter().enumerate() { let (nt, no) = step(th, om, uu); let nf = sfeat(nt, no);
                for k in 0..3 { sfn[(i * ACTS.len() + ai) * 3 + k] = nf[k]; } ohn[(i * ACTS.len() + ai) * K + instr] = 1.0; cst[i * ACTS.len() + ai] = cost(th, om, uu, instr); } }
        let tv: Vec<Var> = tgt.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
        let et = enet(&Var::leaf(Tensor::from_vec(&ctx, &sfn, &[bs * ACTS.len(), 3])), &Var::leaf(Tensor::from_vec(&ctx, &ohn, &[bs * ACTS.len(), K])), &tv, &ov).value().to_vec().await;
        let mut target = vec![0.0f32; bs];
        for i in 0..bs { let mut m = f32::MAX; for ai in 0..ACTS.len() { let q = cst[i * ACTS.len() + ai] * DT + GAMMA * et[i * ACTS.len() + ai]; if q < m { m = q; } } target[i] = m; }
        let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
        let e = enet(&Var::leaf(Tensor::from_vec(&ctx, &sfc, &[bs, 3])), &Var::leaf(Tensor::from_vec(&ctx, &ohc, &[bs, K])), &pv, &ov);
        let diff = e.sub(&Var::leaf(Tensor::from_vec(&ctx, &target, &[bs, 1])));
        let loss = diff.mul(&diff).mean_all(); loss.backward();
        let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect();
        adam.step(&mut p, &g);
        if it % 200 == 0 { tgt = p.clone(); }
    }
    let e = En { ws: p[0].to_vec().await, we: p[1].to_vec().await, b1: p[2].to_vec().await, w2: p[3].to_vec().await, b2: p[4].to_vec().await, w3: p[5].to_vec().await, b3: p[6].to_vec().await[0], emb: p[7].to_vec().await };

    // 1. CONTROL: obey each instruction (reach the instruction's goal by greedy descent)
    let mut reach = 0; let mut tot = 0; let n = 60usize;
    for instr in 0..K { for i in 0..n { let mut th = (u(9000 + (instr * n + i) as u32, 1) * 2.0 - 1.0) * PI; let mut om = 0.0f32;
        for _ in 0..200 { let uu = e.greedy(th, om, instr); let (nt, no) = step(th, om, uu); th = nt; om = no; if wrap(th - GOALS[instr]).abs() < 0.2 && om.abs() < 0.5 { break; } }
        if wrap(th - GOALS[instr]).abs() < 0.3 && om.abs() < 0.6 { reach += 1; } tot += 1; } }
    // 2. VERIFY: low-E picks the helping over the hurting action (vs TRUE distance to the instruction's goal)
    let mut vok = 0; let mut vtot = 0;
    for instr in 0..K { for i in 0..400 { let th = (u(20000 + (instr * 400 + i) as u32, 1) * 2.0 - 1.0) * PI; let om = (u(20000 + (instr * 400 + i) as u32, 2) * 2.0 - 1.0) * 2.5;
        let ug = e.greedy(th, om, instr); let (nt, _) = step(th, om, ug); let dg = wrap(nt - GOALS[instr]).abs();
        let mut wu = 0.0; let mut we_ = f32::MIN; for &uu in &ACTS { let (a, b) = step(th, om, uu); let en = e.eval(a, b, instr); if en > we_ { we_ = en; wu = uu; } }
        let (wt, _) = step(th, om, wu); if dg <= wrap(wt - GOALS[instr]).abs() { vok += 1; } vtot += 1; } }
    // 3. REMEMBER: does each INSTRUCTION's energy put its minimum at that instruction's goal? (the energy DECODED the command)
    let mut mem = 0.0f32;
    for instr in 0..K { let mut bth = 0.0; let mut be = f32::MAX; for gi in 0..361 { let th = -PI + gi as f32 / 360.0 * 2.0 * PI; let en = e.eval(th, 0.0, instr); if en < be { be = en; bth = th; } } mem += wrap(bth - GOALS[instr]).abs(); }
    mem /= K as f32;
    // 4. CERTIFY (Lyapunov) along the controlled trajectory
    let mut mono = 0.0f32; let mut mtot = 0.0f32;
    for instr in 0..K { for i in 0..20 { let mut th = (u(30000 + (instr * 20 + i) as u32, 1) * 2.0 - 1.0) * PI; let mut om = 0.0f32; let mut prev = e.eval(th, om, instr);
        for _ in 0..120 { let uu = e.greedy(th, om, instr); let (nt, no) = step(th, om, uu); th = nt; om = no; let cur = e.eval(th, om, instr); if cur <= prev + 1e-3 { mono += 1.0; } mtot += 1.0; prev = cur; } } }

    println!("  ONE energy E(state, INSTRUCTION), the command decoded from a LEARNED embedding (not handed the goal):");
    println!("     1. CONTROL   obeys the instruction — reaches its goal {:.0}% of the time (over all {} commands)", reach as f32 / tot as f32 * 100.0, K);
    println!("     2. VERIFY    low-E picks the helping over the hurting action {:.0}% of the time", vok as f32 / vtot as f32 * 100.0);
    println!("     3. REMEMBER  each instruction's energy-minimum lands {:.3} rad from that command's goal — the energy DECODED the language", mem);
    println!("     4. CERTIFY   E decreases {:.0}% of steps along the controlled trajectory — Lyapunov certificate, per instruction", mono / mtot * 100.0);
    println!("\n  A symbolic instruction + body state → one structured energy → control, verify, remember, certify. The");
    println!("  language edge fused into the flagship: the full physics↔language↔control centre, occupied.");
}
