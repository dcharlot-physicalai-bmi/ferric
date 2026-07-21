//! EFA energy-first #3 — OOD-GOAL generalization: energy-descent planning vs behavior-cloned feed-forward.
//!
//! The EBM edge is largest exactly where inputs are OOD. Direct test on control: train BOTH a goal-conditioned
//! VALUE/energy V(x,g) (act by energy-descent/MPPI — EFA's planner) AND a matched behavior-cloned feed-forward
//! POLICY π(x,g)→action, on goals drawn ONLY from a restricted region (e.g. the RIGHT half of an open grid).
//! Then test on goals from the LEFT half — OUT of the training goal-distribution. Hypothesis: the energy-descent
//! planner (which composes the learned dynamics with ANY goal at inference) generalizes to unseen goals, while
//! the feed-forward policy (which memorized a state×goal→action map over the training goals) fails OOD. Same
//! world model, same data budget; the only difference is plan-at-test-time vs a reactive map.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_ood_goal --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

const G: usize = 13;
const IN: usize = G * G;
const NA: usize = 4;
const MOVES: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];
const D: usize = 64;

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }
fn clampc(v: i32) -> usize { v.max(0).min(G as i32 - 1) as usize }
fn next_pos(p: (usize, usize), a: usize) -> (usize, usize) { (clampc(p.0 as i32 + MOVES[a].0), clampc(p.1 as i32 + MOVES[a].1)) }
fn cell(seed: u32) -> (usize, usize) { ((u(seed, 1) * G as f32) as usize % G, (u(seed, 2) * G as f32) as usize % G) }
// training goals: RIGHT half only (cols 7..12). test OOD goals: LEFT half (cols 0..5).
fn train_goal(seed: u32) -> (usize, usize) { (7 + (u(seed, 3) * 6.0) as usize % 6, (u(seed, 4) * G as f32) as usize % G) }
fn ood_goal(seed: u32) -> (usize, usize) { ((u(seed, 3) * 6.0) as usize % 6, (u(seed, 4) * G as f32) as usize % G) }
fn blob(p: (usize, usize), out: &mut [f32]) { for y in 0..G { for x in 0..G { let dx = x as f32 - p.0 as f32; let dy = y as f32 - p.1 as f32; out[y * G + x] = (-(dx * dx + dy * dy) / (2.0 * 1.1 * 1.1)).exp(); } } }
fn opt_act(p: (usize, usize), g: (usize, usize)) -> usize { // BFS-free: greedy manhattan (open grid → exact)
    let mut ba = 0; let mut bd = i32::MAX;
    for a in 0..NA { let nc = next_pos(p, a); let d = (nc.0 as i32 - g.0 as i32).abs() + (nc.1 as i32 - g.1 as i32).abs(); if d < bd { bd = d; ba = a; } }
    ba
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let (hid, hv, n) = (256usize, 512usize, 512usize);
    println!("  EFA energy-first — OOD-GOAL generalization: energy-descent planner vs behavior-cloned policy. D={D}");
    println!("  train goals = RIGHT half (cols 7-12); test goals = LEFT half (cols 0-5) = OUT of goal-distribution.");

    // ---- shared world model (encoder + JEPA predictor) on open-grid dynamics (goal-independent) ----
    let mut wp = vec![
        Tensor::from_vec(&ctx, &randn(IN * D, 1, 1.0 / (IN as f32).sqrt()), &[IN, D]), Tensor::zeros(&ctx, &[D]),
        Tensor::from_vec(&ctx, &randn(D * hid, 2, 1.0 / (D as f32).sqrt()), &[D, hid]),
        Tensor::from_vec(&ctx, &randn(NA * hid, 3, 1.0 / (NA as f32).sqrt()), &[NA, hid]), Tensor::zeros(&ctx, &[hid]),
        Tensor::from_vec(&ctx, &randn(hid * D, 4, 1.0 / (hid as f32).sqrt()), &[hid, D]), Tensor::zeros(&ctx, &[D]),
    ];
    let mut wadam = Adam::new(&wp, 0.002);
    let eps = Tensor::from_vec(&ctx, &[1e-6], &[1]);
    let (epsv, one, lamv) = (Tensor::from_vec(&ctx, &[1e-4], &[1]), Tensor::from_vec(&ctx, &[1.0], &[1]), Tensor::from_vec(&ctx, &[0.1], &[1]));
    let enc_v = |o: &Var, p: &[Var], e: &Var| -> Var { let h = o.matmul(&p[0]).add(&p[1]).relu(); let ss = h.mul(&h).sum(&[1]); h.div(&ss.add(e).sqrt()) };
    for step in 0..1400 {
        let (mut ob, mut oh, mut no) = (vec![0.0f32; n * IN], vec![0.0f32; n * NA], vec![0.0f32; n * IN]);
        for i in 0..n { let p = cell(step as u32 * 131 + i as u32); let a = (u(i as u32, step as u32 ^ 7) * NA as f32) as usize % NA; let pn = next_pos(p, a); blob(p, &mut ob[i * IN..(i + 1) * IN]); blob(pn, &mut no[i * IN..(i + 1) * IN]); oh[i * NA + a] = 1.0; }
        let p: Vec<Var> = wp.iter().map(|t| Var::leaf(t.clone())).collect();
        let ev = Var::leaf(eps.clone());
        let ov = Var::leaf(Tensor::from_vec(&ctx, &ob, &[n, IN])); let nv = Var::leaf(Tensor::from_vec(&ctx, &no, &[n, IN])); let ohv = Var::leaf(Tensor::from_vec(&ctx, &oh, &[n, NA]));
        let x = enc_v(&ov, &p, &ev); let target = enc_v(&nv, &p, &ev).detach();
        let hp = x.matmul(&p[2]).add(&ohv.matmul(&p[3])).add(&p[4]).relu(); let o = hp.matmul(&p[5]).add(&p[6]);
        let diff = o.sub(&target); let mse = diff.mul(&diff).mean_all();
        let xc = x.sub(&x.mean(&[0])); let std = xc.mul(&xc).mean(&[0]).add(&Var::leaf(epsv.clone())).sqrt();
        let vloss = Var::leaf(one.clone()).sub(&std).relu().mean_all();
        let loss = mse.add(&vloss.mul(&Var::leaf(lamv.clone()))); loss.backward();
        let grads: Vec<Tensor> = p.iter().map(|v| v.grad().unwrap()).collect(); wadam.step(&mut wp, &grads);
    }
    let e2 = eps.clone();
    let sq = move |t: &Tensor| -> Tensor { let ss = t.mul(t).sum(&[1], true); t.div(&ss.add(&e2).sqrt()) };
    let encode_t = |obs: &Tensor| -> Tensor { sq(&obs.matmul(&wp[0]).add(&wp[1]).relu()) };
    let predict_t = |x: &Tensor, oh: &Tensor| -> Tensor { let hp = x.matmul(&wp[2]).add(&oh.matmul(&wp[3])).add(&wp[4]).relu(); sq(&hp.matmul(&wp[5]).add(&wp[6]).relu()) };
    let enc_batch = |cells: &[(usize, usize)]| -> Tensor { let m = cells.len(); let mut o = vec![0.0f32; m * IN]; for i in 0..m { blob(cells[i], &mut o[i * IN..(i + 1) * IN]); } encode_t(&Tensor::from_vec(&ctx, &o, &[m, IN])) };

    // ---- value/energy V(x,g) via FVI+target, goals sampled ONLY from the training (right) region ----
    let mut vp = vec![
        Tensor::from_vec(&ctx, &randn(D * hv, 10, 1.0 / (D as f32).sqrt()), &[D, hv]),
        Tensor::from_vec(&ctx, &randn(D * hv, 11, 1.0 / (D as f32).sqrt()), &[D, hv]), Tensor::zeros(&ctx, &[hv]),
        Tensor::from_vec(&ctx, &randn(hv, 12, 1.0 / (hv as f32).sqrt()), &[hv, 1]), Tensor::zeros(&ctx, &[1]),
    ];
    let mut vadam = Adam::new(&vp, 0.003);
    let val_v = |xv: &Var, gv: &Var, p: &[Var]| -> Var { let h = xv.matmul(&p[0]).add(&gv.matmul(&p[1])).add(&p[2]).relu(); h.matmul(&p[3]).add(&p[4]) };
    let val_t = |x: &Tensor, g: &Tensor, p: &[Tensor]| -> Tensor { let h = x.matmul(&p[0]).add(&g.matmul(&p[1])).add(&p[2]).relu(); h.matmul(&p[3]).add(&p[4]) };
    let gamma_t = Tensor::from_vec(&ctx, &[0.95f32], &[1]); let bb = 1024usize;
    let mut vp_t: Vec<Tensor> = vp.iter().map(|t| t.clone()).collect();
    for it in 0..9000 {
        let mut cx: Vec<(usize, usize)> = Vec::with_capacity(bb); let mut cg: Vec<(usize, usize)> = Vec::with_capacity(bb);
        for i in 0..bb { let s = it as u32 * 1103 + i as u32; cx.push(cell(s)); cg.push(train_goal(s ^ 0x9e37_79b9)); } // goals: RIGHT only
        let (x, g) = (enc_batch(&cx), enc_batch(&cg));
        let mut target: Option<Tensor> = None;
        for a in 0..NA {
            let ncells: Vec<(usize, usize)> = cx.iter().map(|&c| next_pos(c, a)).collect();
            let reached: Vec<f32> = ncells.iter().zip(&cg).map(|(&nc, &gc)| if nc == gc { 1.0 } else { 0.0 }).collect();
            let vn = val_t(&enc_batch(&ncells), &g, &vp_t);
            let rt = Tensor::from_vec(&ctx, &reached, &[bb, 1]);
            let ta = rt.add(&one.sub(&rt).mul(&vn).mul(&gamma_t));
            target = Some(match target { None => ta, Some(t) => t.maximum(&ta) });
        }
        let target = target.unwrap();
        let p: Vec<Var> = vp.iter().map(|t| Var::leaf(t.clone())).collect();
        let v = val_v(&Var::leaf(x), &Var::leaf(g), &p);
        let diff = v.sub(&Var::leaf(target)); let loss = diff.mul(&diff).mean_all(); loss.backward();
        let grads: Vec<Tensor> = p.iter().map(|vv| vv.grad().unwrap()).collect(); vadam.step(&mut vp, &grads);
        if it % 250 == 249 { vp_t = vp.iter().map(|t| t.clone()).collect(); }
    }

    // ---- behavior-cloned feed-forward policy π(enc(x),enc(g)) → action, SAME goals (right only), SAME data ----
    let mut pp = vec![
        Tensor::from_vec(&ctx, &randn(D * hv, 20, 1.0 / (D as f32).sqrt()), &[D, hv]),
        Tensor::from_vec(&ctx, &randn(D * hv, 21, 1.0 / (D as f32).sqrt()), &[D, hv]), Tensor::zeros(&ctx, &[hv]),
        Tensor::from_vec(&ctx, &randn(hv * NA, 22, 1.0 / (hv as f32).sqrt()), &[hv, NA]), Tensor::zeros(&ctx, &[NA]),
    ];
    let mut padam = Adam::new(&pp, 0.003);
    for it in 0..9000 {
        let mut cx: Vec<(usize, usize)> = Vec::with_capacity(bb); let mut cg: Vec<(usize, usize)> = Vec::with_capacity(bb);
        let mut lab = vec![0.0f32; bb * NA];
        for i in 0..bb { let s = it as u32 * 1301 + i as u32 + 7; let c = cell(s); let gg = train_goal(s ^ 0x1234_5678); cx.push(c); cg.push(gg); lab[i * NA + opt_act(c, gg)] = 1.0; } // expert = greedy-to-goal
        let (x, g) = (enc_batch(&cx), enc_batch(&cg));
        let pv: Vec<Var> = pp.iter().map(|t| Var::leaf(t.clone())).collect();
        let h = Var::leaf(x).matmul(&pv[0]).add(&Var::leaf(g).matmul(&pv[1])).add(&pv[2]).relu();
        let logits = h.matmul(&pv[3]).add(&pv[4]);
        let logp = logits.softmax(1).add(&Var::leaf(eps.clone())).log();
        let loss = Var::leaf(Tensor::from_vec(&ctx, &lab, &[bb, NA])).mul(&logp).mean_all().neg(); // cross-entropy
        loss.backward();
        let grads: Vec<Tensor> = pv.iter().map(|vv| vv.grad().unwrap()).collect(); padam.step(&mut pp, &grads);
    }

    // ---- eval reach on TRAIN goals (right) and OOD goals (left) for both actors ----
    let goalset = |ood: bool| -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
        let pos: Vec<(usize, usize)> = (0..150).map(|i| cell(50 + i as u32 * 3)).collect();
        let goal: Vec<(usize, usize)> = (0..150).map(|i| if ood { ood_goal(900 + i as u32 * 5) } else { train_goal(900 + i as u32 * 5) }).collect();
        (pos, goal)
    };
    let onehot = |a: usize, t: usize| -> Tensor { let mut o = vec![0.0f32; t * NA]; for i in 0..t { o[i * NA + a] = 1.0; } Tensor::from_vec(&ctx, &o, &[t, NA]) };

    let mut pr = [0.0f32; 2]; let mut cr = [0.0f32; 2]; let mut dr = [0.0f32; 2];
    for (oi, ood) in [false, true].into_iter().enumerate() {
        // GOAL-AGNOSTIC energy-descent: E = ‖predict(x,a) − enc(goal)‖²; pick action minimizing latent distance.
        // The encoder/dynamics are goal-independent → this energy is DEFINED for any goal ⇒ should generalize OOD.
        { let (mut pos, goal) = goalset(ood); let t = pos.len(); let xg = enc_batch(&goal); let mut done = vec![false; t];
          for _ in 0..50 { if done.iter().all(|&d| d) { break; }
            let x = enc_batch(&pos); let mut dist: Vec<Vec<f32>> = Vec::new();
            for a in 0..NA { let xa = predict_t(&x, &onehot(a, t)); let df = xa.sub(&xg); dist.push(df.mul(&df).sum(&[1], true).to_vec().await); }
            for i in 0..t { if done[i] { continue; } let (mut ba, mut bd) = (0usize, dist[0][i]); for a in 1..NA { if dist[a][i] < bd { bd = dist[a][i]; ba = a; } } pos[i] = next_pos(pos[i], ba); if pos[i] == goal[i] { done[i] = true; } } }
          dr[oi] = done.iter().filter(|&&d| d).count() as f32 / t as f32; }
        // learned goal-conditioned VALUE planner (depth-2 lookahead) — trained only on train-region goals
        { let (mut pos, goal) = goalset(ood); let t = pos.len(); let xg = enc_batch(&goal); let mut done = vec![false; t];
          for _ in 0..50 { if done.iter().all(|&d| d) { break; }
            let x = enc_batch(&pos); let mut sc: Vec<Vec<f32>> = Vec::new();
            for a in 0..NA { let xa = predict_t(&x, &onehot(a, t)); let mut m = val_t(&xa, &xg, &vp);
                for a2 in 0..NA { m = m.maximum(&val_t(&predict_t(&xa, &onehot(a2, t)), &xg, &vp)); } sc.push(m.to_vec().await); }
            for i in 0..t { if done[i] { continue; } let (mut ba, mut bv) = (0usize, sc[0][i]); for a in 1..NA { if sc[a][i] > bv { bv = sc[a][i]; ba = a; } } pos[i] = next_pos(pos[i], ba); if pos[i] == goal[i] { done[i] = true; } } }
          pr[oi] = done.iter().filter(|&&d| d).count() as f32 / t as f32; }
        // behavior-cloned feed-forward policy
        { let (mut pos, goal) = goalset(ood); let t = pos.len(); let xg = enc_batch(&goal); let mut done = vec![false; t];
          for _ in 0..50 { if done.iter().all(|&d| d) { break; }
            let x = enc_batch(&pos);
            let logits = x.matmul(&pp[0]).add(&xg.matmul(&pp[1])).add(&pp[2]).relu().matmul(&pp[3]).add(&pp[4]).to_vec().await;
            for i in 0..t { if done[i] { continue; } let (mut ba, mut bl) = (0usize, logits[i * NA]); for a in 1..NA { if logits[i * NA + a] > bl { bl = logits[i * NA + a]; ba = a; } } pos[i] = next_pos(pos[i], ba); if pos[i] == goal[i] { done[i] = true; } } }
          cr[oi] = done.iter().filter(|&&d| d).count() as f32 / t as f32; }
    }

    println!("\n  reach success — in-distribution (right goals) vs OOD (left goals):");
    println!("     {:<40} {:>10} {:>10}", "actor", "train-goals", "OOD-goals");
    println!("     {:<40} {:>9.0}% {:>9.0}%", "distance-energy descent (goal-AGNOSTIC)", dr[0] * 100.0, dr[1] * 100.0);
    println!("     {:<40} {:>9.0}% {:>9.0}%", "learned value V(x,g) planner (goal-cond.)", pr[0] * 100.0, pr[1] * 100.0);
    println!("     {:<40} {:>9.0}% {:>9.0}%", "behavior-cloned feed-forward policy", cr[0] * 100.0, cr[1] * 100.0);
    println!("\n  the NUANCE: a goal-AGNOSTIC energy (latent distance) generalizes to OOD goals; a LEARNED goal-conditioned");
    println!("  value is itself a goal-map and fails OOD like the feed-forward policy. The energy FORM is what matters.");
}
