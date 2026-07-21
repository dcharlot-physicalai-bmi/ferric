//! EFA — HARDER maze, PUSH ABSOLUTE REACH: bigger models + more training on the S-corridor.
//!
//! efa_maze_hard showed the "small fail / big succeed" scaling signature (0→1→3→19% at 26k→250k params),
//! but 19% absolute is low — the S-corridor is genuinely hard. Question: does reach keep climbing at REAL
//! sizes with more training, or does the curve bend? Same two-wall S-corridor task; we push D/HID/HV up to
//! ~1M+ params and roughly double both training budgets (world model 1800→3200 steps, FVI 12k→22k iters).
//! The 96/384/1024 point is re-run as an anchor (should reproduce ~19%); the honest test is whether the
//! bigger points exceed it — turning the trend into a real absolute number, not just a slope.
//!
//! Run: `cargo run -p ferric-tensor --example efa_maze_hard_big --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

const G: usize = 13;
const IN: usize = G * G;
const NA: usize = 4;
const MOVES: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];
fn is_wall(x: usize, y: usize) -> bool { (x == 4 && y < 9) || (x == 9 && y >= 4) }

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, scale: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * scale }).collect() }
fn clampc(v: i32) -> usize { v.max(0).min(G as i32 - 1) as usize }
fn next_pos(p: (usize, usize), a: usize) -> (usize, usize) { let nx = clampc(p.0 as i32 + MOVES[a].0); let ny = clampc(p.1 as i32 + MOVES[a].1); if is_wall(nx, ny) { p } else { (nx, ny) } }
fn free_cell(seed: u32) -> (usize, usize) { let mut s = seed; loop { let x = (u(s, 1) * G as f32) as usize % G; let y = (u(s, 2) * G as f32) as usize % G; if !is_wall(x, y) { return (x, y); } s = s.wrapping_add(101); } }
fn blob(p: (usize, usize), out: &mut [f32]) { for y in 0..G { for x in 0..G { let dx = x as f32 - p.0 as f32; let dy = y as f32 - p.1 as f32; out[y * G + x] = (-(dx * dx + dy * dy) / (2.0 * 1.1 * 1.1)).exp(); } } }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA — HARD S-corridor, PUSH ABSOLUTE REACH (bigger models + ~2x training)");
    println!("     {:>18}  {:>10}  {:>12}  {:>12}", "size (D/HID/HV)", "params", "WM 8-step fid", "reach");
    let (wm_steps, fvi_iters) = (3200usize, 22000usize);
    for &(d, hid, hv) in &[(96usize, 384usize, 1024usize), (128, 512, 1536), (160, 640, 2048)] {
        let (fid, reach) = size_run(&ctx, d, hid, hv, wm_steps, fvi_iters).await;
        println!("     {:>18}  {:>10}  {:>11.3}  {:>11.0}%", format!("{d}/{hid}/{hv}"), d * hid + IN * d + d * hv * 2, fid, reach * 100.0);
    }
    println!("\n  Anchor 96/384/1024 should reproduce ~19%; the test is whether the bigger points exceed it.");
    println!("  Reach still climbing → absolute reach is a scale/compute run; bending → a harder RL problem remains.");
}

async fn size_run(ctx: &Arc<ferric_core::Context>, d: usize, hid: usize, hv: usize, wm_steps: usize, fvi_iters: usize) -> (f32, f32) {
    let n = 512usize;
    let mut wp = vec![
        Tensor::from_vec(ctx, &randn(IN * d, 1, 1.0 / (IN as f32).sqrt()), &[IN, d]), Tensor::zeros(ctx, &[d]),
        Tensor::from_vec(ctx, &randn(d * hid, 2, 1.0 / (d as f32).sqrt()), &[d, hid]),
        Tensor::from_vec(ctx, &randn(NA * hid, 3, 1.0 / (NA as f32).sqrt()), &[NA, hid]), Tensor::zeros(ctx, &[hid]),
        Tensor::from_vec(ctx, &randn(hid * d, 4, 1.0 / (hid as f32).sqrt()), &[hid, d]), Tensor::zeros(ctx, &[d]),
    ];
    let mut wadam = Adam::new(&wp, 0.002);
    let eps = Tensor::from_vec(ctx, &[1e-6], &[1]);
    let (epsv, one, lamv) = (Tensor::from_vec(ctx, &[1e-4], &[1]), Tensor::from_vec(ctx, &[1.0], &[1]), Tensor::from_vec(ctx, &[0.1], &[1]));
    let enc_v = |o: &Var, p: &[Var], e: &Var| -> Var { let h = o.matmul(&p[0]).add(&p[1]).relu(); let ss = h.mul(&h).sum(&[1]); h.div(&ss.add(e).sqrt()) };
    for step in 0..wm_steps {
        let (mut ob, mut oh, mut no) = (vec![0.0f32; n * IN], vec![0.0f32; n * NA], vec![0.0f32; n * IN]);
        for i in 0..n { let p = free_cell(step as u32 * 131 + i as u32); let a = (u(i as u32, step as u32 ^ 7) * NA as f32) as usize % NA; let pn = next_pos(p, a); blob(p, &mut ob[i * IN..(i + 1) * IN]); blob(pn, &mut no[i * IN..(i + 1) * IN]); oh[i * NA + a] = 1.0; }
        let p: Vec<Var> = wp.iter().map(|t| Var::leaf(t.clone())).collect();
        let ev = Var::leaf(eps.clone());
        let ov = Var::leaf(Tensor::from_vec(ctx, &ob, &[n, IN])); let nv = Var::leaf(Tensor::from_vec(ctx, &no, &[n, IN])); let ohv = Var::leaf(Tensor::from_vec(ctx, &oh, &[n, NA]));
        let x = enc_v(&ov, &p, &ev); let target = enc_v(&nv, &p, &ev).detach();
        let hp = x.matmul(&p[2]).add(&ohv.matmul(&p[3])).add(&p[4]).relu(); let o = hp.matmul(&p[5]).add(&p[6]);
        let diff = o.sub(&target); let mse = diff.mul(&diff).mean_all();
        let xc = x.sub(&x.mean(&[0])); let std = xc.mul(&xc).mean(&[0]).add(&Var::leaf(epsv.clone())).sqrt();
        let vloss = Var::leaf(one.clone()).sub(&std).relu().mean_all();
        let loss = mse.add(&vloss.mul(&Var::leaf(lamv.clone())));
        loss.backward();
        let grads: Vec<Tensor> = p.iter().map(|v| v.grad().unwrap()).collect(); wadam.step(&mut wp, &grads);
    }
    let e2 = eps.clone();
    let sq = move |t: &Tensor| -> Tensor { let ss = t.mul(t).sum(&[1], true); t.div(&ss.add(&e2).sqrt()) };
    let encode_t = |obs: &Tensor| -> Tensor { sq(&obs.matmul(&wp[0]).add(&wp[1]).relu()) };
    let predict_t = |x: &Tensor, oh: &Tensor| -> Tensor { let hp = x.matmul(&wp[2]).add(&oh.matmul(&wp[3])).add(&wp[4]).relu(); sq(&hp.matmul(&wp[5]).add(&wp[6]).relu()) };
    let enc_batch = |cells: &[(usize, usize)]| -> Tensor { let m = cells.len(); let mut o = vec![0.0f32; m * IN]; for i in 0..m { blob(cells[i], &mut o[i * IN..(i + 1) * IN]); } encode_t(&Tensor::from_vec(ctx, &o, &[m, IN])) };

    let fid = { let ft = 200usize; let mut fpos: Vec<(usize, usize)> = (0..ft).map(|i| free_cell(i as u32 * 5 + 9)).collect();
      let mut x = enc_batch(&fpos);
      for s in 0..8 { let mut oh = vec![0.0f32; ft * NA]; for i in 0..ft { let a = (u(i as u32, 200 + s) * NA as f32) as usize % NA; oh[i * NA + a] = 1.0; fpos[i] = next_pos(fpos[i], a); } x = predict_t(&x, &Tensor::from_vec(ctx, &oh, &[ft, NA])); }
      let tru = enc_batch(&fpos); let (xv, tv) = (x.to_vec().await, tru.to_vec().await);
      let mut c = 0.0f32; for i in 0..ft { let (mut dd, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32); for j in 0..d { let a = xv[i * d + j]; let b = tv[i * d + j]; dd += a * b; na += a * a; nb += b * b; } c += dd / (na.sqrt() * nb.sqrt() + 1e-9); } c / ft as f32 };

    let mut vp = vec![
        Tensor::from_vec(ctx, &randn(d * hv, 10, 1.0 / (d as f32).sqrt()), &[d, hv]),
        Tensor::from_vec(ctx, &randn(d * hv, 11, 1.0 / (d as f32).sqrt()), &[d, hv]), Tensor::zeros(ctx, &[hv]),
        Tensor::from_vec(ctx, &randn(hv, 12, 1.0 / (hv as f32).sqrt()), &[hv, 1]), Tensor::zeros(ctx, &[1]),
    ];
    let mut vadam = Adam::new(&vp, 0.003);
    let val_v = |xv: &Var, gv: &Var, p: &[Var]| -> Var { let h = xv.matmul(&p[0]).add(&gv.matmul(&p[1])).add(&p[2]).relu(); h.matmul(&p[3]).add(&p[4]) };
    let val_t = |x: &Tensor, g: &Tensor, p: &[Tensor]| -> Tensor { let h = x.matmul(&p[0]).add(&g.matmul(&p[1])).add(&p[2]).relu(); h.matmul(&p[3]).add(&p[4]) };
    let gamma_t = Tensor::from_vec(ctx, &[0.97f32], &[1]); let bb = 1024usize;
    let mut vp_t: Vec<Tensor> = Vec::new();
    for t in &vp { vp_t.push(Tensor::from_vec(ctx, &t.to_vec().await, &t.shape)); }
    for it in 0..fvi_iters {
        let mut cx: Vec<(usize, usize)> = Vec::with_capacity(bb);
        let mut cg: Vec<(usize, usize)> = Vec::with_capacity(bb);
        for i in 0..bb { let s = it as u32 * 1103 + i as u32; cx.push(free_cell(s)); cg.push(free_cell(s ^ 0x9e37_79b9)); }
        let (x, g) = (enc_batch(&cx), enc_batch(&cg));
        let mut target: Option<Tensor> = None;
        for a in 0..NA {
            let ncells: Vec<(usize, usize)> = cx.iter().map(|&c| next_pos(c, a)).collect();
            let reached: Vec<f32> = ncells.iter().zip(&cg).map(|(&nc, &gc)| if nc == gc { 1.0 } else { 0.0 }).collect();
            let xa = enc_batch(&ncells);
            let vn = val_t(&xa, &g, &vp_t);
            let rt = Tensor::from_vec(ctx, &reached, &[bb, 1]);
            let ta = rt.add(&one.sub(&rt).mul(&vn).mul(&gamma_t));
            target = Some(match target { None => ta, Some(t) => t.maximum(&ta) });
        }
        let target = target.unwrap();
        let p: Vec<Var> = vp.iter().map(|t| Var::leaf(t.clone())).collect();
        let v = val_v(&Var::leaf(x), &Var::leaf(g), &p);
        let diff = v.sub(&Var::leaf(target)); let loss = diff.mul(&diff).mean_all();
        loss.backward();
        let grads: Vec<Tensor> = p.iter().map(|vv| vv.grad().unwrap()).collect(); vadam.step(&mut vp, &grads);
        if it % 250 == 249 { for i in 0..vp.len() { vp_t[i] = Tensor::from_vec(ctx, &vp[i].to_vec().await, &vp[i].shape); } }
    }

    let (t, maxsteps) = (150usize, 90usize);
    let mut pos: Vec<(usize, usize)> = (0..t).map(|i| {
        let left = u(i as u32, 21) < 0.5;
        if left { ((u(i as u32, 22) * 4.0) as usize % 4, (u(i as u32, 23) * G as f32) as usize % G) }
        else { (10 + (u(i as u32, 24) * 3.0) as usize % 3, (u(i as u32, 25) * G as f32) as usize % G) }
    }).collect();
    let goal: Vec<(usize, usize)> = (0..t).map(|i| {
        let left = pos[i].0 < 5;
        if left { (10 + (u(i as u32, 31) * 3.0) as usize % 3, (u(i as u32, 32) * G as f32) as usize % G) }
        else { ((u(i as u32, 33) * 4.0) as usize % 4, (u(i as u32, 34) * G as f32) as usize % G) }
    }).collect();
    let xg = enc_batch(&goal);
    let onehot = |a: usize| -> Tensor { let mut o = vec![0.0f32; t * NA]; for i in 0..t { o[i * NA + a] = 1.0; } Tensor::from_vec(ctx, &o, &[t, NA]) };
    let mut reached = vec![false; t];
    for _ in 0..maxsteps {
        if reached.iter().all(|&r| r) { break; }
        let x = enc_batch(&pos);
        let mut score: Vec<Vec<f32>> = Vec::with_capacity(NA);
        for a in 0..NA {
            let xa = predict_t(&x, &onehot(a));
            let mut m = val_t(&xa, &xg, &vp);
            for a2 in 0..NA { let x2 = predict_t(&xa, &onehot(a2)); m = m.maximum(&val_t(&x2, &xg, &vp));
                for a3 in 0..NA { m = m.maximum(&val_t(&predict_t(&x2, &onehot(a3)), &xg, &vp)); } }
            score.push(m.to_vec().await);
        }
        for i in 0..t { if reached[i] { continue; } let (mut ba, mut best) = (0usize, score[0][i]); for a in 1..NA { if score[a][i] > best { best = score[a][i]; ba = a; } } pos[i] = next_pos(pos[i], ba); if pos[i] == goal[i] { reached[i] = true; } }
    }
    let reach = reached.iter().filter(|&&r| r).count() as f32 / t as f32;
    (fid, reach)
}
