//! EFA — BENCHMARK vs the EXACT optimal (not a fake published-number comparison).
//!
//! A literal head-to-head vs Dreamer/TD-MPC2 published numbers isn't honest — those are on DeepMind Control
//! (continuous, pixels), a different task. The rigorous benchmark our stack CAN run: compare EFA's MPPI
//! planner against the TRUE optimal, computed exactly by BFS on the maze graph (the 100% ceiling — a
//! stronger baseline than any learned method), plus a naive euclidean-greedy floor. We measure both REACH
//! and PATH-LENGTH OPTIMALITY (EFA steps ÷ shortest-path steps), and sweep the MPPI search budget K to show
//! how test-time compute closes the gap to optimal.
//!
//! Run: `cargo run -p ferric-tensor --example efa_benchmark --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::collections::VecDeque;
use std::sync::Arc;

const G: usize = 13;
const IN: usize = G * G;
const NA: usize = 4;
const MOVES: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];
const D: usize = 64;
fn is_wall(x: usize, y: usize) -> bool { x == 6 && y < 9 }

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, scale: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * scale }).collect() }
fn clampc(v: i32) -> usize { v.max(0).min(G as i32 - 1) as usize }
fn next_pos(p: (usize, usize), a: usize) -> (usize, usize) { let nx = clampc(p.0 as i32 + MOVES[a].0); let ny = clampc(p.1 as i32 + MOVES[a].1); if is_wall(nx, ny) { p } else { (nx, ny) } }
fn free_cell(seed: u32) -> (usize, usize) { let mut s = seed; loop { let x = (u(s, 1) * G as f32) as usize % G; let y = (u(s, 2) * G as f32) as usize % G; if !is_wall(x, y) { return (x, y); } s = s.wrapping_add(101); } }
fn blob(p: (usize, usize), out: &mut [f32]) { for y in 0..G { for x in 0..G { let dx = x as f32 - p.0 as f32; let dy = y as f32 - p.1 as f32; out[y * G + x] = (-(dx * dx + dy * dy) / (2.0 * 1.1 * 1.1)).exp(); } } }

// exact shortest-path distances from `goal` to every free cell (BFS on the maze graph)
fn bfs(goal: (usize, usize)) -> Vec<i32> {
    let mut dist = vec![-1i32; IN];
    let mut q = VecDeque::new();
    dist[goal.1 * G + goal.0] = 0; q.push_back(goal);
    while let Some(c) = q.pop_front() {
        let d0 = dist[c.1 * G + c.0];
        for a in 0..NA { let nc = next_pos(c, a); let idx = nc.1 * G + nc.0; if dist[idx] < 0 { dist[idx] = d0 + 1; q.push_back(nc); } }
    }
    dist
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let (hid, hv, n) = (256usize, 512usize, 512usize);
    println!("  EFA maze BENCHMARK vs EXACT optimal (BFS ceiling) — reach + path-length optimality. D={D}");

    // ---- world model + FVI value (identical to efa_maze) ----
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
        for i in 0..n { let p = free_cell(step as u32 * 131 + i as u32); let a = (u(i as u32, step as u32 ^ 7) * NA as f32) as usize % NA; let pn = next_pos(p, a); blob(p, &mut ob[i * IN..(i + 1) * IN]); blob(pn, &mut no[i * IN..(i + 1) * IN]); oh[i * NA + a] = 1.0; }
        let p: Vec<Var> = wp.iter().map(|t| Var::leaf(t.clone())).collect();
        let ev = Var::leaf(eps.clone());
        let ov = Var::leaf(Tensor::from_vec(&ctx, &ob, &[n, IN])); let nv = Var::leaf(Tensor::from_vec(&ctx, &no, &[n, IN])); let ohv = Var::leaf(Tensor::from_vec(&ctx, &oh, &[n, NA]));
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
    let enc_batch = |cells: &[(usize, usize)]| -> Tensor { let m = cells.len(); let mut o = vec![0.0f32; m * IN]; for i in 0..m { blob(cells[i], &mut o[i * IN..(i + 1) * IN]); } encode_t(&Tensor::from_vec(&ctx, &o, &[m, IN])) };

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
    for it in 0..10000 {
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
            let rt = Tensor::from_vec(&ctx, &reached, &[bb, 1]);
            let ta = rt.add(&one.sub(&rt).mul(&vn).mul(&gamma_t));
            target = Some(match target { None => ta, Some(t) => t.maximum(&ta) });
        }
        let target = target.unwrap();
        let p: Vec<Var> = vp.iter().map(|t| Var::leaf(t.clone())).collect();
        let v = val_v(&Var::leaf(x), &Var::leaf(g), &p);
        let diff = v.sub(&Var::leaf(target)); let loss = diff.mul(&diff).mean_all();
        loss.backward();
        let grads: Vec<Tensor> = p.iter().map(|vv| vv.grad().unwrap()).collect(); vadam.step(&mut vp, &grads);
        if it % 250 == 249 { vp_t = vp.iter().map(|t| t.clone()).collect(); }
    }
    let vt = |x: &Tensor, g: &Tensor| -> Tensor { val_t(x, g, &vp) };

    // ---- benchmark: fixed cross-wall task; measure reach + path/optimal for each method ----
    let (pos0, goal) = make_task(150);
    let dists: Vec<Vec<i32>> = goal.iter().map(|&g| bfs(g)).collect();
    let opt_len: Vec<i32> = (0..pos0.len()).map(|i| dists[i][pos0[i].1 * G + pos0[i].0]).collect(); // shortest steps

    println!("\n  {:<24}  {:>6}  {:>12}  {:>14}", "method", "reach", "opt path", "EFA/opt ratio");
    // OPTIMAL (BFS gradient) — the 100% ceiling
    let (opt_reach, opt_avg, _) = optimal_reach(&pos0, &goal, &dists, &opt_len);
    println!("  {:<24}  {:>5.0}%  {:>11.1}   {:>13}", "optimal (BFS ceiling)", opt_reach * 100.0, opt_avg, "1.00×");
    // NAIVE euclidean-greedy (ignores wall) — the floor
    let (nav_reach, _, nav_ratio) = naive_reach(&pos0, &goal, &dists, &opt_len);
    println!("  {:<24}  {:>5.0}%  {:>11}   {:>12.2}×", "naive euclidean-greedy", nav_reach * 100.0, "—", nav_ratio);
    // EFA MPPI at increasing K
    for &k in &[64usize, 256, 512] {
        let (r, _, ratio) = efa_mppi_reach(&ctx, k, 6, 0.5, &pos0, &goal, &dists, &opt_len, &predict_t, &vt, &enc_batch).await;
        println!("  {:<24}  {:>5.0}%  {:>11}   {:>12.2}×", format!("EFA MPPI K={k}"), r * 100.0, "—", ratio);
    }
    println!("\n  reach vs the true 100% ceiling; EFA/opt ratio = how much longer than the shortest path (1.00 = optimal).");
    println!("  more MPPI compute → reach up, path→shorter: test-time search closing the gap to exact optimal.");
}

fn make_task(t: usize) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
    let pos: Vec<(usize, usize)> = (0..t).map(|i| { let left = u(i as u32, 21) < 0.5;
        if left { ((u(i as u32, 22) * 6.0) as usize % 6, (u(i as u32, 23) * G as f32) as usize % G) }
        else { (7 + (u(i as u32, 24) * 6.0) as usize % 6, (u(i as u32, 25) * G as f32) as usize % G) } }).collect();
    let goal: Vec<(usize, usize)> = (0..t).map(|i| { let left = pos[i].0 < 6;
        if left { (7 + (u(i as u32, 31) * 6.0) as usize % 6, (u(i as u32, 32) * G as f32) as usize % G) }
        else { ((u(i as u32, 33) * 6.0) as usize % 6, (u(i as u32, 34) * G as f32) as usize % G) } }).collect();
    (pos, goal)
}

// follow the BFS distance gradient — exact optimal
fn optimal_reach(pos0: &[(usize, usize)], goal: &[(usize, usize)], dists: &[Vec<i32>], opt_len: &[i32]) -> (f32, f32, f32) {
    let t = pos0.len(); let mut reached = 0; let mut sum_len = 0i64; let mut cnt = 0;
    for i in 0..t {
        let mut p = pos0[i]; let mut steps = 0;
        for _ in 0..80 { if p == goal[i] { reached += 1; sum_len += steps as i64; cnt += 1; break; }
            let mut ba = 0; let mut bd = i32::MAX;
            for a in 0..NA { let nc = next_pos(p, a); let dd = dists[i][nc.1 * G + nc.0]; if dd >= 0 && dd < bd { bd = dd; ba = a; } }
            p = next_pos(p, ba); steps += 1; }
    }
    let _ = opt_len; (reached as f32 / t as f32, if cnt > 0 { sum_len as f32 / cnt as f32 } else { 0.0 }, 1.0)
}

// greedy toward the goal by euclidean distance, ignoring the wall (trapped) — the floor
fn naive_reach(pos0: &[(usize, usize)], goal: &[(usize, usize)], _dists: &[Vec<i32>], opt_len: &[i32]) -> (f32, f32, f32) {
    let t = pos0.len(); let mut reached = 0; let (mut rsum, mut rcnt) = (0.0f32, 0);
    for i in 0..t {
        let mut p = pos0[i]; let mut steps = 0;
        for _ in 0..80 { if p == goal[i] { reached += 1; if opt_len[i] > 0 { rsum += steps as f32 / opt_len[i] as f32; rcnt += 1; } break; }
            let mut ba = 0; let mut bd = f32::MAX;
            for a in 0..NA { let nc = next_pos(p, a); let dx = nc.0 as f32 - goal[i].0 as f32; let dy = nc.1 as f32 - goal[i].1 as f32; let dd = dx * dx + dy * dy; if dd < bd { bd = dd; ba = a; } }
            p = next_pos(p, ba); steps += 1; }
    }
    (reached as f32 / t as f32, 0.0, if rcnt > 0 { rsum / rcnt as f32 } else { 0.0 })
}

// EFA MPPI; return (reach, _, mean path/optimal ratio over successes)
async fn efa_mppi_reach(ctx: &Arc<ferric_core::Context>, kk: usize, hh: usize, lambda: f32, pos0: &[(usize, usize)], goal: &[(usize, usize)], _dists: &[Vec<i32>], opt_len: &[i32], predict_t: &dyn Fn(&Tensor, &Tensor) -> Tensor, val_t: &dyn Fn(&Tensor, &Tensor) -> Tensor, enc_batch: &dyn Fn(&[(usize, usize)]) -> Tensor) -> (f32, f32, f32) {
    let t = pos0.len(); let maxsteps = 60usize;
    let mut pos = pos0.to_vec();
    let mut reached = vec![false; t]; let mut steps_at = vec![0usize; t];
    let m = t * kk;
    for stp in 0..maxsteps {
        if reached.iter().all(|&r| r) { break; }
        let mut pos_rep: Vec<(usize, usize)> = Vec::with_capacity(m);
        let mut goal_rep: Vec<(usize, usize)> = Vec::with_capacity(m);
        for i in 0..t { for _ in 0..kk { pos_rep.push(pos[i]); goal_rep.push(goal[i]); } }
        let mut acts: Vec<Vec<usize>> = vec![vec![0usize; m]; hh];
        for i in 0..t { for k in 0..kk { for h in 0..hh {
            let seed = h32((stp as u32).wrapping_mul(2_654_435_761) ^ (i as u32).wrapping_mul(40_503) ^ (k as u32).wrapping_mul(2_246_822_519) ^ (h as u32).wrapping_mul(3_266_489_917));
            acts[h][i * kk + k] = (seed as usize) % NA;
        } } }
        let first: Vec<usize> = acts[0].clone();
        let mut xh = enc_batch(&pos_rep);
        for h in 0..hh { let mut oh = vec![0.0f32; m * NA]; for r in 0..m { oh[r * NA + acts[h][r]] = 1.0; } xh = predict_t(&xh, &Tensor::from_vec(ctx, &oh, &[m, NA])); }
        let xg = enc_batch(&goal_rep);
        let vend = val_t(&xh, &xg).to_vec().await;
        for i in 0..t { if reached[i] { continue; } let base = i * kk;
            let mut vmax = f32::MIN; for k in 0..kk { if vend[base + k] > vmax { vmax = vend[base + k]; } }
            let mut wsum = [0.0f32; NA]; for k in 0..kk { let w = ((vend[base + k] - vmax) / lambda).exp(); wsum[first[base + k]] += w; }
            let (mut ba, mut best) = (0usize, wsum[0]); for a in 1..NA { if wsum[a] > best { best = wsum[a]; ba = a; } }
            pos[i] = next_pos(pos[i], ba); if pos[i] == goal[i] { reached[i] = true; steps_at[i] = stp + 1; }
        }
    }
    let (mut rsum, mut rcnt) = (0.0f32, 0);
    for i in 0..t { if reached[i] && opt_len[i] > 0 { rsum += steps_at[i] as f32 / opt_len[i] as f32; rcnt += 1; } }
    (reached.iter().filter(|&&r| r).count() as f32 / t as f32, 0.0, if rcnt > 0 { rsum / rcnt as f32 } else { 0.0 })
}
