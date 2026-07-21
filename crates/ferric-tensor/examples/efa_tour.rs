//! EFA — MULTI-ITEM episodic memory in the loop: a landmark TOUR that stresses the Z write/read rules.
//!
//! The POMDP composition run used one goal (M=1, recall exact) so it did NOT stress the memory upgrades.
//! Here the agent is shown K landmarks at t=0 (each with a distinct index-cue), all HIDDEN after, and must
//! visit them IN ORDER. Z holds M=K associations (cue_j → landmark_j); each step it recalls the current
//! waypoint from Z and plans toward it. Now recall quality gates task success, so the efa_memory finding
//! becomes a TASK metric: additive Z interferes across the K landmarks (corrupted recall → wrong cell),
//! delta-rule and Hopfield readout should complete more of the tour. We sweep K and compare
//! additive / delta / Hopfield / blind on fraction-of-tour-completed. Open grid (no wall) so navigation is
//! easy and MEMORY QUALITY, not planning, is what's being measured.
//!
//! Run: `cargo run -p ferric-tensor --example efa_tour --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

const G: usize = 13;
const IN: usize = G * G;
const NA: usize = 4;
const MOVES: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];
const D: usize = 64;

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, scale: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * scale }).collect() }
fn clampc(v: i32) -> usize { v.max(0).min(G as i32 - 1) as usize }
fn next_pos(p: (usize, usize), a: usize) -> (usize, usize) { (clampc(p.0 as i32 + MOVES[a].0), clampc(p.1 as i32 + MOVES[a].1)) } // open grid
fn cell(seed: u32) -> (usize, usize) { ((u(seed, 1) * G as f32) as usize % G, (u(seed, 2) * G as f32) as usize % G) }
fn blob(p: (usize, usize), out: &mut [f32]) { for y in 0..G { for x in 0..G { let dx = x as f32 - p.0 as f32; let dy = y as f32 - p.1 as f32; out[y * G + x] = (-(dx * dx + dy * dy) / (2.0 * 1.1 * 1.1)).exp(); } } }
fn unit(mut v: Vec<f32>) -> Vec<f32> { let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-8); for x in &mut v { *x /= n; } v }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let (hid, hv, n) = (256usize, 512usize, 512usize);
    println!("  EFA landmark TOUR — K goals shown at t=0 then hidden, visit in order; recall from Z. D={D}");

    // ---- world model + FVI value on the OPEN grid (goal = any cell) ----
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
    for it in 0..8000 {
        let mut cx: Vec<(usize, usize)> = Vec::with_capacity(bb);
        let mut cg: Vec<(usize, usize)> = Vec::with_capacity(bb);
        for i in 0..bb { let s = it as u32 * 1103 + i as u32; cx.push(cell(s)); cg.push(cell(s ^ 0x9e37_79b9)); }
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

    // fixed index cues (shared across agents): one unit vector per landmark index
    let cues: Vec<Vec<f32>> = (0..16).map(|j| unit(randn(D, 500 + j as u32, 1.0))).collect();
    let vt = |x: &Tensor, g: &Tensor| -> Tensor { val_t(x, g, &vp) };

    println!("\n  fraction of the K-landmark tour completed (open grid, recall the current waypoint from Z):");
    println!("  [greedy depth-3 planner]");
    println!("     {:>4}   {:>9}   {:>9}   {:>9}   {:>7}", "K", "additive", "delta", "hopfield", "blind");
    for &k in &[4usize, 8, 12, 16] {
        let add = tour(&ctx, "additive", "greedy", 0, k, &cues, &predict_t, &vt, &enc_batch).await;
        let del = tour(&ctx, "delta", "greedy", 0, k, &cues, &predict_t, &vt, &enc_batch).await;
        let hop = tour(&ctx, "hopfield", "greedy", 0, k, &cues, &predict_t, &vt, &enc_batch).await;
        let bli = tour(&ctx, "blind", "greedy", 0, k, &cues, &predict_t, &vt, &enc_batch).await;
        println!("     {:>4}   {:>8.0}%   {:>8.0}%   {:>8.0}%   {:>6.0}%", k, add * 100.0, del * 100.0, hop * 100.0, bli * 100.0);
    }
    println!("\n  [MPPI K=192 planner — better per-landmark reach precision → does the recall-quality gap widen?]");
    println!("     {:>4}   {:>9}   {:>9}   {:>9}", "K", "additive", "delta", "hopfield");
    for &k in &[4usize, 8, 12, 16] {
        let add = tour(&ctx, "additive", "mppi", 192, k, &cues, &predict_t, &vt, &enc_batch).await;
        let del = tour(&ctx, "delta", "mppi", 192, k, &cues, &predict_t, &vt, &enc_batch).await;
        let hop = tour(&ctx, "hopfield", "mppi", 192, k, &cues, &predict_t, &vt, &enc_batch).await;
        println!("     {:>4}   {:>8.0}%   {:>8.0}%   {:>8.0}%", k, add * 100.0, del * 100.0, hop * 100.0);
    }
    println!("\n  delta/hopfield ≥ additive as K grows → the gradient-free memory upgrades pay off IN THE LOOP");
    println!("  (additive Z interferes across the K stored landmarks → corrupted recall → wrong cell).");
}

// run the K-landmark tour for `t` agents; return mean fraction of landmarks reached in order.
async fn tour(ctx: &Arc<ferric_core::Context>, cond: &str, planner: &str, mppi_k: usize, k: usize, cues: &[Vec<f32>], predict_t: &dyn Fn(&Tensor, &Tensor) -> Tensor, val_t: &dyn Fn(&Tensor, &Tensor) -> Tensor, enc_batch: &dyn Fn(&[(usize, usize)]) -> Tensor) -> f32 {
    let t = 100usize;
    let maxsteps = 14 * k + 20;
    // start + K distinct landmark cells per agent
    let mut pos: Vec<(usize, usize)> = (0..t).map(|i| cell(9000 + i as u32)).collect();
    let mut land: Vec<Vec<(usize, usize)>> = vec![Vec::with_capacity(k); t];
    for i in 0..t { let mut s = 3000 + (i * 97) as u32; while land[i].len() < k { let c = cell(s); if !land[i].contains(&c) && c != pos[i] { land[i].push(c); } s = s.wrapping_add(131); } }
    // landmark latents [t*k, d] → cpu
    let mut lm_cells: Vec<(usize, usize)> = Vec::with_capacity(t * k);
    for i in 0..t { for j in 0..k { lm_cells.push(land[i][j]); } }
    let lm_lat = enc_batch(&lm_cells).to_vec().await; // [t*k*d]

    // build per-agent memory once (t=0 write, surprise-gated = all landmarks novel at t=0)
    let mut z_mem: Vec<f32> = Vec::new(); // additive/delta: [t*d*d]
    if cond == "additive" || cond == "delta" {
        z_mem = vec![0.0f32; t * D * D];
        for i in 0..t {
            let zb = i * D * D;
            if cond == "additive" {
                for j in 0..k { for a in 0..D { let va = lm_lat[(i * k + j) * D + a]; if va == 0.0 { continue; } for b in 0..D { z_mem[zb + a * D + b] += va * cues[j][b]; } } }
            } else { // delta rule, 4 consolidation passes, β=1
                for _ in 0..4 { for j in 0..k {
                    let mut pred = [0.0f32; D];
                    for a in 0..D { let mut s = 0.0f32; for b in 0..D { s += z_mem[zb + a * D + b] * cues[j][b]; } pred[a] = s; }
                    for a in 0..D { let err = lm_lat[(i * k + j) * D + a] - pred[a]; if err == 0.0 { continue; } for b in 0..D { z_mem[zb + a * D + b] += err * cues[j][b]; } }
                } }
            }
        }
    }

    let recall = |i: usize, wp: usize, z: &[f32]| -> Vec<f32> {
        match cond {
            "blind" => vec![0.0f32; D],
            "hopfield" => { // v̂ = Σ_j softmax(β cue_wp·cue_j) landmark_j
                let beta = 12.0f32;
                let mut logits: Vec<f32> = (0..k).map(|j| beta * cues[wp].iter().zip(&cues[j]).map(|(a, b)| a * b).sum::<f32>()).collect();
                let lmax = logits.iter().cloned().fold(f32::MIN, f32::max);
                let mut zsum = 0.0f32; for l in &mut logits { *l = (*l - lmax).exp(); zsum += *l; }
                let mut vh = vec![0.0f32; D];
                for j in 0..k { let w = logits[j] / zsum; for a in 0..D { vh[a] += w * lm_lat[(i * k + j) * D + a]; } }
                vh
            }
            _ => { let zb = i * D * D; (0..D).map(|a| { let mut s = 0.0f32; for b in 0..D { s += z[zb + a * D + b] * cues[wp][b]; } s }).collect() }
        }
    };

    let onehot = |a: usize| -> Tensor { let mut o = vec![0.0f32; t * NA]; for i in 0..t { o[i * NA + a] = 1.0; } Tensor::from_vec(ctx, &o, &[t, NA]) };
    let mut prog = vec![0usize; t]; // current waypoint index
    for stp in 0..maxsteps {
        if prog.iter().all(|&p| p >= k) { break; }
        // recall current goal for every agent → xgv [t,d]
        let mut xgv = vec![0.0f32; t * D];
        for i in 0..t { let wp = prog[i].min(k - 1); let r = recall(i, wp, &z_mem); for a in 0..D { xgv[i * D + a] = r[a]; } }
        // pick an action per agent toward its recalled goal
        let mut chosen = vec![0usize; t];
        if planner == "mppi" {
            let m = t * mppi_k;
            let mut pos_rep: Vec<(usize, usize)> = Vec::with_capacity(m);
            for i in 0..t { for _ in 0..mppi_k { pos_rep.push(pos[i]); } }
            let hh = 6usize;
            let mut acts: Vec<Vec<usize>> = vec![vec![0usize; m]; hh];
            for i in 0..t { for kk in 0..mppi_k { for h in 0..hh {
                let seed = h32((stp as u32).wrapping_mul(2_654_435_761) ^ (i as u32).wrapping_mul(40_503) ^ (kk as u32).wrapping_mul(2_246_822_519) ^ (h as u32).wrapping_mul(3_266_489_917));
                acts[h][i * mppi_k + kk] = (seed as usize) % NA;
            } } }
            let first = acts[0].clone();
            let mut xh = enc_batch(&pos_rep);
            for h in 0..hh { let mut oh = vec![0.0f32; m * NA]; for r in 0..m { oh[r * NA + acts[h][r]] = 1.0; } xh = predict_t(&xh, &Tensor::from_vec(ctx, &oh, &[m, NA])); }
            let mut xg_rep = vec![0.0f32; m * D]; for i in 0..t { for kk in 0..mppi_k { for a in 0..D { xg_rep[(i * mppi_k + kk) * D + a] = xgv[i * D + a]; } } }
            let vend = val_t(&xh, &Tensor::from_vec(ctx, &xg_rep, &[m, D])).to_vec().await;
            for i in 0..t { let base = i * mppi_k;
                let mut vmax = f32::MIN; for kk in 0..mppi_k { if vend[base + kk] > vmax { vmax = vend[base + kk]; } }
                let mut wsum = [0.0f32; NA]; for kk in 0..mppi_k { let w = ((vend[base + kk] - vmax) / 0.5).exp(); wsum[first[base + kk]] += w; }
                let (mut ba, mut best) = (0usize, wsum[0]); for a in 1..NA { if wsum[a] > best { best = wsum[a]; ba = a; } }
                chosen[i] = ba;
            }
        } else {
            let xg = Tensor::from_vec(ctx, &xgv, &[t, D]);
            let x = enc_batch(&pos);
            let mut score: Vec<Vec<f32>> = Vec::with_capacity(NA);
            for a in 0..NA { let xa = predict_t(&x, &onehot(a));
                let mut mm = val_t(&xa, &xg);
                for a2 in 0..NA { let x2 = predict_t(&xa, &onehot(a2)); mm = mm.maximum(&val_t(&x2, &xg));
                    for a3 in 0..NA { mm = mm.maximum(&val_t(&predict_t(&x2, &onehot(a3)), &xg)); } }
                score.push(mm.to_vec().await); }
            for i in 0..t { let (mut ba, mut best) = (0usize, score[0][i]); for a in 1..NA { if score[a][i] > best { best = score[a][i]; ba = a; } } chosen[i] = ba; }
        }
        for i in 0..t { if prog[i] >= k { continue; } pos[i] = next_pos(pos[i], chosen[i]); if pos[i] == land[i][prog[i]] { prog[i] += 1; } }
    }
    prog.iter().map(|&p| p as f32 / k as f32).sum::<f32>() / t as f32
}
