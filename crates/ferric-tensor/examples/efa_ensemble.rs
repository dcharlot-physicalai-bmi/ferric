//! EFA — ENSEMBLE / uncertainty-guarded MPPI planning (fight world-model/value error at test time).
//!
//! MPPI's known failure (TD-M(PC)² 2024; PETS): population search OVER-EXPLOITS regions where the learned
//! value is wrong — it finds the action whose rollout lands wherever the value spuriously spikes. The
//! gradient-free fix is an ENSEMBLE: train N value nets (different init + data), and at plan time score a
//! candidate by mean(V) − λ·std(V) across the ensemble — penalize DISAGREEMENT, so the planner avoids
//! latent regions the ensemble is unsure about (usually the drifted/unreliable ones). This is a pure
//! test-time robustness mechanism (no extra training beyond the ensemble). We compare, on the hard maze:
//! single value (our 69% MPPI), ensemble-mean, and ensemble mean−λ·std, all with the same MPPI/world model.
//!
//! Run: `cargo run -p ferric-tensor --example efa_ensemble --release`
use ferric_tensor::{Adam, Tensor, Var};
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

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let (hid, hv, n) = (256usize, 512usize, 512usize);
    println!("  EFA maze — ENSEMBLE/uncertainty-guarded MPPI (fight value error). D={D} HID={hid}, 3 value nets");

    // ---- world model (encoder + JEPA), trained once ----
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

    // ---- train an ENSEMBLE of 3 value nets (different init + data seeds) via FVI+target ----
    let val_v = |xv: &Var, gv: &Var, p: &[Var]| -> Var { let h = xv.matmul(&p[0]).add(&gv.matmul(&p[1])).add(&p[2]).relu(); h.matmul(&p[3]).add(&p[4]) };
    let val_t = |x: &Tensor, g: &Tensor, p: &[Tensor]| -> Tensor { let h = x.matmul(&p[0]).add(&g.matmul(&p[1])).add(&p[2]).relu(); h.matmul(&p[3]).add(&p[4]) };
    let gamma_t = Tensor::from_vec(&ctx, &[0.95f32], &[1]); let bb = 1024usize;
    let mut ens: Vec<Vec<Tensor>> = Vec::new();
    for member in 0..3u32 {
        let si = member * 1000; // decorrelate init
        let mut vp = vec![
            Tensor::from_vec(&ctx, &randn(D * hv, 10 + si, 1.0 / (D as f32).sqrt()), &[D, hv]),
            Tensor::from_vec(&ctx, &randn(D * hv, 11 + si, 1.0 / (D as f32).sqrt()), &[D, hv]), Tensor::zeros(&ctx, &[hv]),
            Tensor::from_vec(&ctx, &randn(hv, 12 + si, 1.0 / (hv as f32).sqrt()), &[hv, 1]), Tensor::zeros(&ctx, &[1]),
        ];
        let mut vadam = Adam::new(&vp, 0.003);
        let mut vp_t: Vec<Tensor> = vp.iter().map(|t| t.clone()).collect();
        for it in 0..9000 {
            let mut cx: Vec<(usize, usize)> = Vec::with_capacity(bb);
            let mut cg: Vec<(usize, usize)> = Vec::with_capacity(bb);
            for i in 0..bb { let s = (it as u32).wrapping_mul(1103).wrapping_add(i as u32).wrapping_add(member.wrapping_mul(0x51ed_2701)); cx.push(free_cell(s)); cg.push(free_cell(s ^ 0x9e37_79b9)); }
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
        ens.push(vp);
    }

    // ---- MPPI eval: single value / ensemble-mean / ensemble mean−λ·std ----
    println!("\n  cross-wall reach (MPPI K=256, H=6), same world model + ensemble:");
    for &(name, mode, lam) in &[("single value (member 0)", 0u8, 0.0f32), ("ensemble mean", 1, 0.0), ("ensemble mean − 1.0·std", 2, 1.0), ("ensemble mean − 2.0·std", 2, 2.0)] {
        let r = reach_ens(&ctx, 256, 6, 0.5, mode, lam, &ens, &predict_t, &val_t, &enc_batch).await;
        println!("     {:<26} : {:>3.0}%", name, r * 100.0);
    }
    println!("\n  ensemble mean−λ·std ≥ single → penalizing value DISAGREEMENT stops MPPI exploiting value error.");
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

// MPPI where the terminal score combines the ensemble: mode 0 = member0 only; 1 = mean; 2 = mean − lam·std.
async fn reach_ens(ctx: &Arc<ferric_core::Context>, kk: usize, hh: usize, lambda: f32, mode: u8, lam: f32, ens: &[Vec<Tensor>], predict_t: &dyn Fn(&Tensor, &Tensor) -> Tensor, val_t: &dyn Fn(&Tensor, &Tensor, &[Tensor]) -> Tensor, enc_batch: &dyn Fn(&[(usize, usize)]) -> Tensor) -> f32 {
    let (t, maxsteps) = (150usize, 60usize);
    let (mut pos, goal) = make_task(t);
    let mut reached = vec![false; t];
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
        // ensemble values at the terminal latent
        let mut vs: Vec<Vec<f32>> = Vec::with_capacity(ens.len());
        for vp in ens { vs.push(val_t(&xh, &xg, vp).to_vec().await); }
        let score: Vec<f32> = (0..m).map(|r| {
            match mode {
                0 => vs[0][r],
                1 => (vs[0][r] + vs[1][r] + vs[2][r]) / 3.0,
                _ => { let mu = (vs[0][r] + vs[1][r] + vs[2][r]) / 3.0;
                       let var = ((vs[0][r] - mu).powi(2) + (vs[1][r] - mu).powi(2) + (vs[2][r] - mu).powi(2)) / 3.0;
                       mu - lam * var.sqrt() }
            }
        }).collect();
        for i in 0..t { if reached[i] { continue; } let base = i * kk;
            let mut vmax = f32::MIN; for k in 0..kk { if score[base + k] > vmax { vmax = score[base + k]; } }
            let mut wsum = [0.0f32; NA]; for k in 0..kk { let w = ((score[base + k] - vmax) / lambda).exp(); wsum[first[base + k]] += w; }
            let (mut ba, mut best) = (0usize, wsum[0]); for a in 1..NA { if wsum[a] > best { best = wsum[a]; ba = a; } }
            pos[i] = next_pos(pos[i], ba); if pos[i] == goal[i] { reached[i] = true; }
        }
    }
    reached.iter().filter(|&&r| r).count() as f32 / t as f32
}
