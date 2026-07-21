//! EFA scaling curve, on the fabric — does size lift fidelity + acting, at real scale on the GPU?
//!
//! The nano (JS) scaling study hinted that world-model multi-step fidelity rises with width. This runs
//! the real EFA world model (Ferric autograd + Adam, batched on the GPU) across several sizes and
//! measures, per size: 4- and 8-step rollout fidelity and energy-descent planning reach. A rising
//! curve turns the nano hint into a real trend on the same cross-fabric stack that runs cloud/browser.
//!
//! Run: `cargo run -p ferric-tensor --example efa_scaling --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

const G: usize = 12;
const IN: usize = G * G;
const NA: usize = 4;
const MOVES: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * scale }).collect()
}
fn clampc(v: i32) -> usize { v.max(0).min(G as i32 - 1) as usize }
fn next_pos(p: (usize, usize), a: usize) -> (usize, usize) { (clampc(p.0 as i32 + MOVES[a].0), clampc(p.1 as i32 + MOVES[a].1)) }
fn blob(p: (usize, usize), out: &mut [f32]) {
    for y in 0..G { for x in 0..G { let dx = x as f32 - p.0 as f32; let dy = y as f32 - p.1 as f32; out[y * G + x] = (-(dx * dx + dy * dy) / (2.0 * 1.1 * 1.1)).exp(); } }
}
fn gen_batch(n: usize, seed: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (mut o, mut oh, mut no) = (vec![0.0f32; n * IN], vec![0.0f32; n * NA], vec![0.0f32; n * IN]);
    for i in 0..n {
        let p = ((u(i as u32, seed) * G as f32) as usize % G, (u(i as u32, seed ^ 77) * G as f32) as usize % G);
        let a = (u(i as u32, seed ^ 131) * NA as f32) as usize % NA;
        let pn = next_pos(p, a);
        blob(p, &mut o[i * IN..(i + 1) * IN]); blob(pn, &mut no[i * IN..(i + 1) * IN]); oh[i * NA + a] = 1.0;
    }
    (o, oh, no)
}

async fn rollout_cos(ctx: &Arc<ferric_core::Context>, d: usize, k: usize, encode_t: &dyn Fn(&Tensor) -> Tensor, predict_t: &dyn Fn(&Tensor, &Tensor) -> Tensor) -> f32 {
    let t = 256usize;
    let mut pos: Vec<(usize, usize)> = (0..t).map(|i| ((u(i as u32, 5) * G as f32) as usize % G, (u(i as u32, 6) * G as f32) as usize % G)).collect();
    let mut ob = vec![0.0f32; t * IN]; for i in 0..t { blob(pos[i], &mut ob[i * IN..(i + 1) * IN]); }
    let mut x = encode_t(&Tensor::from_vec(ctx, &ob, &[t, IN]));
    for step in 0..k {
        let mut oh = vec![0.0f32; t * NA];
        for i in 0..t { let a = (u(i as u32, 100 + step as u32) * NA as f32) as usize % NA; oh[i * NA + a] = 1.0; pos[i] = next_pos(pos[i], a); }
        x = predict_t(&x, &Tensor::from_vec(ctx, &oh, &[t, NA]));
    }
    let mut tob = vec![0.0f32; t * IN]; for i in 0..t { blob(pos[i], &mut tob[i * IN..(i + 1) * IN]); }
    let tru = encode_t(&Tensor::from_vec(ctx, &tob, &[t, IN]));
    let (xv, tv) = (x.to_vec().await, tru.to_vec().await);
    let mut c = 0.0f64;
    for i in 0..t { let (mut dd, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
        for j in 0..d { let a = xv[i * d + j]; let b = tv[i * d + j]; dd += a * b; na += a * a; nb += b * b; }
        c += (dd / (na.sqrt() * nb.sqrt() + 1e-9)) as f64;
    }
    (c / t as f64) as f32
}
async fn plan_reach(ctx: &Arc<ferric_core::Context>, encode_t: &dyn Fn(&Tensor) -> Tensor, predict_t: &dyn Fn(&Tensor, &Tensor) -> Tensor) -> f32 {
    let (t, maxsteps) = (150usize, 50usize);
    let mut pos: Vec<(usize, usize)> = (0..t).map(|i| ((u(i as u32, 11) * G as f32) as usize % G, (u(i as u32, 12) * G as f32) as usize % G)).collect();
    let goal: Vec<(usize, usize)> = (0..t).map(|i| ((u(i as u32, 13) * G as f32) as usize % G, (u(i as u32, 14) * G as f32) as usize % G)).collect();
    let mut gob = vec![0.0f32; t * IN]; for i in 0..t { blob(goal[i], &mut gob[i * IN..(i + 1) * IN]); }
    let xg = encode_t(&Tensor::from_vec(ctx, &gob, &[t, IN]));
    let onehot = |a: usize| -> Tensor { let mut o = vec![0.0f32; t * NA]; for i in 0..t { o[i * NA + a] = 1.0; } Tensor::from_vec(ctx, &o, &[t, NA]) };
    let sqd = |xa: &Tensor| -> Tensor { let dd = xa.sub(&xg); dd.mul(&dd).sum(&[1], true) };
    let mut reached = vec![false; t];
    for _ in 0..maxsteps {
        if reached.iter().all(|&r| r) { break; }
        let mut ob = vec![0.0f32; t * IN]; for i in 0..t { blob(pos[i], &mut ob[i * IN..(i + 1) * IN]); }
        let x = encode_t(&Tensor::from_vec(ctx, &ob, &[t, IN]));
        let mut en: Vec<Vec<f32>> = Vec::with_capacity(NA);
        for a in 0..NA {
            let xa = predict_t(&x, &onehot(a));
            let mut e = sqd(&xa).to_vec().await;
            for a2 in 0..NA { let e2 = sqd(&predict_t(&xa, &onehot(a2))).to_vec().await; for i in 0..t { if e2[i] < e[i] { e[i] = e2[i]; } } }
            en.push(e);
        }
        for i in 0..t { if reached[i] { continue; } let (mut ba, mut best) = (0usize, en[0][i]);
            for a in 1..NA { if en[a][i] < best { best = en[a][i]; ba = a; } }
            pos[i] = next_pos(pos[i], ba); if pos[i] == goal[i] { reached[i] = true; } }
    }
    reached.iter().filter(|&&r| r).count() as f32 / t as f32
}

async fn run_size(ctx: &Arc<ferric_core::Context>, d: usize, hid: usize, steps: usize) -> (usize, f32, f32, f32) {
    let n = 512usize;
    let mut params = vec![
        Tensor::from_vec(ctx, &randn(IN * d, 1, 1.0 / (IN as f32).sqrt()), &[IN, d]),
        Tensor::zeros(ctx, &[d]),
        Tensor::from_vec(ctx, &randn(d * hid, 2, 1.0 / (d as f32).sqrt()), &[d, hid]),
        Tensor::from_vec(ctx, &randn(NA * hid, 3, 1.0 / (NA as f32).sqrt()), &[NA, hid]),
        Tensor::zeros(ctx, &[hid]),
        Tensor::from_vec(ctx, &randn(hid * d, 4, 1.0 / (hid as f32).sqrt()), &[hid, d]),
        Tensor::zeros(ctx, &[d]),
    ];
    let mut adam = Adam::new(&params, 0.002);
    let eps = Tensor::from_vec(ctx, &[1e-6], &[1]);
    let epsv = Tensor::from_vec(ctx, &[1e-4], &[1]);
    let one = Tensor::from_vec(ctx, &[1.0], &[1]);
    let lamv = Tensor::from_vec(ctx, &[0.1], &[1]);
    let encode = |o: &Var, p: &[Var], e: &Var| -> Var { let h = o.matmul(&p[0]).add(&p[1]).relu(); let ss = h.mul(&h).sum(&[1]); h.div(&ss.add(e).sqrt()) };
    for step in 0..steps {
        let (ob, oh, no) = gen_batch(n, step as u32 + 1);
        let p: Vec<Var> = params.iter().map(|t| Var::leaf(t.clone())).collect();
        let ev = Var::leaf(eps.clone());
        let ov = Var::leaf(Tensor::from_vec(ctx, &ob, &[n, IN]));
        let nv = Var::leaf(Tensor::from_vec(ctx, &no, &[n, IN]));
        let ohv = Var::leaf(Tensor::from_vec(ctx, &oh, &[n, NA]));
        let x = encode(&ov, &p, &ev);
        let target = encode(&nv, &p, &ev).detach();
        let hp = x.matmul(&p[2]).add(&ohv.matmul(&p[3])).add(&p[4]).relu();
        let o = hp.matmul(&p[5]).add(&p[6]);
        let diff = o.sub(&target);
        let mse = diff.mul(&diff).mean_all();
        let xc = x.sub(&x.mean(&[0]));
        let std = xc.mul(&xc).mean(&[0]).add(&Var::leaf(epsv.clone())).sqrt();
        let vloss = Var::leaf(one.clone()).sub(&std).relu().mean_all();
        let loss = mse.add(&vloss.mul(&Var::leaf(lamv.clone())));
        loss.backward();
        let grads: Vec<Tensor> = p.iter().map(|v| v.grad().unwrap()).collect();
        adam.step(&mut params, &grads);
    }
    let eps2 = eps.clone();
    let sq_norm = move |t: &Tensor| -> Tensor { let ss = t.mul(t).sum(&[1], true); t.div(&ss.add(&eps2).sqrt()) };
    let encode_t = |obs: &Tensor| -> Tensor { sq_norm(&obs.matmul(&params[0]).add(&params[1]).relu()) };
    let predict_t = |x: &Tensor, oh: &Tensor| -> Tensor { let hp = x.matmul(&params[2]).add(&oh.matmul(&params[3])).add(&params[4]).relu(); sq_norm(&hp.matmul(&params[5]).add(&params[6]).relu()) };
    let f4 = rollout_cos(ctx, d, 4, &encode_t, &predict_t).await;
    let f8 = rollout_cos(ctx, d, 8, &encode_t, &predict_t).await;
    let reach = plan_reach(ctx, &encode_t, &predict_t).await;
    let pcount = IN * d + d + d * hid + NA * hid + hid + hid * d + d;
    (pcount, f4, f8, reach)
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let sizes = [(32usize, 128usize), (48, 192), (64, 256), (96, 384)];
    println!("  EFA scaling on Ferric (GPU) — G={G}, batch=512, light VICReg, 1200 steps/size\n");
    println!("   {:>4} {:>5} {:>9} {:>7} {:>7} {:>8}", "D", "HID", "params", "fid4", "fid8", "reach");
    for (d, hid) in sizes {
        let (p, f4, f8, reach) = run_size(&ctx, d, hid, 1200).await;
        println!("   {:>4} {:>5} {:>9} {:>7.3} {:>7.3} {:>7.0}%", d, hid, p, f4, f8, reach * 100.0);
    }
    println!("\n  ✅ EFA scaling curve on the fabric — fidelity + planning reach vs model width.");
}
