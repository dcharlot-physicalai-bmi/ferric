//! EFA world model, trained natively on the fabric (GPU) — phase 1 of scaling EFA on Ferric.
//!
//! The nano program (JS) proved EFA's mechanisms and showed the one thing that scales is world-model
//! multi-step FIDELITY. This trains the EFA world model — a shared sparse-positive unit-normalized
//! encoder + an action-conditioned JEPA predictor — with Ferric's autograd + Adam, batched on the GPU,
//! at a real scale (D=64, HID=256, batch=512; ~5.7× the widest nano predictor and batched, not
//! per-transition). Loss = JEPA next-latent MSE (stop-grad target) + a VICReg variance term
//! (anti-collapse). Success = the loss falls and multi-step rollout fidelity (cosine to the true next
//! latent, K steps out) is high — the metric the scaling study showed rises with size.
//!
//! Run: `cargo run -p ferric-tensor --example efa_world_model --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;

const G: usize = 12;
const IN: usize = G * G;
const NA: usize = 4;
const MOVES: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];

fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
// zero-mean gaussian init, scaled by 1/sqrt(fan_in)
fn randn(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * scale }).collect()
}
fn clampc(v: i32) -> usize { v.max(0).min(G as i32 - 1) as usize }
fn next_pos(p: (usize, usize), a: usize) -> (usize, usize) { (clampc(p.0 as i32 + MOVES[a].0), clampc(p.1 as i32 + MOVES[a].1)) }
fn blob(p: (usize, usize), out: &mut [f32]) {
    for y in 0..G { for x in 0..G { let dx = x as f32 - p.0 as f32; let dy = y as f32 - p.1 as f32; out[y * G + x] = (-(dx * dx + dy * dy) / (2.0 * 1.1 * 1.1)).exp(); } }
}
// a batch of (obs, action-onehot, next_obs)
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

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    let (d, hid, n, steps) = (64usize, 256usize, 512usize, 1500usize);
    println!("  EFA world model on Ferric — D={d} HID={hid} batch={n}, training on the GPU…");

    // params: Wenc[IN,D] benc[D] | W1x[D,HID] W1a[NA,HID] b1[HID] | W2[HID,D] b2[D]
    let mut params = vec![
        Tensor::from_vec(&ctx, &randn(IN * d, 1, 1.0 / (IN as f32).sqrt()), &[IN, d]),
        Tensor::zeros(&ctx, &[d]),
        Tensor::from_vec(&ctx, &randn(d * hid, 2, 1.0 / (d as f32).sqrt()), &[d, hid]),
        Tensor::from_vec(&ctx, &randn(NA * hid, 3, 1.0 / (NA as f32).sqrt()), &[NA, hid]),
        Tensor::zeros(&ctx, &[hid]),
        Tensor::from_vec(&ctx, &randn(hid * d, 4, 1.0 / (hid as f32).sqrt()), &[hid, d]),
        Tensor::zeros(&ctx, &[d]),
    ];
    let mut adam = Adam::new(&params, 0.002);
    let eps = Tensor::from_vec(&ctx, &[1e-6], &[1]);
    let epsv = Tensor::from_vec(&ctx, &[1e-4], &[1]);
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]);
    let lamv = Tensor::from_vec(&ctx, &[0.1], &[1]);  // light VICReg: prevent collapse without over-spreading (smooth metric for planning)

    // encode a Var of obs → unit-normalized sparse-positive latent
    let encode = |o: &Var, p: &[Var], eps: &Var| -> Var {
        let h = o.matmul(&p[0]).add(&p[1]).relu();
        let ss = h.mul(&h).sum(&[1]);
        h.div(&ss.add(eps).sqrt())
    };

    let mut first = 0.0;
    for step in 0..steps {
        let (ob, oh, no) = gen_batch(n, step as u32 + 1);
        let obst = Tensor::from_vec(&ctx, &ob, &[n, IN]);
        let oht = Tensor::from_vec(&ctx, &oh, &[n, NA]);
        let nobst = Tensor::from_vec(&ctx, &no, &[n, IN]);

        let p: Vec<Var> = params.iter().map(|t| Var::leaf(t.clone())).collect();
        let epsver = Var::leaf(eps.clone());
        let ov = Var::leaf(obst); let nv = Var::leaf(nobst); let ohv = Var::leaf(oht);

        let x = encode(&ov, &p, &epsver);
        let target = encode(&nv, &p, &epsver).detach();               // JEPA target: stop-grad
        let hp = x.matmul(&p[2]).add(&ohv.matmul(&p[3])).add(&p[4]).relu();
        let o = hp.matmul(&p[5]).add(&p[6]);                          // linear next-latent prediction
        let diff = o.sub(&target);
        let mse = diff.mul(&diff).mean_all();
        // VICReg variance term on the current latents — anti-collapse
        let xc = x.sub(&x.mean(&[0]));
        let std = xc.mul(&xc).mean(&[0]).add(&Var::leaf(epsv.clone())).sqrt();
        let vloss = Var::leaf(one.clone()).sub(&std).relu().mean_all();
        let loss = mse.add(&vloss.mul(&Var::leaf(lamv.clone())));

        loss.backward();
        let l = loss.value().to_vec().await[0];
        if step == 0 { first = l; }
        let grads: Vec<Tensor> = p.iter().map(|v| v.grad().unwrap()).collect();
        adam.step(&mut params, &grads);
        if step % 300 == 0 || step == steps - 1 { println!("     step {step:>4}  loss {l:.5}"); }
    }

    // ---- evaluate multi-step rollout fidelity (Tensor-only forward, on the GPU) ----
    let sq_norm = |t: &Tensor| -> Tensor { let ss = t.mul(t).sum(&[1], true); t.div(&ss.add(&eps).sqrt()) };
    let encode_t = |obs: &Tensor| -> Tensor { sq_norm(&obs.matmul(&params[0]).add(&params[1]).relu()) };
    let predict_t = |x: &Tensor, oh: &Tensor| -> Tensor {
        let hp = x.matmul(&params[2]).add(&oh.matmul(&params[3])).add(&params[4]).relu();
        sq_norm(&hp.matmul(&params[5]).add(&params[6]).relu())            // clean(relu(o)) → back on the manifold
    };
    async fn rollout_cos(ctx: &Arc<ferric_core::Context>, k: usize, params: &[Tensor], encode_t: &dyn Fn(&Tensor) -> Tensor, predict_t: &dyn Fn(&Tensor, &Tensor) -> Tensor) -> f32 {
        let t = 256usize;
        let mut pos: Vec<(usize, usize)> = (0..t).map(|i| ((u(i as u32, 5) * G as f32) as usize % G, (u(i as u32, 6) * G as f32) as usize % G)).collect();
        let mut ob = vec![0.0f32; t * IN];
        for i in 0..t { blob(pos[i], &mut ob[i * IN..(i + 1) * IN]); }
        let mut x = encode_t(&Tensor::from_vec(ctx, &ob, &[t, IN]));
        for step in 0..k {
            let mut oh = vec![0.0f32; t * NA];
            for i in 0..t { let a = (u(i as u32, 100 + step as u32) * NA as f32) as usize % NA; oh[i * NA + a] = 1.0; pos[i] = next_pos(pos[i], a); }
            x = predict_t(&x, &Tensor::from_vec(ctx, &oh, &[t, NA]));
        }
        let mut tob = vec![0.0f32; t * IN];
        for i in 0..t { blob(pos[i], &mut tob[i * IN..(i + 1) * IN]); }
        let tru = encode_t(&Tensor::from_vec(ctx, &tob, &[t, IN]));
        let (xv, tv) = (x.to_vec().await, tru.to_vec().await);
        let mut c = 0.0f64;
        for i in 0..t {
            let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
            for j in 0..params[1].numel() { let a = xv[i * 64 + j]; let b = tv[i * 64 + j]; d += a * b; na += a * a; nb += b * b; }
            c += (d / (na.sqrt() * nb.sqrt() + 1e-9)) as f64;
        }
        (c / t as f64) as f32
    }
    // ---- the acting layer: energy-descent PLANNING in the trained latent (MuZero/TD-MPC2 branch) ----
    // Batched over T agents: each step, roll every action 2 steps in the world model, pick the one whose
    // imagined latent gets closest (lowest energy) to the goal latent, act in the real world, repeat.
    async fn plan_reach(ctx: &Arc<ferric_core::Context>, encode_t: &dyn Fn(&Tensor) -> Tensor, predict_t: &dyn Fn(&Tensor, &Tensor) -> Tensor) -> f32 {
        let (t, maxsteps) = (150usize, 50usize);
        let mut pos: Vec<(usize, usize)> = (0..t).map(|i| ((u(i as u32, 11) * G as f32) as usize % G, (u(i as u32, 12) * G as f32) as usize % G)).collect();
        let goal: Vec<(usize, usize)> = (0..t).map(|i| ((u(i as u32, 13) * G as f32) as usize % G, (u(i as u32, 14) * G as f32) as usize % G)).collect();
        let mut gob = vec![0.0f32; t * IN];
        for i in 0..t { blob(goal[i], &mut gob[i * IN..(i + 1) * IN]); }
        let xg = encode_t(&Tensor::from_vec(ctx, &gob, &[t, IN]));
        let onehot = |a: usize| -> Tensor { let mut o = vec![0.0f32; t * NA]; for i in 0..t { o[i * NA + a] = 1.0; } Tensor::from_vec(ctx, &o, &[t, NA]) };
        let sqd = |xa: &Tensor| -> Tensor { let d = xa.sub(&xg); d.mul(&d).sum(&[1], true) };
        let mut reached = vec![false; t];
        for _ in 0..maxsteps {
            if reached.iter().all(|&r| r) { break; }
            let mut ob = vec![0.0f32; t * IN];
            for i in 0..t { blob(pos[i], &mut ob[i * IN..(i + 1) * IN]); }
            let x = encode_t(&Tensor::from_vec(ctx, &ob, &[t, IN]));
            let mut en: Vec<Vec<f32>> = Vec::with_capacity(NA);
            for a in 0..NA {
                let xa = predict_t(&x, &onehot(a));
                let mut e = sqd(&xa).to_vec().await;                       // 1-step energy
                for a2 in 0..NA { let e2 = sqd(&predict_t(&xa, &onehot(a2))).to_vec().await; for i in 0..t { if e2[i] < e[i] { e[i] = e2[i]; } } } // + 2-step lookahead
                en.push(e);
            }
            for i in 0..t {
                if reached[i] { continue; }
                let (mut ba, mut best) = (0usize, en[0][i]);
                for a in 1..NA { if en[a][i] < best { best = en[a][i]; ba = a; } }
                pos[i] = next_pos(pos[i], ba);
                if pos[i] == goal[i] { reached[i] = true; }
            }
        }
        reached.iter().filter(|&&r| r).count() as f32 / t as f32
    }

    let c1 = rollout_cos(&ctx, 1, &params, &encode_t, &predict_t).await;
    let c4 = rollout_cos(&ctx, 4, &params, &encode_t, &predict_t).await;
    let c8 = rollout_cos(&ctx, 8, &params, &encode_t, &predict_t).await;
    let reach = plan_reach(&ctx, &encode_t, &predict_t).await;
    println!("\n  loss {first:.5} → trained.  multi-step rollout fidelity (cosine to true next latent):");
    println!("     1-step {c1:.3}   4-step {c4:.3}   8-step {c8:.3}");
    println!("  acting by energy-descent PLANNING in the trained latent → goals reached: {:.1}%", reach * 100.0);
    assert!(c1 > 0.9 && c8 > 0.7, "world model did not train to fidelity (c1={c1}, c8={c8})");
    assert!(reach > 0.5, "planning did not reach goals (reach={reach})");
    println!("✅ EFA on Ferric (GPU): world model trained (8-step fidelity {c8:.3}) + planning reaches {:.0}% of goals — the full perceive→plan→act loop on the fabric", reach * 100.0);
}
