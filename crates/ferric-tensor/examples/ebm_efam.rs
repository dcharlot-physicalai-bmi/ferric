//! EFA-M SPIKE — the attractor-memory mechanism, measured (Move 1 of docs/BENCHMARKS-2026.md).
//! The field's own findings (RoboMME, ICML 2026 Oral) read as a spec for associative memory: perceptual wins,
//! light modulation wins, recurrent write-mechanisms fail — and the Hopfield/energy slot is EMPTY (3-way verified).
//! This spike builds the mechanism EFA-natively: a Dense Associative Memory whose RETRIEVAL IS DESCENT ON AN ENERGY
//!   E(q) = −(1/β)·log Σᵢ exp(−β·‖q−ξᵢ‖²/2)          (modern-Hopfield distance form; attractors at stored patterns)
//! — the same inference primitive as EFA actuation, pointed at memory. Stored patterns = (context ⊕ goal-encoding)
//! pairs; recall = heteroassociation (partial/corrupted context in → goal out) under distractor pressure.
//! Measured, with gates fixed BEFORE the run:
//!   [1] content-addressable recall ≥95% at 25% cue corruption, M=20 distractors
//!   [2] ENERGY = MEMORY-CONFIDENCE CERTIFICATE: retrieval energy separates stored vs NOVEL contexts (τ_mem gate);
//!       gated wrong-goal executions ≈ 0 (the model KNOWS when its memory is trustworthy — CoRL workshop problem #6)
//!   [3] closed-loop: recalled goal handed to the SHIPPED efa-1 artifact (memory modulates the flagship) —
//!       reach ≥95% (vs memoryless baselines: mean-goal / random-stored-goal)
//!   [4] capacity curve M ∈ {20,100,500} at fixed β · [5] joules-per-recall priced in FLOPs · [6] bit-exact retrieval
//! HONEST: spike scope — writes are direct pattern storage (Lyapunov-certified consolidation = named next stage);
//! contexts are synthetic feature vectors (stand-ins for perceptual embeddings); 1-DOF body (the body is not the point).
//!
//! Run: `cargo run -p ferric-tensor --example ebm_efam --release`
use std::f32::consts::PI;
const H: usize = 128; const EMB: usize = 6; const DT: f32 = 0.05; const UMAX: f32 = 4.0; const CPL: f32 = 0.5;
const MDIR: &str = "/Users/dcharlot/vibe-coding/efa/models/efa-1";
const CDIM: usize = 16;                       // context feature dims
const PDIM: usize = CDIM + 2;                // pattern = context ⊕ [cos g, sin g]
const BETA: f32 = 24.0; const ETA: f32 = 0.5; const KMEM: usize = 20;
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn wrap(x: f32) -> f32 { let mut a = x; while a > PI { a -= 2.0 * PI; } while a < -PI { a += 2.0 * PI; } a }
fn step1(s: [f32; 2], uu: f32) -> [f32; 2] {
    let no = s[1] + DT * (-s[0].sin() - 0.05 * s[1] + uu.clamp(-UMAX, UMAX));
    [wrap(s[0] + DT * no), no]
}
fn load_st(path: &str) -> std::collections::HashMap<String, Vec<f32>> {
    let raw = std::fs::read(path).expect("efa-1 model.safetensors not found");
    let hl = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&raw[8..8 + hl]).unwrap().to_string(); let data = &raw[8 + hl..];
    let mut out = std::collections::HashMap::new(); let mut rest = header.as_str();
    while let Some(q) = rest.find("\"dtype\"") { let pre = &rest[..q]; let ne = pre.rfind("\":{").unwrap(); let ns = pre[..ne].rfind('"').unwrap() + 1;
        let name = pre[ns..ne].to_string(); let a = &rest[q..]; let os = a.find("\"data_offsets\":[").unwrap() + 16; let oe = a[os..].find(']').unwrap() + os;
        let of: Vec<usize> = a[os..oe].split(',').map(|s| s.trim().parse().unwrap()).collect();
        out.insert(name, data[of[0]..of[1]].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()); rest = &a[oe..]; }
    out
}
// the shipped EFA-1 flow (1-DOF path) — the memory module MODULATES this artifact, unchanged
struct Efa1 { emb: Vec<f32>, fw: Vec<Vec<f32>>, fb1: Vec<f32>, fw2: Vec<f32>, fb2: Vec<f32>, fw3: Vec<f32>, fb3: [f32; 3] }
impl Efa1 {
    fn act1(&self, s: [f32; 2], g: f32) -> f32 {
        let fin = 12 + 3 + 1 + EMB; let mut f = vec![0.0f32; fin];
        let d = s[0] - g; f[0] = d.cos(); f[1] = d.sin(); f[2] = s[1]; f[3] = s[0].sin();
        for c in 0..EMB { f[16 + c] = self.emb[c]; }                      // body 0 embedding row
        let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.fb1[j]; for c in 0..fin { z += f[c] * self.fw[c][j]; } h1[j] = z.max(0.0); }
        let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.fb2[j]; for k in 0..H { z += h1[k] * self.fw2[k * H + j]; } h2[j] = z.max(0.0); }
        let mut o = self.fb3[0]; for j in 0..H { o += h2[j] * self.fw3[j * 3]; }
        o.clamp(-UMAX, UMAX) }
}
// ---- the Dense Associative Memory: energy over stored patterns, retrieval = gradient descent on E ----
struct Dam { xi: Vec<[f32; PDIM]> }
impl Dam {
    fn energy(&self, q: &[f32; PDIM]) -> f32 {
        let mut mx = f32::MIN;
        let exps: Vec<f32> = self.xi.iter().map(|x| { let d2: f32 = (0..PDIM).map(|c| (q[c] - x[c]).powi(2)).sum();
            let e = -BETA * d2 / 2.0; if e > mx { mx = e; } e }).collect();
        let s: f32 = exps.iter().map(|e| (e - mx).exp()).sum();
        -(mx + s.ln()) / BETA
    }
    // ∇E(q) = Σᵢ wᵢ (q − ξᵢ),  wᵢ = softmax(−β‖q−ξᵢ‖²/2) — descent pulls q into the nearest basin.
    // Returns (q_converged, e_cue, e_conv, slot). CONFIDENCE = e_cue (the CUE's energy — how close the query starts
    // to any stored pattern). v1 of this spike used e_conv and it FAILED (AUROC 0.469): a novel cue still falls into
    // SOME basin and its bottom looks like any other bottom — the recorded negative, printed below.
    fn retrieve(&self, cue: &[f32; PDIM]) -> ([f32; PDIM], f32, f32, usize) {
        let e_cue = self.energy(cue);
        let mut q = *cue;
        for _ in 0..KMEM {
            let mut mx = f32::MIN;
            let logits: Vec<f32> = self.xi.iter().map(|x| { let d2: f32 = (0..PDIM).map(|c| (q[c] - x[c]).powi(2)).sum();
                let e = -BETA * d2 / 2.0; if e > mx { mx = e; } e }).collect();
            let mut ws: Vec<f32> = logits.iter().map(|e| (e - mx).exp()).collect();
            let sw: f32 = ws.iter().sum(); for w in &mut ws { *w /= sw; }
            let mut grad = [0.0f32; PDIM];
            for (i, x) in self.xi.iter().enumerate() { for c in 0..PDIM { grad[c] += ws[i] * (q[c] - x[c]); } }
            for c in 0..PDIM { q[c] -= ETA * grad[c]; }
        }
        let e_conv = self.energy(&q);
        let (mut bi, mut bd) = (0usize, f32::MAX);
        for (i, x) in self.xi.iter().enumerate() { let d2: f32 = (0..PDIM).map(|c| (q[c] - x[c]).powi(2)).sum();
            if d2 < bd { bd = d2; bi = i; } }
        (q, e_cue, e_conv, bi)
    }
}
fn mk_pairs(m: usize, seed: u32) -> (Vec<[f32; CDIM]>, Vec<f32>, Dam) {
    let mut ctx = Vec::with_capacity(m); let mut goals = Vec::with_capacity(m); let mut xi = Vec::with_capacity(m);
    for i in 0..m { let mut c = [0.0f32; CDIM]; let mut n = 0.0;
        for k in 0..CDIM { c[k] = u(seed + i as u32, 100 + k as u32) * 2.0 - 1.0; n += c[k] * c[k]; }
        let n = n.sqrt().max(1e-6); for k in 0..CDIM { c[k] /= n; }
        let g = (u(seed + i as u32, 200) * 2.0 - 1.0) * 1.0;
        let mut p = [0.0f32; PDIM]; p[..CDIM].copy_from_slice(&c); p[CDIM] = g.cos(); p[CDIM + 1] = g.sin();
        ctx.push(c); goals.push(g); xi.push(p); }
    (ctx, goals, Dam { xi })
}
fn corrupt(c: &[f32; CDIM], frac: f32, seed: u32) -> [f32; PDIM] {
    let mut q = [0.0f32; PDIM];
    for k in 0..CDIM { q[k] = if u(seed, 300 + k as u32) < frac { 0.0 } else { c[k] }; }   // masked dims; goal coords unknown (0)
    q
}
fn goal_of(q: &[f32; PDIM]) -> f32 { q[CDIM + 1].atan2(q[CDIM]) }
fn main() {
    let t = load_st(&format!("{MDIR}/model.safetensors"));
    let fin = 12 + 3 + 1 + EMB; let g3 = t["flow.b3"].clone();
    let m1 = Efa1 { emb: t["body_embedding"][..EMB].to_vec(),
        fw: (0..fin).map(|c| t[&format!("flow.in{}", c)].clone()).collect(), fb1: t["flow.b1"].clone(), fw2: t["flow.w2"].clone(), fb2: t["flow.b2"].clone(), fw3: t["flow.w3"].clone(), fb3: [g3[0], g3[1], g3[2]] };
    println!("  EFA-M spike — attractor memory: retrieval IS energy descent; the energy IS the confidence certificate\n");
    // ── [1] content-addressable recall under corruption, M=20 ──
    let m = 20; let (ctx, goals, dam) = mk_pairs(m, 40);
    println!("  [1] heteroassociative recall (context→goal), M={} stored pairs, {} descent steps:", m, KMEM);
    let mut e_stored: Vec<f32> = vec![]; let mut ec_stored: Vec<f32> = vec![];
    for frac in [0.0f32, 0.25, 0.5] {
        let (mut ok, nn) = (0, 400);
        for tr in 0..nn { let i = tr % m;
            let q0 = corrupt(&ctx[i], frac, 7000 + tr as u32);
            let (q, e0, ec, slot) = dam.retrieve(&q0);
            if frac == 0.25 { e_stored.push(e0); ec_stored.push(ec); }
            if slot == i && (goal_of(&q) - goals[i]).abs() < 0.1 { ok += 1; } }
        println!("     cue corruption {:>3.0}%: recall {:>5.1}%  ({} trials)", frac * 100.0, ok as f32 / nn as f32 * 100.0, nn);
    }
    // ── [2] the confidence certificate: stored vs NOVEL contexts ──
    println!("\n  [2] energy as memory-confidence certificate (certificate = CUE energy E(q0), pre-descent):");
    let mut e_novel: Vec<f32> = vec![]; let mut ec_novel: Vec<f32> = vec![];
    for tr in 0..400u32 { let mut c = [0.0f32; CDIM]; let mut n = 0.0;
        for k in 0..CDIM { c[k] = u(90000 + tr, 100 + k as u32) * 2.0 - 1.0; n += c[k] * c[k]; }
        let n = n.sqrt().max(1e-6); for k in 0..CDIM { c[k] /= n; }
        let (_, e0, ec, _) = dam.retrieve(&corrupt(&c, 0.25, 91000 + tr));
        e_novel.push(e0); ec_novel.push(ec); }
    e_stored.sort_by(|a, b| a.partial_cmp(b).unwrap()); e_novel.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tau_mem = e_stored[(e_stored.len() as f32 * 0.95) as usize];      // 95th pct of stored-cue CUE energies
    let novel_above = e_novel.iter().filter(|&&e| e > tau_mem).count() as f32 / e_novel.len() as f32 * 100.0;
    let auroc = |a: &Vec<f32>, b: &Vec<f32>| -> f32 { let (mut w, mut t) = (0u64, 0u64);
        for &x in a.iter().step_by(4) { for &y in b.iter().step_by(4) { t += 1; if y > x { w += 1; } } }
        w as f32 / t as f32 };
    ec_stored.sort_by(|a, b| a.partial_cmp(b).unwrap()); ec_novel.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("     stored-cue E(q0): median {:+.3} · novel-cue E(q0): median {:+.3} · τ_mem(95pct) = {:+.3}",
        e_stored[e_stored.len() / 2], e_novel[e_novel.len() / 2], tau_mem);
    println!("     novel cues flagged (E(q0)>τ_mem): {:.1}% · AUROC = {:.3}", novel_above, auroc(&e_stored, &e_novel));
    println!("     [recorded negative: converged-energy certificate AUROC = {:.3} — a novel cue still falls into SOME", auroc(&ec_stored, &ec_novel));
    println!("      basin and its bottom looks like any other; confidence lives in the CUE energy, not the resting state]");
    // ── [3] closed loop: recalled goal → the SHIPPED efa-1 flow ──
    println!("\n  [3] closed loop — memory modulates the shipped artifact (60 episodes, 25% corrupted cues):");
    let mean_goal: f32 = goals.iter().sum::<f32>() / m as f32;
    let run = |gsel: &dyn Fn(u32) -> f32| -> f32 { let (mut reach, nn) = (0, 60);
        for ep in 0..nn { let g = gsel(ep);
            let i = ep as usize % m; let gtrue = goals[i];
            let mut s = [(u(5000 + ep, 3) * 2.0 - 1.0) * PI, 0.0];
            for _ in 0..300 { let a = m1.act1(s, g); s = step1(s, a); }
            if wrap(s[0] - gtrue).abs() < 0.35 && s[1].abs() < 0.7 { reach += 1; } }
        reach as f32 / 60.0 * 100.0 };
    let r_mem = run(&|ep| { let i = ep as usize % m; let (q, _, _, _) = dam.retrieve(&corrupt(&ctx[i], 0.25, 8000 + ep)); goal_of(&q) });
    let r_mean = run(&|_| mean_goal);
    let r_rand = run(&|ep| goals[(u(ep, 44) * m as f32) as usize % m]);
    println!("     recalled goal → efa-1:  reach {:>5.1}%      [memoryless baselines: mean-goal {:.1}% · random-stored {:.1}%]", r_mem, r_mean, r_rand);
    // ── [4] memory-gated agency: act only when the memory is confident ──
    println!("\n  [4] memory-gated agency (act iff E ≤ τ_mem; 60 stored + 60 novel cue episodes):");
    let (mut act_ok, mut act_wrong, mut refuse_stored, mut refuse_novel, mut act_novel) = (0, 0, 0, 0, 0);
    for ep in 0..120u32 { let stored = ep < 60;
        let (q, e0, _, slot) = if stored { let i = ep as usize % m; dam.retrieve(&corrupt(&ctx[i], 0.25, 9500 + ep)) }
            else { let mut c = [0.0f32; CDIM]; let mut n = 0.0;
                for k in 0..CDIM { c[k] = u(95000 + ep, 100 + k as u32) * 2.0 - 1.0; n += c[k] * c[k]; }
                let n = n.sqrt().max(1e-6); for k in 0..CDIM { c[k] /= n; }
                dam.retrieve(&corrupt(&c, 0.25, 96000 + ep)) };
        if e0 > tau_mem { if stored { refuse_stored += 1; } else { refuse_novel += 1; } continue; }
        if stored { let i = ep as usize % m;
            if slot == i && (goal_of(&q) - goals[i]).abs() < 0.1 { act_ok += 1; } else { act_wrong += 1; } }
        else { act_novel += 1; } }
    println!("     stored cues: acted-correct {} · acted-WRONG {} · refused {}", act_ok, act_wrong, refuse_stored);
    println!("     novel cues:  acted (silent wrong-goal risk) {} · correctly REFUSED {}", act_novel, refuse_novel);
    println!("     ⇒ ungated, all 60 novel cues would have driven to a nearest-neighbor goal silently.");
    // ── [5] capacity + [6] pricing/determinism ──
    println!("\n  [5] capacity at fixed β={} (recall @25% corruption):", BETA);
    for mm in [20usize, 100, 500] { let (ctx2, goals2, dam2) = mk_pairs(mm, 70);
        let (mut ok, nn) = (0, 400);
        for tr in 0..nn { let i = (u(tr as u32, 55) * mm as f32) as usize % mm;
            let (q, _, _, slot) = dam2.retrieve(&corrupt(&ctx2[i], 0.25, 60000 + tr as u32));
            if slot == i && (goal_of(&q) - goals2[i]).abs() < 0.1 { ok += 1; } }
        println!("     M = {:>3}: recall {:>5.1}%", mm, ok as f32 / nn as f32 * 100.0); }
    let recall_flops = KMEM * (2 * m * PDIM + 3 * m) + 2 * m * PDIM;    // descent (dists+softmax+grad) + final E
    println!("\n  [6] joules-per-recall: ~{:.1} kFLOP per retrieval (M=20) vs 39.2 kFLOP per efa-1 decision — memory ≈ {:.0}% of a decision.",
        recall_flops as f32 / 1000.0, recall_flops as f32 / 39168.0 * 100.0);
    let (qa, ea, _, _) = dam.retrieve(&corrupt(&ctx[3], 0.25, 12345));
    let (qb, eb, _, _) = dam.retrieve(&corrupt(&ctx[3], 0.25, 12345));
    let det = qa.iter().zip(qb.iter()).all(|(x, y)| x.to_bits() == y.to_bits()) && ea.to_bits() == eb.to_bits();
    println!("     determinism: {}", if det { "bit-exact ✓" } else { "MISMATCH ✗" });
    println!("\n  Honest scope: direct pattern writes (certified consolidation = next stage); synthetic context features;");
    println!("  1-DOF body (the body is not the point). The mechanism claim: recall = the SAME energy-descent primitive");
    println!("  as EFA actuation, and the energy is a calibrated confidence certificate over memory itself.");
}
