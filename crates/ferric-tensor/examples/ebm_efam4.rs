//! EFA-M stage 4 — the PERCEPTUAL FRONT-END: pixels → embedding → certified memory → the shipped controller.
//! Stages 1–3 proved the mechanism on synthetic unit-vector contexts. Real perceptual embeddings are nothing like
//! that: correlated, anisotropic, cluttered by task-irrelevant content, and aliasing is a property of the WORLD.
//! This stage runs the whole pipeline on rendered observations and measures whether the assumptions survive:
//!   scene = landmark layout on a ring (the identity) rendered into 24×24 pixels, PLUS the pendulum arm at a random
//!   task-irrelevant pose (clutter that varies across observations of the SAME scene), plus pixel noise.
//!   embedding = frozen random projection (576→32, JL-style; no training, deterministic) + normalization.
//! Measured, gates fixed BEFORE the run:
//!   [1] MANIFOLD STATS: within-scene vs between-scene embedding distances — the margin must exist (reported either
//!       way); gate thresholds CALIBRATED on 10 held-out calibration scenes (quantiles), EVALUATED on 20 fresh ones
//!   [2] recall on fresh observations of stored scenes (new clutter pose + noise) ≥95%; closed loop via efa-1 ≥95%
//!   [3] novel-scene refusal by the cue-energy gate: AUROC ≥0.95
//!   [4] REAL aliasing: near-duplicate scenes (ONE landmark nudged 0.15 rad, contradicting goal) — where do they land
//!       (duplicate zone → refused-as-contradiction · conflict zone → refused-as-aliasing · beyond τ_sep → appended =
//!       SILENT POISON, counted and reported honestly) + parent recall after the aliasing wave
//!   [5] consolidation on real observations (6-shot vs 1-shot store) improves recall margin at high pixel noise
//!   [6] certificates recomputed on the REAL manifold (how much do the synthetic-era guarantees tighten/loosen?) ·
//!       price · bit-exact determinism
//! HONEST: 24×24 renders and random-feature embeddings are the SMALLEST honest perceptual stand-in (real images and
//! learned encoders = the ManiSkill step); thresholds calibrated (disclosed); one seed; the arm is the only clutter.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_efam4 --release`
use std::f32::consts::PI;
const H: usize = 128; const EMB: usize = 6; const DT: f32 = 0.05; const UMAX: f32 = 4.0;
const MDIR: &str = "/Users/dcharlot/vibe-coding/efa/models/efa-1";
const IMG: usize = 24; const NPIX: usize = IMG * IMG;
const CDIM: usize = 32; const PDIM: usize = CDIM + 2;
const BETA: f32 = 24.0; const ETA: f32 = 0.5; const KMEM: usize = 12;
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
struct Efa1 { emb: Vec<f32>, fw: Vec<Vec<f32>>, fb1: Vec<f32>, fw2: Vec<f32>, fb2: Vec<f32>, fw3: Vec<f32>, fb3: [f32; 3] }
impl Efa1 {
    fn act1(&self, s: [f32; 2], g: f32) -> f32 {
        let fin = 12 + 3 + 1 + EMB; let mut f = vec![0.0f32; fin];
        let d = s[0] - g; f[0] = d.cos(); f[1] = d.sin(); f[2] = s[1]; f[3] = s[0].sin();
        for c in 0..EMB { f[16 + c] = self.emb[c]; }
        let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.fb1[j]; for c in 0..fin { z += f[c] * self.fw[c][j]; } h1[j] = z.max(0.0); }
        let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.fb2[j]; for k in 0..H { z += h1[k] * self.fw2[k * H + j]; } h2[j] = z.max(0.0); }
        let mut o = self.fb3[0]; for j in 0..H { o += h2[j] * self.fw3[j * 3]; }
        o.clamp(-UMAX, UMAX) }
}
// ---- the perceptual world ----
#[derive(Clone)]
struct Scene { angs: Vec<f32>, ints: Vec<f32>, goal: f32 }
fn mk_scene(seed: u32) -> Scene {
    let nl = 4;
    Scene { angs: (0..nl).map(|k| (u(seed, 10 + k as u32) * 2.0 - 1.0) * PI).collect(),
            ints: (0..nl).map(|k| 0.6 + 0.4 * u(seed, 20 + k as u32)).collect(),
            goal: (u(seed, 30) * 2.0 - 1.0) * 1.0 }
}
fn render(sc: &Scene, arm_th: f32, noise_seed: u32) -> Vec<f32> {
    let mut px = vec![0.0f32; NPIX];
    let cx = IMG as f32 / 2.0 - 0.5; let cy = cx; let ring = 9.0;
    let mut put = |x: f32, y: f32, v: f32| { let (xi, yi) = (x.round() as i32, y.round() as i32);
        if xi >= 0 && yi >= 0 && (xi as usize) < IMG && (yi as usize) < IMG {
            let idx = yi as usize * IMG + xi as usize; px[idx] = (px[idx] + v).min(1.0); } };
    for (a, i) in sc.angs.iter().zip(&sc.ints) {                         // landmarks: cross blobs on the ring
        let (x, y) = (cx + ring * a.sin(), cy + ring * a.cos());
        put(x, y, *i); put(x + 1.0, y, i * 0.5); put(x - 1.0, y, i * 0.5); put(x, y + 1.0, i * 0.5); put(x, y - 1.0, i * 0.5); }
    for k in 0..8 {                                                      // the arm: task-irrelevant clutter at arm_th
        let r = 1.0 + k as f32; put(cx + r * arm_th.sin(), cy + r * arm_th.cos(), 0.8); }
    for p in px.iter_mut() { *p += (u(noise_seed, 900) * 2.0 - 1.0) * 0.0; }  // per-image DC (kept 0; per-pixel below)
    for (i, p) in px.iter_mut().enumerate() { *p = (*p + (u(noise_seed.wrapping_add(i as u32), 901) * 2.0 - 1.0) * 0.05).clamp(0.0, 1.0); }
    px
}
// frozen random-feature embedding (JL projection), deterministic
struct Proj { w: Vec<f32> }
impl Proj {
    fn new() -> Proj { Proj { w: (0..NPIX * CDIM).map(|i| { let (a, b) = (u(i as u32, 501), u(i as u32, 502));
        (-2.0 * a.ln()).sqrt() * (2.0 * PI * b).cos() / (NPIX as f32).sqrt() }).collect() } }
    fn embed(&self, px: &[f32]) -> [f32; CDIM] {
        let mut c = [0.0f32; CDIM];
        for k in 0..CDIM { let mut z = 0.0; for p in 0..NPIX { z += px[p] * self.w[p * CDIM + k]; } c[k] = z; }
        let n = c.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        for k in 0..CDIM { c[k] /= n; } c
    }
}
// ---- memory (stage-3 machinery: context-space gate + certificates) ----
#[derive(Clone)]
struct Mem { xi: Vec<[f32; PDIM]>, writes: Vec<usize> }
impl Mem {
    fn energy_ctx(&self, q: &[f32; PDIM]) -> f32 {
        let mut mx = f32::MIN;
        let exps: Vec<f32> = self.xi.iter().map(|x| { let d2: f32 = (0..CDIM).map(|c| (q[c] - x[c]).powi(2)).sum();
            let e = -BETA * d2 / 2.0; if e > mx { mx = e; } e }).collect();
        let s: f32 = exps.iter().map(|e| (e - mx).exp()).sum();
        -(mx + s.ln()) / BETA
    }
    fn retrieve(&self, cue: &[f32; PDIM]) -> ([f32; PDIM], usize) {
        let mut q = *cue;
        for _ in 0..KMEM {
            let mut mx = f32::MIN;
            let logits: Vec<f32> = self.xi.iter().map(|x| { let d2: f32 = (0..PDIM).map(|c| (q[c] - x[c]).powi(2)).sum();
                let e = -BETA * d2 / 2.0; if e > mx { mx = e; } e }).collect();
            let mut ws: Vec<f32> = logits.iter().map(|e| (e - mx).exp()).collect();
            let sw: f32 = ws.iter().sum(); for w in &mut ws { *w /= sw; }
            let mut grad = [0.0f32; PDIM];
            for (i, x) in self.xi.iter().enumerate() { for c in 0..PDIM { grad[c] += ws[i] * (q[c] - x[c]); } }
            for c in 0..PDIM { q[c] -= ETA * grad[c]; } }
        let (mut bi, mut bd) = (0usize, f32::MAX);
        for (i, x) in self.xi.iter().enumerate() { let d2: f32 = (0..PDIM).map(|c| (q[c] - x[c]).powi(2)).sum();
            if d2 < bd { bd = d2; bi = i; } }
        (q, bi)
    }
    fn nearest_ctx(&self, p: &[f32; PDIM]) -> (usize, f32) {
        let (mut bi, mut bd) = (0usize, f32::MAX);
        for (i, x) in self.xi.iter().enumerate() { let d2: f32 = (0..CDIM).map(|c| (p[c] - x[c]).powi(2)).sum();
            if d2 < bd { bd = d2; bi = i; } }
        (bi, bd.sqrt())
    }
    fn certify(&self) -> Vec<(f32, f32)> {                                // (ctx separation, ε bound)
        let m = self.xi.len(); let dia = 2.0f32 * (2.0f32).sqrt();
        (0..m).map(|i| { let mut dmin = f32::MAX;
            for j in 0..m { if j != i { let d: f32 = (0..CDIM).map(|c| (self.xi[i][c] - self.xi[j][c]).powi(2)).sum::<f32>().sqrt();
                if d < dmin { dmin = d; } } }
            if m == 1 { dmin = dia; }
            (dmin, (m as f32 - 1.0).max(1.0) * (-BETA * dmin * dmin / 4.0).exp() * dia) }).collect()
    }
}
fn mk_pattern(c: &[f32; CDIM], g: f32) -> [f32; PDIM] {
    let mut p = [0.0f32; PDIM]; p[..CDIM].copy_from_slice(c); p[CDIM] = g.cos(); p[CDIM + 1] = g.sin(); p
}
fn goal_of(q: &[f32; PDIM]) -> f32 { q[CDIM + 1].atan2(q[CDIM]) }
fn main() {
    let t = load_st(&format!("{MDIR}/model.safetensors"));
    let fin = 12 + 3 + 1 + EMB; let g3 = t["flow.b3"].clone();
    let m1 = Efa1 { emb: t["body_embedding"][..EMB].to_vec(),
        fw: (0..fin).map(|c| t[&format!("flow.in{}", c)].clone()).collect(), fb1: t["flow.b1"].clone(), fw2: t["flow.w2"].clone(), fb2: t["flow.b2"].clone(), fw3: t["flow.w3"].clone(), fb3: [g3[0], g3[1], g3[2]] };
    let proj = Proj::new();
    println!("  EFA-M stage 4 — perceptual front-end: pixels → frozen embedding → certified memory → shipped controller\n");
    // RECORDED NEGATIVE (v1 of this stage): a single-frame embedding FAILS — the moving arm dominates the linear
    // features (within-scene d ≈ between-scene d; margin ABSENT; recall 61.5%; aliases appended 14/20). Generic
    // instantaneous features cannot suppress dynamic clutter. Fix (training-free, uses structure every robot has):
    // TEMPORAL AGGREGATION — the scene is static, the clutter moves; average K=8 frames → arm attenuates ~1/K,
    // noise ~1/√K, landmarks persist. Fixation averaging, not segmentation.
    let obs = |sc: &Scene, seed: u32| -> [f32; CDIM] {
        let mut acc = vec![0.0f32; NPIX];
        for f in 0..8u32 { let arm = (u(seed.wrapping_add(f * 131), 41) * 2.0 - 1.0) * PI;
            let px = render(sc, arm, seed.wrapping_add(f * 977));
            for (a, p) in acc.iter_mut().zip(&px) { *a += p / 8.0; } }
        proj.embed(&acc) };
    // ── [1] manifold statistics + threshold calibration on HELD-OUT scenes ──
    let calib: Vec<Scene> = (0..10).map(|i| mk_scene(100 + i)).collect();
    let (mut din, mut dbet) = (vec![], vec![]);
    for (i, sc) in calib.iter().enumerate() {
        for tr in 0..40u32 { let a = obs(sc, 1000 + i as u32 * 100 + tr); let b = obs(sc, 5000 + i as u32 * 100 + tr);
            din.push((0..CDIM).map(|c| (a[c] - b[c]).powi(2)).sum::<f32>().sqrt()); }
        for (j, sc2) in calib.iter().enumerate() { if j <= i { continue; }
            for tr in 0..8u32 { let a = obs(sc, 9000 + i as u32 * 100 + tr); let b = obs(sc2, 9500 + j as u32 * 100 + tr);
                dbet.push((0..CDIM).map(|c| (a[c] - b[c]).powi(2)).sum::<f32>().sqrt()); } } }
    din.sort_by(|a, b| a.partial_cmp(b).unwrap()); dbet.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tau_dup = din[(din.len() as f32 * 0.99) as usize];               // 99th pct within-scene
    let tau_sep = dbet[(dbet.len() as f32 * 0.01) as usize] * 0.9;       // 1st pct between-scene, margin 0.9
    println!("  [1] manifold stats (10 held-out calibration scenes, arm clutter + pixel noise):");
    println!("     within-scene d: median {:.3} / 99pct {:.3} · between-scene d: median {:.3} / 1pct {:.3}",
        din[din.len() / 2], tau_dup, dbet[dbet.len() / 2], dbet[(dbet.len() as f32 * 0.01) as usize]);
    println!("     ⇒ calibrated τ_dup = {:.3} · τ_sep = {:.3} · margin {}", tau_dup, tau_sep,
        if tau_sep > tau_dup { format!("EXISTS ({:.3})", tau_sep - tau_dup) } else { "ABSENT — the embedding cannot support the gate (finding)".into() });
    let goal_agree = 0.4f32;
    // write gate with calibrated thresholds
    let write_gate = |mem: &mut Mem, p: &[f32; PDIM]| -> u8 {
        if mem.xi.is_empty() { mem.xi.push(*p); mem.writes.push(1); return 0; }
        let (i, d) = mem.nearest_ctx(p);
        if d > tau_sep { mem.xi.push(*p); mem.writes.push(1); 0 }        // append
        else if d < tau_dup {
            let g_new = p[CDIM + 1].atan2(p[CDIM]); let g_old = mem.xi[i][CDIM + 1].atan2(mem.xi[i][CDIM]);
            if wrap(g_new - g_old).abs() > goal_agree { return 3; }       // refuse-contradiction
            let k = mem.writes[i] as f32;
            for c in 0..PDIM { mem.xi[i][c] = mem.xi[i][c] * (k / (k + 1.0)) + p[c] / (k + 1.0); }
            let mut n = 0.0; for c in 0..CDIM { n += mem.xi[i][c] * mem.xi[i][c]; }
            let n = n.sqrt().max(1e-6); for c in 0..CDIM { mem.xi[i][c] /= n; }
            let gn = (mem.xi[i][CDIM].powi(2) + mem.xi[i][CDIM + 1].powi(2)).sqrt().max(1e-6);
            mem.xi[i][CDIM] /= gn; mem.xi[i][CDIM + 1] /= gn;
            mem.writes[i] += 1; 1 }                                       // consolidate
        else { 2 }                                                        // refuse-aliasing
    };
    // ── build the store: 20 FRESH scenes (disjoint from calibration), 1-shot writes ──
    let scenes: Vec<Scene> = (0..20).map(|i| mk_scene(700 + i)).collect();
    let mut mem = Mem { xi: vec![], writes: vec![] };
    let mut w_append = 0;
    for (i, sc) in scenes.iter().enumerate() { let c = obs(sc, 30000 + i as u32);
        if write_gate(&mut mem, &mk_pattern(&c, sc.goal)) == 0 { w_append += 1; } }
    println!("\n  store: {}/{} fresh scenes appended (collisions on the real manifold: {})", w_append, scenes.len(), scenes.len() - w_append);
    // ── [2] recall on fresh observations + closed loop ──
    let (mut ok, mut tot) = (0, 0);
    for (i, sc) in scenes.iter().enumerate() { for tr in 0..10u32 { tot += 1;
        let c = obs(sc, 40000 + i as u32 * 100 + tr);
        let (q, _) = mem.retrieve(&mk_pattern(&c, 0.0).map(|x| x));       // goal coords unknown at recall
        let mut cue = mk_pattern(&c, 0.0); cue[CDIM] = 0.0; cue[CDIM + 1] = 0.0;
        let (q2, _) = mem.retrieve(&cue); let _ = q;
        if wrap(goal_of(&q2) - sc.goal).abs() < 0.15 { ok += 1; } } }
    println!("\n  [2] recall from fresh observations (new clutter pose + noise): {:.1}% ({} probes)", ok as f32 / tot as f32 * 100.0, tot);
    let (mut reach, nn) = (0, 40);
    for ep in 0..nn { let i = (u(ep, 21) * scenes.len() as f32) as usize % scenes.len();
        let c = obs(&scenes[i], 60000 + ep);
        let mut cue = mk_pattern(&c, 0.0); cue[CDIM] = 0.0; cue[CDIM + 1] = 0.0;
        let (q, _) = mem.retrieve(&cue); let g = goal_of(&q);
        let mut s = [(u(ep, 3) * 2.0 - 1.0) * PI, 0.0];
        for _ in 0..300 { let a = m1.act1(s, g); s = step1(s, a); }
        if wrap(s[0] - scenes[i].goal).abs() < 0.35 && s[1].abs() < 0.7 { reach += 1; } }
    println!("     closed loop (pixels → recall → shipped efa-1): reach {:.1}%", reach as f32 / nn as f32 * 100.0);
    // ── [3] novel-scene refusal via cue energy ──
    let mut e_st = vec![]; let mut e_nov = vec![];
    for (i, sc) in scenes.iter().enumerate() { for tr in 0..10u32 {
        let c = obs(sc, 70000 + i as u32 * 100 + tr);
        let mut cue = mk_pattern(&c, 0.0); cue[CDIM] = 0.0; cue[CDIM + 1] = 0.0;
        e_st.push(mem.energy_ctx(&cue)); } }
    for k in 0..200u32 { let sc = mk_scene(20000 + k);
        let c = obs(&sc, 80000 + k);
        let mut cue = mk_pattern(&c, 0.0); cue[CDIM] = 0.0; cue[CDIM + 1] = 0.0;
        e_nov.push(mem.energy_ctx(&cue)); }
    e_st.sort_by(|a, b| a.partial_cmp(b).unwrap()); e_nov.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (mut w, mut tt) = (0u64, 0u64);
    for &a in e_st.iter().step_by(2) { for &b in e_nov.iter().step_by(2) { tt += 1; if b > a { w += 1; } } }
    let tau_mem = e_st[(e_st.len() as f32 * 0.95) as usize];
    println!("\n  [3] novel-scene refusal: AUROC {:.3} · novel flagged at τ_mem: {:.1}%",
        w as f32 / tt as f32, e_nov.iter().filter(|&&e| e > tau_mem).count() as f32 / e_nov.len() as f32 * 100.0);
    // ── [4] REAL aliasing: one landmark nudged 0.15 rad, contradicting goal ──
    let (mut al_dup, mut al_conf, mut al_append) = (0, 0, 0);
    let mut mem2 = mem.clone();
    for (i, sc) in scenes.iter().enumerate() {
        let mut al = sc.clone(); al.angs[0] += 0.15; al.goal = wrap(sc.goal + PI * 0.8);
        let c = obs(&al, 90000 + i as u32);
        match write_gate(&mut mem2, &mk_pattern(&c, al.goal)) { 3 => al_dup += 1, 2 => al_conf += 1, 0 => al_append += 1, _ => {} } }
    println!("\n  [4] real aliasing wave (20 near-duplicates, one landmark +0.15 rad, contradicting goal):");
    println!("     refused-as-contradiction {} · refused-as-aliasing {} · APPENDED (silent poison) {}", al_dup, al_conf, al_append);
    let (mut ok2, mut tot2) = (0, 0);
    for (i, sc) in scenes.iter().enumerate() { for tr in 0..5u32 { tot2 += 1;
        let c = obs(sc, 95000 + i as u32 * 100 + tr);
        let mut cue = mk_pattern(&c, 0.0); cue[CDIM] = 0.0; cue[CDIM + 1] = 0.0;
        let (q, _) = mem2.retrieve(&cue);
        if wrap(goal_of(&q) - sc.goal).abs() < 0.15 { ok2 += 1; } } }
    println!("     parent recall AFTER the wave (store incl. any appended aliases): {:.1}%", ok2 as f32 / tot2 as f32 * 100.0);
    // ── [5] consolidation on real observations at doubled pixel noise ──
    // (recall margin under stress: 1-shot store vs 6-shot consolidated store)
    let stress_obs = |sc: &Scene, seed: u32| -> [f32; CDIM] {
        let mut acc = vec![0.0f32; NPIX];
        for f in 0..8u32 { let arm = (u(seed.wrapping_add(f * 131), 41) * 2.0 - 1.0) * PI;
            let mut px = render(sc, arm, seed.wrapping_add(f * 977));
            for (i, p) in px.iter_mut().enumerate() { *p = (*p + (u(seed.wrapping_add(7777 + f * 100 + i as u32), 903) * 2.0 - 1.0) * 0.10).clamp(0.0, 1.0); }
            for (a, p) in acc.iter_mut().zip(&px) { *a += p / 8.0; } }
        proj.embed(&acc) };
    let mut mem_1shot = Mem { xi: vec![], writes: vec![] };
    let mut mem_cons = Mem { xi: vec![], writes: vec![] };
    for (i, sc) in scenes.iter().enumerate() {
        let c = stress_obs(sc, 100000 + i as u32);
        let _ = write_gate(&mut mem_1shot, &mk_pattern(&c, sc.goal));
        for tr in 0..6u32 { let c = stress_obs(sc, 100000 + i as u32 + tr * 1000);
            let _ = write_gate(&mut mem_cons, &mk_pattern(&c, sc.goal)); } }
    let rec = |mm: &Mem| -> f32 { let (mut ok, mut tot) = (0, 0);
        for (i, sc) in scenes.iter().enumerate() { for tr in 0..10u32 { tot += 1;
            let c = stress_obs(sc, 120000 + i as u32 * 100 + tr);
            let mut cue = mk_pattern(&c, 0.0); cue[CDIM] = 0.0; cue[CDIM + 1] = 0.0;
            let (q, _) = mm.retrieve(&cue);
            if wrap(goal_of(&q) - sc.goal).abs() < 0.15 { ok += 1; } } }
        ok as f32 / tot as f32 * 100.0 };
    println!("\n  [5] consolidation under 2× pixel noise: 1-shot store recall {:.1}% · 6-shot consolidated {:.1}%", rec(&mem_1shot), rec(&mem_cons));
    // ── [6] certificates on the real manifold + price + determinism ──
    let cert = mem.certify();
    let mut seps: Vec<f32> = cert.iter().map(|c| c.0).collect(); seps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("\n  [6] certificates on the REAL manifold: ctx separation min {:.3} / median {:.3} (synthetic era: ~1.2) — ε median {:.2e}",
        seps[0], seps[seps.len() / 2], { let mut es: Vec<f32> = cert.iter().map(|c| c.1).collect(); es.sort_by(|a, b| a.partial_cmp(b).unwrap()); es[es.len() / 2] });
    println!("     price: embed {:.1} kFLOP + retrieve {:.1} kFLOP per recall (pixels included)",
        (NPIX * CDIM) as f32 / 1000.0, (KMEM * mem.xi.len() * PDIM * 2) as f32 / 1000.0);
    let ca = obs(&scenes[7], 424242); let cb = obs(&scenes[7], 424242);
    let det = ca.iter().zip(cb.iter()).all(|(x, y)| x.to_bits() == y.to_bits());
    println!("     determinism (render→embed): {}", if det { "bit-exact ✓" } else { "MISMATCH ✗" });
    println!("\n  Honest scope: 24×24 renders + frozen random features = the smallest honest perceptual stand-in (learned");
    println!("  encoders on ManiSkill observations = the public-stake step); thresholds CALIBRATED on held-out scenes");
    println!("  (disclosed); the arm is the only clutter; one seed.");
}
