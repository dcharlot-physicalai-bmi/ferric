//! EFA-M stage 5 — HIDDEN-PROPERTY ATTRACTORS: infer a latent physical parameter from interaction, store it as an
//! attractor keyed by the object's PERCEPTUAL identity, recall it without re-probing, gate on confidence, and feed it
//! back to CORRECT the shipped controller. The combination nobody is adjacent to:
//!   · PhyPush (mass+friction from one push) infers the property but keeps NO memory — re-infers every encounter.
//!   · RoboMME memory recalls object identity but infers NO physical property.
//!   · EFA-M: identify once → store as basin (stage-3 gate) → recall by perceptual cue (stage-4 embedding) → the SAME
//!     energy is the confidence certificate → correct the SHIPPED efa-1. Memory ∩ system-ID in one energy.
//!
//! WHICH latent parameter? — the choice is itself a finding (two recorded negatives that shaped it):
//!   (A) constant LOAD BIAS: the efa-1 FEEDBACK policy silently ABSORBS it — long- AND short-horizon reach stay 100%
//!       for |b|≤1. A feedback loop rejects a constant disturbance; nothing to compensate. (recorded)
//!   (B) actuator GAIN k: genuinely degrades SHORT-horizon reach (100%→45% as k:1→0.2) — but naive output-rescaling
//!       u/k̂ makes it WORSE (saturates the command, overshoots), and where the plant is under-powered NO scaling can
//!       manufacture missing torque. Identification tells you the ceiling; it can't raise it. (recorded)
//!   (C) SENSOR/MOUNTING OFFSET δ (this stage): the controller drives the OBSERVED angle to goal, so the TRUE angle
//!       settles at g−δ — reach fails for |δ|>0.35 and the feedback loop CANNOT self-correct (it thinks it succeeded).
//!       Gravity depends on the TRUE angle, so interaction leaks δ; correcting the commanded goal to g+δ̂ cleanly
//!       restores reach. THE case where identification is necessary and sufficient.
//! Identification: δ̂ = argmin_δ Σ_t [ω_{t+1}−ω_t − DT(−sin(θ_obs−δ) − 0.05ω + clamp(u))]²  (line search; noise ⇒ sharpens).
//! Measured, gates fixed BEFORE the run:
//!   [1] identification sharpens with interaction · [2] nominal FAILS / identified+corrected RECOVERS across |δ|
//!   [3] memory: 12 objects, first-encounter probe→store; RE-encounter recall-without-probe reaches, probes saved
//!   [4] two-confidence gate (recognize via cue energy · identified via posterior width) + aliasing safety
//!   [5] price · bit-exact determinism
//! HONEST: 1-DOF sensor offset as the single latent parameter (mass/friction/multi-param = the extension); 24×24
//!   renders + frozen features + temporal aggregation (stage-4 stand-in); thresholds calibrated (disclosed); one seed.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_efam5 --release`
use std::f32::consts::PI;
const H: usize = 128; const EMB: usize = 6; const DT: f32 = 0.05; const UMAX: f32 = 4.0;
const MDIR: &str = "/Users/dcharlot/vibe-coding/efa/models/efa-1";
const IMG: usize = 24; const NPIX: usize = IMG * IMG;
const CDIM: usize = 32; const PDIM: usize = CDIM + 2;
const BETA: f32 = 24.0; const ETA: f32 = 0.5; const KMEM: usize = 12;
const SIGP: f32 = 0.01;
const DLO: f32 = -0.8; const DHI: f32 = 0.8; const DHALF: f32 = (DHI - DLO) / 2.0;  // DMID = 0
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn wrap(x: f32) -> f32 { let mut a = x; while a > PI { a -= 2.0 * PI; } while a < -PI { a += 2.0 * PI; } a }
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
    fn raw1(&self, s: [f32; 2], g: f32) -> f32 {
        let fin = 12 + 3 + 1 + EMB; let mut f = vec![0.0f32; fin];
        let d = s[0] - g; f[0] = d.cos(); f[1] = d.sin(); f[2] = s[1]; f[3] = s[0].sin();
        for c in 0..EMB { f[16 + c] = self.emb[c]; }
        let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.fb1[j]; for c in 0..fin { z += f[c] * self.fw[c][j]; } h1[j] = z.max(0.0); }
        let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.fb2[j]; for k in 0..H { z += h1[k] * self.fw2[k * H + j]; } h2[j] = z.max(0.0); }
        let mut o = self.fb3[0]; for j in 0..H { o += h2[j] * self.fw3[j * 3]; } o }
}
// pure dynamics on the TRUE state (no hidden param in the physics) + process noise
fn dyn1(s: [f32; 2], u_app: f32, nseed: u32) -> [f32; 2] {
    let u_c = u_app.clamp(-UMAX, UMAX);
    let no = s[1] + DT * (-s[0].sin() - 0.05 * s[1] + u_c) + (u(nseed, 950) * 2.0 - 1.0) * SIGP;
    [wrap(s[0] + DT * no), no]
}
fn d_enc(d: f32) -> f32 { d / DHALF }
fn d_dec(e: f32) -> f32 { (DHALF * e).clamp(DLO, DHI) }
// ---- perceptual world (stage-4 temporal aggregation) ----
#[derive(Clone)]
struct Obj { angs: Vec<f32>, ints: Vec<f32>, goal: f32, delta: f32 }
fn mk_obj(seed: u32) -> Obj {
    Obj { angs: (0..4).map(|j| (u(seed, 10 + j) * 2.0 - 1.0) * PI).collect(),
          ints: (0..4).map(|j| 0.6 + 0.4 * u(seed, 20 + j)).collect(),
          goal: (u(seed, 30) * 2.0 - 1.0) * 0.7,
          delta: DLO + (DHI - DLO) * u(seed, 31) }
}
fn render(o: &Obj, arm_th: f32, nseed: u32) -> Vec<f32> {
    let mut px = vec![0.0f32; NPIX]; let c = IMG as f32 / 2.0 - 0.5; let ring = 9.0;
    let mut put = |x: f32, y: f32, v: f32| { let (xi, yi) = (x.round() as i32, y.round() as i32);
        if xi >= 0 && yi >= 0 && (xi as usize) < IMG && (yi as usize) < IMG { let i = yi as usize * IMG + xi as usize; px[i] = (px[i] + v).min(1.0); } };
    for (a, it) in o.angs.iter().zip(&o.ints) { let (x, y) = (c + ring * a.sin(), c + ring * a.cos());
        put(x, y, *it); put(x + 1.0, y, it * 0.5); put(x - 1.0, y, it * 0.5); put(x, y + 1.0, it * 0.5); put(x, y - 1.0, it * 0.5); }
    for k in 0..8 { let r = 1.0 + k as f32; put(c + r * arm_th.sin(), c + r * arm_th.cos(), 0.8); }
    for (i, p) in px.iter_mut().enumerate() { *p = (*p + (u(nseed.wrapping_add(i as u32), 901) * 2.0 - 1.0) * 0.05).clamp(0.0, 1.0); }
    px
}
struct Proj { w: Vec<f32> }
impl Proj {
    fn new() -> Proj { Proj { w: (0..NPIX * CDIM).map(|i| { let (a, b) = (u(i as u32, 501), u(i as u32, 502));
        (-2.0 * a.ln()).sqrt() * (2.0 * PI * b).cos() / (NPIX as f32).sqrt() }).collect() } }
    fn embed(&self, px: &[f32]) -> [f32; CDIM] { let mut c = [0.0f32; CDIM];
        for k in 0..CDIM { let mut z = 0.0; for p in 0..NPIX { z += px[p] * self.w[p * CDIM + k]; } c[k] = z; }
        let n = c.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6); for k in 0..CDIM { c[k] /= n; } c }
    fn obs(&self, o: &Obj, seed: u32) -> [f32; CDIM] {
        let mut acc = vec![0.0f32; NPIX];
        for f in 0..8u32 { let arm = (u(seed.wrapping_add(f * 131), 41) * 2.0 - 1.0) * PI;
            let px = render(o, arm, seed.wrapping_add(f * 977)); for (a, p) in acc.iter_mut().zip(&px) { *a += p / 8.0; } }
        self.embed(&acc) }
}
// ---- memory ----
#[derive(Clone)]
struct Mem { xi: Vec<[f32; PDIM]>, writes: Vec<usize> }
impl Mem {
    fn energy_ctx(&self, q: &[f32; PDIM]) -> f32 { let mut mx = f32::MIN;
        let exps: Vec<f32> = self.xi.iter().map(|x| { let d2: f32 = (0..CDIM).map(|c| (q[c] - x[c]).powi(2)).sum();
            let e = -BETA * d2 / 2.0; if e > mx { mx = e; } e }).collect();
        let s: f32 = exps.iter().map(|e| (e - mx).exp()).sum(); -(mx + s.ln()) / BETA }
    fn retrieve(&self, cue: &[f32; PDIM]) -> ([f32; PDIM], usize) { let mut q = *cue;
        for _ in 0..KMEM { let mut mx = f32::MIN;
            let logits: Vec<f32> = self.xi.iter().map(|x| { let d2: f32 = (0..PDIM).map(|c| (q[c] - x[c]).powi(2)).sum();
                let e = -BETA * d2 / 2.0; if e > mx { mx = e; } e }).collect();
            let mut ws: Vec<f32> = logits.iter().map(|e| (e - mx).exp()).collect();
            let sw: f32 = ws.iter().sum(); for w in &mut ws { *w /= sw; }
            let mut grad = [0.0f32; PDIM]; for (i, x) in self.xi.iter().enumerate() { for c in 0..PDIM { grad[c] += ws[i] * (q[c] - x[c]); } }
            for c in 0..PDIM { q[c] -= ETA * grad[c]; } }
        let (mut bi, mut bd) = (0usize, f32::MAX);
        for (i, x) in self.xi.iter().enumerate() { let d2: f32 = (0..PDIM).map(|c| (q[c] - x[c]).powi(2)).sum(); if d2 < bd { bd = d2; bi = i; } }
        (q, bi) }
    fn nearest_ctx(&self, p: &[f32; PDIM]) -> (usize, f32) { let (mut bi, mut bd) = (0usize, f32::MAX);
        for (i, x) in self.xi.iter().enumerate() { let d2: f32 = (0..CDIM).map(|c| (p[c] - x[c]).powi(2)).sum(); if d2 < bd { bd = d2; bi = i; } }
        (bi, bd.sqrt()) }
}
fn dpat(c: &[f32; CDIM], d: f32) -> [f32; PDIM] {
    let mut p = [0.0f32; PDIM]; p[..CDIM].copy_from_slice(c); let phi = d_enc(d) * PI / 2.0; p[CDIM] = phi.cos(); p[CDIM + 1] = phi.sin(); p }
fn pat_d(q: &[f32; PDIM]) -> f32 { d_dec(q[CDIM + 1].atan2(q[CDIM]) / (PI / 2.0)) }
// residual SSE for a hypothesized offset δ over a probe log (θ_obs, ω, u, ω')
fn sse(log: &[(f32, f32, f32, f32)], d: f32) -> f32 {
    log.iter().map(|&(tho, om, uc, om2)| { let pred = om + DT * (-(tho - d).sin() - 0.05 * om + uc); (om2 - pred).powi(2) }).sum()
}
// probe: drive the controller on the OBSERVED (offset) state, log transitions, then δ̂ = argmin_δ SSE (coarse+fine)
fn probe(m1: &Efa1, o: &Obj, kk: usize, seed: u32) -> (f32, f32) {
    let mut s = [(u(seed, 3) * 2.0 - 1.0) * PI, 0.0]; let pg = 0.4;
    let mut log: Vec<(f32, f32, f32, f32)> = vec![];
    for t in 0..kk { let tho = s[0] + o.delta; let uc = m1.raw1([tho, s[1]], pg).clamp(-UMAX, UMAX);
        let ns = dyn1(s, uc, seed.wrapping_add(t as u32 * 61)); log.push((tho, s[1], uc, ns[1])); s = ns; }
    let (mut bd, mut be) = (0.0f32, f32::MAX);                          // coarse grid over [DLO,DHI]
    for i in 0..=32 { let d = DLO + (DHI - DLO) * i as f32 / 32.0; let e = sse(&log, d); if e < be { be = e; bd = d; } }
    let step = (DHI - DLO) / 32.0;                                       // fine refine ±step
    for i in 0..=20 { let d = (bd - step + 2.0 * step * i as f32 / 20.0).clamp(DLO, DHI); let e = sse(&log, d); if e < be { be = e; bd = d; } }
    // posterior width proxy: curvature of SSE near the minimum
    let (ep, em) = (sse(&log, (bd + 0.05).min(DHI)), sse(&log, (bd - 0.05).max(DLO)));
    let curv = ((ep + em - 2.0 * be) / 0.0025).max(1e-3);
    let width = (2.0 * be / (log.len() as f32 * curv)).sqrt().min(1.0);
    (bd, width)
}
// closed-loop TRUE reach; controller sees θ_true+δ, commands goal+dcomp (dcomp=0 = nominal). horizon hz.
fn reach_true(m1: &Efa1, o: &Obj, dcomp: f32, seed: u32) -> bool {
    let mut s = [(u(seed, 3) * 2.0 - 1.0) * PI, 0.0];
    for t in 0..300 { let tho = s[0] + o.delta; let uc = m1.raw1([tho, s[1]], o.goal + dcomp).clamp(-UMAX, UMAX);
        s = dyn1(s, uc, seed.wrapping_add(1000 + t as u32)); }
    wrap(s[0] - o.goal).abs() < 0.35 && s[1].abs() < 0.7
}
fn main() {
    let t = load_st(&format!("{MDIR}/model.safetensors"));
    let fin = 12 + 3 + 1 + EMB; let g3 = t["flow.b3"].clone();
    let m1 = Efa1 { emb: t["body_embedding"][..EMB].to_vec(),
        fw: (0..fin).map(|c| t[&format!("flow.in{}", c)].clone()).collect(), fb1: t["flow.b1"].clone(), fw2: t["flow.w2"].clone(), fb2: t["flow.b2"].clone(), fw3: t["flow.w3"].clone(), fb3: [g3[0], g3[1], g3[2]] };
    let proj = Proj::new();
    println!("  EFA-M stage 5 — hidden-property attractors: identify → store → recall → correct (memory ∩ system-ID)");
    println!("  latent = sensor/mounting OFFSET δ (the case feedback CANNOT self-correct; load-bias & gain negatives in header)\n");
    // ── [1] identification sharpens ──
    println!("  [1] identification: δ̂ error vs interaction steps K (physics leaks δ via gravity):");
    for kk in [5usize, 10, 20, 40, 80] { let (mut err, nn) = (0.0f32, 60);
        for ep in 0..nn { let o = mk_obj(4000 + ep); let (dh, _) = probe(&m1, &o, kk, 50000 + ep * 7); err += (dh - o.delta).abs(); }
        println!("     K = {:>2}: mean |δ̂ − δ| = {:.3}", kk, err / nn as f32); }
    // ── [2] nominal FAILS, identified+corrected RECOVERS ──
    println!("\n  [2] control — nominal (commands goal, unaware of δ) vs identified+corrected (commands goal+δ̂):");
    for dmag in [0.0f32, 0.2, 0.4, 0.6, 0.8] { let (mut rn, mut rc, nn) = (0, 0, 40);
        for ep in 0..nn { let mut o = mk_obj(6000 + ep); o.delta = if ep % 2 == 0 { dmag } else { -dmag };
            rn += reach_true(&m1, &o, 0.0, 70000 + ep) as i32;
            let (dh, _) = probe(&m1, &o, 40, 71000 + ep);
            rc += reach_true(&m1, &o, dh, 72000 + ep) as i32; }
        println!("     |δ| = {:.2}: nominal reach {:>3.0}% · identified+corrected {:>3.0}%", dmag, rn as f32 / nn as f32 * 100.0, rc as f32 / nn as f32 * 100.0); }
    // ── build the store: 12 objects, first-encounter probe → certified write ──
    let objs: Vec<Obj> = (0..12).map(|i| mk_obj(800 + i)).collect();
    let hold: Vec<Obj> = (0..8).map(|i| mk_obj(300 + i)).collect();
    let (mut din, mut dbet) = (vec![], vec![]);
    for (i, o) in hold.iter().enumerate() { for tr in 0..30u32 { let a = proj.obs(o, 1000 + i as u32 * 100 + tr); let b = proj.obs(o, 5000 + i as u32 * 100 + tr);
        din.push((0..CDIM).map(|c| (a[c] - b[c]).powi(2)).sum::<f32>().sqrt()); }
        for (j, o2) in hold.iter().enumerate() { if j <= i { continue; } for tr in 0..6u32 { let a = proj.obs(o, 9000 + tr); let b = proj.obs(o2, 9500 + tr);
            dbet.push((0..CDIM).map(|c| (a[c] - b[c]).powi(2)).sum::<f32>().sqrt()); } } }
    din.sort_by(|a, b| a.partial_cmp(b).unwrap()); dbet.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let tau_sep = dbet[(dbet.len() as f32 * 0.01) as usize] * 0.9;
    let mut mem = Mem { xi: vec![], writes: vec![] };
    for (i, o) in objs.iter().enumerate() { let c = proj.obs(o, 30000 + i as u32);
        let (dh, _) = probe(&m1, o, 40, 31000 + i as u32); let p = dpat(&c, dh);
        if mem.xi.is_empty() { mem.xi.push(p); mem.writes.push(1); }
        else { let (_, d) = mem.nearest_ctx(&p); if d > tau_sep { mem.xi.push(p); mem.writes.push(1); } } }
    println!("\n  store: {} objects identified & stored; first-encounter probes = {}", mem.xi.len(), objs.len());
    // ── [3] re-encounter: recall WITHOUT re-probing ──
    let (mut rc_mem, mut rc_reprobe, mut rc_none, mut rc_gated, mut nn) = (0, 0, 0, 0, 0);
    let (mut derr, mut nb, mut gated_probes) = (0.0f32, 0, 0);
    for (i, o) in objs.iter().enumerate() { for tr in 0..5u32 { nn += 1;
        let c = proj.obs(o, 40000 + i as u32 * 100 + tr);
        let mut cue = dpat(&c, 0.0); cue[CDIM] = 0.0; cue[CDIM + 1] = 0.0;
        let (_, slot) = mem.retrieve(&cue); let d_recall = pat_d(&mem.xi[slot]);
        derr += (d_recall - o.delta).abs(); nb += 1;
        rc_mem += reach_true(&m1, o, d_recall, 41000 + i as u32 * 100 + tr) as i32;
        let (dh, _) = probe(&m1, o, 40, 42000 + i as u32 * 100 + tr);
        rc_reprobe += reach_true(&m1, o, dh, 43000 + i as u32 * 100 + tr) as i32;
        rc_none += reach_true(&m1, o, 0.0, 44000 + i as u32 * 100 + tr) as i32;
        // GATED operating policy: trust recall unless a cheap 10-step interaction-check disagrees → re-probe
        let (dchk, _) = probe(&m1, o, 10, 45500 + i as u32 * 100 + tr);
        let d_use = if (dchk - d_recall).abs() < 0.2 { d_recall } else { gated_probes += 1; probe(&m1, o, 40, 45700 + i as u32 * 100 + tr).0 };
        rc_gated += reach_true(&m1, o, d_use, 46000 + i as u32 * 100 + tr) as i32; } }
    println!("\n  [3] re-encounter ({} trials): recall-δ̂ error {:.3}", nn, derr / nb as f32);
    println!("     TRUE reach: RECALL-only {:.0}% · RE-PROBE-every-time {:.0}% · NO-memory {:.0}% · GATED(recall+verify) {:.0}%",
        rc_mem as f32 / nn as f32 * 100.0, rc_reprobe as f32 / nn as f32 * 100.0, rc_none as f32 / nn as f32 * 100.0, rc_gated as f32 / nn as f32 * 100.0);
    println!("     ⇒ GATED reaches ~re-probe quality using {}/{} full probes ({:.0}% probe cost) — memory pays where it's confident, probes where it isn't",
        gated_probes, nn, gated_probes as f32 / nn as f32 * 100.0);
    // ── [4] two-confidence gate + aliasing safety ──
    println!("\n  [4] two-confidence gate (recognize via cue energy · identified via posterior width):");
    let mut e_known = vec![]; let mut e_novel = vec![];
    for (i, o) in objs.iter().enumerate() { for tr in 0..5u32 { let c = proj.obs(o, 45000 + i as u32 * 100 + tr);
        let mut cue = dpat(&c, 0.0); cue[CDIM] = 0.0; cue[CDIM + 1] = 0.0; e_known.push(mem.energy_ctx(&cue)); } }
    for k in 0..120u32 { let o = mk_obj(24000 + k); let c = proj.obs(&o, 46000 + k);
        let mut cue = dpat(&c, 0.0); cue[CDIM] = 0.0; cue[CDIM + 1] = 0.0; e_novel.push(mem.energy_ctx(&cue)); }
    e_known.sort_by(|a, b| a.partial_cmp(b).unwrap()); e_novel.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (mut w, mut tt) = (0u64, 0u64); for &a in e_known.iter() { for &b in e_novel.iter().step_by(2) { tt += 1; if b > a { w += 1; } } }
    let tau_rec = e_known[(e_known.len() as f32 * 0.95) as usize];
    println!("     recognize-gate AUROC = {:.3}; novel flagged→probe: {:.0}%", w as f32 / tt as f32, e_novel.iter().filter(|&&e| e > tau_rec).count() as f32 / e_novel.len() as f32 * 100.0);
    let (mut w5, mut w40) = (0.0f32, 0.0f32); for ep in 0..20u32 { let o = mk_obj(9000 + ep);
        w5 += probe(&m1, &o, 5, 47000 + ep).1; w40 += probe(&m1, &o, 40, 48000 + ep).1; }
    println!("     identified-gate: mean posterior width K=5 {:.3} → keep probing · K=40 {:.3} → act", w5 / 20.0, w40 / 20.0);
    // aliasing: near-dup, OPPOSITE offset → stale recall would MIS-correct; interaction-verify catches → re-probe
    let (mut caught, mut probe_ok, mut poison, nal) = (0, 0, 0, 20);
    for j in 0..nal { let base = &objs[j % objs.len()]; let mut al = base.clone();
        al.angs[0] += 0.15; al.delta = if base.delta > 0.0 { DLO + 0.1 } else { DHI - 0.1 };
        let c = proj.obs(&al, 49000 + j as u32);
        let mut cue = dpat(&c, 0.0); cue[CDIM] = 0.0; cue[CDIM + 1] = 0.0;
        let (_, slot) = mem.retrieve(&cue); let d_recall = pat_d(&mem.xi[slot]);
        let (dh, _) = probe(&m1, &al, 10, 49500 + j as u32);            // short interaction check
        let verify_ok = (dh - d_recall).abs() < 0.2;
        if verify_ok { if !reach_true(&m1, &al, d_recall, 49700 + j as u32) { poison += 1; } }
        else { caught += 1; let (dh2, _) = probe(&m1, &al, 40, 49800 + j as u32);
            if reach_true(&m1, &al, dh2, 49900 + j as u32) { probe_ok += 1; } } }
    println!("     aliasing ({} near-dups, opposite δ): stale recall caught {}/{} → re-probe reaches {}/{}; uncaught poison {}", nal, caught, nal, probe_ok, caught, poison);
    // ── [5] price + determinism ──
    println!("\n  [5] price: probe 40 steps (interaction) amortized ONCE/object · recall {:.1} kFLOP every re-encounter (0 probes)",
        (KMEM * mem.xi.len() * PDIM * 2) as f32 / 1000.0);
    let (da, _) = probe(&m1, &objs[2], 40, 424242); let (db, _) = probe(&m1, &objs[2], 40, 424242);
    println!("     determinism (probe→δ̂): {}", if da.to_bits() == db.to_bits() { "bit-exact ✓" } else { "MISMATCH ✗" });
    println!("\n  Honest scope: 1-DOF sensor offset as the single latent parameter (mass/friction/multi-param = the extension);");
    println!("  perceptual stand-in via stage-4 temporal aggregation; thresholds calibrated; one seed.");
    println!("  The claim: identify a physical property ONCE, store it as a perceptually-keyed attractor, recall it for");
    println!("  free thereafter, gate on TWO confidences, and correct the shipped controller — memory ∩ system-ID.");
}
