//! EFA-1 agency loop — ON THE SHIPPED ARTIFACT. The EFA-1 spec's bleeding-edge mechanism: the model's OWN energy
//! decides when to think harder and when to reach for tools, every path seeded and deterministic:
//!   L1: a = flow K=1 (the cheap path)            — if E(s,a) ≤ τ_body execute
//!   L2: a = flow K=4 (think harder, EBT-style)   — if E ≤ τ execute       (t ≤ 0.75, inside the trained t ≤ 0.9 region)
//!   L3: planner tool — two-stage discrete argmin over the model's OWN potential (the expensive Gᵈ tool)
//!   L4: genetic tool — seeded evolution strategy minimizing E (gradient-free fallback); execute argmin-E of all candidates
//! τ_body is calibrated from validation quantiles of E(s, a_flow) (95th pct) — from the artifact alone, no side data.
//! Measured: (a) in-distribution card eval — reach, escalation rate, priced mean FLOPs/decision (agency must stay cheap
//! when the energy is content); (b) OOD stress — goals |g| ∈ [1.05,1.35], OUTSIDE the training band (±1.0), plain-K=1
//! vs the full agency loop (do the tools buy anything off-distribution? measured, either way); (c) bit-exact determinism
//! of the full ladder. Results → config.json "agency" field. Harness is the cert-validated reconstruction (100/100/100).
//!
//! Run: `cargo run -p ferric-tensor --example ebm_efa1agency --release`
use std::f32::consts::PI;
const H: usize = 128; const EMB: usize = 6; const DT: f32 = 0.05; const UMAX: f32 = 4.0; const CPL: f32 = 0.5;
const NB: usize = 3; const NJ: [usize; NB] = [1, 2, 3];
const MDIR: &str = "/Users/dcharlot/vibe-coding/efa/models/efa-1";
const FLOW_FLOPS: f32 = 39168.0;                       // priced per forward pass (the card's number)
const POT_FLOPS: f32 = 2.0 * (21.0 * 128.0 + 128.0 * 128.0 + 128.0) * 1.0 + 2.0 * 128.0; // ≈38k per E eval
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn wrap(x: f32) -> f32 { let mut a = x; while a > PI { a -= 2.0 * PI; } while a < -PI { a += 2.0 * PI; } a }
fn step(nj: usize, s: [f32; 6], uu: [f32; 3]) -> [f32; 6] {
    let mut th = [s[0], s[1], s[2]]; let om = [s[3], s[4], s[5]]; let mut no = om;
    for i in 0..nj { let mut cpl = 0.0;
        if i > 0 { cpl += CPL * (th[i - 1] - th[i]).sin(); }
        if i + 1 < nj { cpl += CPL * (th[i + 1] - th[i]).sin(); }
        no[i] = om[i] + DT * (-th[i].sin() - 0.05 * om[i] + cpl + uu[i].clamp(-UMAX, UMAX)); }
    for i in 0..nj { th[i] = wrap(th[i] + DT * no[i]); }
    [th[0], th[1], th[2], no[0], no[1], no[2]]
}
fn feat12(nj: usize, s: [f32; 6], g: [f32; 3]) -> [f32; 12] { let mut f = [0.0f32; 12];
    for i in 0..nj { let d = s[i] - g[i]; f[i * 4] = d.cos(); f[i * 4 + 1] = d.sin(); f[i * 4 + 2] = s[3 + i]; f[i * 4 + 3] = s[i].sin(); } f }
fn load_st(path: &str) -> std::collections::HashMap<String, Vec<f32>> {
    let raw = std::fs::read(path).expect("model.safetensors not found");
    let hl = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&raw[8..8 + hl]).unwrap().to_string(); let data = &raw[8 + hl..];
    let mut out = std::collections::HashMap::new(); let mut rest = header.as_str();
    while let Some(q) = rest.find("\"dtype\"") { let pre = &rest[..q]; let ne = pre.rfind("\":{").unwrap(); let ns = pre[..ne].rfind('"').unwrap() + 1;
        let name = pre[ns..ne].to_string(); let a = &rest[q..]; let os = a.find("\"data_offsets\":[").unwrap() + 16; let oe = a[os..].find(']').unwrap() + os;
        let of: Vec<usize> = a[os..oe].split(',').map(|s| s.trim().parse().unwrap()).collect();
        out.insert(name, data[of[0]..of[1]].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()); rest = &a[oe..]; }
    out
}
struct Efa1 { emb: Vec<f32>, fw: Vec<Vec<f32>>, fb1: Vec<f32>, fw2: Vec<f32>, fb2: Vec<f32>, fw3: Vec<f32>, fb3: [f32; 3],
    pw: Vec<Vec<f32>>, pb1: Vec<f32>, pw2: Vec<f32>, pb2: Vec<f32>, pw3: Vec<f32>, pb3: f32 }
impl Efa1 {
    fn flow(&self, bi: usize, s: [f32; 6], g: [f32; 3], a: [f32; 3], t: f32) -> [f32; 3] {
        let nj = NJ[bi]; let ff = feat12(nj, s, g); let fin = 12 + 3 + 1 + EMB; let mut f = vec![0.0f32; fin];
        for c in 0..12 { f[c] = ff[c]; } for j in 0..3 { f[12 + j] = a[j]; } f[15] = t;
        for c in 0..EMB { f[16 + c] = self.emb[bi * EMB + c]; }
        let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.fb1[j]; for c in 0..fin { z += f[c] * self.fw[c][j]; } h1[j] = z.max(0.0); }
        let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.fb2[j]; for k in 0..H { z += h1[k] * self.fw2[k * H + j]; } h2[j] = z.max(0.0); }
        let mut o = self.fb3; for j in 0..H { for c in 0..3 { o[c] += h2[j] * self.fw3[j * 3 + c]; } } o }
    fn act_k(&self, bi: usize, s: [f32; 6], g: [f32; 3], kk: usize) -> [f32; 3] {
        let nj = NJ[bi]; let mut a = [0.0f32; 3];
        for k in 0..kk { let t = k as f32 / kk as f32; let v = self.flow(bi, s, g, a, t);
            for i in 0..nj { a[i] += v[i] / kk as f32; } }
        for i in 0..nj { a[i] = a[i].clamp(-UMAX, UMAX); } for i in nj..3 { a[i] = 0.0; } a }
    fn energy(&self, bi: usize, s: [f32; 6], g: [f32; 3], a: [f32; 3]) -> f32 {
        let nj = NJ[bi]; let ff = feat12(nj, s, g); let pin = 12 + 3 + EMB; let mut f = vec![0.0f32; pin];
        for c in 0..12 { f[c] = ff[c]; } for j in 0..3 { f[12 + j] = a[j]; }
        for c in 0..EMB { f[15 + c] = self.emb[bi * EMB + c]; }
        let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.pb1[j]; for c in 0..pin { z += f[c] * self.pw[c][j]; } h1[j] = z.max(0.0); }
        let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.pb2[j]; for k in 0..H { z += h1[k] * self.pw2[k * H + j]; } h2[j] = z.max(0.0); }
        let mut o = self.pb3; for j in 0..H { o += h2[j] * self.pw3[j]; } o }
    // L3 planner tool: two-stage discrete argmin over the model's OWN potential (coarse 5ⁿ + fine ±0.75 refine)
    fn planner(&self, bi: usize, s: [f32; 6], g: [f32; 3]) -> ([f32; 3], f32) {
        let nj = NJ[bi]; let coarse = [-4.0f32, -2.0, 0.0, 2.0, 4.0];
        let mut best = ([0.0f32; 3], f32::MAX); let mut evals = 0.0f32;
        let mut idx = [0usize; 3]; let total = 5usize.pow(nj as u32);
        for _ in 0..total { let mut a = [0.0f32; 3]; for i in 0..nj { a[i] = coarse[idx[i]]; }
            let e = self.energy(bi, s, g, a); evals += 1.0; if e < best.1 { best = (a, e); }
            let mut c = 0; loop { idx[c] += 1; if idx[c] < 5 { break; } idx[c] = 0; c += 1; if c >= nj { break; } } }
        let base = best.0; let mut idx = [0usize; 3]; let total = 3usize.pow(nj as u32);
        for _ in 0..total { let mut a = [0.0f32; 3];
            for i in 0..nj { a[i] = (base[i] + [-0.75f32, 0.0, 0.75][idx[i]]).clamp(-UMAX, UMAX); }
            let e = self.energy(bi, s, g, a); evals += 1.0; if e < best.1 { best = (a, e); }
            let mut c = 0; loop { idx[c] += 1; if idx[c] < 3 { break; } idx[c] = 0; c += 1; if c >= nj { break; } } }
        (best.0, evals)
    }
    // L4 genetic tool: seeded ES minimizing E — deterministic given (state hash, seed)
    fn genetic(&self, bi: usize, s: [f32; 6], g: [f32; 3], warm: [f32; 3], seed: u32) -> ([f32; 3], f32) {
        let nj = NJ[bi]; let (pop, gens) = (16u32, 8u32); let mut evals = 0.0f32;
        let mut center = warm; let mut best = (warm, self.energy(bi, s, g, warm)); evals += 1.0;
        for gen in 0..gens { let sigma = 1.5 * (0.6f32).powi(gen as i32);
            let mut gbest = best;
            for m in 0..pop { let mut a = [0.0f32; 3];
                for i in 0..nj { let r = u(seed.wrapping_mul(9973).wrapping_add(gen * 1000 + m * 10 + i as u32), 77) * 2.0 - 1.0;
                    a[i] = (center[i] + sigma * r).clamp(-UMAX, UMAX); }
                let e = self.energy(bi, s, g, a); evals += 1.0; if e < gbest.1 { gbest = (a, e); } }
            best = gbest; center = best.0; }
        (best.0, evals)
    }
    // the agency ladder: (action, level, flops)
    fn decide(&self, bi: usize, s: [f32; 6], g: [f32; 3], tau: f32, seed: u32) -> ([f32; 3], usize, f32) {
        let a1 = self.act_k(bi, s, g, 1); let e1 = self.energy(bi, s, g, a1);
        let mut fl = FLOW_FLOPS + POT_FLOPS;
        if e1 <= tau { return (a1, 1, fl); }
        let a2 = self.act_k(bi, s, g, 4); let e2 = self.energy(bi, s, g, a2);
        fl += 4.0 * FLOW_FLOPS + POT_FLOPS;
        if e2 <= tau { return (a2, 2, fl); }
        let (a3, ev3) = self.planner(bi, s, g); let e3 = self.energy(bi, s, g, a3);
        fl += (ev3 + 1.0) * POT_FLOPS;
        if e3 <= tau { return (a3, 3, fl); }
        let (a4, ev4) = self.genetic(bi, s, g, a3, seed); let e4 = self.energy(bi, s, g, a4);
        fl += (ev4 + 1.0) * POT_FLOPS;
        // execute argmin-E among all candidates — deterministic
        let cands = [(a1, e1), (a2, e2), (a3, e3), (a4, e4)];
        let mut b = cands[0]; for c in &cands[1..] { if c.1 < b.1 { b = *c; } }
        (b.0, 4, fl)
    }
}
fn main() {
    let t = load_st(&format!("{MDIR}/model.safetensors"));
    let fin = 12 + 3 + 1 + EMB; let pin = 12 + 3 + EMB;
    let g3 = t["flow.b3"].clone();
    let m = Efa1 { emb: t["body_embedding"].clone(),
        fw: (0..fin).map(|c| t[&format!("flow.in{}", c)].clone()).collect(), fb1: t["flow.b1"].clone(), fw2: t["flow.w2"].clone(), fb2: t["flow.b2"].clone(), fw3: t["flow.w3"].clone(), fb3: [g3[0], g3[1], g3[2]],
        pw: (0..pin).map(|c| t[&format!("potential.in{}", c)].clone()).collect(), pb1: t["potential.b1"].clone(), pw2: t["potential.w2"].clone(), pb2: t["potential.b2"].clone(), pw3: t["potential.w3"].clone(), pb3: t["potential.b3"][0] };
    println!("  EFA-1 agency loop — the energy decides when to think and when to reach for tools (shipped artifact)\n");
    // ── τ calibration: 95th percentile of E(s, flow-K1-action) over 2000 validation contexts per body ──
    let mut taus = [0.0f32; NB];
    for bi in 0..NB { let nj = NJ[bi]; let mut es: Vec<f32> = (0..2000u32).map(|k| {
            let mut s = [0.0f32; 6]; let mut g = [0.0f32; 3];
            for j in 0..nj { let ju = j as u32; s[j] = (u(k, 141 + ju) * 2.0 - 1.0) * PI; s[3 + j] = (u(k, 144 + ju) * 2.0 - 1.0) * 3.0; g[j] = (u(k, 147 + ju) * 2.0 - 1.0) * 1.0; }
            let a = m.act_k(bi, s, g, 1); m.energy(bi, s, g, a) }).collect();
        es.sort_by(|a, b| a.partial_cmp(b).unwrap()); taus[bi] = es[1899]; // 95th pct
        println!("  τ[{}-DOF] = {:+.3}  (95th pct of E(s, a_flow) over 2000 validation contexts)", nj, taus[bi]); }
    // ── evals: in-distribution (card goals) and OOD stress (goals outside the ±1.0 training band) ──
    let gt4 = [[0.8f32, -0.6, 0.5], [-0.7, 0.5, -0.6], [0.5, 0.9, -0.4], [-0.5, -0.8, 0.7]];
    println!("\n     eval            body   reach K1-only   reach AGENCY   esc L2/L3/L4 (% dec.)   mean kFLOP/dec (K1 = {:.0})", (FLOW_FLOPS + POT_FLOPS) / 1000.0);
    let mut aj: Vec<String> = vec![];
    for mode in 0..2 { let label = if mode == 0 { "in-dist " } else { "OOD-goal" };
        for bi in 0..NB { let nj = NJ[bi]; let nn = 60;
            let (mut r_k1, mut r_ag) = (0, 0); let (mut l2, mut l3, mut l4) = (0u64, 0u64, 0u64); let mut dec = 0u64; let mut flops = 0.0f64;
            for k in 0..nn { let sd = 900 + (bi * 100 + k) as u32; let mut s0 = [0.0f32; 6]; let mut g = [0.0f32; 3];
                for j in 0..nj { let ju = j as u32; s0[j] = (u(sd, 7 + ju) * 2.0 - 1.0) * PI;
                    g[j] = if mode == 0 { gt4[k % 4][j] } else {
                        let sgn = if u(sd, 60 + ju) < 0.5 { -1.0 } else { 1.0 }; sgn * (1.05 + u(sd, 63 + ju) * 0.30) }; }
                // plain K=1 baseline
                let mut s = s0; for _ in 0..300 { let a = m.act_k(bi, s, g, 1); s = step(nj, s, a); }
                if (0..nj).all(|i| wrap(s[i] - g[i]).abs() < 0.35 && s[3 + i].abs() < 0.7) { r_k1 += 1; }
                // full agency ladder
                let mut s = s0; for st in 0..300 { let (a, lv, fl) = m.decide(bi, s, g, taus[bi], sd * 1000 + st);
                    dec += 1; flops += fl as f64; match lv { 2 => l2 += 1, 3 => l3 += 1, 4 => l4 += 1, _ => {} }
                    s = step(nj, s, a); }
                if (0..nj).all(|i| wrap(s[i] - g[i]).abs() < 0.35 && s[3 + i].abs() < 0.7) { r_ag += 1; } }
            let (p2, p3, p4) = (l2 as f32 / dec as f32 * 100.0, l3 as f32 / dec as f32 * 100.0, l4 as f32 / dec as f32 * 100.0);
            let kfl = flops / dec as f64 / 1000.0;
            println!("     {}        {}-DOF     {:>3.0}%            {:>3.0}%          {:>4.1} /{:>4.1} /{:>4.1}           {:>6.1}",
                label, nj, r_k1 as f32 / nn as f32 * 100.0, r_ag as f32 / nn as f32 * 100.0, p2, p3, p4, kfl);
            aj.push(format!("{{\"eval\":\"{}\",\"body\":\"{}-DOF\",\"reach_K1_only\":{:.0},\"reach_agency\":{:.0},\"escalation_pct_L2_L3_L4\":[{:.1},{:.1},{:.1}],\"mean_kflop_per_decision\":{:.1}}}",
                label.trim(), nj, r_k1 as f32 / nn as f32 * 100.0, r_ag as f32 / nn as f32 * 100.0, p2, p3, p4, kfl)); }
    }
    // ── determinism of the FULL ladder (force all levels by τ=-inf so every tool runs) ──
    let s0 = [0.3f32, -0.2, 0.1, 0.4, -0.3, 0.2]; let g0 = [1.2f32, -1.1, 1.3];
    let mut det = true;
    for bi in 0..NB { let (a1, _, _) = m.decide(bi, s0, g0, f32::NEG_INFINITY, 42);
        let (a2, _, _) = m.decide(bi, s0, g0, f32::NEG_INFINITY, 42);
        if !a1.iter().zip(a2.iter()).all(|(x, y)| x.to_bits() == y.to_bits()) { det = false; } }
    println!("\n  determinism of the full ladder (flow K=4 + planner + seeded ES, τ=-∞ forces all levels): {}",
        if det { "bit-exact ✓" } else { "MISMATCH ✗" });
    // ── write the agency block into config.json (replace if present) ──
    let cfg = std::fs::read_to_string(format!("{MDIR}/config.json")).unwrap();
    let base = match cfg.find(",\n  \"agency\"") { Some(i) => { let tail = &cfg[i..]; let after = tail.find("]\n").map(|j| i + j + 2).unwrap_or(cfg.len());
            format!("{}{}", &cfg[..i], &cfg[after..]) }, None => cfg };
    let patched = base.trim_end().trim_end_matches('}').trim_end().trim_end_matches(',').to_string()
        + &format!(",\n  \"agency\": {{\n    \"policy\": \"L1 flow K=1 -> if E>tau: L2 flow K=4 -> if E>tau: L3 planner tool (two-stage argmin over the model's own potential) -> if E>tau: L4 seeded ES tool; execute argmin-E among candidates; all levels deterministic (bit-exact, {})\",\n    \"tau_per_body_95pct\": [{:.3}, {:.3}, {:.3}],\n    \"measured\": [{}]\n  }}\n}}\n",
        if det { "verified" } else { "FAILED" }, taus[0], taus[1], taus[2], aj.join(", "));
    std::fs::write(format!("{MDIR}/config.json"), patched).unwrap();
    println!("  agency policy + measurements written into {MDIR}/config.json");
    println!("  HONEST: τ is a global per-body scalar (contrastive E has state-dependent scale — a per-context gate is the finer instrument);");
    println!("  OOD tools inherit the potential's training domain — whatever the numbers above show, they are the measurement.");
}
