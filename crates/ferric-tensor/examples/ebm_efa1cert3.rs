//! EFA-1 certificate v3 — GROW the certified region outward from the proven core (the honest formulation).
//! The v1/v2 record, kept because it defines the method: v1 (small window, metric-field iteration) certified 72.3% on
//! 1-DOF (vs constant-P 31.6%) but drained to 0% on 2/3-DOF — the window is not forward-invariant and edge deaths
//! cascade backward through interpolation. v2 (full circle, periodic θ) drained to 0% EVERYWHERE — and had to: an
//! attracting fixed point on S¹ⁿ×Rⁿ forces a companion unstable set (separatrix) where NO metric contracts; iterating
//! death from those genuinely-dead nodes consumes every inflowing orbit. Both artifacts of the formulation.
//! v3 is the classic expanding-ROA construction, composed with the shipped constant-P certificate:
//!   CORE  = grid nodes inside the constant-P certified ball around x* (one-step contraction in P — cert1's object)
//!   GROW  = repeatedly add any node whose image cell lies ENTIRELY in the certified set and whose funnel metric
//!           M(x) = J(x)ᵀ M̃(f(x)) J(x) + I stays bounded (M̃ = multilinear interp; M ≡ P on the core)
//! Every certified node's orbit provably (grid-sampled) enters the contraction core with bounded expansion along the
//! way — deaths never propagate, growth stops at the separatrix on its own. Report: % of the PHYSICAL domain
//! (θ full circle × ω in the measured envelope) certified, vs the EMPIRICAL basin fraction (rollout-sampled) —
//! the gap is the method's conservatism, stated. 100% is topologically impossible; the empirical basin is the ceiling.
//! HONEST: grid-sampled J + multilinear interp (rigorous = interval bounds); one goal per body; coarse-ω 3-DOF grid.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_efa1cert3 --release`
use std::f32::consts::PI;
const H: usize = 128; const EMB: usize = 6; const DT: f32 = 0.05; const UMAX: f32 = 4.0; const CPL: f32 = 0.5;
const NB: usize = 3; const NJ: [usize; NB] = [1, 2, 3];
const MDIR: &str = "/Users/dcharlot/vibe-coding/efa/models/efa-1";
const CAP: f32 = 1e7;
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
struct Efa1 { emb: Vec<f32>, fw: Vec<Vec<f32>>, fb1: Vec<f32>, fw2: Vec<f32>, fb2: Vec<f32>, fw3: Vec<f32>, fb3: [f32; 3] }
impl Efa1 {
    fn act(&self, bi: usize, s: [f32; 6], g: [f32; 3]) -> [f32; 3] {
        let nj = NJ[bi]; let ff = feat12(nj, s, g); let fin = 12 + 3 + 1 + EMB; let mut f = vec![0.0f32; fin];
        for c in 0..12 { f[c] = ff[c]; } f[15] = 0.0; for c in 0..EMB { f[16 + c] = self.emb[bi * EMB + c]; }
        let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.fb1[j]; for c in 0..fin { z += f[c] * self.fw[c][j]; } h1[j] = z.max(0.0); }
        let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.fb2[j]; for k in 0..H { z += h1[k] * self.fw2[k * H + j]; } h2[j] = z.max(0.0); }
        let mut o = self.fb3; for j in 0..H { for c in 0..3 { o[c] += h2[j] * self.fw3[j * 3 + c]; } }
        let mut a = [0.0f32; 3]; for i in 0..nj { a[i] = o[i].clamp(-UMAX, UMAX); } a }
}
fn fcl(m: &Efa1, bi: usize, x: &[f32], g: [f32; 3]) -> Vec<f32> {
    let nj = NJ[bi]; let mut s = [0.0f32; 6]; for i in 0..nj { s[i] = x[i]; s[3 + i] = x[nj + i]; }
    let ns = step(nj, s, m.act(bi, s, g));
    let mut y = vec![0.0f32; 2 * nj]; for i in 0..nj { y[i] = ns[i]; y[nj + i] = ns[3 + i]; } y
}
fn matmul(a: &[f32], b: &[f32], d: usize) -> Vec<f32> { let mut c = vec![0.0f32; d * d];
    for i in 0..d { for k in 0..d { let aik = a[i * d + k]; for j in 0..d { c[i * d + j] += aik * b[k * d + j]; } } } c }
fn transpose(a: &[f32], d: usize) -> Vec<f32> { let mut t = vec![0.0f32; d * d]; for i in 0..d { for j in 0..d { t[j * d + i] = a[i * d + j]; } } t }
fn lammax(mm: &[f32], d: usize) -> f32 {
    let mut v = vec![1.0f32; d]; let mut lam = 0.0;
    for _ in 0..25 { let mut w = vec![0.0f32; d]; for r in 0..d { for c in 0..d { w[r] += mm[r * d + c] * v[c]; } }
        let n = w.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12); lam = n; for c in 0..d { v[c] = w[c] / n; } }
    lam
}
#[derive(Clone)]
struct Dim { n: usize, lo: f32, span: f32, periodic: bool }
impl Dim {
    fn node(&self, i: usize) -> f32 { if self.periodic { self.lo + self.span * i as f32 / self.n as f32 }
        else { self.lo + self.span * i as f32 / (self.n as f32 - 1.0) } }
    fn cell(&self, x: f32) -> Option<(usize, f32)> {
        if self.periodic { let mut gc = (wrap(x) - self.lo) / self.span * self.n as f32;
            if gc < 0.0 { gc += self.n as f32; } if gc >= self.n as f32 { gc -= self.n as f32; }
            let b = gc.floor() as usize % self.n; Some((b, gc - gc.floor()))
        } else { let gc = (x - self.lo) / self.span * (self.n as f32 - 1.0);
            if gc < 0.0 || gc > self.n as f32 - 1.0 { return None; }
            let b = (gc.floor() as usize).min(self.n - 2); Some((b, gc - b as f32)) } }
    fn corner(&self, b: usize, hi: usize) -> usize { if self.periodic { (b + hi) % self.n } else { b + hi } }
}
fn main() {
    let t = load_st(&format!("{MDIR}/model.safetensors"));
    let fin = 12 + 3 + 1 + EMB; let g3 = t["flow.b3"].clone();
    let m = Efa1 { emb: t["body_embedding"].clone(),
        fw: (0..fin).map(|c| t[&format!("flow.in{}", c)].clone()).collect(), fb1: t["flow.b1"].clone(), fw2: t["flow.w2"].clone(), fb2: t["flow.b2"].clone(), fw3: t["flow.w3"].clone(), fb3: [g3[0], g3[1], g3[2]] };
    println!("  EFA-1 certificate v3 — certified region GROWN outward from the constant-P core (full physical domain)\n");
    let gt = [0.8f32, -0.6, 0.5];
    let th_pts = [0usize, 41, 21, 11]; let om_pts = [0usize, 33, 13, 7];
    let mut certjson: Vec<String> = vec![];
    for bi in 0..NB { let nj = NJ[bi]; let d = 2 * nj; let mut g = [0.0f32; 3]; for j in 0..nj { g[j] = gt[j]; }
        // attractor + measured ω envelope
        let mut xstar = vec![0.0f32; d]; for j in 0..nj { xstar[j] = g[j]; }
        for _ in 0..800 { xstar = fcl(&m, bi, &xstar, g); }
        let mut om_env = 0.0f32;
        for k in 0..60u32 { let mut x = vec![0.0f32; d];
            for j in 0..nj { x[j] = (u(k, 7 + j as u32) * 2.0 - 1.0) * PI; x[nj + j] = (u(k, 17 + j as u32) * 2.0 - 1.0) * 1.5; }
            for _ in 0..300 { x = fcl(&m, bi, &x, g); for j in 0..nj { om_env = om_env.max(x[nj + j].abs()); } } }
        let om_b = (om_env * 1.3).ceil();
        let dims: Vec<Dim> = (0..d).map(|c| if c < nj { Dim { n: th_pts[nj], lo: -PI, span: 2.0 * PI, periodic: true } }
            else { Dim { n: om_pts[nj], lo: -om_b, span: 2.0 * om_b, periodic: false } }).collect();
        let total: usize = dims.iter().map(|dm| dm.n).product();
        let mut stride = vec![1usize; d]; for c in 1..d { stride[c] = stride[c - 1] * dims[c - 1].n; }
        let node_x = |n: usize| -> Vec<f32> { (0..d).map(|c| dims[c].node(n / stride[c] % dims[c].n)).collect() };
        // ---- constant-P core at x* (cert1's construction) ----
        let jac_at = |x: &[f32]| -> Vec<f32> { let hh = 1e-3; let mut jac = vec![0.0f32; d * d];
            for c in 0..d { let mut xp = x.to_vec(); xp[c] += hh; let mut xm = x.to_vec(); xm[c] -= hh;
                let (fp, fm) = (fcl(&m, bi, &xp, g), fcl(&m, bi, &xm, g));
                for r in 0..d { let mut diff = fp[r] - fm[r]; if r < nj { diff = wrap(diff); } jac[r * d + c] = diff / (2.0 * hh); } }
            jac };
        let a = jac_at(&xstar);
        let mut p = vec![0.0f32; d * d]; for i in 0..d { p[i * d + i] = 1.0; }
        let mut term = a.clone();
        for _ in 0..400 { let tt = matmul(&transpose(&term, d), &term, d);
            let tn: f32 = tt.iter().map(|x| x * x).sum::<f32>().sqrt(); for i in 0..d * d { p[i] += tt[i]; }
            if tn < 1e-7 { break; } term = matmul(&term, &a, d); }
        let pnorm = |x: &[f32]| -> f32 { let mut dx = vec![0.0f32; d];
            for c in 0..d { let mut e = x[c] - xstar[c]; if c < nj { e = wrap(e); } dx[c] = e; }
            let mut q = 0.0; for i in 0..d { for j in 0..d { q += dx[i] * p[i * d + j] * dx[j]; } } q.max(0.0).sqrt() };
        // core ball radius: from cert1's shipped numbers (P-norm certified balls per body)
        let core_r = [0.757f32, 0.42, 0.64][bi];
        // ---- per-node J, f(x) ----
        println!("  [{}-DOF] ω envelope {:.2} → ±{:.0} · grid {} nodes · computing J field…", nj, om_env, om_b, total);
        let mut js: Vec<f32> = vec![0.0; total * d * d]; let mut fx: Vec<f32> = vec![0.0; total * d];
        for n in 0..total { let x = node_x(n);
            let jj = jac_at(&x); js[n * d * d..(n + 1) * d * d].copy_from_slice(&jj);
            let y = fcl(&m, bi, &x, g); for c in 0..d { fx[n * d + c] = y[c]; } }
        // ---- grow from the core ----
        let mut state: Vec<u8> = vec![0; total];                       // 0 = unknown, 1 = certified
        let mut mf: Vec<f32> = vec![0.0; total * d * d];
        let mut core_n = 0usize;
        for n in 0..total { let x = node_x(n);
            if pnorm(&x) < core_r { state[n] = 1; mf[n * d * d..(n + 1) * d * d].copy_from_slice(&p); core_n += 1; } }
        let mut certified = core_n;
        for pass in 0..500 {
            let mut added = 0usize;
            for n in 0..total { if state[n] != 0 { continue; }
                let ok = (|| -> Option<Vec<f32>> {
                    let mut base = vec![0usize; d]; let mut frac = vec![0.0f32; d];
                    for c in 0..d { let (b, f) = dims[c].cell(fx[n * d + c])?; base[c] = b; frac[c] = f; }
                    let mut mi = vec![0.0f32; d * d];
                    for corner in 0..(1usize << d) {
                        let mut w = 1.0f32; let mut off = 0usize;
                        for c in 0..d { let hi = (corner >> c) & 1;
                            w *= if hi == 1 { frac[c] } else { 1.0 - frac[c] };
                            off += dims[c].corner(base[c], hi) * stride[c]; }
                        if w < 1e-7 { continue; }
                        if state[off] != 1 { return None; }            // image cell must lie entirely in certified set
                        for k in 0..d * d { mi[k] += w * mf[off * d * d + k]; } }
                    let j = &js[n * d * d..(n + 1) * d * d];
                    let mut jm = vec![0.0f32; d * d];
                    for r in 0..d { for c in 0..d { let mut z = 0.0; for k in 0..d { z += j[k * d + r] * mi[k * d + c]; } jm[r * d + c] = z; } }
                    let mut out = vec![0.0f32; d * d];
                    for r in 0..d { for c in 0..d { let mut z = 0.0; for k in 0..d { z += jm[r * d + k] * j[k * d + c]; } out[r * d + c] = z + if r == c { 1.0 } else { 0.0 }; } }
                    if lammax(&out, d) > CAP { return None; }
                    Some(out)
                })();
                if let Some(out) = ok { mf[n * d * d..(n + 1) * d * d].copy_from_slice(&out); state[n] = 1; added += 1; } }
            certified += added;
            if pass % 50 == 49 { println!("       pass {:>3}: certified {} / {} (+{} this pass)", pass + 1, certified, total, added); }
            if added == 0 { println!("       growth stopped at pass {} — certified {} / {}", pass + 1, certified, total); break; }
        }
        let frac_pct = certified as f32 / total as f32 * 100.0;
        // empirical basin fraction (the ceiling): rollout from a node sample across the WHOLE domain
        let (mut conv, mut tried) = (0, 0); let pick = (total / 400).max(1);
        for n in (0..total).step_by(pick) { tried += 1;
            let mut xx = node_x(n); for _ in 0..600 { xx = fcl(&m, bi, &xx, g); }
            if (0..nj).all(|i| wrap(xx[i] - g[i]).abs() < 0.35 && xx[nj + i].abs() < 0.7) { conv += 1; } }
        let basin = conv as f32 / tried as f32 * 100.0;
        // empirical from CERTIFIED nodes (must be ~100 if the certificate is sound)
        let (mut cconv, mut ctried) = (0, 0);
        for n in (0..total).step_by(pick) { if state[n] != 1 || ctried >= 40 { continue; } ctried += 1;
            let mut xx = node_x(n); for _ in 0..600 { xx = fcl(&m, bi, &xx, g); }
            if (0..nj).all(|i| wrap(xx[i] - g[i]).abs() < 0.35 && xx[nj + i].abs() < 0.7) { cconv += 1; } }
        let cemp = if ctried > 0 { cconv as f32 / ctried as f32 * 100.0 } else { 0.0 };
        println!("     ⇒ {}-DOF: certified {:.1}% of S¹^{}×[−{:.0},{:.0}]^{} (core {} nodes) · EMPIRICAL basin {:.1}% · certified-node rollouts {:.0}% converge\n",
            nj, frac_pct, nj, om_b, om_b, nj, core_n, basin, cemp);
        certjson.push(format!("{{\"body\":\"{}-DOF\",\"domain\":\"S1^{} x [-{:.0},{:.0}]^{}\",\"grid_nodes\":{},\"core_nodes_constant_P\":{},\"certified_fraction\":{:.1},\"empirical_basin_fraction\":{:.1},\"certified_rollout_convergence\":{:.0},\"method\":\"expanding-ROA: constant-P certified core + growth (node joins when its image cell lies entirely in the certified set and funnel metric M=J'M(f)J+I stays bounded); theta periodic; grid-sampled + multilinear interp (rigorous = interval); 100% impossible topologically (separatrix)\"}}",
            nj, nj, om_b, om_b, nj, total, core_n, frac_pct, basin, cemp));
    }
    let cfg = std::fs::read_to_string(format!("{MDIR}/config.json")).unwrap();
    let base = match cfg.find(",\n  \"certificates_M_field\"") { Some(i) => { let tail = &cfg[i..]; let after = tail.find("]\n").map(|j| i + j + 2).unwrap_or(cfg.len()); format!("{}{}", &cfg[..i], &cfg[after..]) }, None => cfg };
    let patched = base.trim_end().trim_end_matches('}').trim_end().trim_end_matches(',').to_string()
        + &format!(",\n  \"certificates_M_field\": [{}]\n}}\n", certjson.join(", "));
    std::fs::write(format!("{MDIR}/config.json"), patched).unwrap();
    println!("  expanding-ROA certificate written into {MDIR}/config.json (replacing the v2 block).");
    println!("  HONEST: grid-sampled; growth conservatism = certified vs empirical-basin gap, printed above; one goal per body.");
}
