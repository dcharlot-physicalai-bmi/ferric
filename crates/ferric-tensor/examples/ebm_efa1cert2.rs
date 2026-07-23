//! EFA-1 certificate upgrade — STATE-DEPENDENT contraction metric field, computed on the shipped artifact.
//! v1 of this experiment validated the method on 1-DOF (72.3% certified vs constant-P's 31.6%) and produced an
//! ARTIFACT on 2/3-DOF (0.0%): the small window wasn't forward-invariant — transients exit through the ω edges, and
//! the conservative interpolation rule cascades edge deaths inward until the window drains. Recorded; the fix is the
//! stronger formulation implemented here: certify over a domain the dynamics cannot leave —
//!   · θ dims on the FULL CIRCLE with periodic interpolation (no θ boundary exists at all)
//!   · ω dims bounded by the loop's MEASURED transient envelope × margin (adaptive, printed)
//! Method (unchanged, sound): Lyapunov operator fixed-point iteration on the grid,
//!   M_{t+1}(x) = J(x)ᵀ M_t(f(x)) J(x) + I,  M_0 = I  — monotone; wherever bounded,
//!   M − JᵀM(f)J = I ≻ 0, i.e. one-step contraction with margin in the loop's own metric field, and the bounded set
//! is a region-of-attraction estimate — now stated as % OF THE PHYSICAL STATE SPACE S¹ⁿ×[−Ω,Ω]ⁿ.
//! HONEST: grid-sampled Jacobians + multilinear interpolation (rigorous = interval bounds); one goal per body;
//! ω-edge exits still drop nodes (rare with the measured envelope, disclosed by the certified fraction itself).
//!
//! Run: `cargo run -p ferric-tensor --example ebm_efa1cert2 --release`
use std::f32::consts::PI;
const H: usize = 128; const EMB: usize = 6; const DT: f32 = 0.05; const UMAX: f32 = 4.0; const CPL: f32 = 0.5;
const NB: usize = 3; const NJ: [usize; NB] = [1, 2, 3];
const MDIR: &str = "/Users/dcharlot/vibe-coding/efa/models/efa-1";
const CAP: f32 = 1e6; const ITERS: usize = 200;
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
fn lammax(mm: &[f32], d: usize) -> f32 {
    let mut v = vec![1.0f32; d]; let mut lam = 0.0;
    for _ in 0..25 { let mut w = vec![0.0f32; d]; for r in 0..d { for c in 0..d { w[r] += mm[r * d + c] * v[c]; } }
        let n = w.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12); lam = n; for c in 0..d { v[c] = w[c] / n; } }
    lam
}
// per-dim grid descriptor: θ dims periodic over the full circle (n cells), ω dims clamped (n−1 cells)
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
    println!("  EFA-1 certificate — state-dependent metric field over the PHYSICAL domain (θ = full circle, periodic)\n");
    let gt = [0.8f32, -0.6, 0.5];
    let th_pts = [0usize, 41, 21, 11]; let om_pts = [0usize, 33, 13, 7];
    let mut certjson: Vec<String> = vec![];
    for bi in 0..NB { let nj = NJ[bi]; let d = 2 * nj; let mut g = [0.0f32; 3]; for j in 0..nj { g[j] = gt[j]; }
        // attractor + MEASURED transient ω envelope from full-circle starts (the invariant bound, adaptive)
        let mut xstar = vec![0.0f32; d]; for j in 0..nj { xstar[j] = g[j]; }
        for _ in 0..800 { xstar = fcl(&m, bi, &xstar, g); }
        let mut om_env = 0.0f32;
        for k in 0..60u32 { let mut x = vec![0.0f32; d];
            for j in 0..nj { x[j] = (u(k, 7 + j as u32) * 2.0 - 1.0) * PI; x[nj + j] = (u(k, 17 + j as u32) * 2.0 - 1.0) * 1.5; }
            for _ in 0..300 { x = fcl(&m, bi, &x, g); for j in 0..nj { om_env = om_env.max(x[nj + j].abs()); } } }
        let om_b = (om_env * 1.3).ceil();
        // grid: θ full circle (periodic), ω ∈ ±om_b (clamped)
        let dims: Vec<Dim> = (0..d).map(|c| if c < nj { Dim { n: th_pts[nj], lo: -PI, span: 2.0 * PI, periodic: true } }
            else { Dim { n: om_pts[nj], lo: -om_b, span: 2.0 * om_b, periodic: false } }).collect();
        let total: usize = dims.iter().map(|dm| dm.n).product();
        let mut stride = vec![1usize; d]; for c in 1..d { stride[c] = stride[c - 1] * dims[c - 1].n; }
        let node_x = |n: usize| -> Vec<f32> { (0..d).map(|c| dims[c].node(n / stride[c] % dims[c].n)).collect() };
        println!("  [{}-DOF] measured ω envelope {:.2} → bound ±{:.0} · grid {} nodes ({}ᶿ×{}ʷ) — computing J field…",
            nj, om_env, om_b, total, th_pts[nj], om_pts[nj]);
        // precompute J(x) and f(x) per node
        let mut js: Vec<f32> = vec![0.0; total * d * d]; let mut fx: Vec<f32> = vec![0.0; total * d];
        for n in 0..total { let x = node_x(n); let h = 1e-3;
            for c in 0..d { let mut xp = x.clone(); xp[c] += h; let mut xm = x.clone(); xm[c] -= h;
                let (fp, fm) = (fcl(&m, bi, &xp, g), fcl(&m, bi, &xm, g));
                for r in 0..d { let mut diff = fp[r] - fm[r]; if r < nj { diff = wrap(diff); } js[n * d * d + r * d + c] = diff / (2.0 * h); } }
            let y = fcl(&m, bi, &x, g); for c in 0..d { fx[n * d + c] = y[c]; } }
        // Lyapunov fixed-point iteration on the metric field
        let eye: Vec<f32> = { let mut e = vec![0.0f32; d * d]; for i in 0..d { e[i * d + i] = 1.0; } e };
        let mut mf: Vec<f32> = (0..total).flat_map(|_| eye.clone()).collect();
        let mut alive: Vec<bool> = vec![true; total];
        let mut certified = 0usize;
        for it in 0..ITERS {
            let mut nmf = vec![0.0f32; total * d * d]; let mut nalive = alive.clone(); let mut changed = 0usize;
            for n in 0..total { if !alive[n] { continue; }
                let ok = (|| -> Option<Vec<f32>> {
                    // locate image cell (θ dims periodic — always inside; ω dims may exit)
                    let mut base = vec![0usize; d]; let mut frac = vec![0.0f32; d];
                    for c in 0..d { let (b, f) = dims[c].cell(fx[n * d + c])?; base[c] = b; frac[c] = f; }
                    // multilinear interp of M at the image
                    let mut mi = vec![0.0f32; d * d];
                    for corner in 0..(1usize << d) {
                        let mut w = 1.0f32; let mut off = 0usize;
                        for c in 0..d { let hi = (corner >> c) & 1;
                            w *= if hi == 1 { frac[c] } else { 1.0 - frac[c] };
                            off += dims[c].corner(base[c], hi) * stride[c]; }
                        if w < 1e-7 { continue; }
                        if !alive[off] { return None; }
                        for k in 0..d * d { mi[k] += w * mf[off * d * d + k]; } }
                    let j = &js[n * d * d..(n + 1) * d * d];
                    let mut jm = vec![0.0f32; d * d];
                    for r in 0..d { for c in 0..d { let mut z = 0.0; for k in 0..d { z += j[k * d + r] * mi[k * d + c]; } jm[r * d + c] = z; } }
                    let mut out = vec![0.0f32; d * d];
                    for r in 0..d { for c in 0..d { let mut z = 0.0; for k in 0..d { z += jm[r * d + k] * j[k * d + c]; } out[r * d + c] = z + if r == c { 1.0 } else { 0.0 }; } }
                    if lammax(&out, d) > CAP { return None; }
                    Some(out)
                })();
                match ok { Some(out) => { nmf[n * d * d..(n + 1) * d * d].copy_from_slice(&out); }
                           None => { nalive[n] = false; changed += 1; } } }
            mf = nmf; alive = nalive;
            certified = alive.iter().filter(|&&a| a).count();
            if it % 50 == 49 { println!("       iter {:>3}: certified {} / {} ({} dropped this sweep)", it + 1, certified, total, changed); }
            if changed == 0 && it > 30 { break; }
        }
        let frac_pct = certified as f32 / total as f32 * 100.0;
        // largest ball around x* fully certified (wrapped-θ Euclid)
        let mut ball = f32::MAX;
        for n in 0..total { if !alive[n] { let x = node_x(n);
            let r: f32 = (0..d).map(|c| { let mut e = x[c] - xstar[c]; if c < nj { e = wrap(e); } e * e }).sum::<f32>().sqrt();
            if r < ball { ball = r; } } }
        if ball == f32::MAX { ball = f32::INFINITY; }
        // empirical from certified nodes
        let (mut conv, mut tried) = (0, 0); let pick = (total / 300).max(1);
        for n in (0..total).step_by(pick) { if !alive[n] || tried >= 40 { continue; } tried += 1;
            let mut xx = node_x(n); for _ in 0..400 { xx = fcl(&m, bi, &xx, g); }
            if (0..nj).all(|i| wrap(xx[i] - g[i]).abs() < 0.35 && xx[nj + i].abs() < 0.7) { conv += 1; } }
        let emp = if tried > 0 { conv as f32 / tried as f32 * 100.0 } else { 0.0 };
        println!("     ⇒ {}-DOF: {:.1}% of the PHYSICAL domain S¹^{}×[−{:.0},{:.0}]^{} certified · ball r = {} · empirical {:.0}% ({} rollouts)\n",
            nj, frac_pct, nj, om_b, om_b, nj, if ball.is_finite() { format!("{:.2}", ball) } else { "all".into() }, emp, tried);
        certjson.push(format!("{{\"body\":\"{}-DOF\",\"domain\":\"S1^{} x [-{:.0},{:.0}]^{} (theta full circle periodic, omega = measured transient envelope x1.3)\",\"grid_nodes\":{},\"certified_fraction_of_physical_domain\":{:.1},\"ball_r\":{:.3},\"empirical_convergence\":{:.0},\"method\":\"state-dependent contraction metric via Lyapunov fixed-point iteration M <- J'M(f(x))J + I (monotone; bounded => one-step contraction with margin I in the loop's own metric field); periodic theta interpolation removes the theta boundary; grid-sampled + multilinear interp (rigorous route = interval bounds)\"}}",
            nj, nj, om_b, om_b, nj, total, frac_pct, if ball.is_finite() { ball } else { -1.0 }, emp));
    }
    let cfg = std::fs::read_to_string(format!("{MDIR}/config.json")).unwrap();
    let base = match cfg.find(",\n  \"certificates_M_field\"") { Some(i) => { let tail = &cfg[i..]; let after = tail.find("]\n").map(|j| i + j + 2).unwrap_or(cfg.len()); format!("{}{}", &cfg[..i], &cfg[after..]) }, None => cfg };
    let patched = base.trim_end().trim_end_matches('}').trim_end().trim_end_matches(',').to_string()
        + &format!(",\n  \"certificates_M_field\": [{}]\n}}\n", certjson.join(", "));
    std::fs::write(format!("{MDIR}/config.json"), patched).unwrap();
    println!("  state-dependent certificate (physical-domain formulation) written into {MDIR}/config.json.");
    println!("  HONEST: grid-sampled J + multilinear M interp; ω-edge exits drop nodes; one goal per body; v1 small-window");
    println!("  artifact (2/3-DOF 0% from non-invariant box + conservative cascade) recorded in the ledger.");
}
