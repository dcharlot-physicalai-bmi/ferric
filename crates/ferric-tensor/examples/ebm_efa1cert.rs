//! EFA-1 certificates — computed ON THE SHIPPED ARTIFACT. Loads models/efa-1/model.safetensors (the exact released
//! weights) and certifies the closed loop per body, in three honest stages:
//!   0. HARNESS VALIDATION — reproduce the shipped card's reach eval (same seeds/criterion) from this CPU
//!      reconstruction; if it doesn't match the card, no certificate number is trustworthy.
//!   1. ATTRACTOR — the loop converges to a neighborhood, not the exact goal point (card criterion = 0.35 rad/0.7 rad/s);
//!      find the true fixed point x* per (body, goal) and its residual ‖x*−(g,0)‖.
//!   2. CONTRACTION IN A LYAPUNOV METRIC — identity-metric one-step contraction FAILS here (first run of this file:
//!      24.5/2.6/0.1% — kept as the recorded negative); the correct lens is σmax(P^½ J P^-½) < 1 where P is the
//!      Lyapunov metric of the closed-loop linearization A at x* (P ≈ Σ (Aᵀ)ᵏAᵏ, so AᵀPA−P=−I). Grid-certify the box,
//!      report the certified ball around x*, and empirically check convergence (card criterion) from inside it.
//! Results REPLACE the "certificates" field in the artifact's config.json — the certificate ships WITH the weights.
//! HONEST: grid-sampled (rigorous route = interval/CROWN), one representative goal certified per body (residuals
//! measured on all four card goals), FD Jacobians.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_efa1cert --release`
use std::f32::consts::PI;
const H: usize = 128; const EMB: usize = 6; const DT: f32 = 0.05; const UMAX: f32 = 4.0; const CPL: f32 = 0.5;
const NB: usize = 3; const NJ: [usize; NB] = [1, 2, 3];
const MDIR: &str = "/Users/dcharlot/vibe-coding/efa/models/efa-1";
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
    let raw = std::fs::read(path).expect("model.safetensors not found — ship EFA-1 first");
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
fn jac_at(m: &Efa1, bi: usize, x: &[f32], g: [f32; 3]) -> Vec<f32> {
    let nj = NJ[bi]; let d = 2 * nj; let h = 1e-3; let mut jac = vec![0.0f32; d * d];
    for c in 0..d { let mut xp = x.to_vec(); xp[c] += h; let mut xm = x.to_vec(); xm[c] -= h;
        let (fp, fm) = (fcl(m, bi, &xp, g), fcl(m, bi, &xm, g));
        for r in 0..d { let mut diff = fp[r] - fm[r]; if r < nj { diff = wrap(diff); } jac[r * d + c] = diff / (2.0 * h); } }
    jac
}
// dense small-matrix helpers (d ≤ 6)
fn matmul(a: &[f32], b: &[f32], d: usize) -> Vec<f32> { let mut c = vec![0.0f32; d * d];
    for i in 0..d { for k in 0..d { let aik = a[i * d + k]; for j in 0..d { c[i * d + j] += aik * b[k * d + j]; } } } c }
fn transpose(a: &[f32], d: usize) -> Vec<f32> { let mut t = vec![0.0f32; d * d]; for i in 0..d { for j in 0..d { t[j * d + i] = a[i * d + j]; } } t }
// cyclic Jacobi eigendecomposition of symmetric a (d≤6): returns (eigvals, eigvecs row-major V with columns = vectors)
fn jacobi_eig(a0: &[f32], d: usize) -> (Vec<f32>, Vec<f32>) {
    let mut a = a0.to_vec(); let mut v = vec![0.0f32; d * d]; for i in 0..d { v[i * d + i] = 1.0; }
    for _ in 0..60 { let mut off = 0.0f32; for i in 0..d { for j in 0..d { if i != j { off += a[i * d + j] * a[i * d + j]; } } }
        if off < 1e-12 { break; }
        for p in 0..d { for q in (p + 1)..d { let apq = a[p * d + q]; if apq.abs() < 1e-9 { continue; }
            let theta = (a[q * d + q] - a[p * d + p]) / (2.0 * apq);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt(); let s = t * c;
            for k in 0..d { let (akp, akq) = (a[k * d + p], a[k * d + q]);
                a[k * d + p] = c * akp - s * akq; a[k * d + q] = s * akp + c * akq; }
            for k in 0..d { let (apk, aqk) = (a[p * d + k], a[q * d + k]);
                a[p * d + k] = c * apk - s * aqk; a[q * d + k] = s * apk + c * aqk; }
            for k in 0..d { let (vkp, vkq) = (v[k * d + p], v[k * d + q]);
                v[k * d + p] = c * vkp - s * vkq; v[k * d + q] = s * vkp + c * vkq; } } } }
    let vals: Vec<f32> = (0..d).map(|i| a[i * d + i]).collect(); (vals, v)
}
// σmax(A) for general small A via power iteration on AᵀA
fn sigmax_mat(a: &[f32], d: usize) -> f32 {
    let mut v = vec![1.0f32; d]; let mut lam = 0.0;
    for _ in 0..40 { let mut w = vec![0.0f32; d]; for r in 0..d { for c in 0..d { w[r] += a[r * d + c] * v[c]; } }
        let mut z = vec![0.0f32; d]; for c in 0..d { for r in 0..d { z[c] += a[r * d + c] * w[r]; } }
        let n = z.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12); lam = n; for c in 0..d { v[c] = z[c] / n; } }
    lam.sqrt()
}
fn main() {
    let t = load_st(&format!("{MDIR}/model.safetensors"));
    let fin = 12 + 3 + 1 + EMB;
    let g3 = t["flow.b3"].clone();
    let m = Efa1 { emb: t["body_embedding"].clone(),
        fw: (0..fin).map(|c| t[&format!("flow.in{}", c)].clone()).collect(), fb1: t["flow.b1"].clone(), fw2: t["flow.w2"].clone(), fb2: t["flow.b2"].clone(), fw3: t["flow.w3"].clone(), fb3: [g3[0], g3[1], g3[2]] };
    println!("  EFA-1 certificates — the SHIPPED closed loop (loaded from {MDIR})\n");
    // ── Stage 0: harness validation — reproduce the shipped card's reach eval exactly ──
    let gt4 = [[0.8f32, -0.6, 0.5], [-0.7, 0.5, -0.6], [0.5, 0.9, -0.4], [-0.5, -0.8, 0.7]];
    print!("  [0] harness validation vs shipped card (must be 100/100/100):  reach =");
    let mut valid = true;
    for bi in 0..NB { let nj = NJ[bi]; let (mut reach, nn) = (0, 60);
        for k in 0..nn { let sd = 900 + (bi * 100 + k) as u32; let mut s = [0.0f32; 6]; let mut g = [0.0f32; 3];
            let gtk = gt4[k % 4]; for j in 0..nj { s[j] = (u(sd, 7 + j as u32) * 2.0 - 1.0) * PI; g[j] = gtk[j]; }
            for _ in 0..300 { let a = m.act(bi, s, g); s = step(nj, s, a); }
            if (0..nj).all(|i| wrap(s[i] - g[i]).abs() < 0.35 && s[3 + i].abs() < 0.7) { reach += 1; } }
        let r = reach as f32 / nn as f32 * 100.0; print!(" {:.0}%", r); if r < 100.0 { valid = false; } }
    println!("{}", if valid { "  ✓ reconstruction matches the artifact" } else { "  ✗ MISMATCH — certificates would be meaningless; aborting" });
    if !valid { return; }
    println!("\n     body   fp residual ‖x*−(g,0)‖ (4 goals)   ρ(A) at x*    P-metric certified   ball r (P-norm)   empirical");
    let mut certjson: Vec<String> = vec![];
    for bi in 0..NB { let nj = NJ[bi]; let d = 2 * nj;
        // ── Stage 1: attractor per card goal — roll long from the goal, then polish; measure residual to (g,0) ──
        let mut residuals = vec![]; let mut xstar_rep = vec![0.0f32; d]; let mut fp_gap_rep = 0.0;
        for (gi, gtk) in gt4.iter().enumerate() { let mut g = [0.0f32; 3]; for j in 0..nj { g[j] = gtk[j]; }
            let mut x = vec![0.0f32; d]; for j in 0..nj { x[j] = g[j]; }
            for _ in 0..800 { x = fcl(&m, bi, &x, g); }
            let fx = fcl(&m, bi, &x, g);
            let fp_gap: f32 = (0..d).map(|c| { let mut e = fx[c] - x[c]; if c < nj { e = wrap(e); } e * e }).sum::<f32>().sqrt();
            let res: f32 = (0..d).map(|c| if c < nj { wrap(x[c] - g[c]).powi(2) } else { x[c] * x[c] }).sum::<f32>().sqrt();
            residuals.push(res);
            if gi == 0 { xstar_rep = x.clone(); fp_gap_rep = fp_gap; } }
        let res_max = residuals.iter().cloned().fold(0.0f32, f32::max);
        // ── Stage 2: Lyapunov metric P at x* (representative goal), then grid-certify σmax(P^½JP^-½)<1 ──
        let mut g = [0.0f32; 3]; for j in 0..nj { g[j] = gt4[0][j]; }
        let a = jac_at(&m, bi, &xstar_rep, g);
        // ρ(A) estimate: ‖Aᵏ‖^(1/k) at k=60
        let mut ak = a.clone(); for _ in 0..59 { ak = matmul(&ak, &a, d); }
        let rho = sigmax_mat(&ak, d).powf(1.0 / 60.0);
        // P = Σ (Aᵀ)ᵏAᵏ truncated — converges iff ρ(A)<1
        let mut p = vec![0.0f32; d * d]; for i in 0..d { p[i * d + i] = 1.0; }
        let mut term = a.clone();
        for _ in 0..400 { let tt = matmul(&transpose(&term, d), &term, d);
            let tn: f32 = tt.iter().map(|x| x * x).sum::<f32>().sqrt(); for i in 0..d * d { p[i] += tt[i]; }
            if tn < 1e-7 { break; } term = matmul(&term, &a, d); }
        // P^½, P^-½ via Jacobi
        let (ev, vv) = jacobi_eig(&p, d);
        let mut ph = vec![0.0f32; d * d]; let mut phi = vec![0.0f32; d * d];
        for i in 0..d { for j in 0..d { for k in 0..d { let lk = ev[k].max(1e-9);
            ph[i * d + j] += vv[i * d + k] * lk.sqrt() * vv[j * d + k];
            phi[i * d + j] += vv[i * d + k] / lk.sqrt() * vv[j * d + k]; } } }
        let sig_p = |jm: &[f32]| -> f32 { sigmax_mat(&matmul(&ph, &matmul(jm, &phi, d), d), d) };
        let sig_at_xstar = sig_p(&a);
        // grid over box around x*: θ_i ∈ x*_i±1.2, ω_i ∈ x*_{nj+i}±1.5
        let ppd = [0usize, 41, 11, 7][nj]; let total = ppd.pow(d as u32);
        let (mut ok, mut tot) = (0usize, 0usize); let mut ball = f32::MAX;
        let pnorm = |x: &[f32]| -> f32 { let mut dxv = vec![0.0f32; d];
            for c in 0..d { let mut e = x[c] - xstar_rep[c]; if c < nj { e = wrap(e); } dxv[c] = e; }
            let mut q = 0.0; for i in 0..d { for j in 0..d { q += dxv[i] * p[i * d + j] * dxv[j]; } } q.max(0.0).sqrt() };
        let mut idx = vec![0usize; d];
        for _ in 0..total {
            let mut x = vec![0.0f32; d];
            for c in 0..d { let f = idx[c] as f32 / (ppd - 1) as f32 * 2.0 - 1.0;
                x[c] = xstar_rep[c] + if c < nj { 1.2 * f } else { 1.5 * f }; }
            let sm = sig_p(&jac_at(&m, bi, &x, g)); tot += 1;
            if sm < 1.0 { ok += 1; } else { let r = pnorm(&x); if r < ball { ball = r; } }
            let mut c = 0; loop { idx[c] += 1; if idx[c] < ppd { break; } idx[c] = 0; c += 1; if c >= d { break; } } }
        if ball == f32::MAX { ball = f32::INFINITY; }
        // empirical: card-criterion convergence from starts inside the certified P-ball
        let (mut conv, nn) = (0, 40);
        for k in 0..nn { let mut x = vec![0.0f32; d];
            for c in 0..d { let f = u(k as u32, 91 + c as u32) * 2.0 - 1.0;
                x[c] = xstar_rep[c] + if c < nj { 1.2 * f } else { 1.5 * f }; }
            // pull the sample inside the certified ball — geometric shrink toward x* terminates unconditionally
            while pnorm(&x) >= ball * 0.95 { for c in 0..d { x[c] = xstar_rep[c] + (x[c] - xstar_rep[c]) * 0.7; } }
            for _ in 0..300 { x = fcl(&m, bi, &x, g); }
            if (0..nj).all(|i| wrap(x[i] - g[i]).abs() < 0.35 && x[nj + i].abs() < 0.7) { conv += 1; } }
        let frac = ok as f32 / tot as f32 * 100.0;
        let res_str = residuals.iter().map(|r| format!("{:.2}", r)).collect::<Vec<_>>().join(",");
        println!("     {}-DOF   [{}] max {:.2}          {:.3}          {:>5.1}% of box       {:.2}            {:>3.0}%",
            nj, res_str, res_max, rho, frac, if ball.is_finite() { ball } else { -1.0 }, conv as f32 / nn as f32 * 100.0);
        println!("             fp-gap ‖f(x*)−x*‖={:.1e} · σ_P(A(x*))={:.3} · grid {} pts", fp_gap_rep, sig_at_xstar, tot);
        certjson.push(format!("{{\"body\":\"{}-DOF\",\"goal\":{:?},\"attractor_residual_max\":{:.3},\"spectral_radius_at_attractor\":{:.3},\"grid_points\":{},\"contraction_fraction_P_metric\":{:.1},\"certified_ball_P_norm\":{:.3},\"empirical_convergence_card_criterion\":{:.0},\"method\":\"one-step contraction sigma_max(P^0.5 J P^-0.5)<1 in the Lyapunov metric of the closed-loop linearization at the measured attractor x*; FD Jacobians, grid-sampled (rigorous route = interval/CROWN); identity-metric contraction fails here (24.5/2.6/0.1% — recorded negative)\"}}",
            nj, &gt4[0][..nj], res_max, rho, tot, frac, if ball.is_finite() { ball } else { -1.0 }, conv as f32 / nn as f32 * 100.0));
    }
    // REPLACE the certificates field in config.json (first run wrote the identity-metric negative)
    let cfg = std::fs::read_to_string(format!("{MDIR}/config.json")).unwrap();
    let base = match cfg.find(",\n  \"certificates\"") { Some(i) => format!("{}\n}}\n", &cfg[..i]), None => cfg.trim_end().trim_end_matches('}').to_string() + "}" };
    let patched = base.trim_end().trim_end_matches('}').to_string() + &format!(",\n  \"certificates\": [{}]\n}}\n", certjson.join(", "));
    std::fs::write(format!("{MDIR}/config.json"), patched).unwrap();
    println!("\n  certificates written into {MDIR}/config.json — the certificate ships WITH the weights.");
    println!("  HONEST: grid-sampled, one representative goal certified per body (residuals on all 4 card goals); not an SMT/interval proof.");
}
