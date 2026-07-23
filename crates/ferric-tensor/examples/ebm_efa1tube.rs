//! EFA-1 ORBIT-TUBE bounds — upgrading the funnel certificate from node-local to BETWEEN-NODE claims.
//! cert4 certified that every grid NODE's orbit enters the contraction core; it claimed nothing about the continuum
//! between nodes. This experiment closes that gap with the standard tube argument:
//!   around each node orbit x_{k+1} = f(x_k), a Euclidean ball of radius δ_k maps into a ball of radius
//!       δ_{k+1} = (σ(J(x_k)) + L·δ_k) · δ_k
//!   (mean-value bound: sup over the segment of ‖J‖ ≤ ‖J(x_k)‖ + L·δ_k, with L the Jacobian's Lipschitz constant),
//!   and the tube CERTIFIES when the whole ball lands inside the P-core:  pnorm(x_k) + √λmax(P)·δ_k ≤ r_core.
//! If every node certifies a tube of initial radius ≥ half its grid-cell diagonal, the balls cover the continuum and
//! the certificate holds for EVERY point of the domain, not just nodes. Where the achievable δ0* is smaller, the
//! deficit tells us exactly the grid spacing a full-coverage run needs — the compute job is priced, not hand-waved.
//! HONEST: L is ESTIMATED from sampled pairs (max over ~300 probes; not a proven global bound — interval arithmetic
//! on the network is the fully rigorous route, named); tubes computed on all nodes for 1-DOF, sampled nodes for
//! 2/3-DOF (cost, disclosed); representative goal (multi-goal breadth handled by ebm_efa1cert5).
//!
//! Run: `cargo run -p ferric-tensor --example ebm_efa1tube --release`
use std::f32::consts::PI;
const H: usize = 128; const EMB: usize = 6; const DT: f32 = 0.05; const UMAX: f32 = 4.0; const CPL: f32 = 0.5;
const NB: usize = 3; const NJ: [usize; NB] = [1, 2, 3];
const MDIR: &str = "/Users/dcharlot/vibe-coding/efa/models/efa-1";
const TMAX: usize = 600; const DCAP: f32 = 10.0;
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
fn sigmax_mat(a: &[f32], d: usize) -> f32 {
    let mut v = vec![1.0f32; d]; let mut lam = 0.0;
    for _ in 0..40 { let mut w = vec![0.0f32; d]; for r in 0..d { for c in 0..d { w[r] += a[r * d + c] * v[c]; } }
        let mut z = vec![0.0f32; d]; for c in 0..d { for r in 0..d { z[c] += a[r * d + c] * w[r]; } }
        let n = z.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12); lam = n; for c in 0..d { v[c] = z[c] / n; } }
    lam.sqrt()
}
fn lammax_sym(mm: &[f32], d: usize) -> f32 {
    let mut v = vec![1.0f32; d]; let mut lam = 0.0;
    for _ in 0..30 { let mut w = vec![0.0f32; d]; for r in 0..d { for c in 0..d { w[r] += mm[r * d + c] * v[c]; } }
        let n = w.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12); lam = n; for c in 0..d { v[c] = w[c] / n; } }
    lam
}
fn main() {
    let t = load_st(&format!("{MDIR}/model.safetensors"));
    let fin = 12 + 3 + 1 + EMB; let g3 = t["flow.b3"].clone();
    let m = Efa1 { emb: t["body_embedding"].clone(),
        fw: (0..fin).map(|c| t[&format!("flow.in{}", c)].clone()).collect(), fb1: t["flow.b1"].clone(), fw2: t["flow.w2"].clone(), fb2: t["flow.b2"].clone(), fw3: t["flow.w3"].clone(), fb3: [g3[0], g3[1], g3[2]] };
    println!("  EFA-1 orbit-tube bounds — from node-local funnels toward continuum coverage\n");
    println!("     body   L̂(J)    δ_need (½ cell diag)   nodes w/ δ0* ≥ δ_need   median δ0*   min δ0*   spacing for full coverage");
    let gt = [0.8f32, -0.6, 0.5];
    let th_pts = [0usize, 41, 21, 11]; let om_pts = [0usize, 33, 13, 7];
    let core_r = [0.757f32, 0.42, 0.64];
    let mut certjson: Vec<String> = vec![];
    for bi in 0..NB { let nj = NJ[bi]; let d = 2 * nj; let mut g = [0.0f32; 3]; for j in 0..nj { g[j] = gt[j]; }
        // attractor + P (cert1 construction) + ω envelope
        let mut xstar = vec![0.0f32; d]; for j in 0..nj { xstar[j] = g[j]; }
        for _ in 0..800 { xstar = fcl(&m, bi, &xstar, g); }
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
        let sqlam = lammax_sym(&p, d).sqrt();
        let pnorm = |x: &[f32]| -> f32 { let mut dx = vec![0.0f32; d];
            for c in 0..d { let mut e = x[c] - xstar[c]; if c < nj { e = wrap(e); } dx[c] = e; }
            let mut q = 0.0; for i in 0..d { for j in 0..d { q += dx[i] * p[i * d + j] * dx[j]; } } q.max(0.0).sqrt() };
        let mut om_env = 0.0f32;
        for k in 0..60u32 { let mut x = vec![0.0f32; d];
            for j in 0..nj { x[j] = (u(k, 7 + j as u32) * 2.0 - 1.0) * PI; x[nj + j] = (u(k, 17 + j as u32) * 2.0 - 1.0) * 1.5; }
            for _ in 0..300 { x = fcl(&m, bi, &x, g); for j in 0..nj { om_env = om_env.max(x[nj + j].abs()); } } }
        let om_b = (om_env * 1.3).ceil();
        // L̂: sampled Jacobian Lipschitz constant (max over ~300 probe pairs at scales 0.01–0.1)
        let mut lhat = 0.0f32;
        for k in 0..300u32 { let mut x = vec![0.0f32; d];
            for c in 0..d { x[c] = if c < nj { (u(k, 31 + c as u32) * 2.0 - 1.0) * PI } else { (u(k, 41 + c as u32) * 2.0 - 1.0) * om_b }; }
            let eps = 0.01 + u(k, 51) * 0.09;
            let mut y = x.clone(); let cdir = (u(k, 52) * d as f32) as usize % d; y[cdir] += eps;
            let (jx, jy) = (jac_at(&x), jac_at(&y));
            let diff: Vec<f32> = jx.iter().zip(&jy).map(|(a, b)| a - b).collect();
            let l = sigmax_mat(&diff, d) / eps;
            if l > lhat { lhat = l; } }
        // grid cell half-diagonal (the δ0 needed for continuum coverage)
        let spacings: Vec<f32> = (0..d).map(|c| if c < nj { 2.0 * PI / th_pts[nj] as f32 } else { 2.0 * om_b / (om_pts[nj] as f32 - 1.0) }).collect();
        let dneed = 0.5 * spacings.iter().map(|s| s * s).sum::<f32>().sqrt();
        // tube certification with binary search on δ0 (sampled nodes for 2/3-DOF)
        let npts: Vec<usize> = (0..d).map(|c| if c < nj { th_pts[nj] } else { om_pts[nj] }).collect();
        let total: usize = npts.iter().product();
        let mut stride = vec![1usize; d]; for c in 1..d { stride[c] = stride[c - 1] * npts[c - 1]; }
        let pick = [0usize, 1, 8, 32][nj];
        let tube_ok = |x0: &[f32], d0: f32| -> bool {
            let mut x = x0.to_vec(); let mut del = d0;
            for _ in 0..TMAX {
                if pnorm(&x) + sqlam * del <= core_r[bi] { return true; }
                let sig = sigmax_mat(&jac_at(&x), d);
                del = (sig + lhat * del) * del;
                if del > DCAP { return false; }
                x = fcl(&m, bi, &x, g);
            }
            false };
        let (mut covered, mut tried) = (0usize, 0usize); let mut d0s: Vec<f32> = vec![];
        for n in (0..total).step_by(pick) { tried += 1;
            let x0: Vec<f32> = (0..d).map(|c| { let i = n / stride[c] % npts[c];
                if c < nj { -PI + 2.0 * PI * i as f32 / npts[c] as f32 } else { -om_b + 2.0 * om_b * i as f32 / (npts[c] as f32 - 1.0) } }).collect();
            if tube_ok(&x0, dneed) { covered += 1; d0s.push(dneed); continue; }
            // binary search the largest certifying δ0 < dneed
            let (mut lo, mut hi) = (0.0f32, dneed);
            for _ in 0..8 { let mid = 0.5 * (lo + hi); if tube_ok(&x0, mid) { lo = mid; } else { hi = mid; } }
            d0s.push(lo); }
        let mut ds = d0s.clone(); ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = ds[ds.len() / 2]; let dmin = ds[0];
        let cov_pct = covered as f32 / tried as f32 * 100.0;
        // spacing that would give full coverage: cell half-diagonal ≤ min δ0*  ⇒  spacing scale factor = dmin/dneed
        let scale = if dneed > 0.0 { dmin / dneed } else { 0.0 };
        let req = format!("{:.3}× current ({} nodes → ~{:.0}M for full sweep)", scale,
            tried, (total as f32 / scale.max(1e-3).powi(d as i32)) / 1e6);
        println!("     {}-DOF   {:.2}      {:.3}                  {:>5.1}% of {}            {:.3}       {:.4}     {}",
            nj, lhat, dneed, cov_pct, tried, med, dmin, req);
        certjson.push(format!("{{\"body\":\"{}-DOF\",\"lipschitz_J_sampled\":{:.2},\"delta_needed_half_cell_diag\":{:.3},\"nodes_covering_pct\":{:.1},\"nodes_tested\":{},\"median_certified_delta0\":{:.3},\"min_certified_delta0\":{:.4},\"full_coverage_spacing_factor\":{:.3},\"method\":\"orbit tubes: delta' = (sigma(J)+L*delta)*delta along each node orbit, certified when pnorm(x)+sqrt(lammax(P))*delta <= r_core; L sampled (300 pairs, NOT proven — interval arithmetic named as the rigorous route); nodes sampled 1/{} for cost\"}}",
            nj, lhat, dneed, cov_pct, tried, med, dmin, scale, pick));
    }
    let cfg = std::fs::read_to_string(format!("{MDIR}/config.json")).unwrap();
    let base = match cfg.find(",\n  \"certificates_tube\"") { Some(i) => { let tail = &cfg[i..]; let after = tail.find("]\n").map(|j| i + j + 2).unwrap_or(cfg.len()); format!("{}{}", &cfg[..i], &cfg[after..]) }, None => cfg };
    let patched = base.trim_end().trim_end_matches('}').trim_end().trim_end_matches(',').to_string()
        + &format!(",\n  \"certificates_tube\": [{}]\n}}\n", certjson.join(", "));
    std::fs::write(format!("{MDIR}/config.json"), patched).unwrap();
    println!("\n  tube certificate written into {MDIR}/config.json — VALIDATE THE JSON before shipping.");
    println!("  HONEST: L̂ sampled not proven; nodes sampled for 2/3-DOF; where δ0* < δ_need the deficit prices the full-coverage grid.");
}
