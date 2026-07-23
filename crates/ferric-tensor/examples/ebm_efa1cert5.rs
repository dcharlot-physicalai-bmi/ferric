//! EFA-1 MULTI-GOAL certification — closes the "one representative goal per body" limit that every certificate so far
//! has carried. For EVERY card goal (4 per body): find the attractor, build the constant-P contraction metric at it,
//! certify the P-core radius by radial sampling (largest r where every sampled shell point has σ_P(J)<1), then run the
//! v4 funnel pass (every node's orbit must enter THAT goal's core) over the full physical domain.
//! Result: the certificate stops being goal-anecdotal — it is per-goal, and the worst goal is the reported number.
//! HONEST: radial shell sampling (~200 pts/shell) instead of a full box grid per goal (cost); 3-DOF funnel on an
//! every-3rd-node subsample per goal (cost, disclosed); grid/node-local sampling limits unchanged from cert4.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_efa1cert5 --release`
use std::f32::consts::PI;
const H: usize = 128; const EMB: usize = 6; const DT: f32 = 0.05; const UMAX: f32 = 4.0; const CPL: f32 = 0.5;
const NB: usize = 3; const NJ: [usize; NB] = [1, 2, 3];
const MDIR: &str = "/Users/dcharlot/vibe-coding/efa/models/efa-1";
const TMAX: usize = 600;
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
fn main() {
    let t = load_st(&format!("{MDIR}/model.safetensors"));
    let fin = 12 + 3 + 1 + EMB; let g3 = t["flow.b3"].clone();
    let m = Efa1 { emb: t["body_embedding"].clone(),
        fw: (0..fin).map(|c| t[&format!("flow.in{}", c)].clone()).collect(), fb1: t["flow.b1"].clone(), fw2: t["flow.w2"].clone(), fb2: t["flow.b2"].clone(), fw3: t["flow.w3"].clone(), fb3: [g3[0], g3[1], g3[2]] };
    println!("  EFA-1 MULTI-GOAL certification — every card goal gets attractor + P-core + full-domain funnel\n");
    let gt4 = [[0.8f32, -0.6, 0.5], [-0.7, 0.5, -0.6], [0.5, 0.9, -0.4], [-0.5, -0.8, 0.7]];
    let th_pts = [0usize, 41, 21, 11]; let om_pts = [0usize, 33, 13, 7];
    let mut certjson: Vec<String> = vec![];
    for bi in 0..NB { let nj = NJ[bi]; let d = 2 * nj;
        println!("  [{}-DOF]  goal        residual   ρ-proxy σ_P(A)   core r (radial)   funnel certified   median entry", nj);
        let mut worst = (0usize, 101.0f32, 0.0f32, 0.0f32); // (goal, funnel%, core_r, residual)
        let mut rows: Vec<String> = vec![];
        // ω envelope measured once per body (goal-independent enough; conservative max over goals)
        let mut om_env = 0.0f32;
        for (gi, gtk) in gt4.iter().enumerate() { let mut g = [0.0f32; 3]; for j in 0..nj { g[j] = gtk[j]; }
            for k in 0..30u32 { let mut x = vec![0.0f32; d];
                for j in 0..nj { x[j] = (u(k + gi as u32 * 100, 7 + j as u32) * 2.0 - 1.0) * PI; x[nj + j] = (u(k + gi as u32 * 100, 17 + j as u32) * 2.0 - 1.0) * 1.5; }
                for _ in 0..300 { x = fcl(&m, bi, &x, g); for j in 0..nj { om_env = om_env.max(x[nj + j].abs()); } } } }
        let om_b = (om_env * 1.3).ceil();
        for (gi, gtk) in gt4.iter().enumerate() {
            let mut g = [0.0f32; 3]; for j in 0..nj { g[j] = gtk[j]; }
            // attractor
            let mut xstar = vec![0.0f32; d]; for j in 0..nj { xstar[j] = g[j]; }
            for _ in 0..800 { xstar = fcl(&m, bi, &xstar, g); }
            let residual: f32 = (0..d).map(|c| { let v = if c < nj { wrap(xstar[c] - g[c]) } else { xstar[c] }; v * v }).sum::<f32>().sqrt();
            // constant P at x*
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
            let (ev, vv) = jacobi_eig(&p, d);
            let mut ph = vec![0.0f32; d * d]; let mut phi = vec![0.0f32; d * d];
            for i in 0..d { for j in 0..d { for k in 0..d { let lk = ev[k].max(1e-9);
                ph[i * d + j] += vv[i * d + k] * lk.sqrt() * vv[j * d + k];
                phi[i * d + j] += vv[i * d + k] / lk.sqrt() * vv[j * d + k]; } } }
            let sig_p = |jm: &[f32]| -> f32 { sigmax_mat(&matmul(&ph, &matmul(jm, &phi, d), d), d) };
            let sig_a = sig_p(&a);
            let pnorm = |x: &[f32]| -> f32 { let mut dx = vec![0.0f32; d];
                for c in 0..d { let mut e = x[c] - xstar[c]; if c < nj { e = wrap(e); } dx[c] = e; }
                let mut q = 0.0; for i in 0..d { for j in 0..d { q += dx[i] * p[i * d + j] * dx[j]; } } q.max(0.0).sqrt() };
            // radial P-core certification: expand shells until a sampled point fails σ_P(J)<1
            let mut core_r = 0.0f32;
            'shell: for sr in 1..=30 { let r = sr as f32 * 0.05;
                for k in 0..200u32 {
                    // random direction in d-dim, P-norm scaled to shell radius r
                    let mut dir: Vec<f32> = (0..d).map(|c| u(k * 7 + c as u32, 900 + gi as u32) * 2.0 - 1.0).collect();
                    let n: f32 = dir.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                    for c in 0..d { dir[c] /= n; }
                    // scale so pnorm(x) = r: factor = r / pnorm(xstar + dir)
                    let mut probe = xstar.clone(); for c in 0..d { probe[c] += dir[c]; }
                    let pn = pnorm(&probe).max(1e-9);
                    let mut x = xstar.clone(); for c in 0..d { x[c] += dir[c] * r / pn; }
                    if sig_p(&jac_at(&x)) >= 1.0 { break 'shell; } }
                core_r = r; }
            // funnel over the full physical domain (3-DOF: every 3rd node — cost, disclosed)
            let npts: Vec<usize> = (0..d).map(|c| if c < nj { th_pts[nj] } else { om_pts[nj] }).collect();
            let total: usize = npts.iter().product();
            let mut stride = vec![1usize; d]; for c in 1..d { stride[c] = stride[c - 1] * npts[c - 1]; }
            let stridepick = if nj == 3 { 3 } else { 1 };
            let (mut cert, mut tried) = (0usize, 0usize); let mut entries: Vec<usize> = vec![];
            for n in (0..total).step_by(stridepick) { tried += 1;
                let mut x: Vec<f32> = (0..d).map(|c| { let i = n / stride[c] % npts[c];
                    if c < nj { -PI + 2.0 * PI * i as f32 / npts[c] as f32 } else { -om_b + 2.0 * om_b * i as f32 / (npts[c] as f32 - 1.0) } }).collect();
                if pnorm(&x) < core_r { cert += 1; entries.push(0); continue; }
                for k in 1..=TMAX { x = fcl(&m, bi, &x, g);
                    if pnorm(&x) < core_r { cert += 1; entries.push(k); break; } } }
            entries.sort_unstable();
            let med = if entries.is_empty() { 0 } else { entries[entries.len() / 2] };
            let fr = cert as f32 / tried as f32 * 100.0;
            println!("           g{} {:?}   {:.2}       {:.3}            {:.2}              {:>5.1}%  ({} nodes)      {:>4} steps",
                gi + 1, &gtk[..nj], residual, sig_a, core_r, fr, tried, med);
            if fr < worst.1 { worst = (gi, fr, core_r, residual); }
            rows.push(format!("{{\"goal\":{:?},\"attractor_residual\":{:.3},\"sigmaP_at_attractor\":{:.3},\"core_r_radial\":{:.2},\"funnel_certified_pct\":{:.1},\"nodes\":{},\"median_entry\":{}}}",
                &gtk[..nj], residual, sig_a, core_r, fr, tried, med));
        }
        println!("     worst goal: g{} — funnel {:.1}% · the per-body number the card must carry\n", worst.0 + 1, worst.1);
        certjson.push(format!("{{\"body\":\"{}-DOF\",\"domain_omega_bound\":{:.0},\"per_goal\":[{}],\"worst_goal_funnel_pct\":{:.1},\"method\":\"per-goal: attractor + constant-P + RADIAL core certification (200 samples/shell, 0.05 steps) + full-domain funnel entry (3-DOF every-3rd-node, disclosed); node-local grid-sampled limits as certificates_funnel\"}}",
            nj, om_b, rows.join(", "), worst.1));
    }
    let cfg = std::fs::read_to_string(format!("{MDIR}/config.json")).unwrap();
    let base = match cfg.find(",\n  \"certificates_multigoal\"") { Some(i) => { let tail = &cfg[i..]; let after = tail.find("]\n").map(|j| i + j + 2).unwrap_or(cfg.len()); format!("{}{}", &cfg[..i], &cfg[after..]) }, None => cfg };
    let patched = base.trim_end().trim_end_matches('}').trim_end().trim_end_matches(',').to_string()
        + &format!(",\n  \"certificates_multigoal\": [{}]\n}}\n", certjson.join(", "));
    std::fs::write(format!("{MDIR}/config.json"), patched).unwrap();
    println!("  multi-goal certificate written into {MDIR}/config.json — VALIDATE THE JSON before shipping.");
}
