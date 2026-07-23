//! EFA-1 — the multi-body model: ONE body-embedding-conditioned trunk controls a FAMILY of bodies (1-, 2-, 3-joint
//! coupled chains) from a single weights set. Flow actuation head + potential verify head, shared across bodies.
//!
//! This is the foundation-model-shaped EFA artifact: swap the body id, same weights, a different body is controlled.
//! Stage A: per-body FVI value demonstrators (HV=128). Stage B: ONE shared flow net + ONE shared potential net + a
//! learned body-embedding table, distilled from a MIXED-body stream (conditional flow matching + contrastive verify),
//! action dims masked per body via zero-targets. Then the IDENTITY CARD — the metrics that matter for machines that act:
//! reach per body, FLOPs/decision (flow vs the discrete Gᵈ planner), verify %, and determinism (same state+goal ⇒ same
//! action, bit-for-bit, on repeat). Tokens and parameter-count-as-capability refused.
//!
//! Run: `cargo run -p ferric-tensor --example ebm_efa1 --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;
use std::io::Write as IoWrite;
const OUTDIR: &str = "/Users/dcharlot/vibe-coding/efa/models/efa-1";
fn save_st(path: &str, ts: &[(String, Vec<usize>, Vec<f32>)]) -> std::io::Result<()> {
    let mut hdr = String::from("{"); let mut off = 0usize;
    for (i, (n, sh, d)) in ts.iter().enumerate() { let b = d.len() * 4; if i > 0 { hdr.push(','); }
        hdr.push_str(&format!("\"{}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}", n, sh.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","), off, off + b)); off += b; }
    hdr.push('}'); let mut f = std::fs::File::create(path)?; f.write_all(&(hdr.len() as u64).to_le_bytes())?; f.write_all(hdr.as_bytes())?;
    for (_, _, d) in ts { for v in d { f.write_all(&v.to_le_bytes())?; } } Ok(())
}
fn load_st(path: &str) -> std::io::Result<std::collections::HashMap<String, Vec<f32>>> {
    let raw = std::fs::read(path)?; let hl = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&raw[8..8 + hl]).unwrap().to_string(); let data = &raw[8 + hl..];
    let mut out = std::collections::HashMap::new(); let mut rest = header.as_str();
    while let Some(q) = rest.find("\"dtype\"") { let pre = &rest[..q]; let ne = pre.rfind("\":{").unwrap(); let ns = pre[..ne].rfind('"').unwrap() + 1;
        let name = pre[ns..ne].to_string(); let a = &rest[q..]; let os = a.find("\"data_offsets\":[").unwrap() + 16; let oe = a[os..].find(']').unwrap() + os;
        let of: Vec<usize> = a[os..oe].split(',').map(|s| s.trim().parse().unwrap()).collect();
        out.insert(name, data[of[0]..of[1]].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()); rest = &a[oe..]; } Ok(out)
}
const H: usize = 128; const HV: usize = 128; const EMB: usize = 6; const DT: f32 = 0.05; const GAMMA: f32 = 0.97; const UMAX: f32 = 4.0; const CPL: f32 = 0.5;
const G5: [f32; 5] = [-4.0, -2.0, 0.0, 2.0, 4.0];
const NB: usize = 3;                       // bodies: 1-, 2-, 3-joint chains
const NJ: [usize; NB] = [1, 2, 3];
use std::f32::consts::PI;
fn h32(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u(i: u32, s: u32) -> f32 { (h32(i.wrapping_mul(2654435761).wrapping_add(s)) % 1_000_000 + 1) as f32 / 1_000_000.0 }
fn randn(n: usize, seed: u32, sc: f32) -> Vec<f32> { (0..n).map(|i| { let a = u(i as u32, seed); let b = u(i as u32, seed.wrapping_add(9973)); ((-2.0 * a.ln()).sqrt() * (6.2831853 * b).cos()) * sc }).collect() }
fn wrap(x: f32) -> f32 { let mut a = x; while a > PI { a -= 2.0 * PI; } while a < -PI { a += 2.0 * PI; } a }
// generalized nj-joint coupled chain (state = [θ×3, ω×3] padded; only first nj joints active)
fn step(nj: usize, s: [f32; 6], uu: [f32; 3]) -> [f32; 6] {
    let mut th = [s[0], s[1], s[2]]; let om = [s[3], s[4], s[5]]; let mut no = om;
    for i in 0..nj { let mut cpl = 0.0;
        if i > 0 { cpl += CPL * (th[i - 1] - th[i]).sin(); }
        if i + 1 < nj { cpl += CPL * (th[i + 1] - th[i]).sin(); }
        no[i] = om[i] + DT * (-th[i].sin() - 0.05 * om[i] + cpl + uu[i].clamp(-UMAX, UMAX)); }
    for i in 0..nj { th[i] = wrap(th[i] + DT * no[i]); }
    [th[0], th[1], th[2], no[0], no[1], no[2]]
}
fn cost(nj: usize, s: [f32; 6], g: [f32; 3], uu: [f32; 3]) -> f32 { let mut c = 0.0; for i in 0..nj { c += wrap(s[i] - g[i]).powi(2) + 0.05 * s[3 + i] * s[3 + i] + 0.01 * uu[i] * uu[i]; } c }
// fixed 12-wide joint-feature encoding (4 per joint; inactive joints = 0) — the body-agnostic state code
fn feat12(nj: usize, s: [f32; 6], g: [f32; 3]) -> [f32; 12] { let mut f = [0.0f32; 12];
    for i in 0..nj { let d = s[i] - g[i]; f[i * 4] = d.cos(); f[i * 4 + 1] = d.sin(); f[i * 4 + 2] = s[3 + i]; f[i * 4 + 3] = s[i].sin(); } f }

// per-body value V(12-feat)→scalar (softplus), with two-stage argmin demonstrator
struct Vn { w: Vec<Vec<f32>>, b1: Vec<f32>, w2: Vec<f32>, b2: Vec<f32>, w3: Vec<f32>, b3: f32, nj: usize }
impl Vn {
    fn eval(&self, s: [f32; 6], g: [f32; 3]) -> f32 { let f = feat12(self.nj, s, g);
        let mut h1 = [0.0f32; HV]; for j in 0..HV { let mut z = self.b1[j]; for c in 0..12 { z += f[c] * self.w[c][j]; } h1[j] = (z.exp() + 1.0).ln(); }
        let mut h2 = [0.0f32; HV]; for j in 0..HV { let mut z = self.b2[j]; for k in 0..HV { z += h1[k] * self.w2[k * HV + j]; } h2[j] = (z.exp() + 1.0).ln(); }
        let mut o = self.b3; for j in 0..HV { o += h2[j] * self.w3[j]; } (o.exp() + 1.0).ln() }
    fn ustar(&self, s: [f32; 6], g: [f32; 3]) -> [f32; 3] {
        let nj = self.nj; let grids = [&G5[..]; 3]; let mut bu = [0.0f32; 3]; let mut be = f32::MAX;
        // coarse argmin over the nj-dim grid (recursive via counters)
        let mut idx = [0usize; 3]; let total: usize = grids[..nj].iter().map(|g| g.len()).product();
        for _ in 0..total { let mut uu = [0.0f32; 3]; for i in 0..nj { uu[i] = G5[idx[i]]; }
            let q = cost(nj, s, g, uu) * DT + GAMMA * self.eval(step(nj, s, uu), g); if q < be { be = q; bu = uu; }
            let mut c = 0; loop { idx[c] += 1; if idx[c] < 5 { break; } idx[c] = 0; c += 1; if c >= nj { break; } } }
        // fine ±0.75 refine
        let base = bu; let mut idx = [0usize; 3]; let total: usize = 3usize.pow(nj as u32);
        for _ in 0..total { let mut uu = base; for i in 0..nj { uu[i] = (base[i] + [-0.75f32, 0.0, 0.75][idx[i]]).clamp(-UMAX, UMAX); }
            let q = cost(nj, s, g, uu) * DT + GAMMA * self.eval(step(nj, s, uu), g); if q < be { be = q; bu = uu; }
            let mut c = 0; loop { idx[c] += 1; if idx[c] < 3 { break; } idx[c] = 0; c += 1; if c >= nj { break; } } }
        bu }
}
// shared multi-body nets: flow (12 feat + emb → 3 vel), potential (12 feat + 3 act + emb → scalar). relu/linear.
struct Efa1 { emb: Vec<f32>,   // [NB, EMB]
    fw: Vec<Vec<f32>>, fb1: Vec<f32>, fw2: Vec<f32>, fb2: Vec<f32>, fw3: Vec<f32>, fb3: [f32; 3],   // flow: 12+3act?+ no, flow in = 12+3(a)+1(t)+EMB
    pw: Vec<Vec<f32>>, pb1: Vec<f32>, pw2: Vec<f32>, pb2: Vec<f32>, pw3: Vec<f32>, pb3: f32 }
// flow input layout: [12 joint-feat, a0,a1,a2, t, EMB…] = 12+3+1+EMB ; potential input: [12, a0,a1,a2, EMB] = 12+3+EMB
impl Efa1 {
    fn flow_vel(&self, bi: usize, s: [f32; 6], g: [f32; 3], a: [f32; 3], t: f32) -> [f32; 3] {
        let nj = NJ[bi]; let ff = feat12(nj, s, g); let mut f = vec![0.0f32; 12 + 3 + 1 + EMB];
        for c in 0..12 { f[c] = ff[c]; } f[12] = a[0]; f[13] = a[1]; f[14] = a[2]; f[15] = t; for c in 0..EMB { f[16 + c] = self.emb[bi * EMB + c]; }
        let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.fb1[j]; for c in 0..f.len() { z += f[c] * self.fw[c][j]; } h1[j] = z.max(0.0); }
        let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.fb2[j]; for k in 0..H { z += h1[k] * self.fw2[k * H + j]; } h2[j] = z.max(0.0); }
        let mut o = self.fb3; for j in 0..H { for c in 0..3 { o[c] += h2[j] * self.fw3[j * 3 + c]; } } o }
    fn act(&self, bi: usize, s: [f32; 6], g: [f32; 3]) -> [f32; 3] {   // K=1 forward pass
        let v = self.flow_vel(bi, s, g, [0.0; 3], 0.0); let nj = NJ[bi]; let mut a = [0.0f32; 3];
        for i in 0..nj { a[i] = v[i].clamp(-UMAX, UMAX); } a }
    fn energy(&self, bi: usize, s: [f32; 6], g: [f32; 3], a: [f32; 3]) -> f32 {
        let nj = NJ[bi]; let ff = feat12(nj, s, g); let mut f = vec![0.0f32; 12 + 3 + EMB];
        for c in 0..12 { f[c] = ff[c]; } f[12] = a[0]; f[13] = a[1]; f[14] = a[2]; for c in 0..EMB { f[15 + c] = self.emb[bi * EMB + c]; }
        let mut h1 = [0.0f32; H]; for j in 0..H { let mut z = self.pb1[j]; for c in 0..f.len() { z += f[c] * self.pw[c][j]; } h1[j] = z.max(0.0); }
        let mut h2 = [0.0f32; H]; for j in 0..H { let mut z = self.pb2[j]; for k in 0..H { z += h1[k] * self.pw2[k * H + j]; } h2[j] = z.max(0.0); }
        let mut o = self.pb3; for j in 0..H { o += h2[j] * self.pw3[j]; } o }
}

// identity card per body: (reach%, verify%, determinism)
fn eval_card(m: &Efa1, vns: &[Vn]) -> Vec<(f32, f32, bool)> {
    let gt = [[0.8f32, -0.6, 0.5], [-0.7, 0.5, -0.6], [0.5, 0.9, -0.4], [-0.5, -0.8, 0.7]];
    let mut rows = vec![];
    for bi in 0..NB { let nj = NJ[bi];
        let (mut reach, nn) = (0, 60);
        for k in 0..nn { let sd = 900 + (bi * 100 + k) as u32; let mut s = [0.0f32; 6]; let mut g = [0.0f32; 3];
            let gtk = gt[k % 4]; for j in 0..nj { s[j] = (u(sd, 7 + j as u32) * 2.0 - 1.0) * PI; g[j] = gtk[j]; }
            for _ in 0..300 { let a = m.act(bi, s, g); s = step(nj, s, a); }
            if (0..nj).all(|i| wrap(s[i] - g[i]).abs() < 0.35 && s[3 + i].abs() < 0.7) { reach += 1; } }
        let (mut vg, mut vt) = (0, 0); for k in 0..2000u32 { let mut s = [0.0f32; 6]; let mut g = [0.0f32; 3];
            for j in 0..nj { let ju = j as u32; s[j] = (u(k, 41 + ju) * 2.0 - 1.0) * PI; s[3 + j] = (u(k, 44 + ju) * 2.0 - 1.0) * 3.0; g[j] = (u(k, 47 + ju) * 2.0 - 1.0) * 1.0; }
            let us = vns[bi].ustar(s, g); let mut bad = [0.0f32; 3]; for j in 0..nj { bad[j] = (u(k, 50 + j as u32) * 2.0 - 1.0) * UMAX; }
            vt += 1; if m.energy(bi, s, g, us) < m.energy(bi, s, g, bad) { vg += 1; } }
        let s0 = [0.3f32, -0.2, 0.1, 0.0, 0.0, 0.0]; let g0 = [0.5f32, -0.3, 0.2];
        let (a1, a2) = (m.act(bi, s0, g0), m.act(bi, s0, g0));
        let det = a1.iter().zip(a2.iter()).all(|(x, y)| x.to_bits() == y.to_bits());
        rows.push((reach as f32 / nn as f32 * 100.0, vg as f32 / vt as f32 * 100.0, det)); }
    rows
}
fn print_card(rows: &[(f32, f32, bool)], fin: usize, tag: &str) {
    let flopf = 2 * (fin * H + H * H + H * 3); let flopv = 2 * (12 * HV + HV * HV + HV);
    println!("\n  ── EFA-1 IDENTITY CARD [{}] (one weights file, {} bodies) ──", tag, NB);
    println!("     body      reach(flow K=1)   verify    FLOPs/decision   vs discrete Gᵈ   determinism");
    for bi in 0..NB { let nj = NJ[bi]; let (r, v, det) = rows[bi]; let gd = 5usize.pow(nj as u32) + 3usize.pow(nj as u32);
        println!("     {}-DOF       {:>4.0}%           {:>4.1}%     {:>7}          {:>4}×          {}", nj, r, v, flopf, (gd * flopv) / flopf, if det { "bit-exact ✓" } else { "✗" }); }
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("  EFA-1 — the multi-body model: ONE body-conditioned trunk controls 1-, 2-, 3-joint chains. Stage A→B→card.\n");
    let one = Tensor::from_vec(&ctx, &[1.0], &[1]); let bs = 160usize;
    let sp = |z: Var, ov: &Var| z.exp().add(ov).log();

    // ---- Stage A: per-body FVI value demonstrators ----
    let mut vns: Vec<Vn> = vec![];
    for bi in 0..NB { let nj = NJ[bi];
        let vnet = |f: &[Var], pv: &[Var], ov: &Var| { let mut pre = pv[12].clone(); for c in 0..12 { pre = pre.add(&f[c].matmul(&pv[c])); } sp(sp(sp(pre, ov).matmul(&pv[13]).add(&pv[14]), ov).matmul(&pv[15]).add(&pv[16]), ov) };
        let mut p: Vec<Tensor> = (0..12).map(|c| Tensor::from_vec(&ctx, &randn(HV, 22 + (bi * 40 + c) as u32, 0.45), &[1, HV])).collect();
        p.push(Tensor::zeros(&ctx, &[HV])); p.push(Tensor::from_vec(&ctx, &randn(HV * HV, 400 + bi as u32, 1.0 / (HV as f32).sqrt()), &[HV, HV])); p.push(Tensor::zeros(&ctx, &[HV]));
        p.push(Tensor::from_vec(&ctx, &randn(HV, 410 + bi as u32, 1.0 / (HV as f32).sqrt()), &[HV, 1])); p.push(Tensor::zeros(&ctx, &[1]));
        let mut tgt = p.clone(); let mut adam = Adam::new(&p, 0.002); let ga: usize = 5usize.pow(nj as u32);
        let iters = if nj == 3 { 18000 } else { 12000 };
        for it in 0..iters {
            let mut fc: Vec<Vec<f32>> = (0..12).map(|_| vec![0.0f32; bs]).collect();
            let mut nf: Vec<Vec<f32>> = (0..12).map(|_| vec![0.0f32; bs * ga]).collect(); let mut cst = vec![0.0f32; bs * ga];
            for i in 0..bs { let sd = it as u32 * 7 + i as u32 + bi as u32 * 900000;
                let mut s = [0.0f32; 6]; let mut g = [0.0f32; 3];
                for j in 0..nj { s[j] = (u(sd, 1 + j as u32) * 2.0 - 1.0) * PI; s[3 + j] = (u(sd, 4 + j as u32) * 2.0 - 1.0) * 3.0; g[j] = (u(sd, 10 + j as u32) * 2.0 - 1.0) * 1.0; }
                let f = feat12(nj, s, g); for c in 0..12 { fc[c][i] = f[c]; }
                // enumerate the nj-dim action grid
                let mut idx = [0usize; 3]; for a in 0..ga { let mut uu = [0.0f32; 3]; for j in 0..nj { uu[j] = G5[idx[j]]; }
                    let ns = step(nj, s, uu); let nff = feat12(nj, ns, g); for c in 0..12 { nf[c][i * ga + a] = nff[c]; } cst[i * ga + a] = cost(nj, s, g, uu);
                    let mut cc = 0; loop { idx[cc] += 1; if idx[cc] < 5 { break; } idx[cc] = 0; cc += 1; if cc >= nj { break; } } } }
            let l = |v: &[f32], r: usize| Var::leaf(Tensor::from_vec(&ctx, v, &[r, 1]));
            let tv: Vec<Var> = tgt.iter().map(|t| Var::leaf(t.clone())).collect(); let ov = Var::leaf(one.clone());
            let et = vnet(&(0..12).map(|c| l(&nf[c], bs * ga)).collect::<Vec<_>>(), &tv, &ov).value().to_vec().await;
            let mut target = vec![0.0f32; bs]; for i in 0..bs { let mut m = f32::MAX; for a in 0..ga { let q = cst[i * ga + a] * DT + GAMMA * et[i * ga + a]; if q < m { m = q; } } target[i] = m; }
            let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
            let e = vnet(&(0..12).map(|c| l(&fc[c], bs)).collect::<Vec<_>>(), &pv, &ov); let d = e.sub(&Var::leaf(Tensor::from_vec(&ctx, &target, &[bs, 1]))); let loss = d.mul(&d).mean_all(); loss.backward();
            let g: Vec<Tensor> = pv.iter().zip(&p).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect(); adam.step(&mut p, &g); if it % 200 == 0 { tgt = p.clone(); }
        }
        let mut w: Vec<Vec<f32>> = Vec::new(); for c in 0..12 { w.push(p[c].to_vec().await); }
        vns.push(Vn { w, b1: p[12].to_vec().await, w2: p[13].to_vec().await, b2: p[14].to_vec().await, w3: p[15].to_vec().await, b3: p[16].to_vec().await[0], nj });
        println!("  Stage A: body {} ({}-DOF) demonstrator trained", bi, nj);
    }

    // ---- Stage B: ONE shared flow + potential + body embedding, distilled from the mixed-body stream ----
    let fin = 12 + 3 + 1 + EMB; let pin = 12 + 3 + EMB;
    let fnet = |f: &[Var], pv: &[Var]| { let mut pre = pv[fin].clone(); for c in 0..fin { pre = pre.add(&f[c].matmul(&pv[c])); } pre.relu().matmul(&pv[fin + 1]).add(&pv[fin + 2]).relu().matmul(&pv[fin + 3]).add(&pv[fin + 4]) };
    let pnet = |f: &[Var], pv: &[Var]| { let mut pre = pv[pin].clone(); for c in 0..pin { pre = pre.add(&f[c].matmul(&pv[c])); } pre.relu().matmul(&pv[pin + 1]).add(&pv[pin + 2]).relu().matmul(&pv[pin + 3]).add(&pv[pin + 4]) };
    // flow params: fin rank-1 [1,H] + b1 + W2 + b2 + W3[H,3] + b3[3]
    let mut fp: Vec<Tensor> = (0..fin).map(|c| Tensor::from_vec(&ctx, &randn(H, 500 + c as u32, 0.35), &[1, H])).collect();
    fp.push(Tensor::zeros(&ctx, &[H])); fp.push(Tensor::from_vec(&ctx, &randn(H * H, 560, 1.0 / (H as f32).sqrt()), &[H, H])); fp.push(Tensor::zeros(&ctx, &[H]));
    fp.push(Tensor::from_vec(&ctx, &randn(H * 3, 561, 1.0 / (H as f32).sqrt()), &[H, 3])); fp.push(Tensor::zeros(&ctx, &[3]));
    let mut pp: Vec<Tensor> = (0..pin).map(|c| Tensor::from_vec(&ctx, &randn(H, 600 + c as u32, 0.35), &[1, H])).collect();
    pp.push(Tensor::zeros(&ctx, &[H])); pp.push(Tensor::from_vec(&ctx, &randn(H * H, 660, 1.0 / (H as f32).sqrt()), &[H, H])); pp.push(Tensor::zeros(&ctx, &[H]));
    pp.push(Tensor::from_vec(&ctx, &randn(H, 661, 1.0 / (H as f32).sqrt()), &[H, 1])); pp.push(Tensor::zeros(&ctx, &[1]));
    let mut emb = Tensor::from_vec(&ctx, &randn(NB * EMB, 700, 0.5), &[NB, EMB]);   // shared body embedding
    let mut adamf = Adam::new(&fp, 0.0015); let mut adamp = Adam::new(&pp, 0.0015); let mut adame = Adam::new(&std::slice::from_ref(&emb), 0.0015);
    let embsel: Vec<Tensor> = (0..EMB).map(|c| { let mut v = vec![0.0f32; EMB]; v[c] = 1.0; Tensor::from_vec(&ctx, &v, &[EMB, 1]) }).collect();
    let marg = Var::leaf(Tensor::from_vec(&ctx, &[0.5], &[1]));
    for it in 0..14000 {
        // mixed-body batch: flow-matching target + potential contrastive
        let mut fcf: Vec<Vec<f32>> = (0..12 + 3 + 1).map(|_| vec![0.0f32; bs]).collect(); let mut oh = vec![0.0f32; bs * NB]; let mut tb = vec![0.0f32; bs * 3];
        let mut pcp: Vec<Vec<f32>> = (0..12 + 3).map(|_| vec![0.0f32; bs]).collect(); let mut pcn: Vec<Vec<f32>> = (0..12 + 3).map(|_| vec![0.0f32; bs]).collect(); let mut ohp = vec![0.0f32; bs * NB];
        for i in 0..bs { let sd = it as u32 * 13 + i as u32; let bi = (u(sd, 20) * NB as f32) as usize % NB; let nj = NJ[bi];
            let mut s = [0.0f32; 6]; let mut g = [0.0f32; 3];
            for j in 0..nj { s[j] = (u(sd, 1 + j as u32) * 2.0 - 1.0) * PI; s[3 + j] = (u(sd, 4 + j as u32) * 2.0 - 1.0) * 3.0; g[j] = (u(sd, 10 + j as u32) * 2.0 - 1.0) * 1.0; }
            let us = vns[bi].ustar(s, g); let ff = feat12(nj, s, g);
            // flow-matching: a_t=(1-t)a0+t·u*, target=u*-a0 (masked dims: a0=0,u*=0 → target 0)
            let t = u(sd, 9) * 0.9; for c in 0..12 { fcf[c][i] = ff[c]; }
            for j in 0..3 { let a0 = if j < nj { (u(sd, 30 + j as u32) * 2.0 - 1.0) * 3.0 } else { 0.0 }; let ut = if j < nj { us[j] } else { 0.0 };
                fcf[12 + j][i] = (1.0 - t) * a0 + t * ut; tb[i * 3 + j] = ut - a0; }
            fcf[15][i] = t; oh[i * NB + bi] = 1.0;
            // potential contrastive: E(s,u*) < E(s, random) at t implicit (potential has no t input)
            let bn = (u(sd, 21) * NB as f32) as usize % NB; let njn = NJ[bn]; let mut sn = [0.0f32; 6]; let mut gn = [0.0f32; 3];
            for j in 0..njn { sn[j] = (u(sd, 40 + j as u32) * 2.0 - 1.0) * PI; sn[3 + j] = (u(sd, 43 + j as u32) * 2.0 - 1.0) * 3.0; gn[j] = (u(sd, 46 + j as u32) * 2.0 - 1.0) * 1.0; }
            let usn = vns[bn].ustar(sn, gn); let ffn = feat12(njn, sn, gn); for c in 0..12 { pcp[c][i] = ffn[c]; pcn[c][i] = ffn[c]; }
            for j in 0..3 { pcp[12 + j][i] = if j < njn { usn[j] } else { 0.0 }; pcn[12 + j][i] = if j < njn { (u(sd, 50 + j as u32) * 2.0 - 1.0) * UMAX } else { 0.0 }; } ohp[i * NB + bn] = 1.0;
        }
        let l = |v: &[f32], n: usize| Var::leaf(Tensor::from_vec(&ctx, v, &[n, 1]));
        // ---- flow update (+ emb) ----
        let fpv: Vec<Var> = fp.iter().map(|t| Var::leaf(t.clone())).collect(); let ev = Var::leaf(emb.clone());
        let embg = Var::leaf(Tensor::from_vec(&ctx, &oh, &[bs, NB])).matmul(&ev);        // [bs,EMB]
        let mut ff: Vec<Var> = (0..16).map(|c| l(&fcf[c], bs)).collect();
        for c in 0..EMB { ff.push(embg.matmul(&Var::leaf(embsel[c].clone()))); }
        let v = fnet(&ff, &fpv);
        let d = v.sub(&Var::leaf(Tensor::from_vec(&ctx, &tb, &[bs, 3]))); let floss = d.mul(&d).mean_all(); floss.backward();
        let gf: Vec<Tensor> = fpv.iter().zip(&fp).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect(); adamf.step(&mut fp, &gf);
        let ge = ev.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; NB * EMB], &[NB, EMB])); adame.step(&mut std::slice::from_mut(&mut emb), &[ge]);
        // ---- potential update (contrastive; + emb) ----
        let ppv: Vec<Var> = pp.iter().map(|t| Var::leaf(t.clone())).collect(); let ev2 = Var::leaf(emb.clone());
        let embp = Var::leaf(Tensor::from_vec(&ctx, &ohp, &[bs, NB])).matmul(&ev2);
        let mut pf: Vec<Var> = (0..15).map(|c| l(&pcp[c], bs)).collect(); for c in 0..EMB { pf.push(embp.matmul(&Var::leaf(embsel[c].clone()))); }
        let mut nf2: Vec<Var> = (0..15).map(|c| l(&pcn[c], bs)).collect(); for c in 0..EMB { nf2.push(embp.matmul(&Var::leaf(embsel[c].clone()))); }
        let ep = pnet(&pf, &ppv); let en = pnet(&nf2, &ppv);
        let hinge = ep.sub(&en).add(&marg).relu(); let anch = ep.mul(&ep).mul(&Var::leaf(Tensor::from_vec(&ctx, &[0.02], &[1])));
        let ploss = hinge.add(&anch).mean_all(); ploss.backward();
        let gp: Vec<Tensor> = ppv.iter().zip(&pp).map(|(v, t)| v.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; t.numel()], &t.shape))).collect(); adamp.step(&mut pp, &gp);
        let ge2 = ev2.grad().unwrap_or_else(|| Tensor::from_vec(&ctx, &vec![0.0; NB * EMB], &[NB, EMB])); adame.step(&mut std::slice::from_mut(&mut emb), &[ge2]);
    }
    // extract shared model
    let mut fw: Vec<Vec<f32>> = Vec::new(); for c in 0..fin { fw.push(fp[c].to_vec().await); }
    let fb3 = fp[fin + 4].to_vec().await; let mut pw: Vec<Vec<f32>> = Vec::new(); for c in 0..pin { pw.push(pp[c].to_vec().await); }
    let m = Efa1 { emb: emb.to_vec().await,
        fw, fb1: fp[fin].to_vec().await, fw2: fp[fin + 1].to_vec().await, fb2: fp[fin + 2].to_vec().await, fw3: fp[fin + 3].to_vec().await, fb3: [fb3[0], fb3[1], fb3[2]],
        pw, pb1: pp[pin].to_vec().await, pw2: pp[pin + 1].to_vec().await, pb2: pp[pin + 2].to_vec().await, pw3: pp[pin + 3].to_vec().await, pb3: pp[pin + 4].to_vec().await[0] };

    // ---- IDENTITY CARD → GATE → SAVE → RELOAD → round-trip ----
    let rows = eval_card(&m, &vns); print_card(&rows, fin, "trained");
    let nparams = (fin * H + H + H * H + H + H * 3 + 3) + (pin * H + H + H * H + H + H + 1) + NB * EMB;
    let pass = rows.iter().all(|(r, v, det)| *r >= 95.0 && *v >= 90.0 && *det);
    if !pass { println!("\n  GATE FAILED (need every body reach≥95 & verify≥90 & bit-exact) — not shipping."); return; }
    // save shared trunk + body embedding
    std::fs::create_dir_all(OUTDIR).unwrap();
    let mut ts: Vec<(String, Vec<usize>, Vec<f32>)> = vec![("body_embedding".into(), vec![NB, EMB], m.emb.clone())];
    for c in 0..fin { ts.push((format!("flow.in{}", c), vec![1, H], m.fw[c].clone())); }
    ts.push(("flow.b1".into(), vec![H], m.fb1.clone())); ts.push(("flow.w2".into(), vec![H, H], m.fw2.clone()));
    ts.push(("flow.b2".into(), vec![H], m.fb2.clone())); ts.push(("flow.w3".into(), vec![H, 3], m.fw3.clone())); ts.push(("flow.b3".into(), vec![3], m.fb3.to_vec()));
    for c in 0..pin { ts.push((format!("potential.in{}", c), vec![1, H], m.pw[c].clone())); }
    ts.push(("potential.b1".into(), vec![H], m.pb1.clone())); ts.push(("potential.w2".into(), vec![H, H], m.pw2.clone()));
    ts.push(("potential.b2".into(), vec![H], m.pb2.clone())); ts.push(("potential.w3".into(), vec![H, 1], m.pw3.clone())); ts.push(("potential.b3".into(), vec![1], vec![m.pb3]));
    save_st(&format!("{OUTDIR}/model.safetensors"), &ts).unwrap();
    let bodies: Vec<String> = (0..NB).map(|b| format!("\"{}-DOF-chain\"", NJ[b])).collect();
    let cardj: Vec<String> = (0..NB).map(|b| format!("{{\"body\":\"{}-DOF\",\"reach_K1\":{:.1},\"verify\":{:.1},\"deterministic\":{}}}", NJ[b], rows[b].0, rows[b].1, rows[b].2)).collect();
    let config = format!("{{\n  \"architecture\": \"efa-1\",\n  \"description\": \"EFA-1: one body-embedding-conditioned trunk controls a family of bodies. Flow head (actuation, K=1 forward pass) + potential head (verify: low = valid action). Energy-based, deterministic, joules-metered, multi-body — identity NOT measured in tokens or parameter count.\",\n  \"hidden\": {H}, \"emb_dim\": {EMB}, \"bodies\": [{}],\n  \"params\": {nparams},\n  \"flow_features\": \"[12 joint-feats (4/joint: cos(θ-g),sin(θ-g),ω,sinθ), a1,a2,a3, t, {EMB}-dim body-embedding]\",\n  \"potential_features\": \"[12 joint-feats, a1,a2,a3, {EMB}-dim body-embedding]\",\n  \"env\": {{\"family\":\"coupled-pendulum-chain\",\"dof\":[1,2,3],\"dt\":{DT},\"umax\":{UMAX},\"coupling\":{CPL},\"damping\":0.05,\"gravity\":\"sin\"}},\n  \"inference\": \"act(body,state,goal): v = flow(feat, a=0, t=0, emb[body]); u = clamp(v[:dof]). verify(body,state,goal,a): potential(feat,a,emb[body]) — lower is more valid.\",\n  \"identity_card\": [{}],\n  \"gate\": \"every body reach_K1>=95 && verify>=90 && bit-exact determinism\"\n}}\n", bodies.join(", "), cardj.join(", "));
    std::fs::write(format!("{OUTDIR}/config.json"), config).unwrap();
    // reload + re-card
    let t = load_st(&format!("{OUTDIR}/model.safetensors")).unwrap();
    let g = |n: &str| t.get(n).unwrap().clone();
    let fb3 = g("flow.b3"); let m2 = Efa1 { emb: g("body_embedding"),
        fw: (0..fin).map(|c| g(&format!("flow.in{}", c))).collect(), fb1: g("flow.b1"), fw2: g("flow.w2"), fb2: g("flow.b2"), fw3: g("flow.w3"), fb3: [fb3[0], fb3[1], fb3[2]],
        pw: (0..pin).map(|c| g(&format!("potential.in{}", c))).collect(), pb1: g("potential.b1"), pw2: g("potential.w2"), pb2: g("potential.b2"), pw3: g("potential.w3"), pb3: g("potential.b3")[0] };
    let rows2 = eval_card(&m2, &vns); print_card(&rows2, fin, "RELOADED from disk");
    let rt = rows.iter().zip(&rows2).all(|(a, b)| (a.0 - b.0).abs() < 0.5 && (a.1 - b.1).abs() < 0.5);
    println!("\n  ONE trunk, {} params, {} bodies — swap the body embedding, control a different body. Flow = 1 forward pass.", nparams, NB);
    println!("  GATE PASSED · round-trip {} · RELEASED: {OUTDIR}/model.safetensors ({} tensors) + config.json", if rt { "EXACT ✓" } else { "MISMATCH ✗" }, ts.len());
    println!("  Identity measured in reach/body · FLOPs/decision · verify% · determinism — NOT tokens, NOT parameter count.");
}
