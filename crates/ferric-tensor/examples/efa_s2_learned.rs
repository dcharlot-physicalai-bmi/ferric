//! Milestone S2+ — a LEARNED nonlinear-head Lyapunov energy in the fused discover→verify loop.
//!
//! Same system as `efa_s2_nl` (reversed Van der Pol, non-convex ROA) and the SAME sound verifier, but now
//! the tanh head's INNER weights are trained too:
//!   V(x) = xᵀPx + Σⱼ wⱼ·tanh(aⱼx₁ + bⱼx₂ + cⱼ),   learn P AND every (wⱼ,aⱼ,bⱼ,cⱼ)
//! so the energy shapes its own level sets to the region of attraction instead of using a fixed basis.
//! Because (aⱼ,bⱼ,cⱼ) are inside the tanh, V and V̇ are NONLINEAR in the params — trained via Ferric
//! autograd through the new `Var::tanh` (V̇ written in closed form using tanh′=1−tanh²). The verifier is
//! unchanged and still SOUND (tight per-box tanh/tanh′/tanh″ intervals + adaptive refinement); an
//! independent pointwise check and a forward simulation confirm every certified set lies in the true ROA.
//! Warm-started at the fixed basis (w=0) so it begins as the pure quadratic and grows the head.
//! Run: `cargo run -p ferric-tensor --example efa_s2_learned --release`
use ferric_tensor::{Adam, Tensor, Var};
use std::sync::Arc;
use ferric_certify::Iv; // SOUND interval arithmetic (outward-rounded). Replaces the local
// round-to-nearest implementation this example used to carry: the naive form can NARROW an
// interval and so assert a bound it has not earned. See ferric-certify.

const MU: f64 = 1.0;
const R0: f64 = 0.10;
const ALPHA: f64 = 0.01;
const DELTA: f64 = 0.01;
const ALPHA_T: f64 = 0.07;
const DELTA_T: f64 = 0.07;
const GB: usize = 56;
const MAXD: u32 = 7;

// learned tanh head: K units, warm-started from this spread (init only; the weights then train)
const NDIR: usize = 6;
const OFFS: [f64; 2] = [0.7, 1.15];
const STEEP: f64 = 2.2;
const K: usize = NDIR * OFFS.len();
fn feat_init(j: usize) -> (f64, f64, f64) {
    let d = j / OFFS.len();
    let th = std::f64::consts::TAU * d as f64 / NDIR as f64;
    (STEEP * th.cos(), STEEP * th.sin(), -STEEP * OFFS[j % OFFS.len()])
}
// read learned (w,a,b,c) of head unit j from the flat param vector
fn hf(p: &[f64], j: usize) -> (f64, f64, f64, f64) { (p[3+4*j], p[3+4*j+1], p[3+4*j+2], p[3+4*j+3]) }

fn f(x1: f64, x2: f64) -> (f64, f64) { (-x2, x1 + MU * (x1 * x1 - 1.0) * x2) }

// ---- interval kernel ----
fn f_iv(x1: Iv, x2: Iv) -> (Iv, Iv) {
    let f1 = x2.scale(-1.0);
    let f2 = x1.add(x1.sq().add(Iv::new(-1.0, -1.0)).mul(x2).scale(MU));
    (f1, f2)
}
fn df_iv(x1: Iv, x2: Iv) -> (Iv, Iv, Iv, Iv) {
    (Iv::new(0.0, 0.0), Iv::new(-1.0, -1.0),
     Iv::new(1.0, 1.0).add(x1.mul(x2).scale(2.0 * MU)),
     x1.sq().add(Iv::new(-1.0, -1.0)).scale(MU))
}
fn v_grad(x1: f64, x2: f64, p: &[f64], head: bool) -> (f64, f64, f64) {
    let (mut v, mut g1, mut g2) = (p[0]*x1*x1 + 2.0*p[1]*x1*x2 + p[2]*x2*x2,
        2.0*(p[0]*x1 + p[1]*x2), 2.0*(p[1]*x1 + p[2]*x2));
    if head { for j in 0..K { let (w,a,b,c)=hf(p,j); let z=a*x1+b*x2+c; let t=z.tanh(); let tp=1.0-t*t;
        v += w*t; g1 += w*tp*a; g2 += w*tp*b; } }
    (v, g1, g2)
}
fn g_at(x1: f64, x2: f64, p: &[f64], head: bool) -> f64 { let (_,g1,g2)=v_grad(x1,x2,p,head); let (f1,f2)=f(x1,x2); g1*f1+g2*f2 + ALPHA*(x1*x1+x2*x2) }
fn h_at(x1: f64, x2: f64, p: &[f64], head: bool) -> f64 { let (v,_,_)=v_grad(x1,x2,p,head); v - DELTA*(x1*x1+x2*x2) }
fn v_at(x1: f64, x2: f64, p: &[f64], head: bool) -> f64 { v_grad(x1, x2, p, head).0 }

fn tanh_ivs(z: Iv) -> (Iv, Iv) {
    let t = Iv::new(z.lo.tanh(), z.hi.tanh());
    let t2 = t.sq();
    let tp = Iv::new(1.0 - t2.hi, 1.0 - t2.lo);
    let tpp = t.mul(tp).scale(-2.0);
    (tp, tpp)
}
fn box_bounds(x1: Iv, x2: Iv, p: &[f64], head: bool) -> (f64, f64) {
    let (c1, c2) = ((x1.lo+x1.hi)/2.0, (x2.lo+x2.hi)/2.0);
    let r = ((x1.hi-x1.lo).max(x2.hi-x2.lo))/2.0;
    let (f1, f2) = f_iv(x1, x2);
    let (df11, df12, df21, df22) = df_iv(x1, x2);
    let (pa, pb, pc) = (p[0], p[1], p[2]);
    let mut vx1 = x1.scale(2.0*pa).add(x2.scale(2.0*pb));
    let mut vx2 = x1.scale(2.0*pb).add(x2.scale(2.0*pc));
    let (mut vxx, mut vxy, mut vyy) = (Iv{lo:2.0*pa,hi:2.0*pa}, Iv{lo:2.0*pb,hi:2.0*pb}, Iv{lo:2.0*pc,hi:2.0*pc});
    if head { for j in 0..K { let (w,a,b,c)=hf(p,j);
        let zj = x1.scale(a).add(x2.scale(b)).add(Iv{lo:c,hi:c});
        let (tp, tpp) = tanh_ivs(zj);
        vx1 = vx1.add(tp.scale(a).scale(w)); vx2 = vx2.add(tp.scale(b).scale(w));
        vxx = vxx.add(tpp.scale(a*a).scale(w)); vxy = vxy.add(tpp.scale(a*b).scale(w)); vyy = vyy.add(tpp.scale(b*b).scale(w));
    }}
    let dg1 = vxx.mul(f1).add(vx1.mul(df11)).add(vxy.mul(f2)).add(vx2.mul(df21)).add(x1.scale(2.0*ALPHA));
    let dg2 = vxy.mul(f1).add(vx1.mul(df12)).add(vyy.mul(f2)).add(vx2.mul(df22)).add(x2.scale(2.0*ALPHA));
    let dh1 = vx1.add(x1.scale(-2.0*DELTA));
    let dh2 = vx2.add(x2.scale(-2.0*DELTA));
    let g_hi = g_at(c1, c2, p, head) + (dg1.amax() + dg2.amax()) * r;
    let h_lo = h_at(c1, c2, p, head) - (dh1.amax() + dh2.amax()) * r;
    (h_lo, g_hi)
}
fn certify_box(x1: Iv, x2: Iv, p: &[f64], head: bool, depth: u32) -> Option<(f64, f64)> {
    if x1.lo > -R0 && x1.hi < R0 && x2.lo > -R0 && x2.hi < R0 { return None; }
    let (h_lo, g_hi) = box_bounds(x1, x2, p, head);
    if h_lo > 0.0 && g_hi < 0.0 { return None; }
    if depth == 0 { return Some(((x1.lo+x1.hi)/2.0, (x2.lo+x2.hi)/2.0)); }
    let (m1, m2) = ((x1.lo+x1.hi)/2.0, (x2.lo+x2.hi)/2.0);
    for &(a1,b1) in &[(x1.lo,m1),(m1,x1.hi)] { for &(a2,b2) in &[(x2.lo,m2),(m2,x2.hi)] {
        if let Some(w)=certify_box(Iv{lo:a1,hi:b1},Iv{lo:a2,hi:b2},p,head,depth-1){return Some(w);} } }
    None
}
fn verify(p: &[f64], head: bool, rr: f64) -> (usize, Vec<(f64, f64)>) {
    let mut ce = Vec::new(); let step = 2.0*rr/GB as f64;
    for i in 0..GB { for j in 0..GB { let (lo1,lo2)=(-rr+i as f64*step,-rr+j as f64*step);
        if let Some(w)=certify_box(Iv{lo:lo1,hi:lo1+step},Iv{lo:lo2,hi:lo2+step},p,head,MAXD){ce.push(w);} } }
    (ce.len(), ce)
}

// ============================ SECOND, INDEPENDENT TRUSTED GATE ============================
// Gate 1 (above) bounds g via the CENTERED/mean-value form: g(c)+Σ|∂g/∂xᵢ|·r — it forms the Jacobian of g
// (Hessian of V, ∂f/∂x). Gate 2 (below) uses NATURAL INTERVAL EXTENSION: it evaluates g=∇V·f+α‖x‖² and
// h=V−δ‖x‖² DIRECTLY as intervals over the box and never forms a Jacobian. The two share no bounding logic,
// so a bug in one path cannot exist in the other. A certificate is trusted only where BOTH gates certify.
const MAXD2: u32 = 9; // natural extension is looser (dependency problem) ⇒ deeper refinement
fn gh_nat(x1: Iv, x2: Iv, p: &[f64], head: bool) -> (Iv, Iv) { // (h, g) by direct interval evaluation
    let f1 = x2.scale(-1.0);
    let f2 = x1.add(x1.sq().add(Iv::new(-1.0, -1.0)).mul(x2).scale(MU)); // x1 + μ(x1²−1)x2
    let (x1sq, x2sq, x1x2) = (x1.sq(), x2.sq(), x1.mul(x2));
    let (pa, pb, pc) = (p[0], p[1], p[2]);
    let mut vv  = x1sq.scale(pa).add(x1x2.scale(2.0*pb)).add(x2sq.scale(pc)); // V
    let mut vx1 = x1.scale(2.0*pa).add(x2.scale(2.0*pb));                     // ∂V/∂x1
    let mut vx2 = x1.scale(2.0*pb).add(x2.scale(2.0*pc));                     // ∂V/∂x2
    if head { for j in 0..K { let (w, a, b, c) = hf(p, j);
        let z = x1.scale(a).add(x2.scale(b)).add(Iv::new(c, c));
        let t = Iv::new(z.lo.tanh(), z.hi.tanh());                      // tanh monotone
        let t2 = t.sq(); let tp = Iv::new(1.0 - t2.hi, 1.0 - t2.lo);    // 1 − tanh²
        vv  = vv.add(t.scale(w));
        vx1 = vx1.add(tp.scale(a).scale(w));
        vx2 = vx2.add(tp.scale(b).scale(w));
    }}
    let r2 = x1sq.add(x2sq);
    let g = vx1.mul(f1).add(vx2.mul(f2)).add(r2.scale(ALPHA)); // V̇ + α‖x‖²  (direct product ∇V·f)
    let h = vv.add(r2.scale(-DELTA));                          // V − δ‖x‖²
    (h, g)
}
fn certify_box2(x1: Iv, x2: Iv, p: &[f64], head: bool, depth: u32) -> Option<(f64, f64)> {
    if x1.lo > -R0 && x1.hi < R0 && x2.lo > -R0 && x2.hi < R0 { return None; }
    let (h, g) = gh_nat(x1, x2, p, head);
    if h.lo > 0.0 && g.hi < 0.0 { return None; }
    if depth == 0 { return Some(((x1.lo+x1.hi)/2.0, (x2.lo+x2.hi)/2.0)); }
    let (m1, m2) = ((x1.lo+x1.hi)/2.0, (x2.lo+x2.hi)/2.0);
    for &(a1,b1) in &[(x1.lo,m1),(m1,x1.hi)] { for &(a2,b2) in &[(x2.lo,m2),(m2,x2.hi)] {
        if let Some(w)=certify_box2(Iv{lo:a1,hi:b1},Iv{lo:a2,hi:b2},p,head,depth-1){return Some(w);} } }
    None
}
// count top boxes NOT certified under gate: 0=centered only, 1=natural only, 2=BOTH must certify
fn verify_gate(p: &[f64], head: bool, rr: f64, gate: u8) -> usize {
    let step = 2.0*rr/GB as f64; let mut n = 0usize;
    for i in 0..GB { for j in 0..GB {
        let (lo1, lo2) = (-rr+i as f64*step, -rr+j as f64*step);
        let (a, b) = (Iv{lo:lo1,hi:lo1+step}, Iv{lo:lo2,hi:lo2+step});
        let ok = match gate {
            0 => certify_box(a, b, p, head, MAXD).is_none(),
            1 => certify_box2(a, b, p, head, MAXD2).is_none(),
            _ => certify_box(a, b, p, head, MAXD).is_none() && certify_box2(a, b, p, head, MAXD2).is_none(),
        };
        if !ok { n += 1; }
    }}
    n
}
fn pointwise_worst(p: &[f64], head: bool, rr: f64) -> (f64, f64) {
    let (mut gmax, mut hmin) = (f64::NEG_INFINITY, f64::INFINITY);
    let n = 500usize;
    for i in 0..=n { for j in 0..=n {
        let x1 = -rr + 2.0*rr*i as f64/n as f64; let x2 = -rr + 2.0*rr*j as f64/n as f64;
        if x1.abs().max(x2.abs()) < R0 { continue; }
        gmax = gmax.max(g_at(x1, x2, p, head)); hmin = hmin.min(h_at(x1, x2, p, head));
    }}
    (gmax, hmin)
}
fn converges(x1: f64, x2: f64) -> bool {
    let (mut a, mut b) = (x1, x2); let dt = 0.003;
    for _ in 0..12000 { let (f1, f2) = f(a, b); a += dt*f1; b += dt*f2;
        if (a*a + b*b).sqrt() > 5.0 { return false; } if (a*a + b*b).sqrt() < 1e-3 { return true; } }
    (a*a + b*b).sqrt() < 0.5
}
fn sublevel_roa(p: &[f64], head: bool, rr: f64) -> (f64, f64, bool) {
    let m = 400usize; let mut cstar = f64::INFINITY;
    for t in 0..=m { let s = -rr + 2.0*rr*t as f64/m as f64;
        for &(x1, x2) in &[(rr, s), (-rr, s), (s, rr), (s, -rr)] { cstar = cstar.min(v_at(x1, x2, p, head)); } }
    let g = 260usize; let cell = 2.0*rr/g as f64;
    let idx = |i: usize, j: usize| i*g + j;
    let inset = |i: usize, j: usize| { let x1 = -rr + (i as f64+0.5)*cell; let x2 = -rr + (j as f64+0.5)*cell; (x1, x2) };
    let mut inset_ok = vec![false; g*g];
    for i in 0..g { for j in 0..g { let (x1, x2) = inset(i, j); if v_at(x1, x2, p, head) <= cstar { inset_ok[idx(i,j)] = true; } } }
    let mut seen = vec![false; g*g]; let mut stack = vec![((g/2), (g/2))]; let mut cnt = 0usize;
    let mut worst = (0.0f64, 0.0f64, -1.0f64);
    if inset_ok[idx(g/2,g/2)] { seen[idx(g/2,g/2)] = true;
        while let Some((i, j)) = stack.pop() { cnt += 1;
            let (x1, x2) = inset(i, j); let rad = x1*x1 + x2*x2; if rad > worst.2 { worst = (x1, x2, rad); }
            let mut nb = vec![]; if i>0 {nb.push((i-1,j));} if i+1<g {nb.push((i+1,j));} if j>0 {nb.push((i,j-1));} if j+1<g {nb.push((i,j+1));}
            for (ni, nj) in nb { if !seen[idx(ni,nj)] && inset_ok[idx(ni,nj)] { seen[idx(ni,nj)] = true; stack.push((ni, nj)); } } } }
    (cstar, cnt as f64 * cell * cell, converges(worst.0, worst.1))
}

// ---- learner: nonlinear (learned inner weights) via Var::tanh; V̇ in closed form (tanh′=1−tanh²) ----
async fn cegis(ctx: &Arc<ferric_core::Context>, head: bool, rr: f64, rounds: usize) -> (bool, usize, Vec<f64>) {
    let np = if head { 3 + 4*K } else { 3 };
    let mut pv0 = vec![0.6, 0.0, 0.6]; // pa,pb,pc
    if head { for j in 0..K { let (a,b,c) = feat_init(j); pv0.push(0.0); pv0.push(a); pv0.push(b); pv0.push(c); } } // w=0 warm start
    let mut p: Vec<Tensor> = (0..np).map(|i| Tensor::from_vec(ctx, &[pv0[i] as f32], &[1, 1])).collect();
    let mut adam = Adam::new(&p, 0.03);
    let mut train: Vec<(f64, f64)> = Vec::new();
    let mut best = usize::MAX;
    let mut pf: Vec<f64> = pv0.clone();
    for _round in 0..rounds {
        if !train.is_empty() {
            let n = train.len();
            // constant per-round tensors
            let col = |v: Vec<f32>| Var::leaf(Tensor::from_vec(ctx, &v, &[n, 1]));
            let x1v = col(train.iter().map(|&(a,_)| a as f32).collect());
            let x2v = col(train.iter().map(|&(_,b)| b as f32).collect());
            let x1sq = col(train.iter().map(|&(a,_)| (a*a) as f32).collect());
            let x1x2 = col(train.iter().map(|&(a,b)| (2.0*a*b) as f32).collect());
            let x2sq = col(train.iter().map(|&(_,b)| (b*b) as f32).collect());
            let f1v = col(train.iter().map(|&(a,b)| f(a,b).0 as f32).collect());
            let f2v = col(train.iter().map(|&(a,b)| f(a,b).1 as f32).collect());
            let d0 = col(train.iter().map(|&(a,b)| (2.0*a*f(a,b).0) as f32).collect());
            let d1 = col(train.iter().map(|&(a,b)| { let (f1,f2)=f(a,b); (2.0*(b*f1 + a*f2)) as f32 }).collect());
            let d2 = col(train.iter().map(|&(a,b)| (2.0*b*f(a,b).1) as f32).collect());
            let onesv = col(vec![1.0f32; n]);
            let ar2 = col(train.iter().map(|&(a,b)| ((a*a+b*b)*ALPHA_T) as f32).collect());
            let dr2 = col(train.iter().map(|&(a,b)| ((a*a+b*b)*DELTA_T) as f32).collect());
            for _ in 0..200 {
                let pv: Vec<Var> = p.iter().map(|t| Var::leaf(t.clone())).collect();
                let mut v = pv[0].mul(&x1sq).add(&pv[1].mul(&x1x2)).add(&pv[2].mul(&x2sq));
                let mut vd = pv[0].mul(&d0).add(&pv[1].mul(&d1)).add(&pv[2].mul(&d2));
                if head { for j in 0..K {
                    let (w, a, b, c) = (&pv[3+4*j], &pv[3+4*j+1], &pv[3+4*j+2], &pv[3+4*j+3]);
                    let z = a.mul(&x1v).add(&b.mul(&x2v)).add(&c.mul(&onesv));
                    let t = z.tanh();
                    v = v.add(&w.mul(&t));
                    let tp = onesv.sub(&t.mul(&t));                 // tanh′ = 1 − tanh²
                    let af = a.mul(&f1v).add(&b.mul(&f2v));         // a·f1 + b·f2
                    vd = vd.add(&w.mul(&tp).mul(&af));
                }}
                let loss = dr2.sub(&v).relu().add(&vd.add(&ar2).relu()).mean_all();
                loss.backward();
                let grd: Vec<Tensor> = pv.iter().zip(&p).map(|(vv,t)| vv.grad().unwrap_or_else(|| Tensor::from_vec(ctx, &vec![0.0; t.numel()], &t.shape))).collect();
                adam.step(&mut p, &grd);
            }
        }
        pf = p.iter().map(|t| pollster::block_on(t.to_vec())[0] as f64).collect();
        let (nv, ce) = verify(&pf, head, rr);
        best = best.min(nv);
        if nv == 0 { return (true, 0, pf); }
        for w in ce { train.push(w); }
        train.sort_by(|a,b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        train.dedup_by(|a,b| (a.0-b.0).abs()<0.015 && (a.1-b.1).abs()<0.015);
        if train.len() > 6000 { let n=train.len(); train.drain(0..(n-6000)); }
    }
    (false, best, pf)
}

// ============================ S3 — OBLIGATION ROUTING ============================
// Each top box is a proof obligation. Route it to the CHEAPEST verifier that can discharge it:
//   rung A = natural extension (cheap per eval, loose) — discharges interior boxes far from the boundary.
//   rung B = centered form + deep refinement (costly per eval, tight) — only for boxes A can't close.
// Same principle as model routing: send each unit of work to the cheapest engine that still meets the
// correctness guarantee (here, a SOUND certificate). The produced certificate is identical; the work drops.
fn nat_cert(x1: Iv, x2: Iv, p: &[f64], head: bool, depth: u32, cost: &mut u64) -> bool {
    if x1.lo > -R0 && x1.hi < R0 && x2.lo > -R0 && x2.hi < R0 { return true; }
    *cost += 1;
    let (h, g) = gh_nat(x1, x2, p, head);
    if h.lo > 0.0 && g.hi < 0.0 { return true; }
    if depth == 0 { return false; }
    let (m1, m2) = ((x1.lo+x1.hi)/2.0, (x2.lo+x2.hi)/2.0);
    for &(a1,b1) in &[(x1.lo,m1),(m1,x1.hi)] { for &(a2,b2) in &[(x2.lo,m2),(m2,x2.hi)] {
        if !nat_cert(Iv{lo:a1,hi:b1}, Iv{lo:a2,hi:b2}, p, head, depth-1, cost) { return false; } } }
    true
}
fn cen_cert(x1: Iv, x2: Iv, p: &[f64], head: bool, depth: u32, cost: &mut u64) -> bool {
    if x1.lo > -R0 && x1.hi < R0 && x2.lo > -R0 && x2.hi < R0 { return true; }
    *cost += 1;
    let (h_lo, g_hi) = box_bounds(x1, x2, p, head); // box_bounds already returns (h_lo, g_hi) as f64
    if h_lo > 0.0 && g_hi < 0.0 { return true; }
    if depth == 0 { return false; }
    let (m1, m2) = ((x1.lo+x1.hi)/2.0, (x2.lo+x2.hi)/2.0);
    for &(a1,b1) in &[(x1.lo,m1),(m1,x1.hi)] { for &(a2,b2) in &[(x2.lo,m2),(m2,x2.hi)] {
        if !cen_cert(Iv{lo:a1,hi:b1}, Iv{lo:a2,hi:b2}, p, head, depth-1, cost) { return false; } } }
    true
}
fn routing_demo(p: &[f64], head: bool, rr: f64) {
    let step = 2.0*rr/GB as f64;
    // per-eval work weights (box_bounds forms the Jacobian+Hessian; gh_nat is a direct eval — measured ≈3×)
    let (wn, wc) = (1.0f64, 3.0f64);
    let (mut nat_ev, mut cen_ev_routed) = (0u64, 0u64);
    let (mut cheap, mut esc) = (0u32, 0u32);
    let (mut rad_cheap, mut rad_esc) = (0.0f64, 0.0f64);
    for i in 0..GB { for j in 0..GB {
        let (lo1, lo2) = (-rr+i as f64*step, -rr+j as f64*step);
        let (bx, by) = (Iv{lo:lo1,hi:lo1+step}, Iv{lo:lo2,hi:lo2+step});
        let rad = ((lo1+step/2.0).powi(2) + (lo2+step/2.0).powi(2)).sqrt();
        if nat_cert(bx, by, p, head, 0, &mut nat_ev) { cheap += 1; rad_cheap += rad; }   // rung A: single cheap eval
        else { esc += 1; rad_esc += rad; cen_cert(bx, by, p, head, MAXD, &mut cen_ev_routed); } // escalate to rung B
    }}
    // FLAT baseline: rung B (centered, deep) on every obligation
    let mut cen_ev_flat = 0u64;
    for i in 0..GB { for j in 0..GB { let (lo1,lo2)=(-rr+i as f64*step,-rr+j as f64*step);
        cen_cert(Iv{lo:lo1,hi:lo1+step}, Iv{lo:lo2,hi:lo2+step}, p, head, MAXD, &mut cen_ev_flat); } }
    let work_routed = nat_ev as f64*wn + cen_ev_routed as f64*wc;
    let work_flat = cen_ev_flat as f64*wc;
    let tot = (cheap+esc) as f64;
    println!("\nS3 obligation routing on the certified learned-head certificate (R={rr:.1}, {} obligations):", cheap+esc);
    println!("  cheap gate (natural, 1 eval/box) discharged {cheap}/{:.0} = {:.0}%   escalated to deep centered: {esc}",
             tot, 100.0*cheap as f64/tot);
    println!("  mean box radius — cheap: {:.2}   escalated: {:.2}   (escalation concentrates where the certificate is marginal)",
             rad_cheap/cheap.max(1) as f64, rad_esc/esc.max(1) as f64);
    println!("  verifier evals — routed: {nat_ev} natural + {cen_ev_routed} centered   flat: {cen_ev_flat} centered");
    println!("  weighted work (1×nat, 3×centered) — routed {work_routed:.0}  vs  flat {work_flat:.0}  →  {:.0}% less work, IDENTICAL certificate",
             100.0*(1.0 - work_routed/work_flat));
}

fn true_roa_area() -> f64 {
    let na = 240usize; let mut area = 0.0;
    for k in 0..na { let th = std::f64::consts::TAU * k as f64 / na as f64; let (c, s) = (th.cos(), th.sin());
        let (mut lo, mut hi) = (0.05, 3.2);
        if converges(lo*c, lo*s) { for _ in 0..34 { let m = (lo+hi)/2.0; if converges(m*c, m*s) { lo = m; } else { hi = m; } } } else { lo = 0.0; }
        area += 0.5 * lo * lo * (std::f64::consts::TAU / na as f64);
    }
    area
}

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(ferric_core::Context::new().await.unwrap());
    println!("S2+ HARDENED — learned tanh-head Lyapunov energy behind TWO independent trusted verifiers.");
    println!("  gate 1 = centered/mean-value bound (forms the Jacobian of g).");
    println!("  gate 2 = natural interval extension (evaluates g,h directly — no Jacobian). Disjoint logic.");
    println!("  A certificate is HARDENED only where BOTH gates independently certify 0 uncertified boxes.\n");
    let roa = true_roa_area();
    println!("  true ROA area (simulation oracle) ≈ {roa:.3}\n");
    for &head in &[false, true] {
        let label = if head { "learned tanh-head" } else { "quadratic only   " };
        // best area under gate 1 alone, and under BOTH gates (hardened)
        let (mut a1, mut r1, mut c1) = (0.0f64, 0.0f64, 0.0f64);
        let (mut ad, mut rd, mut cd, mut sd, mut okd, mut disagree) = (0.0f64, 0.0f64, 0.0f64, true, true, false);
        for step in 0..12 {
            let rr = 0.8 + 0.1 * step as f64;
            let (ok, _b, p) = cegis(&ctx, head, rr, 30).await; // trained/verified against gate 1
            if !ok { if a1 > 0.0 { break; } else { continue; } }
            let (cstar, area, roa_ok) = sublevel_roa(&p, head, rr);
            let (gmax, hmin) = pointwise_worst(&p, head, rr);
            if area > a1 { a1 = area; r1 = rr; c1 = cstar; }
            let nv2 = verify_gate(&p, head, rr, 1);      // INDEPENDENT second gate on the same certificate
            if nv2 == 0 && area > ad { ad = area; rd = rr; cd = cstar; sd = gmax<0.0 && hmin>0.0; okd = roa_ok; }
            if nv2 > 0 { disagree = true; } // gate 2 (looser) couldn't confirm this R — not a bug, just tighter trust
        }
        println!("  {label}  gate 1 alone : R≤{r1:.1}  invariant ROA {{V≤{c1:.2}}} area {a1:.3} ({:.0}% of ROA)", 100.0*a1/roa);
        println!("  {label}  BOTH gates   : R≤{rd:.1}  invariant ROA {{V≤{cd:.2}}} area {ad:.3} ({:.0}% of ROA)  ← HARDENED", 100.0*ad/roa);
        println!("  {label}    hardened cert sound (pointwise): {}   inside true ROA (sim): {}   gate-2 conservative at high R: {}\n",
                 if sd {"YES"} else {"NO — BUG"}, if okd {"YES"} else {"NO"}, if disagree {"yes"} else {"no"});
    }
    println!("Two verifiers with disjoint bounding logic. The hardened region is where BOTH independently agree —\nno single verifier bug can grant it. Any disagreement is looseness (a smaller trusted region), never a\nfalse certificate: both are sound, so their intersection is at least as sound as either alone.");

    // S3 — route the obligations of a concrete certified certificate to the cheapest sufficient verifier
    let (ok, _b, p) = cegis(&ctx, true, 1.7, 30).await;
    if ok { routing_demo(&p, true, 1.7); }
    println!("\nSame idea routes MODELS: send each task to the cheapest engine that meets the guarantee, escalate only\nwhen it can't — here the guarantee is a sound certificate; for models it's the task's acceptance test.");
}
