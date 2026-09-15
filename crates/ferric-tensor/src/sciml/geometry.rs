//! **Domains, boundaries and boundary conditions** — the ergonomic layer every PINN user meets first:
//! sample the interior of a shape, sample its boundary with outward normals, and turn Dirichlet /
//! Neumann / Robin / periodic conditions into loss terms without hand-writing each one.
//!
//! Shapes are [`Geometry`] values with a membership test and a boundary sampler; [`Union`],
//! [`Difference`] and [`Intersection`] compose them (constructive solid geometry), and the composite's
//! boundary is the parts' boundaries filtered by membership in the other part — an annulus is a disk
//! minus a disk, and its boundary is the outer circle plus the inner circle with the inner normal flipped.
//!
//! ⭐ Every sampler is checked against its own geometry: a boundary point stepped a little along its
//! normal must leave the shape, and stepped against it must stay inside. That is what makes a Neumann
//! condition mean something — a normal pointing the wrong way is a flux condition of the wrong sign, and
//! nothing about a loss going down would reveal it.

use super::mse;
use super::util::{leaf, u01};
use super::deriv;
use crate::Var;
use ferric_core::Context;
use std::sync::Arc;

/// A domain in `d` dimensions.
pub trait Geometry {
    fn dim(&self) -> usize;
    fn inside(&self, x: &[f64]) -> bool;
    /// Axis-aligned bounding box, for rejection sampling.
    fn bbox(&self) -> (Vec<f64>, Vec<f64>);
    /// `n` points on the boundary with their outward unit normals, both flattened row-major.
    fn boundary(&self, n: usize, seed: u32) -> (Vec<f64>, Vec<f64>);
}

/// Hash-uniform points inside `g` by rejection from its bounding box, flattened as `f32` for the device.
pub fn interior(g: &dyn Geometry, n: usize, seed: u32) -> Vec<f32> {
    let (lo, hi) = g.bbox();
    let d = g.dim();
    let mut out = Vec::with_capacity(n * d);
    let mut i = 0u32;
    let mut tries = 0u32;
    while out.len() < n * d {
        let x: Vec<f64> = (0..d).map(|a| lo[a] + (hi[a] - lo[a]) * u01(i * d as u32 + a as u32, seed) as f64).collect();
        i += 1;
        tries += 1;
        assert!(tries < 200 * n as u32 + 10_000, "rejection sampling is not terminating: is the shape empty?");
        if g.inside(&x) {
            out.extend(x.iter().map(|&v| v as f32));
        }
    }
    out
}

/// Boundary points and normals of `g` as `f32`, ready for the device.
pub fn boundary_f32(g: &dyn Geometry, n: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
    let (p, nrm) = g.boundary(n, seed);
    (p.iter().map(|&v| v as f32).collect(), nrm.iter().map(|&v| v as f32).collect())
}

/// An axis-aligned box `lo..hi`.
pub struct BoxDomain {
    pub lo: Vec<f64>,
    pub hi: Vec<f64>,
}
impl Geometry for BoxDomain {
    fn dim(&self) -> usize {
        self.lo.len()
    }
    fn inside(&self, x: &[f64]) -> bool {
        x.iter().zip(&self.lo).zip(&self.hi).all(|((&v, &l), &h)| v >= l && v <= h)
    }
    fn bbox(&self) -> (Vec<f64>, Vec<f64>) {
        (self.lo.clone(), self.hi.clone())
    }
    fn boundary(&self, n: usize, seed: u32) -> (Vec<f64>, Vec<f64>) {
        let d = self.dim();
        let (mut p, mut nrm) = (Vec::with_capacity(n * d), Vec::with_capacity(n * d));
        for i in 0..n {
            // pick a face uniformly by area
            let areas: Vec<f64> = (0..d).map(|a| (0..d).filter(|&b| b != a).map(|b| self.hi[b] - self.lo[b]).product()).collect();
            let total: f64 = 2.0 * areas.iter().sum::<f64>();
            let mut r = u01(i as u32, seed) as f64 * total;
            let mut face = 0usize;
            let mut side = 0usize;
            'faces: for (a, &area) in areas.iter().enumerate() {
                for s in 0..2 {
                    if r < area {
                        face = a;
                        side = s;
                        break 'faces;
                    }
                    r -= area;
                }
            }
            for a in 0..d {
                if a == face {
                    p.push(if side == 0 { self.lo[a] } else { self.hi[a] });
                    nrm.push(if side == 0 { -1.0 } else { 1.0 });
                } else {
                    p.push(self.lo[a] + (self.hi[a] - self.lo[a]) * u01(i as u32 * 17 + a as u32 + 1, seed) as f64);
                    nrm.push(0.0);
                }
            }
        }
        (p, nrm)
    }
}

/// A disk (2-D) of radius `r` about `c`.
pub struct Disk {
    pub c: [f64; 2],
    pub r: f64,
}
impl Geometry for Disk {
    fn dim(&self) -> usize {
        2
    }
    fn inside(&self, x: &[f64]) -> bool {
        (x[0] - self.c[0]).powi(2) + (x[1] - self.c[1]).powi(2) <= self.r * self.r
    }
    fn bbox(&self) -> (Vec<f64>, Vec<f64>) {
        (vec![self.c[0] - self.r, self.c[1] - self.r], vec![self.c[0] + self.r, self.c[1] + self.r])
    }
    fn boundary(&self, n: usize, seed: u32) -> (Vec<f64>, Vec<f64>) {
        let (mut p, mut nrm) = (Vec::with_capacity(2 * n), Vec::with_capacity(2 * n));
        for i in 0..n {
            let th = std::f64::consts::TAU * u01(i as u32, seed) as f64;
            let (cx, sx) = (th.cos(), th.sin());
            p.extend([self.c[0] + self.r * cx, self.c[1] + self.r * sx]);
            nrm.extend([cx, sx]);
        }
        (p, nrm)
    }
}

/// `A ∪ B`.
pub struct Union<A: Geometry, B: Geometry>(pub A, pub B);
/// `A \ B`.
pub struct Difference<A: Geometry, B: Geometry>(pub A, pub B);
/// `A ∩ B`.
pub struct Intersection<A: Geometry, B: Geometry>(pub A, pub B);

fn merge_bbox(a: &dyn Geometry, b: &dyn Geometry, union: bool) -> (Vec<f64>, Vec<f64>) {
    let ((al, ah), (bl, bh)) = (a.bbox(), b.bbox());
    if union {
        (al.iter().zip(&bl).map(|(x, y)| x.min(*y)).collect(), ah.iter().zip(&bh).map(|(x, y)| x.max(*y)).collect())
    } else {
        (al, ah)
    }
}

/// Keep the boundary points of `g` for which `keep(x)` holds, `n` of them, flipping normals if asked.
fn filtered_boundary(g: &dyn Geometry, keep: &dyn Fn(&[f64]) -> bool, flip: bool, n: usize, seed: u32) -> (Vec<f64>, Vec<f64>) {
    let d = g.dim();
    let (mut p, mut nrm) = (Vec::with_capacity(n * d), Vec::with_capacity(n * d));
    let mut round = 0u32;
    while p.len() < n * d {
        let (bp, bn) = g.boundary(n, seed.wrapping_add(round.wrapping_mul(7919)));
        round += 1;
        assert!(round < 1000, "the composite boundary is empty on this part");
        for i in 0..n {
            let x = &bp[i * d..(i + 1) * d];
            if keep(x) {
                p.extend_from_slice(x);
                nrm.extend(bn[i * d..(i + 1) * d].iter().map(|&v| if flip { -v } else { v }));
                if p.len() >= n * d {
                    break;
                }
            }
        }
    }
    (p, nrm)
}

impl<A: Geometry, B: Geometry> Geometry for Union<A, B> {
    fn dim(&self) -> usize {
        self.0.dim()
    }
    fn inside(&self, x: &[f64]) -> bool {
        self.0.inside(x) || self.1.inside(x)
    }
    fn bbox(&self) -> (Vec<f64>, Vec<f64>) {
        merge_bbox(&self.0, &self.1, true)
    }
    fn boundary(&self, n: usize, seed: u32) -> (Vec<f64>, Vec<f64>) {
        let half = n / 2;
        let (mut p, mut nrm) = filtered_boundary(&self.0, &|x| !self.1.inside(x), false, half, seed);
        let (p2, n2) = filtered_boundary(&self.1, &|x| !self.0.inside(x), false, n - half, seed ^ 0xABCD);
        p.extend(p2);
        nrm.extend(n2);
        (p, nrm)
    }
}
impl<A: Geometry, B: Geometry> Geometry for Difference<A, B> {
    fn dim(&self) -> usize {
        self.0.dim()
    }
    fn inside(&self, x: &[f64]) -> bool {
        self.0.inside(x) && !self.1.inside(x)
    }
    fn bbox(&self) -> (Vec<f64>, Vec<f64>) {
        merge_bbox(&self.0, &self.1, false)
    }
    fn boundary(&self, n: usize, seed: u32) -> (Vec<f64>, Vec<f64>) {
        let half = n / 2;
        let (mut p, mut nrm) = filtered_boundary(&self.0, &|x| !self.1.inside(x), false, half, seed);
        // the hole's boundary, where it lies inside A, with the normal pointing INTO the hole (outward from A\B)
        let (p2, n2) = filtered_boundary(&self.1, &|x| self.0.inside(x), true, n - half, seed ^ 0xABCD);
        p.extend(p2);
        nrm.extend(n2);
        (p, nrm)
    }
}
impl<A: Geometry, B: Geometry> Geometry for Intersection<A, B> {
    fn dim(&self) -> usize {
        self.0.dim()
    }
    fn inside(&self, x: &[f64]) -> bool {
        self.0.inside(x) && self.1.inside(x)
    }
    fn bbox(&self) -> (Vec<f64>, Vec<f64>) {
        merge_bbox(&self.0, &self.1, false)
    }
    fn boundary(&self, n: usize, seed: u32) -> (Vec<f64>, Vec<f64>) {
        let half = n / 2;
        let (mut p, mut nrm) = filtered_boundary(&self.0, &|x| self.1.inside(x), false, half, seed);
        let (p2, n2) = filtered_boundary(&self.1, &|x| self.0.inside(x), false, n - half, seed ^ 0xABCD);
        p.extend(p2);
        nrm.extend(n2);
        (p, nrm)
    }
}

// ---------------------------------------------------------------- boundary conditions ------------

/// `u = g` on the given boundary points (`[N, d]`), as a mean-square loss.
pub fn dirichlet(ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, pts: &[f32], d: usize, g: &[f32]) -> Var {
    let n = pts.len() / d;
    mse(&fwd(&leaf(ctx, pts, &[n, d])).sub(&leaf(ctx, g, &[n, 1])))
}

/// The outward normal derivative `∂u/∂n = ∇u · n` at boundary points, as `[N, 1]`.
pub fn normal_derivative(ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, pts: &[f32], normals: &[f32], d: usize) -> Var {
    let n = pts.len() / d;
    let xv = leaf(ctx, pts, &[n, d]);
    let u = fwd(&xv);
    deriv(&u, &xv).mul(&leaf(ctx, normals, &[n, d])).sum(&[1]).reshape(&[n, 1])
}

/// `∂u/∂n = h` on the boundary — a flux condition. ⛔ Its sign is the normal's sign; see the module note.
pub fn neumann(ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, pts: &[f32], normals: &[f32], d: usize, h: &[f32]) -> Var {
    let n = pts.len() / d;
    mse(&normal_derivative(ctx, fwd, pts, normals, d).sub(&leaf(ctx, h, &[n, 1])))
}

/// `a·u + b·∂u/∂n = h` on the boundary.
#[allow(clippy::too_many_arguments)]
pub fn robin(ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, pts: &[f32], normals: &[f32], d: usize, a: f32, b: f32, h: &[f32]) -> Var {
    let n = pts.len() / d;
    let u = fwd(&leaf(ctx, pts, &[n, d]));
    let un = normal_derivative(ctx, fwd, pts, normals, d);
    let lhs = u.mul(&super::scalar(&u, a)).add(&un.mul(&super::scalar(&un, b)));
    mse(&lhs.sub(&leaf(ctx, h, &[n, 1])))
}

/// `u(left) = u(right)` for paired points — periodicity in value.
pub fn periodic(ctx: &Arc<Context>, fwd: &dyn Fn(&Var) -> Var, left: &[f32], right: &[f32], d: usize) -> Var {
    let n = left.len() / d;
    mse(&fwd(&leaf(ctx, left, &[n, d])).sub(&fwd(&leaf(ctx, right, &[n, d]))))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A boundary point stepped along its normal must leave the shape and stepped against it must stay
    /// inside — for the primitives AND for every composite. This is the check that gives a Neumann
    /// condition its sign.
    fn check_normals(g: &dyn Geometry, n: usize, eps: f64) {
        let d = g.dim();
        let (p, nrm) = g.boundary(n, 3);
        assert_eq!(p.len(), n * d);
        let mut bad = 0;
        for i in 0..n {
            let x = &p[i * d..(i + 1) * d];
            let v = &nrm[i * d..(i + 1) * d];
            let norm: f64 = v.iter().map(|a| a * a).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-9, "normal must be unit: {v:?}");
            let out: Vec<f64> = x.iter().zip(v).map(|(a, b)| a + eps * b).collect();
            let inn: Vec<f64> = x.iter().zip(v).map(|(a, b)| a - eps * b).collect();
            if g.inside(&out) || !g.inside(&inn) {
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "{bad} of {n} boundary points have a normal that does not point out of the shape");
    }

    #[test]
    fn primitives_and_csg_composites_have_outward_normals_and_consistent_membership() {
        let bx = BoxDomain { lo: vec![-1.0, -1.0], hi: vec![1.0, 1.0] };
        let disk = Disk { c: [0.0, 0.0], r: 0.5 };
        check_normals(&bx, 400, 1e-6);
        check_normals(&disk, 400, 1e-6);
        // annulus: the disk of radius 1 minus the disk of radius 0.5 — an inner boundary with flipped normal
        let annulus = Difference(Disk { c: [0.0, 0.0], r: 1.0 }, Disk { c: [0.0, 0.0], r: 0.5 });
        check_normals(&annulus, 400, 1e-6);
        assert!(annulus.inside(&[0.75, 0.0]) && !annulus.inside(&[0.25, 0.0]) && !annulus.inside(&[1.25, 0.0]));
        // a box with a disk bite taken out of one corner, and a union of two overlapping disks
        let bite = Difference(BoxDomain { lo: vec![0.0, 0.0], hi: vec![1.0, 1.0] }, Disk { c: [1.0, 1.0], r: 0.4 });
        check_normals(&bite, 400, 1e-6);
        let pair = Union(Disk { c: [-0.3, 0.0], r: 0.5 }, Disk { c: [0.3, 0.0], r: 0.5 });
        check_normals(&pair, 400, 1e-6);
        assert!(pair.inside(&[0.0, 0.0]) && pair.inside(&[-0.7, 0.0]) && !pair.inside(&[0.0, 0.6]));
        let lens = Intersection(Disk { c: [-0.3, 0.0], r: 0.5 }, Disk { c: [0.3, 0.0], r: 0.5 });
        check_normals(&lens, 400, 1e-6);
        assert!(lens.inside(&[0.0, 0.0]) && !lens.inside(&[-0.7, 0.0]));
        // interior sampling respects membership and fills the shape
        let pts = interior(&annulus, 2000, 5);
        let mut rs: Vec<f64> = pts.chunks(2).map(|p| ((p[0] as f64).powi(2) + (p[1] as f64).powi(2)).sqrt()).collect();
        rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(rs[0] >= 0.5 && rs[rs.len() - 1] <= 1.0, "interior points must lie in the annulus: {} .. {}", rs[0], rs[rs.len() - 1]);
        assert!(rs[10] < 0.55 && rs[rs.len() - 11] > 0.95, "and cover it: {} .. {}", rs[10], rs[rs.len() - 11]);
    }

    /// ⭐ **Laplace on an annulus by CSG with Dirichlet data** — exact `u = a + b ln r`, so the composite
    /// boundary sampler, the interior sampler and the Dirichlet builder are checked together against a
    /// closed form. Inner circle `u = 0`, outer `u = 1`: `u = ln(r/½) / ln 2`.
    #[ignore = "trains a PINN on the GPU (~4 min); run with -- --ignored"]
    #[test]
    fn laplace_on_an_annulus_from_csg_matches_the_closed_form() {
        use crate::sciml::util::{col, dcol, rel_l2, step, vars};
        use crate::sciml::{Act, FourierNet, LossBalancer};
        use crate::Adam;
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let (r_in, r_out) = (0.5f64, 1.0f64);
            let annulus = Difference(Disk { c: [0.0, 0.0], r: r_out }, Disk { c: [0.0, 0.0], r: r_in });
            let exact = |x: f64, y: f64| ((x * x + y * y).sqrt() / r_in).ln() / (r_out / r_in).ln();
            let colloc = interior(&annulus, 1500, 7);
            let n = colloc.len() / 2;
            let (bp, _) = boundary_f32(&annulus, 400, 11);
            let g: Vec<f32> = bp.chunks(2).map(|p| exact(p[0] as f64, p[1] as f64) as f32).collect();
            let net = FourierNet::new(&ctx, 2, 32, &[0.5, 1.5], &[64, 64], 1, Act::Tanh, 1);
            let mut wp = net.params.clone();
            let mut adam = Adam::new(&wp, 1e-3);
            let mut bal = LossBalancer::new(2, 0.1);
            for it in 0..4000usize {
                let pv = vars(&wp);
                let fwd = |x: &Var| net.forward(&pv, x);
                let xv = leaf(&ctx, &colloc, &[n, 2]);
                let u = fwd(&xv);
                let lap = dcol(&ctx, &dcol(&ctx, &u, &xv, n, 2, 0), &xv, n, 2, 0).add(&dcol(&ctx, &dcol(&ctx, &u, &xv, n, 2, 1), &xv, n, 2, 1));
                let l_res = mse(&lap);
                let l_bc = dirichlet(&ctx, &fwd, &bp, 2, &g);
                if it % 100 == 0 {
                    bal.update(&[l_res.clone(), l_bc.clone()], &pv).await;
                }
                let loss = bal.combine(&[l_res, l_bc]);
                step(&ctx, &loss, &pv, &mut wp, &mut adam).await;
            }
            let test = interior(&annulus, 2000, 99);
            let pred = net.forward(&vars(&wp), &leaf(&ctx, &test, &[2000, 2])).value().to_vec().await;
            let truth: Vec<f32> = test.chunks(2).map(|p| exact(p[0] as f64, p[1] as f64) as f32).collect();
            let e = rel_l2(&pred, &truth);
            eprintln!("  Laplace on the annulus (CSG, Dirichlet): rel-L2 {e:.4} against ln(r/½)/ln 2");
            assert!(e < 0.05, "the annulus problem must be solved through the geometry layer: {e:.4}");
            let _ = col(&ctx, &[0.0]);
        });
    }

    /// The Neumann builder's sign, on the 1-D problem `u'' = 0`, `u(0) = 0`, `u'(1) = 1`, exact `u = x`:
    /// a wrong normal sign would solve `u'(1) = −1` and return `u = −x`.
    #[ignore = "trains a small PINN on the GPU (~1 min); run with -- --ignored"]
    #[test]
    fn the_neumann_builder_has_the_sign_of_the_outward_normal() {
        use crate::sciml::util::{col, step, vars};
        use crate::sciml::{Act, Mlp};
        use crate::Adam;
        pollster::block_on(async {
            let ctx = Arc::new(Context::new().await.unwrap());
            let seg = BoxDomain { lo: vec![0.0], hi: vec![1.0] };
            let colloc = interior(&seg, 64, 3);
            let net = Mlp::new(&ctx, &[1, 32, 32, 1], 2);
            let mut wp = net.params.clone();
            let mut adam = Adam::new(&wp, 2e-3);
            for _ in 0..2500usize {
                let pv = vars(&wp);
                let fwd = |x: &Var| Mlp::forward_act(&pv, x, Act::Tanh);
                let xv = col(&ctx, &colloc);
                let u = fwd(&xv);
                let uxx = deriv(&deriv(&u, &xv), &xv);
                let l_res = mse(&uxx);
                let l_d = dirichlet(&ctx, &fwd, &[0.0], 1, &[0.0]);
                // the right end: outward normal +1, flux +1
                let l_n = neumann(&ctx, &fwd, &[1.0], &[1.0], 1, &[1.0]);
                let loss = l_res.add(&l_d.add(&l_n).mul(&super::super::scalar(&l_d, 20.0)));
                step(&ctx, &loss, &pv, &mut wp, &mut adam).await;
            }
            let pv = vars(&wp);
            let u_half = Mlp::forward_act(&pv, &col(&ctx, &[0.5]), Act::Tanh).value().to_vec().await[0];
            eprintln!("  Neumann sign: u(½) = {u_half:.4} (exact +0.5; a flipped normal gives −0.5)");
            assert!((u_half - 0.5).abs() < 0.05, "u(½) must be +0.5, got {u_half:.4}");
        });
    }
}
