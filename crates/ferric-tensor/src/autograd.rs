//! Reverse-mode automatic differentiation over the general tensor runtime — the layer that turns
//! Ferric from an inference demo into a fabric that can **train**. A `Var` wraps a `Tensor` and
//! records how to backpropagate through each op; `backward()` walks the graph in reverse and
//! accumulates gradients (broadcasting-aware). Params live as plain `Tensor`s and are re-wrapped
//! each step, so an optimizer is just tensor arithmetic.
//!
//! **Second-order.** Each op also records a *differentiable* VJP (a gradient expressed in `Var`
//! ops), so `grad(output, wrt)` returns gradients as `Var`s that can themselves be backpropped —
//! enabling Hessian-vector products, energy-conserving nets (loss on ∂H/∂x), and training-through-
//! optimization (Energy-Based Transformers). The fast first-order `backward()` path is unchanged:
//! it still uses the raw-`Tensor` closures, so ordinary training pays nothing for this.
//!
//! Validated by a finite-difference gradient check and by actually training an MLP (loss ↓).

use crate::Tensor;
use ferric_core::Context;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

type BackFn = dyn Fn(&Tensor, &[Var]);
// Differentiable VJP: given the upstream grad as a Var and the parents, return each parent's grad
// contribution as a Var (possibly still broadcast — `grad()` un-broadcasts to the parent's shape).
type VjpFn = dyn Fn(&Var, &[Var]) -> Vec<Var>;

struct Inner {
    value: Tensor,
    grad: RefCell<Option<Tensor>>,
    parents: Vec<Var>,
    backward: Option<Box<BackFn>>,
    vjp: Option<Box<VjpFn>>,
}

#[derive(Clone)]
pub struct Var(Rc<Inner>);

impl Var {
    /// A differentiable leaf (a parameter or input we want gradients for).
    pub fn leaf(t: Tensor) -> Var {
        Var(Rc::new(Inner { value: t, grad: RefCell::new(None), parents: vec![], backward: None, vjp: None }))
    }
    /// First-order-only node (no differentiable VJP recorded).
    fn node(value: Tensor, parents: Vec<Var>, backward: Box<BackFn>) -> Var {
        Var(Rc::new(Inner { value, grad: RefCell::new(None), parents, backward: Some(backward), vjp: None }))
    }
    /// Node with both the fast Tensor backward AND a differentiable Var VJP (supports second order).
    fn node_d(value: Tensor, parents: Vec<Var>, backward: Box<BackFn>, vjp: Box<VjpFn>) -> Var {
        Var(Rc::new(Inner { value, grad: RefCell::new(None), parents, backward: Some(backward), vjp: Some(vjp) }))
    }
    pub fn value(&self) -> &Tensor { &self.0.value }
    pub fn grad(&self) -> Option<Tensor> { self.0.grad.borrow().clone() }

    fn ctx(&self) -> Arc<Context> { self.0.value.ctx_arc() }
    fn zeros_like(&self) -> Var { Var::leaf(Tensor::from_vec(&self.ctx(), &vec![0.0; self.0.value.numel()], &self.0.value.shape)) }
    fn accumulate(&self, g: &Tensor) {
        let g = unbroadcast(g, &self.0.value.shape);
        let mut slot = self.0.grad.borrow_mut();
        *slot = Some(match slot.take() {
            Some(prev) => prev.add(&g),
            None => g,
        });
    }

    /// Backpropagate from this (scalar-ish) output: seed grad = ones, walk reverse-topo. (First order.)
    pub fn backward(&self) {
        let mut topo: Vec<Var> = vec![];
        let mut seen: Vec<*const Inner> = vec![];
        topo_sort(self, &mut topo, &mut seen);
        let ones = Tensor::from_vec(&self.ctx(), &vec![1.0; self.0.value.numel()], &self.0.value.shape);
        *self.0.grad.borrow_mut() = Some(ones);
        for v in topo.iter().rev() {
            if let Some(bw) = &v.0.backward {
                let g = v.0.grad.borrow().clone().expect("grad set before backward");
                bw(&g, &v.0.parents);
            }
        }
    }

    // ---- differentiable ops (each carries both a Tensor backward and a Var VJP) ----
    pub fn add(&self, o: &Var) -> Var {
        let out = self.0.value.add(&o.0.value);
        Var::node_d(out, vec![self.clone(), o.clone()],
            Box::new(|g, p| { p[0].accumulate(g); p[1].accumulate(g); }),
            Box::new(|g, _p| vec![g.clone(), g.clone()]))
    }
    pub fn sub(&self, o: &Var) -> Var {
        let out = self.0.value.sub(&o.0.value);
        Var::node_d(out, vec![self.clone(), o.clone()],
            Box::new(|g, p| { p[0].accumulate(g); p[1].accumulate(&g.neg()); }),
            Box::new(|g, _p| vec![g.clone(), g.neg()]))
    }
    pub fn mul(&self, o: &Var) -> Var {
        let out = self.0.value.mul(&o.0.value);
        let (a, b) = (self.0.value.clone(), o.0.value.clone());
        Var::node_d(out, vec![self.clone(), o.clone()],
            Box::new(move |g, p| { p[0].accumulate(&g.mul(&b)); p[1].accumulate(&g.mul(&a)); }),
            Box::new(|g, p| vec![g.mul(&p[1]), g.mul(&p[0])]))
    }
    /// Matmul (last two dims; equal or absent batch dims).
    pub fn matmul(&self, o: &Var) -> Var {
        let out = self.0.value.matmul(&o.0.value);
        let (a, b) = (self.0.value.clone(), o.0.value.clone());
        Var::node_d(out, vec![self.clone(), o.clone()],
            Box::new(move |g, p| {
                let r = a.rank();
                p[0].accumulate(&g.matmul(&b.transpose(r - 1, r - 2)));   // dA = g · Bᵀ
                p[1].accumulate(&a.transpose(r - 1, r - 2).matmul(g));    // dB = Aᵀ · g
            }),
            Box::new(|g, p| {
                let r = p[0].value().rank();
                vec![g.matmul(&p[1].transpose(r - 1, r - 2)), p[0].transpose(r - 1, r - 2).matmul(g)]
            }))
    }
    /// 2D convolution (NHWC × HWIO → NHWO, like [`Tensor::conv2d`]). First-order gradients only
    /// (no double-backward): dX is the stride-aware transposed convolution, dW the input×grad
    /// correlation — both portable WGSL. The forward routes through the tensor units under
    /// `FERRIC_METAL4` like every conv; the backward stays on the portable floor.
    pub fn conv2d(&self, w: &Var, stride: (usize, usize), pad: (usize, usize)) -> Var {
        let out = self.0.value.conv2d(&w.0.value, stride, pad);
        let (x, wt) = (self.0.value.clone(), w.0.value.clone());
        Var::node(out, vec![self.clone(), w.clone()],
            Box::new(move |g, p| {
                p[0].accumulate(&g.conv2d_dx(&wt, stride, pad, (x.shape[1], x.shape[2])));
                p[1].accumulate(&x.conv2d_dw(g, stride, pad, (wt.shape[0], wt.shape[1])));
            }))
    }

    pub fn relu(&self) -> Var {
        let out = self.0.value.relu();
        let x = self.0.value.clone();
        Var::node_d(out, vec![self.clone()],
            Box::new(move |g, p| { p[0].accumulate(&g.mul(&x.relu_mask())); }),
            // relu'' = 0 ⇒ the mask is a constant (detached leaf) — correct for second order.
            Box::new(|g, p| vec![g.mul(&Var::leaf(p[0].value().relu_mask()))]))
    }
    pub fn neg(&self) -> Var {
        let out = self.0.value.neg();
        Var::node_d(out, vec![self.clone()],
            Box::new(|g, p| { p[0].accumulate(&g.neg()); }),
            Box::new(|g, _p| vec![g.neg()]))
    }
    /// Transpose two dims; gradient transposes back.
    pub fn transpose(&self, a: usize, b: usize) -> Var {
        let out = self.0.value.transpose(a, b);
        Var::node_d(out, vec![self.clone()],
            Box::new(move |g, p| { p[0].accumulate(&g.transpose(a, b)); }),
            Box::new(move |g, _p| vec![g.transpose(a, b)]))
    }
    /// Reshape; gradient reshapes back. (Used by second-order un-broadcasting.)
    pub fn reshape(&self, shape: &[usize]) -> Var {
        let out = self.0.value.reshape(shape);
        let ishape = self.0.value.shape.clone();
        let ishape2 = ishape.clone();
        Var::node_d(out, vec![self.clone()],
            Box::new(move |g, p| { p[0].accumulate(&g.reshape(&ishape)); }),
            Box::new(move |g, _p| vec![g.reshape(&ishape2)]))
    }
    pub fn div(&self, o: &Var) -> Var {
        let out = self.0.value.div(&o.0.value);
        let (a, b) = (self.0.value.clone(), o.0.value.clone());
        Var::node_d(out, vec![self.clone(), o.clone()],
            Box::new(move |g, p| {
                p[0].accumulate(&g.div(&b));                                  // dA = g/B
                p[1].accumulate(&g.mul(&a).neg().div(&b.mul(&b)));            // dB = -g·A/B²
            }),
            Box::new(|g, p| vec![g.div(&p[1]), g.mul(&p[0]).neg().div(&p[1].mul(&p[1]))]))
    }
    pub fn exp(&self) -> Var {
        let out = self.0.value.exp();
        let o2 = out.clone();
        Var::node_d(out, vec![self.clone()],
            Box::new(move |g, p| { p[0].accumulate(&g.mul(&o2)); }),
            Box::new(|g, p| vec![g.mul(&p[0].exp())]))   // d/dx eˣ = eˣ (differentiable in x)
    }
    pub fn log(&self) -> Var {
        let out = self.0.value.log();
        let x = self.0.value.clone();
        Var::node_d(out, vec![self.clone()],
            Box::new(move |g, p| { p[0].accumulate(&g.div(&x)); }),
            Box::new(|g, p| vec![g.div(&p[0])]))          // d/dx ln x = 1/x
    }
    /// Elementwise sin. d/dx sin x = cos x. (Enables angle composition — e.g. goal = f(instruction) in-graph.)
    pub fn sin(&self) -> Var {
        let out = self.0.value.sin();
        let x = self.0.value.clone();
        Var::node_d(out, vec![self.clone()],
            Box::new(move |g, p| { p[0].accumulate(&g.mul(&x.cos())); }),
            Box::new(|g, p| vec![g.mul(&p[0].cos())]))    // d/dx sin x = cos x (differentiable in x)
    }
    /// Elementwise cos. d/dx cos x = −sin x.
    pub fn cos(&self) -> Var {
        let out = self.0.value.cos();
        let x = self.0.value.clone();
        Var::node_d(out, vec![self.clone()],
            Box::new(move |g, p| { p[0].accumulate(&g.mul(&x.sin()).neg()); }),
            Box::new(|g, p| vec![g.mul(&p[0].sin()).neg()]))  // d/dx cos x = −sin x (differentiable in x)
    }
    /// Elementwise sqrt. d/dx √x = 0.5/√x. (Enables L2-normalization and VICReg std.)
    pub fn sqrt(&self) -> Var {
        let out = self.0.value.sqrt();
        let o2 = out.clone();
        Var::node_d(out, vec![self.clone()],
            Box::new(move |g, p| { let half = o2.scalar(0.5); p[0].accumulate(&g.mul(&half).div(&o2)); }),
            Box::new(|g, p| { let s = p[0].sqrt(); vec![g.mul(&Var::leaf(s.0.value.scalar(0.5))).div(&s)] }))
    }
    /// Sum over `axes` (keepdim); gradient broadcasts back to the input shape.
    pub fn sum(&self, axes: &[usize]) -> Var {
        let out = self.0.value.sum(axes, true);
        let ishape = self.0.value.shape.clone();
        Var::node_d(out, vec![self.clone()],
            Box::new(move |g, p| { p[0].accumulate(&g.broadcast_to(&ishape)); }),
            // broadcast g back to the input shape via zeros(ishape) + g (add broadcasts) — differentiable.
            Box::new(|g, p| vec![p[0].zeros_like().add(g)]))
    }
    pub fn mean(&self, axes: &[usize]) -> Var {
        let n: f32 = axes.iter().map(|&d| self.0.value.shape[d]).product::<usize>() as f32;
        let s = self.sum(axes);
        s.mul(&Var::leaf(s.0.value.scalar(1.0 / n)))
    }
    /// A non-differentiable copy (stop-gradient) — for the detached max in a stable softmax.
    pub fn detach(&self) -> Var { Var::leaf(self.0.value.clone()) }
    /// Numerically-stable softmax over `axis`, fully differentiable (built from primitives).
    pub fn softmax(&self, axis: usize) -> Var {
        let m = Var::leaf(self.0.value.max(&[axis], true)); // detached max
        let e = self.sub(&m).exp();
        e.div(&e.sum(&[axis]))
    }
    pub fn sum_all(&self) -> Var {
        let axes: Vec<usize> = (0..self.0.value.rank()).collect();
        let out = self.0.value.sum(&axes, false);
        let shape = self.0.value.shape.clone();
        Var::node_d(out, vec![self.clone()],
            Box::new(move |g, p| {
                let full = Tensor::from_vec(&p[0].ctx(), &vec![0.0; numel(&shape)], &shape);
                p[0].accumulate(&full.add(g)); // add broadcasts scalar over full
            }),
            Box::new(|g, p| vec![p[0].zeros_like().add(g)]))  // broadcast scalar [1] over input
    }
    pub fn mean_all(&self) -> Var {
        let n = self.0.value.numel() as f32;
        let s = self.sum_all();
        let inv = Var::leaf(Tensor::from_vec(&self.ctx(), &[1.0 / n], &[1]));
        s.mul(&inv)
    }
}

/// **Second-order gradient.** Returns d`output`/d`wrt[i]` as *differentiable* `Var`s: the returned
/// gradients are themselves graph nodes, so a loss built from them can be `backward()`-ed to get the
/// second derivative (Hessian-vector products, ∂H/∂x losses, training-through-optimization).
/// `grad_output` seeds the reverse pass (defaults to ones). Requires every op on the path to `wrt`
/// to carry a VJP — all elementwise/matmul/reduce primitives above do.
pub fn grad(output: &Var, wrt: &[Var], grad_output: Option<&Var>) -> Vec<Var> {
    let mut topo: Vec<Var> = vec![];
    let mut seen: Vec<*const Inner> = vec![];
    topo_sort(output, &mut topo, &mut seen);
    // accumulator: node ptr -> grad Var
    let mut gmap: Vec<(*const Inner, Var)> = vec![];
    let seed = match grad_output {
        Some(g) => g.clone(),
        None => Var::leaf(Tensor::from_vec(&output.ctx(), &vec![1.0; output.0.value.numel()], &output.0.value.shape)),
    };
    gmap.push((Rc::as_ptr(&output.0), seed));
    for v in topo.iter().rev() {
        let gv = gmap.iter().find(|(p, _)| *p == Rc::as_ptr(&v.0)).map(|(_, g)| g.clone());
        let gv = match gv { Some(g) => g, None => continue };
        if let Some(vjp) = &v.0.vjp {
            let contribs = vjp(&gv, &v.0.parents);
            for (par, contrib) in v.0.parents.iter().zip(contribs) {
                let c = unbroadcast_var(&contrib, &par.0.value.shape);
                let pp = Rc::as_ptr(&par.0);
                if let Some(entry) = gmap.iter_mut().find(|(p, _)| *p == pp) {
                    entry.1 = entry.1.add(&c);
                } else {
                    gmap.push((pp, c));
                }
            }
        }
    }
    wrt.iter().map(|w| {
        gmap.iter().find(|(p, _)| *p == Rc::as_ptr(&w.0)).map(|(_, g)| g.clone())
            .unwrap_or_else(|| w.zeros_like())
    }).collect()
}

fn topo_sort(v: &Var, topo: &mut Vec<Var>, seen: &mut Vec<*const Inner>) {
    let p = Rc::as_ptr(&v.0);
    if seen.contains(&p) { return; }
    seen.push(p);
    for par in &v.0.parents { topo_sort(par, topo, seen); }
    topo.push(v.clone());
}

fn numel(s: &[usize]) -> usize { s.iter().product() }

/// Differentiable reverse of broadcasting: sum a grad `Var` back down to `target` (Var-space twin of
/// `unbroadcast`). Sums the extra leading dims and the broadcast (size-1) dims, then reshapes.
fn unbroadcast_var(g: &Var, target: &[usize]) -> Var {
    let gs = g.0.value.shape.clone();
    if gs == target { return g.clone(); }
    let off = gs.len().saturating_sub(target.len());
    let mut axes: Vec<usize> = (0..off).collect();
    for i in 0..target.len() { if target[i] == 1 && gs[off + i] > 1 { axes.push(off + i); } }
    let summed = if axes.is_empty() { g.clone() } else { g.sum(&axes) }; // keepdim → 1s where summed
    if summed.0.value.shape == target { summed } else { summed.reshape(target) }
}

/// Sum a gradient back down to `shape` (reverse of broadcasting). First-order (Tensor) path.
fn unbroadcast(g: &Tensor, shape: &[usize]) -> Tensor {
    let mut out = g.clone();
    while out.rank() > shape.len() {
        out = out.sum(&[0], false);
    }
    let axes: Vec<usize> = (0..shape.len()).filter(|&d| shape[d] == 1 && out.shape[d] != 1).collect();
    if !axes.is_empty() {
        out = out.sum(&axes, true);
    }
    if out.shape != shape {
        out = out.reshape(shape);
    }
    out
}
