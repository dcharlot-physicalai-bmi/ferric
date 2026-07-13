//! Reverse-mode automatic differentiation over the general tensor runtime — the layer that turns
//! Ferric from an inference demo into a fabric that can **train**. A `Var` wraps a `Tensor` and
//! records how to backpropagate through each op; `backward()` walks the graph in reverse and
//! accumulates gradients (broadcasting-aware). Params live as plain `Tensor`s and are re-wrapped
//! each step, so an optimizer is just tensor arithmetic.
//!
//! Validated by a finite-difference gradient check and by actually training an MLP (loss ↓).

use crate::Tensor;
use ferric_core::Context;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

type BackFn = dyn Fn(&Tensor, &[Var]);

struct Inner {
    value: Tensor,
    grad: RefCell<Option<Tensor>>,
    parents: Vec<Var>,
    backward: Option<Box<BackFn>>,
}

#[derive(Clone)]
pub struct Var(Rc<Inner>);

impl Var {
    /// A differentiable leaf (a parameter or input we want gradients for).
    pub fn leaf(t: Tensor) -> Var {
        Var(Rc::new(Inner { value: t, grad: RefCell::new(None), parents: vec![], backward: None }))
    }
    fn node(value: Tensor, parents: Vec<Var>, backward: Box<BackFn>) -> Var {
        Var(Rc::new(Inner { value, grad: RefCell::new(None), parents, backward: Some(backward) }))
    }
    pub fn value(&self) -> &Tensor { &self.0.value }
    pub fn grad(&self) -> Option<Tensor> { self.0.grad.borrow().clone() }

    fn ctx(&self) -> Arc<Context> { self.0.value.ctx_arc() }
    fn accumulate(&self, g: &Tensor) {
        let g = unbroadcast(g, &self.0.value.shape);
        let mut slot = self.0.grad.borrow_mut();
        *slot = Some(match slot.take() {
            Some(prev) => prev.add(&g),
            None => g,
        });
    }

    /// Backpropagate from this (scalar-ish) output: seed grad = ones, walk reverse-topo.
    pub fn backward(&self) {
        // post-order DFS → topo order
        let mut topo: Vec<Var> = vec![];
        let mut seen: Vec<*const Inner> = vec![];
        fn visit(v: &Var, topo: &mut Vec<Var>, seen: &mut Vec<*const Inner>) {
            let p = Rc::as_ptr(&v.0);
            if seen.contains(&p) { return; }
            seen.push(p);
            for par in &v.0.parents { visit(par, topo, seen); }
            topo.push(v.clone());
        }
        visit(self, &mut topo, &mut seen);
        let ones = Tensor::from_vec(&self.ctx(), &vec![1.0; self.0.value.numel()], &self.0.value.shape);
        *self.0.grad.borrow_mut() = Some(ones);
        for v in topo.iter().rev() {
            if let Some(bw) = &v.0.backward {
                let g = v.0.grad.borrow().clone().expect("grad set before backward");
                bw(&g, &v.0.parents);
            }
        }
    }

    // ---- differentiable ops ----
    pub fn add(&self, o: &Var) -> Var {
        let out = self.0.value.add(&o.0.value);
        Var::node(out, vec![self.clone(), o.clone()], Box::new(|g, p| { p[0].accumulate(g); p[1].accumulate(g); }))
    }
    pub fn sub(&self, o: &Var) -> Var {
        let out = self.0.value.sub(&o.0.value);
        Var::node(out, vec![self.clone(), o.clone()], Box::new(|g, p| { p[0].accumulate(g); p[1].accumulate(&g.neg()); }))
    }
    pub fn mul(&self, o: &Var) -> Var {
        let out = self.0.value.mul(&o.0.value);
        let (a, b) = (self.0.value.clone(), o.0.value.clone());
        Var::node(out, vec![self.clone(), o.clone()], Box::new(move |g, p| {
            p[0].accumulate(&g.mul(&b));
            p[1].accumulate(&g.mul(&a));
        }))
    }
    /// Matmul (last two dims; equal or absent batch dims).
    pub fn matmul(&self, o: &Var) -> Var {
        let out = self.0.value.matmul(&o.0.value);
        let (a, b) = (self.0.value.clone(), o.0.value.clone());
        Var::node(out, vec![self.clone(), o.clone()], Box::new(move |g, p| {
            let r = a.rank();
            p[0].accumulate(&g.matmul(&b.transpose(r - 1, r - 2)));   // dA = g · Bᵀ
            p[1].accumulate(&a.transpose(r - 1, r - 2).matmul(g));    // dB = Aᵀ · g
        }))
    }
    pub fn relu(&self) -> Var {
        let out = self.0.value.relu();
        let x = self.0.value.clone();
        Var::node(out, vec![self.clone()], Box::new(move |g, p| {
            // grad * (x > 0)  ==  grad * (relu(x)/x) avoided; use step = (x>0) via max(sign,0)
            let mask = x.relu_mask();
            p[0].accumulate(&g.mul(&mask));
        }))
    }
    pub fn sum_all(&self) -> Var {
        let axes: Vec<usize> = (0..self.0.value.rank()).collect();
        let out = self.0.value.sum(&axes, false);
        let shape = self.0.value.shape.clone();
        Var::node(out, vec![self.clone()], Box::new(move |g, p| {
            // g is scalar [1]; broadcast to input shape
            let full = Tensor::from_vec(&p[0].ctx(), &vec![0.0; numel(&shape)], &shape);
            p[0].accumulate(&full.add(g)); // add broadcasts scalar over full
        }))
    }
    pub fn mean_all(&self) -> Var {
        let n = self.0.value.numel() as f32;
        let s = self.sum_all();
        let inv = Var::leaf(Tensor::from_vec(&self.ctx(), &[1.0 / n], &[1]));
        s.mul(&inv)
    }
}

fn numel(s: &[usize]) -> usize { s.iter().product() }

/// Sum a gradient back down to `shape` (reverse of broadcasting).
fn unbroadcast(g: &Tensor, shape: &[usize]) -> Tensor {
    let mut out = g.clone();
    // collapse extra leading dims
    while out.rank() > shape.len() {
        out = out.sum(&[0], false);
    }
    // sum dims that were broadcast (target size 1, grad size > 1)
    let axes: Vec<usize> = (0..shape.len()).filter(|&d| shape[d] == 1 && out.shape[d] != 1).collect();
    if !axes.is_empty() {
        out = out.sum(&axes, true);
    }
    if out.shape != shape {
        out = out.reshape(shape);
    }
    out
}
