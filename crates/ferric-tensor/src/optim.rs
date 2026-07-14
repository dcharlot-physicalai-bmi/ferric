//! Optimizers — plain tensor arithmetic over the general runtime. Adam keeps per-parameter first/
//! second moment estimates and applies bias-corrected updates entirely on the GPU.

use crate::Tensor;

pub struct Adam {
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    t: i32,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl Adam {
    pub fn new(params: &[Tensor], lr: f32) -> Adam {
        let m = params.iter().map(|p| Tensor::zeros(&p.ctx_arc(), &p.shape)).collect();
        let v = params.iter().map(|p| Tensor::zeros(&p.ctx_arc(), &p.shape)).collect();
        Adam { lr, b1: 0.9, b2: 0.999, eps: 1e-8, t: 0, m, v }
    }

    /// One update step: `params[i] -= lr · m̂ / (√v̂ + eps)`, replacing each param tensor in place.
    pub fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        self.t += 1;
        let bc1 = 1.0 / (1.0 - self.b1.powi(self.t));
        let bc2 = 1.0 / (1.0 - self.b2.powi(self.t));
        for i in 0..params.len() {
            let g = &grads[i];
            let sc = |t: &Tensor, s: f32| t.mul(&t.scalar(s));
            // m = b1·m + (1-b1)·g ;  v = b2·v + (1-b2)·g²
            self.m[i] = sc(&self.m[i], self.b1).add(&sc(g, 1.0 - self.b1));
            self.v[i] = sc(&self.v[i], self.b2).add(&sc(&g.mul(g), 1.0 - self.b2));
            let mhat = sc(&self.m[i], bc1);
            let vhat = sc(&self.v[i], bc2);
            let update = mhat.div(&vhat.sqrt().add(&vhat.scalar(self.eps)));
            params[i] = params[i].sub(&sc(&update, self.lr));
        }
    }
}
