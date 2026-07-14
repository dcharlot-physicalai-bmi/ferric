//! Kernel fusion via runtime WGSL codegen — the seed of Ferric's optimizing compiler. An elementwise
//! expression over input tensors is compiled to ONE WGSL kernel and dispatched once: no per-op
//! intermediate buffers, no per-op dispatch/readback. `E::input(0).silu().mul(&E::input(1))` (a SwiGLU
//! gate) becomes a single kernel instead of two ops + a temp. SSA codegen with common-subexpression
//! sharing (each DAG node emitted once). This is what closes the perf gap with hand-fused C++/CUDA.

use crate::{empty, groups, run, unibuf, Tensor};
use std::rc::Rc;

enum Node {
    Input(usize),
    Scalar(f32),
    Un(&'static str, E),
    Bin(&'static str, E, E),
}

/// An elementwise expression (a DAG over input tensors). Build it, then `eval` compiles + runs it.
#[derive(Clone)]
pub struct E(Rc<Node>);

impl E {
    pub fn input(i: usize) -> E { E(Rc::new(Node::Input(i))) }
    pub fn scalar(s: f32) -> E { E(Rc::new(Node::Scalar(s))) }
    pub fn exp(&self) -> E { self.un("exp") }
    pub fn relu(&self) -> E { self.un("relu") }
    pub fn sigmoid(&self) -> E { self.un("sigmoid") }
    pub fn silu(&self) -> E { self.un("silu") }
    pub fn neg(&self) -> E { self.un("neg") }
    pub fn add(&self, o: &E) -> E { self.bin("+", o) }
    pub fn sub(&self, o: &E) -> E { self.bin("-", o) }
    pub fn mul(&self, o: &E) -> E { self.bin("*", o) }
    pub fn div(&self, o: &E) -> E { self.bin("/", o) }
    pub fn max(&self, o: &E) -> E { self.bin("max", o) }
    fn un(&self, op: &'static str) -> E { E(Rc::new(Node::Un(op, self.clone()))) }
    fn bin(&self, op: &'static str, o: &E) -> E { E(Rc::new(Node::Bin(op, self.clone(), o.clone()))) }
}

/// Compile the expression to WGSL (SSA, CSE by node identity) → (shader source, input count).
fn codegen(e: &E) -> (String, usize) {
    let mut body = String::new();
    let mut seen: Vec<(*const Node, usize)> = Vec::new();
    let mut counter = 0usize;
    let mut n_in = 0usize;
    fn emit(e: &E, body: &mut String, seen: &mut Vec<(*const Node, usize)>, counter: &mut usize, n_in: &mut usize) -> usize {
        let ptr = Rc::as_ptr(&e.0);
        if let Some(&(_, id)) = seen.iter().find(|(p, _)| *p == ptr) { return id; }
        let expr = match &*e.0 {
            Node::Input(i) => { *n_in = (*n_in).max(i + 1); format!("in{i}[gid]") }
            Node::Scalar(s) => format!("f32({s:?})"),
            Node::Un(op, a) => {
                let v = format!("v{}", emit(a, body, seen, counter, n_in));
                match *op {
                    "exp" => format!("exp({v})"),
                    "relu" => format!("max({v}, 0.0)"),
                    "sigmoid" => format!("1.0 / (1.0 + exp(-{v}))"),
                    "silu" => format!("{v} / (1.0 + exp(-{v}))"),
                    "neg" => format!("-{v}"),
                    _ => v,
                }
            }
            Node::Bin(op, a, b) => {
                let (x, y) = (format!("v{}", emit(a, body, seen, counter, n_in)), format!("v{}", emit(b, body, seen, counter, n_in)));
                if *op == "max" { format!("max({x}, {y})") } else { format!("({x} {op} {y})") }
            }
        };
        let id = *counter; *counter += 1;
        body.push_str(&format!("    let v{id} = {expr};\n"));
        seen.push((ptr, id));
        id
    }
    let root = emit(e, &mut body, &mut seen, &mut counter, &mut n_in);
    let mut binds = String::new();
    for k in 0..n_in { binds.push_str(&format!("@group(0) @binding({k}) var<storage,read> in{k}: array<f32>;\n")); }
    let shader = format!(
        "{binds}@group(0) @binding({n_in}) var<storage,read_write> out: array<f32>;\n\
         @group(0) @binding({}) var<uniform> info: vec4<u32>;\n\
         @compute @workgroup_size(64)\n\
         fn main(@builtin(global_invocation_id) g: vec3<u32>) {{\n\
         \x20   let gid = g.x; if (gid >= info.x) {{ return; }}\n{body}    out[gid] = v{root};\n}}\n",
        n_in + 1
    );
    (shader, n_in)
}

/// Compile the expression to one kernel and run it over `inputs` (all same shape). Returns one tensor.
pub fn eval(inputs: &[&Tensor], e: &E) -> Tensor {
    let (shader, n_in) = codegen(e);
    assert!(n_in <= inputs.len(), "expression uses more inputs than given");
    assert!(n_in + 1 <= 4, "fused kernel exceeds the 4-storage-buffer limit ({n_in} inputs)");
    let ctx = &inputs[0].ctx;
    let n = inputs[0].numel();
    let cs: Vec<Tensor> = inputs.iter().map(|t| t.contiguous()).collect();
    let out = empty(ctx, n);
    let info = unibuf(ctx, &[n as u32, 0, 0, 0]);
    let mut binds: Vec<&wgpu::Buffer> = cs.iter().map(|t| t.buf.as_ref()).collect();
    binds.push(&out);
    binds.push(&info);
    run(ctx, &shader, "fused", &binds, groups(n));
    Tensor::from_parts(ctx, out, inputs[0].shape.clone())
}
