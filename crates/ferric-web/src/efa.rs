//! EFA open-weight models on browser WebGPU — the runtime-grade path for the released checkpoints.
//!
//! The pure-JS demo runs the 21k/11k-param policies on CPU; this module runs the SAME safetensors through the SAME
//! Ferric tensor ops (and second-order autograd for ∇ₐE) that run natively on Metal — now on the tab's WebGPU. For
//! models this small the CPU wins on latency (dispatch overhead dominates) and the page shows both numbers honestly;
//! the point is the runtime path that scales when the models outgrow JS.
//!
//! Exports: `efa_load(which, bytes)` (0 = hybrid-arm2, 1 = flow-arm3), `efa_hybrid_act(...)`, `efa_flow_act(...)`.
use ferric_core::Context;
use ferric_tensor::{grad, Tensor, Var};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

const H: usize = 96;
const KAPPA: f32 = 2.0;
const UMAX: f32 = 4.0;

struct Net { win: Tensor, b1: Tensor, w2: Tensor, b2: Tensor, w3: Tensor, b3: Tensor }
struct Efa { ctx: Arc<Context>, pot: Option<Net>, cor: Option<Net>, flow: Option<Net>,
    e1_flow: Option<Net>, e1_pot: Option<Net>, e1_emb: Option<Vec<f32>> }
thread_local! { static EFA: RefCell<Option<Efa>> = RefCell::new(None); }

fn parse_st(bytes: &[u8]) -> HashMap<String, Vec<f32>> {
    let hl = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&bytes[8..8 + hl]).unwrap().to_string();
    let data = &bytes[8 + hl..];
    let mut out = HashMap::new();
    let mut rest = header.as_str();
    while let Some(q) = rest.find("\"dtype\"") {
        let pre = &rest[..q]; let name_end = pre.rfind("\":{").unwrap(); let name_start = pre[..name_end].rfind('"').unwrap() + 1;
        let name = pre[name_start..name_end].to_string(); let after = &rest[q..];
        let of_s = after.find("\"data_offsets\":[").unwrap() + 16; let of_e = after[of_s..].find(']').unwrap() + of_s;
        let offs: Vec<usize> = after[of_s..of_e].split(',').map(|s| s.trim().parse().unwrap()).collect();
        let vals: Vec<f32> = data[offs[0]..offs[1]].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        out.insert(name, vals); rest = &after[of_e..];
    }
    out
}
// stack per-feature rank-1 rows "prefix.in0..in{n-1}" (each [1,h]) into one [n,h] input matrix
fn build_net_h(ctx: &Arc<Context>, t: &HashMap<String, Vec<f32>>, prefix: &str, nin: usize, nout: usize, h: usize) -> Net {
    let mut win = Vec::with_capacity(nin * h);
    for c in 0..nin { win.extend_from_slice(&t[&format!("{prefix}.in{c}")]); }
    Net {
        win: Tensor::from_vec(ctx, &win, &[nin, h]),
        b1: Tensor::from_vec(ctx, &t[&format!("{prefix}.b1")], &[h]),
        w2: Tensor::from_vec(ctx, &t[&format!("{prefix}.w2")], &[h, h]),
        b2: Tensor::from_vec(ctx, &t[&format!("{prefix}.b2")], &[h]),
        w3: Tensor::from_vec(ctx, &t[&format!("{prefix}.w3")], &[h, nout]),
        b3: Tensor::from_vec(ctx, &t[&format!("{prefix}.b3")], &[nout]),
    }
}
fn build_net(ctx: &Arc<Context>, t: &HashMap<String, Vec<f32>>, prefix: &str, nin: usize, nout: usize) -> Net {
    build_net_h(ctx, t, prefix, nin, nout, H)
}
fn fwd(n: &Net, f: &Var) -> Var {
    f.matmul(&Var::leaf(n.win.clone())).add(&Var::leaf(n.b1.clone())).relu()
        .matmul(&Var::leaf(n.w2.clone())).add(&Var::leaf(n.b2.clone())).relu()
        .matmul(&Var::leaf(n.w3.clone())).add(&Var::leaf(n.b3.clone()))
}

/// Initialize the WebGPU context and load a released EFA checkpoint. which: 0 = hybrid-arm2, 1 = flow-arm3.
#[wasm_bindgen]
pub async fn efa_load(which: u32, bytes: Vec<u8>) -> Result<String, JsValue> {
    console_error_panic_hook::set_once();
    if EFA.with(|e| e.borrow().is_none()) {
        let ctx = Arc::new(Context::new().await.map_err(|e| JsValue::from_str(&format!("{e:?}")))?);
        EFA.with(|e| *e.borrow_mut() = Some(Efa { ctx, pot: None, cor: None, flow: None, e1_flow: None, e1_pot: None, e1_emb: None }));
    }
    let t = parse_st(&bytes);
    EFA.with(|e| {
        let mut e = e.borrow_mut(); let efa = e.as_mut().unwrap();
        if which == 0 { efa.pot = Some(build_net(&efa.ctx, &t, "potential", 11, 1)); efa.cor = Some(build_net(&efa.ctx, &t, "correction", 11, 2)); }
        else if which == 1 { efa.flow = Some(build_net(&efa.ctx, &t, "flow", 16, 3)); }
        else { // 2 = efa-1: one trunk (H=128), three bodies via learned embedding
            efa.e1_flow = Some(build_net_h(&efa.ctx, &t, "flow", 22, 3, 128));
            efa.e1_pot = Some(build_net_h(&efa.ctx, &t, "potential", 21, 1, 128));
            efa.e1_emb = Some(t["body_embedding"].clone()); }
    });
    Ok(format!("loaded {} tensors on WebGPU", t.len()))
}

fn feat2(th1: f32, th2: f32, om1: f32, om2: f32, g1: f32, g2: f32, a1: f32, a2: f32, t: f32) -> Vec<f32> {
    let (d1, d2) = (th1 - g1, th2 - g2);
    vec![d1.cos(), d1.sin(), om1, d2.cos(), d2.sin(), om2, th1.sin(), th2.sin(), a1, a2, t]
}

/// One hybrid decision on WebGPU: K=2 steps of a ← a + (−κ∇ₐE + w)/K, then the verify readout from the SAME potential.
/// Returns [u1, u2, e_policy, e_random, ms].
#[wasm_bindgen]
pub async fn efa_hybrid_act(th1: f32, th2: f32, om1: f32, om2: f32, g1: f32, g2: f32, r1: f32, r2: f32) -> Result<Vec<f32>, JsValue> {
    let t0 = js_sys::Date::now();
    let (ctx, pot, cor) = EFA.with(|e| { let e = e.borrow(); let f = e.as_ref().unwrap();
        (f.ctx.clone(), f.pot.as_ref().map(|n| (n.win.clone(), n.b1.clone(), n.w2.clone(), n.b2.clone(), n.w3.clone(), n.b3.clone())).unwrap(),
         f.cor.as_ref().map(|n| (n.win.clone(), n.b1.clone(), n.w2.clone(), n.b2.clone(), n.w3.clone(), n.b3.clone())).unwrap()) });
    let pot = Net { win: pot.0, b1: pot.1, w2: pot.2, b2: pot.3, w3: pot.4, b3: pot.5 };
    let cor = Net { win: cor.0, b1: cor.1, w2: cor.2, b2: cor.3, w3: cor.4, b3: cor.5 };
    let (mut a1, mut a2) = (0.0f32, 0.0f32);
    for k in 0..2 {
        let tt = k as f32 / 2.0;
        let fv = Var::leaf(Tensor::from_vec(&ctx, &feat2(th1, th2, om1, om2, g1, g2, a1, a2, tt), &[1, 11]));
        let e = fwd(&pot, &fv);
        let gr = grad(&e.sum_all(), &[fv.clone()], None);          // ∇feat E on WebGPU (2nd-order autograd path)
        let gvec = gr[0].value().to_vec().await;
        let w = fwd(&cor, &fv).value().to_vec().await;
        a1 += (-KAPPA * gvec[8] + w[0]) / 2.0;
        a2 += (-KAPPA * gvec[9] + w[1]) / 2.0;
    }
    a1 = a1.clamp(-UMAX, UMAX); a2 = a2.clamp(-UMAX, UMAX);
    // verify: the same potential at t=1 scores the chosen action vs a caller-supplied random action
    let ep = fwd(&pot, &Var::leaf(Tensor::from_vec(&ctx, &feat2(th1, th2, om1, om2, g1, g2, a1, a2, 1.0), &[1, 11]))).value().to_vec().await[0];
    let er = fwd(&pot, &Var::leaf(Tensor::from_vec(&ctx, &feat2(th1, th2, om1, om2, g1, g2, r1, r2, 1.0), &[1, 11]))).value().to_vec().await[0];
    Ok(vec![a1, a2, ep, er, (js_sys::Date::now() - t0) as f32])
}

// EFA-1 body-agnostic 12-wide joint encoding (4/joint: cos(θ−g), sin(θ−g), ω, sinθ; inactive joints zero)
fn feat12(nj: usize, th: &[f32], om: &[f32], g: &[f32]) -> [f32; 12] {
    let mut f = [0.0f32; 12];
    for i in 0..nj { let d = th[i] - g[i]; f[i * 4] = d.cos(); f[i * 4 + 1] = d.sin(); f[i * 4 + 2] = om[i]; f[i * 4 + 3] = th[i].sin(); } f
}
/// One EFA-1 decision on WebGPU with the shipped agency gate: flow K=1, the model's OWN potential scores it, and only
/// if E > τ does it think harder (K=4 — the L2 escalation from config.json). body = joint count (1..3).
/// r1..r3 = caller-supplied random action for the verify readout. Returns [u1,u2,u3, e, escalated, e_policy, e_random, ms].
#[wasm_bindgen]
pub async fn efa1_act(body: u32, th1: f32, th2: f32, th3: f32, o1: f32, o2: f32, o3: f32,
                      g1: f32, g2: f32, g3: f32, tau: f32, r1: f32, r2: f32, r3: f32) -> Result<Vec<f32>, JsValue> {
    let t0 = js_sys::Date::now();
    let nj = body.clamp(1, 3) as usize;
    let (ctx, fl, pot, emb) = EFA.with(|e| { let e = e.borrow(); let f = e.as_ref().unwrap();
        (f.ctx.clone(),
         f.e1_flow.as_ref().map(|n| (n.win.clone(), n.b1.clone(), n.w2.clone(), n.b2.clone(), n.w3.clone(), n.b3.clone())).unwrap(),
         f.e1_pot.as_ref().map(|n| (n.win.clone(), n.b1.clone(), n.w2.clone(), n.b2.clone(), n.w3.clone(), n.b3.clone())).unwrap(),
         f.e1_emb.clone().unwrap()) });
    let fl = Net { win: fl.0, b1: fl.1, w2: fl.2, b2: fl.3, w3: fl.4, b3: fl.5 };
    let pot = Net { win: pot.0, b1: pot.1, w2: pot.2, b2: pot.3, w3: pot.4, b3: pot.5 };
    let (th, om, g) = ([th1, th2, th3], [o1, o2, o3], [g1, g2, g3]);
    let ff = feat12(nj, &th, &om, &g); let eb = &emb[(nj - 1) * 6..(nj - 1) * 6 + 6];
    // async matmuls can't live in a sync closure — feature builders here, K-step integration awaited inline below
    let flow_once = |a: [f32; 3], t: f32| {
        let mut f = Vec::with_capacity(22); f.extend_from_slice(&ff); f.extend_from_slice(&a); f.push(t); f.extend_from_slice(eb); f
    };
    let energy_feat = |a: [f32; 3]| {
        let mut f = Vec::with_capacity(21); f.extend_from_slice(&ff); f.extend_from_slice(&a); f.extend_from_slice(eb); f
    };
    // K=1
    let mut a = [0.0f32; 3];
    let v = fwd(&fl, &Var::leaf(Tensor::from_vec(&ctx, &flow_once(a, 0.0), &[1, 22]))).value().to_vec().await;
    for i in 0..nj { a[i] = v[i].clamp(-UMAX, UMAX); }
    let e = fwd(&pot, &Var::leaf(Tensor::from_vec(&ctx, &energy_feat(a), &[1, 21]))).value().to_vec().await[0];
    let mut escalated = 0.0f32;
    if e > tau { // L2: think harder — 4-step integration of the SAME field (t ≤ 0.75, inside the trained region)
        escalated = 1.0; let mut ak = [0.0f32; 3];
        for k in 0..4 {
            let v = fwd(&fl, &Var::leaf(Tensor::from_vec(&ctx, &flow_once(ak, k as f32 / 4.0), &[1, 22]))).value().to_vec().await;
            for i in 0..nj { ak[i] += v[i] / 4.0; } }
        for i in 0..nj { a[i] = ak[i].clamp(-UMAX, UMAX); }
    }
    // verify readout: the same potential scores the chosen action vs the caller's random one
    let ep = fwd(&pot, &Var::leaf(Tensor::from_vec(&ctx, &energy_feat(a), &[1, 21]))).value().to_vec().await[0];
    let mut rr = [0.0f32; 3]; for (i, r) in [r1, r2, r3].iter().enumerate() { if i < nj { rr[i] = *r; } }
    let er = fwd(&pot, &Var::leaf(Tensor::from_vec(&ctx, &energy_feat(rr), &[1, 21]))).value().to_vec().await[0];
    Ok(vec![a[0], a[1], a[2], e, escalated, ep, er, (js_sys::Date::now() - t0) as f32])
}

/// One flow decision on WebGPU (K=1 forward pass). Returns [u1, u2, u3, ms].
#[wasm_bindgen]
pub async fn efa_flow_act(t1: f32, t2: f32, t3: f32, o1: f32, o2: f32, o3: f32, g1: f32, g2: f32, g3: f32) -> Result<Vec<f32>, JsValue> {
    let t0 = js_sys::Date::now();
    let (ctx, fl) = EFA.with(|e| { let e = e.borrow(); let f = e.as_ref().unwrap();
        (f.ctx.clone(), f.flow.as_ref().map(|n| (n.win.clone(), n.b1.clone(), n.w2.clone(), n.b2.clone(), n.w3.clone(), n.b3.clone())).unwrap()) });
    let fl = Net { win: fl.0, b1: fl.1, w2: fl.2, b2: fl.3, w3: fl.4, b3: fl.5 };
    let (d1, d2, d3) = (t1 - g1, t2 - g2, t3 - g3);
    let f = vec![d1.cos(), d1.sin(), o1, d2.cos(), d2.sin(), o2, d3.cos(), d3.sin(), o3, t1.sin(), t2.sin(), t3.sin(), 0.0, 0.0, 0.0, 0.0];
    let v = fwd(&fl, &Var::leaf(Tensor::from_vec(&ctx, &f, &[1, 16]))).value().to_vec().await;
    Ok(vec![v[0].clamp(-UMAX, UMAX), v[1].clamp(-UMAX, UMAX), v[2].clamp(-UMAX, UMAX), (js_sys::Date::now() - t0) as f32])
}
