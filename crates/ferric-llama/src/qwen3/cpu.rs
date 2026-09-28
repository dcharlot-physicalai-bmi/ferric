//! **The Dense runtime on the CPU fabric** — `FERRIC_CPU=1`, and every caller of [`Qwen3`] (the CLI,
//! the server, the conformance gates) runs the model on CPU cores instead of the GPU.
//!
//! ## Why a tier, and not a CPU `Tensor`
//!
//! Ferric's `Tensor` IS a wgpu buffer: its ~110 ops each record a GPU dispatch, and a CPU storage
//! variant would touch every one of them. This codebase already has the pattern for a second compute
//! device running this model — the NVIDIA native tier (`native_step`/`native_prefill`): the runtime's
//! entry points hand the step to the device when it is selected, and the device's graph is built from
//! the runtime's OWN decisions. This module follows it, closer than the NVIDIA tier can:
//!
//!   - it is a child module of `qwen3`, so it calls the runtime's private helpers rather than copying
//!     them: the embedding rows (`embed_rows`, Gemma's √d and Glimmer's weightless norm included), the
//!     rope angles (`rope_host`, the authors' float32 inverse frequencies, Llama-3 / YaRN / linear /
//!     LongRoPE scaling and the attention factor), the per-layer rope base / window / NoPE
//!     (`layer_attn_plan`, with the `FERRIC_ONE_ROPE` / `FERRIC_NO_SWA` controls), the pairing
//!     (`rope_plan`, with `FERRIC_NEOX` / `FERRIC_ROPE_NORM`), the LongRoPE cache refill and the cache's
//!     position/history accounting (`longrope_refill`, `advance`);
//!   - its weights are loaded INSTEAD of the GPU ones (`Qwen3::load` with `FERRIC_CPU=1`), so the model
//!     is not resident twice;
//!   - what it does not implement it REFUSES by name at load (multimodal rope) or per call (LoRA
//!     adapters, a quantized KV cache, the stepped and embeddings-in forwards), never approximates.
//!
//! The one thing that still touches wgpu is the return type: the entry points return a `Tensor`, so a
//! context is created and the logits uploaded. [`Qwen3::cpu_logits`] is the host-side entry that skips
//! that round trip (what a GPU-less host, and the benchmark, call).
//!
//! ## Numerics
//!
//! Everything outside the matmuls runs in f32 on the host with the GPU path's formulas: RMSNorm
//! (`x / sqrt(mean(x²) + eps) · w`, the mean accumulated in f64), rotation from the shared host tables,
//! softmax attention with `1/sqrt(head_dim)` scaling, SwiGLU / GEGLU(tanh), the post-norms and softcaps.
//! The matmuls are `ferric_tensor::cpu_q` — see its module doc for the int8-activation arithmetic on
//! quantized weights and `FERRIC_CPU_F32ACT`.
use super::{Cache, Cfg, Qwen3, RopePlan, RopeRows, layer_attn_plan, rope_on_device, rope_plan};
use ferric_gguf::GgufSource;
use ferric_tensor::cpu_q::{self, QWeight, SyncPtr, WType};
use ferric_tensor::Tensor;

/// Whether dense models load onto the CPU fabric: `FERRIC_CPU` set to anything but ""/"0".
///
/// ⚠ An env var, like `FERRIC_CUDA`, because it must reach every caller (CLI, server, gates) without a
/// signature change. `FERRIC_CPU=` (set but empty) means OFF — `is_ok()` would read it as on.
pub(crate) fn selected() -> bool {
    matches!(std::env::var("FERRIC_CPU").as_deref(), Ok(v) if !v.is_empty() && v != "0")
}

fn vecf(g: &impl GgufSource, name: &str) -> Result<Vec<f32>, String> { g.dequant(name) }

/// A weight in its stored format when this fabric has a kernel for it; otherwise dequantized to F32 —
/// said once per format, because a 4x memory cost should not be silent.
fn qw(g: &impl GgufSource, name: &str) -> Result<QWeight, String> {
    let t = g.tensor(name).ok_or_else(|| format!("no tensor '{name}'"))?;
    let (ty, cols, rows) = (t.ggml_type, t.dims[0] as usize, t.dims.get(1).copied().unwrap_or(1) as usize);
    match WType::from_ggml(ty) {
        Some(w) if cols % w.block().0 == 0 => QWeight::new(w, rows, cols, g.raw(name)?),
        _ => {
            static SAID: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());
            if let Ok(mut s) = SAID.lock() {
                if !s.contains(&ty) {
                    s.push(ty);
                    eprintln!("cpu: ggml type {ty} ({name}) has no CPU kernel — dequantized to F32 at load (4 bytes/weight)");
                }
            }
            Ok(QWeight::from_f32(rows, cols, &g.dequant(name)?))
        }
    }
}

struct CpuLayer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    wq: QWeight,
    wk: QWeight,
    wv: QWeight,
    /// Qwen2's q | k | v biases, each its own width.
    bias: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    wo: QWeight,
    gate: QWeight,
    up: QWeight,
    down: QWeight,
    attn_gate: Option<QWeight>,
    post_attn_norm: Option<Vec<f32>>,
    post_ffn_norm: Option<Vec<f32>>,
    rope_base: f32,
    window: usize,
    rope: bool,
}

impl CpuLayer {
    fn load(g: &impl GgufSource, cfg: &Cfg, il: usize) -> Result<CpuLayer, String> {
        let b = |s: &str| format!("blk.{il}.{s}");
        let (q_out, kv_out) = (cfg.n_head * cfg.head_dim, cfg.n_head_kv * cfg.head_dim);
        // Phi-3 stores q|k|v as ONE `attn_qkv` and gate|up as ONE `ffn_up` (gate first). Rows are
        // independent, so the CPU splits them into their parts at load — a byte split, exact.
        let (wq, wk, wv) = if g.tensor(&b("attn_qkv.weight")).is_some() {
            let mut p = qw(g, &b("attn_qkv.weight"))?.split_rows(&[q_out, kv_out, kv_out])?.into_iter();
            (p.next().unwrap(), p.next().unwrap(), p.next().unwrap())
        } else {
            (qw(g, &b("attn_q.weight"))?, qw(g, &b("attn_k.weight"))?, qw(g, &b("attn_v.weight"))?)
        };
        let (gate, up) = if g.tensor(&b("ffn_gate.weight")).is_some() {
            (qw(g, &b("ffn_gate.weight"))?, qw(g, &b("ffn_up.weight"))?)
        } else {
            let mut p = qw(g, &b("ffn_up.weight"))?.split_rows(&[cfg.n_ff, cfg.n_ff])?.into_iter();
            (p.next().unwrap(), p.next().unwrap())
        };
        let (rope_base, window, rope) = layer_attn_plan(g, cfg, il);
        Ok(CpuLayer {
            attn_norm: vecf(g, &b("attn_norm.weight"))?,
            ffn_norm: vecf(g, &b("ffn_norm.weight"))?,
            q_norm: if cfg.has_qk_norm { Some(vecf(g, &b("attn_q_norm.weight"))?) } else { None },
            k_norm: if cfg.has_qk_norm { Some(vecf(g, &b("attn_k_norm.weight"))?) } else { None },
            bias: if cfg.qkv_bias {
                Some((vecf(g, &b("attn_q.bias"))?, vecf(g, &b("attn_k.bias"))?, vecf(g, &b("attn_v.bias"))?))
            } else { None },
            wq, wk, wv,
            wo: qw(g, &b("attn_output.weight"))?,
            gate, up,
            down: qw(g, &b("ffn_down.weight"))?,
            attn_gate: if g.tensor(&b("attn_gate.weight")).is_some() { Some(qw(g, &b("attn_gate.weight"))?) } else { None },
            post_attn_norm: if cfg.post_norms { Some(vecf(g, &b("post_attention_norm.weight"))?) } else { None },
            post_ffn_norm: if cfg.post_norms { Some(vecf(g, &b("post_ffw_norm.weight"))?) } else { None },
            rope_base, window, rope,
        })
    }
    fn nbytes(&self) -> usize {
        [&self.wq, &self.wk, &self.wv, &self.wo, &self.gate, &self.up, &self.down].iter().map(|w| w.nbytes()).sum::<usize>()
            + self.attn_gate.as_ref().map_or(0, |w| w.nbytes())
    }
}

/// The model's weights on the host, in their GGUF formats.
pub(crate) struct CpuModel {
    layers: Vec<CpuLayer>,
    out_norm: Vec<f32>,
    lm_head: QWeight,
}

impl CpuModel {
    pub(crate) fn load(g: &impl GgufSource, cfg: &Cfg) -> Result<CpuModel, String> {
        if cfg.mrope_sections.is_some() { return Err("multimodal rope is not on the CPU fabric".into()); }
        let head = if g.tensor("output.weight").is_some() { "output.weight" } else { "token_embd.weight" };
        let layers = (0..cfg.n_layer).map(|il| CpuLayer::load(g, cfg, il)).collect::<Result<Vec<_>, _>>()?;
        let m = CpuModel { layers, out_norm: vecf(g, "output_norm.weight")?, lm_head: qw(g, head)? };
        let mut fmts: Vec<&str> = m.layers.iter()
            .flat_map(|l| [l.wq.ty, l.wk.ty, l.wv.ty, l.wo.ty, l.gate.ty, l.up.ty, l.down.ty])
            .chain([m.lm_head.ty]).map(|t| t.name()).collect();
        fmts.sort(); fmts.dedup();
        eprintln!("cpu: {} layers, {:.1} MB of weights ({}), {} threads, {} kernels",
                  m.layers.len(), (m.layers.iter().map(|l| l.nbytes()).sum::<usize>() + m.lm_head.nbytes()) as f64 / 1e6,
                  fmts.join("/"), cpu_q::pool().threads(), cpu_q::kernel_family());
        Ok(m)
    }
}

/// One sequence's K/V on the host: per layer, `[len, n_head_kv·head_dim]` row-major K and V.
#[derive(Default)]
pub struct CpuKv {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    len: usize,
}

impl CpuKv {
    fn new(n_layer: usize) -> CpuKv {
        CpuKv { k: vec![Vec::new(); n_layer], v: vec![Vec::new(); n_layer], len: 0 }
    }
    /// Bring the store in line with the cache before a forward. Rows past `pos` are dead (a rewind —
    /// speculative drafts rejected, `Cache::truncate`, a LongRoPE refill) and are dropped.
    ///
    /// Rows are STORED rows, not positions — as in the GPU store (`KvBuf`): a cache positioned by hand
    /// at P with nothing in it (`lm_logits`' `position_offset`, the authors' `position_ids = P..`) holds
    /// the new rows at 0.., rotated for P... and attends over just them. What cannot happen is rows in
    /// the GPU store (`gpu_rows`, a prefix-cache seed / `set_layers`) that this store lacks: attending
    /// over that hole would be fluent and wrong, so it panics, naming why.
    fn sync(&mut self, pos: usize, width: usize, gpu_rows: usize) {
        if self.len > pos {
            for l in self.k.iter_mut().chain(self.v.iter_mut()) { l.truncate(pos * width); }
            self.len = pos;
        }
        assert!(gpu_rows <= self.len,
                "CPU fabric: the GPU K/V of this sequence holds {gpu_rows} rows the CPU store does not ({}) — rows \
                 were installed by a path that does not run on the CPU fabric (a prefix-cache seed, set_layers)", self.len);
    }
    /// Bytes held (f32 rows).
    pub fn bytes(&self) -> usize { self.k.iter().chain(&self.v).map(|l| l.len() * 4).sum() }
}

/// Which rows of a forward get the LM head.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Out {
    /// Every fed row (`forward_cached`, `forward`).
    All,
    /// The last row (`forward_cached_last`).
    Last,
    /// No head: `out_norm(x)` of every row (`forward_hidden`).
    Hidden,
}

// ---------------------------------------------------------------------------------------------------
// host ops
// ---------------------------------------------------------------------------------------------------

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) { for i in 0..8 { s[i] += x[i] * y[i]; } }
    let mut t = ((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7]));
    for (x, y) in ra.iter().zip(rb) { t += x * y; }
    t
}

#[inline]
fn axpy(y: &mut [f32], a: f32, x: &[f32]) { for (o, &v) in y.iter_mut().zip(x) { *o += a * v; } }

/// `rows` rows of width `d`: `x / sqrt(mean(x²) + eps) · w`. The mean is accumulated in f64: it costs
/// nothing at these widths and removes the reduction order from the comparison with the authors.
fn rmsnorm_row(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ms = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / ((ms as f32) + eps).sqrt();
    for ((o, &v), &g) in out.iter_mut().zip(x).zip(w) { *o = v * inv * g; }
}

/// Run `f(r0, r1)` over row ranges of `rows`, in parallel when the work is worth a dispatch.
fn par_rows(rows: usize, row_cost: usize, f: impl Fn(usize, usize) + Sync) {
    if rows == 0 { return; }
    if rows * row_cost < (1 << 16) || rows == 1 { f(0, rows); return; }
    let n = cpu_q::pool().threads();
    let chunk = rows.div_ceil(n * 2).max(1);
    let items = rows.div_ceil(chunk);
    cpu_q::pool().for_each(items, |i| f(i * chunk, ((i + 1) * chunk).min(rows)));
}

fn rmsnorm_rows(x: &[f32], w: &[f32], eps: f32, d: usize, out: &mut [f32]) {
    let op = SyncPtr(out.as_mut_ptr());
    par_rows(x.len() / d, d, |r0, r1| {
        for r in r0..r1 {
            let o = unsafe { std::slice::from_raw_parts_mut(op.get().add(r * d), d) };
            rmsnorm_row(&x[r * d..(r + 1) * d], w, eps, o);
        }
    });
}

/// Rotate `heads` heads of width `hd` in each of `t` rows (row stride `rs`), angles from the `[t, hd/2]`
/// tables. NEOX pairs (c, c + hd/2); NORM pairs (2c, 2c + 1). The GPU's `rope_with_table`, element for
/// element.
fn rotate_rows(x: &mut [f32], t: usize, rs: usize, heads: usize, hd: usize, cos: &[f32], sin: &[f32], norm_pairs: bool) {
    let half = hd / 2;
    let xp = SyncPtr(x.as_mut_ptr());
    par_rows(t, heads * hd, |r0, r1| {
        for r in r0..r1 {
            let (c, s) = (&cos[r * half..(r + 1) * half], &sin[r * half..(r + 1) * half]);
            for h in 0..heads {
                let v = unsafe { std::slice::from_raw_parts_mut(xp.get().add(r * rs + h * hd), hd) };
                for i in 0..half {
                    let (p0, p1) = if norm_pairs { (2 * i, 2 * i + 1) } else { (i, i + half) };
                    let (x1, x2) = (v[p0], v[p1]);
                    v[p0] = x1 * c[i] - x2 * s[i];
                    v[p1] = x2 * c[i] + x1 * s[i];
                }
            }
        }
    });
}

/// Causal (optionally windowed, optionally softcapped) attention, each query against its own
/// sequence's cache. `q` is `[t, nh·hd]`; query `r` belongs to sequence `seq[r]`, is stored at row
/// `pos[r]` there, and attends rows `max(0, pos−window+1) ..= pos` of `ks[seq]`/`vs[seq]` — the GPU's
/// masks (`decode_attention_win`, `causal_attention_win`) are in stored rows too.
#[allow(clippy::too_many_arguments)]
fn attention(q: &[f32], seq: &[usize], pos: &[usize], ks: &[&[f32]], vs: &[&[f32]],
             nh: usize, nkv: usize, hd: usize, window: usize, softcap: f32, out: &mut [f32]) {
    let t = seq.len();
    let (group, kvw, qw) = (nh / nkv, nkv * hd, nh * hd);
    let scale = 1.0 / (hd as f32).sqrt();
    let span = |r: usize| { let p = pos[r]; let lo = if window > 0 { (p + 1).saturating_sub(window) } else { 0 }; (lo, p + 1) };
    let op = SyncPtr(out.as_mut_ptr());
    let one = |r: usize, h: usize, sc: &mut Vec<f32>| {
        let (lo, hi) = span(r);
        let (kc, vc) = (ks[seq[r]], vs[seq[r]]);
        let g = h / group;
        let qh = &q[r * qw + h * hd..r * qw + (h + 1) * hd];
        sc.clear();
        let mut mx = f32::NEG_INFINITY;
        for j in lo..hi {
            let mut s = dot(qh, &kc[j * kvw + g * hd..j * kvw + (g + 1) * hd]) * scale;
            if softcap > 0.0 { s = softcap * (s / softcap).clamp(-15.0, 15.0).tanh(); }
            mx = mx.max(s);
            sc.push(s);
        }
        let mut sum = 0f32;
        for s in sc.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
        let o = unsafe { std::slice::from_raw_parts_mut(op.get().add(r * qw + h * hd), hd) };
        o.fill(0.0);
        let inv = 1.0 / sum;
        for (jj, j) in (lo..hi).enumerate() { axpy(o, sc[jj] * inv, &vc[j * kvw + g * hd..j * kvw + (g + 1) * hd]); }
    };
    let work: usize = (0..t).map(|r| { let (lo, hi) = span(r); hi - lo }).sum::<usize>() * nh * hd;
    if work < (1 << 16) {
        let mut sc = Vec::new();
        for r in 0..t { for h in 0..nh { one(r, h, &mut sc); } }
        return;
    }
    thread_local! { static SCORES: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) }; }
    cpu_q::pool().for_each(t * nh, |i| SCORES.with(|sc| one(i / nh, i % nh, &mut sc.borrow_mut())));
}

/// `FERRIC_CPU_PROFILE=1`: wall time per phase of the CPU forward, summed over calls and printed every
/// 64 calls — to attribute a step between the matmuls, attention and the serial host work between
/// dispatches. One env read per forward when off.
struct Prof { on: bool, t: std::time::Instant, acc: [f64; 8] }
const PHASES: [&str; 8] = ["norm+host", "qkv", "rope+kv", "attention", "wo", "gate|up", "act+down", "head"];
static PROF_ACC: std::sync::Mutex<([f64; 8], usize)> = std::sync::Mutex::new(([0.0; 8], 0));
impl Prof {
    fn new() -> Prof { Prof { on: std::env::var("FERRIC_CPU_PROFILE").is_ok(), t: std::time::Instant::now(), acc: [0.0; 8] } }
    #[inline]
    fn mark(&mut self, phase: usize) {
        if !self.on { return; }
        let now = std::time::Instant::now();
        self.acc[phase] += now.duration_since(self.t).as_secs_f64();
        self.t = now;
    }
    /// Add this span's times; `forward_done` counts one forward (the head's span does not).
    fn flush(self, forward_done: bool) {
        if !self.on { return; }
        let mut g = PROF_ACC.lock().unwrap();
        for i in 0..8 { g.0[i] += self.acc[i]; }
        if !forward_done { return; }
        g.1 += 1;
        if g.1 % 64 == 0 {
            let tot: f64 = g.0.iter().sum();
            let parts: Vec<String> = (0..8).map(|i| format!("{} {:.0}us ({:.0}%)", PHASES[i], g.0[i] / g.1 as f64 * 1e6, 100.0 * g.0[i] / tot)).collect();
            eprintln!("cpu profile, per forward over {} forwards: {:.0}us = {}", g.1, tot / g.1 as f64 * 1e6, parts.join(", "));
        }
    }
}

/// The matmul options for one projection role. `FERRIC_CPU_F32ACT_ROLES=qkv,wo,gateup,down,head` keeps f32
/// activations for the listed roles only (the rest at `FERRIC_CPU_ACT`) — the instrument that attributed
/// the int8 activation error: it is spread over every role (Qwen3-0.6B Q8_0, mean |dlogit| to the
/// authors 0.1055 int8 everywhere, 0.0644 f32; f32 at any ONE role still 0.089..0.104).
fn role_opts(role: &str) -> cpu_q::Opts {
    let mut o = cpu_q::Opts::from_env();
    if let Ok(list) = std::env::var("FERRIC_CPU_F32ACT_ROLES") {
        if list.split(',').any(|r| r.trim() == role) { o.act = cpu_q::ActPrec::F32; }
    }
    o
}

#[inline]
fn silu(v: f32) -> f32 { v / (1.0 + (-v).exp()) }
#[inline]
fn gelu_tanh(v: f32) -> f32 {
    let a = 0.797_884_6_f32 * (v + 0.044715 * v * v * v);
    0.5 * v * (1.0 + a.clamp(-15.0, 15.0).tanh())
}

// ---------------------------------------------------------------------------------------------------
// the forward
// ---------------------------------------------------------------------------------------------------

impl Qwen3 {
    /// Whether this model's weights live on the CPU fabric (`FERRIC_CPU=1` at load).
    pub fn on_cpu(&self) -> bool { self.cpu.is_some() }

    /// The cos/sin tables for `rows` at `base`, as the GPU path builds them — or, under
    /// `FERRIC_ROPE_DEVICE=1` (the precision gates' can-fail arm), the device's old derivation
    /// (`exp(-2c/d · ln base)` in f32, angle and trig in f32), so that control still moves here.
    fn cpu_rope(&self, rows: &RopeRows, t: usize, base: f32, long: Option<bool>) -> (Vec<f32>, Vec<f32>) {
        let (mut c, mut s) = if !rope_on_device() {
            self.rope_host(rows, t, base, long)
        } else {
            let hd = self.cfg.head_dim;
            let lb = base.ln();
            let mult: Vec<f32> = match (&self.longrope, long) {
                (Some(lr), Some(l)) => (if l { &lr.long_ext } else { &lr.short_ext }).iter().map(|&x| 1.0 / x).collect(),
                _ => match &self.rope_scale {
                    super::RopeScale::None | super::RopeScale::Ext(_) => vec![1.0; hd / 2],
                    super::RopeScale::Div(d) => d.iter().map(|&x| 1.0 / x).collect(),
                    super::RopeScale::Mul(m) => m.clone(),
                },
            };
            let af = match &self.longrope {
                Some(lr) if std::env::var("FERRIC_LONGROPE_NO_ATTN_FACTOR").is_err() => lr.attn_factor,
                _ => 1.0,
            };
            let (mut c, mut s) = (Vec::with_capacity(t * hd / 2), Vec::with_capacity(t * hd / 2));
            for r in 0..t {
                let p = match rows { RopeRows::Run(s0, _) => (s0 + r) as f32, RopeRows::At(ps) => ps[r] as f32 };
                for i in 0..hd / 2 {
                    let a = p * ((-2.0 * i as f32 / hd as f32 * lb).exp() * mult[i]);
                    c.push(a.cos() * af);
                    s.push(a.sin() * af);
                }
            }
            (c, s)
        };
        // YaRN scales the rotated q and k by (1 + 0.1·ln factor) — see `Qwen3::rope`.
        if self.cfg.yarn_factor > 1.0 {
            let m = 1.0 + 0.1 * self.cfg.yarn_factor.ln();
            for v in c.iter_mut().chain(s.iter_mut()) { *v *= m; }
        }
        (c, s)
    }

    /// Every layer over rows `x` (`[t, n_embd]`), row `r` in sequence `seq[r]`, stored at row `row[r]`
    /// of that sequence's K/V (its rotation comes from `rope_rows`), appending each row's K/V. Returns
    /// the hidden state after the last layer.
    fn cpu_layers(&self, mut x: Vec<f32>, seq: &[usize], row: &[usize], kvs: &mut [&mut CpuKv],
                  rope_rows: &RopeRows, long: Option<bool>) -> Vec<f32> {
        let cm = self.cpu.as_ref().expect("cpu_layers without CPU weights");
        let c = &self.cfg;
        let (t, d, hd, nh, nkv) = (seq.len(), c.n_embd, c.head_dim, c.n_head, c.n_head_kv);
        let (q_out, kv_out) = (nh * hd, nkv * hd);
        let neox_override = std::env::var("FERRIC_NEOX").is_ok();
        let norm_pairs = match rope_plan(c, self.rope_freqs.is_some(), neox_override) {
            RopePlan::Norm | RopePlan::ScaledNorm => true,
            RopePlan::Neox | RopePlan::ScaledNeox => false,
            RopePlan::Mrope(..) => panic!("multimodal rope is not on the CPU fabric"),
        };
        let nonope = std::env::var("FERRIC_NONOPE").is_ok();
        let nowindow = std::env::var("FERRIC_NOWINDOW").is_ok();
        let nogate = std::env::var("FERRIC_NOGATE").is_ok();
        let mut tabs: Vec<(u32, Vec<f32>, Vec<f32>)> = Vec::new();
        let n_ff = cm.layers.first().map_or(0, |l| l.gate.rows);
        let (mut h, mut ao, mut f) = (vec![0f32; t * d], vec![0f32; t * d], vec![0f32; t * d]);
        let (mut q, mut k, mut v, mut o) = (vec![0f32; t * q_out], vec![0f32; t * kv_out], vec![0f32; t * kv_out], vec![0f32; t * q_out]);
        let (mut gb, mut ub) = (vec![0f32; t * n_ff], vec![0f32; t * n_ff]);
        let mut pf = Prof::new();
        for (il, l) in cm.layers.iter().enumerate() {
            // ---- attention ----
            rmsnorm_rows(&x, &l.attn_norm, c.eps, d, &mut h);
            pf.mark(0);
            cpu_q::matmul_opts(&mut [(&l.wq, &mut q), (&l.wk, &mut k), (&l.wv, &mut v)], &h, t, role_opts("qkv"));
            pf.mark(1);
            if let Some((bq, bk, bv)) = &l.bias {
                for r in 0..t {
                    for (a, b) in q[r * q_out..(r + 1) * q_out].iter_mut().zip(bq) { *a += b; }
                    for (a, b) in k[r * kv_out..(r + 1) * kv_out].iter_mut().zip(bk) { *a += b; }
                    for (a, b) in v[r * kv_out..(r + 1) * kv_out].iter_mut().zip(bv) { *a += b; }
                }
            }
            // QK-norm (Qwen3): each head normalised over head_dim.
            if let (Some(qn), Some(kn)) = (&l.q_norm, &l.k_norm) {
                let mut tmp = vec![0f32; hd];
                for hrow in q.chunks_exact_mut(hd) { rmsnorm_row(hrow, qn, c.eps, &mut tmp); hrow.copy_from_slice(&tmp); }
                for hrow in k.chunks_exact_mut(hd) { rmsnorm_row(hrow, kn, c.eps, &mut tmp); hrow.copy_from_slice(&tmp); }
            }
            if l.rope || nonope {
                let key = l.rope_base.to_bits();
                if !tabs.iter().any(|(b, _, _)| *b == key) {
                    let (cs, sn) = self.cpu_rope(rope_rows, t, l.rope_base, long);
                    tabs.push((key, cs, sn));
                }
                let (_, cs, sn) = tabs.iter().find(|(b, _, _)| *b == key).unwrap();
                rotate_rows(&mut q, t, q_out, nh, hd, cs, sn, norm_pairs);
                rotate_rows(&mut k, t, kv_out, nkv, hd, cs, sn, norm_pairs);
            }
            pf.mark(2);
            // Append each row's K/V to ITS sequence, at ITS row.
            for r in 0..t {
                let kv = &mut kvs[seq[r]];
                assert_eq!(kv.k[il].len(), row[r] * kv_out, "CPU K/V of layer {il} is not at row {}", row[r]);
                kv.k[il].extend_from_slice(&k[r * kv_out..(r + 1) * kv_out]);
                kv.v[il].extend_from_slice(&v[r * kv_out..(r + 1) * kv_out]);
            }
            {
                let ks: Vec<&[f32]> = kvs.iter().map(|kv| &kv.k[il][..]).collect();
                let vs: Vec<&[f32]> = kvs.iter().map(|kv| &kv.v[il][..]).collect();
                let win = if nowindow { 0 } else { l.window };
                pf.mark(2);
                attention(&q, seq, row, &ks, &vs, nh, nkv, hd, win, c.attn_softcap, &mut o);
                pf.mark(3);
            }
            // Gated GQA (Muse Glimmer): sigmoid of a projection of the layer's NORMED INPUT.
            if let (Some(wg), false) = (&l.attn_gate, nogate) {
                let mut gt = vec![0f32; t * q_out];
                cpu_q::matmul(wg, &h, t, &mut gt);
                for (a, g) in o.iter_mut().zip(&gt) { *a *= 1.0 / (1.0 + (-g).exp()); }
            }
            pf.mark(0);
            cpu_q::matmul_opts(&mut [(&l.wo, &mut ao)], &o, t, role_opts("wo"));
            pf.mark(4);
            // ---- residual + FFN ----
            if let (Some(pa), Some(pfn)) = (&l.post_attn_norm, &l.post_ffn_norm) {
                // Gemma / Glimmer: x = x + post_attn_norm(attn); x = x + post_ffn_norm(ffn(ffn_norm(x))).
                let pe = c.post_norm_eps;
                rmsnorm_rows(&ao.clone(), pa, pe, d, &mut ao);
                for (a, b) in x.iter_mut().zip(&ao) { *a += b; }
                self.cpu_ffn(l, &x, t, &mut h, &mut gb, &mut ub, &mut f, &mut pf);
                rmsnorm_rows(&f.clone(), pfn, pe, d, &mut f);
            } else {
                for (a, b) in x.iter_mut().zip(&ao) { *a += b; }
                self.cpu_ffn(l, &x, t, &mut h, &mut gb, &mut ub, &mut f, &mut pf);
            }
            for (a, b) in x.iter_mut().zip(&f) { *a += b; }
            pf.mark(0);
        }
        pf.flush(true);
        x
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_ffn(&self, l: &CpuLayer, x: &[f32], t: usize, h: &mut [f32], gb: &mut [f32], ub: &mut [f32], f: &mut [f32], pf: &mut Prof) {
        let c = &self.cfg;
        rmsnorm_rows(x, &l.ffn_norm, c.eps, c.n_embd, h);
        pf.mark(0);
        cpu_q::matmul_opts(&mut [(&l.gate, &mut *gb), (&l.up, &mut *ub)], h, t, role_opts("gateup"));
        pf.mark(5);
        let gemma = c.is_gemma;
        let gp = SyncPtr(gb.as_mut_ptr());
        let n = l.gate.rows;
        par_rows(t, n, |r0, r1| {
            let g = unsafe { std::slice::from_raw_parts_mut(gp.get().add(r0 * n), (r1 - r0) * n) };
            let u = &ub[r0 * n..r1 * n];
            if gemma { for (a, &b) in g.iter_mut().zip(u) { *a = gelu_tanh(*a) * b; } }
            else { for (a, &b) in g.iter_mut().zip(u) { *a = silu(*a) * b; } }
        });
        cpu_q::matmul_opts(&mut [(&l.down, &mut *f)], gb, t, role_opts("down"));
        pf.mark(6);
    }

    /// Final norm + LM head (+ logit scale, + final softcap) over `rows` rows of `x`.
    fn cpu_head(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let mut pf = Prof::new();
        let cm = self.cpu.as_ref().unwrap();
        let c = &self.cfg;
        let mut n = vec![0f32; rows * c.n_embd];
        rmsnorm_rows(x, &cm.out_norm, c.eps, c.n_embd, &mut n);
        let mut lg = vec![0f32; rows * cm.lm_head.rows];
        cpu_q::matmul_opts(&mut [(&cm.lm_head, &mut lg)], &n, rows, role_opts("head"));
        if c.logit_scale != 1.0 && std::env::var("FERRIC_NOLOGITSCALE").is_err() {
            for v in lg.iter_mut() { *v *= c.logit_scale; }
        }
        if c.final_softcap > 0.0 {
            let cap = c.final_softcap;
            for v in lg.iter_mut() { *v = cap * (*v / cap).clamp(-15.0, 15.0).tanh(); }
        }
        pf.mark(7);
        pf.flush(false);
        lg
    }

    fn cpu_refuse(&self, cache: &Cache) {
        assert!(cache.lora.is_empty(), "LoRA adapters are not on the CPU fabric (unset FERRIC_CPU to run them on the GPU)");
        assert!(cache.fmt.is_none() && cache.q.is_empty(),
                "a quantized KV cache (FERRIC_KVQ) is not on the CPU fabric — its K/V are f32 host rows");
    }

    /// **One forward on the CPU fabric, host in and out.** `tokens` continue `cache`; returns the
    /// logits of the rows `out` selects (`[rows, n_vocab]`, or `[rows, n_embd]` for `Out::Hidden`) and
    /// that row count. The LongRoPE crossing recomputes the whole sequence, as on the GPU path.
    pub(crate) fn cpu_run(&self, tokens: &[u32], cache: &mut Cache, out: Out) -> (Vec<f32>, usize) {
        self.cpu_refuse(cache);
        let refill = self.longrope_refill(tokens, cache);
        let toks: &[u32] = refill.as_deref().unwrap_or(tokens);
        let keep_from = toks.len() - tokens.len();
        let (t, d) = (toks.len(), self.cfg.n_embd);
        let kv_w = self.cfg.n_head_kv * self.cfg.head_dim;
        let pos0 = cache.pos;
        let gpu_rows = cache.kv.first().map_or(0, |p| p.0.len());
        let kv = cache.cpu.get_or_insert_with(|| CpuKv::new(self.cfg.n_layer));
        kv.sync(pos0, kv_w, gpu_rows);
        let x = self.embed_rows(toks, true);
        let seq = vec![0usize; t];
        let row: Vec<usize> = (kv.len..kv.len + t).collect();
        let long = self.longrope_long(pos0, t);
        let x = self.cpu_layers(x, &seq, &row, &mut [&mut *kv], &RopeRows::Run(pos0, t), long);
        kv.len += t;
        self.advance(cache, Some(toks), t);
        match out {
            Out::Hidden => {
                let cm = self.cpu.as_ref().unwrap();
                let mut n = vec![0f32; t * d];
                rmsnorm_rows(&x, &cm.out_norm, self.cfg.eps, d, &mut n);
                (n[keep_from * d..].to_vec(), t - keep_from)
            }
            Out::Last => (self.cpu_head(&x[(t - 1) * d..], 1), 1),
            Out::All => (self.cpu_head(&x[keep_from * d..], t - keep_from), t - keep_from),
        }
    }

    /// [`Self::cpu_run`] wrapped as the `Tensor` the runtime's entry points return.
    pub(crate) fn cpu_forward(&self, tokens: &[u32], cache: &mut Cache, out: Out) -> Tensor {
        let (v, rows) = self.cpu_run(tokens, cache, out);
        let w = v.len() / rows.max(1);
        Tensor::from_vec(&self.ctx, &v, &[rows, w])
    }

    /// **Host-side logits** for `tokens` continuing `cache`, on the CPU fabric: `[n_vocab]` for the last
    /// row when `last_only`, else `[t, n_vocab]`. No GPU round trip — the entry a GPU-less caller (and
    /// a CPU benchmark, which must not time a buffer upload and readback per token) uses.
    pub fn cpu_logits(&self, tokens: &[u32], cache: &mut Cache, last_only: bool) -> Vec<f32> {
        assert!(self.on_cpu(), "cpu_logits on a model loaded for the GPU (set FERRIC_CPU=1 before Qwen3::load)");
        self.cpu_run(tokens, cache, if last_only { Out::Last } else { Out::All }).0
    }

    /// Batched decode on the CPU fabric: one token per sequence, the projections shared across rows,
    /// attention per sequence against its own history. See `Qwen3::forward_batch`.
    pub(crate) fn cpu_forward_batch(&self, tokens: &[u32], caches: &mut [&mut Cache]) -> Tensor {
        let kv_w = self.cfg.n_head_kv * self.cfg.head_dim;
        let n = tokens.len();
        for c in caches.iter_mut() {
            self.cpu_refuse(c);
            let (p, gpu_rows) = (c.pos, c.kv.first().map_or(0, |q| q.0.len()));
            c.cpu.get_or_insert_with(|| CpuKv::new(self.cfg.n_layer)).sync(p, kv_w, gpu_rows);
        }
        let pos: Vec<usize> = caches.iter().map(|c| c.pos).collect();
        let row: Vec<usize> = caches.iter().map(|c| c.cpu.as_ref().unwrap().len).collect();
        let seq: Vec<usize> = (0..n).collect();
        let x = self.embed_rows(tokens, true);
        let rows = RopeRows::At(pos.iter().map(|&p| p as u32).collect());
        let x = {
            let mut kvs: Vec<&mut CpuKv> = caches.iter_mut().map(|c| c.cpu.as_mut().unwrap()).collect();
            let x = self.cpu_layers(x, &seq, &row, &mut kvs, &rows, None);
            for kv in kvs.iter_mut() { kv.len += 1; }
            x
        };
        for (c, &tk) in caches.iter_mut().zip(tokens) { self.advance(c, Some(&[tk]), 1); }
        let lg = self.cpu_head(&x, n);
        Tensor::from_vec(&self.ctx, &lg, &[n, lg.len() / n])
    }
}
