//! Ferric L3 — a pure-Rust ONNX importer. Parses an ONNX model (prost-decoded protobuf) and runs its
//! graph on Ferric's on-GPU tensor ops. Starter op set: MatMul, Add, Relu — enough to run real
//! MLP/linear graphs; the transformer op set (attention decomposition, LayerNorm, etc.) extends this.

pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

use ferric_core::{Context, Tensor};
use prost::Message;
use std::collections::HashMap;

pub struct Model {
    pub graph: onnx::GraphProto,
}

/// Parse ONNX bytes into a Model.
pub fn load(bytes: &[u8]) -> Result<Model, String> {
    let m = onnx::ModelProto::decode(bytes).map_err(|e| format!("onnx decode: {e}"))?;
    let graph = m.graph.ok_or_else(|| "model has no graph".to_string())?;
    Ok(Model { graph })
}

/// Extract f32 data + shape from a TensorProto (raw_data little-endian, or float_data).
fn tensor_data(t: &onnx::TensorProto) -> (Vec<f32>, Vec<usize>) {
    let shape: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();
    let data = if !t.raw_data().is_empty() {
        t.raw_data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    } else {
        t.float_data.clone()
    };
    (data, shape)
}

/// Extract i64 data + shape from an INT64 TensorProto (raw_data little-endian, or int64_data).
fn int_data(t: &onnx::TensorProto) -> (Vec<i64>, Vec<usize>) {
    let shape: Vec<usize> = t.dims.iter().map(|&d| d as usize).collect();
    let data = if !t.raw_data().is_empty() {
        t.raw_data().chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect()
    } else {
        t.int64_data.clone()
    };
    (data, shape)
}

fn attr_i(node: &onnx::NodeProto, name: &str, default: i64) -> i64 {
    node.attribute.iter().find(|a| a.name() == name).map(|a| a.i()).unwrap_or(default)
}
fn attr_f(node: &onnx::NodeProto, name: &str, default: f32) -> f32 {
    node.attribute.iter().find(|a| a.name() == name).map(|a| a.f()).unwrap_or(default)
}
fn norm_axis(a: i64, rank: usize) -> usize { (if a < 0 { a + rank as i64 } else { a }) as usize }
/// Squeeze/Unsqueeze axes: opset-13+ pass them as an int64 input, older ones as an `axes` attribute.
fn axes_of(node: &onnx::NodeProto, ints: &HashMap<String, (Vec<i64>, Vec<usize>)>) -> Vec<i64> {
    if node.input.len() > 1 {
        ints.get(&node.input[1]).map(|(v, _)| v.clone()).unwrap_or_default()
    } else {
        node.attribute.iter().find(|a| a.name() == "axes").map(|a| a.ints.clone()).unwrap_or_default()
    }
}

impl Model {
    /// Run the graph. `inputs` maps graph input names → (data, shape). Returns (output_name, values).
    pub async fn run(&self, ctx: &Context, inputs: &HashMap<String, (Vec<f32>, Vec<usize>)>) -> Result<(String, Vec<f32>), String> {
        let mut env: HashMap<String, Tensor> = HashMap::new();
        for (name, (d, s)) in inputs {
            env.insert(name.clone(), ctx.tensor(d, s));
        }
        // INT64 initializers (shapes, indices, axes) live host-side as ints; everything else is an f32 GPU tensor.
        let mut ints: HashMap<String, (Vec<i64>, Vec<usize>)> = HashMap::new();
        for init in &self.graph.initializer {
            if init.data_type() == onnx::tensor_proto::DataType::Int64 as i32 {
                ints.insert(init.name().to_string(), int_data(init));
            } else {
                let (d, s) = tensor_data(init);
                env.insert(init.name().to_string(), ctx.tensor(&d, &s));
            }
        }
        for node in &self.graph.node {
            let get = |name: &str| env.get(name).ok_or_else(|| format!("missing value '{name}'"));
            let out = match node.op_type() {
                "MatMul" => {
                    let a = get(&node.input[0])?;
                    let b = get(&node.input[1])?;
                    let (ra, rb) = (a.shape.len(), b.shape.len());
                    let m = a.shape[ra - 2] as u32;
                    let k = a.shape[ra - 1] as u32;
                    let n = b.shape[rb - 1] as u32;
                    ctx.mm(a, b, m, k, n)
                }
                "Add" => {
                    let (a, b) = (get(&node.input[0])?, get(&node.input[1])?);
                    let (big, small) = if a.len() >= b.len() { (a, b) } else { (b, a) };
                    if big.len() == small.len() {
                        ctx.add_t(big, small)
                    } else if small.len() > 0 && big.len() % small.len() == 0 {
                        ctx.add_bias_t(big, small) // per-row bias broadcast [.,d] + [d]
                    } else {
                        return Err(format!("Add broadcast {:?}+{:?} not yet supported", a.shape, b.shape));
                    }
                }
                "Relu" => ctx.relu_t(get(&node.input[0])?),
                "Sigmoid" => ctx.sigmoid_t(get(&node.input[0])?),
                "Sqrt" => ctx.sqrt_t(get(&node.input[0])?),
                "Gelu" => ctx.gelu_t(get(&node.input[0])?),
                "Sub" => {
                    let (a, b) = (get(&node.input[0])?, get(&node.input[1])?);
                    if a.len() != b.len() { return Err(format!("Sub broadcast {:?}-{:?} not supported", a.shape, b.shape)); }
                    ctx.sub_t(a, b)
                }
                "Div" => {
                    let (a, b) = (get(&node.input[0])?, get(&node.input[1])?);
                    if a.len() != b.len() { return Err(format!("Div broadcast {:?}/{:?} not supported", a.shape, b.shape)); }
                    ctx.div_t(a, b)
                }
                // Fused ONNX LayerNormalization (opset 17+): normalize last `d` dims, scale+bias.
                "LayerNormalization" => {
                    let x = get(&node.input[0])?;
                    let scale = get(&node.input[1])?;
                    let eps = attr_f(node, "epsilon", 1e-5);
                    let d = scale.len();
                    let rows = (x.len() / d) as u32;
                    // ONNX bias (input[2]) is optional; synthesize zeros when absent.
                    let zeros;
                    let bias = if node.input.len() > 2 && !node.input[2].is_empty() {
                        get(&node.input[2])?
                    } else {
                        zeros = ctx.tensor(&vec![0.0f32; d], &[d]);
                        &zeros
                    };
                    ctx.layernorm_t(x, scale, bias, rows, d as u32, eps)
                }
                // RMSNorm — modern Llama/SmolVLA norm. Fused ONNX op (opset 23) or the ORT contrib op.
                "RMSNormalization" | "SimplifiedLayerNormalization" => {
                    let x = get(&node.input[0])?;
                    let scale = get(&node.input[1])?;
                    let eps = attr_f(node, "epsilon", 1e-5);
                    let d = scale.len();
                    let rows = (x.len() / d) as u32;
                    ctx.rmsnorm_t(x, scale, rows, d as u32, eps)
                }
                "Gemm" => {
                    // Y = alpha·(A[·ᵀ] · B[·ᵀ]) + beta·C   (Linear layers export as this)
                    let a = get(&node.input[0])?;
                    let b = get(&node.input[1])?;
                    let (alpha, trans_a, trans_b) = (attr_f(node, "alpha", 1.0), attr_i(node, "transA", 0), attr_i(node, "transB", 0));
                    let a_t;
                    let a = if trans_a != 0 { a_t = ctx.transpose2d_t(a, a.shape[0] as u32, a.shape[1] as u32); &a_t } else { a };
                    let m = a.shape[a.shape.len() - 2] as u32;
                    let k = a.shape[a.shape.len() - 1] as u32;
                    let mut y = if trans_b != 0 {
                        ctx.mm_bt(a, b, m, b.shape[0] as u32, k, alpha) // B is [n,k]; Y = α·A·Bᵀ
                    } else {
                        let n = b.shape[b.shape.len() - 1] as u32;
                        let y = ctx.mm(a, b, m, k, n);
                        if alpha != 1.0 { let s = ctx.tensor(&[alpha], &[1]); ctx.mul_scalar_t(&y, &s) } else { y }
                    };
                    if node.input.len() > 2 {
                        y = ctx.add_bias_t(&y, get(&node.input[2])?); // + beta·C (beta assumed 1)
                    }
                    y
                }
                "Mul" => {
                    let (a, b) = (get(&node.input[0])?, get(&node.input[1])?);
                    if b.len() == 1 { ctx.mul_scalar_t(a, b) }
                    else if a.len() == 1 { ctx.mul_scalar_t(b, a) }
                    else if a.len() == b.len() { ctx.mul_t(a, b) }
                    else { return Err(format!("Mul broadcast {:?}*{:?} not supported", a.shape, b.shape)); }
                }
                "Transpose" => {
                    let x = get(&node.input[0])?;
                    if x.shape.len() != 2 { return Err("Transpose: only 2D supported yet".into()); }
                    ctx.transpose2d_t(x, x.shape[0] as u32, x.shape[1] as u32)
                }
                "Softmax" => {
                    let x = get(&node.input[0])?;
                    if x.shape.len() != 2 { return Err("Softmax: only 2D (last-axis) supported yet".into()); }
                    ctx.softmax_t(x, x.shape[0] as u32, x.shape[1] as u32)
                }
                // ---- structural / int tier: reshape data movement, no numeric change ----
                "Reshape" => {
                    let x = get(&node.input[0])?;
                    let (shp, _) = ints.get(&node.input[1]).ok_or("Reshape: shape must be an int64 initializer")?;
                    let total = x.len() as i64;
                    let (mut dims, mut neg, mut known) = (Vec::new(), None, 1i64);
                    for (i, &s) in shp.iter().enumerate() {
                        if s == -1 { neg = Some(i); dims.push(0); }
                        else if s == 0 { dims.push(x.shape[i]); known *= x.shape[i] as i64; }
                        else { dims.push(s as usize); known *= s; }
                    }
                    if let Some(i) = neg { dims[i] = (total / known) as usize; }
                    ctx.dup(x, dims)
                }
                "Unsqueeze" => {
                    let x = get(&node.input[0])?;
                    let axes = axes_of(node, &ints);
                    let rank = x.shape.len() + axes.len();
                    let mut norm: Vec<usize> = axes.iter().map(|&a| norm_axis(a, rank)).collect();
                    norm.sort_unstable();
                    let mut dims = x.shape.clone();
                    for a in norm { dims.insert(a.min(dims.len()), 1); }
                    ctx.dup(x, dims)
                }
                "Squeeze" => {
                    let x = get(&node.input[0])?;
                    let axes = axes_of(node, &ints);
                    let dims: Vec<usize> = if axes.is_empty() {
                        x.shape.iter().copied().filter(|&d| d != 1).collect()
                    } else {
                        let rank = x.shape.len();
                        let drop: Vec<usize> = axes.iter().map(|&a| norm_axis(a, rank)).collect();
                        x.shape.iter().enumerate().filter(|(i, _)| !drop.contains(i)).map(|(_, &d)| d).collect()
                    };
                    ctx.dup(x, dims)
                }
                "Cast" | "Identity" => { let x = get(&node.input[0])?; ctx.dup(x, x.shape.clone()) }
                "Gather" => {
                    let data = get(&node.input[0])?;
                    let (idx, idx_shape) = ints.get(&node.input[1]).ok_or("Gather: indices must be an int64 initializer")?;
                    if attr_i(node, "axis", 0) != 0 { return Err("Gather: only axis 0 supported yet".into()); }
                    let rows = data.shape[0] as i64;
                    let d: usize = data.shape[1..].iter().product::<usize>().max(1);
                    let u: Vec<u32> = idx.iter().map(|&v| (if v < 0 { v + rows } else { v }) as u32).collect();
                    let out = ctx.gather0(data, &u, d);
                    let mut sh = idx_shape.clone();
                    sh.extend_from_slice(&data.shape[1..]);
                    if sh.is_empty() { sh.push(d); }
                    ctx.dup(&out, sh)
                }
                // Host-side concat along `axis` (readback→concat→upload; a fusion/perf TODO).
                "Concat" => {
                    let axis = attr_i(node, "axis", 0);
                    let mut parts = Vec::new();
                    for inp in &node.input { parts.push(get(inp)?); }
                    let ax = norm_axis(axis, parts[0].shape.len());
                    let outer: usize = parts[0].shape[..ax].iter().product();
                    let inner_each: Vec<usize> = parts.iter().map(|p| p.shape[ax..].iter().product()).collect();
                    let mut oshape = parts[0].shape.clone();
                    oshape[ax] = parts.iter().map(|p| p.shape[ax]).sum();
                    let out_inner: usize = oshape[ax..].iter().product();
                    let mut datas = Vec::new();
                    for p in &parts { datas.push(ctx.to_vec(p).await?); }
                    let mut out = vec![0f32; oshape.iter().product()];
                    for o in 0..outer {
                        let mut off = 0;
                        for (pi, &bp) in inner_each.iter().enumerate() {
                            out[o * out_inner + off..o * out_inner + off + bp]
                                .copy_from_slice(&datas[pi][o * bp..o * bp + bp]);
                            off += bp;
                        }
                    }
                    ctx.tensor(&out, &oshape)
                }
                other => return Err(format!("unsupported op '{other}'")),
            };
            env.insert(node.output[0].clone(), out);
        }
        let out_name = self.graph.output.first().and_then(|o| o.name.clone()).ok_or("no graph output")?;
        let out = env.get(&out_name).ok_or_else(|| format!("output '{out_name}' not produced"))?;
        let v = ctx.to_vec(out).await?;
        Ok((out_name, v))
    }

    pub fn ops(&self) -> Vec<&str> {
        self.graph.node.iter().map(|n| n.op_type()).collect()
    }
}
