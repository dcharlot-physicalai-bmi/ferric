//! Plain-Rust reference on logical (row-major) data — the source of truth for the GPU tensor runtime.
//! Operates on the same logical layout `Tensor::to_vec` returns, so comparisons are apples-to-apples.

fn unravel(mut i: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0usize; shape.len()];
    for d in (0..shape.len()).rev() {
        idx[d] = i % shape[d];
        i /= shape[d];
    }
    idx
}
fn ravel(idx: &[usize], shape: &[usize]) -> usize {
    let mut i = 0;
    for d in 0..shape.len() {
        i = i * shape[d] + idx[d];
    }
    i
}
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
    let r = a.len().max(b.len());
    (0..r).map(|i| {
        let da = if i + a.len() >= r { a[i + a.len() - r] } else { 1 };
        let db = if i + b.len() >= r { b[i + b.len() - r] } else { 1 };
        da.max(db)
    }).collect()
}
// map an out multi-index to the linear index of a right-aligned (broadcast) operand
fn proj(out_idx: &[usize], shape: &[usize]) -> usize {
    let r = out_idx.len();
    let idx: Vec<usize> = (0..shape.len()).map(|d| {
        let od = out_idx[d + r - shape.len()];
        if shape[d] == 1 { 0 } else { od }
    }).collect();
    ravel(&idx, shape)
}

pub fn binary(a: &[f32], ash: &[usize], b: &[f32], bsh: &[usize], op: &str) -> (Vec<f32>, Vec<usize>) {
    let sh = broadcast_shapes(ash, bsh);
    let n: usize = sh.iter().product();
    let out = (0..n).map(|i| {
        let idx = unravel(i, &sh);
        let (x, y) = (a[proj(&idx, ash)], b[proj(&idx, bsh)]);
        match op { "+" => x + y, "-" => x - y, "*" => x * y, "/" => x / y, "max" => x.max(y), _ => unreachable!() }
    }).collect();
    (out, sh)
}

pub fn unary(x: &[f32], op: &str) -> Vec<f32> {
    x.iter().map(|&v| match op { "exp" => v.exp(), "neg" => -v, "relu" => v.max(0.0), "sqrt" => v.sqrt(), "sin" => v.sin(), "cos" => v.cos(), _ => v }).collect()
}

pub fn reduce(x: &[f32], shape: &[usize], axes: &[usize], op: &str, keepdim: bool) -> (Vec<f32>, Vec<usize>) {
    let keep: Vec<usize> = (0..shape.len()).filter(|d| !axes.contains(d)).collect();
    let oshape: Vec<usize> = if keepdim {
        (0..shape.len()).map(|d| if axes.contains(&d) { 1 } else { shape[d] }).collect()
    } else if keep.is_empty() { vec![1] } else { keep.iter().map(|&d| shape[d]).collect() };
    let on: usize = oshape.iter().product();
    let mut acc = vec![if op == "max" { f32::NEG_INFINITY } else { 0.0 }; on];
    for i in 0..x.len() {
        let idx = unravel(i, shape);
        let oidx: Vec<usize> = keep.iter().map(|&d| idx[d]).collect();
        let o = if oidx.is_empty() { 0 } else { ravel(&oidx, &keep.iter().map(|&d| shape[d]).collect::<Vec<_>>()) };
        if op == "max" { acc[o] = acc[o].max(x[i]); } else { acc[o] += x[i]; }
    }
    if op == "mean" {
        let red: usize = axes.iter().map(|&d| shape[d]).product();
        for v in acc.iter_mut() { *v /= red as f32; }
    }
    (acc, oshape)
}

pub fn matmul(a: &[f32], ash: &[usize], b: &[f32], bsh: &[usize]) -> (Vec<f32>, Vec<usize>) {
    let (ra, rb) = (ash.len(), bsh.len());
    let (m, k, n) = (ash[ra - 2], ash[ra - 1], bsh[rb - 1]);
    let batch = broadcast_shapes(&ash[..ra - 2], &bsh[..rb - 2]);
    let bn: usize = batch.iter().product::<usize>().max(1);
    let oshape: Vec<usize> = batch.iter().chain([m, n].iter()).copied().collect();
    let mut out = vec![0.0f32; bn * m * n];
    for bt in 0..bn {
        let bidx = unravel(bt, &batch);
        let ab = proj(&bidx, &ash[..ra - 2]) * m * k;
        let bb = proj(&bidx, &bsh[..rb - 2]) * k * n;
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for l in 0..k { s += a[ab + i * k + l] * b[bb + l * n + j]; }
                out[bt * m * n + i * n + j] = s;
            }
        }
    }
    (out, oshape)
}
