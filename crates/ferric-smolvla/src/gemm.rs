//! Row-parallel f32 GEMM. Each row is a sequential dot, so the result matches
//! `ferric_core::matmul_cpu` on the transposed layout.

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    let mut i = 0;
    let n = a.len();
    while i + 4 <= n {
        acc += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < n {
        acc += a[i] * b[i];
        i += 1;
    }
    acc
}

/// `C[m, n] = A[m, k] * B[n, k]^T`. `B` is stored as rows of length `k`.
pub fn gemm_nt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    let mut c = vec![0.0f32; m * n];
    if m == 0 || n == 0 {
        return c;
    }
    let threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
        .min(m);
    let base = m / threads;
    let extra = m % threads;
    std::thread::scope(|scope| {
        let mut rest = &mut c[..];
        let mut start = 0;
        for t in 0..threads {
            let rows = base + usize::from(t < extra);
            let (head, tail) = rest.split_at_mut(rows * n);
            rest = tail;
            let a_rows = &a[start * k..(start + rows) * k];
            start += rows;
            scope.spawn(move || {
                for i in 0..rows {
                    let ar = &a_rows[i * k..(i + 1) * k];
                    for j in 0..n {
                        head[i * n + j] = dot(ar, &b[j * k..(j + 1) * k]);
                    }
                }
            });
        }
    });
    c
}

/// `C[m, n] = A[m, k] * B[k, n]`.
pub fn gemm_nn(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut c = vec![0.0f32; m * n];
    if m == 0 {
        return c;
    }
    let threads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
        .min(m);
    let base = m / threads;
    let extra = m % threads;
    std::thread::scope(|scope| {
        let mut rest = &mut c[..];
        let mut start = 0;
        for t in 0..threads {
            let rows = base + usize::from(t < extra);
            let (head, tail) = rest.split_at_mut(rows * n);
            rest = tail;
            let a_rows = &a[start * k..(start + rows) * k];
            start += rows;
            scope.spawn(move || {
                for i in 0..rows {
                    let crow = &mut head[i * n..(i + 1) * n];
                    for p in 0..k {
                        let ap = a_rows[i * k + p];
                        let brow = &b[p * n..(p + 1) * n];
                        for j in 0..n {
                            crow[j] += ap * brow[j];
                        }
                    }
                }
            });
        }
    });
    c
}

/// `Y = X W^T + b` with `W` stored `[out, in]`.
pub fn linear(
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
    seq: usize,
    inn: usize,
    out: usize,
) -> Vec<f32> {
    let mut y = gemm_nt(x, w, seq, inn, out);
    if let Some(bias) = bias {
        for s in 0..seq {
            for o in 0..out {
                y[s * out + o] += bias[o];
            }
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::matmul_cpu;

    #[test]
    fn gemm_nt_matches_matmul_cpu() {
        let m = 3;
        let k = 5;
        let n = 4;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1 - 0.4).collect();
        let b: Vec<f32> = (0..n * k).map(|i| (i as f32) * -0.07 + 0.2).collect();
        let mut bt = vec![0.0f32; k * n];
        for j in 0..n {
            for p in 0..k {
                bt[p * n + j] = b[j * k + p];
            }
        }
        let got = gemm_nt(&a, &b, m, k, n);
        let expect = matmul_cpu(&a, &bt, m, k, n);
        for (g, e) in got.iter().zip(&expect) {
            assert!((g - e).abs() < 1e-5, "{g} {e}");
        }
    }
}
