//! **A BERT-family encoder, stage by stage, on ids the model's AUTHORS produced** — the Ferric side of
//! `scripts/nomic_bert_conformance.sh`.
//!
//!   bert_stages <model.gguf | HF checkpoint dir> <id,id,...> <row,row,...> <col,col,...>
//!
//! Prints, for every stage (`inp_norm`, then each block's `attn_out_norm` / `layer_out_norm`):
//!   `ROWS <stage> <v...>`   the requested rows at the requested columns, row-major
//!   `SUMS <stage> <v...>`   every row's sum, `SSQS` every row's sum of squares
//! then `EMB` — the declared pooling (`<arch>.pooling_type`) of the last stage — and `NORM`, its L2
//! normalisation. An HF directory is read through `ferric_load::hf` with no conversion step.
use ferric_llama::bert::Bert;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }

fn csv(s: &str) -> Vec<u32> { s.split(',').filter(|x| !x.trim().is_empty()).map(|x| x.trim().parse().expect("int")).collect() }
fn line(tag: &str, name: &str, v: &[f32]) {
    println!("{tag} {name} {}", v.iter().map(|x| format!("{x:e}")).collect::<Vec<_>>().join(" "));
}

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    assert!(a.len() == 5, "usage: bert_stages <model.gguf | hf-dir> <ids> <rows> <cols>");
    let (ids, rows, cols) = (csv(&a[2]), csv(&a[3]), csv(&a[4]));
    let ctx = Arc::new(ferric_core::Context::new().await.expect("gpu"));
    let p = std::path::Path::new(&a[1]);
    let m = if p.is_dir() {
        Bert::load(&ctx, &ferric_load::hf::HfCheckpoint::open(p).expect("open HF checkpoint")).expect("load")
    } else {
        Bert::load(&ctx, &ferric_gguf::GgufFile::open(&a[1]).expect("open gguf")).expect("load")
    };
    let (d, t) = (m.cfg.d, ids.len());
    eprintln!("{} ({}): {} layers, d {}, rope {:?}, pooling {}, {t} tokens", a[1], m.cfg.arch, m.cfg.n_layer, d,
              m.cfg.rope_base, m.cfg.pooling);
    let (h, taps) = m.forward_taps(&ids).expect("forward");
    for (name, x) in &taps {
        let v = x.to_vec().await;
        assert_eq!(v.len(), t * d, "{name}: {} values for {t} x {d}", v.len());
        let pick: Vec<f32> = rows.iter().flat_map(|&r| cols.iter().map(move |&c| (r, c)))
            .map(|(r, c)| v[r as usize * d + c as usize]).collect();
        line("ROWS", name, &pick);
        line("SUMS", name, &(0..t).map(|r| v[r * d..(r + 1) * d].iter().sum()).collect::<Vec<f32>>());
        line("SSQS", name, &(0..t).map(|r| v[r * d..(r + 1) * d].iter().map(|x| x * x).sum()).collect::<Vec<f32>>());
    }
    let v = h.to_vec().await;
    let e = ferric_llama::pooling::pool(&v, t, d, m.cfg.pooling).expect("pool");
    line("EMB", "pooled", &e);
    let n = e.iter().map(|x| x * x).sum::<f32>().sqrt();
    line("NORM", "pooled", &e.iter().map(|x| x / n).collect::<Vec<f32>>());
}
