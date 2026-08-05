//! **The same model, the same tokens, from every backing** — file, memory, and browser-staged.
//!
//! `stream_generate` proved streaming works from a file. This proves the *embodiment* does not matter:
//! the identical checkpoint read through a file handle, an in-memory buffer, and an asynchronously-staged
//! set of ranges — the browser path — must produce byte-identical logits.
//!
//! The staged case is the browser, simulated honestly. `StagedBacking` is fed the same way a wasm caller
//! would feed it (fetch a range, hand it over) and read the same way (synchronously, from the forward
//! pass), including the property that matters most: **an un-staged read is a named error, never zeros.**
//! That is asserted here rather than assumed, because zeros would produce a model that runs and is
//! quietly wrong.
//!
//!   cargo run -p ferric-llama --example stream_embodiments --release
use ferric_core::{max_abs_diff, Context};
use ferric_gguf::backed::{header_probe, GgufBacked};
use ferric_gguf::{GgufFile, GgufSource};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_llama::stream::layer_runs_of;
use ferric_tier::{Backing, FileBacking, SliceBacking, StagedBacking, TierError};
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();
    let path = format!("{home}/.cache/ferric/hub/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf");
    let total = std::fs::metadata(&path).unwrap().len();
    let prompt: Vec<u32> = vec![785, 6722, 315, 9625, 374]; // "The capital of France is"

    println!("One checkpoint, three embodiments — {:.1} MB\n", total as f64 / 1e6);

    // ---------- 1. native: a file handle ----------
    let file_b: Arc<dyn Backing + Send + Sync> = Arc::new(FileBacking::open(&path).unwrap());
    let (header, hlen) = header_probe(&*file_b, total, 1 << 20, 64 << 20).unwrap();
    println!("  header probe: {:.2} MB of {:.1} MB ({:.2}% of the file) before anything else can be planned",
             hlen as f64 / 1e6, total as f64 / 1e6, 100.0 * hlen as f64 / total as f64);

    let runs = layer_runs_of(
        &GgufBacked::new(header.clone(), Arc::clone(&file_b)).unwrap().tensors,
        GgufBacked::new(header.clone(), Arc::clone(&file_b)).unwrap().data_start(),
    ).unwrap();
    let layer_bytes: u64 = runs.iter().map(|r| r.bytes).sum();

    let logits_of = |b: Arc<dyn Backing + Send + Sync>| {
        let g = GgufBacked::new(header.clone(), b).unwrap();
        let m = Qwen3::load(&ctx, &g).unwrap();
        let mut c = Cache::new(&m.cfg);
        m.forward_cached(&prompt, &mut c)
    };

    let reference = logits_of(Arc::clone(&file_b)).to_vec().await;
    println!("\n  {:<28} {:>12}  {:>14}   {}", "embodiment", "resident", "max|Δ| vs file", "argmax");
    println!("  {:-<74}", "");
    let arg = |v: &[f32]| v.iter().enumerate().fold((0usize, f32::MIN), |a, (i, &x)| if x > a.1 { (i, x) } else { a }).0;
    let ref_arg = arg(&reference[reference.len() - 151936..]);
    println!("  {:<28} {:>12}  {:>14}   {ref_arg}", "native · FileBacking", "on disk", "—");

    // ---------- 2. anywhere: the whole model in memory ----------
    let mem: Arc<dyn Backing + Send + Sync> = Arc::new(SliceBacking::new(std::fs::read(&path).unwrap()));
    let l_mem = logits_of(mem).to_vec().await;
    let d_mem = max_abs_diff(&reference, &l_mem);
    println!("  {:<28} {:>11.1}M  {:>14.3e}   {}", "in-memory · SliceBacking",
             total as f64 / 1e6, d_mem, arg(&l_mem[l_mem.len() - 151936..]));
    assert_eq!(d_mem, 0.0, "in-memory backing changed the logits");

    // ---------- 3. browser: asynchronously staged ranges ----------
    // Fed exactly as a wasm caller would: fetch a range, stage it. Nothing here reads the file during
    // the forward pass — only what was staged in advance is available, which is the browser's constraint.
    let staged = Arc::new(StagedBacking::new());
    staged.stage(0, header.clone());
    // A real browser stages the resident set (embeddings, norms, head) plus the layer runs it needs. This
    // stages every run, i.e. the "fetched it all, still streaming within it" case; a budgeted browser
    // would release runs as the tier evicts them.
    let src = GgufBacked::new(header.clone(), Arc::clone(&file_b)).unwrap();
    let mut staged_bytes = header.len() as u64;
    for t in &src.tensors {
        let (off, sz) = src.extent(&t.name).unwrap();
        let mut buf = vec![0u8; sz];
        file_b.read_at(off, &mut buf).unwrap();
        staged.stage(off, buf);
        staged_bytes += sz as u64;
    }
    let staged_dyn: Arc<dyn Backing + Send + Sync> = staged.clone();
    let l_stg = logits_of(staged_dyn).to_vec().await;
    let d_stg = max_abs_diff(&reference, &l_stg);
    println!("  {:<28} {:>11.1}M  {:>14.3e}   {}", "browser · StagedBacking",
             staged_bytes as f64 / 1e6, d_stg, arg(&l_stg[l_stg.len() - 151936..]));
    assert_eq!(d_stg, 0.0, "the staged (browser) backing changed the logits");
    assert_eq!(arg(&l_stg[l_stg.len() - 151936..]), ref_arg, "staged backing changed the predicted token");

    // ---------- the property the browser path lives or dies on ----------
    let empty = StagedBacking::new();
    let e = empty.read_at(1 << 20, &mut vec![0u8; 64]).unwrap_err();
    let named = matches!(e, TierError::NotStaged { .. });
    println!("\n  un-staged read → {}", e);
    assert!(named, "an un-staged read must be a NAMED error, not zeros and not a stall");

    println!("\n  ✅ IDENTICAL LOGITS from a file handle, from memory, and from asynchronously staged");
    println!("     ranges. One reader (`GgufBacked`), one tier policy, three embodiments — so the browser");
    println!("     runs the same code path as native rather than a wasm-only reimplementation of it.");
    println!("\n  The browser's constraint is real and handled rather than hidden: read_at is synchronous");
    println!("  and fetch is not, so bytes must be staged BEFORE the forward pass reads them. A miss is");
    println!("  `{}`,", "TierError::NotStaged");
    println!("  which names the range — because returning zeros would produce a model that runs and lies.");
    println!("\n  Layer weights: {:.1} MB of {:.1} MB.", layer_bytes as f64 / 1e6, total as f64 / 1e6);

    // ---------- 4. the STEPPING forward, which is what bounds a browser's peak ----------
    // A monolithic forward cannot pause, so every streamed layer must be resident before it starts —
    // which is what pinned the browser's peak at the whole model regardless of budget. Stepping lets a
    // caller stage one layer, apply it, release the previous. It must produce identical logits.
    let g = GgufBacked::new(header.clone(), Arc::clone(&file_b)).unwrap();
    let m = Qwen3::load(&ctx, &g).unwrap();
    let mut c1 = Cache::new(&m.cfg);
    let whole = m.forward_cached(&prompt, &mut c1).to_vec().await;

    let mut c2 = Cache::new(&m.cfg);
    let mut st = m.step_begin(&prompt, &c2);
    let mut applied = 0usize;
    while let Some(il) = st.next_layer() {
        assert_eq!(il, applied, "steps must run in order");
        // A browser awaits a fetch here, and releases layer il-1 after the call.
        if m.step_layer(&mut st, &mut c2) { break; }
        applied += 1;
    }
    let stepped = m.step_finish(st, &mut c2).to_vec().await;
    let d_step = max_abs_diff(&whole, &stepped);
    println!("  {:<28} {:>12}  {:>14.3e}   {}", "stepping forward", "per-layer", d_step,
             arg(&stepped[stepped.len() - 151936..]));
    assert_eq!(d_step, 0.0, "the stepping forward changed the logits");
    assert_eq!(c1.pos, c2.pos, "stepping left the KV cache at a different position");
    // ---------- 5. streamed EMBEDDINGS: the table is 21% of this checkpoint ----------
    // embed() only ever gathers the prompt's rows, so holding the whole 144.6 MB table is pure resident
    // weight for no benefit. Fetching rows on demand must not change a single logit.
    let staged_e = Arc::new(StagedBacking::new());
    staged_e.stage(0, header.clone());
    let src_e = GgufBacked::new(header.clone(), Arc::clone(&file_b)).unwrap();
    let (ebase, esz) = src_e.extent("token_embd.weight").unwrap();
    for t in &src_e.tensors {
        if t.name == "token_embd.weight" { continue; }         // deliberately NOT staged in full
        let (off, sz) = src_e.extent(&t.name).unwrap();
        let mut b = vec![0u8; sz];
        file_b.read_at(off, &mut b).unwrap();
        staged_e.stage(off, b);
    }
    let se_dyn: Arc<dyn Backing + Send + Sync> = staged_e.clone();
    let g_e = GgufBacked::new(header.clone(), Arc::clone(&se_dyn)).unwrap();
    // token_embd is absent from the staged set, so a resident load would fail here — stage the header
    // range it needs by loading through the FILE, then switch the table to streamed.
    let g_full = GgufBacked::new(header.clone(), Arc::clone(&file_b)).unwrap();
    let mut m_e = Qwen3::load(&ctx, &g_full).unwrap();
    m_e.stream_embeddings(Arc::clone(&se_dyn), ebase);
    // Stage ONLY the rows this prompt touches.
    let mut row_bytes_total = 0usize;
    for &t in &prompt {
        let (off, rb) = m_e.embd_row_extent(t, ebase);
        let mut b = vec![0u8; rb];
        file_b.read_at(off, &mut b).unwrap();
        staged_e.stage(off, b);
        row_bytes_total += rb;
    }
    let _ = g_e;
    let mut c3 = Cache::new(&m_e.cfg);
    let l_emb = m_e.forward_cached(&prompt, &mut c3).to_vec().await;
    let d_emb = max_abs_diff(&reference, &l_emb);
    println!("  {:<28} {:>11.1}K  {:>14.3e}   {}", "streamed embeddings",
             row_bytes_total as f64 / 1e3, d_emb, arg(&l_emb[l_emb.len() - 151936..]));
    assert_eq!(d_emb, 0.0, "streaming the embedding table changed the logits");
    println!("     ({} rows staged = {:.1} KB, versus {:.1} MB for the whole table — {:.0}x less)",
             prompt.len(), row_bytes_total as f64 / 1e3, esz as f64 / 1e6,
             esz as f64 / row_bytes_total as f64);

    println!("\n  ✅ The STEPPING forward is bit-identical to the monolithic one, so a browser can stage");
    println!("     one layer at a time and release the previous — bounding peak residency to the pinned");
    println!("     set plus one layer instead of the whole model.");
}

