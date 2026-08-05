use ferric_gguf::{GgufFile, GgufSource};
fn main() { pollster::block_on(async {
    let ctx = std::sync::Arc::new(ferric_core::Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();
    let p = format!("{home}/.cache/ferric/hub/Qwen_Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf");
    let g = GgufFile::open(&p).unwrap();
    let m = ferric_llama::qwen3::Qwen3::load(&ctx, &g).unwrap();
    let vn = m.cfg.n_vocab;
    let am = |l: &[f32]| l[l.len()-vn..].iter().enumerate().fold((0usize,f32::MIN),|a,(i,&x)| if x>a.1 {(i,x)} else {a}).0 as u32;
    const N: usize = 30;

    // ---- A: plain loop, one timer ----
    let mut c = ferric_llama::qwen3::Cache::new(&m.cfg);
    let _ = m.forward_cached(&[10u32,20,30], &mut c).to_vec().await;
    ferric_tensor::reset_host_ns();
    let t0 = std::time::Instant::now();
    let mut next = 100u32;
    for _ in 0..N { next = am(&m.forward_cached(&[next], &mut c).to_vec().await); }
    let plain = t0.elapsed().as_secs_f64()*1000.0 / N as f64;
    let (pl, bg, enc, buf) = ferric_tensor::host_ns();
    let ms = |n: u64| n as f64 / 1e6 / N as f64;

    // ---- B: same loop, split into build / await ----
    let mut c2 = ferric_llama::qwen3::Cache::new(&m.cfg);
    let _ = m.forward_cached(&[10u32,20,30], &mut c2).to_vec().await;
    let (mut b_ms, mut w_ms) = (0.0f64, 0.0f64);
    let mut next2 = 100u32;
    for _ in 0..N {
        let t1 = std::time::Instant::now();
        let t = m.forward_cached(&[next2], &mut c2);
        b_ms += t1.elapsed().as_secs_f64()*1000.0;
        let t2 = std::time::Instant::now();
        let l = t.to_vec().await;
        w_ms += t2.elapsed().as_secs_f64()*1000.0;
        next2 = am(&l);
    }
    println!("  A  plain loop total      {plain:>8.2} ms/token");
    println!("     instrumented host     {:>8.2} ms/token  ({:.0}%)", ms(pl+bg+enc+buf), 100.0*ms(pl+bg+enc+buf)/plain);
    println!("       info buffers        {:>8.2}", ms(buf));
    println!("       create_bind_group   {:>8.2}", ms(bg));
    println!("       pipeline lookup     {:>8.2}", ms(pl));
    println!("       encode pass         {:>8.2}", ms(enc));
    println!("  B  build phase           {:>8.2} ms/token", b_ms / N as f64);
    println!("     await phase           {:>8.2} ms/token", w_ms / N as f64);
    println!("     B total               {:>8.2} ms/token", (b_ms+w_ms) / N as f64);
    assert_eq!(next, next2, "the two loops diverged");
}); }
