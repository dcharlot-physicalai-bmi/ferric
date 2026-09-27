//! **Convert a LoRA adapter between PEFT and llama.cpp's GGUF adapter format**, checked against its base.
//!
//!   lora_convert <base.gguf> <adapter: PEFT dir | .gguf> <out: .gguf | PEFT dir> [base_model_id]
//!
//! The base supplies the architecture and head counts the q/k row order depends on (a `llama` GGUF stores
//! q/k permuted, so a PEFT adapter's q/k `lora_B` rows are permuted on the way in and back on the way out)
//! and every pair's shape is checked against it before anything is written.
//!
//! ⚠ PEFT → GGUF writes `adapter.lora.alpha = scaling · r`, so an rslora adapter keeps PEFT's `alpha/√r`
//! under llama.cpp's `alpha / rank` arithmetic. llama.cpp's own `convert_lora_to_gguf.py` writes
//! `lora_alpha` unchanged and loses it.
use ferric_gguf::{GgufFile, GgufSource, Meta};
use ferric_load::lora::{LoraAdapter, RowOrder};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        eprintln!("usage: lora_convert <base.gguf> <adapter> <out.gguf | out_dir> [base_model_id]");
        std::process::exit(2);
    }
    let die = |e: String| -> ! { eprintln!("⛔ {e}"); std::process::exit(1) };
    let g = GgufFile::open(&a[1]).unwrap_or_else(|e| die(e));
    let md = g.metadata();
    let arch = match md.get("general.architecture") { Some(Meta::Str(s)) => s.clone(), _ => die("base has no architecture".into()) };
    let u = |k: &str| match md.get(&format!("{arch}.{k}")) { Some(Meta::U(v)) => *v as usize, _ => die(format!("base: no {arch}.{k}")) };
    let (nh, nkv) = (u("attention.head_count"), u("attention.head_count_kv"));
    let ad = LoraAdapter::open(&a[2]).unwrap_or_else(|e| die(e));
    let out = std::path::Path::new(&a[3]);
    let to_gguf = out.extension().is_some_and(|x| x == "gguf");
    let bound = ad.bind(&arch, nh, nkv, if to_gguf { RowOrder::Gguf } else { RowOrder::Hf }).unwrap_or_else(|e| die(e));
    // Shapes are checked in the base's own order, whichever way the file is going.
    ad.bind(&arch, nh, nkv, RowOrder::Gguf).and_then(|b| b.check_base(&g)).unwrap_or_else(|e| die(e));
    if to_gguf {
        bound.save_gguf(out, &arch).unwrap_or_else(|e| die(e));
    } else {
        let id = a.get(4).cloned().or(ad.base_model.clone()).unwrap_or_else(|| die("give a base_model_id".into()));
        bound.save_peft(out, &id).unwrap_or_else(|e| die(e));
    }
    eprintln!("{} ({:?}, {} pairs, alpha {} r {} rslora {}) -> {}", ad.name, ad.format, ad.targets.len(),
              ad.alpha, ad.r, ad.use_rslora, out.display());
}
