//! END-TO-END Cosmos 3 Edge ACTION ROLLOUT in pure Rust — the capstone. Runs Ferric's full packing
//! forward through Ferric's UniPC sampler for 4 denoising steps and matches the action chunk produced
//! by the REAL `Cosmos3OmniTransformer` driven by the REAL `UniPCMultistepScheduler`
//! (`cosmos_rollout_ref.py`). guidance_scale=1 (CFG verified separately in `cosmos_action_loop`).
//! usage: cargo run -p ferric-llama --example cosmos_action_rollout --release -- <cosmos3-edge-dir>
use ferric_core::Context;
use ferric_llama::cosmos::interleaved_mrope;
use ferric_llama::unipc::UniPc;
use ferric_load::safetensors_filtered;
use ferric_tensor::{nn, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

fn main() { pollster::block_on(run()); }
async fn run() {
    let dir = std::env::args().nth(1)
        .unwrap_or_else(|| format!("{}/.cache/ferric/cosmos3-edge", std::env::var("HOME").unwrap()));
    let gp = format!("{}/.cache/ferric/cosmos_ref/action_rollout_golden.json", std::env::var("HOME").unwrap());
    let jg = json_min::parse(&std::fs::read_to_string(&gp).expect("run cosmos_rollout_ref.py first"));
    let (alen, adim, lc) = (jg.get("ALEN").as_usize(), jg.get("ADIM").as_usize(), jg.get("LC").as_usize());
    let und_len = jg.get("und_len").as_usize();
    let seq_len = jg.get("seq_len").as_usize();
    let input_ids: Vec<u32> = jg.get("input_ids").as_f64_vec().iter().map(|x| *x as u32).collect();
    let pos = jg.get("position_ids").as_f64_mat();
    let vision_latent = jg.get("vision_latent").as_f64_vec();
    let init_action = jg.get("init_action").as_f64_vec();
    let timesteps = jg.get("timesteps").as_f64_vec();       // [995,973,798,128] (rounded — for temb)
    let final_ref = jg.get("final_action").as_f64_vec();

    let ctx = Arc::new(Context::new().await.unwrap());
    let (nh, nkv, hd, eps, d) = (16usize, 8usize, 128usize, 1e-5f32, 2048usize);

    // ---- load weights once ----
    let keep = |n: &str| n.starts_with("layers.") || matches!(n,
        "embed_tokens.weight" | "proj_in.weight" | "proj_in.bias" | "norm_moe_gen.weight"
        | "action_modality_embed" | "action_proj_in.fc.weight" | "action_proj_in.bias.weight"
        | "action_proj_out.fc.weight" | "action_proj_out.bias.weight") || n.starts_with("time_embedder.");
    let mut w: HashMap<String, ferric_load::STensor> = HashMap::new();
    for e in std::fs::read_dir(format!("{dir}/transformer")).unwrap() {
        let p = e.unwrap().path();
        if p.extension().and_then(|x| x.to_str()) == Some("safetensors") {
            w.extend(safetensors_filtered(&std::fs::read(&p).unwrap(), keep).unwrap());
        }
    }
    let g = |n: &str| { let s = &w[n]; Tensor::from_vec(&ctx, &s.data, &s.shape) };
    let raw = |n: &str| -> &[f32] { &w[n].data };

    // ---- fixed pieces (constant across denoising steps) ----
    let (pt, ph, pw): (Vec<i64>, Vec<i64>, Vec<i64>) = (
        pos[0].iter().map(|x| *x as i64).collect(),
        pos[1].iter().map(|x| *x as i64).collect(),
        pos[2].iter().map(|x| *x as i64).collect());
    let (cos, sin) = interleaved_mrope(&pt, &ph, &pw, hd, 1e8, (24, 20, 20));
    let cos_t = Tensor::from_vec(&ctx, &cos, &[seq_len, hd]);
    let sin_t = Tensor::from_vec(&ctx, &sin, &[seq_len, hd]);
    let cos_und = cos_t.narrow(0, 0, und_len);
    let sin_und = sin_t.narrow(0, 0, und_len);
    let cos_gen = cos_t.narrow(0, und_len, seq_len - und_len);
    let sin_gen = sin_t.narrow(0, und_len, seq_len - und_len);
    let und0 = g("embed_tokens.weight").gather_rows(&input_ids); // [L,2048] text (fixed)
    // vision patch (fixed) -> proj_in (temb added per step)
    let patch_dim = 4 * lc;
    let mut patch = vec![0f32; patch_dim];
    for c in 0..lc { for p_ in 0..2 { for q in 0..2 { patch[(p_ * 2 + q) * lc + c] = vision_latent[c * 4 + p_ * 2 + q] as f32; }}}
    let vis_base = Tensor::from_vec(&ctx, &patch, &[1, patch_dim]).matmul_bt(&g("proj_in.weight")).add(&g("proj_in.bias").reshape(&[1, d]));
    let (ta, tg) = (und_len, seq_len - und_len);
    let fc_in = raw("action_proj_in.fc.weight");
    let b_in = raw("action_proj_in.bias.weight");
    let fc_out = raw("action_proj_out.fc.weight");
    let b_out = raw("action_proj_out.bias.weight");

    // one full packing forward: (current action latents, step timestep) -> velocity [ALEN*ADIM]
    let forward = |lat: &[f64], timestep: f64| -> Vec<f32> {
        let sigma = (timestep as f32) * 0.001;
        let mut tp = vec![0f32; 256];
        for i in 0..128 { let f = (-(10000f32.ln()) * i as f32 / 128.0).exp(); tp[i] = (sigma * f).cos(); tp[128 + i] = (sigma * f).sin(); }
        let temb = Tensor::from_vec(&ctx, &tp, &[1, 256])
            .matmul_bt(&g("time_embedder.linear_1.weight")).add(&g("time_embedder.linear_1.bias").reshape(&[1, d]))
            .silu().matmul_bt(&g("time_embedder.linear_2.weight")).add(&g("time_embedder.linear_2.bias").reshape(&[1, d]));
        // action codec (host, domain 0) on the current latents
        let mut z = vec![0f32; alen * d];
        for a in 0..alen { for b in 0..d { let mut acc = b_in[b];
            for k in 0..adim { acc += lat[a * adim + k] as f32 * fc_in[k * d + b]; } z[a * d + b] = acc; }}
        let vis = vis_base.add(&temb);
        let act = Tensor::from_vec(&ctx, &z, &[alen, d]).add(&g("action_modality_embed").reshape(&[1, d])).add(&temb);
        let mut xu = und0.clone();
        let mut xg = vis.cat(&act, 0);
        let mlp = |h: &Tensor, u: &Tensor, dn: &Tensor| h.matmul_bt(u).relu2().matmul_bt(dn);
        for il in 0..28 {
            let b = |s: &str| g(&format!("layers.{il}.{s}"));
            let hu = xu.rmsnorm(&b("input_layernorm.weight"), eps);
            let qu = hu.matmul_bt(&b("self_attn.to_q.weight")).apply_rope_costable(&cos_und, &sin_und, nh, hd);
            let k_und = hu.matmul_bt(&b("self_attn.to_k.weight"));
            let ku = k_und.apply_rope_costable(&cos_und, &sin_und, nkv, hd);
            let ku_gen = k_und.reshape(&[ta, nkv, hd]).rmsnorm(&b("self_attn.k_norm_und_for_gen.weight"), eps).reshape(&[ta, nkv * hd]).apply_rope_costable(&cos_und, &sin_und, nkv, hd);
            let vu = hu.matmul_bt(&b("self_attn.to_v.weight"));
            let ou = nn::causal_attention(&qu, &ku, &vu, nh, nkv, 0.0);
            let xu1 = xu.add(&ou.matmul_bt(&b("self_attn.to_out.weight")));
            xu = xu1.add(&mlp(&xu1.rmsnorm(&b("post_attention_layernorm.weight"), eps), &b("mlp.up_proj.weight"), &b("mlp.down_proj.weight")));
            let hg = xg.rmsnorm(&b("input_layernorm_moe_gen.weight"), eps);
            let qg = hg.matmul_bt(&b("self_attn.add_q_proj.weight")).reshape(&[tg, nh, hd]).rmsnorm(&b("self_attn.norm_added_q.weight"), eps).reshape(&[tg, nh * hd]).apply_rope_costable(&cos_gen, &sin_gen, nh, hd);
            let kg = hg.matmul_bt(&b("self_attn.add_k_proj.weight")).reshape(&[tg, nkv, hd]).rmsnorm(&b("self_attn.norm_added_k.weight"), eps).reshape(&[tg, nkv * hd]).apply_rope_costable(&cos_gen, &sin_gen, nkv, hd);
            let vg = hg.matmul_bt(&b("self_attn.add_v_proj.weight"));
            let og = nn::full_attention_kv(&qg, &ku_gen.cat(&kg, 0), &vu.cat(&vg, 0), nh, nkv);
            let xg1 = xg.add(&og.matmul_bt(&b("self_attn.to_add_out.weight")));
            xg = xg1.add(&mlp(&xg1.rmsnorm(&b("post_attention_layernorm_moe_gen.weight"), eps), &b("mlp_moe_gen.up_proj.weight"), &b("mlp_moe_gen.down_proj.weight")));
        }
        let gen_out = xg.rmsnorm(&g("norm_moe_gen.weight"), eps);
        let act_hidden = pollster::block_on(gen_out.narrow(0, 1, alen).to_vec()); // [ALEN,2048]
        let mut vel = vec![0f32; alen * adim];
        for a in 0..alen { for b in 0..adim { let mut acc = b_out[b];
            for k in 0..d { acc += act_hidden[a * d + k] * fc_out[k * adim + b]; } vel[a * adim + b] = acc; }}
        vel
    };

    // ---- 4-step UniPC rollout ----
    let mut sched = UniPc::new(4);
    let mut lat: Vec<f64> = init_action.clone();
    let t0 = std::time::Instant::now();
    for i in 0..4 {
        let v: Vec<f64> = forward(&lat, timesteps[i]).iter().map(|x| *x as f64).collect();
        lat = sched.step(&v, &lat);
        println!("  step {i} t={:.0}: latent[0,:4]={:?}", timesteps[i], round(&lat[..4], 5));
    }
    println!("rollout (4× packing forward + UniPC) {:?}", t0.elapsed());
    let maxerr = lat.iter().zip(&final_ref).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    println!("Ferric final action[0,:6] = {:?}", round(&lat[..6], 5));
    println!("real   final action[0,:6] = {:?}", round(&final_ref[..6], 5));
    println!("Ferric sum={:.5}  real sum={:.5}", lat.iter().sum::<f64>(), final_ref.iter().sum::<f64>());
    println!("\nMAX final-chunk Δ vs real transformer+UniPC rollout = {maxerr:.3e}  ->  {}",
             if maxerr < 3e-3 { "MATCH ✓" } else { "MISMATCH ✗" });
    assert!(maxerr < 3e-3, "Ferric action rollout diverged from the real Cosmos 3 Edge rollout");
}

fn round(a: &[f64], dgt: i32) -> Vec<f64> { let m = 10f64.powi(dgt); a.iter().map(|x| (x * m).round() / m).collect() }

mod json_min {
    pub struct Val(nd::Node);
    pub fn parse(s: &str) -> Val { Val(nd::parse(s)) }
    impl Val {
        pub fn get(&self, k: &str) -> Val { Val(self.0.get(k)) }
        pub fn as_usize(&self) -> usize { self.0.as_f64() as usize }
        pub fn as_f64_vec(&self) -> Vec<f64> { self.0.as_vec().iter().map(|n| n.as_f64()).collect() }
        pub fn as_f64_mat(&self) -> Vec<Vec<f64>> {
            self.0.as_vec().iter().map(|r| r.as_vec().iter().map(|n| n.as_f64()).collect()).collect()
        }
    }
    mod nd {
        #[derive(Clone)]
        pub enum Node { Num(f64), Arr(Vec<Node>), Obj(Vec<(String, Node)>), Null }
        impl Node {
            pub fn get(&self, k: &str) -> Node {
                if let Node::Obj(m) = self { for (kk, v) in m { if kk == k { return v.clone(); } } }
                Node::Null
            }
            pub fn as_f64(&self) -> f64 { if let Node::Num(n) = self { *n } else { f64::NAN } }
            pub fn as_vec(&self) -> Vec<Node> { if let Node::Arr(a) = self { a.clone() } else { vec![] } }
        }
        pub fn parse(s: &str) -> Node { let b = s.as_bytes(); let mut i = 0; pv(b, &mut i) }
        fn ws(b: &[u8], i: &mut usize) { while *i < b.len() && (b[*i] as char).is_whitespace() { *i += 1; } }
        fn pv(b: &[u8], i: &mut usize) -> Node {
            ws(b, i);
            match b[*i] { b'{' => po(b, i), b'[' => pa(b, i), b'"' => { ps(b, i); Node::Null } _ => pn(b, i) }
        }
        fn po(b: &[u8], i: &mut usize) -> Node {
            *i += 1; let mut m = Vec::new();
            loop {
                ws(b, i); if b[*i] == b'}' { *i += 1; break; }
                let k = ps(b, i); ws(b, i); *i += 1; let v = pv(b, i); m.push((k, v));
                ws(b, i); if b[*i] == b',' { *i += 1; } else if b[*i] == b'}' { *i += 1; break; }
            }
            Node::Obj(m)
        }
        fn pa(b: &[u8], i: &mut usize) -> Node {
            *i += 1; let mut a = Vec::new();
            loop {
                ws(b, i); if b[*i] == b']' { *i += 1; break; }
                a.push(pv(b, i)); ws(b, i);
                if b[*i] == b',' { *i += 1; } else if b[*i] == b']' { *i += 1; break; }
            }
            Node::Arr(a)
        }
        fn ps(b: &[u8], i: &mut usize) -> String {
            *i += 1; let s = *i; while b[*i] != b'"' { *i += 1; }
            let r = String::from_utf8_lossy(&b[s..*i]).to_string(); *i += 1; r
        }
        fn pn(b: &[u8], i: &mut usize) -> Node {
            let s = *i;
            while *i < b.len() && matches!(b[*i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') { *i += 1; }
            Node::Num(std::str::from_utf8(&b[s..*i]).unwrap().parse().unwrap())
        }
    }
}
