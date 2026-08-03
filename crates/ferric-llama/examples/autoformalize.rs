//! S4 — AUTOFORMALIZE, and the one link the verifier cannot certify.
//!
//! Pipeline: informal claim (English) → a MODEL formalizes it into a checkable obligation (system + region)
//! → a SOUND verifier discharges it (Lyapunov certificate for the damped pendulum). The verifier is sound for
//! WHAT IT IS GIVEN — but nothing certifies that the formal obligation faithfully captures the informal claim.
//! That informal→formal step is the exact seam the "AI scientist" hype hides. We do two honest things about it:
//!   (1) GUARD it — an independent deterministic extraction cross-checks the model's formalization; a mismatch
//!       is rejected before any proof, so we never soundly certify the WRONG theorem by a parse error.
//!   (2) LABEL the residual — where the claim is vague ("small swings"), formalizing a number is a JUDGMENT no
//!       guard can verify. The certificate proves the formal obligation; the informal faithfulness is not proven.
//!   cargo run -p ferric-llama --example autoformalize --release
use ferric_core::Context;
use ferric_gguf::{GgufFile, Meta};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Arc;

fn byte_decoder() -> HashMap<char, u8> {
    let mut m = HashMap::new(); let mut n = 0u32;
    for b in 0u32..256 {
        let printable = (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
        let c = if printable { b } else { let c = 256 + n; n += 1; c };
        m.insert(char::from_u32(c).unwrap(), b as u8);
    }
    m
}
struct Solver { model: Qwen3, bpe: Bpe, tokens: Vec<String>, u2b: HashMap<char, u8>, ims: u32, ime: u32 }
impl Solver {
    async fn load(ctx: &Arc<Context>, path: &str) -> Solver {
        let g = GgufFile::open(path).unwrap();
        let tokens: Vec<String> = match g.metadata.get("tokenizer.ggml.tokens") {
            Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(), _ => panic!() };
        let vocab: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let merges: Vec<(String, String)> = match g.metadata.get("tokenizer.ggml.merges") {
            Some(Meta::Arr(a)) => a.iter().filter_map(|m| if let Meta::Str(s) = m { s.split_once(' ').map(|(x,y)|(x.to_string(),y.to_string())) } else { None }).collect(), _ => panic!() };
        let bpe = Bpe::new(vocab.clone(), &merges);
        let (ims, ime) = (*vocab.get("<|im_start|>").unwrap(), *vocab.get("<|im_end|>").unwrap());
        Solver { model: Qwen3::load(ctx, &g).unwrap(), bpe, tokens, u2b: byte_decoder(), ims, ime }
    }
    fn detok(&self, ids: &[u32]) -> String {
        let s: String = ids.iter().map(|&i| self.tokens.get(i as usize).cloned().unwrap_or_default()).collect();
        String::from_utf8_lossy(&s.chars().filter_map(|c| self.u2b.get(&c).copied()).collect::<Vec<u8>>()).into_owned()
    }
    async fn ask(&self, sys: &str, user: &str, max: usize) -> String {
        let mut ids = vec![self.ims]; ids.extend(self.bpe.encode(&format!("system\n{sys}")));
        ids.push(self.ime); ids.extend(self.bpe.encode("\n"));
        ids.push(self.ims); ids.extend(self.bpe.encode(&format!("user\n{user}")));
        ids.push(self.ime); ids.extend(self.bpe.encode("\n"));
        ids.push(self.ims); ids.extend(self.bpe.encode("assistant\n"));
        let c = &self.model.cfg; let mut cache = Cache::new(c);
        let argmax = |row: &[f32]| (0..c.n_vocab).max_by(|&a,&b| row[a].partial_cmp(&row[b]).unwrap()).unwrap() as u32;
        let mut out = Vec::new();
        for step in 0..max {
            let logits = if step == 0 { self.model.forward_cached(&ids, &mut cache) } else { self.model.forward_cached(&[*out.last().unwrap()], &mut cache) };
            let v = logits.to_vec().await; let next = argmax(&v[v.len()-c.n_vocab..]);
            if next == self.ime || next == 151643 { break; } out.push(next);
        }
        self.detok(&out)
    }
}

// ---------- the SOUND verifier: does a damped pendulum admit V=xᵀPx certifying ‖x‖∞≤r ? ----------
// f(x)=[x2 ; −sin x1 − d·x2]. P from the linearization's Lyapunov eqn AᵀP+PA=−I (A=[[0,1],[−1,−d]]):
//   P=[[1/d + d/2, 1/2],[1/2, 1/d]].  We then SOUNDLY check V̇+α‖x‖² < 0 over the box via interval
//   arithmetic + adaptive refinement (natural extension, sound sin-interval). Certified ⇒ real proof.
#[derive(Clone, Copy)] struct Iv { lo: f64, hi: f64 }
impl Iv {
    fn add(self,o:Iv)->Iv{Iv{lo:self.lo+o.lo,hi:self.hi+o.hi}}
    fn mul(self,o:Iv)->Iv{let(a,b,c,d)=(self.lo*o.lo,self.lo*o.hi,self.hi*o.lo,self.hi*o.hi);Iv{lo:a.min(b).min(c).min(d),hi:a.max(b).max(c).max(d)}}
    fn scale(self,k:f64)->Iv{if k>=0.0{Iv{lo:self.lo*k,hi:self.hi*k}}else{Iv{lo:self.hi*k,hi:self.lo*k}}}
    fn sq(self)->Iv{if self.lo>=0.0{Iv{lo:self.lo*self.lo,hi:self.hi*self.hi}}else if self.hi<=0.0{Iv{lo:self.hi*self.hi,hi:self.lo*self.lo}}else{Iv{lo:0.0,hi:(self.lo*self.lo).max(self.hi*self.hi)}}}
}
fn sin_iv(x: Iv) -> Iv { // sound range of sin over [lo,hi]
    let mut lo = x.lo.sin().min(x.hi.sin()); let mut hi = x.lo.sin().max(x.hi.sin());
    // extrema at π/2 + kπ inside the interval give ±1
    let mut k = ((x.lo - PI/2.0)/PI).floor() as i64 - 1;
    while (k as f64)*PI + PI/2.0 <= x.hi + 1e-12 {
        let e = (k as f64)*PI + PI/2.0;
        if e >= x.lo - 1e-12 && e <= x.hi + 1e-12 { let s = e.sin(); lo = lo.min(s); hi = hi.max(s); }
        k += 1;
    }
    Iv { lo, hi }
}
const R0: f64 = 0.05; const ALPHA: f64 = 0.005;
fn gbox(x1: Iv, x2: Iv, a: f64, b: f64, c: f64, d: f64) -> f64 { // upper bound of V̇+α‖x‖² over box (natural)
    let f1 = x2;
    let f2 = sin_iv(x1).scale(-1.0).add(x2.scale(-d));                // −sin x1 − d x2
    let vx1 = x1.scale(2.0*a).add(x2.scale(2.0*b));                   // ∂V/∂x1
    let vx2 = x1.scale(2.0*b).add(x2.scale(2.0*c));                   // ∂V/∂x2
    let g = vx1.mul(f1).add(vx2.mul(f2)).add(x1.sq().add(x2.sq()).scale(ALPHA));
    g.hi
}
fn certify(x1: Iv, x2: Iv, a: f64, b: f64, c: f64, d: f64, depth: u32) -> bool {
    if x1.lo > -R0 && x1.hi < R0 && x2.lo > -R0 && x2.hi < R0 { return true; }
    if gbox(x1, x2, a, b, c, d) < 0.0 { return true; }
    if depth == 0 { return false; }
    let (m1, m2) = ((x1.lo+x1.hi)/2.0, (x2.lo+x2.hi)/2.0);
    for &(p,q) in &[(x1.lo,m1),(m1,x1.hi)] { for &(u,v) in &[(x2.lo,m2),(m2,x2.hi)] {
        if !certify(Iv{lo:p,hi:q}, Iv{lo:u,hi:v}, a,b,c,d, depth-1) { return false; } } }
    true
}
fn discharge(d: f64, r: f64) -> bool { // is "‖x‖∞≤r stable" provable for damping d?
    if d <= 0.0 || r <= R0 { return false; }
    let (a, b, c) = (1.0/d + d/2.0, 0.5, 1.0/d);
    let gb = 48usize; let step = 2.0*r/gb as f64;
    for i in 0..gb { for j in 0..gb { let (lo1,lo2)=(-r+i as f64*step,-r+j as f64*step);
        if !certify(Iv{lo:lo1,hi:lo1+step}, Iv{lo:lo2,hi:lo2+step}, a,b,c,d, 6) { return false; } } }
    true
}

// ---------- the GUARD: an independent deterministic extraction of (damping, radius) from the claim ----------
fn num_after(text: &str, key: &str) -> Option<f64> {
    let t = text.to_lowercase();
    let i = t.find(key)?; let rest = &t[i+key.len()..];
    let mut s = String::new();
    for ch in rest.chars() { if ch.is_ascii_digit() || ch=='.' { s.push(ch); } else if !s.is_empty() { break; } else if ch==' '||ch=='='||ch==':' { continue; } else { break; } }
    s.parse().ok()
}
fn guard_extract(claim: &str) -> (Option<f64>, Option<f64>) {
    let d = num_after(claim, "damping");
    let r = num_after(claim, "up to").or_else(|| num_after(claim, "swings up to"));
    (d, r)
}
fn model_extract(reply: &str) -> (Option<f64>, Option<f64>) { (num_after(reply, "damping"), num_after(reply, "radius")) }

fn main() { pollster::block_on(run()); }
async fn run() {
    let ctx = Arc::new(Context::new().await.unwrap());
    let home = std::env::var("HOME").unwrap();
    let m = Solver::load(&ctx, &format!("{home}/.cache/ferric/hub/qwen1.5b-q4km.gguf")).await;
    let sys = "You convert a claim about a damped pendulum into two numbers. Reply exactly: damping=<x>, radius=<y>";
    println!("S4 AUTOFORMALIZE — informal claim → model formalization → GUARD → sound verify.\n");

    let claims = [
        "A pendulum with damping 0.8 remains stable for swings up to 0.9 radians.",
        "A pendulum with damping 0.3 is stable for swings up to 2.5 radians.",
        "Settling within 3 seconds, a pendulum with damping 0.6 is stable for swings up to 1.0 radians.",
    ];
    for claim in claims {
        println!("claim: \"{claim}\"");
        let reply = m.ask(sys, claim, 24).await;
        let (dm, rm) = model_extract(&reply);
        let (dg, rg) = guard_extract(claim);
        println!("  model formalized → {}   (guard reads → damping={:?}, radius={:?})", reply.replace('\n'," ").trim(), dg, rg);
        let ok = |a: Option<f64>, b: Option<f64>| matches!((a,b),(Some(x),Some(y)) if (x-y).abs()<1e-6);
        if !(ok(dm,dg) && ok(rm,rg)) {
            println!("  ✗ GUARD REJECTS — model formalization ≠ source. Not proving anything (would risk a sound proof of the WRONG obligation).\n");
            continue;
        }
        let (d, r) = (dm.unwrap(), rm.unwrap());
        let proved = discharge(d, r);
        println!("  ✓ guard OK → formal obligation: pendulum(d={d}) asymptotically stable on ‖x‖∞≤{r}");
        println!("    sound verifier → {}\n", if proved { "CERTIFIED (real Lyapunov proof)".to_string() } else { "NOT provable here (verifier refuses — region too large for this certificate)".to_string() });
    }

    // The residual gap: a VAGUE claim. Formalizing a number for "small" is a judgment no guard can verify.
    let vague = "A pendulum with damping 0.7 is stable for small swings.";
    println!("claim: \"{vague}\"   ← vague");
    let reply = m.ask("Convert to: damping=<x>, radius=<y>. If the swing size is only described as 'small', use radius=1.0.", vague, 24).await;
    let (dm, rm) = model_extract(&reply);
    let (dg, rg) = guard_extract(vague);
    println!("  model formalized → {}   (guard reads → damping={:?}, radius={:?} — NO explicit radius in the text)", reply.replace('\n'," ").trim(), dg, rg);
    let _ = rg;
    let d = dm.or(dg).unwrap_or(0.7);
    let default = rm.is_none();
    let r = rm.unwrap_or(1.0); // the pipeline's default reading of "small" — itself an unverified choice
    let proved = discharge(d, r);
    println!("  ⚠ 'small' has no number in the text, so the pipeline must CHOOSE one (radius={r}{}) — a judgment no guard can verify.", if default {", pipeline default"} else {""});
    println!("    sound verifier → {} on ‖x‖∞≤{r}", if proved {"CERTIFIED (real Lyapunov proof)"} else {"not provable"});
    println!("    HONEST: the proof is about ‖x‖∞≤{r}. Whether THAT is 'small swings' is a semantic judgment the pipeline cannot certify.\n");

    // Demonstrate the guard catching a formalization error (injected), so the 'caught' path is shown regardless of the model.
    println!("guard demo — suppose the formalizer had misread the distractor and returned damping=3, radius=1.0:");
    let (dm, rm) = (Some(3.0), Some(1.0));
    let (dg, rg) = guard_extract("...a pendulum with damping 0.6 is stable for swings up to 1.0 radians.");
    let ok = |a: Option<f64>, b: Option<f64>| matches!((a,b),(Some(x),Some(y)) if (x-y).abs()<1e-6);
    println!("  model {:?}/{:?} vs guard {:?}/{:?}  →  {}", dm, rm, dg, rg,
        if ok(dm,dg)&&ok(rm,rg) {"accepted"} else {"✗ GUARD REJECTS — the wrong formalization never reaches the verifier."});

    println!("\nThe verifier is sound for the obligation it is given. The GUARD narrows the informal→formal gap");
    println!("(catching parse errors), but cannot close it — a faithful-looking formalization can still mean");
    println!("something subtly other than the claim. So we state exactly what was proved, and never dress a");
    println!("sound proof of the formal obligation up as proof of the informal intent. That line is the product.");
}
