//! **Persistent KV sessions** — resume a long prompt from disk instead of re-prefilling it.
//!
//! A 25k-token agent prompt costs tens of seconds of prefill and is very nearly the same bytes on every
//! request. Storing the attention state and resuming from it turns that into a disk read. The state is
//! opaque here: this module owns the *identity and safety* of a checkpoint, not its layout, so it works
//! for any model.
//!
//! ## The key is the rendered BYTE prefix, not the token sequence
//!
//! This is the design decision everything else follows from, and it is not the obvious one.
//!
//! A model samples a token whose text a client may later send back as *two* differently-tokenised
//! tokens. Key on tokens and that request misses, re-prefills, and looks like a cache that "just doesn't
//! work sometimes". Key on the rendered bytes and it hits, because the bytes are what both sides agree
//! on. So a checkpoint answers exactly one question: **are these bytes a prefix of the incoming prompt?**
//!
//! ## The trap that silently breaks resumption
//!
//! Having matched a byte prefix, it is tempting to take the already-tokenised full prompt and slice off
//! the tokens past that byte offset. **That is wrong.** BPE merges across the boundary: the token
//! spanning the seam is generally not the concatenation of a prefix token and a suffix token. The
//! resumed sequence then differs from what the model actually cached, and the divergence is silent —
//! output degrades without any error.
//!
//! The suffix must be tokenised *after* the cache decision, from the raw text. [`ResumePlan`] returns the
//! exact prefix tokens and the byte range that still needs tokenising, and never a token slice.

use crate::TierError;
use std::path::{Path, PathBuf};

/// Why a checkpoint was written. Feeds eviction: deliberate anchors outlive incidental waypoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Reason {
    Unknown = 0,
    /// Written after a long first prompt, before generation — the reusable scaffolding.
    Cold = 1,
    /// A periodic waypoint inside a long generation.
    Continued = 2,
    /// Written because an unrelated request was about to replace the live session.
    Evict = 3,
    Shutdown = 4,
}

impl Reason {
    fn from_u8(v: u8) -> Reason {
        match v {
            1 => Reason::Cold,
            2 => Reason::Continued,
            3 => Reason::Evict,
            4 => Reason::Shutdown,
            _ => Reason::Unknown,
        }
    }
    /// Anchors are worth keeping: they are the checkpoints a future session is most likely to land on.
    fn is_anchor(self) -> bool {
        matches!(self, Reason::Cold | Reason::Evict | Reason::Shutdown)
    }
}

const MAGIC: [u8; 3] = *b"KVC";
const VERSION: u8 = 1;
/// Bumping this invalidates every stored checkpoint. It must change whenever the payload layout changes,
/// because a payload from an older layout is not detectably wrong — it is merely the wrong numbers.
pub const PAYLOAD_ABI: u8 = 1;
/// Fixed header, byte-for-byte. Written and read against this table rather than by appending fields in
/// whatever order they were thought of — the two drifted apart the first time and only a size assertion
/// caught it. Reserved bytes exist so a field can be added without moving anything after it.
///
/// ```text
///  0..3  magic "KVC"        3     version
///  4     reserved           5     reason
///  6     reserved           7     payload ABI
///  8..12 token count       12..16 hit count
/// 16..24 model id          24..32 created (unix s)
/// 32..40 last used         40..48 payload length
/// ```
const HEADER: usize = 48;

/// Identity of the model a checkpoint belongs to. A checkpoint is meaningless to a different model, and
/// nothing in the bytes themselves would reveal the mistake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelId(pub u64);

#[derive(Debug, Clone)]
pub struct Entry {
    pub sha: [u8; 20],
    pub text_bytes: usize,
    pub tokens: u32,
    pub hits: u32,
    pub created: u64,
    pub last_used: u64,
    pub file_size: u64,
    pub reason: Reason,
    pub model: ModelId,
}

#[derive(Debug, Clone, Copy)]
pub struct StoreOptions {
    /// Below this, a checkpoint is not worth its bytes.
    pub min_tokens: usize,
    /// Tokens trimmed from the tail before storing.
    ///
    /// The last few tokens of a prompt are the most likely to retokenise differently once more text is
    /// appended, so storing right up to the frontier maximises the chance of a near-miss.
    pub boundary_trim: usize,
    /// Store length is aligned down to a multiple of this.
    ///
    /// Independent sessions then land on the *same* offsets and can share checkpoints, instead of each
    /// writing a slightly different frontier that nothing else can ever hit.
    pub boundary_align: usize,
    /// Half-life for the hit counter in eviction scoring.
    pub hit_half_life_secs: f64,
}

impl Default for StoreOptions {
    fn default() -> Self {
        Self { min_tokens: 512, boundary_trim: 32, boundary_align: 2048, hit_half_life_secs: 6.0 * 3600.0 }
    }
}

/// What a caller must do to resume from a checkpoint.
#[derive(Debug, Clone)]
pub struct ResumePlan {
    /// Tokens to reuse verbatim — exactly what the model cached.
    pub prefix_tokens: Vec<u32>,
    /// Byte range of the prompt that still needs tokenising.
    ///
    /// **A byte range, deliberately, not a token slice.** See the module docs: slicing the already
    /// tokenised prompt at this boundary is the silent-corruption bug this type exists to prevent.
    pub suffix_bytes: std::ops::Range<usize>,
    pub payload: Vec<u8>,
}

/// Content-addressed store of KV checkpoints on disk.
#[derive(Debug)]
pub struct KvStore {
    dir: PathBuf,
    budget: u64,
    opts: StoreOptions,
    model: ModelId,
}

impl KvStore {
    pub fn open(dir: impl AsRef<Path>, budget_bytes: u64, model: ModelId, opts: StoreOptions) -> Result<Self, TierError> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir).map_err(|e| TierError::Io(format!("mkdir {}: {e}", dir.display())))?;
        Ok(Self { dir, budget: budget_bytes, opts, model })
    }

    fn path_for(&self, sha: &[u8; 20]) -> PathBuf { self.dir.join(format!("{}.kv", hex(sha))) }

    /// Length to store at: trimmed back from the frontier and aligned down, or `None` if too short.
    pub fn store_len(&self, tokens: usize) -> Option<usize> {
        let t = tokens.checked_sub(self.opts.boundary_trim)?;
        let aligned = if self.opts.boundary_align > 0 { t - (t % self.opts.boundary_align) } else { t };
        (aligned >= self.opts.min_tokens).then_some(aligned)
    }

    /// Write a checkpoint for `rendered_text` (the exact bytes the model consumed) and its `tokens`.
    ///
    /// Returns `false` when the checkpoint is not worth storing. Written to a temp file and renamed, so a
    /// crash mid-write leaves the previous checkpoint intact rather than a half-file that parses.
    pub fn store(
        &self,
        rendered_text: &str,
        tokens: &[u32],
        payload: &[u8],
        reason: Reason,
    ) -> Result<bool, TierError> {
        if tokens.len() < self.opts.min_tokens { return Ok(false); }
        let sha = sha1(rendered_text.as_bytes());
        let mut buf = Vec::with_capacity(HEADER + rendered_text.len() + payload.len() + 8);
        let t = now();
        buf.resize(HEADER, 0);
        buf[0..3].copy_from_slice(&MAGIC);
        buf[3] = VERSION;
        buf[5] = reason as u8;
        buf[7] = PAYLOAD_ABI;
        buf[8..12].copy_from_slice(&(tokens.len() as u32).to_le_bytes());
        buf[12..16].copy_from_slice(&0u32.to_le_bytes()); // hits
        buf[16..24].copy_from_slice(&self.model.0.to_le_bytes());
        buf[24..32].copy_from_slice(&t.to_le_bytes());
        buf[32..40].copy_from_slice(&t.to_le_bytes());
        buf[40..48].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(rendered_text.len() as u32).to_le_bytes());
        buf.extend_from_slice(rendered_text.as_bytes());
        buf.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
        for t in tokens { buf.extend_from_slice(&t.to_le_bytes()); }
        buf.extend_from_slice(payload);

        self.evict_for(buf.len() as u64, &sha)?;
        let final_path = self.path_for(&sha);
        let tmp = self.dir.join(format!("{}.tmp{}", hex(&sha), std::process::id()));
        std::fs::write(&tmp, &buf).map_err(|e| TierError::Io(format!("write {}: {e}", tmp.display())))?;
        std::fs::rename(&tmp, &final_path)
            .map_err(|e| TierError::Io(format!("rename into {}: {e}", final_path.display())))?;
        Ok(true)
    }

    /// Every valid checkpoint currently on disk.
    pub fn entries(&self) -> Vec<Entry> {
        let mut out = Vec::new();
        let Ok(rd) = std::fs::read_dir(&self.dir) else { return out };
        for e in rd.flatten() {
            let p = e.path();
            let Some(name) = p.file_name().and_then(|s| s.to_str()) else { continue };
            // Only files whose name IS their content hash. Anything else in the directory is ignored
            // rather than parsed, so a stray temp file can never be mistaken for a checkpoint.
            let Some(stem) = name.strip_suffix(".kv") else { continue };
            let Some(sha) = unhex(stem) else { continue };
            let Ok(bytes) = std::fs::read(&p) else { continue };
            if let Some(ent) = parse_header(&bytes, &sha) { out.push(ent); }
        }
        out
    }

    /// Find the longest checkpoint whose stored text is a byte prefix of `prompt_text`.
    pub fn lookup(&self, prompt_text: &str) -> Option<Entry> {
        let mut best: Option<Entry> = None;
        for e in self.entries() {
            if e.model != self.model || e.text_bytes > prompt_text.len() { continue; }
            // Cheap reject on the hash before touching the file body.
            if sha1(&prompt_text.as_bytes()[..e.text_bytes]) != e.sha { continue; }
            if best.as_ref().is_none_or(|b| {
                (e.text_bytes, e.tokens) > (b.text_bytes, b.tokens)
            }) {
                best = Some(e);
            }
        }
        best
    }

    /// Load a checkpoint and produce the plan for resuming it.
    ///
    /// The validation ladder here is deliberately paranoid, because every rung guards a failure that is
    /// otherwise silent: a checkpoint from another model, from another payload layout, or whose contents
    /// no longer match the name it is filed under.
    pub fn resume(&self, prompt_text: &str, entry: &Entry) -> Result<ResumePlan, TierError> {
        let path = self.path_for(&entry.sha);
        let b = std::fs::read(&path).map_err(|e| TierError::Io(format!("read {}: {e}", path.display())))?;
        let ent = parse_header(&b, &entry.sha)
            .ok_or_else(|| TierError::Io("checkpoint header failed validation".into()))?;
        if ent.model != self.model {
            return Err(TierError::Io("checkpoint belongs to a different model".into()));
        }

        let mut p = HEADER;
        let tl = rd_u32(&b, &mut p)? as usize;
        if p + tl > b.len() { return Err(TierError::Io("truncated rendered text".into())); }
        let text = &b[p..p + tl];
        p += tl;

        // Re-verify that the stored text really hashes to the filename. Guards the case where a file was
        // copied, renamed, or half-written by an older build: without this, a checkpoint could be served
        // for bytes it does not actually represent.
        if sha1(text) != entry.sha {
            return Err(TierError::Io("stored text does not match its own filename hash".into()));
        }
        // And that it is genuinely a prefix of THIS prompt. The hash comparison in `lookup` already
        // implies it, but a literal compare costs nothing and turns a hash collision from a wrong answer
        // into an error.
        if text.len() > prompt_text.len() || text != &prompt_text.as_bytes()[..text.len()] {
            return Err(TierError::Io("checkpoint text is not a prefix of this prompt".into()));
        }

        let nt = rd_u32(&b, &mut p)? as usize;
        if p + nt * 4 > b.len() { return Err(TierError::Io("truncated token vector".into())); }
        let mut prefix_tokens = Vec::with_capacity(nt);
        for i in 0..nt {
            prefix_tokens.push(u32::from_le_bytes(b[p + i * 4..p + i * 4 + 4].try_into().unwrap()));
        }
        p += nt * 4;
        if prefix_tokens.len() as u32 != ent.tokens {
            return Err(TierError::Io("token count disagrees with the header".into()));
        }

        Ok(ResumePlan {
            prefix_tokens,
            suffix_bytes: text.len()..prompt_text.len(),
            payload: b[p..].to_vec(),
        })
    }

    /// Free space for an incoming checkpoint of `incoming` bytes.
    ///
    /// Score is **token density with a decaying hit bonus**: `(hits' + 1) · tokens / bytes`, where
    /// `hits' = hits · 2^(−age/half_life)`, doubled for anchors. Density rather than raw size, because
    /// the point of the cache is tokens-not-prefilled per byte spent; decay rather than a raw count, so a
    /// checkpoint that was popular yesterday does not hold space forever.
    fn evict_for(&self, incoming: u64, keep: &[u8; 20]) -> Result<(), TierError> {
        let mut es = self.entries();
        let total: u64 = es.iter().map(|e| e.file_size).sum();
        if total + incoming <= self.budget { return Ok(()); }

        let now = now();
        let score = |e: &Entry| -> f64 {
            let age = now.saturating_sub(e.last_used) as f64;
            let decayed = e.hits as f64 * 2f64.powf(-age / self.opts.hit_half_life_secs);
            let d = (decayed + 1.0) * e.tokens as f64 / e.file_size.max(1) as f64;
            if e.reason.is_anchor() { d * 2.0 } else { d }
        };
        // Lowest score first; ties broken by least-recently-used.
        es.sort_by(|a, b| {
            score(a).total_cmp(&score(b)).then(a.last_used.cmp(&b.last_used))
        });

        let mut freed = 0u64;
        for e in es {
            if &e.sha == keep { continue; }
            if total + incoming - freed <= self.budget { break; }
            let p = self.path_for(&e.sha);
            if std::fs::remove_file(&p).is_ok() { freed += e.file_size; }
        }
        Ok(())
    }
}

fn parse_header(b: &[u8], expect_sha: &[u8; 20]) -> Option<Entry> {
    if b.len() < HEADER || b[0..3] != MAGIC || b[3] != VERSION { return None; }
    // Reject a payload layout this build cannot interpret. The bytes would parse and be wrong.
    if b[7] != PAYLOAD_ABI { return None; }
    let tokens = u32::from_le_bytes(b[8..12].try_into().ok()?);
    if tokens == 0 { return None; }
    Some(Entry {
        sha: *expect_sha,
        text_bytes: {
            let mut p = HEADER;
            rd_u32(b, &mut p).ok()? as usize
        },
        tokens,
        hits: u32::from_le_bytes(b[12..16].try_into().ok()?),
        model: ModelId(u64::from_le_bytes(b[16..24].try_into().ok()?)),
        created: u64::from_le_bytes(b[24..32].try_into().ok()?),
        last_used: u64::from_le_bytes(b[32..40].try_into().ok()?),
        file_size: b.len() as u64,
        reason: Reason::from_u8(b[5]),
    })
}

fn rd_u32(b: &[u8], p: &mut usize) -> Result<u32, TierError> {
    if *p + 4 > b.len() { return Err(TierError::Io("truncated checkpoint".into())); }
    let v = u32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
    *p += 4;
    Ok(v)
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn hex(b: &[u8; 20]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex(s: &str) -> Option<[u8; 20]> {
    if s.len() != 40 { return None; }
    let mut o = [0u8; 20];
    for i in 0..20 {
        o[i] = u8::from_str_radix(s.get(i * 2..i * 2 + 2)?, 16).ok()?;
    }
    Some(o)
}

/// SHA-1, inline. A dependency-free hash keeps this crate buildable for wasm and keeps the on-disk names
/// interoperable with the convention the frontier engines use. Not used for anything adversarial.
fn sha1(msg: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x6745_2301, 0xEFCD_AB89, 0x98BA_DCFE, 0x1032_5476, 0xC3D2_E1F0];
    let mut data = msg.to_vec();
    let bitlen = (msg.len() as u64) * 8;
    data.push(0x80);
    while data.len() % 64 != 56 { data.push(0); }
    data.extend_from_slice(&bitlen.to_be_bytes());

    for chunk in data.chunks_exact(64) {
        let mut w = [0u32; 80];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A82_7999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
                _ => (b ^ c ^ d, 0xCA62_C1D6),
            };
            let t = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(wi);
            e = d; d = c; c = b.rotate_left(30); b = a; a = t;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
    let mut o = [0u8; 20];
    for i in 0..5 { o[i * 4..i * 4 + 4].copy_from_slice(&h[i].to_be_bytes()); }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir()
            .join(format!("ferric-kv-{tag}-{}-{}", std::process::id(), now()));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn sha1_matches_known_vectors() {
        assert_eq!(hex(&sha1(b"")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(hex(&sha1(b"abc")), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(
            hex(&sha1(b"The quick brown fox jumps over the lazy dog")),
            "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12"
        );
        // Multi-block, to exercise the padding path.
        assert_eq!(hex(&sha1(&vec![b'a'; 1000])), "291e9a6c66994949b57ba5e650361e98fc36b1ba");
    }

    fn store(dir: &Path, model: u64) -> KvStore {
        KvStore::open(dir, 1 << 20, ModelId(model),
            StoreOptions { min_tokens: 4, boundary_trim: 0, boundary_align: 0, ..Default::default() })
            .unwrap()
    }

    #[test]
    fn round_trips_and_resumes_from_a_byte_prefix() {
        let d = tmpdir("rt");
        let s = store(&d, 7);
        let text = "SYSTEM PROMPT scaffolding that never changes.";
        let toks: Vec<u32> = (1..=9).collect();
        assert!(s.store(text, &toks, b"payload-bytes", Reason::Cold).unwrap());

        let prompt = format!("{text} and then the new user turn");
        let e = s.lookup(&prompt).expect("should hit on a byte prefix");
        let plan = s.resume(&prompt, &e).unwrap();
        assert_eq!(plan.prefix_tokens, toks);
        assert_eq!(plan.payload, b"payload-bytes");
        assert_eq!(&prompt[plan.suffix_bytes.clone()], " and then the new user turn");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_plan_never_hands_back_a_token_slice() {
        // THE trap. A checkpoint's text ends mid-word; BPE would merge across that seam, so the tokens of
        // the full prompt are NOT prefix_tokens ++ (some suffix of the full tokenisation). The plan
        // therefore returns a BYTE RANGE, forcing the caller to retokenise. This test encodes the shape
        // of the contract; a `Vec<u32>` suffix field would make the bug expressible again.
        let d = tmpdir("bpe");
        let s = store(&d, 1);
        let text = "the quick brown fo";           // deliberately mid-word
        s.store(text, &[10, 11, 12, 13, 14], b"p", Reason::Cold).unwrap();
        let prompt = "the quick brown fox jumps";
        let e = s.lookup(prompt).unwrap();
        let plan = s.resume(prompt, &e).unwrap();
        assert_eq!(&prompt[plan.suffix_bytes.clone()], "x jumps");
        assert_eq!(plan.suffix_bytes.start, text.len(), "suffix must begin exactly at the stored bytes");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_checkpoint_from_another_model_is_refused() {
        let d = tmpdir("model");
        let a = store(&d, 111);
        a.store("shared text prefix", &[1, 2, 3, 4, 5], b"x", Reason::Cold).unwrap();
        let b = store(&d, 222);
        assert!(b.lookup("shared text prefix and more").is_none(), "served another model's checkpoint");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_payload_from_another_abi_is_refused() {
        // The failure this prevents is the nastiest kind: bytes from an older layout parse cleanly and
        // are simply the wrong numbers.
        let d = tmpdir("abi");
        let s = store(&d, 5);
        s.store("some text prefix here", &[1, 2, 3, 4, 5], b"payload", Reason::Cold).unwrap();
        let f = std::fs::read_dir(&d).unwrap().next().unwrap().unwrap().path();
        let mut b = std::fs::read(&f).unwrap();
        b[7] = PAYLOAD_ABI + 1;
        std::fs::write(&f, &b).unwrap();
        assert!(s.lookup("some text prefix here and more").is_none(), "accepted a foreign payload ABI");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_file_whose_contents_do_not_match_its_name_is_refused() {
        // Guards a copied/renamed checkpoint being served for bytes it does not represent.
        let d = tmpdir("rename");
        let s = store(&d, 3);
        s.store("original text content", &[1, 2, 3, 4, 5], b"p", Reason::Cold).unwrap();
        let f = std::fs::read_dir(&d).unwrap().next().unwrap().unwrap().path();
        let mut b = std::fs::read(&f).unwrap();
        let off = HEADER + 4;
        b[off] = b'X'; // corrupt the stored text, leave the filename alone
        std::fs::write(&f, &b).unwrap();
        let prompt = "Xriginal text content plus more";
        if let Some(e) = s.lookup(prompt) {
            assert!(s.resume(prompt, &e).is_err(), "served a file whose text does not hash to its name");
        }
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn longest_matching_prefix_wins() {
        let d = tmpdir("longest");
        let s = store(&d, 9);
        s.store("aaaa", &[1, 2, 3, 4], b"short", Reason::Cold).unwrap();
        s.store("aaaaaaaa", &[1, 2, 3, 4, 5, 6, 7, 8], b"long", Reason::Cold).unwrap();
        let e = s.lookup("aaaaaaaaaaaa").unwrap();
        assert_eq!(e.tokens, 8, "picked the shorter checkpoint and threw away reusable prefill");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn store_len_trims_and_aligns_so_sessions_can_share_checkpoints() {
        let d = tmpdir("align");
        let s = KvStore::open(&d, 1 << 20, ModelId(1), StoreOptions::default()).unwrap();
        // 5000 tokens -> trim 32 -> 4968 -> align down to 2048 -> 4096.
        assert_eq!(s.store_len(5000), Some(4096));
        // Two sessions at different frontiers land on the SAME offset, which is the point: otherwise each
        // writes a unique frontier that nothing else can ever hit.
        assert_eq!(s.store_len(4500), s.store_len(4300));
        assert_eq!(s.store_len(100), None, "below min_tokens must not be stored");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn eviction_keeps_dense_anchors_over_sparse_waypoints() {
        // Budget forces a choice. A big low-token 'continued' waypoint should go before a compact
        // high-token 'cold' anchor: the cache's job is tokens-not-prefilled per byte held.
        let d = tmpdir("evict");
        let opts = StoreOptions { min_tokens: 4, boundary_trim: 0, boundary_align: 0, ..Default::default() };
        // Room for roughly one big entry plus the small one.
        let s = KvStore::open(&d, 6000, ModelId(1), opts).unwrap();
        s.store("anchor text", &(0..200).collect::<Vec<u32>>(), &[0u8; 64], Reason::Cold).unwrap();
        s.store("waypoint text", &(0..10).collect::<Vec<u32>>(), &[0u8; 4000], Reason::Continued).unwrap();
        // This third store forces eviction.
        s.store("third text", &(0..50).collect::<Vec<u32>>(), &[0u8; 3000], Reason::Cold).unwrap();
        let names: Vec<u32> = s.entries().iter().map(|e| e.tokens).collect();
        assert!(names.contains(&200), "evicted the dense anchor: {names:?}");
        assert!(!names.contains(&10), "kept the sparse waypoint: {names:?}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_stray_temp_file_is_never_mistaken_for_a_checkpoint() {
        let d = tmpdir("stray");
        let s = store(&d, 1);
        s.store("real checkpoint text", &[1, 2, 3, 4, 5], b"p", Reason::Cold).unwrap();
        std::fs::write(d.join("garbage.tmp999"), b"not a checkpoint at all").unwrap();
        std::fs::write(d.join("deadbeef.kv"), b"short").unwrap(); // wrong-length name
        assert_eq!(s.entries().len(), 1, "parsed something that was not a checkpoint");
        let _ = std::fs::remove_dir_all(&d);
    }
}
