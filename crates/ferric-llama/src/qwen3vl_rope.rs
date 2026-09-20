//! **Qwen3-VL's mRoPE position index for a mixed text+image sequence.**
//!
//! Three position components (temporal, height, width) per token. Text tokens carry the same value
//! in all three; image tokens carry their grid coordinates, offset by wherever the image starts.
//!
//! ⛔ THE LINE THAT IS NOT GUESSABLE: after an image the position advances by
//! `max(grid_h, grid_w) / merge` — **not** by the number of image tokens it contributed. A 4x8 patch
//! grid is 8 LLM tokens but advances the position by 4. Advancing by the token count instead leaves
//! every later text token shifted, which is fluent, position-scaled and invisible to every shape
//! check — the same failure mode as [`crate::qwen3vl_vision`]'s row reorder.
//!
//! ⚠ Video (`mm_token_type_id == 2`) is refused rather than folded into the image path: the
//! published code splits video grids by timestamp first, and that is a different function.

/// Per-token positions, section-major, ready for an mRoPE kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RopeIndex {
    pub t: Vec<i64>,
    pub h: Vec<i64>,
    pub w: Vec<i64>,
    /// `max(position) + 1 - seq_len` — what a later decode step adds to continue the sequence.
    pub delta: i64,
}

/// `types[i]` is 0 for text and 1 for an image token; `grids` are `(t, h, w)` PATCH grids, one per
/// image, in order.
pub fn rope_index(types: &[u8], grids: &[(usize, usize, usize)], merge: usize)
    -> Result<RopeIndex, String>
{
    if merge == 0 { return Err("merge size is 0".into()); }
    if types.is_empty() { return Err("empty sequence".into()); }
    if let Some(bad) = types.iter().find(|&&x| x > 1) {
        return Err(format!("modality id {bad} is not text(0) or image(1) — video is not supported"));
    }
    let n = types.len();
    let (mut t, mut h, mut w) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    let mut pos: i64 = 0;
    let mut g = 0usize;
    let mut i = 0usize;
    while i < n {
        let kind = types[i];
        let mut j = i;
        while j < n && types[j] == kind { j += 1; }
        let run = j - i;
        if kind == 0 {
            for k in 0..run {
                let p = pos + k as i64;
                t.push(p); h.push(p); w.push(p);
            }
            pos += run as i64;
        } else {
            let (gt, gh, gw) = *grids.get(g)
                .ok_or_else(|| format!("image run at {i} has no grid: {} supplied", grids.len()))?;
            g += 1;
            if gh % merge != 0 || gw % merge != 0 {
                return Err(format!("grid {gh}x{gw} is not a multiple of the merge size {merge}"));
            }
            let (lt, lh, lw) = (gt, gh / merge, gw / merge);
            // ⛔ This is also the guard against ADJACENT IMAGE RUNS. The published code groups
            // consecutive tokens by modality, so two images with nothing between them are ONE run and
            // consume ONE grid — it then emits half the positions it should. The chat template always
            // puts <vision_start>/<vision_end> text between images so it never happens there, but a
            // port that does not check returns a short array and the caller zips it against the wrong
            // tokens. Refused, with the arithmetic in the message.
            if lt * lh * lw != run {
                return Err(format!(
                    "image run of {run} tokens at {i} does not match grid ({gt},{gh},{gw}) = {} tokens \
                     at merge {merge} — two adjacent images with no text between them look like one run",
                    lt * lh * lw));
            }
            for ti in 0..lt {
                for hi in 0..lh {
                    for wi in 0..lw {
                        t.push(ti as i64 + pos);
                        h.push(hi as i64 + pos);
                        w.push(wi as i64 + pos);
                    }
                }
            }
            // ⛔ max of the PATCH dims then divided, exactly as published — not the token count.
            pos += (gh.max(gw) / merge) as i64;
        }
        i = j;
    }
    if g != grids.len() {
        return Err(format!("{} grids supplied, {g} consumed", grids.len()));
    }
    let hi = t.iter().chain(&h).chain(&w).copied().max().unwrap_or(0);
    Ok(RopeIndex { t, h, w, delta: hi + 1 - n as i64 })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Case { name: String, types: Vec<u8>, grids: Vec<(usize, usize, usize)>,
                  want: Option<(Vec<i64>, Vec<i64>, Vec<i64>, i64)> }

    /// Parse the fixture emitted by `examples/refgen/qwen3vl_rope_index.py`, which runs the
    /// PUBLISHED function bodies under a numpy shim rather than a reimplementation of them.
    fn cases() -> Vec<Case> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/qwen3vl_vision/rope_index.txt");
        let txt = std::fs::read_to_string(path).expect("rope_index.txt fixture");
        let nums = |s: &str| -> Vec<i64> { s.split(',').map(|v| v.trim().parse().unwrap()).collect() };
        let mut out = Vec::new();
        for block in txt.split("\ncase ") {
            let b = block.trim_start_matches("case ").trim();
            if b.is_empty() { continue; }
            let mut name = String::new();
            let mut types = Vec::new();
            let mut grids = Vec::new();
            let (mut t, mut h, mut w, mut d, mut err) = (vec![], vec![], vec![], 0i64, false);
            for line in b.lines() {
                let (k, v) = line.split_once(' ').unwrap_or((line, ""));
                match k {
                    "types" => types = v.bytes().map(|c| c - b'0').collect(),
                    "grids" => grids = v.split(';').filter(|s| !s.is_empty())
                        .map(|g| { let n = nums(g); (n[0] as usize, n[1] as usize, n[2] as usize) }).collect(),
                    "pos_t" => t = nums(v), "pos_h" => h = nums(v), "pos_w" => w = nums(v),
                    "delta" => d = v.trim().parse().unwrap(),
                    "ERROR" => err = true,
                    _ => if name.is_empty() { name = k.to_string() },
                }
            }
            out.push(Case { name, types, grids, want: if err { None } else { Some((t, h, w, d)) } });
        }
        out
    }

    #[test]
    fn rope_index_matches_the_published_function() {
        let cs = cases();
        assert!(cs.len() >= 9, "only {} cases parsed — fixture shrank", cs.len());
        let (mut ok, mut refused) = (0, 0);
        for c in &cs {
            let got = rope_index(&c.types, &c.grids, 2);
            match &c.want {
                Some((t, h, w, d)) => {
                    let g = got.unwrap_or_else(|e| panic!("{}: unexpected refusal: {e}", c.name));
                    assert_eq!(&g.t, t, "{} temporal", c.name);
                    assert_eq!(&g.h, h, "{} height", c.name);
                    assert_eq!(&g.w, w, "{} width", c.name);
                    assert_eq!(g.delta, *d, "{} delta", c.name);
                    assert_eq!(g.t.len(), c.types.len(), "{} length", c.name);
                    ok += 1;
                }
                None => { assert!(got.is_err(), "{}: should be refused, got {got:?}", c.name); refused += 1; }
            }
        }
        // ⛔ The refusal case is the one the published code gets WRONG (it returns a short array), so
        // a fixture that lost it would still pass everything else.
        assert!(ok >= 8 && refused >= 1, "{ok} matched, {refused} refused — coverage shifted");
    }

    /// ⛔ The advance after an image is `max(h, w) / merge`, NOT the token count. Stated as its own
    /// test because it is the single line a reimplementation gets wrong, and because a square image
    /// hides it: 4x4 -> 4 tokens and advance 2, but 4x8 -> 8 tokens and advance 4.
    #[test]
    fn an_image_advances_the_position_by_its_longest_side_not_its_token_count() {
        // one text token, then the image, then one text token
        let probe = |gh: usize, gw: usize| -> i64 {
            let ntok = (gh / 2) * (gw / 2);
            let mut types = vec![0u8]; types.extend(std::iter::repeat(1).take(ntok)); types.push(0);
            let r = rope_index(&types, &[(1, gh, gw)], 2).unwrap();
            *r.t.last().unwrap()   // the trailing text token's position
        };
        // 4x4: 4 tokens, advance max(4,4)/2 = 2 -> trailing text at 1 + 2 = 3
        assert_eq!(probe(4, 4), 3);
        // 4x8: 8 tokens, advance max(4,8)/2 = 4 -> 5, NOT 1 + 8 = 9
        assert_eq!(probe(4, 8), 5);
        // 8x4: also 8 tokens, also advance 4 — the count and the advance are independent
        assert_eq!(probe(8, 4), 5);
        // 12x2: 6 tokens, advance max(12,2)/2 = 6 -> 7, which EXCEEDS the token count
        assert_eq!(probe(12, 2), 7);
    }

    /// Image tokens are raster over the MERGED grid, and the three components differ from each other
    /// — a port that filled all three with the same ramp would pass a text-only fixture.
    #[test]
    fn image_tokens_carry_grid_coordinates_not_a_ramp() {
        let types: Vec<u8> = std::iter::repeat(1).take(6).collect();
        let r = rope_index(&types, &[(1, 4, 6)], 2).unwrap();   // merged 2 x 3
        assert_eq!(r.t, vec![0, 0, 0, 0, 0, 0], "one frame: temporal is constant");
        assert_eq!(r.h, vec![0, 0, 0, 1, 1, 1], "height is the slow axis");
        assert_eq!(r.w, vec![0, 1, 2, 0, 1, 2], "width is the fast axis");
        assert_ne!(r.h, r.w, "height and width must not be the same sequence");
    }

    #[test]
    fn mismatched_and_missing_grids_are_refused_with_the_arithmetic() {
        // 4 image tokens but a grid that yields 9
        let e = rope_index(&[0, 1, 1, 1, 1, 0], &[(1, 6, 6)], 2).unwrap_err();
        assert!(e.contains("does not match grid") && e.contains("9 tokens"), "unhelpful: {e}");
        // an image run with no grid at all
        assert!(rope_index(&[0, 1, 1, 1, 1], &[], 2).unwrap_err().contains("no grid"));
        // a grid nobody used
        assert!(rope_index(&[0, 0, 0], &[(1, 4, 4)], 2).unwrap_err().contains("consumed"));
        // video is refused, not folded into the image path
        assert!(rope_index(&[0, 2, 2], &[(1, 4, 4)], 2).unwrap_err().contains("video"));
        // a grid that is not a multiple of the merge size
        assert!(rope_index(&[1, 1, 1], &[(1, 3, 6)], 2).unwrap_err().contains("multiple of the merge"));
    }
}
