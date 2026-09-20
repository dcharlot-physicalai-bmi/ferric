//! **Qwen3-VL's image preprocessing** — the part that decides WHICH pixels become WHICH token.
//!
//! Qwen3-VL ships no image processor of its own: its `preprocessor_config.json` names
//! `Qwen2VLImageProcessorFast`, so the rules below are Qwen2-VL's, read from that file and from the
//! published processor rather than from a description of either.
//!
//! ⛔ **The resampler is NOT implemented here and is not claimed.** See `Plan` — this module computes
//! the target size and turns an already-resized image into patch rows. Everything it does do is exact
//! integer or elementwise arithmetic and is tested against the published function's own output; the
//! bicubic downscale is the one step where two published references disagree with each other (PIL's
//! `BICUBIC` vs torchvision's antialiased `bicubic`), so folding it in silently would put an
//! unmeasured difference inside a module that otherwise matches exactly.
use ferric_tensor::image::Rgb8;

/// The preprocessing constants, read from a checkpoint's `preprocessor_config.json`.
#[derive(Debug, Clone)]
pub struct PreprocCfg {
    pub patch: usize,
    pub temporal_patch: usize,
    pub merge: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub rescale: f32,
}

impl PreprocCfg {
    pub fn load(dir: &str) -> Result<PreprocCfg, String> {
        let txt = std::fs::read_to_string(format!("{dir}/preprocessor_config.json"))
            .map_err(|e| format!("preprocessor_config.json: {e}"))?;
        let j: serde_json::Value = serde_json::from_str(&txt).map_err(|e| format!("preproc: {e}"))?;
        let u = |k: &str| j.get(k).and_then(|x| x.as_u64()).map(|x| x as usize)
            .ok_or_else(|| format!("preprocessor_config: missing {k}"));
        let arr3 = |k: &str| -> Result<[f32; 3], String> {
            let a = j.get(k).and_then(|x| x.as_array())
                .ok_or_else(|| format!("preprocessor_config: missing {k}"))?;
            if a.len() != 3 { return Err(format!("{k} has {} entries, expected 3", a.len())); }
            let mut o = [0f32; 3];
            for (i, v) in a.iter().enumerate() {
                o[i] = v.as_f64().ok_or_else(|| format!("{k}[{i}] is not a number"))? as f32;
            }
            Ok(o)
        };
        // ⚠ `min_pixels`/`max_pixels` are the legacy spelling and `size.shortest_edge`/`longest_edge`
        // the current one; the published __init__ lets the legacy pair OVERRIDE `size`. This reads the
        // same way round, and refuses if neither is present rather than inventing a default budget —
        // the budget decides the token count, so a wrong default is a wrong prompt length.
        let edge = |legacy: &str, modern: &str| -> Result<usize, String> {
            if let Ok(v) = u(legacy) { return Ok(v); }
            j.get("size").and_then(|s| s.get(modern)).and_then(|x| x.as_u64()).map(|x| x as usize)
                .ok_or_else(|| format!("preprocessor_config: neither {legacy} nor size.{modern}"))
        };
        Ok(PreprocCfg {
            patch: u("patch_size")?,
            temporal_patch: u("temporal_patch_size")?,
            merge: u("merge_size")?,
            min_pixels: edge("min_pixels", "shortest_edge")?,
            max_pixels: edge("max_pixels", "longest_edge")?,
            mean: arr3("image_mean")?,
            std: arr3("image_std")?,
            rescale: j.get("rescale_factor").and_then(|x| x.as_f64()).unwrap_or(1.0 / 255.0) as f32,
        })
    }

    /// `patch_size * merge_size` — the multiple both output dimensions must land on.
    pub fn factor(&self) -> usize { self.patch * self.merge }
}

/// What a source image of a given size turns into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    /// Resize the source to exactly this, with BICUBIC, before calling `patchify`.
    pub out_h: usize,
    pub out_w: usize,
    pub grid_h: usize,
    pub grid_w: usize,
}

impl Plan {
    /// Patch rows the tower consumes — `grid_h * grid_w`.
    pub fn patches(&self) -> usize { self.grid_h * self.grid_w }
    /// Tokens the LANGUAGE MODEL consumes, after the 2x2 merge.
    pub fn tokens(&self, merge: usize) -> usize { self.patches() / (merge * merge) }
}

/// **Half to EVEN**, which is what Python's `round()` does — and is not what `f64::round()` does.
///
/// ⛔ This single line changes the number of image tokens in the prompt. At factor 32 every input of
/// the form `32k+16` is exactly on a half, and the two rules disagree on alternate ones: an 80x80
/// source becomes 64x64 = **4** tokens under Python's rule and 96x96 = **9** under half-away-from-zero.
/// Over the 64 sizes in `tests/fixtures/qwen3vl_vision/smart_resize.csv`, **32 come out different**.
/// Nothing downstream can catch it — both are legal grids, the tower runs on either, and the model
/// answers about the image either way.
#[inline]
fn round_half_even(x: f64) -> f64 { x.round_ties_even() }

/// The published `smart_resize`: both sides a multiple of `factor`, total pixels inside
/// `[min_pixels, max_pixels]`, aspect ratio preserved as closely as that allows.
pub fn smart_resize(h: usize, w: usize, factor: usize, min_px: usize, max_px: usize)
    -> Result<(usize, usize), String>
{
    if h == 0 || w == 0 { return Err(format!("image is {h}x{w}")); }
    if factor == 0 { return Err("factor is 0".into()); }
    let (hi, lo) = (h.max(w) as f64, h.min(w) as f64);
    if hi / lo > 200.0 {
        return Err(format!("absolute aspect ratio must be smaller than 200, got {}", hi / lo));
    }
    let f = factor as f64;
    let mut hb = round_half_even(h as f64 / f) * f;
    let mut wb = round_half_even(w as f64 / f) * f;
    if hb * wb > max_px as f64 {
        let beta = ((h * w) as f64 / max_px as f64).sqrt();
        hb = f.max((h as f64 / beta / f).floor() * f);
        wb = f.max((w as f64 / beta / f).floor() * f);
    } else if hb * wb < min_px as f64 {
        // ⚠ No clamp on this branch and no re-check against max_px — matching the published code.
        // Ferric does not "improve" it: a different budget is a different token count.
        let beta = (min_px as f64 / (h * w) as f64).sqrt();
        hb = (h as f64 * beta / f).ceil() * f;
        wb = (w as f64 * beta / f).ceil() * f;
    }
    Ok((hb as usize, wb as usize))
}

/// Plan a source image of `h x w`.
pub fn plan(h: usize, w: usize, cfg: &PreprocCfg) -> Result<Plan, String> {
    let (out_h, out_w) = smart_resize(h, w, cfg.factor(), cfg.min_pixels, cfg.max_pixels)?;
    Ok(Plan { out_h, out_w, grid_h: out_h / cfg.patch, grid_w: out_w / cfg.patch })
}

/// `(row, col)` of the patch that token `i` carries, in spatial-merge-block order.
///
/// ⛔ Shared with the tower on purpose — see `qwen3vl_vision::sweep_rc`. The preprocessor's
/// `permute(0, 2, 5, 3, 6, 1, 4, 7)` and the model's `get_vision_position_ids` are two independent
/// statements of this same order, and the patch rows must be built in it because nothing downstream
/// re-checks: the tower does not reorder, and the merger's reshape is happy with any four
/// consecutive rows.
use crate::qwen3vl_vision::sweep_rc;

/// Turn an **already-resized** image into the patch rows the tower consumes.
///
/// `img` must be exactly `plan.out_h x plan.out_w`. Returns `[grid_h*grid_w, 3*tps*patch*patch]`
/// row-major, in block order, rescaled and normalised.
pub fn patchify(img: &Rgb8, plan: &Plan, cfg: &PreprocCfg) -> Result<Vec<f32>, String> {
    if img.h != plan.out_h || img.w != plan.out_w {
        return Err(format!("image is {}x{}, plan says {}x{} — resize first (BICUBIC)",
                           img.h, img.w, plan.out_h, plan.out_w));
    }
    let (p, m, tps) = (cfg.patch, cfg.merge, cfg.temporal_patch);
    let (gh, gw) = (plan.grid_h, plan.grid_w);
    let row = 3 * tps * p * p;
    let mut out = vec![0f32; gh * gw * row];
    for i in 0..gh * gw {
        let (pr, pc) = sweep_rc(m, gw, i);           // which patch of the grid this token is
        let (y0, x0) = (pr * p, pc * p);
        for c in 0..3 {
            for dy in 0..p {
                for dx in 0..p {
                    let px = img.px[((y0 + dy) * img.w + (x0 + dx)) * 3 + c] as f32;
                    let v = (px * cfg.rescale - cfg.mean[c]) / cfg.std[c];
                    // ⚠ The temporal axis is an `expand` of one frame in the published code, not two
                    // frames: a still image repeats. Video would put frame t here, and is refused by
                    // the tower rather than silently folded.
                    for t in 0..tps {
                        out[i * row + ((c * tps + t) * p + dy) * p + dx] = v;
                    }
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PreprocCfg {
        // This checkpoint's own preprocessor_config.json.
        PreprocCfg { patch: 16, temporal_patch: 2, merge: 2, min_pixels: 4096, max_pixels: 1_310_720,
                     mean: [0.5; 3], std: [0.5; 3], rescale: 1.0 / 255.0 }
    }

    /// Against the PUBLISHED function's own output, generated by
    /// `examples/refgen/qwen3vl_smart_resize.py`, which pastes it verbatim rather than re-deriving it.
    #[test]
    fn smart_resize_matches_the_published_table() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/qwen3vl_vision/smart_resize.csv");
        let txt = std::fs::read_to_string(path).expect("smart_resize.csv fixture");
        let c = cfg();
        let (mut ok, mut refused, mut halves) = (0, 0, 0);
        for line in txt.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').collect();
            assert_eq!(f.len(), 4, "malformed row {line:?}");
            let (h, w) = (f[0].parse::<usize>().unwrap(), f[1].parse::<usize>().unwrap());
            let got = smart_resize(h, w, c.factor(), c.min_pixels, c.max_pixels);
            if f[2] == "ERR" {
                assert!(got.is_err(), "{h}x{w} should be refused (aspect ratio), got {got:?}");
                refused += 1;
            } else {
                let want = (f[2].parse::<usize>().unwrap(), f[3].parse::<usize>().unwrap());
                assert_eq!(got.as_ref().unwrap(), &want, "{h}x{w}");
                ok += 1;
            }
            if h % c.factor() == c.factor() / 2 { halves += 1; }
        }
        // ⛔ A table that happened to contain no exact halves would pass on `f64::round()` too, which
        // is the whole thing this test exists to catch. Assert the subject is present.
        assert!(ok >= 60 && refused >= 4, "table shrank: {ok} sized, {refused} refused");
        assert!(halves >= 20, "only {halves} exact-half rows — the rounding rule is barely exercised");
    }

    /// The rule itself, stated as the difference it makes, so the constant cannot be "simplified".
    #[test]
    fn rounding_is_half_to_even_and_it_changes_the_token_count() {
        let c = cfg();
        let f = |h, w| smart_resize(h, w, c.factor(), c.min_pixels, c.max_pixels).unwrap();
        // 80/32 = 2.5 -> 2 (even), not 3.
        assert_eq!(f(80, 80), (64, 64));
        // 112/32 = 3.5 -> 4 (even), which agrees with away-from-zero — both must hold, or the
        // implementation is just `floor`.
        assert_eq!(f(112, 112), (128, 128));
        // and the consequence, which is what actually reaches the language model
        let p80 = plan(80, 80, &c).unwrap();
        assert_eq!(p80.tokens(c.merge), 4);
        assert_eq!(plan(112, 112, &c).unwrap().tokens(c.merge), 16);
    }

    #[test]
    fn the_pixel_budget_is_enforced_on_both_sides() {
        let c = cfg();
        // far too big -> clamped under max_pixels, still on the factor grid
        let (h, w) = smart_resize(4000, 3000, c.factor(), c.min_pixels, c.max_pixels).unwrap();
        assert!(h * w <= c.max_pixels, "{h}x{w} = {} > max {}", h * w, c.max_pixels);
        assert_eq!((h % c.factor(), w % c.factor()), (0, 0));
        // far too small -> lifted to at least min_pixels
        let (h, w) = smart_resize(8, 8, c.factor(), c.min_pixels, c.max_pixels).unwrap();
        assert!(h * w >= c.min_pixels, "{h}x{w} = {} < min {}", h * w, c.min_pixels);
        assert_eq!((h % c.factor(), w % c.factor()), (0, 0));
        // and the aspect-ratio refusal is a refusal, not a clamp
        assert!(smart_resize(1, 201, c.factor(), c.min_pixels, c.max_pixels).is_err());
        assert!(smart_resize(1, 200, c.factor(), c.min_pixels, c.max_pixels).is_ok());
    }

    /// Patch rows, checked against the OTHER derivation of the block sweep (nested loops) and with
    /// every pixel uniquely identifying its own (y, x, channel).
    #[test]
    fn patchify_puts_each_patch_where_the_permute_puts_it() {
        let c = cfg();
        let (gh, gw) = (4usize, 6usize);
        let (oh, ow) = (gh * c.patch, gw * c.patch);
        // px[y][x][ch] is a distinct byte for as many cells as a u8 allows; the test reads back the
        // exact source coordinate rather than comparing two computed layouts.
        let px: Vec<u8> = (0..oh * ow * 3)
            .map(|i| { let (y, x, ch) = (i / (ow * 3), (i / 3) % ow, i % 3); ((y * 7 + x * 13 + ch * 101) % 251) as u8 })
            .collect();
        let img = Rgb8 { w: ow, h: oh, px };
        let p = Plan { out_h: oh, out_w: ow, grid_h: gh, grid_w: gw };
        let rows = patchify(&img, &p, &c).unwrap();
        let row = 3 * c.temporal_patch * c.patch * c.patch;
        assert_eq!(rows.len(), gh * gw * row);

        // the OTHER derivation: four nested loops, as in permute(0,2,5,3,6,1,4,7)
        let mut expect = Vec::new();
        for by in (0..gh).step_by(c.merge) {
            for bx in (0..gw).step_by(c.merge) {
                for dy in 0..c.merge { for dx in 0..c.merge { expect.push((by + dy, bx + dx)); } }
            }
        }
        assert_eq!(expect.len(), gh * gw);

        for (i, &(pr, pc)) in expect.iter().enumerate() {
            for ch in 0..3 {
                for dy in 0..c.patch {
                    for dx in 0..c.patch {
                        let src = img.px[((pr * c.patch + dy) * ow + (pc * c.patch + dx)) * 3 + ch] as f32;
                        let want = (src * c.rescale - c.mean[ch]) / c.std[ch];
                        for t in 0..c.temporal_patch {
                            let got = rows[i * row + ((ch * c.temporal_patch + t) * c.patch + dy) * c.patch + dx];
                            assert!((got - want).abs() < 1e-6,
                                    "token {i} patch ({pr},{pc}) ch{ch} t{t} @({dy},{dx}): {got} vs {want}");
                        }
                    }
                }
            }
        }
        // ⚠ and the sweep is NOT raster order on this grid, or everything above would pass on an
        // implementation that never reorders at all.
        let raster: Vec<(usize, usize)> = (0..gh * gw).map(|i| (i / gw, i % gw)).collect();
        assert_ne!(expect, raster, "the {gh}x{gw} sweep must differ from raster order");
        assert_eq!(expect[2], (1, 0), "token 2 must be the row below token 0, not two columns along");
    }

    /// The parser, against this checkpoint's actual `preprocessor_config.json` contents (copied in,
    /// so the test runs everywhere rather than skipping when the 4 GB checkpoint is absent — a guard
    /// that can never fire is not coverage).
    #[test]
    fn preproc_cfg_reads_the_published_json() {
        let dir = std::env::temp_dir().join("ferric_qwen3vl_preproc_test");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("preprocessor_config.json"), r#"{
          "image_processor_type": "Qwen2VLImageProcessorFast",
          "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
          "max_pixels": 1310720, "min_pixels": 4096,
          "merge_size": 2, "patch_size": 16, "temporal_patch_size": 2,
          "rescale_factor": 0.00392156862745098, "resample": 3,
          "size": {"longest_edge": 1310720, "shortest_edge": 4096}
        }"#).unwrap();
        let c = PreprocCfg::load(dir.to_str().unwrap()).expect("load");
        assert_eq!((c.patch, c.temporal_patch, c.merge), (16, 2, 2));
        assert_eq!((c.min_pixels, c.max_pixels), (4096, 1_310_720));
        assert_eq!(c.factor(), 32, "factor is patch*merge, and it sets the whole grid");
        assert!((c.rescale - 1.0 / 255.0).abs() < 1e-9);
        assert_eq!((c.mean, c.std), ([0.5; 3], [0.5; 3]));

        // ⚠ the modern spelling alone must also work — `size.*_edge` with no legacy keys
        std::fs::write(dir.join("preprocessor_config.json"), r#"{
          "merge_size": 2, "patch_size": 16, "temporal_patch_size": 2,
          "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
          "size": {"longest_edge": 999424, "shortest_edge": 1024}
        }"#).unwrap();
        let c = PreprocCfg::load(dir.to_str().unwrap()).expect("modern spelling");
        assert_eq!((c.min_pixels, c.max_pixels), (1024, 999_424));

        // ⛔ and neither spelling present is a REFUSAL, not a default budget: the budget decides how
        // many image tokens the prompt gets, so guessing it produces a confidently wrong sequence.
        std::fs::write(dir.join("preprocessor_config.json"), r#"{
          "merge_size": 2, "patch_size": 16, "temporal_patch_size": 2,
          "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]
        }"#).unwrap();
        let e = PreprocCfg::load(dir.to_str().unwrap()).unwrap_err();
        assert!(e.contains("min_pixels") && e.contains("shortest_edge"), "unhelpful error: {e}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn patchify_refuses_an_image_that_is_not_the_planned_size() {
        let c = cfg();
        let p = Plan { out_h: 64, out_w: 64, grid_h: 4, grid_w: 4 };
        let img = Rgb8 { w: 63, h: 64, px: vec![0; 63 * 64 * 3] };
        assert!(patchify(&img, &p, &c).unwrap_err().contains("resize first"));
    }
}
