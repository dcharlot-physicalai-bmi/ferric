//! **How a `[t, n]` hidden-state buffer becomes one `n`-vector** — the rule the CHECKPOINT declares,
//! in ONE place.
//!
//! ⛔⛔ **WHY THIS IS ITS OWN MODULE.** This rule was written twice and only one copy was right.
//! `ferric-web` read `<arch>.pooling_type` and refused a type it could not honour; `ferric-serve`'s
//! `/v1/embeddings` hardcoded last-token pooling and never read the key at all. Same rule, two
//! crates, two answers — which is the exact shape of the `is_spm` defect found in this tree on
//! 2026-09-20 (ferric-web listed `llama|gemma4|t5`, ferric-serve listed `llama`, and one Gemma-4
//! file tokenized two ways by two front ends). A rule maintained in two places is a rule that is
//! wrong in one of them; the only durable fix is that there is nowhere for the second copy to live.
//!
//! ## What the defect actually costs
//!
//! Hardcoded LAST is correct for Qwen3-Embedding (which declares 3) and **silently wrong for the
//! MEAN-pooling family** — BGE, E5, GTE and most sentence-transformer exports — where pooling the
//! final position of a bidirectionally-trained encoder returns a vector with no trained meaning.
//! ⭐ The symptom is the dangerous kind: the vector has exactly the right length and roughly the
//! right magnitude, cosine similarities come back in `[-1, 1]` looking perfectly ordinary, and the
//! ranking is arbitrary. Nothing throws. Nothing looks wrong.
//!
//! ## The values, as llama.cpp writes them
//!
//! | `pooling_type` | meaning | implemented here |
//! |---|---|---|
//! | 0 | NONE — caller wants the raw `[t, n]` state | ⛔ refused; there is no single vector to return |
//! | 1 | MEAN — average down each channel | ✅ |
//! | 2 | CLS — the FIRST token | ✅ |
//! | 3 | LAST — the final token | ✅ |
//! | 4 | RANK — a classifier/reranker head, not a pooling | ⛔ refused; needs [`crate::bert::Reranker`] |
//!
//! ⚠ **0 and 4 refuse on purpose.** Falling back to LAST for either is precisely the defect above.

/// Pool a `[t, n]` row-major hidden-state buffer into one `n`-vector, per the checkpoint's
/// `<arch>.pooling_type`.
///
/// `v` is `t * n` floats, token-major: row `r` channel `c` is `v[r * n + c]`.
///
/// # Errors
/// - the buffer is shorter than `t * n`, or either dimension is zero — refused before it can slice
///   out of bounds;
/// - `kind` is not one of MEAN (1), CLS (2), LAST (3) — including NONE (0) and RANK (4), which are
///   real llama.cpp values that this function deliberately will not guess at.
pub fn pool(v: &[f32], t: usize, n: usize, kind: u32) -> Result<Vec<f32>, String> {
    if n == 0 || t == 0 || v.len() < t * n {
        return Err(format!("hidden state is {} floats, too short for {t}x{n}", v.len()));
    }
    match kind {
        1 => Ok((0..n).map(|c| (0..t).map(|r| v[r * n + c]).sum::<f32>() / t as f32).collect()),
        2 => Ok(v[0..n].to_vec()),
        3 => Ok(v[(t - 1) * n..t * n].to_vec()),
        other => Err(format!(
            "this checkpoint declares pooling_type {other}, which embed() does not implement. \
             Refusing rather than pooling the wrong position and returning cosine scores that look \
             ordinary and rank arbitrarily (0 = NONE, 4 = RANK need a reranker head).")),
    }
}

/// The checkpoint's declared pooling, read from GGUF by KEY SUFFIX rather than by guessing the arch.
///
/// ⚠ The key is architecture-prefixed (`bert.pooling_type`, `qwen3.pooling_type`, …), so matching on
/// the suffix is what lets one reader serve every family. Returns `None` when the file does not
/// declare one — the CALLER decides what that means, because the right default differs by front end
/// and a default chosen here would be invisible at the call site.
pub fn declared_pooling<'a>(md: impl IntoIterator<Item = (&'a String, &'a ferric_gguf::Meta)>) -> Option<u32> {
    md.into_iter()
        .find(|(k, _)| k.ends_with(".pooling_type"))
        .and_then(|(_, v)| match v {
            ferric_gguf::Meta::U(n) => Some(*n as u32),
            ferric_gguf::Meta::I(n) => Some(*n as u32),
            _ => None,
        })
}

#[cfg(test)]
mod pool_tests {
    use super::pool;

    #[test]
    fn mean_pooling_averages_down_COLUMNS_not_along_rows() {
        // Rows are tokens, columns are channels. The transposed version of this loop returns a vector
        // of the right length and the right magnitude and no meaning — which is why it is worth a
        // test with an asymmetric shape, where the two readings cannot coincide.
        let v = vec![1.0, 2.0, 3.0,
                     5.0, 6.0, 7.0];              // t = 2, n = 3
        assert_eq!(pool(&v, 2, 3, 1).unwrap(), vec![3.0, 4.0, 5.0]);
        assert_eq!(pool(&v, 2, 3, 2).unwrap(), vec![1.0, 2.0, 3.0], "CLS is the FIRST token");
        assert_eq!(pool(&v, 2, 3, 3).unwrap(), vec![5.0, 6.0, 7.0], "LAST is the final token");
    }

    #[test]
    fn a_pooling_type_we_cannot_honour_is_an_error_not_a_guess() {
        // The whole point. Falling back to LAST for a MEAN checkpoint is precisely the defect this
        // change exists to remove, so an unknown type must refuse rather than default.
        for bad in [0u32, 4, 9] {
            let e = pool(&[1.0, 2.0], 1, 2, bad).unwrap_err();
            assert!(e.contains(&bad.to_string()), "the error must name the type it refused: {e}");
        }
    }

    #[test]
    fn a_hidden_state_too_short_for_its_shape_is_refused_before_it_panics() {
        assert!(pool(&[1.0, 2.0], 4, 3, 3).is_err(), "would slice out of bounds");
        assert!(pool(&[], 1, 1, 3).is_err());
        assert!(pool(&[1.0], 0, 1, 3).is_err(), "zero tokens has no last token to pool");
    }

    /// ⛔ THE REGRESSION THAT MOTIVATED MOVING THIS HERE. A MEAN checkpoint pooled as LAST returns a
    /// vector that is the right length, in range, and wrong. This pins that the two rules are
    /// genuinely different on a shape where they cannot coincide — so a front end that silently
    /// defaults to LAST is detectably doing something else.
    #[test]
    fn mean_and_last_disagree_so_a_hardcoded_last_is_detectable() {
        let v = vec![0.0, 10.0,
                     2.0, 0.0,
                     4.0, 2.0];                   // t = 3, n = 2
        let mean = pool(&v, 3, 2, 1).unwrap();
        let last = pool(&v, 3, 2, 3).unwrap();
        assert_eq!(mean, vec![2.0, 4.0]);
        assert_eq!(last, vec![4.0, 2.0]);
        assert_ne!(mean, last, "if these ever coincide the fixture is too symmetric to prove anything");
    }
}
