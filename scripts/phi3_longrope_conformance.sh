#!/usr/bin/env bash
# Phi-3 / Phi-3.5 LongRoPE: Ferric vs THE MODEL AUTHORS' OWN CODE (transformers `Phi3ForCausalLM`), float32 AND
# float64, on Phi-3.5-mini-instruct converted at F32 from their own weights:
#     convert_hf_to_gguf.py <microsoft/Phi-3.5-mini-instruct snapshot> --outtype f32
#
# LongRoPE is two per-dimension frequency tables — `short_factor` while the forward's last position + 1 is
# <= original_max_position_embeddings (4096), `long_factor` beyond — and a cos/sin scale (1.1902) applied at
# EVERY length. Before this gate Ferric applied none of it: on a 155-token prompt it missed the authors by
# 6.26 in the logits (89,397x their float32 floor), argmax 118/155.
#
# Three checks (scripts/lm_floor_conformance.sh does each; see it for the band and the controls):
#   1. 155 tokens — the SHORT table and the attention factor. Controls: the long table, no factor, pairing.
#   2. 4,650 tokens — the LONG table over the whole prompt (their forward switches the whole table, rows
#      before 4096 included). Controls: the short table past 4096, no factor, pairing; and device-derived
#      rope angles must leave the band (8.4x here: at position 4,600 an ulp of inv_freq is visible).
#   3. The CROSSING: the first 4,129 of those ids prefilled to 4096 and decoded one token at a time. At
#      position 4096 the table changes, and the rows from there on must equal the authors' full prefill —
#      the cache recomputed with the long table (their `generate`'s "enforce re-compute cache"), not
#      extended with short-table K/V. Control: the stale cache.
#
# ~9 loads of a 15 GB model: 20-40 minutes on a shared M-series machine.
#
#   scripts/phi3_longrope_conformance.sh <Phi-3.5-mini-instruct F32 .gguf> [fixture dir]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="${1:?usage: phi3_longrope_conformance.sh <Phi-3.5-mini-instruct F32 gguf> [fixture dir]}"
D="${2:-$ROOT/crates/ferric-llama/tests/fixtures/lm_floor}"
ok=0
echo "== 1. short prompt (short table) =="
"$ROOT/scripts/lm_floor_conformance.sh" "$M" "$D/phi-3.5-mini-short.json.gz" || ok=1
echo "== 2. long prompt (long table) =="
"$ROOT/scripts/lm_floor_conformance.sh" "$M" "$D/phi-3.5-mini-long.json.gz" || ok=1
echo "== 3. the crossing: prefill 4096, decode to 4128 =="
"$ROOT/scripts/lm_floor_conformance.sh" "$M" "$D/phi-3.5-mini-long.json.gz" --decode-from 4096 --upto 4129 || ok=1
[ $ok -eq 0 ] && echo "✅ Phi-3 LongRoPE matches the authors: short, long, and across the switch" \
             || echo "⛔ Phi-3 LongRoPE conformance FAILED"
exit $ok
