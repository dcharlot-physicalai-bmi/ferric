#!/usr/bin/env python3
"""Emit the Qwen3-VL vision-tower reference seams for `examples/qwen3vl_vision_ref.rs`.

⛔ THIS LIVES IN THE REPO ON PURPOSE. An earlier run kept it in a scratch directory; the directory was
wiped between sessions and took both the generator and the reference data with it, leaving a check that
could not be re-run. A verification artifact that only exists in scratch is a verification you cannot
repeat.

⛔ PIXELS ARE SYNTHESISED HERE, not produced by the image preprocessor. Preprocessing has its own rules
(smart-resize to patch multiples, normalisation constants) and the two reference implementations are
known to disagree on the pixel budget, so folding it in would leave any mismatch with two possible
causes. The tower is tested alone.

Writes, for prefix P:
    P            pooler_output    [n/merge^2, out_hidden]  merged + projected — what the LM consumes
    P.deepK      deepstack k      [n/merge^2, out_hidden]  from the deepstack block indices
    P.hidden     last_hidden      [n, hidden]              per-patch rows BEFORE the merger
    P.px         pixel rows       [n, 3*tps*patch^2]       the input, so the Rust side feeds the same
⚠ `pooler_output` is the tower's real output. `last_hidden_state` is the field a reader's hand reaches
for and it is the WRONG one here: pre-merge rows at vision width, not merged rows at text width.

    python3 qwen3vl_vision_ref.py <out-prefix> [model-id] [grid_h] [grid_w]
"""
import struct, sys, torch
from transformers import AutoModel

out = sys.argv[1]
mid = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-VL-Embedding-2B"
gh = int(sys.argv[3]) if len(sys.argv) > 3 else 4
gw = int(sys.argv[4]) if len(sys.argv) > 4 else 4

m = AutoModel.from_pretrained(mid, local_files_only=True, dtype=torch.float32).eval()
vis = m.visual if hasattr(m, "visual") else m.model.visual
c = vis.config
print(f"patch {c.patch_size} temporal {c.temporal_patch_size} merge {c.spatial_merge_size} "
      f"hidden {c.hidden_size} depth {c.depth} heads {c.num_heads} ff {c.intermediate_size}")
print("deepstack layers", c.deepstack_visual_indexes)
if gh % c.spatial_merge_size or gw % c.spatial_merge_size:
    raise SystemExit(f"grid {gh}x{gw} must be a multiple of the merge size {c.spatial_merge_size}")

row = 3 * c.temporal_patch_size * c.patch_size ** 2
px = torch.tensor([[((i * 37 + j * 11) % 255) / 255.0 for j in range(row)] for i in range(gh * gw)],
                  dtype=torch.float32)
with torch.no_grad():
    o = vis(px, torch.tensor([[1, gh, gw]]))

def dump(path, t):
    t = t.flatten().contiguous()
    with open(path, "wb") as f:
        f.write(struct.pack("<I", t.numel()))
        f.write(t.numpy().astype("<f4").tobytes())
    print(f"  wrote {path} ({t.numel()} floats)")

dump(out, o.pooler_output)
for k, d in enumerate(o.deepstack_features): dump(f"{out}.deep{k}", d)
dump(f"{out}.hidden", o.last_hidden_state)
dump(f"{out}.px", px)
print("pooled first:", [round(float(x), 6) for x in o.pooler_output.flatten()[:6]])
