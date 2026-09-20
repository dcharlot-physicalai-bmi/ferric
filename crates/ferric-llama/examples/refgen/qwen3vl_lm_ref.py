#!/usr/bin/env python3
"""Emit the Qwen3-VL LM-SIDE seams: image-token replacement + deepstack injection.

⛔ PIXELS ARE SYNTHESISED, not preprocessed — same reason as the tower reference: preprocessing has
its own disagreements (see the a=-0.5 vs -0.75 bicubic finding) and folding it in would leave any
mismatch with two causes.

Seams, in pipeline order:
    P.embed    [seq, d_text]  inputs_embeds AFTER image tokens are replaced by merger output
    P.layerK   [seq, d_text]  hidden states after LM layer K, i.e. AFTER deepstack K is added
    P.final    [seq, d_text]  last_hidden_state (post-norm)
    P.meta     text           ids, grid, image positions, dims
"""
import struct, sys, torch, numpy as np
from transformers import AutoModel, AutoConfig

out = sys.argv[1]
mid = "Qwen/Qwen3-VL-Embedding-2B"
gh, gw = 4, 4

cfg = AutoConfig.from_pretrained(mid, local_files_only=True)
m = AutoModel.from_pretrained(mid, local_files_only=True, dtype=torch.float32).eval()
base = m.model if hasattr(m, "model") else m          # Qwen3VLModel
vc = base.config.vision_config
tc = base.config.text_config
IMG = base.config.image_token_id
VS, VE = base.config.vision_start_token_id, base.config.vision_end_token_id
merge = vc.spatial_merge_size
n_img = (gh // merge) * (gw // merge)

# ⛔⛔ THE <vision_start> MARKER IS LOAD-BEARING AND ITS ABSENCE IS SILENT.
# This generator first emitted `[text, text, IMG*4, text...]` with no marker. transformers 5.1.0
# locates images by scanning for vision_start_token_id and reading the token AFTER it, so with no
# marker it finds ZERO images, falls through to the text branch, and hands the rotary plain
# positions 0..9 on all three axes. The forward still runs, the image embeddings are still spliced
# in, deepstack is still applied — only mRoPE silently never happens. The resulting fixture looks
# like a multimodal reference and is a text-position one, and anything checked against it is being
# asked the wrong question. The chat template always emits these markers; so does this.
ids = [9707, 11, VS] + [IMG] * n_img + [VE, 1526, 264, 2168, 13]
input_ids = torch.tensor([ids], dtype=torch.long)
pos_img = [i for i, t in enumerate(ids) if t == IMG]

row = 3 * vc.temporal_patch_size * vc.patch_size ** 2
px = torch.tensor([[((i * 37 + j * 11) % 255) / 255.0 for j in range(row)] for i in range(gh * gw)],
                  dtype=torch.float32)
grid = torch.tensor([[1, gh, gw]], dtype=torch.long)

caught = {}
def hook(k):
    def f(mod, args, output):
        h = output[0] if isinstance(output, tuple) else output
        caught[k] = h.detach()[0].clone()
    return f
# NOTE: a decoder layer's hook fires BEFORE _deepstack_process, which the loop applies to the
# RETURNED tensor. So capture at the NEXT layer's input instead — that is the post-injection state.
handles = [base.language_model.layers[i].register_forward_pre_hook(
    lambda mod, args, k=i: caught.__setitem__(f"in{k}", args[0].detach()[0].clone())) for i in range(5)]

with torch.no_grad():
    o = base(input_ids=input_ids, pixel_values=px, image_grid_thw=grid)
for h in handles: h.remove()

def dump(path, t):
    t = t.flatten().contiguous().float()
    with open(path, "wb") as f:
        f.write(struct.pack("<I", t.numel())); f.write(t.numpy().astype("<f4").tobytes())
    print(f"  wrote {path} ({t.numel()} floats)")

dump(f"{out}.embed", caught["in0"])          # layer 0 input = embeddings after image replacement
for k in (1, 2, 3):
    dump(f"{out}.layer{k-1}", caught[f"in{k}"])   # input to layer k = output of layer k-1 + deepstack
dump(f"{out}.final", o.last_hidden_state[0])
dump(f"{out}.px", px)
# ⛔ A GUARD, because the failure above was invisible: if the three position axes are identical,
# mRoPE did not happen and this is a text-position capture wearing a multimodal costume.
_p, _d = base.get_rope_index(input_ids, image_grid_thw=grid)
assert not (torch.equal(_p[0], _p[1]) and torch.equal(_p[1], _p[2])), (
    "all three position axes are identical -> get_rope_index found NO image; the <vision_start> "
    "marker is missing and this fixture would not exercise mRoPE at all")
print("mrope positions t/h/w:", [_p[k, 0].tolist() for k in range(3)])
with open(f"{out}.meta", "w") as f:
    f.write(f"ids {','.join(map(str, ids))}\n")
    f.write(f"image_token_id {IMG}\n")
    f.write(f"grid {1},{gh},{gw}\n")
    f.write(f"image_positions {','.join(map(str, pos_img))}\n")
    f.write(f"types {''.join('1' if t == IMG else '0' for t in ids)}\n")
    f.write(f"vision_start_token_id {VS}\nvision_end_token_id {VE}\n")
    f.write(f"d_text {tc.hidden_size}\n")
    f.write(f"n_layers {tc.num_hidden_layers}\n")
    f.write(f"deepstack_visual_indexes {','.join(map(str, vc.deepstack_visual_indexes))}\n")
print("meta:", open(f"{out}.meta").read().strip().replace("\n", " | "))
print("embed[img0][:4]:", [round(float(x), 6) for x in caught["in0"][pos_img[0]][:4]])
print("final[-1][:4]  :", [round(float(x), 6) for x in o.last_hidden_state[0][-1][:4]])
