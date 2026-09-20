#!/usr/bin/env python3
"""End-to-end reference: a REAL image file through the real processor and the real model.

Unlike the other fixtures, the pixels here are NOT synthesised — they come from
`tests/fixtures/qwen3vl_preproc/probe.ppm` via `Qwen2VLImageProcessorFast`, so this exercises the
preprocessing -> tower seam that nothing else does.

    python3 e2e_ref.py <out-prefix> <image.png>
"""
import struct, sys, torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

out, imgp = sys.argv[1], sys.argv[2]
ip = AutoImageProcessor.from_pretrained("Qwen/Qwen3-VL-Embedding-2B", local_files_only=True)
m = AutoModel.from_pretrained("Qwen/Qwen3-VL-Embedding-2B", local_files_only=True,
                              dtype=torch.float32).eval()
c = m.config
IMG, VS, VE = c.image_token_id, c.vision_start_token_id, c.vision_end_token_id

enc = ip(images=Image.open(imgp).convert("RGB"), return_tensors="pt")
pv, grid = enc["pixel_values"], enc["image_grid_thw"]
t, gh, gw = grid[0].tolist()
n_img = (gh // ip.merge_size) * (gw // ip.merge_size)
print(f"grid {t}x{gh}x{gw} -> {pv.shape[0]} patches, {n_img} image tokens")

ids = [9707, 11, VS] + [IMG] * n_img + [VE, 1526, 264, 2168, 13]
input_ids = torch.tensor([ids], dtype=torch.long)
with torch.no_grad():
    o = m(input_ids=input_ids, pixel_values=pv, image_grid_thw=grid)

# ⛔ guard: if the three position axes agree, get_rope_index found no image and mRoPE never fired
p, _ = m.get_rope_index(input_ids, image_grid_thw=grid)
assert not (torch.equal(p[0], p[1]) and torch.equal(p[1], p[2])), "mRoPE did not fire"
print("mrope t/h/w:", [p[k, 0].tolist() for k in range(3)])

def dump(path, x):
    x = x.flatten().contiguous().float()
    with open(path, "wb") as f:
        f.write(struct.pack("<I", x.numel())); f.write(x.numpy().astype("<f4").tobytes())
    print(f"  wrote {path} ({x.numel()} floats)")

dump(f"{out}.final", o.last_hidden_state[0])
dump(f"{out}.pixel_values", pv)
with open(f"{out}.meta", "w") as f:
    f.write(f"ids {','.join(map(str, ids))}\n")
    f.write(f"types {''.join('1' if x == IMG else '0' for x in ids)}\n")
    f.write(f"grid {t},{gh},{gw}\n")
    f.write(f"image_positions {','.join(str(i) for i, x in enumerate(ids) if x == IMG)}\n")
    f.write(f"vision_start_token_id {VS}\nvision_end_token_id {VE}\n")
print("meta:", open(f"{out}.meta").read().strip().replace("\n", " | "))
