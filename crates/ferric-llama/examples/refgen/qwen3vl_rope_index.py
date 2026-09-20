#!/usr/bin/env python3
"""Emit `get_rope_index` reference cases for `qwen3vl_rope.rs`.

⛔ THE TWO FUNCTIONS BELOW ARE THE PUBLISHED BODIES, PASTED UNCHANGED from
`transformers/models/qwen3_vl/modeling_qwen3_vl.py`. Only the ARRAY PRIMITIVES are shimmed to numpy
(torch is not installed on this machine) — the control flow, the index arithmetic and the order of
operations are byte-for-byte the originals. Re-deriving the algorithm here would produce an oracle
that agrees with a wrong port, which is the failure this whole fixture set exists to prevent.

    python3 qwen3vl_rope_index.py <out.txt>
"""
import itertools, sys
import numpy as np


class _T:  # the few torch surface bits these two bodies touch
    long = np.int64
    @staticmethod
    def arange(n, device=None): return np.arange(n, dtype=np.int64)
    @staticmethod
    def meshgrid(*a, indexing="ij"): return np.meshgrid(*a, indexing=indexing)
    @staticmethod
    def stack(a, dim=0): return np.stack(a, axis=dim)
    @staticmethod
    def cat(a, dim=0): return np.concatenate(a, axis=dim)
    @staticmethod
    def zeros(*shape, dtype=None, device=None): return np.zeros(shape, dtype=np.int64)
torch = _T


class _Cell(int):
    """`grid_thw[i]` is a 0-d tensor in the original, so `.item()` is called on it."""
    def item(self): return int(self)


def get_vision_position_ids(start_position, grid_thw, temp_merge_size=1, spatial_merge_size=1,
                            time_interval=1.0, device=None):
    # ---- published body, unchanged ----
    llm_grid_t, llm_grid_h, llm_grid_w = (
        grid_thw[0].item() // temp_merge_size,
        grid_thw[1].item() // spatial_merge_size,
        grid_thw[2].item() // spatial_merge_size,
    )
    position_temporal = (torch.arange(llm_grid_t, device=device) * time_interval).astype(np.int64)
    position_height = torch.arange(llm_grid_h, device=device) + start_position
    position_width = torch.arange(llm_grid_w, device=device) + start_position
    T_grid, H_grid, W_grid = torch.meshgrid(position_temporal, position_height, position_width, indexing="ij")
    vision_position_ids = torch.stack([T_grid, H_grid, W_grid], dim=0).reshape(3, -1)
    vision_position_ids[0] += start_position  # must be after time_interval multiply
    return vision_position_ids
    # ---- end published body ----


def get_rope_index(input_ids, mm_token_type_ids, image_grid_thw, spatial_merge_size):
    # ---- published body, unchanged except: video dropped (refused in Ferric), `self.config...`
    #      hoisted to a parameter, and `.to(device)` / `torch.tensor(...)` wrappers removed ----
    mrope_position_deltas = []
    position_ids = torch.zeros(3, input_ids.shape[0], input_ids.shape[1])
    grid_iters = {1: iter(image_grid_thw) if image_grid_thw is not None else None, 2: None}
    for batch_idx, current_input_ids in enumerate(input_ids):
        input_token_type = mm_token_type_ids[batch_idx]
        input_type_group = []
        for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
            group = list(group)
            start_index = group[0][0]
            end_index = group[-1][0] + 1
            input_type_group.append((key, start_index, end_index))
        current_pos = 0
        llm_pos_ids_list = []
        for modality_type, start_idx, end_idx in input_type_group:
            if modality_type == 0:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    np.broadcast_to(torch.arange(text_len).reshape(1, -1), (3, text_len)) + current_pos
                )
                current_pos += text_len
            else:
                grid_thw = next(grid_iters[modality_type])
                vision_position_ids = get_vision_position_ids(
                    current_pos, grid_thw, 1, spatial_merge_size
                )
                llm_pos_ids_list.append(vision_position_ids)
                current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[:, batch_idx] = llm_positions
        mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
    return position_ids, np.array(mrope_position_deltas).reshape(-1, 1)
    # ---- end published body ----


MERGE = 2
# (name, per-token modality ids, [(t,h,w) per image])  — 1 = image, 0 = text
CASES = [
    ("text_only",        [0] * 7,                                   []),
    ("one_image_mid",    [0, 0] + [1] * 4 + [0, 0, 0],              [(1, 4, 4)]),
    ("image_first",      [1] * 6 + [0] * 4,                         [(1, 4, 6)]),
    ("image_last",       [0] * 3 + [1] * 6,                         [(1, 6, 4)]),
    ("wide_image",       [0] + [1] * 8 + [0, 0],                    [(1, 4, 8)]),
    ("tall_image",       [0] + [1] * 8 + [0, 0],                    [(1, 8, 4)]),
    ("two_images",       [0] + [1] * 4 + [0, 0] + [1] * 9 + [0],    [(1, 4, 4), (1, 6, 6)]),
    ("adjacent_images",  [1] * 4 + [1] * 4,                         [(1, 4, 4), (1, 4, 4)]),  # ⚠ one run!
    ("single_token_img", [0, 0, 1, 0],                              [(1, 2, 2)]),
]

out = sys.argv[1]
with open(out, "w") as f:
    for name, types, grids in CASES:
        n = len(types)
        ids = np.arange(n, dtype=np.int64).reshape(1, n)
        tt = np.array(types, dtype=np.int64).reshape(1, n)
        gl = [[_Cell(t), _Cell(h), _Cell(w)] for (t, h, w) in grids] if grids else None
        try:
            pos, delta = get_rope_index(ids, tt, gl, MERGE)
        except (StopIteration, ValueError) as e:
            # ⛔ NOT a shim artefact. `itertools.groupby` merges CONSECUTIVE image tokens into ONE
            # run, so two images with no text between them consume one grid and emit half the
            # positions. The chat template always puts <vision_start>/<vision_end> text between
            # images, so the published code never meets this — but a port that does not REFUSE it
            # silently returns a short position array.
            f.write(f"case {name}\ntypes {''.join(map(str,types))}\n"
                    f"grids {';'.join(f'{t},{h},{w}' for t,h,w in grids)}\nERROR adjacent_image_runs\n\n")
            print(f"  {name:18} -> REFUSED ({type(e).__name__}): adjacent image runs collapse")
            continue
        p = pos[:, 0, :]
        f.write(f"case {name}\n")
        f.write(f"types {''.join(map(str, types))}\n")
        f.write(f"grids {';'.join(f'{t},{h},{w}' for t, h, w in grids)}\n")
        for k, lab in enumerate("thw"):
            f.write(f"pos_{lab} {','.join(map(str, p[k].tolist()))}\n")
        f.write(f"delta {int(delta[0][0])}\n\n")
        print(f"  {name:18} t={p[0].tolist()} h={p[1].tolist()} w={p[2].tolist()} delta={int(delta[0][0])}")
print(f"\nwrote {out}")
