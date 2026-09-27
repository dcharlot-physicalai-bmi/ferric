"""Fixture images + the pixels the MODEL AUTHORS' pipeline sees for each (transformers' `load_image`:
PIL.Image.open -> ImageOps.exif_transpose -> convert("RGB"), which every HF vision processor calls).

    python make_images.py <out-dir>      # writes the images and reference.json.gz

Synthetic content with edges, gradients and noise at odd sizes, so chroma subsampling, block edges and
palette/alpha handling are all exercised. Deterministic (fixed seed)."""
import base64, gzip, hashlib, io, json, os, sys
import numpy as np
from PIL import Image
from transformers.image_utils import load_image

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(20260927)
H, W = 45, 67
y, x = np.mgrid[0:H, 0:W]
rgb = np.stack([(x * 255 // (W - 1)), (y * 255 // (H - 1)), ((x + y) * 7 % 256)], -1).astype(np.int32)
rgb[10:30, 20:40] = [250, 20, 30]                       # a hard-edged block
rgb = np.clip(rgb + rng.integers(-12, 13, rgb.shape), 0, 255).astype(np.uint8)
alpha = ((x * 3 + y * 5) % 256).astype(np.uint8)

base = Image.fromarray(rgb, "RGB")
items = {}
def save(name, img, **kw):
    p = os.path.join(out, name); img.save(p, **kw); items[name] = p

save("rgb.png", base)
save("rgba.png", Image.fromarray(np.dstack([rgb, alpha]), "RGBA"))
save("palette.png", base.quantize(colors=50))
pal_t = base.quantize(colors=16); pal_t.info["transparency"] = 3
save("palette_trns.png", pal_t, transparency=3)
save("gray.png", base.convert("L"))
save("gray_alpha.png", Image.fromarray(np.dstack([np.asarray(base.convert("L")), alpha]), "LA"))
save("baseline420.jpg", base, quality=90)
save("q444.jpg", base, quality=95, subsampling=0)
save("q422.jpg", base, quality=85, subsampling=1)
save("progressive.jpg", base, quality=88, progressive=True)
save("gray.jpg", base.convert("L"), quality=90)
exif = Image.Exif(); exif[0x0112] = 6                   # orientation 6: rotate 90 degrees clockwise to display
save("exif6.jpg", base, quality=90, exif=exif.tobytes())
exif3 = Image.Exif(); exif3[0x0112] = 3
save("exif3.jpg", base, quality=90, exif=exif3.tobytes())

ref = {}
for name, p in items.items():
    a = np.asarray(load_image(p))
    assert a.dtype == np.uint8 and a.ndim == 3 and a.shape[2] == 3, (name, a.shape)
    ref[name] = {"h": a.shape[0], "w": a.shape[1], "rgb": base64.b64encode(a.tobytes()).decode(),
                 "file_sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
with gzip.open(os.path.join(out, "reference.json.gz"), "wt") as f:
    json.dump({"generator": "transformers.image_utils.load_image", "pillow": Image.__version__, "images": ref}, f)
print(len(ref), "images;", {k: (v["h"], v["w"]) for k, v in ref.items()})
