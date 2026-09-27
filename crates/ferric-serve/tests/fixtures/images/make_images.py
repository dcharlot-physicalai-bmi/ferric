"""Fixture images + the pixels the MODEL AUTHORS' pipeline sees for each (transformers' `load_image`:
PIL.Image.open -> ImageOps.exif_transpose -> convert("RGB"), which every HF vision processor calls).

    python make_images.py <out-dir>      # writes the images and reference.json.gz

Synthetic content with edges, gradients and noise at odd sizes, so chroma subsampling, block edges and
palette/alpha handling are all exercised. Deterministic (fixed seed). Pillow 12.2.0 (libjpeg-turbo 3.1.4.1).

JPEGs come from two encoders:
- PIL's own (libjpeg-turbo): qualities 1..100, 4:4:4/4:2:2/4:2:0, progressive (jpeg_simple_progression, which
  uses successive approximation), restart intervals, optimized Huffman tables, RGB colour space (keep_rgb),
  16-bit quantisation tables (SOF1), EXIF orientations 2-8, XMP orientation, sizes 1x1 / 17x9 / 333x217, and a
  crop of the MiMo-Embodied demo photo re-encoded with that photo's own quantisation tables.
- `enc()` below, a small baseline encoder, for what PIL cannot write: 4:1:1, 4:4:0, 4:2:2-vertical, luma
  subsampled below chroma, 4x2 / 2x4 ratios, widths where the chroma plane is <= 2 samples (box upsampling),
  'R','G','B' component ids, Adobe APP14 without JFIF, no DHT (Motion-JPEG: the Annex K tables apply),
  non-interleaved scans, restart intervals in both, greyscale with 2x2 sampling factors, a DQT redefined
  between scans (libjpeg latches each component's table at its first scan), and one-block files at the
  edges of the IDCT range where libjpeg-turbo's C and SIMD paths agree.
Every one is decoded by PIL for the reference; the encoder only has to produce a valid stream.

Reference pixels: base64 RGB for images up to 20000 pixels, a CRC32 per row above that. The full demo photo
(1083x722, not committed) is recorded under "external" by CRC32 of its bytes and of each row PIL gives; the
Rust test checks it when the Hugging Face cache has it.
"""
import base64, glob, gzip, hashlib, io, json, math, os, struct, sys, zlib
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

def write(name, data):
    p = os.path.join(out, name)
    with open(p, "wb") as f: f.write(data)
    items[name] = p

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

# ── PIL-encoded variants ──
for q in (1, 50, 75, 95, 100):
    save(f"q{q}.jpg", base, quality=q)
save("q100_444.jpg", base, quality=100, subsampling=0)
save("q30_progressive.jpg", base, quality=30, progressive=True)
save("q100_progressive.jpg", base, quality=100, progressive=True)
save("optimized.jpg", base, quality=80, optimize=True)
save("restart_blocks.jpg", base, quality=85, restart_marker_blocks=5)
save("restart_rows_progressive.jpg", base, quality=85, progressive=True, restart_marker_rows=1)
save("rgb_colorspace.jpg", base, quality=90, keep_rgb=True)   # Adobe APP14 transform 0: no YCbCr
save("gray_progressive.jpg", base.convert("L"), quality=85, progressive=True)
# Custom tables with values > 255: libjpeg writes 16-bit DQT and SOF1 (extended sequential).
qbig = [[min(4 + 9 * i, 700) for i in range(64)], [min(8 + 11 * i, 900) for i in range(64)]]
save("sof1_dqt16.jpg", base, qtables=qbig)

# Sizes that end mid-block and mid-MCU, down to a single pixel.
def scene(h, w, seed):
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    a = 128 + 90 * np.sin(xx / 7.0) * np.cos(yy / 11.0)
    b = 128 + 100 * np.cos((xx + 2 * yy) / 13.0)
    c = 255 * ((xx - w / 2) ** 2 + (yy - h / 2) ** 2 < (min(h, w) / 3) ** 2)
    img = np.stack([a, b, 0.5 * c + 0.5 * (xx * 255 / max(w - 1, 1))], -1)
    img[(yy.astype(int) // 9 + xx.astype(int) // 13) % 5 == 0] = [240, 240, 30]  # hard edges
    return Image.fromarray(np.clip(img + r.integers(-4, 5, img.shape), 0, 255).astype(np.uint8), "RGB")
for (h, w) in ((1, 1), (9, 17), (217, 333)):
    im = scene(h, w, h * 1000 + w)
    save(f"s{w}x{h}_420.jpg", im, quality=85)
    save(f"s{w}x{h}_422.jpg", im, quality=85, subsampling=1)
    save(f"s{w}x{h}_444.jpg", im, quality=85, subsampling=0)
    save(f"s{w}x{h}_progressive.jpg", im, quality=85, progressive=True)

# Orientation: every EXIF value, PIL's type rules, and the XMP fallback of Image.getexif.
for o in (2, 4, 5, 7, 8):
    e = Image.Exif(); e[0x0112] = o
    save(f"exif{o}.jpg", base, quality=90, exif=e.tobytes())
def tiff(endian, typ, count, value):
    """A one-entry IFD0 holding tag 0x0112."""
    p = "<" if endian == b"II" else ">"
    return endian + struct.pack(p + "HI", 42, 8) + struct.pack(p + "H", 1) + struct.pack(p + "HHI", 0x0112, typ, count) + value + b"\0\0\0\0"
def with_exif(name, tiff_bytes):
    b = io.BytesIO(); base.save(b, "JPEG", quality=90, exif=b"Exif\0\0" + tiff_bytes); write(name, b.getvalue())
with_exif("exif_be_long8.jpg", tiff(b"MM", 4, 1, struct.pack(">I", 8)))           # LONG, big-endian: rotates
with_exif("exif_byte6.jpg", tiff(b"II", 1, 1, b"\x06\0\0\0"))                   # BYTE loads as bytes: does NOT rotate
with_exif("exif_short2x.jpg", tiff(b"II", 3, 2, struct.pack("<HH", 5, 1)))      # 2 values: PIL takes the first
rat = tiff(b"II", 5, 1, struct.pack("<I", 26))[:-4] + b"\0\0\0\0" + struct.pack("<II", 12, 2)  # RATIONAL 12/2 at offset 26
with_exif("exif_rational6.jpg", rat)
xmp6 = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description xmlns:tiff="http://ns.adobe.com/tiff/1.0/" tiff:Orientation="6"/></rdf:RDF></x:xmpmeta>'
save("xmp6.jpg", base, quality=90, xmp=xmp6)
e1 = Image.Exif(); e1[0x0112] = 1
save("exif1_xmp6.jpg", base, quality=90, exif=e1.tobytes(), xmp=xmp6)          # EXIF has the tag: XMP ignored

# A real photo: a crop of the MiMo-Embodied demo image, re-encoded with the photo's own tables at 4:2:0.
demo = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-Embodied-7B/snapshots/*/assets/demo.jpg")))
if demo:
    src = Image.open(demo[0])
    save("demo_crop.jpg", src.crop((301, 157, 301 + 245, 157 + 163)), qtables=src.quantization, subsampling=2)

# ── a small baseline encoder, for layouts PIL cannot write ──
ZIGZAG = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,
          7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39,
          46, 53, 60, 61, 54, 47, 55, 62, 63]
Q_LUM = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22,
         29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103,
         121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]
Q_CHR = [17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99,
         99, 99, 99, 99, 99] + [99] * 32
STD = {  # Annex K: (bits[1..16], values)
    ("dc", 0): ([0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], list(range(12))),
    ("dc", 1): ([0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], list(range(12))),
    ("ac", 0): ([0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d], bytes.fromhex(
        "01020300041105122131410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435"
        "363738393a434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a92939495969798"
        "999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4"
        "f5f6f7f8f9fa")),
    ("ac", 1): ([0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77], bytes.fromhex(
        "000102031104052131061241510761711322328108144291a1b1c109233352f0156272d10a162434e125f11718191a262728292a"
        "35363738393a434445464748494a535455565758595a636465666768696a737475767778797a82838485868788898a9293949596"
        "9798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4"
        "f5f6f7f8f9fa")),
}
def codes(bits, vals):
    c, code, out = 0, 0, {}
    for l, n in enumerate(bits, 1):
        for _ in range(n):
            out[vals[c]] = (code, l); code += 1; c += 1
        code <<= 1
    return out
assert all(sum(b) == len(v) for b, v in STD.values())
CODES = {k: codes(*v) for k, v in STD.items()}
DCT = np.array([[(math.sqrt(0.5) if u == 0 else 1.0) / 2 * math.cos((2 * x + 1) * u * math.pi / 16) for x in range(8)] for u in range(8)])

class Bits:
    def __init__(self): self.out, self.acc, self.n = bytearray(), 0, 0
    def put(self, v, l):
        self.acc, self.n = (self.acc << l) | (v & ((1 << l) - 1)), self.n + l
        while self.n >= 8:
            self.n -= 8; b = (self.acc >> self.n) & 0xFF
            self.out.append(b)
            if b == 0xFF: self.out.append(0)
    def flush(self):
        if self.n: self.put((1 << (8 - self.n)) - 1, 8 - self.n)

def seg(m, body): return bytes([0xFF, m]) + struct.pack(">H", len(body) + 2) + body

def enc(img_planes, factors, ids=(1, 2, 3), jfif=True, adobe=None, dht=True, restart=0, interleaved=True, quality_scale=1.0,
        redefine_q0=False):
    """img_planes: full-resolution uint8 planes (Y,Cb,Cr or R,G,B or one grey). Baseline, Annex K tables."""
    Hh, Ww = img_planes[0].shape
    hmax, vmax = max(f[0] for f in factors), max(f[1] for f in factors)
    mcux, mcuy = -(-Ww // (8 * hmax)), -(-Hh // (8 * vmax))
    qt = [[max(1, min(255, round(v * quality_scale))) for v in t] for t in (Q_LUM, Q_CHR)]
    comps = []
    for ci, (p, (h, v)) in enumerate(zip(img_planes, factors)):
        fx, fy = hmax // h, vmax // v
        assert fx * h == hmax and fy * v == vmax
        dw, dh = -(-Ww * h // hmax), -(-Hh * v // vmax)
        full = np.pad(p.astype(np.float64), ((0, dh * fy - Hh), (0, dw * fx - Ww)), mode="edge")
        ds = full.reshape(dh, fy, dw, fx).mean(axis=(1, 3))
        gh, gw = mcuy * v * 8, mcux * h * 8
        ds = np.pad(ds, ((0, gh - dh), (0, gw - dw)), mode="edge") - 128
        tq = 0 if ci == 0 else 1
        blocks = {}
        for by in range(gh // 8):
            for bx in range(gw // 8):
                f = DCT @ ds[by * 8:by * 8 + 8, bx * 8:bx * 8 + 8] @ DCT.T
                blocks[(bx, by)] = [int(round(f.flat[z] / qt[tq][z])) for z in ZIGZAG]
        comps.append(dict(h=h, v=v, tq=tq, blocks=blocks, bw=-(-dw // 8), bh=-(-dh // 8), tbl=min(ci, 1)))
    def block(bits, blk, pred, t):
        diff = blk[0] - pred
        s = abs(diff).bit_length()
        c, l = CODES[("dc", t)][s]; bits.put(c, l)
        if s: bits.put(diff if diff > 0 else diff - 1, s)
        run = 0
        for k in range(1, 64):
            a = blk[k]
            if a == 0: run += 1; continue
            while run > 15:
                c, l = CODES[("ac", t)][0xF0]; bits.put(c, l); run -= 16
            s = abs(a).bit_length()
            c, l = CODES[("ac", t)][(run << 4) | s]; bits.put(c, l); bits.put(a if a > 0 else a - 1, s)
            run = 0
        if run:
            c, l = CODES[("ac", t)][0]; bits.put(c, l)
        return blk[0]
    o = bytearray(b"\xff\xd8")
    if jfif: o += seg(0xE0, b"JFIF\0\x01\x01\0\0\x01\0\x01\0\0")
    if adobe is not None: o += seg(0xEE, b"Adobe" + struct.pack(">HHHB", 100, 0, 0, adobe))
    o += seg(0xDB, b"".join(bytes([i]) + bytes(qt[i][z] for z in ZIGZAG) for i in range(2)))
    o += seg(0xC0, struct.pack(">BHHB", 8, Hh, Ww, len(comps)) + b"".join(bytes([ids[i], c["h"] << 4 | c["v"], c["tq"]]) for i, c in enumerate(comps)))
    if restart: o += seg(0xDD, struct.pack(">H", restart))
    if dht:
        o += seg(0xC4, b"".join(bytes([(0x10 if k == "ac" else 0) | t]) + bytes(STD[(k, t)][0]) + bytes(STD[(k, t)][1]) for (k, t) in STD))
    scans = [list(range(len(comps)))] if interleaved else [[i] for i in range(len(comps))]
    for sc in scans:
        o += seg(0xDA, bytes([len(sc)]) + b"".join(bytes([ids[i], comps[i]["tbl"] << 4 | comps[i]["tbl"]]) for i in sc) + b"\x00\x3f\x00")
        bits, pred, n, rst = Bits(), [0] * len(comps), 0, 0
        if len(sc) == 1:
            c = comps[sc[0]]
            units = [[(sc[0], bx, by)] for by in range(c["bh"]) for bx in range(c["bw"])]
        else:
            units = [[(i, mx * comps[i]["h"] + bx, my * comps[i]["v"] + by) for i in sc for by in range(comps[i]["v"]) for bx in range(comps[i]["h"])]
                     for my in range(mcuy) for mx in range(mcux)]
        for u in units:
            if restart and n and n % restart == 0:
                bits.flush(); bits.out += bytes([0xFF, 0xD0 + rst]); rst = (rst + 1) & 7; pred = [0] * len(comps)
            for (i, bx, by) in u:
                pred[i] = block(bits, comps[i]["blocks"][(bx, by)], pred[i], comps[i]["tbl"])
            n += 1
        bits.flush(); o += bits.out
        if redefine_q0 and sc == scans[0]:  # a new table 0 AFTER luma's scan: luma keeps the table it latched
            o += seg(0xDB, bytes([0]) + bytes(min(255, 3 * qt[0][z]) for z in ZIGZAG))
    return bytes(o + b"\xff\xd9")

big = np.asarray(scene(58, 83, 7))
ycc = [np.asarray(Image.fromarray(big, "RGB").convert("YCbCr"))[:, :, i] for i in range(3)]
def e(name, planes=None, **kw): write(name, enc(planes or ycc, **kw))
e("enc_411.jpg", factors=[(4, 1), (1, 1), (1, 1)])                       # int_upsample 4x1 (box)
e("enc_440.jpg", factors=[(1, 2), (1, 1), (1, 1)])                       # h1v2_fancy_upsample
e("enc_422v.jpg", factors=[(2, 2), (2, 1), (2, 1)])                      # h1v2 at hmax 2
e("enc_c12.jpg", factors=[(2, 2), (1, 2), (1, 2)])                       # h2v1_fancy with vmax 2
e("enc_y11_c22.jpg", factors=[(1, 1), (2, 2), (2, 2)])                   # LUMA is the upsampled one
e("enc_42.jpg", factors=[(4, 2), (1, 1), (1, 1)])                        # int_upsample 4x2, 10 blocks/MCU
e("enc_24.jpg", factors=[(2, 4), (1, 1), (1, 1)])                        # int_upsample 2x4
e("enc_restart3.jpg", factors=[(2, 2), (1, 1), (1, 1)], restart=3)       # RST0..7 wrap several times
e("enc_noninterleaved.jpg", factors=[(2, 2), (1, 1), (1, 1)], interleaved=False)
e("enc_noninterleaved_restart.jpg", factors=[(2, 1), (1, 1), (1, 1)], interleaved=False, restart=4)
e("enc_dqt_between_scans.jpg", factors=[(2, 2), (1, 1), (1, 1)], interleaved=False, redefine_q0=True)
e("enc_no_dht.jpg", factors=[(2, 2), (1, 1), (1, 1)], dht=False)         # Motion-JPEG: Annex K tables
e("enc_coarse.jpg", factors=[(2, 2), (1, 1), (1, 1)], quality_scale=6.0)
rgbp = [big[:, :, i] for i in range(3)]
e("enc_rgb_ids.jpg", planes=rgbp, factors=[(1, 1)] * 3, ids=(82, 71, 66), jfif=False)   # 'R','G','B': no transform
e("enc_rgb_ids_422.jpg", planes=rgbp, factors=[(2, 1), (1, 1), (1, 1)], ids=(82, 71, 66), jfif=False)
e("enc_adobe_ycc.jpg", factors=[(2, 1), (1, 1), (1, 1)], jfif=False, adobe=1)
e("enc_adobe_unknown.jpg", factors=[(2, 2), (1, 1), (1, 1)], jfif=False, adobe=7)  # unknown transform: YCbCr
e("enc_gray_h2.jpg", planes=[ycc[0]], factors=[(2, 2)])                  # 1 component, 2x2 factors
# Saturated quadrants, so a 2-sample-wide chroma plane really differs between box and fancy upsampling.
quad = np.zeros((6, 4, 3), np.uint8)
quad[:3, :2], quad[:3, 2:], quad[3:, :2], quad[3:, 2:] = [255, 0, 0], [0, 0, 255], [0, 255, 0], [255, 255, 0]
qycc = [np.asarray(Image.fromarray(quad).convert("YCbCr"))[:, :, i] for i in range(3)]
e("enc_4x6_quads_420.jpg", planes=qycc, factors=[(2, 2), (1, 1), (1, 1)])   # chroma 2 wide: h2v2 BOX
e("enc_4x6_quads_422.jpg", planes=qycc, factors=[(2, 1), (1, 1), (1, 1)])   # h2v1 BOX
e("enc_3x6_quads_420.jpg", planes=[p[:, :3].copy() for p in qycc], factors=[(2, 2), (1, 1), (1, 1)])
save("s4x6_quads_420.jpg", Image.fromarray(quad), quality=95)             # PIL's own encoder, same case
save("s6x6_quads_420.jpg", Image.fromarray(np.pad(quad, ((0, 0), (0, 2), (0, 0)), mode="edge")), quality=95)  # width 3: fancy
for (h, w) in ((5, 3), (3, 4), (7, 5), (2, 2)):                          # chroma width <= 2: box upsampling
    sm = [p[:h, :w].copy() for p in ycc]
    e(f"enc_{w}x{h}_420.jpg", planes=sm, factors=[(2, 2), (1, 1), (1, 1)])
    e(f"enc_{w}x{h}_422.jpg", planes=sm, factors=[(2, 1), (1, 1), (1, 1)])

# One 8x8 grey block with a chosen DC coefficient and DC quantiser: the IDCT output is DC*q/8, placed at the
# edges of the range where libjpeg-turbo's C code and its SIMD kernels agree (the decoder refuses beyond:
# its tests craft those same blocks with 512, -513 and 16-bit overflows and expect a refusal).
def one_block(dc, q):
    b = Bits(); s = abs(dc).bit_length(); b.put(s, 4)
    if s: b.put(dc if dc > 0 else dc - 1, s)
    b.put(0, 1); b.flush()
    return (b"\xff\xd8" + seg(0xDB, bytes([0, q] + [1] * 63)) + seg(0xC0, struct.pack(">BHHB", 8, 8, 8, 1) + bytes([1, 0x11, 0]))
            + seg(0xC4, bytes([0x00, 0, 0, 0, 15] + [0] * 12 + list(range(15)) + [0x10, 1] + [0] * 15 + [0]))
            + seg(0xDA, bytes([1, 1, 0x00, 0, 63, 0])) + bytes(b.out) + b"\xff\xd9")
write("craft_x511.jpg", one_block(2044, 2))      # output 511: C's table and a saturating kernel both give 255
write("craft_x-512.jpg", one_block(-2048, 2))    # output -512: both give 0
write("craft_x500.jpg", one_block(2000, 2))

ref = {}
for name, p in items.items():
    a = np.asarray(load_image(p))
    assert a.dtype == np.uint8 and a.ndim == 3 and a.shape[2] == 3, (name, a.shape)
    ref[name] = {"h": a.shape[0], "w": a.shape[1], "file_sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
    if a.shape[0] * a.shape[1] <= 20000:
        ref[name]["rgb"] = base64.b64encode(a.tobytes()).decode()
    else:  # larger images: a CRC32 per row keeps the file small and still says WHICH rows differ
        ref[name]["rows_crc32"] = [zlib.crc32(row.tobytes()) for row in a]
external = {}
if demo:
    a = np.asarray(load_image(demo[0]))
    external["demo.jpg"] = {"hf_cache": "models--XiaomiMiMo--MiMo-Embodied-7B/snapshots/*/assets/demo.jpg",
                            "h": a.shape[0], "w": a.shape[1], "file_crc32": zlib.crc32(open(demo[0], "rb").read()),
                            "rows_crc32": [zlib.crc32(row.tobytes()) for row in a]}
with gzip.GzipFile(os.path.join(out, "reference.json.gz"), "wb", mtime=0) as f:
    f.write(json.dumps({"generator": "transformers.image_utils.load_image", "pillow": Image.__version__,
                        "images": ref, "external": external}).encode())
print(len(ref), "images;", {k: (v["h"], v["w"]) for k, v in ref.items()})
