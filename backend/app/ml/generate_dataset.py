"""
Synthetic dataset generation.

Per assessment section 8 ("generate controlled image-quality degradations
from clean images"), this script:
  1. Procedurally generates a diverse set of "clean" base images (no
     external dataset dependency, no network calls -> reproducible offline).
  2. Applies controlled degradations (blur, under/over-exposure, noise,
     corruption/compression artifacts) at randomized severities.
  3. Writes out (image_bytes, multi-label targets, severity) pairs used by
     train.py.

Base image diversity: gradients, geometric shapes, checkerboards, text,
random textures, and photographic-like Perlin-ish noise fields, at varied
resolutions and color palettes, so the model learns quality cues rather
than memorizing content.
"""
import io
import os
import random
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

random.seed(42)
np.random.seed(42)

ISSUE_TYPES = ["blur", "underexposure", "overexposure", "noise", "corruption"]


def _random_color():
    return tuple(np.random.randint(0, 256, size=3).tolist())


def _gen_gradient(size):
    w, h = size
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    c1, c2 = np.random.randint(0, 256, 3), np.random.randint(0, 256, 3)
    horizontal = random.random() < 0.5
    for i in range(w if horizontal else h):
        t = i / max(1, (w if horizontal else h) - 1)
        color = (c1 * (1 - t) + c2 * t).astype(np.uint8)
        if horizontal:
            arr[:, i, :] = color
        else:
            arr[i, :, :] = color
    return Image.fromarray(arr)


def _gen_shapes(size):
    img = Image.new("RGB", size, _random_color())
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(5, 25)):
        shape = random.choice(["rectangle", "ellipse", "line"])
        x0, y0 = random.randint(0, size[0]), random.randint(0, size[1])
        x1, y1 = random.randint(0, size[0]), random.randint(0, size[1])
        color = _random_color()
        xs, ys = sorted([x0, x1]), sorted([y0, y1])
        box = [xs[0], ys[0], xs[1], ys[1]]
        if xs[0] == xs[1]:
            box[2] += 1
        if ys[0] == ys[1]:
            box[3] += 1
        if shape == "rectangle":
            draw.rectangle(box, fill=color)
        elif shape == "ellipse":
            draw.ellipse(box, fill=color)
        else:
            draw.line([x0, y0, x1, y1], fill=color, width=random.randint(1, 6))
    return img


def _gen_checkerboard(size, cell=None):
    w, h = size
    cell = cell or random.choice([8, 16, 32, 64])
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    c1, c2 = _random_color(), _random_color()
    for y in range(h):
        for x0 in range(0, w, cell):
            pass
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    mask = ((xx // cell) + (yy // cell)) % 2 == 0
    arr[mask] = c1
    arr[~mask] = c2
    return Image.fromarray(arr)


def _gen_texture(size):
    w, h = size
    base = np.random.randint(0, 256, (h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    img = Image.fromarray(base, "RGB").resize((w, h), Image.BICUBIC)
    return img


def _gen_text(size):
    img = Image.new("RGB", size, _random_color())
    draw = ImageDraw.Draw(img)
    words = ["QUALITY", "TEST", "SAMPLE", "IMAGE", "DATA", "VISION", "MODEL"]
    for _ in range(random.randint(3, 10)):
        text = random.choice(words)
        pos = (random.randint(0, size[0]), random.randint(0, size[1]))
        draw.text(pos, text, fill=_random_color())
    return img


GENERATORS = [_gen_gradient, _gen_shapes, _gen_checkerboard, _gen_texture, _gen_text]


def generate_clean_image(size=None):
    size = size or random.choice([(256, 256), (320, 240), (400, 300), (300, 400), (512, 384)])
    gen = random.choice(GENERATORS)
    img = gen(size)
    # Slight smoothing so base images are not pathologically noisy to begin with
    if random.random() < 0.5:
        img = img.filter(ImageFilter.SMOOTH)
    return img


# ---------------- Degradations ----------------

def apply_blur(img: Image.Image, severity: float):
    radius = 1 + severity * 8  # 1..9
    return img.filter(ImageFilter.GaussianBlur(radius))


def apply_underexposure(img: Image.Image, severity: float):
    arr = np.asarray(img).astype(np.float64)
    factor = 1.0 - severity * 0.85  # dim towards near-black
    arr = arr * factor
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_overexposure(img: Image.Image, severity: float):
    arr = np.asarray(img).astype(np.float64)
    factor = 1.0 + severity * 2.5
    arr = arr * factor + severity * 60
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_noise(img: Image.Image, severity: float):
    arr = np.asarray(img).astype(np.float64)
    sigma = severity * 60
    noise = np.random.normal(0, sigma, arr.shape)
    arr = arr + noise
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_corruption(img: Image.Image, severity: float):
    arr = np.asarray(img).copy()
    h, w = arr.shape[:2]
    # random block corruption + heavy jpeg re-encoding to simulate corruption/artifacts
    n_blocks = int(severity * 15) + 1
    for _ in range(n_blocks):
        bh, bw = random.randint(5, max(6, h // 6)), random.randint(5, max(6, w // 6))
        y0 = random.randint(0, max(0, h - bh))
        x0 = random.randint(0, max(0, w - bw))
        if random.random() < 0.5:
            arr[y0:y0 + bh, x0:x0 + bw] = np.random.randint(0, 256, (bh, bw, 3))
        else:
            arr[y0:y0 + bh, x0:x0 + bw] = 0
    out = Image.fromarray(arr)
    quality = max(2, int(30 - severity * 28))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


DEGRADERS = {
    "blur": apply_blur,
    "underexposure": apply_underexposure,
    "overexposure": apply_overexposure,
    "noise": apply_noise,
    "corruption": apply_corruption,
}


def make_sample():
    """Return (image_bytes, labels: dict[issue]->severity in [0,1] or 0 if absent)."""
    img = generate_clean_image()
    labels = {k: 0.0 for k in ISSUE_TYPES}

    r = random.random()
    if r < 0.22:
        # clean / acceptable sample, no degradation
        pass
    else:
        n_issues = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        issues = random.sample(ISSUE_TYPES, k=n_issues)
        for issue in issues:
            severity = float(np.clip(np.random.beta(2, 2), 0.05, 1.0))
            img = DEGRADERS[issue](img, severity)
            labels[issue] = severity

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue(), labels


def generate_dataset(n_samples: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for i in range(n_samples):
        img_bytes, labels = make_sample()
        fname = f"sample_{i:05d}.png"
        with open(os.path.join(out_dir, fname), "wb") as f:
            f.write(img_bytes)
        manifest.append({"file": fname, "labels": labels})
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "data", "synthetic")
    manifest = generate_dataset(1400, out)
    print(f"Generated {len(manifest)} samples in {out}")