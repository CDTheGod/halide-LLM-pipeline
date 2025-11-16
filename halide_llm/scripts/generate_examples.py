#!/usr/bin/env python3
"""
generate_examples.py

Generates 5 procedurally-created 32x32 RGB input images (bird, mountain, abstract,
fruit bowl, portrait-blob), computes expected outputs for each example task
using pure NumPy reference implementations (option 2), saves PNGs, and writes
an augmented examples JSON file.

Usage:
    python scripts/generate_examples.py
"""

import os
import json
from typing import List, Dict, Tuple
import numpy as np
import imageio


# -------------------------
# Utilities
# -------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES_DIR = os.path.join(ROOT, "examples")
IMAGES_DIR = os.path.join(EXAMPLES_DIR, "images")
SRC_EXAMPLES_JSON = os.path.join(EXAMPLES_DIR, "halide_examples.json")
OUT_EXAMPLES_JSON = os.path.join(EXAMPLES_DIR, "halide_examples_augmented.json")

os.makedirs(IMAGES_DIR, exist_ok=True)


def save_png(arr: np.ndarray, path: str) -> None:
    """Save float image in [0,1] as PNG uint8."""
    arr = np.clip(arr, 0.0, 1.0)
    imageio.imwrite(path, (arr * 255.0).astype(np.uint8))


def clamp01(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0)


def as_list(a: np.ndarray) -> List:
    return a.astype(np.float32).tolist()


# -------------------------
# Simple image helpers
# -------------------------
def make_grid(h=32, w=32):
    y = np.linspace(0, 1, h, dtype=np.float32)
    x = np.linspace(0, 1, w, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return yy, xx


def radial_gradient(h=32, w=32, cx=0.5, cy=0.4, radius=0.6):
    yy, xx = make_grid(h, w)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
    return np.clip(1.0 - d, 0.0, 1.0)


def box_blur3(img: np.ndarray) -> np.ndarray:
    """3x3 box blur, channel-last, edge clamped."""
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    pad = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out += pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w, :]
    out /= 9.0
    return out


def draw_circle_mask(h, w, cx, cy, r):
    yy, xx = make_grid(h, w)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return (d <= r).astype(np.float32)


def smooth_noise(h=32, w=32, scale=4.0, seed=None):
    rng = np.random.RandomState(seed)
    base = rng.randn(h, w).astype(np.float32)
    # smooth by repeated small box blur (cheap)
    out = base.copy()
    for _ in range(4):
        out = box_blur3(np.stack([out] * 3, axis=-1))[:, :, 0]
    # normalize to [0,1]
    out = (out - out.min()) / (out.max() - out.min() + 1e-9)
    return out


# -------------------------
# Procedural 32x32 images
# -------------------------
def gen_bird_sky(h=32, w=32) -> np.ndarray:
    """Pseudo real bird on sky:
       - radial sky gradient
       - sun glow
       - soft cloud wisps
       - bird silhouette (ellipse + wing)
    """
    yy, xx = make_grid(h, w)
    # sky: light blue to slightly darker
    sky = np.zeros((h, w, 3), dtype=np.float32)
    grad = radial_gradient(h, w, cx=0.5, cy=0.4, radius=1.2)
    sky[:, :, 0] = 0.55 * (0.8 + 0.2 * (1 - grad))  # R small
    sky[:, :, 1] = 0.78 * (0.9 + 0.08 * (1 - grad))  # G small
    sky[:, :, 2] = 1.0 * (0.95 - 0.05 * (1 - grad))  # B
    # sun glow
    sun = radial_gradient(h, w, cx=0.8, cy=0.2, radius=0.25)
    sky = sky + np.expand_dims(sun, -1) * np.array([1.0, 0.95, 0.6])[None, None, :]
    # clouds: soft noise added
    clouds = smooth_noise(h, w, seed=12)
    sky = sky + (np.expand_dims(clouds, -1) - 0.5) * 0.08
    sky = clamp01(sky)
    # bird silhouette: ellipse + wing
    mask = np.zeros((h, w), dtype=np.float32)
    # body ellipse
    bx, by = 0.38, 0.45
    body = ((xx - bx) ** 2) / (0.06 ** 2) + ((yy - by) ** 2) / (0.04 ** 2)
    mask += (body <= 1.0).astype(np.float32)
    # wing triangle/shape (simple polygon approx via distance to line)
    wing = ((xx - 0.45) ** 2) + ((yy - 0.4) ** 2) * 4
    mask += (wing <= 0.02).astype(np.float32)
    mask = np.clip(mask, 0.0, 1.0)
    # bird color: dark silhouette
    bird_col = np.array([0.05, 0.05, 0.06])[None, None, :]
    out = sky * (1 - np.expand_dims(mask, -1)) + bird_col * np.expand_dims(mask, -1)
    return clamp01(out)


def gen_mountain(h=32, w=32) -> np.ndarray:
    """Mountain + sun landscape. Layered silhouettes + sun + grassy foreground."""
    yy, xx = make_grid(h, w)
    img = np.zeros((h, w, 3), dtype=np.float32)
    # sky gradient
    img[:, :, 2] = 0.9 - 0.3 * yy  # blue channel decreases downward
    img[:, :, 0] = 0.6 - 0.2 * yy
    img[:, :, 1] = 0.75 - 0.25 * yy
    # sun
    sun = radial_gradient(h, w, cx=0.7, cy=0.2, radius=0.13)
    img += np.expand_dims(sun, -1) * np.array([1.0, 0.95, 0.5])[None, None, :]
    # mountains: layered - create depth by thresholding sinusoidal ridges
    for i, depth in enumerate([0.25, 0.40, 0.55]):
        ridge = 0.5 + 0.12 * np.sin(10 * xx + (i * 0.7)) * (1 - yy * (i + 0.3))
        mask = (yy > ridge).astype(np.float32)
        shade = 0.12 + 0.12 * i
        img = img * (1 - mask[..., None]) + (np.array([0.08 + shade, 0.06 + shade, 0.05 + shade])[None, None, :]) * mask[..., None] + img * 0.0
    # grassy foreground: low-frequency green noise
    grass = smooth_noise(h, w, seed=21) * 0.25
    img[:, :, 1] = np.clip(img[:, :, 1] + grass, 0.0, 1.0)
    return clamp01(img)


def gen_abstract(h=32, w=32) -> np.ndarray:
    """Colorful abstract pattern using sin/cos warps and a palette."""
    yy, xx = make_grid(h, w)
    # swirl fields
    u = np.sin(6 * xx + 4 * yy)
    v = np.cos(5 * xx - 3 * yy)
    r = 0.5 + 0.5 * (np.sin(u * 3.14) + v * 0.2)
    g = 0.5 + 0.5 * (np.cos(v * 3.14) + u * 0.2)
    b = 0.3 + 0.7 * (0.5 * np.sin(u + v) + 0.5)
    # mix palette teal/purple/orange/yellow style by weighting
    img = np.stack([r, g, b], axis=-1)
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    # add smoothed noise
    noise = smooth_noise(h, w, seed=7) * 0.15
    img += noise[..., None]
    img = clamp01(img)
    return img


def gen_fruit_bowl(h=32, w=32) -> np.ndarray:
    """Synthetic fruit bowl: circular apples, bowl arc, shading."""
    yy, xx = make_grid(h, w)
    img = np.ones((h, w, 3), dtype=np.float32) * 0.9 * np.expand_dims(1 - 0.2 * yy, -1)
    # bowl arc (brown)
    bowl_mask = ((yy - 0.75) ** 2 + (xx - 0.5) ** 2) <= (0.45 ** 2)
    img[bowl_mask] = img[bowl_mask] * 0.6 + np.array([0.45, 0.32, 0.2])[None]
    # apples: 3 circles with shading
    apples = [
        (0.42, 0.55, 0.09, np.array([0.9, 0.12, 0.1])),  # red
        (0.6, 0.58, 0.08, np.array([0.1, 0.8, 0.12])),  # green
        (0.49, 0.42, 0.07, np.array([0.95, 0.6, 0.1])),  # yellow/orange
    ]
    for cx, cy, r, col in apples:
        mask = draw_circle_mask(h, w, cx, cy, r)
        # shading: darker underside via yy
        shade = 0.2 * (yy - cy)
        shade = np.clip(shade, -0.15, 0.15)
        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - mask) + (col[c] + shade * mask) * mask + img[:, :, c] * 0.0
    img = clamp01(img)
    return img


def gen_portrait_blob(h=32, w=32) -> np.ndarray:
    """Portrait-ish blob with skin tone gradient, head+shoulder silhouette and vignette."""
    yy, xx = make_grid(h, w)
    # background dark
    img = np.zeros((h, w, 3), dtype=np.float32) + 0.06
    # silhouette mask: oval head+shoulders
    cx, cy = 0.5, 0.42
    mask = ((xx - cx) ** 2) / (0.13 ** 2) + ((yy - cy) ** 2) / (0.18 ** 2)
    mask = (mask <= 1.0).astype(np.float32)
    # skin gradient
    skin = np.zeros((h, w, 3), dtype=np.float32)
    skin[:, :, 0] = 0.86 - 0.06 * yy  # r
    skin[:, :, 1] = 0.70 - 0.05 * yy  # g
    skin[:, :, 2] = 0.56 - 0.04 * yy  # b
    # apply
    img = img * (1 - mask[..., None]) + skin * (mask[..., None])
    # vignette: darken edges
    vign = radial_gradient(h, w, cx=0.5, cy=0.45, radius=0.9)
    img = img * vign[..., None]
    return clamp01(img)


# -------------------------
# Reference operation functions
# -------------------------
def ref_brighten2(img: np.ndarray) -> np.ndarray:
    return clamp01(2.0 * img)


def ref_box_blur3(img: np.ndarray) -> np.ndarray:
    return box_blur3(img)


def ref_crop_topleft_10(img: np.ndarray) -> np.ndarray:
    # produce a 10x10 HWC output (top-left)
    h, w = img.shape[:2]
    out = img[0:10, 0:10]
    return out.copy()


def ref_invert(img: np.ndarray) -> np.ndarray:
    return clamp01(1.0 - img)


def ref_grayscale(img: np.ndarray) -> np.ndarray:
    """
    EXACT Halide-style luminance:
        gray = 0.299*R + 0.587*G + 0.114*B
    Returns float32 HxW array.
    """
    gray = (
        0.299 * img[:, :, 0]
        + 0.587 * img[:, :, 1]
        + 0.114 * img[:, :, 2]
    )
    return gray.astype(np.float32)



def ref_swap_rb(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    out = out[:, :, [2, 1, 0]]
    return out


def ref_hflip(img: np.ndarray) -> np.ndarray:
    return img[:, ::-1, :].copy()


# -------------------------
# Map tasks -> ref functions and expected output shape specifics
# -------------------------
TASK_FN_MAP = {
    "Brighten an image by 2x": (ref_brighten2, True),  # returns RGB HWC
    "Apply a 3x3 box blur": (ref_box_blur3, True),
    "Crop an image to 10x10 region": (ref_crop_topleft_10, True),
    "Invert image colors": (ref_invert, True),
    "Convert image to grayscale": (ref_grayscale, False),  # returns single channel HxW
    "Swap red and blue channels": (ref_swap_rb, True),
    "Apply horizontal flip": (ref_hflip, True),
}


# -------------------------
# Driver: produce images and build JSON entries
# -------------------------
def generate_base_images() -> List[Tuple[str, np.ndarray]]:
    imgs = []
    imgs.append(("bird_sky", gen_bird_sky()))
    imgs.append(("mountain_sun", gen_mountain()))
    imgs.append(("abstract_pattern", gen_abstract()))
    imgs.append(("fruit_bowl", gen_fruit_bowl()))
    imgs.append(("portrait_blob", gen_portrait_blob()))
    return imgs


def augment_examples():
    # load source examples JSON
    if not os.path.exists(SRC_EXAMPLES_JSON):
        raise FileNotFoundError(f"Missing source examples JSON: {SRC_EXAMPLES_JSON}")
    with open(SRC_EXAMPLES_JSON, "r", encoding="utf-8") as f:
        src = json.load(f)

    base_images = generate_base_images()

    # ensure images dir
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # for each task in source JSON, find matching prompt and augment
    out_examples = []
    for example in src:
        prompt = example.get("prompt", "")
        halide_code = example.get("halide_code", "")
        # default: keep the original small cases but replace with 5 large ones
        if prompt not in TASK_FN_MAP:
            # keep original (no change)
            out_examples.append(example)
            continue

        ref_fn, is_rgb = TASK_FN_MAP[prompt]

        test_cases = []
        # loop over base images
        for i, (name, img) in enumerate(base_images):
            # compute expected
            expected = ref_fn(img)
            # if grayscale, expected will be HxW not HxWx3
            if is_rgb:
                expected_output = expected
            else:
                expected_output = expected  # HxW (single channel)

            # save input and expected images under per-task folder
            task_img_dir = os.path.join(IMAGES_DIR, prompt.replace(" ", "_"))
            os.makedirs(task_img_dir, exist_ok=True)

            input_png = os.path.join(task_img_dir, f"image_{i+1}_{name}_input.png")
            expected_png = os.path.join(task_img_dir, f"image_{i+1}_{name}_expected.png")

            save_png(img, input_png)
            # For grayscale expected convert to single-channel image for saving as PNG
            if not is_rgb:
                # expected is HxW float
                save_png(np.stack([expected_output] * 3, axis=-1), expected_png)
            else:
                save_png(expected_output, expected_png)

            # record case JSON
                        # Record input shape normally (32x32x3)
            case = {
                "format": "HWC",
                "dtype": "float32",
                "shape": list(img.shape),     # always 32×32×3 input
                "input": as_list(img),
            }

            if is_rgb:
                # RGB outputs use standard HWC shape
                case["expected_output"] = as_list(expected_output)
            else:
                # Grayscale → H×W single-channel output
                case["expected_output"] = expected_output.astype(np.float32).tolist()
                case["shape_expected"] = list(expected_output.shape)  # e.g. [32, 32]


            case["notes"] = f"Auto-generated input '{name}' (32x32). Expected computed with NumPy ref {ref_fn.__name__}."
            test_cases.append(case)

        # Build new example entry preserving halide_code but replacing test_cases
        new_ex = {
            "prompt": prompt,
            "halide_code": halide_code,
            "test_cases": test_cases
        }
        out_examples.append(new_ex)

    # write augmented JSON
    with open(OUT_EXAMPLES_JSON, "w", encoding="utf-8") as f:
        json.dump(out_examples, f, indent=2)

    print(f"Augmented examples written to: {OUT_EXAMPLES_JSON}")
    print(f"Images saved to: {IMAGES_DIR}")
    return out_examples


if __name__ == "__main__":
    print("Generating 5 base 32x32 images and augmenting examples.json with NumPy reference outputs...")
    augmented = augment_examples()
    print("Done.")
