#!/usr/bin/env python3

"""
Clean halide_examples_augmented.json by removing notes,
cleaning Halide code, and ensuring structural consistency.

IMPORTANT:
- Does NOT compress pixel arrays.
- Preserves all input and expected_output arrays exactly.
"""

import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXAMPLES_DIR = os.path.join(ROOT, "examples")

SRC = os.path.join(EXAMPLES_DIR, "halide_examples_augmented.json")
DST = os.path.join(EXAMPLES_DIR, "halide_examples_clean.json")


def clean_code(code: str) -> str:
    """Remove markdown fences, inline comments, blank lines."""
    clean = []
    for line in code.splitlines():
        stripped = line.strip()

        # remove markdown fences
        if stripped.startswith("```"):
            continue
        # strip comments
        if "#" in line:
            line = line.split("#")[0]

        # skip empty
        if stripped == "":
            continue

        clean.append(line.rstrip())

    return "\n".join(clean)


def infer_shape(arr):
    """Infer shape from nested lists."""
    if not isinstance(arr, list):
        return None

    # Expect HWC
    H = len(arr)
    W = len(arr[0]) if H > 0 else 0

    # Grayscale case?
    first = arr[0][0]
    if isinstance(first, list):
        C = len(first)
        return [H, W, C]
    else:
        # grayscale 2D
        return [H, W]


def clean_test_case(tc: dict) -> dict:
    """Remove notes, ensure dtype/format/shape consistency."""
    out = dict(tc)

    # 1. Remove notes
    out.pop("notes", None)

    # 2. Ensure dtype/format
    out["dtype"] = "float32"
    out["format"] = "HWC"

    # 3. Clean shape (input)
    if "input" in out:
        shape = infer_shape(out["input"])
        if shape is not None:
            out["shape"] = shape

    # 4. Expected_output shape is implicitly correct (do not alter pixel arrays!)
    if "expected_output" in out:
        # if grayscale (HxW), shape_expected tracks that
        shape_e = infer_shape(out["expected_output"])
        if shape_e is not None and len(shape_e) == 2:
            out["shape_expected"] = shape_e

    return out


def clean_examples():
    with open(SRC, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = []

    for ex in data:
        new_ex = {}

        # Copy prompt
        new_ex["prompt"] = ex["prompt"]

        # Clean code
        new_ex["halide_code"] = clean_code(ex.get("halide_code", ""))

        # Clean all test cases (pixel arrays preserved)
        raw_cases = ex.get("test_cases", [])
        new_cases = [clean_test_case(tc) for tc in raw_cases]
        new_ex["test_cases"] = new_cases

        cleaned.append(new_ex)

    with open(DST, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2)

    print(f"✔ Cleaned examples saved to:\n{DST}")


if __name__ == "__main__":
    clean_examples()
