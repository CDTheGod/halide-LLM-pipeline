def build_prompt(user_input: str) -> str:
    """
    Build a compact, structured Halide + Python reasoning prompt.
    Enforces:
      - precise reasoning,
      - full runnable Halide code generation,
      - 1 analytical test + 5 computed 32×32 test cases with explicit expected outputs.
    """
    return f"""
You are an expert Halide + Python developer specializing in image processing.

Your task is to generate **fully runnable Halide Python code** and **verified test cases**
for the following operation:
"{user_input}"

---

## 🧩 Phase 1 — Mathematical Reasoning

1️⃣ Restate the operation precisely as pixel math.  
   Examples:
   - Brighten ×2 → output(x,y,c) = min(2 * input(x,y,c), 1.0)
   - 3×3 blur → mean of neighborhood (edge clamped)
   - Invert → 1 - input(x,y,c)

2️⃣ Work through one **manual numeric example** on a 3×3 or 4×4 RGB image:
   - Show the formula applied to at least one pixel.
   - Round all intermediate values to 3 decimals.
   - Clamp to [0.0, 1.0].

---

## ⚙️ Phase 2 — Generate Halide Code

Produce **one runnable Halide Python script** that performs this operation.
Follow these exact rules:

- Imports (in order):
  ```python
  import halide as hl
  import imageio
  import numpy as np
Use:

python
Copy code
input = hl.ImageParam(hl.Float(32), 3)
x, y, c = hl.Var('x'), hl.Var('y'), hl.Var('c')
f = hl.Func('f')
Transpose input HWC→CHW before creating hl.Buffer, and output CHW→HWC before saving.

Clamp indices with hl.clamp(), and pixel values with hl.min(..., 1.0) if scaling.

Input filename = "input.png", output = "output.png".

The script must be standalone — no functions, markdown, or comments.

Use only Halide primitives (hl.Func, hl.Var, hl.ImageParam, hl.clamp, hl.min, etc.).

🧪 Phase 3 — Test Case Generation
Generate exactly 7 JSON test cases:

2 small analytical tests (3×3 or 5×5) with manually computed expected values.

5 larger 32×32 RGB tests, each with explicitly computed expected_output.

Each case must include:

json
Copy code
{{
  "format": "HWC",
  "dtype": "float32",
  "shape": [H, W, C],
  "input": [[[r,g,b], ...], ...],
  "expected_output": [[[r,g,b], ...], ...],
  "notes": "Short explanation of what this case checks"
}}
Rules:
All pixel values must be in [0.0, 1.0].

Round all floats to 3 decimals.

Dtype = "float32", format = "HWC".

For 32×32 cases, list the actual computed numeric arrays (no descriptions or summaries).

Every expected output must be computed explicitly using the same operation.

📦 Phase 4 — Output Format
Return one JSON object only (no markdown, no explanations):

json
Copy code
{{
  "halide_code": "<full runnable Python code>",
  "test_cases": [ ... ]
}}
Now, reason step-by-step and output your final JSON object only.
Do not include markdown formatting, code fences, or additional commentary.
"""