def build_prompt(user_input: str) -> str:
    """
    Builds the full reasoning + instruction prompt for the LLM,
    inserting the user's operation into the correct location.
    """
    return f"""
You are an expert Halide + Python developer and image processing researcher.

================= REASONING DEBUG =================
You are a precise Halide + Python image processing assistant.

Your job is to:
1. Generate correct Halide Python code for the operation described.
2. Then generate multiple mathematically verified test cases that confirm correctness.

Follow this exact reasoning protocol before you output anything.

### 🧩 Step 1: Restate and Formalize the Operation
Briefly restate the user’s image operation and the exact pixel-level math.  
For example:
- Brighten by 2× → `output(x,y,c) = min(2 * input(x,y,c), 1.0)`
- 3×3 Box Blur → `output(x,y,c) = average of all valid input(x+i,y+j,c)` for i,j in [-1,1], edge-clamped
- Invert → `output(x,y,c) = 1.0 - input(x,y,c)`

Always state the pixel formula clearly before proceeding.

---

### 🧮 Step 2: Compute an Example by Hand
Take a **small test image** (3×3 or 4×4, RGB or single-channel) and compute **at least one output pixel** numerically:
- Show the neighboring values you averaged or scaled.
- Compute the exact numeric result step by step.
- Round all intermediate values to 3 decimals.
- Clamp to [0.0, 1.0].

This step ensures mathematical correctness before generating test cases.

---

### 🧪 Step 3: Generate Test Cases
Now generate **at least 5 test cases** that thoroughly test the operation:

| Case | Description | Size | Purpose |
|------|--------------|------|----------|
| 1 | Uniform image | 3×3 | Should remain same (for blur) or scale evenly (for brightening) |
| 2 | Gradient | 3×3 or 4×4 | Checks interpolation and scaling accuracy |
| 3 | Checkerboard | 3×3 or 4×4 | Detects incorrect neighborhood logic |
| 4 | Edge bright pixel | 5×5 | Tests edge clamping |
| 5 | Random values | 4×4 | Tests robustness of arithmetic |

For each test case:
- Include `format`, `dtype`, `shape`, `input`, and `expected_output`.
- Compute each `expected_output` **numerically** using the formula.
- Round outputs to **3 decimals**.
- ENSURE ALL VALUES LIE BETWEEN 0.0 AND 1.0
- Add a short `"notes"` field explaining what this case checks.

---

### ⚙️ Step 4: Output Format
After reasoning, output **only** the final Halide code and JSON test cases in the format below:

⚠️ CRITICAL RULES — ALWAYS FOLLOW THESE EXACTLY:
1️⃣ Halide expects CHW (channel, height, width) layout.
   - Transpose input from HWC → CHW before creating the Halide Buffer:
     `img_np = np.transpose(img_np, (2, 0, 1)).copy()`
   - Transpose output from CHW → HWC after realization:
     `output = np.transpose(output, (1, 2, 0))`
2️⃣ Clamp values > 1.0 with `hl.min(expr, 1.0)` whenever brightening or multiplying.
3️⃣ Input image filename must be `"input.png"`, output `"output.png"`.
4️⃣ Do not define functions or classes. The output must be **one runnable script**.
5️⃣ Use only Halide primitives (`hl.Func`, `hl.Var`, `hl.ImageParam`, `hl.min`, `hl.clamp`, etc.).
6️⃣ Keep the import order and general structure identical to the reference.
7️⃣ Never use Python slicing syntax (like `input[x-1:x+2]`).
8️⃣ All test cases must represent 3-channel RGB images with values from 0.0 to 1.0, unless it is mentioned to be in grayscale in which case it should be single channel values

---

### REFERENCE BASELINE
(The following Halide program simply copies the input image; modify only logic as needed.)

import halide as hl
import imageio
import numpy as np

input = hl.ImageParam(hl.Float(32), 3)
f = hl.Func('f')
x, y, c = hl.Var('x'), hl.Var('y'), hl.Var('c')
f[x, y, c] = input[x, y, c]

img_np = imageio.imread('input.png').astype(np.float32) / 255.0
img_np = np.ascontiguousarray(img_np)
img_np = np.transpose(img_np, (2, 0, 1)).copy()
img = hl.Buffer(img_np)
input.set(img)

output = f.realize([img.width(), img.height(), img.channels()])
output = np.array(output)
output = np.transpose(output, (1, 2, 0))
imageio.imsave('output.png', (output * 255.0).astype(np.uint8))

---

### Here is one EXAMPLE FOR REFERENCE for box blur
# (for context only — DO NOT REPEAT IT unless useful)

import halide as hl
import imageio
import numpy as np

# Step 1: Load and normalize the input image
img_np = imageio.imread("input.png").astype(np.float32) / 255.0

# Step 2: Ensure shape is (height, width, channels) and memory is contiguous
if img_np.ndim == 2:
    img_np = img_np[:, :, np.newaxis]  # grayscale fallback
img_np = np.ascontiguousarray(img_np)

# Step 3: Transpose to match Halide's layout: [channels, height, width]
img_for_halide = np.transpose(img_np, (2, 0, 1)).copy()
input_buf = hl.Buffer(img_for_halide)

# Step 4: Declare Halide Vars and ImageParam
x, y, c = hl.Var("x"), hl.Var("y"), hl.Var("c")
input = hl.ImageParam(hl.Float(32), 3)
input.set(input_buf)

# Step 5: Clamp edges to avoid out-of-bounds access
clamped = hl.Func("clamped")
clamped[x, y, c] = input[
    hl.clamp(x, 0, input_buf.width() - 1),
    hl.clamp(y, 0, input_buf.height() - 1),
    c
]

# Step 6: Horizontal blur
blur_x = hl.Func("blur_x")
blur_x[x, y, c] = (clamped[x - 1, y, c] + clamped[x, y, c] + clamped[x + 1, y, c]) / 3.0

# Step 7: Vertical blur
blur_y = hl.Func("blur_y")
blur_y[x, y, c] = (blur_x[x, y - 1, c] + blur_x[x, y, c] + blur_x[x, y + 1, c]) / 3.0

# Step 8: Realize and save
output = blur_y.realize([input_buf.width(), input_buf.height(), input_buf.channels()])
output_np = np.array(output)
output_np = np.transpose(output_np, (1, 2, 0))
imageio.imsave("output.png", (output_np * 255.0).astype(np.uint8))

---

### TASK
Now, write **full runnable Python code** that performs the operation:
"{user_input}"

Then, produce realistic test cases that logically demonstrate the operation.

---

### TEST CASE GENERATION (REASON CAREFULLY)

### 🧪 Step 3: Generate Test Cases

Now generate **6–8 test cases**, divided into two categories:

#### 🧩 Category A: Analytical micro-tests (3×3 or 5×5)
Used to verify exact mathematical correctness.  
- These should be small enough that you can manually show expected pixel values.
- Typical patterns: uniform, gradient, checkerboard, edge pixel, random small patch.

#### 🖼️ Category B: Realistic macro-tests (≥32×32)
Used to demonstrate the visual behavior of the operation.  
- Generate 2 realistic synthetic images:
  - One 32×32 gradient or color ramp image.
  - One 32×32 or 64×64 random noise or pattern image.
- Compute expected outputs algorithmically (using vectorized logic, not manual listing).
- Mention `"notes": "Larger synthetic test for realism"` in these cases.

For **each test case**:
- Include `"format"`, `"dtype"`, `"shape"`, `"input"`, `"expected_output"`, and `"notes"`.
- Use `"float32"` dtype and `"format": "HWC"`.
- For large cases, summarize inputs (e.g. show first few rows, or describe generation rule) rather than listing all pixels.
- Clamp outputs between 0.0 and 1.0.
- Always return **valid JSON** (no markdown fences, no comments).
- Describe your reasoning in one short sentence.
- Show the exact numerical input and manually computed expected output.
- Use small arrays (e.g., 3×3, 4×4, or 5×5 RGB images).
- Ensure the `"shape"`, `"format"`, and `"dtype"` fields are consistent.
- Use **float32** dtype and `"format": "HWC"`.
- Round output values to 3 decimals.
- Clamp outputs between 0.0 and 1.0.

Let’s reason step-by-step:
1. For each pattern type above, imagine the pixel grid.
2. Apply the image operation formula to each pixel.
3. Write down the resulting expected output array.
4. Finally, produce a JSON list of all test cases in this format:

---

### OUTPUT FORMAT (strict JSON only, no markdown fences)
{{
  "halide_code": "full runnable Python Halide code (escaped properly)",
  "test_cases": [
    {{
      "format": "HWC",
      "dtype": "float32",
      "shape": [3, 3, 3],
      "input": [[[r,g,b], ...], ...],
      "expected_output": [[[r,g,b], ...], ...],
      "notes": "short description"
    }}
  ]
}}

---

Now, **reason step-by-step** about how pixel values will change, and only then output your final JSON object.
"""