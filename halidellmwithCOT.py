import os
import re
import json
import dspy
import halide as hl
import numpy as np
import subprocess
import tempfile
import imageio.v2 as imageio
from dspy.teleprompt import BootstrapFewShot


# ======================================================
# 1. DSPy Model Configuration
# ======================================================

lm = dspy.LM(
    "openai/llama3.1:8b",
    api_base="http://172.27.21.160:11434/v1",
    api_key="ollama",
    model_type="chat",
    temperature=0.2,
)
dspy.configure(lm=lm)


# ======================================================
# 2. Signature and Pipeline Setup
# ======================================================

with open("halide_examples.json") as f:
    raw_examples = json.load(f)

reference_examples = [
    dspy.Example(
        prompt=ex["prompt"],
        halide_code=ex["halide_code"],
        test_cases=json.dumps(ex["test_cases"])
    ).with_inputs("prompt")
    for ex in raw_examples
]


class HalideCodeGen(dspy.Signature):
    """LLM signature for generating Halide code and test cases."""
    prompt = dspy.InputField(desc="User request describing the image processing operation to perform.")
    thoughts = dspy.OutputField(desc="Step-by-step reasoning explaining how to approach the problem.")
    halide_code = dspy.OutputField(
        format="code",
        desc="Full valid Python Halide code string that runs end-to-end."
    )
    test_cases = dspy.OutputField(
        format="json",
        desc=(
            "A JSON array of multiple test cases (HWC format). "
            "Each test case must contain 'format', 'dtype', 'shape', 'input', 'expected_output', and 'notes'. "
            "Use small clean numbers like 0.0, 0.25, 0.5, 0.75, 1.0."
        )
    )


class HalidePipeline(dspy.Module):
    """
    Generates Halide code using Chain-of-Thought reasoning and adaptive refinement.
    If a validation round fails, the LLM's reasoning and validator feedback are
    automatically fed back into the next iteration.
    """

    def __init__(self):
        super().__init__()
        # Use Chain-of-Thought generator (supports reasoning + final output)
        self.gen = dspy.ChainOfThought(HalideCodeGen)
        with open("halide_examples.json", "r") as f:
            self.examples = json.load(f)

    def forward(self, full_prompt=None, user_input=None, feedback=None, prev_thoughts=None, prompt=None):
        """
        Generate Halide code and test cases, using reasoning from examples and previous feedback.
        Compatible with both DSPy bootstrapping (which passes `prompt=`) and runtime loop (which passes `full_prompt, user_input=`).
        """

        # --- Backward compatibility for DSPy training ---
        if user_input is None and prompt is not None:
            user_input = prompt
        if full_prompt is None:
            full_prompt = user_input or prompt or ""

        # --- Guard: prevent NoneType issues ---
        if not user_input:
            user_input = ""
        if not full_prompt:
            full_prompt = ""

        # 🧠 Compare only using user_input text (safe)
        def overlap(a, b):
            return len(set(a.lower().split()) & set(b.lower().split()))

        # 🧠 Compare only using user_input, not the full prompt
        best_match = max(
            self.examples,
            key=lambda ex: len(
                set(user_input.lower().split()) & set(ex["prompt"].lower().split())
            )
        )

        # Construct a reasoning-rich reference context
        ref = f"""
    {full_prompt}

    Reference Example for this particular task is:
    Prompt: {best_match['prompt']}
    Halide Code:
    {best_match['halide_code']}
    Test Cases:
    {best_match['test_cases']}

    Instructions:
    1. Think step by step about how this reference relates to the new prompt.
    2. Adapt it carefully while preserving correct image I/O, transposition rules, and data types.
    3. You MUST generate atleast 4 test cases, ideal range of number of test cases is 4-5
    4. Always clamp output values between 0.0 and 1.0.
    5. If feedback is given, fix your previous mistakes accordingly.
    """

        if feedback:
            ref += f"\nPrevious Feedback:\n{feedback}\n"
        if prev_thoughts:
            ref += f"\nPrevious Reasoning (for self-reflection):\n{prev_thoughts}\n"

        ref += """
    Now, reason carefully step-by-step before coding.
    Then output:
    1. Your reasoning as plain text.
    2. The final Halide code.
    3. A JSON array of test cases (input + expected output).
    """

        # Run reasoning + code generation
        result = self.gen(prompt=ref)

        # Print reasoning for debugging
        print("\n================= REASONING DEBUG =================")
        print(result.thoughts)
        print("===================================================\n")

        return result


# Bootstrap few-shot learning with existing examples
tele = BootstrapFewShot(metric=None)
refined_pipeline = tele.compile(HalidePipeline(), trainset=reference_examples)


# ======================================================
# 3. JSON Utility Functions
# ======================================================

def extract_halide_code(text):
    """Extract and clean the halide_code block from model output."""
    if not isinstance(text, str):
        return text

    # 🧹 Remove any markdown fences like ```python or ```json
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*", "", text)  # remove leading ```python or ```json
    text = text.replace("```", "")  # remove closing or stray ```
    
    # 🔍 Try to extract code from a JSON-style response
    match = re.search(r'"halide_code"\s*:\s*"([\s\S]*?)"\s*,\s*"?test_cases"?', text)
    if match:
        code = match.group(1)
        code = code.encode("utf-8").decode("unicode_escape")
        return code.strip()

    # 🧩 Fallback: if not JSON-wrapped, return the cleaned text block
    return text.strip()


def _extract_json_block(text):
    """Extract a valid JSON object or array from text."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text)
    text = re.sub(r"```$", "", text)
    match = re.search(r"(\{.*\})", text, flags=re.S)
    return match.group(1).strip() if match else text.strip()


def _safe_json(obj):
    """Convert numpy types for safe JSON serialization."""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


import json, re, ast

def _clean_json_array_block(raw_text: str):
    """
    Extracts and cleans a JSON array of test cases from model output.
    Handles code fences, Python-style dicts, stray text, and bracket mismatches.
    """
    if not isinstance(raw_text, str):
        return raw_text

    text = raw_text.strip()
    # Remove Markdown fences (```json, ```python, etc.)
    text = re.sub(r"^```[a-zA-Z]*", "", text)
    text = re.sub(r"```$", "", text)
    text = text.replace("```", "")

    # Extract the first JSON-like array block
    match = re.search(r"(\[[\s\S]*\])", text)
    if not match:
        raise ValueError("No JSON array found in text.")
    cleaned = match.group(1).strip()

    # Stage 1: try direct JSON
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Stage 2: try Python-style (single quotes)
    try:
        parsed = ast.literal_eval(cleaned)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # Stage 3: cleanup and truncate
    repaired = cleaned.replace("'", '"')
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)

    # Truncate to last bracket if malformed
    last_bracket = repaired.rfind("]")
    if last_bracket != -1:
        repaired = repaired[:last_bracket + 1]

    # Final attempt
    try:
        return json.loads(repaired)
    except Exception as e:
        snippet = cleaned[:200].replace("\n", " ")
        raise ValueError(f"JSON parsing failed after cleanup: {e}\nSnippet: {snippet}")





# ======================================================
# 4. Halide Validator Loop
# ======================================================

class HalideValidatorLoop:
    def __init__(self, generator, max_attempts=2):
        self.generator = generator
        self.max_attempts = max_attempts

    def _run_python(self, code: str, cwd: str, filename: str):
        """Run generated Halide code and capture stdout/stderr for debugging."""
        path = os.path.join(cwd, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            output = subprocess.check_output(
                ["python", path],
                stderr=subprocess.STDOUT,
                cwd=cwd,
                timeout=20,
            )
            return True, output.decode("utf-8", errors="ignore")

        except subprocess.CalledProcessError as e:
            # 🔍 Print first 40 lines of error for readability
            decoded = e.output.decode("utf-8", errors="ignore")
            print("\n---- ⚠️ RUNTIME ERROR LOG ----")
            print("\n".join(decoded.splitlines()[:40]))
            print("---- END OF ERROR LOG ----\n")

            full_error = (
                f"Exit code: {e.returncode}\n"
                f"Output:\n{decoded}"
            )
            return False, full_error

        except Exception as e:
            print("\n---- ⚠️ UNEXPECTED PYTHON ERROR ----")
            print(repr(e))
            print("---- END OF ERROR ----\n")
            return False, f"Unexpected runtime error: {repr(e)}"



    def _write_image(self, arr, path):
        arr = np.array(arr, dtype=np.float32)

        # Handle scalar or 1D input (invalid)
        if arr.ndim == 0:
            arr = np.full((1, 1, 3), arr)
        elif arr.ndim == 1:
            arr = np.tile(arr[:, None, None], (1, 1, 3))
        elif arr.ndim == 2:
            # Convert grayscale 2D to RGB by stacking
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            # Convert single-channel 3D to RGB
            arr = np.repeat(arr, 3, axis=-1)

        # Clamp and save
        arr = np.clip(arr, 0.0, 1.0)
        imageio.imwrite(path, (arr * 255.0).astype(np.uint8))



    def _load_image(self, path):
        return imageio.imread(path).astype(np.float32) / 255.0

    def _compare(self, actual, expected, atol=0.05):
        actual = np.clip(actual, 0.0, 1.0)
        expected = np.clip(expected, 0.0, 1.0)
        if actual.shape != expected.shape:
            return 0.0, {"error": f"shape mismatch {actual.shape} vs {expected.shape}"}
        diff = np.abs(actual - expected)
        max_diff, mean_diff = float(diff.max()), float(diff.mean())
        passed = np.allclose(actual, expected, atol=atol)
        score = 1.0 if passed else max(0.0, 1.0 - mean_diff * 10)
        return score, {"max_diff": max_diff, "mean_diff": mean_diff}


    def _validate_multiple_cases(self, halide_code: str, test_cases_json: str):
        print("\n================= JSON Parsing Debug =================")
        print(test_cases_json[:600])
        print("=====================================================\n")

        try:
            parsed = _clean_json_array_block(test_cases_json)
            # ✅ Handle case where LLM wraps test cases inside an object
            if isinstance(parsed, dict) and "test_cases" in parsed:
                cases = parsed["test_cases"]
            elif isinstance(parsed, list):
                cases = parsed
            else:
                raise ValueError(f"Unexpected JSON structure: {type(parsed)}")
            print(f"✅ Extracted {len(cases)} test case(s) successfully.")
        except Exception as e:
            print("JSON parse error:", e)
            return {"syntax_ok": False, "error": f"Bad JSON: {e}", "results": []}

        all_results = []

        for idx, case in enumerate(cases):
            if not isinstance(case, dict):
                print(f"⚠️ Skipping malformed case #{idx}: not a dict ({type(case)})")
                continue

            print(f"\nRunning Test Case #{idx+1}: {case.get('notes', '(no notes)')}")

            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    input_arr = np.array(case["input"], dtype=case.get("dtype", "float32"))
                    expected_arr = np.array(case["expected_output"], dtype=case.get("dtype", "float32"))

                    if input_arr.ndim == 2:
                        input_arr = np.expand_dims(input_arr, axis=-1)
                    if expected_arr.ndim == 2:
                        expected_arr = np.expand_dims(expected_arr, axis=-1)

                    if input_arr.ndim == 4 and input_arr.shape[0] == 1:
                        input_arr = np.squeeze(input_arr, axis=0)
                    if expected_arr.ndim == 4 and expected_arr.shape[0] == 1:
                        expected_arr = np.squeeze(expected_arr, axis=0)

                    print(f"   Input shape: {input_arr.shape}, Expected shape: {expected_arr.shape}")

                except Exception as e:
                    all_results.append({"idx": idx, "error": f"Bad arrays: {e}"})
                    continue

                self._write_image(input_arr, os.path.join(tmpdir, "input.png"))
                self._write_image(expected_arr, os.path.join(tmpdir, "expected_output.png"))

                ok, err = self._run_python(halide_code, tmpdir, "halide_code.py")
                if not ok:
                    snippet = "\n".join(err.splitlines()[:8])  # only first 8 lines
                    all_results.append({"idx": idx, "error": f"Halide code failed:\n{snippet}"})
                    continue

                output_path = os.path.join(tmpdir, "output.png")
                if not os.path.exists(output_path):
                    all_results.append({"idx": idx, "error": "Halide code ran but did not produce output.png"})
                    continue

                try:
                    actual = self._load_image(output_path)
                    expected = self._load_image(os.path.join(tmpdir, "expected_output.png"))
                except Exception as e:
                    all_results.append({"idx": idx, "error": f"Image load failed: {e}"})
                    continue

                score, info = self._compare(actual, expected)
                if score < 1.0 and info["max_diff"] < 0.1:
                    # Accept small rounding errors as pass
                    score = 1.0

                result = {
                    "idx": idx,
                    "correctness_index": round(score, 3),
                    "max_diff": info.get("max_diff"),
                    "mean_diff": info.get("mean_diff"),
                    "expected_mean": float(np.round(expected.mean(), 3)),
                    "actual_mean": float(np.round(actual.mean(), 3))
                }
                all_results.append(result)

                if score < 1.0:
                    print(f"❌ Test #{idx+1} failed")
                    print(f"   Expected mean: {result['expected_mean']}")
                    print(f"   Actual mean: {result['actual_mean']}")
                    print(f"   Max diff: {result['max_diff']}")
                    print(f"   Input array (sample): {np.round(input_arr[:2, :2, :], 3)}")
                    print(f"   Expected output (sample): {np.round(expected_arr[:2, :2, :], 3)}")
                    print(f"   Actual output (sample): {np.round(actual[:2, :2, :], 3)}")
                    result["input_sample"] = np.round(input_arr, 3).tolist()
                    result["expected_output_sample"] = np.round(expected_arr, 3).tolist()
                    result["actual_output_sample"] = np.round(actual, 3).tolist()
                else:
                    print(f"✅ Test #{idx+1} passed")


        syntax_ok = all(r.get("error") is None for r in all_results)
        return {"syntax_ok": syntax_ok, "results": all_results}


# ======================================================
# 5. Feedback + React Loop
# ======================================================

def react_loop_with_code_feedback(base_prompt, pipeline, validator, max_rounds=3):
    last_error = None
    last_halide_code = None
    last_test_cases = None
    results = []

    for i in range(max_rounds):
        print(f"\n{'='*80}")
        print(f"🌀 ROUND {i+1}")
        print(f"{'='*80}\n")

                # Always keep the same user prompt across rounds (avoid box-blur fallback)
        user_prompt = base_prompt.strip().split("Now, write **full runnable Python code** that performs the operation:")[-1].strip().split("\n")[0].strip('" ')
        
        print(f"\n[Debug] Using user prompt for this round: {user_prompt}\n")

        # Construct context for pipeline (without full reference clutter)
        feedback_context = None
        if last_error:
            feedback_context = f"Previous feedback:\n{last_error}"

        result = pipeline(base_prompt, user_prompt, feedback=feedback_context)


        # Run model

        # ---- Debug Print ----
        print("\n================= RAW MODEL OUTPUT DEBUG =================")
        print(result.halide_code)
        print("\n--- TEST CASES ---\n")
        print(result.test_cases)
        print("==========================================================\n")

        # Extract and validate
        halide_code = extract_halide_code(result.halide_code)
        test_cases_json = result.test_cases

        validation = validator._validate_multiple_cases(halide_code, test_cases_json)
        results.append(validation)

        print("\n🔍 ROUND SUMMARY")
        print("-" * 60)

        # 1️⃣ Show syntax or JSON errors immediately
        if not validation.get("syntax_ok", True):
            print("❌ JSON/Syntax Error:")
            print(validation.get("error", "Unknown error"))
            print("-" * 60)
        else:
            # 2️⃣ Loop over test case results and show differences
            for r in validation["results"]:
                if "error" in r:
                    print(f"❌ Test case #{r.get('idx', '?')} failed to run:")
                    print(r["error"])
                    print("-" * 60)
                elif r.get("correctness_index", 1.0) < 0.99:
                    print(f"❌ Test #{r['idx']} failed numerical comparison:")
                    print(f"   Max diff: {r['max_diff']}")
                    print(f"   Mean diff: {r['mean_diff']}")
                    print(f"   Expected mean: {r['expected_mean']}")
                    print(f"   Actual mean:   {r['actual_mean']}")
                    if "input_sample" in r:
                        print("\n--- INPUT (sample) ---")
                        print(np.round(np.array(r['input_sample']), 3))
                        print("\n--- EXPECTED OUTPUT (sample) ---")
                        print(np.round(np.array(r['expected_output_sample']), 3))
                        print("\n--- ACTUAL OUTPUT (sample) ---")
                        print(np.round(np.array(r['actual_output_sample']), 3))
                    print("-" * 60)
                else:
                    print(f"✅ Test #{r['idx']} passed!")

        print("\n✅ ROUND COMPLETE")
        print("=" * 80)

        # 3️⃣ Prepare feedback for next round
        passed_all = (
            validation.get("syntax_ok", False)
            and all(
                (r.get("correctness_index", 0) >= 0.99)
                for r in validation.get("results", [])
            )
        )


        if passed_all:
            print("\n🎉 All test cases passed! Stopping early.")
            break

        # Pass detailed feedback (including samples) to next round
        fail_examples = [
            {
                "test_idx": r.get("idx"),
                "error": r.get("error"),
                "max_diff": r.get("max_diff"),
                "mean_diff": r.get("mean_diff"),
                "input_sample": r.get("input_sample", []),
                "expected_output_sample": r.get("expected_output_sample", []),
                "actual_output_sample": r.get("actual_output_sample", []),
            }
            for r in validation["results"]
            if (r.get("correctness_index", 1.0) < 0.99) or ("error" in r)
        ]

        if fail_examples:
            last_error = json.dumps(fail_examples, indent=2, default=_safe_json)
        else:
            last_error = None

        last_halide_code = halide_code
        last_test_cases = test_cases_json

    return results


# ======================================================
# 6. Main Interactive Entry
# ======================================================

def main():
    print("\nHalide Code Generator + Validator")
    print("Enter a description of the image operation you want Halide to perform.")
    print("Example: 'Apply a 3x3 box blur' or 'Invert image colors' or 'Brighten image by 2x'\n")

    user_input = input("Enter your operation prompt: ").strip()
    if not user_input:
        print("No prompt entered. Exiting.")
        return

    base_prompt = f"""
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

Now, let's generate comprehensive and diverse test cases step-by-step (COT-style).
We'll reason explicitly before producing the JSON.

---
First, think about what the operation does mathematically.  
For example, if it’s a blur, each pixel becomes the average of its neighbors.  
If it’s brightness scaling, each pixel is multiplied by a constant, etc.

Then, design **at least 4–5 distinct test cases** that help verify correctness.

Let's consider the following patterns for coverage and clarity:

1. **Flat Pattern** – all pixels have identical values (tests stability when no change is expected).
2. **Gradient Pattern** – values gradually increase across rows or columns (tests interpolation and averaging).
3. **Checkerboard Pattern** – alternating high and low values (tests neighborhood logic).
4. **Edge/Corner Highlight** – single bright pixel in dark image (tests edge clamping).
5. **Random/Mixed Values** – random but clamped values between 0.0 and 1.0 (tests overall robustness).

For each test case:
- Describe your reasoning in one short sentence.
- Show the exact numerical input and manually computed expected output.
- Use small arrays (e.g., 3×3, 4×4, or 5×5 RGB images).
- Ensure the `"shape"`, `"format"`, and `"dtype"` fields are consistent.
- Use **float32** dtype and `"format": "HWC"`.
- Round output values to 3 decimals.
- Clamp outputs between 0.0 and 1.0.
- Always return **valid JSON** (no markdown fences, no comments).

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


    validator = HalideValidatorLoop(generator=refined_pipeline, max_attempts=3)

    results = react_loop_with_code_feedback(
        base_prompt=base_prompt,
        pipeline=refined_pipeline,
        validator=validator,
        max_rounds=5
    )

    # 🧹 Only print the final round summary
    if isinstance(results, list) and len(results) > 0:
        final_result = results[-1]
        print("\n==============================")
        print("✅ FINAL ROUND SUMMARY")
        print("==============================")

        syntax_ok = final_result.get("syntax_ok", False)
        print(f"Syntax OK: {syntax_ok}")

        for r in final_result.get("results", []):
            if "error" in r:
                print(f"❌ Test #{r.get('idx', '?')} failed:")
                print(r["error"])
            else:
                ci = r.get("correctness_index", 0)
                if ci >= 0.99:
                    print(f"✅ Test #{r['idx']} passed (mean diff={r['mean_diff']:.4f}, max diff={r['max_diff']:.4f})")
                else:
                    print(f"⚠️ Test #{r['idx']} partial pass (correctness={ci:.2f}, mean diff={r['mean_diff']:.4f})")

        print("==============================\n")
    else:
        print("⚠️ No final results available.")


# ======================================================
# 7. Entry Point
# ======================================================

if __name__ == "__main__":
    main()
