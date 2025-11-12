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
    temperature=0.2
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
    prompt = dspy.InputField(desc="Natural language description of the image processing task.")
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
    """Generates Halide code, using examples when prompt matches."""
    def __init__(self):
        super().__init__()
        self.gen = dspy.Predict(HalideCodeGen)
        with open("halide_examples.json", "r") as f:
            self.examples = json.load(f)

    def forward(self, prompt):
        best_match = max(
            self.examples,
            key=lambda ex: len(set(prompt.lower().split()) & set(ex["prompt"].lower().split()))
        )
        ref = f"""
    Reference Example:
    Prompt: {best_match['prompt']}
    Code:
    {best_match['halide_code']}
    Test case:
    {best_match['test_cases']}
    Now modify it to achieve:
    {prompt}
    """
        return self.gen(prompt=ref)



# Bootstrap few-shot learning with existing examples
tele = BootstrapFewShot(metric=None)
refined_pipeline = tele.compile(HalidePipeline(), trainset=reference_examples)


# ======================================================
# 3. JSON Utility Functions
# ======================================================

def extract_halide_code(text):
    """Extract the value of halide_code from JSON-like model output."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text)
    text = re.sub(r"```$", "", text)
    match = re.search(r'"halide_code"\s*:\s*"([\s\S]*?)"\s*,\s*"?test_cases"?', text)
    if match:
        code = match.group(1)
        code = code.encode("utf-8").decode("unicode_escape")
        return code.strip()
    return text


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
    Handles double encoding, Python-style dicts, and extra text.
    """
    if not isinstance(raw_text, str):
        return raw_text

    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text)
    text = re.sub(r"```$", "", text)

    match = re.search(r"(\[[\s\S]*\])", text)
    if not match:
        raise ValueError("No JSON array found in text.")

    cleaned = match.group(1).strip()

    # Attempt JSON first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt to interpret as Python literal (single quotes or True/False)
    try:
        parsed = ast.literal_eval(cleaned)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # Last-ditch cleanup: fix common issues
    repaired = cleaned.replace("'", '"')
    repaired = re.sub(r"(\bTrue\b)", "true", repaired)
    repaired = re.sub(r"(\bFalse\b)", "false", repaired)
    repaired = re.sub(r"(\bNone\b)", "null", repaired)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        raise ValueError(f"JSON parsing failed after cleanup. First 200 chars:\n{cleaned[:200]}")





# ======================================================
# 4. Halide Validator Loop
# ======================================================

class HalideValidatorLoop:
    def __init__(self, generator, max_attempts=2):
        self.generator = generator
        self.max_attempts = max_attempts

    def _run_python(self, code: str, cwd: str, filename: str):
        path = os.path.join(cwd, filename)
        with open(path, "w") as f:
            f.write(code)
        try:
            subprocess.check_output(["python", path], stderr=subprocess.STDOUT, cwd=cwd)
            return True, None
        except subprocess.CalledProcessError as e:
            return False, e.output.decode()

    def _write_image(self, arr, path):
        arr = np.round(arr, 3)
        arr = np.clip(arr, 0.0, 1.0)
        imageio.imwrite(path, (arr * 255.0).astype(np.uint8))

    def _load_image(self, path):
        return imageio.imread(path).astype(np.float32) / 255.0

    def _compare(self, actual, expected, atol=0.01):
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
                    all_results.append({"idx": idx, "error": f"Halide code failed: {err}"})
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

        prompt = base_prompt
        if last_halide_code:
            prompt += f"\n\nPrevious Halide code:\n{last_halide_code}"
        if last_test_cases:
            prompt += f"\n\nPrevious test cases:\n{last_test_cases}"
        if last_error:
            prompt += f"\n\nError encountered:\n{last_error}\nPlease fix it."

        # Run model
        result = pipeline(prompt)

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
You are an expert Halide code generator.

You are given a **reference Python Halide example**. You must output similar code, 
adjusting the logic or constants as requested by the user.

⚠️ CRITICAL RULE: Halide expects CHW (channel, height, width) layout, 
while imageio/numpy use HWC (height, width, channel). 
You must **always**:
1. Transpose the numpy image from HWC → CHW before creating the Halide Buffer.
   Example: img_np = np.transpose(img_np, (2, 0, 1)).copy()
2. After realizing the Halide output, transpose it back from CHW → HWC before saving.
   Example: output = np.transpose(output, (1, 2, 0))
3. If there is a chance that in the task, any value can become greater then one, then all values must be clamped to 1. For eg. while brightening, you must use f[x, y, c] = hl.min(2 * input[x, y, c], 1.0)
4. The output image must be named 'output.png' and the input must be taken as 'input.png' always

If you forget this, the Halide output dimensions will be wrong.

--- Reference Example - this halide program will just print the same image , you must use this as your base halide program and make necessary changes in this as per the need of the prompt---
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

output = f.realize([img.width(),img.height(),img.channels()])
output = np.array(output)
output = np.transpose(output, (1, 2, 0))
imageio.imsave('output.png', (output * 255.0).astype(np.uint8))

Here is an example where we apply a 3x3 box blur on an image

import halide as hl
import imageio
import numpy as np

# Step 1: Load and normalize the input image
img_np = imageio.imread("input.png").astype(np.float32) / 255.0

# Step 2: Ensure shape is (height, width, channels) and memory is contiguous
if img_np.ndim == 2:
    img_np = img_np[:, :, np.newaxis]  # grayscale fallback
img_np = np.ascontiguousarray(img_np)

# Step 3: Transpose to match Halide's layout: [width, height, channels]
img_for_halide = np.transpose(img_np, (2,0,1)).copy()
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

# Step 6: Horizontal blur (blur across X)
blur_x = hl.Func("blur_x")
blur_x[x, y, c] = (clamped[x - 1, y, c] + clamped[x, y, c] + clamped[x + 1, y, c]) / 3.0

# Step 7: Vertical blur (blur across Y)
blur_y = hl.Func("blur_y")
blur_y[x, y, c] = (blur_x[x, y - 1, c] + blur_x[x, y, c] + blur_x[x, y + 1, c]) / 3.0

# Step 8: Realize the output buffer using correct shape
output = blur_y.realize([input_buf.width(), input_buf.height(), input_buf.channels()])

# Step 9: Convert Halide Buffer to NumPy array and transpose back to HWC
output_np = np.array(output)
output_np = np.transpose(output_np, (1,2,0))  # [height, width, channels]

# Step 10: Save the blurred image
imageio.imsave("output.png", (output_np * 255.0).astype(np.uint8))
print("Blurred image saved as output.png")

# Step 11: Debug pixel values
print("Original pixel:", img_np[100, 100])
print("Blurred pixel:", output_np[100, 100])

--- Task ---
Now, write **full runnable Python code** that performs the operation:
"{user_input}"

You must strictly follow the structure and layout of the reference Halide example.
- Do not define new functions.
- Do not add or remove imports.
- Do not use Python slicing syntax (e.g., input[x-1:x+2]).
- Use only Halide primitives (hl.Func, hl.Var, hl.ImageParam, hl.min, hl.clamp, etc.).
- Always use the same image loading, transposing, and saving logic as in the example.
- Do not invent new control flow or logic unrelated to the examples.


Adjust constants or formulas as needed (e.g., use 3x instead of 2x if asked).
Keep the structure, imports, and function layout the same.

You must output a strict JSON object with this structure:
{{
  "halide_code": "full runnable Python Halide code (escaped properly)",
  "test_cases": [
    {{
      "format": "HWC",
      "dtype": "float32",
      "shape": [H, W, 3],
      "input": [[[r,g,b], ...], ...],
      "expected_output": [[[r,g,b], ...], ...],
      "notes": "short description"
    }}
  ]
}}

Rules for test_cases:
- Use exactly one image per test case.
- Each image must be 3D (height, width, channels).
- Use simple 2x2 or 3x3 examples for clarity.
- Clamp values above 1.0 to 1.0.
- Use RGB values like 0.0, 0.25, 0.5, 0.75, 1.0.
- Ensure your JSON parses with Python json.loads() without errors.

All values in the test cases must be clamped to 1.0
"""


    validator = HalideValidatorLoop(generator=refined_pipeline, max_attempts=3)

    results = react_loop_with_code_feedback(
        base_prompt=base_prompt,
        pipeline=refined_pipeline,
        validator=validator,
        max_rounds=5
    )

    print("\nFinal Results:\n", json.dumps(results, indent=2, default=_safe_json))


# ======================================================
# 7. Entry Point
# ======================================================

if __name__ == "__main__":
    main()
