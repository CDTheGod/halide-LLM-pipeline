import os
import dspy
import halide as hl
import numpy as np
import subprocess
import tempfile
import json, re, ast
import base64, webbrowser
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
    max_tokens=10000
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
        """

        if user_input is None and prompt is not None:
            user_input = prompt
        if full_prompt is None:
            full_prompt = user_input or prompt or ""

        user_input = user_input or ""
        full_prompt = full_prompt or ""

        # 🧠 Find best match example for adaptation
        best_match = max(
            self.examples,
            key=lambda ex: len(set(user_input.lower().split()) & set(ex["prompt"].lower().split()))
        )

        # 🧩 Construct reasoning reference
        ref = f"""
        {full_prompt}

        Reference Example:
        Prompt: {best_match['prompt']}
        Halide Code:
        {best_match['halide_code']}
        Test Cases:
        {best_match['test_cases']}

        Instructions:
        1. Think step by step about how this reference relates to the new prompt.
        2. Adapt it carefully while preserving correct image I/O, transposition, and data types.
        3. Generate **6–8 test cases** total.
        - 3–4 small analytical ones (3×3 or 5×5).
        - 2 large synthetic ones (≥32×32).
        4. For large ones, describe input patterns (e.g. gradient, noise) but ensure JSON stays valid.
        5. Always clamp values between 0.0 and 1.0.
        6. Always include the code and JSON **without markdown fences**.
        7. The Halide code **must be valid Python**, not C++ (no 'int main', no ';', no types).
        8. Never use `f` suffixes like `1.0f`; only plain floats.
        9. Imports must be exactly:
        import halide as hl
        import imageio
        import numpy as np

        csharp
        Copy code
        10. Always include `input.png` / `output.png` for I/O.
        11. Keep structure identical to reference; just change computation logic.
        12. If feedback is provided, correct your previous mistakes accordingly.

        Now, reason carefully step-by-step.
        Then output:
        1. Your reasoning as plain text.
        2. The final Halide code (runnable Python).
        3. A JSON array of test cases (input + expected output).
        """

        if feedback:
            ref += f"\nPrevious Feedback:\n{feedback}\n"
        if prev_thoughts:
            ref += f"\nPrevious Reasoning (for self-reflection):\n{prev_thoughts}\n"

        # Run generation
        result = self.gen(prompt=ref)

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

def strip_before_first_import(code: str) -> str:
    """
    Removes all text before the first valid Python 'import' statement.
    This helps clean up LLM outputs that contain reasoning or markdown.
    """
    if not isinstance(code, str):
        return code

    lines = code.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("import "):
            return "\n".join(lines[i:]).strip()
    return code.strip()

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
        return strip_before_first_import(code.strip())

    # 🧩 Fallback: if not JSON-wrapped, return the cleaned text block
    return strip_before_first_import(text.strip())


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

def sanitize_json_like(text: str) -> str:
    """Remove junk, comments, ellipses, and np.random from JSON-like model output."""
    text = text.strip()
    text = re.sub(r"^```(json|python)?", "", text)
    text = re.sub(r"```$", "", text)
    text = re.sub(r"#.*", "", text)
    text = re.sub(r"\.\.\.", "0.0", text)
    text = re.sub(r"np\.random\.[A-Za-z_]+\([^\)]*\)", "0.5", text)
    text = re.sub(r",(\s*[\]\}])", r"\1", text)  # remove trailing commas
    return text

def _clean_json_array_block(raw_text: str):
    """
    Extracts and cleans a JSON array of test cases from model output.
    Handles code fences, Python-style dicts, stray text, and bracket mismatches.
    """
    if not isinstance(raw_text, str):
        return raw_text
    
    text = sanitize_json_like(raw_text) 
    text = text.strip()
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

        # If shapes don't match, return numeric placeholders but log the error
        if actual.shape != expected.shape:
            return 0.0, {
                "error": f"shape mismatch {actual.shape} vs {expected.shape}",
                "max_diff": 1.0,
                "mean_diff": 1.0
            }

        diff = np.abs(actual - expected)
        max_diff = float(diff.max()) if diff.size > 0 else 0.0
        mean_diff = float(diff.mean()) if diff.size > 0 else 0.0

        passed = np.allclose(actual, expected, atol=atol)
        score = 1.0 if passed else max(0.0, 1.0 - mean_diff * 10)

        return score, {"max_diff": max_diff, "mean_diff": mean_diff}




    def _validate_multiple_cases(self, halide_code: str, test_cases_json: str, operation_name="operation"):
        print("\n================= JSON Parsing Debug =================")
        print(test_cases_json)
        print("=====================================================\n")

        try:
            parsed = _clean_json_array_block(test_cases_json)
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

        # 🔧 Augment with large tests if missing
        def expand_small_test_cases(cases):
            augmented = list(cases)
            if not any(np.prod(c["shape"]) >= 1024 for c in cases):
                h, w = 32, 32
                y, x = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
                grad = np.stack([x, y, 0.5 * np.ones_like(x)], axis=-1).astype(np.float32)
                inv_grad = 1.0 - grad
                augmented.append({
                    "format": "HWC",
                    "dtype": "float32",
                    "shape": [h, w, 3],
                    "input": grad.tolist(),
                    "expected_output": inv_grad.tolist(),
                    "notes": "Synthetic 32×32 gradient test for realism"
                })
            return augmented

        cases = expand_small_test_cases(cases)
        cases = cases[0:3]
        all_results = []

        base_run_dir = os.path.join(os.getcwd(), "runs", operation_name.replace(" ", "_"))
        os.makedirs(base_run_dir, exist_ok=True)

        for idx, case in enumerate(cases):
            if not isinstance(case, dict):
                continue

            case_dir = os.path.join(base_run_dir, f"case_{idx+1}")
            os.makedirs(case_dir, exist_ok=True)

            print(f"\nRunning Test Case #{idx+1}: {case.get('notes', '(no notes)')}")
            try:
                input_arr = np.array(case["input"], dtype=case.get("dtype", "float32"))
                expected_arr = np.array(case["expected_output"], dtype=case.get("dtype", "float32"))
                expected_arr = np.clip(expected_arr, 0.0, 1.0)
                input_arr = np.clip(input_arr, 0.0, 1.0)

            except Exception as e:
                all_results.append({"idx": idx, "error": f"Array load failed: {e}"})
                continue

            self._write_image(input_arr, os.path.join(case_dir, "input.png"))
            self._write_image(expected_arr, os.path.join(case_dir, "expected_output.png"))

            # Save Halide code for inspection
            with open(os.path.join(case_dir, "halide_code.py"), "w", encoding="utf-8") as f:
                f.write(halide_code)

            ok, err = self._run_python(halide_code, case_dir, "halide_code.py")
            if not ok:
                all_results.append({"idx": idx, "error": f"Halide code failed:\n{err}"})
                continue

            output_path = os.path.join(case_dir, "output.png")
            if not os.path.exists(output_path):
                all_results.append({"idx": idx, "error": "Halide code ran but did not produce output.png"})
                continue

            actual = self._load_image(output_path)
            expected = self._load_image(os.path.join(case_dir, "expected_output.png"))
            actual = np.clip(actual, 0.0, 1.0)          
            expected = np.clip(expected, 0.0, 1.0)      
            if actual.ndim == 2:
                # actual is HxW -> expand to HxWx1 then maybe to 3 channels if expected is RGB
                actual = actual[:, :, np.newaxis]
            if expected.ndim == 2:
                expected = expected[:, :, np.newaxis]

            # If one is single-channel and the other has 3 channels, broadcast single->3
            if actual.ndim == 3 and expected.ndim == 3:
                if actual.shape[2] == 1 and expected.shape[2] == 3:
                    actual = np.repeat(actual, 3, axis=2)
                elif actual.shape[2] == 3 and expected.shape[2] == 1:
                    expected = np.repeat(expected, 3, axis=2)
            score, info = self._compare(actual, expected)

            if score < 1.0 and isinstance(info, dict) and "max_diff" in info:
                if info["max_diff"] < 0.05:
                    score = 1.0  # allow small tolerance


            result = {
                "idx": idx,
                "correctness_index": round(score, 3),
                "max_diff": info["max_diff"],
                "mean_diff": info["mean_diff"],
                "expected_mean": float(np.round(expected.mean(), 3)),
                "actual_mean": float(np.round(actual.mean(), 3)),
                "path": case_dir
            }
            all_results.append(result)

            if score >= 0.99:
                print(f"✅ Test #{idx+1} passed! (saved to {case_dir})")
            else:
                print(f"❌ Test #{idx+1} failed (saved to {case_dir})")
                print(f"   max diff: {info['max_diff']}, mean diff: {info['mean_diff']}")

        syntax_ok = all("error" not in r for r in all_results)
                # 🧭 Generate HTML report for visual inspection
        try:
            report_path = self._generate_html_gallery(operation_name, all_results)
        except Exception as e:
            print(f"⚠️ Failed to generate HTML report: {e}")
            report_path = None
        return {"syntax_ok": syntax_ok, "results": all_results}
    

    def _generate_html_gallery(self, operation_name, all_results):
        """
        Create and open an HTML report showing input, expected, and output images side-by-side
        for each test case.
        """
        base_dir = os.path.join(os.getcwd(), "runs", operation_name.replace(" ", "_"))
        os.makedirs(base_dir, exist_ok=True)
        html_path = os.path.join(base_dir, "report.html")

        def img_to_base64(path):
            if not os.path.exists(path):
                return ""
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        html = [
            "<html><head><meta charset='utf-8'>",
            "<style>",
            "body { font-family: 'Segoe UI', Arial; background: #f8f9fa; color: #222; padding: 20px; }",
            "h1 { color: #283e4a; }",
            ".case { margin: 25px 0; background: #fff; border-radius: 12px; padding: 16px; box-shadow: 0 3px 8px rgba(0,0,0,0.08); }",
            ".images { display: flex; justify-content: space-evenly; flex-wrap: wrap; gap: 20px; }",
            "img { border-radius: 8px; width: 280px; height: auto; border: 1px solid #ccc; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }",
            "h2 { margin-top: 0; }",
            ".meta { color: #555; font-size: 0.9em; margin-bottom: 10px; }",
            "h3 { margin: 6px 0; text-align: center; }",
            "</style>",
            "</head><body>",
            f"<h1>🧠 Halide LLM Validation Report — {operation_name}</h1>"
        ]

        for res in all_results:
            case_dir = res.get("path", "")
            if not os.path.exists(case_dir):
                continue

            idx = res["idx"] + 1
            correctness = res.get("correctness_index", 0)
            mean_diff = res.get("mean_diff", 0)
            max_diff = res.get("max_diff", 0)

            input_img = img_to_base64(os.path.join(case_dir, "input.png"))
            expected_img = img_to_base64(os.path.join(case_dir, "expected_output.png"))
            output_img = img_to_base64(os.path.join(case_dir, "output.png"))

            html.append(f"""
            <div class='case'>
                <h2>Test Case #{idx}</h2>
                <div class='meta'>
                    ✅ <b>Correctness:</b> {correctness:.3f} |
                    📊 <b>Mean diff:</b> {mean_diff:.4f} |
                    ⚖️ <b>Max diff:</b> {max_diff:.4f}
                </div>
                <div class='images'>
                    <div><h3>Input</h3><img src='data:image/png;base64,{input_img}'></div>
                    <div><h3>Expected Output</h3><img src='data:image/png;base64,{expected_img}'></div>
                    <div><h3>Actual Output</h3><img src='data:image/png;base64,{output_img}'></div>
                </div>
            </div>
            """)

        html.append("</body></html>")

        with open(html_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))

        print(f"\n🖼️  Visual report generated at: {html_path}")
        try:
            webbrowser.open(f"file://{html_path}")
            print("🌐 Opened report in browser.")
        except Exception as e:
            print(f"⚠️ Could not auto-open browser: {e}")

        return html_path



# ======================================================
# 5. Feedback + React Loop
# ======================================================

def react_loop_with_code_feedback(base_prompt, pipeline, validator, max_rounds=3):
    last_error = None
    last_halide_code = None
    last_test_cases = None
    results = []
    latest_report_path = None   # ✅ track last successful report

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
        if "report" in validation and validation["report"]:
            latest_report_path = validation["report"]

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
    if results:
        final_round = results[-1]
        report_path = final_round.get("report")
        if report_path and os.path.exists(report_path):
            print(f"\n🌐 Opened visual report in browser:\n{report_path}")
        elif latest_report_path and os.path.exists(latest_report_path):
            print(f"\n🖼️ Using last successful visual report:\n{latest_report_path}")
        else:
            print("\n⚠️ No HTML report generated.")


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
