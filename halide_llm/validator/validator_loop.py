import os,imageio, json, subprocess, numpy as np, base64, webbrowser
from utils.io_utils import write_image, load_image
from utils.json_utils import _clean_json_array_block, _safe_json

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




    def _validate_multiple_cases(self, halide_code: str, test_cases_json: str, operation_name="operation", base_output_dir: str = None):
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

        all_results = []
        root = base_output_dir if base_output_dir else os.getcwd()
        base_run_dir = os.path.join(root, "runs", operation_name.replace(" ", "_"))
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

            if score >= 0.88:
                print(f"✅ Test #{idx+1} passed! (saved to {case_dir})")
            else:
                print(f"❌ Test #{idx+1} failed (saved to {case_dir})")
                print(f"   max diff: {info['max_diff']}, mean diff: {info['mean_diff']}")

        syntax_ok = all("error" not in r for r in all_results)
                # 🧭 Generate HTML report for visual inspection
        # try:
        #     report_path = self._generate_html_gallery(operation_name, all_results)
        # except Exception as e:
        #     print(f"⚠️ Failed to generate HTML report: {e}")
        #     report_path = None
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
