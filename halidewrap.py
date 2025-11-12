import dspy
import halide as hl
import numpy as np
import cv2
import json


# Use a strong base model (can be GPT, Llama, etc.)
lm = dspy.LM(
    "openai/llama3.1:8b",  # model name is arbitrary but should match your endpoint
    api_base="http://172.27.21.160:11434/v1",  # your local server
    api_key="ollama",  # if your server requires this token
    model_type="chat"  # use 'chat' for instruct-style models
)

dspy.configure(lm=lm)
# Load reference examples from JSON
with open("halide_examples.json") as f:
    raw_examples = json.load(f)

reference_examples = [dspy.Example(**ex) for ex in raw_examples]


class HalideCodeGen(dspy.Signature):
    prompt = dspy.InputField(desc="Natural language description of the image processing task")
    halide_code = dspy.OutputField(desc="Write a full Halide code in python with imports, buffer setup, and realization, that reads 'input.png' and writes 'output.png'")
    test_case = dspy.OutputField(
        desc=(
            "Return a structured JSON (NOT runnable python) describing a test case. "
            "JSON must contain keys: 'format', 'dtype', 'shape', 'input', 'expected_output', 'notes'. "
            "Use small numeric values (e.g. 0.0, 0.25, 0.5, 1.0, 2.0, 5.0). "
            "Example: {\"format\":\"HWC\",\"dtype\":\"float32\",\"shape\":[3,3,3],\"input\":...,\"expected_output\":...}"
        )
    )


class HalidePipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.gen = dspy.Predict(HalideCodeGen)

    def forward(self, prompt):
        return self.gen(prompt=prompt)
# Initialize DSPy with your model (e.g., OpenAI, Together, etc.)
# Example: dspy.settings.configure(openai_api_key="...", model="gpt-4")
from dspy.teleprompt import BootstrapFewShot

tele = BootstrapFewShot(metric=None)  # You can plug in halide_feedback later
refined_pipeline = tele.compile(HalidePipeline(), trainset=reference_examples)

import subprocess, tempfile, os, numpy as np, imageio
from typing import List

import subprocess, tempfile, os, numpy as np, imageio

class HalideValidatorLoop:
    def __init__(self, generator, max_attempts=10):
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

    def _load_image(self, path):
        return imageio.imread(path).astype(np.float32) / 255.0

    def _compare_pixels(self, actual, expected):
        if actual.shape != expected.shape:
            return 0.0
        error = np.linalg.norm(actual - expected)
        return max(0.0, 1.0 - error / 10.0)

    def _validate_code(self, halide_code: str, test_case_code: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Run test case to generate input.png and expected_output.png
            ok, err = self._run_python(test_case_code, tmpdir, "test_case.py")
            if not ok:
                return {"syntax_ok": False, "error": f"Test case failed: {err}", "correctness_index": 0.0}

            # Step 2: Run Halide code to generate actual output
            ok, err = self._run_python(halide_code, tmpdir, "halide_code.py")
            if not ok:
                return {"syntax_ok": False, "error": f"Halide code failed: {err}", "correctness_index": 0.0}

            try:
                actual = self._load_image(os.path.join(tmpdir, "output.png"))
                expected = self._load_image(os.path.join(tmpdir, "expected_output.png"))
                score = self._compare_pixels(actual, expected)
                return {
                    "syntax_ok": True,
                    "error": None,
                    "correctness_index": round(score, 3),
                    "expected_pixel": expected[1, 1].tolist(),
                    "actual_pixel": actual[1, 1].tolist()
                }
            except Exception as e:
                return {"syntax_ok": True, "error": f"Pixel comparison failed: {str(e)}", "correctness_index": 0.0}

    def run(self, prompt: str):
        results = []
        for i in range(self.max_attempts):
            print(f"\n🔁 Attempt {i+1}")
            result = self.generator(prompt)
            halide_code = result.halide_code
            test_case_code = result.test_case

            print("📄 Halide Code:\n", halide_code)
            print("\n🧪 Test Case Code:\n", test_case_code)

            validation = self._validate_code(halide_code, test_case_code)

            print("✅ Syntax OK:", validation["syntax_ok"])
            print("🧪 Correctness Index:", validation["correctness_index"])
            print("🔍 Expected Pixel:", validation.get("expected_pixel"))
            print("🔍 Actual Pixel:", validation.get("actual_pixel"))
            if validation["error"]:
                print("❌ Error:", validation["error"])

            results.append({
                "attempt": i+1,
                "halide_code": halide_code,
                "test_case": test_case_code,
                "syntax_ok": validation["syntax_ok"],
                "correctness_index": validation["correctness_index"],
                "error": validation["error"],
                "expected_pixel": validation.get("expected_pixel"),
                "actual_pixel": validation.get("actual_pixel")
            })

            if validation["correctness_index"] == 1.0:
                print("🎯 Perfect match achieved. Stopping early.")
                break

        return results
    

def react_loop_with_code_feedback(base_prompt, pipeline, validator, max_rounds=10):
    last_error = None
    last_halide_code = None
    last_test_case_code = None
    results = []

    for i in range(max_rounds):
        print(f"\n🔁 Round {i+1}")

        # Build prompt with full feedback
        prompt = base_prompt
        if last_halide_code:
            prompt += f"\n\nPrevious Halide code:\n{last_halide_code}"
        if last_test_case_code:
            prompt += f"\n\nPrevious test case code:\n{last_test_case_code}"
        if last_error:
            prompt += f"\n\nError encountered:\n{last_error}\nPlease revise both codes to fix this."

        # Generate new attempt
        result = pipeline(prompt)
        halide_code = result.halide_code
        test_case_code = result.test_case

        print("📄 Halide Code:\n", halide_code)
        print("\n🧪 Test Case Code:\n", test_case_code)
        print(f"\n🧠 Injected Prompt for Round {i+1}:\n{prompt}")

        # Validate
        validation = validator._validate_code(halide_code, test_case_code)

        print("✅ Syntax OK:", validation["syntax_ok"])
        print("🧪 Correctness Index:", validation["correctness_index"])
        print("🔍 Expected Pixel:", validation.get("expected_pixel"))
        print("🔍 Actual Pixel:", validation.get("actual_pixel"))
        if validation["error"]:
            print("❌ Error:", validation["error"])

        results.append({
            "round": i+1,
            "halide_code": halide_code,
            "test_case": test_case_code,
            "syntax_ok": validation["syntax_ok"],
            "correctness_index": validation["correctness_index"],
            "error": validation["error"],
            "expected_pixel": validation.get("expected_pixel"),
            "actual_pixel": validation.get("actual_pixel")
        })

        # Stop if perfect
        if validation["correctness_index"] == 1.0:
            print("🎯 Perfect match achieved. Stopping early.")
            break

        # Update feedback for next round
        last_error = validation["error"]
        last_halide_code = halide_code
        last_test_case_code = test_case_code

    return results

pipeline = refined_pipeline  # from BootstrapFewShot or manual setup
print("Invoking the validator loop")
validator = HalideValidatorLoop(generator=pipeline, max_attempts=1)

results = react_loop_with_code_feedback(
    base_prompt="Write a halide program in python to Brighten an image by 2x. Output should be saved as output.png",
    pipeline=pipeline,
    validator=validator,
    max_rounds=10
)