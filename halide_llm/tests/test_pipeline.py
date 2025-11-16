# tests/test_pipeline.py
import json
import traceback

from model.pipeline import refined_pipeline

# A simple fake validator-like helper to test JSON correctness only.
REQUIRED_KEYS = {"format", "dtype", "shape", "input", "expected_output", "notes"}

def test_stage_A_code_generation():
    print("\n=== TEST: Stage A — Halide code generation ===")
    op = "Apply a 3x3 box blur to the input image"
    out = refined_pipeline.generate_code(op)

    assert isinstance(out, dict), "Stage A did not return a dict"
    code = out.get("halide_code")
    assert code is not None, "Stage A returned no halide_code"
    assert isinstance(code, str), f"halide_code is not a string: {type(code)}"
    assert "import halide as hl" in code, "Missing required Halide import"
    assert "hl.Func" in code, "Missing Halide Func"

    print("✔ Stage A passed.")


def test_stage_B_test_generation():
    print("\n=== TEST: Stage B — Test case JSON generation ===")
    op = "Apply a 3x3 box blur to the input image"

    # First get code
    code_resp = refined_pipeline.generate_code(op)
    code = code_resp.get("halide_code")

    # Then get test cases
    test_resp = refined_pipeline.generate_tests(op, code)
    tc_raw = test_resp.get("test_cases")

    assert isinstance(tc_raw, str), "test_cases output is not a string"

    # Try parsing JSON
    try:
        tc_list = json.loads(tc_raw)
    except Exception:
        print("\n❌ JSON could not be parsed:")
        print(tc_raw)
        traceback.print_exc()
        raise

    assert isinstance(tc_list, list), "Top-level JSON is not a list"
    assert len(tc_list) == 5, f"Expected 5 test cases, got {len(tc_list)}"

    # Validate structure
    for i, tc in enumerate(tc_list):
        assert isinstance(tc, dict), f"Test case {i} is not an object"
        missing = REQUIRED_KEYS - tc.keys()
        assert not missing, f"Test case {i} missing keys: {missing}"

    print("✔ Stage B passed.")


if __name__ == "__main__":
    print("Running pipeline sanity tests...\n")
    test_stage_A_code_generation()
    test_stage_B_test_generation()
    print("\n🎉 All pipeline tests passed successfully.\n")
