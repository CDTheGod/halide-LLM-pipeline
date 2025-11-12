import os, json, numpy as np
from utils.code_utils import extract_halide_code
from utils.json_utils import _safe_json

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