import os
import json
import shutil
from halide_llm.model.pipeline import refined_pipeline
from halide_llm.validator.validator_loop import HalideValidatorLoop
from halide_llm.utils.code_utils import extract_halide_code
from halide_llm.react_loop import react_loop_with_code_feedback

# ======================================================
# CONFIGURATION
# ======================================================

TASKS_PATH = "data/tasks.json"
OUTPUT_ROOT = "evaluation_runs"
MAX_ROUNDS = 5
TASKS_PER_MACHINE = 10

# ======================================================
# HELPER
# ======================================================

def load_tasks(start_idx=0, end_idx=10):
    with open(TASKS_PATH, "r") as f:
        tasks = json.load(f)
    return tasks[start_idx:end_idx]

def make_dirs_for_task(task_id):
    base_dir = os.path.join(OUTPUT_ROOT, f"task_{task_id}")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "codes"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "runs"), exist_ok=True)
    return base_dir

# ======================================================
# MAIN EVALUATION LOOP
# ======================================================

def run_batch(start_idx=0, end_idx=10):
    tasks = load_tasks(start_idx, end_idx)
    print(f"🧪 Running tasks {start_idx+1}–{end_idx}")

    results_summary = []

    for task in tasks:
        task_id = task["task_id"]
        prompt = task["description"]
        print(f"\n=== 🧩 Task {task_id}: {prompt} ===")

        task_dir = make_dirs_for_task(task_id)
        codes_dir = os.path.join(task_dir, "codes")
        runs_dir = os.path.join(task_dir, "runs")

        validator = HalideValidatorLoop(generator=refined_pipeline, max_attempts=3)

        # Run main loop (up to MAX_ROUNDS)
        results = react_loop_with_code_feedback(
            base_prompt=prompt,
            pipeline=refined_pipeline,
            validator=validator,
            max_rounds=MAX_ROUNDS
        )

        # Collect per-round details
        per_round_data = []
        for i, round_result in enumerate(results):
            round_dir = os.path.join(runs_dir, f"round_{i+1}")
            os.makedirs(round_dir, exist_ok=True)

            # Save halide code
            halide_path = os.path.join(codes_dir, f"round_{i+1}.py")
            with open(halide_path, "w", encoding="utf-8") as f:
                f.write(extract_halide_code(round_result.get("halide_code", "")))

            # Save JSON summary of test results
            json_path = os.path.join(round_dir, "results.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(round_result, f, indent=2)

            # Copy run images (optional)
            for r in round_result.get("results", []):
                if "path" in r and os.path.exists(r["path"]):
                    dest = os.path.join(round_dir, f"test_{r['idx']}")
                    if not os.path.exists(dest):
                        shutil.copytree(r["path"], dest)

            per_round_data.append({
                "round": i + 1,
                "syntax_ok": round_result.get("syntax_ok", False),
                "num_tests": len(round_result.get("results", [])),
                "num_passed": sum(
                    1 for t in round_result.get("results", [])
                    if t.get("correctness_index", 0) >= 0.99
                )
            })

        # Final summary for this task
        final_round = results[-1] if results else {}
        num_passed = sum(
            1 for t in final_round.get("results", [])
            if t.get("correctness_index", 0) >= 0.99
        )

        summary_entry = {
            "task_id": task_id,
            "description": prompt,
            "status": "pass" if num_passed > 0 else "fail",
            "compiled": final_round.get("syntax_ok", False),
            "num_tests_passed": num_passed,
            "num_tests_total": len(final_round.get("results", [])),
            "rounds_run": len(results)
        }
        results_summary.append(summary_entry)

        with open(os.path.join(task_dir, "task_summary.json"), "w") as f:
            json.dump(summary_entry, f, indent=2)

    # Write master summary
    summary_path = os.path.join(OUTPUT_ROOT, f"summary_{start_idx+1}_{end_idx}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n✅ Finished tasks {start_idx+1}–{end_idx}")
    print(f"📄 Summary written to: {summary_path}")


# ======================================================
# ENTRY POINT
# ======================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Halide LLM evaluation batch.")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--end", type=int, default=10, help="End index (non-inclusive)")
    args = parser.parse_args()

    run_batch(args.start, args.end)
