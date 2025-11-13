import os
import json
import shutil
from model.config import get_model
from model.pipeline import refined_pipeline
from validator.validator_loop import HalideValidatorLoop
from utils.code_utils import extract_halide_code
from react_loop import react_loop_with_code_feedback

# ======================================================
# CONFIGURATION
# ======================================================

get_model()

# Project root (one level above halide_llm)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUTPUT_ROOT = os.path.join(ROOT_DIR, "evaluation_runs")
TASKS_PATH = os.path.join(ROOT_DIR, "data", "tasks.json")

MAX_ROUNDS = 5

# ======================================================
# HELPERS
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

def save_code(path, code):
    with open(path, "w", encoding="utf-8") as f:
        f.write(extract_halide_code(code or ""))

# ======================================================
# MAIN EVALUATION LOOP
# ======================================================

def run_batch(start_idx=0, end_idx=1):
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

        # pass operation_name and base_output_dir to the react_loop
        results = react_loop_with_code_feedback(
            base_prompt=prompt,
            pipeline=refined_pipeline,
            validator=validator,
            max_rounds=MAX_ROUNDS,
            operation_name=f"task_{task_id}",
            base_output_dir=task_dir  # ✅ ensures validator writes inside evaluation_runs/task_<id>
        )


        # Save per-round data
        per_round_data = []
        for i, round_result in enumerate(results):
            round_dir = os.path.join(runs_dir, f"round_{i+1}")
            os.makedirs(round_dir, exist_ok=True)

            # Save halide code
            halide_code_content = round_result.get("_model_halide_code") or round_result.get("halide_code", "")
            halide_path = os.path.join(codes_dir, f"round_{i+1}.py")
            save_code(halide_path, halide_code_content)

            # Save round results
            with open(os.path.join(round_dir, "results.json"), "w", encoding="utf-8") as f:
                json.dump(round_result, f, indent=2)

            # Copy case folders from validator output into round_dir
            for r in round_result.get("results", []):
                case_path = r.get("path")
                if case_path and os.path.exists(case_path):
                    dest = os.path.join(round_dir, f"case_{r['idx']+1}")
                    if not os.path.exists(dest):
                        shutil.copytree(case_path, dest)

            per_round_data.append({
                "round": i + 1,
                "syntax_ok": round_result.get("syntax_ok", False),
                "num_tests": len(round_result.get("results", [])),
                "num_passed": sum(
                    1 for t in round_result.get("results", [])
                    if t.get("correctness_index", 0) >= 0.99
                )
            })

        # Pick best round
        best_round = None
        if per_round_data:
            best_round = max(
                per_round_data,
                key=lambda x: (x["num_passed"], -x["round"])
            )

            # Copy final.py
            src_code = os.path.join(codes_dir, f"round_{best_round['round']}.py")
            dst_code = os.path.join(codes_dir, "final.py")
            shutil.copy(src_code, dst_code)

        # Write summary
        summary_entry = {
            "task_id": task_id,
            "description": prompt,
            "rounds_run": len(results),
            "best_round": best_round["round"] if best_round else None,
            "best_passed": best_round["num_passed"] if best_round else 0,
            "best_total": best_round["num_tests"] if best_round else 0,
            "status": "pass" if best_round and best_round["num_passed"] > 0 else "fail"
        }

        with open(os.path.join(task_dir, "task_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_entry, f, indent=2)

        results_summary.append(summary_entry)

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
