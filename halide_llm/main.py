from model.config import get_model
from model.pipeline import get_pipeline
from validator.validator_loop import HalideValidatorLoop
from react_loop import react_loop_with_code_feedback

def main():
    print("\n🧠 Halide Code Generator + Validator")
    user_input = input("Enter operation prompt: ").strip()
    if not user_input:
        print("No prompt entered. Exiting.")
        return

    get_model()
    pipeline = get_pipeline()
    validator = HalideValidatorLoop(generator=pipeline, max_attempts=3)

    from prompts.full_prompt import build_prompt
    base_prompt = build_prompt(user_input)

    react_loop_with_code_feedback(base_prompt, pipeline, validator, max_rounds=5)

if __name__ == "__main__":
    main()
