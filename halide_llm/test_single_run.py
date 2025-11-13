from model.config import get_model
lm = get_model() 
from model.pipeline import refined_pipeline
from validator.validator_loop import HalideValidatorLoop
from react_loop import react_loop_with_code_feedback

# Example image operation to test
prompt = "Convert the input RGB image to grayscale"

# Initialize validator
validator = HalideValidatorLoop(generator=refined_pipeline, max_attempts=3)

# Run one test round to verify everything works
results = react_loop_with_code_feedback(
    base_prompt=prompt,
    pipeline=refined_pipeline,
    validator=validator,
    max_rounds=3
)
