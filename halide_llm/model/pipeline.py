import dspy
import json
from dspy.teleprompt import BootstrapFewShot
from model.config import get_model

get_model()

class HalideCodeGen(dspy.Signature):
    """Signature: Generate Halide code + test cases."""
    prompt = dspy.InputField(desc="User request describing the image operation.")
    thoughts = dspy.OutputField(desc="Step-by-step reasoning.")
    halide_code = dspy.OutputField(format="code", desc="Full runnable Python Halide code.")
    test_cases = dspy.OutputField(format="json", desc="JSON array of test cases verifying correctness.")


class HalidePipeline(dspy.Module):
    """Halide LLM Pipeline with reference-based reasoning."""
    def __init__(self):
        super().__init__()
        self.gen = dspy.ChainOfThought(HalideCodeGen)
        with open("examples/halide_examples.json", "r") as f:
            self.examples = json.load(f)

    def forward(self, full_prompt=None, user_input=None, feedback=None, prev_thoughts=None, prompt=None):
        user_input = user_input or prompt or ""
        full_prompt = full_prompt or user_input

        best_match = max(
            self.examples,
            key=lambda ex: len(set(user_input.lower().split()) & set(ex["prompt"].lower().split()))
        )

        ref = f"""
        {full_prompt}

        Reference Example:
        Prompt: {best_match['prompt']}
        Halide Code:
        {best_match['halide_code']}
        Test Cases:
        {best_match['test_cases']}

        Instructions:
        1. Think step by step and adapt this example for the new prompt.
        2. Always output runnable Python Halide code and JSON test cases.
        3. Keep code valid (no C++ types, no markdown fences).
        4. Use imports:
           import halide as hl
           import imageio
           import numpy as np
        """

        if feedback:
            ref += f"\nPrevious Feedback:\n{feedback}\n"
        if prev_thoughts:
            ref += f"\nPrevious Reasoning:\n{prev_thoughts}\n"

        result = self.gen(prompt=ref)

        print("\n================= REASONING DEBUG =================")
        print(result.thoughts)
        print("===================================================\n")

        return result


def get_pipeline():
    """Bootstrap the Halide pipeline using few-shot examples."""
    with open("examples/halide_examples.json") as f:
        raw_examples = json.load(f)

    reference_examples = [
        dspy.Example(
            prompt=ex["prompt"],
            halide_code=ex["halide_code"],
            test_cases=json.dumps(ex["test_cases"])
        ).with_inputs("prompt")
        for ex in raw_examples
    ]

    tele = BootstrapFewShot(metric=None)
    return tele.compile(HalidePipeline(), trainset=reference_examples)

# bootstrap pipeline at import time for convenience
refined_pipeline = get_pipeline()
