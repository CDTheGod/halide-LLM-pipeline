import dspy

def get_model():
    """Configure and return the DSPy LM for Halide code generation."""
    lm = dspy.LM(
        "openai/llama3.1:8b",
        api_base="http://172.27.21.160:11434/v1",
        api_key="ollama",
        model_type="chat",
        temperature=0.2,
        max_tokens=10000
    )
    dspy.configure(lm=lm)
    return lm
