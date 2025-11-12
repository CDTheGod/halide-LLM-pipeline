import json, re, ast, numpy as np

def sanitize_json_like(text: str) -> str:
    """Clean JSON-like text from LLM output."""
    text = text.strip()
    text = re.sub(r"^```(json|python)?", "", text)
    text = re.sub(r"```$", "", text)
    text = re.sub(r"#.*", "", text)
    text = re.sub(r"\.\.\.", "0.0", text)
    text = re.sub(r"np\.random\.[A-Za-z_]+\([^\)]*\)", "0.5", text)
    text = re.sub(r",(\s*[\]\}])", r"\1", text)
    return text


def _clean_json_array_block(raw_text: str):
    """Extract and fix malformed JSON array of test cases."""
    if not isinstance(raw_text, str):
        return raw_text

    text = sanitize_json_like(raw_text)
    text = re.sub(r"^```[a-zA-Z]*", "", text).replace("```", "")

    match = re.search(r"(\[[\s\S]*\])", text)
    if not match:
        raise ValueError("No JSON array found in text.")
    cleaned = match.group(1).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(cleaned)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    repaired = cleaned.replace("'", '"')
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)

    try:
        return json.loads(repaired)
    except Exception as e:
        raise ValueError(f"JSON parsing failed: {e}") from e


def _safe_json(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
