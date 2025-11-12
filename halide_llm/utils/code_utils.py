import re

def strip_before_first_import(code: str) -> str:
    """Removes any text before the first valid import statement."""
    if not isinstance(code, str):
        return code
    lines = code.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("import "):
            return "\n".join(lines[i:]).strip()
    return code.strip()


def extract_halide_code(text):
    """Extract halide_code block and clean it."""
    if not isinstance(text, str):
        return text

    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*", "", text)
    text = text.replace("```", "")

    match = re.search(r'"halide_code"\s*:\s*"([\s\S]*?)"\s*,\s*"?test_cases"?', text)
    if match:
        code = match.group(1)
        code = code.encode("utf-8").decode("unicode_escape")
        return strip_before_first_import(code.strip())

    return strip_before_first_import(text.strip())
