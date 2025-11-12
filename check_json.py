import json
import re
from pathlib import Path

FILE = Path("halide_examples.json")
backup = FILE.with_suffix(".bak.json")

print(f"🔍 Reading: {FILE}")
raw = FILE.read_text(encoding="utf-8")

# --- 1️⃣ Create a backup just in case ---
backup.write_text(raw, encoding="utf-8")
print(f"💾 Backup saved as: {backup}")

# --- 2️⃣ Remove trailing commas before ] or } ---
fixed = re.sub(r",\s*(?=[}\]])", "", raw)

# --- 3️⃣ Remove stray JSON fragments after full array ---
fixed = re.sub(r"\]\s*\]\s*$", "]]", fixed)  # double closing bracket at EOF
fixed = re.sub(r"\}\s*\}\s*$", "}}", fixed)

# --- 4️⃣ Try to reformat compactly ---
try:
    data = json.loads(fixed)
except json.JSONDecodeError as e:
    print(f"❌ Still invalid JSON: {e}")
    lines = fixed.splitlines()
    start = max(0, e.lineno - 5)
    end = min(len(lines), e.lineno + 5)
    print("\n--- Context ---")
    for i in range(start, end):
        print(f"{i+1:03d}: {lines[i]}")
    exit(1)

# --- 5️⃣ Normalize test case formatting ---
for ex in data:
    if "test_cases" in ex:
        for tc in ex["test_cases"]:
            # ensure clean numeric rounding
            if "input" in tc:
                tc["input"] = json.loads(json.dumps(tc["input"], separators=(",", ":")))
            if "expected_output" in tc:
                tc["expected_output"] = json.loads(json.dumps(tc["expected_output"], separators=(",", ":")))

# --- 6️⃣ Write clean file ---
FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
print("✅ JSON cleaned and reformatted successfully!")
print(f"🧩 {len(data)} example(s) cleaned and saved.")
