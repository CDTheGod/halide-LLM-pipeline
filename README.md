# 📦 Halide-LLM Pipeline
*A modular, test-driven code generation system for producing correct Halide image-processing pipelines using LLMs.*

This project integrates **Halide**, **NumPy**, **imageio**, and **DSPy** to build a robust framework that can:

- Generate Halide pipelines from natural-language prompts  
- Produce runnable Python/Halide code  
- Auto-generate multiple JSON test cases  
- Validate the generated code using image-based comparisons  
- Run iterative feedback loops (ReAct-style)  
- Log all outputs, failures, and intermediate rounds  

The goal is to create a **fully automated Halide code-gen system** capable of generalizing across a wide range of image transformations.

---

## 🚀 Features

### ✅ Natural-language → Halide code  
Give the system a description (e.g., “Apply a 3×3 box blur”), and it outputs runnable Python Halide code.

### ✅ Automatic Test Case Generation  
The LLM also emits a **JSON array of HWC test cases**, used by the validator.

### ✅ Image-based Validation  
Each test case is executed by a Python validator which:

- Writes inputs as PNG  
- Runs the generated Halide pipeline  
- Loads expected outputs  
- Computes differences  
- Reports pass/fail  

### ✅ Multi-Round ReAct Loop  
Wrong outputs cause automatic retries with feedback.

### ✅ Example-Guided Reasoning  
The LLM is conditioned on a set of curated reference examples (`halide_examples_clean.json`).

### ✅ No DSPy Bootstrapping  
The pipeline uses **direct example retrieval**, avoiding DSPy training steps.

---

## 📂 Project Structure

halide_llm/
│
├── eval/
│ ├── run_evaluation.py # Main evaluation runner
│ ├── report_generator.py # Creates HTML reports
│ └── react_loop.py # ReAct feedback loops
│
├── validator/
│ ├── validator_loop.py # Executes Halide code + checks outputs
│ └── helpers.py # Utility routines
│
├── examples/
│ └── halide_examples_clean.json # Curated reference examples
│
├── model/
│ └── config.py # Model configuration (OpenAI, Anthropic, etc.)
│
├── pipeline.py # HalidePipeline + LLM interaction logic
└── README.md # Documentation

text

## 🛠 Installation

### **1. Clone the repo**
```bash
git clone https://github.com/yourusername/halide-llm.git
cd halide-llm
```

### **2. Install dependencies**
Requires Python ≥ 3.10:

```bash
pip install -r requirements.txt
```

### **3. Install Halide**
You must install the Python bindings for Halide.

Simplest:

```bash
pip install halide
```
Or build from source:
https://halide-lang.org

### **4. Running the Evaluation**
To run tasks 0 → 5:

```bash
python -m eval.run_evaluation --start 0 --end 5
```
This will:

Load a task prompt

Retrieve the best matching reference example

Generate Halide code + JSON test cases

Validate by running Halide

Retry if wrong (ReAct feedback loop)

Save outputs under evaluation_runs/

Generate an HTML report per task

All outputs and logs go into:

text
evaluation_runs/
task_#/runs/

### **🧪 Test Case Format**
Each generated test case looks like:

json
{
  "format": "HWC",
  "dtype": "float32", 
  "shape": [32, 32, 3],
  "input": [...],
  "expected_output": [...],
  "notes": "Optional description"
}
Arrays contain raw float values in [0.0, 1.0]

Shape must match array dimensions

Validator saves them as PNG

🧠 How the LLM Works
Input to the LLM:

User prompt

Retrieved reference example

Short instruction block

Optional validator feedback

Output from the LLM:

python
thoughts: step-by-step reasoning
halide_code: runnable Python Halide program
test_cases: JSON array (validated by parser)
🐛 Debugging Tools
Debug printouts include:

Reasoning (step-by-step thoughts)

Raw code from model

JSON extraction logs

PNG I/O logs

Validator diffs

Failures never silently stop — they continue until retries end.

📜 License
MIT License