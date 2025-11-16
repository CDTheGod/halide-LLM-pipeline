import halide as hl

def generate_halide_code(test_cases):
    halide_code = ""
    for i, test_case in enumerate(test_cases):
        input_array = test_case['input']
        expected_output_array = test_case['expected_output']

        # Generate Halide code for this test case
        halide_code += f"// Test case {i+1}\n"
        halide_code += "Var x('x'), y('y');\n"
        halide_code += "Func input(x, y);\n"
        halide_code += "input(x, y) = {\n"
        for j in range(input_array.shape[0]):
            for k in range(input_array.shape[1]):
                halide_code += f"  {j}, {k}: {input_array[j, k]}\n"
        halide_code += "};\n"

        # Generate Halide code for the expected output
        halide_code += "Func expected_output(x, y);\n"
        halide_code += "expected_output(x, y) = {\n"
        for j in range(expected_output_array.shape[0]):
            for k in range(expected_output_array.shape[1]):
                halide_code += f"  {j}, {k}: {expected_output_array[j, k]}\n"
        halide_code += "};\n"

    return halide_code

halide_code = generate_halide_code(test_cases)
print(halide_code)