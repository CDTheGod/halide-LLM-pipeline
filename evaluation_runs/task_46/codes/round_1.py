import halide as hl

# Define the input function
input_func = hl.Func('input')
input_func[hhl.Range(0, 10), hhl.Range(0, 10), hhl.Range(0, 10)] = hl.Var('x') + hl.Var('y') + hl.Var('z')

# Define the output function
output_func = hl.Func('output')
output_func[hhl.Range(0, 10), hhl.Range(0, 10), hhl.Range(0, 10)] = input_func * input_func

# Compute the output
output_func.compute_root()

# Print the Halide code
print(hl.codegen(output_func))