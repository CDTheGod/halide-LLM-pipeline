import halide as hl

# Define the input image
img = hl.ImageParam(hl.float32, 3)

# Define the output function
output = hl.Func('output')

# Compute the maximum of each pixel's color channels
output[hx, hy] = hl.max(img[x, y, 0], img[x, y, 1], img[x, y, 2])

# Generate Halide code for the computation graph
hl.codegen(output)

# Compile and run the generated code
compiled_output = hl.compile_and_run(output)