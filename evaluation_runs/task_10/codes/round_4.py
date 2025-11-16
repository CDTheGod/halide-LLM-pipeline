import halide as hl

# Define the input buffer
input_buffer = hl.Buffer(hl.UInt(8), 256, 256, 3)

# Define the output buffer
output_buffer = hl.Buffer(hl.UInt(8), 256, 256, 3)

# Define the color inversion transformation
def invert_color(x):
    r = x[0]
    g = x[1]
    b = x[2]
    return [255 - r, 255 - g, 255 - b]

# Apply the transformation to the input buffer
output_buffer = hl.Func("invert_color")(input_buffer)

# Compile and run the Halide code
hl.compile_to_c(output_buffer)