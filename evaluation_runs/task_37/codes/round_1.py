import halide as hl

# Define the input and output buffers
input_buffer = hl.Buffer(hl.float32, [3, 256, 256])
output_buffer = hl.Buffer(hl.float32, [3, 256, 256])

# Create a buffer from the given data
buffer = hl.Buffer(hl.float32, [3, 256, 256], input_data)

# Define a custom function to apply a filter on the buffer
def apply_filter(x, y):
    return (input_buffer(x, y, 0) + input_buffer(x, y, 1) + input_buffer(x, y, 2)) / 3.0

# Apply the filter on the buffer
hl.For('x', 0, 256).store(output_buffer, apply_filter, 'y')

# Define a Halide function to perform the operation
def halide_function():
    hl.For('x', 0, 256).For('y', 0, 256).store(output_buffer, apply_filter, 'x')