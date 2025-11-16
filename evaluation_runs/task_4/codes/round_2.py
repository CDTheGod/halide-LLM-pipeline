import halide as hl

# Define the input buffer
buf = hl.Buffer(hl.UInt(8), [16, 256, 256])

# Flatten the buffer into a 1D array
flattened_buf = buf.flatten()

# Compute the sum of all elements in the flattened buffer
sum_result = flattened_buf.sum()

# Define the output buffer
output_buf = hl.Buffer(hl.UInt(32), [1])

# Compute the final result by storing the sum in the output buffer
hl.Store(output_buf, sum_result)