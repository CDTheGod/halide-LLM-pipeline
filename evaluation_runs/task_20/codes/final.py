import halide as hl

# Define the input buffer
buf = hl.Buffer(hl.float32, [3, 256, 256])

# Split the buffer into separate planes for each color channel
r = buf.split(0)
g = r.split(1)
b = g.split(2)

# Extract the color channel values
red = b.project(hl.Range(), hl.DimTag.y, hl.DimTag.x)
green = b.project(hl.Range(), hl.DimTag.y, hl.DimTag.x)
blue = b.project(hl.Range(), hl.DimTag.y, hl.DimTag.x)

# Define the output buffers
out_red = hl.Buffer(hl.float32, [256, 256])
out_green = hl.Buffer(hl.float32, [256, 256])
out_blue = hl.Buffer(hl.float32, [256, 256])

# Compute the color channel values
hl.Func('compute').add_input(buf).add_output(out_red).add_output(out_green).add_output(out_blue)