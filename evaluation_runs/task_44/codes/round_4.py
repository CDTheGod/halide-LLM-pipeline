import halide as hl

# Define the input image
img = hl.ImageParam(hl.float32, [3, 256, 256])

# Define a simple kernel (a small matrix)
kernel = hl.Func('kernel')
kernel[hx, hy] = hl.Select([hx == 0, hx == 1], [hl.float32(2), hl.float32(-1)])

# Perform the convolution
convolved_img = hl.Func('convolved_img')
convolved_img[hx, hy, c] = hl.RangeReduce(hl.Min, 0, 3,
    lambda x: kernel[x[0] - 1, x[1] - 1] * img[x[0], x[1], c])

# Define the output image
output = convolved_img

# Compile and run the code
hl.compile_to_c(output)