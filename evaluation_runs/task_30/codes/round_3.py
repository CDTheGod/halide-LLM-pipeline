import halide as hl

# Define the input and output images
input_image = hl.ImageParam(hl.float32, 3)
output_image = hl.Image(output_image.ctype(), output_image.dim(0).cast_to_int(),
                        output_image.dim(1).cast_to_int())

# Apply blur operation using convolve function
blur_kernel = hl.Func("blur_kernel")
blur_kernel[hx, hy] = (hl.sin(hx) + hl.cos(hy)) / 16.0

output_image = hl.Convolve(blur_kernel)(input_image)

# Define the pipeline
pipe = hl.Pipeline(output_image)