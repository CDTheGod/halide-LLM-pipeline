import halide as hl

# Define the input image
input = hl.ImageParam(hl.float32, [3, 256, 256])

# Resize the input image
resized_input = hl.image_resize(input, 128, 128)

# Blur the resized image
blurred_image = hl.image_blur(resized_input, 2, 5)

# Threshold the blurred image
thresholded_image = hl.image_threshold(blurred_image, 0.5)

# Define the output function
def output(x, y):
    return thresholded_image(x, y)

# Compile and run the code
output.compile_jit().run()