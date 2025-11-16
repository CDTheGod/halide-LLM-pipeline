import halide as hl

# Define the input and output image types
input_image = hl.image(hl.float32)
output_image = hl.image(hl.float32)

# Define the size of the kernel (filter) and the radius of the filter
kernel_size = 5
radius = 2

# Apply a Gaussian blur filter to the input image
def gaussian_blur(input):
    return halide_filter_gaussian(input, kernel_size, radius)

# Create a Halide pipeline that applies the Gaussian blur filter
pipe = hl.Pipeline()
input_var = pipe.input(hl.float32, "input")
output_var = pipe.output(hl.float32, "output")

# Define the function to apply the Gaussian blur filter
@hl.func
def gaussian_blur_func(input):
    return halide_filter_gaussian(input, kernel_size, radius)

# Create a Halide generator that applies the Gaussian blur filter
gen = hl.Generator("gaussian_blur")
gen.add_variable("input", input_var)
gen.add_variable("output", output_var)
gen.add_function(gaussian_blur_func)

# Compile and run the pipeline
pipe.compile_jit()