import halide as hl

# Define the input image as a Halide buffer
input_image = hl.Buffer(hl.UInt(8), [1024, 1024, 3])

# Apply Gaussian blur filter to each channel
def gaussian_blur(input):
    # Create a kernel for the Gaussian blur filter
    kernel = hl.Function('kernel', hl.Vector(2), input)
    kernel = hl.Convolve(kernel, hl.GaussianBlur(5))
    
    # Apply the kernel to each channel of the input image
    output = hl.For('x', 0, input.shape[0], hl.For('y', 0, input.shape[1]))
        .For('c', 0, input.shape[2])
        .Output(hl.Select(input[x, y, c], kernel(x, y)))
    
    return output

# Define the Halide pipeline
pipeline = gaussian_blur(input_image)

# Compile and run the pipeline
hl.compile_pipeline(pipeline)