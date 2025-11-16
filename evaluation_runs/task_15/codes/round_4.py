import halide as hl

# Define the input image
input_image = hl.Buffer(hl.UInt(8), [256, 256, 3])

# Define a function that performs the convolution operation
def convolve(input):
    # Create a new buffer to store the output
    output = hl.Buffer(hl.UInt(8), [256, 256, 3])
    
    # Perform the convolution operation using the Convolution operator
    kernel = hl.Const([0.1, 0.2, 0.1], hl.Float32)
    convolve = hl.Convolution(input, kernel, hl.BorderMode.halide_constant(0))
    
    # Store the output in the new buffer
    output = convolve
    
    return output

# Define the Halide pipeline
pipeline = hl.Pipeline()
input_buffer = pipeline.input(hl.UInt(8), "input", [256, 256, 3])
output_buffer = convolve(input_buffer)
pipeline.return_(output_buffer)

# Compile and run the pipeline
hl.compile_pipeline(pipeline)