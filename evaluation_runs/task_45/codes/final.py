import halide as hl

# Define the shape of the input image
input_shape = (10, 20, 3)

# Create a Halide pipeline that takes the input image and produces the output image after resizing
def resize_image(input):
    # Define the output shape
    output_shape = (20, 40, 3)
    
    # Create a Halide buffer to store the output image
    output = hl.Buffer(hl.UInt(8), output_shape)
    
    # Perform the resizing operation using Halide's built-in functions
    input_buffer = hl.Buffer(hl.UInt(8), input_shape)
    hl.image.resize(input_buffer, output, 2.0)
    
    return output

# Compile the pipeline to generate the Halide code
pipeline = resize_image
hl.codegen(pipeline, "resize_image")