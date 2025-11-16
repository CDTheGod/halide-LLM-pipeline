import halide as hl

def process_image(image):
    # Define the Halide pipeline
    pipe = hl.Pipeline('image')
    
    # Define the input and output variables
    in_image = pipe.input(hl.UInt(8), 'in', image.shape)
    out_image = pipe.output(hl.UInt(8), 'out', image.shape)
    
    # Apply a blur operation to the image
    blurred_image = hl.Convolve(in_image, 3, 3)
    
    # Assign the blurred image to the output variable
    out_image = blurred_image
    
    # Compile and run the pipeline
    pipe.compile_jit()
    processed_image = pipe.run(in_image.asnumpy())
    
    return processed_image

# Example usage:
image_data = [[[0.5, 0.7], [0.2, 0.1]], [[0.9, 0.8], [0.6, 0.4]]]
processed_image = process_image(image_data)
print(processed_image)