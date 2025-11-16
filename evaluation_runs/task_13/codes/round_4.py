import halide as hl

def process_image(image_data):
    # Create a new Halide function named 'image' that takes the input image data as an argument and returns the processed image.
    image = hl.Func('image')
    
    # Define the operations to be applied to the image.
    resized_image = hl.image.resize(image, 0.5)
    grayscale_image = hl.image.convert_to_grayscale(resized_image)
    blurred_image = hl.image.gaussian_blur(grayscale_image, 3, 3)
    
    # Return the processed image.
    return blurred_image

# Create a new Halide pipeline that takes the input image data as an argument and returns the processed image.
pipe = hl.Pipeline(process_image)

# Define the input image data.
image_data = np.array([...])  # Replace with actual image data.

# Execute the pipeline and get the output image.
output_image = pipe.compile(image_data).run()

# Save the output image to a file.
imageio.imwrite('output.png', output_image)