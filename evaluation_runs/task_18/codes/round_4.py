import halide as hl

# Define the input image type
input_image = hl.ImageParam(hl.float32, [2, 4])

# Create a new image with twice the resolution in each dimension
new_image = hl.Image(hl.float32, [4, 8], "new_image")

# Perform interpolation using the interpolate function from the image_processing module
interpolated_pixel = hl.image_processing.interpolate(input_image, new_image)

# Define the output function
def interpolated_image(x, y):
    return interpolated_pixel[x, y]

# Compile and run the Halide code
hl.compile_to_function(interpolated_image)