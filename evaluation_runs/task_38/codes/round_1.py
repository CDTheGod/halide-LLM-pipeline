import halide as hl

# Define the input and output variables
input_image = hl.Var('input_image')
output_image = hl.Var('output_image')

# Create a new function that takes the input image as a parameter
def blur_and_convert(input):
    # Apply Gaussian blur to the input image
    blurred = hl.GaussianBlur(input, 2.0)
    
    # Convert the blurred image from RGB to grayscale
    gray = hl.Cast(blurred, hl.Float(32))
    gray = hl.ReduceMean(gray, 3)
    
    return gray

# Create a new function that takes the input and output variables as parameters
def main(input_image, output_image):
    # Apply the blur_and_convert operation on the input image
    blurred_gray = blur_and_convert(input_image)
    
    # Assign the result to the output variable
    output_image = blurred_gray
    
    return output_image

# Compile the Halide code and run it on a sample image
input_data = np.random.rand(256, 256, 3).astype(np.float32) / 255.0
output_data = main(input_data, None)