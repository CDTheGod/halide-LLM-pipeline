import halide as hl

# Define the input image as a Halide buffer
input_image = hl.Buffer(hl.UInt(8), 5*5*3)

# Convert the RGB values of each pixel to a single grayscale value
grayscale_image = hl.RgbToGrayscale(input_image)

# Define the output function
def convert_to_grayscale():
    return grayscale_image

# Compile and run the function
convert_to_grayscale()