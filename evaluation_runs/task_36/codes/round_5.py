import halide as hl

# Define the input and output images
input_image = hl.ImageParam(hl.UInt(8), 'xy')
output_image = hl.Image('o', hl.UInt(8), 'xy')

# Resize the input image to a new size (e.g., 1/4 of the original size)
new_size = hl.Divide(hl.MakeVector(2, 2), hl.MakeVector(4, 4))
halide_image_resize(input_image, output_image, new_size)

# Compile and run the code
hl.compile_to_c(output_image).run({'input': input_image})