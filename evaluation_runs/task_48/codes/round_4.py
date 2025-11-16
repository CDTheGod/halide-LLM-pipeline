import halide as hl

# Define the input image
img = hl.ImageParam(hl.scalar.float32(), hl.make_vector(0, 100), 'img')

# Blur the image using filter_box
blurred_img = hl.filter_box(img, 3)

# Compile and run the code
hl.compile_to_file('blur_image', blurred_img)