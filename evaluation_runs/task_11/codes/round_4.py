import halide as hl

# Define the dimensions and data type of the input image
input_image = hl.Func('input_image')
input_image.type.set(hl.TypeFloat(0))

# Specify the bounds of the image
x_range = hl.Range(0, 256)
y_range = hl.Range(0, 256)
z_range = hl.Range(0, 3)

# Define the input and output images
input_img = hl.ImageParam(input_image.type, 'input')
output_img = hl.ImageParam(input_image.type, 'output')

# Create a Halide pipeline that converts the input image to grayscale
pipeline = hl.pipeline()
pipeline.add(hl.Cast(output_img, input_img))

# Generate Halide code for the pipeline
hl_code = pipeline.generate()

print(hl_code)