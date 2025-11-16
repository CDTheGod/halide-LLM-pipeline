import halide as hl

# Define the input and output variables
x = hl.Var('x')
y = hl.Var('y')

# Define the input and output images
input_image = hl.ImageParam(hl.float32, 'xy', "input")
output_image = hl.ImageParam(hl.float32, 'xy', "output")

# Define the Gaussian blur kernel
kernel = hl.Func("kernel")
kernel[x, y] = hl.exp(-(x*x + y*y) / 2)

# Apply the Gaussian blur to the input image
blurred_image = hl.Func("blurred_image")
blurred_image[x, y] = hl.ImgDiv(hl.ImgSum(input_image[x-1:x+2, y-1:y+2], kernel), 16)

# Define the pipeline
pipeline = hl.Pipeline()
pipeline.add_input(input_image)
pipeline.add_output(output_image)
pipeline.add_func(blurred_image)

# Compile and run the pipeline
hl.compile_pipeline(pipeline)