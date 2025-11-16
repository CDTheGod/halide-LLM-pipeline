import halide as hl

# Define the input and output variables
input_image = hl.ImageParam(hl.float32, [5, 5])
output_image = hl.Buffer(hl.float32, [5 * 5])

# Convert the image from RGB to grayscale using Halide's built-in function
grayscale_image = hlRgbToGrayscale(input_image)

# Define the computation for each pixel in the output image
with hl.Stage():
    hl.For('x', 0, 5).store(output_image, 'x',
        hl.Select(grayscale_image, 0, x) + hl.Select(grayscale_image, 1, x) * 0.299 +
        hl.Select(grayscale_image, 2, x) * 0.587 + hl.Select(grayscale_image, 3, x) * 0.114)

# Define the Halide pipeline
pipeline = hl.make_pipeline(
    hl.Input(input_image),
    hl.Output(output_image)
)

# Compile and run the pipeline
hl.compile(pipeline).run()