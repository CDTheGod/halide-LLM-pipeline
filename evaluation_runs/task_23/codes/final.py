import halide as hl

# Define the pipeline for the input image
def create_pipeline():
    x = hl.Var('x')
    y = hl.Var('y')
    c = hl.Var('c')

    input_image = hl.Func('input_image')
    output_image = hl.Func('output_image')

    input_image[x, y] = hl.Cast(hl.UInt(8), 0)
    for i in range(-1, 2):
        for j in range(-1, 2):
            input_image[x + i, y + j] += hl.Load(input_image, x + i, y + j)

    output_image[x, y] = hl.Cast(hl.UInt(8), 0)
    for i in range(-1, 2):
        for j in range(-1, 2):
            output_image[x + i, y + j] += input_image[x + i, y + j]

    return output_image

# Create the pipeline
pipeline = create_pipeline()

# Compile and run the pipeline
hl.compile_pipeline(pipeline)