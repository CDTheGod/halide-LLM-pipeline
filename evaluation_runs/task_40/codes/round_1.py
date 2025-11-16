import halide as hl

def process_image(image):
    # Define the pipeline stages
    input_stage = hl.Input('input', image.shape)
    resize_stage = hl.ResizeBilinear(input_stage, 256, 256)
    convert_stage = hl.Convert(resize_stage, 'grayscale')
    output_stage = hl.Output(convert_stage)

    # Create the pipeline
    pipeline = hl.Pipeline(output_stage)

    return pipeline

# Generate Halide code for the pipeline
pipeline = process_image(np.random.rand(512, 512, 3))
halide_code = pipeline.generate_halide_code()

print(halide_code)