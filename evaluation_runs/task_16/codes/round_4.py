import halide as hl

def generate_halide_code():
    # Create input image buffer
    input_image = hl.Buffer(hl.float32, [1024, 768])

    # Define resized image buffer
    resized_image = hl.Buffer(hl.float32, [512, 384])

    # Convert image data to float32
    input_image.copy_from(np.random.rand(1024, 768).astype(np.float32))

    # Apply operations to input image
    output_image = hl.image(input_image)
    output_image.resize(resized_image)

    # Define blur function
    def blur(image):
        return hl.image(image).blur(hl.Radius(2), hl.Radius(2))

    # Apply blur operation to resized image
    blurred_image = blur(output_image)

    # Convert blurred image data back to uint8
    blurred_image.copy_to(np.random.rand(512, 384).astype(np.uint8))

    return input_image, output_image, resized_image, blurred_image

input_image, output_image, resized_image, blurred_image = generate_halide_code()