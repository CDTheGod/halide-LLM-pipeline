import halide as hl

# Define the generate_image function
def generate_image(input_img):
    # Initialize the output image
    output = hl.image(hl.float32, 256, 256)

    # Resize the input image to 128x128 pixels
    resized_img = hl.image(hl.float32, 128, 128)
    resized_img.load(0, 0).cast_to(hl.float32) = input_img.load(0, 0).cast_to(hl.float32)
    resized_img.save("resized.png")

    # Blur the resized image
    blurred_img = hl.image(hl.float32, 128, 128)
    for i in range(1, 127):
        for j in range(1, 127):
            blurred_img.load(i, j).cast_to(hl.float32) = (resized_img.load(i-1, j).cast_to(hl.float32) + resized_img.load(i+1, j).cast_to(hl.float32) +
                                                           resized_img.load(i, j-1).cast_to(hl.float32) + resized_img.load(i, j+1).cast_to(hl.float32)) / 4.0
    blurred_img.save("blurred.png")

    # Convert the blurred image to grayscale
    gray_img = hl.image(hl.float32, 128, 128)
    for i in range(128):
        for j in range(128):
            gray_img.load(i, j).cast_to(hl.float32) = (blurred_img.load(i, j).red() + blurred_img.load(i, j).green() + blurred_img.load(i, j).blue()) / 3.0
    gray_img.save("gray.png")

# Load the input image
input_img = hl.image(hl.uint8, 256, 256)
input_img.load(0, 0) = hl.uint8(255)

# Generate the output image
generate_image(input_img)