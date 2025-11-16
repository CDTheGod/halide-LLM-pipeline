import halide as hl

# Define the function generate_image that takes no arguments and returns a buffer
def generate_image():
    # Create a buffer with shape (6, 10, 3) and type uint8_t
    img = hl.Buffer(hl.UInt(8), [6, 10, 3])

    # Iterate over each pixel in the image
    for y in range(6):
        for x in range(10):
            for c in range(3):
                # Assign color value to buffer based on given data
                img[y, x, c] = data[y, x, c]

    return img

# Define the data array
data = np.array([
    [[0.2, 0.5, 0.1], [0.4, 0.7, 0.3], [0.6, 0.9, 0.2],
     [0.8, 0.1, 0.4], [0.3, 0.6, 0.5], [0.7, 0.8, 0.9]],
    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9],
     [0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 0.1]],
    [[0.3, 0.4, 0.5], [0.6, 0.7, 0.8], [0.9, 0.1, 0.2],
     [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.1, 0.2, 0.3]],
    [[0.5, 0.6, 0.7], [0.8, 0.9, 0.1], [0.2, 0.3, 0.4],
     [0.6, 0.7, 0.8], [0.9, 0.1, 0.2], [0.3, 0.4, 0.5]],
    [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6],
     [0.8, 0.9, 0.1], [0.2, 0.3, 0.4], [0.5, 0.6, 0.7]],
    [[0.9, 0.1, 0.2], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8],
     [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
])

# Create a Halide pipeline
pipe = generate_image()

# Compile the pipeline to C++
pipe.compile_to_c("generate_image")

# Run the compiled code and save the output as an image file
imageio.imwrite('output.png', pipe)