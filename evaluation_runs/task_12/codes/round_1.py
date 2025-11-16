import halide as hl

def threshold_image(image):
    # Define the input image
    x = hl.Var('x')
    y = hl.Var('y')
    c = hl.Var('c')

    # Define the output image
    out = hl.Func('out')
    out[x, y] = hl.If(image[x, y, 0] > 0.5, 1, 0)

    return out

# Create a sample input image
image = np.random.rand(10, 10, 3).astype(np.float32)
image[::2, ::2, 0] = 1.0  # Set some pixels to 1

# Run the thresholding function
thresholded_image = threshold_image(image)

# Convert the output image back to a numpy array
result = np.array(thresholded_image).astype(np.uint8)