import halide as hl

def reshape_3d_to_2d(image):
    # Define the input and output variables
    x = hl.Var('x')
    y = hl.Var('y')
    z = hl.Var('z')

    # Reshape the 3D image into a 2D array
    reshaped_image = hl.Range(0, image.shape[0]) * hl.Range(image.shape[1], image.shape[2])

    return reshaped_image

# Test the function with a sample 3D image
image = np.random.rand(10, 10, 10)
reshaped_image = reshape_3d_to_2d(image)

print(reshaped_image)