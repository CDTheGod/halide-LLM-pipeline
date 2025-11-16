import halide as hl

# Define the function that takes the input image as a 2D array
def avg_neighbors(input_image):
    # Get the dimensions of the input image
    x = input_image.dim(0)
    y = input_image.dim(1)

    # Create a new 2D array to store the averaged values
    output_image = hl.Func('output_image')
    output_image[x, y] = (input_image[x-1, y] + input_image[x+1, y] +
                         input_image[x, y-1] + input_image[x, y+1]) / 4

    return output_image

# Define the main function that takes the input image as a Halide variable
def main():
    # Create a new Halide variable to store the input image
    input_image = hl.Var('input_image')

    # Define the pipeline that takes the input image and produces the averaged image
    pipeline = avg_neighbors(input_image)

    # Compile the pipeline into a function
    func = pipeline.compile_to_function()

    # Load the input image from a file (not shown in this example)
    img = imageio.imread('input.png')

    # Convert the input image to a Halide variable
    input_var = hl.Image(img, 'float32')

    # Run the pipeline on the input image
    output_img = func(input_var)

    # Save the output image to a file (not shown in this example)
    imageio.imwrite('output.png', output_img)

if __name__ == '__main__':
    main()