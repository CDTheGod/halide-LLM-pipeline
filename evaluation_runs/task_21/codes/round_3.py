import halide as hl

# Define the input image
img = hl.ImageParam(hl.float32, 'xy', (1024, 768))

# Define the gradient operators
sobel_x = hl.Func('sobel_x')
sobel_y = hl.Func('sobel_y')

# Compute gradients in x and y directions using Sobel operator
sobel_x(img) = hl.Min(
    hl.Max(img(x+1, y-1), img(x, y-1)),
    hl.Max(img(x+1, y), img(x, y))
)

sobel_y(img) = hl.Min(
    hl.Max(img(x-1, y+1), img(x, y+1)),
    hl.Max(img(x, y+1), img(x, y))
)

# Compute gradient magnitude
grad_mag = hl.Func('grad_mag')
grad_mag(img) = hl.Sqrt(sobel_x(img)**2 + sobel_y(img)**2)

# Define the output image
output = grad_mag(img)

# Compile and run the function
output.compile_jit()