import halide as hl

# Define the input image
img = hl.ImageParam(hl.float32, [3])

# Compute the gradient in the x direction using the Sobel operator
gx = hl.Func('gx')
gx[hx, hy] = (hl.if_(hx == 0, 0,
                    hl.if_(hx < img.width() - 1, img(hx + 1, hy) - img(hx - 1, hy),
                           0)) +
             hl.if_(hy == 0, 0,
                    hl.if_(hy < img.height() - 1, img(hx, hy + 1) - img(hx, hy - 1),
                           0)))

# Compute the gradient in the y direction using the Sobel operator
gy = hl.Func('gy')
gy[hx, hy] = (hl.if_(hx == 0, 0,
                    hl.if_(hx < img.width() - 1, img(hx + 1, hy) - img(hx - 1, hy),
                           0)) +
             hl.if_(hy == 0, 0,
                    hl.if_(hy < img.height() - 1, img(hx, hy + 1) - img(hx, hy - 1),
                           0)))

# Define the output image
output = hl.ImageParam(hl.float32, [3])

# Compute the gradient magnitude
grad_mag = hl.Func('grad_mag')
grad_mag[hx, hy] = hl.sqrt(gx(hx, hy) * gx(hx, hy) + gy(hx, hy) * gy(hx, hy))

# Define the pipeline
pipe = hl.Pipeline(img)
pipe.output(grad_mag)

# Compile and run the pipeline
pipe.compile_to_file('grad_mag')