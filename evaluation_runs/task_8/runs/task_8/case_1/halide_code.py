import halide as hl
import imageio
import numpy as np

# Constructing Halide functions statically.
input = hl.ImageParam(hl.Float(32), 3)
f = hl.Func('f')
x, y, c = hl.Var('x'), hl.Var('y'), hl.Var('c')

# Multiply the values by 1.5 and clamp them to 1.0
f[x, y, c] = hl.min(1.5 * input[x, y, c], 1.0)

# Actually compiling/executing the Halide functions.
# Setup the input by loading an image (Halide assumes Fortran ordering).
img_np = imageio.imread('input.png').astype(np.float32) / 255.0
# Ensure shape is (height, width, channels) and memory is contiguous
img_np = np.ascontiguousarray(img_np)

img = hl.Buffer(img_np)
input.set(img)
# Process the input by calling f.realize
output = f.realize([img.width(), img.height(), img.channels()])
# Save the image to a file by converting to a numpy array.
output = np.array(output)
imageio.imsave('output.png', (output * 255).astype(np.uint8))