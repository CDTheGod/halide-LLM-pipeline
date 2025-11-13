import halide as hl
import imageio
import numpy as np

input = hl.ImageParam(hl.Float(32), 3)
x, y, c = hl.Var('x'), hl.Var('y'), hl.Var('c')

clamped = hl.Func('clamped')
clamped[x, y, c] = input[hl.clamp(x, 0, input.width() - 1), hl.clamp(y, 0, input.height() - 1), c]

blur_x = hl.Func('blur_x')
blur_x[x, y, c] = (clamped[x - 1, y, c] + clamped[x, y, c] + clamped[x + 1, y, c]) / 3.0

blur_y = hl.Func('blur_y')
blur_y[x, y, c] = (blur_x[x, y - 1, c] + blur_x[x, y, c] + blur_x[x, y + 1, c]) / 3.0

img_np = imageio.imread('input.png').astype(np.float32) / 255.0
img_np = np.ascontiguousarray(img_np)
img_np = np.transpose(img_np, (2, 0, 1)).copy()
img = hl.Buffer(img_np)
input.set(img)

output = blur_y.realize([img.width(), img.height(), img.channels()])
output = np.array(output)
output = np.transpose(output, (1, 2, 0))
imageio.imsave('output.png', (output * 255.0).astype(np.uint8))