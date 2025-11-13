import halide as hl
import imageio
import numpy as np

input = hl.ImageParam(hl.Float(32), 3)
x, y, c = hl.Var('x'), hl.Var('y'), hl.Var('c')

# Calculate weights for Gaussian blur
sigma = 1.5
weights_x = [hl.exp(-((i - 0) ** 2) / (2 * sigma ** 2)) for i in range(-1, 2)]
weights_y = [hl.exp(-((j - 0) ** 2) / (2 * sigma ** 2)) for j in range(-1, 2)]

# Normalize weights
sum_weights_x = sum(weights_x)
sum_weights_y = sum(weights_y)

weights_x = [w / sum_weights_x for w in weights_x]
weights_y = [w / sum_weights_y for w in weights_y]

clamped = hl.Func('clamped')
clamped[x, y, c] = input[hl.clamp(x, 0, input.width() - 1), hl.clamp(y, 0, input.height() - 1), c]

blur_x = hl.Func('blur_x')
blur_x[x, y, c] = (weights_x[0] * clamped[x - 1, y, c] +
                  weights_x[1] * clamped[x, y, c] +
                  weights_x[2] * clamped[x + 1, y, c])

blur_y = hl.Func('blur_y')
blur_y[x, y, c] = (weights_y[0] * blur_x[x, y - 1, c] +
                  weights_y[1] * blur_x[x, y, c] +
                  weights_y[2] * blur_x[x, y + 1, c])

img_np = imageio.imread('input.png').astype(np.float32) / 255.0
img_np = np.ascontiguousarray(img_np)
img_np = np.transpose(img_np, (2, 0, 1)).copy()
img = hl.Buffer(img_np)
input.set(img)

output = blur_y.realize([img.width(), img.height(), img.channels()])
output = np.array(output)
output = np.transpose(output, (1, 2, 0))
imageio.imsave('output.png', (output * 255.0).astype(np.uint8))