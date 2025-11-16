import halide as hl

def bilateral_filter(input_buffer):
    radius = 2
    bilateral_func = hl.Func('bilateral')
    bilateral_func[hx, hy] = input_buffer[hx, hy] * (1 - hl.exp(-((input_buffer[hx, hy] - input_buffer[hx + radius, hy])**2) / (2 * radius**2))) \
                            + input_buffer[hx + radius, hy] * hl.exp(-((input_buffer[hx, hy] - input_buffer[hx + radius, hy])**2) / (2 * radius**2))
    return bilateral_func

input_buffer = hl.Buffer(hl.float32, [1024, 1024])
gaussian_blur_func = bilateral_filter(input_buffer)
output_buffer = gaussian_blur_func.realize([1024, 1024])

print(output_buffer)