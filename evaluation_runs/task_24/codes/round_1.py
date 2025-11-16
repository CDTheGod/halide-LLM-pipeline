import halide as hl

x = 256
y = 256
z = 3

kernel_size = (5, 5, 5)
kernel_weights = np.random.rand(*kernel_size)

def convolve(x, y, z):
    input_buffer = hl.Buffer(hl.UInt(8), x * y * z)
    output_buffer = hl.Buffer(hl.UInt(8), x * y * z)

    convolve_kernel = hl.Func('convolve_kernel')
    convolve_kernel[x, y, z] = hl.Min(x, y, z) + 1

    convolve_func = hl.Convolve(convolve_kernel, input_buffer, output_buffer)
    return convolve_func