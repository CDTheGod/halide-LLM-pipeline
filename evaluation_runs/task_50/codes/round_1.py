import halide as hl

# Define the input image
input_image = hl.Var('input_image')

# Compute the gradient in x direction
grad_x = hl.Func('grad_x')
grad_x[hil.BoundaryConditions.periodic(0)] = hl.grad(input_image, 0)

# Compute the gradient in y direction
grad_y = hl.Func('grad_y')
grad_y[hil.BoundaryConditions.periodic(1)] = hl.grad(input_image, 1)

# Compute the gradient magnitude
gradient_magnitude = hl.sqrt(grad_x * grad_x + grad_y * grad_y)