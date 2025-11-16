import halide as hl

# Define the input and output variables
input_var = hl.Var('x')
input_var_2d = hl.Var('y')
input_var_3d = hl.Var('z')

output_var = hl.Var('x')
output_var_2d = hl.Var('y')
output_var_3d = hl.Var('z')

# Define the input and output types
input_type = hl.BoundedInt(0, 255)
output_type = hl.BoundedInt(0, 255)

# Create a Halide function
func = hl.Func('my_func')

# Define the body of the function
func[hx, hy, hz] = hl.Select(hl.Or(hx == 0, hx == input_var.x + 1), 
                             hl.Select(hy == 0, output_type(255),
                                       hl.Select(hy == input_var.y + 1,
                                                 output_type(0),
                                                 hl.Min(input_var[hy], input_var[hy+1]))),
                             hl.Select(hy == 0, output_type(255),
                                       hl.Select(hy == input_var.y + 1,
                                                 output_type(0),
                                                 hl.Max(input_var[hy-1], input_var[hy]))))

# Define the schedule
func.schedule[hx] = hl.Divide(hl.Input(hl.Int32), 2)
func.schedule[hy] = hl.Divide(hl.Input(hl.Int32), 2)

# Compile and run the function
hl.compile_to_c(func, 'my_func')