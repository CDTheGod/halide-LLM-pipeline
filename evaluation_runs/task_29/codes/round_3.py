import halide as hl

# Define the input type
input_type = hl.BlasFloat(3)

# Define the output type
output_type = hl.BlasFloat(1)

# Create the function
func = hl.Func('grayscale')

# Define the body of the function
body = func.root()
body = hl.Cast(hl.RGBToGray(body), output_type)

# Define the input and output variables
input_var = hl.Var('input')
output_var = hl.Var('output')

# Define the schedule
schedule = hl.Schedule(func)
schedule.parallel(input_var, 4)
schedule.parallel(output_var, 4)

# Create the Halide code
code = func.compile_to_c()

print(code)