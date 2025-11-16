import halide as hl

# Define the input volume
def generate_volume():
    x = 10
    y = 10
    z = 10
    volume = hl.Buffer(hl.float32, [x, y, z])
    
    # Initialize the volume with some values (just for demonstration)
    for i in range(x):
        for j in range(y):
            for k in range(z):
                volume[i, j, k] = i + j * 10 + k * 100
    
    return volume

# Define a function to extract a slice from the volume
def generate_slice(volume, x_index, y_index, z_index):
    # Create a new array that represents the slice
    slice_array = hl.Array_2(hl.float32, [x_index + 1, y_index + 1])
    
    # Extract the slice from the volume
    for i in range(x_index, x_index + 1):
        for j in range(y_index, y_index + 1):
            slice_array[i - x_index, j - y_index] = volume[i, j, z_index]
    
    return slice_array

# Generate a sample volume and extract slices
volume = generate_volume()
slice_0 = generate_slice(volume, 5, 5, 5)
slice_1 = generate_slice(volume, 7, 3, 2)

# Print the extracted slices
print("Slice 0:")
hl.print(slice_0)
print("\nSlice 1:")
hl.print(slice_1)