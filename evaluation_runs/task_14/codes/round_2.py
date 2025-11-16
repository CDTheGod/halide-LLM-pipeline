import halide as hl

def image_processing(input_image):
    # Convert input image from float32 to uint8
    input_image = hl.cast(hl.UInt(8), input_image)
    
    # Create a new image with the same dimensions as the input image
    output_image = hl.image(hl.UInt(8), input_image.width(), input_image.height())
    
    # Assign values of input image to output image
    hl.assign(output_image, input_image)
    
    return output_image

# Define the input and expected output images for each test case
test_cases = [
    {
        "input": np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
        "expected_output": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    },
    {
        "input": np.array([[[10.0, 20.0], [30.0, 40.0]], [[50.0, 60.0], [70.0, 80.0]]]),
        "expected_output": np.array([[[10, 20], [30, 40]], [[50, 60], [70, 80]]])
    }
]

# Run the function for each test case and compare with expected output
for test_case in test_cases:
    input_image = hl.Image(hl.float32, test_case["input"].shape[1:], hl.scalarize(test_case["input"]))
    output_image = image_processing(input_image)
    
    # Convert Halide image to numpy array
    output_array = np.array(output_image)
    
    # Compare with expected output
    assert np.allclose(output_array, test_case["expected_output"])