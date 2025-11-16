import halide as hl

# Define a function to calculate the square root of a number
def sqrt(x):
    return x * (1 + x) / 2

# Define a function to calculate the Euclidean distance between two points
def euclidean_distance(point):
    x, y, z = point
    return sqrt((x - 0)**2 + (y - 0)**2 + (z - 0)**2)

# Define the main function to calculate the distances from each point in the input array to the origin
def calculate_distances(points):
    distances = []
    for point in points:
        distance = euclidean_distance(point)
        distances.append(distance)
    return distances

# Test the function with a sample input
points = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
distances = calculate_distances(points)

print(distances)