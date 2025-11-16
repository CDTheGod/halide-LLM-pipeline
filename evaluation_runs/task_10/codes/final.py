import halide as hl

# Define a function to calculate the Euclidean distance between two points
def euclidean_distance(x1, y1, z1, x2, y2, z2):
    return (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2

# Define a function to find the nearest point to the origin from each set of points
def find_nearest_point(points):
    min_distances = []
    for point in points:
        distance = euclidean_distance(0, 0, 0, point[0], point[1], point[2])
        min_distances.append(distance)
    return min_distances

# Define the input points
points = [[[0.5, 0.3, 0.2],
           [0.7, 0.4, 0.6],
           [0.9, 0.1, 0.8]],
          [[0.2, 0.5, 0.3],
           [0.6, 0.7, 0.9],
           [0.1, 0.8, 0.4]]]

# Find the nearest point to the origin from each set of points
nearest_points = find_nearest_point(points)

# Print the result
print(nearest_points)