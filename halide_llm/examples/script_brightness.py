import numpy as np
import math

def rainbow_horizontal(h=32, w=32):
    arr = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            x = j / (w - 1)
            arr[i, j, 0] = x                # Red increases horizontally
            arr[i, j, 1] = 1.0 - x          # Green decreases horizontally
            arr[i, j, 2] = 0.5 + 0.5 * math.sin(3 * math.pi * x)  # Blue wave
    return arr


def dual_axis_gradient(h=32, w=32):
    arr = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            arr[i, j, 0] = i / (h - 1)               # R depends on row
            arr[i, j, 1] = j / (w - 1)               # G depends on column
            arr[i, j, 2] = (i + j) / (2 * (h - 1))   # B depends on both
    return arr


def colored_checkerboard(h=32, w=32, block=4):
    arr = np.zeros((h, w, 3), dtype=np.float32)
    c1 = np.array([0.9, 0.6, 0.2])  # orange
    c2 = np.array([0.1, 0.8, 0.7])  # teal
    for i in range(h):
        for j in range(w):
            if ((i // block + j // block) % 2) == 0:
                arr[i, j] = c1
            else:
                arr[i, j] = c2
    return arr


def colored_center_circle(h=32, w=32, radius=10):
    arr = np.zeros((h, w, 3), dtype=np.float32)
    cx = (h - 1) / 2
    cy = (w - 1) / 2
    for i in range(h):
        for j in range(w):
            d = math.hypot(i - cx, j - cy)
            if d < radius:
                arr[i, j] = [0.0, 0.9, 0.8]       # inner: teal
            elif d < radius + 3:
                arr[i, j] = [1.0, 0.4, 0.0]       # ring: orange/red
            else:
                arr[i, j] = [0.0, 0.1, 0.3]       # outer: navy blue
    return arr


def square_plus_diagonal(h=32, w=32):
    arr = np.zeros((h, w, 3), dtype=np.float32)

    # Background colorful gradient
    for i in range(h):
        for j in range(w):
            arr[i, j, 0] = 0.2 + 0.6 * (i / (h - 1))             # Red ramp vertical
            arr[i, j, 1] = 0.2 + 0.6 * (j / (w - 1))             # Green ramp horizontal
            arr[i, j, 2] = 0.2 + 0.6 * ((i + j) / (2*(h - 1)))   # Blue diagonal ramp

    # bright green square
    arr[4:12, 4:12] = [0.0, 1.0, 0.0]

    # magenta diagonal
    for k in range(min(h, w)):
        arr[k, k] = [1.0, 0.0, 1.0]

    return arr
