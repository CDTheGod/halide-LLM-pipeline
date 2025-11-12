import imageio.v2 as imageio
import numpy as np
import os

def write_image(arr, path):
    arr = np.array(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    arr = np.clip(arr, 0.0, 1.0)
    imageio.imwrite(path, (arr * 255.0).astype(np.uint8))

def load_image(path):
    return imageio.imread(path).astype(np.float32) / 255.0
