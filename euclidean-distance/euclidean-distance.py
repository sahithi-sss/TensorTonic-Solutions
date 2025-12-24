import numpy as np

def euclidean_distance(x, y):
    """
    Compute Euclidean (L2) distance between two vectors x and y.
    """
    # Convert inputs to NumPy arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Compute Euclidean distance
    return np.sqrt(np.sum((x - y) ** 2))
