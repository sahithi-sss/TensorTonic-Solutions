import numpy as np
import math

def gelu(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2)))
