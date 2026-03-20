import numpy as np

def softmax(x):
    # Convert list or array-like to a NumPy array first
    x = np.asarray(x)
    
    # Handle axis and keepdims for both 1D and 2D
    # Using axis=-1 handles both the vector and the rows of a matrix
    x_max = np.max(x, axis=-1, keepdims=True)
    
    e_x = np.exp(x - x_max)
    
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
