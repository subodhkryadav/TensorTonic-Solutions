import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Convert input to a numpy array to ensure vectorized operations
    x = np.array(x)
    
    # Return x where x >= 0, and alpha * x where x < 0
    return np.where(x >= 0, x, alpha * x)
