import numpy as np

def swish(x):
    """
    Implement Swish activation function: f(x) = x * sigmoid(x)
    """
    # Convert input to a numpy array to ensure vectorized operations
    x = np.array(x, dtype=float)
    
    # Numerically stable Sigmoid implementation
    # sigma(x) = 1 / (1 + exp(-x))
    # For very small x, exp(-x) can overflow; 
    # however, for Swish, x * sigmoid(x) naturally handles 
    # large negative values by approaching 0.
    sigmoid = 1 / (1 + np.exp(-x))
    
    # Calculate Swish
    result = x * sigmoid
    
    # Ensure it returns an array even for scalars to match requirements
    return np.atleast_1d(result)
