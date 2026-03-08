import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Convert input to a numpy array to handle scalars/lists consistently
    x = np.atleast_1d(np.array(x, dtype=float))
    
    # Calculate e^x and e^-x
    exp_x = np.exp(x)
    exp_neg_x = np.exp(-x)
    
    # Apply the formula: (e^x - e^-x) / (e^x + e^-x)
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
