import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    y=np.array(x)
    e_x=np.exp(-y)

    output=1/(1+e_x)
    return output

    