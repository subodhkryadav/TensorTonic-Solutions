import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here

    # now used to builden method of the numpy
    mat=np.array(A)
    mat_t=mat.T
    return mat_t