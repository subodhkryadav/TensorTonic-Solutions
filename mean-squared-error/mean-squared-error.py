import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    #first convert into the np_array
    y_pred_new=np.array(y_pred)
    y_true_new=np.array(y_true)

    # then subtract it
    sub=np.subtract(y_pred_new,y_true_new)

    # then find the square
    sq=squared_arr = np.square(sub)

    #now find the sum of these subtraction
    total=np.sum(sq)

    return (total/len(sq))
    

    