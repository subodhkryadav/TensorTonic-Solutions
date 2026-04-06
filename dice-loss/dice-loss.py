import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    p: Predicted probabilities (N,) or (H,W)
    y: Binary mask {0,1}, same shape as p
    eps: Smoothing epsilon for stability
    """
    # Convert inputs to numpy arrays (handles lists or nested lists)
    p = np.asarray(p).astype(np.float32)
    y = np.asarray(y).astype(np.float32)

    # Calculate the intersection: sum(P * Y)
    intersection = np.sum(p * y)
    
    # Calculate the sums of each array
    p_sum = np.sum(p)
    y_sum = np.sum(y)

    # Dice Coefficient formula: (2 * intersection + eps) / (sum(P) + sum(Y) + eps)
    dice_coeff = (2. * intersection + eps) / (p_sum + y_sum + eps)

    # Dice Loss is 1 - Dice Coefficient
    loss = 1.0 - dice_coeff
    
    return float(loss)
