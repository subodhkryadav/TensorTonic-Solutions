import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X without using np.cov.
    """
    X = np.array(X)
    
    # Check for invalid input: must be 2D and have at least 2 samples
    if X.ndim != 2 or X.shape[0] < 2:
        return None
    
    N, D = X.shape
    
    # Step 1: Center the data (subtract mean of each feature)
    # X - mu has shape (N, D)
    X_centered = X - np.mean(X, axis=0)
    
    # Step 2: Compute covariance matrix Σ = (1 / (N - 1)) * (X_centered.T @ X_centered)
    # Resulting shape is (D, D)
    covariance = (X_centered.T @ X_centered) / (N - 1)
    
    return covariance
