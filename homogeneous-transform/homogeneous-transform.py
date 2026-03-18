import numpy as np

def apply_homogeneous_transform(T, points):
    # Convert T to numpy array to use .T and @ operator
    T = np.asanyarray(T)
    pts = np.atleast_2d(points)
    
    # Create homogeneous coordinates (N, 4)
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])
    
    # Transform: pts_h @ T.T is equivalent to T @ p for every row
    transformed_h = pts_h @ T.T
    
    # Extract (x, y, z)
    result = transformed_h[:, :3]
    
    # Return (3,) for single point, (N, 3) for batch
    return result[0] if np.ndim(points) == 1 else result
