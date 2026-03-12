def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    if not points:
        return []
        
    dim = len(points[0])
    # Initialize centroids and a counter for points in each cluster
    centroids = [[0.0] * dim for _ in range(k)]
    counts = [0] * k
    
    # Sum up points for each cluster
    for point, cluster_idx in zip(points, assignments):
        counts[cluster_idx] += 1
        for i in range(dim):
            centroids[cluster_idx][i] += point[i]
            
    # Calculate the mean
    for j in range(k):
        if counts[j] > 0:
            for i in range(dim):
                centroids[j][i] /= counts[j]
        else:
            # If no points, return a zero vector as per requirements
            centroids[j] = [0.0] * dim
            
    return centroids
