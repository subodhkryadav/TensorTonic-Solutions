def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Convert lists to sets to handle duplicates and enable set operations
    s1 = set(set_a)
    s2 = set(set_b)
    
    # Handle the edge case: if both sets are empty, return 0.0
    if not s1 and not s2:
        return 0.0
    
    # Calculate intersection and union
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    
    # Return the ratio as a float
    return float(intersection / union)
