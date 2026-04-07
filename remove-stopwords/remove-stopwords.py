def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    # Convert stopwords to a set for O(1) average lookup speed
    stopset = set(stopwords)
    
    # Use a list comprehension to filter tokens
    return [token for token in tokens if token not in stopset]
