def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x = x0
    for _ in range(steps):
        # The derivative of ax^2 + bx + c is 2ax + b
        gradient = 2 * a * x + b
        # Update x: x_new = x_old - learning_rate * gradient
        x = x - lr * gradient
        
    return float(x)
