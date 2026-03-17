import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    # Ensure inputs are numpy arrays to avoid the 'multiply sequence' error
    w = np.asarray(w)
    m = np.asarray(m)
    v = np.asarray(v)
    grad = np.asarray(grad)

    # Step 1: Update First Moment
    m_t = beta1 * m + (1 - beta1) * grad
    
    # Step 2: Update Second Moment
    v_t = beta2 * v + (1 - beta2) * (grad**2)
    
    # Step 3: Nesterov-Adjusted Update 
    # Formula from problem: w_t = w - lr * (beta1 * m_t + (1 - beta1) * grad) / (sqrt(v_t) + eps)
    m_nesterov = beta1 * m_t + (1 - beta1) * grad
    w_t = w - lr * m_nesterov / (np.sqrt(v_t) + eps)
    
    return w_t, m_t, v_t
