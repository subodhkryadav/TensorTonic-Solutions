import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    # 1. Handle shapes: Convert 1D inputs to 2D (batch_size=1) for uniform matrix math
    D = params["Wz"].shape[0]
    H = params["Wz"].shape[1]
    
    x_2d, was_1d_x = _as2d(x, D)
    h_prev_2d, was_1d_h = _as2d(h_prev, H)

    # 2. Update Gate (z_t)
    # z_t = sigmoid(x_t * Wz + h_{t-1} * Uz + bz)
    z_t = _sigmoid(x_2d @ params["Wz"] + h_prev_2d @ params["Uz"] + params["bz"])

    # 3. Reset Gate (r_t)
    # r_t = sigmoid(x_t * Wr + h_{t-1} * Ur + br)
    r_t = _sigmoid(x_2d @ params["Wr"] + h_prev_2d @ params["Ur"] + params["br"])

    # 4. Candidate Hidden State (h_tilde_t)
    # h_tilde_t = tanh(x_t * Wh + (r_t * h_{t-1}) * Uh + bh)
    # Note: * denotes element-wise multiplication (Hadamard product)
    gated_h = r_t * h_prev_2d
    h_tilde_t = np.tanh(x_2d @ params["Wh"] + gated_h @ params["Uh"] + params["bh"])

    # 5. Final Hidden State (h_t)
    # h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t
    h_t = (1 - z_t) * h_prev_2d + z_t * h_tilde_t

    # 6. Return to original shape if input was 1D
    if was_1d_x or was_1d_h:
        return h_t.flatten()
    
    return h_t