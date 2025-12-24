import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Perform one forward step of a vanilla tanh RNN.

    Parameters
    ----------
    x_t : array, shape (D,)
        Input at time t
    h_prev : array, shape (H,)
        Previous hidden state
    Wx : array, shape (D, H)
        Input-to-hidden weight matrix
    Wh : array, shape (H, H)
        Hidden-to-hidden (recurrent) weight matrix
    b : array, shape (H,)
        Bias vector

    Returns
    -------
    h_t : array, shape (H,)
        New hidden state
    """
    # Ensure NumPy arrays (no in-place modification)
    x_t = np.asarray(x_t, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)

    # Compute pre-activation
    pre_act = x_t @ Wx + h_prev @ Wh + b

    # Apply non-linearity
    h_t = np.tanh(pre_act)

    return h_t
