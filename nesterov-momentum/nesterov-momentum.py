import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    w = np.asarray(w,dtype = float)
    v = np.asarray(v,dtype = float)
    grad = np.asarray(grad,dtype = float)

    #wl = w - (momentum * v)
    v = (momentum * v) + (lr * grad)
    w = w  - v
    return (w,v)