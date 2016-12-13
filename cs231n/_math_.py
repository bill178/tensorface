import numpy as np

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - arg_x: Input data, of shape (N, C) where arg_x[i, j] is the score for the jth class
    for the ith input.
    - arg_y: Vector of labels, of shape (N,) where arg_y[i] is the label for arg_x[i] and
    0 <= arg_y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to arg_x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    dx = probs.copy()

    loss = -np.sum([np.log(probs[i, int(y[i])]) for i in xrange(N)]) / N
    for i in xrange(N):
        dx[i, int(y[i])] -= 1
    dx /= N
    
    return loss, dx