import numpy as np

def prox_id(X, step):
    """Identity proximal operator
    """
    return X

def prox_zero(X, step):
    """Proximal operator to project onto zero
    """
    return np.zeros_like(X)

def prox_hard(X, step, thresh=0):
    """Hard thresholding

    X if X >= thresh, otherwise 0
    NOTE: modifies X in place
    """
    below = X - thresh*step < 0
    X[below] = 0
    return X

def prox_plus(X, step):
    """Projection onto non-negative numbers
    """
    return prox_hard(X, step)

def prox_min(X, step, thresh=0):
    """Projection onto numbers above `thresh`
    """
    below = X - thresh*step < 0
    X[below] = thresh*step
    return X

def prox_max(X, step, thresh=0):
    """Projection onto numbers below `thresh`
    """
    above = X - thresh*step > 0
    X[above] = thresh*step
    return X

def prox_soft(X, step, thresh=0):
    """Soft thresholding proximal operator
    """
    return np.sign(X)*prox_plus(np.abs(X) - thresh*step, step)

def prox_soft_plus(X, step, l=0):
    """Soft thresholding with projection onto non-negative numbers
    """
    return prox_plus(prox_soft(X, step, thresh=l), step)

def prox_unity(X, step, axis=0):
    """Projection onto sum=1 along an axis
    """
    return X / np.sum(X, axis=axis, keepdims=True)

def prox_unity_plus(X, step, axis=0):
    """Non-negative Projection onto sum=1 along an axis
    """
    return prox_unity(prox_plus(X, step), step, axis=axis)
