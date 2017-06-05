from __future__ import print_function, division
import logging
import numpy as np

logging.basicConfig()
logger = logging.getLogger("proxmin.operators")

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
    return prox_plus(prox_soft(X, step, l=l), step)

def prox_unity(X, step, axis=0):
    """Projection onto sum=1 along an axis
    """
    return X / np.sum(X, axis=axis, keepdims=True)

def prox_unity_plus(X, step, axis=0):
    """Non-negative Projection onto sum=1 along an axis
    """
    return prox_unity(prox_plus(X, step), step, axis=axis)

def prox_likelihood(X, step, prox_g, allX, xidx, grad_likelihood):
    """Apply proximal operator to a gradient step
    """
    if xidx==0:
        #logger.debug("{0}, {1}".format(grad_likelihood, prox_g))
        #logger.debug("step: {0}".format(step))
        #logger.debug("grad_likelihood: \n{0}".format(grad_likelihood(X, allX, xidx)))
        #logger.debug("step*grad_likelihood: \n{0}".format(step*grad_likelihood(X, allX, xidx)))
        #logger.debug("X-step*grad_likelihood: \n{0}".format(X-step*grad_likelihood(X, allX, xidx)))
        pass

    return prox_g(X-step*grad_likelihood(X, allX, xidx), step=step)
