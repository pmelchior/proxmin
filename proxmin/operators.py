import numpy as np

#### CAUTION ####
def _step_gamma(step, gamma):
    """Update gamma parameter for use inside of continuous proximal operator.

    Every proximal operator for a function with a continuous parameter,
    e.g. gamma ||x||_1, needs to update that parameter to account for the
    stepsize of the algorithm.

    Returns:
        gamma * step
    """
    return gamma * step
#################

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
    thresh_ = _step_gamma(step, thresh)
    below = X - thresh_ < 0
    X[below] = 0
    return X

def prox_plus(X, step):
    """Projection onto non-negative numbers
    """
    return prox_hard(X, step)

def prox_min(X, step, thresh=0):
    """Projection onto numbers above `thresh`
    """
    thresh_ = _step_gamma(step, thresh)
    below = X - thresh_ < 0
    X[below] = thresh_
    return X

def prox_max(X, step, thresh=0):
    """Projection onto numbers below `thresh`
    """
    thresh_ = _step_gamma(step, thresh)
    above = X - thresh_ > 0
    X[above] = thresh_
    return X

def prox_soft(X, step, thresh=0):
    """Soft thresholding proximal operator
    """
    thresh_ = _step_gamma(step, thresh)
    return np.sign(X)*prox_plus(np.abs(X) - thresh_, step)

def prox_soft_plus(X, step, thresh=0):
    """Soft thresholding with projection onto non-negative numbers
    """
    return prox_plus(prox_soft(X, step, thresh=thresh), step)

def prox_unity(X, step, axis=0):
    """Projection onto sum=1 along an axis
    """
    return X / np.sum(X, axis=axis, keepdims=True)

def prox_unity_plus(X, step, axis=0):
    """Non-negative projection onto sum=1 along an axis
    """
    return prox_unity(prox_plus(X, step), step, axis=axis)

def prox_max_entropy(X, step, gamma=1):
    """Proximal operator for maximum entropy regularization.

    g(x) = gamma \sum_i x_i ln(x_i)

    has the analytical solution of gamma W(1/gamma exp((X-gamma)/gamma)), where
    W is the Lambert W function. This would *minimize* the entropy g(x),
    maximizing requires a few sign flips.
    """
    from scipy.special import lambertw
    gamma_ = _step_gamma(step, gamma)
    # minimize entropy: return gamma_ * np.real(lambertw(np.exp((X - gamma_) / gamma_) / gamma_))

    return - gamma_ * np.real(lambertw(np.exp(-(X + gamma_) / gamma_) / -gamma_))
