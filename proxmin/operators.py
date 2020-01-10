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
    X[:] = np.zeros(X.shape, dtype=X.dtype)
    return X


def prox_plus(X, step):
    """Projection onto non-negative numbers
    """
    below = X < 0
    X[below] = 0
    return X


def prox_unity(X, step, axis=0):
    """Projection onto sum=1 along an axis
    """
    X[:] = X / np.sum(X, axis=axis, keepdims=True)
    return X


def prox_unity_plus(X, step, axis=0):
    """Non-negative projection onto sum=1 along an axis
    """
    X[:] = prox_unity(prox_plus(X, step), step, axis=axis)
    return X


def prox_min(X, step, thresh=0, type="relative"):
    """Projection onto numbers above `thresh`

    If type == 'relative', the penalty is expressed in units of the function value;
    if type == 'absolute', it's expressed in units of the variable `X`.
    """
    assert type in ["relative", "absolute"]
    if type == "relative":
        thresh_ = _step_gamma(step, thresh)
    else:
        thresh_ = thresh
    below = X - thresh_ < 0
    X[below] = thresh_
    return X


def prox_max(X, step, thresh=0, type="relative"):
    """Projection onto numbers below `thresh`

    If type == 'relative', the penalty is expressed in units of the function value;
    if type == 'absolute', it's expressed in units of the variable `X`.
    """
    assert type in ["relative", "absolute"]
    if type == "relative":
        thresh_ = _step_gamma(step, thresh)
    else:
        thresh_ = thresh
    above = X - thresh_ > 0
    X[above] = thresh_
    return X


def prox_components(X, step, prox=None, axis=0):
    """Split X along axis and apply prox to each chunk.

    prox can be a list.
    """
    K = X.shape[axis]

    if not hasattr(prox_list, "__iter__"):
        prox = [prox] * K
    assert len(prox_list) == K

    if axis == 0:
        Pk = [prox_list[k](X[k], step) for k in range(K)]
    if axis == 1:
        Pk = [prox_list[k](X[:, k], step) for k in range(K)]
    X[:] = np.stack(Pk, axis=axis)
    return X


#### Regularization function below ####


def prox_hard(X, step, thresh=0, type="relative"):
    """Hard thresholding

    X if |X| >= thresh, otherwise 0
    NOTE: modifies X in place

    If type == 'relative', the penalty is expressed in units of the function value;
    if type == 'absolute', it's expressed in units of the variable `X`.
    """
    assert type in ["relative", "absolute"]
    if type == "relative":
        thresh_ = _step_gamma(step, thresh)
    else:
        thresh_ = thresh
    below = np.abs(X) < thresh_
    X[below] = 0
    return X


def prox_hard_plus(X, step, thresh=0, type="relative"):
    """Hard thresholding with projection onto non-negative numbers

    If type == 'relative', the penalty is expressed in units of the function value;
    if type == 'absolute', it's expressed in units of the variable `X`.
    """
    X[:] = prox_plus(prox_hard(X, step, thresh=thresh, type=type), step)
    return X


def prox_soft(X, step, thresh=0, type="relative"):
    """Soft thresholding proximal operator

    If type == 'relative', the penalty is expressed in units of the function value;
    if type == 'absolute', it's expressed in units of the variable `X`.
    """
    assert type in ["relative", "absolute"]
    if type == "relative":
        thresh_ = _step_gamma(step, thresh)
    else:
        thresh_ = thresh
    X[:] = np.sign(X) * prox_plus(np.abs(X) - thresh_, step)
    return X


def prox_soft_plus(X, step, thresh=0, type="relative"):
    """Soft thresholding with projection onto non-negative numbers

    If type == 'relative', the penalty is expressed in units of the function value;
    if type == 'absolute', it's expressed in units of the variable `X`.
    """
    X[:] = prox_plus(prox_soft(X, step, thresh=thresh, type=type), step)
    return X


def prox_max_entropy(X, step, gamma=1, type="relative"):
    """Proximal operator for maximum entropy regularization.

    g(x) = gamma sum_i x_i ln(x_i)

    has the analytical solution of gamma W(1/gamma exp((X-gamma)/gamma)), where
    W is the Lambert W function.

    If type == 'relative', the penalty is expressed in units of the function value;
    if type == 'absolute', it's expressed in units of the variable `X`.
    """
    from scipy.special import lambertw

    assert type in ["relative", "absolute"]
    if type == "relative":
        gamma_ = _step_gamma(step, gamma)
    else:
        gamma_ = gamma
    # minimize entropy: return gamma_ * np.real(lambertw(np.exp((X - gamma_) / gamma_) / gamma_))
    above = X > 0
    X[above] = gamma_ * np.real(lambertw(np.exp(X[above] / gamma_ - 1) / gamma_))
    return X


class AlternatingProjections(object):
    """Combine several proximal operators in the form of Alternating Projections

    This implements the simple POCS method with several repeated executions of
    the projection sequence.

    Note: The operators are executed in the "natural" order, i.e. the first one
    in the list is applied last.
    """

    def __init__(self, prox_list=None, repeat=1):
        self.operators = []
        self.repeat = repeat
        if prox_list is not None:
            self.operators += prox_list

    def __call__(self, X, step):
        # simple POCS method, no Dykstra or averaging
        # TODO: no convergence test
        # NOTE: inline updates
        for r in range(self.repeat):
            # in reverse order (first one last, as expected from a sequence of ops)
            for prox in self.operators[::-1]:
                X = prox(X, step)
        return X

    def find(self, cls):
        import functools

        for i in range(len(self.operators)):
            prox = self.operators[i]
            if isinstance(prox, functools.partial):
                if prox.func is cls:
                    return i
            else:
                if prox is cls:
                    return i
        return -1
