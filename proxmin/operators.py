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
    return np.zeros(X.shape, dtype=X.dtype)

def prox_plus(X, step):
    """Projection onto non-negative numbers
    """
    below = X < 0
    X[below] = 0
    return X

def prox_unity(X, step, axis=0):
    """Projection onto sum=1 along an axis
    """
    return X / np.sum(X, axis=axis, keepdims=True)

def prox_unity_plus(X, step, axis=0):
    """Non-negative projection onto sum=1 along an axis
    """
    return prox_unity(prox_plus(X, step), step, axis=axis)

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

def prox_components(X, step, prox=None, axis=0):
    """Split X along axis and apply prox to each chunk.

    prox can be a list.
    """
    K = X.shape[axis]

    if not hasattr(prox_list, '__iter__'):
        prox = [prox] * K
    assert len(prox_list) == K

    if axis == 0:
        Pk = [prox_list[k](X[k], step) for k in range(K)]
    if axis == 1:
        Pk = [prox_list[k](X[:,k], step) for k in range(K)]
    return np.stack(Pk, axis=axis)


#### Regularization function below ####

def prox_hard(X, step, thresh=0):
    """Hard thresholding

    X if |X| >= thresh, otherwise 0
    NOTE: modifies X in place
    """
    thresh_ = _step_gamma(step, thresh)
    below = np.abs(X) < thresh_
    X[below] = 0
    return X

def prox_hard_plus(X, step, thresh=0):
    """Hard thresholding with projection onto non-negative numbers
    """
    return prox_plus(prox_hard(X, step, thresh=thresh), step)

def prox_soft(X, step, thresh=0):
    """Soft thresholding proximal operator
    """
    thresh_ = _step_gamma(step, thresh)
    return np.sign(X)*prox_plus(np.abs(X) - thresh_, step)

def prox_soft_plus(X, step, thresh=0):
    """Soft thresholding with projection onto non-negative numbers
    """
    return prox_plus(prox_soft(X, step, thresh=thresh), step)

def prox_max_entropy(X, step, gamma=1):
    """Proximal operator for maximum entropy regularization.

    g(x) = gamma \sum_i x_i ln(x_i)

    has the analytical solution of gamma W(1/gamma exp((X-gamma)/gamma)), where
    W is the Lambert W function.
    """
    from scipy.special import lambertw
    gamma_ = _step_gamma(step, gamma)
    # minimize entropy: return gamma_ * np.real(lambertw(np.exp((X - gamma_) / gamma_) / gamma_))
    above = X > 0
    X[above] = gamma_ * np.real(lambertw(np.exp(X[above]/gamma_ - 1) / gamma_))
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

def get_gradient_y(shape, py):
    """Calculate the gradient in the y direction to the line at py

    The y gradient operator is a block matrix, where each block is the size of the image width.
    The matrix itself is made up of (img_height x img_height) blocks, most of which are all zeros.
    """
    import scipy.sparse

    height, width = shape
    rows = []
    empty = scipy.sparse.dia_matrix((width, width))
    identity = scipy.sparse.identity(width)

    # Create the blocks by row, beginning with blocks leading up to the peak row from the top
    for n in range(py):
        row = [empty]*n
        row += [-identity, identity]
        row += [empty]*(height-n-2)
        rows.append(row)
    # Set all elements in the peak row to zero
    rows.append([empty]*height)
    # Create the blocks for the rows leading up to the peak row from the bottom
    for n in range(height-py-1):
        row = [empty]*(py+n)
        row += [identity, -identity]
        row += [empty]*(height-py-n-2)
        rows.append(row)
    return scipy.sparse.bmat(rows)

def get_gradient_x(shape, px):
    """Calculate the gradient in the x direction to the line at px

    The y gradient operator is a block diagonal matrix, where each block is the size of the image width.
    The matrix itself is made up of (img_height x img_height) blocks, most of which are all zeros.
    """
    import scipy.sparse

    height, width = shape
    size = height * width

    # Set the diagonal to -1, except for the value at the peak, which is zero
    c = -np.ones((width,))
    c[px] = 0
    # Set the pixels leading up to the peak from the left
    r = np.zeros(c.shape, dtype=c.dtype)
    r[:px] = 1
    # Set the pixels leading up to the peak from the right
    l = np.zeros(c.shape, dtype=c.dtype)
    l[px:] = 1
    # Make a block for a single row in the image
    block = scipy.sparse.diags([l, c, r], [-1, 0,1], shape=(width,width))
    # Use the same block for each row
    op = scipy.sparse.block_diag([block for n in range(height)])
    return op
