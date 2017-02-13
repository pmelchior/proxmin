import numpy as np

# identity
def prox_id(X, l=None):
    return X

# projection onto 0
def prox_zero(X,l=None):
    return np.zeros_like(X)

# hard thresholding: X if X >= k, otherwise 0
# NOTE: modified X in place
def prox_hard(X, l):
    below = X-l < 0
    X[below] = 0
    return X

# projection onto non-negative numbers
def prox_plus(X, l=None):
    return prox_hard(X, 0)

# soft thresholding operator
def prox_soft(X, l):
    return np.sign(X)*prox_plus(np.abs(X) - l)

# same but with projection onto non-negative
def prox_soft_plus(X, l):
    return prox_plus(np.abs(X) - l)

# projection onto sum=1 along each axis
def prox_unity1(X, l=None, axis=0):
    return X / np.sum(X, axis=axis, keepdims=True)

# same but with projection onto non-negative
def prox_unity1_plus(X, l=None, axis=0):
    return prox_unity1(prox_plus(X), axis=axis)

def l2sq(x):
    return (x**2).sum()

def l2(x):
    return np.sqrt((x**2).sum())

def grad_likelihood_A(A, S, Y, W=None, P=None):
    if W is None:
        return np.dot(np.dot(A, S) - Y, S.T)
    else:
        return np.dot(W*(np.dot(A, S) - Y), S.T)

def grad_likelihood_S(S, A, Y, W=None, P=None):
    if W is None:
        return np.dot(A.T, np.dot(A, S) - Y)
    else:
        return np.dot(A.T, W*(np.dot(A, S) - Y))

# executes one proximal step of likelihood gradient, followed by prox_g
def prox_likelihood_A(A, S, Y, prox_g, step, W=None, P=None):
    return prox_g(A - step*grad_likelihood_A(A, S, Y, W=W, P=None), l=step)

def prox_likelihood_S(S, A, Y, prox_g, step, W=None, P=None):
    return prox_g(S - step*grad_likelihood_S(S, A, Y, W=W, P=None), l=step)

# split X into K components along axis
# apply prox_list[k] to each component k
# stack results to reconstruct shape of X
def prox_components(X, prox_list, step, axis=0):
    K = len(prox_list)
    assert X.shape[axis] == K
    if axis == 0:
        PK = [prox_list[i](XK[i]) for i in range(K)]
    if axis == 1:
        PK = [prox_list[i](XK[:,i]) for i in range(K)]
    return np.stack(PK, axis=axis)
    
def init_A():
    pass

def init_S():
    pass

def nmf(A, S, Y, W=None, P=None):
    if P is not None:
        raise NotImplementedError("PSF convolution not implemented!")
