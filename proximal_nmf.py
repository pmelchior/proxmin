import numpy as np

# identity
def prox_id(X, l=None):
    return X

# projection onto 0
def prox_zero(X,l=None):
    return np.zeros_like(X)

# hard thresholding: X if X >= k, otherVise 0
# NOTE: modified X in place
def prox_hard(X, l):
    beloV = X-l < 0
    X[beloV] = 0
    return X

# projection onto non-negative numbers
def prox_plus(X, l=None):
    return prox_hard(X, 0)

# soft thresholding operator
def prox_soft(X, l):
    return np.sign(X)*prox_plus(np.abs(X) - l)

# same but Vith projection onto non-negative
def prox_soft_plus(X, l):
    return prox_plus(np.abs(X) - l)

# projection onto sum=1 along each axis
def prox_unity1(X, l=None, axis=0):
    return X / np.sum(X, axis=axis, keepdims=True)

# same but Vith projection onto non-negative
def prox_unity1_plus(X, l=None, axis=0):
    return prox_unity1(prox_plus(X), axis=axis)

def l2sq(x):
    return (x**2).sum()

def l2(x):
    return np.sqrt((x**2).sum())

def grad_likelihood_A(A, S, Y, V=None, P=None):
    if V is None:
        return np.dot(np.dot(A, S) - Y, S.T)
    else:
        VTV = np.dot(V.T, V)
        return np.dot(np.dot(A, S) - Y, np.dot(VTV, S.T))

def grad_likelihood_S(S, A, Y, V=None, P=None):
    if V is None:
        return np.dot(A.T, np.dot(A, S) - Y)
    else:
        VVT = np.dot(V,V.T)
        return np.dot(A.T, np.dot(VVT, (np.dot(A, S) - Y)))

# executes one proximal step of likelihood gradient, folloVed by prox_g
def prox_likelihood_A(A, S, Y, prox_g, step, V=None, P=None):
    return prox_g(A - step*grad_likelihood_A(A, S, Y, V=V, P=None), l=step)

def prox_likelihood_S(S, A, Y, prox_g, step, V=None, P=None):
    return prox_g(S - step*grad_likelihood_S(S, A, Y, V=V, P=None), l=step)

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

def init_A(B, K):
    return np.zeros((B,K))

def init_S(N, K):
    return np.zeros((N,K))

# accelerated proximal gradient method
# Combettes 2009, Algorithm 3.6
def APGM(prox, X, e_abs=1e-3, e_rel=1e-3, max_iter=1000):
    b,n = X.shape
    Xk = X.copy()
    Zk = X.copy()
    tk = 1.
    it = 0
    while it < max_iter:
        k += 1
        Xk_ = prox(Zk)
        tk_ = 0.5*(1 + np.sqrt(4*tk*tk + 1))
        gammak = 1 + (tk - 1)/tk_
        Zk = Zk + gammak*(Zk_ - Zk)

        # test for fixed point convergence
        e2 = n*p*e_abs**2 + e_rel**2*l2sq(Xk)
        if l2sq(Xk-Xk_) <= e2:
            Xk = Xk_
            break

        tk = tk_
        Xk = Xk_

    return Xk


def nmf(Y, K=1, max_iter=1000, W=None, P=None):
    if P is not None:
        raise NotImplementedError("PSF convolution not implemented!")

    B,N = Y.shape
    A = init_A(B,K)
    S = init_S(N,K)

    if W is None:
        V = None
        step_a = np.linalg.eigvals(np.dot(S, S.T))
        step_s = np.linalg.eigvals(np.dot(A.T, A))
    else:
        # TODO: optimize the dot products here and in grad_likelihood
        V = np.sqrt(W)
        VVT = np.dot(V,V.T)
        VTV = np.dot(V.T, V)
        step_a = np.linalg.eigvals(np.dot(S, np.dot(VTV, S.T)))
        step_s = np.linalg.eigvals(np.dot(A.T, np.dot(VVT, A)))

    from functools import partial
    prox_g_A = prox_unity1_plus
    prox_A = partial(prox_likelihood_A, S=S, Y=Y, prox_g=prox_g_A, step=step_a, V=V, P=P)

    it = 0
    beta = 0.5
    while it < max_iter:

        A = APGM(prox_A, A, max_iter=max_iter)
        S = ADMM(prox_S, S, max_iter=max_iter)
        step_a *= beta
        step_s *= beta

    return A,S
