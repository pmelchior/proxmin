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
def prox_likelihood_A(A, step, S=None, Y=None, prox_g=None, V=None, P=None):
    return prox_g(A - step*grad_likelihood_A(A, S, Y, V=V, P=None), l=step)

def prox_likelihood_S(S, step, A=None, Y=None, prox_g=None, V=None, P=None):
    return prox_g(S - step*grad_likelihood_S(S, A, Y, V=V, P=None), l=step)

# split X into K components along axis
# apply prox_list[k] to each component k
# stack results to reconstruct shape of X
def prox_components(X, step, prox_list=[], axis=0):
    K = X.shape[axis]
    assert len(step) == K
    assert len(prox_list) == K

    if axis == 0:
        PK = [prox_list[i](XK[i], step[i]) for i in range(K)]
    if axis == 1:
        PK = [prox_list[i](XK[:,i], step[i]) for i in range(K)]
    return np.stack(PK, axis=axis)

# accelerated proximal gradient method
# Combettes 2009, Algorithm 3.6
def APGM(prox, X, step, e_rel=1e-3, max_iter=1000):
    b,n = X.shape
    Xk = X.copy()
    Zk = X.copy()
    tk = 1.
    for it in range(max_iter):
        Xk_ = prox(Zk, step)
        tk_ = 0.5*(1 + np.sqrt(4*tk*tk + 1))
        gammak = 1 + (tk - 1)/tk_
        Zk = Xk + gammak*(Xk_ - Xk)

        # test for fixed point convergence
        if l2sq(Xk - Xk_) <= e_rel**2*l2sq(Xk):
            Xk = Xk_
            break

        tk = tk_
        Xk = Xk_

    return Xk

# Alternating direction method of multipliers
# initial: initial guess of solution
# K: number of iterations
# A: minimizes f(x) + g(Ax)
# See Boyd+2011, Section 3, with arbitrary A, B=-Id, c=0
def ADMM(prox_f, step_f, prox_g, step_g, X, max_iter=1000, A=None, e_abs=1e-6, e_rel=1e-3):
    if A is None:
        Uk = np.zeros_like(X)
        Zk = initial.copy()
        p,n = initial.size, initial.size
    else:
        Xk = X.copy()
        Zk = np.dot(A, Xk)
        Uk = np.zeros_like(Zk)
        p,n = A.shape

    for it in range(max_iter):
        if A is None:
            Xk = prox_f(Zk - Uk, step=step_f)
            Ak = Xk
        else:
            Xk = prox_f(Xk - step_f/step_g*np.dot(A.T, np.dot(A, Xk) - Zk + Uk), step=step_f)
            Ak = np.dot(A, Xk)
        Zk_ = prox_g(Ak + Uk, l=step_g)
        # this uses relaxation parameter of 1
        Uk = Uk + Ak - Zk_

        # compute prime residual rk and dual residual sk
        Rk = Ak - Zk_
        if A is None:
            Sk = -(Zk_ - Zk)
        else:
            Sk = -np.dot(A.T, Zk_ - Zk)
        Zk = Zk_

        # stopping criteria from Boyd+2011, sect. 3.3.1
        e_pri2 = p*e_abs**2 + e_rel**2*max(l2sq(Ak), l2sq(Zk))
        if A is None:
            e_dual2 = n*e_abs**2 + e_rel**2*l2sq(Uk)
        else:
            e_dual2 = n*e_abs**2 + e_rel**2*l2sq(np.dot(A.T, Uk))
        if l2sq(Rk) <= e_pri2 and l2sq(Sk) <= e_dual2:
            break

    return Xk,Zk,Uk


def steps_AS(A,S,V=None):
    # TODO: optimize the dot products here and in grad_likelihood
    if V is None:
        step_A = 1./np.linalg.eigvals(np.dot(S, S.T)).max()
        step_S = 1./np.linalg.eigvals(np.dot(A.T, A)).max()
    else:
        VVT = np.dot(V,V.T)
        VTV = np.dot(V.T, V)
        print VVT
        print VTV
        step_A = 1./np.linalg.eigvals(np.dot(S, np.dot(VTV, S.T))).max()
        step_S = 1./np.linalg.eigvals(np.dot(A.T, np.dot(VVT, A))).max()
    return step_A, step_S


def nmf_AS(Y, A, S, max_iter=1000, constraints=None, W=None, P=None):

    if W is None:
        V = None
    else:
        V = np.sqrt(W)
    step_A, step_S = steps_AS(A,S,V=V)
    print step_A, step_S

    from functools import partial
    # define proximal operators:
    # A: ||A_k||_2 = 1 with A_ik >= 0 for all columns k
    prox_g_A = prox_unity1_plus

    # S: L0 sparsity plus linear
    # TODO: individual sparsity or global?
    prox_g_S = prox_hard

    # additional constraint for each component of S
    if constraints is not None:
        # ... initialize the constraint matrices ...
        Cs = []

        # calculate step sizes for each constraint matrix
        lCs = np.array([np.linalg.eigvals(np.dot(C.T, C)).max() for C in Cs])

        prox_constraints = {
            "M": prox_plus,  # positive gradients
            "S": prox_zero   # zero deviation of mirrored pixels
        }
        prox_Cs = [prox_constraints[c] for c in constraints]

    beta = 0.5
    for it in range(max_iter):
        print it
        # A: simple gradient method; need to rebind S each time
        prox_A = partial(prox_likelihood_A, S=S, Y=Y, prox_g=prox_g_A, V=V, P=P)
        A = APGM(prox_A, A, step_A, max_iter=max_iter)
        print A

        # A: either gradient or ADMM, depending on additional constraints
        prox_S = partial(prox_likelihood_S, A=A, Y=Y, prox_g=prox_g_S, V=V, P=P)
        if constraints is None:
            S = APGM(prox_S, S, step_S, max_iter=max_iter)
        else:
            # split constraints along each row = component
            # need step sizes for each component
            step_S2 = step_S * lCs
            prox_S2 = partial(prox_components, prox_list=prox_Cs, axis=1)
            S = ADMM(prox_S, step_S, prox_S2, step_S2, S, max_iter=max_iter)

        # recompute step_sizes
        step_A, step_S = steps_AS(A,S,V=V)
        step_A *= beta
        step_S *= beta

    return A,S

def init_A(B, K):
    return np.zeros((B,K))

def init_S(N, K):
    return np.zeros((N,K))

def nmf(Y, K=1, max_iter=1000, constraints=None, W=None, P=None):
    if P is not None:
        raise NotImplementedError("PSF convolution not implemented!")
    B,N = Y.shape
    A = init_A(B,K)
    S = init_S(N,K)
    nmf_AS(Y, A, S, max_iter=1000, constraints=None, W=W, P=P)
