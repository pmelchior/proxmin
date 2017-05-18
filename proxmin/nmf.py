from __future__ import print_function, division
import logging
import numpy as np

logging.basicConfig()
logger = logging.getLogger("proxmin.nmf")

def delta_data(A, S, Y, W=1):
    return W*(np.dot(A,S) - Y)

def grad_likelihood_A(A, S, Y, W=1):
    D = delta_data(A, S, Y, W=W)
    return np.dot(D, S.T)

def grad_likelihood_S(S, A, Y, W=1):
    D = delta_data(A, S, Y, W=W)
    return np.dot(A.T, D)

# executes one proximal step of likelihood gradient, followed by prox_g
def prox_likelihood_A(A, step, S=None, Y=None, prox_g=None, W=1):
    return prox_g(A - step*grad_likelihood_A(A, S, Y, W=W), step)

def prox_likelihood_S(S, step, A=None, Y=None, prox_g=None, W=1):
    return prox_g(S - step*grad_likelihood_S(S, A, Y, W=W), step)

# split X into K components along axis
# apply prox_list[k] to each component k
# stack results to reconstruct shape of X
def prox_components(X, step, prox_list=[], axis=0):
    assert X.shape[axis] == len(prox_list)
    K = X.shape[axis]

    if np.isscalar(step):
        step = [step for k in range(K)]

    if axis == 0:
        Pk = [prox_list[k](X[k], step[k]) for k in range(K)]
    if axis == 1:
        Pk = [prox_list[k](X[:,k], step[k]) for k in range(K)]
    return np.stack(Pk, axis=axis)

def dot_components(C, X, axis=0, transpose=False):
    assert X.shape[axis] == len(C)
    K = X.shape[axis]

    if axis == 0:
        if not transpose:
            CX = [C[k].dot(X[k]) for k in range(K)]
        else:
            CX = [C[k].T.dot(X[k]) for k in range(K)]
    if axis == 1:
        if not transpose:
            CX = [C[k].dot(X[:,k]) for k in range(K)]
        else:
            CX = [C[k].T.dot(X[:,k]) for k in range(K)]
    return np.stack(CX, axis=axis)


def nmf(Y, A0, S0, prox_A, prox_S, prox_S2=None, M2=None, lM2=None,
        max_iter=1000, W=None, e_rel=1e-3, algorithm='ADMM',
        outer_max_iter=50, min_iter=10):

    K = S0.shape[0]
    A = A0.copy()
    S = S0.copy()
    S_ = S0.copy() # needed for convergence test
    beta = 1. # 0.75    # TODO: unclear how to chose 0 < beta <= 1

    if W is not None:
        W_max = W.max()
    else:
        W = W_max = 1

    all_errors = []
    all_norms = []
    for it in range(outer_max_iter):
        # A: simple gradient method; need to rebind S each time
        prox_like_A = partial(prox_likelihood_A, S=S, Y=Y, prox_g=prox_A, W=W)
        step_A = beta**it / lipschitz_const(S) / W_max
        it_A = APGM(A, prox_like_A, step_A, max_iter=max_iter)

        # S: either gradient or ADMM, depending on additional constraints
        prox_like_S = partial(prox_likelihood_S, A=A, Y=Y, prox_g=prox_S, W=W)
        step_S = beta**it / lipschitz_const(A) / W_max
        if prox_S2 is None or algorithm == "APGM":
            it_S = APGM(S_, prox_like_S, step_S, max_iter=max_iter)
            errors = []
        elif algorithm == "ADMM":
            # steps set to upper limit per component
            step_S2 = step_S * lM2
            it_S, S_, _, _, errors = ADMM(S_, prox_like_S, step_S, prox_S2, step_S2, A=M2,
                                          max_iter=max_iter, e_rel=e_rel)
        elif algorithm == "SDMM":
            # TODO: Check that we are properly setting the step size.
            # Currently I am just using the same form as ADMM, with a slightly modified
            # lM2 in nmf_deblender
            step_S2 = step_S * lM2
            it_S, S_, _, _, errors = SDMM(S_, prox_like_S, step_S, prox_S2, step_S2,
                                          constraints=M2, max_iter=max_iter, e_rel=e_rel)
        else:
            raise Exception("Unexpected 'algorithm' to be 'APGM', 'ADMM', or 'SDMM'")

        logger.info("{0} {1} {2} {3} {4} {5}".format(it, step_A, it_A, step_S, it_S,
                                                     [(S[i,:] > 0).sum()for i in range(S.shape[0])]))

        if it_A == 0 and it_S == 0:
            break

        ## Convergence crit from Langville 2014, section 5 ?
        NMF_converge, norms = check_NMF_convergence(it, S_, S, e_rel, K, min_iter)
        all_errors += errors
        all_norms.append(norms)

        # Store norms and errors

        if NMF_converge:
            break

        S[:,:] = S_[:,:]
    S[:,:] = S_[:,:]
    return A,S, [all_norms, all_errors]
