from __future__ import print_function, division
import logging
import numpy as np
from . import utils

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

def prox_likelihood(X, step, Xs=None, j=None, Y=None, W=None, prox_g_S=prox_id, prox_g_A=prox_id):
    if j == 0:
        return prox_likelihood_A(X, step, S=Xs[1], Y=Y, prox_g=prox_g_A, W=W)
    else:
        return prox_likelihood_S(X, step, A=Xs[0], Y=Y, prox_g=prox_g_S, W=W)

def steps_AS(Xs=None, j=None, Wmax=1):
    if j == 0:
        L = utils.get_spectral_norm(Xs[1].T) * Wmax  # ||S*S.T||
    else:
        L = utils.get_spectral_norm(Xs[0]) * Wmax # ||A.T * A||
    return 1./L

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


def nmf(Y, A0, S0, proxs_A=None, proxs_S=None, W=None, Ls=None, l0_thresh=None, l1_thresh=None, max_iter=1000, min_iter=10, e_rel=1e-3, traceback=False):

    if W is not None:
        Wmax = W.max()
    else:
        W = Wmax = 1

    from functools import partial
    f = partial(prox_likelihood, Y=Y, W=W, prox_g_S=prox_g_S)
    steps_f = partial(steps_AS, Wmax=Wmax)

    # sanitize and merge proxs_A and proxs_S:
    # require at least non-negativity
    proxs_g = []
    if proxs_A is None:
        proxs_g.append([prox_plus])
    else:
        proxs_g.append(proxs_A)
    if proxs_S is None:
        # S: non-negativity or L0/L1 sparsity plus ...
        if l0_thresh is None and l1_thresh is None:
            proxs_g.append([prox_plus])
        else:
            # L0 has preference
            if l0_thresh is not None:
                if l1_thresh is not None:
                    logger.warn("Warning: l1_thresh ignored in favor of l0_thresh")
                prox_S = partial(prox_hard, l=l0_thresh)
            else:
                prox_S = partial(prox_soft_plus, l=l1_thresh)
    else:
        proxs_g.append(proxs_S)

    # set step sizes and Ls to None as default
    steps_g = [[None] * len(proxs_g[j]) for j in range(2)]
    if Ls is None:
        Ls = [[None] * len(proxs_g[j]) for j in range(2)]

    Xs = [A0.copy(), S0.copy()]
    res = pa.glmm(Xs, f, steps_f, proxs_g, steps_g, Ls=Ls, max_iter=max_iter, e_rel=e_rel, traceback=traceback)

    if not traceback:
        return res[0], res[1]
    else:
        return res[0][0], res[0][1], res[1]
