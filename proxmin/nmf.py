from __future__ import print_function, division
import logging
import numpy as np
from . import operators
from . import utils
from . import algorithms

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

def prox_likelihood(X, step, Xs=None, j=None, Y=None, W=None, prox_S=operators.prox_id, prox_A=operators.prox_id):
    if j == 0:
        return prox_likelihood_A(X, step, S=Xs[1], Y=Y, prox_g=prox_A, W=W)
    else:
        return prox_likelihood_S(X, step, A=Xs[0], Y=Y, prox_g=prox_S, W=W)

def steps_AS(Xs=None, j=None, Wmax=1):
    if j == 0:
        L = utils.get_spectral_norm(Xs[1].T) * Wmax  # ||S*S.T||
    else:
        L = utils.get_spectral_norm(Xs[0]) * Wmax # ||A.T * A||
    return 1./L

def nmf(Y, A0, S0, prox_A=operators.prox_plus, prox_S=None, proxs_g=None, W=None, Ls=None, l0_thresh=None, l1_thresh=None, max_iter=1000, min_iter=10, e_rel=1e-3, traceback=False):

    # for S: use non-negative or sparsity constraints directly
    from functools import partial
    if prox_S is not None:
        if l0_thresh is not None or l1_thresh is not None:
            logger.warn("Warning: l0_thresh or l1_thresh ignored because prox_S is set")
    else:
        # L0 has preference
        if l0_thresh is not None:
            if l1_thresh is not None:
                logger.warn("Warning: l1_thresh ignored in favor of l0_thresh")
            prox_S = partial(operators.prox_hard, l=l0_thresh)
        elif l1_thresh is not None:
            prox_S = partial(operators.prox_soft_plus, l=l1_thresh)
        else:
            prox_S = operators.prox_plus

    # get max of W
    if W is not None:
        Wmax = W.max()
    else:
        W = Wmax = 1

    # gradient step, followed by direct application of prox
    f = partial(prox_likelihood, Y=Y, W=W, prox_S=prox_S, prox_A=prox_A)
    steps_f = partial(steps_AS, Wmax=Wmax)

    N = 2
    # set step sizes and Ls to None
    if proxs_g is None:
        proxs_g = [[operators.prox_id]] * N
    steps_g = [[None]] * N
    if Ls is None:
        Ls = [[None]] * N

    Xs = [A0.copy(), S0.copy()]
    res = algorithms.glmm(Xs, f, steps_f, proxs_g, steps_g, Ls=Ls, max_iter=max_iter, e_rel=e_rel, traceback=traceback)

    if not traceback:
        return res[0], res[1]
    else:
        return res[0][0], res[0][1], res[1]
