from __future__ import print_function, division
import numpy as np
from . import operators
from . import utils
from . import algorithms

import logging
logger = logging.getLogger("proxmin")

def grad_likelihood(*X, Y=0, W=1):
    A, S = X
    D = W*(A.dot(S) - Y)
    return D.dot(S.T), A.T.dot(D)

def step(*X, it=None):
    A, S = X
    return 1./utils.get_spectral_norm(S.T), 1./utils.get_spectral_norm(A)

def nmf(Y, A, S, W=1, prox_A=operators.prox_plus, prox_S=operators.prox_plus, proxs_g=None, steps_g=None, Ls=None, slack=0.9, steps_g_update='steps_f', max_iter=1000, e_rel=1e-3, e_abs=0, callback=None):
    """Non-negative matrix factorization.

    This method solves the NMF problem
        minimize || Y - AS ||_2^2
    under an arbitrary number of constraints on A and/or S.

    Args:
        Y:  target matrix MxN
        A: initial amplitude matrix MxK, will be updated
        S: initial source matrix KxN, will be updated
        W: (optional weight matrix MxN)
        prox_A: direct projection contraint of A
        prox_S: direct projection constraint of S
        proxs_g: list of constraints for A or S for ADMM-type optimization
            [[prox_A_0, prox_A_1...],[prox_S_0, prox_S_1,...]]
        steps_g: specific value of step size for proxs_g (experts only!)
        Ls: list of linear operators for the constraint functions proxs_g
            If set, needs to have same format as proxs_g.
            Matrices can be numpy.array, scipy.sparse, or None (for identity).
        slack: tolerance for (re)evaluation of Lipschitz constants
            See Steps_AS() for details.
        update_order: list of factor indices in update order
            j=0 -> A, j=1 -> S
        max_iter: maximum iteration number, irrespective of current residuals
        e_rel: relative error threshold for primal and dual residuals
        e_abs: absolute error threshold for primal and dual residuals
        callback: arbitrary logging function
            Signature: callback(*X, it=None)

    Returns:
        converged: convence test for A,S
        errors: difference between latest and previous iterations for A,S

    See also:
        algorithms.bsdmm for update_order and steps_g_update
        utils.AcceleratedProxF for Nesterov acceleration

    Reference:
        Moolekamp & Melchior, 2017 (arXiv:1708.09066)

    """
    from functools import partial
    grad = partial(grad_likelihood, Y=Y, W=W)

    X = [A, S]
    prox = [prox_A, prox_S]
    # use accelerated block-PGM if there's no proxs_g
    if proxs_g is None or not utils.hasNotNone(proxs_g):
        return algorithms.pgm(X, grad, step, prox=prox, accelerated=True, max_iter=max_iter, e_rel=e_rel, callback=callback)
    else:
        return algorithms.bsdmm(X, f, steps_f, proxs_g, steps_g=steps_g, Ls=Ls, update_order=update_order, steps_g_update=steps_g_update, max_iter=max_iter, e_rel=e_rel, e_abs=e_abs, traceback=traceback)
