from __future__ import print_function, division
import numpy as np
import scipy.sparse, scipy.sparse.linalg
from . import operators
from . import utils
from . import algorithms

import logging

logger = logging.getLogger("proxmin")


def log_likelihood(*X, Y=0, W=1):
    """Log-likelihood of NMF assuming Gaussian error model.

    Args:
        X:  tuple of (A,S) matrix factors
        Y:  target matrix
        W: (optional weight matrix MxN)

    Returns:
        float
    """
    A, S = X
    return np.sum(W * (Y - A.dot(S)) ** 2) / 2


def grad_likelihood(*X, Y=0, W=1):
    """Gradient of the log-likelihood of NMF assuming Gaussian error model.

    Args:
        X:  tuple of (A,S) matrix factors
        Y:  target matrix
        W: (optional weight matrix MxN)

    Returns:
        grad_A f, grad_S f
    """
    A, S = X
    D = W * (A.dot(S) - Y)
    return D.dot(S.T), A.T.dot(D)


def step_A(A, S):
    return 1 / utils.get_spectral_norm(S.T)


def step_S(A, S):
    return 1 / utils.get_spectral_norm(A)


def step_pgm(*X, it=None, W=1):
    """Step sizes for PGM of NMF assuming Gaussian error model.

    Args:
        X:  tuple of (A,S) matrix factors
        it:  iteration counter
        W: (optional weight matrix MxN)

    Returns:
        step_A, step_S
    """
    A, S = X
    if W is 1:
        return step_A(A, S), step_S(A, S)
    else:
        C, K = A.shape
        K, N = S.shape
        Sigma_1 = scipy.sparse.diags(W.flatten())

        # Lipschitz constant for grad_A = || S Sigma_1 S.T||_s
        PS = scipy.sparse.block_diag([S.T for c in range(C)])
        SSigma_1S = PS.T.dot(Sigma_1.dot(PS))
        LA = np.real(
            scipy.sparse.linalg.eigs(SSigma_1S, k=1, return_eigenvectors=False)[0]
        )

        # Lipschitz constant for grad_S = || A.T Sigma_1 A||_s
        PA = scipy.sparse.bmat(
            [[scipy.sparse.identity(N) * A[c, k] for k in range(K)] for c in range(C)]
        )
        ASigma_1A = PA.T.dot(Sigma_1.dot(PA))
        LS = np.real(
            scipy.sparse.linalg.eigs(ASigma_1A, k=1, return_eigenvectors=False)[0]
        )
        LA, LS

        return 1 / LA, 1 / LS


def step_adaprox(*X, it=None):
    A, S = X
    return (np.mean(A, axis=0) / 10, S.mean(axis=1)[:, None] / 10)


def nmf(
    Y,
    A,
    S,
    W=1,
    prox_A=operators.prox_plus,
    prox_S=operators.prox_plus,
    algorithm=algorithms.pgm,
    step=None,
    max_iter=1000,
    e_rel=1e-3,
    callback=None,
    **algorithm_args
):
    """Non-negative matrix factorization.

    This method solves the matrix factorization problem
        minimize || Y - AS ||_2^2
    under an arbitrary number of constraints on A and/or S.

    Args:
        Y: target matrix MxN
        A: initial amplitude matrix MxK, will be updated
        S: initial source matrix KxN, will be updated
        W: (optional weight matrix MxN)
        prox_A: direct contraint of A
        prox_S: direct constraint of S
        algorithm: an algorithm from `proxmin.algorithms`
        step: function to compute step sizes for A and S
            Signature: step(*X, it)
        max_iter: maximum iteration number, irrespective of current residuals
        e_rel: relative error threshold for primal and dual residuals
        callback: arbitrary logging function
            Signature: callback(*X, it=None)
        algorithm_args: further arguments passed to the algorithm

    Returns:
        return arguments of algorithm
        A, S are updated inline

    Reference:
        Moolekamp & Melchior, 2018 (arXiv:1708.09066)

    """

    assert algorithm in [algorithms.pgm, algorithms.adaprox, algorithms.bsdmm]

    # setup
    from functools import partial

    grad = partial(grad_likelihood, Y=Y, W=W)
    X = [A, S]
    prox = [prox_A, prox_S]

    if algorithm is algorithms.pgm:
        if step is None:
            step = partial(step_pgm, W=W)
        return algorithm(
            X,
            grad,
            step,
            prox=prox,
            max_iter=max_iter,
            e_rel=e_rel,
            callback=callback,
            **algorithm_args
        )

    if algorithm is algorithms.adaprox:
        if step is None:
            step = step_adaprox
        return algorithm(
            X,
            grad,
            step,
            prox=prox,
            max_iter=max_iter,
            e_rel=e_rel,
            callback=callback,
            **algorithm_args
        )

    if algorithm is algorithms.bsdmm:

        # transform gradient steps into prox
        def prox_f(X, step, Xs=None, j=None):
            # that's a bit of a waste since we compute all gradients
            grads = grad(*Xs)
            # ...but only use one
            return prox[j](X - step * grads[j], step)

        if step is None:
            step_ = partial(step_pgm, W=W)

            def step_f(Xs, j=None):
                return step_(*Xs)[j]

            step = step_f

        return algorithms.bsdmm(
            X,
            prox_f,
            step_f,
            max_iter=max_iter,
            e_rel=e_rel,
            callback=callback,
            **algorithm_args
        )
