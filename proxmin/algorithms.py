from __future__ import print_function, division
import logging
import numpy as np
from functools import partial

from . import utils

logging.basicConfig()
logger = logging.getLogger("proxmin.algorithms")

def pgm(X0, prox_f, step_f, relax=1.49, e_rel=1e-6, max_iter=1000, traceback=False):
    """Proximal Gradient Method

    Adapted from Combettes 2009, Algorithm 3.4
    """
    X = X0.copy()
    Z = X0.copy()

    history = []
    for it in range(max_iter):

        # Optionally store the current state
        if traceback:
            history.append(X)

        _X = prox_f(Z, step_f)
        Z = X + relax*(_X - X)
        # test for fixed point convergence
        if utils.l2sq(X - _X) <= e_rel**2*utils.l2sq(X):
            X = _X
            break

        X = _X

    if it+1 == max_iter:
        logger.warning("Solution did not converge")
    logger.info("Completed {0} iterations".format(it+1))

    if not traceback:
        return X
    else:
        tr = utils.Traceback(it=it, history=history)
        return X, tr


def apgm(X0, prox_f, step_f, e_rel=1e-6, max_iter=1000, traceback=False):
    """Accelerated Proximal Gradient Method

    Adapted from Combettes 2009, Algorithm 3.6
    """
    X = X0.copy()
    Z = X0.copy()
    t = 1.
    history = []
    for it in range(max_iter):

        # Optionally store the current state
        if traceback:
            history.append(X)

        _X = prox_f(Z, step_f)
        t_ = 0.5*(1 + np.sqrt(4*t*t + 1))
        gamma = 1 + (t - 1)/t_
        Z = X + gamma*(_X - X)
        # test for fixed point convergence
        if utils.l2sq(X - _X) <= e_rel**2*utils.l2sq(X):
            X = _X
            break

        t = t_
        X = _X

    if it+1 == max_iter:
        logger.warning("Solution did not converge")
    logger.info("Completed {0} iterations".format(it+1))

    if not traceback:
        return X
    else:
        tr = utils.Traceback(it=it, history=history)
        return X, tr


def admm(X0, prox_f, step_f, prox_g, step_g, L=None, e_rel=1e-6, max_iter=1000, traceback=False):

    """Alternating Direction Method of Multipliers

    Adapted from Parikh and Boyd (2009).
    """

    # use matrix adapter for convenient & fast notation
    _L = utils.MatrixOrNone(L)
    # determine spectral norm of matrix
    norm_L2 = utils.get_spectral_norm(_L)
    # get/check compatible step size for g
    step_g = utils.get_step_g(step_f, norm_L2, step_g=step_g)

    # init
    X,Z,U = utils.initXZU(X0, _L)

    errors = []
    history = []
    for it in range(max_iter):

        # Optionally store the current state
        if traceback:
            history.append(X.copy())

        # Update the variables, return LX and primal/dual residual
        LX, R, S = utils.update_variables(X, Z, U, prox_f, step_f, prox_g, step_g, _L)
        # convergence criteria, adapted from Boyd 2011, Sec 3.3.1
        convergence, error = utils.check_constraint_convergence(_L, LX, Z, U, R, S, e_rel)
        # if primal residual does not improve: decrease step_f and step_g, and restart
        if it > 0:
            if (R == R_).all():
                step_f /= 2
                step_g = utils.get_step_g(step_f, norm_L2, step_g=step_g)
                X,Z,U  = utils.initXZU(X0, _L)
                logger.warning("Restarting with step_f = %.3f" % step_f)
        R_ = R

        # store the errors
        if traceback:
            errors.append(error)

        if convergence:
            break

    if it+1 == max_iter:
        logger.warning("Solution did not converge")
    logger.info("Completed {0} iterations".format(it+1))

    if not traceback:
        return X
    else:
        tr = utils.Traceback(it=it, Z=Z, U=U, errors=errors, history=history)
        return X, tr


def sdmm(X0, prox_f, step_f, proxs_g, steps_g, Ls=None, e_rel=1e-6, max_iter=1000, traceback=False):

    """Implement Simultaneous-Direction Method of Multipliers

    This implements the SDMM algorithm derived from Algorithm 7.9 from Combettes and Pesquet (2009),
    Section 4.4.2 in Parikh and Boyd (2013), and Eq. 2.2 in Andreani et al. (2007).

    In Combettes and Pesquet (2009) they use a matrix inverse to solve the problem.
    In our case that is the inverse of a sparse matrix, which is no longer sparse and too
    costly to implement.
    The `scipy.sparse.linalg` module does have a method to solve a sparse matrix equation,
    using Algorithm 7.9 directly still does not yield the correct result,
    as the treatment of penalties due to constraints are on equal footing with our likelihood
    proximal operator and require a significant change in the way we calculate step sizes to converge.

    Instead we calculate the constraint vectors (as in SDMM) but extend the update of the ``X`` matrix
    using a modified version of the ADMM X update function (from Parikh and Boyd, 2009),
    using an augmented Lagrangian for multiple linear constraints as given in Andreani et al. (2007).

    """
    if not hasattr(proxs_g, "__iter__"):
        proxs_g = [proxs_g]
    if not hasattr(steps_g, "__iter__"):
        steps_g = [steps_g]
    if not hasattr(Ls, "__iter__"):
        Ls = [Ls]
    M = len(proxs_g)
    assert len(steps_g) == M
    assert len(Ls) == M

    # get/check compatible step sizes for g
    # use matrix adapter for convenient & fast notation
    _L = []
    norm_L2 = []
    for i in range(M):
        _L.append(utils.MatrixOrNone(Ls[i]))
        norm_L2.append(utils.get_spectral_norm(_L[i]))
        # get/check compatible step size for g
        steps_g[i] = utils.get_step_g(step_f, norm_L2[i], step_g=steps_g[i])

    # Initialization
    X = X0.copy()
    Z = []
    U = []
    for i in range(M):
        Z.append(_L[i].dot(X).copy())
        U.append(np.zeros_like(Z[i]))
    all_errors = []
    history = []

    for it in range(max_iter):
        # Optionally store the current state
        if traceback:
            history.append(X.copy())

        # update the variables
        LX, R, S = utils.update_variables(X, Z, U, prox_f, step_f, proxs_g, steps_g, _L)
        # convergence criteria, adapted from Boyd 2011, Sec 3.3.1
        convergence, errors = utils.check_constraint_convergence(_L, LX, Z, U, R, S, e_rel)

        if traceback:
            all_errors.append(errors)

        if convergence:
            break

    if it+1 == max_iter:
        logger.warning("Solution did not converge")
    logger.info("Completed {0} iterations".format(it+1))

    if not traceback:
        return X
    else:
        tr = utils.Traceback(it=it, Z=Z, U=U, errors=all_errors, history=history)
        return X, tr


def glmm(X0s, proxs_f, steps_f_cb, proxs_g, steps_g, Ls, min_iter=10, max_iter=1000, e_rel=1e-6, traceback=False):
    """General Linearized Method of Multipliers.

    TODO: proxs_f must have signature prox(X,step, j=None, Xs=None)
    TODO: steps_f_cb(j=None, Xs=None) -> Reals
    """
    # Set up
    N = len(X0s)
    if np.isscalar(e_rel):
        e_rel = [e_rel] * N
    M = [0] * N
    steps_f = [None] * N
    assert len(proxs_g) == N
    assert len(steps_g) == N
    assert len(Ls) == N
    for j in range(N):
        if not hasattr(proxs_g[j], "__iter__"):
            proxs_g[j] = [proxs_g[j]]
        M[j] = len(proxs_g[j])
        if not hasattr(steps_g[j], "__iter__"):
            steps_g[j] = [steps_g[j]]
        if not hasattr(Ls[j], "__iter__"):
            Ls[j] = [Ls[j]]
        assert len(steps_g[j]) == M[j]
        assert len(Ls[j]) == M[j]

    # use matrix adapters
    _L = [[ utils.MatrixOrNone(Ls[j][i]) for i in range(M[j])] for j in range(N)]
    norm_L2 = [[ utils.get_spectral_norm(_L[j][i]) for i in range(M[j])] for j in range(N)]

    # Initialization
    X = [X0s[j].copy() for j in range(N)]
    Z = []
    U = []
    LX = [None] * N
    R = [None] * N
    S = [None] * N
    for j in range(N):
        Z.append([])
        U.append([])
        for i in range(M[j]):
            Z[j].append(_L[j][i].dot(X[j]).copy())
            U[j].append(np.zeros_like(Z[j][i]))
    history = []
    all_errors = []
    convergence = [None] * N
    errors = [None] * N

    for it in range(max_iter):
        # Optionally store the current state
        if traceback:
            history.append([X[j].copy() for j in range(N)])

        # get compatible step sizes for f and g
        for j in range(N):
            steps_f[j] = steps_f_cb(j=j, Xs=X)
            for i in range(M[j]):
                steps_g[j][i] = utils.get_step_g(steps_f[j], norm_L2[j][i], step_g=steps_g[j][i])

            # update the variables
            proxs_f_j = partial(proxs_f, j=j, Xs=X)
            LX[j], R[j], S[j] = utils.update_variables(X[j], Z[j], U[j], proxs_f_j, steps_f[j], proxs_g[j], steps_g[j], _L[j])
            # convergence criteria, adapted from Boyd 2011, Sec 3.3.1
            convergence[j], errors[j] = utils.check_constraint_convergence(_L[j], LX[j], Z[j], U[j], R[j], S[j], e_rel[j])

            # TODO: do we need a X - X_ convergence criterion?
            # If so, we need X_ above
            """
            # Convergence crit from Langville 2014, section 5
            iter_norms = []
            likelihood_convergence = []
            for n,Xk_ in enumerate(_allXk):
                convergence, norms = utils.check_convergence(it, Xk_, allXk[n], e_rel[n], min_iter)
                iter_norms.append(norms)
                likelihood_convergence.append(convergence)
            """

        if traceback:
            all_errors.append(errors)

        if np.all(convergence) and it >= min_iter:
            break

    if it+1 == max_iter:
        logger.warning("Solution did not converge")
    logger.info("Completed {0} iterations".format(it+1))

    if not traceback:
        return X
    else:
        tr = utils.Traceback(it=it, Z=Z, U=U, errors=all_errors, history=history)
        return X, tr
