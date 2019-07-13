from __future__ import print_function, division
import numpy as np
from functools import partial

from . import utils

import logging
logger = logging.getLogger("proxmin")

def _copy_tuple(X):
    return tuple(item.copy() for item in X)

def pgm(X, grad, step, prox=None, accelerated=False, relax=None, e_rel=1e-6, max_iter=1000, callback=None):
    """Proximal Gradient Method

    Adapted from Combettes 2009, Algorithm 3.4.
    The accelerated version is Algorithm 3.6 with modifications
    from Xu & Yin (2015).

    Args:
        X: initial X, will be updated
        grad: gradient function of f wrt to X
        step: function to compute step size.
            Should be smaller than 2/L with L the Lipschitz constant of grad
            Signature: step(*X, it=it) -> float
        prox: proximal operator for penalty functions
        accelerated: if Nesterov acceleration should be used
        relax: (over)relaxation parameter, 0 < relax < 1.5
        e_rel: relative error of X sufficient for convergence
        max_iter: maximum iteration
        callback: arbitrary logging function
            Signature: callback(*X, it=it)

    Returns:
        converged: whether the optimizer has converged within e_rel
        error: X^it - X^it-1
    """
    # Set up: turn X and prox into tuples
    if type(X) not in (list, tuple):
        X = (X,)
    if type(prox) not in (list, tuple):
        prox = (prox,)

    N = len(X)
    if np.isscalar(e_rel):
        e_rel = (e_rel,) * N

    assert len(prox) == len(X)
    assert len(e_rel) == len(X)

    if relax is not None:
        assert relax > 0 and relax < 1.5

    # init
    stepper = utils.NesterovStepper(accelerated=accelerated)

    for it in range(max_iter):

        if callback is not None:
            callback(*X, it=it)

        # use Nesterov acceleration (if omega > 0), automatically incremented
        omega = stepper.omega
        if omega > 0:
            _X = tuple(X[j] + omega*(X[j] - X_[j]) for j in range(N))
        else:
            _X = X

        # make copy for convergence test and acceleration
        X_ = _copy_tuple(X)

        # (P)GM step
        G = grad(*_X)
        S = step(*_X, it=it)

        for j in range(N):
            _X[j][:] -= S[j] * G[j]

            if prox[j] is not None:
                X[j][:] = prox[j](_X[j], S[j])

            if relax is not None:
                X[j][:] += (relax-1)*(X[j] - X_[j])

        # test for fixed point convergence
        errors = tuple(X[j] - X_[j] for j in range(N))
        converged = tuple(utils.l2sq(errors[j]) <= e_rel[j]**2*utils.l2sq(X[j]) for j in range(N))
        if all(converged):
            break

    logger.info("Completed {0} iterations".format(it+1))
    if not all(converged):
        logger.warning("Solution did not converge")

    return converged, errors


def adam(X, grad, step, prox=None, algorithm="adam", b1=0.9, b2=0.999, eps=10**-8, p=0.25, e_rel=1e-6, max_iter=1000, callback=None):
    """Proximal Adam and variants

    Adam (Kingma & Ba 2015)
    AMSGrad (Reddi, Kale & Kumar 2018)
    PAdam (Chen & Gu 2018)
    AdamX (Phuong & Phong 2019)

    Uses sub-iterations to satisfy penalty.

    Args:
        X: initial X, will be updated
        grad: gradient function of f wrt to X
        step: function to compute step size.
            Should be smaller than 2/L with L the Lipschitz constant of grad
            Signature: step(X, it) -> float
        prox: proximal operator of penalty function
        algorithm: one of ["adam", "adamx", "amsgrad", "padam"]
        b1: (float or array) first moment momentum decay
        b2: second moment momentum decay
        eps: softening of second moment (only for algorithm == "adam")
        p: power of econd moment (only for algorithm == "padam")
        e_rel: relative error of X
        max_iter: maximum iteration, irrespective of residual error
        traceback: utils.Traceback to hold variable histories

    Returns:
        converged: whether the optimizer has converged within e_rel
        error: X^it - X^it-1
    """
    if not hasattr(b1, '__iter__'):
        b1 = np.array([b1,] * max_iter)

    assert len(b1) == max_iter
    assert (b1 >= 0).all() and (b1 < 1).all()
    assert b2 >= 0 and b2 < 1
    assert eps >= 0
    assert p > 0 and p <= 0.5
    assert algorithm in ["adam", "adamx", "amsgrad", "padam"]

    m = np.zeros(X.shape, X.dtype)
    v = np.zeros(X.shape, X.dtype)
    vhat = np.zeros(X.shape, X.dtype)

    for it in range(max_iter):

        if callback is not None:
            callback(X, it)

        X_ = X.copy()
        g = grad(X)
        s = step(X, it)

        m = (1 - b1[it]) * g + b1[it] * m
        v = (1 - b2) * (g**2) + b2 * v

        if algorithm == "adam":
            s *= np.sqrt(1 - b2**(it + 1)) / (1 - b1[it]**(it+1))
            vhat = v
        else:
            factor = 1
            if it > 0 and algorithm == "adamx":
                factor = (1 - b1[it])**2 / (1 - b1[it-1])**2
            vhat = np.maximum(v, factor * vhat)

        if algorithm == "padam":
            denom = vhat**p
        else:
            denom = np.sqrt(vhat)

        if algorithm == "adam":
            denom += eps

        X -= s * m / denom

        prox_it = 0
        if prox is not None:
            h = np.sqrt(vhat)
            # projected gradients
            beta = np.max(h**2)
            gamma = 1 / beta
            Z = X.copy()
            for prox_it in range(1, max_iter):
                # h-metric norm
                Z_ = prox(Z - gamma * h * (Z - X), gamma)

                if utils.l2sq(Z_ - Z) <= e_rel**2*utils.l2sq(Z):
                    break

                Z = Z_

            X[:] = Z

        converged = utils.l2sq(X_ - X) <= e_rel**2*utils.l2sq(X)
        if converged:
            break

    logger.info("Completed {0} iterations".format(it+1))
    if not converged:
        logger.warning("Solution did not converge")

    return converged, X-X_


def admm(X, prox_f, step_f, prox_g=None, step_g=None, L=None, e_rel=1e-6, e_abs=0, max_iter=1000, callback=None):
    """Alternating Direction Method of Multipliers

    This method implements the linearized ADMM from Parikh & Boyd (2014).

    Args:
        X: initial X will be updated
        prox_f: proxed function f
        step_f: function to compute step size.
            Should be smaller than 2/L with L the Lipschitz constant of grad
            Signature: step(X, it) -> float
        prox_g: proxed function g
        step_g: specific value of step size for prox_g (experts only!)
            By default, set to the maximum value of step_f * ||L||_s^2.
        L: linear operator of the argument of g.
            Matrix can be numpy.array, scipy.sparse, or None (for identity).
        e_rel: relative error threshold for primal and dual residuals
        e_abs: absolute error threshold for primal and dual residuals
        max_iter: maximum iteration number, irrespective of current residuals
        traceback: utils.Traceback to hold variable histories

    Returns:
        converged: whether the optimizer has converged within e_rel
        error: X^it - X^it-1

    Reference:
        Moolekamp & Melchior, Algorithm 1 (arXiv:1708.09066)
    """

    # use matrix adapter for convenient & fast notation
    _L = utils.MatrixAdapter(L)

    # init
    Z,U = utils.initZU(X, _L)
    it = 0
    slack = 1.

    while it < max_iter:

        if callback is not None:
            callback(X, it)

        step_f_ = slack * step_f(X, it)

        # get/check compatible step size for g
        if prox_g is not None and step_g is None:
            step_g_ = utils.get_step_g(step_f_, _L.spectral_norm)
        else:
            step_g_ = step_g

        # Update the variables, return LX and primal/dual residual
        LX, R, S = utils.update_variables(X, Z, U, prox_f, step_f_, prox_g, step_g_, _L)

        # convergence criteria, adapted from Boyd 2011, Sec 3.3.1
        converged, error = utils.check_constraint_convergence(X, _L, LX, Z, U, R, S,
                                                                step_f, step_g, e_rel, e_abs)

        if converged:
            break

        it += 1

        # if X and primal residual does not change: decrease step_f and step_g, and restart
        if prox_g is not None:
            if it > 1:
                if (X == X_).all() and (R == R_).all():
                    slack /= 2

                    # re-init
                    it = 0
                    Z,U  = utils.initZU(X, _L)
                    logger.info("Restarting with step size slack = %.3f" % slack)
            X_ = X.copy()
            R_ = R

    logger.info("Completed {0} iterations".format(it+1))
    if not converged:
        logger.warning("Solution did not converge")

    return converged, error


def sdmm(X, prox_f, step_f, proxs_g=None, steps_g=None, Ls=None, e_rel=1e-6, e_abs=0, max_iter=1000, callback=None):
    """Simultaneous-Direction Method of Multipliers

    This method is an extension of the linearized ADMM for multiple constraints.

    Args:
        X: initial X, will be updated
        prox_f: proxed function f
        step_f: step size for prox_f
        proxs_g: list of proxed functions
        steps_g: specific value of step size for proxs_g (experts only!)
            If set, needs to have same format as proxs_g.
            By default, set to the maximum value of step_f * ||L_i||_s^2.
        Ls: linear operators of the argument of g_i.
            If set, needs to have same format as proxs_g.
            Matrices can be numpy.array, scipy.sparse, or None (for identity).
        e_rel: relative error threshold for primal and dual residuals
        e_abs: absolute error threshold for primal and dual residuals
        max_iter: maximum iteration number, irrespective of current residuals
        traceback: utils.Traceback to hold variable histories

    Returns:
        converged: whether the optimizer has converged within e_rel
        error: X^it - X^it-1

    See also:
        algorithms.admm

    Reference:
        Moolekamp & Melchior, Algorithm 2 (arXiv:1708.09066)
    """

    # fall-back to simple ADMM
    if proxs_g is None or not hasattr(proxs_g, '__iter__'):
        return admm(X, prox_f, step_f, prox_g=proxs_g, step_g=steps_g, L=Ls, e_rel=e_rel, max_iter=max_iter, callback=callback)

    # from here on we know that proxs_g is a list
    M = len(proxs_g)

    # if Ls are None or single: create M duplicates
    if not hasattr(Ls, "__iter__"):
        Ls = [Ls] * M
    assert len(Ls) == M

    # get/check compatible step sizes for g
    # use matrix adapter for convenient & fast notation
    _L = []
    for i in range(M):
        _L.append(utils.MatrixAdapter(Ls[i]))

    # Initialization
    Z,U = utils.initZU(X, _L)
    it, omega = 0, 0
    slack = 1.

    while it < max_iter:

        if callback is not None:
            callback(X, it)

        step_f_ = slack * step_f(X, it)

        # get/check compatible step size for g
        # get/check compatible step size for g
        if steps_g is None:
            steps_g_ = [ utils.get_step_g(step_f_, _L[i].spectral_norm, M=M) for i in range(M) ]
        else:
            steps_g_ = steps_g

        # update the variables
        LX, R, S = utils.update_variables(X, Z, U, prox_f, step_f_, proxs_g, steps_g_, _L)

        # convergence criteria, adapted from Boyd 2011, Sec 3.3.1
        converged, errors = utils.check_constraint_convergence(X, _L, LX, Z, U, R, S, step_f_, steps_g_, e_rel, e_abs)

        if converged:
            break

        it += 1

        # if X and primal residual does not change: decrease step_f and step_g, and restart
        if it > 1:
            if (X == X_).all() and all([(R[i] == R_[i]).all() for i in range(M)]):
                slack /= 2

                # re-init
                it = 0
                Z,U  = utils.initZU(X, _L)
                logger.info("Restarting with step size slack = %.3f" % slack)

        R_ = R
        X_ = X.copy()

    logger.info("Completed {0} iterations".format(it+1))
    if not converged:
        logger.warning("Solution did not converge")

    return converged, errors


def bpgm(X, proxs_f, steps_f_cb, update_order=None, accelerated=False, relax=None, max_iter=1000, e_rel=1e-6, callback=None):
    """Block Proximal Gradient Method.

    Also know as Alternating Proximal Gradient Method, it performs Proximal
    gradient (forward-backward) updates on each block in alternating fashion.

    Args:
        X: list of initial Xs, will be updated
        proxs_f: proxed function f
            Signature prox(X,step, j=None, Xs=None) -> X'
        step_f: step size, < 1/L with L being the Lipschitz constant of grad f
        steps_f_cb: callback function to compute step size for proxs_f[j]
            Signature: steps_f_cb(j, Xs) -> Reals
        update_order: list of components to update in desired order.
        accelerated: If Nesterov acceleration should be used for all variables
        relax: (over)relaxation parameter for each variable, < 1.5
        e_rel: relative error of X
        max_iter: maximum iteration, irrespective of residual error
        traceback: utils.Traceback to hold variable histories

    Returns:
        converged: whether the optimizer has converged within e_rel
        error: X^it - X^it-1
    """
    # Set up
    N = len(X)
    if np.isscalar(e_rel):
        e_rel = [e_rel] * N

    if relax is not None:
        assert relax > 0 and relax < 1.5

    if update_order is None:
        update_order = range(N)
    else:
        # we could check that every component is in the list
        # but one can think of cases when a component is *not* to be updated.
        #assert len(update_order) == N
        pass

    # init
    X_ = [None] * N
    stepper = utils.NesterovStepper(accelerated=accelerated)

    for it in range(max_iter):

        if callback is not None:
            callback(X, it)

        # use Nesterov acceleration (if omega > 0), automatically incremented
        omega = stepper.omega

        # iterate over blocks X_j
        for j in update_order:

            # tell prox the state of other variables
            proxs_f_j = partial(proxs_f, j=j, Xs=X)
            steps_f_j = steps_f_cb(j, X)

            # acceleration?
            # check for resizing: if resize ocurred, temporily skip acceleration
            if omega > 0 and X[j].shape == X_[j].shape:
                _X = X[j] + omega*(X[j] - X_[j])
            else:
                _X = X[j]

            # keep copy for convergence test (and acceleration)
            X_[j] = X[j].copy()

            # PGM step, force inline update
            X[j][:] = proxs_f_j(_X, steps_f_j)

            if relax is not None:
                X[j] += (relax-1)*(X[j] - X_[j])


        # test for fixed point convergence
        # allowing for transparent resizing of X: need to check shape of X_
        errors = [X[j] - X_[j] if X[j].shape == X_[j].shape else X[j] for j in range(N)]
        converged = [utils.l2sq(errors[j]) <= e_rel[j]**2*utils.l2sq(X[j]) for j in range(N)]
        if all(converged):
            break

    logger.info("Completed {0} iterations".format(it+1))
    if not all(converged):
        logger.warning("Solution did not converge")

    return converged, errors


def bsdmm(X, proxs_f, steps_f_cb, proxs_g=None, steps_g=None, Ls=None, update_order=None, steps_g_update='steps_f', max_iter=1000, e_rel=1e-6, e_abs=0, callback=None):
    """Block-Simultaneous Method of Multipliers.

    This method is an extension of the linearized SDMM, i.e. ADMM for multiple
    constraints, for functions f with several arguments.
    The function f needs to be proper convex in every argument.
    It performs a block optimization for each argument while propagating the
    changes to other arguments.

    Args:
        X: list of initial Xs, will be updated
        proxs_f: proxed function f
            Signature prox(X,step, j=None, Xs=None) -> X'
        steps_f_cb: callback function to compute step size for proxs_f[j]
            Signature: steps_f_cb(j, Xs) -> Reals
        proxs_g: list of proxed functions
            [[prox_X0_0, prox_X0_1...],[prox_X1_0, prox_X1_1,...],...]
        steps_g: specific value of step size for proxs_g (experts only!)
            If set, needs to have same format as proxs_g.
            By default, set to the maximum value of step_f_j * ||L_i||_s^2.
        Ls: linear operators of the argument of g_i.
            If set, needs to have same format as proxs_g.
            Matrices can be numpy.array, scipy.sparse, or None (for identity).
        update_order: list of components to update in desired order.
        steps_g_update: relation between steps_g and steps_f (experts only!)
            'steps_f':  update steps_g as required by most conservative limit
            'fixed':    never update initial value of steps_g
            'relative': update initial values of steps_g propertional to changes
                        of steps_f
        e_rel: relative error threshold for primal and dual residuals
        e_abs: absolute error threshold for primal and dual residuals
        max_iter: maximum iteration number, irrespective of current residuals
        traceback: utils.Traceback to hold variable histories

    Returns:
        converged: whether the optimizer has converged within e_rel
        error: X^it - X^it-1

    Warning:
        Because of the potentially large list of optimization variables,
        setting traceback may exhaust memory. It should thus be run
        with a sufficiently small max_iter.

    See also:
        algorithms.sdmm

    Reference:
        Moolekamp & Melchior, Algorithm 3 (arXiv:1708.09066)
    """

    # Set up
    N = len(X)
    if proxs_g is None:
        proxs_g = [None] * N
    assert len(proxs_g) == N
    assert steps_g_update.lower() in ['steps_f', 'fixed', 'relative']

    if np.isscalar(e_rel):
        e_rel = [e_rel] * N
    if np.isscalar(e_abs):
        e_abs = [e_abs] * N
    steps_f = [None] * N

    if update_order is None:
        update_order = range(N)
    else:
        # we could check that every component is in the list
        # but one can think of cases when a component is *not* to be updated.
        #assert len(update_order) == N
        pass

    if steps_g_update.lower() == 'steps_f':
        if steps_g is not None:
            logger.debug("Setting steps_g = None for update strategy 'steps_f'.")
            steps_g = None
    if steps_g_update.lower() in ['fixed', 'relative']:
        if steps_g is None:
            logger.debug("Ignoring steps_g update strategy '%s' because steps_g is None." % steps_g_update)
            steps_g_update = 'steps_f'

    # if steps_g / Ls are None or single: create N duplicates
    if not hasattr(steps_g, "__iter__"):
        steps_g = [steps_g] * N
    if not hasattr(Ls, "__iter__"):
        Ls = [Ls] * N
    # check for cases in which a list was given
    assert len(steps_g) == N
    assert len(Ls) == N

    M = [0] * N
    for j in range(N):
        if proxs_g[j] is not None:
            if not hasattr(proxs_g[j], "__iter__"):
                proxs_g[j] = [proxs_g[j]]
            M[j] = len(proxs_g[j])
            if not hasattr(steps_g[j], "__iter__"):
                steps_g[j] = [steps_g[j]] * M[j]
            if not hasattr(Ls[j], "__iter__"):
                Ls[j] = [Ls[j]] * M[j]
            assert len(steps_g[j]) == M[j]
            assert len(Ls[j]) == M[j]

    # need container for current-iteration steps_g and matrix adapters
    steps_g_ = []
    _L = []
    for j in range(N):
        if proxs_g[j] is None:
            steps_g_.append(None)
            _L.append(utils.MatrixAdapter(None))
        else:
            steps_g_.append([[None] for i in range(M[j])])
            _L.append([ utils.MatrixAdapter(Ls[j][m]) for m in range(M[j])])

    # Initialization
    Z, U = [],[]
    LX, R, S = [None] * N, [None] * N, [None] * N
    for j in range(N):
        Zj, Uj = utils.initZU(X[j], _L[j])
        Z.append(Zj)
        U.append(Uj)

    # containers
    converged, errors = [None] * N, [None] * N
    slack = [1.] * N
    it = 0

    while it < max_iter:

        if callback is not None:
            callback(X, it)

        # iterate over blocks X_j
        for j in update_order:
            proxs_f_j = partial(proxs_f, j=j, Xs=X)
            steps_f_j = steps_f_cb(j, X) * slack[j]

            # update steps_g relative to change of steps_f ...
            if steps_g_update.lower() == 'relative':
                for i in range(M[j]):
                    steps_g[j][i] *= steps_f_j / steps_f[j]
            steps_f[j] = steps_f_j
            # ... or update them as required by the most conservative limit
            if steps_g_update.lower() == 'steps_f':
                for i in range(M[j]):
                    steps_g_[j][i] = utils.get_step_g(steps_f[j], _L[j][i].spectral_norm, N=N, M=M[j])

            # update the variables
            LX[j], R[j], S[j] = utils.update_variables(X[j], Z[j], U[j], proxs_f_j, steps_f[j], proxs_g[j], steps_g_[j], _L[j])

            # convergence criteria, adapted from Boyd 2011, Sec 3.3.1
            converged[j], errors[j] = utils.check_constraint_convergence(X[j], _L[j], LX[j], Z[j], U[j],
                R[j], S[j], steps_f[j],steps_g_[j],e_rel[j], e_abs[j])

        if all(converged):
            break
        it += 1

    logger.info("Completed {0} iterations".format(it+1))
    if not all(converged):
        logger.warning("Solution did not converge")

    return converged, errors
