from __future__ import print_function, division
import numpy as np
from functools import partial

from . import utils

import logging

logger = logging.getLogger("proxmin")


def _copy_tuple(X):
    return tuple(item.copy() for item in X)


def _as_tuple(X):
    if type(X) in [list, tuple]:
        return X
    else:
        return (X,)


def pgm(
    X,
    grad,
    step,
    prox=None,
    accelerated=False,
    relax=None,
    e_rel=1e-6,
    max_iter=1000,
    callback=None,
):
    """Proximal Gradient Method

    Adapted from Combettes 2009, Algorithm 3.4.
    The accelerated version is Algorithm 3.6 with modifications
    from Xu & Yin (2015).

    Args:
        X: initial X, will be updated
        grad: gradient function of f wrt to X
        step: function to compute step size.
            Should be smaller than 2/L with L the Lipschitz constant of grad
            Signature: step(*X, it=None) -> float
        prox: proximal operator for penalty functions
        accelerated: if Nesterov acceleration should be used
        relax: (over)relaxation parameter, 0 < relax < 1.5
        e_rel: relative error of X sufficient for convergence
        max_iter: maximum iteration
        callback: arbitrary logging function
            Signature: callback(*X, it=None)

    Returns:
        converged: whether the optimizer has converged within e_rel
        gradient: last iteration gradients
        step: last iteration steps
    """
    # Set up: turn X and prox into tuples
    X = _as_tuple(X)
    N = len(X)
    prox = _as_tuple(prox)
    if len(prox) == 1:
        prox = prox * N
    assert len(prox) == len(X)

    if np.isscalar(e_rel):
        e_rel = (e_rel,) * N

    assert len(e_rel) == len(X)

    if relax is not None:
        assert relax > 0 and relax < 1.5

    if callback is None:
        callback = utils.NullCallback()

    # init
    stepper = utils.NesterovStepper(accelerated=accelerated)

    for it in range(max_iter):

        try:
            callback(*X, it=it)

            # use Nesterov acceleration (if omega > 0), automatically incremented
            omega = stepper.omega
            if omega > 0:
                _X = tuple(X[j] + omega * (X[j] - X_[j]) for j in range(N))
            else:
                _X = X

            # make copy for convergence test and acceleration
            X_ = _copy_tuple(X)

            # (P)GM step
            G = _as_tuple(grad(*_X))
            S = _as_tuple(step(*_X, it=it))

            for j in range(N):
                _X[j][:] -= S[j] * G[j]

                if prox[j] is not None:
                    X[j][:] = prox[j](_X[j], S[j])

                if relax is not None:
                    X[j][:] += (relax - 1) * (X[j] - X_[j])

            # test for fixed point convergence
            converged = tuple(
                utils.l2sq(X[j] - X_[j]) <= e_rel[j] ** 2 * utils.l2sq(X[j])
                for j in range(N)
            )
            if all(converged):
                break

        except StopIteration:
            break

    logger.info("Completed {0} iterations".format(it + 1))
    if not all(converged):
        logger.warning("Solution did not converge")

    return converged, G, S


def _adam_phi_psi(it, G, M, V, Vhat, b1, b2, eps, p):
    # moving averages
    M[:] = (1 - b1[it]) * G + b1[it] * M
    V[:] = (1 - b2) * (G ** 2) + b2 * V

    # bias correction
    t = it + 1
    Phi = M / (1 - b1[it] ** t)
    Psi = np.sqrt(V / (1 - b2 ** t)) + eps
    return Phi, Psi


def _amsgrad_phi_psi(it, G, M, V, Vhat, b1, b2, eps, p):
    # moving averages
    M[:] = (1 - b1[it]) * G + b1[it] * M
    V[:] = (1 - b2) * (G ** 2) + b2 * V

    Phi = M
    if Vhat is None:
        Vhat = V
    else:
        Vhat[:] = np.maximum(Vhat, V)
    # sanitize zero-gradient elements
    if eps > 0:
        Vhat = np.maximum(Vhat, eps)
    Psi = np.sqrt(Vhat)
    return Phi, Psi


def _padam_phi_psi(it, G, M, V, Vhat, b1, b2, eps, p):
    # moving averages
    M[:] = (1 - b1[it]) * G + b1[it] * M
    V[:] = (1 - b2) * (G ** 2) + b2 * V

    Phi = M
    if Vhat is None:
        Vhat = V
    else:
        Vhat[:] = np.maximum(Vhat, V)
    # sanitize zero-gradient elements
    if eps > 0:
        Vhat = np.maximum(Vhat, eps)
    Psi = Vhat ** p
    return Phi, Psi


def _adamx_phi_psi(it, G, M, V, Vhat, b1, b2, eps, p):
    # moving averages
    M[:] = (1 - b1[it]) * G + b1[it] * M
    V[:] = (1 - b2) * (G ** 2) + b2 * V

    Phi = M
    if Vhat is None:
        Vhat = V
    else:
        factor = (1 - b1[it]) ** 2 / (1 - b1[it - 1]) ** 2
        Vhat[:] = np.maximum(factor * Vhat, V)
    # sanitize zero-gradient elements
    if eps > 0:
        Vhat = np.maximum(Vhat, eps)
    Psi = np.sqrt(Vhat)
    return Phi, Psi


def _radam_phi_psi(it, G, M, V, Vhat, b1, b2, eps, p):
    rho_inf = 2 / (1 - b2) - 1

    # moving averages
    M[:] = (1 - b1[it]) * G + b1[it] * M
    V[:] = (1 - b2) * (G ** 2) + b2 * V

    # bias correction
    t = it + 1
    Phi = M / (1 - b1[it] ** t)
    rho = rho_inf - 2 * t * b2 ** t / (1 - b2 ** t)

    if rho > 4:
        Psi = np.sqrt(V / (1 - b2 ** t))
        r = np.sqrt(
            (rho - 4) * (rho - 2) * rho_inf / (rho_inf - 4) / (rho_inf - 2) / rho
        )
        Psi /= r
    else:
        Psi = np.ones(G.shape, G.dtype)
    # sanitize zero-gradient elements
    if eps > 0:
        Psi = np.maximum(Psi, np.sqrt(eps))
    return Phi, Psi


def adaprox(
    X,
    grad,
    step,
    prox=None,
    scheme="adam",
    b1=0.9,
    b2=0.999,
    eps=1e-8,
    check_convergence=True,
    p=0.25,
    e_rel=1e-6,
    max_iter=1000,
    prox_max_iter=1000,
    callback=None,
):
    """Adaptive Proximal Gradient Method

    Uses multiple variants of adaptive quasi-Newton gradient descent

        * Adam (Kingma & Ba 2015)
        * AMSGrad (Reddi, Kale & Kumar 2018)
        * PAdam (Chen & Gu 2018)
        * AdamX (Phuong & Phong 2019)
        * RAdam (Liu et al. 2019)

    and PGM sub-iterations to satisfy feasibility and optimality.

    Args:
        X: initial X, will be updated
        grad: gradient function of f wrt to X
        step: function to compute step size.
            Should be smaller than 2/L with L the Lipschitz constant of grad
            Signature: step(*X, it=None) -> float
        prox: proximal operator of penalty function
        scheme: one of ["adam", "adamx", "amsgrad", "padam","radam"]
        b1: (float or array) first moment momentum decay
        b2: second moment momentum decay
        eps: softening of second moment (only for algorithm == "adam")
        p: power of second moment (only for algorithm == "padam")
        e_rel: relative error of X
        check_convergence: whether convergence checks are performed
        max_iter: maximum iteration, irrespective of residual error
        prox_max_iter: maximum proximal sub-iteration error
        callback: arbitrary logging function
            Signature: callback(*X, it=None)

    Returns:
        converged: whether the optimizer has converged within e_rel
        gradient: last iteration gradient
        gradient2: last iteration squared gradient
    """
    X = _as_tuple(X)
    N = len(X)
    prox = _as_tuple(prox)
    if len(prox) == 1:
        prox = prox * N
    assert len(prox) == len(X)

    if np.isscalar(e_rel):
        e_rel = (e_rel,) * N
    assert len(e_rel) == len(X)

    if not hasattr(b1, "__iter__"):
        b1 = np.array((b1,) * max_iter)
    assert len(b1) == max_iter
    assert (b1 >= 0).all() and (b1 < 1).all()

    assert b2 >= 0 and b2 < 1
    assert eps >= 0
    assert p > 0 and p <= 0.5
    scheme = scheme.lower()
    assert scheme in ["adam", "adamx", "amsgrad", "padam", "radam"]

    phi_psi = {
        "adam": _adam_phi_psi,
        "amsgrad": _amsgrad_phi_psi,
        "padam": _padam_phi_psi,
        "adamx": _adamx_phi_psi,
        "radam": _radam_phi_psi,
    }

    M = [np.zeros(x.shape, x.dtype) for x in X]
    V = [np.zeros(x.shape, x.dtype) for x in X]
    Vhat = [None] * N
    Sub_iter = [0] * N

    if callback is None:
        callback = utils.NullCallback()

    for it in range(max_iter):

        try:
            callback(*X, it=it)
            G = _as_tuple(grad(*X))
            Alpha = _as_tuple(step(*X, it=it))
            if check_convergence:
                X_ = _copy_tuple(X)

            for j in range(N):
                Phi, Psi = phi_psi[scheme](
                    it, G[j], M[j], V[j], Vhat[j], b1, b2, eps, p
                )
                X[j][:] -= Alpha[j] * Phi / Psi

                if prox[j] is not None:

                    # proximal subiterations to solve for optimality
                    z = X[j].copy()
                    gamma = Alpha[j] / np.max(Psi)

                    for tau in range(1, prox_max_iter + 1):
                        z_ = prox[j](z - gamma / Alpha[j] * Psi * (z - X[j]), gamma)

                        converged = utils.l2sq(z_ - z) <= e_rel[j] ** 2 * utils.l2sq(z)
                        z = z_

                        if converged:
                            break

                    logger.debug(
                        "Proximal sub-iterations for variable {}: {}".format(j, tau)
                    )
                    Sub_iter[j] += tau

                    X[j][:] = z

            # test for fixed point convergence
            if check_convergence:
                converged = tuple(
                    utils.l2sq(X[j] - X_[j]) <= e_rel[j] ** 2 * utils.l2sq(X[j])
                    for j in range(N)
                )

                if all(converged):
                    break

        except StopIteration:
            break

    logger.info(
        "Completed {0} iterations and {1} sub-iterations".format(it + 1, Sub_iter)
    )
    if check_convergence and not all(converged):
        logger.warning("Solution did not converge")
    if not check_convergence:
        converged = (None,) * N

    return converged, M, V


def admm(
    X,
    prox_f,
    step_f,
    prox_g=None,
    step_g=None,
    L=None,
    e_rel=1e-6,
    e_abs=0,
    max_iter=1000,
    callback=None,
):
    """Alternating Direction Method of Multipliers

    This method implements the linearized ADMM from Parikh & Boyd (2014).

    Args:
        X: initial X will be updated
        prox_f: proxed function f
        step_f: function to compute step size.
            Should be smaller than 2/L with L the Lipschitz constant of grad
            Signature: step(*X, it=None) -> float
        prox_g: proxed function g
        step_g: specific value of step size for prox_g (experts only!)
            By default, set to the maximum value of step_f * ||L||_s^2.
        L: linear operator of the argument of g.
            Matrix can be numpy.array, scipy.sparse, or None (for identity).
        e_rel: relative error threshold for primal and dual residuals
        e_abs: absolute error threshold for primal and dual residuals
        max_iter: maximum iteration number, irrespective of current residuals
        callback: arbitrary logging function
            Signature: callback(*X, it=None)

    Returns:
        converged: whether the optimizer has converged within e_rel
        error: X^it - X^it-1

    Reference:
        Moolekamp & Melchior, Algorithm 1 (arXiv:1708.09066)
    """

    # use matrix adapter for convenient & fast notation
    _L = utils.MatrixAdapter(L)

    # init
    Z, U = utils.initZU(X, _L)
    it = 0
    slack = 1.0

    if callback is None:
        callback = utils.NullCallback()

    while it < max_iter:

        callback(X, it=it)

        step_f_ = slack * step_f(X, it=it)

        # get/check compatible step size for g
        if prox_g is not None and step_g is None:
            step_g_ = utils.get_step_g(step_f_, _L.spectral_norm)
        else:
            step_g_ = step_g

        # Update the variables, return LX and primal/dual residual
        LX, R, S = utils.update_variables(X, Z, U, prox_f, step_f_, prox_g, step_g_, _L)

        # convergence criteria, adapted from Boyd 2011, Sec 3.3.1
        converged, error = utils.check_constraint_convergence(
            X, _L, LX, Z, U, R, S, step_f, step_g, e_rel, e_abs
        )

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
                    Z, U = utils.initZU(X, _L)
                    logger.info("Restarting with step size slack = %.3f" % slack)
            X_ = X.copy()
            R_ = R

    logger.info("Completed {0} iterations".format(it + 1))
    if not converged:
        logger.warning("Solution did not converge")

    return converged, error


def sdmm(
    X,
    prox_f,
    step_f,
    proxs_g=None,
    steps_g=None,
    Ls=None,
    e_rel=1e-6,
    e_abs=0,
    max_iter=1000,
    callback=None,
):
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
        callback: arbitrary logging function
            Signature: callback(*X, it=None)

    Returns:
        converged: whether the optimizer has converged within e_rel
        error: X^it - X^it-1

    See also:
        algorithms.admm

    Reference:
        Moolekamp & Melchior, Algorithm 2 (arXiv:1708.09066)
    """

    # fall-back to simple ADMM
    if proxs_g is None or not hasattr(proxs_g, "__iter__"):
        return admm(
            X,
            prox_f,
            step_f,
            prox_g=proxs_g,
            step_g=steps_g,
            L=Ls,
            e_rel=e_rel,
            max_iter=max_iter,
            callback=callback,
        )

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
    Z, U = utils.initZU(X, _L)
    it, omega = 0, 0
    slack = 1.0

    if callback is None:
        callback = utils.NullCallback()

    while it < max_iter:

        callback(X, it=it)

        step_f_ = slack * step_f(X, it=it)

        # get/check compatible step size for g
        # get/check compatible step size for g
        if steps_g is None:
            steps_g_ = [
                utils.get_step_g(step_f_, _L[i].spectral_norm, M=M) for i in range(M)
            ]
        else:
            steps_g_ = steps_g

        # update the variables
        LX, R, S = utils.update_variables(
            X, Z, U, prox_f, step_f_, proxs_g, steps_g_, _L
        )

        # convergence criteria, adapted from Boyd 2011, Sec 3.3.1
        converged, errors = utils.check_constraint_convergence(
            X, _L, LX, Z, U, R, S, step_f_, steps_g_, e_rel, e_abs
        )

        if converged:
            break

        it += 1

        # if X and primal residual does not change: decrease step_f and step_g, and restart
        if it > 1:
            if (X == X_).all() and all([(R[i] == R_[i]).all() for i in range(M)]):
                slack /= 2

                # re-init
                it = 0
                Z, U = utils.initZU(X, _L)
                logger.info("Restarting with step size slack = %.3f" % slack)

        R_ = R
        X_ = X.copy()

    logger.info("Completed {0} iterations".format(it + 1))
    if not converged:
        logger.warning("Solution did not converge")

    return converged


def bsdmm(
    X,
    proxs_f,
    steps_f_cb,
    proxs_g=None,
    steps_g=None,
    Ls=None,
    update_order=None,
    steps_g_update="steps_f",
    max_iter=1000,
    e_rel=1e-6,
    e_abs=0,
    callback=None,
):
    """Block-Simultaneous Method of Multipliers.

    This method is an extension of the linearized SDMM, i.e. ADMM for multiple
    constraints, for functions f with several arguments.
    The function f needs to be proper convex in every argument.
    It performs a block optimization for each argument while propagating the
    changes to other arguments.

    Args:
        X: list of initial Xs, will be updated
        proxs_f: proxed function f
            Signature prox(X,step, Xs=None, j=None) -> X'
        steps_f_cb: callback function to compute step size for proxs_f[j]
            Signature: steps_f_cb(Xs, j=None) -> Reals
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
        callback: arbitrary logging function
            Signature: callback(*X, it=None)

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
    steps_g_update = steps_g_update.lower()
    assert steps_g_update in ["steps_f", "fixed", "relative"]

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
        # assert len(update_order) == N
        pass

    if steps_g_update == "steps_f":
        if steps_g is not None:
            logger.debug("Setting steps_g = None for update strategy 'steps_f'.")
            steps_g = None
    if steps_g_update in ["fixed", "relative"]:
        if steps_g is None:
            logger.debug(
                "Ignoring steps_g update strategy '%s' because steps_g is None."
                % steps_g_update
            )
            steps_g_update = "steps_f"

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
            _L.append([utils.MatrixAdapter(Ls[j][m]) for m in range(M[j])])

    # Initialization
    Z, U = [], []
    LX, R, S = [None] * N, [None] * N, [None] * N
    for j in range(N):
        Zj, Uj = utils.initZU(X[j], _L[j])
        Z.append(Zj)
        U.append(Uj)

    # containers
    converged, errors = [None] * N, [None] * N
    slack = [1.0] * N
    it = 0

    if callback is None:
        callback = utils.NullCallback()

    while it < max_iter:

        callback(*X, it=it)

        # iterate over blocks X_j
        for j in update_order:
            proxs_f_j = partial(proxs_f, j=j, Xs=X)
            steps_f_j = steps_f_cb(X, j=j) * slack[j]

            # update steps_g relative to change of steps_f ...
            if steps_g_update == "relative":
                for i in range(M[j]):
                    steps_g[j][i] *= steps_f_j / steps_f[j]
            steps_f[j] = steps_f_j
            # ... or update them as required by the most conservative limit
            if steps_g_update == "steps_f":
                for i in range(M[j]):
                    steps_g_[j][i] = utils.get_step_g(
                        steps_f[j], _L[j][i].spectral_norm, N=N, M=M[j]
                    )

            # update the variables
            LX[j], R[j], S[j] = utils.update_variables(
                X[j], Z[j], U[j], proxs_f_j, steps_f[j], proxs_g[j], steps_g_[j], _L[j]
            )

            # convergence criteria, adapted from Boyd 2011, Sec 3.3.1
            converged[j], errors[j] = utils.check_constraint_convergence(
                X[j],
                _L[j],
                LX[j],
                Z[j],
                U[j],
                R[j],
                S[j],
                steps_f[j],
                steps_g_[j],
                e_rel[j],
                e_abs[j],
            )

        it += 1

        if all(converged):
            break

    logger.info("Completed {0} iterations".format(it))
    if not all(converged):
        logger.warning("Solution did not converge")

    return converged
