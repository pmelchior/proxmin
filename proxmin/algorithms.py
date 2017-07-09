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

    if traceback:
        tr = utils.Traceback()
        tr.update_history(0, X=X, Z=Z, step_f=step_f)

    for it in range(max_iter):

        _X = prox_f(Z, step_f)
        Z = X + relax*(_X - X)

        if traceback:
            tr.update_history(it+1, X=_X, Z=Z, step_f=step_f)

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
        return X, tr


def apgm(X0, prox_f, step_f, e_rel=1e-6, max_iter=1000, traceback=False):
    """Accelerated Proximal Gradient Method

    Adapted from Combettes 2009, Algorithm 3.6
    """
    X = X0.copy()
    Z = X0.copy()
    t = 1.

    if traceback:
        tr = utils.Traceback()
        tr.update_history(0, X=X, Z=Z, step_f=step_f, t=t, gamma=1.)

    for it in range(max_iter):
        _X = prox_f(Z, step_f)
        t_ = 0.5*(1 + np.sqrt(4*t*t + 1))
        gamma = 1 + (t - 1)/t_

        if traceback:
            tr.update_history(it+1, X=_X, Z=Z, step_f=step_f, t=t_, gamma=gamma)

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
        return X, tr


def admm(X0, prox_f, step_f, prox_g, step_g=None, L=None, e_rel=1e-6, max_iter=1000, traceback=False):

    """Alternating Direction Method of Multipliers

    Adapted from Parikh and Boyd (2009).
    """

    # use matrix adapter for convenient & fast notation
    _L = utils.MatrixAdapter(L)
    # determine spectral norm of matrix
    norm_L2 = utils.get_spectral_norm(_L.L)
    # get/check compatible step size for g
    step_g = utils.get_step_g(step_f, norm_L2, step_g=step_g)

    # init
    X,Z,U = utils.initXZU(X0, _L)
    it = 0

    if traceback:
        tr = utils.Traceback()
        tr.update_history(it, X=X, Z=Z, U=U, R=np.zeros_like(Z), S=np.zeros_like(X),
                          step_f=step_f, step_g=step_g)

    while it < max_iter:
        # Update the variables, return LX and primal/dual residual
        LX, R, S = utils.update_variables(X, Z, U, prox_f, step_f, prox_g, step_g, _L)

        # Optionally store the variables in the history
        if traceback:
            tr.update_history(it+1, X=X, Z=Z, U=U, R=R, S=S, step_f=step_f, step_g=step_g)

        # convergence criteria, adapted from Boyd 2011, Sec 3.3.1
        convergence, error = utils.check_constraint_convergence(_L, LX, Z, U, R, S, e_rel)

        if convergence:
            break

        it += 1

        # if X and primal residual does not change: decrease step_f and step_g, and restart
        if it > 1:
            if (X == X_).all() and (R == R_).all():
                step_f /= 2
                step_g /= 2
                # re-init
                it = 0
                tr.reset()

                X,Z,U  = utils.initXZU(X0, _L)
                logger.warning("Restarting with step_f = %.3f" % step_f)
                tr.update_history(it, X=X, Z=Z, U=U, R=np.zeros_like(Z), S=np.zeros_like(X),
                                  step_f=step_f, step_g=step_g)
        R_ = R
        X_ = X.copy()

    if it+1 == max_iter:
        logger.warning("Solution did not converge")
    logger.info("Completed {0} iterations".format(it+1))

    if not traceback:
        return X
    else:
        return X, tr


def sdmm(X0, prox_f, step_f, proxs_g, steps_g=None, Ls=None, e_rel=1e-6, max_iter=1000, traceback=False):

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
    M = len(proxs_g)

    # if steps_g / Ls are None or single: create M duplicates
    if not hasattr(steps_g, "__iter__"):
        steps_g = [steps_g] * M
    if not hasattr(Ls, "__iter__"):
        Ls = [Ls] * M
    # check for cases in which a list was given
    assert len(steps_g) == M
    assert len(Ls) == M

    # get/check compatible step sizes for g
    # use matrix adapter for convenient & fast notation
    _L = []
    norm_L2 = []
    for i in range(M):
        _L.append(utils.MatrixAdapter(Ls[i]))
        norm_L2.append(utils.get_spectral_norm(_L[i].L))
        # get/check compatible step size for g
        steps_g[i] = utils.get_step_g(step_f, norm_L2[i], step_g=steps_g[i], M=M)

    # Initialization
    X,Z,U = utils.initXZU(X0, _L)
    it = 0

    if traceback:
        tr = utils.Traceback()
        tr.update_history(it, X=X, step_f=step_f)
        tr.update_history(it, M=M, Z=Z, U=U, R=np.zeros_like(Z),
                          S=[np.zeros_like(X) for n in range(M)], steps_g=steps_g)

    while it < max_iter:
        # update the variables
        LX, R, S = utils.update_variables(X, Z, U, prox_f, step_f, proxs_g, steps_g, _L)

        # Optionally update the new state
        if traceback:
            tr.update_history(it+1, X=X, step_f=step_f)
            tr.update_history(it+1, M=M, Z=Z, U=U, R=R, S=S, steps_g=steps_g)

        # convergence criteria, adapted from Boyd 2011, Sec 3.3.1
        convergence, errors = utils.check_constraint_convergence(_L, LX, Z, U, R, S, e_rel)

        if convergence:
            break

        it += 1

        # if X and primal residual does not change: decrease step_f and step_g, and restart
        if it > 1:
            if (X == X_).all() and all([(R[i] == R_[i]).all() for i in range(M)]):
                step_f /= 2
                for i in range(M):
                    steps_g[i] /= 2

                # re-init
                it = 0
                tr.reset()

                X,Z,U  = utils.initXZU(X0, _L)
                tr.update_history(it, X=X, step_f=step_f)
                tr.update_history(it, M=M, Z=Z, U=U, R=np.zeros_like(Z),
                                  S=[np.zeros_like(X) for n in range(M)], steps_g=steps_g)
                logger.warning("Restarting with step_f = %.3f" % step_f)

        R_ = R
        X_ = X.copy()

    if it+1 == max_iter:
        logger.warning("Solution did not converge")
    logger.info("Completed {0} iterations".format(it+1))

    if not traceback:
        return X
    else:
        return X, tr


def glmm(X0s, proxs_f, steps_f_cb, proxs_g, steps_g=None, Ls=None,
         max_iter=1000, e_rel=1e-6, traceback=False, steps_g_update='steps_f'):
    """General Linearized Method of Multipliers.

    proxs_f must have signature prox(X,step, j=None, Xs=None)
    steps_f_cb(j, Xs) -> Reals
    steps_g_update in ['steps_f', 'fixed', 'relative']:
        steps_f:  update steps_g as required by the most conservative limit (recommended)
        fixed:    never updated initial value of steps_g
        relative: update initial values of steps_g propertional to changes of steps_f
    """
    # Set up
    N = len(X0s)
    assert len(proxs_g) == N

    if np.isscalar(e_rel):
        e_rel = [e_rel] * N
    steps_f = [None] * N

    # if steps_g / Ls are None or single: create N duplicates
    if steps_g_update.lower() == 'steps_f':
        if steps_g is not None:
            logger.warning("Setting steps_g = None for update strategy 'steps_f'.")
            steps_g = None
    if steps_g_update.lower() in ['fixed', 'relative']:
        if steps_g is None:
            logger.warning("Ignoring steps_g update strategy '%s' because steps_g is None." % steps_g_update)
            steps_g_update = 'steps_f'

    if not hasattr(steps_g, "__iter__"):
        steps_g = [steps_g] * N
    if not hasattr(Ls, "__iter__"):
        Ls = [Ls] * N
    # check for cases in which a list was given
    assert len(steps_g) == N
    assert len(Ls) == N

    M = [0] * N
    for j in range(N):
        if not hasattr(proxs_g[j], "__iter__"):
            proxs_g[j] = [proxs_g[j]]
        M[j] = len(proxs_g[j])
        if not hasattr(steps_g[j], "__iter__"):
            steps_g[j] = [steps_g[j]] * M[j]
        if not hasattr(Ls[j], "__iter__"):
            Ls[j] = [Ls[j]] * M[j]
        assert len(steps_g[j]) == M[j]
        assert len(Ls[j]) == M[j]

    # need container for current-iteration steps_g
    steps_g_ = [[[None] for i in range(M[j])] for j in range(N)]

    # use matrix adapters
    _L = [[ utils.MatrixAdapter(Ls[n][m]) for m in range(M[n])] for n in range(N)]
    norm_L2 = [[ utils.get_spectral_norm(_L[n][m].L) for m in range(M[n])] for n in range(N)]

    # Initialization
    X, Z, U = [],[],[]
    LX, R, S = [None] * N, [None] * N, [None] * N
    for j in range(N):
        Xj, Zj, Uj = utils.initXZU(X0s[j], _L[j])
        X.append(Xj)
        Z.append(Zj)
        U.append(Uj)
    # containers
    convergence, errors = [None] * N, [None] * N
    slack = [1.] * N
    it = 0

    if traceback:
        tr = utils.Traceback(N)
        for j in range(N):
            tr.update_history(it, j=j, X=X[j], steps_f=steps_f[j])
            tr.update_history(it, j=j, M=M[j], steps_g=steps_g_[j], Z=Z[j], U=U[j],
                              R=np.zeros_like(Z[j]),
                              S=[np.zeros_like(X[j]) for n in range(M[j])])

    while it < max_iter:
        # get compatible step sizes for f and g
        for j in range(N):
            step_f_j = steps_f_cb(j, X) * slack[j]
            # update steps_g relative to change of steps_f ...
            if steps_g_update.lower() == 'relative':
                for i in range(M[j]):
                    steps_g[j][i] *= step_f_j / steps_f[j]
            steps_f[j] = step_f_j
            # ... or update them as required by the most conservative limit
            if steps_g_update.lower() == 'steps_f':
                for i in range(M[j]):
                    steps_g_[j][i] = utils.get_step_g(steps_f[j], norm_L2[j][i], N=N, M=M[j])

            # update the variables
            proxs_f_j = partial(proxs_f, j=j, Xs=X)
            LX[j], R[j], S[j] = utils.update_variables(X[j], Z[j], U[j], proxs_f_j, steps_f[j],
                                                       proxs_g[j], steps_g_[j], _L[j])
            # convergence criteria, adapted from Boyd 2011, Sec 3.3.1
            convergence[j], errors[j] = utils.check_constraint_convergence(_L[j], LX[j], Z[j], U[j],
                                                                           R[j], S[j], e_rel[j])
            # Optionally update the new state
            if traceback:
                tr.update_history(it+1, j=j, X=X[j], steps_f=steps_f[j])
                tr.update_history(it+1, j=j, M=M[j], steps_g=steps_g_[j], Z=Z[j], U=U[j], R=R[j], S=S[j])

            # TODO: do we need a X - X_ convergence criterion?
            """
            # Convergence crit from Langville 2014, section 5
            iter_norms = []
            likelihood_convergence = []
            for n,Xk_ in enumerate(_allXk):
                convergence, norms = utils.check_convergence(it, Xk_, allXk[n], e_rel[n], min_iter)
                iter_norms.append(norms)
                likelihood_convergence.append(convergence)
            """

        if all(convergence):
            break

        it += 1

        # if X and primal residual does not change: decrease step_f and step_g, and restart
        if it > 1:
            # perform step size update for each Xj independently
            for j in range(N):
                if (X[j] == X_[j]).all() and all([(R[j][i] == R_[j][i]).all() for i in range(M[j])]):
                    slack[j] /= 2

                    # re-init
                    it = 0

                    X[j],Z[j],U[j]  = utils.initXZU(X0s[j], _L[j])
                    logger.warning("Restarting with step_f[%d] = %.3f" % (j,steps_f[j]*slack[j]))

                    # TODO: Traceback needs to have offset for each j!
                    tr.reset()
                    tr.update_history(it, X=X, step_f=step_f)
                    tr.update_history(it, M=M[j], Z=Z, U=U, R=np.zeros_like(Z),
                                      S=[np.zeros_like(X) for n in range(M)], steps_g=steps_g)

        R_ = R
        X_ = [X[j].copy() for j in range(N)]

    if it+1 >= max_iter:
        logger.warning("Solution did not converge")
    logger.info("Completed {0} iterations".format(it+1))

    if not traceback:
        return X
    else:
        return X, tr
