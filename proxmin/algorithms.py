from __future__ import print_function, division
import logging
import numpy as np
from functools import partial

from . import utils

logging.basicConfig()
logger = logging.getLogger("proxmin")

def pgm(X0, prox_f, step_f, accelerated=False, relax=None, e_rel=1e-6, max_iter=1000, traceback=False):
    """Proximal Gradient Method

    Adapted from Combettes 2009, Algorithm 3.4.
    The accelerated version is Algorithm 3.6 with modifications
    from Xu & Yin (2015).

    Args:
        X0: initial X
        prox_f: proxed function f (the forward-backward step)
        step_f: step size, < 1/L with L being the Lipschitz constant of grad f
        accelerated: If Nesterov acceleration should be used
        relax: (over)relaxation parameter, < 1.5
        e_rel: relative error of X
        max_iter: maximum iteration, irrespective of residual error
        traceback: whether a record of all optimization variables is kept

    Returns:
        X: optimized value
        X, trace: adds utils.Traceback if traceback is True

    See also:
        utils.AcceleratedProxF
    """

    # init
    X = X0.copy()
    Xk = X.copy()
    if accelerated:
        prox_f_ = utils.AcceleratedProxF(prox_f)
    else:
        prox_f_ = prox_f

    if relax is not None:
        assert relax < 1.5

    if traceback:
        tr = utils.Traceback()
        tr.update_history(0, X=X, step_f=step_f)
        if accelerated:
            tr.update_history(0, omega=prox_f_.omega)
        if relax is not None:
            tr.update_history(0, relax=relax)

    for it in range(max_iter):

        X = prox_f_(X, step_f)

        if relax is not None:
            X += (relax-1)*(X - Xk)

        if traceback:
            tr.update_history(it+1, X=X, step_f=step_f)
            if accelerated:
                tr.update_history(it+1, omega=prox_f_.omega)
            if relax is not None:
                tr.update_history(it+1, relax=relax)

        # test for fixed point convergence
        if utils.l2sq(X - Xk) <= e_rel**2*utils.l2sq(X):
            break

        Xk = X.copy()

    if it+1 == max_iter:
        logger.warning("Solution did not converge")
    logger.info("Completed {0} iterations".format(it+1))

    if not traceback:
        return X
    else:
        return X, tr


def admm(X0, prox_f, step_f, prox_g=None, step_g=None, L=None, accelerated=False, e_rel=1e-6,
         e_abs=0, max_iter=1000, traceback=False):
    """Alternating Direction Method of Multipliers

    This method implements the linearized ADMM from Parikh & Boyd (2014).

    Args:
        X0: initial X
        prox_f: proxed function f
        step_f: step size for prox_f
        prox_g: proxed function g
        step_g: specific value of step size for prox_g (experts only!)
            By default, set to the maximum value of step_f * ||L||_s^2.
        L: linear operator of the argument of g.
            Matrix can be numpy.array, scipy.sparse, or None (for identity).
        accelerated: If Nesterov acceleration should be used
        e_rel: relative error threshold for primal and dual residuals
        e_abs: absolute error threshold for primal and dual residuals
        max_iter: maximum iteration number, irrespective of current residuals
        traceback: whether a record of all optimization variables is kept

    Returns:
        X: optimized value
        X, trace: adds utils.Traceback if traceback is True

    See also:
        utils.AcceleratedProxF

    Reference:
        Moolekamp & Melchior, Algorithm 1 (arXiv:1708.09066)
    """

    # use matrix adapter for convenient & fast notation
    _L = utils.MatrixAdapter(L)
    # get/check compatible step size for g
    if prox_g is not None and step_g is None:
        step_g = utils.get_step_g(step_f, _L.spec_norm)

    # init
    X,Z,U = utils.initXZU(X0, _L)
    Xk = X.copy()
    it = 0

    if accelerated and prox_g is None:
        prox_f_ = utils.AcceleratedProxF(prox_f)
    else:
        if accelerated:
            logger.debug("Ignoring acceleration because of prox_g")
            accelerated = False
        prox_f_ = prox_f

    if traceback:
        tr = utils.Traceback()
        tr.update_history(it, X=X, step_f=step_f, Z=Z, U=U, R=np.zeros_like(Z),
                          S=np.zeros_like(X), step_g=step_g)
        if accelerated:
            tr.update_history(it, omega=prox_f_.omega)

    while it < max_iter:

        # Update the variables, return LX and primal/dual residual
        LX, R, S = utils.update_variables(X, Z, U, prox_f_, step_f, prox_g, step_g, _L)

        # Optionally store the variables in the history
        if traceback:
            tr.update_history(it+1, X=X, step_f=step_f, Z=Z, U=U, R=R, S=S,  step_g=step_g)
            if accelerated:
                tr.update_history(it+1, omega=prox_f_.omega)

        # convergence criteria, adapted from Boyd 2011, Sec 3.3.1
        convergence, error = utils.check_constraint_convergence(X, _L, LX, Z, U, R, S,
                                                                step_f, step_g, e_rel, e_abs)

        if convergence:
            break

        it += 1

        # if X and primal residual does not change: decrease step_f and step_g, and restart
        if prox_g is not None:
            if it > 1:
                if (X == Xk).all() and (R == Rk).all():
                    step_f /= 2
                    step_g /= 2
                    # re-init
                    it = 0
                    tr.reset()

                    X,Z,U  = utils.initXZU(X0, _L)
                    if accelerated:
                        prox_f_.t = 1.
                        prox_f_.omega = 0.
                    logger.info("Restarting with step_f = %.3f" % step_f)
                    tr.update_history(it, X=X, Z=Z, U=U, R=np.zeros_like(Z), S=np.zeros_like(X),
                                      step_f=step_f, step_g=step_g)
                    if accelerated:
                        tr.update_history(it, omega=prox_f_.omega)


            Rk = R
        Xk = X.copy() # used for acceleration and non-convergence test

    if it+1 == max_iter:
        logger.warning("Solution did not converge")
    logger.info("Completed {0} iterations".format(it+1))

    if not traceback:
        return X
    else:
        return X, tr


def sdmm(X0, prox_f, step_f, proxs_g=None, steps_g=None, Ls=None, e_rel=1e-6, e_abs=0, max_iter=1000,
         traceback=False):
    """Simultaneous-Direction Method of Multipliers

    This method is an extension of the linearized ADMM for multiple constraints.

    Args:
        X0: initial X
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
        traceback: whether a record of all optimization variables is kept

    Returns:
        X: optimized value
        X, trace: adds utils.Traceback if traceback is True

    See also:
        algorithms.admm

    Reference:
        Moolekamp & Melchior, Algorithm 2 (arXiv:1708.09066)
    """

    # fall-back to simple ADMM, try acceleration as it doesn't harm
    if proxs_g is None or not hasattr(proxs_g, '__iter__'):
        return admm(X0, prox_f, step_f, prox_g=proxs_g, step_g=steps_g, L=Ls, accelerated=True, e_rel=e_rel,
                    max_iter=max_iter, traceback=traceback)

    # from here on we know that proxs_g is a list
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
    for i in range(M):
        _L.append(utils.MatrixAdapter(Ls[i]))
        # get/check compatible step size for g
        if steps_g[i] is None:
            steps_g[i] = utils.get_step_g(step_f, _L[i].spec_norm, M=M)

    # Initialization
    X,Z,U = utils.initXZU(X0, _L)
    Xk = X.copy()
    it, omega = 0, 0

    if traceback:
        tr = utils.Traceback()
        tr.update_history(it, X=X, step_f=step_f, omega=omega)
        tr.update_history(it, M=M, Z=Z, U=U, R=np.zeros_like(Z),
                          S=[np.zeros_like(X) for n in range(M)], steps_g=steps_g)

    while it < max_iter:

        # update the variables
        LX, R, S = utils.update_variables(X, Z, U, prox_f, step_f, proxs_g, steps_g, _L)

        if traceback:
            tr.update_history(it+1, X=X, step_f=step_f, omega=omega)
            tr.update_history(it+1, M=M, Z=Z, U=U, R=R, S=S, steps_g=steps_g)

        # convergence criteria, adapted from Boyd 2011, Sec 3.3.1
        convergence, errors = utils.check_constraint_convergence(X, _L, LX, Z, U, R, S, step_f, steps_g,
                                                                 e_rel, e_abs)

        if convergence:
            break

        it += 1

        # if X and primal residual does not change: decrease step_f and step_g, and restart
        if it > 1:
            if (X == Xk).all() and all([(R[i] == Rk[i]).all() for i in range(M)]):
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
                logger.info("Restarting with step_f = %.3f" % step_f)

        Rk = R
        Xk = X.copy()

    if it+1 == max_iter:
        logger.warning("Solution did not converge")
    logger.info("Completed {0} iterations".format(it+1))

    if not traceback:
        return X
    else:
        return X, tr


def bsdmm(X0s, proxs_f, steps_f_cb, proxs_g=None, steps_g=None, Ls=None, accelerated=False, update='cascade',
          update_order=None, steps_g_update='steps_f', max_iter=1000, e_rel=1e-6, e_abs=0, traceback=False):
    """Block-Simultaneous Method of Multipliers.

    This method is an extension of the linearized SDMM, i.e. ADMM for multiple
    constraints, for functions f with several arguments.
    The function f needs to be proper convex in every argument.
    It performs a block optimization for each argument while propagating the
    changes to other arguments.

    Args:
        X0s: list of initial Xs
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
        accelerated: If Nesterov acceleration should be used for each variable
        update: update sequence between the blocks
            'cascade': proxs_f are evaluated sequentually,
                       update for X_j^{k+1} is aware of X_l^{k+1} for l < j.
            'block':   proxs_f are evaluated independently
                       i.e. update for X_j^{k+1} is aware of X_l^k forall l.
        update_order: list of components to update in desired order.
                      Only relevant if update=='cascade'.
        steps_g_update: relation between steps_g and steps_f (experts only!)
            'steps_f':  update steps_g as required by most conservative limit
            'fixed':    never update initial value of steps_g
            'relative': update initial values of steps_g propertional to changes
                        of steps_f
        e_rel: relative error threshold for primal and dual residuals
        e_abs: absolute error threshold for primal and dual residuals
        max_iter: maximum iteration number, irrespective of current residuals
        traceback: whether a record of all optimization variables is kept

    Returns:
        Xs: list of optimized values
        X, trace: adds utils.Traceback if traceback is True

    Warning:
        Because of the potentially large list of optimization variables,
        setting traceback=True may exhaust memory. It should thus be run
        with a sufficiently small max_iter.

    See also:
        algorithms.sdmm, utils.AcceleratedProxF

    Reference:
        Moolekamp & Melchior, Algorithm 3 (arXiv:1708.09066)
    """
    # Set up
    N = len(X0s)

    if accelerated and proxs_g is not None:
        logger.debug("Ignoring acceleration because of proxs_g")
        accelerated = False
    if accelerated:
        proxs_f_acc = [utils.AcceleratedProxF(None) for j in range(N)]

    # allow proxs_g to be None
    if proxs_g is None:
        proxs_g = [proxs_g] * N
    if(len(proxs_g) != N):
        raise ValueError("len(proxs_g)={0} != N={1}".format(len(proxs_g), N))

    if np.isscalar(e_rel):
        e_rel = [e_rel] * N
    if np.isscalar(e_abs):
        e_abs = [e_abs] * N
    steps_f = [None] * N

    assert update.lower() in ['cascade', 'block']
    assert steps_g_update.lower() in ['steps_f', 'fixed', 'relative']

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
    X, Z, U = [],[],[]
    Xk = []
    LX, R, S = [None] * N, [None] * N, [None] * N
    for j in range(N):
        Xj, Zj, Uj = utils.initXZU(X0s[j], _L[j])
        X.append(Xj)
        Z.append(Zj)
        U.append(Uj)
        Xk.append(Xj.copy())

    # containers
    convergence, errors = [None] * N, [None] * N
    slack = [1.] * N
    it = 0
    delta, l, Lmax = 0.999, 0, np.empty(N)

    if traceback:
        tr = utils.Traceback(N)
        for j in update_order:
            if M[j]>0:
                _S = [np.zeros_like(X[j]) for n in range(M[j])]
            else:
                _S = np.zeros_like(X[j])
            tr.update_history(it, j=j, X=X[j], steps_f=steps_f[j])
            tr.update_history(it, j=j, M=M[j], steps_g=steps_g_[j], Z=Z[j], U=U[j], R=np.zeros_like(Z[j]),
                              S=[np.zeros_like(X[j]) for n in range(M[j])])
            if accelerated:
                tr.update_history(it, j=j, omega=proxs_f_acc[j].omega)

    while it < max_iter:

        # cascading or blocking updates?
        if update.lower() == 'block':
            X_ = [X[j].copy() for j in range(N)]
        else:
            X_ = X

        # iterate over blocks X_j
        for j in update_order:
            proxs_f_j = partial(proxs_f, j=j, Xs=X_)
            steps_f_j = steps_f_cb(j, X_) * slack[j]

            if accelerated:
                proxs_f_acc[j].prox_f = proxs_f_j
                proxs_f_j_ = proxs_f_acc[j]
                # making sure that no block gets 1/Lipschitz speeds:
                # Xu & Yin (2015, eq. 3.5)
                #Lmax[j] = max(l, 1./steps_f_j)
                #proxs_f_acc[j].omega = min(proxs_f_acc[j].omega, delta*np.sqrt(l/Lmax[j]))
                #print (it,j,"%.4f\t%.4f\t%.4f" % (proxs_f_acc[j].omega, l, 1./steps_f_j))
                # need to save omega here because it'll be updated in __call__
                if traceback:
                    tr.update_history(it+1, j=j, omega=proxs_f_acc[j].omega)
            else:
                proxs_f_j_ = proxs_f_j

            # update steps_g relative to change of steps_f ...
            if steps_g_update.lower() == 'relative':
                for i in range(M[j]):
                    steps_g[j][i] *= steps_f_j / steps_f[j]
            steps_f[j] = steps_f_j
            # ... or update them as required by the most conservative limit
            if steps_g_update.lower() == 'steps_f':
                for i in range(M[j]):
                    steps_g_[j][i] = utils.get_step_g(steps_f[j], _L[j][i].spec_norm, N=N, M=M[j])

            # update the variables
            LX[j], R[j], S[j] = utils.update_variables(X_[j], Z[j], U[j], proxs_f_j_, steps_f[j], proxs_g[j],
                                                       steps_g_[j], _L[j])

            # convergence criteria, adapted from Boyd 2011, Sec 3.3.1
            convergence[j], errors[j] = utils.check_constraint_convergence(X_[j], _L[j], LX[j], Z[j], U[j],
                R[j], S[j], steps_f[j],steps_g_[j],e_rel[j], e_abs[j])
            # Optionally update the new state
            if traceback:
                tr.update_history(it+1, j=j, X=X_[j], steps_f=steps_f[j])
                tr.update_history(it+1, j=j, M=M[j], steps_g=steps_g_[j], Z=Z[j], U=U[j], R=R[j], S=S[j])
                if accelerated:
                    tr.update_history(it+1, omega=proxs_f_j_.omega)


        if update.lower() == 'block':
            for j in range(N):
                X[j] = X_[j]

        if all(convergence):
            break

        it += 1
        Xk = [X[j].copy() for j in range(N)]
        l = Lmax.min()

    if it+1 >= max_iter:
        logger.warning("Solution did not converge")
    logger.info("Completed {0} iterations".format(it+1))

    if not traceback:
        return X
    else:
        return X, tr
