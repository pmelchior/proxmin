from . import utils
from . import algorithms
from . import operators

def nmf(Y, A0, S0, W=None, prox_A=operators.prox_plus, prox_S=operators.prox_plus, proxs_g=None, steps_g=None, Ls=None, slack=0.9, update_order=None, steps_g_update='steps_f', accelerated=False, max_iter=1000, e_rel=1e-3, e_abs=0, traceback=False):
    """Non-negative matrix factorization.

    This method solves the NMF problem
        minimize || Y - AS ||_2^2
    under an arbitrary number of constraints on A and/or S.

    Args:
        Y:  target matrix MxN
        A0: initial amplitude matrix MxK
        S0: initial source matrix KxN
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
        accelerated: If Nesterov acceleration should be used for A and S
        max_iter: maximum iteration number, irrespective of current residuals
        e_rel: relative error threshold for primal and dual residuals
        e_abs: absolute error threshold for primal and dual residuals
        traceback: whether a record of all optimization variables is kept

    Returns:
        A, S: updated amplitude and source matrices
        A, S, trace: adds utils.Traceback if traceback is True

    See also:
        algorithms.bsdmm for update_order and steps_g_update
        utils.AcceleratedProxF for Nesterov acceleration

    Reference:
        Moolekamp & Melchior, 2017 (arXiv:1708.09066)

    """
    from . import utils_nmf as ut

    # create stepsize callback, needs max of W
    if W is not None:
        Wmax = W.max()
    else:
        W = Wmax = 1
    steps_f = ut.Steps_AS(Wmax=Wmax, slack=slack, update_order=update_order)

    # gradient step, followed by direct application of prox_S or prox_A
    from functools import partial
    f = partial(ut.prox_likelihood, Y=Y, W=W, prox_S=prox_S, prox_A=prox_A)

    Xs = [A0, S0]
    res = algorithms.bsdmm(Xs, f, steps_f, proxs_g, steps_g=steps_g, Ls=Ls,
                           update_order=update_order, steps_g_update=steps_g_update, accelerated=accelerated,
                           max_iter=max_iter, e_rel=e_rel, e_abs=e_abs, traceback=traceback)

    if not traceback:
        return res[0], res[1]
    else:
        return res[0][0], res[0][1], res[1]
