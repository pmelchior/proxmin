from __future__ import print_function, division
import logging
import numpy as np
from . import operators
from . import utils
from . import algorithms

def delta_data(A, S, Y, W=1):
    return W*(A.dot(S) - Y)

def grad_likelihood_A(A, S, Y, W=1):
    D = delta_data(A, S, Y, W=W)
    return D.dot(S.T)

def grad_likelihood_S(S, A, Y, W=1):
    D = delta_data(A, S, Y, W=W)
    return A.T.dot(D)

# executes one proximal step of likelihood gradient, followed by prox_g
def prox_likelihood_A(A, step, S=None, Y=None, prox_g=None, W=1):
    return prox_g(A - step*grad_likelihood_A(A, S, Y, W=W), step)

def prox_likelihood_S(S, step, A=None, Y=None, prox_g=None, W=1):
    return prox_g(S - step*grad_likelihood_S(S, A, Y, W=W), step)

def prox_likelihood(X, step, Xs=None, j=None, Y=None, WA=None, WS=None, prox_S=operators.prox_id, prox_A=operators.prox_id):
    if j == 0:
        return prox_likelihood_A(X, step, S=Xs[1], Y=Y, prox_g=prox_A, W=WA)
    else:
        return prox_likelihood_S(X, step, A=Xs[0], Y=Y, prox_g=prox_S, W=WS)

class Steps_AS:
    def __init__(self, WA=1, WS=1, slack=0.1, max_stride=100):
        """Helper class to compute the Lipschitz constants of grad f.

        The __call__ function compute the spectral norms of A or S, which
        determine the Lipschitz constant of the respective update steps.

        If a weight matrix is used, the stepsize will be upper bounded by
        assuming the maximum value of the weights. In the case of varying
        weights, it is generally advised to normalize the weight matrix
        differently for the A and S updates, therefore two maximum numbers
        (WAMax, WSmax) can be set.

        Because the spectral norm is expensive to compute, it will only update
        the step_size if relative changes of L exceed slack/2.
        If not, which is usually the case after only a few iterations, it will
        report a previous value for the next several iterations. The stride
        between updates is set by
            stride -> stride * (slack/2 / rel_error
        i.e. it increases more strongly if the rel_error is much below the
        slack budget.
        """
        import scipy.sparse
        if WA is 1:
            self.WA = WA
        else:
            self.WA = scipy.sparse.diags(WA.reshape(-1))
        if WS is 1:
            self.WS = WS
        else:
            self.WS = scipy.sparse.diags(WS.reshape(-1))

        # two independent caches for Lipschitz constants
        self._cb = [utils.ApproximateCache(self._one_over_lipschitzA, slack=slack, max_stride=max_stride),
                    utils.ApproximateCache(self._one_over_lipschitzS, slack=slack, max_stride=max_stride)]

    def _one_over_lipschitzA(self, Xs):
        A,S = Xs
        if self.WA is 1:
            return 1./utils.get_spectral_norm(S.T)
        else: # full weight matrix, need to serialize S along k
            import scipy.sparse
            Ss = scipy.sparse.block_diag([S.T for b in range(len(A))])
            # Lipschitz constant for grad_A = || S Sigma_1 S.T||_s
            SSigma_1S = Ss.T.dot(self.WA.dot(Ss))
            LA = np.real(scipy.sparse.linalg.eigs(SSigma_1S, k=1, return_eigenvectors=False)[0])
            return 1./LA

    def _one_over_lipschitzS(self, Xs):
        A,S = Xs
        if self.WA is 1:
            return 1./utils.get_spectral_norm(A)
        else:
            import scipy.sparse
            N = S.shape[1]
            As = scipy.sparse.bmat([[scipy.sparse.identity(N) * A[b,k] for k in range(A.shape[1])] for b in range(A.shape[0])])
            ASigma_1A = As.T.dot(self.WS.dot(As))
            LS = np.real(scipy.sparse.linalg.eigs(ASigma_1A, k=1, return_eigenvectors=False)[0])
            return 1./LS

    def __call__(self, j, Xs):
        return self._cb[j](Xs)

def normalizeMatrix(M, axis):
    if axis == 1:
        norm = np.sum(M, axis=axis)
        norm = np.broadcast_to(norm, M.T.shape)
        norm = norm.T
    else:
        norm = np.sum(M, axis=axis)
        norm = np.broadcast_to(norm, M.shape)
    return norm

def nmf(Y, A0, S0, W=None, prox_A=operators.prox_plus, prox_S=operators.prox_plus, proxs_g=None, steps_g=None, Ls=None, slack=0.9, update_order=None, steps_g_update='steps_f', max_iter=1000, e_rel=1e-3, e_abs=0, return_errors=False, traceback=None):
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
        max_iter: maximum iteration number, irrespective of current residuals
        e_rel: relative error threshold for primal and dual residuals
        e_abs: absolute error threshold for primal and dual residuals
        return_errors: whether convergence and variable errors are returned
        traceback: utils.Traceback to hold variable histories

    Returns:
        A, S: updated amplitude and source matrices
        convergence, errors: convergence test and iteration differences (
            (only with `return_errors`)

    See also:
        algorithms.bsdmm for update_order and steps_g_update
        utils.AcceleratedProxF for Nesterov acceleration

    Reference:
        Moolekamp & Melchior, 2017 (arXiv:1708.09066)

    """

    # create stepsize callback, needs max of W
    if W is not None:
        # normalize in pixel and band directions to have similar update speeds
        WA = normalizeMatrix(W, 1)
        WS = normalizeMatrix(W, 0)
    else:
        WA = WS = 1
    steps_f = Steps_AS(WA=WA, WS=WS, slack=slack)

    # gradient step, followed by direct application of prox_S or prox_A
    from functools import partial
    f = partial(prox_likelihood, Y=Y, WA=WA, WS=WS, prox_S=prox_S, prox_A=prox_A)

    Xs = [A0, S0]
    # use accelerated block-PGM if there's no proxs_g
    update = 'cascade'
    if proxs_g is None or not utils.hasNotNone(proxs_g):
        res = algorithms.bpgm(Xs, f, steps_f, accelerated=True, update=update, update_order=update_order, max_iter=max_iter, e_rel=e_rel, traceback=traceback)
    else:
        res = algorithms.bsdmm(Xs, f, steps_f, proxs_g, steps_g=steps_g, Ls=Ls, update=update, update_order=update_order, steps_g_update=steps_g_update, max_iter=max_iter, e_rel=e_rel, e_abs=e_abs, traceback=traceback)

    if return_errors:
        return res
    else:
        return res[0]
