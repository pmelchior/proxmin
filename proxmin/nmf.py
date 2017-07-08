from __future__ import print_function, division
import logging
import numpy as np
from . import operators
from . import utils
from . import algorithms

logging.basicConfig()
logger = logging.getLogger("proxmin.nmf")

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

def prox_likelihood(X, step, Xs=None, j=None, Y=None, W=None,
                    prox_S=operators.prox_id, prox_A=operators.prox_id):
    if j == 0:
        return prox_likelihood_A(X, step, S=Xs[1], Y=Y, prox_g=prox_A, W=W)
    else:
        return prox_likelihood_S(X, step, A=Xs[0], Y=Y, prox_g=prox_S, W=W)

class Steps_AS:
    def __init__(self, slack=0.5, Wmax=1):
        """Helper class to compute the Lipschitz constants of grad f.

        Because the spectral norm is expensive to compute, it will only update
        the step_size if relative changes of L exceed (1-slack)/2.
        If not, which is usually the case after only a few iterations, it will
        report a previous value for the next several iterations. The stride
        beteen updates is set by
            stride -> stride * (1-slack)/2 / rel_error
        i.e. it increases more strongly if the rel_error is much below the
        slack budget.
        """

        self.Wmax = Wmax
        self.slack = slack
        self.it = 0
        N = 2
        self.stride = [1] * N
        self.last = [-1] * N
        self.stored = [None] * N # last update of L

    def __call__(self, j, Xs):
        if self.it >= self.last[j] + self.stride[j]:
            self.last[j] = self.it
            if j == 0:
                L = utils.get_spectral_norm(Xs[1].T) * self.Wmax  # ||S*S.T||
            else:
                L = utils.get_spectral_norm(Xs[0]) * self.Wmax # ||A.T * A||
                self.it += 1 # iteration counter

            # increase stride when rel. changes in L are smaller than (1-slack)/2
            if self.it > 1:
                rel_error = np.abs(self.stored[j] - L) / self.stored[j]
                budget = (1-self.slack)/2
                if rel_error < budget:
                    self.stride[j] += max(1,int(budget/rel_error * self.stride[j]))
            # updated last value
            self.stored[j] = L
        elif j == 1:
            self.it += 1

        return self.slack / self.stored[j]

def nmf(Y, A0, S0, W=None, prox_A=operators.prox_plus, prox_S=operators.prox_plus,
        proxs_g=None, steps_g=None, Ls=None,
        max_iter=1000, e_rel=1e-3, traceback=False):

    # create stepsize callback, needs max of W
    if W is not None:
        Wmax = W.max()
    else:
        W = Wmax = 1
    steps_f = Steps_AS(Wmax=Wmax)

    # gradient step, followed by direct application of prox_S or prox_A
    from functools import partial
    f = partial(prox_likelihood, Y=Y, W=W, prox_S=prox_S, prox_A=prox_A)

    Xs = [A0.copy(), S0.copy()]
    res = algorithms.glmm(Xs, f, steps_f, proxs_g, steps_g=steps_g, Ls=Ls,
                          max_iter=max_iter, e_rel=e_rel, traceback=traceback)

    if not traceback:
        return res[0], res[1]
    else:
        return res[0][0], res[0][1], res[1]
