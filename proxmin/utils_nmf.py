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
    def __init__(self, slack=0.9, Wmax=1, max_stride=100, update_order=None):
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
        assert slack > 0 and slack <= 1

        self.slack = slack
        self.Wmax = Wmax
        self.max_stride = max_stride
        # need to knwo when to advance the iterations counter
        if update_order is None:
            self.advance_index = 1
        else:
            self.advance_index = update_order[-1]

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
            if j == self.advance_index:
                self.it += 1

            # increase stride when rel. changes in L are smaller than (1-slack)/2
            if self.it > 1 and self.slack < 1:
                rel_error = np.abs(self.stored[j] - L) / self.stored[j]
                budget = (1-self.slack)/2
                if rel_error < budget and rel_error > 0:
                    self.stride[j] += max(1,int(budget/rel_error * self.stride[j]))
                    self.stride[j] = min(self.max_stride, self.stride[j])
            # updated last value
            self.stored[j] = L
        elif j == self.advance_index:
            self.it += 1

        return self.slack / self.stored[j]
