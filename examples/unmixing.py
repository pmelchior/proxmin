# from proxmin import nmf
import numpy as np
import proxmin
from proxmin.utils import Traceback
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import time, sys
from functools import partial

# initialize and run NMF
import logging

logging.basicConfig()
logger = logging.getLogger("proxmin")
logger.setLevel(logging.INFO)


def generateComponent(m):
    """Creates oscillating components to be mixed"""
    freq = 25 * np.random.random()
    phase = 2 * np.pi * np.random.random()
    x = np.arange(m)
    return np.cos(x / freq - phase) ** 2


def generateAmplitudes(k):
    """Makes mixing coefficients"""
    res = np.array([np.random.random() for i in range(k)])
    return res / res.sum()


def add_noise(Y, sigma):
    """Adds noise to Y"""
    return Y + np.random.normal(0, sigma, Y.shape)


def match(A, S, trueS):
    """Rearranges columns of S to best fit the components they likely represent (maximizes sum of correlations)"""
    cov = np.cov(trueS, S)
    k = S.shape[0]
    corr = np.zeros([k, k])
    for i in range(k):
        for j in range(k):
            corr[i][j] = cov[i + k][j] / np.sqrt(cov[i + k][i + k] * cov[j][j])
    arrangement = linear_sum_assignment(-corr)
    resS = np.zeros_like(S)
    resAT = np.zeros_like(A.T)
    for t in range(k):
        resS[arrangement[1][t]] = S[arrangement[0][t]]
        resAT[arrangement[1][t]] = A.T[arrangement[0][t]]
    return resAT.T, resS


def plotData(trueS, Y, S):

    # show data and model
    fig, axs = plt.subplots(1, 3, sharey=True)
    axs[0].plot(trueS.T)
    axs[1].plot(Y.T, c="k", alpha=0.15)
    axs[2].plot(S.T)
    axs[0].set_xlabel("Feature")
    axs[1].set_xlabel("Feature")
    axs[2].set_xlabel("Feature")
    axs[0].text(0.03, 0.97, "True S", ha="left", va="top", transform=axs[0].transAxes)
    axs[1].text(0.03, 0.97, "Y", ha="left", va="top", transform=axs[1].transAxes)
    axs[2].text(
        0.03, 0.97, "Best-fit S", ha="left", va="top", transform=axs[2].transAxes
    )
    axs[0].set_ylim(top=1.1)
    axs[1].set_ylim(top=1.1)
    axs[2].set_ylim(top=1.1)
    fig.subplots_adjust(wspace=0)
    fig.tight_layout()
    fig.show()


def plotLoss(trace, Y, ax=None, label=None):

    # convergence plot from traceback
    loss = []
    feasible = []
    for At, St in traceback.trace:
        loss.append(proxmin.nmf.log_likelihood(At, St, Y=Y))

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.semilogy(loss, label=label)


if __name__ == "__main__":

    if len(sys.argv) == 2:
        problem = sys.argv[1]
        if problem not in ["nmf", "mixmf"]:
            raise ValueError("Expected either 'nmf' or 'mixmf' as an argument")
    else:
        problem = "mixmf"

    n = 50  # component resolution
    k = 3  # number of components
    b = 100  # number of observations
    noise = 0.02  # stdev of added noise
    np.random.seed(101)

    # set up test data
    trueA = np.array([generateAmplitudes(k) for i in range(b)])
    trueS = np.array([generateComponent(n) for i in range(k)])
    Y = add_noise(np.dot(trueA, trueS), noise)

    A0 = np.random.rand(b, k)
    A0 /= A0.sum(axis=1)[:, None]
    S0 = np.random.rand(k, n)

    # mixture model: amplitudes positive
    # and sum up to one at every pixel
    pA = partial(proxmin.operators.prox_unity_plus, axis=1)
    pS = proxmin.operators.prox_plus

    if problem == "nmf":
        prox = [pS, pS]
    elif problem == "mixmf":
        prox = [pA, pS]

    grad = partial(proxmin.nmf.grad_likelihood, Y=Y)

    traceback = Traceback()
    all_args = {"prox": prox, "max_iter": 1000, "callback": traceback, "e_rel": 1e-4}
    b1 = 0.9
    b2 = 0.999
    adaprox_args = {"b1": b1, "b2": b2, "prox_max_iter": 100}
    runs = (
        (proxmin.pgm, all_args, "PGM"),
        (proxmin.adaprox, dict(all_args, **adaprox_args, scheme="adam"), "Adam"),
        (
            proxmin.adaprox,
            dict(all_args, **adaprox_args, scheme="padam", p=0.125),
            "PAdam",
        ),
        (proxmin.adaprox, dict(all_args, **adaprox_args, scheme="amsgrad"), "AMSGrad"),
    )

    best_AS = None
    best_loss = np.inf

    for i, alpha in enumerate([0.01, 0.1]):
        step = {
            proxmin.pgm: proxmin.nmf.step_pgm,
            proxmin.adaprox: lambda *X, it: (alpha, alpha),
        }

        for alg, kwargs, label in runs:
            A = A0.copy()
            S = S0.copy()
            traceback.clear()
            try:
                alg((A, S), grad, step[alg], **kwargs)
                loss = proxmin.nmf.log_likelihood(A, S, Y=Y)
                print("{}: final loss = {}\n".format(label, loss))

                if loss < best_loss:
                    best_loss = loss
                    best_AS = (A.copy(), S.copy())

            except np.linalg.LinAlgError:
                pass

    A, S = match(*best_AS, trueS)
    plotData(trueS, Y, S)
