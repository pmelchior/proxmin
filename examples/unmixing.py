#from proxmin import nmf
import numpy as np
import proxmin
from proxmin.utils import Traceback
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import time
from functools import partial

# initialize and run NMF
import logging
logging.basicConfig()
logger = logging.getLogger('proxmin')
logger.setLevel(logging.INFO)

def generateComponent(m):
    """Creates oscillating components to be mixed"""
    freq = 25*np.random.random()
    phase = 2*np.pi*np.random.random()
    x = np.arange(m)
    return np.cos(x/freq-phase)**2

def generateAmplitudes(k):
    """Makes mixing coefficients"""
    res = np.array([np.random.random() for i in range(k)])
    return res/res.sum()

def add_noise(Y, sigma):
    """Adds noise to Y"""
    return Y + np.random.normal(0, sigma, Y.shape)

def match(A, S, trueS):
    """Rearranges columns of S to best fit the components they likely represent (maximizes sum of correlations)"""
    cov = np.cov(trueS, S)
    k = S.shape[0]
    corr = np.zeros([k,k])
    for i in range(k):
        for j in range(k):
            corr[i][j] = cov[i + k][j]/np.sqrt(cov[i + k][i + k]*cov[j][j])
    arrangement = linear_sum_assignment(-corr)
    resS = np.zeros_like(S)
    resAT = np.zeros_like(A.T)
    for t in range(k):
        resS[arrangement[1][t]] = S[arrangement[0][t]]
        resAT[arrangement[1][t]] = A.T[arrangement[0][t]]
    return resAT.T, resS

def plotResults(trace, Y, A, S, trueS, trueA):

    # sort components to best match inputs
    A, S = match(A, S, trueS)

    # show data and model
    fig = plt.figure(figsize=(6,7))
    ax = fig.add_subplot(311)
    ax.set_title("True Components S")
    ax.plot(trueS.T)
    ax2 = fig.add_subplot(312)
    ax2.set_title("Data Y")
    ax2.plot(Y.T)
    ax3 = fig.add_subplot(313)
    ax3.set_title("Found Components S")
    ax3.set_xlabel("Pixel")
    ax3.plot(S.T)
    fig.subplots_adjust(bottom=0.07, top=0.95, hspace=0.35)
    fig.show()

    # convergence plot from traceback
    loss = []
    feasible = []
    trueY = np.dot(trueA, trueS)
    for At,St in traceback.trace:
        Yt = np.dot(At, St)
        loss.append(((Yt - trueY)**2).sum())
        feasible_A = np.all(At >=0 ) & np.allclose(np.sum(At, axis=1), 1)
        feasible_S = np.all(St >=0 )
        feasible.append((feasible_A, feasible_S))
    print ("final loss: {}".format(loss[-1]))
    print ("solution feasible: {}".format(feasible[-1]))

    fig2 = plt.figure(figsize=(6,4))
    ax4 = fig2.add_subplot(111)
    ax4.set_title("Loss")
    ax4.semilogy(loss)
    ax4.set_ylabel("$||Y-AS||^2$")
    ax4.set_xlabel("Iterations")
    ax4.text(0.95, 0.95, "Loss=%.3f" % loss[-1], ha='right', va='top', transform=ax4.transAxes)
    fig2.show()


if __name__ == "__main__":
    n = 50 			# component resolution
    k = 3 			# number of components
    b = 100			# number of observations
    noise = 0.02    # stdev of added noise
    np.random.seed(101)

    # set up test data
    trueA = np.array([generateAmplitudes(k) for i in range(b)])
    trueS = np.array([generateComponent(n) for i in range(k)])
    Y = add_noise(np.dot(trueA,trueS), noise)

    # mixture model: amplitudes positive
    # and sum up to one at every pixel
    pA = partial(proxmin.operators.prox_unity_plus, axis=1)
    pS = proxmin.operators.prox_plus

    A = np.random.rand(b,k)
    S = np.random.rand(k,n)
    traceback = Traceback()
    proxmin.nmf.nmf(Y, A, S, prox_A=pA, prox_S=pS, e_rel=1e-3, callback=traceback)
    plotResults(traceback, Y, A, S, trueS, trueA)
