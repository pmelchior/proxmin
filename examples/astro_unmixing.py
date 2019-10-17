#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import proxmin
from proxmin.utils import Traceback
from functools import partial

import logging
logging.basicConfig()
logger = logging.getLogger('proxmin')
logger.setLevel(logging.INFO)

def generateComponent(size, pos, dim):
    """Creates 2D Gaussian component"""
    x = np.arange(dim)
    c = np.exp(-((x - pos[0])[:,None]**2 + (x - pos[1])[None,:]**2) / (2*size**2))
    return c.flatten() / c.sum()

def generateAmplitude(flux, dim):
    """Creates normalized SED"""
    return flux * np.random.dirichlet(np.ones(dim))

def add_noise(Y, sky):
    """Adds Poisson noise to Y"""
    Y += sky[:,None]
    Y = np.random.poisson(Y).astype('float64')
    Y -= sky[:,None]
    return Y

def plotLoss(trace, Y, W, ax=None, label=None, plot_max=None):

    # convergence plot from traceback
    loss = []
    for At,St in traceback.trace:
        loss.append(proxmin.nmf.log_likelihood(At, St, Y=Y, W=W))

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.semilogy(loss, label=label)

def plotData(Y, A, S):

    c, n = Y.shape
    nx = ny = np.int(np.sqrt(n))

    # reasonable mapping from 5 observed bands to rgb channels
    filter_weights = np.zeros((3, c))
    filter_weights[0, 4] = 1
    filter_weights[0, 3] = 0.667
    filter_weights[1, 3] = 0.333
    filter_weights[1, 2] = 1
    filter_weights[1, 1] = 0.333
    filter_weights[2, 1] = 0.667
    filter_weights[2, 0] = 1
    filter_weights /= 1.667

    rgb = np.dot(filter_weights, Y)

    try:
        from astropy.visualization import make_lupton_rgb
        Q = 1
        stretch = Y.max()/2

        fig = plt.figure(figsize=(9,3))
        ax0 = fig.add_axes([0, 0, 0.33, 1], frameon=False)
        ax1 = fig.add_axes([0.333, 0, 0.33, 1], frameon=False)
        ax2 = fig.add_axes([0.666, 0, 0.33, 1], frameon=False)

        ax0.imshow(make_lupton_rgb(*np.split(rgb, 3, axis=0)[::-1], Q=Q, stretch=stretch).reshape(ny,nx,3))

        best_Y = np.dot(A, S)
        rgb =np.dot(filter_weights, best_Y)
        ax1.imshow(make_lupton_rgb(*np.split(rgb, 3, axis=0)[::-1], Q=Q, stretch=stretch).reshape(ny,nx,3))

        residual = Y - best_Y
        rgb = np.dot(filter_weights, residual)
        rgb -= rgb.min()
        rgb /= rgb.max()
        ax2.imshow(rgb.reshape(ny,nx,3))

        ax0.text(0.05, 0.95, 'Data', color='w', va='top', ha='left', transform=ax0.transAxes)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax1.text(0.05, 0.95, 'Model', color='w', va='top', ha='left', transform=ax1.transAxes)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.text(0.05, 0.95, 'Residuals', color='w', va='top', ha='left', transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])
        fig.show()

    except ImportError:
        print ("please install astropy for plotting")

if __name__ == "__main__":

    N = 30 # component pixel number per dimension
    K = 7  # number of components
    C = 5  # number of channels

    factor = 1e4
    np.random.seed(8)
    sky = factor*np.random.rand(C)
    np.random.seed(10)

    # set up test data
    sizes = np.array([1,2,2,3,4,5,10])
    fluxes = factor * sizes**2
    pad = 3
    poss = (pad + np.random.rand(K, 2) * (N - 2*pad)).astype(np.int)

    components = [(generateAmplitude(flux, C), generateComponent(size, pos, N)) for flux, size, pos in zip(fluxes, sizes, poss)]
    Y = np.sum([a[:,None] * s[None,:] for a,s in components], axis=0)
    W = np.ones_like(Y) / sky[:,None]
    Y = add_noise(Y, sky)

    trueA = np.stack([a for a,s in components], axis=1)
    trueS = np.stack([s for a,s in components], axis=0)

    # intialize with approximate parameters
    meas_sizes = sizes * (1 + (np.random.rand(K) - 0.5))
    meas_poss = poss + np.random.normal(0, scale=np.ones(2)[None,:] * meas_sizes[:,None]) / 8
    A0 = np.sqrt(2*np.pi)*meas_sizes**2 * np.stack([Y[:, N*posy + posx] for posy,posx in meas_poss.astype('int')], axis=1) # SED at peak location
    S0 = np.stack([ generateComponent(size, pos, N) for size,pos in zip(meas_sizes, meas_poss)], axis=0)


    # optimize loss under constraints
    grad = partial(proxmin.nmf.grad_likelihood, Y=Y, W=W)

    s_step = 1e-5
    step = {
        proxmin.pgm: partial(proxmin.nmf.step_pgm, W=W),
        proxmin.adaprox: lambda A, S, it: (np.mean(A, axis=0)/10, s_step)
    }

    pA = proxmin.operators.prox_plus

    def proxS(X, step, thresh=1e-4):
        # hard thresholding
        X = np.where(X > thresh, X, 0)
        # unit normalization
        X = proxmin.operators.prox_unity_plus(X, step, axis=1)
        return X

    prox = [pA, proxS]

    traceback = Traceback()
    all_args = {'prox': prox, 'max_iter': 1000, 'callback': traceback, 'e_rel': 1e-3}
    b1 = 0.9
    b2 = 0.999
    runs = (
        (proxmin.pgm, all_args, 'PGM'),
        (proxmin.adaprox, dict(all_args, scheme="adam", b1=b1, b2=b2, prox_max_iter=100), 'Adam'),
        (proxmin.adaprox, dict(all_args, scheme="padam", b1=b1, b2=b2, prox_max_iter=100, p=0.45), 'PAdam'),
        (proxmin.adaprox, dict(all_args, scheme="amsgrad", b1=b1, b2=b2, prox_max_iter=100), 'AMSGrad'),
    )

    best_AS = None
    best_loss = np.inf
    for alg, kwargs, label in runs:
        A = A0.copy()
        S = S0.copy()
        traceback.clear()
        try:
            alg((A,S), grad, step[alg], **kwargs)
            loss = proxmin.nmf.log_likelihood(A, S, Y=Y, W=W)
            print ("{}: final loss = {}\n".format(label, loss))

            if loss < best_loss:
                best_loss = loss
                best_AS = (A.copy(), S.copy())

        except np.linalg.LinAlgError:
            pass

    plotData(Y, *best_AS)
