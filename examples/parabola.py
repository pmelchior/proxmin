import sys, numpy as np
import proxmin
from functools import partial

import logging
logging.basicConfig()
logger = logging.getLogger('proxmin')
logger.setLevel(logging.INFO)

# location of true minimum of f
dX = np.array([1,0.5])

def f(X):
    """Shifted parabola"""
    return np.sum((X - dX)**2, axis=-1)

def grad_f(X):
    return 2*(X - dX)

def step_f(X, it=0):
    L = 2
    slowdown = 0.1 # to see behavior better
    return slowdown * 1 / L

def prox_circle(X, step):
    """Projection onto circle"""
    center = np.array([0,0])
    dX = X - center
    radius = 0.5
    phi = np.arctan2(dX[1], dX[0])
    return center + radius*np.array([np.cos(phi), np.sin(phi)])

def prox_xline(x, step):
    """Projection onto line in x"""
    if not np.isscalar(x):
        x= x[0]
    if x > 0.5:
        return np.array([0.5])
    else:
        return np.array([x])

def prox_yline(y, step):
    """Projection onto line in y"""
    if not np.isscalar(y):
        y= y[0]
    if y > -0.75:
        return np.array([-0.75])
    else:
        return np.array([y])

def prox_line(X, step):
    """2D projection onto 2 lines"""
    return np.concatenate((prox_xline(X[0], step), prox_yline(X[1], step)))

def prox_lim(X, step, boundary=None):
    """Proximal projection operator"""
    if boundary == "circle":
        return prox_circle(X, step)
    if boundary == "line":
        return prox_line(X, step)
    # default: do nothing
    return X

def prox_gradf(X, step):
    """Gradient step"""
    return X-step*grad_f(X)

def prox_gradf_lim(X, step, boundary=None):
    """Forward-backward step: gradient, followed by projection"""
    return prox_lim(prox_gradf(X,step), step, boundary=boundary)


def plotResults(trace, label="", boundary=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    lims = -2,2
    X = np.dstack(np.meshgrid(np.linspace(lims[0],lims[1],1000), np.linspace(lims[0], lims[1],1000)))
    r = f(X)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111,aspect='equal')
    ax.contourf(r, 20, extent=(lims[0], lims[1], lims[0], lims[1]), cmap='Reds')

    traj = np.array(trace).reshape(len(trace), -1)

    ax.plot(traj[:,0], traj[:,1], 'b.', markersize=4, ls='-')

    if boundary is not None:
        if boundary == "circle":
            circ = patches.Circle((0, 0), radius=0.5, fc="none", ec='k', ls='dotted')
            ax.add_artist(circ)
        if boundary == "line":
            ax.plot(lims, [-0.75, -0.75], 'k:')
            ax.plot([0.5,0.5], lims, 'k:')

    ax.scatter(traj[-1][0], traj[-1][1], marker='x', s=30, c='r')
    ax.text(0.05, 0.95, 'it %d: (%.3f, %.3f)' % (len(traj), traj[-1][0], traj[-1][1]),
            transform=ax.transAxes, color='k', ha='left', va='top')
    ax.set_title(label)
    plt.show()


if __name__ == "__main__":
    X0 = np.array([-1.,-1])
    if len(sys.argv)==2:
        boundary = sys.argv[1]
        if boundary not in ["line", "circle"]:
            raise ValueError("Expected either 'line' or 'circle' as an argument")
    else:
        boundary = "circle"

    prox = partial(prox_lim, boundary=boundary)
    max_iter = 1000
    traceback = proxmin.utils.Traceback()

    # PGM without boundary
    X = X0.copy()
    proxmin.pgm(X, grad_f, step_f, max_iter=max_iter, relax=1, callback=traceback)
    plotResults(traceback.trace, "PGM no boundary")

    # PGM
    X = X0.copy()
    traceback.clear()
    proxmin.pgm(X, grad_f, step_f, prox=prox, max_iter=max_iter, callback=traceback, accelerated=False)
    plotResults(traceback.trace, "PGM", boundary=boundary)

    # APGM
    X = X0.copy()
    traceback.clear()
    proxmin.pgm(X, grad_f, step_f, prox=prox, max_iter=max_iter, accelerated=True,  callback=traceback)
    plotResults(traceback.trace, "PGM accelerated", boundary=boundary)

    # Adaptive moments methods: Adam and friends
    for scheme in ["adam", "adamx", "amsgrad", "padam", "radam"]:
        X = X0.copy()
        traceback.clear()
        b1 = 0.0
        if scheme != "adam":
            b1 = b1 ** np.arange(1, max_iter+1)
        proxmin.adaprox(X, grad_f, step_f, prox=prox, b1=b1, b2=0.5, max_iter=max_iter, callback=traceback, scheme=scheme, p=0.125)
        plotResults(traceback.trace, scheme.upper(), boundary=boundary)

    # ADMM
    X = X0.copy()
    traceback.clear()
    proxmin.admm(X, prox_gradf, step_f, prox_g=prox, max_iter=max_iter, callback=traceback)
    plotResults(traceback.trace, "ADMM", boundary=boundary)

    # ADMM with direct constraint projection
    prox_direct = partial(prox_gradf_lim, boundary=boundary)
    X = X0.copy()
    traceback.clear()
    proxmin.admm(X, prox_direct, step_f, prox_g=None, max_iter=max_iter, callback=traceback)
    plotResults(traceback.trace, "ADMM direct", boundary=boundary)

    # SDMM
    M = 2
    proxs_g = [prox] * M # using same constraint several, i.e. M, times
    X = X0.copy()
    traceback.clear()
    proxmin.sdmm(X, prox_gradf, step_f, proxs_g=proxs_g, max_iter=max_iter, callback=traceback)
    plotResults(traceback.trace, "SDMM", boundary=boundary)
