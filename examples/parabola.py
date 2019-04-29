import sys, numpy as np
from functools import partial
from proxmin import algorithms as pa
from proxmin.utils import Traceback

import logging
logging.basicConfig()
logger = logging.getLogger('proxmin')
logger.setLevel(logging.INFO)

# location of true minimum of f
dx,dy = 1,0.5

def f(x,y):
    """Shifted parabola"""
    return (x-dx)**2 + (y-dy)**2

def grad_fx(x,y):
    """Gradient of f wrt x"""
    return 2*x - 2*dx

def grad_fy(x,y):
    """Gradient of f wrt y"""
    return 2*y - 2*dy

def grad_f(xy):
    """Gradient of f"""
    return np.array([grad_fx(xy[0],xy[1]),grad_fy(xy[0],xy[1])])

def prox_circle(xy, step):
    """Projection onto circle"""
    center = np.array([0,0])
    dxy = xy - center
    radius = 0.5
    # exclude interior of circle
    #if (dxy**2).sum() < radius**2:
    # exclude everything other than perimeter of circle
    if 1:
        phi = np.arctan2(dxy[1], dxy[0])
        return center + radius*np.array([np.cos(phi), np.sin(phi)])
    else:
        return xy

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

def prox_line(xy, step):
    """2D projection onto 2 lines"""
    return np.concatenate((prox_xline(xy[0], step), prox_yline(xy[1], step)))

def prox_lim(xy, step, boundary=None):
    """Proximal projection operator"""
    if boundary == "circle":
        return prox_circle(xy, step)
    if boundary == "line":
        return prox_line(xy, step)
    # default: do nothing
    return xy

def prox_gradf(xy, step):
    """Gradient step"""
    return xy-step*grad_f(xy)

def prox_gradf_lim(xy, step, boundary=None):
    """Forward-backward step: gradient, followed by projection"""
    return prox_lim(prox_gradf(xy,step), step, boundary=boundary)

# for GLMM only: x1 and x2 treated separately
def prox_gradf12(x, step, j=None, Xs=None):
    """1D gradient operator for x or y"""
    if j == 0:
        return x - step*grad_fx(Xs[0][0], Xs[1][0])
    if j == 1:
        y = x
        return y - step*grad_fy(Xs[0][0], Xs[1][0])
    raise NotImplementedError

def prox_circle12(x, step, j=None, Xs=None):
    # this is the experimental non-separable constraint
    if j == 0:
        xy = np.array([x[0], Xs[1][0]])
    if j == 1:
        xy = np.array([Xs[0][0], x[0]])
    return [prox_circle(xy, step)[j]]

def prox_lim12(x, step, j=None, Xs=None, boundary=None):
    # separable constraints
    if boundary == "line":
        if j == 0:
            return prox_xline(x, step)
        if j == 1:
            return prox_yline(x, step)

    # this is the "illegal" non-separable constraint: raise exception
    raise NotImplementedError
    """
    if boundary == "circle":
        if j == 0:
            xy = np.array([x[0], Xs[1][0]])
        if j == 1:
            xy = np.array([Xs[0][0], x[0]])
        return [prox_circle(xy, step)[j]]
    """


def prox_gradf_lim12(x, step, j=None, Xs=None, boundary=None):
    """1D projection operator"""
    # TODO: split boundary in x1 and x2 and use appropriate operator
    if j == 0:
        x -= step*grad_fx(Xs[0][0], Xs[1][0])
    if j == 1:
        y = x
        y -= step*grad_fy(Xs[0][0], Xs[1][0])
    return prox_lim12(x, step, j=j, Xs=Xs, boundary=boundary)


def steps_f12(j=None, Xs=None):
    """Stepsize for f update given current state of Xs"""
    # Lipschitz const is always 2
    L = 2
    slack = 0.1# 1.
    return slack / L


def plotResults(tr, label="", boundary=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    lims = -2,2
    x,y = np.meshgrid(np.linspace(lims[0],lims[1],1000), np.linspace(lims[0], lims[1],1000))
    r = f(x,y)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111,aspect='equal')
    ax.contourf(r, 20, extent=(lims[0], lims[1], lims[0], lims[1]), cmap='Reds')

    if tr.N == 1:
        traj = tr["X"]
    else:
        traj = np.dstack([tr["X",j] for j in range(tr.N)])[:,0,:]

    if tr.offset == 0:
        ax.plot(traj[:,0], traj[:,1], 'b.', markersize=4, ls='-')
    else:
        ax.plot(traj[:tr.offset,0], traj[:tr.offset,1], 'b.', markersize=4, ls='--')
        ax.plot(traj[tr.offset:,0], traj[tr.offset:,1], 'b.', markersize=4, ls='-')

    if boundary is not None:
        if boundary == "circle":
            circ = patches.Circle((0, 0), radius=0.5, fc="none", ec='k', ls='dotted')
            ax.add_artist(circ)
        if boundary == "line":
            ax.plot(lims, [-0.75, -0.75], 'k:')
            ax.plot([0.5,0.5], lims, 'k:')

    ax.scatter(traj[-1][0], traj[-1][1], marker='x', s=30, c='r')
    ax.text(0.05, 0.95, 'it %d: (%.3f, %.3f)' % (tr.it, traj[-1][0], traj[-1][1]),
            transform=ax.transAxes, color='k', ha='left', va='top')
    ax.set_title(label)
    plt.show()

if __name__ == "__main__":
    xy0 = np.array([-1.,-1.])
    if len(sys.argv)==2:
        boundary = sys.argv[1]
        if boundary not in ["line", "circle"]:
            raise ValueError("Expected either 'line' or 'circle' as an argument")
    else:
        boundary = "circle"
    max_iter = 100

    # step sizes and proximal operators for boundary
    step_f = steps_f12()

    prox_g = partial(prox_lim, boundary=boundary)
    prox_gradf_ = partial(prox_gradf_lim, boundary=boundary)

    # PGM without boundary
    tr = Traceback()
    xy = xy0.copy()
    pa.pgm(xy, prox_gradf, step_f, max_iter=max_iter, relax=1, traceback=tr)
    plotResults(tr, "PGM no boundary")

    # PGM
    tr = Traceback()
    xy = xy0.copy()
    pa.pgm(xy, prox_gradf_, step_f, max_iter=max_iter, traceback=tr)
    plotResults(tr, "PGM", boundary=boundary)

    # APGM
    tr = Traceback()
    xy = xy0.copy()
    pa.pgm(xy, prox_gradf_, step_f, max_iter=max_iter, accelerated=True,  traceback=tr)
    plotResults(tr, "PGM accelerated", boundary=boundary)

    # ADMM
    tr = Traceback()
    xy = xy0.copy()
    pa.admm(xy, prox_gradf, step_f, prox_g, max_iter=max_iter, traceback=tr)
    plotResults(tr, "ADMM", boundary=boundary)

    # ADMM with direct constraint projection
    prox_g_direct = None
    tr = Traceback()
    xy = xy0.copy()
    pa.admm(xy, prox_gradf_, step_f, prox_g_direct, max_iter=max_iter, traceback=tr)
    plotResults(tr, "ADMM direct", boundary=boundary)

    # SDMM
    M = 2
    proxs_g = [prox_g] * M # using same constraint several, i.e. M, times
    tr = Traceback()
    xy = xy0.copy()
    pa.sdmm(xy, prox_gradf, step_f, proxs_g, max_iter=max_iter, traceback=tr)
    plotResults(tr, "SDMM", boundary=boundary)

    # Block-SDMM
    if boundary == "line":
        N = 2
        M1 = 7
        M2 = 2
        proxs_g = [[prox_xline]*M1, [prox_yline]*M2]
        tr = Traceback(N)
        XY = [np.array([xy0[0]]), np.array([xy0[1]])]
        pa.bsdmm(XY, prox_gradf12, steps_f12, proxs_g, max_iter=max_iter, traceback=tr)
        plotResults(tr, "bSDMM", boundary=boundary)

        # bSDMM with direct constraint projection
        prox_gradf12_ = partial(prox_gradf_lim12, boundary=boundary)
        prox_g_direct = None
        tr = Traceback(N)
        XY = [np.array([xy0[0]]), np.array([xy0[1]])]
        pa.bsdmm(XY, prox_gradf12_, steps_f12, prox_g_direct, max_iter=max_iter, traceback=tr)
        plotResults(tr, "bSDMM direct", boundary=boundary)

        # BPGM
        tr = Traceback(N)
        XY = [np.array([xy0[0]]), np.array([xy0[1]])]
        pa.bpgm(XY, prox_gradf12_, steps_f12, max_iter=max_iter, accelerated=True, traceback=tr)
        plotResults(tr, "bPGM", boundary=boundary)
