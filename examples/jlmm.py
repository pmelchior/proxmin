import numpy as np
from functools import partial
from proxmin import algorithms as pa
from proxmin import operators as po
from proxmin.utils import MathList
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

def prox_gradf(Xs, steps):
    x,y = Xs
    gradfs = MathList([grad_fx(x,y), grad_fy(x,y)])
    return Xs - gradfs*steps

def prox_circle(Xs, steps):
    """Projection onto circle"""
    center = MathList([0,0])
    dxy = Xs - center
    radius = 0.5
    phi = np.arctan2(dxy[1], dxy[0])
    return center + MathList([np.cos(phi), np.sin(phi)])*radius

def prox_gradf_circle(Xs, steps):
    return prox_circle(prox_gradf(Xs, steps), steps)

def plotResults(tr, label, boundary=None):
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

    #ax.scatter(traj[:,0], traj[:,1], s=4, c=np.arange(1,len(traj)+1), cmap='Blues_r')
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

"""
def prox_joint(Xs, steps, prox_list=None):
    return MathList([prox_list[i](Xs[i], steps[i]) for i in range(len(X))])
"""

if __name__ == "__main__":
    boundary = "circle"

    """Proximal gradient method with 2 blocks and operators that
    - have access to beoth blocks
    - update both atomically
    """

    # does not converge to the correct location if the step sizes are different!
    # the problem is that the projection onto the circle doesn't know that the
    # gradient step has different stepsizes, it's therefore too strong on the
    # "slow" y direction, even though that particular prox operator doesn't use
    # the step size argument.
    max_iter = 100
    Xs = MathList([np.array(-1.),np.array(-0.75)])
    steps = MathList([0.5, 0.1])

    x, tr = pa.pgm(Xs, prox_gradf_circle, steps, max_iter=max_iter, relax=1, traceback=True)
    plotResults(tr, "PGM joint", boundary=boundary)

    # adopt the minimum stepsize (so as to now break non-expansiveness)
    steps = MathList([0.1, 0.1])
    x, tr = pa.pgm(Xs, prox_gradf_circle, steps, max_iter=max_iter, relax=1, traceback=True)
    plotResults(tr, "PGM joint min stepsize", boundary=boundary)

    x, tr = pa.apgm(Xs, prox_gradf_circle, steps, max_iter=max_iter, traceback=True)
    plotResults(tr, "APGM joint min stepsize", boundary=boundary)
