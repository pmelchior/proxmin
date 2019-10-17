[![PyPI](https://img.shields.io/pypi/v/proxmin.svg)](https://pypi.org/project/proxmin/)
[![License](https://img.shields.io/github/license/pmelchior/proxmin.svg)](https://github.com/pmelchior/proxmin/blob/master/LICENSE.md)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs11081--018--9380--y-blue.svg)](https://doi.org/10.1007/s11081-018-9380-y)
[![arXiv](https://img.shields.io/badge/arxiv-1708.09066-red.svg)](http://arxiv.org/abs/1708.09066)

# Proximal Minimization

The methods in this package provide solvers for constrained optimization problems. All of them use proximal operators to deal with non-smooth penalty functions.

The algorithms:

* **Proximal Gradient Method (PGM)**: forward-backward splitting with a single smooth function with a Lipschitz-continuous gradient and a single (non-smooth) penalty function. Includes multi-block optimization and Nesterov acceleration.
* **Adam and derivatives (AdamX, AMSGrad, PAdam)**: forward-backward splitting with adaptive gradient steps for single- and multi-block optimization.
* **Alternating Direction Method of Multipliers (ADMM)**: Rachford-Douglas splitting for two potentially non-smooth functions. We use its linearized form to solve for additional linear mappings in the penalty functions.
* **Simultaneous Direction Method of Multipliers (SDMM)**: Extension of linearized ADMM for several penalty functions.
* **Block-Simultaneous Direction Method of Multipliers (bSDMM)**: Extension of SDMM to work with objective functions that are convex in several arguments. It's a proximal version of Block coordinate descent methods.

Two-block PGM or bSDMM is used as backend solvers for Non-negative Matrix Factorization (NMF). As the algorithms allow any proxable function as constraint on each of the matrix factors, we prefer the term Constrained Matrix Factorization.

Details can be found in the [paper](https://doi.org/10.1007/s11081-018-9380-y) *"Block-Simultaneous Direction Method of Multipliers - A proximal primal-dual splitting algorithm for nonconvex problems with multiple constraints"* by Fred Moolekamp and Peter Melchior.

We ask that any published work that utilizes this package cites:
```
@ARTICLE{proxmin,
    author="{Moolekamp}, Fred and {Melchior}, Peter",
    title="Block-simultaneous direction method of multipliers: a proximal primal-dual splitting algorithm for nonconvex problems with multiple constraints",
    journal="Optimization and Engineering",
    year="2018",
    month="Dec",
    volume=19,
    issue=4,
    pages={871-885},
    doi="10.1007/s11081-018-9380-y",
    url="https://doi.org/10.1007/s11081-018-9380-y"
    archivePrefix="arXiv",
    eprint={1708.09066},
    primaryClass="math.OC"
}
```
Also, let us know (e.g. [@peter_melchior](https://twitter.com/peter_melchior)), we're curious.

## Installation and Dependencies

```
pip install proxmin
```

 For the latest development version, clone this repository and execute `python setup.py install`.

The code works on python>2.7 and requires numpy and scipy. It is fully compatible with gradient computation by `autograd`.

## Approach

The gradient-based methods PGM and Adam expect two callback function: one to compute the gradients, the other to compute step sizes. In the former case, the step sizes are bound between 0 and 2/L, where L is the Lipschitz constant of the gradient.

The penalty functions are given as proximal mappings: `X <- prox(X, step)`. 

Many proximal operators can be constructed analytically, see e.g. [Parikh & Boyd (2014)](https://web.stanford.edu/~boyd/papers/prox_algs.html). We provide a number of common ones in `proxmin.operators`. An important class of constraints are indicator functions of convex sets, for which the proximal operator, given some point **X**, returns the closest point to **X** in the Euclidean norm that is in the set. 

**Example:** find the minimum of a shifted parabola on the unit circle in 2D

```python
import numpy as np
import proxmin

dX = np.array([1.,0.5])
radius = 1

def f(X):
    """Shifted parabola"""
    return np.sum((X - dX)**2, axis=-1)

def grad_f(X):
    return 2*(X - dX)

def step_f(X, it=0):
    L = 2. # Lipschitz constant of grad f
    return 1 / L

def prox_circle(X, step):
    """Projection onto circle"""
    center = np.array([0,0])
    dX = X - center
    # exclude everything other than perimeter of circle
    phi = np.arctan2(dX[1], dX[0])
    return center + radius*np.array([np.cos(phi), np.sin(phi)])

X = np.array([-1.,-1.]) # or whereever
converged, grad, step = proxmin.pgm(X, grad_f, step_f, prox=prox_circle)
```

Since the objective function is smooth and there is only one constraint, one can simply perform a sequence of *forward-backward* steps: a step in gradient direction, followed by a projection onto the constraint subset. That is, in essence, the proximal gradient method.

If both functions are not smooth, one can use ADMM. It therefore operates on two proxed functions. Unlike PGM, feasibility is only achieved at the end of the optimization and only within some error tolerance.

Continuing the example above, the smooth function gets turned into a proxed function by performing the gradient step internally and returning the updated position:

```python
def prox_gradf(X, step):
    """Proximal gradient step"""
    return X-step*grad_f(X)

converged = proxmin.admm(X, prox_gradf, step_f, prox_g=prox_circle, e_rel=1e-3, e_abs=1e-3)
```

## Constrained matrix factorization (CMF)

Matrix factorization seeks to approximate a target matrix `Y` as a product of `np.dot(A,S)`. If those constraints are only non-negativity, the method is known as NMF.

We have extended the capabilities by allowing for an arbitrary number of constraints to be enforced on either matrix factor:

```python
# PGM approach for each factor
prox_A = ... # a single constraint on A, solved by projection
prox_S = ... # a single constraint on S, solved by projection
A0, S0 = ... # initialization
proxmin.nmf.nmf(Y, A0, S0, prox_A=prox_A, prox_S=prox_S)

# same with AdaProx-AMSGrad
proxmin.nmf.nmf(Y, A0, S0, prox_A=prox_A, prox_S=prox_S, algorithm=proxmin.algorithms.adaprox, scheme="amsgrad")

# for multiple constraints, solved by ADMM-style split
proxs_g = [[...], # list of proxs for A
           [...]] # list of proxs for S
A, S = proxmin.nmf.nmf(Y, A0, S0, algorithm=proxmin.algorithms.bsdmm, proxs_g=proxs_g)
# or a combination
A, S = proxmin.nmf.nmf(Y, A0, S0, algorithm=proxmin.algorithms.bsdmm, prox_A=prox_A, prox_S=prox_S, proxs_g=proxs_g)
```
