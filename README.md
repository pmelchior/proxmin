[![PyPI](https://img.shields.io/pypi/v/proxmin.svg)](https://pypi.org/project/proxmin/)
[![License](https://img.shields.io/github/license/pmelchior/proxmin.svg)](https://github.com/pmelchior/proxmin/blob/master/LICENSE.md)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs11081--018--9380--y-blue.svg)](https://doi.org/10.1007/s11081-018-9380-y)
[![arXiv](https://img.shields.io/badge/arxiv-1708.09066-red.svg)](http://arxiv.org/abs/1708.09066)

# Proximal Minimization

The methods in this package provide solvers for constrained optimization problems. All of them use proximal operators to deal with non-smooth constraint functions.

The algorithms:

* **Proximal Gradient Method (PGM)**: *forward-backward* split with a single smooth function with a Lipschitz-continuous gradient and a single (non-smooth) constraint function. Nesterov acceleration is available.
* **Block/Alternating Proximal Gradient Method (bPGM)**: Extension of PGM to objective functions that are convex in several arguments; optional Nesterov acceleration.
* **Alternating Direction Method of Multipliers (ADMM)**: Rachford-Douglas split for two potentially non-smooth functions. We use the linearized form of it solve for additional linear mappings in the constraint functions.
* **Simultaneous Direction Method of Multipliers (SDMM)**: Extension of linearized ADMM for several constraint functions.
* **Block-Simultaneous Direction Method of Multipliers (bSDMM)**: Extension of SDMM to work with objective functions that are convex in several arguments. It's a proximal version of Block coordinate descent methods.

In addition, bSDMM is used as the backend of a solver for Non-negative Matrix Factorization (NMF). As our algorithm allows an arbitrary number of constraints on each of the matrix factors, we prefer the term Constrained Matrix Factorization.

Details can be found in the [paper](https://doi.org/10.1007/s11081-018-9380-y) *"Block-Simultaneous Direction Method of Multipliers - A proximal primal-dual splitting algorithm for nonconvex problems with multiple constraints"* by Fred Moolekamp and Peter Melchior.

We ask that any published work that utilizes this package cites:
```
@ARTICLE{proxmin,
    author="{Moolekamp}, Fred and {Melchior}, Peter",
    title="Block-simultaneous direction method of multipliers: a proximal primal-dual splitting algorithm for nonconvex problems with multiple constraints",
    journal="Optimization and Engineering",
    year="2018",
    month="Mar",
    day="20",
    issn="1573-2924",
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

The code works on python>2.7 and requires numpy and scipy.

## Approach

All algorithms accept functions only in their proxed form, i.e. they call the respective proximal operators. As user you have to provide the proximal operator(s) for your problem and the step size(s).

An example (minimum of a shifted parabola on the unit circle):

```python
dx,dy,radius = 1,0.5,1

def f(X):
    """Shifted parabola"""
    x,y = X
    return (x-dx)**2 + (y-dy)**2

def grad_fx(X):
    """Gradient of f wrt x"""
    xy = X
    return 2*x - 2*dx

def grad_fy(X):
    """Gradient of f wrt y"""
    x,y = X
    return 2*y - 2*dy

def grad_f(X):
    """Gradient of f"""
    return np.array([grad_fx(X),grad_fy(X)])

def prox_circle(X, step):
    """Projection onto circle of radius r"""
    center = np.array([0,0])
    dxy = X - center
    phi = np.arctan2(dxy[1], dxy[0])
    return center + radius*np.array([np.cos(phi), np.sin(phi)])
```

Many proximal operators can be constructed analytically, see e.g. [Parikh & Boyd (2014)](https://web.stanford.edu/~boyd/papers/prox_algs.html). We provide a number of common ones in `proxmin.operators`. An important class of constraints are indicator functions of convex sets, for which the proximal operator, given some point **X**, returns the closes point to **X** in the Euclidean norm that is in the set. That is what `prox_circle` above does.

If the objective function is smooth and there is only one constraint, one can simply perform a sequence of *forward-backward* steps:  step in gradient direction, followed by a projection onto the constraint.

```python
from proxmin import algorithms as pa
def prox_gradf(X, step):
    """Proximal gradient step"""
    return X-step*grad_f(X)

def prox_gradf_circle(X, step):
    """Proximal torward-backward step"""
    return prox_circle(prox_gradf(X,step), step)

# Run proximal gradient method
L = 2         # Lipschitz constant of grad f
step_f = 1./L # maximum step size of smooth function: 1/L
X0 = np.array([-1,0])
# X: updated quantity
# convergence: if iterate difference are smaller than relative error
# error: X^{it} - X^{it-1}
X, convergence, error = pa.pgm(X0, prox_gradf_circle, step_f)
# or with Nesterov acceleration
X, convergence, error = pa.apgm(X0, prox_gradf_circle, step_f)  
```

If the objective function is not smooth, one can use ADMM. This also allows for two functions (the objective and one constraint ) to be satisfied, but it treats them *separately*. Unlike PGM, the constraint is only met at the end of the optimization and only within some error tolerance.

```python
X, convergence, error = pa.admm(X, prox_gradf, step_f, prox_circle, e_rel=1e-3, e_abs=1e-3)
```

A fully working example to demonstrate the principle of operations is [examples/parabola.py] that find the minimum of a 2D parabola under hard boundary constraints (on a shifted circle or the intersection of lines).

## Constrained matrix factorization (CMF)

We have developed this package with a few application cases in mind. One is matrix factorization under constraints on the matrix factors, i.e. describing a target matrix **Y** as a product of **A S**. If those constraints are only non-negativity, the method is known as NMF.

We have extended the capabilities substantially by allowing for an arbitrary number of constraints to be enforced. As above, the constraints and the objective function will be accessed through their proximal operators only.

For a solver, you can simply do this:

```python
from proxmin import nmf
# PGM-like approach for each factor
prox_A = ... # a single constraint on A, solved by projection
prox_S = ... # a single constraint on S, solved by projection
A0, S0 = ... # initialization
A, S = nmf(Y, A0, S0, prox_A=prox_A, prox_S=prox_S)
# for multiple constraints, solved by ADMM-style split
proxs_g = [[...], # list of proxs for A
           [...]] # list of proxs for S
A, S = nmf(Y, A0, S0, proxs_g=proxs_g)
# or a combination
A, S = nmf(Y, A0, S0, prox_A=prox_A, prox_S=prox_S, proxs_g=proxs_g)
```

A complete and practical example is given in [these notebooks](https://github.com/fred3m/hyperspectral) of the hyperspectral unmixing study from our paper.
