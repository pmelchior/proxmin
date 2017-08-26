# Proximal Minimization

The methods in this pure python(>=2.7) package provide solvers for constrained optimization problems. All of them use proximal operators to deal with non-smooth constraint functions.

The algorithms:

* Proximal Gradient Method (PGM): *forward-backward* split with a single smooth function with a Lipschitz-continuous gradient and a single (non-smooth) constraint function. Nesterov acceleration is available.
* Alternating Direction Method of Multipliers (ADMM): Rachford-Douglas split for two potentially non-smooth functions. We use the linearized form of it solve for additional linear mappings in the constraint functions.
* Simultaneous Direction Method of Multipliers (SDMM): Extension of linearized ADMM for several constraint functions.
* Block-Simultaneous Direction Method of Multipliers (bSDMM): Extension of SDMM to work with objective functions that are convex in several arguments. It's a proximal version of Block coordinate descent methods.

In addition, bSDMM is used as the backend of a solver for Non-negative Matrix Factorization (NMF). It allows the employment of an arbitrary number of constraints on each of the matrix factors.

Details can be found in the [paper](http://arxiv.org/abs/XXXXXX) "Block-Simultaneous Direction Method of Multipliers — A proximal splitting algorithm for multiple constraints and multiple variables" by Fred Moolekamp and Peter Melchior.

The code is licensed under the permissive MIT license. We ask that any published work that utilizes this package cites the paper above.

## Dependencies

numpy and scipy