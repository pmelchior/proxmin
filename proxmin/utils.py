from __future__ import print_function, division
import logging

import numpy as np

logging.basicConfig()
logger = logging.getLogger("proxmin.utils")


def l2sq(x):
    """Sum the matrix elements squared
    """
    return (x**2).sum()

def l2(x):
    """Square root of the sum of the matrix elements squared
    """
    return np.sqrt((x**2).sum())

def lipschitz_const(X):
    """Calculate the Lipschitz constant to determine the step size
    """
    return np.real(np.linalg.eigvals(X.dot(X.T)).max())

def get_linearization(constraint, X, Z, U, dot_components):
    """Linearize the constraint on X
    
    Z is the primal variable, U is the dual variable
    """
    return dot_components(constraint.T, dot_components(constraint,X) - Z + U)

def update_variables(X, Z, U, prox_f, step_f, prog_g, step_g, constraints, dot_components):
    """Update the primal and dual variables
    """
    # Lienarize each constraint
    linearization = [step_f/step_g[i] * get_linearization(c, X, Z[i], U[i])
                     for i, c in enumerate(constraints)]
    # Apply the proximal operator to update the variable X^k
    X_ = prox_f(X - np.sum(linearization, axis=0), step=step_f)
    # Iterate over the different constraints
    CX = []
    U_ = Y.copy()
    for i in range(len(constraints)):
        # Apply the constraint for each peak to the peak intensities
        CXi = dot_components[i](constraints[i], X_)
        Z_[i] = proxOps[i](CXi+U[i], step=step_g[i])
        U[i] = U[i] + CXi - Z_[i]
        CX.append(CXi)
    return X_ ,Z_, U, CX

def get_variable_errors(A, AX, Z, U, e_rel, dot_components):
    """Get the errors in a single multiplier method step
    
    For a given linear operator A, (and its dot product with X to save time),
    calculate the errors in the prime and dual variables, used by the
    Boyd 2011 Section 3 stopping criteria.
    """
    e_pri2 = e_rel**2*np.max([l2sq(AX), l2sq(Z), 1])
    if A is None:
        e_dual2 = e_rel**2*l2sq(U)
    else:
        e_dual2 = e_rel**2*l2sq(dot_components(A.T,U))
    return e_pri2, e_dual2

def check_convergence(it, newX, oldX, e_rel, min_iter=10, history=False, **kwargs):
    """Check that the algorithm converges using Langville 2014 criteria
    
    Uses the check from Langville 2014, Section 5, to check if the NMF
    algorithm has converged.
    """
    # Calculate the norm for columns and rows, which can be used for debugging
    # Otherwise skip, since it takes extra processing time
    new_old = newX*oldX
    old2 = oldX**2
    norms = [np.sum(new_old), np.sum(old2)]
    if history:
        norms += [new_old, old2]

    convergent = (it > min_iter) and (norms[0] >= (1-e_rel**2)*norms[1])
    return convergent, norms

def check_column_convergence(it, newX, oldX, e_rel, min_iter=10, **kwargs):
    """Check that the columns of the algorithm converge
    
    Uses the check from Langville 2014, Section 5, to check if the NMF
    algorithm has converged.
    """
    K = newX.shape(0)
    norms = np.zeros((2, K))
    norms[0,:] = [newX[k].dot(oldX[k]) for k in range(K)]
    norms[1,:] = [l2sq(oldX[k]) for k in range(K)]

    convergent = it > min_iter and np.all([ct >= (1-e_rel**2)*o2 for ct,o2 in norms])
    return convergent, norms

def check_row_convergence(it, newX, oldX, e_rel, min_iter=10, **kwargs):
    """Check that the rows of the algorithm converge
    
    Uses the check from Langville 2014, Section 5, to check if the NMF
    algorithm has converged.
    """
    K = newX.shape(1)
    norms = np.zeros((2, K))
    norms[0,:] = [newX[:,k].dot(oldX[:,k]) for k in range(K)]
    norms[1,:] = [l2sq(oldX[:,k]) for k in range(K)]

    convergent = it > min_iter and np.all([ct >= (1-e_rel**2)*o2 for ct,o2 in norms])
    return convergent, norms

def check_diff_convergence(it, newX, oldX, e_rel, min_iter=10, history=False, **kwargs):
    """Check that the algorithm converges using the difference
    
    Uses the differences between X and the old and new step to check for convergence.
    """
    # Calculate the norm for columns and rows, which can be used for debugging
    # Otherwise skip, since it takes extra processing time
    diff2 = (oldX-newX)**2
    old2 = oldX**2
    norms = [np.sum(old2), np.sum(diff2)]
    if history:
        norms += [diff2, old2]

    convergent = (it > min_iter) and (norms[0] >= (1-e_rel**2)*norms[1])
    return convergent, norms

def unpack_convergence_norms(norms, axis=0):
    """Unpack the convergence norms for a given axis
    
    In most schemes, for example NMF, either the rows or columns of a matrix represent
    a given feature, so this allows the user to extract the parameters used to
    calculate convergence.
    """
    if norms[2] is None or norms[3] is None:
        return norms[:2]
    axis_norms = np.zeros((2, norms.shape[axis]))
    axis_norms[0,:] = np.sum(norms[2], axis=axis)
    axis_norms[1,:] = np.sum(norms[3], axis=axis)
    return axis_norms

def check_constraint_convergence(step_f, step_g, X, CX, Z_, Z, U, constraints, e_rel):
    """Calculate if all constraints have converged
    
    Using the stopping criteria from Boyd 2011, Sec 3.3.1, calculate whether the
    variables for each constraint have converged.
    """
    # compute prime residual rk and dual residual sk
    R = [cx-Z_[i] for i, cx in enumerate(CX)]
    S = [-(step_f/step_g[i]) * c.T.dot(Z_[i] - Z[i]) for i, c in enumerate(constraints)]
    # Calculate the error for each constraint
    errors = np.zeros((len(constraints), 4))
    errors[:,:2] = np.array([get_variable_errors(c, CX[i], Z[i], U[i], e_rel)
                                for i, c in enumerate(constraints)])
    errors[:,2] = [l2sq(r) for r in R]
    errors[:,3] = [l2sq(s) for s in S]

    # Check the constraints for convergence
    convergence = [e[2]<=e[0] and e[3]<=e[1] for e in errors]
    return np.all(convergence), errors

def unpack_residual_errors(errors):
    raise NotImplemented