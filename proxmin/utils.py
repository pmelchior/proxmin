from __future__ import print_function, division
import logging
import numpy as np

logging.basicConfig()
logger = logging.getLogger("proxmin.utils")

class MatrixOrNone(object):
    """Matrix adapter to deal with the absence of a matrix.
    """
    def __init__(self, L):
        # prevent cascade
        if isinstance(L, MatrixOrNone):
            self.L = L.L
        else:
            self.L = L

    @property
    def T(self):
        if self.L is None:
            return self # NOT: self.L !!!
        return self.L.T

    def dot(self, X):
        if self.L is None:
             # CAVEAT: This is not a copy (for performance reasons)
             # so make sure you're not binding it to another variable
             # OK for all temporary arguments X
            return X
        return self.L.dot(X)

class Traceback(object):
    """Container structure for traceback of algorithm behavior.
    """
    def __init__(self, it=None, Z=None, U=None, errors=None, history=None):
        self.it = it
        self.Z = Z
        self.U = U
        self.errors = errors
        self.history = history

    def __repr__(self):
        message = "Traceback:\n"
        for k,v in self.__dict__.iteritems():
            message += "\t%s: %r\n" % (k,v)
        return message

def initXZU(X0, L):
    X = X0.copy()
    if not isinstance(L, list):
        Z = L.dot(X).copy()
        U = np.zeros_like(Z)
    else:
        Z = []
        U = []
        for i in range(len(L)):
            Z.append(L[i].dot(X).copy())
            U.append(np.zeros_like(Z[i]))
    return X,Z,U

def l2sq(x):
    """Sum the matrix elements squared
    """
    return (x**2).sum()

def l2(x):
    """Square root of the sum of the matrix elements squared
    """
    return np.sqrt((x**2).sum())

def get_spectral_norm(L):
    if L is None:
        return 1
    else: # linearized ADMM
        LTL = L.T.dot(L)
        # need spectral norm of L
        import scipy.sparse
        if scipy.sparse.issparse(L):
            if min(L.L.shape) <= 2:
                L2 = np.linalg.eigvals(LTL.toarray()).max()
            else:
                import scipy.sparse.linalg
                L2 = np.real(scipy.sparse.linalg.eigs(LTL, k=1, return_eigenvectors=False)[0])
        else:
            L2 = np.linalg.eigvals(LTL).max()
        return L2

def get_step_g(step_f, norm_L2, step_g=None):
    """Get step_g compatible with step_f (and L) for ADMM, SDMM, GLMM.
    """
    if step_g is None:
        return step_f * norm_L2
    else:
        assert step_f <= step_g / norm_L2
        return step_g

def get_step_f(step_f, lR2, lS2):
    """Update the stepsize of given the primal and dual errors.

    See Boyd (2011), section 3.4.1
    """
    mu, tau = 10, 2
    if lR2 > mu*lS2:
        return step_f * tau
    elif lS2 > mu*lR2:
        return step_f / tau
    return step_f

def do_the_mm(X, step_f, Z, U, prox_g, step_g, L):
    LX = L.dot(X)
    Z_ = prox_g(LX + U, step_g)
    # primal and dual errors
    R = LX - Z_
    S = -step_f/step_g * L.T.dot(Z_ - Z)
    Z[:] = Z_[:] # force the copy
    # this uses relaxation parameter of 1
    U[:] += R
    return LX, R, S

def update_variables(X, Z, U, prox_f, step_f, prox_g, step_g, L):
    """Update the primal and dual variables

    Note: X, Z, U are updated inline

    Returns: LX, R, S
    """
    if not isinstance(prox_g, list):
        dX = step_f/step_g * L.T.dot(L.dot(X) - Z + U)
        X[:] = prox_f(X - dX, step_f)
        LX, R, S = do_the_mm(X, step_f, Z, U, prox_g, step_g, L)
    else:
        M = len(prox_g)
        dX = np.sum([step_f/step_g[i] * L[i].T.dot(L[i].dot(X) - Z[i] + U[i]) for i in range(M)], axis=0)
        X[:] = prox_f(X - dX, step_f)
        LX = [None] * M
        R = [None] * M
        S = [None] * M
        for i in range(M):
            LX[i], R[i], S[i] = do_the_mm(X, step_f, Z[i], U[i], prox_g[i], step_g[i], L[i])
    return LX, R, S

def get_variable_errors(L, LX, Z, U, e_rel):
    """Get the errors in a single multiplier method step

    For a given linear operator A, (and its dot product with X to save time),
    calculate the errors in the prime and dual variables, used by the
    Boyd 2011 Section 3 stopping criteria.
    """
    e_pri2 = e_rel**2*np.max([l2sq(LX), l2sq(Z)])
    e_dual2 = e_rel**2*l2sq(L.T.dot(U))
    return e_pri2, e_dual2

def check_constraint_convergence(L, LX, Z, U, R, S, e_rel):
    """Calculate if all constraints have converged.

    Using the stopping criteria from Boyd 2011, Sec 3.3.1, calculate whether the
    variables for each constraint have converged.
    """

    if isinstance(L, list):
        M = len(L)
        convergence = True
        errors = []
        # recursive call
        for i in range(M):
            c, e = check_constraint_convergence(L[i], LX[i], Z[i], U[i], R[i], S[i], e_rel)
            convergence &= c
            errors.append(e)
        return convergence, errors
    else:
        # check convergence of prime residual R and dual residual S
        e_pri2, e_dual2 = get_variable_errors(L, LX, Z, U, e_rel)
        lR2 = l2sq(R)
        lS2 = l2sq(S)
        convergence = (lR2 <= e_pri2 or np.isclose(lR2, e_pri2, atol=e_rel**2)) and (lS2 <= e_dual2 or np.isclose(lS2, e_dual2, atol=e_rel**2))
        return convergence, (e_pri2, e_dual2, lR2, lS2)

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




def unpack_residual_errors(errors):
    raise NotImplemented
