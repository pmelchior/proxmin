from __future__ import print_function, division
import numpy as np


def get_spectral_norm(L):
    if L is None:
        return 1
    elif hasattr(L, "spectral_norm"):
        return L.spectral_norm
    else:  # linearized ADMM
        LTL = L.T.dot(L)
        # need spectral norm of L
        import scipy.sparse

        if scipy.sparse.issparse(L):
            if min(L.shape) <= 2:
                L2 = np.real(np.linalg.eigvals(LTL.toarray()).max())
            else:
                import scipy.sparse.linalg

                L2 = np.real(
                    scipy.sparse.linalg.eigs(LTL, k=1, return_eigenvectors=False)[0]
                )
        else:
            L2 = np.real(np.linalg.eigvals(LTL).max())
        return L2


class MatrixAdapter(object):
    """Matrix adapter to deal with None and per-component application.
    """

    def __init__(self, L, axis=None):
        # prevent cascade
        spec_norm = None
        while isinstance(L, MatrixAdapter):
            spec_norm = L._spec_norm
            axis = L.axis
            L = L.L
        self.L = L
        self.axis = axis
        self._spec_norm = spec_norm

    @property
    def spectral_norm(self):
        if self._spec_norm is None:
            if self.L is not None:
                self._spec_norm = get_spectral_norm(self.L)
            else:
                self._spec_norm = 1
        return self._spec_norm

    @property
    def T(self):
        if self.L is None:
            return self  # NOT: self.L !!!
        # because we need to preserve axis for dot(), create a new adapter
        return MatrixAdapter(self.L.T, axis=self.axis)

    def dot(self, X):
        if self.L is None:
            # CAVEAT: This is not a copy (for performance reasons)
            # so make sure you're not binding it to another variable
            # OK for all temporary arguments X
            return X

        if self.axis is None:
            return self.L.dot(X)

        # axis=0 is not needed because it can be done with a normal matrix
        # dot product
        if self.axis == 1:
            return self.L.dot(X.reshape(-1)).reshape(X.shape[0], -1)
        raise NotImplementedError(
            "MatrixAdapter.dot() is not useful with axis=0.\n"
            "Use regular matrix dot product instead!"
        )

    def __len__(self):
        return len(self.L)

    @property
    def shape(self):
        return self.L.shape

    @property
    def size(self):
        return self.L.size

    @property
    def ndim(self):
        return self.L.ndim


class Traceback(object):
    def __init__(self):
        self._trace = []

    def __call__(self, *X, it=None):
        self._trace.append(tuple(x.copy() for x in X))

    @property
    def trace(self):
        return self._trace

    def clear(self):
        self._trace = []


class NullCallback(object):
    def __call__(self, *X, it):
        pass


class ApproximateCache(object):
    """Cache function evaluations that don't change much

    Certain calculations take a long time but have little change after a few
    iterations, so that these functions become an unnecessary time sink.
    This class allows the user to store these (and similar) values and
    only calculate them when there is a substantial change in the value.

    This method works by comparing the stored value to the last value calculated.
    If the relative difference between the two is less than `slack`/2, then the
    change is deemed to be "small" and the `stride` (number of iterations to skip)
    is increased to skip calculating the value for several iterations.
    """

    def __init__(self, func, slack=0.1, max_stride=100):
        """Constructor

        Parameters
        ----------
        func: function
            The slow function that calculates the value
        slack: float, default=0.1
            A measure of how much the value can differ before
            it needs to be recalculated
        max_stride: int, default=100
            Maximum `stride` between calculations of the value.
        """
        self.func = func
        assert slack >= 0 and slack < 1
        self.slack = slack
        self.max_stride = max_stride
        self.it = 0
        self.stride = 1
        self.last = -1
        self.stored = None

    def __len__(self):
        """The current `stride` value
        """
        return len(self.stride)

    def __call__(self, *args, **kwargs):
        """Calculate the value (if necessary)

        This method checks whether or not the value needs to be recalculated,
        and if so, calculates it and sets the number of `stride` (iterations)
        until it is calculated again.
        """
        if self.slack == 0:
            self.it += 1
            return self.func(*args, **kwargs)
        if self.it >= self.last + self.stride:
            self.last = self.it
            val = self.func(*args, **kwargs)

            # increase stride when rel. changes in L are smaller than (1-slack)/2
            if self.it > 1 and self.slack > 0:
                rel_error = np.abs(self.stored - val) / self.stored
                budget = self.slack / 2
                if rel_error < budget and rel_error > 0:
                    self.stride += max(1, int(budget / rel_error * self.stride))
                    self.stride = min(self.max_stride, self.stride)
            # updated last value
            self.stored = val
        else:
            self.it += 1
        return self.stored


class NesterovStepper(object):
    def __init__(self, accelerated=False):
        self.t = 1.0
        self.accelerated = accelerated

    @property
    def omega(self):
        if self.accelerated:
            t_ = 0.5 * (1 + np.sqrt(4 * self.t * self.t + 1))
            om = (self.t - 1) / t_
            self.t = t_
            return om
        else:
            return 0


def initZU(X, L):
    if not isinstance(L, list):
        Z = L.dot(X).copy()
        U = np.zeros(Z.shape, dtype=Z.dtype)
    else:
        Z = []
        U = []
        for i in range(len(L)):
            Z.append(L[i].dot(X).copy())
            U.append(np.zeros(Z[i].shape, dtype=Z[i].dtype))
    return Z, U


def l2sq(x):
    """Sum the matrix elements squared
    """
    return (x ** 2).sum()


def l2(x):
    """Square root of the sum of the matrix elements squared
    """
    return np.sqrt((x ** 2).sum())


def get_step_g(step_f, norm_L2, N=1, M=1):
    """Get step_g compatible with step_f (and L) for ADMM, SDMM, GLMM.
    """
    # Nominally: minimum step size is step_f * norm_L2
    # see Parikh 2013, sect. 4.4.2
    #
    # BUT: For multiple constraints, need to multiply by M.
    # AND: For multiple variables, need to multiply by N.
    # Worst case of constraints being totally correlated, otherwise Z-updates
    # overwhelm X-updates entirely -> blow-up
    return step_f * norm_L2 * N * M


def get_step_f(step_f, lR2, lS2):
    """Update the stepsize of given the primal and dual errors.

    See Boyd (2011), section 3.4.1
    """
    mu, tau = 10, 2
    if lR2 > mu * lS2:
        return step_f * tau
    elif lS2 > mu * lR2:
        return step_f / tau
    return step_f


def do_the_mm(X, step_f, Z, U, prox_g, step_g, L):
    LX = L.dot(X)
    Z_ = prox_g(LX + U, step_g)
    # primal and dual errors
    R = LX - Z_
    S = -1 / step_g * L.T.dot(Z_ - Z)
    Z[:] = Z_[:]  # force the copy
    # this uses relaxation parameter of 1
    U[:] += R
    return LX, R, S


def update_variables(X, Z, U, prox_f, step_f, prox_g, step_g, L):
    """Update the primal and dual variables

    Note: X, Z, U are updated inline

    Returns: LX, R, S
    """
    if not hasattr(prox_g, "__iter__"):
        if prox_g is not None:
            dX = step_f / step_g * L.T.dot(L.dot(X) - Z + U)
            X[:] = prox_f(X - dX, step_f)
            LX, R, S = do_the_mm(X, step_f, Z, U, prox_g, step_g, L)
        else:
            # fall back to simple fixed-point method for f
            # see do_the_mm for normal definitions of LX,Z,R,S
            S = -X.copy()
            X[:] = prox_f(X, step_f)
            LX = X
            Z[:] = X[:]
            R = np.zeros(X.shape, dtype=X.dtype)
            S += X

    else:
        M = len(prox_g)
        dX = np.sum(
            [
                step_f / step_g[i] * L[i].T.dot(L[i].dot(X) - Z[i] + U[i])
                for i in range(M)
            ],
            axis=0,
        )
        X[:] = prox_f(X - dX, step_f)
        LX = [None] * M
        R = [None] * M
        S = [None] * M
        for i in range(M):
            LX[i], R[i], S[i] = do_the_mm(
                X, step_f, Z[i], U[i], prox_g[i], step_g[i], L[i]
            )
    return LX, R, S


def get_variable_errors(X, L, LX, Z, U, step_g, e_rel, e_abs=0):
    """Get the errors in a single multiplier method step

    For a given linear operator A, (and its dot product with X to save time),
    calculate the errors in the prime and dual variables, used by the
    Boyd 2011 Section 3 stopping criteria.
    """
    n = X.size
    p = Z.size
    e_pri2 = np.sqrt(p) * e_abs / L.spectral_norm + e_rel * np.max([l2(LX), l2(Z)])
    if step_g is not None:
        e_dual2 = np.sqrt(n) * e_abs / L.spectral_norm + e_rel * l2(L.T.dot(U) / step_g)
    else:
        e_dual2 = np.sqrt(n) * e_abs / L.spectral_norm + e_rel * l2(L.T.dot(U))
    return e_pri2, e_dual2


def check_constraint_convergence(X, L, LX, Z, U, R, S, step_f, step_g, e_rel, e_abs):
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
            c, e = check_constraint_convergence(
                X, L[i], LX[i], Z[i], U[i], R[i], S[i], step_f, step_g[i], e_rel, e_abs
            )
            convergence &= c
            errors.append(e)
        return convergence, errors
    else:
        # check convergence of prime residual R and dual residual S
        e_pri, e_dual = get_variable_errors(X, L, LX, Z, U, step_g, e_rel, e_abs)
        lR = l2(R)
        lS = l2(S)
        convergence = (lR <= e_pri) and (lS <= e_dual)
        return convergence, (e_pri, e_dual, lR, lS)


def check_convergence(newX, oldX, e_rel):
    """Check that the algorithm converges using Langville 2014 criteria

    Uses the check from Langville 2014, Section 5, to check if the NMF
    algorithm has converged.
    """
    # Calculate the norm for columns and rows, which can be used for debugging
    # Otherwise skip, since it takes extra processing time
    new_old = newX * oldX
    old2 = oldX ** 2
    norms = [np.sum(new_old), np.sum(old2)]
    convergent = norms[0] >= (1 - e_rel ** 2) * norms[1]
    return convergent, norms


def hasNotNone(l):
    i = 0
    for ll in l:
        if ll is not None:
            if hasattr(ll, "__iter__"):
                for lll in ll:
                    if lll is not None:
                        return len(l) - i
        i += 1
    return 0
