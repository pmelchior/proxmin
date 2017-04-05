from __future__ import print_function, division
import numpy as np, scipy.sparse, scipy.sparse.linalg
from functools import partial

import logging

logging.basicConfig()
logger = logging.getLogger("lsst.meas.deblender.proximal_nmf")

# identity
def prox_id(X, step):
    return X

# projection onto 0
def prox_zero(X, step):
    return np.zeros_like(X)

# hard thresholding: X if X >= k, otherwise 0
# NOTE: modifies X in place
def prox_hard(X, step, l=0):
    below = X - l*step < 0
    X[below] = 0
    return X

# projection onto non-negative numbers
def prox_plus(X, step):
    return prox_hard(X, step, l=0)

# projection onto numbers above l
# NOTE: modifies X in place
def prox_min(X, step, l=0):
    below = X - l*step < 0
    X[below] = l*step
    return X

# soft thresholding operator
def prox_soft(X, step, l=0):
    return np.sign(X)*prox_plus(np.abs(X) - l*step, step)

# same but with projection onto non-negative
def prox_soft_plus(X, step, l=0):
    return prox_plus(prox_soft(X, step, l=l), step)

# projection onto sum=1 along each axis
def prox_unity(X, step, axis=0):
    return X / np.sum(X, axis=axis, keepdims=True)

# same but with projection onto non-negative
def prox_unity_plus(X, step, axis=0):
    return prox_unity(prox_plus(X, step), step, axis=axis)

def l2sq(x):
    return (x**2).sum()

def l2(x):
    return np.sqrt((x**2).sum())

def convolve_band(P, I):
    if isinstance(P, list) is False:
        return P.dot(I.T).T
    else:
        PI = np.empty(I.shape)
        B = I.shape[0]
        for b in range(B):
            PI[b] = P[b].dot(I[b])
        return PI

def delta_data(A, S, Y, W=1, P=None):
    if P is None:
        return W*(np.dot(A,S) - Y)
    else:
        # all the tranposes are needed to allow for the sparse matrix dot
        # products to be callable
        if isinstance(P, list) is False:
            return P.T.dot(W.T*(P.dot(np.dot(S.T, A.T)) - Y.T)).T
        else:
            B,N = A.shape[0], S.shape[1]
            EWP = np.empty((B,N))
            for b in range(B):
                EWP[b] = P[b].T.dot(W[b]*(P[b].dot(np.dot(S.T, A[b])) - Y[b]))
            return EWP

def grad_likelihood_A(A, S, Y, W=1, P=None):
    D = delta_data(A, S, Y, W=W, P=P)
    return np.dot(D, S.T)

def grad_likelihood_S(S, A, Y, W=1, P=None):
    D = delta_data(A, S, Y, W=W, P=P)
    return np.dot(A.T, D)

# executes one proximal step of likelihood gradient, folloVed by prox_g
def prox_likelihood_A(A, step, S=None, Y=None, prox_g=None, W=1, P=None):
    return prox_g(A - step*grad_likelihood_A(A, S, Y, W=W, P=P), step)

def prox_likelihood_S(S, step, A=None, Y=None, prox_g=None, W=1, P=None):
    return prox_g(S - step*grad_likelihood_S(S, A, Y, W=W, P=P), step)

# split X into K components along axis
# apply prox_list[k] to each component k
# stack results to reconstruct shape of X
def prox_components(X, step, prox_list=[], axis=0):
    assert X.shape[axis] == len(prox_list)
    K = X.shape[axis]

    if np.isscalar(step):
        step = [step for k in range(K)]

    if axis == 0:
        Pk = [prox_list[k](X[k], step[k]) for k in range(K)]
    if axis == 1:
        Pk = [prox_list[k](X[:,k], step[k]) for k in range(K)]
    return np.stack(Pk, axis=axis)

def dot_components(C, X, axis=0, transpose=False):
    assert X.shape[axis] == len(C)
    K = X.shape[axis]

    if axis == 0:
        if not transpose:
            CX = [C[k].dot(X[k]) for k in range(K)]
        else:
            CX = [C[k].T.dot(X[k]) for k in range(K)]
    if axis == 1:
        if not transpose:
            CX = [C[k].dot(X[:,k]) for k in range(K)]
        else:
            CX = [C[k].T.dot(X[:,k]) for k in range(K)]
    return np.stack(CX, axis=axis)



# accelerated proximal gradient method
# Combettes 2009, Algorithm 3.6
def APGM(X, prox, step, e_rel=1e-6, max_iter=1000):
    Z = X.copy()
    t = 1.
    for it in range(max_iter):
        X_ = prox(Z, step)
        t_ = 0.5*(1 + np.sqrt(4*t*t + 1))
        gamma = 1 + (t - 1)/t_
        Z = X + gamma*(X_ - X)

        # test for fixed point convergence
        if l2sq(X - X_) <= e_rel**2*l2sq(X):
            X[:] = X_[:]
            break

        t = t_
        X[:] = X_[:]

    return it

def check_NMF_convergence(it, newX, oldX, e_rel, K):
    """Check the that NMF converges
    
    Uses the check from Langville 2014, Section 5, to check if the NMF
    deblender has converged
    """
    result = it > 10 and np.array([np.dot(newX[k], oldX[k]) > (1-e_rel**2)*l2sq(newX[k])
                                   for k in range(K)]).all()
    return result

def get_variable_errors(A, AX, Z, U, e_rel):
    """Get the errors in a single multiplier method step
    
    For a given linear operator A, (and its dot product with X to save time),
    calculate the errors in the prime and dual variables, used by the
    Boyd 2011 Section 3 stopping criteria.
    """
    e_pri2 = e_rel**2*max(l2sq(AX), l2sq(Z))
    if A is None:
        e_dual2 = e_rel**2*l2sq(U)
    else:
        e_dual2 = e_rel**2*l2sq(dot_components(A, U, transpose=True))
    return e_pri2, e_dual2

def get_quadratic_regularization(A, X, Z, U):
    """Get the quadratic regularization term for a linear operator
    
    Using the method of augmented Lagrangians requires a quadratic regularization used
    to updated the X matrix in ADMM. This method calculates those terms.
    """
    return dot_components(A, dot_components(A, X) - Z + U, transpose=True)

# Alternating direction method of multipliers
# initial: initial guess of solution
# K: number of iterations
# A: KxN*M: minimizes f(X) + g(AX) for each component k (i.e. rows of X)
# See Boyd+2011, Section 3, with arbitrary A, B=-Id, c=0
def ADMM(X0, prox_f, step_f, prox_g, step_g, A=None, max_iter=1000, e_rel=1e-3):

    if A is None:
        U = np.zeros_like(X0)
        Z = X0.copy()
    else:
        X = X0.copy()
        Z = dot_components(A, X)
        U = np.zeros_like(Z)

    for it in range(max_iter):
        if A is None:
            X = prox_f(Z - U, step_f)
            AX = X
        else:
            X = prox_f(X - (step_f/step_g)[:,None] * get_quadratic_regularization(A, X, Z, U), step_f)
            AX = dot_components(A, X)
        Z_ = prox_g(AX + U, step_g)
        # this uses relaxation parameter of 1
        U = U + AX - Z_

        # compute prime residual rk and dual residual sk
        R = AX - Z_
        if A is None:
            S = -(Z_ - Z)
        else:
            S = -(step_f/step_g)[:,None] * dot_components(A, Z_ - Z, transpose=True)
        Z = Z_

        # stopping criteria from Boyd+2011, sect. 3.3.1
        # only relative errors
        e_pri2, e_dual2 = get_variable_errors(A, AX, Z, U, e_rel)
        if l2sq(R) <= e_pri2 and l2sq(S) <= e_dual2:
            break

    return it, X, Z, U

def update_sdmm_variables(X, Y, Z, prox_f, step_f, proxOps, proxSteps, constraints):
    """Update the prime and dual variables for multiple linear constraints
    
    Both SDMM and GLM require the same method of updating the prime and dual
    variables for the intensity matrix linear constraints.
    
    """
    linearization = [step_f/proxSteps[i] * get_quadratic_regularization(c, X, Y[i], Z[i])
                     for i, c in enumerate(constraints)]
    X_ = prox_f(X - np.sum(linearization, axis=0), step=step_f)
    # Iterate over the different constraints
    CX = []
    for i in range(len(constraints)):
        # Apply the constraint for each peak to the peak intensities
        CXi = dot_components(constraints[i], X_)
        # TODO: the following is wrong!
        #CXi = dot_components(constraints[i], X)
        Y[i] = proxOps[i](CXi+Z[i], step=proxSteps[i])
        Z[i] = Z[i] + CXi - Y[i]
        CX.append(CXi)
    return X_ ,Y, Z, CX

def test_multiple_constraint_convergence(step_f, step_g, X, CX, Z_, Z, U, constraints, e_rel):
    """Calculate if all constraints have converged
    
    Using the stopping criteria from Boyd 2011, Sec 3.3.1, calculate whether the
    variables for each constraint have converged.
    """
    # compute prime residual rk and dual residual sk
    R = [cx-Z_[i] for i, cx in enumerate(CX)]
    S = [-(step_f/step_g[i]) * dot_components(c, Z_[i] - Z[i], transpose=True)
         for i, c in enumerate(constraints)]
    # Calculate the error for each constraint
    errors = [get_variable_errors(c, CX[i], Z[i], U[i], e_rel) for i, c in enumerate(constraints)]
    # Check the constraints for convergence
    convergence = [l2sq(R[i])<= e_pri2 and l2sq(S[i])<= e_dual2
                    for i, (e_pri2, e_dual2) in enumerate(errors)]
    return np.all(convergence), convergence

def SDMM(X0, prox_f, step_f, prox_g, step_g, constraints, max_iter=1000, e_rel=1e-3):
    """Implement Simultaneous-Direction Method of Multipliers
    
    This implements the SDMM algorithm derived from Algorithm 7.9 from Combettes and Pesquet (2009),
    Section 4.4.2 in Parikh and Boyd (2013), and Eq. 2.2 in Andreani et al. (2007).
    
    In Combettes and Pesquet (2009) they use a matrix inverse to solve the problem.
    In our case that is the inverse of a sparse matrix, which is no longer sparse and too
    costly to implement.
    The `scipy.sparse.linalg` module does have a method to solve a sparse matrix equation,
    using Algorithm 7.9 directly still does not yield the correct result,
    as the treatment of penalties due to constraints are on equal footing with our likelihood
    proximal operator and require a significant change in the way we calculate step sizes to converge.
    
    Instead we calculate the constraint vectors (as in SDMM) but extend the update of the ``X`` matrix
    using a modified version of the ADMM X update function (from Parikh and Boyd, 2009),
    using an augmented Lagrangian for multiple linear constraints as given in Andreani et al. (2007).
    
    In the language of Combettes and Pesquet (2009), constraints = list of Li,
    proxOps = list of ``prox_{gamma g i}``.
    """
    # Initialization
    X = X0.copy()
    N,M = X0.shape
    Z = np.zeros((len(constraints), N, M))
    U = np.zeros_like(Z)
    for c, C in enumerate(constraints):
        Z[c] = dot_components(C,X)

    # Update the constrained matrix
    for n in range(max_iter):
        # Update the variables
        X_, Z_, U, CX = update_sdmm_variables(X, Z, U, prox_f, step_f, prox_g, step_g, constraints)
        # Check for convergence
        convergence, c = test_multiple_constraint_convergence(step_f, step_g, X, CX, Z_, Z,
                                                              U, constraints, e_rel)

        X = X_
        Z = Z_
        if convergence:
            break
    return n, X, Z, U

def GLM(data, X10, X20, W, P,
        prox_f1, prox_f2, prox_g1, prox_g2,
        constraints1, constraints2, lM1, lM2, max_iter=1000, e_rel=1e-3, beta=1):
    """ Solve for both the SED and Intensity Matrices at the same time
    """
    # Initialize SED matrix
    X1 = X10.copy()
    N1, M1 = X1.shape
    # TODO: Allow for constraints
    #Y1 = np.zeros((len(constraints1), N1, M1))
    #Z1 = np.zeros_like(Y1)
    Z1 = np.zeros_like(X1)
    U1 = np.zeros_like(Z1)

    # Initialize Intensity matrix
    X2 = X20.copy()
    N2, M2 = X2.shape
    Z2 = np.zeros((len(constraints2), N2, M2))
    U2 = np.zeros_like(Z2)

    # Initialize Other Parameters
    K = X2.shape[0]
    if W is not None:
        W_max = W.max()
    else:
        W = W_max = 1

    # Evaluate the solution
    logger.info("Beginning Loop")
    for it in range(max_iter):
        # Step updates might need to be fixed, this is just a guess using the step updates from ALS
        step_f1 = beta**it / lipschitz_const(X2) / W_max

        # Update SED matrix
        prox_like_f1 = partial(prox_likelihood_A, S=X2, Y=data, prox_g=prox_f1, W=W, P=P)
        # TODO: Implement the more general version using `update_sdmm_variables`
        X1 = prox_like_f1(Z1-U1, step_f1)
        Z1 = prox_f1(X1+U1, step_f1)
        U1 = U1 + X1 - Z1

        # Update Intensity Matrix
        step_f2 = beta**it / lipschitz_const(X1) / W_max
        step_g2 = step_f2 * lM2
        prox_like_f2 = partial(prox_likelihood_S, A=X1, Y=data, prox_g=prox_f2, W=W, P=P)
        X2_, Z2_, U2, CX = update_sdmm_variables(X2, Z2, U2, prox_like_f2, step_f2, prox_g2, step_g2, constraints2)

        ## Convergence crit from Langville 2014, section 5
        #X2_converge = check_intensity_convergence(it, X2_, X2, e_rel, K)
        X2_converge = check_NMF_convergence(it, X2_, X2, e_rel, K)

        # ADMM Convergence Criteria, adapted from Boyd 2011, Sec 3.3.1
        convergence, c = test_multiple_constraint_convergence(step_f2, step_g2, X2_, CX, Z2_, Z2,
                                                              U2, constraints2, e_rel)

        # For now just use the Langville 2014 convergence criteria
        if X2_converge and convergence:
            X2 = X2_
            break

        X2 = X2_
        Z2 = Z2_
    logger.info("{0} iterations".format(it))
    return X1, X2

def lipschitz_const(M):
    return np.real(np.linalg.eigvals(np.dot(M, M.T)).max())

def getPeakSymmetry(shape, px, py, fillValue=0):
    """Build the operator to symmetrize a the intensities for a single row
    """
    center = (np.array(shape)-1)/2.0
    # If the peak is centered at the middle of the footprint,
    # make the entire footprint symmetric
    if px==center[1] and py==center[0]:
        return scipy.sparse.coo_matrix(np.fliplr(np.eye(shape[0]*shape[1])))

    # Otherwise, find the bounding box that contains the minimum number of pixels needed to symmetrize
    if py<(shape[0]-1)/2.:
        ymin = 0
        ymax = 2*py+1
    elif py>(shape[0]-1)/2.:
        ymin = 2*py-shape[0]+1
        ymax = shape[0]
    else:
        ymin = 0
        ymax = shape[0]
    if px<(shape[1]-1)/2.:
        xmin = 0
        xmax = 2*px+1
    elif px>(shape[1]-1)/2.:
        xmin = 2*px-shape[1]+1
        xmax = shape[1]
    else:
        xmin = 0
        xmax = shape[1]

    fpHeight, fpWidth = shape
    fpSize = fpWidth*fpHeight
    tWidth = xmax-xmin
    tHeight = ymax-ymin
    extraWidth = fpWidth-tWidth
    pixels = (tHeight-1)*fpWidth+tWidth

    # This is the block of the matrix that symmetrizes intensities at the peak position
    subOp = np.eye(pixels, pixels)
    for i in range(0,tHeight-1):
        for j in range(extraWidth):
            idx = (i+1)*tWidth+(i*extraWidth)+j
            subOp[idx, idx] = fillValue
    subOp = np.fliplr(subOp)

    smin = ymin*fpWidth+xmin
    smax = (ymax-1)*fpWidth+xmax
    if fillValue!=0:
        symmetryOp = np.identity(fpSize)*fillValue
    else:
        symmetryOp = np.zeros((fpSize, fpSize))
    symmetryOp[smin:smax,smin:smax] = subOp

    # Return a sparse matrix, which greatly speeds up the processing
    return scipy.sparse.coo_matrix(symmetryOp)

def getPeakSymmetryOp(shape, px, py, fillValue=0):
    """Operator to calculate the difference from the symmetric intensities
    """
    symOp = getPeakSymmetry(shape, px, py, fillValue)
    diffOp = scipy.sparse.identity(symOp.shape[0])-symOp
    # In cases where the symmetry operator is very small (eg. a small isolated source)
    # scipy doesn't return a sparse matrix, so we test whether or not the matrix is sparse
    # and if it is, use a sparse matrix that works best with the proximal operators.
    if hasattr(diffOp, "tocoo"):
        diffOp = diffOp.tocoo()
    return diffOp

def getOffsets(width, coords=None):
    """Get the offset and slices for a sparse band diagonal array

    For an operator that interacts with its neighbors we want a band diagonal matrix,
    where each row describes the 8 pixels that are neighbors for the reference pixel
    (the diagonal). Regardless of the operator, these 8 bands are always the same,
    so we make a utility function that returns the offsets (passed to scipy.sparse.diags).

    See `diagonalizeArray` for more on the slices and format of the array used to create
    NxN operators that act on a data vector.
    """
    # Use the neighboring pixels by default
    if coords is None:
        coords = [(-1,-1), (-1,0), (-1, 1), (0,-1), (0,1), (1, -1), (1,0), (1,1)]
    offsets = [width*y+x for y,x in coords]
    slices = [slice(None, s) if s<0 else slice(s, None) for s in offsets]
    slicesInv = [slice(-s, None) if s<0 else slice(None, -s) for s in offsets]
    return offsets, slices, slicesInv

def diagonalizeArray(arr, shape=None, dtype=np.float64):
    """Convert an array to a matrix that compares each pixel to its neighbors

    Given an array with length N, create an 8xN array, where each row will be a
    diagonal in a diagonalized array. Each column in this matrix is a row in the larger
    NxN matrix used for an operator, except that this 2D array only contains the values
    used to create the bands in the band diagonal matrix.

    Because the off-diagonal bands have less than N elements, ``getOffsets`` is used to
    create a mask that will set the elements of the array that are outside of the matrix to zero.

    ``arr`` is the vector to diagonalize, for example the distance from each pixel to the peak,
    or the angle of the vector to the peak.

    ``shape`` is the shape of the original image.
    """
    if shape is None:
        height, width = arr.shape
        data = arr.flatten()
    elif len(arr.shape)==1:
        height, width = shape
        data = np.copy(arr)
    else:
        raise ValueError("Expected either a 2D array or a 1D array and a shape")
    size = width * height

    # We hard code 8 rows, since each row corresponds to a neighbor
    # of each pixel.
    diagonals = np.zeros((8, size), dtype=dtype)
    mask = np.ones((8, size), dtype=bool)
    offsets, slices, slicesInv = getOffsets(width)
    for n, s in enumerate(slices):
        diagonals[n][slicesInv[n]] = data[s]
        mask[n][slicesInv[n]] = 0

    # Create a mask to hide false neighbors for pixels on the edge
    # (for example, a pixel on the left edge should not be connected to the
    # pixel to its immediate left in the flattened vector, since that pixel
    # is actual the far right pixel on the row above it).
    mask[0][np.arange(1,height)*width] = 1
    mask[2][np.arange(height)*width-1] = 1
    mask[3][np.arange(1,height)*width] = 1
    mask[4][np.arange(1,height)*width-1] = 1
    mask[5][np.arange(height)*width] = 1
    mask[7][np.arange(1,height-1)*width-1] = 1

    return diagonals, mask

def diagonalsToSparse(diagonals, shape, dtype=np.float64):
    """Convert a diagonalized array into a sparse diagonal matrix

    ``diagonalizeArray`` creates an 8xN array representing the bands that describe the
    interactions of a pixel with its neighbors. This function takes that 8xN array and converts
    it into a sparse diagonal matrix.

    See `diagonalizeArray` for the details of the 8xN array.
    """
    height, width = shape
    offsets, slices, slicesInv = getOffsets(width)
    diags = [diag[slicesInv[n]] for n, diag in enumerate(diagonals)]

    # This block hides false neighbors for the edge pixels (see comments in diagonalizeArray code)
    # For now we assume that the mask in diagonalizeArray has already been applied, making these
    # lines redundant and unecessary, but if that changes in the future we can uncomment them
    #diags[0][np.arange(1,height-1)*width-1] = 0
    #diags[2][np.arange(height)*width] = 0
    #diags[3][np.arange(1,height)*width-1] = 0
    #diags[4][np.arange(1,height)*width-1] = 0
    #diags[5][np.arange(height)*width] = 0
    #diags[7][np.arange(1,height-1)*width-1] = 0

    diagonalArr = scipy.sparse.diags(diags, offsets, dtype=dtype)
    return diagonalArr

def getRadialMonotonicOp(shape, px, py, useNearest=True, minGradient=1):
    """Create an operator to constrain radial monotonicity

    This version of the radial monotonicity operator selects all of the pixels closer to the peak
    for each pixel and weights their flux based on their alignment with a vector from the pixel
    to the peak. In order to quickly create this using sparse matrices, its construction is a bit opaque.
    """
    # Calculate the distance between each pixel and the peak
    size = shape[0]*shape[1]
    x = np.arange(shape[1])
    y = np.arange(shape[0])
    X,Y = np.meshgrid(x,y)
    X = X - px
    Y = Y - py
    distance = np.sqrt(X**2+Y**2)

    # Find each pixels neighbors further from the peak and mark them as invalid
    # (to be removed later)
    distArr, mask = diagonalizeArray(distance, dtype=np.float64)
    relativeDist = (distance.flatten()[:,None]-distArr.T).T
    invalidPix = relativeDist<=0

    # Calculate the angle between each pixel and the x axis, relative to the peak position
    # (also avoid dividing by zero and set the tan(infinity) pixel values to pi/2 manually)
    inf = X==0
    tX = X.copy()
    tX[inf] = 1
    angles = np.arctan2(-Y,-tX)
    angles[inf&(Y!=0)] = 0.5*np.pi*np.sign(angles[inf&(Y!=0)])

    # Calcualte the angle between each pixel and it's neighbors
    xArr, m = diagonalizeArray(X)
    yArr, m = diagonalizeArray(Y)
    dx = (xArr.T-X.flatten()[:, None]).T
    dy = (yArr.T-Y.flatten()[:, None]).T
    # Avoid dividing by zero and set the tan(infinity) pixel values to pi/2 manually
    inf = dx==0
    dx[inf] = 1
    relativeAngles = np.arctan2(dy,dx)
    relativeAngles[inf&(dy!=0)] = 0.5*np.pi*np.sign(relativeAngles[inf&(dy!=0)])

    # Find the difference between each pixels angle with the peak
    # and the relative angles to its neighbors, and take the
    # cos to find its neighbors weight
    dAngles = (angles.flatten()[:, None]-relativeAngles.T).T
    cosWeight = np.cos(dAngles)
    # Mask edge pixels, array elements outside the operator (for offdiagonal bands with < N elements),
    # and neighbors further from the peak than the reference pixel
    cosWeight[invalidPix] = 0
    cosWeight[mask] = 0

    if useNearest:
        # Only use a single pixel most in line with peak
        cosNorm = np.zeros_like(cosWeight)
        columnIndices =  np.arange(cosWeight.shape[1])
        maxIndices = np.argmax(cosWeight, axis=0)
        indices = maxIndices*cosNorm.shape[1]+columnIndices
        indices = np.unravel_index(indices, cosNorm.shape)
        cosNorm[indices] = minGradient
        # Remove the reference for the peak pixel
        cosNorm[:,px+py*shape[1]] = 0
    else:
        # Normalize the cos weights for each pixel
        normalize = np.sum(cosWeight, axis=0)
        normalize[normalize==0] = 1
        cosNorm = (cosWeight.T/normalize[:,None]).T
        cosNorm[mask] = 0
    cosArr = diagonalsToSparse(cosNorm, shape)

    # The identity with the peak pixel removed represents the reference pixels
    diagonal = np.ones(size)
    diagonal[px+py*shape[1]] = -1
    monotonic = cosArr-scipy.sparse.diags(diagonal)

    return monotonic.tocoo()

def getPSFOp(psfImg, imgShape, threshold=1e-2):
    """Create an operator to convolve intensities with the PSF

    Given a psf image ``psfImg`` and the shape of the blended image ``imgShape``,
    make a banded matrix out of all the pixels in ``psfImg`` above ``threshold``
    that acts as the PSF operator.

    TODO: Optimize this algorithm to
    """
    height, width = imgShape
    size = width * height

    # Hide pixels in the psf below the threshold
    psf = np.copy(psfImg)
    psf[psf<threshold] = 0
    logger.info("Total psf pixels: {0}".format(np.sum(psf>0)))

    # Calculate the coordinates of the pixels in the psf image above the threshold
    indices = np.where(psf>0)
    indices = np.dstack(indices)[0]
    cy, cx = np.unravel_index(np.argmax(psf), psf.shape)
    coords = indices-np.array([cy,cx])

    # Create the PSF Operator
    offsets, slices, slicesInv = getOffsets(width, coords)
    psfDiags = [psf[y,x] for y,x in indices]
    psfOp = scipy.sparse.diags(psfDiags, offsets, shape=(size, size), dtype=np.float64)
    psfOp = psfOp.tolil()

    # Remove entries for pixels on the left or right edges
    cxRange = np.unique([cx for cy,cx in coords])
    for h in range(height):
        for y,x in coords:
            # Left edge
            if x<0 and width*(h+y)+x>=0 and h+y<=height:
                psfOp[width*h, width*(h+y)+x] = 0

                # Pixels closer to the left edge
                # than the radius of the psf
                for x_ in cxRange[cxRange<0]:
                    if (x<x_ and
                        width*h-x_>=0 and
                        width*(h+y)+x-x_>=0 and
                        h+y<=height
                    ):
                        psfOp[width*h-x_, width*(h+y)+x-x_] = 0

            # Right edge
            if x>0 and width*(h+1)-1>=0 and width*(h+y+1)+x-1>=0 and h+y<=height and width*(h+1+y)+x-1<size:
                psfOp[width*(h+1)-1, width*(h+y+1)+x-1] = 0

                for x_ in cxRange[cxRange>0]:
                    # Near right edge
                    if (x>x_ and
                        width*(h+1)-x_-1>=0 and
                        width*(h+y+1)+x-x_-1>=0 and
                        h+y<=height and
                        width*(h+1+y)+x-x_-1<size
                    ):
                        psfOp[width*(h+1)-x_-1, width*(h+y+1)+x-x_-1] = 0

    # Return the transpose, which correctly convolves the data with the PSF
    return psfOp.T.tocoo()


def nmf(Y, A0, S0, prox_A, prox_S, prox_S2=None, M2=None, lM2=None,
        max_iter=1000, W=None, P=None, e_rel=1e-3, algorithm='ADMM'):

    K = S0.shape[0]
    A = A0.copy()
    S = S0.copy()
    S_ = S0.copy() # needed for convergence test
    beta = 1. # 0.75    # TODO: unclear how to chose 0 < beta <= 1

    if W is not None:
        W_max = W.max()
    else:
        W = W_max = 1

    for it in range(max_iter):
        # A: simple gradient method; need to rebind S each time
        prox_like_A = partial(prox_likelihood_A, S=S, Y=Y, prox_g=prox_A, W=W, P=P)
        step_A = beta**it / lipschitz_const(S) / W_max
        it_A = APGM(A, prox_like_A, step_A, max_iter=max_iter)

        # S: either gradient or ADMM, depending on additional constraints
        prox_like_S = partial(prox_likelihood_S, A=A, Y=Y, prox_g=prox_S, W=W, P=P)
        step_S = beta**it / lipschitz_const(A) / W_max
        if prox_S2 is None or algorithm == "APGM":
            it_S = APGM(S_, prox_like_S, step_S, max_iter=max_iter)
        elif algorithm == "ADMM":
            # steps set to upper limit per component
            step_S2 = step_S * lM2
            it_S, S_, _, _ = ADMM(S_, prox_like_S, step_S, prox_S2, step_S2, A=M2,
                                  max_iter=max_iter, e_rel=e_rel)
        elif algorithm == "SDMM":
            # TODO: Check that we are properly setting the step size.
            # Currently I am just using the same form as ADMM, with a slightly modified
            # lM2 in nmf_deblender
            step_S2 = step_S * lM2
            it_S, S_, _, _ = SDMM(S_, prox_like_S, step_S, prox_S2, step_S2,
                                  constraints=M2, max_iter=max_iter, e_rel=e_rel)
        else:
            raise Exception("Unexpected 'algorithm' to be 'APGM', 'ADMM', or 'SDMM'")

        logger.info("{0} {1} {2} {3} {4} {5}".format(it, step_A, it_A, step_S, it_S,
                                                     [(S[i,:] > 0).sum()for i in range(S.shape[0])]))

        if it_A == 0 and it_S == 0:
            break

        # Convergence crit from Langville 2014, section 5
        if it > 10 and np.array([np.dot(S_[k],S[k]) > (1-e_rel**2)*l2sq(S[k]) for k in range(K)]).all():
            break
        if it>12:
            import IPython; IPython.embed()

        S[:,:] = S_[:,:]
    S[:,:] = S_[:,:]
    return A,S

def init_A(B, K, peaks=None, I=None):
    # init A from SED of the peak pixels
    if peaks is None:
        A = np.random.rand(B,K)
    else:
        assert I is not None
        assert len(peaks) == K
        A = np.empty((B,K))
        for k in range(K):
            px,py = peaks[k]
            A[:,k] = I[:,py,px]
    A = prox_unity_plus(A, 0)
    return A

def init_S(N, M, K, peaks=None, I=None):
    # init S with intensity of peak pixels
    if peaks is None:
        S = np.random.rand(K,N*M)
    else:
        assert I is not None
        assert len(peaks) == K
        S = np.zeros((K,N*M))
        tiny = 1e-10
        for k in range(K):
            px,py = peaks[k]
            S[k,py*M+px] = np.abs(I[:,py,px].mean()) + tiny
    return S

def adapt_PSF(P, B, shape, threshold=1e-2):
    # Simpler for likelihood gradients if P = const across B
    if isinstance(P, list) is False: # single matrix
        return getPSFOp(P, shape, threshold=threshold)

    P_ = []
    for b in range(B):
        P_.append(getPSFOp(P[b], shape, threshold=threshold))
    return P_


def get_constraints(constraint, (px, py), (N, M), useNearest=True, fillValue=1):
    """Get appropriate constraint operator
    """
    if constraint == " ":
        return scipy.sparse.identity(N*M)
    if constraint.upper() == "M":
        return getRadialMonotonicOp((N,M), px, py, useNearest=useNearest)
    if constraint.upper() == "S":
        return getPeakSymmetryOp((N,M), px, py, fillValue=fillValue)
    raise ValueError("'constraint' should be in [' ', 'M', 'S'] but received '{0}'".format(constraint))

def nmf_deblender(I, K=1, max_iter=1000, peaks=None, constraints=None, W=None, P=None, sky=None,
                  l0_thresh=None, l1_thresh=None, gradient_thresh=0, e_rel=1e-3, psf_thresh=1e-2,
                  monotonicUseNearest=False, nonSymmetricFill=1, algorithm="ADMM"):

    # vectorize image cubes
    B,N,M = I.shape
    if sky is None:
        Y = I.reshape(B,N*M)
    else:
        Y = (I-sky).reshape(B,N*M)
    if W is None:
        W_ = W
    else:
        W_ = W.reshape(B,N*M)
    if P is None:
        P_ = P
    else:
        P_ = adapt_PSF(P, B, (N,M), threshold=psf_thresh)

    # init matrices
    A = init_A(B, K, I=I, peaks=peaks)
    S = init_S(N, M, K, I=I, peaks=peaks)

    # define constraints for A and S via proximal operators
    # A: ||A_k||_2 = 1 with A_ik >= 0 for all columns k
    prox_A = prox_unity_plus

    # S: non-negativity or L0/L1 sparsity plus ...
    if l0_thresh is None and l1_thresh is None:
        prox_S = prox_plus
    else:
        # L0 has preference
        if l0_thresh is not None:
            if l1_thresh is not None:
                logger.warn("Warning: l1_thresh ignored in favor of l0_thresh")
            prox_S = partial(prox_hard, l=l0_thresh)
        else:
            prox_S = partial(prox_soft_plus, l=l1_thresh)

    # ... additional constraint for each component of S
    if constraints is not None:
        prox_constraints = {
            " ": prox_id,    # do nothing
            "M": partial(prox_min, l=gradient_thresh), # positive gradients
            "S": prox_zero   # zero deviation of mirrored pixels
        }
        M2 = []
        # Expand the constraints if the user passed an abbreviated format and
        # build the constraint operators and proximal operators
        if algorithm=="ADMM":
            if len(constraints)==1:
                constraints = constraints*K
            elif len(constraints)!=K:
                raise ValueError("'constraints' in ADMM should either be a single constraint to"
                                 "use on each peak, or a string of constraints, with an entry for each peak")
            M2 = [get_constraints(constraints[pk], peak, (N, M),
                                  monotonicUseNearest, nonSymmetricFill) for pk, peak in enumerate(peaks)]
            lM2 = np.array([np.real(scipy.sparse.linalg.eigs(np.dot(C.T,C), k=1,
                                    return_eigenvectors=False)[0]) for C in M2])
            prox_S2 = partial(prox_components, prox_list=[prox_constraints[c] for c in constraints], axis=0)

        elif algorithm=="SDMM" or algorithm=="GLM":
            if isinstance(constraints, basestring):
                constraints = [constraint*K for constraint in constraints]
            elif all([len(constraint)==K for constraint in constraints]):
                raise ValueError("'constraints' in SDMM must either be a string of constraints to apply to"
                                 "each peak or a list of constraints for each peak")
            M2 = []
            lM2 = []
            prox_S2 = []
            for constraint in constraints:
                M2.append([])
                lM2.append([])
                for pk, peak in enumerate(peaks):
                    C = get_constraints(constraint[pk], peak, (N, M), monotonicUseNearest, nonSymmetricFill)
                    M2[-1].append(C)
                    lM2[-1].append(C.T.dot(C))
                prox_S2.append(partial(prox_components, prox_list=[prox_constraints[c] for c in constraint], axis=0))
            lM2 = np.sum(lM2, axis=0)
            lM2 = np.array([np.real(scipy.sparse.linalg.eigs(l, k=1, return_eigenvectors=False)[0]) for l in lM2])
    else:
        prox_S2 = M2 = lM2 = None

    # run the NMF with those constraints
    if algorithm=="ADMM" or algorithm=="SDMM":
        A,S = nmf(Y, A, S, prox_A, prox_S, prox_S2=prox_S2, M2=M2, lM2=lM2,
                  max_iter=max_iter, W=W_, P=P_, e_rel=e_rel, algorithm=algorithm)
    elif algorithm=="GLM":
        # TODO: Improve this, the following is for testing purposes only
        A, S = GLM(data=Y, X10=A, X20=S, W=W_, P=P_,
                   prox_f1=prox_A, prox_f2=prox_S, prox_g1=None, prox_g2=prox_S2,
                   constraints1=None, constraints2=M2, lM1=1, lM2=lM2, max_iter=max_iter, e_rel=e_rel, beta=1.0)

    # reshape to have shape B,N,M
    model = np.dot(A,S)
    if P is not None:
        model = convolve_band(P_, model)
    model = model.reshape(B,N,M)
    S = S.reshape(K,N,M)

    return A,S,model,P_