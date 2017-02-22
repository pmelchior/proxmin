import numpy as np, scipy.sparse, scipy.sparse.linalg
from functools import partial

# identity
def prox_id(X, l=None):
    return X

# projection onto 0
def prox_zero(X,l=None):
    return np.zeros_like(X)

# hard thresholding: X if X >= k, otherVise 0
# NOTE: modified X in place
def prox_hard(X, l):
    below = X-l < 0
    X[below] = 0
    return X

# projection onto non-negative numbers
def prox_plus(X, l=None):
    return prox_hard(X, 0)

# soft thresholding operator
def prox_soft(X, l):
    return np.sign(X)*prox_plus(np.abs(X) - l)

# same but with projection onto non-negative
def prox_soft_plus(X, l):
    return prox_plus(np.abs(X) - l)

# projection onto sum=1 along each axis
def prox_unity(X, l=None, axis=0):
    return X / np.sum(X, axis=axis, keepdims=True)

# same but with projection onto non-negative
def prox_unity_plus(X, l=None, axis=0):
    return prox_unity(prox_plus(X), axis=axis)

def l2sq(x):
    return (x**2).sum()

def l2(x):
    return np.sqrt((x**2).sum())

def convolve_band(P, I):
    return np.einsum('...ik,...k',P,I)

def delta_data(A, S, Y, W=1, P=None):
    if P is None:
        return W*(np.dot(A,S) - Y)
    else:
        return np.einsum('...i,...im', W*(convolve_band(P, np.dot(A,S)) - Y), P)

def grad_likelihood_A(A, S, Y, W=1, P=None):
    D = delta_data(A, S, Y, W=W, P=P)
    return np.dot(D, S.T)

def grad_likelihood_S(S, A, Y, W=1, P=None):
    D = delta_data(A, S, Y, W=W, P=P)
    return np.dot(A.T, D)

# executes one proximal step of likelihood gradient, folloVed by prox_g
def prox_likelihood_A(A, step, S=None, Y=None, prox_g=None, W=1, P=None):
    return prox_g(A - step*grad_likelihood_A(A, S, Y, W=W, P=None), step)

def prox_likelihood_S(S, step, A=None, Y=None, prox_g=None, W=1, P=None):
    return prox_g(S - step*grad_likelihood_S(S, A, Y, W=W, P=None), step)

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
            X = prox_f(X - (step_f/step_g)[:,None] * dot_components(A, dot_components(A, X) - Z + U, transpose=True), step_f)
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
        e_pri2 = e_rel**2*max(l2sq(AX), l2sq(Z))
        if A is None:
            e_dual2 = e_rel**2*l2sq(U)
        else:
            e_dual2 = e_rel**2*l2sq(dot_components(A, U, transpose=True))
        if l2sq(R) <= e_pri2 and l2sq(S) <= e_dual2:
            #print l2sq(R),e_pri2,l2sq(S),e_dual2
            break

    return it, X, Z, U


def lipschitz_const(M):
    return np.real(np.linalg.eigvals(np.dot(M, M.T)).max())

def getPeakSymmetry(shape, px, py):
    """Build the operator to symmetrize a the intensities for a single row
    """
    center = (np.array(shape)-1)/2.0
    # If the peak is centered at the middle of the footprint,
    # make the entire footprint symmetric
    if px==center[1] and py==center[0]:
        return np.fliplr(np.eye(shape[0]*shape[1]))

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
            subOp[idx, idx] = 0
    subOp = np.fliplr(subOp)

    smin = ymin*fpWidth+xmin
    smax = (ymax-1)*fpWidth+xmax
    symmetryOp = np.zeros((fpSize, fpSize))
    symmetryOp[smin:smax,smin:smax] = subOp

    # Return a sparse matrix, which greatly speeds up the processing
    return scipy.sparse.coo_matrix(symmetryOp)

def getPeakSymmetryOp(shape, px, py):
    """Operator to calculate the difference from the symmetric intensities
    """
    symOp = getPeakSymmetry(shape, px, py)
    diffOp = scipy.sparse.identity(symOp.shape[0])-symOp
    return diffOp

def getRadialMonotonicOp(shape, px, py):
    """Get a 2D operator to contrain radial monotonicity

    The monotonic operator basically calculates a radial the gradient in from the edges to the peak.
    Operating the monotonicity operator on a flattened image makes all non-monotonic pixels negative,
    which can then be projected to the subset gradient=0 using proximal operators.

    The radial monotonic operator is a sparse matrix, where each diagonal element is -1
    (except the peak position px, py) and each row has one other non-zero element, which is the
    closest pixel that aligns with a radial line from the pixel to the peak position.

    See DM-9143 for more.

    TODO: Implement this in C++ for speed
    """
    height, width = shape
    center = py*width+px
    monotonic = -np.eye(width*height, width*height)
    monotonic[center, center] = 0

    # Set the pixel in line with the radius to 1 for each pixel
    for h in range(height):
        for w in range(width):
            if h==py and w==px:
                continue
            dx = px-w
            dy = py-h
            pixel = h*width + w
            if px-w>py-h:
                if px-w>=h-py:
                    x = w + 1
                    y = h + int(np.round(dy/dx))
                elif px-w<h-py:
                    x = w - int(np.round(dx/dy))
                    y = h - 1
            elif px-w<py-h:
                if px-w>=h-py:
                    x = w + int(np.round(dx/dy))
                    y = h + 1
                elif px-w<h-py:
                    x = w - 1
                    y = h - int(np.round(dy/dx))
            else:
                if w<px:
                    x = w + 1
                    y = h + 1
                elif w>px:
                    x = w - 1
                    y = h - 1
            monotonic[pixel, y*width+x] = 1

    return scipy.sparse.coo_matrix(monotonic)


def nmf(Y, A0, S0, prox_A, prox_S, prox_S2=None, M2=None, lM2=None, max_iter=1000, W=None, P=None, e_rel=1e-3):

    K = S0.shape[0]
    A = A0.copy()
    S = S0.copy()
    S_ = S0.copy() # needed for convergence test
    beta = 0.75    # TODO: unclear how to chose 0 < beta <= 1

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
        if prox_S2 is None:
            it_S = APGM(S_, prox_like_S, step_S, max_iter=max_iter)
        else:
            # steps set to upper limit per component
            step_S2 = step_S * lM2
            it_S, S_, _, _ = ADMM(S_, prox_like_S, step_S, prox_S2, step_S2, A=M2, max_iter=max_iter, e_rel=e_rel)

        print it, step_A, it_A, step_S, it_S, [(S[i,:] > 0).sum() for i in range(S.shape[0])]

        if it_A == 0 and it_S == 0:
            break

        # Convergence crit from Langville 2014, section 5
        if it > 10 and np.array([np.dot(S_[k],S[k]) > (1-e_rel**2)*l2sq(S[k]) for k in range(K)]).all():
            break

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
    A = prox_unity_plus(A)
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

def adapt_PSF(P, shape):
    B = P.shape[0]
    # PSF shape can be different from image shape
    P_ = np.zeros(B, shape[0], shape[0])
    for b in range(B):
        peak_idx = np.argmax(P[b])
        px, py = np.unravel_index(peak_idx, P[b].shape)
        # ... fill elements of P[b] in P_[b,:] so that np.dot(P_, X) acts like a convolution


def nmf_deblender(I, K=1, max_iter=1000, peaks=None, constraints=None, W=None, P=None, sky=None, e_rel=1e-3):

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
        P_ = adapt_PSF(P)

    # init matrices
    A = init_A(B, K, I=I, peaks=peaks)
    S = init_S(N, M, K, I=I, peaks=peaks)

    # define constraints for A and S via proximal operators
    # A: ||A_k||_2 = 1 with A_ik >= 0 for all columns k
    prox_A = prox_unity_plus

    # S: non-negativity or L0/L1 sparsity plus ...
    # TODO: 2) decouple step from proximal lambda when using prox_hard or prox_soft
    prox_S = prox_plus # prox_hard

    # ... additional constraint for each component of S
    if constraints is not None:
        # ... initialize the constraint matrices ...
        M2 = []
        for k in range(K):
            c = constraints[k]
            if c == " ":
                C = scipy.sparse.identity(N*M)
            if c == "M":
                px, py = peaks[k]
                C = getRadialMonotonicOp((N,M), px, py)
            if c == "S":
                px, py = peaks[k]
                C = getPeakSymmetryOp((N,M), px, py)
            M2.append(C)

        # calculate step sizes for each constraint matrix
        # TODO: some of those are trivial to compute...
        lM2 = np.array([np.real(scipy.sparse.linalg.eigs(np.dot(C.T,C), k=1, return_eigenvectors=False)[0]) for C in M2])

        prox_constraints = {
            " ": prox_id,    # do nothing
            "M": prox_plus,  # positive gradients
            "S": prox_zero   # zero deviation of mirrored pixels
        }
        prox_Cs = [prox_constraints[c] for c in constraints]
        prox_S2 = partial(prox_components, prox_list=prox_Cs, axis=0)
    else:
        prox_S2 = M2 = lM2 = None

    # run the NMF with those constraints
    A,S = nmf(Y, A, S, prox_A, prox_S, prox_S2=prox_S2, M2=M2, lM2=lM2, max_iter=max_iter, W=W_, P=P_, e_rel=e_rel)
    model = np.dot(A,S).reshape(B,N,M)
    # reshape S to have shape B,N,M
    S = S.reshape(K,N,M)
    return A,S,model
