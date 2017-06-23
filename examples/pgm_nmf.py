from proxmin import nmf
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(101)

def generateComponent(m, sigma=1):
	"""Creates components to be mixed, option to smooth with gaussian kernel"""
	res = np.array([np.random.random() for i in range(m)])		
	if sigma > 0:
		original = res
		res = np.zeros_like(res)
		for i in range(-2*sigma, 2*sigma):
			res = res + np.roll(original, i)*gaussian(i,0,sigma)
	return res

def gaussian(x, mu, sigma):
	"""Gaussian with mean mu, stdev sigma evaluated at x"""
	return 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-((x - mu)/sigma)**2./2)

def generateAmplitudes(k):
	"""Makes mixing coefficients"""
	res = np.array([np.random.random() for i in range(k)])
	return res/res.sum()

def add_noise(Y, sigma):
	"""Adds noise to Y"""
	return Y + np.random.normal(0, sigma, Y.shape)

def match(A, S, trueS):
	"""Rearranges columns of S to best fit the components they likely represent (maximizes sum of correlations)"""
	cov = np.cov(trueS, S)
	k = S.shape[0]
	corr = np.zeros([k,k])
	for i in range(k):
		for j in range(k):
			corr[i][j] = cov[i + k][j]/np.sqrt(cov[i + k][i + k]*cov[j][j])
	arrangement = linear_sum_assignment(-corr)
	resS = np.zeros_like(S)
	resA = np.zeros_like(A)
	for t in range(k):
		resS[arrangement[1][t]] = S[arrangement[0][t]]
		for i in range(A.shape[0]):
			resA[i][arrangement[1][t]] = resA[i][arrangement[0][t]]
	return resA, resS

def l2sq(X):
	length = len(X)
	res = 0
	for i in range(length):
		res += ((X[i]**2).sum())
	return res

def subtract(A, B):
	length = len(A)
	res = []
	for i in range(length):
		res.append(A[i] - B[i])
	return res

def multiply(A, B):
	length = len(A)
	res = []
	for i in range(length):

		res.append(A[i]*B[i])
	return res

def multiply_scalar(A, B):
	length = len(A)
	res = []
	for i in range(length):

		res.append(A[i]*B)
	return res

def add(A, B):
	length = len(A)
	res = []
	for i in range(length):
		res.append(A[i] + B[i])
	return res

def pgm(X0, prox_f, relax=1.49, e_rel=1e-4, max_iter=1000, traceback=False):
	"""Proximal Gradient Method

	Adapted from Combettes 2009, Algorithm 3.4
	"""
	X = X0.copy()
	Z = X0.copy()

	history = []
	for it in range(max_iter):

		_X = prox_f(Z)
		Z = add(X, multiply_scalar(subtract(_X, X), relax))
		if traceback:
			history.append([X[0], X[1]])
		if l2sq(subtract(X,_X)) <= e_rel**2*l2sq(X):
			X = _X
			break
		X = _X
	if not traceback:
		return X
	return X, history

def prox_grad(AS):
	return prox(grad(AS))

def prox(AS):
	return (AS + np.abs(AS))/2

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

def grad(AS): 
	A = AS[0]
	S = AS[1]
	L_A = get_spectral_norm(A)
	L_S = get_spectral_norm(S.T)
	grad = np.array([(A @ S - Y) @ S.T , A.T @ (A @ S - Y)])
	return subtract(AS, [grad[0]/L_S, grad[1]/L_A])

def nmf(Y, A0, S0, max_iter=1000, traceback=False):
	return pgm([A0, S0], prox_grad, max_iter=max_iter, relax=1, traceback=traceback)

if __name__ == "__main__":
	m = 50 			# component resolution
	k = 3 			# number of components
	n = 20			# number of observations
	smoothS = 3 	# stdev of gaussian kernel to smoothen S, set to 0 for no effect
	trueA = np.array([generateAmplitudes(k) for i in range(n)])
	trueS = np.array([generateComponent(m, smoothS) for i in range(k)])
	trueY = trueA @ trueS
	A0 = np.array([generateAmplitudes(k) for i in range(n)])
	S0 = np.array([generateComponent(m, smoothS) for i in range(k)])

	# Tests
	testcase = True
	convergence_plot = False
	noiseplot = False
	runtime = False
	if testcase:
		noise = 0.02		# stdev of added noise 
		Y = add_noise(trueY, noise)
		A, S = nmf(Y, A0, S0)
		A, S = match(A, S, trueS)
		fig = plt.figure(figsize=(6,6))
		ax = fig.add_subplot(311)
		ax.set_title("True Components (S) (PGM)")
		ax.plot(trueS.T)
		ax2 = fig.add_subplot(312)
		ax2.set_title("Data (Y)")
		ax2.plot(Y.T)
		ax3 = fig.add_subplot(313)
		ax3.set_title("Found Components (S)")
		ax3.plot(S.T)
		fig.show()
	if convergence_plot:
		res, history = nmf(Y, A0, S0, traceback=True)
		convergences = []
		for h in history:
			Y = h[0] @ h[1]
			convergences.append(((Y - trueY)**2).sum())
		fig2 = plt.figure(figsize=(6,6))
		ax4 = fig2.add_subplot(111)
		ax4.set_title("Convergence (PGM)")
		ax4.plot(convergences)
		ax4.set_ylabel("||Y-AS||")
		ax4.set_xlabel("Iterations")
		ax4.set_xlim([10,1000])
		ax4.set_ylim([0,0.5])
		fig2.show()
	if noiseplot:
		noises = np.linspace(0,0.2,20)
		A_chi_squared = []
		S_chi_squared = []
		for e in noises:
			Y = add_noise(trueY, e)
			A, S = nmf(Y, A0, S0)
			A, S = match(A, S, trueS)
			A_chi_squared.append(np.sum((A - trueA)**2))
			S_chi_squared.append(np.sum((S - trueS)**2))
		end = time.time()
		fig3 = plt.figure(figsize=(6,6))
		ax5 = fig3.add_subplot(111)
		ax5.plot(noises, S_chi_squared, label="S Chi-squared")
		ax5.plot(noises, A_chi_squared, label="A Chi-squared")
		ax5.set_title("S and A robustness to noise (PGM)")
		ax5.legend()
		ax5.set_ylabel("Chi-squared")
		ax5.set_xlabel("Standard deviation of noise")
		fig3.show()
	if runtime:
		noise = 0.02
		trials = 20
		A0s = []
		S0s = []
		Ys = []
		for t in range(trials):
			trueA = np.array([generateAmplitudes(k) for i in range(n)])
			trueS = np.array([generateComponent(m, smoothS) for i in range(k)])
			trueY = trueA @ trueS
			A0s.append(np.array([generateAmplitudes(k) for i in range(n)]))
			S0s.append(np.array([generateComponent(m, smoothS) for i in range(k)]))
			Ys.append(add_noise(trueY, noise))
		start = time.time()
		for t in range(trials):
			A, S = nmf(Ys[t], A0s[t], S0s[t], max_iter=1000)
		end = time.time()
		print("Runtime: {0}".format(end - start))
	plt.show()