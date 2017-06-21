from proxmin import nmf
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt

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
	arrangment = linear_sum_assignment(-corr)[1]
	resS = np.array([S[arrangment[i]] for i in range(k)])
	resA = np.array([[A[i][arrangment[j]] for j in range(A.shape[1])] for i in range(A.shape[0])])
	return resA, resS

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

	noises = np.linspace(0,0.2,20)
	A_chi_squared = []
	S_chi_squared = []
	for n in noises:
		Y = add_noise(trueY, n)
		A, S = nmf.nmf(Y, A0, S0)
		A, S = match(A, S, trueS)
		A_chi_squared.append(np.sum((A - trueA)**2))
		S_chi_squared.append(np.sum((S - trueS)**2))
	""" For individual test cases
	noise = 0.02		# stdev of added noise 
	Y = add_noise(trueY, noise)
	A, S = nmf.nmf(Y, A0, S0)
	A, S = match(A, S, trueS)
	fig = plt.figure(figsize=(6,8))
	ax = fig.add_subplot(311)
	ax.set_title("True Components (S)")
	ax.plot(trueS.T)
	ax2 = fig.add_subplot(312)
	ax2.set_title("Data (Y)")
	ax2.plot(Y.T)
	ax3 = fig.add_subplot(313)
	ax3.set_title("Found Components (S)")
	ax3.plot(S.T)
	fig.show()
	"""
	plt.plot(noises, S_chi_squared, label="S Chi-squared")
	plt.plot(noises, A_chi_squared, label="A Chi-squared")
	plt.title("S and A robustness to noise")
	plt.legend()
	plt.ylabel("Chi-squared")
	plt.xlabel("Standard deviation of noise")
	plt.show()