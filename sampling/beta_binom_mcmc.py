#Ideas taken from https://people.duke.edu/~ccc14/sta-663/MCMC.html

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.stats import norm
from scipy.stats import beta
from scipy.stats import binom
# plot true posterior
a = 10
b = 10
n = 100
h = 61
thetas = np.linspace(0,1, 200)
betaVals = beta.pdf(thetas, h+a, b+n-h)
plt.plot(thetas, betaVals, label = "Real Distribution")

# mcmc begins here
def target(theta,a,b,h,n):
    if theta<0 or theta>1:
        return 0
    return beta.pdf(theta,a,b)*binom.pmf(k = h, n = n, p = theta)
theta = 0.1

niters = 1000
samples = np.zeros(niters+1)
samples[0] = theta
for i in range(niters):
    theta_p = theta + norm(0, 0.3).rvs()
    try:
        rho = min(1, (target(theta_p,a,b,h,n)/target(theta,a,b,h,n)))
    except ZeroDivisionError:
        print("i = ", i, " Whoops", " theta_p = ", theta_p, " theta = ", theta)
    u = np.random.uniform()
    if u< rho:
        theta = theta_p
    samples[i+1] = theta
nmcmc = len(samples)//2
plt.hist(samples[nmcmc:], density = True, histtype='step', label = "Approximate Distribution")
plt.legend()
plt.show()

