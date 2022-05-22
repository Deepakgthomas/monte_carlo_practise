# Ideas taken from https://people.duke.edu/~ccc14/sta-663/MCMC.html

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.stats import poisson
from scipy.stats import norm

#real distribution
x = np.linspace(0,20,2000)
y = np.array([4, 4, 5, 8, 3])
n = len(y)

alpha = 1
beta = 1
gamma_vals = gamma.pdf(x = x, a = (n*np.mean(y)) + alpha, scale = 1/(1+(1/beta)))

plt.plot(x, gamma_vals)

#metropolis hastings starts here
def target(lambda_, alpha = alpha, beta = beta):
    if lambda_<0:
        print("woo")
        return 0
    return poisson.pmf(k = n*np.mean(y),mu = lambda_)*gamma.pdf(x = lambda_, a = alpha, scale = beta)

niters = 10000
lambda_ = 1.3
# print("Check ",poisson.pmf(k = n*np.mean(y),mu = lambda_))

samples = np.zeros(niters+1)
samples[0] = lambda_
for i in range(niters):
    lambda_p = lambda_ + norm(0, 0.3).rvs()
    rho = min(1, (target(lambda_p)/target(lambda_)))
    u = np.random.uniform()
    if u<rho:
        lambda_ = lambda_p
    samples[i+1] = lambda_
nmcmc = len(samples)//2

plt.hist(samples[nmcmc:], density = True,histtype='step')
plt.show()