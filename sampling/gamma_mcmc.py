##### The real distribution ####
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
data = np.array([4, 4, 5, 8, 3])
n = len(data)
#Gamma(shape = 25, rate = 6)
lik = st.gamma.pdf(a = 25, scale = 1/6, x = np.linspace(0, 10, num=100))
plt.plot(lik)
plt.show()

def target(lambda_val, log_val = True):
    if log_val == False:
        return ((lambda_val)**(n*np.mean(data)+a+1))*np.exp(-lambda_val*(b+n))
    else:
        return np.exp((n*np.mean(data)+a+1)*np.log(lambda_val) - lambda_val*(n+b))

### Using MCMC to sample from it #####
niters = 1000
theta = 40

a = 1
b = 1
samples = np.zeros(niters+1)
samples[0] = 5
naccept = 0
for i in range(niters):
    theta_p = theta + st.norm(0,3).rvs()
    rho= min(1, target(theta_p)/target(theta))
    u = np.random.uniform()
    if u<rho:
        naccept += 1
        theta = theta_p
    samples[i+1] = theta
nmcmc = len(samples)//2
print("Efficiency = ", naccept/niters)

plt.hist(samples[nmcmc:], 40)
plt.show()

# print(samples.shape)