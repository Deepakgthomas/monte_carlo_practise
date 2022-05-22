# Adapted from here - https://www.jarad.me/teaching/2013/10/03/rejection-sampling

from scipy.special import gamma, factorial
import numpy as np
import matplotlib.pyplot as plt

def target_dsns(x, alpha, beta):
    value = (alpha-1)*np.log(x)+(beta-1)*np.log(1-x)-np.log(gamma(alpha))-np.log(gamma(beta))+np.log(gamma(alpha+beta))
    return np.exp(value)
def proposal_dsns(b, a):
    return np.exp(np.log(1) - np.log(b-a))

def M(alpha, beta):
    value =  target_dsns(((alpha-1)/(alpha+beta-2)),alpha, beta)
    return value

alpha = 7
beta = 20
a = 0
b = 1
uniform_values = np.asarray(np.random.uniform(0,1,1000))
x_values = np.asarray(np.random.uniform(0,1,1000))
accept_vals = []
reject_vals = []
for count, x in enumerate(x_values):

    if uniform_values[count]<=target_dsns(x, alpha, beta)/(M(alpha, beta)*proposal_dsns(b, a)):
        accept_vals.append((x,M(alpha, beta)*uniform_values[count]))
    else:
        reject_vals.append((x, M(alpha, beta)*uniform_values[count]))

x1, y1 = zip(*accept_vals)
x2, y2 = zip(*reject_vals)
plotting_vals = np.linspace(0.0001, .999, 1000)
proposal_vals = []
plt.scatter(plotting_vals, target_dsns(plotting_vals, alpha, beta))
plt.scatter(plotting_vals, [proposal_dsns(b,a)]*len(plotting_vals))
plt.scatter(plotting_vals, [M(alpha, beta)*proposal_dsns(b,a)]*len(plotting_vals))
plt.scatter(x1, y1)
plt.scatter(x2, y2)
plt.show()

