"""
MLE vs MAP demo

Scenario 1: You see N trials of a Bermoulli RV (coin tosses). If you consider 
probability distributions of the form product of Bernoulli's with param theta,
what is the MLE of theta?

Secnario 2: Same setup as above, but you assume a Beta prior with params a and
b. What is the MAP of theta?
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

N = 100    # N tosses
T = 10       # N trials
p = .6       # probability of heads
a = 1    # beta distribution params for MAP prior
b = 1 

X = np.random.randint(2, size = N)
X = (np.random.random(size = (N,T)) < p).astype(int)

# Compute MLE
theta_MLE = np.zeros_like(X, dtype = float)
theta_MAP = np.zeros_like(X, dtype = float)
for i in range(0, N):

    n_heads = np.sum(X[0:i+1,:], axis=0)
    n_total = (i+1)
    theta_MLE[i,:] = n_heads/n_total
    theta_MAP[i,:] = (n_heads + a - 1)/(n_total + a + b -2)
    
# compute Beta distribution for plot
t = np.linspace(0,1,1000)
Beta = t**(a-1)*(1-t)**(b-1)/(gamma(a)*gamma(b)/gamma(a+b))

fig, ax = plt.subplots(num=3)
ax.plot(t, Beta)
ax.set(title='Beta Distribution')

    
fig, ax = plt.subplots(num=1)
ax.plot(range(N), theta_MLE)
ax.set(xlabel='num tosses', ylabel='MLE', title='MLE')
ax.grid()

fig, ax = plt.subplots(num=2)
ax.plot(range(N), theta_MAP)
ax.set(xlabel='num tosses', ylabel='MAP', title='MAP')

ax.grid()

plt.show()