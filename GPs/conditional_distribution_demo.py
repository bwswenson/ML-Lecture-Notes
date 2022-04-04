import numpy as np
import matplotlib.pyplot as plt

'''
In this demo, we see the conditional distribution of a jointly Gaussian RV, where
the conditioning is of the first coordinate conditioned on the second coordinate. 

Consider a 2-d Gaussian with mean mu and covariance Sigma = (a, b; c, d)
Let mu_tilde and sigma_tilde be the mean of x1 conditoined on x2. We have

mu_tilde = mu_1 + b/d(x2 - mu_2)
sigma_tilde = a - bc/d

See Bishop book p.87 for derivation. 
'''

# First, look at a scatter plot of our joint Gaussian
N = 1000   # Number of samples

# these need to be chosen to form a positive definite matrix
a = 1
b = -(np.sqrt(100)-1)  # should be less than sqrt(a*d)
c = b
d = 100


mu = [0,0]
Sigma = [[a, b], [c, d]]
X = np.random.multivariate_normal(mean = mu, cov = Sigma, size = N)
fig, ax = plt.subplots(num=1)
fig.clf()
fig, ax = plt.subplots(num=1)
ax.scatter(X[:,0], X[:,1])
x_max = 3*np.sqrt(max((a,d)))
y_max = 3*np.sqrt(max((a,d)))
ax.set(xlim=(-x_max, x_max), ylim=(-y_max, y_max))
plt.show()

# Now, draw a single sample from the joint Gaussian, and look at the
# Distribution of x1 conditioned on x2

x = np.random.multivariate_normal(mean = mu, cov = Sigma, size = 1).squeeze()
mu_tilde = mu[0] + b/d*(x[1] - mu[1])
sigma_tilde = a - b*c/d

print(f'Value of x2 = {x[1]}')

# plot the conditional distribution of x1|x2
z = np.linspace(-3,3,1000)
y = 1/(2*np.pi*sigma_tilde)**.5*np.exp(-1/2*(z - mu_tilde)**2/sigma_tilde)
fig, ax = plt.subplots(num=2)
fig.clear()
fig, ax = plt.subplots(num=2)
plt.plot(z,y)
ax.set(title = f'conditional distribution of x1\n  \
               x2 = {x[1]:.2f}\n \
               mu_tilde = {mu_tilde:.2f}\n \
               sigma_tilde = {sigma_tilde:2f}')
