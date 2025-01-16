"""
This code gives an example of gaussian process regression in two dimensions. Points are randomly sampled.
GP is fit using standard scikit learn fit (to choose length scale, noise parameter, and actually do the matrix inversion
for computing posterior mean and variance).

Plots the scatter points, the mean function, the actual function, and the estimator variance.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from matplotlib import cm

# TODO: Use a white noise kernel to automatically learn noise parameter?
# Set up latin hypercube sampling

N = 300
# Generate sample data
# np.random.seed(1)
X = np.random.rand(2, N) * 10 - 5  #TODO: Change this to latin hypercube sampling
f = lambda x: np.sin(x[0]) + np.cos(x[1])
y = f(X) + np.random.randn(N) * 0.4

# Create the Gaussian Process Regressor
kernel = RBF()
gpr = GaussianProcessRegressor()

# Fit the model
gpr.fit(X.T, y)
print(f'alpha={gpr.alpha}')

# Scatter plot points
fig, ax = plt.subplots(num=1, subplot_kw={"projection": "3d"})
ax.scatter(X[0], X[1], y, color='red')

# Plot the GP mean surface + scatter points
x1, x2 = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
X_grid = np.c_[x1.ravel(), x2.ravel()]
fig, ax = plt.subplots(num=2, subplot_kw={"projection": "3d"})
y_pred, y_std = gpr.predict(X_grid, return_std=True)
ax.plot_surface(x1, x2, y_pred.reshape(50, 50),
                       linewidth=0, antialiased=False)

# plot GP variance
fig, ax = plt.subplots(num=3)
plt.imshow(y_std.reshape(50,50))
plt.colorbar(label='variance')

# plot original function
fig, ax = plt.subplots(num=4, subplot_kw={"projection": "3d"})

ax.plot_surface(x1, x2, f(X_grid.T).reshape(50,50))

# plot error function
#TODO

# plt.show()
