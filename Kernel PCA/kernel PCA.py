import numpy as np
from numpy import pi, cos, sin, exp
import matplotlib.pyplot as plt
import itertools
from scipy.linalg import eig
from itertools import product

# params
ker_type = 'rbf'
poly_deg = 2
poly_c = 10
gamma = 1/1  # rbf kernel inverse bandwidth

# generate sphere data
m = 100  # num points
m_half = m  # save this m. We're going to double (or triple) the dataset size.

eta = .0  # width of annulus
eps = .0  # size of inner ball
r = 1.0 + np.random.rand(m)*eta
theta = np.random.rand(m)*2*pi
x = np.array([r*cos(theta), r*sin(theta)]).T
# z = np.random.rand(m,2)*eps - eps/2
# r = 2 + np.random.rand(m)*eps
# theta = np.random.rand(m)*2*pi
# z = np.array([r*cos(theta), r*sin(theta)]).T
# x = np.vstack((x,z))
#
# # add an outer ring
# r = 3 + np.random.rand(m)*eps
# theta = np.random.rand(m)*2*pi
# z = np.array([r*cos(theta), r*sin(theta)]).T
# x = np.vstack((x,z))
# m = 3*m  # make this 2m if outer ring is missing


# params controlling manifold visualization
manifold_viz_on = True
w = 3  # grid width to use for constructing data manifold
n_grid_pts = 20  # careful, the number of points goes up as the square of this.


# # Generate some v shaped data
# a = 1
# eps = .05
# x1 = np.random.rand(m_half)
# y1 = a*x1
# z1 = np.vstack((x1, y1))
# zshape = z1.shape
# z1 += np.random.rand(*zshape)*eps
# x2 = np.random.rand(m_half)*-1
# y2 = -a*x2
# z2 = np.vstack((x2, y2))
# z2 += np.random.rand(*zshape)*eps
# x = np.hstack((z1, z2)).T
# z = np.random.rand(m,2)*np.array([[1, .01]])

# # construct quadratic data
# eps = .1
# x = np.random.rand(m)*2-1
# y = x**2 + np.random.rand(m)*eps
# x = np.vstack((x, y)).T

# plot points in regular space
n_labels = int(m*1)  # percentage of labels to keep
true_c = ['r' for i in range(m_half)] #+ ['b' for i in range(m_half)] +  ['g' for i in range(m_half)] # set colors for each half of plot
c = ['k' for _ in range(m)]
for i in np.random.choice(range(m), size=n_labels, replace=False):
    c[i] = true_c[i]
fig, ax = plt.subplots(num=1)
ax.scatter(x[:,0], x[:,1], color=c)

###### kernelize data ######

# pick kernel
ker = None
if ker_type == 'rbf':
    def ker(x, z):
        return exp( -gamma*sum((x-z)**2) )
elif ker_type == 'poly':
    d = poly_deg
    def ker(x, z):
        return (np.dot(x, z) + poly_c)**d


# construct gram matrix
K = np.zeros(shape=(m,m))
for i, j in itertools.product(range(m), repeat=2):
    K[i,j] = ker(x[i], x[j])

# center the kernel matrix
K = K - 1/m*np.ones(shape=(m,m)) @ K - 1/m*K@np.ones(shape=(m,m)) + (1/m**2)*np.ones(shape=(m,m)) @ K @ np.ones(shape=(m,m))
lam, vecs = eig(K)
a = np.real(vecs.T)  # this puts eigvecs in rows. But makes indexing easier.
a2 = a[:2]  # pick first two principal components
a3 = a[:3]  # first three principal components

# compute projections
y2d = a2 @ K
y3d = np.real(a3 @ K)
fig, ax = plt.subplots(num=2)
ax.scatter(y2d[0], y2d[1], color=c)

fig = plt.figure(num=3)
ax = fig.add_subplot(projection='3d')
ax.scatter(y3d[0], y3d[1], y3d[2], color=c)

# visualize the manifold
if manifold_viz_on:
    # create data on a grid
    pts = []
    for s, t in product(np.linspace(-w,w,n_grid_pts).tolist(), repeat=2):
        new_x = np.array([t, s])
        k = []
        for xv in x:
            k.append(ker(new_x, xv))
        p = a3 @ np.array(k)
        pts.append(p)
    pts = np.array(pts).T

    # try computing offset
    k=[]
    for xv in x:
        k.append(ker(x[0], xv))
    offset = a3 @ K[0] - a3 @ np.array(k)
    pts = (pts.T + offset).T

    # fig = plt.figure(num=4)
    # ax = fig.add_subplot(projection='3d')
    ax.scatter(pts[0], pts[1], pts[2])
plt.show()
