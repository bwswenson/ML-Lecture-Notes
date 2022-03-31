from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=4)
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from numpy import diag
from numpy import dot
from numpy import zeros

# Generate some random data
N = 100
offset = 0
x = np.random.uniform(0,1,N)
y = x + .1*np.random.uniform(-1,1,N) + offset

################################
# plot data and run built in PCA
fig = plt.figure(1)
fig.clf()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.scatter(x,y)
ax1.plot(x,x)

Z = np.vstack((x,y))

pca = PCA(n_components=1)
pca.fit(Z.T)
Z_temp = pca.transform(Z.T)
Z_new = (pca.inverse_transform(Z_temp)).T
ax2.scatter(Z_new[0,:],Z_new[1,:])

#############################
# Next: Compute PCA decomposition via SVD
U, S, V_transpose = np.linalg.svd(Z)
V = V_transpose.T
print(f"Dimensions -- U: {U.shape}, V: {V.shape}, S: {S.shape}")

# Confirm that we have right SVD
S_full = np.zeros((2,100))
S_full[(0,0)] = S[0]
S_full[(1,1)] = S[1]
Z_hat = U @ S_full @ V.T
print(f'SVD matches = {(Z_hat - Z < 1e-7).all()}')


# Extract vectors for low rank approximation
# Note: Need to correct for np's stupid dimension alterations
u0 = U[:,0]; u0 = u0[:,np.newaxis]
u1 = U[:,1]; u1 = u1[:,np.newaxis]
v0 = V[:,0]; v0 = v0[:,np.newaxis]
v1 = V[:,1]; v1 = v1[:,np.newaxis]

Z_full = S[0]*u0 @ v0.T + S[1]*u1 @ v1.T
print(f'Full rank approximation matches = {(Z_full - Z < 1e-7).all()}')

# construct and plot low rank approximation
Z_low_rank = S[0]*u0 @ v0.T

fig = plt.figure(2)
fig.clf()
ax1 = fig.add_subplot(111)
ax1.scatter(x,y)
ax1.scatter(Z_low_rank[0], Z_low_rank[1], color = 'red')

############################
# Use the "PCA transform" to send a single point to its low dim representation
# and then back again
W = np.random.normal(size = (2,20))  # some random points
U0 = u0         # normally, you would stack more vectors in here
W_reconstructed = U0 @ U0.T @ W
fig = plt.figure(3)
fig.clf()
ax1 = ax1 = fig.add_subplot(111)
ax1.scatter(W[0], W[1])
ax1.scatter(W_reconstructed[0], W_reconstructed[1], color = 'red')

#########################
# A note of caution about centering

# Add offset to data
offset = 1
y = y + offset
Z = np.vstack((x,y))

# Compute PCA via SVD
U, S, V_transpose = np.linalg.svd(Z)
V = V_transpose.T
u0 = U[:,0]; u0 = u0[:,np.newaxis]
v0 = V[:,0]; v0 = v0[:,np.newaxis]
U_proj = u0 @ u0.T 
Z_low = U_proj @ Z

# plots
fig = plt.figure(4)
fig.clf()
ax1 = fig.add_subplot(111)
ax1.scatter(Z[0], Z[1])
ax1.plot(Z_low[0], Z_low[1], color = 'red')

# repeat, but "center" data first
mu = np.mean(Z,1)
mu = mu[:,np.newaxis]
Z = Z - mu

# Compute PCA via SVD
U, S, V_transpose = np.linalg.svd(Z)
V = V_transpose.T
u0 = U[:,0]; u0 = u0[:,np.newaxis]
U_proj = u0 @ u0.T 
Z_low = U_proj @ Z

# plots
fig = plt.figure(5)
fig.clf()
ax1 = fig.add_subplot(111)
ax1.scatter(Z[0], Z[1])
ax1.plot(Z_low[0], Z_low[1], color = 'red')

############################
# PCA representation gives "uncorrelated" variables

# Construct remaining vector for full rank reconstruction (we're in dim 2, so have to have full rank
# reconstruction for this part of example to be meaningful.)
u1 = U[:,1]; u1 = u1[:,np.newaxis]

U2 = np.array([u0, u1]).squeeze() # rank 2 reconstrution matrix

# The "PCA reprentation" (normally reduced dimension, but technically doesn't have to be)
Z_transformed = U2.T @ Z

# compute covariance matrix (not sure if we need to recenter since we already did...)
mu = np.mean(Z_transformed,1)
mu = mu[:,np.newaxis]
Z_new = Z_transformed - mu
Z_cov = Z_new @ Z_new.T
print(f'Covariance Matrix: {Z_cov}')












#print(z_proj[0])

#data = np.load('subset.npy')
#data = data.T

## we need to scale the data first
#scaler = StandardScaler()
#scaler.fit(data)
#data_c = scaler.transform(data)
#
## we then conduct PCA
#pca = PCA(n_components=3)
#pca.fit(data_c)
#print('The proportion of total variance by the principle components are:', pca.explained_variance_ratio_)
#components = pca.components_ # n_components x n_features
#x_proj = pca.fit_transform(data_c) # n_samples x n_components
#
## Hard way to display images
#fig, (ax1,ax2,ax3) = plt.subplots(3,1)
#ax1.imshow(x_proj[:,0].reshape(200,200))
#ax2.imshow(x_proj[:,1].reshape(200,200))
#ax3.imshow(x_proj[:,2].reshape(200,200))
#
#im3 = x_proj[:,2].reshape(200,200)
#
## Easy way to display images
