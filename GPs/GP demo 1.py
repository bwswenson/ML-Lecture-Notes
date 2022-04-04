# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 07:29:59 2021

@author: swens
"""

import numpy as np
import itertools
from scipy.linalg import sqrtm
from matplotlib import pyplot as plt

dim = 100
n_samples = 2

Sigma_squared = np.zeros((dim,dim))
for i,j in itertools.product(range(dim), range(dim)):
    Sigma_squared[i,j] = i * j # np.min((i,j)) #1 - np.abs(i-j)*.01
    # Sigma[i,j] = i*j
Sigma = sqrtm(Sigma_squared)

x = np.random.randn(dim,n_samples)
y = Sigma @ x

print(f'Cov = {Sigma @ Sigma.T}')
print(f'sample covariance: {1/(n_samples - 1)*y @ y.T}')

plt.figure(1).clf()
fig, ax = plt.subplots(1,1, num=1)


for i in range(n_samples):
    ax.plot(y[:,i], '-')


