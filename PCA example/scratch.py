# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 07:06:28 2020

@author: swens
"""

import numpy as np

A=['1','2','3']

for a in A:

  print(2*a)

#data = np.load('covid-19.npy')
#Ts, ts = data[:,0], data[:,1]
#taus = Ts - ts
#m = np.mean(taus)
#v = np.mean(taus**2)
#
#def moments_to_parameters(m, v):
#    hat_mu = np.exp(2*np.log(m) - 1/2*np.log(v))
#    hat_sigma = np.sqrt(np.log(v) - 2*np.log(m))
#    return hat_mu, hat_sigma
#
#m_hat, v_hat = moments_to_parameters(m, v)
#
#
#print(f"Estimated parameter values: {moments_to_parameters(m, v)}")
#print(f"squared sigma: {v_hat**2}")
#
#
#def Omega_tau(tau, Lambda1=10):
#    t0 = 0
#    return np.maximum(t0, Lambda1 - tau) / Lambda1
#
#weights = 1/Omega_tau(taus)
#
#m_ = np.sum(taus*weights) / np.sum(weights)
#v_ = np.sum(taus**2*weights) / np.sum(weights)
#
#print(f"Compensated estimates: {moments_to_parameters(m_, v_)}")