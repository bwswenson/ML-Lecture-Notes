"""
Demo: This shows how points sampled uniformly from a high dimensional sphere 
tend to be far aprt. To make it easier to draw samples, points are drawn from
the L\infty ball and distances are measured in terms of L\infty norm. 
"""

import numpy as np
import matplotlib.pyplot as plt
d = 1000 # dimension of space
N = 100 # num samples

x = np.random.uniform(size=(N,d))

dist_list = []
for i in range(N):
    for j in range(i+1, N):
        dist_list.append(np.linalg.norm(x[i] - x[j], ord=np.inf))

plt.hist(dist_list, bins='auto')
plt.show()