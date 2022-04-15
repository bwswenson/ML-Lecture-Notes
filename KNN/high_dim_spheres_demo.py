"""
Demo:  
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