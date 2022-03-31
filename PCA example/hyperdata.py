# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import AffinityPropagation
from matplotlib import cm
import time

hyperdata = np.load('subset.npy')
start = time.time()
N = 5
cutdata = hyperdata[:,:N*200]
af = AffinityPropagation(damping=0.75).fit(cutdata.T)
centers = af.cluster_centers_indices_
labels = af.labels_
centers = af.cluster_centers_

label_counts = Counter(labels)
most_frequent_labels = sorted(label_counts.keys(),key=lambda item: -label_counts[item])

#plt.scatter(cutdata[0,:],cutdata[1,:],c=labels)
joint = np.zeros((N,200,4))
for i in range(len(labels)):
    joint[int(i/200),int(i%200),:] = cm.hsv(labels[i]/len(label_counts))
print('{} total clusters'.format(len(label_counts)))
plt.imshow(joint)
print('Finished after {:.2f} seconds'.format(time.time()-start))