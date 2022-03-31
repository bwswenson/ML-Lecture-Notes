# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 21:03:14 2020

@author: swens
"""

from sklearn import datasets
faces = datasets.fetch_olivetti_faces()

from matplotlib import pyplot as plt
fig = plt.figure(1,figsize=(8, 6))

# plot several images
for i in range(15):
    ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(faces.images[i], cmap=plt.cm.bone)

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(faces.data,
#        faces.target, random_state=0)
#
#print(X_train.shape, X_test.shape)

from sklearn import decomposition
pca = decomposition.PCA(n_components=150, whiten=True)
pca.fit(faces.data)

fig = plt.figure(2,figsize=(16, 6))
for i in range(30):
    ax = fig.add_subplot(3, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(pca.components_[i].reshape(faces.images[0].shape),
              cmap=plt.cm.bone)
    
