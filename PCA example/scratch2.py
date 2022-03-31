# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:27:28 2020

@author: swens
"""

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pickle

fig = plt.figure(6)
fig.clear()
ax = fig.gca(projection='3d')

X, Y = np.meshgrid(mesh_vec2, mesh_vec1)

# Plot the surface.
surf = ax.plot_surface(X_new, Y_new, stable_man_new, linewidth=0, antialiased=False)
plt.show()
ax.set_xlabel('x3')
ax.set_ylabel('x1')
ax.set_zlabel('x2')


fig = plt.figure(6)
ax = fig.gca(projection='3d')
ax.set_xlabel('x3')
ax.set_ylabel('x2')
ax.set_zlabel('x1')

#fig = plt.figure(2)
#fig.clear()
#ax = fig.gca(projection='3d')
#X= np.concatenate((X1,X2))
#Y= np.concatenate((Y1,Y2))
#stable_man = np.concatenate((SM1,SM2))
#surf = ax.plot_surface(X, Y, stable_man, linewidth=0, antialiased=False)
#plt.show()


## Save plot parameters if I like it

my_data = (X,Y,stable_man)