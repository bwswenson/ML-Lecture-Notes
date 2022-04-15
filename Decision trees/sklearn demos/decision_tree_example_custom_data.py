"""
Info here.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from mlxtend.plotting import plot_decision_regions
# import matplotlib.gridspec as gridspec

from manual_data_input import custom_data

# load data
custom_dataset = custom_data()
X, y = custom_dataset['data'], custom_dataset['targets']

# fit decision tree
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X, y)
tree.plot_tree(clf)

# fit RF
clf2 = RandomForestClassifier(random_state=0)
clf2.fit(X, y)

# Plot decision surface
fig = plt.figure(figsize=(10,8), num=5)
ax = plt.subplot()
fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
plt.title("Decision tree")

# Plot decision surface
fig = plt.figure(figsize=(10,8), num=6)
ax = plt.subplot()
fig = plot_decision_regions(X=X, y=y, clf=clf2, legend=2)
plt.title("Random forest")


RF_trees = []
for _ in range(10):
    
    # Bootstrap new dataset
    n_datapoints = X.shape[0]
    indices = np.random.choice(range(n_datapoints), shape=(n_datapoints,))
    X_bootstrap = X[indices]
    y_bootstrap = y[indices]
    
    # train new tree
    clf = DecisionTreeClassifier()
    clf.fit(X_bootstrap, y_bootstrap)
    RF_trees.append(clf)

# To do: Need to finish this line. Take majority vote here. Not positive how to do this efficiently. 
# custom_RF = lambda x: # To do

