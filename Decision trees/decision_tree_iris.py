"""
This code gives a simple illustration of computing a decision tree
using scikit learn. 

All that is done here is to load the classical iris dataset and 
train a decision tree with the standard fit function. I believe 
this uses the gini index as a gain measure. As an alternative, you 
can specify using the entropy instead. 

TODO: 
    - Look into running pruning. How is it done? How does it improve 
    performance on different datasets
    - Make some notes about how to choose good hyperparameters 
    (mainly, depth. But also the other ones you can choose. Like,
     min number of nodes to have in a split, and other stuff. See 
     scikit learn website/decision trees.
    - Run an example of a random forest. Undertsand the strengths 
    and weaknesses of this. 
       )


"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np

# import graphviz 
iris = load_iris()
X, y = iris.data, iris.target

# train/test split
x_train, x_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=.5)

clf = tree.DecisionTreeClassifier(min_samples_split=2)
clf = clf.fit(x_train, y_train)

acc = clf.score(x_test, y_test)
print(f'acc = {acc}')
# tree.plot_tree(clf)

