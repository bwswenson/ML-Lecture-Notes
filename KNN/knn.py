import numpy as np
from collections import Counter

def distance_metric(x, z, norm_type = "euclidean"):
    assert norm_type in ["euclidean"] # To do: Add other distance metrics here
    
    if norm_type == "euclidean":
        return ( np.sum( (x - z)**2 ) )**.5
        

class KNN():
    
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
        
    def _predict(self, x):
        # compute distances to each training point
        distances = [distance_metric(x, x_train) for x_train in self.X_train]
        
        # get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # get most common class label with majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
        