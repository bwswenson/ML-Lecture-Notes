knn.py contains a simple implementation of k-NN from scratch (using only numpy)

knn_test.py contains some code to run KNN from knn.py on the classic iris dataset

knn_boundary_demo uses a sklearn implementation of knn to demostrate what the boundary looks like. The code has some comments giving suggestions for a demo. 

Some comments:
 - Note that knn.py is not optimized. It uses nested lists to compute the the distance to all other points. This could be improved to use a vectorized implementation. But it would still involve a brute force search. For large datasets, it is faster to use KD-trees or ball-trees as implemented in scikit learn.  