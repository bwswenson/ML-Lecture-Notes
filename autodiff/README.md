# autodiff basics
 Basic examples of autodiff and backprop in numpy
 
Lecture 1 and autodiff_basics.py cover the basic idea of autodiff. 

Lecture 2 briefly reviews logistic regression and the cross entropy loss as the negative log likelihood for multi-class classification problems. 

Lecture 3 and autodiff_MNIST_np.py consider the problem of coding up a feedforward network from scratch (as in, using only numpy) to classify MNIST with network parameters tuned using gradient descent. The gradient is computed using the principles of reverse mode autodiff, with each intermediate derivative explicitly computed. (Optimizations for matrix-vector products are not considered.) This is the same as the backprop algorithm. 
