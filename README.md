This repository contains lecture notes and code demos for core ML topics. 

Topics with written notes (handwritten or LaTeX), most with accompanying code:
- Linear algebra (SVD, pseudoinverse, low-rank approximation): `Latex notes/Some personal notes` (personal notes)
- Calculus and analysis (the Laplacian, Lagrange multipliers, change of variables, polar coordinates): `Latex notes/Some personal notes` (personal notes)
- Least squares and linear regression (pseudoinverse solution, Bayesian linear regression): `Lecture notes/Least squares notes`, `Latex notes/Some personal notes` (personal notes)
- Ridge regression: `Ridge regression`
- PCA (maximum variance perspective, connection to the SVD, kernel PCA): `Latex notes/PCA`, `Lecture notes/Least squares notes`, `Latex notes/Some personal notes` (personal notes), `Kernel PCA`
- Automatic differentiation, logistic regression, MNIST with NumPy: `autodiff/lecture notes`
- Gaussian processes: `GPs/lecture notes`
- Variational autoencoders and the reparameterization trick: `VAEs/Latex notes` (LaTeX source only), `Reparameterization trick lecture code`
- Support vector machines: `SVM`
- Policy gradient and REINFORCE: `Policy gradient`
- PAC learning: `PAC learning`
- High-dimensional spaces and concentration of measure (volume of balls and cubes, high-dimensional Gaussians): `concentration of measure`, `Latex notes/Some personal notes` (personal notes)

Topics with code only:
- MLE and MAP estimation: `MLE and MAP`
- k-nearest neighbors: `KNN`
- AdaBoost: `adaboost`
- MCMC: `MCMC lecture code`
- Numerically stable softmax: `stable softmax`

Some of these notes were prepared for lectures in an ML seminar. The personal notes, found at [`Latex notes/Some personal notes`](Latex%20notes/Some%20personal%20notes), contain notes on some topics of interest. These are not polished and have not been reviewed. But may contain useful ideas that could be converted to readable lecture notes. 

The folder "Other topics" contains notes or code written by others, kept here only as a useful reference. Topics include: Bayes by Backprop, Bayesian inference (hierarchical linear Gaussian models), conjugate gradient, decision trees, naive Bayes, Karpathy micrograd repo

Books used in writing these notes include
- S. Shalev-Shwartz and S. Ben-David, *Understanding Machine Learning: From Theory to Algorithms*, Cambridge University Press, 2014.
- C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.
- C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine Learning*, MIT Press, 2006.
- K. P. Murphy, *Probabilistic Machine Learning: An Introduction*, MIT Press, 2022.
- A. Blum, J. Hopcroft, and R. Kannan, *Foundations of Data Science* (Chapter 1, High-Dimensional Space).
- S. Roch, *Mathematical Methods in Data Science* ([online textbook](https://mmids-textbook.github.io/)).

<!--
Other sources include
- D. P. Kingma and M. Welling, [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114), 2013.
- D. M. Blei, A. Kucukelbir, and J. D. McAuliffe, Variational Inference: A Review for Statisticians, 2017.
- X. Glorot and Y. Bengio, [Understanding the Difficulty of Training Deep Feedforward Neural Networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf), AISTATS 2010.
- UCL Smart Geometry, [Deep Learning for Graphics VAE notebook](https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/variational_autoencoder.ipynb).
- G. Strang, figure on the action of the pseudoinverse.
- R. Tibshirani, notes on ridge regression and LASSO.
-->