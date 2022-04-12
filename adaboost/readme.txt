Basic implementation of AdaBoost

Note: This code is not optimized. It could run faster by using the trick mentioned in the UML book to fit decision stumps quickly. (TODO: Fix this.) 

Ideas for demoing the code:
- try increasing the number of classifiers AdaBoost fits. Watch what happens with the test error. It should actually go down for a while before it goes up. Also, plot the train error vs the AdaBoost loss, vs test error. Note that the test error continues to go down after the train error has hit zero. Explain why this is. (Notice what adaboost is using for a loss function internally. It doesn't really care if the training error on the ensemble classifier is zero if the most recent classifier it produced is still making mistakes.) 

