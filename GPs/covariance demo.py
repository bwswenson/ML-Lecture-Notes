import math
import numpy as np
import matplotlib.pyplot as plt


'''
This is a demo giving some basic intution on covariance. 
First, we consider discrete random varialbes X and Y with joint distribution:

    y1| a | b |
    y2| c | d |
       x1   x2

This script shows how the covariance and correlation coeff
change as we vary the parameters. 
'''
# Set up X and Y
x1 = -1
x2 = 1
y1 = -1
y2 = 1

a = .25
b = .5
c = .25
d = 1-(a+b+c)

# Compute first and second moments and standard deviation
EX = x1*(a+c) + x2*(1-(a+c))
EY = y1*(c+d) + y2*(1-(c+d))

varX = x1**2*(a+c) + x2**2*(1-(a+c)) - EX**2
varY = y1**2*(c+d) + y2**2*(1-(c+d)) - EY**2
sigmaX = math.sqrt(varX)
sigmaY = math.sqrt(varY)

# compute covariance
corr = a*(x1 - EX)*(y2 - EY) \
    + b*(x2 - EX)*(y2 - EY) \
    + c*(x1 - EX)*(y1 - EY) \
    + d*(x2 - EX)*(y1 - EY) 
    
print(f'corr = {corr}, rho = {corr/(sigmaX*sigmaY)}')


'''
Next, we see examples of RVs that are uncorrelated (Cov = 0), but not
independent
'''

# Example 1: X~U(-1,1), Y = X^2
N = 1000 # number of samples
X = np.random.uniform(low=-1,high=1,size = N)
Y = X**2

fig, ax = plt.subplots(num=1)
ax.scatter(X,Y)
plt.show()

