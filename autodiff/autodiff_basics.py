import numpy as np
import time

"""
Illustration of basics of autodiff with neural networks. 
The gradient for a neural net with one hidden layer is computed. The output
of the network is the logits (and so it typically higher than 1 dimensional).


The network has the following structure
f1 = W1@x + b1
f2 = sigma(f1)
f3 = W2@f2 + b2
out = f3

n0: dimension of inputs
n1: number of hidden units
n2: dimension of output

The gradient is computed with respect to x (for illustration. In practice, you 
want to compute the grad with respect to parameters, but the principles are the
same.). The gradient is the product of three Jacobians; in bad
notation: Df1 x Df2 x Df3, where Dfi is the derivative of the i'th function 
above. See notes. 

The jacobians have dimensions (n0 x n1) x (n1 x n1) x (n1 x n2). The middle 
Jacobian is from the elementwise sigmoid, that's why it preserves dimension.

If n0 is allowed to vary, forward mode complexity grows as n0^3 and reverse 
mode grows as n0^2. This is where reverse mode is better and is the typical
scenario in neural nets. 

If n2 is allowed to vary (so the output of the function is high dimensional)
then forward mode is faster. Forward mode grows as n2^2 while reverse mode 
grows as n2^3. This is good to know, but doesn't really arise in deep learning.

Play around with increasing n0 and n2 separately. 

Derivations of the closed form derivatives used here can be found in the notes.
"""


# Layer sizes
n0 = 500000
n1 = 1000
n2 = 10

# Make a random input for now
x = np.random.normal(size=(n0,1)) # replace this with mnist stuff in training loop

# init W's and b's
W1 = np.random.normal(size=(n1,n0))
b1 = np.random.normal(size=(n1,1))
W2 = np.random.normal(size=(n2,n1))
b2 = np.random.normal(size=(n2,1))

# define functions used in computation
def sigma(z):
    return 1/(1+np.exp(-z))

def f1(y):
    return W1@y + b1

def f2(y):
    return sigma(y)

def f3(y):
    return W2@y + b2

def f(y):
    return f3(f2(f1(y)))

# Define derivatives. Have to figure these out on paper ahead of time. 
def sigma_prime(z):
    return sigma(z)*(1-sigma(z))

def Df1(y):
    return W1

def Df2(y):
    return np.diag(sigma_prime(y).squeeze())

def Df3(y):
    return W2

t_start = time.time()
# Compute df3/dx using "forward mode" evaluation of chain rule
# Quantities are deleted once we're done with them to emphasize that there's
# no memory requirement. This doesn't really effect runtime. 

# Compute first Jacobian
f1_x = f1(x)
Df1_temp = W1

# Compute second Jacobian
f2_y = f2(f1_x)
Df2_temp = np.diag(sigma_prime(f1_x).squeeze())
del f1_x

# Update the right to left matrix accumulation
D_temp_RtL = Df2_temp@Df1_temp
del Df1_temp, Df2_temp

# Compute third Jacobian
f3_y = f3(f2_y)
del f2_y
Df3_temp = W2
del f3_y
D_temp_RtL = Df3_temp@D_temp_RtL
del Df3_temp

# Final grad out
grad_out_forward = D_temp_RtL
t_forward = time.time() - t_start


# Recompute gradient using "reverse mode" eval
t_start = time.time()
# First, do a forward pass and save all the intermediate values
f1_x = f1(x)
f2_x = f2(f1_x)
f3_x = f3(f2_x)

# Left to right accumulation
D_temp_LtR = Df3(f2_x) @ Df2(f1_x)
D_temp_LtR = D_temp_LtR @ Df1(x)
grad_out_reverse = D_temp_LtR
t_back = time.time() - t_start

###########################################################################
# Uncomment the section below to sanity check the derivative computations
# with a finite differences check.
###########################################################################
# # Double check using finite difference approx
# derivative_check = np.zeros_like(grad_out_forward)
# h = .00001 # small perturbation param
# for i, x_i in enumerate(x):
#     x_plus_h = x.copy()
#     x_plus_h[i] = x_plus_h[i] + h
    
#     x_minus_h = x.copy()
#     x_minus_h[i] = x_minus_h[i] - h
    
#     # compute row of the Jacobian    
#     derivative_in_xi = (f(x_plus_h) - f(x_minus_h))/(2*h)
#     derivative_check[:,i] = derivative_in_xi.squeeze()

# print(f'forward error = {np.linalg.norm(derivative_check - grad_out_forward)}')
# print(f'reverse error = {np.linalg.norm(derivative_check - grad_out_reverse)}')
############################################################################
# End finite differences check.
############################################################################


print(f'forward time = {1000*t_forward:.2f}ms, reverse time = {1000*t_back:2f}ms')
