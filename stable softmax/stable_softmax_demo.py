"""
Demo of the logsumexp trick and how to make a stable softmax.

If you set the offset near zero, computing a naive softmax works fine.
But if you crank it up to +/-1000 things break. The logsumexp trick
just uses some algebra so you the only thing you have to exponentiate is
difference between inputs and some constant (like the mean of the inputs). 
This tends to be a lot closer to zero (and hence more stable under
exponentiation) than the raw inputs.
"""

import numpy as np


def softmax(x):
    const = np.sum(np.exp(x))
    return np.exp(x)/const

def logsumexp(x):
    c = np.min(x)
    return c + np.log(np.sum(np.exp(x-c)))

def stable_softmax(x):
    intermediate = x - logsumexp(x)
    return np.exp(intermediate)

if __name__ == "__main__":

    offset = 100_000
    x = offset + np.random.rand(5)

    print(f'stable softmax = {stable_softmax(x)}')
    print(f'sum = {np.sum(stable_softmax(x))}')
    print(f'softmax = {softmax(x)}')
    print(f'sum = {np.sum(softmax(x))}')