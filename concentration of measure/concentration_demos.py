import numpy as np
import matplotlib.pyplot as plt

# Chebychev's inequality demo
plt.figure(1).clf()
fig, ax = plt.subplots(2,3, num = 1)
n_trials = 1000

# Set number of samples for illustration and value for delta
powers = range(6)
#m_list = [10*3**i for i in powers]
delta = .0001
m_list = [2*i + 1 for i in range(6)]

# indices for plots
index_list = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]

# Set title
fig.suptitle(f'Chebychev bound illustration:\n 1-delta = {1-delta:.3} of mass inside dashed lines')

# Make subplots
for i, m in enumerate(m_list):
    Z_vec = np.sqrt(12)*(np.random.uniform(size = (m,n_trials)) - .5)
    #Z_vec = np.random.normal(size = (m,n_trials))
    Z = np.mean(Z_vec,0)
    ax_now = ax[index_list[i]]
    ax_now.hist(Z, bins = 20)
    ax_now.set_title(f'{m} samples')
    ax_now.axvline(x = 1/np.sqrt(delta*m), color = 'k', linestyle = '--')
    ax_now.axvline(x = -1/np.sqrt(delta*m), color = 'k', linestyle = '--')


# Repeat, but illustrate Hoeffding bound

# initialize fig
plt.figure(2).clf()
fig, ax = plt.subplots(2,3, num = 2)

# Set title
fig.suptitle(f'Hoeffding bound illustration:\n 1-delta = {1-delta:.3} of mass inside dashed lines')

# Make subplots
for i, m in enumerate(m_list):
    Z_vec = np.sqrt(12)*(np.random.uniform(size = (m,n_trials)) - .5)
    #Z_vec = np.random.normal(size = (m,n_trials))
    Z = np.mean(Z_vec,0)
    ax_now = ax[index_list[i]]
    ax_now.hist(Z, bins = 20)
    ax_now.set_title(f'{m} samples')
    ax_now.axvline(x = (-np.log(.5*delta)/2*m)**.5, color = 'k', linestyle = '--')
    ax_now.axvline(x = -(-np.log(.5*delta)/2*m)**.5, color = 'k', linestyle = '--')

'''
Notes:
    - This illustrates Lemma B.2 in UML book and Hoeffding's inequality
'''


X = np.random.uniform(size = (10000), low = 0, high = 10)
#X = np.random.normal(size = (10000))
Y = np.mean((X-0)**1) # Make a positive RV

# What standard Markov tells us:
a = 6
print(f'Prob(Y > a) < {np.mean(Y)/a}')

# If we use a Chernoff bound:
lamb = .75
Z = np.exp(lamb*Y)
print(f'Prob(Y > a) < {np.mean(Z)/np.exp(lamb*a)}')





