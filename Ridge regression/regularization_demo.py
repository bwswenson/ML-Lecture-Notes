import torch
import torch.optim as optim
import numpy as np
import itertools
import matplotlib.pyplot as plt

#optimizer stuff
tau = .1    # for manual GD
LR = .01    # for ADAM optimizer
N_steps = 4_000
tol = 1e-6 # tolerance for solving optimization problem

# other params
N_points = 5
degree = 5
reg_coeff = .1

# compute features
x = np.linspace(0,1,N_points)
y = x + (np.random.rand(N_points)-.5)*1
powers = [i for i in range(degree)]

def regress(x, y, powers, lamb, fig_num):
    N_points = x.shape[0]
    alpha = torch.rand((len(powers),), requires_grad = True)
    optimizer = optim.Adam([alpha], lr = LR)
    targets = torch.tensor(y)
    X = [x**i for i in powers]
    X = torch.tensor(X, dtype = torch.float32)
    for t in range(N_steps):
        optimizer.zero_grad()
        y_hat = torch.matmul(alpha,X)
        loss = 1/N_points*torch.sum((y_hat - targets)**2 + lamb*torch.norm(alpha))
        loss.backward()
        optimizer.step()
        
        if t % 10000 == 0:
            print('\r' + 30*' ', end = '') # clear out the stupid line so I can reprint to it
            print(f'\rt = {t}, loss = {loss.item():.2}', end='')
            
        if loss < tol:
            print('\r' + 30*' ', end = '') # clear out the stupid line so I can reprint to it
            print(f'\rt = {t}, loss = {loss.item():.2}', end='')
            break
        
    z = np.linspace(0,1,1000)
    Z = [z**i for i in powers]
    f = alpha.detach().numpy() @ Z
    
    plt.figure(fig_num).clf()
    fig, ax = plt.subplots(num=fig_num)
    ax.scatter(x,y)
    ax.plot(z,f,x,x)
    ax.set_title(f'lambda = {lamb}, loss = {loss}')
    print(f'\nfig num {fig_num}: alpha = {alpha.data}')
    
    
def norm_eq_soln(x, y, powers, lamb, fig_num):
    # Compute directly with matrix inversion
    N_points = x.shape[0]
    m = N_points
    X = np.array([x**i for i in powers])
    y = np.array(y)
    
    alpha = np.linalg.inv(X.T @ X + 2*m*np.eye(degree)*lamb) @ X.T @ y
    
    empirical_loss = 1/N_points*sum( (alpha @ X - y)**2)
    
    z = np.linspace(0,1,1000)
    Z = np.array([z**i for i in powers])
    f = alpha @ Z
    
    plt.figure(fig_num).clf()
    fig, ax = plt.subplots(num = fig_num)
    ax.scatter(x,y)
    ax.plot(z,f, x,x)
    ax.set_title(f'Normal eq soln\nLambda = {lamb} empirical loss = {empirical_loss:.2f}')
    print(f'\nfig num {fig_num}: alpha = {alpha}')
    
regress(x,y,powers, lamb = 0, fig_num = 1)
regress(x,y,powers,lamb = reg_coeff, fig_num = 2)
norm_eq_soln(x, y, powers, lamb = reg_coeff, fig_num = 3)
norm_eq_soln(x, y, powers, lamb = 0, fig_num = 4)