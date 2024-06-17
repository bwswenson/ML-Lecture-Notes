"""
This code is an attempt to code up the bayes by backprop algorithm from the paper weight uncertainty in neural networks. The implementation is buggy still. Need to troubleshoot. 
"""

import numpy as np
from numpy import sin, pi
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

class BBBPNet(nn.Module):
    def __init__(self, pi, sigma_p1, sigma_p2):
        super(BBBPNet, self).__init__()
        # self.net = nn.Sequential(nn.Linear(1, 10))
        #                     # nn.ReLU(),
        #                     # nn.Linear(10, 1))
        self.layer1 = nn.Linear(1, 1)
        # self.layer2 = nn.Linear(1, 1)
        self.weights = [p for p in self.parameters()]
        self.rhos = [nn.Parameter(torch.randn_like(p), requires_grad=True) for p in self.parameters()]  # change this initialization to Xavier or something
        self.mus = [nn.Parameter(torch.randn_like(p), requires_grad=True) for p in self.parameters()]
        self.pi = pi
        self.sigma_p1 = sigma_p1
        self.sigma_p2 = sigma_p2
        self.optimized_params = (p for p in (self.rhos + self.mus))

    @staticmethod
    def norm_pdf(x, sigma):
        return 1/np.sqrt(2*np.pi) * 1 / sigma * torch.exp(-.5 *(x)**2/sigma**2)

    def forward(self, x):
        x = self.layer1(x)
        # x = F.relu(x)
        # x = self.layer2(x)
        return x

    def fetch_loss_sample(self, X, Y):
        M = X.shape[0]
        KL_loss = torch.tensor(0.0, requires_grad=True)
        prior_loss = torch.tensor(0.0, requires_grad=True)

        # compute weights sample
        self.eps_vals = []
        sigma_sum = 0.0
        for p, rho, mu in zip(self.weights, self.rhos, self.mus):
            eps = torch.randn_like(p)
            self.eps_vals.append(eps)
            sigma = torch.log(1+torch.exp(rho))
            w = mu + sigma*eps
            p.data = w
            sigma_sum += sigma.sum()  # track sum_i sigma_i. After collecting them all, we'll take the log.
            KL_loss = KL_loss + sum((-.5*(mu - p)**2/sigma**2).flatten()) - sum((-0.5 *(p)**2).flatten())   # just use N(0,1) prior. See if this maybe helps with numerical issues
            # prior_loss = prior_loss - sum((torch.log(self.pi*self.norm_pdf(p, self.sigma_p1) + (1-self.pi)*self.norm_pdf(p, self.sigma_p2))).flatten())

        KL_loss += torch.log(sigma_sum)  # log(sum_i sigma_i)
        posterior_loss = - sum((-.5*(self.forward(X) - Y)**2).flatten())
        return KL_loss, prior_loss, posterior_loss

    def sample_forward(self, x):

        # draw a weights sample
        for p, rho, mu in zip(self.weights, self.rhos, self.mus):
            eps = torch.randn_like(p)
            sigma = torch.log(1+torch.exp(rho))
            w = mu + sigma*eps
            p.data = w

        return self.forward(x)

    def grad_update(self, lr):

        # mu update
        for p, mu in zip(self.weights, self.mus):
            grad = p.grad + mu.grad
            with torch.no_grad():
                mu.data -= lr*grad

        # rho update
        for p, rho, eps_val in zip(self.weights, self.rhos, self.eps_vals):
            grad = p.grad*eps_val/(1+torch.exp(-rho)) + rho.grad
            with torch.no_grad():
                rho -= lr*grad

    def overwrite_grads(self):

        # mu update
        for p, mu in zip(self.weights, self.mus):
            grad = p.grad + mu.grad
            mu.grad = grad

        # rho update
        for p, rho, eps_val in zip(self.weights, self.rhos, self.eps_vals):
            grad = p.grad*eps_val/(1+torch.exp(-rho)) + rho.grad
            rho.grad = grad

if __name__ == "__main__":

    # build a toy dataset
    N = 10_000
    sigma = .1  # noise var
    x = np.random.rand(N)*2-1
    f = lambda x: 2*x
    # f = lambda x: (sin(x) + .5*sin(3*x - pi/4))*np.exp(-abs(.5*x))
    y = f(x) + sigma*np.random.normal(size=N)
    X = torch.tensor(x).float().view(-1, 1)
    Y = torch.tensor(y).float().view(-1, 1)

    plt.scatter(x, y)
    vec = np.linspace(-2, 3, 100)
    # plt.plot(vec, f(vec))
    # plt.show()



    # Maybe try and do this with Bayesian linear regression? Could also try and do it with bayesian regression with a polynomial.
    # And could then move to a kernelized version with a polynomial kernel? Then, from there, move to a neural network?

    mynet = BBBPNet(.5, .5, .5)

    optim = torch.optim.Adam(mynet.optimized_params, lr=1e-6)  # pass in only the rho and mu parameters

    for i in range(1000):

        mynet.zero_grad()
        losses = mynet.fetch_loss_sample(X, Y)
        loss = sum(losses)
        loss.backward()
        mynet.grad_update(lr=1e-5)
        # mynet.overwrite_grads()
        # optim.step()

        if i % 500 == 0:
            print(f'loss={loss}')
            print(f'weights = {sum([sum((p.data**2).flatten()) for p in mynet.weights])}')

    # print(f'mu={torch.tensor(mynet.mus)}, sig={torch.log(1+torch.exp(torch.tensor(mynet.rhos)))}')

# display network output

c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  # color scheme (copied this)

n_nets = 50
y = []
for _ in range(n_nets):
    x = torch.linspace(-1, 1, 40).view(-1, 1)
    y.append(mynet.sample_forward(x).tolist())
y = torch.tensor(y)
means = y.mean(dim=0).squeeze()
vars = y.var(dim=0).squeeze()
plt.plot(x.squeeze().numpy(), means)
plt.fill_between(x.squeeze(), means, means + vars, color = c[0], alpha = 0.3, label = r'$\sigma(y^*|x^*)$')
plt.fill_between(x.squeeze(), means, means - vars, color = c[0], alpha = 0.3, label = r'$\sigma(y^*|x^*)$')
plt.scatter(X.squeeze(), Y.squeeze())
plt.show()