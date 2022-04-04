
import numpy as np
# import autograd.numpy as anp
# import jax.numpy as np
# import jax
from scipy.linalg import block_diag
from torch import nn
import matplotlib.pyplot as plt
import time

# These are only used to load datasets. No computations. 
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pickle

# helper function
def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

# Define some layers and their derivatives
def softmax(x):
    temp = x - logsumexp(x)
    return np.exp(temp)
    # return np.exp(x)/sum(np.exp(x))

def D_softmax(x):
    temp = logsumexp(x)
    c = 1/(np.exp(temp))**2
    M = -np.outer(np.exp(x), np.exp(x))
    # v = np.exp(x)*sum(np.exp(x))
    v = np.exp(x)*np.exp(temp) # previous line works, but I think this should be more stable?
    return c*(M + np.diag(v))

def linear(W, b, x):
    return W@x + b
    
def sigmoid(x):
    return np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))

def sigma_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def D_sigmoid(z):
    return np.diag(sigma_prime(z).squeeze())

def D_x_linear(W, b, x):
    return W

def D_W_linear(x, W):
    m = W.shape[0]
    n = W.shape[1]
    assert x.shape[0] == n
    temp_tuple = tuple([x.copy() for i in range(m)])
    return block_diag(*temp_tuple)

def D_b_linear(b):
    return np.eye(b.shape[0])

def CE_loss(y_hat, y_target):
    return -sum(y_target*np.log(y_hat))

def CE_loss2(y_hat, y_target_index):
    return -np.log(y_hat[y_target_index])

def CE_loss_torch(y_hat, y_target):
    return -sum((y_target*torch.log(y_hat)).squeeze())

CE_loss_torch2 = nn.CrossEntropyLoss()

def D_CE_loss(y_hat, y_target):
    return -y_target*1/y_hat

# Helper function
def onehot(i: int, N: int):
    # N = Length of onehot vector
    out = np.zeros(N)
    out[i] = 1
    return out

def test_grad(x, f, N, f_out_dim):
   
    derivative_check = np.zeros((f_out_dim, N))
    h = .00001 # small perturbation param
    for i, x_i in enumerate(x):
        x_plus_h = x.copy()
        x_plus_h[i] = x_plus_h[i] + h
        
        x_minus_h = x.copy()
        x_minus_h[i] = x_minus_h[i] - h
        
        # compute row of the Jacobian    
        derivative_in_xi = (f(x_plus_h) - f(x_minus_h))/(2*h)
        derivative_check[:,i] = derivative_in_xi.squeeze()
    
    return derivative_check

def test_bunch_of_gradients():
    
    print('%%%%% runing some test %%%%%')
    # generate test data
    N = 5
    x = np.random.random(N)
    W = np.random.random((N,N))
    b = np.random.random(N)
    
    # Test cross entropy loss
    y = np.random.random(N)
    f = lambda z: CE_loss(z,y)
    d_check = test_grad(x, f, N, f_out_dim=1)
    d_back = D_CE_loss(x, y)
    print(f'cross entropy error = {np.linalg.norm(d_check - d_back)}')
    
    # Test softmax layer
    f = softmax
    d_check = test_grad(x, f, N, f_out_dim=N)
    d_back = D_softmax(x)
    print(f'softmax error = {np.linalg.norm(d_check - d_back)}')
    
    # Test linear layer in x
    f = lambda z: W @ z
    d_check = test_grad(x, f, N, f_out_dim=N)
    d_back = D_x_linear(W, b, x)
    print(f'linear in x error = {np.linalg.norm(d_check - d_back)}')
    
    # Test linear layer in W
    f = lambda z: z.reshape(N,N) @ x + b
    d_check = test_grad(W.reshape(-1), f, N**2, f_out_dim=N)
    d_back = D_W_linear(x, W)
    print(f'linear in W error = {np.linalg.norm(d_check - d_back)}')
    
    # Test linear layer in b
    f = lambda z: W @ x + z
    d_check = test_grad(b, f, N, f_out_dim=N)
    d_back = D_b_linear(b)
    print(f'linear in b error = {np.linalg.norm(d_check - d_back)}')
    
    # Test sigmoid layer
    f = sigmoid
    d_check = test_grad(x, f, N, f_out_dim=N)
    d_back = D_sigmoid(x)
    print(f'sigmoid error = {np.linalg.norm(d_check - d_back)}')
    
    print('%%%%% end tests %%%%%')
    
# Define a network
class Net():
    def __init__(self, n0, n1, n2, n3):
            
        # self.W1 = np.random.normal(size=(n1,n0))*1/np.sqrt(n0)
        # self.b1 = np.zeros(shape=(n1,))
        # self.W2 = np.random.normal(size=(n2,n1))*1/np.sqrt(n1)
        # self.b2 = np.zeros(shape=(n2,))
        # self.W3 = np.random.normal(size=(n3,n2))*1/np.sqrt(n2)
        # self.b3 = np.zeros(shape=(n3,))
        
        # param_dict = {'W1': self.W1,
        #                'b1': self.b1,
        #                'W2': self.W2,
        #                'b2': self.b2,
        #                'W3': self.W3,
        #                'b3': self.b3}
        
        # with open('model_params.pickle', 'wb') as f:
        #     pickle.dump(param_dict, f)
            
        with open('model_params.pickle', 'rb') as f:
            d = pickle.load(f)
        self.W1 = d['W1']
        self.b1 = d['b1']
        self.W2 = d['W2']
        self.b2 = d['b2']
        self.W3 = d['W3']
        self.b3 = d['b3']
    
    def set_target_in_loss(self, y):
        self.y_target = y
    
    def evaluate(self, x):
        x0 = x
        x1 = linear(self.W1, self.b1, x0)
        x2 = sigmoid(x1)
        x3 = linear(self.W2, self.b2, x2)
        x4 = sigmoid(x3)
        x5 = linear(self.W3, self.b3, x4)
        x6 = softmax(x5)
        return x0, x1, x2, x3, x4, x5, x6
    
    def forward(self, x):
        x0, x1, x2, x3, x4, x5, x6 = self.evaluate(x)
        self.x0 = x0
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.x5 = x5
        self.x6 = x6
        return self.x6
    
    def backward(self):
        # bookkeeping 
        W1, W2, W3 = self.W1, self.W2, self.W3
        b1, b2, b3 = self.b1, self.b2, self.b3
        
        # Compute gradient of last layer params
        J_LtR = D_CE_loss(self.x6, self.y_target)
        J_LtR = J_LtR @ D_softmax(self.x5)
        DW3 = J_LtR @ D_W_linear(self.x4, W3)
        DW3 = DW3.reshape(W3.shape[0], W3.shape[1])
        Db3 = J_LtR @ D_b_linear(b3)
        
        # compute gradient of 2nd layer params
        J_LtR = J_LtR @ D_x_linear(W3, b3, self.x4)
        J_LtR = J_LtR @ D_sigmoid(self.x3)
        DW2 = J_LtR @ D_W_linear(self.x2, W2)
        DW2 = DW2.reshape(W2.shape[0], W2.shape[1])
        Db2 = J_LtR @ D_b_linear(b2)
        
        # gradient of 1st layer params
        J_LtR = J_LtR @ D_x_linear(W2, b2, self.x2)
        J_LtR = J_LtR @ D_sigmoid(self.x1)
        DW1 = J_LtR @ D_W_linear(self.x0, W1)
        DW1 = DW1.reshape(W1.shape[0], W1.shape[1])
        Db1 = J_LtR @ D_b_linear(b1)
        
        # Derivative in input, just for fun
        self.Dx = J_LtR @ D_x_linear(W1, b1, self.x1)
        
        # collect gradients
        grad_dict = {'W1': DW1,
                     'W2': DW2,
                     'W3': DW3,
                     'b1': Db1,
                     'b2': Db2,
                     'b3': Db3,
                    }
        
        return grad_dict, [('W1', DW1), 
                            ('b1', Db1), 
                            ('W2', DW2), 
                            ('b2', Db2), 
                            ('W3', DW3), 
                            ('b3', Db3)]
    
    def params(self):
        
        # collect paramers for easy iteration
        param_dict = {'W1': self.W1,
                      'W2': self.W2,
                      'W3': self.W3,
                      'b1': self.b1,
                      'b2': self.b2,
                      'b3': self.b3
                     }
        
        return iter(param_dict.items())

    def parameter_list(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        
    def grad_sanity_check(self,x):
        derivative_check = np.zeros_like(x)
        h = .0_001 # small perturbation param
        for i, x_i in enumerate(x):
            x_plus_h = x.copy()
            x_plus_h[i] = x_plus_h[i] + h
            
            x_minus_h = x.copy()
            x_minus_h[i] = x_minus_h[i] - h
            
            # compute row of the Jacobian   
            *_, f_plus = self.evaluate(x_plus_h)
            loss_plus = CE_loss(f_plus, self.y_target)
            *_, f_minus = self.evaluate(x_minus_h)
            loss_minus = CE_loss(f_minus, self.y_target)
            derivative_in_xi = (loss_plus - loss_minus)/(2*h)
            derivative_check[i] = derivative_in_xi
        return derivative_check
    
if __name__ == "__main__":
    
    # # Test closed form gradient computations
    # test_bunch_of_gradients()

    batch_size = 1
    
    # Some crap to load dataset
    if True:
        # trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
        transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
        torch.manual_seed(1)
        trainset = datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=1)
        testset = datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                  shuffle=True, num_workers=1)
        dataset_size = trainloader.dataset.data.shape[0]
    
    # Layer sizes
        n0 = 784
        n1 = 32
        n2 = 10
        n3 = 10
        
    # instantiate our net
    net = Net(n0, n1, n2, n3)
    
    # Copy of net in pytorch for comparison 
    torch_net = nn.Sequential(
                              nn.Linear(n0, n1),
                              nn.Sigmoid(),
                              nn.Linear(n1,n2),
                              nn.Sigmoid(),
                              nn.Linear(n2,n3)
                              )
    
    #step size for GD
    alpha = (1e-3)*2**4               
    
    # # compute initial test set accuracy
    # test_sum = 0
    # for i, (images, labels) in enumerate(testloader):
    #     images = images.view(-1).numpy()
    #     y_hat = net.forward(images).argmax()
    #     test_sum += int(y_hat == labels.item())
    #     n=i
    # test_acc = test_sum/(n+1)
    # print(f'test acc = {test_acc}')
    
    # GD loop
    t_start = time.time()
    avg_loss = 5
    for epoch in range(5):
        # exponentially decaying step size per epoch
        alpha = alpha*2**(-epoch)
        for i, (images, labels) in enumerate(trainloader):
            
            # Do forward pass
            images = images.view(-1).numpy()
            y_hat = net.forward(images)
            
            # Set cost function using true label
            labels_onehot = onehot(labels, N=10)
            net.set_target_in_loss(labels_onehot)
            
            # Do backwards pass
            gradients, g_list = net.backward()
            
            # # check gradients with pytorch
            if False:
                # Get my net parameters
                p_list = net.parameter_list()
                
                # copy params to pytorch
                for i, p in enumerate(torch_net.parameters()):
                    p_my_model = p_list[i]
                    p.data.copy_(torch.tensor(p_my_model))
                    
                # check copy operation
                no_error = True
                for i, p in enumerate(torch_net.parameters()):
                    error = torch.tensor(p_list[i]) - p
                    if error.max() > 1e5:
                        no_error = False
                if no_error:
                    print('copy good')
                else:
                    print('copy error')
                    
                # compute gradient in pytorch
                y_hat_torch = torch_net(torch.tensor(images)).unsqueeze(0)
                softmax_torch = nn.Softmax()
                loss1 = CE_loss_torch(softmax_torch(y_hat_torch), torch.tensor(labels_onehot))
                loss2 = CE_loss_torch2(y_hat_torch, labels)
                torch_net.zero_grad()
                loss2.backward()
                # loss2.backward()
                
                # Check against pytorch gradient
                for i, p in enumerate(torch_net.parameters()):
                    this_grad = g_list[i][1]
                    error = this_grad - p.grad.numpy()
                    print(f'error = {error.max()}')
                    print('pausing...')
                    
                # zero pytorch gradients
                torch_net.zero_grad()
            
            # compute current value of loss (to monitor loss)
            # For first 5000 iterations, compute average; exponential moving average thereafter 
            current_loss = CE_loss(y_hat, labels_onehot)
            if epoch == 0 and i < 5_000:
                avg_loss = ((1-1/(i+1))*avg_loss + 1/(i+1)*current_loss)
            else:
                avg_loss = (1-.0001)*avg_loss + .0001*current_loss
            if i % 100 == 0:
                print( (f'\repoch = {epoch}, i = {i}/{dataset_size}, ' +
                      f'avg loss = {avg_loss:.3f}'), end='')
            
            if i*batch_size > 5_000:
                t_run = time.time() - t_start
                print(f'\nruntime = {t_run} Seconds')
                break
            
            # take gradient step
            for p, g in zip(net.params(), iter(gradients.items())):
                param = p[1]
                grad = g[1]
                param += -alpha*grad
        
        print('')
        test_sum = 0
        for i, (images, labels) in enumerate(testloader):
            images = images.view(-1).numpy()
            y_hat = net.forward(images).argmax()
            test_sum += int(y_hat == labels.item())
            n=i
        test_acc = test_sum/(n+1)
        print(f'test acc = {test_acc}')
        
        # fig = plt.figure
        # plt.imshow(images.reshape(28,28), cmap='gray')
        # plt.show()