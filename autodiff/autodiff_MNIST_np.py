
import numpy as np
from scipy.linalg import block_diag # This is the only non-numpy thing used. 
import time

# These are only used to load datasets. No computations. 
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# This is used to visualize data
from torch.utils.tensorboard import SummaryWriter

# # This fixes an os error I get on some machines
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# helper function
def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

# Define some layers and their derivatives
def softmax(x):
    temp = x - logsumexp(x)
    return np.exp(temp)
    # return np.exp(x)/sum(np.exp(x))

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

def D_softmax(x):
    temp = logsumexp(x)
    c = 1/(np.exp(temp))**2
    M = -np.outer(np.exp(x), np.exp(x))
    # v = np.exp(x)*sum(np.exp(x))
    v = np.exp(x)*np.exp(temp) # previous line works, but I think this should be more stable?
    return c*(M + np.diag(v))

def CE_loss(y_hat, y_target):
    return -sum(y_target*np.log(y_hat))

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
        """
        Initialize weights using Xavier initialization scheme: 
            http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf. 
        """
        bound1 = np.sqrt(6)/np.sqrt(n0 + n1)
        self.W1 = np.random.uniform(size=(n1, n0), low=-bound1, high=bound1)
        self.b1 = np.zeros(shape=(n1,))
        
        bound2 = np.sqrt(6)/np.sqrt(n1 + n2)
        self.W2 = np.random.uniform(size=(n2, n1), low=-bound2, high=bound2)
        self.b2 = np.zeros(shape=(n2,))
        
        bound3 = np.sqrt(6)/np.sqrt(n2 + n3)
        self.W3 = np.random.uniform(size=(n3, n2), low=-bound3, high=bound3)
        self.b3 = np.zeros(shape=(n3,))
    
    def set_target_in_loss(self, y):
        self.y_target = y
    
    def evaluate(self, x):
        """
        Perform the forward pass of the network without saving intermediate values. 
        
        The difference between forward and evaluate is that forward 
        actually saves the intermediate values from the forward pass. 
        I did this so I could call evaluate and get the loss without having
        to save intermediate values if I want. 
        """
        
        x0 = x
        x1 = linear(self.W1, self.b1, x0)
        x2 = sigmoid(x1)
        x3 = linear(self.W2, self.b2, x2)
        x4 = sigmoid(x3)
        x5 = linear(self.W3, self.b3, x4)
        x6 = softmax(x5)
        return x0, x1, x2, x3, x4, x5, x6
    
    def forward(self, x):
        """
        Perform a forward pass of the network. 
        """
        
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
        """
        Compute gradients for each parameter. 
        
        This must be called after forward so the intermediate values 
        (base points for derivative computation) are known/computed.
        """
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
        
        # Derivative in input, just for kicks
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
        """
        Return the current parameters of the network. 
        
        Simple function for convenience. 
        """
        param_dict = {'W1': self.W1,
                      'W2': self.W2,
                      'W3': self.W3,
                      'b1': self.b1,
                      'b2': self.b2,
                      'b3': self.b3
                     }
        
        return iter(param_dict.items())
    
    
class Optim():
    def __init__(self, params, opt_method, lr, eta = None):
        
        assert opt_method in ['SGD', 'SGDmomentum', 'Adam']
        self.opt_method = opt_method
        
        # set up parameters
        if self.opt_method == 'SGD':
            self.alpha = lr
        elif self.opt_method == 'SGDmomentum':
            self.alpha = lr
            if eta == None:
                self.eta = 0.9 # This is supposed to be a good default value
            else:
                self.eta = eta
            
            # initialize the variable you track over time for momentum
            w = []
            for p in params:
                name = p[0]
                param = p[1]
                w.append(np.zeros_like(param))
            self.w = w
            
        elif self.opt_method == 'Adam':
            # Initialize parameters
            self.beta1 = 0.9
            self.beta2 = .999
            self.alpha = lr
            
            # initialize the variables you track over time
            m_list = []
            v_list = []
            for p in params:
                name = p[0]
                param = p[1]
                m_list.append(np.zeros_like(param))
                v_list.append(np.zeros_like(param))
            self.m_list = m_list
            self.v_list = v_list
            
        
    def step(self, params, grads):
        # choose type of optimization step to take
        if self.opt_method == 'SGD':
            self._SGD_step(params, grads)
        elif self.opt_method == 'SGDmomentum':
            self._SGD_momentum_step(params, grads)
        elif self.opt_method == 'Adam':
            self._Adam_step(params, grads)
    
    def _SGD_step(self, params, grads):
        alpha = self.alpha
        for p, g in zip(params, grads):
            # unpack parameter data (for debugging)
            name_param = p[0]
            param = p[1]
            name_grad = g[0]
            grad = g[1]
            
            # take optimization step
            param += -alpha*grad
        
    def _SGD_momentum_step(self, params, grads):
        
        alpha = self.alpha
        eta = self.eta 
        temp_var = self.w
        
        for p, g, w in zip(params, grads, temp_var):
            # unpack parameter data (for debugging)
            name_param = p[0]
            param = p[1]
            name_grad = g[0]
            grad = g[1]
            
            # take optimization step
            w_new = eta*w - alpha*grad
            np.copyto(dst=w, src=w_new) # IMPORTANT: w = eta*w - alpha*grad doesn't write to same w. Need to do inplace operation. 
            param += w
        
    def _Adam_step(self, params, grads):
        beta1 = self.beta1
        beta2 = self.beta2
        alpha = self.alpha
        m_list = self.m_list
        v_list = self.v_list
        
        for p, g, m, v in zip(params, grads, m_list, v_list):
            # unpack parameter data (for debugging)
            name_param = p[0]
            param = p[1]
            name_grad = g[0]
            grad = g[1]
            
            # take optimization step
            m_new = beta1*m + (1-beta1)*grad
            v_new = beta2*v + (1-beta2)*grad**2
            m_hat = m_new/(1-beta1)
            v_hat = v_new/(1-beta2)
            np.copyto(m, m_new)
            np.copyto(v, v_new)
            param += -alpha*m_hat/(np.sqrt(v_hat) + 1e-8)
    
if __name__ == "__main__":
    
    # Test closed form gradient computations
    test_bunch_of_gradients()

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
    
    # Set up tensorboard to track plots
    writer = SummaryWriter()
    
    # Layer sizes
    n0 = 784
    n1 = 32
    n2 = 10
    n3 = 10
        
    # instantiate our net
    net = Net(n0, n1, n2, n3)
    
    #step size for GD
    alpha = .001 # OK for Adam
    alpha = (1e-3)*2**4 # Good for SGD
    
    # instantiate our optimizer
    optimizer = Optim(net.params(), 
                      opt_method = 'Adam', #use SGD or SGDmomentum or Adam
                      lr=alpha)
    
    # compute initial test set accuracy
    test_sum = 0
    for i, (images, labels) in enumerate(testloader):
        images = images.view(-1).numpy()
        y_hat = net.forward(images).argmax()
        test_sum += int(y_hat == labels.item())
        n=i
    test_acc = test_sum/(n+1)
    print(f'test acc = {test_acc}')
    
    # GD loop
    t_start = time.time()
    avg_loss = 5
    loss_plot = []
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
            
            # take gradient step
            optimizer.step(net.params(), iter(gradients.items()))
                
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
                loss_plot.append(avg_loss)
                
                writer.add_scalar('loss', avg_loss, epoch*60_000 + i)
        print('')
        
        # Check accuracy of trained network
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