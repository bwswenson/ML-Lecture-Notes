import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class MyNet(nn.Module):
    
    def __init__(self):
        super(MyNet, self).__init__()
        n0 = 784
        n1 = 32
        n2 = 10
        n3 = 10
        self.net = nn.Sequential(nn.Linear(n0, n1),
                                 nn.BatchNorm1d(n1),
                                 nn.ReLU(),
                                 nn.Linear(n1, n2),
                                 nn.BatchNorm1d(n2),
                                 nn.ReLU(),
                                 nn.Linear(n2, n3)
            )
        
    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    
    batch_size = 32
    
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
    
    net = MyNet()
    
    optimizer = optim.Adam(net.parameters(), lr=.001)
    
    avg_loss = 0
    
    loss = nn.CrossEntropyLoss()
    for i, (images, labels) in enumerate(trainloader):
        images = images.view(-1, 28*28)
        y_hat = net(images)
        current_loss = loss(y_hat, labels)
        current_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # compute average loss
        current_loss = current_loss.item()
        if i < 5_000:
            avg_loss = ((1-1/(i+1))*avg_loss + 1/(i+1)*current_loss)
        else:
            avg_loss = (1-.0001)*avg_loss + .0001*current_loss
        
        if i % 100 == 0:
            print( (f'\repoch = {0}, i = {i*batch_size}/{dataset_size}, ' +
                  f'avg loss = {avg_loss:.3f}'), end='')
        