import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

import math
import os
import time
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Decide if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 100

# Define Discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv6 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv8 = nn.Conv2d(128, 128, 3, 2, 1)
        self.pool1 = nn.MaxPool2d(4, 4)
        
        self.norm1 = nn.LayerNorm([32, 32])
        self.norm2 = nn.LayerNorm([16, 16])
        self.norm3 = nn.LayerNorm([16, 16])
        self.norm4 = nn.LayerNorm([8, 8])
        self.norm5 = nn.LayerNorm([8, 8])
        self.norm6 = nn.LayerNorm([8, 8])
        self.norm7 = nn.LayerNorm([8, 8])
        self.norm8 = nn.LayerNorm([4, 4])

        self.fc1 = nn.Linear(128, 1)
        self.fc10 = nn.Linear(128, 10)

    def forward(self, x, extract_features=0):
        x = F.leaky_relu(self.norm1(self.conv1(x)))
        x = F.leaky_relu(self.norm2(self.conv2(x)))
        x = F.leaky_relu(self.norm3(self.conv3(x)))
        x = F.leaky_relu(self.norm4(self.conv4(x)))
        x = F.leaky_relu(self.norm5(self.conv5(x)))
        x = F.leaky_relu(self.norm6(self.conv6(x)))
        x = F.leaky_relu(self.norm7(self.conv7(x)))
        x = self.conv8(x)
        if extract_features == 8:
            x = F.max_pool2d(x, 4, 4)
            x = x.view(-1, 128)
            return x
        x = self.pool1(F.leaky_relu(self.norm8(x)))
        x = x.view(-1, 128)
        
        return [self.fc1(x), self.fc10(x)]

def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    
    return fig

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = enumerate(testloader)

model = torch.load('tempD.model', map_location="cpu")
model.to(device)
model.eval()

batch_idx, (X_batch, Y_batch) = testloader.__next__()
X_batch = Variable(X_batch,requires_grad=True).to(device)
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).to(device)
Y_batch = Variable(Y_batch).to(device)

X = X_batch.mean(dim=0)
X = X.repeat(batch_size,1,1,1)

Y = torch.arange(batch_size).type(torch.int64)
Y = Variable(Y).to(device)

lr = 0.1
weight_decay = 0.001
for i in range(200):
    output = model(X, 8)

    loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).to(device),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('max_features.png', bbox_inches='tight')
plt.close(fig)
