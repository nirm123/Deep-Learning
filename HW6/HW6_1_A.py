import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import math
import os
import time
import numpy as np
import sys

# Hyper paramaters
batch_size = 128
num_epochs = 100

# Define Discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        
        # Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv1 = nn.Conv2d(3, 196, 3, 1, 1)
        self.l1 = nn.LayerNorm([32, 32])
        
        self.conv2 = nn.Conv2d(196, 196, 3, 2, 1)
        self.l2 = nn.LayerNorm([16, 16])
        
        self.conv3 = nn.Conv2d(196, 196, 3, 1, 1)
        self.l3 = nn.LayerNorm([16, 16])
        
        self.conv4 = nn.Conv2d(196, 196, 3, 2, 1)
        self.l4 = nn.LayerNorm([8, 8])
        
        self.conv5 = nn.Conv2d(196, 196, 3, 1, 1)
        self.l5 = nn.LayerNorm([8, 8])
        
        self.conv6 = nn.Conv2d(196, 196, 3, 1, 1)
        self.l6 = nn.LayerNorm([8, 8])
        
        self.conv7 = nn.Conv2d(196, 196, 3, 1, 1)
        self.l7 = nn.LayerNorm([8, 8])
        
        self.conv8 = nn.Conv2d(196, 196, 3, 2, 1)
        self.l8 = nn.LayerNorm([4, 4])

        self.pool = nn.MaxPool2d(4, 4, 0)

        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, 10)

    def forward(self, x):
        x = F.leaky_relu(self.l1(self.conv1(x)))
        x = F.leaky_relu(self.l2(self.conv2(x)))
        x = F.leaky_relu(self.l3(self.conv3(x)))
        x = F.leaky_relu(self.l4(self.conv4(x)))
        x = F.leaky_relu(self.l5(self.conv5(x)))
        x = F.leaky_relu(self.l6(self.conv6(x)))
        x = F.leaky_relu(self.l7(self.conv7(x)))
        x = F.leaky_relu(self.l8(self.conv8(x)))
        x = self.pool(x)
        x = x.view(-1, 196)

        fake = self.fc1(x)
        clas = self.fc10(x)

        return [clas, fake]

# Define Generator
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()

        self.fc1 = nn.Linear(100, 196*4*4)
        
        self.conv1 = nn.ConvTranspose2d(196, 196, 4, 2, 1)
        self.norm1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(196, 196, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(8)

        self.conv3 = nn.Conv2d(196, 196, 3, 1, 1)
        self.norm3 = nn.BatchNorm2d(8)

        self.conv4 = nn.Conv2d(196, 196, 3, 1, 1)
        self.norm4 = nn.BatchNorm2d(8)

        self.conv5 = nn.ConvTranspose2d(196, 196, 4, 2, 1)
        self.norm5 = nn.BatchNorm2d(16)

        self.conv6 = nn.Conv2d(196, 196, 3, 1, 1)
        self.norm6 = nn.BatchNorm2d(16)

        self.conv7 = nn.ConvTranspose2d(196, 196, 4, 2, 1)
        self.norm7 = nn.BatchNorm2d(32)

        self.conv8 = nn.Conv2d(196, 3, 3, 1, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.view(4, 4, 196)
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = F.relu(self.norm4(self.conv4(x)))
        x = F.relu(self.norm5(self.conv5(x)))
        x = F.relu(self.norm6(self.conv6(x)))
        x = F.relu(self.norm7(self.conv7(x)))
        x = F.tanh(self.conv8(x))

        return x

# Data augmentation
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
        brightness=0.1*torch.randn(1),
        contrast=0.1*torch.randn(1),
        saturation=0.1*torch.randn(1),
        hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

# Data loading
trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = discriminator()
model.to(device)
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(num_epochs):
    if(epoch==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0
    if(epoch==75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/100.0

    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

        if(Y_train_batch.shape[0] < batch_size):
                continue

        X_train_batch = Variable(X_train_batch).to(device)
        Y_train_batch = Variable(Y_train_batch).to(device)
        output = model(X_train_batch)

        loss = criterion(output[0], Y_train_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000

torch.save(model, 'cifar10.model')
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for X_test_batch, Y_test_batch in trainloader:
        X_test_batch = X_test_batch.to(device)
        Y_test_batch = Y_test_batch.to(device)
        output = model(X_test_batch)
        _, predicted = torch.max(output[0].data, 1)
        total += Y_test_batch.size(0)
        correct += (predicted == Y_test_batch).sum().item()

        break

    print('Test Accuracy: ' + str((correct/total) * 100) + '%')
