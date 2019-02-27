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

# Hyper-parameters
num_epochs = 1
learning_rate = 0.01
batch_size = 128
DIM = 32
no_of_hidden_units = 196

# Define Model
class cifarModel(nn.Module):
    def __init__(self): 
        super(cifarModel, self).__init__()
        # (w - k + 2p)/s + 1
        self.conv1 = nn.Conv2d(3, 64, 4, 1, 2) # (32 - 4 + 4)/1 + 1 = 33
        self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 4, 1, 2) # (32 - 4 + 4)/1 + 1 = 34
        
        self.pool = nn.MaxPool2d(2, 2) # (32 - 0 + 0)/2 = 17
        self.drop1 = nn.Dropout(0.5)
        
        self.conv3 = nn.Conv2d(64, 64, 4, 1, 2) # (17 - 4 + 4)/1 + 1 = 18
        self.norm2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 5, 1, 2) # (18 - 5 + 4)/1 + 1 = 18
        
        # (self.pool1) (18 - 0 + 0)/2 = 9
        self.drop2 = nn.Dropout(0.5)

        self.conv5 = nn.Conv2d(64, 64, 4, 1, 2) # (9 - 4 + 4)/1 + 1 = 10
        self.norm3 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3, 1) # (10 - 3 + 0)/1 + 1 = 8

        self.drop3 = nn.Dropout(0.5)

        self.conv7 = nn.Conv2d(64, 64, 3, 1) # (8 - 3 + 0)/1 + 1 = 6
        self.norm4 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 64, 3, 1) # (8 - 3 + 0)/1 + 1 = 4

        self.norm5 = nn.BatchNorm2d(64)
        self.drop4 = nn.Dropout(0.5)

        self.full1 = nn.Linear(64 * 4 * 4, 500)
        self.full2 = nn.Linear(500, 250)
        self.full3 = nn.Linear(250, 10)

    def forward(self, x):
        x = F.relu(self.conv2(self.norm1(F.relu(self.conv1(x)))))
        x = self.drop1(self.pool(x))

        x = F.relu(self.conv4(self.norm2(F.relu(self.conv3(x)))))
        x = self.drop2(self.pool(x))

        x = self.drop3(F.relu(self.conv6(self.norm3(F.relu(self.conv5(x))))))

        x = F.relu(self.conv8(self.norm4(F.relu(self.conv7(x)))))

        x = self.drop4(self.norm5(x))

        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.full1(x))
        x = F.relu(self.full2(x))
        x = self.full3(x)
        return x

# Data Augmentation
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(DIM, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Test Set Standardization
transform_test = transforms.Compose([
    transforms.CenterCrop(DIM),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load Training Data
trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

# Load Testing Data
testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

use_gpu = torch.cuda.is_available()

model = cifarModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_time = time.time()

loss = []
accuracy = []

model.train()

# Train the model
for epoch in range(num_epochs):
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
        if(Y_train_batch.shape[0]<batch_size):
            continue
        
        #if use_gpu:
        #    X_train_batch = Variable(X_train_batch).cuda()
        #    Y_train_batch = Variable(Y_train_batch).cuda()
        
        output = model(X_train_batch)
        curr_loss = criterion(output, Y_train_batch)
        loss.append(curr_loss.item())

        optimizer.zero_grad()
        curr_loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        correct = (predicted == Y_train_batch).sum().item()
        accuracy.append(correct/Y_train_batch.size(0))
        if batch_idx % 100 == 0:
            print('Epoch: ' + str(epoch+1) + '/' + str(num_epochs) + ', Step: ' + str(batch_idx+1) + '/' + str(len(trainloader)) + ', Loss: ' + str(curr_loss.item()) + ', Accuracy: ' + str(correct/Y_train_batch.size(0)*100) + '%')

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for X_test_batch, Y_test_batch in trainloader:
        output = model(X_test_batch)
        _, predicted = torch.max(output.data, 1)
        total += Y_test_batch.size(0)
        correct += (predicted == Y_test_batch).sum().item()

    print('Test Accuracy: ' + str((correct/total) * 100) + '%')
