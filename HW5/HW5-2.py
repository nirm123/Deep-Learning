import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torch.autograd import Variable

import math
import os
import time
import numpy as np

# Hyper-parameters
num_epochs = 20
learning_rate = 0.001
batch_size = 128
DIM = 224
no_of_hidden_units = 196

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
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Test Set Standardization
transform_test = transforms.Compose([
    transforms.CenterCrop(DIM),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Load Training Data
trainset = torchvision.datasets.CIFAR100(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

# Load Testing Data
testset = torchvision.datasets.CIFAR100(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 100)

model.to(device)
criterion = nn.CrossEntropyLoss()

params = list(model.layer4.parameters()) + list(model.fc.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)

start_time = time.time()

loss = []
accuracy = []

model.train()

# Train the model
for epoch in range(num_epochs):
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
        if(Y_train_batch.shape[0]<batch_size):
            continue

        X_train_batch = X_train_batch.to(device)
        Y_train_batch = Y_train_batch.to(device)

        with torch.no_grad():
            h = model.conv1(X_train_batch)
            h = model.layer3(model.layer2(model.layer1(model.maxpool(model.relu(model.bn1(h))))))
        h = model.layer4(h)
        h = model.avgpool(h)
        h = h.view(h.size(0), -1)
        output = model.fc(h)

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
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for X_test_batch, Y_test_batch in trainloader:
        X_test_batch = X_test_batch.to(device)
        Y_test_batch = Y_test_batch.to(device)
        output = model(X_test_batch)
        _, predicted = torch.max(output.data, 1)
        total += Y_test_batch.size(0)
        correct += (predicted == Y_test_batch).sum().item()

    print('Test Accuracy: ' + str((correct/total) * 100) + '%')

