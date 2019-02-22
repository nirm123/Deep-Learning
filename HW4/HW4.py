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
num_epochs = 100
learning_rate = 0.0001
batch_size = 128
DIM = 32
no_of_hidden_units = 196

# Define Model
class cifarModel(nn.module):
    def __init__(self):
        super(cifarModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 1, 2)
        self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 4, 1, 2)
        
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv2d(3, 64, 4, 1, 2)
        self.norm2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4, 1, 2)
        
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(0.2)

        self.conv5 = nn.Conv2d(3, 64, 4, 1, 2)
        self.norm3 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3, 1)

        self.drop3 = nn.Dropout(0.2)

        self.conv7 = nn.Conv2d(64, 64, 3, 1)
        self.norm4 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 64, 3, 1)

        self.norm5 = nn.BatchNorm2d(64)
        self.drop4 = nn.Dropout(0.2)

        self.full1 = nn.Linear(

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


start_time = time.time()

# Train the model
for epoch in range(0,num_epochs):


    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

        if(Y_train_batch.shape[0]<batch_size):
            continue

        X_train_batch = Variable(X_train_batch).cuda()
        Y_train_batch = Variable(Y_train_batch).cuda()

