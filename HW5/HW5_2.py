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
    transforms.Resize(DIM, interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Load Training Data
trainset = torchvision.datasets.CIFAR100(root='/projects/training/bawc/CIFAR100/Dataset', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

# Load Testing Data
testset = torchvision.datasets.CIFAR100(root='/projects/training/bawc/CIFAR100/Dataset', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

# CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load pretrained model
model = models.resnet18(pretrained=True)

# Change fully connected layer
model.fc = nn.Linear(512, 100)

# Put model on GPU/CPU
model.to(device)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define layers to optimize
params = list(model.layer4.parameters()) + list(model.fc.parameters())

# Define ADAM as optimizer
optimizer = torch.optim.Adam(params, lr=learning_rate)

# Record start time
start_time = time.time()

# List for loss and accuracy
loss = []
accuracy = []

# Start training
model.train()

# Train the model
for epoch in range(num_epochs):
    # Learning rate schedule
    if epoch == 11:
        learning_rate /= 10

    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
        if(Y_train_batch.shape[0]<batch_size):
            continue

        # Load data and target to GPU
        X_train_batch = X_train_batch.to(device)
        Y_train_batch = Y_train_batch.to(device)

        # Don't calculate gradient of lower layers
        with torch.no_grad():
            h = model.conv1(X_train_batch)
            h = model.layer3(model.layer2(model.layer1(model.maxpool(model.relu(model.bn1(h))))))
        
        # Store gradient for 4th and fully connected layers
        h = model.layer4(h)
        h = model.avgpool(h)
        h = h.view(h.size(0), -1)

        # Calculate forward pass
        output = model.fc(h)

        # Calculate loss
        curr_loss = criterion(output, Y_train_batch)
        loss.append(curr_loss.item())

        # Calculate gradient and take step
        optimizer.zero_grad()
        curr_loss.backward()
        optimizer.step()

        # Keep track of accuracy
        predicted = F.softmax(output, dim = 1)
        predicted = predicted.data.max(1)[1]
        acc = float(predicted.eq(Y_train_batch.data).sum())
        accuracy.append(acc/Y_train_batch.size(0))

        # Every 100 iterations, print stats
        if batch_idx % 100 == 0:
            print('Epoch: ' + str(epoch+1) + '/' + str(num_epochs) + ', Step: ' + str(batch_idx+1) + '/' + str(len(trainloader)) + ', Loss: ' + str(curr_loss.item()) + ', Accuracy: ' + str(acc/Y_train_batch.size(0)*100) + '%')
        
        # Clear variables for ADAM in bluewaters
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000

# Test model
total_acc = 0.0
total_loss = 0.0
total_num = 0
batch_size = 0

# Put model in eval mode
model.eval()
for X_test_batch, Y_test_batch in testloader:
    batch_size = Y_test_batch.size(0)
    with torch.no_grad():
        data, target = Variable(X_test_batch).to(device), Variable(Y_test_batch).to(device)
        out = model(data)
        loss = criterion(out, target)
        pred = F.softmax(out, dim = 1)
        pred = pred.data.max(1)[1]
        total_acc += float(pred.eq(target.data).sum())
        total_loss += loss.item()
        total_num += batch_size

# Calculate total accuracy and loss
total_acc /= total_num
total_loss /= (total_num/batch_size)

# Store end and elapsed time
end_time = time.time()
elapsed = end_time - start_time

# Print test results
print("TEST ACCURACY:  " + str(total_acc*100.0) + "%\nTEST LOSS: " + str(total_loss))
print("TIME: " + str(elapsed/60))
print("\n\n\n" + str(total_acc*total_num) + " " + str(total_num))
