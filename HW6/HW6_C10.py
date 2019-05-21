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
batch_size = 128
learning_rate = 0.0001

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

    def forward(self, x):
        x = F.leaky_relu(self.norm1(self.conv1(x)))
        x = F.leaky_relu(self.norm2(self.conv2(x)))
        x = F.leaky_relu(self.norm3(self.conv3(x)))
        x = F.leaky_relu(self.norm4(self.conv4(x)))
        x = F.leaky_relu(self.norm5(self.conv5(x)))
        x = F.leaky_relu(self.norm6(self.conv6(x)))
        x = F.leaky_relu(self.norm7(self.conv7(x)))
        x = self.pool1(F.leaky_relu(self.norm8(self.conv8(x))))
        x = x.view(-1, 128)
        
        return [self.fc1(x), self.fc10(x)]

# Define Generator
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(100, 128*4*4)

        self.conv1 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv5 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.conv6 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv7 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.conv8 = nn.Conv2d(128, 3, 3, 1, 1)

        self.norm1 = nn.BatchNorm2d(8)
        self.norm2 = nn.BatchNorm2d(8)
        self.norm3 = nn.BatchNorm2d(8)
        self.norm4 = nn.BatchNorm2d(8)
        self.norm5 = nn.BatchNorm2d(16)
        self.norm6 = nn.BatchNorm2d(16)
        self.norm7 = nn.BatchNorm2d(32)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 128, 4, 4)
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        x = F.relu(self.norm4(self.conv4(x)))
        x = F.relu(self.norm5(self.conv5(x)))
        x = F.relu(self.norm6(self.conv6(x)))
        x = F.relu(self.norm7(self.conv7(x)))
        x = self.tanh(self.conv8(x))

        return x

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

model =  discriminator()
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# List of loss and accuracy
loss_t = []
accuracy = []

# Train the model
for epoch in range(0,num_epochs):
    if(epoch==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0
    if(epoch==75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/100.0
    model.train()
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

        if(Y_train_batch.shape[0] < batch_size):
            continue

        X_train_batch = Variable(X_train_batch).cuda()
        Y_train_batch = Variable(Y_train_batch).cuda()
        output = model(X_train_batch)
        output = output[1]

        loss = criterion(output, Y_train_batch)
        loss_t.append(loss.item())
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # Calculate accuracy
        predicted = F.softmax(output, dim = 1)
        predicted = predicted.data.max(1)[1]
        acc = float(predicted.eq(Y_train_batch.data).sum())
        accuracy.append(acc/Y_train_batch.size(0))
        
        # Print training stats
        if batch_idx % 100 == 0:
            print('Epoch: ' + str(epoch+1) + '/' + str(num_epochs) + ', Step: ' + str(batch_idx+1) + '/' + str(len(trainloader)) + ', Loss: ' + str(loss.item()) + ', Accuracy: ' + str(acc/Y_train_batch.size(0)*100) + '%')
           
 
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000

torch.save(model,'cifar10.model')

model.eval()
for X_test_batch, Y_test_batch in testloader:
    batch_size = Y_test_batch.size(0)
    with torch.no_grad():
        data, target = Variable(X_test_batch).cuda(), Variable(Y_test_batch).cuda()
        out = model(data)
        out = out[1]
        loss = criterion(out, target)
        pred = F.softmax(out, dim = 1)
        pred = pred.data.max(1)[1]
        total_acc += float(pred.eq(target.data).sum())
        total_loss += loss.item()
        total_num += batch_size

# Calculate total loss and accuracy
total_acc /= total_num
total_loss /= (total_num/batch_size)

# Store end and elapsed time
end_time = time.time()
elapsed = end_time - start_time

# Print test stats
print("TEST ACCURACY:  " + str(total_acc*100.0) + "%\nTEST LOSS: " + str(total_loss))
print("TIME: " + str(elapsed/60))
