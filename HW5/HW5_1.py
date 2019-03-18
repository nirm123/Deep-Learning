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
num_epochs = 20
num_epochs = 0
learning_rate = 0.001#0.0001#/10.0
batch_size = 128
DIM = 32
no_of_hidden_units = 196


# Define basic block for residual network
class basic_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(basic_block, self).__init__()
        self.down = nn.Conv2d(in_channel, out_channel, 1, 2)

        self.down_flag = False
        cur_stride = 1
        if in_channel != out_channel:
            self.down_flag = True
            cur_stride = 2

        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, cur_stride, 1)
        self.norm1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        orig_x = x
        if self.down_flag:
            orig_x = self.down(orig_x)

        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))

        x += orig_x

        x = F.relu(x)

        return x


# Define Model
class resNet_cifar100(nn.Module):
    def __init__(self):
        super(resNet_cifar100, self).__init__()
        # (w - k + 2p)/s + 1
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1) # (32 - 3 + 2)/1 + 1 = 32
        self.norm1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout(0.5)

        self.conv2_1 = basic_block(32, 32)
        self.conv2_2 = basic_block(32, 32)

        self.conv3_1 = basic_block(32, 64)
        self.conv3_2 = basic_block(64, 64)
        self.conv3_3 = basic_block(64, 64)
        self.conv3_4 = basic_block(64, 64)

        self.conv4_1 = basic_block(64, 128)
        self.conv4_2 = basic_block(128, 128)
        self.conv4_3 = basic_block(128, 128)
        self.conv4_4 = basic_block(128, 128)

        self.conv5_1 = basic_block(128, 256)
        self.conv5_2 = basic_block(256, 256)

        self.pool = nn.MaxPool2d(4, 4)
        self.full = nn.Linear(256, 100)

    def forward(self, x):
        x = self.drop1(F.relu(self.norm1(self.conv1(x))))
        x = self.conv2_2(self.conv2_1(x))
        x = self.conv3_4(self.conv3_3(self.conv3_2(self.conv3_1(x))))
        x = self.conv4_4(self.conv4_3(self.conv4_2(self.conv4_1(x))))
        x = self.conv5_2(self.conv5_1(x))
        x = self.pool(x)
        x = x.view(-1, 256)
        x = self.full(x)
        return x


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

transform_test = transforms.Compose([
    transforms.CenterCrop(DIM),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR100(root='/projects/training/bawc/CIFAR100/Dataset', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)


testset = torchvision.datasets.CIFAR100(root='/projects/training/bawc/CIFAR100/Dataset', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resNet_cifar100()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

start_time = time.time()

loss = []
accuracy = []

model.train()

# Train the model
for epoch in range(0,num_epochs):
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

        if(Y_train_batch.shape[0]<batch_size):
            continue

        X_train_batch = Variable(X_train_batch).to(device)
        Y_train_batch = Variable(Y_train_batch).to(device)

        output = model(X_train_batch)
        curr_loss = criterion(output, Y_train_batch)
        loss.append(curr_loss.item())

        optimizer.zero_grad()
        curr_loss.backward()
        optimizer.step()

        predicted = F.softmax(output, dim = 1)
        predicted = predicted.data.max(1)[1]
        acc = float(pred.eq(Y_train_batch.data).sum())
        #los = curr_loss.item()
        #_, predicted = torch.max(output.data, 1)
        #correct = (predicted == Y_train_batch).sum().item()
        #accuracy.append(correct/Y_train_batch.size(0))
        if batch_idx % 100 == 0:
            print('Epoch: ' + str(epoch+1) + '/' + str(num_epochs) + ', Step: ' + str(batch_idx+1) + '/' + str(len(trainloader)) + ', Loss: ' + str(curr_loss.item()) + ', Accuracy: ' + str(acc/Y_train_batch.size(0)*100) + '%')
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
total_acc /= total_num
total_loss /= (total_num/batch_size)

end_time = time.time()
elapsed = end_time - start_time


print("TEST ACCURACY:  " + str(total_acc*100.0) + "%\nTEST LOSS: " + str(total_loss))
print("TIME: " + str(elapsed/60))

