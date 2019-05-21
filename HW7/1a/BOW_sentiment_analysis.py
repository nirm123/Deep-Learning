import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

import time
import os
import sys
import io

from BOW_model import BOW_model

#imdb_dictionary = np.load('../preprocessed_data/imdb_dictionary.npy')

## Number of unique words we are considering
vocab_size = 8000

## List of training samples
x_train = []

## Open preprocessed training data
with io.open('../preprocessed_data/imdb_train.txt','r',encoding='utf-8') as f:
    lines = f.readlines()

## Additional processing for each sample
for line in lines:
    ## Convert sentence of token ids into list
    line = line.strip()
    line = line.split(' ')

    ## Convert list into numpy array
    line = np.asarray(line,dtype=np.int)

    ## Tokens greater than vocab size forced to unknown
    line[line>vocab_size] = 0

    ## Append sample to training list
    x_train.append(line)

## Only care about first 25000 labled samples
x_train = x_train[0:25000]

## Correct classification (first half positive, second half negative
y_train = np.zeros((25000,))
y_train[0:12500] = 1

## List of testing samples
x_test = []

## Open preprocessed testing data
with io.open('../preprocessed_data/imdb_test.txt','r',encoding='utf-8') as f:
    lines = f.readlines()

## Additional processing for each sample
for line in lines:
    ## Convert sentence of token ids into list
    line = line.strip()
    line = line.split(' ')

    ## Convert list into numpy array
    line = np.asarray(line,dtype=np.int)

    ## Tokens greater than vocab size forced to unknown
    line[line>vocab_size] = 0

    ## Append sample to training list
    x_test.append(line)

## Correct classification (first half positive, second half negative
y_test = np.zeros((25000,))
y_test[0:12500] = 1

## Allow 8000 words + 1 unknown
vocab_size += 1

## Define model with 500 hidden units
model = BOW_model(vocab_size,600)
model.cuda()

## Define optimizer
# opt = 'sgd'
# LR = 0.01
opt = 'adam'
LR = 0.001
if(opt=='adam'):
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif(opt=='sgd'):
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

## Hyper-parameters
batch_size = 200
no_of_epochs = 6

## Length of train/test set
L_Y_train = len(y_train)
L_Y_test = len(y_test)

## Put model in training mode
model.train()

## Lists to keep track of training stats
train_loss = []
train_accu = []
test_accu = []

## Training Loop
for epoch in range(no_of_epochs):
    ## Put model in training mode
    model.train()

    ## Current epoch stats
    epoch_acc = 0.0
    epoch_loss = 0.0
    epoch_counter = 0

    ## Start time
    time1 = time.time()
    
    ## Randomly order training sample access
    I_permutation = np.random.permutation(L_Y_train)

    ## Run through data
    for i in range(0, L_Y_train, batch_size):
        ## Select corresponding input and output from random permutation
        x_input = [x_train[j] for j in I_permutation[i:i+batch_size]]
        y_input = np.asarray([y_train[j] for j in I_permutation[i:i+batch_size]],dtype=np.int)
        
        ## Move output to GPU
        target = Variable(torch.FloatTensor(y_input)).cuda()

        ## Zero out optimizer
        optimizer.zero_grad()

        ## Forward step
        loss, pred = model(x_input,target)
     
        ## Calculate backpropogation
        loss.backward()

        ## Update weights
        optimizer.step() 

        ## Convert prediction from real to binary        
        prediction = pred >= 0.0

        ## Convert expected output to binary 
        truth = target >= 0.5

        ## Number of correct predictions
        acc = prediction.eq(truth).sum().cpu().data.numpy()

        ## Update running epoch stats
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

    ## Calculate full epoch stats
    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    ## Store stats
    train_loss.append(epoch_loss)
    train_accu.append(epoch_acc)

    ## Print stats
    print(epoch, "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss, "%.4f" % float(time.time()-time1))

    ## Test mid training
    model.eval()

    ## Epoch test set stats
    epoch_acc = 0.0
    epoch_loss = 0.0
    epoch_counter = 0

    ## Start test
    time1 = time.time()
    
    ## Random permutation of test data
    I_permutation = np.random.permutation(L_Y_test)

    ## Iterate through batches of test data
    for i in range(0, L_Y_test, batch_size):
        ## Get corrext input/output test data
        x_input = [x_test[j] for j in I_permutation[i:i+batch_size]]
        y_input = np.asarray([y_test[j] for j in I_permutation[i:i+batch_size]],dtype=np.int)

        ## Move output to GPU
        target = Variable(torch.FloatTensor(y_input)).cuda()

        ## Forward step without gradient calculation
        with torch.no_grad():
            loss, pred = model(x_input,target)
        
        ## Convert predicted real output into binary
        prediction = pred >= 0.0

        ## Convert expected output to binary
        truth = target >= 0.5

        ## Calculate number of correct predictions
        acc = prediction.eq(truth).sum().cpu().data.numpy()

        ## Calculate running test set stats
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

    ## Calculate full test set stats
    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    ## Store accuracy
    test_accu.append(epoch_acc)

    ## Calculate test set time
    time2 = time.time()
    time_elapsed = time2 - time1

    ## Print stats
    print("  ", "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss)

## Save model and output
torch.save(model,'BOW4.model')
data = [train_loss,train_accu,test_accu]
data = np.asarray(data)
np.save('data4.npy',data)
