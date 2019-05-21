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

from RNN_model import RNN_model

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
model = torch.load('RNN3.model')
model.cuda()

## Hyper-parameters
batch_size = 200
no_of_epochs = 10

## Length of train/test set
L_Y_train = len(y_train)
L_Y_test = len(y_test)

## Training Loop
for epoch in range(no_of_epochs):
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

    ## Sequence Length
    sequence_length = (epoch+1)*50

    ## Iterate through batches of test data
    for i in range(0, L_Y_test, batch_size):
        ## Select corresponding input and output from random permutation
        x_input2 = [x_test[j] for j in I_permutation[i:i+batch_size]]
        x_input = np.zeros((batch_size, sequence_length),dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if sl < sequence_length:
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl - sequence_length + 1)
                x_input[j,:] = x[start_index:(start_index + sequence_length)]
        y_input = y_test[I_permutation[i:i+batch_size]]

        ## Move output to GPU
        data = Variable(torch.LongTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        ## Forward step without gradient calculation
        with torch.no_grad():
            loss, pred = model(data, target, train=False)
        
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

    ## Calculate test set time
    time2 = time.time()
    time_elapsed = time2 - time1

    ## Print stats
    print(str(sequence_length) + " " + str(epoch_acc*100.0) + " " + str(epoch_loss))
