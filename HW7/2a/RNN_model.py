import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

## Class that stores internal cell and hidden state
class StatefulLSTM(nn.Module):
    def __init__(self,in_size,out_size):
        super(StatefulLSTM,self).__init__()
        
        ## Contains LSTM weights 
        self.lstm = nn.LSTMCell(in_size,out_size)

        ## Keep track of ouput size
        self.out_size = out_size
        
        ## Store state
        self.h = None
        self.c = None

    def reset_state(self):
        ## Reset internal state
        self.h = None
        self.c = None

    def forward(self,x):
        ## Determine batch size based on input
        batch_size = x.data.size()[0]

        ## If no stored state, zero out internal state
        if self.h is None:
            state_size = [batch_size, self.out_size]
            self.c = Variable(torch.zeros(state_size)).cuda()
            self.h = Variable(torch.zeros(state_size)).cuda()

        ## Compute forward step through LSTM unit and store state
        self.h, self.c = self.lstm(x,(self.h,self.c))

        ## Return hidden state
        return self.h

## Class to deal with dropout
class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout,self).__init__()

        ## Variable storing mask
        self.m = None

    def reset_state(self):
        ## Reset mask
        self.m = None

    def forward(self, x, dropout=0.5, train=True):
        ## If we are not training, there is no mask
        if train==False:
            return x

        ## We are training, but the mask has not been set
        if(self.m is None):
            ## Generate a new mask
            self.m = x.data.new(x.size()).bernoulli_(1 - dropout)

        ## Apply mask to input
        mask = Variable(self.m, requires_grad=False) / (1 - dropout)
        return mask * x

## RNN Model definition
class RNN_model(nn.Module):
    def __init__(self,vocab_size,no_of_hidden_units):
        super(RNN_model, self).__init__()

        ## Embedding from token to hidden units
        self.embedding = nn.Embedding(vocab_size,no_of_hidden_units)#,padding_idx=0)

        ## LSTM unit
        self.lstm1 = StatefulLSTM(no_of_hidden_units,no_of_hidden_units)

        ## Batch normalization
        self.bn_lstm1= nn.BatchNorm1d(no_of_hidden_units)

        ## Dropout
        self.dropout1 = LockedDropout() #torch.nn.Dropout(p=0.5)

        #self.lstm2 = StatefulLSTM(no_of_hidden_units,no_of_hidden_units)
        #self.bn_lstm2= nn.BatchNorm1d(no_of_hidden_units)
        #self.dropout2 = LockedDropout() #torch.nn.Dropout(p=0.5)

        ## Final output layer
        self.fc_output = nn.Linear(no_of_hidden_units, 1)

        ## Loss function
        #self.loss = nn.CrossEntropyLoss()
        self.loss = nn.BCEWithLogitsLoss()

    def reset_state(self):
        ## Reset LSTM network
        self.lstm1.reset_state()
        self.dropout1.reset_state()
        #self.lstm2.reset_state()
        #self.dropout2.reset_state()

    def forward(self, x, t, train=True):
        ## Calculate embedding for given tokens
        embed = self.embedding(x) # batch_size, time_steps, features

        ## Calculate number of tokens in sentence
        no_of_timesteps = embed.shape[1]

        ## Reset lstm state
        self.reset_state()

        ## Store outputs from LSTM unit
        outputs = []

        ## Iterate through timestamps
        for i in range(no_of_timesteps):
            ## Forward step through one LSTM unit
            h = self.lstm1(embed[:,i,:])
            h = self.bn_lstm1(h)
            h = self.dropout1(h,dropout=0.5,train=train)

            #h = self.lstm2(h)
            #h = self.bn_lstm2(h)
            #h = self.dropout2(h,dropout=0.3,train=train)

            outputs.append(h)

        ## Stack all outputs into a tensor and rearrange dimensions
        outputs = torch.stack(outputs) # (time_steps,batch_size,features)
        outputs = outputs.permute(1,2,0) # (batch_size,features,time_steps)

        ## Max Pool along timesteps
        pool = nn.MaxPool1d(no_of_timesteps)
        h = pool(outputs)

        ## Collapse last dimension (only 1)
        h = h.view(h.size(0),-1)
        #h = self.dropout(h)

        ## Final output layer
        h = self.fc_output(h)

        ## Return loss and output of forward step
        return self.loss(h[:,0],t), h[:,0]#F.softmax(h, dim=1)
