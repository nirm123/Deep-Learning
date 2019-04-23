import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

## Define bag of words model
class BOW_model(nn.Module):
    def __init__(self, vocab_size, no_of_hidden_units):
        super(BOW_model, self).__init__()
       
        ## Linear layer
        self.fc_hidden = nn.Linear(300, no_of_hidden_units)

        ## Additional fc
        self.fc_2 = nn.Linear(no_of_hidden_units,no_of_hidden_units)

        ## Additional fc
        self.fc_3 = nn.Linear(no_of_hidden_units,no_of_hidden_units)

        ## Batch normalization
        self.bn_hidden = nn.BatchNorm1d(no_of_hidden_units)
        
        ## Additional Batch normalization
        self.bn_2 = nn.BatchNorm1d(no_of_hidden_units)

        ## Dropout
        self.dropout = torch.nn.Dropout(p=0.5)
        
        ## Additional Dropout
        self.dropout2 = torch.nn.Dropout(p=0.5)

        ## Final output
        self.fc_output = nn.Linear(no_of_hidden_units, 1)
        
        ## Loss function
        self.loss = nn.BCEWithLogitsLoss()
 
    def forward(self, x, t):
        ## Pass through single layer neural network
        #h = self.dropout(F.relu(self.bn_hidden(self.fc_hidden(x))))
        h = F.relu(self.fc_hidden(x))
        
        ## Pass through single layer neural network
        #h = self.dropout2(F.relu(self.bn_2(self.fc_2(h))))
        h = F.relu(self.fc_2(h))
        
        ## Pass through single layer neural network
        h = F.relu(self.fc_3(h))
        
        ## Pass through layer bringing down to 1 output
        h = self.fc_output(h)

        ## Return loss and output
        return self.loss(h[:,0], t), h[:,0]
