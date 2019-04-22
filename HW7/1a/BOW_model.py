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
       
        ## Embedding that converts word into d dimensional vector
        self.embedding = nn.Embedding(vocab_size,no_of_hidden_units)

        ## Linear layer
        self.fc_hidden = nn.Linear(no_of_hidden_units,no_of_hidden_units)

        ## Additional fc
        self.fc_2 = nn.Linear(no_of_hidden_units,no_of_hidden_units)

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
        ## Empty array
        bow_embedding = []

        ## Calculate BOW for each element in batch
        for i in range(len(x)):
            ## Isolate single sample
            lookup = Variable(torch.LongTensor(x[i])).cuda()
       
            ## Find embedding
            embed = self.embedding(lookup)

            ## Calculate BOW
            embed = embed.mean(dim=0)

            ## Store 
            bow_embedding.append(embed)

        ## Stack all batches
        bow_embedding = torch.stack(bow_embedding)

        ## Pass through single layer neural network
        #h = self.dropout(F.relu(self.bn_hidden(self.fc_hidden(bow_embedding))))
        h = F.relu(self.fc_hidden(bow_embedding))
        
        ## Pass through single layer neural network
        #h = self.dropout2(F.relu(self.bn_2(self.fc_2(h))))
        h = F.relu(self.fc_2(h))
        
        ## Pass through layer bringing down to 1 output
        h = self.fc_output(h)

        ## Return loss and output
        return self.loss(h[:,0], t), h[:,0]
