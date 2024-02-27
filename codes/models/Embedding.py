import math
import torch
import collections
import numpy as np
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from torch.autograd import Variable

##----------------------------------Embedding模块--------------------------------------##
class CNN_Embedding(nn.Module):
    def __init__(self):
        super(CNN_Embedding, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 22), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 11))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 1, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))



    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0, 1, 3, 2)
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.3)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.3)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = torch.squeeze(x, dim=1)
        return x,x.shape[-1]

if __name__ == "__main__":
    input = torch.randn(200,22,1000)
    model = CNN_Embedding()
    out,fe = model(input)