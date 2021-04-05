import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import numpy as np
import importlib
import sklearn
import random
import torch
import math

import helpers
importlib.reload(helpers)

from helpers import *


class bilinearImputation(torch.nn.Module):
    '''
    Class to create the social signature image
    '''
    def __init__(self, X):
        super(bilinearImputation, self).__init__()
        self.W = torch.nn.Parameter(torch.tensor(np.arange(0, X.shape[1]), dtype = torch.float32, requires_grad=True))
        self.outDim = [224,224]
        self.inDim = math.ceil(math.sqrt(X.shape[1]))

    def forward(self, batchX):
        # print("    W at beginning: ", torch.tensor(self.W, dtype = torch.int)) 
        taken = torch.take(batchX, construct_indices(torch.clamp(torch.tensor(self.W, dtype = torch.int64), 0, 29), batchX.shape[0], self.W.shape[0])).cuda()
        batchX.data = batchX.data.copy_(taken.data)        
        inDataSize = self.W.shape[0] #Data we have per dimension
        targetSize = self.inDim ** 2
        paddingOffset = targetSize - inDataSize
        paddedInX = torch.nn.functional.pad(input=batchX, pad=(0,paddingOffset), mode="constant", value=0)
        buildImage = torch.reshape(paddedInX,(batchX.shape[0], 1, self.inDim, self.inDim))   
        return torch.nn.functional.interpolate(buildImage, size=([self.outDim[0], self.outDim[1]]), mode='bilinear')


class bilinearImputationNoDrop(torch.nn.Module):
    '''
    Class to create the social signature image
    '''
    def __init__(self, X):
        super(bilinearImputationNoDrop, self).__init__()
        self.W = torch.nn.Parameter(torch.tensor(np.arange(0, X.shape[1])*.001, dtype = torch.float32, requires_grad=True))
        # self.outDim = [10,10]
        self.outDim = [224,224]
        self.inDim = math.ceil(math.sqrt(X.shape[1]))

    def forward(self, batchX):
        # print("    W at beginning: ", torch.tensor(self.W)) 
        taken = torch.take(batchX, construct_noOverlap_indices(torch.tensor(self.W, dtype = torch.float32), batchX.shape[0], self.W.shape[0])).cuda()
        batchX.data = batchX.data.copy_(taken.data)   

        inDataSize = self.W.shape[0] #Data we have per dimension
        targetSize = self.inDim ** 2
        paddingOffset = targetSize - inDataSize
        paddedInX = torch.nn.functional.pad(input=batchX, pad=(0,paddingOffset), mode="constant", value=0)
        buildImage = torch.reshape(paddedInX,(batchX.shape[0], 1, self.inDim, self.inDim))   

        return torch.nn.functional.interpolate(buildImage, size=([self.outDim[0], self.outDim[1]]), mode='bilinear')


class scoialSigNet_NoDrop(torch.nn.Module):
    '''
    SocialSigNet
    Mocks the ResNet101_32x8d architecture
    '''
    def __init__(self, X, outDim, resnet):
        super().__init__()
        self.SocialSig = bilinearImputationNoDrop(X=X)      
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.linear = torch.nn.Linear(in_features=2048, out_features=1, bias = True)

    def forward(self, X, epoch):

        out = self.SocialSig(X)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.flatten(start_dim=1)
        out = self.linear(out)

        return out

        