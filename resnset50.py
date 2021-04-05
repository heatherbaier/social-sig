import torchvision.models as models
from sklearn import preprocessing
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import numpy as np
import importlib
import argparse
import sklearn
import random
import torch
import math

import socialSigNoDrop
importlib.reload(socialSigNoDrop)
from helpers import *



resnext101_32x8d = models.resnext101_32x8d(pretrained=True)


class SocialSigNet101_32x8d(torch.nn.Module):
    '''
    SocialSigNet
    Mocks the ResNet101_32x8d architecture
    '''
    def __init__(self, X, outDim, resnset):
        super().__init__()
        self.SocialSig = bilinearImputationNoDrop(X=X)      
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = resnset.bn1
        self.relu = resnset.relu
        self.maxpool = resnset.maxpool
        self.layer1 = resnset.layer1
        self.layer2 = resnset.layer2
        self.layer3 = resnset.layer3
        self.layer4 = resnset.layer4
        self.avgpool = resnset.avgpool
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




print(resnext101_32x8d)





# ####### Load our Data
# #y - 'number_moved'
# #x - 'everything else that is or can be represented as a float.'
# devSet = pd.read_csv("./us_migration.csv")
# devSet = devSet.loc[:, ~devSet.columns.str.contains('^Unnamed')]
# devSet = devSet.apply(lambda x: pd.to_numeric(x, errors='coerce'))
# devSet = devSet.dropna(axis=1)
# devSet = devSet.drop(['sending'], axis = 1)

# y = torch.Tensor(devSet['US_MIG_05_10'].values)
# X = devSet.loc[:, devSet.columns != "US_MIG_05_10"].values

# mMScale = preprocessing.MinMaxScaler()
# X = mMScale.fit_transform(X)



# ####### Build and fit the Model
# lr = 1e-6
# batchSize = 50
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = SocialSigNet50(X=X, outDim = batchSize, resnet50 = resnet50).to(device)
# epochs = 50
# criterion = torch.nn.MSELoss(reduction='sum')
# optimizer = torch.optim.SGD(model.parameters(), lr = lr)



# # Prep the training and validation sets
# x_train, y_train, x_val, y_val = train_test_split(X, y, .80)

# train = [(k,v) for k,v in zip(x_train, y_train)]
# val = [(k,v) for k,v in zip(x_val, y_val)]

# train = torch.utils.data.DataLoader(train, batch_size = batchSize, shuffle = True)
# val = torch.utils.data.DataLoader(val, batch_size = batchSize, shuffle = True)



# model_wts = train_model(model, train, val, criterion, optimizer, epochs, batchSize, device, lr)

