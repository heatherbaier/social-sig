import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import numpy as np
import importlib
import sklearn
import random
import torch
import math

import socialSig
importlib.reload(socialSig)
from helpers import *




####### Load our Data
from sklearn import preprocessing
devSet = pd.read_csv("./us_migration.csv")
devSet = devSet.loc[:, ~devSet.columns.str.contains('^Unnamed')]
devSet = devSet.apply(lambda x: pd.to_numeric(x, errors='coerce'))
devSet = devSet.dropna(axis=1)

y = torch.Tensor(devSet['US_MIG_05_10'].values)
X = devSet.loc[:, devSet.columns != "US_MIG_05_10"].values

mMScale = preprocessing.MinMaxScaler()
X = mMScale.fit_transform(X)


#y - 'number_moved'
#x - 'everything else that is or can be represented as a float.'


####### Build and fit the Model
lr = 1e-7
batchSize = 200
model = socialSig.SocialSigNet(X=X, outDim = batchSize)



criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr = lr)


for t in range(30):

    # Prep the batch for a forward pass
    batchObs = random.sample(range(0, len(X)), batchSize)
    # batchObs = [i for i in range(0, batchSize)]
    modelX = X[batchObs]
    modelX = torch.tensor(list(modelX), requires_grad = True, dtype = torch.float32)
    modely = torch.tensor(y[batchObs], dtype = torch.float32)

    # Forward pass
    y_pred = model(modelX, t)
    loss = criterion(y_pred, modely)  
    
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    grad = torch.autograd.grad(outputs=loss, inputs=modelX, retain_graph = True)
    loss.backward()
    optimizer.step()

    # Update the coordinate weights
    # https://discuss.pytorch.org/t/updatation-of-parameters-without-using-optimizer-step/34244/4
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name == 'SocialSig.W':
                new_val = update_function(p, grad[0], loss, lr)
                p.copy_(new_val)
    

    print("EPOCH: ", t)
    print("    Loss:     ", loss.item(), "     MAE: ", mae(y_pred, modely).item())
    print("\n")

    if loss.item() < 0:
        break





