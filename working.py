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
lr = 1e-9
batchSize = 500
model = socialSig.SocialSigNet(X=X, outDim = batchSize)
epochs = 30

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr = lr)


for t in range(epochs):
    for k in range(math.ceil(len(y)/batchSize)):
        # Prep the batch for a forward pass
        batchObs = random.sample(range(0, len(y)), batchSize)
        #batchObs = [i for i in range(0, batchSize)]
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
        

        print("Epoch: " + str(t) + " Batch: " + str(k))
        print("    Loss:     ", loss.item(), "     MAE: ", mae(y_pred, modely).item())
        print("\n")

        if loss.item() < 0:
            break





def get_final_columns(new_val, devSet):
    print("\n")
    indices = list(torch.clamp(torch.tensor(new_val, dtype = torch.int64), 0, len(new_val)).detach().numpy())
    print("Columns kept: ", list(devSet.columns[list(set(indices))]))
    droppped_indices = [i for i in range(0, len(devSet.columns)) if i not in indices]
    print("\n")
    print("Columns dropped: ", list(devSet.columns[droppped_indices]))
    print("\n")
    # print(indices)
    dup_indicies = list(set([i for i in indices if indices.count(i)>1]))
    print("Duplicated columns: ", list(devSet.columns[dup_indicies]))
    print("\n")


get_final_columns(new_val, devSet.loc[:, devSet.columns != "US_MIG_05_10"])