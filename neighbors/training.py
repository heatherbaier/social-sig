from sklearn import preprocessing
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import numpy as np
import importlib
import sklearn
import random
import torch
import math
import os

import socialSigN
importlib.reload(socialSigN)
from helpers import *


import warnings
warnings.filterwarnings('ignore')


####### Load our Data
#y - 'number_moved'
#x - 'everything else that is or can be represented as a float.'
devSet = pd.read_csv("../us_migration.csv")
devSet = devSet.loc[:, ~devSet.columns.str.contains('^Unnamed')]
devSet = devSet.apply(lambda x: pd.to_numeric(x, errors='coerce'))
devSet = devSet.dropna(axis=1)
# devSet = devSet.drop(['sending'], axis = 1)

y = torch.Tensor(devSet['US_MIG_05_10'].values)
X = devSet.loc[:, devSet.columns != "US_MIG_05_10"].values

mMScale = preprocessing.MinMaxScaler()
X = mMScale.fit_transform(X)


xy = devSet[['sending', 'US_MIG_05_10']]
sending = devSet['sending'].to_list()

avail_sending = []
for i in os.listdir("./inputs/"):
    avail_sending.append(i.split(".")[0])

avail_sending = [i for i in avail_sending if i not in ['nan', '']]
avail_sending = [i for i in avail_sending if int(i) in sending]




####### Build and fit the Model
lr = 1e-5
batchSize = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = socialSigN.SocialSigNet(X=torch.reshape(torch.tensor(read_file(1001), dtype = torch.float32, requires_grad = True), (1, 243)), 
                                outDim = batchSize).to(device)
epochs = 50

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr = lr)


x_train, y_train, x_val, y_val = train_test_split(avail_sending, .80, xy)


train = [(k,v) for k,v in zip(x_train, y_train)]
val = [(k,v) for k,v in zip(x_val, y_val)]

train = torch.utils.data.DataLoader(train, batch_size = batchSize, shuffle = True)
val = torch.utils.data.DataLoader(val, batch_size = batchSize, shuffle = True)



best_mae = 9000000000000000000
best_model_wts = deepcopy(model.state_dict())


print(device)



for epoch in range(epochs):

    for phase in ['train','val']:

        if phase == 'train':

            c = 1
            running_train_mae, running_train_loss = 0, 0

            for inputs, output in train:

                if len(inputs) == batchSize:
                    
                    inputs = inputs.to(device)
                    output = output.to(device)

                    # print("Epoch: ", epoch)

                    # Forward pass
                    y_pred = model(inputs, str(epoch))
                    loss = criterion(y_pred, output)  

                    # print("    Loss: ", loss)
                    # print("    MAE: ", mae(y_pred, output).item())
                    # print("    Predictions: ", y_pred)
                    # print("    True: ", output)

                    # Zero gradients, perform a backward pass, and update the weights.
                    optimizer.zero_grad()
                    grad = torch.autograd.grad(outputs = loss, inputs = inputs, retain_graph = True)
                    # print(grad)
                    loss.backward()
                    optimizer.step()

                    # Update the coordinate weights
                    # https://discuss.pytorch.org/t/updatation-of-parameters-without-using-optimizer-step/34244/4
                    with torch.no_grad():
                        for name, p in model.named_parameters():
                            if name == 'SocialSig.W':
                                new_val = update_function(p, grad[0], loss, lr)
                                p.copy_(new_val)


                    running_train_mae += mae(y_pred, output).item()
                    running_train_loss += loss.item()

                    c += 1

        if phase == 'val':

            d = 1
            running_val_mae, running_val_loss,  = 0, 0

            # print("In validation")

            for inputs, output in val:

                if len(inputs) == batchSize:

                    inputs = inputs.to(device)
                    output = output.to(device)

                    # Forward pass
                    y_pred = model(inputs, 4444)
                    loss = criterion(y_pred, output)  

                    running_val_mae += mae(y_pred, output).item()
                    running_val_loss += loss.item()
                    d += 1
                    
                    if mae(y_pred, output).item() < best_mae:
                        best_mae = mae(y_pred, output).item()
                        best_model_wts = deepcopy(model.state_dict())

            


    print("\n")

    print("Epoch: ", epoch)  
    print("  Train:")
    print("    Loss: ", running_train_loss / c)      
    print("    MAE: ", running_train_mae / c)
    print("  Val:")
    print("    Loss: ", running_val_loss / d)      
    print("    MAE: ", running_val_mae / d)