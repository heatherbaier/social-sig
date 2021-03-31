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

import socialSigNoDrop
importlib.reload(socialSigNoDrop)
from helpers import *


####### Load our Data
#y - 'number_moved'
#x - 'everything else that is or can be represented as a float.'
devSet = pd.read_csv("./us_migration.csv")
devSet = devSet.loc[:, ~devSet.columns.str.contains('^Unnamed')]
devSet = devSet.apply(lambda x: pd.to_numeric(x, errors='coerce'))
devSet = devSet.dropna(axis=1)
devSet = devSet.drop(['sending'], axis = 1)

print(devSet.head())

y = torch.Tensor(devSet['US_MIG_05_10'].values)
X = devSet.loc[:, devSet.columns != "US_MIG_05_10"].values

mMScale = preprocessing.MinMaxScaler()
X = mMScale.fit_transform(X)



####### Build and fit the Model
lr = 1e-6
batchSize = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = socialSigNoDrop.SocialSigNet(X=X, outDim = batchSize).to(device)
epochs = 1

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr = lr)



x_train, y_train, x_val, y_val = train_test_split(X, y, .80)

train = [(k,v) for k,v in zip(x_train, y_train)]
val = [(k,v) for k,v in zip(x_val, y_val)]

train = torch.utils.data.DataLoader(train, batch_size = batchSize, shuffle = True)
val = torch.utils.data.DataLoader(val, batch_size = batchSize, shuffle = True)




best_mae = 9000000000000000000
best_model_wts = deepcopy(model.state_dict())


for epoch in range(epochs):

    for phase in ['train','val']:

        if phase == 'train':

            c = 1
            running_train_mae, running_train_loss = 0, 0

            for inputs, output in train:

                if len(inputs) == batchSize:

                    inputs = torch.tensor(inputs, dtype = torch.float32, requires_grad = True)
                    output = torch.reshape(torch.tensor(output, dtype = torch.float32, requires_grad = True), (batchSize,1))

                    inputs = inputs.to(device)
                    output = output.to(device)

                    # Forward pass
                    y_pred = model(inputs, str(epoch) + str(c))
                    loss = criterion(y_pred, output)  
                    
                    # Zero gradients, perform a backward pass, and update the weights.
                    optimizer.zero_grad()
                    grad = torch.autograd.grad(outputs = loss, inputs = inputs, retain_graph = True)
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

            for inputs, output in val:

                if len(inputs) == batchSize:

                    c += 1

                    inputs = torch.tensor(inputs, dtype = torch.float32, requires_grad = True)
                    output = torch.reshape(torch.tensor(output, dtype = torch.float32, requires_grad = True), (batchSize,1))

                    # Forward pass
                    y_pred = model(inputs, 1)
                    loss = criterion(y_pred, output)  

                    running_val_mae += mae(y_pred, output).item()
                    running_val_loss += loss.item()

                    d += 1
                    
                    if mae(y_pred, output).item() < best_mae:
                        best_mae = mae(y_pred, output).item()
                        best_model_wts = deepcopy(model.state_dict())

                    
                    
    print("Epoch: ", epoch)  
    print("  Train:")
    print("    Loss: ", running_train_loss / c)      
    print("    MAE: ", running_train_mae / c)
    print("  Val:")
    print("    Loss: ", running_val_loss / d)      
    print("    MAE: ", running_val_mae / d)





model.load_state_dict(best_model_wts)

torch.save({
            'epoch': 50,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, "./trained_models/socialSigNoDrop_50epochs.torch")