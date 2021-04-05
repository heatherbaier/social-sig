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




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-batchSize", help="Size of each batch", default = 50)
    parser.add_argument("-epochs", help="Number of epochs", default = 50)
    args = parser.parse_args()


    ####### Load our Data
    #y - 'number_moved'
    #x - 'everything else that is or can be represented as a float.'
    devSet = pd.read_csv("./us_migration_allvars.csv")
    devSet = devSet.fillna(0)
    devSet = devSet.loc[:, ~devSet.columns.str.contains('^Unnamed')]
    devSet = devSet.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    devSet = devSet.dropna(axis=1)
    devSet = devSet.drop(['sending'], axis = 1)

    print(devSet.head())

    with open("vars.txt", "r") as vars_file:
        vars = vars_file.read()

    devSet = devSet[vars.splitlines()]

    y = torch.Tensor(devSet['num_persons_to_us'].values)
    X = devSet.loc[:, devSet.columns != "num_persons_to_us"].values

    mMScale = preprocessing.MinMaxScaler()
    X = mMScale.fit_transform(X)


    ####### Build and fit the Model
    lr = 1e-6
    batchSize = int(args.batchSize)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = socialSigNoDrop.SocialSigNet(X=X, outDim = batchSize).to(device)
    resnet50 = models.resnet50(pretrained=True)
    model = socialSigNoDrop.scoialSigNet_NoDrop(X=X, outDim = batchSize, resnet = resnet50).to(device)
    epochs = int(args.epochs)
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)



    # Prep the training and validation sets
    x_train, y_train, x_val, y_val = train_test_split(X, y, .80)

    train = [(k,v) for k,v in zip(x_train, y_train)]
    val = [(k,v) for k,v in zip(x_val, y_val)]
    
    train = torch.utils.data.DataLoader(train, batch_size = batchSize, shuffle = True)
    val = torch.utils.data.DataLoader(val, batch_size = batchSize, shuffle = True)

    model_wts = train_model(model, train, val, criterion, optimizer, epochs, batchSize, device, lr)
    model.load_state_dict(model_wts)
    torch.save({
                'epoch': 50,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, "./trained_models/socialSigNoDrop_50epochs.torch")