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
import os

import socialSigN
importlib.reload(socialSigN)
from helpers import *


import warnings
warnings.filterwarnings('ignore')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-batchSize", help="Size of each batch", default = 50)
    parser.add_argument("-epochs", help="Number of epochs", default = 50)
    args = parser.parse_args()


    ####### Load our Data
    #y - 'number_moved'
    #x - 'everything else that is or can be represented as a float.'
    devSet = pd.read_csv("./us_migration.csv")
    devSet = devSet.loc[:, ~devSet.columns.str.contains('^Unnamed')]
    devSet = devSet.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    devSet = devSet.dropna(axis=1)

    y = torch.Tensor(devSet['US_MIG_05_10'].values)
    X = devSet.loc[:, devSet.columns != "US_MIG_05_10"].values

    mMScale = preprocessing.MinMaxScaler()
    X = mMScale.fit_transform(X)


    # Reference xy dct for data loading later on and sending ID list
    xy = devSet[['sending', 'US_MIG_05_10']]
    sending = devSet['sending'].to_list()


    # Get the municipality ID's that have available distance data
    avail_sending = []
    for i in os.listdir("./neighbors/inputs/"):
        avail_sending.append(i.split(".")[0])

    avail_sending = [i for i in avail_sending if i not in ['nan', '']]
    avail_sending = [i for i in avail_sending if int(i) in sending]


    ####### Build and fit the Model
    lr = 1e-5
    batchSize = int(args.batchSize)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = socialSigN.SocialSigNet(X=torch.reshape(torch.tensor(read_file(1001), dtype = torch.float32, requires_grad = True), (1, 243)), 
                                    outDim = batchSize).to(device)
    epochs = int(args.epochs)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)


    # Prep the training and validation sets
    x_train, y_train, x_val, y_val = train_test_split(avail_sending, .80, xy)

    train = [(k,v) for k,v in zip(x_train, y_train)]
    val = [(k,v) for k,v in zip(x_val, y_val)]

    train = torch.utils.data.DataLoader(train, batch_size = batchSize, shuffle = True)
    val = torch.utils.data.DataLoader(val, batch_size = batchSize, shuffle = True)


    # Train the model
    model_wts = train_model(model, train, val, criterion, optimizer, epochs, batchSize, device, lr)
    model.load_state_dict(model_wts)
    torch.save({
                'epoch': 50,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, "./trained_models/socialSigN_VB_50epochs.torch")


