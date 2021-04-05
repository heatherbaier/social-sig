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
import os

import socialSigN_VB
importlib.reload(socialSigN_VB)
from helpersN_VB import *


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
    devSet = pd.read_csv("./us_migration_allvars.csv")
    devSet = devSet.fillna(0)
    devSet = devSet.loc[:, ~devSet.columns.str.contains('^Unnamed')]
    devSet = devSet.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    devSet = devSet.dropna(axis=1)

    with open("vars.txt", "r") as vars_file:
        vars = vars_file.read()

    cols_list = vars.splitlines()
    cols_list = list(cols_list) + ['sending']
    devSet = devSet[cols_list]

    y = torch.Tensor(devSet['num_persons_to_us'].values)
    X = devSet.loc[:, devSet.columns != "num_persons_to_us"].values

    mMScale = preprocessing.MinMaxScaler()
    X = mMScale.fit_transform(X)


    # Reference xy dct for data num_persons_to_us later on and sending ID list
    xy = devSet[['sending', 'num_persons_to_us']]
    sending = devSet['sending'].to_list()


    # Get the municipality ID's that have available distance data
    avail_sending = []
    for i in os.listdir("./neighbors/inputs/"):
        avail_sending.append(i.split(".")[0])

    avail_sending = [i for i in avail_sending if i not in ['nan', '']]
    avail_sending = [i for i in avail_sending if int(i) in sending]


    # Prep the training and validation sets
    batchSize = int(args.batchSize)
    
    x_train, y_train, x_val, y_val = train_test_split(avail_sending, .80, xy)

    train = [(k,v) for k,v in zip(x_train, y_train)]
    val = [(k,v) for k,v in zip(x_val, y_val)]

    input_size = train[0][0].shape[0]

    train = torch.utils.data.DataLoader(train, batch_size = batchSize, shuffle = True)
    val = torch.utils.data.DataLoader(val, batch_size = batchSize, shuffle = True)


    ####### Build and fit the Model
    lr = 1e-5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet50 = models.resnet50(pretrained=True)
    model = socialSigN_VB.SocialSigNet(X=torch.reshape(torch.tensor(read_file(1001), dtype = torch.float32, requires_grad = True), (1, input_size)), 
                                    outDim = batchSize, 
                                    resnet = resnet50).to(device)
    epochs = int(args.epochs)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)





    # Train the model
    model_wts = train_model(model, train, val, criterion, optimizer, epochs, batchSize, device, lr)
    model.load_state_dict(model_wts)
    torch.save({
                'epoch': 50,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, "./trained_models/socialSigN_VB_50epochs.torch")


