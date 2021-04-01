from sklearn import preprocessing
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import numpy as np
import importlib
import sklearn
import random
import pickle
import torch
import math
import os

import socialSigN
importlib.reload(socialSigN)
from helpers import *


####### Load our Data
#y - 'number_moved'
#x - 'everything else that is or can be represented as a float.'
devSet = pd.read_csv("./us_migration.csv")
devSet = devSet.loc[:, ~devSet.columns.str.contains('^Unnamed')]
devSet = devSet.apply(lambda x: pd.to_numeric(x, errors='coerce'))
devSet = devSet.dropna(axis=1)
# devSet = devSet.drop(['sending'], axis = 1)


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


# Assemble model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = socialSigN.SocialSigNet(X=torch.reshape(torch.tensor(read_file(1001), dtype = torch.float32, requires_grad = True), (1, 243)), 
                                    outDim = 1).to(device)


# Read in trained weights
checkpoint = torch.load("./trained_models/socialSigN_VB_50epochs.torch")
model.load_state_dict(checkpoint['model_state_dict'])


# Evaluate model and save predictions
eval_df = eval_model(avail_sending, (1,243), device, xy, model)
eval_df.to_csv("./predictions/socialSigN_VB_preds.csv", index = False)