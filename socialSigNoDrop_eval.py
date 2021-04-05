import torchvision.models as models
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

import socialSigNoDrop
importlib.reload(socialSigNoDrop)
from helpers import *


####### Load our Data
#y - 'number_moved'
#x - 'everything else that is or can be represented as a float.'
devSet = pd.read_csv("./us_migration_allvars.csv")
devSet = devSet.fillna(0)
devSet = devSet.loc[:, ~devSet.columns.str.contains('^Unnamed')]
devSet = devSet.apply(lambda x: pd.to_numeric(x, errors='coerce'))
devSet = devSet.dropna(axis=1)
sending = devSet['sending'].to_list()
devSet = devSet.drop(['sending'], axis = 1)

print(devSet.head())

with open("vars.txt", "r") as vars_file:
    vars = vars_file.read()

devSet = devSet[vars.splitlines()]

y = torch.Tensor(devSet['num_persons_to_us'].values)
X = devSet.loc[:, devSet.columns != "num_persons_to_us"].values

mMScale = preprocessing.MinMaxScaler()
X = mMScale.fit_transform(X)



# Prep model with trained weights
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet50 = models.resnet50(pretrained=True)
model = socialSigNoDrop.scoialSigNet_NoDrop(X=torch.reshape(torch.tensor(X[0], dtype = torch.float32), (1, X[0].shape[0])), outDim = 1, resnet = resnet50).to(device)
checkpoint = torch.load("./trained_models/socialSigNoDrop_50epochs.torch")
model.load_state_dict(checkpoint['model_state_dict'])


# Evaluate model and save predictions
eval_df = eval_model(X, y, sending, (1, X[0].shape[0]), model, device)
eval_df.to_csv("./predictions/socialSigNoDrop_preds.csv", index = False)
