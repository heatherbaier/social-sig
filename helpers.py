import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import numpy as np
import importlib
import sklearn
import random
import torch
import math

from helpers import *


def scale(x, out_range=(0, 29)):
    '''
    Takes as input the coordinate weights and scales them between 0 and len(weights)
    '''
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    to_ret = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
    return to_ret.astype(int)


def construct_indices(weights, dim, length):
    '''
    The coordinate weights are between 0-len(weights) but the size of X is len(weights) * batch size so the torch.taken
    function will only take items at indices between  0 & len(weights) meaning only the first item in the batch. This function
    adds len(weights) to each index so taken grabs from every batch
    ^^ fix that explanation yo lol
    '''
    indices = []
    weights = scale(weights.clone().detach().numpy())
    print(weights.size)
    for i in range(0, dim):
        to_add = i * length
        cur_indices = [i + to_add for i in weights]
        indices.append(cur_indices)
    return torch.tensor(indices, dtype = torch.int64)

def scale_noOverlap(x, out_range=(0, 29)):
    '''
    Takes as input the coordinate weights and scales them between 0 and len(weights)
    Dan removed int rounding from this one.
    '''
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    to_ret = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
    return to_ret

def construct_noOverlap_indices(weights, dim, length):
    '''
    The coordinate weights are between 0-len(weights) but the size of X is len(weights) * batch size so the torch.taken
    function will only take items at indices between  0 & len(weights) meaning only the first item in the batch. This function
    adds len(weights) to each index so taken grabs from every batch
    Dan then modified whatever the above was to ensure the rounding only occurs to available indices, precluding
    drop out. 
    ^^ fix that explanation yo lol
    '''
    indices = []
    weights = scale_noOverlap(weights.clone().detach().numpy())
    indices = 200*[[x for _,x in sorted(zip(weights,range(0,dim)))]]
    for i in range(0,len(indices)):
        indices[i] = [x+(i*length) for x in indices[i]]
    return torch.tensor(indices, dtype = torch.int64)

def update_function(param, grad, loss, learning_rate):
    '''
    Calculates the new coordinate weights based on the LR and gradient
    '''
    return param - learning_rate * grad.mean(axis = 0)


def mae(real, pred):
    '''
    Calculates MAE of an epoch
    '''
    return torch.abs(real - pred).mean()


def show_image(best_epoch):
    '''
    Takes as input an epoch number and displays the SocialSig from that epoch
    '''
    df = pd.read_csv("./figs/im" + str(best_epoch) + ".csv")
    df["0"] = df["0"].str.split("(").str[1].str.split(",").str[0].astype(float)
    plt.imshow(np.reshape(np.array(df["0"]), (10, 10)))




class dataLoader():

    '''
    DataLoader used in resnetWorking
    Loads socialSig footprint images as 1D raster arrays for feeding into 1D resnet
    '''

    def __init__(self, dir, df):
        self.data = []
        self.labels = []
        df = pd.read_csv(df)

        for i in os.listdir(dir):
            fname = os.path.join(dir, i)
            im = np.array(rio.open(fname).read(1))
            im = torch.from_numpy(im)
            im = torch.reshape(im, (1, 224, 224)).numpy()
            num_mig = df[df['sending'] == int(fname.split("m")[1].split(".")[0])]['US_MIG_05_10'].to_list()[0]

            self.data.append(im)
            self.labels.append(num_mig)

    def train_val_split(self, split):
        train_num = int(len(self.data) * split)
        train_indices = random.sample(range(0, len(self.data)), train_num)
        val_indices = [i for i in range(0, len(self.data)) if i not in train_indices]
        x_train, y_train = [self.data[i] for i in train_indices], [self.labels[i] for i in train_indices]
        x_val, y_val = [self.data[i] for i in val_indices], [self.labels[i] for i in val_indices]
        return x_train, y_train, x_val, y_val 