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
    for i in range(0, dim):
        to_add = i * length
        cur_indices = [i + to_add for i in weights]
        indices.append(cur_indices)
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