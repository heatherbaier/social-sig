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
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    to_ret = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
    return to_ret.astype(int)


def construct_indices(weights, dim, length):
    indices = []
    weights = scale(weights.clone().detach().numpy())
    for i in range(0, dim):
        to_add = i * length
        cur_indices = [i + to_add for i in weights]
        indices.append(cur_indices)
    return torch.tensor(indices, dtype = torch.int64)



def update_function(param, grad, loss, learning_rate):
    # print(grad.mean(axis = 0))
    # print("    WEIGHT UPDATES: ", param - learning_rate * grad.mean(axis = 0)[0])
    return param - learning_rate * grad.mean(axis = 0)


def mae(real, pred):
    return torch.abs(real - pred).mean()


def show_image(best_epoch):
    df = pd.read_csv("./figs/im" + str(best_epoch) + ".csv")
    df["0"] = df["0"].str.split("(").str[1].str.split(",").str[0].astype(float)
    plt.imshow(np.reshape(np.array(df["0"]), (10, 10)))