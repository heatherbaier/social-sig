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
    print(dim)
    print(length)
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
    weights = scale_noOverlap(weights.clone().cpu().detach().numpy())
    indices = dim*[[x for _,x in sorted(zip(weights,range(0,length)))]]
    for i in range(0,len(indices)):
        indices[i] = [x+(i*length) for x in indices[i]]
    return torch.tensor(indices, dtype = torch.int64).to("cuda:0")

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



def read_file(shape_id):
    fname = "./neighbors/inputs/" + str(shape_id) + ".0.txt"
    with open(fname, "r") as f:
        f = f.read()

    to_return = []
        
    for i in f.splitlines():
        splt = i.split(" ")
        [to_return.append(float(val)) if val not in ['inf', '-inf', 'nan'] else to_return.append(0) for val in splt]

    return to_return


def load_data(xy, batch, sending_ids):

    batch = [sending_ids[i] for i in batch]
    # print("    Batch ID's: ", batch)
    inputs = [read_file(i) for i in batch]
    inputs = torch.reshape(torch.tensor(inputs, dtype = torch.float32, requires_grad = True), (len(inputs), 243))

    ys = []

    for i in batch:
        # print(i)
        cur = xy[xy['sending'] == int(i)]
        # print(list(cur['US_MIG_05_10'])[0])
        ys.append(list(cur['US_MIG_05_10'])[0])

    ys = torch.reshape(torch.tensor(ys, dtype = torch.float32, requires_grad = True), (len(inputs), 1))

    return inputs, ys




def train_test_split(sending_ids, split, xy):

    train_num = int(len(sending_ids) * split)
    val_num = int(len(sending_ids) - train_num)

    train_indices = random.sample(range(len(sending_ids)), train_num)
    val_indices = [i for i in range(len(sending_ids)) if i not in train_indices]

    x_train, y_train = load_data(xy, train_indices, sending_ids)


    x_val, y_val = load_data(xy, val_indices, sending_ids)

    return x_train, y_train, x_val, y_val




def train_model(model, train, val, criterion, optimizer, epochs, batchSize, device, lr):

    best_mae = 9000000000000000000
    best_model_wts = deepcopy(model.state_dict())


    for epoch in range(epochs):

        for phase in ['train','val']:

            if phase == 'train':

                c = 1
                running_train_mae, running_train_loss = 0, 0

                for inputs, output in train:

                    if len(inputs) == batchSize:

                        inputs = inputs.to(device)
                        output = output.to(device)

                        inputs = torch.tensor(inputs, dtype = torch.float32, requires_grad = True)
                        output = torch.reshape(torch.tensor(output, dtype = torch.float32, requires_grad = True), (batchSize,1))

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

                        inputs = inputs.to(device)
                        output = output.to(device)

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
        print("\n")

    return best_model_wts



def eval_model(X, size, device, xy, model):

    preds, ids, true_vals = [], [], []

    # Run each row through the data frame
    for i in X:
        true_val = xy[xy['sending'] == int(i)]
        true_val = list(true_val['US_MIG_05_10'])[0]
        input = torch.reshape(torch.tensor(read_file(i), dtype = torch.float32), size).to(device)
        model.eval()
        pred = model(input, 1).detach().cpu().numpy()[0][0]
        true_vals.append(true_val)
        preds.append(pred)
        ids.append(i)

    # Make data frame
    df = pd.DataFrame()
    df['sending_id'] = ids
    df['true'] = true_vals
    df['pred'] = preds
    df['abs_error'] = abs(df['true'] - df['pred'])
    df['error'] = df['true'] - df['pred']

    return df




