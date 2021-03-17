import numpy as np
import torch
import math
import importlib
# import socialSigLayers
# importlib.reload(socialSigLayers)
import pandas as pd
import random
import matplotlib.pyplot as plt
import sklearn
from copy import deepcopy


# def scale(x, out_range=(0, 29)):
#     domain = np.min(x), np.max(x)
#     y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
#     to_ret = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
#     return to_ret.astype(int)


# def construct_indices(weights, dim, length):
#     indices = []
#     weights = scale(weights.clone().detach().numpy())
#     for i in range(0, dim):
#         to_add = i * length
#         cur_indices = [i + to_add for i in weights]
#         indices.append(cur_indices)
#     return torch.tensor(indices, dtype = torch.int64)



def train_val_split(X, y, split):
    train_num = int(len(X) * split)
    train_indices = random.sample(range(0, len(X)), train_num)
    val_indices = [i for i in range(0, len(X)) if i not in train_indices]
    x_train = X[train_indices]
    y_train = y[train_indices]
    x_val = X[val_indices]
    y_val = y[val_indices]
    return x_train, y_train, x_val, y_val 



def construct_indices(weights, dim, length):
    indices = []
    # weights = weights.clone().detach().numpy()
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



class bilinearImputation(torch.nn.Module):
    '''
    Class to create the social signature image
    '''
    def __init__(self, X):
        super(bilinearImputation, self).__init__()
        self.W = torch.nn.Parameter(torch.tensor(np.arange(0, X.shape[1]), dtype = torch.float32, requires_grad=True))
        self.outDim = [10,10]
        self.inDim = math.ceil(math.sqrt(X.shape[1]))

    def forward(self, batchX):
        
        print("    W at beginning: ", torch.tensor(self.W, dtype = torch.int)) 

        # taken = torch.take(batchX, construct_indices(self.W, batchX.shape[0], self.W.shape[0]))
        taken = torch.take(batchX, construct_indices(torch.clamp(torch.tensor(self.W, dtype = torch.int64), 0, 29), batchX.shape[0], self.W.shape[0]))

        batchX.data = batchX.data.copy_(taken.data)
        # print("batchX.data: ", batchX.data.copy_(taken.data))  
        
        inDataSize = self.W.shape[0] #Data we have per dimension
        targetSize = self.inDim ** 2
        paddingOffset = targetSize - inDataSize
        paddedInX = torch.nn.functional.pad(input=batchX, pad=(0,paddingOffset), mode="constant", value=0)
        buildImage = torch.reshape(paddedInX,(batchX.shape[0], 1, self.inDim, self.inDim))   
        return torch.nn.functional.interpolate(buildImage, size=([self.outDim[0], self.outDim[1]]), mode='bilinear')




# class bilinearImputation(torch.nn.Module):
#     '''
#     Class to create the social signature image
#     '''
#     def __init__(self, X):
#         super(bilinearImputation, self).__init__()
#         self.W = torch.nn.Parameter(torch.tensor(np.arange(0,X.shape[1]), dtype = torch.float32, requires_grad=True))
#         self.outDim = [10,10]
#         self.inDim = math.ceil(math.sqrt(X.shape[1]))

#     def forward(self, batchX):
        
#         # print("    W at beginning: ", torch.tensor(self.W, dtype = torch.int)) 

#         self.X = batchX
#         # xTemp = torch.stack([self.X, self.W.clone().repeat(self.X.shape[0],1).data])

#         # print(self.W.shape)

#         print(construct_indices(self.W, self.X.shape[0], self.W.shape[0]))

#         # print("    TAKE: ", torch.take(self.X, construct_indices(self.W, self.X.shape[0], self.W.shape[0])))
#         taken = torch.take(self.X, construct_indices(self.W, self.X.shape[0], self.W.shape[0]))
        
#         # XSort, indices = torch.sort(xTemp, dim=1, descending=False)

#         # print("self.X.data: ", self.X.data.copy_(taken.data)) 

#         # print("    XSort: ", XSort[0].shape)

#         self.X.data = self.X.data.copy_(taken.data)
#         # print("self.X.data: ", self.X.data.copy_(taken.data))       
        
#         inDataSize = self.W.shape[0] #Data we have per dimension
#         targetSize = self.inDim ** 2
#         paddingOffset = targetSize - inDataSize
#         paddedInX = torch.nn.functional.pad(input=self.X, pad=(0,paddingOffset), mode="constant", value=0)
#         buildImage = torch.reshape(paddedInX,(self.X.shape[0], 1, self.inDim, self.inDim))   
#         return torch.nn.functional.interpolate(buildImage, size=([self.outDim[0], self.outDim[1]]), mode='bilinear')


###### Define our model
class SocialSigNet(torch.nn.Module):
    def __init__(self, X):
        super().__init__()
        self.SocialSig = bilinearImputation(X=X)                
        self.conv2d = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxPool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)#,
            # torch.nn.Sequential(
            #     torch.nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False),
            #     torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            # )
        )
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.seqBlock1 = torch.nn.Sequential(self.block1, self.block1)
        self.seqBlock2 = torch.nn.Sequential(self.block2, self.block3)
    

        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            #   (downsample): Sequential(
            #     (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            #     (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            #   )
        )
        self.block5 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.seqBlock3 = torch.nn.Sequential(self.block4, self.block5)

        self.block6 = torch.nn.Sequential(
                torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            #   (downsample): Sequential(
            #     (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            #     (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            #   )
            )
        self.block7 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.seqBlock4 = torch.nn.Sequential(self.block6, self.block7)

        self.linear = torch.nn.Linear(2560, 5)

        
    def forward(self, X, epoch):
        out = self.SocialSig(X) # OUT:  torch.Size([100, 1, 10, 10])

        # pd.DataFrame(out.clone()[0].flatten()).to_csv("./figs/im" + str(epoch) + ".csv")

        out = self.conv2d(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxPool(out)
        out = self.seqBlock1(out)
        out = self.seqBlock2(out)
        out = self.seqBlock3(out)
        out = self.seqBlock4(out)
        out = self.relu(out)
        out = out.flatten()

        self.linear = torch.nn.Linear(out.shape[0], X.shape[0])
        out = self.linear(out)

        # print("OUT: ", out.shape)
        # print("OUT: ", out)
        return out