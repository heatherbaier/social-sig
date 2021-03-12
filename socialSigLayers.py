import numpy as np
import torch
import math

class bilinearImputation(torch.nn.Module):
    '''
    Class to create the social signature image
    '''
    def __init__(self, X):
        super(bilinearImputation, self).__init__()
        self.W = torch.nn.Parameter(torch.tensor(np.arange(0,X.shape[1]), dtype = torch.float32, requires_grad=True))
        self.outDim = [10,10]
        self.inDim = math.ceil(math.sqrt(X.shape[1]))

    def forward(self, batchX):
        self.X = torch.tensor(list(batchX))        
        #print("    W at beginning: ", self.W) 
        xTemp = torch.stack([self.X, self.W.repeat(self.X.shape[0],1).data])
        XSort = torch.sort(xTemp, dim=2, descending=False)
        inDataSize = XSort[0][0].shape[1] #Data we have per dimension
        targetSize = self.inDim ** 2
        paddingOffset = targetSize - inDataSize
        paddedInX = torch.nn.functional.pad(input=XSort[0][0], pad=(0,paddingOffset), mode="constant", value=0)
        buildImage = torch.reshape(paddedInX,(self.X.shape[0], 1, self.inDim, self.inDim))   
        return torch.nn.functional.interpolate(buildImage, size=([self.outDim[0], self.outDim[1]]), mode='bilinear')


