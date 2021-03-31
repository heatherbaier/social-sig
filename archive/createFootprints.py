import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from libtiff import TIFF
from PIL import Image
import pandas as pd
import torch
import math
import json



####### Load our Data
from sklearn import preprocessing
devSet = pd.read_csv("./us_migration.csv")
devSet = devSet.loc[:, ~devSet.columns.str.contains('^Unnamed')]
devSet = devSet.apply(lambda x: pd.to_numeric(x, errors='coerce'))
devSet = devSet.dropna(axis=1)

sending = devSet['sending']
print(sending[0:5])

y = torch.Tensor(devSet['US_MIG_05_10'].values)
X = devSet.loc[:, devSet.columns != "US_MIG_05_10"].values

mMScale = preprocessing.MinMaxScaler()
X = mMScale.fit_transform(X)


with open("./weights.json", 'r') as w:
    w = json.load(w)

weights = w['weights']


def create_footprints(weights, X):
    X = torch.tensor(X)
    weights = torch.tensor(weights, dtype = torch.int64)
    taken = torch.take(X, torch.clamp(weights, 0, 29))#, batchX.shape[0], self.W.shape[0])
    inDim = math.ceil(math.sqrt(X.shape[0]))
    outDim = [224,224]
    X.data = X.data.copy_(taken.data) 
    inDataSize = X.shape[0] #Data we have per dimension
    targetSize = inDim ** 2
    paddingOffset = targetSize - inDataSize
    paddedInX = torch.nn.functional.pad(input=X, pad=(0,paddingOffset), mode="constant", value=0)
    buildImage = torch.reshape(paddedInX,(1, 1, inDim, inDim))   
    image = torch.nn.functional.interpolate(buildImage, size=([outDim[0], outDim[1]]), mode='bilinear')
    return image.detach().numpy()


for i in range(0, len(sending)):
    cur = create_footprints(weights, X[i])[0][0]
    # mpimg.imsave("./im" + str(sending[i]) + ".tiff", cur)
    tiff = TIFF.open("./final_pics/im" + str(sending[i]) + ".tiff", mode='w')
    tiff.write_image(cur)
    tiff.close()