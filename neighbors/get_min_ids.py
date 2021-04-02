import pandas as pd
import numpy as np

from helpersN_VB import *


df = pd.read_csv("./MEX/mex_distance_matrix.csv")
df = df.drop(['Unnamed: 0', 'shapeID'], axis = 1)

min_list, num, c = [], 8, 0


for col, row in df.iterrows():
    
    # Get the indices of the num closest polygons to the currect one
    mins = get_mins(row, num, df.columns[c])
    
    # The get_min algo does not include the current polygon so drop that for the list of column we are going to grab from using the min indices
    ref_cols = [i for i in df.columns if i != df.columns[c]]
    
    # Get the shapeID's of the num closest polygons, append them to the overall list and increment c
    min_ids = [ref_cols[i] for i in mins]
    min_list.append(min_ids)
    c += 1
    

# Make the final data frame
min_df = pd.DataFrame(min_list)
min_df = min_df.drop([8,9,10,11,12,13,14,15,16,17,18,19,20,21], axis = 1)
min_df.columns = ['closest' + str(i) for i in range(0, num)]
min_df = min_df.reset_index()
min_df['index'] = df.columns#[0:5]
min_df = min_df.rename(columns = {'index':'shapeID'})
min_df.head()


min_df.to_csv("./MEX/closest_polygons2.csv", index = False)