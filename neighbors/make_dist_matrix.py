import geopandas as gpd
import pandas as pd
import argparse

import multiprocessing
from joblib import Parallel, delayed
from functools import partial
import os


def make_distance_matrix(gpd_path):
    
    mex = gpd.read_file("./MEX/MEX_ADM2_fixedInternalTopology.shp")
    polygons = mex['geometry'].to_list()
    shapeIDs = mex['shapeID'].to_list()[0:15]
    
    all_distances = []

    for poly in range(15):
        print("Working on poly ", poly, " of ", len(polygons))
        cur_distances = []
        for dist_to in range(15):
            cur_dist = polygons[poly].distance(polygons[dist_to])
            cur_distances.append(cur_dist)
        all_distances.append(cur_distances)
        
    dist_df = pd.DataFrame(all_distances)
    dist_df.columns = shapeIDs
    dist_df = dist_df.reset_index()
    dist_df['index'] = shapeIDs
    dist_df = dist_df.rename(columns = {'index':'shapeID'})

    return dist_df





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("gpd_path", help="Country ISO")
    parser.add_argument("out_path", help="ADM level")
    args = parser.parse_args()
    
    num_cores = multiprocessing.cpu_count()
    output = Parallel(n_jobs=num_cores)(delayed(make_distance_matrix)(gpd_path = args.gpd_path))
    
    output.to_csv(args.out_path, index = False)
