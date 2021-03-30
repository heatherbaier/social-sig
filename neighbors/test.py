from joblib import Parallel, delayed
import multiprocessing

import geopandas as gpd
import pandas as pd


def processInput(i):
    print("Working on poly ", i, " of ", len(polygons))
    cur_distances = []
    for dist_to in range(len(polygons)):
        cur_dist = polygons[i].distance(polygons[dist_to])
        cur_distances.append(cur_dist)
    return cur_distances


num_cores = multiprocessing.cpu_count()
print("Num Cores: ", num_cores)

mex = gpd.read_file("./MEX/MEX_ADM2_fixedInternalTopology.shp")
polygons = mex['geometry'].to_list()#[0:50]
shapeIDs = mex['shapeID'].to_list()#[0:50]

results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in range(len(polygons)))

print("Done calculating distances.")

dist_df = pd.DataFrame(results)
dist_df.columns = shapeIDs
dist_df = dist_df.reset_index()
dist_df['index'] = shapeIDs
dist_df = dist_df.rename(columns = {'index':'shapeID'})
print(dist_df.head())
dist_df.to_csv("./MEX/mex_distance_matrix.csv")
