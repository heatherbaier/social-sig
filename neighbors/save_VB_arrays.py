import pandas as pd
import numpy as np

df = pd.read_csv("../us_migration_allvars.csv")
df = df.fillna(0)
df = df.replace(np.nan, 0)
with open("../vars.txt", "r") as vars_file:
    vars = vars_file.read()
cols_list = vars.splitlines()
cols_list = list(cols_list) + ['sending']
df = df[cols_list]


var_cols = [i for i in df.columns if i not in ['Unnamed: 0', 'sending', 'num_persons_to_us']]
# var_cols = var_cols[0:219]

print(len(var_cols))


for i in var_cols:
    df[i] = df[i] / max(df[i])
df = df.replace(np.nan, 0)


match = pd.read_csv("./MEX/gB_IPUMS_match.csv")
match = match[['shapeID', 'MUNI2015']]

ref_dict = dict(zip(match['shapeID'], match['MUNI2015']))

dist_df = pd.read_csv("./MEX/closest_polygons.csv")
dist_df = dist_df


for i in dist_df.columns:
    dist_df[i] = dist_df[i].map(ref_dict)


print(df.head())

print(dist_df.head())


# print(hajhsgjkhajh)


file_num = 0

for col, row in dist_df.iterrows():
    
    print("Municipality: ", row['shapeID'], "    |    #", file_num)
    
    all_vals = []
    
    for col_name in var_cols:
        tmp_vals = []
        for i in row:
            cur_dta = df[df['sending'] == i]
            if len(cur_dta) != 0:
                cur_val = list(cur_dta[col_name])[0]
                tmp_vals.append(cur_val)
            else:
                cur_val = 0
                tmp_vals.append(cur_val)
        all_vals.append(tmp_vals)

    
    ar_vals = []    
    [ar_vals.append(np.reshape(np.array(i), (3,3))) for i in all_vals]


    n = 3
    indices = [i for i in range(len(ar_vals))]
    indices = [indices[i:i + n] for i in range(0, len(indices), n)]
    final_dta = []

    for i in indices:
        tmp = np.hstack([ar_vals[i[0]], ar_vals[i[1]], ar_vals[i[2]]])
        final_dta.append(tmp)
        
    fname = "./inputs2/" + str(row['shapeID']) + ".txt"
    # np.savetxt(fname, np.reshape(np.array(final_dta), (27,9)))
    np.savetxt(fname, np.reshape(np.array(final_dta), (219,9)))
    
    file_num += 1
    