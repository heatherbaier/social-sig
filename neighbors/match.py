import pandas as pd

def get_id(x):
    return int(x.partition('0')[2])

devSet = pd.read_csv("../us_migration_allvars.csv")
devSet['GEO2_MX'] = devSet['GEO2_MX'].apply(lambda x: get_id(str(x)))
devSet = devSet.drop(["Unnamed: 0"], axis = 1)
devSet = devSet.rename(columns = {'GEO2_MX': 'sending'})

devSet.to_csv("../us_migration_allvars.csv")

print(devSet.head(20))