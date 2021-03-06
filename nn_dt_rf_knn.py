from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd
import random


devSet = pd.read_csv("./us_migration.csv")
devSet = devSet.loc[:, ~devSet.columns.str.contains('^Unnamed')]
devSet = devSet.apply(lambda x: pd.to_numeric(x, errors='coerce'))
devSet = devSet.dropna(axis=1)

y = devSet['US_MIG_05_10'].values
X = devSet.loc[:, devSet.columns != "US_MIG_05_10"].values

mMScale = preprocessing.MinMaxScaler()
X = mMScale.fit_transform(X)

def train_val_split(X, y, split):
    train_num = int(len(X) * split)
    train_indices = random.sample(range(0, len(X)), train_num)
    val_indices = [i for i in range(0, len(X)) if i not in train_indices]
    x_train, y_train = [X[i] for i in train_indices], [y[i] for i in train_indices]
    x_val, y_val = [X[i] for i in val_indices], [y[i] for i in val_indices]
    return x_train, y_train, x_val, y_val 


x_train, y_train, x_val, y_val = train_val_split(X, y, .80)


DT_regression = tree.DecisionTreeRegressor(min_samples_split= 20, max_features ='auto', max_depth= 5, criterion= 'mae', ccp_alpha= 0.0)
DT_regressionFit = DT_regression.fit(x_train, y_train)


RF_regression = RandomForestRegressor(n_estimators= 500, min_samples_split= 5, min_samples_leaf= 50, max_features= 'auto', max_depth= 10, random_state=146)
RF_regressionFit = RF_regression.fit(x_train, y_train)


neigh = KNeighborsRegressor(n_neighbors=2)
neighFit = neigh.fit(x_train, y_train)


mlp_regr = MLPRegressor(max_iter=500)
mlp_regrFit = mlp_regr.fit(x_train, y_train)


DT_MAD = mean_absolute_error(y_val, DT_regressionFit.predict(x_val))
KNN_MAD = mean_absolute_error(y_val, neighFit.predict(x_val))
RF_MAD = mean_absolute_error(y_val, RF_regressionFit.predict(x_val))
MLP_MAD = mean_absolute_error(y_val, mlp_regrFit.predict(x_val))


print("MAE")
print('Decision Tree MAD: ' + str(DT_MAD))
print('KNN MAD ' + str(KNN_MAD))
print('RF MAD ' + str(RF_MAD))
print('MLP MAD ' + str(MLP_MAD))
print("\n")


DT_MAPE = mean_absolute_percentage_error(y_val, DT_regressionFit.predict(x_val))
KNN_MAPE = mean_absolute_percentage_error(y_val, neighFit.predict(x_val))
RF_MAPE = mean_absolute_percentage_error(y_val, RF_regressionFit.predict(x_val))
MLP_MAPE = mean_absolute_percentage_error(y_val, mlp_regrFit.predict(x_val))


print("MAPE")
print('Decision Tree MAPE: ' + str(DT_MAPE * 100))
print('KNN MAPE ' + str(KNN_MAPE * 100))
print('RF MAPE ' + str(RF_MAPE * 100))
print('MLP MAPE ' + str(MLP_MAPE))
print("\n")


DT_R2 = r2_score(y_val, DT_regressionFit.predict(x_val))
KNN_R2 = r2_score(y_val, neighFit.predict(x_val))
RF_R2 = r2_score(y_val, RF_regressionFit.predict(x_val))
MLP_R2 = r2_score(y_val, mlp_regrFit.predict(x_val))


print("R2")
print('Decision Tree R2: ' + str(DT_R2))
print('KNN R2 ' + str(KNN_R2))
print('RF R2 ' + str(RF_R2))
print('MLP R2 ' + str(MLP_R2))
print("\n")


def allocation_error(ytruelist, ypredlist):
    return abs((sum(ytruelist) - sum(ypredlist)) / len(ypredlist))


DT_AE = allocation_error(y_val, DT_regressionFit.predict(x_val))
KNN_AE = allocation_error(y_val, neighFit.predict(x_val))
RF_AE = allocation_error(y_val, RF_regressionFit.predict(x_val))
MLP_AE = allocation_error(y_val, mlp_regrFit.predict(x_val))


print("ALLOCATION ERROR")
print('Decision Tree AE: ' + str(DT_AE))
print('KNN AE ' + str(KNN_AE))
print('RF AE ' + str(RF_AE))
print('MLP AE ' + str(MLP_AE))
print("\n")


def quantity_error(ytruelist, ypredlist): 
    return abs(sum(ytruelist) - sum(ypredlist))


DT_QE = quantity_error(y_val, DT_regressionFit.predict(x_val))
KNN_QE = quantity_error(y_val, neighFit.predict(x_val))
RF_QE = quantity_error(y_val, RF_regressionFit.predict(x_val))
MLP_QE = quantity_error(y_val, mlp_regrFit.predict(x_val))


print("QUANTITY ERROR")
print('Decision Tree QE: ' + str(DT_QE))
print('KNN QE ' + str(KNN_QE))
print('RF QE ' + str(RF_QE))
print('MLP QE ' + str(MLP_QE))
print("\n")



## Plots
plt.scatter(y_val, DT_regressionFit.predict(x_val))
plt.ylim([0, 4000])
plt.title("Decision Tree")
plt.savefig("./model_plots/DT.png")
plt.clf()

plt.scatter(y_val, RF_regressionFit.predict(x_val))
plt.ylim([0, 4000])
plt.title("Random Forest")
plt.savefig("./model_plots/RF.png")
plt.clf()

plt.scatter(y_val, neighFit.predict(x_val))
plt.ylim([0, 4000])
plt.title("KNN")
plt.savefig("./model_plots/KNN.png")
plt.clf()

plt.scatter(y_val, mlp_regrFit.predict(x_val))
plt.ylim([0, 4000])
plt.title("MLP")
plt.savefig("./model_plots/MLP.png")
plt.clf()



