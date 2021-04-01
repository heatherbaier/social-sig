import matplotlib.pyplot as plt
import pandas as pd


ss_NoDrop = pd.read_csv("./predictions/socialSigNoDrop_preds.csv")
plt.scatter(ss_NoDrop['true'], ss_NoDrop['pred'])
plt.ylim([0, 5000])
plt.title("socialSig No Drop")
plt.savefig("./model_plots/socialSig_NoDrop.png")
plt.clf()


ssN_VB = pd.read_csv("./predictions/socialSigN_VB_preds.csv")
plt.scatter(ssN_VB['true'], ssN_VB['pred'])
plt.ylim([0, 5000])
plt.title("SocialSig Value Based Neighbors")
plt.savefig("./model_plots/socialSigN_VB.png")
plt.clf()