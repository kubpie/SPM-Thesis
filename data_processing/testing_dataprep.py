import os
import numpy as np
from pathlib import Path
import pandas as  pd
import matplotlib.pyplot as plt
import xgboost as xgb
from joblib import load
from sklearn.model_selection import train_test_split
from pycebox.ice import ice, ice_plot

from data_analysis import ClassImbalance, PlotCorrelation, plot_ice_grid, ICEPlot
from data_prep import FeatDuct, FeatBathy, FeatSSPVec, FeatSSPId, FeatSSPStat, FeatSSPOnDepth
from data_prep import LoadData, UndersampleData, SMOTSampling
from data_prep import CreateModelSplits, EncodeData
from ssp_features import SSPId

PATH = os.getcwd()
path = Path(PATH+"/data/")
ALLDATA = LoadData(path)

#####################################################
### Messy script for testing & plotting data prep ###
### and feature vector generating functions       ###
#####################################################

# SSP Identification -> produces plots
SSP_Input = pd.read_excel(str(path)+"\env.xlsx", sheet_name = "SSP")
#SSP_Grad = SSPGrad(SSP_Input, path, save = False)
#SSP_Stat = SSPStat(SSP_Input, path, plot = True, save = False)
SSP_Prop = SSPId(SSP_Input, str(path), plot = True, save = False)


#data = FeatDuct(ALLDATA, Input_Only = True)
#data = FeatBathy(data, path)
#data = FeatSSPVec(data, path)
#data_sspid = FeatSSPId(data, path, src_cond = True)
#data4 = FeatSSPStat(data3,path)
#data = FeatSSPOnDepth(data_sspid, path, save = False)
#data = pd.read_csv(str(path)+"\data_complete.csv")
#data_enc = EncodeData(data)
"""
target = 'num_rays'
features = data.columns.tolist()
features.remove(target)
features_enc = data_enc.columns.tolist()
features_enc.remove(target)
X, y = data_enc[features_enc], data_enc[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123, shuffle = True, stratify = y)
"""        
#y_pop = ClassImbalance(data, plot_density = True)
#plt.show()
#PlotCorrelation(data_enc,features_enc, annotate = False)
#print(y_pop)

# SMOT Upsampling
#X_smot, y_smot = SMOTSampling(X_train, y_train)
#X_smot.to_csv(str(path) + '/xgbsets/dataset_smot.csv')


"""
# Plot to see the effect of subsampling SSP for Grakn input
plot_sampling = False
if plot_sampling == True:
    fig, axes = plt.subplots(nrows = 3, ncols = 8, figsize = (15,20), sharey = True)
    axes = axes.flat
    axes[0].invert_yaxis()
    
    # This downsampling is used now in GRAKN. 0 is skipped in favor of first point at 10m
    # because grad at 0 is 0, which breaks the SSP-vec 'triplet'
    
    max_depth = [0, 50, 150, 250, 350, 450, 600, 750, 900, 1050, 1200, 1500]
    depth = SSP_Input.iloc[:,0]

    for i, ssp in enumerate(SSP_Input.iloc[:,1:]):
        axes[i].plot(SSP_Input.iloc[:,1:][ssp], depth, linewidth = 2, label = 'Sound Speed Profile' )
        axes[i].plot(SSP_Input.iloc[:,1:][ssp][depth.isin(max_depth)], depth[depth.isin(max_depth)], linewidth = 1, label = 'Subsampled Sound Speed Profile')
        axes[i].set_title("{}. {}".format(i, ssp))
"""
"""
# POLYFIT
best, allres = PolyfitSSP(SSP_Input)
deg = range(1,11)
z = np.array(SSP_Input.iloc[:,0]).astype(float)
znew = np.linspace(z[0], z[-1], num=len(z)*5)

#plt.plot(xnew,ffit,x,y)
fig, axes = plt.subplots(nrows = 3, ncols = 8, figsize = (15,20), sharey = True)
axes = axes.flat
axes[0].invert_yaxis()
for i, ssp in enumerate(SSP_Input.iloc[:,1:]):
    coeff = best[i][2]
    ffit = poly.polyval(znew, coeff) 
    
    axes[i].plot(ffit, znew)
    axes[i].plot(SSP_Input.iloc[:,i], z)
    axes[i].set_title("{}. {}".format(i, ssp))
"""