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

PATH = os.getcwd()
path = Path(PATH+"/data/")
ALLDATA = LoadData(path)

#####################################################
### Messy script for testing & plotting data prep ###
### and feature vector generating functions       ###
#####################################################
"""
# SSP Identification
SSP_Input = pd.read_excel(path+"env.xlsx", sheet_name = "SSP")
SSP_Grad = SSPGrad(SSP_Input, path, save = False)
SSP_Stat = SSPStat(SSP_Input, path, plot = True, save = False)
SSP_Prop = SSPId(SSP_Input, path, plot = True, save = False)
"""

#data = FeatDuct(ALLDATA, Input_Only = True)
#data = FeatBathy(data, path)
#data = FeatSSPVec(data, path)
#data_sspid = FeatSSPId(data, path, src_cond = True)
#data4 = FeatSSPStat(data3,path)
#data = FeatSSPOnDepth(data_sspid, path, save = False)
data = pd.read_csv(str(path)+"\data_complete.csv")
data_enc = EncodeData(data)

target = 'num_rays'
features = data.columns.tolist()
features.remove(target)
features_enc = data_enc.columns.tolist()
features_enc.remove(target)
X, y = data_enc[features_enc], data_enc[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123, shuffle = True, stratify = y)
"""        
y_pop = ClassImbalance(data, plot = True)
PlotCorrelation(data_enc,features_enc, annotate = False)
print(y_pop)

# SMOT Upsampling
#X_smot, y_smot = SMOTSampling(X_train, y_train)
#X_smot.to_csv(str(path) + '/xgbsets/dataset_smot.csv')
"""

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

# ICE PLOTS
#plot_features = features[0:5] + features[6:10]
#target = {np.unique(y)[0]:300, np.unique(y)[1]:300, np.unique(y)[2]:300}

# Undersampling to uncrowd the ICE plot
#undersample = UndersampleData()
#Xdf = pd.DataFrame(X_train, columns = features)
"""
model = xgb.XGBClassifier(
            silent = 0,
            learning_rate = 0.1,
            n_estimators= 250, #250
            max_depth= 10,
            min_child_weight=1.0, 
            min_split_loss=0.0,
            subsample= 0.8,
            colsample_bytree=0.8,
            reg_alpha = 0.0,
            reg_lambda= 1.0,
            objective= 'multi:softprob',
            num_class = 17,
            n_jobs = -1)
best_params = load(resultpath + 'best_params.dat')
model = model.set_params(**best_params)
"""
resultpath = Path(PATH+"/XGB/results/xgb_class/")
resultpath = str(resultpath) + '\\' 
model = load(resultpath+'xgb_class_final_model.dat')

# create dict of ICE data for grid of ICE plots
plot_features  = features_enc[:20]
#Xdf =  pd.DataFrame(X_train.iloc[:,:20], columns = plot_features)
#Xdf.sort_index(axis = 1, inplace = True)
model.fit(X_train,y_train)
ICEdict = ICEPlot(X_train, model, plot_features)

train_ice_dfs = {feat: ice(data=Xdf, column=feat, predict=model.predict) 
                 for feat in plot_features}

fig = plot_ice_grid(train_ice_dfs, Xdf, plot_features,
                    ax_ylabel='Pred. Ray Num', alpha=0.3, plot_pdp=True,
                    pdp_kwargs={'c': 'blue', 'linewidth': 3},
                    linewidth=0.5, c='dimgray')
fig.tight_layout()
fig.suptitle('ICE plot: XGBClassifier')
fig.subplots_adjust(top=0.89)