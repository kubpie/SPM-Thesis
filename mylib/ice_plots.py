# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:05:10 2020

@author: kubap
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from joblib import dump
from joblib import load
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from data_prep import FeatDuct, FeatBathy, FeatSSPVec, FeatSSPId, FeatSSPStat, FeatSSPOnDepth
from data_prep import LoadData, UndersampleData, SMOTSampling
from data_prep import CreateModelSplits, EncodeData
from data_analysis import PlotCorrelation, ICEPlot
from sklearn.model_selection import train_test_split
import os
from pathlib import Path


""""
A PDP is the average of the lines of an ICE plot.
Unlike partial dependence plots, ICE curves can uncover heterogeneous relationships.
PDPs can obscure a heterogeneous relationship created by interactions. 
PDPs can show you what the average relationship between a feature and the prediction looks like. This only works well if the interactions between the features for which the PDP is calculated and the other features are weak. In case of interactions, the ICE plot will provide much more insight.
"""
PATH = os.getcwd()
path = Path(PATH+"/data/")
#ALLDATA = LoadData(path
#data = FeatDuct(ALLDATA, Input_Only = True)
#data = FeatBathy(data, path)
#data = FeatSSPVec(data, path)
#data_sspid = FeatSSPId(data, path, src_cond = True)
#data = FeatSSPOnDepth(data_sspid, path, save = False)
data = pd.read_csv(str(path)+"\data_complete.csv")
data_enc = EncodeData(data)
data_enc = data_enc.fillna(0) #ICE plot func has problems with NaNs :(

target = 'num_rays'
features = data_enc.columns.tolist()
features.remove(target)
seasons = ['Autumn', 'Spring', 'Summer', 'Winter']
locations = ['Labrador-Sea', 'Mediterranean-Sea', 'North-Pacific-Ocean',
       'Norwegian-Sea', 'South-Atlantic-Ocean', 'South-Pacific-Ocean']
ice_features = [ feat for feat in features if feat not in locations + seasons ]
X, y = data_enc[features], data_enc[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123, shuffle = True, stratify = y)

"""
# Undersampling of the TEST SET to avoid overcrowding the ICE plot
# It reduces all clases to the size of the smallest class 7000: 20 samples
undersample = RandomUnderSampler(sampling_strategy='auto')
Xt_under, yt_under = undersample.fit_resample(Xt, yt)
Xt_under_df = pd.DataFrame(Xt_under, columns = features)
print(Counter(yt))
print(Counter(yt_under))
"""

### XGB MODEL SETUP
resultpath = Path(PATH+"/XGB/results/xgb_class/")
resultpath = str(resultpath) + '\\' 
model = load(resultpath+'xgb_class_final_model.dat')

### ICE PLOT FOR THE WHOLE DATASET
# Model 'fit' before 'predict' inside ICE plot function
model = model.fit(X_train.values, y_train.values)
# Ice plot for the whole dataset
ICEdict = ICEPlot(X_train, model, ice_features)
"""
### ICE PLOTS FOR SPLITS
SplitSets ,_ = CreateSplits(data_enc, level_out = 1, remove_outliers = True, replace_outliers = True, plot_distributions = False, plot_correlations = False)
for s,subset in enumerate(SplitSets):
    
    sub_features = subset.columns.tolist()
    sub_features.remove(target)
    ice_sub_features = [ feat for feat in sub_features if feat not in locations + seasons ]
    [dtrain, dtest] = TrainTestSplit(subset, test_size = 0.20) #reduced training/test split to 20% because smaller datasets

    Xs = dtrain[sub_features]
    ys = dtrain[target]
    Xst = dtest[sub_features]
    yst = dtest[target]

    eval_set = [(Xs.values, ys.values),(Xst.values, yst.values)]
    submodel_trained = model.fit(Xs.values, ys.values, eval_set=eval_set, eval_metric = feval, verbose=0, early_stopping_rounds = early_stop)
    results = submodel_trained.evals_result()
    print(f'Best iteration: {submodel_trained.best_iteration}\nF-score: {1-submodel_trained.best_score}')

    # Ice plot for the whole dataset
    ICEdict = ICEPlot(Xst, submodel_trained, ice_sub_features)

"""
# an attempt to split the plots further down on wdep_min
# there's a clear correlation between shallow channel and high ray nr
# however there are also not enough sample to create a separate model
"""

SplitSets_test, data_dist = CreateSplits(data_enc, level_out = 1, remove_outliers = False, replace_outliers = False, plot_distributions = False, plot_correlations = False)
split_neg2 = SplitSets_test[2]
split_0 = SplitSets_test[0]
split_neg2_shallow = split_neg2.loc[data['water_depth_min'] == 50] #only 50m shadowing problem
split_0_shallow = split_0.loc[data['water_depth_min'] == 50]


from collections import defaultdict
import operator
  
for split in SplitSets_test:
    ind_feature = 'water_depth_min'
    f_ind = np.unique(split[ind_feature]).tolist()
    f_dep = np.unique(split['num_rays']).tolist()
    calc = pd.DataFrame(np.zeros([len(split),len(f_ind)]), columns = f_ind)
    Fdict = dict.fromkeys(f_ind)
    Fmean = dict.fromkeys(f_ind)
    for fi in f_ind:
        value_vec = split.loc[split['water_depth_min'] == fi,'num_rays']
        yclass, ycount = np.unique(value_vec, return_counts=True)
        yper = ycount/sum(ycount)*100

        empty =  [ val for val in f_dep if val not in yclass ]
        empty_it = [f_dep.index(it) for it in empty]        
        ycount = ycount.tolist()
        yper = yper.tolist()
        for it in empty_it:
            ycount.insert(it,0)
            yper.insert(it,0)
        y_population = dict(zip(f_dep, zip(ycount, yper)))
        
        Fdict[fi] = y_population 
        Fmean[fi] = np.mean(value_vec)
    
    # TODO: Plot with 2 y-axes and mean value over the distributions
    fig,ax1 = plt.subplots()
    ax1.plot(list(Fmean.keys()), list(Fmean.values()), label = 'mean')
    ax1.set_title(['Mean num. rays vs ' + ind_feature])
    
    Rdict = defaultdict(list)
    for key in list(Fdict.keys()):
        for ray in list(Fdict[key].keys()):
            Rdict[ray].append(Fdict[key][ray][1])
    
    fig,ax = plt.subplots()
    for raynr in list(Rdict.keys()    ):
        ax.plot(list(Fdict.keys()), list(Rdict[raynr]), label = f'{raynr}')
    ax.legend()

"""