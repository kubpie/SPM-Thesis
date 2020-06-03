# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 20:34:47 2020

@author: kubap
"""
import numpy as np
import pandas as pd
import seaborn as sns
from os import listdir

import xgboost as xgb
from xgb_mylib import ModelFit
from data_prep import LoadData, FeatDuct, EncodeData, CreateSplits, TrainTestSplit, FeatBathy, FeatSSP, FeatSSPId


# XGBOOST DATABASE PROCESSING (incl. feature selection)
# load data
import os
path = os.getcwd()+'\data\\'
rawdata = LoadData(path)
data = FeatDuct(rawdata, Input_Only = True) #just to leave only input data
target = 'num_rays'

xgb_class = xgb.XGBClassifier(
        silent = 0,
        learning_rate = 0.1, #0.05
        n_estimators=1000, #3000
        max_depth=10, #9 updated 
        min_child_weight=0.0, #1 updated
        min_split_loss=0.0, #0 updated
        subsample= 1.0,
        colsample_bytree=1.0,
        reg_alpha = 0.0,
        reg_lambda= 1.0,
        objective= 'multi:softprob',
        seed=27,
        n_jobs = -1
        )


""" ORDER OF ACTIONS """"""
1. add missing bathy features
2. decide to keep profile (name) as categorical feat or replace by other representation of ssp
3a. if keeping the profile:
    - add other features like SSP-id
    - encode the profile with one-hot-encoder 
3b. if replacing profile with ssp:
    - make sure to remove the profile inside FeatSSP() function
    - no need to encode, because no more categorical vars
4. update feature column name list:    
    features = data_enc.columns.tolist()
    features.remove(target)
5. decide if the problem is split into subproblems based on slope value
6a. for no splits:
    - create train\test split  
    - run ModelFit
6b. run CreateSplits:
    - the script will take care of fixing outliers
    - and remove redundant features
    - you can plot data correlations from within that script
    - create subsets and create train\test split for subsets 
7. (optional) Plot feature correlations for the data file PlotCorrelation(dat, features, annotate = True))
    - for splits, plot from CreateSplits() function

"""

# Adding Bathy features for each - filling missing info
data = FeatBathy(data, path)


### 1. XGB wihout splits
data_enc = EncodeData(data)
features = data_enc.columns.tolist()
features.remove(target)
[dtrain, dtest] = TrainTestSplit(data_enc, test_size = 0.25)
_, _, _, = ModelFit(xgb_class, dtrain, dtest, features, target, early_stop = 100,
verbose=True, learningcurve = True, importance = True, plottree = False, savename = False)

### 2. XGB with SSP-vec directly in feature vector (implicit SSP features) & missin bathy info 
data_ssp = FeatSSP(data, path)
features = data_ssp.columns.tolist()
features.remove(target)
[dtrain, dtest] = TrainTestSplit(data_ssp, test_size = 0.25)

_, _, _, = ModelFit(xgb_class, dtrain, dtest, features, target, early_stop = 100,
verbose=True, learningcurve = True, importance = True, plottree = False, savename = False)

### XGB with SSP-id and without SSP-vec
# 3A. Without checking src depth condition
data_sspid_noc = FeatSSPId(data, path, src_cond = False)
sspid_enc = EncodeData(data_sspid_noc)
features = sspid_enc.columns.tolist()
features.remove(target)
[dtrain, dtest] = TrainTestSplit(sspid_enc, test_size = 0.25)
_, _, _, = ModelFit(xgb_class, dtrain, dtest, features, target, early_stop = 100,
verbose=True, learningcurve = True, importance = True, plottree = False, savename = False)

# 3B. With src condition => full acoustic duct identification for each scenario
data_sspid_con = FeatSSPId(data, path, src_cond = True)
sspid_enc = EncodeData(data_sspid_con)
features = sspid_enc.columns.tolist()
features.remove(target)
[dtrain, dtest] = TrainTestSplit(sspid_enc, test_size = 0.25)
_, _, _, = ModelFit(xgb_class, dtrain, dtest, features, target, early_stop = 100,
verbose=True, learningcurve = True, importance = True, plottree = False, savename = False)

# 4. sspvec + sspid with duct check
data_sspid_con = FeatSSPId(data, path, src_cond = True)
data_ssp = FeatSSP(data_sspid_con, path)
features = data_ssp.columns.tolist()
features.remove(target)
[dtrain, dtest] = TrainTestSplit(data_ssp, test_size = 0.25)

_, _, _, = ModelFit(xgb_class, dtrain, dtest, features, target, early_stop = 100,
verbose=True, learningcurve = True, importance = True, plottree = False, savename = False)

""""
### XGB with sub-problem splits on slope
SplitSets, data_dist = CreateSplits(data_enc, level_out = 1, remove_outliers = True, replace_outliers = True, plot_distributions = False, plot_correlations = False)
for subset in SplitSets:
    
    [dtrain, dtest] = TrainTestSplit(subset, test_size = 0.25)
    
    sub_features = subset.columns.tolist()
    sub_features.remove(target)
    _, _, _, = ModelFit(xgb_class, dtrain, dtest, sub_features, target, early_stop = 100,
    verbose=True, learningcurve = True, importance = True, plottree = False, savename = False)
"""
