# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:18:29 2020

@author: Jakub Pietrak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import os, sys
from pathlib import Path
from joblib import dump
from joblib import load

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
#from sklearn.multiclass import OneVsRestClassifier

#from xgb_mylib import ModelFit
#from xgb_mylib import HyperParamGS
#from xgb_mylib import PlotGS
from xgb_mylib import accuracy_rounding_score
from xgb_mylib import f1_rounding_score

# Load data and define paths
PATH = os.getcwd()
datapath = Path(PATH+"/data/")
resultpath = Path(PATH+"/XGB/results/")
sys.path.insert(1, PATH + '/mylib/')
from data_prep import LoadData, FeatDuct, EncodeData, FeatBathy, FeatSSPVec, FeatSSPId, FeatSSPStat, FeatSSPOnDepth
DATA = LoadData(datapath)
TARGET = 'num_rays' # target variable label

"""
##### HYPERPARAMETER TUNING #####

### model complexity ###
# max_depth: maximum depth of a tree. 
# min_child_weight: minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. 
# In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be.
# minimum_split_loss: reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.
param1 = {
 'max_depth': np.arange(10,22,2),
 'min_child_weight' : np.arange(0.0, 2.2, 0.4), # range: [0,∞] [default=1]
 'min_split_loss': np.arange(0.0, 2.2, 0.4) #range: [0,∞] [default=0]
}

### overfitting ###
# subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting.
# colsample_bytree is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
param2 = {
 'subsample': np.arange(1.0, 0.2, -0.2), #range: (0,1]  [default=1]
 'colsample_bytree': np.arange(1.0, 0.4, -0.1) # range: (0, 1] [default=1]
}

### regularization ###
# lambda: L2 regularization term on weights. Increasing this value will make model more conservative.
# alpha: L1 regularizationterm on weights. Increasing this value will make model more conservative.
# L2 reg - generally better than L1 unless solution is sparse
param4 = {
    'reg_lambda':np.arange(0.1,4.0,0.3), #[default=1]
    'reg_alpha': np.arange(0.0, 1.1, 0.1) #[default=0]
}
"""
# Classifier model   
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
    num_class = 17,
    n_jobs = -1
)
scoring_class = {
    'Accuracy': make_scorer(accuracy_score),
    'F1-macro': make_scorer(f1_score, average='macro')
    }
"""
# Regression model
xgb_reg = xgb.XGBRegressor(
    silent = 0,
    learning_rate = 0.05, #0.1
    n_estimators=1000,#4000,
    max_depth = 10, 
    min_child_weight= 1.,
    min_split_loss= 0.,
    subsample = 1.0,
    colsample_bytree=1.0,
    objective= 'reg:squarederror',
    reg_alpha = 0.,
    reg_lambda= 1.,
    seed=27,
    n_jobs = -1
) 
scoring_reg = {
    'Accuracy': make_scorer(accuracy_rounding_score),
    'F1-macro': make_scorer(f1_rounding_score, average='macro')
    }
"""
param_test = {
    'min_child_weight' : np.arange(0.0, 1.5, 0.5),
    'min_split_loss': np.arange(0.0, 1.5, 0.5)
}

### 1. XGB wihout splits
data = FeatDuct(DATA, Input_Only = True) #just to leave only input data
data = FeatBathy(data,datapath)
data = EncodeData(data)
features = data.columns.tolist()
features.remove(TARGET)
##################
###  PIPELINE  ###
##################
# `outer_cv` creates K folds for estimating generalization model error
outer_cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
# when we train on a certain fold, use a second cross-validation split in order to choose best hyperparameters
inner_cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)

X, y = data[features], data[TARGET]


#[dtrain, dtest] = TrainTestSplit(data_enc, test_size = 0.25)
#_, _, _, = ModelFit(model, dtrain, dtest, features, target, early_stop = 100,
#verbose=True, learningcurve = True, importance = True, plottree = False, savename = False)

