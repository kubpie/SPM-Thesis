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

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, GridSearchCV
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
from data_prep import LoadData, XGBSets, FeatDuct, EncodeData, FeatBathy, FeatSSPVec, FeatSSPId, FeatSSPOnDepth
DATA = LoadData(datapath)

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
### for tuning all at once ### 
param_all = {
    'max_depth': np.arange(10,22,2),
    'min_child_weight' : np.arange(0.0, 2.2, 0.4), # range: [0,∞] [default=1]
    'min_split_loss': np.arange(0.0, 2.2, 0.4), #range: [0,∞] [default=0]
    'colsample_bytree': np.arange(1.0, 0.4, -0.1) # range: (0, 1] [default=1]
    'reg_lambda':np.arange(0.1,4.0,0.3), #[default=1]
}
"""
models_and_scorers = {
    # Classifier model
    'xgb_class':(
        xgb.XGBClassifier(
        silent = 0,
        learning_rate = 0.1, #0.1
        n_estimators=100, #1000
        max_depth=10, #10
        min_child_weight=1.0, 
        min_split_loss=0.0,
        subsample= 1.0,
        colsample_bytree=1.0,
        reg_alpha = 0.0,
        reg_lambda= 1.0,
        objective= 'multi:softprob',
        num_class = 17,
        n_jobs = -1),

        'f1_macro'),
    
    # Regressesor model
    'xgb_reg':(
        xgb.XGBRegressor(
        silent = 0,
        learning_rate = 0.1,
        n_estimators=100,
        max_depth = 10, 
        min_child_weight= 1.,
        min_split_loss= 0.0,
        subsample = 1.0,
        colsample_bytree=1.0,
        objective= 'reg:squarederror',
        reg_alpha = 0.,
        reg_lambda= 1.,
        n_jobs = -1), 

        'mean_squared_error')
        #{#'Accuracy': make_scorer(accuracy_rounding_score),
        #'F1-macro': make_scorer(f1_rounding_score, greater_is_better = True, average='macro')})
}

param_test = {
    'min_child_weight' : np.arange(0.0, 1.5, 0.5),
    'min_split_loss': np.arange(0.0, 1.5, 0.5)
}
###############################
### FEATURES REPRESENTATION ###
###############################
data = FeatDuct(DATA, Input_Only = True) #just to leave only input data
data = FeatBathy(data, datapath) #also add slope length everywhere
datasspid = FeatSSPId(data, datapath, src_cond = True)
datasets = {
    'data_sspcat': (data,[]),                        # 1. categorical ssp
    'data_sspvec': (FeatSSPVec(data, datapath),[]),   # 2. ssp vector + categorical
    'data_sspid':  (FeatSSPOnDepth(datasspid, datapath, save = False),[]) #ssp_id + categorical + selected_depths
}

# ALWAYS leave out 20% of the whole dataset as test set that won't be used for tuning
size_test = 0.2
target  = 'num_rays'
for setname, (setsamples,xgbsets) in datasets.items():
        dataset = EncodeData(setsamples)
        features = dataset.columns.tolist()
        features.remove(target)
        X, y = dataset[features], dataset[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size_test, random_state = 123, shuffle = True, stratify = y)
        xgbsets.append(X_train, X_test, y_train, y_test)
   

##################
###  PIPELINE  ###
##################

# `outer_cv` creates K folds for estimating generalization model error
outer_cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
# when we train on a certain fold, use a second cross-validation split in order to choose best hyperparameters
inner_cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 24)

average_scores_across_outer_folds_for_each_model = dict()

for name, (model, scorer) in models_and_scorers.items():

    hyperparam_optimizer = GridSearchCV(estimator = model, #verbose = 2,
                            param_grid = param_test, n_jobs=-1, 
                            cv=inner_cv, scoring=scorer)
                            #return_train_score=True)
    for setname, (setsamples,xgbsets) in datasets.items():

        name = name + setname
        X_train = xgbsets[0,:]
        y_train = xgbsets[2,:]

        scores_across_outer_folds = cross_val_score(
                                hyperparam_optimizer,
                                X_train, y_train, cv=outer_cv, scoring=scorer)

        # get the mean MSE across each of outer_cv's 3 folds
        average_scores_across_outer_folds_for_each_model[name] = np.mean(scores_across_outer_folds)
        error_summary = 'Model: {name}\nMean F1-macro score in the outer folds: {scores}.\nAverage error: {avg}'
        print(error_summary.format(
            name=name, scores=scores_across_outer_folds,
            avg=np.mean(scores_across_outer_folds)))
        print()


print('Average score across the outer folds: ',
      average_scores_across_outer_folds_for_each_model)
