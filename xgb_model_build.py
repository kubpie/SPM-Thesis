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
from data_prep import LoadData, FeatDuct, EncodeData, CreateSplits, TrainTestSplit


# XGBOOST DATABASE PROCESSING (incl. feature selection)
# load data
import os
path = os.getcwd()+'\data\\'
rawdata = LoadData(path)
data = FeatDuct(rawdata, Input_Only = True) #just to leave only input data
y = data['num_rays']


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


data_enc = EncodeData(data)
SplitSets, data_dist = CreateSplits(data_enc, level_out = 1, replace=True, plot_distributions = False, plot_correlations = False)

for subset in SplitSets:
    
    [dtrain, dtest] = TrainTestSplit(subset, test_size = 0.25)
    target = 'num_rays'
    features = subset.columns.tolist()
    features.remove(target)
    bst_model, fullresult_class, output_class = \
    ModelFit(xgb_class, dtrain, dtest, features, target, early_stop = 100,
    verbose=True, learningcurve = True, importance = True, plottree = False, savename = False)
