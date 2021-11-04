# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:18:29 2020

@author: Jakub Pietrak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from joblib import dump
from joblib import load
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.multiclass import OneVsRestClassifier

from xgb_mylib import ModelFit
from xgb_mylib import HyperParamGS
from xgb_mylib import PlotGS
from xgb_mylib import accuracy_rounding_score
from xgb_mylib import f1_rounding_score


# load data
filepath = r'C:\Users\kubap\Documents\THESIS\XGBoost\data\\'
resultpath = r'C:\Users\kubap\Documents\THESIS\XGBoost\results\tuning\\'

dtrain = pd.read_csv(filepath +'dtrain_new2.csv')
dtest = pd.read_csv(filepath +'dtest_new2.csv')
target = 'num_rays'
features = [x for x in dtrain.columns if x not in [target, 'criterion', 'residual','runID']]

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

# Tree regression model
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
       

##### HYPERPARAMETER TUNING #####
param1 = {
 'min_child_weight' : np.arange(0.0, 2.2, 0.2), # range: [0,∞] [default=1]
 'min_split_loss': np.arange(0.0, 2.2, 0.2) #range: [0,∞] [default=0]
 #alias: gamma 20 is extremly high
}
param2 = {
 'subsample': np.arange(1.0, 0.2, -0.2), #range: (0,1]  [default=1]
 'colsample_bytree': np.arange(1.0, 0.4, -0.1) # range: (0, 1] [default=1]
}
param3 = {
 'max_depth': np.arange(10,22,2),
 'reg_lambda':np.arange(0.1,4.0,0.3) #[default=1]
 #L2 reg - generally better than L1 unless solution is sparse
}
param4 = {
    'reg_alpha': np.arange(0.0, 1.1, 0.1)
}

param0 = {'subsample': np.arange(1.0, 0.7, -0.1),
            'colsample_bytree': np.arange(1.0, 0.8, -0.2)}

"""
it = 0
for param in [param4]:
    it = it + 1
    r_res, r_param = HyperParamGS(xgb_reg, param, dtrain, features, target, scoring_reg, refit = 'F1-macro')
    dump(r_res, resultpath + "reg_param_L1" + str(it) + ".dat")
    xgb_reg = xgb_reg.set_params(**r_param)
    PlotGS(r_res, param, scoring_reg, modeltype='reg_L1')

dump(xgb_reg, "L1_reg.dat")

it = 0
for param in [param4]:
    it = it + 1
    c_res, c_param = HyperParamGS(xgb_class, param, dtrain, features, target, scoring_class, refit = 'F1-macro')
    dump(c_res, resultpath + "class_param_L1" + str(it) + ".dat")
    xgb_class = xgb_class.set_params(**c_param)
    PlotGS(c_res, param, scoring_class, modeltype='class_L1')

dump(xgb_class, "L1_fine.dat")


#param = param3
#class_res, class_param = HyperParamGS(xgb_class, param, dtrain, features, target, scoring_class, refit = 'F1-macro')
#dump(class_res, resultpath + "param4_class.dat")
#xgb_class = xgb_class.set_params(**class_param)
#PlotGS(class_res, param, scoring_class)
"""


param_test = {
    'learning_rate': 0.05,
    'n_estimators': 2000
}
#xgb_reg = load(resultpath+"coarse_reg.dat")
#xgb_class = load(resultpath+"coarse_class.dat")

#xgb_class = xgb_class.set_params(**param_test)
#xgb_reg = xgb_reg.set_params(**param_test)

##### TRAIN THE BEST MODEL ON THE FULL DATASET AND FIT ON THE TEST SET #####
bst_model_class, fullresult_class, output_class = ModelFit(xgb_class, dtrain, dtest, features, target, early_stop = 100, verbose=True, learningcurve = True, importance = True, plottree = False, savename = "class_feat")
#bst_model_reg, fullresults_reg, output_reg  = ModelFit(xgb_reg, dtrain, dtest, features, target, early_stop = 200, verbose=True, learningcurve = True, importance = True, plottree = False, savename = False)
