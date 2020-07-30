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

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
#from sklearn.multiclass import OneVsRestClassifier

from xgb_mylib import ModelFit
from xgb_mylib import HyperParamGS
#from xgb_mylib import PlotGS
#from xgb_mylib import accuracy_rounding_score
#from xgb_mylib import f1_rounding_score
import time

# Load data and define paths
PATH = os.getcwd()
datapath = Path(PATH+"/data/")
resultpath = Path(PATH+"/XGB/results/")
sys.path.insert(1, PATH + '/mylib/')
from data_prep import LoadData, FeatDuct, EncodeData, FeatBathy, FeatSSPVec, FeatSSPId, FeatSSPOnDepth, SMOTSampling, CreateModelSplits
from data_analysis import ClassImbalance
DATA = LoadData(datapath)

###############################
### FEATURES REPRESENTATION ###
###############################
print('*** CREATING DATASETS ***')
#data = FeatDuct(DATA, Input_Only = True) #just to leave only input data
#data = FeatBathy(data, datapath) #also add slope length everywhere
#data_sspid = FeatSSPId(data, datapath, src_cond = True) #ssp identification algoritm, takes some time
#data_complete = FeatSSPOnDepth(data_sspid, datapath, save = False)
data_complete = pd.read_csv(str(datapath)+"\data_complete.csv")
data_enc = EncodeData(data_complete)

# ALWAYS leave out 20% of the whole dataset as test set that won't be used for tuning
size_test = 0.2
target  = 'num_rays'
best_model = xgb.XGBClassifier(
            silent = 0,
            learning_rate = 0.1,
            n_estimators= 100, #250
            max_depth= 10,
            min_child_weight=1.0, 
            min_split_loss=0.0,
            subsample= 0.8,
            colsample_bytree=0.8,
            reg_alpha = 0.0,
            reg_lambda= 1.0,
            objective= 'multi:softprob',
            #num_class = 17,
            n_jobs = -1)
model_type = 'xgb_class'

SplitSets, SplitDistributions =  CreateModelSplits(data_enc, level_out = .99,
         remove_outliers = True, replace_outliers = True,
         feature_dropout = True, plot_distributions = False, 
         plot_correlations = False)
#split = pd.concat([SplitSets[1],SplitSets[2]],axis=0)
split = SplitSets[1]
split_nr = '1'
#for split_nr, split, in enumerate(SplitSets):
features = split.columns.tolist()
features.remove(target)
X, y = split[features], split[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size_test, random_state = 123, shuffle = True, stratify = y)
ClassImbalance(split, plot=True)
#plt.show()
###############################
### MODEL TUNING & TRAINING ###
###############################

start_time_tuning=time.time() # time HS grid search and prediction

param_tuning = {
    'learning_rate': [0.1, 0.05, 0.03],
    'max_depth': [7, 8, 9],
    #'min_child_weight' : [1.0, 5.0, 10.0], # range: [0,∞] [default=1]
    'min_split_loss': [0., 0.5, 1.0], #range: [0,∞] [default=0]
    'subsample': [1.0, 0.9, 0.8],  #range: (0,1]  [default=1]
    'colsample_bytree': [1.0, 0.9, 0.8], # range: (0, 1] [default=1]
    'reg_lambda':[1.0, 3.0, 5.0], #[default=1]
    #'reg_alpha': [0., 0.5, 1.0] #[default=0]
} 
# 3^7 * 2 folds * 3 models = 13122 iterations

GS_results, best_params = HyperParamGS(best_model, X_train, y_train, model_type, param_tuning, cv = 2) #perform serach over param grid

dump(GS_results, f'{resultpath}\\splits\\{split_nr}\\GSCV_results.dat')
dump(best_params, f'{resultpath}\\splits\\{split_nr}\\best_params.dat')

end_time_tuning = time.time() - start_time_tuning
print(f'-> Elapsed time for HP tuning: {end_time_tuning}')    
# 2. Model training, validation and prediction on left-out test set
print(f'\n*** MODEL TRAINING, VALIDATION & TESTING {split_nr} ***')
start_time_training = time.time()
increase_learning = {
    'n_estimators': 500 # 500  increase nr of estimators
}
# use only for testing the loop 
#best_params = load(f'{resultpath}\\splits\\{split_nr}\\best_params.dat') #load from previous run
tuned_model = best_model.set_params(**best_params)
tuned_model = tuned_model.set_params(**increase_learning)

final_model, train_results, pred_output = ModelFit(tuned_model, model_type, 
            X_train, y_train, X_test, y_test, 
            early_stop=500, 
            cv = 0,
            split_nr = split_nr,
            val_size = 0.2,
            learningcurve = True, 
            importance = True, 
            plottree = True, 
            savemodel = True,
            verbose = 1)
dump(train_results, f'{resultpath}\\splits\\{split_nr}\\training_results.dat')
dump(pred_output, f'{resultpath}\\splits\\{split_nr}\\prediction_results.dat')

end_time_training= time.time() - start_time_training
print(f'-> Elapsed time for training, validation & testing: {end_time_training}')

