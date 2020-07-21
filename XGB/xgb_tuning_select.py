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

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate, cross_val_score, GridSearchCV
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
        n_estimators=50, #1000
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

        'scoring_i''f1_macro'),
    
    # Regressesor model
    'xgb_reg':(
        xgb.XGBRegressor(
        silent = 0,
        learning_rate = 0.1,
        n_estimators=50,
        max_depth = 10, 
        min_child_weight= 1.,
        min_split_loss= 0.0,
        subsample = 1.0,
        colsample_bytree=1.0,
        objective= 'reg:squarederror',
        reg_alpha = 0.0,
        reg_lambda= 1.0,
        n_jobs = -1), 

        'neg_root_mean_squared_error')
        #{#'Accuracy': make_scorer(accuracy_rounding_score),
        #'F1-macro': make_scorer(f1_rounding_score, greater_is_better = True, average='macro')})
}

param_test = {
    'max_depth': np.arange(5,20,5),
    'min_child_weight' : np.arange(0.0, 1.0, 0.5),
    'min_split_loss': np.arange(0.0, 1.0, 0.5)
}
###############################
### FEATURES REPRESENTATION ###
###############################
data = FeatDuct(DATA, Input_Only = True) #just to leave only input data
data = FeatBathy(data, datapath) #also add slope length everywhere
datasspid = FeatSSPId(data, datapath, src_cond = True)
datasets = {
    'data-sspcat': (data,[]),                        # 1. categorical ssp
    'data-sspvec': (FeatSSPVec(data, datapath),[])   # 2. ssp vector + categorical
    #'data-sspid':  (FeatSSPOnDepth(datasspid, datapath, save = False),[]) #ssp_id + categorical + selected_depths
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
        datasets[setname][1].append([X_train, X_test, y_train, y_test])
   
##################
###  PIPELINE  ###
##################

# `outer_cv` creates K folds for estimating generalization model error
outer_cv = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 42)
# when we train on a certain fold, use a second cross-validation split in order to choose best hyperparameters
inner_cv = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 24)

estimators_and_average_scores_across_outer_folds = {
    'xgb_class': {'names':[],'scores':[], 'parameters':[]},
    'xgb_reg': {'names':[],'scores':[], 'parameters':[]}
}

print('Evaluating feature-selection dependent generalisation error in nested CV setup')
for modeltype, (model, scorer) in models_and_scorers.items():

    hyperparam_optimizer = GridSearchCV(estimator = model,
                            param_grid = param_test, n_jobs=-1, 
                            cv=inner_cv, scoring=scorer, verbose = 1)
                            #return_train_score=True)
    
    for setname, (setsamples,xgbsets) in datasets.items():
        name = modeltype + '_' + setname
        X_train = xgbsets[0][0]
        y_train = xgbsets[0][2]

        #scores_across_outer_folds = cross_val_score(
        #                        hyperparam_optimizer,
        #                        X_train, y_train, cv=outer_cv, scoring=scorer,
        #                        verbose = 1)
        outer_cv_scores = cross_validate(
                                hyperparam_optimizer, 
                                X_train, y_train, cv=outer_cv, scoring=scorer,
                                verbose = 1, return_estimator=True)

        # gather labelled results in the dictionary of scores & estimators
        estimators_and_average_scores_across_outer_folds[modeltype]['names'].append(name)
        # get the mean MSE across each of outer_cv's 3 folds
        scores_across_outer_folds = outer_cv_scores['test_score']
        estimators_and_average_scores_across_outer_folds[modeltype]['scores'].append(np.mean(scores_across_outer_folds))
        if modeltype == 'xgb_class':
            error_summary = 'Model: {name}\nMean F1-macro score in the outer folds: {scores}.\nAverage score: {avg} (+/-{std})'
            print(error_summary.format(
                name=name, scores=scores_across_outer_folds,
                avg=np.mean(scores_across_outer_folds),
                std=np.std(scores_across_outer_folds)))
        else:
            error_summary = 'Model: {name}\nMean RMSE score in the outer folds: {scores}.\nAverage error: {avg} (+/-{std})'
            print(error_summary.format(
                name=name, scores=scores_across_outer_folds,
                avg=np.mean(scores_across_outer_folds),
                std=np.std(scores_across_outer_folds)))

        # get a guess for an estimator with parameters tuned in the nested CV procedure
        best_iter = np.argmax(scores_across_outer_folds)        
        nested_cv_estimator = outer_cv_scores['estimator'][best_iter]
        estimators_and_average_scores_across_outer_folds[modeltype]['parameters'].append(nested_cv_estimator.best_params_)

best_models_nested_CV = []
best_scores_nested_CV = []
best_params_guess_nested_CV = []
# Due to different metrics best regression and classification models need to be chosen separately beased on min\max
for modeltype in ['xgb_class','xgb_reg']:
    print('\nAverage score across the outer folds: ',
    (estimators_and_average_scores_across_outer_folds[modeltype]['names'], 
    estimators_and_average_scores_across_outer_folds[modeltype]['scores']),
    )
    #estimators_and_average_scores_across_outer_folds[modeltype]['parameters']))
    best_idx = np.argmax(estimators_and_average_scores_across_outer_folds[modeltype]['scores'])
    best_models_nested_CV.append(estimators_and_average_scores_across_outer_folds[modeltype]['names'][best_idx])
    best_scores_nested_CV.append(estimators_and_average_scores_across_outer_folds[modeltype]['scores'][best_idx])
    best_params_guess_nested_CV.append(estimators_and_average_scores_across_outer_folds[modeltype]['parameters'][best_idx])

#TODO: Plot stability? Save results?

for (best_model_name, best_model_avg_score) in zip(best_models_nested_CV, best_scores_nested_CV):
    print(f'Best model: {best_model_name}')
    print(f'Estimation of its generalization error: {best_model_avg_score:.3}')

    best_model_dataset_name = best_model_name.split('_')[-1]
    best_model_data = datasets[best_model_dataset_name][1]
    X_train_best, y_train_best = best_model_data[0][0], best_model_data[0][2]
    X_test_best, y_test_best = best_model_data[0][1], best_model_data[0][3]

    # now we refit this best model on the whole train dataset so that we can start
    # making predictions on other data, and now we have a reliable estimate of
    # this model's generalization error and we are confident this is the best model
    # among the ones we have tried

    ### NORMAL CROSS VALIDATION WITH FULL PARAMETER GRID & CUSTOM METRICS
    # at this step rounded F1 will be implemented for a regression model to help 
    # with comparison against classifier
    
    #TODO: 
    # 1. Implement custom scorers here
    # 2. Params guess from nested-CV??
    # 3. Increase nr of estimators and implement early stopping

    final_model = GridSearchCV(best_model, param_test, cv = inner_cv, n_jobs=-1)
    final_model.fit(X_train_best, y_train_best)

    print('Best parameter choice for this model: \n\t{params}'
        '\n(according to cross-validation `{cv}` on the whole dataset).'.format(
        params=final_model.best_params_, cv=inner_cv))


#final_model.predict(X_test_best)
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")