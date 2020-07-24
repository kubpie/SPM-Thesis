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
from data_prep import LoadData, FeatDuct, EncodeData, FeatBathy, FeatSSPVec, FeatSSPId, FeatSSPOnDepth, SMOTSampling
DATA = LoadData(datapath)

"""
##### HYPERPARAMETER TUNING #####

### model complexity ###
# max_depth: maximum depth of a tree. 
# min_child_weight: minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. 
# In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be.
b# minimum_split_loss (gamma):  pseudo-regularization hyperparameter in gradient boosting, reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.
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
    'learning_rate': [0.1, 0.01],
    'max_depth': [6, 8, 10, 12],
    'min_child_weight' : [1, 5, 10], # range: [0,∞] [default=1]
    'min_split_loss': [0, 1, 5], #range: [0,∞] [default=0]
    'subsample': [1, 0.8],  #range: (0,1]  [default=1]
    'colsample_bytree': [1, 0.8] # range: (0, 1] [default=1]
    'reg_lambda':[1, 5, 10], #[default=1]
}
"""
models_and_scorers = {
    # Classifier model
    'xgb_class':(
        xgb.XGBClassifier(
        silent = 0,
        learning_rate = 0.1, #0.1
        n_estimators= 200, #change to at least 100 #TODO: FIX PARAM BEFORE RUN!!
        max_depth= 5, #10 #TODO: FIX PARAM BEFORE RUN!!
        min_child_weight=1.0, 
        min_split_loss=0.0,
        subsample= 0.8,
        colsample_bytree=0.8,
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
        n_estimators=200, #change to at least 100 #TODO: FIX PARAM BEFORE RUN!!
        max_depth = 5, #TODO: FIX PARAM BEFORE RUN!!
        min_child_weight= 1.,
        min_split_loss= 0.0,
        subsample = 0.8,
        colsample_bytree=0.8,
        objective= 'reg:squarederror',
        reg_alpha = 0.0,
        reg_lambda= 1.0,
        n_jobs = -1), 

        'neg_root_mean_squared_error')
}
# parameters for model complexity + learning rate to control overfitting
# also by default subsample and colsample_bytree set to 0.8 to add randomness
param_test = {
    'max_depth': [8, 12],
    'min_split_loss': [0, 5],
    'min_child_weight' : [1, 5],
    'learning_rate': [0.1, 0.01]
}

timelist = [] #all training times are going to be gathered for performance eval.

###############################
### FEATURES REPRESENTATION ###
###############################
print('Creating datasets for evaluating feature selection.')
data = FeatDuct(DATA, Input_Only = True) #just to leave only input data
data = FeatBathy(data, datapath) #also add slope length everywhere
data_sppvec = FeatSSPVec(data, datapath)
data_sspid = FeatSSPId(data, datapath, src_cond = True) #ssp identification algoritm, takes some time
data_complete = FeatSSPOnDepth(data_sspid, datapath, save = False)
datasets = {
    'data-sspcat': (data,[]),                          # 1. categorical ssp
    'data-sspvec': (data_sppvec,[]),                   # 2. ssp vector + categorical
    'data-sspid':  (data_complete,[]),                 # 3. ssp_id + categorical + selected_depths
    'data-sspid-upsampled-100': (data_complete,[]),    # 4. ssp_id + categorical + selected_depth + upsampling in minority class
    'data-sspid-upsampled-200': (data_complete,[])     # 5. same as above but minority upsampled to 300
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
        if setname == 'data-sspid-upsampled-100':
            X_train, y_train = SMOTSampling(X_train, y_train, min_class_size = 100)
        elif setname == 'data-sspid-upsampled-200':
            X_train, y_train = SMOTSampling(X_train, y_train, min_class_size = 200)
        datasets[setname][1].append([X_train, X_test, y_train, y_test])

##########################
###  FEATURE SELECTION ###
##########################

# start time for nested CV
start_time=time.time()

Kin = 5 # number of k-folds in inner CV
Kout = 5 # number of k-folds in outer CV

# `outer_cv` creates K folds for estimating generalization model error
outer_cv = StratifiedKFold(n_splits = Kout, shuffle = True, random_state = 42)
# when we train on a certain fold, use a second cross-validation split in order to choose best hyperparameters
inner_cv = StratifiedKFold(n_splits = Kin, shuffle = True, random_state = 24)

estimators_and_average_scores_across_outer_folds = {
    'xgb_class': {'names':[],'scores':[], 'parameters':[]},
    'xgb_reg': {'names':[],'scores':[], 'parameters':[]}
}

best_models_nested_CV = []
best_scores_nested_CV = []
best_params_guess_nested_CV = []

for modeltype, (model, scorer) in models_and_scorers.items():

    print(f'\n*** FEATURE SELECTION & GENERALISATION ERROR EVALUATION {modeltype} ***')
    
    hyperparam_optimizer = GridSearchCV(estimator = model,
                            param_grid = param_test, n_jobs=-1, 
                            cv=inner_cv, scoring=scorer, verbose = 1)
                            
    nested_means = []
    nested_sds = []
    test_scores_per_folds = np.zeros([len(datasets),Kout]) 
    
    it = 0
    for setname, (setsamples,xgbsets) in datasets.items():
        name = modeltype + '_' + setname
        X_train = xgbsets[0][0]
        y_train = xgbsets[0][2]

        outer_cv_scores = cross_validate(
                                hyperparam_optimizer, 
                                X_train, y_train, cv=outer_cv, scoring=scorer,
                                verbose = 1, return_estimator=True)

        # gather labelled results in the dictionary of scores & estimators
        estimators_and_average_scores_across_outer_folds[modeltype]['names'].append(name)
        # get the mean MSE across each of outer_cv's k folds
        scores_across_outer_folds = outer_cv_scores['test_score']
        if modeltype == 'xgb_class':
            estimators_and_average_scores_across_outer_folds[modeltype]['scores'].append(np.mean(scores_across_outer_folds))
            error_summary = '\nModel: {name}\nMean F1-macro score in the outer folds: {scores}\nAverage score: {avg} (+/-{std})\n'
            print(error_summary.format(
                name=name, scores=scores_across_outer_folds,
                avg=np.mean(scores_across_outer_folds),
                std=np.std(scores_across_outer_folds)))
        else:
            estimators_and_average_scores_across_outer_folds[modeltype]['scores'].append(np.mean(scores_across_outer_folds))
            error_summary = '\nModel: {name}\nMean RMSE score in the outer folds: {scores}.\nAverage score in all folds: {avg} (+/-{std})\n'
            print(error_summary.format(
                name=name, scores=[abs(score) for score in scores_across_outer_folds],
                avg=abs(np.mean(scores_across_outer_folds)),
                std=np.std(scores_across_outer_folds)))

        # get a guess for an estimator HPs (gimodel complexity) that were tuned in the nested CV procedure
        best_iter = np.argmax(scores_across_outer_folds)        
        nested_cv_estimator = outer_cv_scores['estimator'][best_iter]
        estimators_and_average_scores_across_outer_folds[modeltype]['parameters'].append(nested_cv_estimator.best_params_)
        
        # track stability of nested CV
        nested_means.append(np.mean(scores_across_outer_folds))
        nested_sds.append(np.std(scores_across_outer_folds))
        test_scores_per_folds[it,:] = scores_across_outer_folds
        it +=1
    
    
    best_idx = np.argmax(estimators_and_average_scores_across_outer_folds[modeltype]['scores'])
    best_models_nested_CV.append(estimators_and_average_scores_across_outer_folds[modeltype]['names'][best_idx])
    best_scores_nested_CV.append(estimators_and_average_scores_across_outer_folds[modeltype]['scores'][best_idx])
    best_params_guess_nested_CV.append(estimators_and_average_scores_across_outer_folds[modeltype]['parameters'][best_idx])

    # save results of nested CV in case of pipeline crash \ redos
    best_nested = {'best_models_nested_CV': estimators_and_average_scores_across_outer_folds[modeltype]['names'][best_idx],
        'best_scores_nested_CV': estimators_and_average_scores_across_outer_folds[modeltype]['scores'][best_idx],
        'best_params_guess_nested_CV': estimators_and_average_scores_across_outer_folds[modeltype]['parameters'][best_idx]}
    dump(estimators_and_average_scores_across_outer_folds[modeltype], f'{resultpath}\\{modeltype}\\nested_scores_and_models.dat')
    dump(best_nested, f'{resultpath}\\{modeltype}\\best_nested_score_and_model.dat')


    # plot stability of nested CV
    fig, ax = plt.subplots(figsize=(10,7))
    #ax.fill_between(range(len(nested_means)), np.array(nested_means)-np.array(nested_sds), np.array(nested_means)+np.array(nested_sds), facecolor='lightblue', label = 'score std. dev.')
    for p in range(1, np.size(test_scores_per_folds,0)+1):
        ax.scatter(p*np.ones(np.size(test_scores_per_folds,1)), test_scores_per_folds[p-1,:], color='r', label='Outer K-fold Scores')
    ax.plot(range(1, np.size(test_scores_per_folds,0)+1), nested_means, linewidth = 1, color = 'k', label = 'Mean Score')
    xticklabels = ['sspcat','sspevec','sspid','ups-100', 'ups-200']
    ax.set_xticks(range(1, np.size(test_scores_per_folds,0)+1))
    ax.set_xticklabels(xticklabels, rotation = 45)
    #ax.set_xlabel('Dataset Index')
    ax.set_ylabel(f'Score {scorer}')
    ax.set_title(f'{modeltype} Feature Selection: Generalisation Error Per Dataset')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[0],handles[-1]], [labels[0],labels[-1]], loc = 'best')
    ax.grid()
    plt.savefig(f'{resultpath}\\{modeltype}\\nested_cv_output.png')
                
    print('Mean scores of outer folds per each dataset:\n',
    estimators_and_average_scores_across_outer_folds[modeltype]['names'],'\n',
    estimators_and_average_scores_across_outer_folds[modeltype]['scores'], '\n')

    # time nested CV procedure
    nested_time=time.time() - start_time
    print(f'-> Elapsed time for Nested CV: {nested_time}')
    timelist.append(nested_time) 

"""
# BREAK: take results of nested CV and apply to normal CV & Extensive HP Tuning
best_model_name = 'xgb_class_data-sspcat'
best_model_avg_score =123.23523523
best_model_params = {}
"""

for (best_model_name, best_model_avg_score, best_model_params) in zip(best_models_nested_CV, best_scores_nested_CV, best_params_guess_nested_CV):
    
    # retrieve information about the best model-dataset combination chosen in nested CV procedure
    best_model_dataset_name = best_model_name.split('_')[-1]
    model_type = best_model_name.replace('_' + best_model_dataset_name, '') 
    best_model_data = datasets[best_model_dataset_name][1]
    X_train_best, y_train_best = best_model_data[0][0], best_model_data[0][2]
    X_test_best, y_test_best = best_model_data[0][1], best_model_data[0][3]

    print(f'\nBest dataset for {model_type} model: {best_model_dataset_name}')
    print(f'Estimation of its generalization error: {best_model_avg_score:.3}')
    print(f'Guess for model depth (model complexity): {best_model_params}\n')

    ###############################
    ### MODEL TUNING & TRAINING ###
    ###############################

    # now we refit this best model on the whole train dataset so that we can start
    # making predictions on other data, and now we have a reliable estimate of
    # this model's generalization error and we are confident this is the best model
    # among the ones we have tried
    # at this step rounded F1 will be implemented for a regression model to help 
    # with comparison against classifier

    # 1. Hyperparameter tuning
    print(f'\n*** HYPERPARAMETER TUNING {model_type} ***')
    start_time_tuning=time.time() # time HS grid search and prediction

    increase_learning = {
        'n_estimators': 200 # 200 increase nr of estimators, 200 seems like a good guess with most of model trainings stopping around 150
    }

    param_tuning = {
        'learning_rate': [0.1, 0.05, 0.01],
        'max_depth': [8, 10, 12],
        'min_child_weight' : [1, 3, 5], # range: [0,∞] [default=1]
        'min_split_loss': [0, 3, 5], #range: [0,∞] [default=0]
        'subsample': [1, 0.8, 0.6],  #range: (0,1]  [default=1]
        'colsample_bytree': [1, 0.8, 0.6], # range: (0, 1] [default=1]
        'reg_lambda':[0.1, 1, 10] #[default=1]
    } 
    # 3^7 = 2187 * 5 folds * 2 models = 21870

    best_model = models_and_scorers[model_type][0]
    best_model = best_model.set_params(**increase_learning)
    GS_results, best_params = HyperParamGS(best_model, X_train, y_train, model_type, param_tuning, inner_cv)

    dump(GS_results, f'{resultpath}\\{model_type}\\GSCV_results.dat')
    dump(best_params, f'{resultpath}\\{model_type}\\best_params.dat')

    end_time_tuning = time.time() - start_time_tuning
    timelist.append(end_time_tuning)
    print(f'-> Elapsed time for HP tuning: {end_time_tuning}')

    # 2. Model training, validation and prediction on left-out test set
    print(f'\n*** MODEL TRAINING, VALIDATION & TESTING {model_type} ***')
    start_time_training = time.time()

    increase_learning = {
        'n_estimators': 500 # 500  increase nr of estimators
    }

    tuned_model = best_model#.set_params(**best_params)
    tuned_model = tuned_model.set_params(**increase_learning)
    tuned_model = best_model
    final_model, train_results, pred_output = ModelFit(tuned_model, model_type, 
                X_train_best, y_train_best, X_test_best, y_test_best, 
                early_stop=50, 
                learningcurve = True, 
                importance = True, 
                plottree = True, 
                savemodel = True,
                verbose = 1)

    dump(train_results, f'{resultpath}\\{model_type}\\training_results.dat')
    dump(pred_output, f'{resultpath}\\{model_type}\\prediction_results.dat')

    end_time_training= time.time() - start_time_training
    print(f'-> Elapsed time for training, validation & testing: {end_time_training}')
    timelist.append(end_time_training)

dump(timelist, f'{resultpath}\\total_timing.dat')




