# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:05:10 2020

@author: kubap
"""

from pycebox.ice import ice, ice_plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from joblib import dump
from joblib import load
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import os
from data_prep import LoadData, FeatDuct, EncodeData, CreateSplits, TrainTestSplit, FeatBathy, FeatSSP, FeatSSPId
from xgb_mylib import f1_eval_class

def plot_ice_grid(dict_of_ice_dfs, data_df, features, ax_ylabel='', nrows=3, 
                  ncols=3, figsize=(12, 12), sharex=False, sharey=True, 
                  subplots_kws={}, rug_kws={'color':'k'}, **ice_plot_kws):
    """A function that plots ICE plots for different features in a grid."""
    fig, axes = plt.subplots(nrows=nrows, 
                             ncols=ncols, 
                             figsize=figsize,
                             sharex=sharex,
                             sharey=sharey,
                             **subplots_kws)
    # for each feature plot the ice curves and add a rug at the bottom of the 
    # subplot
    for f, ax in zip(features, axes.flatten()):
        ice_plot(dict_of_ice_dfs[f], ax=ax, **ice_plot_kws)
        # add the rug
        sns.distplot(data_df[f], ax=ax, hist=False, kde=False, 
                     rug=True, rug_kws=rug_kws)
        #ax.set_title('feature = ' + f)
        ax.set_ylabel(ax_ylabel)
        ax.set_ylim(0, 15000)
        sns.despine()
        
    # get rid of blank plots
    for i in range(len(features), nrows*ncols):
        axes.flatten()[i].axis('off')
    return fig

def ICEPlot(data, model, features):
    # create dict of ICE data for grid of ICE plots
    train_ice_dfs = {feat: ice(data=data, column=feat, predict=model.predict) 
                     for feat in features}
    
    fig = plot_ice_grid(train_ice_dfs, X, features,
                        ax_ylabel='Pred. Ray Num.', 
                        nrows=4, 
                        ncols=4,
                        alpha=0.3, plot_pdp=True,
                        pdp_kwargs={'c': 'blue', 'linewidth': 2.0},
                        linewidth=0.5, c='dimgray')
    #fig.tight_layout()
    fig.suptitle('ICE plot: Classification - all training data')
    fig.subplots_adjust(top=0.89)
    
    return train_ice_dfs
""""
A PDP is the average of the lines of an ICE plot.
Unlike partial dependence plots, ICE curves can uncover heterogeneous relationships.
PDPs can obscure a heterogeneous relationship created by interactions. 
PDPs can show you what the average relationship between a feature and the prediction looks like. This only works well if the interactions between the features for which the PDP is calculated and the other features are weak. In case of interactions, the ICE plot will provide much more insight.
"""

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

model = xgb_class

path = os.getcwd()+'\data\\'
rawdata = LoadData(path)
data = FeatDuct(rawdata, Input_Only = True) #just to leave only input data
data = FeatBathy(data, path)
data = FeatSSPId(data, path, src_cond = True)
data_fin = FeatSSP(data, path)
#data_fin = EncodeData(data)

data_fin = data_fin.fillna(0)

target = 'num_rays'
features = data_fin.columns.tolist()
features.remove(target)

[dtrain, dtest] = TrainTestSplit(data_fin, test_size = 0.25)
X = dtrain[features]
y = dtrain[target]
Xt = dtest[features]
yt = dtest[target]
# Undersampling of the TEST SET to avoid overcrowding the ICE plot
# It reduces all clases to the size of the smallest class 7000: 20 samples
undersample = RandomUnderSampler(sampling_strategy='auto')
Xt_under, yt_under = undersample.fit_resample(Xt, yt)
Xt_under_df = pd.DataFrame(Xt_under, columns = features)
print(Counter(yt))
print(Counter(yt_under))


# TODO: Check if deliberate fit-predict => 100% correct prediction gives you 
# real data analysis instead of model analysis

### XGB MODEL SETUP
# Model 'fit' before 'predict' inside ICE plot function
eval_set = [(dtrain[features].values, dtrain[target].values),(dtest[features].values, dtest[target].values)]
eval_metric = ["f1_err","merror"] #the last item in eval_metric will be used for early stopping
feval = f1_eval_class
early_stop = 100

model_trained = model.fit(X.values, y.values, eval_set=eval_set, eval_metric = feval, verbose=0, early_stopping_rounds = early_stop)
results = model_trained.evals_result()
print(f'Best iteration: {model_trained.best_iteration}\nF-score: {1-model_trained.best_score}')

ice_features = [ feat for feat in features if "SSP" not in feat ]

ICEdict1 = ICEPlot(Xt, model_trained, ice_features)

# TODO: Modify ice/pdp plots to verify the correctness of prediction!
# - Joris suggestion, gotta do.
# PLot line over bar/surface indicating sample population
# Problem with ICE plots:\
# correlation, some points are invalid, i.e. min_depth > max_depth, source_depth > max_depth

