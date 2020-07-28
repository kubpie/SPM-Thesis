# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 14:33:09 2020

@author: kubap
"""

from joblib import dump
from joblib import load
import numpy as np
import matplotlib.pyplot as plt

import os
from pathlib import Path

model_type = 'xgb_class' #'xgb_class' 'xgb_reg'
path = os.getcwd()
resultpath = Path(path+"/XGB/results/" + model_type + '/run3_corrected')
resultpath = str(resultpath) + '\\' 

best_nested_score_and_model = load(resultpath + 'best_nested_score_and_model.dat')
all_nested_scores = load(resultpath + 'nested_scores_and_models.dat')

gridsearch_results = load(resultpath + 'GSCV_results.dat')
best_params = load(resultpath + 'best_params.dat')

training_results= load(resultpath + 'training_results.dat')
prediction_results= load(resultpath + 'prediction_results.dat')
final_model = load(resultpath + model_type + '_final_model.dat')
print('loaded')