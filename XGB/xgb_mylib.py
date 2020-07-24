# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:47:55 2020

@author: kubap
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import xgboost as xgb
from joblib import dump
from joblib import load
import os
from pathlib import Path
import pandas as pd
import seaborn as sns

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import fbeta_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import check_consistent_length
from sklearn.utils.sparsefuncs import count_nonzero
from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics.classification import _check_targets
from sklearn.metrics.classification import _weighted_sum



def ModelFit(best_model, model_type, 
            X_train, y_train, X_test, y_test,
            early_stop, 
            learningcurve = False, 
            importance = False, 
            plottree = False, 
            savemodel = False,
            verbose = 1
            ):


    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 321, shuffle = True, stratify = y_train)
    dtrainDM = xgb.DMatrix(X_train, label=y_train)
    eval_set = [(X_train, y_train),(X_val, y_val)]
    class_labels = np.unique(y_test)

    #Path for saves
    path = os.getcwd()
    resultpath = Path(path+"/XGB/results/" + model_type)
    resultpath = str(resultpath) + '\\' 

    if model_type == "xgb_class":
        eval_metrics = ['merror','f1_err'] 
        eval_label = ['Mean Class. Error', 'F1-macro Error']
        feval = f1_eval_class
    elif model_type == "xgb_reg":
        eval_metrics = ['rmse', 'f1_err']
        eval_label = ['RMSE', 'F1-macro Error']
        feval = f1_eval_reg
        #best_model.set_params()
    best_model.fit(X_train, y_train, 
                early_stopping_rounds = early_stop, 
                eval_set=eval_set,
                eval_metric=feval, 
                verbose=verbose)
    results = best_model.evals_result()
    print(f'Training stopped at iteration: {best_model.best_iteration} \nEvaluation Error: {best_model.best_score}')    

    #Call predict on the estimator with the best found parameters.
    y_pred = best_model.predict(X_test)
    #print(y_pred)
        
    output = []
    if model_type == "xgb_reg":
        prediction = np.zeros(y_pred.size)
        for p in range(0,y_pred.size):
            prediction[p] = min(class_labels, key=lambda x:abs(x-y_pred[p]))

        rounding_error = abs(prediction-y_pred)
        print("Mean classification rounding error: %.2f" % (np.mean(rounding_error)))
        rmse = np.sqrt(mean_squared_error(y_test, prediction))
        print("RMSE after rounding: %.2f" % (rmse))
        output = [y_pred, prediction]
        y_pred = prediction
        # TODO: make a plot of prediction vs rounded showing the residual
        
    #Print model report:
    print('\nPrediction on the test set')
    report = classification_report(y_test, y_pred, digits=2)
    print(report)

    #Plot confustion matrix
    cmatrix = confusion_matrix(y_test, y_pred)      
    print(cmatrix)
    df_cm = pd.DataFrame(cmatrix, index = [c for c in class_labels], columns = [c for c in class_labels])
    plt.figure(figsize = (12,10))
    plt.title("Confustion Matrix")
    hm = sns.heatmap(df_cm, annot = True, cmap="YlGnBu", fmt="d", linewidths  = 0.5)
    hm.set_yticklabels(hm.get_yticklabels(), rotation = 0)
    plt.savefig(resultpath + model_type +'_cmatrix.png')

    if savemodel:
         #save model to file
         dump(best_model, resultpath + model_type + "_final_model.dat")
         print("Saved model to:" + resultpath + model_type + "_final_model.dat")
            #TODO: FOR LATER USE
            #load model from file
            #loaded_model = load("pima.joblib.dat")
            #print("Loaded model from: pima.joblib.dat")  
            
    if importance:
        #Available importance_types = ["weight", "gain", "cover", "total_gain", "total_cover"]
        # default for 'True' is 'weight'
        types = ["weight","total_gain","total_cover"]
        fig, axes = plt.subplots(nrows=1, 
                         ncols=3, 
                         figsize=(20,5)
                         )
        plt.subplots_adjust(wspace = 0.4, bottom = 0.15, left = 0.125,)
        for t, ax in zip(types, axes):
            xgb.plot_importance(best_model, importance_type=t, title = t, show_values = False,  ylabel=None, ax = ax)
        plt.savefig(resultpath + model_type +"_feature_importance.png")
        
    if learningcurve:
        #retrieve performance metrics
        epochs = len(results['validation_0'][eval_metrics[0]])
        x_axis = range(0, epochs)
        # plot F1_err
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0'][eval_metrics[0]], label='Train', color = 'dodgerblue')
        ax.plot(x_axis, results['validation_1'][eval_metrics[0]], label='Validation', color = 'orangered')
        plt.axvline(x=best_model.best_iteration, color = 'k', linestyle='--', linewidth = 0.5, label = 'Best iteration')
        ax.set_xlim(0,epochs)
        ax.legend()
        plt.ylabel(eval_label[0])
        plt.xlabel('Epoch')
        plt.title("Learning curve for " + model_type +' - '+ eval_label[0])
        plt.savefig(resultpath + model_type +'_'+ eval_metrics[0] + '.png')
        
        # plot classification error
        fig, ax = plt.subplots()   
        ax.plot(x_axis, results['validation_0'][eval_metrics[1]], label='Train', color = 'dodgerblue')
        ax.plot(x_axis, results['validation_1'][eval_metrics[1]], label='Validation', color = 'orangered')
        plt.axvline(x=best_model.best_iteration, color = 'k', linestyle='--', linewidth = 0.5, label = 'Best iteration')
        ax.set_xlim(0,epochs)
        ax.legend()
        plt.ylabel(eval_label[1])
        plt.xlabel('Epoch')
        plt.title("Learning curve for " + model_type +' - '+ eval_label[1])
        plt.savefig(resultpath +  model_type + '_' + eval_metrics[1] + '.png')
                        
        
    if plottree:
        plt.figure(figsize=(20,5))
        xgb.plot_tree(best_model, rankdir='LR')
        #tree is saved to pdf because it's too large to display details in python
        plt.savefig(resultpath + model_type +'_tree.png', format='png', dpi=800)
             
    output.append(report), output.append(cmatrix)
    return(best_model, results, output)

    
def HyperParamGS(model, X_train, y_train, model_type, param_tuning, cv):
    
    if model_type == "xgb_class":
        scoring = 'f1_macro'
       
    elif model_type == "xgb_reg":
        scoring = 'neg_root_mean_squared_error'
           
    model = GridSearchCV(model, 
                        param_grid = param_tuning, 
                        cv=cv, scoring=scoring,
                        verbose = 1, n_jobs=-1,
                        return_train_score=False)
    model.fit(X_train,y_train)

    #Train the model on the whole dataset in GridSearchCV setup to find best hyperparameters #
    results = model.cv_results_
    means = results['mean_test_score']
    stds = results['std_test_score']
    print('\nHyperparameter Tuning Results')
    for mean, std, params in zip(means,stds,results['params']):
        print("%0.3f (+/-%0.03f)  for %r"
              % (abs(mean), std * 2, params))

    print(f'Best hyperparameters found in CV Grid Search on the whole training dataset: \n{model.best_params_}')
    
    return(results, model.best_params_)

def PlotGS(results, param, scoring, resultpath):
# TODO: Make a save option for offline tuning, then plt.show(False) and save(True)    
    
    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV Scorer Evaluation",
              fontsize=16)
    ax = plt.gca()
    ax.set_ylim(0, 1.2)
    plt.ylabel("Score")
    ax.set_xlim(0, len(results['params'])-1)       
    plt.xlabel([key for key in param.keys()])
    X_axis  = np.arange(0,np.size(results['params']))

    if len(param) > 1:
        xlabels = []
        for param in results['params']:
            xlabels.append(tuple(param.values()))
        xlabels = np.round(xlabels,2)
        ax.set_xticks(X_axis)
        ax.set_xticklabels(xlabels, fontdict ={'fontsize': 8})
        plt.xticks(rotation=90)
    elif len(param) == 1:
        value = param.values()
        ax.set_xticks(X_axis)
        ax.set_xticklabels(value)

    for scorer, color in zip(sorted(scoring), ['g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))
    
        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]
    
        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)
    
        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))
    plt.legend(loc="best")
    plt.grid(False)
    plt.savefig(resultpath + "/GS/" + list(param.keys())[0]+'.png')
    return

def accuracy_rounding_score(y_true, y_pred, normalize=True, sample_weight=None):
    for p in range(0,y_pred.size):
        class_labels = np.unique(y_true)
        y_pred[p] = min(class_labels, key=lambda x:abs(x-y_pred[p]))    
    # Compute accuracy for each possible representation
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    check_consistent_length(y_true, y_pred, sample_weight)
    if y_type.startswith('multilabel'):
        differing_labels = count_nonzero(y_true - y_pred, axis=1)
        score = differing_labels == 0
    else:
        score = y_true == y_pred

    return _weighted_sum(score, sample_weight, normalize)

def f1_rounding_score(y_true, y_pred):
    class_labels = np.unique(y_true)
    for p in range(0,y_pred.size):
        y_pred[p] = min(class_labels, key=lambda x:abs(x-y_pred[p]))    
    return fbeta_score(y_true, y_pred, beta=1, labels=class_labels, average='macro')
    
def f1_eval_class(y_pred, dtrainDM):
        y_true = dtrainDM.get_label()
        index_max = np.argmax(y_pred, axis = 1)
        err = 1 - f1_score(y_true, index_max, average = 'macro')
        return 'f1_err', err
    
def f1_eval_reg(y_pred, dtrainDM):
        y_true = dtrainDM.get_label()
        err = 1 - f1_rounding_score(y_true, y_pred)
        return 'f1_err', err
