# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:47:55 2020

@author: kubap
"""

import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import os
from joblib import dump
from joblib import load
from sklearn.model_selection import GridSearchCV
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

from sklearn.metrics import _check_targets
from sklearn.metrics.classification import _weighted_sum

from matplotlib.ticker import FormatStrFormatter


def ModelFit(bst_model, dtrain, dtest, features, target, early_stop, verbose, learningcurve, importance, plottree, savename):
    #dtrainDM = xgb.DMatrix(dtrain[features], dtrain[target])
    eval_set = [(dtrain[features], dtrain[target]),(dtest[features], dtest[target])]
    modeltype = str(type(bst_model))
    class_names = np.unique(dtest[target])

    if "Classifier" in modeltype:
        modeltype = "class"
        eval_metric = ["f1_err","merror"] #the last item in eval_metric will be used for early stopping
        feval = f1_eval_class
    
    elif "Regressor" in modeltype:
        modeltype = "reg"
        eval_metric = ["f1_err","rmse"] 
        feval = f1_eval_reg

    #Fit the algorithm with tuned hyperparameters on the full training data
    bst_model = bst_model.fit(dtrain[features], dtrain[target], eval_set=eval_set, eval_metric = feval,
               verbose=verbose, early_stopping_rounds = early_stop)
    results = bst_model.evals_result()
    print(bst_model.best_iteration, bst_model.best_score)
    
    print("\nModel Summary")
    print(bst_model)
    
    #Predict test set:
    y_pred = bst_model.predict(dtest[features])
    #print(y_pred)
    #dtrain_predprob = bst_model.predict_proba(dtest[features])
    output = []
    
    if modeltype == "reg":
        prediction = np.zeros(y_pred.size)
        for p in range(0,y_pred.size):
            #prediction[p] = min(np.arange(0,17, dtype=np.float32), key=lambda x:abs(x-y_pred[p]))          
            prediction[p] = min(class_names, key=lambda x:abs(x-y_pred[p]))

        rounding_error = abs(prediction-y_pred)
        print("Mean rounding error: %.2f" % (np.mean(rounding_error)))
        rmse = np.sqrt(mean_squared_error(dtest[target].values, prediction))
        print("RMSE: %.2f" % (rmse))
        output = [y_pred, prediction]
        y_pred = prediction
        # TODO: make a plot of prediction vs rounded showing the residual
        
    #Print model report:
    print("\nModel Report")
    report = classification_report(dtest[target].values, y_pred, digits=2)
    print(report)
    #Plot confustion matrix
    cmatrix = confusion_matrix(dtest[target].values, y_pred)      
    print(cmatrix)
    """
    disp = plot_confusion_matrix(bst_model, dtest[features], y_pred,
                             display_labels=class_names,
                             cmap=plt.cm.Blues,
                             normalize=None)
    disp.ax_.set_title("Confustion Matrix")
    #print(disp.confusion_matrix)
    plt.show()
    """
    #Path for plots
    resultpath = os.getcwd()+'\results\\'

    if savename:
         #save model to file
         dump(bst_model, resultpath + savename + ".dat")
         print("Saved model to:" + resultpath + savename + ".dat")
            #TODO: FOR LATER USE
            #load model from file
            #loaded_model = load("pima.joblib.dat")
            #print("Loaded model from: pima.joblib.dat")  
            
    if importance:
        #Available importance_types = ["weight", "gain", "cover", "total_gain", "total_cover"]
        # default for 'True' is 'weight'
        """
        xgb.plot_importance(bst_model, importance_type=importance)
        plt.rcParams['figure.figsize'] = 10, 5
        """
        types = ["weight","total_gain","total_cover"]
        fig, axes = plt.subplots(nrows=1, 
                         ncols=3, 
                         figsize=(30,8)
                         )
        for t, ax in zip(types, axes):
            xgb.plot_importance(bst_model, importance_type=t, title = t, 
                            show_values = False,  ylabel=None, ax = ax)
        
    if learningcurve:
        #retrieve performance metrics
        epochs = len(results['validation_0'][eval_metric[0]])
        x_axis = range(0, epochs)
        
        # plot F1_err
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0'][eval_metric[0]], label='Train')
        ax.plot(x_axis, results['validation_1'][eval_metric[0]], label='Test')
        plt.axvline(x=bst_model.best_iteration, color = 'k', linewidth = 0.5, label = 'Best run')
        ax.legend()
        plt.ylabel(eval_metric[0])
        plt.xlabel('Epoch')
        plt.title("Learning curve with: " + eval_metric[0])
        plt.savefig(resultpath + modeltype + eval_metric[0] + '.png')
        plt.show()
        
        # plot classification error
        fig, ax = plt.subplots()   
        ax.plot(x_axis, results['validation_0'][eval_metric[1]], label='Train')
        ax.plot(x_axis, results['validation_1'][eval_metric[1]], label='Test')
        #plot stopline
        #TODO: plot range only until x=bst_model.best_iteration ?? 
        plt.axvline(x=bst_model.best_iteration, color = 'k', linewidth = 0.5, label = 'Best run')
        ax.legend()
        plt.ylabel(eval_metric[1])
        plt.xlabel('Epoch')
        plt.title("Learning curve with: " + eval_metric[1])
        plt.savefig(resultpath + modeltype + eval_metric[1] + '.png')
        plt.show()    
                        
        
    if plottree:
        xgb.plot_tree(bst_model,num_trees=0)
        #tree is saved to pdf because it's too large to display details in python
        plt.savefig(resultpath + modeltype + '_tree.pdf', format='pdf', dpi=2000)
             
    output.append(report), output.append(cmatrix)
    return(bst_model, results, output)

    
def HyperParamGS(model, param, dtrain, features, kfold, target, scoring, refit):
    # The function evaluates different model paramaters in a grid search setup 
    # and creates a scorer that registers metrics in 'scoring' dict 

    gs_model = GridSearchCV(estimator = model, verbose = 2,
                            param_grid = param, n_jobs=-1, 
                            cv=kfold, scoring=scoring, refit=refit, 
                            return_train_score=True)
    
    gs_model.fit(dtrain[features],dtrain[target])
    
    print(gs_model.best_index_,gs_model.best_params_, gs_model.best_score_)
    GSresults = gs_model.cv_results_
    return(GSresults, gs_model.best_params_)

def PlotGS(results, param, scoring, modeltype):
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
    resultpath = resultpath = os.getcwd()+'results\tuning\\'
    plt.savefig(resultpath + list(param.keys())[0]+modeltype+'.png')
    #plt.show(False)

def accuracy_rounding_score(y_true, y_pred, normalize=True, sample_weight=None):
    for p in range(0,y_pred.size):
        labels = [  500.,  1000.,  1500.,  2000.,  2500.,  3000.,  3500.,  4000.,
        4500.,  5000.,  6000.,  7000.,  8000.,  9000., 10000., 12500., 15000.]
        y_pred[p] = min(labels, key=lambda x:abs(x-y_pred[p]))    
    # Compute accuracy for each possible representation
    #y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    check_consistent_length(y_true, y_pred, sample_weight)
    if y_type.startswith('multilabel'):
        differing_labels = count_nonzero(y_true - y_pred, axis=1)
        score = differing_labels == 0
    else:
        score = y_true == y_pred

    return _weighted_sum(score, sample_weight, normalize)

def f1_rounding_score(y_true, y_pred, labels=None, pos_label=1, average='binary',
    sample_weight=None, zero_division="warn"):
    labels = [  500.,  1000.,  1500.,  2000.,  2500.,  3000.,  3500.,  4000.,
        4500.,  5000.,  6000.,  7000.,  8000.,  9000., 10000., 12500., 15000.]
    for p in range(0,y_pred.size):
        y_pred[p] = min(labels, key=lambda x:abs(x-y_pred[p]))    


    return fbeta_score(y_true, y_pred, 1, labels=labels,
                           pos_label=pos_label, average=average,
                           sample_weight=sample_weight,
                           zero_division=zero_division)
    
def f1_eval_class(y_pred, dtrainDM):
        y_true = dtrainDM.get_label()
        index_max = np.argmax(y_pred, axis = 1)
        err = 1 - f1_score(y_true, index_max, average = 'macro')
        return 'f1_err', err
    
def f1_eval_reg(y_pred, dtrainDM):
        y_true = dtrainDM.get_label()
        err = 1 - f1_rounding_score(y_true, y_pred, average = 'macro')
        return 'f1_err', err
