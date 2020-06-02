# -*- coding: utf-8 -*-
"""
Created on Sun May 31 16:31:29 2020

@author: kubap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
import seaborn as sns
import xgboost as xgb

from ssp_features import SSPGrad, SSPStat, SSPId
from data_prep import LoadData, FeatDuct
from xgb_mylib import ModelFit



"""
This separate piece of code takes care of data pre-processing. 
1. load filtered XGB data (only convergent results)
2. convert to DataFrame format
3. categorical data one-hot encoding
4. label encoding class [target_values] -> class [0,16]
5. create a 80/20 traning test split
6. save training/test/upsampled data in separate .csv files
---- UPSAMPLING ----
7. upsample higher classes to the size of the 0-class (500 rays) resulting in 17*3089 = 52513 dataset 
8. test dataset distribution is a 'real-life' distribution and remains unchanged
"""

def EncodeData(data):
    SeasonList = []
    LocationList = []
    
    for ssp in data['profile']:
        seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
        season = next((s for s in seasons if s in ssp), False)
        SeasonList.append(season)
        location = ssp.replace(season, '')[:-1]
        location = location.replace(' ', '-')
        LocationList.append(location)
    
    data_enc = data.drop(columns = 'profile')
        
    for f, feature in enumerate([SeasonList, LocationList]):
        label_encoder = LabelEncoder()
        feature = label_encoder.fit_transform(feature)
        feature = feature.reshape(feature.shape[0], 1)
        onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        feature = onehot_encoder.fit_transform(feature)
        name = ["season", "location"]
        feature = pd.DataFrame(feature)
        feature.columns = [name[f]+ "-" + str(i) for i in range(feature.shape[1])]    
        data_enc = pd.concat((data_enc, feature), axis=1)
        
    #TODO: See if encoding target is still necessary with the new method
    # encode y for k-folded valdiation
    #label_encoder = LabelEncoder()
    #label_encoder = label_encoder.fit(y)
    #y_enc = label_encoder.transform(y)
    #y_enc = y
    return data_enc

def ClassImbalance(data, plot = False):
    target = 'num_rays'

    yclass, ycount = np.unique(data[target], return_counts=True)
    yper = ycount/sum(ycount)*100
    y_population = dict(zip(yclass, zip(ycount, yper)))
    
    #print("y-variance: ", data[target].var())
    #print("y-mean:",  data[target].mean())
    #data.describe()
    
    if plot:
        fig, ax = plt.subplots()
        width = 0.5
        x = np.arange(len(yclass))
        bars = ax.bar(x, ycount, width, label='Class Distribution')
        ax.set_ylabel('Number of Samples')
        ax.set_xlabel('Class: Number of Rays')
        ax.set_title('Class Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(yclass)
        ax.grid()
        #ax.legend()
        
        for b, bar in enumerate(bars):
            height = bar.get_height()
            ax.annotate('{:.2f}%'.format(yper[b]),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom')
        
        fig, ax = plt.subplots()
        x = np.arange(len(yclass))
        ax.plot(x, np.cumsum(yper), '-ok')
        ax.set_ylabel('Per-class Percentage of Total Dataset [%]')
        ax.set_xlabel('Class: Number of Rays')
        ax.set_xticks(x)
        ax.set_xticklabelsmet(yclass)
        ax.set_title('Cumulative sum plot of class distributions')
        ax.grid()
    
        for i, txt in enumerate(np.cumsum(yper)):
            ax.annotate('{:.2f}%'.format(txt),
            xy=(x[i], np.cumsum(yper)[i]), 
            xytext=(x[i]-0.65, np.cumsum(yper)[i]+0.2), 
            arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))
    
    return y_population

def PlotCorrelation(dat, features, annotate = True):

        correlation_matrix = dat[features].corr(method = 'spearman').abs()
        # Set font scale
        sns.set(font_scale = 1)
        # Set the figure size
        f, ax = plt.subplots(figsize=(12, 12))
        # Plot heatmap
        sns.heatmap(correlation_matrix, cmap= 'YlGnBu', square=True, annot=annotate)
        # Tight layout
        f.tight_layout()
        plt.show()  


def CreateSplits(data, level_out = 1, replace = True, plot_distributions = False, plot_correlations = False):
    """
    1. Create 3 separate data splits based on the 'wedge_slope' value
    --- OUTLIERS ---
    2. Investigate the distribution of each set
    3. Fix outliers based on 'level_out'% threshold, i.e. classes < 1% of the subset. 
    If replace = True, the outliers will be propagated to the closest higher class
    up to 2! classes up. 
    If after propagating up, the class is still < 1% it will be discared as an outlier.
    --- FEATURES ---
    4. Remove features that become redundant withing subsets i.e. water depth max, or slope value
    
    """

    data_00 = data.loc[data['wedge_slope'] == 0]
    data_2U = data.loc[data['wedge_slope'] == 2] #2 deg up
    data_2D = data.loc[data['wedge_slope'] == -2] #2 deg down
    
    ### Outliers Correction     
    distributions = []
    SplitSets = []
    
    def remove_outliers(dat,rayclass):
        outliers = dat.index[dat['num_rays'] == rayclass]
        dat = dat.drop(index = outliers)
        return dat
    
    
    def constant_features(X, frac_constant_values = 0.90):
        # Get number of rows in X
        num_rows = X.shape[0]
        # Get column labels
        allLabels = X.columns.tolist()
        # Make a dict to store the fraction describing the value that occurs the most
        constant_per_feature = {label: X[label].value_counts().iloc[0]/num_rows for label in allLabels}
        # Determine the features that contain a fraction of missing values greater than threshold
        labels = [label for label in allLabels if constant_per_feature [label] > frac_constant_values]
        
        return labels
        
    #check the datasets statistics: class popualtion, ...
    for t, dat in enumerate([data_00, data_2U, data_2D]):
        ystat = ClassImbalance(dat, plot = False)
        distributions.append(ystat)
        classlist = list(ystat.keys())
        for r, rayclass in enumerate(ystat):
            ystatnew = ClassImbalance(dat, plot = False)
            #remove outliers when sample size < 1% of total samples
            if ystatnew[rayclass][1] < level_out:              
                if replace and r <= len(ystat)-3:
                    propagated_class = ystat[classlist[r]][1] + ystat[classlist[r+1]][1]
                    if propagated_class >= level_out:
                        dat.loc[dat['num_rays'] == rayclass, ('num_rays')] = classlist[r+1] 
                    else:
                        dat.loc[dat['num_rays'] == rayclass, ('num_rays')] = classlist[r+1] 
                        propagated_class = ystat[classlist[r+1]][1] + ystat[classlist[r+2]][1]
                        if propagated_class >= level_out:
                            dat.loc[dat['num_rays'] == classlist[r+1], ('num_rays')][1:] = classlist[r+2] 

                        else:
                            dat = remove_outliers(dat,rayclass)
    
                if replace and r == len(ystat)-2: #second last class can be propagated only once
                    propagated_class = ystat[classlist[r]][1] + ystat[classlist[r+1]][1]
                    if propagated_class >= level_out:
                        dat.loc[dat['num_rays'] == rayclass, ('num_rays')] = classlist[r+1] 
                    else:
                        dat = remove_outliers(dat,rayclass)
                            
                if replace and r == len(ystat)-1: #last class can be only removed if it's still < 1%
                    dat = remove_outliers(dat,rayclass)
                    
                if not replace: #if replace = False then always remove outliers
                    dat = remove_outliers(dat,rayclass)
        
        #1. TODO: Dirty fix for dat00, for some reason 6000 gets propagated all classes from below
        if t == 0:
            dat = remove_outliers(dat, 6000)
            
        ystat = ClassImbalance(dat, plot = plot_distributions)
        distributions.append(ystat)  
        SplitSets.append(dat)  
    SplitSets[0] = remove_outliers(SplitSets[0], 6000)    
    
    #2. TODO : A value is trying to be set on a copy of a slice from a DataFrame.
    #          Try using .loc[row_indexer,col_indexer] = value instead
    
    ### End of Outlier Correction
    
    ### Feature dropout
    # Remove redundant features with constant values in each set 
    for i in range(len(SplitSets)):
        features = data.columns.tolist()
        redF = constant_features(SplitSets[i][features], frac_constant_values = 0.99)
        print('Removed constant features ' + f'{redF} '  'for SplitSets ' f'{i}' )
        SplitSets[i] = SplitSets[i].drop(columns = redF)
        features.remove('num_rays')
        features = [f for f in features if f not in redF]
        
        if plot_correlations:
            PlotCorrelation(SplitSets[i], features, annotate = True)

    
    return SplitSets, distributions

def TrainTestSplit(data, save = False, seed = 27, test_size = 0.25):
    # divide dataset into test & training subsets
    target = 'num_rays'
    predictors = [x for x in data.columns if x not in target]
    X_train, X_test, y_train, y_test = train_test_split(data[predictors], data[target], test_size=test_size, random_state=seed, stratify =  data[target])
    # stratified split ensures that the class distribution in training\test sets is as similar as possible
    dtrain = pd.concat((X_train, y_train), axis = 1)
    dtest = pd.concat((X_test, y_test), axis = 1)
    
    if save:
        # save into separate .csv files
        filepath = os.getcwd()+'\data\\xgboost\\'
        dtest.to_csv(filepath + 'dtest_25.csv', index = None, header = True)
        dtrain.to_csv(filepath + 'dtrain_75.csv', index = None, header = True)
        #dtrainup.to_csv(filepath + 'dtrainup.csv', index = None, header = True)
        #dtrain_smot.to_csv(filepath + 'dtrain_smot.csv', index = None, header = True)
        print("New datafiles have been created!")
    
    return dtrain, dtest


"""
# Upsampling with SMOT-ENC technique that can handle both cont. and categorical variables
#categorical_var = np.hstack([2, np.arange(5,33)])
categorical_var = np.hstack([2,np.arange(5,33)])
minority = np.arange(4,17)
samplenr = 250
population_target = dict(zip(minority, (np.ones(len(minority))*samplenr).astype(int)))
smote_nc = SMOTENC(categorical_features=categorical_var, sampling_strategy=population_target, random_state=42)
#smote_nc_max = SMOTENC(categorical_features=categorical_var, sampling_strategy='auto', random_state=42)
X_smot, y_smot = smote_nc.fit_resample(X_train, y_train)
dtrain_smot = pd.concat((X_smot, y_smot), axis =1)
dtrain_smot = dtrain_smot.sample(frac = 1) #shuffle the upsampled dataset

"""


# XGBOOST DATABASE PROCESSING (incl. feature selection)
# load data
import os
path = os.getcwd()+'\data\\'
rawdata = LoadData(path)
data = FeatDuct(rawdata, Input_Only = True)
y = data['num_rays']

data_enc = EncodeData(data)
SplitSets, data_dist = CreateSplits(data_enc, level_out = 1, replace=True, plot_distributions = False, plot_correlations = False)

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
"""
for subset in SplitSets:
    [dtrain, dtest] = TrainTestSplit(subset, test_size = 0.25)
    target = 'num_rays'
    features = subset.columns
    features.remove(target)
    bst_model, fullresult_class, output_class = \
    ModelFit(xgb_class, dtrain, dtest, features, target, early_stop = 100,
    verbose=True, learningcurve = True, importance = True, plottree = False, savename = False)
    """