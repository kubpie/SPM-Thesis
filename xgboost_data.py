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

from ssp_features import SSPGrad, SSPStat, SSPId
from data_prep import LoadData, FeatDuct

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
    
def TrainTestSplit(data, save = False):
    # divide dataset into test & training subsets
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

def ClassImbalance(data, target, plot = False):
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
        ax.set_xticklabels(yclass)
        ax.set_title('Cumulative sum plot of class distributions')
        ax.grid()
    
        for i, txt in enumerate(np.cumsum(yper)):
            ax.annotate('{:.2f}%'.format(txt),
            xy=(x[i], np.cumsum(yper)[i]), 
            xytext=(x[i]-0.65, np.cumsum(yper)[i]+0.2), 
            arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))
    
    return y_population
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

def CreateSplits(data, level_out = 1, replace=False):
    """
    1. Create 3 separate data splits based on the 'wedge_slope' value
    2. Investigate the distribution of each set
    3. Fix outliers based on 'level_out'% threshold, i.e. classes < 1% of the subset. 
    If replace = True, the outliers will be propagated to the closest higher class
    up to 2! classes up. 
    If after propagating up, the class is still < 1% it will be discared as an outlier.
    """

    data_00 = data.loc[data['wedge_slope'] == 0]
    data_2U = data.loc[data['wedge_slope'] == 2] #2 deg up
    data_2D = data.loc[data['wedge_slope'] == -2] #2 deg down
        
    distributions = []
    SplitSets = []
    
    def remove_outliers(dat,rayclass):
        outliers = dat.index[dat['num_rays'] == rayclass]
        dat = dat.drop(index = outliers)
        return dat
    
    
    #check the datasets statistics: class popualtion, ...
    for dat in [data_00, data_2U, data_2D]:
        ystat = ClassImbalance(dat, target, plot = False)
        distributions.append(ystat)
        classlist = list(ystat.keys())
        for r, rayclass in enumerate(ystat):
            ystatnew = ClassImbalance(dat, target, plot = False)
            #remove outliers when sample size < 1% of total samples
            if ystatnew[rayclass][1] < level_out:              
                if replace and r <= len(ystat)-3:
                    propagated_class = ystat[classlist[r]][1] + ystat[classlist[r+1]][1]
                    if propagated_class >= level_out:
                        dat.loc[dat['num_rays'] == rayclass, ('num_rays')] = classlist[r+1] 
                    else:
                        dat.loc[dat['num_rays'] == rayclass, ('num_rays')] = classlist[r+1] 
                        ystatnew = ClassImbalance(dat, target, plot = False)
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
    
        ystat = ClassImbalance(dat, target, plot = False)
        distributions.append(ystat)  
        SplitSets.append(dat)  
    
    return SplitSets, distributions
    
"""
    #LABEL TARGET AND FEATURES
    target = 'num_rays'
    features = data.columns.tolist()
    features.remove(target)
    
    redundant_feat = []
    alldata_new = []
    features_new = []
    # Remove redundant features in separate dataset (with constant values)
    for i, dat in enumerate(SplitSets):
        #corr_matrix = dat[features].corr(method = 'spearman').abs()
        redF = constant_features(dat[features], frac_constant_values = 0.90)
        redundant_feat.append(redF)
        alldata_new.append(dat.drop(columns = redF))
        featnames = [x for x in features if x not in redF]
        features_new.append(featnames)
                    
        
    features = features_new
    SplitSets = alldata_new

    # divide dataset into test & training subsets
    seed = 233
    test_size = 0.25
    
    dtrain = []
    dtest = []
    
    for i, dat in enumerate(alldata):
        X_train, X_test, y_train, y_test = train_test_split(dat[features[i]], dat[target], test_size=test_size, random_state=seed, stratify = dat[target])
        # stratified split makes sure that class distribution in training\test sets is as similar as possible
        dtrain.append(pd.concat((X_train, y_train), axis = 1))
        dtest.append(pd.concat((X_test, y_test), axis = 1))
        corr_matrix = dat[features[i]].corr(method = 'spearman').abs()
        
        if plot_corr:
            plot_correlation(corr_matrix)

    
    """
     #, dtrain, dtest, features

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

def plot_correlation(corrmat):
    for c in corrmat:
        # Set font scale
        sns.set(font_scale = 1)
        # Set the figure size
        f, ax = plt.subplots(figsize=(12, 12))
        sns.heatmap(c, cmap= 'YlGnBu', square=True)
        # Tight layout
        f.tight_layout()
        

# XGBOOST DATABASE PROCESSING (incl. feature selection)
# load data
import os
path = os.getcwd()+'\data\\'
rawdata = LoadData(path)
data = FeatDuct(rawdata, Input_Only = True)
y = data['num_rays']
target = 'num_rays'

data_enc = EncodeData(data)
ClassImbalance(data,target, plot = True)
SplitSets, data_dist = CreateSplits(data, level_out = 1, replace=True)
