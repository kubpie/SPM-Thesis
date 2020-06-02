# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:42:50 2020

@author: kubap
"""
import numpy as np
import pandas as pd
import seaborn as sns
from os import listdir

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
#from imblearn.over_sampling import SMOTENC

from data_analysis_lib import ClassImbalance, PlotCorrelation, Multipage

def LoadData(path): 
    
    def CheckCol(lst):
        return lst[1:] == lst[:-1] 
    
    flist = [f for f in listdir(path) if "Dataset" in f]
    col = []
    frames = []
    
    for file in flist: #to check fix it latest
        raw_data = pd.read_csv(path+file, index_col=None, header=0)
        col.append(raw_data.columns.tolist())
        frames.append(raw_data)
        test_col = CheckCol(col)
        if test_col == False:
            print("Column names are inconsistent!")
    
    mergedData = pd.concat(frames, ignore_index = True)
    # fill NaN
    mergedData.fillna(0, inplace = True)
    # dropping ALL duplicte values 
    mergedData.drop_duplicates(subset = None, keep = 'first', inplace = True) 
    # calculate the number of non-converged scenarios, can be useful for showing stat later
    nonconverged = mergedData[(mergedData['num_rays'] == 15000) & (mergedData['criterion'] == 0)]
    # leave out only converged scenarios
    convData = mergedData[(mergedData['num_rays'] != 20000) & (mergedData['criterion'] == 1)] #ALL RAW UNPROCESSED DATA
     # remove nonimportant cols
    convData = convData.drop(columns = ['runID','residual','runtime','criterion'])
    # reset index values (loses info from the initial set)
    convData.reset_index(drop=True, inplace=True)

    return convData
    
# FORMATTING
def FeatDuct(data, Input_Only = True):   
    # merge SD/BD features into one to emphasize duct propagation mode
    duct_cols = ['duct_prop_type','duct_width_if_sourceinduct', 'duct_SSP_if_sourceinduct']
    duct_df = pd.DataFrame(0, index=np.arange(len(data)), columns=duct_cols)
   
    data = pd.concat((data,duct_df), axis = 1)
    
    data.loc[data['duct_type'] == 'SD', 'duct_prop_type'] = 1
    data.loc[data['duct_type'] == 'SD', 'duct_width_if_sourceinduct'] =  data.loc[data['duct_type'] == 'SD', 'surface_duct_depth']
    data.loc[data['duct_type'] == 'SD', 'duct_SSP_if_sourceinduct'] = data.loc[data['duct_type'] == 'SD', 'surface_duct_SSP']

    data.loc[data['duct_type'] == 'BD', 'duct_prop_type'] = -1
    data.loc[data['duct_type'] == 'BD', 'duct_width_if_sourceinduct'] =  data.loc[data['duct_type'] == 'BD', 'bottom_duct_width']
    data.loc[data['duct_type'] == 'BD', 'duct_SSP_if_sourceinduct'] = data.loc[data['duct_type'] == 'BD', 'bottom_duct_SSP']

    data = data.drop(columns = ['duct_type', 'surface_duct', 'bottom_duct', 'source_in_duct','surface_duct_depth','surface_duct_SSP','bottom_duct_width','bottom_duct_depth','bottom_duct_SSP'])        
               
    #DROPPING LOTS OF COLUMNS HERE TO LEAVE OUT PLAIN SIMULATION I/O DATA
    if Input_Only == True:
        data = data.drop(columns = ['deep_CH_axis','deep_CH_SSP','shallow_CH_axis','shallow_CH_SSP'])
        data = data.drop(columns = ['waveguide','CHmax_axis','SSP_CHmax'])
        data = data.drop(columns = 'SSP_source')
        data = data.drop(columns = duct_cols)
    return data

def FeatBathySSP(data):
    rmax = 44000
    dmax = 1500
    
    path = r'C:\Users\kubap\Documents\THESIS\DATA\\'

    ssp = pd.read_excel(path+"env.xlsx", sheet_name = "SSP")
    depth = ssp['DEPTH'].values.tolist()
    
    dmin = data['water_depth_min'].values
    profile = data['profile'].values
    slope = data['wedge_slope'].values
    srcdepth = data['source_depth'].values
    srcssp = data['SSP_source'].values
    
    cmat = np.zeros([len(data),len(depth)]) #segmented & interpolated sound speed profile vector 
    weight = np.zeros([len(data),len(depth)]) #weights on the SSP
    wedge = np.zeros([len(data),2]) #wedge parameters, bathymetry info
    area = np.zeros([len(data),4]) #duct area parameters 
    src = np.zeros([len(data),2]) #source-to-channel dimension ratios
    
    for row in range(len(data)):
        d = min(depth, key=lambda x:abs(x-dmin[row]))
        idx = depth.index(d)+1
        cmat[row,0:idx] = ssp[profile[row]].iloc[0:idx]
        weight[row,0:idx] = 1.0
        
        src[row,0] = srcdepth[row]/dmin[row] 
        d300 = min(depth, key=lambda x:abs(x-300))
        idx300 = depth.index(d300)
        avg300 = np.mean(ssp[profile[row]].iloc[:idx300])
        src[row,1] = srcssp[row]/avg300    #ssp_source/mean ssp of 300m depth, src pos goes only up to 300m so it indicated wheter src is in the duct
          
        area[row,0] = 0
        area[row,1] = 0.5
        area[row,2] = 0.5
        area[row,3] = rmax*dmin[row]
                
        if slope[row] != 0: 
            for dz in range(idx,len(depth)):
                wedge_range = np.round((dmax-dmin[row])/np.tan(np.deg2rad(abs(slope[row]))))
                start_wedge = 0.5*(rmax - wedge_range)
                ds = np.round((dmax-depth[dz])/np.tan(np.deg2rad(abs(slope[row]))))
                
                #SSP weight matrix
                gamma = 1.0
                weight[row,dz] = ((rmax - gamma*(start_wedge+wedge_range-ds))/rmax) #weight proportional to the area covered by SSP
                cmat[row,dz] = ssp[profile[row]].iloc[dz]*weight[row,dz]
    
                if slope[row] > 0:
                    wedge[row,0] = start_wedge
                    wedge[row,1] = start_wedge + wedge_range
                    area[row,0] = (0.5*(wedge_range)*(dmax-dmin[row]))+wedge_range*dmin[row]
                    area[row,1] = dmax*start_wedge
                    area[row,2] = dmin[row]*(rmax - start_wedge - wedge_range)
                    area[row,3] = sum(area[row,0:3]) #total area
                    src[row,0] = srcdepth[row]/dmax
    
                else:
                    wedge[row,0] = start_wedge
                    wedge[row,1] = start_wedge + wedge_range
                    area[row,0] = (0.5*(abs(wedge_range))*(dmax-dmin[row]))+abs(wedge_range)*dmin[row]
                    area[row,1] = dmin[row]*abs(start_wedge)
                    area[row,2] = dmax*(rmax - abs(start_wedge) - abs(wedge_range))
                    area[row,3] = sum(area[row,0:3]) #total area
                    src[row,0] = srcdepth[row]/dmin[row] 
                              
    colnames = []         
    for i in range(len(depth)):
        colnames.append("SSPd-"+str(depth[i]))    
    df_cmat = pd.DataFrame(cmat)
    df_cmat.columns = colnames
    
    df_wedge = pd.DataFrame(wedge)
    df_wedge.columns = ['r1','r2']
    
    df_area = pd.DataFrame(area)
    df_area.columns = ['wedge_area','inlet_area','outlet_area', 'total_area'] 
    
    df_src = pd.DataFrame(src)
    df_src.columns = ['src_pos_ratio', 'src_ssp_avg300']
    
    #Overwrite data fiel with new feature columns
    data = data.drop(columns = 'profile') 
    data = pd.concat([data, df_src, df_area['total_area'], df_cmat], axis=1, sort=False) #df_wedge is out       

    return data

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

import os
path = os.getcwd()+'\data\\'

rawdata = LoadData(path)
ssp = pd.read_excel(path+"env.xlsx", sheet_name = "SSP")
ssp_grad = pd.read_excel(path+"env.xlsx", sheet_name = "SSP_GRAD")
ssp_prop = pd.read_excel(path+"env.xlsx",  sheet_name = "SSP_PROP")
data = FeatDuct(rawdata, Input_Only = True)
