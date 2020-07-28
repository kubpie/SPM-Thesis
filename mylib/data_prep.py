# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:42:50 2020

@author: kubap
"""
import numpy as np
import pandas as pd
import seaborn as sns
import os

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC

from ssp_features import SSPStat
from data_analysis import ClassImbalance
""" 
### DATA PREPARATION & FEATURE VECTOR CREATION LIBRARY ###
Contents:
1. LoadData: load data from csv and select convergent scenarios
2. UndersampleData: creates a stratified subset less or equal (=<) n samples in each ray-class
3. XGB data prep:
    - EncodeData: One Hot encoder for categorical feautes and optionally the target feature
    - TrainTestSplit: Splits for XGB
    - CreateModelSplits: split data on slope value to create 3 sub-models
4. SSP Features:
    - FeatDuct: add or remove features connected to ducted propagation, used mostly with InputOnly = True to leave only 'raw' BELLHOP input data
    - FeatBathy: add missing bathymetry featues slope/flat bottom seg. lengths
    - FeatSSPId: creates feature vector from SSPId function of Deep Sound Channels Axis and Sonic Layer depths & gradients
    - FeatSSPStat: creates feature vector of statistical properties of SSP function - mean and std #TODO: Make it read from csv too instead of function
    - FeatSSPVec: creates feature vector representation of all SSP values until dmax (XGB Application)
    - FeatSSPOnDepth: retrieves SSP values only for critical depths appearing in the input (XGB Application, alternative to SSPVec, closest repr. to KGCN model)

IMPORTANT: Most of the functions depend on previously created env.xlsx to save time with relatively slow SSPId script!
Make sure to have env.xlsx created from ssp_features.py or downlaoded before running.
"""


def LoadData(path):   
        
    flist = [f for f in os.listdir(path) if "Dataset" in f]
    col = []
    frames = []

    def CheckCol(lst):
        return lst[1:] == lst[:-1] 
    
    for file in flist: #to check fix it latest
        raw_data = pd.read_csv( str(path)+"/"+file, index_col=None, header=0)
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
    
    #print(f'Total size of the database: {len(mergedData)}')
    #print(f'Number of converged scenarios: {len(convData)}')
    #print(f'Number of nonconverged scenarios: {len(nonconverged)}')
    
    return convData
    
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

def FeatBathy(data, path):    
    Bathy = pd.read_excel(str(path)+"\env.xlsx", sheet_name = "BATHY")
    wedge = np.zeros([len(data),2]) #wedge parameters, bathymetry info
    
    for dmin, dmax, slope, row in zip(data['water_depth_min'], data['water_depth_max'], data['wedge_slope'], range(len(data)) ): 
        ### Wedge Loop
        if slope == 0 or slope == -2:
            dstart = dmin
            dend = dmax
        else:
            dstart = dmax
            dend = dmin     
        find_lenflat = Bathy.loc[(Bathy['d_start'] == dstart) & (Bathy['d_end'] == dend), 'len_flat']
        lenflat = find_lenflat.values[0]        
        find_lenslope = Bathy.loc[(Bathy['d_start'] == dstart) & (Bathy['d_end'] == dend), 'len_slope']
        lenslope = find_lenslope.values[0]
        wedge[row, 0] = lenflat
        wedge[row, 1] = lenslope        
            
    df_wedge = pd.DataFrame(wedge)
    df_wedge.columns = ['len_flat','len_slope']
    # Choose only slope length, len_flat is redundant
    data = pd.concat([data, df_wedge.iloc[:,1]], axis=1, sort=False)    
    return data
    
def FeatSSPVec(data, path):
    ssp = pd.read_excel(str(path)+"\env.xlsx", sheet_name = "SSP_LEVITUS") #changed for original sampling!!
    depth = ssp['DEPTH'].values.tolist()
    
    cmat = np.zeros([len(data),len(depth)]) #segmented & interpolated sound speed profile vector   
    weight = np.zeros([len(data),len(depth)]) #weights on the SSP
    
    
    for dmin, dmax, profile, slope, row in zip(data['water_depth_min'], data['water_depth_max'], data['profile'], data['wedge_slope'], range(len(data)) ):

        ### SSP-vec Loop        
        d = min(depth, key=lambda x:abs(x-dmin))
        # d is a depth approximation in case that ssp sampling doesn't match the grid in Bellhop
        idx = depth.index(d)+1
        # idx matches the index of ssp-vec entry with max_depth in each scenarion
        # so in flat-bottom scn only a part of ssp is used
        
        if slope == 0:
            weight[row,0:idx] = 1.0
            cmat[row,0:idx] = ssp[profile].iloc[0:idx]
                
        else:
            rmax = 44000
            dmax = 1500 
            for dz in range(len(depth)):
                wedge_range = np.round((dmax-dmin)/np.tan(np.deg2rad(abs(slope))))
                start_wedge = 0.5*(rmax - wedge_range)
                ds = np.round((dmax-depth[dz])/np.tan(np.deg2rad(abs(slope))))
                
                # SSP weight matrix for changing the values of SSP-vec with respect to 
                # 'totale coverage' of the water column, so kinda fittign on the bathymetry shape
                # The weight matrix influence is controlled by weight-parameter
                # gamma! If gamma = 0/0 weight is 1.0 everywhere, effectively turning off the 
                # influence of the weight matrix
                
                gamma = 0.0
                
                weight[row,dz] = ((rmax - gamma*(start_wedge+wedge_range-ds))/rmax) 
                cmat[row,dz] = ssp[profile].iloc[dz]*weight[row,dz]
           
                              
    colnames = []         
    for i in range(len(depth)):
        colnames.append("SSPd-"+str(depth[i]))    
    df_cmat = pd.DataFrame(cmat)
    df_cmat.columns = colnames
    data = pd.concat([data, df_cmat], axis=1, sort=False)    
    #data = data.drop(columns = 'profile')
    return data

def FeatSSPId(data, path, src_cond):
    
    ssp_prop = pd.read_excel(str(path)+"\env.xlsx",  sheet_name = "SSP_PROP")
    # beacause 0 is a meaningful value, to allocate space, use an array of NaNs
    dat = np.empty([len(data), len(ssp_prop.columns)])
    dat[:,:] = np.nan
    df_sid = pd.DataFrame(dat, columns = ssp_prop.columns)
    
    for profile,dmax,row in zip(data['profile'], data['water_depth_max'], range(len(data))):
        match = ssp_prop.loc[(profile == ssp_prop['SSP']) & (dmax == ssp_prop['dmax']), :]
        if not match.empty:
            df_sid.loc[row,:] = match.values[0]
            
    df_sid = df_sid.drop(columns = ['SSP', 'dmax'])
    data = pd.concat([data, df_sid], axis=1, sort=False)
    # With this condition the src position will be checked to indicate whether sound propagates 
    # in specific duct type SD/BD/DC and not only if those ducts exists on a given SSP
    # This includes EXPERT KNWOLEDGE in creation of feature vector beyond SSP identification
    # The SSP_id feature entries of the table will be turned to NaN if the conditions are violated
    # In effect the SSP_id features are more sparse but represent the 'reality' more precisely
    if src_cond:
        dccols = [col for col in data.columns if 'DC' in col ]
        sldcols = [col for col in data.columns if 'SLD' in col ]
  
        for src, sld, dctop, dcbot, row in zip(data['source_depth'], data['SLD_depth'], data['DC_top'], data['DC_bot'], range(len(data))):
            # no SD/BD propagation
            if sld < 30 or src >= sld: 
                data.loc[row,sldcols] = np.nan   
            #no DC propagation
            if src < dctop or src > dcbot:
                data.loc[row,dccols] = np.nan
                
    return data

def FeatSSPStat(data, path):
    
    SSP_Input = pd.read_excel(path+"env.xlsx", sheet_name = "SSP")
    ssp = pd.read_excel(path+"env.xlsx", sheet_name = "SSP")
    SSP_Stat = SSPStat(SSP_Input, path, plot = False, save = False)
    depth = ssp['DEPTH'].values.tolist()
    
    stats = ['mean_SSP','stdev_SSP','mean_grad','stdev_grad']
    df_stat = pd.DataFrame(np.zeros([len(data), 4]), columns = stats)

    for profile, dmax, row in zip(data['profile'], data['water_depth_max'], range(len(data))):                
        df_stat.iloc[row,:] = SSP_Stat[profile].iloc[depth.index(dmax)].values
    
    data = pd.concat([data, df_stat], axis=1, sort=False)
    
    return data      

def FeatSSPOnDepth(data_sspid, path, save = False):
    # Retrieve SSP values only for critical depths appearing in the input
    SSP_Input = pd.read_excel(str(path)+"\env.xlsx", sheet_name = "SSP")
    crit_depths = ['water_depth_min', 'water_depth_max', 'source_depth', 'DC_axis', 'DC_bot', 'DC_top', 'SLD_depth']
    crit_ssp = ['SSP_wmin', 'SSP_wmax', 'SSP_src', 'SSP_dcax', 'SSP_dcb', 'SSP_dct', 'SSP_sld']
    dat = np.empty([len(data_sspid),len(crit_ssp)])
    dat[:,:] = np.nan
    df_sdep = pd.DataFrame(dat,columns = crit_ssp)
    
    for col in data_sspid[crit_depths].columns:
        for row, dep, profile in zip(range(len(data_sspid)), data_sspid[crit_depths].loc[:,col], data_sspid['profile']):         
            sspcol = crit_ssp[crit_depths.index(col)]
            if not np.isnan(dep):    
                d = min(SSP_Input['DEPTH'], key=lambda x:abs(x-dep))
                dep = d
            sspval  = SSP_Input[profile].loc[SSP_Input['DEPTH'] == dep]
            if not sspval.empty:
                df_sdep[sspcol].iloc[row] = sspval.values[0]
    
    data = pd.concat([data_sspid,df_sdep], axis = 1, sort = False)
    if save:
        data.to_csv(str(path) + '/data_complete.csv', index = None, header = True)
        print("New datafiles have been created!")                

    return data

def EncodeData(data):
    SeasonList = []
    LocationList = []
    # Split 'profile' into features 'location' and 'season' and remove 'profile' permamently
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
        alph_srt = sorted(np.unique(feature))
        feature = label_encoder.fit_transform(feature)
        feature = feature.reshape(feature.shape[0], 1)
        onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        feature = onehot_encoder.fit_transform(feature)
        #name = ["season", "location"]
        feature = pd.DataFrame(feature)
        #feature.columns = [name[f]+ "-" + str(i) for i in range(feature.shape[1])]
        feature.columns = alph_srt
        data_enc = pd.concat((data_enc, feature), axis=1)
        
    #TODO: See if encoding target is still necessary with the new method
    # encode y for k-folded valdiation
    #label_encoder = LabelEncoder()
    #label_encoder = label_encoder.fit(y)
    #y_enc = label_encoder.transform(y)
    #y_enc = y
    return data_enc 

def UndersampleData(data, max_sample):
    
    target = np.unique(data['num_rays'])
    random_state = 27
    y_population = ClassImbalance(data,plot = False)
    
    data_sampled = pd.DataFrame(columns = data.columns)
    
    for raynr in target:
        if y_population[raynr][0] > max_sample:
            data_slice = data.loc[data['num_rays'] == raynr]
            data_sample = data_slice.sample(n = max_sample, random_state = random_state)
            data_sampled = data_sampled.append(data_sample, ignore_index = False)
        else:
            data_sampled = data_sampled.append(data.loc[data['num_rays'] == raynr], ignore_index= False)
            
    return data_sampled

def CreateModelSplits(data, level_out = 1, remove_outliers = True, replace_outliers = True, feature_dropout = False, plot_distributions = False, plot_correlations = False):
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
    
    
    def constant_features(X, frac_constant_values = 0.95):
        # Get number of rows in X
        num_rows = X.shape[0]
        # Get column labels
        allLabels = X.columns.tolist()
        # Make a dict to store the fraction describing the value that occurs the most
        constant_per_feature = {label: X[label].value_counts().iloc[0]/num_rows for label in allLabels}
        # Determine the features that contain a fraction of missing values greater than threshold
        labels = [label for label in allLabels if constant_per_feature [label] > frac_constant_values]
        
        return labels
    
    # using more articulate names for optins in the func. def only
    replace = replace_outliers
    remove = remove_outliers
    if remove == False and replace != True:
        replace = remove #can't replace without starting remove procedure
                        #remove = False means no interruption into original distr.
    if remove == False and replace == True:
        print('Set remove to True, to allow replacement. This will allow to remove empty classes.')
        print('No outliers have been corrected')
    if remove:
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
        
              
            ystat = ClassImbalance(dat, plot = plot_distributions)
            distributions.append(ystat)  
            SplitSets.append(dat)  
        
        SplitSets[0] = remove_outliers(SplitSets[0], 6000)    
        distributions[1] = ClassImbalance(SplitSets[0], plot = plot_distributions)
        
    #2. TODO : A value is trying to be set on a copy of a slice from a DataFrame.
    #          Try using .loc[row_indexer,col_indexer] = value instead
    
    ### End of Outlier Correction
    
    ### Feature dropout
    if feature_dropout:
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


def SMOTSampling(X_train, y_train, min_class_size = 100):
    # UPSAMPLING 
    #check the label population - important for the value of max folds possible
    yclass, ycount = np.unique(y_train, return_counts=True)
    yper = ycount/sum(ycount)*100
    y_population = dict(zip(yclass, zip(ycount, yper)))

    # Upsampling with SMOT-ENC technique that can handle both cont. and categorical variables
    #categorical_var = np.hstack([2, np.arange(5,33)])
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    locations = ['Labrador-Sea','Mediterranean-Sea','North-Pacific-Ocean','Norwegian-Sea','South-Atlantic-Ocean','South-Pacific-Ocean']
    bottom = ['bottom_type']
    categorical_var =[]
    for cat in locations+seasons+bottom:
        cat_idx = X_train.columns.get_loc(cat)
        categorical_var.append(cat_idx)
    min_class_size = min_class_size
    population_target = dict.fromkeys(y_population.keys())
    for ray, nrsamples in y_population.items():
        if nrsamples[0] < min_class_size:
            population_target[ray] = min_class_size
        else:
            population_target[ray] = y_population[ray][0]
    smote_nc = SMOTENC(categorical_features=categorical_var, sampling_strategy=population_target, random_state=42)
    X_train = X_train.fillna(0) #smotenc doesn't work for NaNs, which is not good and changes sspid logics
    X_smot, y_smot = smote_nc.fit_resample(X_train, y_train)
    return X_smot, y_smot
