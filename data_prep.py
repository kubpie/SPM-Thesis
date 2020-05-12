# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 13:42:50 2020

@author: kubap
"""
import numpy as np
import pandas as pd
from os import listdir

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns


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
def FeatDuct(data, SSP_info = False):
        
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
    if SSP_info == False:
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


def CreateSets(data, plot_corr = False):
    
    data0 = data.loc[data['wedge_slope'] == 0]
    data2 = data.loc[data['wedge_slope'] == 2]
    dataN2 = data.loc[data['wedge_slope'] == -2]
        
    distributions = []
    alldata = []
    #check the datasets statistics: class popualtion, ...
    for dat in [data0, data2, dataN2]:
        yclass, ycount = np.unique(dat['num_rays'], return_counts=True)
        yper = ycount/sum(ycount)*100
        ystat = dict(zip(yclass, zip(ycount, yper)))
        distributions.append(ystat)
        #remove outliers on condition: predict classes with 10 or more samples
        outlier = np.where(ycount <= 10)
        for x in outlier[0]:
            out = dat.index[dat['num_rays'] == yclass[x]]     
            dat = dat.drop(out)
        alldata.append(dat)   
    #data0 = data0.loc[data['num_rays'] <= 2500]
    #data2 = data2.loc[data['num_rays'] <= 10000]    

    #LABEL TARGET AND FEATURES
    target = 'num_rays'
    features = data.columns.tolist()
    features.remove(target)
    
    redundant_feat = []
    alldata_new = []
    features_new = []
    # Remove redundant features in separate dataset (with constant values)
    for i, dat in enumerate(alldata):
        #corr_matrix = dat[features].corr(method = 'spearman').abs()
        redF = constant_features(dat[features], frac_constant_values = 0.90)
        redundant_feat.append(redF)
        alldata_new.append(dat.drop(columns = redF))
        featnames = [x for x in features if x not in redF]
        features_new.append(featnames)
                    
        
    features = features_new
    alldata = alldata_new

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

    
    
    return alldata, dtrain, dtest, features

from matplotlib.backends.backend_pdf import PdfPages

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

        
    
#path = #change path  #r'C:\Users\kubap\Documents\THESIS\DATA\\'

import os
path = os.getcwd()+'\\'

rawdata = LoadData(path)
ssp = pd.read_excel(path+"env.xlsx", sheet_name = "SSP")
ssp_grad = pd.read_excel(path+"env.xlsx", sheet_name = "SSP_GRAD")
ssp_prop = pd.read_excel(path+"env.xlsx",  sheet_name = "SSP_PROP")
data = FeatDuct(rawdata, SSP_info = False)
