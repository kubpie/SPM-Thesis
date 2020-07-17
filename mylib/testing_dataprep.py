import os
from pathlib import Path
import pandas as  pd
import matplotlib.pyplot as plt
from data_analysis import ClassImbalance, PlotCorrelation
from data_prep import FeatDuct, FeatBathy, FeatSSPId, FeatSSPStat, FeatSSPOnDepth
from data_prep import LoadData, UndersampleData
from data_prep import CreateModelSplits, TrainTestSplit, EncodeData
PATH = os.getcwd() #+'\data\\'
path = Path(PATH+"/data/")
ALLDATA = LoadData(path)

#####################################################
### Messy script for testing & plotting data prep ###
### and feature vector generating functions       ###
#####################################################
# TODO: may need an update!


data = FeatDuct(ALLDATA, Input_Only = True)
data = FeatBathy(data, path)
data = EncodeData(data)
#data3 = FeatSSPId(data2, path, src_cond = True)
#data4 = FeatSSPStat(data3,path)
#data5 = FeatSSPOnDepth(data4, path, save = True)

target = 'num_rays'
features = data.columns.tolist()
features.remove(target)
PlotCorrelation(data,features, annotate = False)


#ClassImbalance(data, plot = True)

"""
# SSP Identification
SSP_Input = pd.read_excel(path+"env.xlsx", sheet_name = "SSP")
SSP_Grad = SSPGrad(SSP_Input, path, save = False)
SSP_Stat = SSPStat(SSP_Input, path, plot = True, save = False)
SSP_Prop = SSPId(SSP_Input, path, plot = True, save = False)
"""
"""
# Feature Vector
#data1 = FeatDuct(rawdata, Input_Only = True)
#data2 = FeatBathy(data1, path)
#data3 = FeatSSPId(data2, path, src_cond = True)
#data4 = FeatSSPStat(data3,path)
#data5 = FeatSSPOnDepth(data4, path, save = True)
#data = UndersampleData(data, 100)
"""

"""
# Plot to see the effect of subsampling SSP for Grakn input
plot_sampling = False
if plot_sampling == True:
    fig, axes = plt.subplots(nrows = 3, ncols = 8, figsize = (15,20), sharey = True)
    axes = axes.flat
    axes[0].invert_yaxis()
    
    # This downsampling is used now in GRAKN. 0 is skipped in favor of first point at 10m
    # because grad at 0 is 0, which breaks the SSP-vec 'triplet'
    
    max_depth = [0, 50, 150, 250, 350, 450, 600, 750, 900, 1050, 1200, 1500]
    depth = SSP_Input.iloc[:,0]

    for i, ssp in enumerate(SSP_Input.iloc[:,1:]):
        axes[i].plot(SSP_Input.iloc[:,1:][ssp], depth, linewidth = 2, label = 'Sound Speed Profile' )
        axes[i].plot(SSP_Input.iloc[:,1:][ssp][depth.isin(max_depth)], depth[depth.isin(max_depth)], linewidth = 1, label = 'Subsampled Sound Speed Profile')
        axes[i].set_title("{}. {}".format(i, ssp))
"""

"""
# POLYFIT
best, allres = PolyfitSSP(SSP_Input)
deg = range(1,11)
z = np.array(SSP_Input.iloc[:,0]).astype(float)
znew = np.linspace(z[0], z[-1], num=len(z)*5)

#plt.plot(xnew,ffit,x,y)
fig, axes = plt.subplots(nrows = 3, ncols = 8, figsize = (15,20), sharey = True)
axes = axes.flat
axes[0].invert_yaxis()
for i, ssp in enumerate(SSP_Input.iloc[:,1:]):
    coeff = best[i][2]
    ffit = poly.polyval(znew, coeff) 
    
    axes[i].plot(ffit, znew)
    axes[i].plot(SSP_Input.iloc[:,i], z)
    axes[i].set_title("{}. {}".format(i, ssp))
"""

