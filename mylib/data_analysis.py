# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 20:36:24 2020

@author: kubap
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pycebox.ice import ice, ice_plot

def ClassImbalance(data, plot = False):
    target = 'num_rays'

    yclass, ycount = np.unique(data[target], return_counts=True)
    yper = ycount/sum(ycount)*100
    y_population = dict(zip(yclass, zip(ycount, yper)))
    
    #print("y-variance: ", data[target].var())
    #print("y-mean:",  data[target].mean())
    #data.describe()
    
    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        width = 0.5
        x = np.arange(len(yclass))
        bars = ax.bar(x, ycount, width, label='Class Distribution')
        ax.set_ylabel('Number of Samples')
        ax.set_xlabel('Class: Number of Rays')
        ax.set_title('Class Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(yclass)
        ax.grid()
        autolabel(bars)
        #ax.legend()
        
        for b, bar in enumerate(bars):
            height = bar.get_height()
            ax.annotate('{:.2f}%'.format(yper[b]),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom')
        
        fig2, ax2 = plt.subplots()
        x = np.arange(len(yclass))
        ax2.plot(x, np.cumsum(yper), '-ok')
        ax2.set_ylabel('Per-class Percentage of Total Dataset [%]')
        ax2.set_xlabel('Class: Number of Rays')
        ax2.set_xticks(x)
        ax2.set_xticklabels(yclass)
        ax2.set_title('Cumulative sum plot of class distributions')
        ax2.grid()
        
        plt.rcParams.update({'font.size': 20})

        fig3,ax3 = plt.subplots()
        width = 0.5
        x = np.arange(len(yclass))
        bars3 = ax3.bar(x, ycount, width, label = 'Number of samples')
        ax3.set_ylabel('Number of Samples')
        ax3.set_xlabel('Class: Number of Rays')
        ax3.set_title('Class Distribution')
        ax3.set_xticks(x)
        ax3.set_xticklabels(yclass)
        ax3.grid()
        #ax3.legend()
        autolabel(bars3)

        ax4 = ax3.twinx() 
        ax4.plot(x, np.cumsum(yper), '-ok', label = 'Cumulative sum')
        ax4.set_ylabel('Cumulative sum [%]')
        #ax4.legend()
        for i, txt in enumerate(np.cumsum(yper)):
            ax2.annotate('{:.2f}%'.format(txt),
            xy=(x[i], np.cumsum(yper)[i]), 
            xytext=(x[i]-0.65, np.cumsum(yper)[i]+0.2), 
            arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))
            
            ax4.annotate('{:.2f}%'.format(txt),
            xy=(x[i], np.cumsum(yper)[i]), 
            xytext=(x[i]-1, np.cumsum(yper)[i]+0.35), 
            arrowprops=dict(arrowstyle="-", connectionstyle="arc3"))
            
    return y_population

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    fig, ax = plt.subplots()
    plt.close()
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')        
 

def PlotCorrelation(dat, features, annotate = True):

        correlation_matrix = dat[features].corr(method = 'spearman').abs()
        # Set font scale
        sns.set(font_scale = 2) #2 for visibility
        # Set the figure size
        f, ax = plt.subplots(figsize=(10, 10))
        # Plot heatmap
        sns.heatmap(correlation_matrix, cmap= 'YlGnBu', square=True, annot=annotate, annot_kws = {"size": 12}, xticklabels = [])
        # Tight layout
        plt.title("Correlation Matrix")
        f.tight_layout()
        plt.show() 

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
    #max value for y-axis based on water-depth-min which goes through the whole range
    ymax = dict_of_ice_dfs[features[0]].values.max()
    
    for f, ax in zip(features, axes.flatten()):
        ice_plot(dict_of_ice_dfs[f], ax=ax, **ice_plot_kws)
        # add the rug
        sns.distplot(data_df[f], ax=ax, hist=False, kde=False, 
                     rug=True, rug_kws=rug_kws)
        #ax.set_title('feature = ' + f)
        ax.set_ylabel(ax_ylabel)
        ax.set_ylim(0, ymax)
        sns.despine()
        
    # get rid of blank plots
    for i in range(len(features), nrows*ncols):
        axes.flatten()[i].axis('off')
    return fig

def ICEPlot(data, model, features):
    # create dict of ICE data for grid of ICE plots
    train_ice_dfs = {feat: ice(data=data, column=feat, predict=model.predict) 
                     for feat in features}
    
    fig = plot_ice_grid(train_ice_dfs, data, features,
                        ax_ylabel='Pred. Ray Num.', 
                        nrows=5, 
                        ncols=4,
                        alpha=0.3, plot_pdp=True,
                        pdp_kwargs={'c': 'blue', 'linewidth': 2.0},
                        linewidth=0.5, c='dimgray')
    #fig.tight_layout()
    fig.suptitle('ICE plot: Classification - all training data')
    fig.subplots_adjust(top=0.89)
    
    return train_ice_dfs
    
from matplotlib.backends.backend_pdf import PdfPages

def Multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    
    
def dataframe_difference(df1, df2, which=None):
    """Find rows which are different between two DataFrames."""
    comparison_df = df1.merge(df2,
                          indicator=True,
                          how='outer')
    if which is None:
        diff_df = comparison_df[comparison_df['_merge'] != 'both']
    else:
        diff_df = comparison_df[comparison_df['_merge'] == which]
    #diff_df.to_csv('data/diff.csv')
    return diff_df

import os
from matplotlib.lines import Line2D

def bathy_plot():
    path = os.getcwd()+'\data\\'
    bathy =  pd.read_excel(path+ "env.xlsx", sheet_name = "BATHY")
    bathy = bathy.loc[bathy['bottom_type']==1]
    bathy1 = bathy.iloc[0:9,:]
    bathy2 = bathy.iloc[10:19,:]
    bathy3 = bathy.iloc[20:,:]
    bathy4 = bathy.iloc[[6,16],:]
    xt = [0] + np.unique(bathy['len_flat']).tolist() + np.unique(bathy['len_slope']+bathy['len_flat']).tolist()
    yt = [0] + bathy['d_start'][:11].tolist()
    fig,ax = plt.subplots()
    custom_lines = [Line2D([0], [0], color='lightcoral'),
                Line2D([0], [0], color='skyblue'),
                Line2D([0], [0], color='chartreuse')]
    #ax.legend(custom_lines, ["slope -2","slope 2", "slope 0"])
    for bat, clr in zip([bathy1,bathy2,bathy3], ['lightcoral','skyblue','chartreuse']):
        for dstart, dend, lenflat, lenslope in zip(bat['d_start'],bat['d_end'],bat['len_flat'],bat['len_slope']):
            ax.plot([0,lenflat,lenflat+lenslope,44000],[dstart,dstart,dend,dend],clr,lw=0.5)
            #plt.ylim(0,1550)
            #plt.gca().invert_yaxis()
    
    bathy4 = bathy.iloc[[6,16],:]
    for dstart, dend, lenflat, lenslope,clr in zip(bathy4['d_start'],bathy4['d_end'],bathy4['len_flat'],bathy4['len_slope'],['r','#0055ffff']):
        ax.plot([0,lenflat,lenflat+lenslope,44000],[dstart,dstart,dend,dend],color=clr,lw=2)
        ax.set_ylim(0,1560)
        ax.invert_yaxis()
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_title('Bathymetry Configurations')
    ax.set_xlabel('Range [m]')
    ax.set_ylabel('Depth [m]')
    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(custom_lines, ["slope -2","slope 2", "slope 0"], loc='upper center', bbox_to_anchor=(0.5, -0.115),
          fancybox=True, shadow=True, ncol=3)
    fig.set_size_inches(10,5)
    
