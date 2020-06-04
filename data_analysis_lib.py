# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 20:36:24 2020

@author: kubap
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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