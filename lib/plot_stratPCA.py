
from matplotlib import pyplot as plt

import os
import numpy as np
import scipy
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.kde import gaussian_kde
import sklearn
from matplotlib.pylab  import savefig
import pandas as pd
import matplotlib.colors as colors


import lib.plot_pca_sk as plot_pca_sk

def plot_stratPCA(data, cyto, CytoDict, StimDict, HAVE_PHA = True, prefix = 'FigureX', addlabel='(b) ', OutputDir= os.getcwd()):


    df_cur = data.copy()
    df_cyto = df_cur[df_cur.index.isin(CytoDict[cyto], level=0)]
    clust_cyto = df_cyto.index.get_level_values(0).values

    

    
    
    if cyto == 'Pro-inflammatory':
        stimgroups = ['All', 'Bacterial']
        fig = plt.figure(figsize=(12,6))
        gs = gridspec.GridSpec(1, 2,
                       width_ratios=[1,1])
    elif cyto == 'Anti-Viral':
        stimgroups = ['All', 'Viral']
        fig = plt.figure(figsize=(12,6))
        gs = gridspec.GridSpec(1, 2,
                       width_ratios=[1,1])
    else:
        stimgroups = ['All', 'Viral','Bacterial']
        fig = plt.figure(figsize=(12,12))
        gs = gridspec.GridSpec(2, 2,
                       width_ratios=[1,1],
                       height_ratios=[1,1])
    plot_weight = 1        
    for i,stim in enumerate(stimgroups):
        if stim == "All":
            stimlist = StimDict[stim]
            if HAVE_PHA == False:
                stimlist.remove('PHA')
                subtitle = stim + " stimuli (with log-media-normalisation), excl. T-cells (PHA)"
            else:
                subtitle = stim + " stimuli as features"
            legendon = 'best'
        else:
            stimlist = StimDict[stim]
            subtitle = stim + " stimuli as features"
            legendon = False

        ax = fig.add_subplot(gs[i])
        if cyto == 'Pro-inflammatory':
            if stim == 'Bacterial':
                k_repel = 0.01
                extra_w = 1.6
            elif stim == 'Viral':
                k_repel = 0.02
                extra_w = 1
            else:
                k_repel = 0.005
                extra_w = 1.2
        elif cyto == 'Th1':
            if stim == 'Bacterial':
                k_repel = 0.01
                extra_w = 1.5
            elif stim == 'Viral':
                k_repel = 0.02
                extra_w = 1
            else:
                k_repel = 0.01
                extra_w = 1.1
        elif cyto == 'Anti-Viral':
#             plot_weight = -1  # shift left and right so that right panel has more variables
            if stim == 'Bacterial':
                k_repel = 0.03
                extra_w = 1.55
            elif stim == 'Viral':
                k_repel = 0.001
                extra_w = 1
            else:
                k_repel = 0.004
                extra_w = 1.2
        elif cyto == 'Th2/proTh2':
            if stim == 'Bacterial':
                k_repel = 0.01
                extra_w = 1.5
            elif stim == 'Viral':
                k_repel = 0.02
                extra_w = 1
            else:
                k_repel = 0.01
                extra_w = 1.1         
        ax = plot_pca_sk.plot_pca_repel(df_cyto.loc[:,stimlist],df_cyto.index.get_level_values(0).values, 
                      markevery=2,alpha =1, plot_weight = plot_weight,legendon = legendon, 
                      ax=ax, PCname = [1,2],extra_w = extra_w,k_repel = k_repel,centered = True)
        ax.set_title(subtitle)

        ax.set_aspect('equal')

    plt.suptitle(addlabel + 'PCA on ' + cyto + ' Cytokine Data',fontsize = 18,fontweight='bold')
    if HAVE_PHA == True:
        fname = cyto + "June18.pdf"
    else:
        fname =  'NOMEDIANOPHA'+ cyto+"Cyto"+"June18.pdf"
    fname = prefix + fname
    savefig(OutputDir + fname.replace('/','-'), bbox_inches="tight", dpi = 300)
