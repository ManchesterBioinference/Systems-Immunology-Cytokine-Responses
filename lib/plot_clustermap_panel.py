
import numpy as np
import seaborn as sns
import scipy
import sklearn
import pandas as pd

import matplotlib.colors as colors
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
from matplotlib.pylab  import savefig

from sklearn.decomposition import PCA
from sklearn import metrics
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform




def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap



def plot_clustermap(datapanel,dist='seuclidean',mean = None, fname = None,
                    cbar_title = None,vmin = None, vmax=None, gp=True, 
                    fontsize = 'medium',addlabel=None, cmap = None,
                    cbar_ticks = None, cbar_ticklabels = None, cbar_labelsize = 10):
    sns.set(style="dark")
    row_linkage, col_linkage = my_linkage(datapanel= datapanel, dist = dist)
    if mean is None:
        CS_mean = datapanel.mean().stack('cytokine')
    else:
        CS_mean = mean.copy()
#     new_cyto_label = [c+':   ' +inv_CytoDict[c][0] for c in CS_mean.index.values]
    CS_mean_newCyto = CS_mean.copy()
    if gp == True:
        new_cyto_label = [c+':   ' + ', '.join(inv_CytoDict[c]) for c in CS_mean_newCyto.index.values]
        CS_mean_newCyto.loc[:,'Cytokines and their groups'] = new_cyto_label
        CS_mean_newCyto = CS_mean_newCyto.reset_index(drop=True).set_index('Cytokines and their groups',drop=True)
    
 

    if cmap is None:
        cmap = plt.get_cmap('RdBu_r') 
        new_cmap = truncate_colormap(cmap, 0.5,1,n=28*16)
    else:
        new_cmap = cmap

    clm=sns.clustermap(CS_mean_newCyto,vmax = vmax, vmin = vmin,# norm=MidpointNormalize(midpoint=0), 
                       cmap=new_cmap, #mask = mask,
#                        norm = colors.PowerNorm(gamma=1.0/3),
#                        norm=colors.SymLogNorm(linthresh=.1, linscale=.1),#, vmin=-1.0, vmax=1.0),
    #                    CS_mean_newCyto.max().max()
                       figsize=(10,10),col_linkage=col_linkage,row_linkage=row_linkage,
                       cbar_kws = {'orientation':'vertical','format':'%1i',
                                   'ticks':cbar_ticks,
                                  }#,'label':''}# cblar label and tick format
                       )
    clm.cax.tick_params(labelsize=cbar_labelsize)                           
    if cbar_title is not None:
        clm.cax.set_title(cbar_title)
#     if cbar_ticks is not None:
#         clm.cax.set_yticks(cbar_ticks)
    if cbar_ticklabels is not None:    
        clm.cax.set_yticklabels(cbar_ticklabels)
        
    ylbl = clm.ax_heatmap.get_yticklabels()
    _=clm.ax_heatmap.set_yticklabels(ylbl,rotation=0,fontsize = fontsize)
    if fontsize == 'large':
        xlbl = clm.ax_heatmap.get_xticklabels()
        _=clm.ax_heatmap.set_xticklabels(xlbl,fontsize = fontsize,rotation = 25)

    if addlabel is not None:
        plt.suptitle(addlabel,fontsize=18,fontweight='bold')
    
    if fname is not None:
        clm.savefig(fname,bbox_inches='tight')
        
        

# from numbapro import jit, float32

def distcorr(X, Y):
    """ Compute the distance correlation function
    
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor


def my_linkage(datapanel,dist = 'correlation',return_dist = False):
    
    # stimuli
    df = datapanel.stack('cytokine')
    df.index.names = ['number', 'cytokine']
    df.columns.names = ['stimulus']
    X = df.values.T
    
    if dist == 'seuclidean':
        V= np.var(X,axis=0)
    #     V= np.linalg.norm(X,axis=0)
        V[V==0]=1e-6
        Y = pdist(X, 'seuclidean',V=V)
    elif dist == 'c_correlation': # data centered correlation
        V= np.mean(X,axis=0)
        X = X-V
        Y = pdist(X, 'correlation')
    else:
        Y = pdist(X, dist)
    col_dist = Y
    col_linkage=linkage(Y,method='average')

    # cytokine linkage

    df = datapanel.stack('stimulus')
    df.index.names = ['number', 'stimulus']
    df.columns.names = ['cytokine']
    X = df.values.T

    if dist in ['seuclidean', 'euclidean'] :
        V= np.var(X,axis=0)
        V[V==0]=1e-6
        Y = pdist(X, 'seuclidean',V=V)
    elif dist == 'c_correlation': # data centered correlation
        V= np.mean(X,axis=0)
        X = X-V
        Y = pdist(X, 'correlation')
    else:
        Y = pdist(X, dist)

    row_dist = Y
    row_linkage=linkage(Y,method='average')

    if return_dist == True:
        return row_linkage, col_linkage, row_dist,col_dist
    else:
        return row_linkage, col_linkage







