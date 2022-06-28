
import scipy
# from matplotlib.mlab import PCA
from sklearn.decomposition import PCA
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.pylab  import savefig
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx
import lib.adjustText as adjustText

## Global 
# mycolor = scipy.io.loadmat('mycolor.mat') 
# mycolor = mycolor['mycolor']
# mycolor = mycolor.tolist()[0][-1].tolist() # largest color set
# mycolor.remove(mycolor[3])  # remove black
# mycolor.remove(mycolor[-5])  # remove light yellow
mycolor = ['#0000FF', '#FF0000', '#00FF00', '#00002C', '#FF1AB9', '#FFD300',
       '#005800', '#8484FF', '#9E4F46', '#00FFC1', '#008495', '#00007B',
       '#95D34F', '#F69EDC', '#D312FF', '#7B1A6A', '#F61261', '#FFC184',
       '#232309', '#8DA77B', '#F68409', '#847200', '#72F6FF', '#9EC1FF',
       '#72617B', '#9E0000', '#004FFF', '#004695', '#D3FF00', '#B94FD3',
       '#3E001A', '#EDFFB0', '#FF7B61', '#46FF7B', '#12A761', '#D3A7A7',
       '#D34F84', '#6A00C1', '#2C6146', '#0095F6', '#093E4F', '#A75809',
       '#72613E', '#099500', '#9E6AB9', '#FFFF72', '#A7F6CA', '#95B0B9',
       '#B0B009', '#2C004F']

mycolor.remove(mycolor[3])  # remove black
mycolor.remove(mycolor[-5])  # remove light yellow

# paircolors = sns.color_palette("Paired",12) #'Set1',9 'Set2', 9 'Set3', 12 'Paired',12
# paircolors.remove(paircolors[10]) # remove yellow


# # Ref: http://xkcd.com/color/rgb/
# paircolors =  sns.xkcd_palette(['purple','dark blue','blue','red',"windows blue", "amber", 
#                                        "faded green", "indigo",'royal blue',"faded green",'black']) 

paircolors = sns.xkcd_palette(['purple', 'blue','brown','dark green','red','orange', 'royal blue', 
                              'olive','violet','wine','forest','deep pink'])

def plot_pca(X, clust, clust_lbl = None, clabel=None, markevery=2, legendon = True,
             alpha =0.65,ax=None, plot_weight = 1, fname = None, ftitle = None,
             PCname = [1,2],extra_w = 1):
#     if X is None:
#         if data is None:
#             error = "Cannot plot with `X` or `data`"
#             raise ValueError(error)
#         else:
#             X = data.values
#             clabel = data.columns.values
    if isinstance(X, pd.DataFrame):
        if plot_weight!=0:
            clabel = X.columns.values
        X = X.values
    elif (clabel is None) and (plot_weight !=0):
        error = 'need column label `clabel` to plot weights'
        raise ValueError(error)
        

#     results = PCA(X)
#     fracs = results.fracs
#     Wt = results.Wt    # n_redim, n_feature
#     Y = results.Y      # n_sample, n_redim

    pca = PCA()
    pca.fit(X)
    fracs = pca.explained_variance_ratio_
    Y = pca.fit_transform(X)
    Wt = pca.components_  # n_redim, n_feature
    
    PC = np.int32(np.asarray(PCname)-1)
    f = [float("{0:.2f}".format(i*100)) for i in fracs]
    w1 = Wt[0]
    if sum(w1<0)*1.0/len(w1) > .5:  #too many negative ele's
        Wt[PC[0]] = -1*Wt[PC[0]]
        Y[:,PC[0]] = -1*Y[:,PC[0]]
        
    x = Y[:,PC[0]]
    y = Y[:,PC[1]]
    
    if ax==None:
        fig1 = plt.figure() # Make a plotting figure
        fig1.set_size_inches(8,8)
        ax = fig1.add_subplot(111)

    pltData = [x,y]    
    
    
    if plot_weight != 0:
        #     # weight color
        w_col = itertools.cycle(paircolors)

        W = Wt[PC,:]
        ratio = max(np.linalg.norm(pltData,axis=0))/max(np.linalg.norm(W,axis=0))  # 1.5 picked by eyes
        W = W*ratio
        a_range = np.linspace(.3,1,W.shape[1])
        ylist = np.array(np.infty)   # weight y list
        
        xmin = min(x)
        xmax = max(x)
        ymin = min(y)
        ymax = max(y)
        
        maxl = max(ymax-ymin,xmax-xmin)
   
        for i,(w,wc) in enumerate(zip(W.T,w_col)):
            d = abs(w[1] - ylist)
            ct = len(d[d < 1.0*maxl/100]) # number of 'closeby' points
            ylist = np.append(ylist, w[1])
            
            if w[0] < 0:
                horizontalalignment = 'right'
                linespacing = np.power(1.3,min(ct+1,3)) # default 1.2
#                 shift = (ct+1)*np.sign(w[0])*(np.max(abs(x)))/8  # value 10 is picked up by eyes
                ratio2 = np.power(1.05,min(ct,3))
                if ct > 1:
                    shift = 0
                else:
                    shift = (xmax-xmin)/20  # value 10 is picked up by eyes
                
            else:
                horizontalalignment = 'left'
                linespacing = np.power(1.25,min(ct+1,3))
                shift = 0*(xmax-xmin)/60    # no need for right half plane
                ratio2 = np.power(1.02,min(ct,3))
#                 shift = (ct+1)*np.sign(w[0])*(np.max(abs(x)))/30    # no need for right half plane
            w = w*extra_w  # manual hacking
            wLine = ((0,w[0]), (0,w[1])) 
            ax.plot(wLine[0], wLine[1],  color = wc, linestyle='-', linewidth=.5)
        
            xl,yl = w*ratio2
            ## comment the following since hl and linespacing work better than my ratio and shift
#             ax.text(xl+shift*np.sign(w[0]), yl,clabel[i],color=wc, 
#                     fontsize ='x-small',alpha = 1)
            ax.text(xl, yl,clabel[i],color=wc, 
                    fontsize ='xx-small',alpha = 1,
                    horizontalalignment = horizontalalignment,
                   linespacing = linespacing, fontweight  = 'semibold')


    ax = plot_model2d([x,y], clust, clust_lbl = clust_lbl, 
                      legendon = legendon,ax = ax, markevery = markevery, alpha = alpha)
        
    
    # make simple, bare axis lines through space:
    xAxisLine = ((min(pltData[0]), max(pltData[0])),  (0,0)) # 2 points make the x-axis line at the data extrema along x-axis 
    ax.plot(xAxisLine[0], xAxisLine[1],  'grey') # make a red line for the x-axis.
    yAxisLine = ((0, 0), (min(pltData[1]), max(pltData[1]))) # 2 points make the y-axis line at the data extrema along y-axis
    ax.plot(yAxisLine[0], yAxisLine[1],  'grey') # make a red line for the y-axis.

    # label the axes 
    xlbl = 'PC' + str(PCname[0]) +' ' + str(f[PC[0]]) + '\%'
    ylbl = 'PC' + str(PCname[1]) +' ' + str(f[PC[1]]) + '\%'
    ax.set_xlabel(xlbl) 
    ax.set_ylabel(ylbl) 

    if ftitle != None:
        ax.set_title(ftitle)
    ax.grid(color='grey', linestyle='-', linewidth=.2)
    ax.set_aspect('equal')      
#     ax.set_aspect(1./ax.get_data_ratio())

    
    if fname != None:
        savefig(fname,bbox_inches='tight')
    return ax

def plot_model2d(X, clust, clust_lbl = None, legendon = True, ax=None, markevery=2, alpha = 1):    
    if ax==None:
        fig1 = plt.figure() # Make a plotting figure
        fig1.set_size_inches(8,8)
        ax = fig1.add_subplot(111)

    # colors choices
    colors = itertools.cycle(mycolor)
    # markers choices
#     markers = itertools.cycle(mlines.Line2D.filled_markers)
    markers = itertools.cycle(['o','+','s','^','v','<','>'])
    # prevent sorting elements
    elements,indexes = np.unique(clust,return_index=True)
    elements = [clust[index] for index in sorted(indexes)]
    x = X[0]
    y = X[1]
    for i,(el,marker,col) in enumerate(zip(elements,markers,colors)):
        if clust_lbl == None:
            label = el
        else:
            label = '  \\textbf{' + el + '}' +':\n %s ' % ',\n'.join(map(str, clust_lbl[el])) 
        ax.set_aspect('equal')        
        ax.plot(x[clust==el],y[clust==el],linestyle ='None',
                marker =  marker, markevery=markevery,
                markerfacecolor='None',
                markeredgecolor=col,
                markersize=4, markeredgewidth=1 ,
                alpha=alpha, label= label)
    if legendon == True:
        ax.legend( markerscale =1.5, loc='center left',bbox_to_anchor=(1,0.5),
                   labelspacing=.8, fancybox=True, shadow=True, ncol=1, fontsize ='x-small')
    elif legendon == 'best':
        ax.legend( markerscale =1.5, loc='best',#bbox_to_anchor=(1,0.5),
                   labelspacing=.8, fancybox=True, shadow=True, ncol=1, fontsize ='x-small')
        
    ax.set_facecolor('white')

    return ax



def plot_pca_filt(X, clust, filt, clust_lbl = None, clabel=None, markevery=2, legendon = True,
             alpha =0.65,ax=None, plot_weight = 1, fname = None, ftitle = None,
             PCname = [1,2],extra_w = 1, k_repel= None,centered = False):
#     if X is None:
#         if data is None:
#             error = "Cannot plot with `X` or `data`"
#             raise ValueError(error)
#         else:
#             X = data.values
#             clabel = data.columns.values
    if isinstance(X, pd.DataFrame):
        if plot_weight!=0:
            clabel = X.columns.values
        X = X.values
    elif (clabel is None) and (plot_weight !=0):
        error = 'need `clabel` to plot weights'
        raise ValueError(error)
    if centered == True:
        X = X.astype('float')  # Since X is object
#         m = (X.T).mean(0)
#         X = (X.T - m).T

        X = X - X.mean(0)
        X = X/X.std(0)
    pca = PCA()
    pca.fit(X)
    fracs = pca.explained_variance_ratio_
    Y = pca.fit_transform(X)
    Wt = pca.components_  # n_redim, n_feature
    
    PC = np.int32(np.asarray(PCname)-1)
    f = [float("{0:.2f}".format(i*100)) for i in fracs]
    w1 = Wt[0]
    if sum(w1<0)*1.0/len(w1) > .5:  #too many negative ele's
        Wt[PC[0]] = -1*Wt[PC[0]]
        Y[:,PC[0]] = -1*Y[:,PC[0]]
        
    x = Y[:,PC[0]]
    y = Y[:,PC[1]]
    
   
    
    if ax==None:
        fig1 = plt.figure() # Make a plotting figure
        fig1.set_size_inches(8,8)
        ax = fig1.add_subplot(111)
        

        
    
    # colors choices
    colors = itertools.cycle(mycolor)
    # markers choices
#     markers = itertools.cycle(mlines.Line2D.filled_markers)
    markers = itertools.cycle(['o','+','s','^','v','<','>'])

    # prevent sorting elements
    elements,indexes = np.unique(clust,return_index=True)
    elements = [clust[index] for index in sorted(indexes)]

    for i,(el,marker,col) in enumerate(zip(elements,markers,colors)):
        if el in filt: 
            if clust_lbl == None:
                label = el
            else:
                label = '  \\textbf{' + el + '}' +':\n %s ' % ',\n'.join(map(str, clust_lbl[el])) 
            ax.set_aspect('equal')        
            ax.plot(x[clust==el],y[clust==el],linestyle ='None',
                    marker =  marker, markevery=markevery,
                    markerfacecolor='None',
                    markeredgecolor=col,
                    markersize=4, markeredgewidth=1 ,
                    alpha=alpha, label= label)
    if legendon == True:
        ax.legend( markerscale =1.5, loc='center left',bbox_to_anchor=(1,0.5),
                   labelspacing=.8, fancybox=True, shadow=True, ncol=1, fontsize ='x-small')
    elif legendon == 'best':
        ax.legend( markerscale =1.5, loc='best',#bbox_to_anchor=(1,0.5),
                   labelspacing=.8, fancybox=True, shadow=True, ncol=1, fontsize ='x-small')
        
    ax.set_facecolor('white')
    pltData = [x,y]
    # make simple, bare axis lines through space:
    xAxisLine = ((min(pltData[0]), max(pltData[0])),  (0,0)) # 2 points make the x-axis line at the data extrema along x-axis 
    ax.plot(xAxisLine[0], xAxisLine[1],  'grey') # make a red line for the x-axis.
    yAxisLine = ((0, 0), (min(pltData[1]), max(pltData[1]))) # 2 points make the y-axis line at the data extrema along y-axis
    ax.plot(yAxisLine[0], yAxisLine[1],  'grey') # make a red line for the y-axis.
 
    if plot_weight != 0:
        #     # weight color
        if k_repel == None:
            ax = _plot_weight(ax,Wt,PC,pltData,clabel,extra_w)
        else:
#             ax = _plot_weight_repel(ax, Wt,PC,pltData,extra_w,clabel,k_repel)
            ax = _plot_weight_partialrepel(ax, Wt,PC,pltData,extra_w,clabel,k_repel)
                
    

    # label the axes 
    xlbl = 'PC' + str(PCname[0]) +' ' + str(f[PC[0]]) + '\%'
    ylbl = 'PC' + str(PCname[1]) +' ' + str(f[PC[1]]) + '\%'
    ax.set_xlabel(xlbl) 
    ax.set_ylabel(ylbl) 

    if ftitle != None:
        ax.set_title(ftitle)
    ax.set_aspect('equal')      
    ax.grid(color='grey', linestyle='-', linewidth=.2)
    if fname != None:
        savefig(fname,bbox_inches='tight')
    return ax

def _plot_weight(ax,Wt,PC,pltData,clabel,extra_w):
    w_col = itertools.cycle(paircolors)

    W = Wt[PC,:]
    ratio = max(np.linalg.norm(pltData,axis=0))/max(np.linalg.norm(W,axis=0))  # 1.5 picked by eyes
    W = W*ratio
    a_range = np.linspace(.3,1,W.shape[1])
    ylist = np.array(np.infty)   # weight y list
    x = pltData[0]
    y = pltData[1]
    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = min(y)

    maxl = max(ymax-ymin,xmax-xmin)

    for i,(w,wc) in enumerate(zip(W.T,w_col)):
        d = abs(w[1] - ylist)
        ct = len(d[d < 1.0*maxl/100]) # number of 'closeby' points
        ylist = np.append(ylist, w[1])

        if w[0] < 0:
            horizontalalignment = 'right'
            linespacing = np.power(1.3,min(ct+1,3)) # default 1.2
#                 shift = (ct+1)*np.sign(w[0])*(np.max(abs(x)))/8  # value 10 is picked up by eyes
            ratio2 = np.power(1.05,min(ct,3))
            if ct > 1:
                shift = 0
            else:
                shift = (xmax-xmin)/20  # value 10 is picked up by eyes

        else:
            horizontalalignment = 'left'
            linespacing = np.power(1.25,min(ct+1,3))
            shift = 0*(xmax-xmin)/60    # no need for right half plane
            ratio2 = np.power(1.02,min(ct,3))
#                 shift = (ct+1)*np.sign(w[0])*(np.max(abs(x)))/30    # no need for right half plane
        w = w*extra_w  # manual hacking
        wLine = ((0,w[0]), (0,w[1])) 
        ax.plot(wLine[0], wLine[1],  color = wc, linestyle='-', linewidth=.5)

        xl,yl = w*ratio2
        ## comment the following since hl and linespacing work better than my ratio and shift
#             ax.text(xl+shift*np.sign(w[0]), yl,clabel[i],color=wc, 
#                     fontsize ='x-small',alpha = 1)
        ax.text(xl, yl,clabel[i],color=wc, 
                fontsize ='xx-small',alpha = 1,
                horizontalalignment = horizontalalignment,
               linespacing = linespacing, fontweight  = 'semibold')
    return ax


 
def _unrep_labels(ax, x, y, labels, colors):
    for _x, _y, _label in zip(x,y,labels):
        if _x < 0:
            horizontalalignment = 'right'
            linespacing = 1.2
        else:
            horizontalalignment = 'left'
            linespacing = 1.2
        
        ax.text(_x, _y,_label,color=colors[_label], 
                fontsize ='x-small',
                horizontalalignment = horizontalalignment,
               linespacing = linespacing, fontweight  = 'semibold')
    return ax

 
def _plot_weight_org(ax,Wt,PC,pltData,clabel,extra_w=1):
    w_col = itertools.cycle(paircolors)

    W = Wt[PC,:]
    ratio = max(np.linalg.norm(pltData,axis=0))/max(np.linalg.norm(W,axis=0))  # 1.5 picked by eyes
    W = W*ratio

    for i,(w,wc) in enumerate(zip(W.T,w_col)):
        if w[0] < 0:
            horizontalalignment = 'right'
            linespacing = 1.2
        else:
            horizontalalignment = 'left'
            linespacing = 1.2
        w = w*extra_w  # manual hacking
        wLine = ((0,w[0]), (0,w[1])) 
        ax.plot(wLine[0], wLine[1],  color = wc, linestyle='-', linewidth=.5)

        ax.text(w[0], w[1],clabel[i],color=wc, 
                fontsize ='xx-small',alpha = 1,
                horizontalalignment = horizontalalignment,
               linespacing = linespacing, fontweight  = 'semibold')
    return ax

def plot_pca_repel(X, clust, clust_lbl = None, clabel=None, markevery=2, legendon = True,
             alpha =0.65,ax=None, plot_weight = 1, fname = None, ftitle = None,
             PCname = [1,2],extra_w = 1,k_repel = None,centered = False):
    if isinstance(X, pd.DataFrame):
        if plot_weight!=0:
            clabel = X.columns.values
        X = X.values
    elif (clabel is None) and (plot_weight !=0):
        error = 'need column label `clabel` to plot weights'
        raise ValueError(error)
        
    if centered == True:
        X = X.astype('float')  # Since X is object
#         m = (X.T).mean(0)
#         X = (X.T - m).T

        X = X - X.mean(0)
        X = X/X.std(0)
    pca = PCA()
    pca.fit(X)
    fracs = pca.explained_variance_ratio_
    Y = pca.fit_transform(X)
    Wt = pca.components_  # n_redim, n_feature
    
    PC = np.int32(np.asarray(PCname)-1)
    f = [float("{0:.2f}".format(i*100)) for i in fracs]
    w1 = Wt[0]
#     if ((sum(w1<0)*1.0/len(w1) > .5 ) or (plot_weight == -1)) :  #too many negative ele's on x axis
    if plot_weight == -1 :  #too many negative ele's on x axis
#         print('weight %g' % plot_weight)
#         print (sum(w1<0)*1.0/len(w1))
#         print('shift left to right')
        Wt[PC[0]] = -1*Wt[PC[0]]
        Y[:,PC[0]] = -1*Y[:,PC[0]]
        
    x = Y[:,PC[0]]
    y = Y[:,PC[1]]
    
    if ax==None:
        fig1 = plt.figure() # Make a plotting figure
        fig1.set_size_inches(8,8)
        ax = fig1.add_subplot(111)

    pltData = [x,y]    
    
    
    ax = plot_model2d([x,y], clust, clust_lbl = clust_lbl, 
                      legendon = legendon,ax = ax, markevery = markevery, alpha = alpha)
    if plot_weight != 0:
        
#         ax = _plot_weight_repel(ax, Wt,PC,pltData,extra_w,clabel,k_repel)
        ax = _plot_weight_partialrepel(ax, Wt,PC,pltData,extra_w,clabel,k_repel)
        
    
#     # make simple, bare axis lines through space:
    xAxisLine = ((min(pltData[0]), max(pltData[0])),  (0,0)) # 2 points make the x-axis line at the data extrema along x-axis 
    ax.plot(xAxisLine[0], xAxisLine[1],  'grey') # make a red line for the x-axis.
    yAxisLine = ((0, 0), (min(pltData[1]), max(pltData[1]))) # 2 points make the y-axis line at the data extrema along y-axis
    ax.plot(yAxisLine[0], yAxisLine[1],  'grey') # make a red line for the y-axis.

    # label the axes 
    xlbl = 'PC' + str(PCname[0]) +' ' + str(f[PC[0]]) + '\%'
    ylbl = 'PC' + str(PCname[1]) +' ' + str(f[PC[1]]) + '\%'
    ax.set_xlabel(xlbl) 
    ax.set_ylabel(ylbl) 

    if ftitle != None:
        ax.set_title(ftitle)
      
     
# ax.set_aspect(1./ax.get_data_ratio())

    ax.grid(color='grey', linestyle='-', linewidth=.2)
    ax.set_aspect('equal')
    if fname != None:
        savefig(fname,bbox_inches='tight')
    return ax
    


def _plot_weight_repel(ax, Wt,PC,pltData,extra_w,clabel,k_repel=None):
    W = Wt[PC,:]
    ratio = max(np.linalg.norm(pltData,axis=0))/max(np.linalg.norm(W,axis=0))  
    W = W*ratio*extra_w
    color_dict = {}
    for label,col in zip(clabel,itertools.cycle(paircolors)):
        color_dict[label] = col

    for i,(w,label) in enumerate(zip(W.T,clabel)):
        wLine = ((0,w[0]), (0,w[1]))
        ax.plot(wLine[0], wLine[1],  color = color_dict[label], linestyle='-', linewidth=.5)
    if k_repel != None:
        ax = repel_labels(ax, W[0], W[1], clabel, colors =color_dict, k=k_repel)
    else:
        xs = W[0]
        ys = W[1]
        texts = []
        for _x, _y, _t in zip(xs, ys, clabel):
            texts.append(ax.text(_x,_y,_t))
        reload(adjustText)
        adjustText.adjust_text(xs, ys, texts,color_dict,ax=ax, 
                               prefer_move = 'x', only_move = {'text':['x'],'points':['x']},
                               #arrowprops=dict(arrowstyle="-", color='k', lw=0.5),
                                bbox={'pad':0, 'alpha':0}, size=7)
    return ax

def repel_labels(ax, x, y, labels,colors, k=0.01): # colors has to be a dictionary
    G = nx.DiGraph()
    data_nodes = []
    init_pos = {}
    
    for xi, yi, label in zip(x, y, labels):
        data_str = 'data_{0}'.format(label)
        G.add_node(data_str)
        G.add_node(label)
        G.add_edge(label, data_str)
        data_nodes.append(data_str)
        init_pos[data_str] = (xi, yi)
        init_pos[label] = (xi, yi)
        

    pos = nx.spring_layout(G, pos=init_pos, fixed=data_nodes, k=k)

    # undo spring_layout's rescaling
    pos_after = np.vstack([pos[d] for d in data_nodes])
    pos_before = np.vstack([init_pos[d] for d in data_nodes])
    scale, shift_x = np.polyfit(pos_after[:,0], pos_before[:,0], 1)
    scale, shift_y = np.polyfit(pos_after[:,1], pos_before[:,1], 1)
    shift = np.array([shift_x, shift_y])
    for key, val in pos.items():
        pos[key] = (val*scale) + shift

    for label, data_str in G.edges():
        _xy = pos[data_str]
        if _xy[0] < 0:
            ha = 'right'
        else:
            ha = 'left'
        ax.annotate(label, color = colors[label],fontsize ='x-small',
                     fontweight  = 'semibold',
                    xy=pos[data_str], xycoords='data',
                    xytext=pos[label], textcoords='data',
                    horizontalalignment = ha,)
#                     arrowprops=dict(arrowstyle="->",
#                                 connectionstyle="angle3,angleA=0,angleB=-90"))
#                     arrowprops=dict(arrowstyle="->",
#                                     shrinkA=0, shrinkB=0,
#                                     connectionstyle="arc3", 
#                                     color='black') )
#     # expand limits
#     all_pos = np.vstack(pos.values())
#     x_span, y_span = np.ptp(all_pos, axis=0)
#     mins = np.min(all_pos-x_span*0.15, 0)
#     maxs = np.max(all_pos+y_span*0.15, 0)
#     ax.set_xlim([mins[0], maxs[0]])
#     ax.set_ylim([mins[1], maxs[1]])
    return ax



def _plot_weight_partialrepel(ax, Wt,PC,pltData,extra_w,clabel,k_repel=None):
    W = Wt[PC,:]
    ratio = max(np.linalg.norm(pltData,axis=0))/max(np.linalg.norm(W,axis=0))  
    W = W*ratio*extra_w
    color_dict = {}
    pos = {}
    for label,w,col in zip(clabel,W.T,itertools.cycle(paircolors)):
        color_dict[label] = col
        pos[label] = w
            
    for i,(w,label) in enumerate(zip(W.T,clabel)):
        wLine = ((0,w[0]), (0,w[1]))
        ax.plot(wLine[0], wLine[1],  color = color_dict[label], linestyle='-', linewidth=.5)

    if k_repel != None:
        
        x = scipy.spatial.KDTree(W.T)
        xymax = max(np.ptp(pltData,axis = 1))
        p =  x.query_pairs(xymax/15.0)  # hand picked 100
#         print len(p)
        if len(p) != 0:
            rep_loc = np.unique(np.vstack(list(p)).flatten())
        #     rep_loc = np.unique(np.vstack(list(p))[:,0])
#             print len(rep_loc)
            rep_label = [clabel[i] for i in rep_loc]
            unrep_label = [label for label in clabel if label not in rep_label ]
            rep_W = W[:,rep_loc]
            unrep_W = np.delete(W,rep_loc,axis=1)

            ax = repel_labels(ax, rep_W[0], rep_W[1], rep_label, colors =color_dict, k=k_repel)
    #         ax = repel_labels(ax, unrep_W[0], unrep_W[1], unrep_label, colors =color_dict, k=0.001)
            ax = _unrep_labels(ax, unrep_W[0], unrep_W[1], unrep_label, colors =color_dict)
        else:
            ax = _unrep_labels(ax, W[0], W[1], clabel, colors =color_dict)
            
    else:
        xs = W[0]
        ys = W[1]
        texts = []
        for _x, _y, _t in zip(xs, ys, clabel):
            texts.append(ax.text(_x,_y,_t))
        reload(adjustText)
        adjustText.adjust_text(xs, ys, texts,color_dict,ax=ax, 
                               prefer_move = 'x', only_move = {'text':['x'],'points':['x']},
                               #arrowprops=dict(arrowstyle="-", color='k', lw=0.5),
                                bbox={'pad':0, 'alpha':0}, size=7)
    return ax