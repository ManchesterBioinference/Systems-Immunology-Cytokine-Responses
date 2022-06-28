import pandas as pd
# from matplotlib.mlab import PCA
from sklearn.decomposition import PCA

def print_pca_table(X, n_wt = 10,centered = False):
    if isinstance(X, pd.DataFrame):
        X = X.values
#     results = PCA(X)
#     fracs = results.fracs
#     Wt = results.Wt
#     Y = results.Y
        
    if centered == True:
        X = X.astype('float')  # Since X is object
#         m = (X.T).mean(0)
#         X = (X.T - m).T

        X = X - X.mean(0)
        X = X/X.std(0)
        
    pca = PCA()
    pca.fit(X)
    fracs = pca.explained_variance_ratio_
    f = [float("{0:.4f}".format(i)) for i in fracs]
    Y = pca.fit_transform(X)
    Wt = pca.components_  # n_redim, n_feature

    w1 = Wt[0]
    if sum(w1<0)*1.0/len(w1) > .5:  #too many negative ele's
        Wt[0] = -1*Wt[0]
        Y[:,0] = -1*Y[:,0]

    return f,Wt