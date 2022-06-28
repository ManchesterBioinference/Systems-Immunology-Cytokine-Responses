### Fit GMM with cov. type and no. of components chosen by BIC scores
import sklearn.mixture
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['figure.figsize'] =  (8,5)
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'serif'
from matplotlib.pylab  import savefig

### Fit GMM with cov. type and no. of components chosen by BIC scores
def gmm_fit(X,n_components_range = [5],
            cv_types = ['diag', 'spherical','tied', 'full'],
            n_init = 1000,
            random_state = None, #1234, # only for reproducibility of published results
            n_iter = 1000):
    lowest_bic = np.infty
    bic = []
#     np.random.seed(random_state) 
    for cv_type in cv_types:
        for n_components in n_components_range:
            np.random.seed(random_state) # this is different from using random_state in mixture.GMM

            ## Fit a mixture of Gaussians with EM
            gmm = sklearn.mixture.GMM(n_components=n_components, 
                                        covariance_type=cv_type,
                                        random_state = None,  # this actually make n_init meaningless
                                        n_init = n_init,n_iter=n_iter)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    print(best_gmm.covariance_type, best_gmm.score(X).mean())
    return best_gmm


## option to change the min_covar_range
def gmm_fit_min(X,n_components_range = [5],nr_init = 200, min_covar_range=[0.001]):
    np.random.seed(1234)
    lowest_bic = np.infty
    bic = []
    cv_types = ['diag', 'spherical', 'tied', 'full']
    for min_covar in min_covar_range:
        for cv_type in cv_types:
            for n_components in n_components_range:
    #             np.random.seed(1234)
                ## Fit a mixture of Gaussians with EM
                largest_lgr = -np.infty
                lgr = []
                for times in range(nr_init):
                    gmm_t = sklearn.mixture.GMM(n_components=n_components, 
                                        covariance_type=cv_type,n_iter=1000,
                                        min_covar = min_covar)
                    gmm_t.fit(X)
                    lgr.append(gmm_t.score(X).sum())
                    if lgr[-1] > largest_lgr:
                        largest_lgr = lgr[-1]
                        gmm = gmm_t
                bic.append(gmm.bic(X))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm
    return best_gmm

# arrange the cluster label so that the higher number, the more counts 
import numpy as np
def shift_idx(clust):
    clust_s = clust.copy()
    unique, counts = np.unique(clust_s, return_counts=True)
    I = np.argsort(counts)
    I = 1000*np.argsort(I)
    for u, i in zip(unique,I):
        clust_s[clust==u] = i
    clust_s = clust_s/1000  # label starting from 1
    clust_s = clust_s.astype(int)   
    unique, counts = np.unique(clust_s, return_counts = True)
    print(' %d  %s' % ( len(unique), str(counts.tolist())))
    return clust_s # label starting from 0
