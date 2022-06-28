"""standard setup.py used by all notebook in Project/code
   This loads commonly used functions/packages
"""

import h5py
import pdb

import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina' #'svg'
matplotlib.rcParams['figure.figsize'] =  (8,5)
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'sans-serif'
from matplotlib.pylab  import savefig

from collections import OrderedDict
import string
import scipy.io
import itertools
import setuptools   #to fixed Problems with weave compilation; issue #159 in GPy
# import GPy

print("System info:")
import sys
print("{}".format(sys.version))

import IPython
print("IPython version is {}".format(IPython.version_info))

print("Package Versions:")
import sklearn; print( "  scikit-learn:", sklearn.__version__)
import scipy; print( "  scipy:", scipy.__version__)
import statsmodels; print( "  statsmodels:", statsmodels.__version__)
import numpy as np; print ("  Numpy:", np.__version__)
import seaborn as sns; print( "  Seaborn:", sns.__version__)
import pandas as pd; print( "  Pandas:", pd.__version__)
#sns.set(rc={'image.cmap': 'cubehelix'})