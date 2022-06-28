import pandas as pd
import numpy as np
import scipy.io



def load_labels(dataDir):

    cytoStimList = scipy.io.loadmat(dataDir+'/cytoStimList.mat')   # from save cytoStimList.mat allCyto stStim
    allCyto = cytoStimList['allCyto']
    allCyto = allCyto.tolist()
    allCyto = allCyto[0]
    stStim = cytoStimList['stStim']
    stStim = stStim.tolist()  
    stStim = stStim[0]

    stimlabel = []
    for stim in stStim[:]:
        stimlabel.append(str(stim[0])) 
    cytolabel = []
    for stim in allCyto[:]:
        cytolabel.append(str(stim[0])) 
    
    return cytolabel, stimlabel    

def _mkDF(d,num,cyto,stim):
#     d has the shape 28*307*16
    d = d.swapaxes(0,1) #get the shape of 307 x 28 x 16
    d = d.reshape(d.shape[0],np.prod(d.shape[1:3]))

    df = pd.DataFrame(
        data = d,
        index = num,
        columns = pd.MultiIndex.from_product([cyto,stim])
    )
    df.columns.names = ['cytokine','stimulus']
    df.index.names = ['number']
    return df

def load_meanFoldChange(dataDir):
    MFmat = scipy.io.loadmat(dataDir+'/meanFold.mat')   # H_sigPairs
    meanFold = MFmat['meanFold']
    cytolabel, stimlabel = load_labels(dataDir)
    meanFold = pd.DataFrame(meanFold,index = cytolabel,columns=stimlabel[1:])
    return meanFold
    
def load_cytokine(dataDir,batch_adjust=True):

    
    

    # cytoData.mat is gernerated in MATLAB file dataPreprocessing.m with the following lines:
    # %% Save the data
    # save cytoData.mat   resp_raw     resp_log      resp_nrm  ...
    #     resp_raw_ad   resp_log_ad   resp_nrm_ad  ...
    #     S_Rcon_log  S_Rcon_nrm  ...
    #     batch stStim allCyto  number  dtl
    # % The following comments were in MATLAB file
    # % resp_raw: raw data 28*307*16 without any replacement for zeros
    # % resp_log: log normalized data with 0 replaced by half detction limit
    # % reps_nrm: normalise data with media so 28*307*15
    # % resp_raw_ad: raw data with batch effect removed 28*307*16
    # % resp_log_ad: log data with batch adjusted data
    # % resp_nrm_ad: normalise data first and then do batch adjustment; this will
    # %    be different from normalise resp_log_ad
    # % batach: batch data 307*1
    # % number: children number 307*1
    # % dtl: detecion limits 28 * 2(two batches)
    # % S_Rcon_log: impute missing data in resp_log_ad
    # % S_Rcon_rnm: impute missing data in resp_nrm_ad; so this will be different
    # %   for nomalise S_Rcon_log 
    cytolabel, stimlabel = load_labels(dataDir)
    Data = scipy.io.loadmat(dataDir+'/cytoData.mat')   # 
    
    number = Data['number'].flatten()
    number[number==22947] = 22497
    batch = Data['batch'].flatten()
    
    if batch_adjust is False:
        resp_raw = Data['resp_raw']
        resp_nrm = Data['resp_nrm']
        resp_log = Data['resp_log']
    else:
        resp_raw = Data['resp_raw_ad']
        resp_nrm = Data['resp_nrm_ad']
        resp_log = Data['resp_log_ad']
    
    

    
	        
    # raw data (only adjusted batch effect)
    df_raw = _mkDF(resp_raw,number,cytolabel,stimlabel)
    df_raw_nrm = _mkDF(resp_nrm,number,cytolabel,stimlabel[1:])
    df_raw_log = _mkDF(resp_log,number,cytolabel,stimlabel)
    
    
    
    S_Rcon_nrm = Data['S_Rcon_nrm']
    df_nrm = _mkDF(S_Rcon_nrm,number,cytolabel,stimlabel[1:])
    
    S_Rcon_log = Data['S_Rcon_log']
    df_log = _mkDF(S_Rcon_log,number,cytolabel,stimlabel)
    

    return df_raw, df_raw_nrm, df_raw_log, df_nrm, df_log








