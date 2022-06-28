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


def load_meanFoldChange(dataDir):
    MFmat = scipy.io.loadmat(dataDir+'/meanFold.mat')   # H_sigPairs
    meanFold = MFmat['meanFold']
    cytolabel, stimlabel = load_labels(dataDir)
    meanFold = pd.DataFrame(meanFold,index = cytolabel,columns=stimlabel[1:])
    return meanFold
    
def load_cytokine(dataDir,batch_adjust=True):

    
    cytolabel, stimlabel = load_labels(dataDir)

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
    datapanel_raw = pd.Panel(resp_raw, items=cytolabel, major_axis=number, minor_axis=stimlabel[:])  # media not used since data normalised
    datapanel_raw.items.name = 'cytokine'
    datapanel_raw.major_axis.name = 'number'
    datapanel_raw.minor_axis.name = 'stimulus'
    
    datapanel_raw_nrm = pd.Panel(resp_nrm, items=cytolabel, major_axis=number, minor_axis=stimlabel[1:])  # media not used since data normalised
    datapanel_raw_nrm.items.name = 'cytokine'
    datapanel_raw_nrm.major_axis.name = 'number'
    datapanel_raw_nrm.minor_axis.name = 'stimulus'
    
    datapanel_raw_log = pd.Panel(resp_log, items=cytolabel, major_axis=number, minor_axis=stimlabel[:])  # media not used since data normalised
    datapanel_raw_log.items.name = 'cytokine'
    datapanel_raw_log.major_axis.name = 'number'
    datapanel_raw_log.minor_axis.name = 'stimulus'
    
    S_Rcon_nrm = Data['S_Rcon_nrm']
    datapanel_nrm = pd.Panel(S_Rcon_nrm, items=cytolabel, major_axis=number, minor_axis=stimlabel[1:])  # media not used since data normalised
    datapanel_nrm.items.name = 'cytokine'
    datapanel_nrm.major_axis.name = 'number'
    datapanel_nrm.minor_axis.name = 'stimulus'
    
    S_Rcon_log = Data['S_Rcon_log']
    datapanel_log = pd.Panel(S_Rcon_log, items=cytolabel, major_axis=number, minor_axis=stimlabel[:])  # media not used since data normalised
    datapanel_log.items.name = 'cytokine'
    datapanel_log.major_axis.name = 'number'
    datapanel_log.minor_axis.name = 'stimulus'

    return datapanel_raw, datapanel_raw_nrm, datapanel_raw_log, datapanel_nrm, datapanel_log








