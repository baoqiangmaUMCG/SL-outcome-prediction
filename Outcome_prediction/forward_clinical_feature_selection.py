import pandas as pd
import numpy as np
from sklearn import preprocessing
import sys
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from opts_cox  import parse_opts
#import seaborn as sb

from lifelines.plotting import plot_lifetimes      # Lifeline package for the Survival Analysis
from lifelines import KaplanMeierFitter

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.utils import resample
from lifelines.plotting import add_at_risk_counts
import pickle
import random

def main():
    opt = parse_opts()
    losses_str = '_'.join(opt.losses)
    
    ValidDataInd = pickle.load(open('/data/pg-dl_radioth/scripts/MultilabelLearning_OPC_Radiomics/OPC-Radiomics/ValidDataInd_clinical.d', 'rb'))
    
    opt.HNSCC_ptnum = 606
    ptlist = range(opt.HNSCC_ptnum)
    train_val_list, test_list, all_list = Train_test_split(ptlist,ValidDataInd)
    print (len(train_val_list),len(test_list))

    train_list = all_list[0:374]
    # the path saving the clinical features and extracted features by autoencoder
    opt.result_path_ct = opt.result_path_linux + '_test_good_normalization_model' + str(opt.model) + '_inputtype' + str(opt.input_type)+ '_inputmodality0'+'_fold' + str(opt.fold) \
                          + '_lr' + str(opt.learning_rate) + '_optim_' + str(opt.optimizer)+ '_bs' + str(opt.batch_size) \
                          + '_z_size' + str(opt.z_size) + '_md' + str(opt.model_depth) + '_sepconv_' + losses_str + '_' + opt.lr_scheduler
    
    ct_feature_list =  [str(i)+'_ct_gtv' for i in range(0,1024)]
    
    # potential clinical predictors
    latent_feature_list = ['AGE','GESLACHT_codes','Smoking_codes','MODALITY_codes','TSTAD_codes_123VS4','NSTAD_codes_N01VSN2VSN3','P16_codes','WHO_SCORE_codes_0VS123']
   
    event_columns_code = ['OS_code','TumorSpecificSurvival_code','MET_code','LR_code','RR_code','LRR_code','DFS_code'] # DFS here is not correct
    survival_columns = ['TIME_OS','TIME_TumorSpecificSurvival','TIME_MET','TIME_LR','TIME_RR','TIME_LRR','TIME_DFS']
    
    # potential image feature predictors and the output of clinical model (need to build clinical model first)
    #latent_feature_list = ['clinical_model_risk_'+event_columns_code[opt.outcome]] + ct_feature_list
    opcdata = pd.read_csv(opt.result_path_ct+'/latent_feature_total.csv')
    #opcdata = pd.read_csv(opt.result_path_ct+'/latent_feature_total'+event_columns_code[opt.outcome]+'.csv')
    print (opcdata)
    traindata = opcdata.loc[train_list].reset_index(drop=True) 

    # feature selection
    Feature_freq_sorted = {}
    for event_cols,survival_cols in zip([event_columns_code[opt.outcome]],[survival_columns[opt.outcome]]):
        
        savename = './forward_selection/'+str(opt.sub)+str(event_cols)+'_clinical.csv'
    
        ## Create dummy variables
        trainD = traindata[latent_feature_list+[event_cols]+[survival_cols]]
        X = np.asarray(traindata[latent_feature_list])
        traindata[event_cols+'_boolen']=traindata[event_cols].apply(lambda x: True if x == 1 else False )
        y = traindata[[event_cols+'_boolen']+[survival_cols]].apply(tuple,axis=1)
        y = np.asarray(y,dtype=[('event', bool), ('time', float)])
        # bootstrapping
        Feature_freq = {}.fromkeys(traindata[latent_feature_list].columns, 0) 
        N = len(X)

        for m in range(100):
            print ('bootstrapping epoch:',m)
            index = resample(np.array(range(N)))
            trainD_resample = trainD.loc[index]   
            #trainD_resample = trainD
            # Using Cox Proportional Hazards model
            # from lifelines import CoxPHFitter

            from sksurv.linear_model import CoxnetSurvivalAnalysis
            # from sksurv.linear_model import CoxPHSurvivalAnalysis
            # cpsa = CoxPHSurvivalAnalysis()
            # sfs = SequentialFeatureSelector(cpsa, n_features_to_select=4) 
            # sfs.fit(X[index],y[index])
            # sfs_col = X.columns[sfs.get_support()]

            cnsa = CoxnetSurvivalAnalysis()
            sfs = SequentialFeatureSelector(cnsa, n_features_to_select=5) # set the number of selected features
            sfs.fit(X[index],y[index])
            sfs_col = traindata[latent_feature_list].columns[sfs.get_support()]


            from lifelines import CoxPHFitter
            cph = CoxPHFitter()   ## Instantiate the class to create a cph object

            try:
                cph.fit(trainD_resample[list(sfs_col)+[event_cols]+[survival_cols]], survival_cols, event_col=event_cols)   ## Fit the data to train the model
                # cph.print_summary()    ## HAve a look at the significance of the features
            except:
                continue

            for f in sfs_col[cph._compute_p_values()<0.05]:
                Feature_freq[f]=Feature_freq[f]+1

        Feature_freq_sorted[event_cols] = Feature_freq


    Feature_freq_sorted = pd.DataFrame.from_dict(Feature_freq_sorted)       
    Feature_freq_sorted.to_csv(savename, index = True, header=True)
    '''
    # This is for combined multi feature selection results
    csv = pd.read_csv('./forward_selection/'+str(opt.sub)+
                      event_columns_code[opt.outcome]+'_clinical.csv',header=0)
    for sub in range(2,11): 
        savename = opt.result_path_ct+'/forward_selection/'+str(sub)+event_columns_code[opt.outcome]+'_clinical.csv'
        sub_csv = pd.read_csv(savename,header=0)
        csv[event_columns_code[opt.outcome]] = csv[event_columns_code[opt.outcome]] + sub_csv[event_columns_code[opt.outcome]]
    csv.to_csv('./forward_selection/total_'+event_columns_code[opt.outcome]+'_clinical.csv', index = True, header=True)
    '''
def Train_test_split(pt_list,ValidDataInd):

	ValidDataInd = np.asarray(ValidDataInd).astype(int)
	all_list = np.asarray(pt_list)[np.where(ValidDataInd>0)[0]]
	trainval_list = np.asarray(pt_list)[list(set(np.where(ValidDataInd<200)[0])- set(np.where(ValidDataInd==0)[0])) ]
	test_list = np.asarray(pt_list)[np.where(ValidDataInd>199)[0]]

	return trainval_list, test_list, all_list    
    

if __name__ == '__main__':
	main()
