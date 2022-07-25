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
import lifelines
from lifelines.calibration import survival_probability_calibration

from sklearn.linear_model import LinearRegression

def main():
    opt = parse_opts()
    losses_str = '_'.join(opt.losses)
    
    ValidDataInd = pickle.load(open('/data/pg-dl_radioth/scripts/MultilabelLearning_OPC_Radiomics/OPC-Radiomics/ValidDataInd_clinical.d', 'rb'))
    
    opt.HNSCC_ptnum = 606
    ptlist = range(opt.HNSCC_ptnum)
    train_val_list, test_list, all_list = Train_test_split(ptlist,ValidDataInd)
    
    #print (len(train_val_list),len(test_list))

    train_list = test_list[0:200]
    val_list  = test_list[200:]
    test_list = test_list[200:]
    '''
    train_list = train_val_list
    val_list  = test_list
    test_list = test_list
    '''
    opt.result_path_ct = opt.result_path_linux + '_test_good_normalization_model' + str(opt.model) + '_inputtype' + str(opt.input_type)+ '_inputmodality0'+'_fold' + str(opt.fold) \
                          + '_lr' + str(opt.learning_rate) + '_optim_' + str(opt.optimizer)+ '_bs' + str(opt.batch_size) \
                          + '_z_size' + str(opt.z_size) + '_md' + str(opt.model_depth) + '_sepconv_' + losses_str + '_' + opt.lr_scheduler+'_200train'
                          
    opcdata_ct = pd.read_csv(opt.result_path_ct+'/latent_feature.csv',header=0,index_col ='index')

    opcdata = pd.read_csv('/data/pg-dl_radioth/scripts/MultilabelLearning_OPC_Radiomics/opcradiomics_digits_202103_specific.csv',header=0,index_col ='Unnamed: 0')

    #-------------
    #   bootstrapping for feature selection
    
    # grouping different levels of the categorical variables
   

    clinical_para_strat = [ 'WHO_SCORE_codes_0VS123','AGE','GESLACHT_codes','Smoking_codes','TSTAD_DEF_codes_T123VS4', 'NSTAD_DEF_codes_N01VSN2VSN3', 'P16_codes']
    event_columns_code = ['OS_code','TumorSpecificSurvival_code','MET_code','LR_code','RR_code','LRR_code','DFS_code'] # DFS here is not correct
    survival_columns = ['TIME_OS','TIME_TumorSpecificSurvival','TIME_MET','TIME_LR','TIME_RR','TIME_LRR','TIME_DFS']
    real_event_names = ['Overall survival','Tumor-specific survival','Distant metastasis-free survival','Local control','Regional control','Locoregional control',
        'Disease-free survival','ULCER_code']
    
    event_columns_code_2year = ['OS_2year','TumorSpecificSurvival_2year','MET_code_2year','LR_code_2year','RR_code_2year','LRR_code_2year','DFS_code_2year']
    event_columns_code_2year_uncensoring = ['OS_2year_uncensoring','TumorSpecificSurvival_2year_uncensoring','MET_code_2year_uncensoring',
                                            'LR_code_2year_uncensoring','RR_code_2year_uncensoring','LRR_code_2year_uncensoring',
                                            'DFS_code_2year_uncensoring']
    event_2year = event_columns_code_2year[opt.outcome]
    event_2year_uncensoring = event_columns_code_2year_uncensoring[opt.outcome]
    

    opcdata.dropna()

    opcdata = opcdata.ix[all_list]

    ct_feature_list =  [str(i)+'_ct_gtv' for i in range(0,1024)]

    opcdata[ct_feature_list] = opcdata_ct[[str(i) for i in range(0,1024)]]
    
    #opcdata.to_csv(opt.result_path_ct+'/latent_feature_total.csv')
  
    latent_feature_list = ['AGE','GESLACHT_codes','Smoking_codes','Smoking_codes_noVSyes','MODALITY_codes','TSTAD_codes','TSTAD_codes_123VS4','TSTAD_codes_12VS34', 'NSTAD_codes','NSTAD_codes_N01VSN2VSN3','NSTAD_codes_012VS3', 
                     'P16_codes', 'P16_codes_combine', 'WHO_SCORE_codes','WHO_SCORE_codes_0VS123']
    latent_feature_list_opcradiomics  = ['WHO_SCORE_codes_0VS123','AGE','GESLACHT_codes','TSTAD_codes_123VS4', 'NSTAD_codes_N01VSN2VSN3', 'P16_codes_combine','Smoking_codes']
    traindata = opcdata.loc[train_list].reset_index(drop=True) 
    valdata = opcdata.loc[val_list].reset_index(drop=True)  
    testdata = opcdata.loc[test_list].reset_index(drop=True) 
    
    event_columns_code = event_columns_code
    survival_columns = survival_columns

    if opt.outcome ==0:
             para_tokeep = ['AGE',
'P16_codes_combine',
'WHO_SCORE_codes_0VS123',
'TSTAD_codes_123VS4',
'NSTAD_codes_N01VSN2VSN3',
]
    if opt.outcome ==1:
       
        para_tokeep = ['P16_codes_combine',
'NSTAD_codes_N01VSN2VSN3',
'AGE',
'TSTAD_codes_123VS4',]
          
    if opt.outcome ==2:
       para_tokeep =['NSTAD_codes_N01VSN2VSN3','WHO_SCORE_codes_0VS123',
]       
    if opt.outcome ==3:
       para_tokeep =['AGE',
'P16_codes_combine',
]
    if opt.outcome ==4:
       para_tokeep =[ 'NSTAD_codes_N01VSN2VSN3',
'P16_codes_combine',
]
    if opt.outcome ==5:
       para_tokeep =['P16_codes_combine',
'NSTAD_codes_N01VSN2VSN3',
]
    if opt.outcome ==6:
       para_tokeep =['P16_codes_combine',
'AGE',
'WHO_SCORE_codes_0VS123',
'TSTAD_codes_123VS4',]


    # clinical model
    from lifelines import CoxPHFitter
    trainD = traindata[para_tokeep+[event_columns_code[opt.outcome]]+[survival_columns[opt.outcome]]]
    cphAll = CoxPHFitter()   ## Instantiate the class to create a cph object
    cphAll.fit(trainD, survival_columns[opt.outcome], event_col=event_columns_code[opt.outcome],step_size=0.5)   ## Fit the data to train the model
    cphAll.print_summary()    ## HAve a look at the significance of the features
    cphAll.plot()
    
    #predicted_survival_time  = cphAll.predict_survival_function(trainD[para_tokeep])
    #print ('predicted_survival_time:', predicted_survival_time)
    

    print(cphAll.score(traindata[para_tokeep+[event_columns_code[opt.outcome]]+[survival_columns[opt.outcome]]],scoring_method='concordance_index'))
    print(cphAll.score(valdata[para_tokeep+[event_columns_code[opt.outcome]]+[survival_columns[opt.outcome]]],scoring_method='concordance_index'))
    #print(cphAll.score(testdata[para_tokeep+[event_columns_code[opt.outcome]]+[survival_columns[opt.outcome]]],scoring_method='concordance_index'))
    print (cphAll.log_likelihood_ratio_test())
    
    opcdata['clinical_model_risk_'+event_columns_code[opt.outcome]] = cphAll.predict_log_partial_hazard(opcdata[para_tokeep])
    
    
    # combine model
    traindata = opcdata.loc[train_list].reset_index(drop=True) 
    valdata = opcdata.loc[val_list].reset_index(drop=True)  
    testdata = opcdata.loc[test_list].reset_index(drop=True) 
    if opt.outcome == 0:        
       para_tokeep_combine =['clinical_model_risk_'+event_columns_code[opt.outcome],
'695_ct_gtv','517_ct_gtv','956_ct_gtv'
] 
    if opt.outcome == 1:        
       para_tokeep_combine =['clinical_model_risk_'+event_columns_code[opt.outcome],'956_ct_gtv'
                             
]
    if opt.outcome == 2:        
       para_tokeep_combine =['clinical_model_risk_'+event_columns_code[opt.outcome],
'911_ct_gtv','469_ct_gtv'
]
    if opt.outcome == 3:        
       para_tokeep_combine =['clinical_model_risk_'+event_columns_code[opt.outcome],

'956_ct_gtv','567_ct_gtv'

]
    if opt.outcome == 4:        
       para_tokeep_combine =['clinical_model_risk_'+event_columns_code[opt.outcome],
                             '614_ct_gtv','333_ct_gtv','160_ct_gtv']
    if opt.outcome == 5:        
        para_tokeep_combine =['clinical_model_risk_'+event_columns_code[opt.outcome],
                             '567_ct_gtv','444_ct_gtv'
] 
    if opt.outcome == 6:        
        para_tokeep_combine =['clinical_model_risk_'+event_columns_code[opt.outcome],
                              '220_ct_gtv',
'695_ct_gtv',

]
    trainD = traindata[para_tokeep_combine+[event_columns_code[opt.outcome]]+[survival_columns[opt.outcome]]]
    cphAll_combine = CoxPHFitter()   ## Instantiate the class to create a cph object
    cphAll_combine.fit(trainD, survival_columns[opt.outcome], event_col=event_columns_code[opt.outcome],step_size=0.5)   ## Fit the data to train the model
    cphAll_combine.print_summary()    ## HAve a look at the significance of the features
    #cphAll_combine.plot()
    #cphAll_combine.predict_survival_function(trainD[para_tokeep]).plot()

    print(cphAll_combine.score(traindata[para_tokeep_combine+[event_columns_code[opt.outcome]]+[survival_columns[opt.outcome]]],scoring_method='concordance_index'))
    print(cphAll_combine.score(valdata[para_tokeep_combine+[event_columns_code[opt.outcome]]+[survival_columns[opt.outcome]]],scoring_method='concordance_index'))
    
    
    envents_num = traindata[[event_columns_code[opt.outcome]]].sum()    
    median_followup = traindata[[survival_columns[opt.outcome]]].median()
    events_percent = envents_num/ traindata[[event_columns_code[opt.outcome]]].count()
    print ('Training set median follow-up, events_num, events percentage:',median_followup, envents_num, events_percent )
    envents_num = valdata[[event_columns_code[opt.outcome]]].sum()    
    median_followup = valdata[[survival_columns[opt.outcome]]].median()
    events_percent = envents_num/ valdata[[event_columns_code[opt.outcome]]].count()
    print ('Validation set median follow-up, events_num, events percentage:',median_followup, envents_num, events_percent )
    
    # high and los risk groups, KM-curves
    
    print (' risk of training set:',cphAll_combine.predict_log_partial_hazard(trainD[para_tokeep_combine]))
    risk_scores_testing = cphAll_combine.predict_log_partial_hazard(valdata[para_tokeep_combine])
    print (' risk of testing set:',risk_scores_testing)
    median_risk = np.median(cphAll_combine.predict_log_partial_hazard(trainD[para_tokeep_combine]))
    print ('Median risk of training set:', median_risk) 
    
    highrisk_group_list = risk_scores_testing[np.where(risk_scores_testing>median_risk)[0]].index
    lowrisk_group_list = risk_scores_testing[np.where(risk_scores_testing<median_risk)[0]].index
    print ('high_risk_group:', highrisk_group_list , len(highrisk_group_list))
    print ('low_risk_group:', lowrisk_group_list , len(lowrisk_group_list))
    
    event = event_columns_code[opt.outcome]
    time_e = survival_columns[opt.outcome]
    event_name  = ['OS','TSS','DMFS','LC','RC','LRC','DFS'][opt.outcome]

        # log rank test
    from lifelines.statistics import logrank_test
    results = logrank_test(valdata.loc[highrisk_group_list][time_e], valdata.loc[lowrisk_group_list][time_e]
                           , event_observed_A=valdata.loc[highrisk_group_list][event], event_observed_B=valdata.loc[lowrisk_group_list][event])

    results.print_summary()
    print('p-value:', results.p_value)        # 0.7676
    print(results.test_statistic) # 0.0872
    

    # new calibration curves for all patients
    plt.figure()
    ax = plt.subplot(111)
    kmf_0 = KaplanMeierFitter()
    
    # just draw calibratin cureves before 60 months
    valdata.loc[valdata[time_e] > 60 ,event] = 0
    valdata.loc[valdata[time_e] > 60 ,time_e] = 60
    
    ax = kmf_0.fit(valdata[time_e], valdata[event],label='Kaplan-Meier').plot_survival_function(ax=ax,show_censors=True,ci_show=True,color='blue')
    plt.tight_layout()
    plt.ylim([0.0,1.0])
   
    ax.set_ylabel(event_name + ' rate', fontsize = 16, fontweight = 'bold')
    ax.set_xlabel('Time (months)', fontsize = 16, fontweight = 'bold')   
    ax.tick_params(axis='both',labelsize = 14)
    plt.title("Independent internal test" , fontsize = 16, fontweight = 'bold')
    predicted_survival_function = cphAll_combine.predict_survival_function(valdata[para_tokeep_combine],times= np.asarray(sorted(list(set(valdata[time_e])))))
    #print ('predicted_survival_function',predicted_survival_function)
    ax = predicted_survival_function.mean(1).plot(color='red' , label = 'Predicted')
    # draw ci of predicted
    ci = 1.96 * np.std(np.array(predicted_survival_function), axis=1)/np.sqrt(np.array(predicted_survival_function).shape[1])
    x = np.array(predicted_survival_function.mean(1).index)
    y = np.array(predicted_survival_function.mean(1))
    ax.fill_between(x, (y -ci), (y + ci), color='red', alpha=.3)
    
    add_at_risk_counts(kmf_0, ax=ax, fontsize = 14)
    ax.legend(loc='lower left' , prop={ 'size': 14})
    plt.savefig("/data/pg-dl_radioth/scripts/Autoencoder_opcradiomics/Calibration/" + "Calibration_"+ event + '_opcradiomics.svg', bbox_inches = 'tight')
    
    # draw calibration curve at 2-year
    valdata = valdata.loc[valdata[event_2year_uncensoring] == 1] # oncly select uncensored patients at 2-year
    valdata = valdata.reset_index(drop=True) 
    
    predicted_survival_function = np.asarray(cphAll_combine.predict_survival_function(valdata[para_tokeep_combine],times= [24.0]))[0]
    real_survival = -np.asarray(valdata[event_2year]) + 1 
    
    print ('predicted_survival_function', predicted_survival_function)
    print ('real_survival', real_survival)
    
    # HosmerLemeshow test
    out_hosmer, prob_true , prob_pred = HosmerLemeshow(np.asarray(predicted_survival_function),np.asarray(real_survival), 3, strategy = "uniform") # "quantile" or "uniform"
    print ('out_hosmer p-value:', out_hosmer["p - value"][0])
    
    reg = LinearRegression().fit(np.array(prob_pred)[:, np.newaxis], prob_true)
    slope, intercept = reg.coef_[0], reg.intercept_
    print ('slope, intercept:' ,slope, intercept)
    
    plt.figure()      
    plt.plot(prob_pred, prob_true,'o',markersize = 10, markerfacecolor = 'blue')
    plt.plot((0, 1), slope*np.array([0,1]) + intercept , linestyle = '-',color ='blue', label = 'Real calibration') # predict line
    
    #plt.plot(prob_pred, prob_true)
    plt.plot((0, 1), (0, 1), linestyle = '-',color ='red' , label = 'Ideal calibration') # ideal line
    plt.ylabel("Observed actual 2-year " + event_name + ' rate', fontsize = 16, fontweight = 'bold')
    plt.xlabel("Predicted 2-year " + event_name + ' rate', fontsize = 16, fontweight = 'bold')
    plt.title("Independent internal test", fontsize = 16, fontweight = 'bold')
    plt.legend(loc='upper left', prop={ 'size': 14})

    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    
    plt.text(0.55, 0.20, 'Slope = '+str('%.3f' % slope), fontsize=16 )
    plt.text(0.55, 0.12, 'Intercept = '+str('%.3f' % intercept), fontsize=16 )
    plt.text(0.55, 0.04, 'p = '+str('%.3f' % out_hosmer["p - value"][0]), fontsize=16 )
    
    figname = "/data/pg-dl_radioth/scripts/Autoencoder_opcradiomics/Calibration/" + "Calibration_2year_"+ event  + '_opcradiomics.svg'
    plt.savefig(figname, bbox_inches = 'tight')
    
   
    
    ### external test 
    ValidDataInd = pickle.load(open('/data/p303924/UMCGOPC/VolPatch_Clinical/ValidDataInd_clinical.d','rb'))
    
    opcdata_ct = pd.read_csv(opt.result_path_ct+'/latent_feature_external.csv',header=0,index_col ='index')
    #print ("external ct features:" , opcdata_ct)
    
    clic_data = pd.read_csv("/data/pg-dl_radioth/scripts/MultilabelLearning_umcgopc/OPCdigits_2018split_202101_specific.csv")	
    

    patinets_401 = pd.read_excel("/data/pg-dl_radioth/scripts/MultilabelLearning_umcgopc/struct_paths_+CT_PET_E_401.xlsx")    
    patinets_name = list(patinets_401['UMCG-ID'])    
    clic_data['UMCG'] = [str(int(i)).zfill(7) for i in clic_data['UMCG']]
    indexs = []
    #print (patinets_name, clic_data['UMCG'] )
    for patinet_name in patinets_name:

        index = np.where(clic_data['UMCG'] == str(patinet_name).zfill(7))[0]        
        indexs.append(index[0])       
    opcdata = clic_data.loc[np.asarray(indexs)].reset_index(drop = True)

       
    opcdata = opcdata.ix[list(set(range(401))-set([0,16,42,55,60 ]))]
    opcdata[ct_feature_list] = opcdata_ct[[str(i) for i in range(0,1024)]]
    
    valdata = opcdata[171:]
    # only select subgrop of external tesing sets (such as only positive patients)
    valdata = valdata.loc[valdata['P16_codes'] != 0] # exclude hpv unknown patients
    #valdata = opcdata
    
    envents_num = valdata[[event_columns_code[opt.outcome]]].sum()    
    median_followup = valdata[[survival_columns[opt.outcome]]].median()
    events_percent = envents_num/ valdata[[event_columns_code[opt.outcome]]].count()
    print ('External validation set median follow-up, events_num, events percentage:',median_followup, envents_num, events_percent )
    
    valdata = valdata.reset_index(drop=True) 
    
    valdata['clinical_model_risk_'+event_columns_code[opt.outcome]] = cphAll.predict_log_partial_hazard(valdata[para_tokeep])
    risk_scores_testing = cphAll_combine.predict_log_partial_hazard(valdata[para_tokeep_combine])
    print (' risk of testing set:',risk_scores_testing)
    
    highrisk_group_list = risk_scores_testing[np.where(risk_scores_testing>median_risk)[0]].index
    lowrisk_group_list = risk_scores_testing[np.where(risk_scores_testing<median_risk)[0]].index
    print ('high_risk_group:', highrisk_group_list , len(highrisk_group_list))
    print ('low_risk_group:', lowrisk_group_list , len(lowrisk_group_list))
    
    event = event_columns_code[opt.outcome]
    time_e = survival_columns[opt.outcome]
    

    # log rank test
    from lifelines.statistics import logrank_test
    results = logrank_test(valdata.loc[highrisk_group_list][time_e], valdata.loc[lowrisk_group_list][time_e]
                           , event_observed_A=valdata.loc[highrisk_group_list][event], event_observed_B=valdata.loc[lowrisk_group_list][event])

    results.print_summary()
    print('p-value:', results.p_value)        # 0.7676
    print(results.test_statistic) # 0.0872
    
 
    # new calibration curves for all patients
    
    # just draw calibratin cureves before 60 months
    valdata.loc[valdata[time_e] > 60 ,event] = 0
    valdata.loc[valdata[time_e] > 60 ,time_e] = 60
    
    plt.figure()
    ax = plt.subplot(111)
    kmf_0 = KaplanMeierFitter()
    ax = kmf_0.fit(valdata[time_e], valdata[event],label='Kaplan-Meier').plot_survival_function(ax=ax,show_censors=True,ci_show=True,color='blue')
    plt.tight_layout()
    plt.ylim([0.0,1.0])
    ax.set_ylabel(event_name + ' rate', fontsize = 16, fontweight = 'bold')
    ax.set_xlabel('Time (months)', fontsize = 16, fontweight = 'bold')   
    ax.tick_params(axis='both',labelsize = 14)
    plt.title("External test" , fontsize = 16, fontweight = 'bold')
    predicted_survival_function = cphAll_combine.predict_survival_function(valdata[para_tokeep_combine],times= np.asarray(sorted(list(set(valdata[time_e])))))
    print ('predicted_survival_function', predicted_survival_function, np.array(predicted_survival_function).shape)
    ax = predicted_survival_function.mean(1).plot(color='red' , label = 'Predicted')
    # draw ci of predicted
    ci = 1.96 * np.std(np.array(predicted_survival_function), axis=1)/np.sqrt(np.array(predicted_survival_function).shape[1])
    x = np.array(predicted_survival_function.mean(1).index)
    y = np.array(predicted_survival_function.mean(1))
    ax.fill_between(x, (y -ci), (y + ci), color='red', alpha=.3)
    
    print ('predicted_survival_function_mean', np.array(predicted_survival_function.mean(1).index), np.array(predicted_survival_function.mean(1)).shape)
    add_at_risk_counts(kmf_0, ax=ax, fontsize = 14)
    ax.legend(loc='lower left' , prop={ 'size': 14})
    plt.savefig("/data/pg-dl_radioth/scripts/Autoencoder_opcradiomics/Calibration/" + "Calibration_"+ event + '_umcgopc_(external).svg', bbox_inches = 'tight')
    
    # draw calibration curve at 2-year
    valdata = valdata.loc[valdata[event_2year_uncensoring] == 1] # oncly select uncensored patients at 2-year
    valdata = valdata.reset_index(drop=True) 
    predicted_survival_function = np.asarray(cphAll_combine.predict_survival_function(valdata[para_tokeep_combine],times= [24.0]))[0]
    real_survival = -np.asarray(valdata[event_2year]) + 1 
    #print ('predicted_survival_function', predicted_survival_function)
    #print ('real_survival', real_survival)
    # HosmerLemeshow test
    out_hosmer, prob_true , prob_pred = HosmerLemeshow(np.asarray(predicted_survival_function),np.asarray(real_survival), 5 , strategy = "uniform") # "quantile" or "uniform"
    print ('out_hosmer p-value:', out_hosmer["p - value"][0])
    
    reg = LinearRegression().fit(np.array(prob_pred)[:, np.newaxis], prob_true)
    slope, intercept = reg.coef_[0], reg.intercept_
    print ('slope, intercept:' ,slope, intercept)
   
    plt.figure()      
    plt.plot(prob_pred, prob_true,'o',markersize = 10, markerfacecolor = 'blue')
    
    plt.plot((0, 1), slope*np.array([0,1]) + intercept , linestyle = '-',color ='blue', label = 'Real calibration') # predict line
    
    #plt.plot(prob_pred, prob_true)
    plt.plot((0, 1), (0, 1), linestyle = '-',color ='red' , label = 'Ideal calibration') # ideal line
    plt.ylabel("Observed actual 2-year " + event_name + ' rate', fontsize = 16, fontweight = 'bold')
    plt.xlabel("Predicted 2-year " + event_name + ' rate', fontsize = 16, fontweight = 'bold')
    plt.title("External test", fontsize = 16, fontweight = 'bold')
    plt.legend(loc='upper left', prop={ 'size': 14})
    #new 
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.text(0.55, 0.20, 'Slope = '+str('%.3f' % slope), fontsize=16 )
    plt.text(0.55, 0.12, 'Intercept = '+str('%.3f' % intercept), fontsize=16 )
    plt.text(0.55, 0.04, 'p = '+str('%.3f' % out_hosmer["p - value"][0]), fontsize=16 )
    
    figname = "/data/pg-dl_radioth/scripts/Autoencoder_opcradiomics/Calibration/" + "Calibration_2year_"+ event  + '_external(umcgopc).svg'
    plt.savefig(figname, bbox_inches = 'tight')
    
  
def calibration_at_years(survial_curves, month  = 12):
    # survial_curves: the figure incluidng four survival curves- High risk real','Low risk real','High risk predicted','Low risk predicted      
    # evaluate calibration at month (12, 24, 60)
    ax= survial_curves
    month =month
    times_high_risk_real , probs_high_risk_real = ax.lines[0].get_data()
    times_low_risk_real , probs_low_risk_real   = ax.lines[1].get_data()
    times_high_risk_predict , probs_high_risk_predict = ax.lines[2].get_data()
    times_low_risk_predict , probs_low_risk_predict   = ax.lines[3].get_data()
    probs_high_risk_real = np.min(probs_high_risk_real[np.where(times_high_risk_real<=month)[0]])
    probs_high_risk_predict = np.min(probs_high_risk_predict[np.where(times_high_risk_predict<=month)[0]])
    probs_low_risk_real = np.min(probs_low_risk_real[np.where(times_low_risk_real<=month)[0]])
    probs_low_risk_predict = np.min(probs_low_risk_predict[np.where(times_low_risk_predict<=month)[0]])
    
    probs_high_risk_difference = probs_high_risk_real - probs_high_risk_predict
    probs_low_risk_difference = probs_low_risk_real - probs_low_risk_predict
  
    return probs_high_risk_real, probs_high_risk_predict, probs_high_risk_difference, probs_low_risk_real, probs_low_risk_predict, probs_low_risk_difference
 

def Train_test_split(pt_list,ValidDataInd):

	ValidDataInd = np.asarray(ValidDataInd).astype(int)
	all_list = np.asarray(pt_list)[np.where(ValidDataInd>0)[0]]
	trainval_list = np.asarray(pt_list)[list(set(np.where(ValidDataInd<200)[0])- set(np.where(ValidDataInd==0)[0])) ]
	test_list = np.asarray(pt_list)[np.where(ValidDataInd>199)[0]]

	return trainval_list, test_list, all_list    
    
from scipy.stats import chi2
def HosmerLemeshow(obseved ,expected,bins = 5, strategy = "quantile") :
    pihat=obseved
    Y = expected
    pihatcat=pd.cut(pihat, np.percentile(pihat,[0,20,40,60,80,100]),labels = False,include_lowest=True) #here we've chosen only 4 groups
    
    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, bins + 1)
        pihatcat = np.percentile(obseved, quantiles * 100)
    elif strategy == "uniform":
        pihatcat = np.linspace(0.0, 1.0, bins + 1)
    pihatcat = np.searchsorted(pihatcat[1:-1], obseved)

    meanprobs =[0]*bins 
    expevents =[0]*bins
    obsevents =[0]*bins 
    meanprobs2=[0]*bins 
    expevents2=[0]*bins
    obsevents2=[0]*bins 
    #points
    expprobs =[0]*bins
    obsprobs =[0]*bins 
    

    for i in range(bins):
       meanprobs[i]=np.mean(pihat[pihatcat==i])
       expevents[i]=np.sum(pihatcat==i)*np.array(meanprobs[i])
       obsevents[i]=np.sum(Y[pihatcat==i])
       meanprobs2[i]=np.mean(1-pihat[pihatcat==i])
       expevents2[i]=np.sum(pihatcat==i)*np.array(meanprobs2[i])
       obsevents2[i]=np.sum(1-Y[pihatcat==i]) 
       
       expprobs[i] = np.sum(Y[pihatcat==i]) / len(Y[pihatcat==i])
       obsprobs[i] = np.mean(pihat[pihatcat==i])

    data1={'meanprobs':meanprobs,'meanprobs2':meanprobs2}
    data2={'expevents':expevents,'expevents2':expevents2}
    data3={'obsevents':obsevents,'obsevents2':obsevents2}
    m=pd.DataFrame(data1)
    e=pd.DataFrame(data2)
    o=pd.DataFrame(data3)
    
    # The statistic for the test, which follows, under the null hypothesis,
    # The chi-squared distribution with degrees of freedom equal to amount of groups - 2. Thus 4 - 2 = 2
    tt=sum(sum((np.array(o)-np.array(e))**2/np.array(e))) 
    pvalue=1-chi2.cdf(tt,int(bins) - 2)

    return pd.DataFrame([[chi2.cdf(tt,2).round(2), pvalue.round(2)]],
                        columns = ["Chi2", "p - value"]), expprobs, obsprobs #expevents,  obsevents

if __name__ == '__main__':
	main()
