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
from lifelines.utils import concordance_index

def main():
    opt = parse_opts()
    losses_str = '_'.join(opt.losses)
    ValidDataInd = pickle.load(open('../Data_preprocessing/VolPatch_clinical/ValidDataInd_clinical.d', 'rb'))
    opt.HNSCC_ptnum = 606
    ptlist = range(opt.HNSCC_ptnum)
    train_val_list, test_list, all_list = Train_test_split(ptlist,ValidDataInd)
    
    #print (len(train_val_list),len(test_list))

    train_list = test_list[0:200]
    val_list  = test_list[200:]
    test_list = test_list[200:]

    opt.result_path_ct = opt.result_path_linux + '_test_good_normalization_model' + str(opt.model) + '_inputtype' + str(opt.input_type)+ '_inputmodality0'+'_fold' + str(opt.fold) \
                          + '_lr' + str(opt.learning_rate) + '_optim_' + str(opt.optimizer)+ '_bs' + str(opt.batch_size) \
                          + '_z_size' + str(opt.z_size) + '_md' + str(opt.model_depth) + '_sepconv_' + losses_str + '_' + opt.lr_scheduler
                          
    opcdata_ct = pd.read_csv(opt.result_path_ct+'/latent_feature.csv',header=0,index_col ='index')

    opcdata = pd.read_csv('../Data_preprocessing/opcradiomics_digits_202103.csv',header=0,index_col ='Unnamed: 0')

   
    clinical_para_strat = [ 'WHO_SCORE_codes_0VS123','AGE','GESLACHT_codes','Smoking_codes','TSTAD_codes_T123VS4', 'NSTAD_codes_N01VSN2VSN3', 'P16_codes']
    event_columns_code = ['OS_code','TumorSpecificSurvival_code','MET_code','LR_code','RR_code','LRR_code','DFS_code'] # DFS here is not correct
    survival_columns = ['TIME_OS','TIME_TumorSpecificSurvival','TIME_MET','TIME_LR','TIME_RR','TIME_LRR','TIME_DFS']
    real_event_names = ['Overall survival','Tumor-specific survival','Distant metastasis-free survival','Local control','Regional control','Locoregional control',
        'Disease-free survival','ULCER_code']
    real_event_names_short = ['OS','TSS','DMFS','LC','RC','LRC', 'DFS','ULCER_code']

    opcdata.dropna()

    opcdata = opcdata.ix[all_list]
    
    # statistics of outcomes opcradiomics
    mean_follow_up = opcdata[survival_columns[opt.outcome]].mean()
    print ('mean_follow_up_opcradiomics:', mean_follow_up)
    events_num = opcdata[event_columns_code[opt.outcome]].sum()
    print ('events_num_opcradiomics:', events_num)
    print ('events_percent_opcradiomics:', events_num/ len(all_list))

    ct_feature_list =  [str(i)+'_ct_gtv' for i in range(0,1024)]

    opcdata[ct_feature_list] = opcdata_ct[[str(i) for i in range(0,1024)]]
    
    #opcdata.to_csv(opt.result_path_ct+'/latent_feature_total.csv')
  
    latent_feature_list = ['AGE','GESLACHT_codes','Smoking_codes','Smoking_codes_noVSyes','MODALITY_codes','TSTAD_codes','TSTAD_codes_123VS4','TSTAD_codes_12VS34', 'NSTAD_codes','NSTAD_codes_N01VSN2VSN3','NSTAD_codes_012VS3', 'P16_codes', 'P16_codes_combine', 'WHO_SCORE_codes','WHO_SCORE_codes_0VS123']
    latent_feature_list_opcradiomics  = ['WHO_SCORE_codes_0VS123','AGE','GESLACHT_codes','TSTAD_codes_123VS4', 'NSTAD_codes_N01VSN2VSN3', 'P16_codes_combine','Smoking_codes']
    traindata = opcdata.loc[train_list].reset_index(drop=True) 
    valdata = opcdata.loc[val_list].reset_index(drop=True)  
    testdata = opcdata.loc[test_list].reset_index(drop=True) 
    
    event_columns_code = event_columns_code
    survival_columns = survival_columns
    
    # input the selected clinical features in para_tokeep for buliding clinical model
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
    #para_tokeep = list(set(para_tokeep) - set(['P16_codes']))
    from lifelines import CoxPHFitter
    trainD = traindata[para_tokeep+[event_columns_code[opt.outcome]]+[survival_columns[opt.outcome]]]
    cphAll = CoxPHFitter()   ## Instantiate the class to create a cph object
    cphAll.fit(trainD, survival_columns[opt.outcome], event_col=event_columns_code[opt.outcome],step_size=0.5)   ## Fit the data to train the model
    cphAll.print_summary()    ## HAve a look at the significance of the features
    cphAll.plot()
    
    
    print(cphAll.score(traindata[para_tokeep+[event_columns_code[opt.outcome]]+[survival_columns[opt.outcome]]],scoring_method='concordance_index'))
    print(cphAll.score(valdata[para_tokeep+[event_columns_code[opt.outcome]]+[survival_columns[opt.outcome]]],scoring_method='concordance_index'))
    #print(cphAll.score(testdata[para_tokeep+[event_columns_code[opt.outcome]]+[survival_columns[opt.outcome]]],scoring_method='concordance_index'))
    print (cphAll.log_likelihood_ratio_test())
    
    opcdata['clinical_model_risk_'+event_columns_code[opt.outcome]] = cphAll.predict_log_partial_hazard(opcdata[para_tokeep]) # save clinical model output 
    opcdata.to_csv(opt.result_path_ct+'/latent_feature_total'+event_columns_code[opt.outcome]+'.csv')
    
    # high and low risk groups, KM-curves
    risk_scores_train = cphAll.predict_log_partial_hazard(traindata[para_tokeep])
    # calculate 95%Ci of C-index 
    ci_cindex,cindex_values = conf_cindex(-risk_scores_train , traindata[survival_columns[opt.outcome]], traindata[event_columns_code[opt.outcome]])
    print ("clinical model ci_cindex train:",ci_cindex)
    # save the 1000 cindex values
    cindex_values = pd.DataFrame(cindex_values)
    cindex_values.loc[0, '95%_CI'] = str(ci_cindex)
    cindex_values.to_csv("./results/" + "cindexs_"+ event_columns_code[opt.outcome]  + '_clinical_train.csv')
    
    
    risk_scores_testing = cphAll.predict_log_partial_hazard(valdata[para_tokeep])
    # calculate 95%Ci of C-index 
    ci_cindex,cindex_values = conf_cindex(-risk_scores_testing , valdata[survival_columns[opt.outcome]], valdata[event_columns_code[opt.outcome]])
    print ("clinical model ci_cindex internal:",ci_cindex)
    # save the 1000 cindex values
    cindex_values = pd.DataFrame(cindex_values)
    cindex_values.loc[0, '95%_CI'] = str(ci_cindex)
    cindex_values.to_csv("./results/" + "cindexs_"+ event_columns_code[opt.outcome]  + '_clinical_opcradiomics.csv')
    median_risk = np.median(cphAll.predict_log_partial_hazard(trainD[para_tokeep]))
    
    highrisk_group_list = risk_scores_testing[np.where(risk_scores_testing>median_risk)[0]].index
    lowrisk_group_list = risk_scores_testing[np.where(risk_scores_testing<=median_risk)[0]].index
    
    event = event_columns_code[opt.outcome]
    time_e = survival_columns[opt.outcome]
    
    #log rank test
    from lifelines.statistics import logrank_test
    results = logrank_test(valdata.loc[highrisk_group_list][time_e], valdata.loc[lowrisk_group_list][time_e]
                           , event_observed_A=valdata.loc[highrisk_group_list][event], event_observed_B=valdata.loc[lowrisk_group_list][event])
    
    results.print_summary()
    print('p-value:', results.p_value)        # 0.7676
    print(results.test_statistic) # 0.0872

       
    plt.figure()
    ax = plt.subplot(111)
    kmf_0 = KaplanMeierFitter()
    ax = kmf_0.fit(valdata.loc[highrisk_group_list][time_e], valdata.loc[highrisk_group_list][event],label='High risk').plot_survival_function(ax=ax,show_censors=True,ci_show=False)
    kmf_1 = KaplanMeierFitter()
    ax = kmf_1.fit(valdata.loc[lowrisk_group_list][time_e], valdata.loc[lowrisk_group_list][event],label='Low risk').plot_survival_function(ax=ax,show_censors=True,ci_show=False)
    add_at_risk_counts(kmf_0, kmf_1, ax=ax, fontsize = 14)
    plt.tight_layout()
    plt.ylim([0.0,1.0])
    ax.set_ylabel(str(real_event_names_short[opt.outcome]) + ' rate', fontsize = 16, fontweight = 'bold') # new
    ax.set_xlabel('Time (months)',fontsize = 16, fontweight = 'bold')
    ax.legend( loc='lower left', prop={ 'size': 14})
    
    #ax.legend(loc='lower left', prop={ 'size': 14})
    # new
    #ax.set_xticks(labelsize = 14)
    #ax.set_yticks(labelsize = 14)
    ax.tick_params(axis='both',labelsize = 14)
    plt.title("Independent internal test" , fontsize = 16, fontweight = 'bold') # new
    if results.p_value < 0.001:
       plt.text(105, 0.05, 'p = '+str(results.p_value)[:5] +str(results.p_value)[-4:], fontsize=14)
    else:
       plt.text(105, 0.05, 'p = '+str(round(results.p_value, 3)), fontsize=14 )

    figname = "./results/" + "KM5_clinical_"+ event + '_opcradiomics.svg'
    plt.savefig(figname, bbox_inches = 'tight')
    
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
#

    trainD = traindata[para_tokeep_combine+[event_columns_code[opt.outcome]]+[survival_columns[opt.outcome]]]
    cphAll_combine = CoxPHFitter()   ## Instantiate the class to create a cph object
    cphAll_combine.fit(trainD, survival_columns[opt.outcome], event_col=event_columns_code[opt.outcome],step_size=0.5)   ## Fit the data to train the model
    cphAll_combine.print_summary()    ## HAve a look at the significance of the features
    #cphAll_combine.plot()
    #cphAll_combine.predict_survival_function(trainD[para_tokeep]).plot()

    print(cphAll_combine.score(traindata[para_tokeep_combine+[event_columns_code[opt.outcome]]+[survival_columns[opt.outcome]]],scoring_method='concordance_index'))
    print(cphAll_combine.score(valdata[para_tokeep_combine+[event_columns_code[opt.outcome]]+[survival_columns[opt.outcome]]],scoring_method='concordance_index'))
    
    
    # high and los risk groups, KM-curves
    risk_scores_train = cphAll_combine.predict_log_partial_hazard(traindata[para_tokeep_combine])
    # calculate 95%Ci of C-index 
    ci_cindex,cindex_values = conf_cindex(-risk_scores_train , traindata[survival_columns[opt.outcome]], traindata[event_columns_code[opt.outcome]])
    print ("combine model ci_cindex train:",ci_cindex)
    # save the 1000 cindex values
    cindex_values = pd.DataFrame(cindex_values)
    cindex_values.loc[0, '95%_CI'] = str(ci_cindex)
    cindex_values.to_csv("./results/" + "cindexs_"+ event  + '_combine_train.csv')
    
   
    risk_scores_testing = cphAll_combine.predict_log_partial_hazard(valdata[para_tokeep_combine])
    # calculate 95%Ci of C-index 
    ci_cindex,cindex_values = conf_cindex(-risk_scores_testing , valdata[survival_columns[opt.outcome]], valdata[event_columns_code[opt.outcome]])
    print ("combine model ci_cindex internal:",ci_cindex)
    # save the 1000 cindex values
    cindex_values = pd.DataFrame(cindex_values)
    cindex_values.loc[0, '95%_CI'] = str(ci_cindex)
    cindex_values.to_csv("./results/" + "cindexs_"+ event  + '_combine_opcradiomics.csv')
    
    #print (' risk of testing set:',risk_scores_testing)
    median_risk = np.median(cphAll_combine.predict_log_partial_hazard(trainD[para_tokeep_combine]))
    #print ('Median risk of training set:', median_risk) 
    
    highrisk_group_list = risk_scores_testing[np.where(risk_scores_testing>median_risk)[0]].index
    lowrisk_group_list = risk_scores_testing[np.where(risk_scores_testing<=median_risk)[0]].index
    #print ('high_risk_group:', highrisk_group_list , len(highrisk_group_list))
    #print ('low_risk_group:', lowrisk_group_list , len(lowrisk_group_list))
    
    event = event_columns_code[opt.outcome]
    time_e = survival_columns[opt.outcome]

    # log rank test
    from lifelines.statistics import logrank_test
    results = logrank_test(valdata.loc[highrisk_group_list][time_e], valdata.loc[lowrisk_group_list][time_e]
                           , event_observed_A=valdata.loc[highrisk_group_list][event], event_observed_B=valdata.loc[lowrisk_group_list][event])

    results.print_summary()
    print('p-value:', results.p_value)        # 0.7676
    print(results.test_statistic) # 0.0872

    plt.figure()
    ax = plt.subplot(111)
    kmf_0 = KaplanMeierFitter()
    ax = kmf_0.fit(valdata.loc[highrisk_group_list][time_e], valdata.loc[highrisk_group_list][event],label='High risk').plot_survival_function(ax=ax,show_censors=True,ci_show=False)
    kmf_1 = KaplanMeierFitter()
    ax = kmf_1.fit(valdata.loc[lowrisk_group_list][time_e], valdata.loc[lowrisk_group_list][event],label='Low risk').plot_survival_function(ax=ax,show_censors=True,ci_show=False)
    add_at_risk_counts(kmf_0, kmf_1, ax=ax, fontsize = 14)
    plt.tight_layout()
    plt.ylim([0.0,1.0])
    ax.set_ylabel(str(real_event_names_short[opt.outcome]) + ' rate', fontsize = 16, fontweight = 'bold') # new
    ax.set_xlabel('Time (months)',fontsize = 16, fontweight = 'bold')
    ax.legend( loc='lower left', prop={ 'size': 14})
    # new
    #ax.set_xticks(labelsize = 14)
    #ax.set_yticks(labelsize = 14)
    ax.tick_params(axis='both',labelsize = 14)
    plt.title("Independent internal test" , fontsize = 16, fontweight = 'bold') # new
    if results.p_value < 0.001:
       plt.text(105, 0.05, 'p = '+str(results.p_value)[:5] +str(results.p_value)[-4:], fontsize=14)
    else:
       plt.text(105, 0.05, 'p = '+str(round(results.p_value, 3)), fontsize=14 )
    
    
    figname = "./results/" + "KM5_combine_"+ event + '_opcradiomics.svg'
    plt.savefig(figname, bbox_inches = 'tight')
    
    
    # new calibration curves for all patients
    event = event_columns_code[opt.outcome]
    time_e = survival_columns[opt.outcome]
    event_name  = ['OS','TSS','DMFS','LC','RC','LRC','DFS'][opt.outcome]
    
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
    plt.savefig("./results/" + "Calibration_"+ event + '_opcradiomics.svg', bbox_inches = 'tight')
    
    # draw calibration curve at 2-year
    event_columns_code_2year = ['OS_2year','TumorSpecificSurvival_2year','MET_code_2year','LR_code_2year','RR_code_2year','LRR_code_2year','DFS_code_2year']
    event_columns_code_2year_uncensoring = ['OS_2year_uncensoring','TumorSpecificSurvival_2year_uncensoring','MET_code_2year_uncensoring',
                                        'LR_code_2year_uncensoring','RR_code_2year_uncensoring','LRR_code_2year_uncensoring',
                                        'DFS_code_2year_uncensoring']
    event_2year = event_columns_code_2year[opt.outcome]
    event_2year_uncensoring = event_columns_code_2year_uncensoring[opt.outcome]
    
    valdata = valdata.loc[valdata[event_2year_uncensoring] == 1] # oncly select uncensored patients at 2-year
    valdata = valdata.reset_index(drop=True) 
    
    predicted_survival_function = np.asarray(cphAll_combine.predict_survival_function(valdata[para_tokeep_combine],times= [24.0]))[0]
    real_survival = -np.asarray(valdata[event_2year]) + 1 
    
    print ('predicted_survival_function', predicted_survival_function)
    print ('real_survival', real_survival)
    
    # HosmerLemeshow test
    out_hosmer, prob_true , prob_pred = HosmerLemeshow(np.asarray(predicted_survival_function),np.asarray(real_survival), 3, strategy = "uniform") # "quantile" or "uniform"
    print ('out_hosmer p-value:', out_hosmer["p - value"][0])
    from sklearn.linear_model import LinearRegression
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
    
    figname = "./results/" + "Calibration_2year_"+ event  + '_opcradiomics.svg'
    plt.savefig(figname, bbox_inches = 'tight')
    


def conf_cindex(test_predictions, ground_truth_y,ground_truth_e, bootstrap=1000, seed=None,  confint=0.95):
    """Takes as input test predictions, ground truth, number of bootstraps, seed, and confidence interval"""
    #inspired by https://stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals by ogrisel
    bootstrapped_scores = []
    rng = np.random.RandomState(seed)
    if confint>1:
        confint=confint/100
    for i in range(bootstrap):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(test_predictions) - 1, len(test_predictions))
        if len(np.unique(ground_truth_y[indices])) < 2:
            continue

        #score = metrics.roc_auc_score(ground_truth[indices], test_predictions[indices])
        try:
           # For RC, sometimes no event selected, so mistake happens
           score = concordance_index(ground_truth_y[indices], test_predictions[indices], ground_truth_e[indices])
           bootstrapped_scores.append(score)
        except:
           continue
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    lower_bound=(1-confint)/2
    upper_bound=1-lower_bound
    confidence_lower = sorted_scores[int(lower_bound * len(sorted_scores))]
    confidence_upper = sorted_scores[int(upper_bound * len(sorted_scores))]
    auc = concordance_index(ground_truth_y, test_predictions, ground_truth_e)
    print("{:0.0f}% confidence interval for the score: [{:0.3f} - {:0.3}] and your cindex is: {:0.3f}".format(confint*100, confidence_lower, confidence_upper, auc))
    confidence_interval = (confidence_lower, auc, confidence_upper)
    return confidence_interval, sorted_scores
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
