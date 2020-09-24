# -*- coding: utf-8 -*-
import pandas as pd
import os
import random
import pickle
import numpy as np
from joblib import dump, load
import warnings
from sklearn.metrics import roc_curve
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import shap
#%%

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.decomposition import PCA

from tqdm import tqdm

from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import (SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC,
                                    KMeansSMOTE)
from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier, EasyEnsembleClassifier
from imblearn.under_sampling import RandomUnderSampler

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, precision_score, confusion_matrix

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from pyod.models.ocsvm import OCSVM #One Class SVM
from pyod.models.lscp import LSCP
from pyod.models.lof import LOF
from pyod.models.xgbod import XGBOD
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from xgboost import XGBClassifier

#%%
localpath = ''
basepath = '../Processed Data'
savepath = ''

def load_data(basepath):
    data = pd.read_csv(os.path.join(basepath,'CURIAL_Processed_2.csv'))
    data = data[~data['Blood_Test HAEMOGLOBIN'].isnull()]
    data.reset_index(inplace = True)
    """ Convert NaNs to Negative Results in Virology Columns """
    data.loc[:,'Covid-19 Positive'].fillna(0,inplace=True)
    data.loc[:,'Flu Positive'].fillna(0,inplace=True)
    data.loc[:,'Other Positive'].fillna(0,inplace=True)
    data.loc[:,'Respiratory'].fillna(0,inplace=True)
    #data.drop(columns=['Delta_BASOPHILS'], inplace=True)
    return data

def load_case_data(data,scenario):
    """ Obtain Case Cohort """
    case_condition0 = data['Covid-19 Positive']==True
    if scenario in ['covid_or_flu_vs_all_non_flu']:
        case_condition1 = data['Flu Positive']==True
        combined_condition = case_condition0 | case_condition1
    else:
        combined_condition = case_condition0
    
    case_indices = data.index[combined_condition]
    case_data = data.iloc[case_indices,:]
    return case_data,case_indices

#%%
def load_control_data(data,scenario):
    """ Obtain Control Cohort """
    if scenario in ['covid_vs_all_non_flu','covid_or_flu_vs_all_non_flu']:
        control_condition0 = data['Covid-19 Positive']==False
        #control_condition1 = data['No Virology']==True
        """ Guarantees Control to Come From Pre-Covid Era """
        control_condition2 = data['ArrivalDateTime'] < '2019-12-01'
        """ Do Not Include Flu Positive People """
        control_condition3 = data['Flu Positive']==False 
        combined_condition = control_condition0 & control_condition2 & control_condition3
    elif scenario in ['covid_vs_all']:
        control_condition0 = data['Covid-19 Positive']==False
        control_condition1 = data['ArrivalDateTime'] < '2019-12-01'
        combined_condition = control_condition0 & control_condition1
    elif scenario in ['covid_vs_flu_only']:
        control_condition0 = data['Covid-19 Positive']==False
        control_condition1 = data['Flu Positive']==True
        control_condition2 = data['ArrivalDateTime'] < '2019-12-01'
        combined_condition = control_condition0 & control_condition1 & control_condition2
    
    control_indices = data.index[combined_condition] 
    control_data = data.iloc[control_indices,:]
  #  print(len(control_data))
    return control_data

# #%%
# """ Check if Patient Overlap Occurs Between 2 Cohorts - SHOULD NOT """
# cohort_overlap = case_data['ClusterID'].isin(control_data['ClusterID']).sum()
# assert cohort_overlap == 0

#%%
""" Obtain Matched Control Cohort """

def load_matched_control_cohort(scenario,match_cohort,match_number,save_control_indices,control_data,data,case_indices, resp = False, admission= False, bytimepn =False):
  #  print(match_number,scenario,match_cohort)
    if scenario in ['covid_vs_all_non_flu','covid_or_flu_vs_all_non_flu','covid_vs_all'] and match_cohort == True:
    #    print(match_number,scenario)
        try:
            """ Try Loading Already Saved Indices """
#            if resp:
#                 with open(os.path.join(localpath,'control_indices_%i_%s_resp.pkl' % (match_number,scenario)),'rb') as f:
 #                   matched_cohort_indices = pickle.load(f)
#            if admission:
#                 with open(os.path.join(localpath,'control_indices_%i_%s_adm.pkl' % (match_number,scenario)),'rb') as f:
#                    matched_cohort_indices = pickle.load(f)
#                    
#            else:
            
            with open(os.path.join(localpath,'control_indices_%i_%s.pkl' % (match_number,scenario)),'rb') as f:
                matched_cohort_indices = pickle.load(f)
            
        #    if bytimepn:
        #        ind2 = np.where((pd.DatetimeIndex(control_data['ArrivalDateTime']).month ==5)|
       #                           ((pd.DatetimeIndex(control_data['ArrivalDateTime']).month ==4) &(pd.DatetimeIndex(control_data['ArrivalDateTime']).day >=20)))[0]
  #              matched_cohort_indices = np.concatenate((matched_cohort_indices,ind2))
            control_cohort = control_data.loc[matched_cohort_indices,:]
        except:
            """ If Saved Indices Do Not Exist, Generate Them """
            matched_cohort_indices = []
            for index in case_indices:
            #    print(index)
                patient_data = data.iloc[index,:]
                
                patient_age = patient_data['Age']
                gender = patient_data['Gender']
                ethnicity = patient_data['Ethnicity']
                
                age_condition1 = control_data['Age'] < patient_age + 2
                age_condition2 = control_data['Age'] > patient_age - 2
                gender_condition = control_data['Gender'] == gender
                ethnicity_condition = control_data['Ethnicity'] == ethnicity
             #   print(np.where(case_data[['Covid-19 Positive']].isnull()))
                if resp & admission:
                    adm_condition = (~control_data['EpisodeID'].isnull() & ~control_data['Vital_Sign Temperature Tympanic'].isnull())
                    respirotory_condition = control_data['Respiratory'] == 1
                   # print(np.where(respirotory_condition==True))
                    matched_indices = control_data.index[age_condition1 & age_condition2 & gender_condition & ethnicity_condition & respirotory_condition & adm_condition]
                elif resp:
                    respirotory_condition = control_data['Respiratory'] == 1
                   # print(np.where(respirotory_condition==True))
                    matched_indices = control_data.index[age_condition1 & age_condition2 & gender_condition & ethnicity_condition & respirotory_condition]
                elif admission:
                    adm_condition = (~control_data['EpisodeID'].isnull() & ~control_data['Vital_Sign Temperature Tympanic'].isnull())
                    matched_indices = control_data.index[age_condition1 & age_condition2 & gender_condition & ethnicity_condition & adm_condition]
                else:
                    matched_indices = control_data.index[age_condition1 & age_condition2 & gender_condition & ethnicity_condition]
                
                matched_indices = matched_indices.tolist()
                random.seed(0)
                matched_indices = random.sample(matched_indices,len(matched_indices))        
                
                valid_indices = [index for index in matched_indices if index not in matched_cohort_indices][:match_number]
                matched_cohort_indices.extend(valid_indices)
            
            if save_control_indices == True:
  #              if resp:
 #                   with open(os.path.join(localpath,'control_indices_%i_%s_resp.pkl' % (match_number,scenario)),'wb') as f:
  #                      pickle.dump(matched_cohort_indices,f)
 #               elif admission:
 #                   with open(os.path.join(localpath,'control_indices_%i_%s_adm.pkl' % (match_number,scenario)),'wb') as f:
#                        pickle.dump(matched_cohort_indices,f)
#                else:
                with open(os.path.join(localpath,'control_indices_%i_%s.pkl' % (match_number,scenario)),'wb') as f:
                    pickle.dump(matched_cohort_indices,f)
            
        #    if bytimepn:
             #   ind2 = np.where((pd.DatetimeIndex(control_data['ArrivalDateTime']).month ==5)|
              #                    ((pd.DatetimeIndex(control_data['ArrivalDateTime']).month ==4) &(pd.DatetimeIndex(control_data['ArrivalDateTime']).day >=20)))[0]
         #       matched_cohort_indices = np.concatenate((matched_cohort_indices,ind2))
          #  print(len(matched_cohort_indices))
            control_cohort = control_data.loc[matched_cohort_indices,:] #index name not location
    else:
        control_cohort = control_data
    
    return control_cohort

#%%
""" Load Featres and Labels """

def load_combined_cohort(case_data,control_cohort):
    """ Combined Cohorts """
    case_cohort = case_data
    df = pd.concat((case_cohort,control_cohort),0)
    return df

def load_features_and_labels(df, include_dummy_variable=False):
    """ Input Features """
    X = df.loc[:,'Blood_Test ALBUMIN':]
    X['CCI']=df.loc[:,'CCI']
    X.drop(['Vital_Sign AVPU Scale',
       'Vital_Sign Best Verbal Response', 'Vital_Sign Delivery device used',
       'Vital_Sign Eye Opening Response', 'Vital_Sign GCS Assessed',
       'Vital_Sign Glasgow Coma Score', 'Vital_Sign Humidified monitoring',
        'Vital_Sign Motor Response','Vital_Sign T&T Total Score', 'Vital_Sign TAT AVPU Scale',
       'Vital_Sign TAT Oxygen Saturation', 'Vital_Sign TAT Pulse Rate',
       'Vital_Sign TAT Respiratory Rate','Vital_Sign TAT Systolic Blood Pressure', 
        'Vital_Sign TAT Temperature','Vital_Sign Tracheostomy mask monitoring'],axis=1,inplace=True)
    
#four level of oxygen support
               
    Y = df['Covid-19 Positive']
    Z = df.iloc[:,:5] #demographic inform

    """ Dummy Variable Added to Features to Help Determine Feature Importance """
    if include_dummy_variable == True:
        np.random.seed(0)
        dummy_variable = pd.DataFrame(np.random.uniform(0,1,size=X.shape[0]),index=X.index,columns=['Dummy'])
        X = pd.concat((X,dummy_variable),1)
    
    return X, Y, Z

#%%
""" Load Phase Ids """
def load_phase_ids(df, match_number, bytime = False, bytimepn=False,flag='no'):
    """ Train/Val/Test Patient IDs """
    if bytime:
        df_pos = df[df['Covid-19 Positive']==1]
        df_pos = df_pos.drop_duplicates('ClusterID')
        pos_ids = df_pos[['ClusterID']].values
        train_ids_p = pos_ids[np.where((pd.DatetimeIndex(df_pos['ArrivalDateTime']).month ==3)|
                              ((pd.DatetimeIndex(df_pos['ArrivalDateTime']).month ==4) &(pd.DatetimeIndex(df_pos['ArrivalDateTime']).day <10)))[0]]
       # print(len(train_ids_p))
        val_ids_p = pos_ids[np.where((pd.DatetimeIndex(df_pos['ArrivalDateTime']).month==4) &(pd.DatetimeIndex(df_pos['ArrivalDateTime']).day >=10) &(pd.DatetimeIndex(df_pos['ArrivalDateTime']).day <20))[0]]
        test_ids_p =  pos_ids[np.where((pd.DatetimeIndex(df_pos['ArrivalDateTime']).month ==5)|
                              ((pd.DatetimeIndex(df_pos['ArrivalDateTime']).month ==4) &(pd.DatetimeIndex(df_pos['ArrivalDateTime']).day >=20)))[0]]
        df_neg = df[df['Covid-19 Positive']!=1]
        unique_ids = df_neg['ClusterID'].unique().tolist()
        random.seed(0)
        shuffled_ids = random.sample(unique_ids,len(unique_ids))
        ntotal_ids = len(shuffled_ids)
        train_ratio,val_ratio,test_ratio = 0.6, 0.2, 0.2
        train_amount,val_amount,test_amount = int(train_ratio*ntotal_ids), int(val_ratio*ntotal_ids), int(test_ratio*ntotal_ids)
        train_ids,val_ids,test_ids = shuffled_ids[:train_amount], shuffled_ids[train_amount:train_amount+val_amount], shuffled_ids[-test_amount:]
        
        train_ids = np.concatenate((train_ids,train_ids_p[:,0]))
        test_ids =  np.concatenate((test_ids,test_ids_p[:,0]))
        val_ids =  np.concatenate((val_ids, val_ids_p[:,0]))
        
        
    #    print(len(train_ids),len(val_ids),len(test_ids),len(train_ids_p),len(val_ids_p),len(test_ids_p) )
    
    if bytimepn:
        df_pos = df.drop_duplicates('ClusterID')
        ids = df_pos[['ClusterID']].values
        train_ids = ids[np.where((pd.DatetimeIndex(df_pos['ArrivalDateTime']).month ==3)|
                              ((pd.DatetimeIndex(df_pos['ArrivalDateTime']).month ==4) &(pd.DatetimeIndex(df_pos['ArrivalDateTime']).day <10)))[0]]
       # print(len(train_ids_p))
        val_ids = ids[np.where((pd.DatetimeIndex(df_pos['ArrivalDateTime']).month==4) &(pd.DatetimeIndex(df_pos['ArrivalDateTime']).day >=10) &(pd.DatetimeIndex(df_pos['ArrivalDateTime']).day <20))[0]]
        
        if flag == 'ICU':
            test_ids =  ids[np.where((pd.DatetimeIndex(df_pos['ArrivalDateTime']).month ==5)|
                              ((pd.DatetimeIndex(df_pos['ArrivalDateTime']).month ==4) &(pd.DatetimeIndex(df_pos['ArrivalDateTime']).day >=20))& (df_pos['ICU'] == 1))[0]]
            
        elif flag == 'Discharged':
            test_ids =  ids[np.where((pd.DatetimeIndex(df_pos['ArrivalDateTime']).month ==5)|
                              ((pd.DatetimeIndex(df_pos['ArrivalDateTime']).month ==4) &(pd.DatetimeIndex(df_pos['ArrivalDateTime']).day >=20))& (df_pos['Discharged'] == 1))[0]]
        elif flag == 'Death':
            test_ids =  ids[np.where((pd.DatetimeIndex(df_pos['ArrivalDateTime']).month ==5)|
                              ((pd.DatetimeIndex(df_pos['ArrivalDateTime']).month ==4) &(pd.DatetimeIndex(df_pos['ArrivalDateTime']).day >=20))& (df_pos['death'] == 1))[0]]
        elif flag == 'resp':
            test_ids =  ids[np.where((pd.DatetimeIndex(df_pos['ArrivalDateTime']).month ==5)|
                              ((pd.DatetimeIndex(df_pos['ArrivalDateTime']).month ==4) &(pd.DatetimeIndex(df_pos['ArrivalDateTime']).day >=20))& (df_pos['Respiratory'] == 1))[0]]
        else:
            test_ids =  ids[np.where((pd.DatetimeIndex(df_pos['ArrivalDateTime']).month ==5)|
                              ((pd.DatetimeIndex(df_pos['ArrivalDateTime']).month ==4) &(pd.DatetimeIndex(df_pos['ArrivalDateTime']).day >=20)))[0]]
        
        train_ids = np.reshape(train_ids,(len(train_ids[:,0],)))
        test_ids = np.reshape(test_ids,(len(test_ids[:,0],)))
        val_ids = np.reshape(val_ids,(len(val_ids[:,0],)))
        print(len(test_ids))
        #
    else:

        unique_ids = df['ClusterID'].unique().tolist()
        random.seed(0)
        shuffled_ids = random.sample(unique_ids,len(unique_ids))
        ntotal_ids = len(shuffled_ids)
        train_ratio,val_ratio,test_ratio = 0.6, 0.2, 0.2
        train_amount,val_amount,test_amount = int(train_ratio*ntotal_ids), int(val_ratio*ntotal_ids), int(test_ratio*ntotal_ids)
        train_ids,val_ids,test_ids = shuffled_ids[:train_amount], shuffled_ids[train_amount:train_amount+val_amount], shuffled_ids[-test_amount:]
        
        ########
   #     dff =  df[df['ClusterID'].isin(test_ids)]
   #     df_pos = dff[dff['Covid-19 Positive']==1]
   #     df_pos = df_pos['ClusterID'].unique().tolist()
   #     ll = len(df_pos)*match_number
   #     df_neg = dff[dff['Covid-19 Positive']!=1]
   #     df_neg = df_neg['ClusterID'].unique().tolist()[:ll]
   #     test_ids = df_neg +  df_pos
        
   #     dff =  df[df['ClusterID'].isin(train_ids)]
   #     df_pos = dff[dff['Covid-19 Positive']==1]
   #     df_pos = df_pos['ClusterID'].unique().tolist()
   #     ll = len(df_pos)*match_number
   #     df_neg = dff[dff['Covid-19 Positive']!=1]
   #     df_neg = df_neg['ClusterID'].unique().tolist()[:ll]
   #     train_ids = df_neg +  df_pos
        
   #     dff =  df[df['ClusterID'].isin(val_ids)]
   #     df_pos = dff[dff['Covid-19 Positive']==1]
   #     df_pos = df_pos['ClusterID'].unique().tolist()
   #     ll = len(df_pos)*match_number
   #     df_neg = dff[dff['Covid-19 Positive']!=1]
   #     df_neg = df_neg['ClusterID'].unique().tolist()[:ll]
   #     val_ids = df_neg +  df_pos
      #  print(match_number,ll,len(test_ids))
        #######
    return train_ids,val_ids,test_ids


#%%
def load_data_splits(df,X,Y,Z,train_ids,val_ids,test_ids):
    """ Train/Val/Test Features and Labels """
   # print(df.columns)
    train_rows_bool,val_rows_bool,test_rows_bool = df['ClusterID'].isin(train_ids),  df['ClusterID'].isin(val_ids),  df['ClusterID'].isin(test_ids)
    
    X_train, X_val, X_test = X.loc[train_rows_bool,:], X.loc[val_rows_bool,:], X.loc[test_rows_bool,:]
    
    Y_train, Y_val, Y_test = Y.loc[train_rows_bool], Y.loc[val_rows_bool], Y.loc[test_rows_bool]
    Z_train, Z_val, Z_test = Z.loc[train_rows_bool], Z.loc[val_rows_bool], Z.loc[test_rows_bool]
    return X_train,Y_train,Z_train, X_val,Y_val,Z_val, X_test,Y_test,Z_test
    
#%%
""" Imputation (Before Relative Value Calculations) """
def impute_missing_features(imputation_method,X_train,X_val,X_test,Z_train,Z_val,Z_test):#,columns_to_keep):
    if imputation_method in ['mean','median','MICE']:
        if imputation_method == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif imputation_method == 'median':
            imputer = SimpleImputer(strategy='median')
        elif imputation_method == 'MICE':
            imputer = IterativeImputer(max_iter=50)
        
        columns_to_keep = X_train.columns
    
        imputer.fit(X_train)
        X_train = pd.DataFrame(imputer.transform(X_train))
        X_val = pd.DataFrame(imputer.transform(X_val))
        X_test = pd.DataFrame(imputer.transform(X_test))
            
        """ Imputation Will Remove Columns Where Statistic is np.nan, Therefore Keep Track of Remaining Columns """
        if isinstance(imputer,SimpleImputer):
            columns_to_keep = columns_to_keep[np.where(~np.isnan(imputer.statistics_))[0]]
    
        X_train.columns = columns_to_keep
        X_val.columns = columns_to_keep
        X_test.columns = columns_to_keep     
   
    elif imputation_method == 'age-based mean':
        feature_columns = X_train.columns
    #    print(X_train.columns)
        """ Create Age-Band Column for Training Set """
        df_train = pd.concat((X_train,Z_train),1)
        age_bands = [0,45,69,120] #ensures everyone is placed into bucket 
        #df_train['AgeBin'] = pd.cut(df_train['Age'], [int(df_train['Age'].min()),45,69,int(df_train['Age'].max())], include_lowest=True)
        df_train['AgeBin'] = pd.cut(df_train['Age'], age_bands, include_lowest=True)        
        """ Create Age-Band Column for Validation Set """
        df_val = pd.concat((X_val,Z_val),1)
        #df_val['AgeBin'] = pd.cut(df_val['Age'], [int(df_train['Age'].min()),45,69,int(df_train['Age'].max())], include_lowest=True)
        df_val['AgeBin'] = pd.cut(df_val['Age'], age_bands, include_lowest=True)
        """ Create Age-Band Column for Test Set """
        df_test = pd.concat((X_test,Z_test),1)
        #df_test['AgeBin'] = pd.cut(df_test['Age'], [int(df_train['Age'].min()),45,69,int(df_train['Age'].max())], include_lowest=True)
        df_test['AgeBin'] = pd.cut(df_test['Age'], age_bands, include_lowest=True)
        """ Impute Values in Training Set """
        X_train = df_train.groupby('AgeBin')[feature_columns].transform(lambda grp: grp.fillna(np.mean(grp)))
        """ Obtain Mean Values from Training Set """
        grp_means = df_train.groupby('AgeBin')[feature_columns].mean()
        """ Impute Values in Validation Set Using Training Set Means """
        val_mean_matrix = grp_means.loc[df_val['AgeBin']]
        val_mean_matrix.index = X_val.index
        X_val = X_val.fillna(val_mean_matrix)
        """ Impute Values in Test Set Using Training Set Means """
        test_mean_matrix = grp_means.loc[df_test['AgeBin']]
        test_mean_matrix.index = X_test.index
        X_test = X_test.fillna(test_mean_matrix)
        """ Drop Column with ALL NaNs """
        #cols_to_drop = np.unique(np.where(X_train.isna())[1])
        X_train = X_train.dropna(axis=1,how='all')
        X_val = X_val.dropna(axis=1,how='all')
        X_test = X_test.dropna(axis=1,how='all')
    
    features = X_train.columns.tolist()
    return X_train,X_val,X_test,features



def filter_features(df,feature_type,add_demographics): #apply this to each phase input set 
    #""" Used for Old Curial Dataset """
    #absolute_features = df.loc[:,'25-OH VITAMIN D':'eGFR']
    #baseline_features = df.loc[:,'Baseline 25-OH VITAMIN D':'Baseline eGFR'].rename(lambda x:x[9:],axis=1)
    #relative_features = absolute_features - baseline_features
    #relative_features = relative_features.add_prefix('delta ')

    #""" Remove NaN Column Due to Absent Column in Absolute or Baseline Features """
    #relative_features = relative_features.dropna(axis=1,how='all')
    
    """ Use This For CURIAL_Processed.csv """
    BloodTest_cols_bool = df.columns.str.contains('Blood_Test ')
    BloodTestDelta_cols_bool = df.columns.str.contains('Blood_Test_Delta')
    BloodGas_cols_bool = df.columns.str.contains('Blood_Gas')
    VitalSign_cols_bool = df.columns.str.contains('Vital_Sign')
    CCI_cols_bool = df.columns.str.contains('CCI')
    
    BloodTest_features = df.loc[:,BloodTest_cols_bool]
    BloodTestDelta_features = df.loc[:,BloodTestDelta_cols_bool]
    BloodGas_features = df.loc[:,BloodGas_cols_bool]
    VitalSign_features = df.loc[:,VitalSign_cols_bool]
    CCI_features = df.loc[:,CCI_cols_bool]
    if feature_type == 'Blood':
        X =  BloodTest_features
    elif feature_type == 'Blood_Delta':
        X =  BloodTestDelta_features
    elif feature_type == 'Blood & Blood_Delta':
        X = pd.concat(( BloodTest_features, BloodTestDelta_features),1)
    elif feature_type == 'Blood & Blood_Gas':
        X = pd.concat(( BloodTest_features, BloodGas_features),1)
    elif feature_type == 'Blood & Blood_Delta & Blood_Gas':
        X = pd.concat(( BloodTest_features, BloodTestDelta_features, BloodGas_features),1)
    elif feature_type == 'All except CCI':
        X = pd.concat(( BloodTest_features, BloodTestDelta_features, BloodGas_features, VitalSign_features),1)
    elif feature_type == 'Blood & Blood_Gas & Vitals':
        X = pd.concat(( BloodTest_features, BloodGas_features, VitalSign_features),1) 
    elif feature_type == 'All Features':
        X = pd.concat(( BloodTest_features, BloodTestDelta_features, BloodGas_features, VitalSign_features,CCI_features),1)
    elif feature_type == 'Vital_Signs':
        X =  VitalSign_features
    elif feature_type == 'Blood_Gas':
        X =  BloodGas_features
    elif feature_type == 'Blood & Vitals':
        X = pd.concat(( BloodTest_features, VitalSign_features),1)
     
    if add_demographics == True:
        demographic_data = df.loc[:,['Age','Gender','Ethnicity']]
        X = pd.concat((X,demographic_data),1)
    
    return X

#%%
""" Feature Scaling/Standardization """
def standardize_features(X_train,X_val,X_test,feature_type):
    scaler = StandardScaler()
    scaler.fit(X_train)
    name = 'norm_' + feature_type + '.pkl' 
    with open(os.path.join(localpath,name),'wb') as f:
        pickle.dump(scaler,f)
    X_train = pd.DataFrame(scaler.transform(X_train))
    X_val = pd.DataFrame(scaler.transform(X_val))
    X_test = pd.DataFrame(scaler.transform(X_test))
    return X_train,X_val,X_test

def obtain_sampler(sampler_type):
    if sampler_type == 'SMOTE':
        sampler = SMOTE(random_state=0)
    elif sampler_type == 'SVMSMOTE':
        sampler = SVMSMOTE(random_state=0)
    elif sampler_type == 'SMOTEENN':
        sampler = SMOTEENN(random_state=0)
    else:
        sampler = None
    return sampler

def obtain_model(model_type,sampler,contamination,use_sampler=False):
    if model_type == 'LR':
        model = LogisticRegression(max_iter=500)            
    elif model_type == 'SVM':
        model = SVC(kernel='rbf',max_iter=500)
    elif model_type == 'RF':
        model = RandomForestClassifier(70)
    elif model_type == 'XGB':
        model = GradientBoostingClassifier(n_estimators = 70)
    elif model_type == 'DT':
        model = DecisionTreeClassifier()
    elif model_type == 'BalancedRF':
        model = BalancedRandomForestClassifier(random_state=0)
    elif model_type == 'BalancedBag':
        model = BalancedBaggingClassifier()
    elif model_type == 'EasyEnsemble':
        model = EasyEnsembleClassifier()
    elif model_type == 'LOF':
        model = LocalOutlierFactor(novelty=True)
    elif model_type == 'IF':
        model = IsolationForest()
    elif model_type == 'OCSVM':
        model = OCSVM()
    elif model_type == 'LSCP':
        detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10),
                     LOF(n_neighbors=20), LOF(n_neighbors=40)]
        model = LSCP(detector_list, random_state=0)
    elif model_type == 'XGBOD':
        model = XGBOD()
    elif model_type == 'HBOS':
        model = HBOS(contamination=contamination)
    elif model_type == 'CBLOF':
        model = CBLOF(contamination=contamination,random_state=0)
        
    if use_sampler == True:
        model = make_pipeline(sampler,model)
    
    return model

def find_threshold_at_metric(model,inputs,outputs,best_threshold,metric_of_interest,value_of_interest,results_df,fold_number,error,match_number):
    ground_truth = outputs['eval']
    """ Probability Values for Predictions """
    probs = model.predict_proba(inputs['eval'])[:,1]
    threshold_metrics = pd.DataFrame(np.zeros((500,8)),index=np.linspace(0,1,500),columns=['Recall','Precision','F1-Score','Accuracy','Specificity','PPV','NPV','AUC'])
    prev = 1/(match_number+1)
    for t in np.linspace(0,1,500):
        preds = np.where(probs>t,1,0)
       # precision = precision_score(ground_truth,preds,zero_division=0)
        recall = recall_score(ground_truth,preds,zero_division=0)
        accuracy = accuracy_score(ground_truth,preds)
        auc = roc_auc_score(ground_truth,probs)
        tn, fp, fn, tp = confusion_matrix(ground_truth,preds).ravel()
        specificity = tn/(tn+fp)
        ppv = (recall* (prev))/(recall * prev + (1-specificity) * (1-prev))
        precision = ppv
        f1score = 2*(precision*recall)/(precision+recall)
        if tn== 0 and fn==0:
            npv = 0
        else:
            npv = (specificity* (1-prev))/(specificity * (1-prev) + (1-recall) * (prev))
        threshold_metrics.loc[t,['Recall','Precision','F1-Score','Accuracy','Specificity','PPV','NPV','AUC']] = [recall,precision,f1score,accuracy,specificity,ppv,npv,auc]
    
    """ Identify Results that Satisfy Constraints and Best Threshold """
 #   value_of_interest = threshold_metrics.loc[:,metric_of_interest].max()
    condition1 = threshold_metrics.loc[:,metric_of_interest] < value_of_interest + error
    condition2 = threshold_metrics.loc[:,metric_of_interest] > value_of_interest - error
    combined_condition = condition1 & condition2
    if metric_of_interest == 'Recall':
        sort_col = 'Precision'
    elif metric_of_interest == 'Precision':
        sort_col = 'Recall'
    elif metric_of_interest == 'F1-Score':
        sort_col = 'F1-Score'
    sorted_results = threshold_metrics[combined_condition].sort_values(by=sort_col,ascending=False)
  #  print(sorted_results)
    if len(sorted_results) > 0:
        """ Only Record Value if Condition is Satisfied """
        results_df.loc[['Recall','Precision','F1-Score','Accuracy','Specificity','PPV','NPV','AUC'],fold_number] = sorted_results.iloc[0,:]   
        best_threshold.iloc[fold_number] = sorted_results.iloc[0,:].name
    else:
        print('No Threshold Found for Constraint!')
    
    return best_threshold, results_df

def calculate_results(metrics_dict,outputs,preds,results_df,fold_number):
    for metric_name,metric in metrics_dict.items():
        result = metric(outputs['eval'],preds)
        results_df.loc[metric_name,fold_number] = result
    return results_df

def perform_classification_one_fold(inputs,outputs,metrics_dict,results_df,best_threshold,confusion_array,importance_array,fold_number,model,model_type,sampler,match_number,metric_of_interest='recall',value_of_interest=0.80,error=0.04,find_best_threshold=False):
    if model_type in ['LR','SVM','RF','XGB','BalancedRF','BalancedBag','EasyEnsemble','XGBOD','DT']:
        """ Supervised Methods """
        model.fit(inputs['train'],outputs['train'])
        preds = model.predict(inputs['eval'])
        if model_type in ['BalancedRF','RF','DT','XGB']:
            feature_importances = model.feature_importances_
            importance_array[:,fold_number] = feature_importances
    elif model_type in ['LOF','IF']:
        """ Outlier Detection Methods - Unsupervised """
        model.fit(inputs['train'])
        preds = model.predict(inputs['eval'])
        preds = np.where(preds==1,0,1) #1 in original formulation means inlier         
    elif model_type in ['OCSVM','LSCP']:
        model.fit(inputs['train'])
        preds = model.predict(inputs['eval'])
    
    if find_best_threshold == True:
        """ Find Best Threshold """
        best_threshold, results_df = find_threshold_at_metric(model, inputs, outputs, best_threshold, metric_of_interest, value_of_interest, results_df, fold_number, error,match_number)
    else:
        """ Use Default Threshold (Argmax) """
        results_df = calculate_results(metrics_dict, outputs, preds, results_df, fold_number)
        confusion_array[:,:,fold_number] = confusion_matrix(outputs['eval'],preds,labels=[1,0])
    
    return results_df, confusion_array, importance_array, best_threshold
   
def load_data_and_model_one_configuration(data,scenario,match_number,save_control_indices,imputation_method,feature_type,sampler_type,model_type,use_sampler,add_demographics=False,include_dummy_variable=False, resp=False, bytime=False, admission= False, bytimepn=False,flag='no'):
    """ Load Case Data """
    case_data, case_indices = load_case_data(data, scenario)
    if admission:
        case_data = case_data[(~case_data['EpisodeID'].isnull() & ~case_data['Vital_Sign Temperature Tympanic'].isnull())]
    """ Load Control Data """
    control_data = load_control_data(data, scenario)
   
    """ Load Matched Controls (If Needed) """
    control_cohort = load_matched_control_cohort(scenario, True, match_number, save_control_indices, control_data, data, case_indices,resp,admission,bytimepn)
    """ Load Combined Cohort """
    df = load_combined_cohort(case_data, control_cohort)
    
    """ Load Inputs and Outputs """
    X, Y, Z = load_features_and_labels(df, include_dummy_variable=include_dummy_variable)
    """ Load Patient Ids For DataSplits """
    train_ids, val_ids, test_ids = load_phase_ids(df, match_number,bytime, bytimepn,flag)
    
    """ Split Data For Training """
    X_train,Y_train,Z_train, X_val,Y_val,Z_val, X_test,Y_test,Z_test = load_data_splits(df, X, Y, Z, train_ids, val_ids, test_ids)
    """ Impute Missing Features Based on Training Data """
    X_train, X_val, X_test, features = impute_missing_features(imputation_method, X_train, X_val, X_test, Z_train, Z_val, Z_test)
    
    """ Filter Features e.g. Absolute, Relative, etc. """
    X_train, X_val, X_test = filter_features(X_train, feature_type, add_demographics), filter_features(X_val, feature_type, add_demographics), filter_features(X_test, feature_type, add_demographics)
    """ Standardize Features """
    features = X_train.columns.tolist()
    X_train, X_val, X_test = standardize_features(X_train, X_val, X_test,feature_type)
    
    #balance train & validation set
 #   res = RandomUnderSampler(random_state=42)
#    X_train, Y_train = res.fit_resample(X_train, Y_train)
 #   Z_train = Z_train.loc[res.sample_indices_,:]
  #  X_val, Y_val = res.fit_resample(X_val, Y_val)
 #   Z_val = Z_val.loc[res.sample_indices_,:]
    
    """ Load Under/Over Sampler """
    sampler = obtain_sampler(sampler_type)
    """ Load Classification Model """
    contamination = np.histogram(Y_train,2)[0][1]/np.sum(np.histogram(Y_train,2)[0]) #only used for OD methods
    model = obtain_model(model_type, sampler, contamination, use_sampler)    
    
    return X_train,Y_train, X_val,Y_val, X_test,Y_test,Z_test, sampler, model, features

def perform_classification_kfolds(nfolds,X_train,X_val,Y_train,Y_val,model,model_type,sampler,match_number,metric_of_interest='Recall',value_of_interest=0.80,error=0.04,find_best_threshold=True):
    """ Function Implemented When Evaluation = False, and Inference = False """

    """ Combined Train/Val For KFold CV """
    kf = StratifiedKFold(n_splits=nfolds,random_state=0,shuffle=True)
    combined_features = pd.concat((X_train,X_val),0,ignore_index=True)
    combined_labels = pd.concat((Y_train,Y_val),0,ignore_index=True)
    
    """ DataFrame to Store Results for All Folds """
    results_df = pd.DataFrame(np.zeros((len(metric_names),nfolds)),index=metric_names)
    best_threshold = pd.Series(np.zeros((nfolds)))
    confusion_array = np.zeros((2,2,nfolds)) 
    importance_array = np.zeros((combined_features.shape[1],nfolds))
    
    """ Perform KFold Classification """
    inputs,outputs = dict(), dict()
    
    for fold_number,(train_index, val_index) in tqdm(enumerate(kf.split(combined_features,combined_labels))):
        x_tr,y_tr = combined_features.iloc[train_index,:], combined_labels.iloc[train_index]
        x_val,y_val = combined_features.iloc[val_index,:], combined_labels.iloc[val_index]
        
        """ Novelty Detection Trains Exclusively on ONE Class """
        if model_type in ['OCSVM']: #novelty detection requires this modification
            positive_indices_in_train = y_tr.index[y_tr==1]
            x_val2,y_val2 = x_tr.iloc[positive_indices_in_train,:], y_tr.iloc[positive_indices_in_train]
            x_val,y_val = pd.concat((x_val,x_val2),0), pd.concat((y_val,y_val2),0)
            x_tr,y_tr = x_tr.drop(positive_indices_in_train), y_tr.drop(positive_indices_in_train)
        
        """ Shuffle Fold Data """
        random.seed(0)
        shuffled_indices = random.sample(list(np.arange(len(y_tr))),len(y_tr))
        x_tr,y_tr = x_tr.iloc[shuffled_indices,:], y_tr.iloc[shuffled_indices]
        
        inputs['train'], inputs['eval'] = [x_tr,x_val]
        outputs['train'], outputs['eval'] = [y_tr,y_val]
        
        """ Perform Fold Classification """
        results_df, confusion_array, importance_array, best_threshold = perform_classification_one_fold(inputs,outputs,metrics_dict,results_df,best_threshold,confusion_array,importance_array,fold_number,model,model_type,sampler,match_number,metric_of_interest=metric_of_interest,value_of_interest=value_of_interest,error=error,find_best_threshold=find_best_threshold)

    return results_df, best_threshold


def fit_and_evaluate_on_test_set(X_train, X_val, Y_train, Y_val, X_test, Y_test, model, model_type, sampler,chosen_threshold,match_number):
    """ Function Implemented When Evaluation = True, and Inference = False """
    
    """ Combined Training and Validation Data From KFold CV """
    combined_features = pd.concat((X_train,X_val),0,ignore_index=True)
    combined_labels = pd.concat((Y_train,Y_val),0,ignore_index=True)
    
    inputs, outputs = dict(), dict()
    inputs['train'], inputs['eval'] = [combined_features, X_test]
    outputs['train'], outputs['eval'] = [combined_labels, Y_test]
        
    """ Fit Model and Make Predictions """
    model.fit(inputs['train'],outputs['train'])
    probs = model.predict_proba(inputs['eval'])[:,1]
    ground_truth = Y_test
    
    """ Perform Evaluation """
    preds = np.where(probs>chosen_threshold,1,0)
   # precision = precision_score(ground_truth,preds,zero_division=0)
    recall = recall_score(ground_truth,preds,zero_division=0)
    accuracy = accuracy_score(ground_truth,preds)
    auc = roc_auc_score(ground_truth,probs) 
    ns_fpr, ns_tpr, _ = roc_curve(ground_truth, probs)
  #  ns_fpr, ns_tpr, _ = precision_recall_curve(ground_truth,probs)
    tn, fp, fn, tp = confusion_matrix(ground_truth,preds).ravel()
    prev = 1/(match_number+1)
    specificity = tn/(tn+fp)
    ppv = (recall* (prev))/(recall * prev + (1-specificity) * (1-prev))
    precision = ppv
    npv = (specificity* (1-prev))/(specificity * (1-prev) + (1-recall) * (prev))
    f1score = 2*(precision*recall)/(precision+recall)
    """ Calculate Feature Importances if Possible """
    if model_type in ['RF','DT','XGB']:
        importance_array = model.feature_importances_
    elif model_type in ['LR']:
        importance_array = model.coef_
        importance_array = np.reshape(importance_array,(len(importance_array[0]),))
    else:
        importance_array = []
    explainer = shap.TreeExplainer(model)
    tmp = pd.DataFrame(data=X_test.values)
    shap_values = explainer.shap_values(tmp)
    importance_array = shap_values
    
    results_df = pd.DataFrame([recall,precision,f1score,accuracy,specificity, ppv,npv,auc])
    
    return results_df, importance_array, model, ns_fpr,ns_tpr

def perform_inference_on_test_set(fitted_models_dict,confusion_matrix_list,labels_list,preds_list,cluster_ids_list,configuration_name,X_test,Y_test,Z_test,chosen_threshold,match_number,features):
    """ Function Implemented When Evaluation = True, and Inference = True """

    """ Load Appropriate Model """
    model = fitted_models_dict[str(configuration_name)]
    
    """ Make Predictions """
    cluster_ids = np.array(Z_test['Encounter'].tolist())
    ground_truth = Y_test
    probs = model.predict_proba(X_test)[:,1]
    
    explainer = shap.TreeExplainer(model)
    tmp = pd.DataFrame(data=X_test.values,columns=features)
    shap_values = explainer.shap_values(tmp)
#    print(shap_values)
 #   f = plt.figure()
 #   shap.summary_plot(shap_values, tmp)
 #   f.savefig("./shap"+str(configuration_name)+".png", bbox_inches='tight', dpi=600)
#    shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(X_test)
#    shap.summary_plot(shap_interaction_values,tmp)
    """ Perform Evaluation """
    preds = np.where(probs>chosen_threshold,1,0)
  #  precision = precision_score(ground_truth,preds,zero_division=0)
    recall = recall_score(ground_truth,preds,zero_division=0)
    accuracy = accuracy_score(ground_truth,preds)
    auc = roc_auc_score(ground_truth,probs)    
    conf_matrix = confusion_matrix(ground_truth,preds)
    tn, fp, fn, tp = confusion_matrix(ground_truth,preds).ravel()
    specificity = tn/(tn+fp)
    prev = 1/(match_number+1)
    ppv = (recall* (prev))/(recall * prev + (1-specificity) * (1-prev))
    precision = ppv
    npv = (specificity* (1-prev))/(specificity * (1-prev) + (1-recall) * (prev))
    f1score = 2*(precision*recall)/(precision+recall)
    """ Calculate Feature Importances if Possible """
    if isinstance(model,RandomForestClassifier):
        importance_array = model.feature_importances_
    elif isinstance(model,LogisticRegression):
        importance_array = model.coef_#,model.intercept_]
    elif isinstance(model,DecisionTreeClassifier):
        importance_array = model.feature_importances_
    else:
        importance_array = []
    results_df = pd.DataFrame([recall,precision,f1score,accuracy,specificity, npv,ppv,auc])
    importance_array = shap_values
    """ Store Preds, Ground Truths, and Confusion Matrix """
    confusion_matrix_list.append(conf_matrix)
    labels_list.append(np.array(ground_truth.tolist()))
    preds_list.append(preds)
    cluster_ids_list.append(cluster_ids)
    
    return results_df, importance_array, confusion_matrix_list, labels_list, preds_list, cluster_ids_list

#%%
def load_previous_configurations(scenario,evaluation,perform_inference,results_path):
    if evaluation == True and perform_inference == True:
        """ Clean Slate If Performing Final Test Set Inference Only """
        configurations_name_list = []
        results_configurations_list = []        
    else:
        """ Load Previously Saved Configuration Names and Results To Append To """
        try:
            with open(os.path.join(results_path,'completed_configurations_%s.pkl' % scenario),'rb') as f:
                configurations_name_list = pickle.load(f)
            with open(os.path.join(results_path,'completed_results_%s.pkl' % scenario),'rb') as f:
                results_configurations_list = pickle.load(f)
        except:
            configurations_name_list = []
            results_configurations_list = []
    return configurations_name_list,results_configurations_list

def save_configurations(scenario,configurations_name_list,results_configurations_list,evaluation,perform_inference,results_path):
    """ Conditions to Allow for Saving """
    condition0 = evaluation == False
    condition1 = evaluation == True and perform_inference == False
    combined_condition = condition0 | condition1
    if combined_condition:
        """ Save Running List of Results and Configurations """
        with open(os.path.join(results_path,'completed_configurations_%s.pkl' % scenario),'wb') as f:
            pickle.dump(configurations_name_list,f)
        with open(os.path.join(results_path,'completed_results_%s.pkl' % scenario),'wb') as f:
            pickle.dump(results_configurations_list,f)

def save_fitted_models(fitted_models_dict,results_path):
    """ Save Models Fitted on All Folds """
    with open(os.path.join(results_path,'fitted_models_dict.pkl'),'wb') as f:
        pickle.dump(fitted_models_dict,f)

def load_fitted_models(results_path):
    with open(os.path.join(results_path,'fitted_models_dict.pkl'),'rb') as f:
        fitted_models_dict = pickle.load(f)
    return fitted_models_dict

#ind_to_keep = np.where([el[1]==False for el in configurations_name_list])[0]
#%%
def run_experiments(data, scenario_list, match_number_list, imputation_method_list, feature_type_list, model_type_list, sampler_type_list, evaluation, perform_inference, results_path, value_of_interest=0.80, include_dummy_variable=False, resp=False, bytime=False, admission= False, bytimepn=False,flag='no'):
    
    nconfig = 0
    for scenario in scenario_list:

        configurations_name_list,results_configurations_list = load_previous_configurations(scenario, evaluation, perform_inference, results_path)
        importance_array_list = []
        fitted_models_dict = dict()
        features_list, labels_list, preds_list, confusion_matrix_list, cluster_ids_list = [], [], [], [], []
        if perform_inference == True:
            """ Load Previously Fitted Models """
            fitted_models_dict = load_fitted_models(results_path)
            #with open(os.path.join(results_path,'fitted_models_dict.pkl'),'rb') as f:
            #    fitted_models_dict = pickle.load(f)
 
        for match_number in match_number_list:
         #   print(match_number)
            for imputation_method in imputation_method_list:
                
                for feature_type in feature_type_list:
                    
                    for model_type in model_type_list:
                        #if model_type in ['LR','SVM','RF','XGB']:
                        #    use_sampler_list = [False,True]
                        #else: #models for which sampler is not compatible 
                        #    use_sampler_list = [False]
                        use_sampler_list = [False]
                        
                        for use_sampler in use_sampler_list:
                            if use_sampler == True:
                                current_sampler_type_list = sampler_type_list
                            elif use_sampler == False:
                                current_sampler_type_list = [None]
                                
                            for sampler_type in current_sampler_type_list:
                                print('Config #%i' % nconfig)
    
                                configuration_name = [scenario,evaluation,match_number,imputation_method,feature_type,model_type,use_sampler,sampler_type]
                                """ Skip Configuration if Already Performed """
                                if configuration_name in configurations_name_list and evaluation == False: #remove last condition to ensure you NEVER repeat
                                    print('Skipped!')
                                    nconfig += 1
                                    continue
                                
                                """ Load Data and Model """    
                                #print(match_number)
                                X_train,Y_train, X_val,Y_val, X_test,Y_test,Z_test, sampler, model, features = load_data_and_model_one_configuration(data, scenario, match_number, save_control_indices, imputation_method, feature_type, sampler_type, model_type, use_sampler, include_dummy_variable=include_dummy_variable,resp=resp,bytime=bytime, admission= admission, bytimepn=bytimepn,flag=flag)
                                
                                
                                """ Perform either KFold CV or Evaluation on Test Set """
                                if evaluation == False:
                                    """ Perform KFold Classification """
                                    results_df, best_threshold = perform_classification_kfolds(nfolds, X_train, X_val, Y_train, Y_val, model, model_type, sampler, match_number, metric_of_interest=metric_of_interest,value_of_interest=value_of_interest,error=error,find_best_threshold=find_best_threshold)
                                    current_results = results_df.mean(axis=1).tolist()
                                    current_results.append((best_threshold.mean()))
                                    importance_array = []
                                    #features_list = []
                                elif evaluation == True:
                                    """ Load CV Results to Obtain Prediction Threshold """
                                    print(results_path)
                                    mean_cv_results = pd.read_csv(os.path.join(results_path,'10CV_Mean_Multiple_Imputation.csv'))
                                    condition1 = mean_cv_results['Model']==model_type
                                    condition2 = mean_cv_results['Match Number']==match_number
                                    combined_conditions = condition1 & condition2
                                    test_threshold = mean_cv_results[combined_conditions]['Threshold'].item()
                                    if perform_inference == False:
                                        """ Use Threshold to Fit and Evalute on Test Set - Once """
                                        results_df, importance_array, model,fpr,tpr = fit_and_evaluate_on_test_set(X_train, X_val, Y_train, Y_val, X_test, Y_test, model, model_type, sampler, test_threshold,match_number)
                                        
                                        #roc auc
                                   #     plt.plot(fpr, tpr, label=imputation_method)
                                   #     plt.xlabel('False Positive Rate')
                                   #     plt.ylabel('True Positive Rate')
                                    #    plt.legend()  
                                    #    plt.savefig(results_path + '/roc_'+str(match_number)+'.pdf')
                                        
                                    #    plt.plot(tpr, fpr, label=imputation_method)
                                    #    plt.xlabel('Recall')
                                    #    plt.ylabel('Precision')
                                    #    plt.legend() 
                                    #    plt.savefig(results_path + '/prc_'+str(match_number)+'.pdf')
                                    #    if nconfig%3 ==2:
                                    #        plt.close()
                                        """ Store Fitted Model Into Dict """
                                        fitted_models_dict[str(configuration_name)] = model
                                        save_fitted_models(fitted_models_dict,results_path)
                                    elif perform_inference == True:
                                        """ Only Perform Inference on Test Set - No Fitting """
                                        results_df, importance_array, confusion_matrix_list, labels_list, preds_list, cluster_ids_list = perform_inference_on_test_set(fitted_models_dict, confusion_matrix_list, labels_list, preds_list, cluster_ids_list, configuration_name, X_test, Y_test, Z_test, test_threshold,match_number,features)
                                    
                                    current_results = results_df.mean(axis=1).tolist()
                                    current_results.append(test_threshold)                                    
                                
                           
                                """ Store Results and Configuration Name """
                                results_configurations_list.append(current_results)
                                configurations_name_list.append(configuration_name)
                                
                                
                                importance_array_list.append(importance_array)
                                features_list = features



                                """ Save Configurations """
                                save_configurations(scenario, configurations_name_list, results_configurations_list, evaluation, perform_inference, results_path)
    
                                nconfig += 1 #only used for printing
                              
    return results_configurations_list, configurations_name_list, importance_array_list, features_list, confusion_matrix_list, labels_list, preds_list, cluster_ids_list

def obtain_results_path(savepath,value_of_interest,feature,metric_of_interest):
    """ Create Results Path if it Doesn't Exist """
 #   if resp:
 ##       if bytime:
  #          results_path = os.path.join(savepath,'Results','Sensitivity_%.3f_resp_time' % value_of_interest)
   #     else:        
    #        results_path = os.path.join(savepath,'Results','Sensitivity_%.3f_resp' % value_of_interest)
  #  elif admission:
  #      if bytime:
  #          results_path = os.path.join(savepath,'Results','Sensitivity_%.3f_adm_time' % value_of_interest)
  #      else:        
  #          results_path = os.path.join(savepath,'Results','Sensitivity_%.3f_adm' % value_of_interest)
  #  else:            
  #      if bytime:
  #          results_path = os.path.join(savepath,'Results','Sensitivity_%.3f_time' % value_of_interest)
  #      else:        
    results_path = os.path.join(savepath,'Results', feature, metric_of_interest + '_%.3f' % value_of_interest)
    if os.path.isdir(results_path) == False:
        os.makedirs(results_path)
    return results_path

#%%
def save_results_df(results_path,configurations_name_list,results_configurations_list,importance_array_list,features_list,metric_names):
    """ Convert Lists of Results into DataFrame """
    configurations_name_df = pd.DataFrame(configurations_name_list,columns=['Scenario','Evaluation','Match Number','Imputation','Features','Model','Use Sampler','Sampler Type'])
    results_configurations_df = pd.DataFrame(results_configurations_list,columns=metric_names+['Threshold'])
    performance_df = pd.concat((configurations_name_df,results_configurations_df),1)
    augmented_metric_names = metric_names + ['Threshold']

    """ Multiple Imputation Strategy """
    if evaluation == False:
        """ MIS for CV - Mean and Std Across Imputation Strategies """
        cv_df = performance_df[performance_df['Evaluation']==False]
        mean_multiple_imputation = cv_df.groupby(['Match Number','Model'])[augmented_metric_names].mean()
        std_multiple_imputation = cv_df.groupby(['Match Number','Model'])[augmented_metric_names].std()
        
        """ This Needs to be Saved Because Thresholds Are Accessed During Final Evaluation """
        mean_multiple_imputation.to_csv(os.path.join(results_path,'10CV_Mean_Multiple_Imputation.csv'))
        std_multiple_imputation.to_csv(os.path.join(results_path,'10CV_Std_Multiple_Imputation.csv'))
    elif evaluation == True:
        
        """ MIS for Test - Mean and Std Across Imputation Strategies """
        test_df = performance_df[performance_df['Evaluation']==True]
        mean_multiple_imputation = test_df.groupby(['Match Number','Model'])[augmented_metric_names].mean()
        std_multiple_imputation = test_df.groupby(['Match Number','Model'])[augmented_metric_names].std()
        
        """ Save Final Test Performance """
        mean_multiple_imputation.to_csv(os.path.join(results_path,'Test_Mean_Multiple_Imputation.csv'))
        std_multiple_imputation.to_csv(os.path.join(results_path,'Test_Std_Multiple_Imputation.csv'))
    
        """ Feature Importance Based on Final Fitting Procedure """
        
        f = plt.figure()
        shap.summary_plot(np.mean(importance_array_list,0), features_list, plot_type="bar")
        f.savefig("./shap.png", bbox_inches='tight', dpi=600)
        
        for i in range(len(importance_array_list)):
            importance_array_list[i] = np.mean(np.abs(importance_array_list[i]),0)
            
        test_configurations_df = test_df.loc[:,:'Sampler Type'].reset_index()
        importance_df = pd.DataFrame(importance_array_list,columns=features_list)
        full_importance_df = pd.concat((test_configurations_df,importance_df),1)
       # print(np.sort(full_importance_df.columns.tolist()))
        # """ MIS for Test Importance - Average Importance Across Imputation Strategies """
      #  print(full_importance_df['Blood_Test ALBUMIN'])
        mean_importance = full_importance_df.groupby(['Match Number','Model'])[features_list].mean()
        std_importance = full_importance_df.groupby(['Match Number','Model'])[features_list].std()
    #    print(mean_importance.columns)
        # """ Save Final Test Importance """
        mean_importance.to_csv(os.path.join(results_path,'Test_Mean_Importance.csv'))
        std_importance.to_csv(os.path.join(results_path,'Test_Std_Importance.csv'))
        

#%%
""" Obtain Cluster IDs for Patients Correctly and Miss-classified - For Further Inspection of Bias """
def obtain_classification_cluster_ids(test_df,labels_list,preds_list,cluster_ids_list,match_number_list):
    if evaluation == True and perform_inference == True:    
        labels_dict = dict(zip(test_df.index,labels_list))
        preds_dict = dict(zip(test_df.index,preds_list))
        cluster_id_dict = dict(zip(test_df.index,cluster_ids_list))
        
        fp_cluster_id_dict, fn_cluster_id_dict, correct_cluster_id_dict = dict(), dict(), dict()
        for match_number in match_number_list:
            
            match_indices = test_df[test_df['Match Number'] == match_number].index.tolist()
            for index in match_indices:         
                current_labels = labels_dict[index]
                current_preds = preds_dict[index]
                current_cluster_ids = cluster_id_dict[index]
                """ FP Pathway - Predict 1 Actual 0 """
                fp_patients_indices = np.where(np.logical_and(current_labels==0,current_preds==1))[0]
                """ FN Pathway - Predict 0 Actual 1 """
                fn_patients_indices = np.where(np.logical_and(current_labels==1,current_preds==0))[0]
                """ Correct Pathway """
                tn_patients_indices = np.where(np.logical_and(current_labels==0,current_preds==0))[0]
                """ FN Pathway - Predict 0 Actual 1 """
                tp_patients_indices = np.where(np.logical_and(current_labels==1,current_preds==1))[0]
                """ Obtain Cluster ID for those with FP and FN """
                fp_patients = current_cluster_ids[fp_patients_indices]
                fn_patients = current_cluster_ids[fn_patients_indices]
                """ Obtain Cluster ID for those with TPs and TNs"""
                tp_patients = current_cluster_ids[tp_patients_indices]
                tn_patients = current_cluster_ids[tn_patients_indices]                
                correct_patients = np.concatenate((tp_patients,tn_patients))
                """ Store Cluster IDs for use Later """
                fp_cluster_id_dict[index] = fp_patients
                fn_cluster_id_dict[index] = fn_patients
                correct_cluster_id_dict[index] = correct_patients
    
    return fp_cluster_id_dict, fn_cluster_id_dict, correct_cluster_id_dict

def  save_miss_classified(results_path,configurations_name_list,results_configurations_list,labels_list,preds_list,cluster_ids_list,match_number):
        """ Convert Lists of Results into DataFrame """
        configurations_name_df = pd.DataFrame(configurations_name_list,columns=['Scenario','Evaluation','Match Number','Imputation','Features','Model','Use Sampler','Sampler Type'])
        results_configurations_df = pd.DataFrame(results_configurations_list,columns=metric_names+['Threshold'])
        performance_df = pd.concat((configurations_name_df,results_configurations_df),1)
    
        test_df = performance_df[performance_df['Evaluation']==True]
        fp_cluster_id_dict, fn_cluster_id_dict, correct_cluster_id_dict = obtain_classification_cluster_ids(test_df,labels_list,preds_list,cluster_ids_list,match_number)
        cluster_id_dict = {'fp_encounter_id_dict':fp_cluster_id_dict,'fn_encounter_id_dict':fn_cluster_id_dict,'correct_encounter_id_dict':correct_cluster_id_dict}
      #  print(cluster_id_dict)
        with open(os.path.join(results_path,'encounter_id_dict.pkl'),'wb') as f:
            pickle.dump(cluster_id_dict,f)
    
    
#%%
scenario_list = ['covid_vs_all'] #['covid_vs_all_non_flu']#,'covid_or_flu_vs_all_non_flu'] #options: 'covid_vs_flu_only' | 'covid_vs_all_non_flu' | 'covid_or_flu_vs_all_non_flu' | 'covid_vs_all'

imputation_method_list = ['mean','median','age-based mean']#,'MICE'] #options: 'mean' | 'median' | 'MICE'
nfolds = 10

model_type_list = ['XGB']#,'RF','LR']#,'XGB','BalancedRF','OCSVM']#,'LSCP'] #options: 'LR' | 'RF' | 'SVM' | 'LOF' | 'BalancedRF' | 'BalancedBag' | 'EasyEnsemble' | 'OCSVM' | 'LSCP' | 'HBOS' | 'CBLOF' 
#use_sampler = False
sampler_type_list = ['SMOTE']#,'SMOTEENN'] #options: 'SMOTE' | 'SVMSMOTE' | 'SMOTEENN'
export_data = False
#print('Model Type: %s - Sampler Type: %s' % (model_type,sampler_type))

feature_type_list = ['Blood & Vitals']#,'Blood', 'Blood_Delta', 'Blood & Blood_Delta', 'Blood & Blood_Delta & Blood_Gas', 'All except CCI', 'All Features', 'Vital_Signs', 'Blood & Blood_Gas','Blood_Gas'] #options: 'Blood_Test' | 'Blood_Test_Delta' | 'both' | 'AllBlood' | 'All' | 'CCI'
#|Vital_Signs | 'Blood_Test_Gas' | 'Blood_Gas'
add_demographics = False
include_dummy_variable = True
match_cohort = True

prevalence_list =[0.005]#[0.01,0.02, 0.05, 0.10, 0.20, 0.25, 0.33, 0.50, 1.00] #np.arange(0.05,0.55,0.1)

#AS: Correction to match number; subtracting one so that match number = 1/prevalence
match_number_list = [int(np.round(1/prevalence)) for prevalence in prevalence_list]
save_control_indices = True #do you want to save control cohort indices

metric_names = ['Recall','Precision','F1-Score','Accuracy','Specificity','PPV','NPV','AUC']
metrics = [recall_score,precision_score,accuracy_score,roc_auc_score]#,confusion_matrix]
metrics_dict = dict(zip(metric_names,metrics))

""" Determine Ideal Metric and Value """
find_best_threshold = True #False = use default threshold, True = find best threshold using below params
metric_of_interest = 'Recall'
value_of_interest_list = [0.80]#[0.70,0.75,0.80,0.85,0.90] #not using this value but max
error = 0.07

evaluation = False #true means perform final fit on ALL FOLDS and evaluate on held-out test set
perform_inference = False #true means perform inference on held-out test set

flag = 'Death'

bytime = False
resp = False
adm = False
bytimepn = True

if __name__ == '__main__':
    """ Load Original DataFrame - Done Once """
    data = load_data(basepath)
    """ Iterate Over Range of Sensitivity Targets """
    for value_of_interest in value_of_interest_list:
        for f in feature_type_list:
            """ Obtain Results Path - Where All Results are Saved """
            results_path = obtain_results_path(savepath,value_of_interest,f,metric_of_interest)
            """ Perform All Experiments """
            results_configurations_list, configurations_name_list, importance_array_list, features_list, confusion_matrix_list, labels_list, preds_list, cluster_ids_list = run_experiments(data, scenario_list, match_number_list, imputation_method_list, [f], model_type_list, sampler_type_list, evaluation, perform_inference, results_path, value_of_interest=value_of_interest, include_dummy_variable=include_dummy_variable,resp=resp,bytime=bytime, admission= adm, bytimepn=bytimepn,flag=flag)
            """ Save Result for This Iteration """
            if perform_inference == True:
                save_miss_classified(results_path,configurations_name_list,results_configurations_list,labels_list,preds_list,cluster_ids_list,match_number_list)
            else:
                save_results_df(results_path, configurations_name_list, results_configurations_list, importance_array_list, features_list, metric_names)
                  

#fp_cluster_id_dict, fn_cluster_id_dict, correct_cluster_id_dict = obtain_classification_cluster_ids(test_df,labels_list,preds_list,cluster_ids_list)
#cluster_id_dict = {'fp_encounter_id_dict':fp_cluster_id_dict,'fn_encounter_id_dict':fn_cluster_id_dict,'correct_encounter_id_dict':correct_cluster_id_dict}

# with open(os.path.join(savepath,'Results','encounter_id_dict.pkl'),'wb') as f:
#     pickle.dump(cluster_id_dict,f)
            

            
 