# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 14:25:17 2022

@author: dht
"""


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from nproc import npc
import os


train_path = 'TwoLeadECG/TwoLeadECG_TRAIN.tsv'
test_path = 'TwoLeadECG/TwoLeadECG_TEST.tsv'
train_data = pd.read_csv(train_path, sep='\t', header = None).values
test_data = pd.read_csv(test_path, sep='\t', header = None).values
y_train, x_train = train_data[:, 0], train_data[:, 1:]
y_test, x_test = test_data[:, 0], test_data[:, 1:]
y_train[y_train != 1] = 0
y_test[y_test != 1] = 0




#未进行特征工程
def data(x_test, x_train):
    X_train = x_train
    X_test = x_test
    return X_train,  X_test


#特征工程提取时序特征
def extract(x_test, x_train):
    x = np.vstack((x_train, x_test))   
    N = x.shape[0]
    extraction_settings = ComprehensiveFCParameters()
    master_train_df = pd.DataFrame({'feature': x.flatten(),
                              'id': np.arange(N).repeat(x.shape[1])})
    
    # 时间序列特征工程
    X = extract_features(timeseries_container=master_train_df, n_jobs=0, column_id='id', impute_function=impute,
                         default_fc_parameters=extraction_settings)
    
    
    
    X_train,  X_test = X[0:x_train.shape[0]], X[x_train.shape[0]:x.shape[0]] 
    
    return X_train,  X_test


#提取特征并选择特征
def extract_select(x_test, x_train):
    x = np.vstack((x_train, x_test))
    y = np.hstack((y_train, y_test))
    N = x.shape[0]
    extraction_settings = ComprehensiveFCParameters()
    master_train_df = pd.DataFrame({'feature': x.flatten(),
                              'id': np.arange(N).repeat(x.shape[1])})
    
    # 时间序列特征工程
    X = extract_features(timeseries_container=master_train_df, n_jobs=0, column_id='id', impute_function=impute,
                         default_fc_parameters=extraction_settings)
    
    #特征选择
    X = select_features(X, y)
    X_train,  X_test = X[0:x_train.shape[0]], X[x_train.shape[0]:x.shape[0]] 
    
    return X_train,  X_test




if __name__ == '__main__':
    X_train,  X_test = extract(x_test, x_train)
    net1 = GaussianNB()
    net1.fit(X_train, y_train)
    y_pre1 = net1.predict(X_test)
    acc = accuracy_score(y_pre1, y_test)        
    print(acc)
    
    
    







