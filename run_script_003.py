# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 18:35:49 2022

@author: Chris
"""
# imports block:
# import pandas to read csv file:
import pandas as pd

# import HOST model:
import host

# ----
# data block:
# import data file:
raw_data = pd.read_csv('E:/Research/9_seus_arma/host_model/datasets/example_data.csv')
raw_data.drop('Unnamed: 0', axis=1, inplace=True)


# ----
# HOST usage block:
# Using model for multiple cases in data file:
    
# generate columns names to use in for loop, for multiple calculations
columns = list(raw_data.columns)

# prepare overview results tables for main outputs only
# for occurrence training results:
occ_bf_res = pd.DataFrame(columns=['accuracy','best_type','f1score','precision',
                                   'recall'])
# occurrence testing results:
occ_test_res = pd.DataFrame(columns=['accuracy','precision','recall','f1score'])
# flow training results:
flow_bf_res = pd.DataFrame(columns=['best_type','RMSE','corr','chi2_stat',
                                    'MWU_stat'])
# and flow testing results:
flow_test_res = pd.DataFrame(columns=['MWU_stat','MWU_p-val','chi2_stat',
                                      'chi2_p-val', 'corr','corr_p-val','MSE',
                                      'RMSE'])

# run in loop: for each column generate trainig and testing set and run HOST 
# analysis: 
for column in columns:
    # assign flow data from column in dataframe:
    flow = raw_data[column]
    
    # preprocess the data using provided event type and assigns date index ranges
    # to time series
    training, testing, testx = host.preprocessor.preprocess(data=flow,
                                                            event='lf', 
                                                            beginning='2-1-1979')
    
    # generate best fitted HOST model for low flow occurrence (categorical) data:
    occ_bf, occ_pr, occ_pa, funct, lim = host.train_occurrence(training)
    # stores: best_fitted function accuracy statistics in occ_bf variable
    # stores: predicted values for best_fitted function in occ_pr variable
    # stores: best_fitted function parameters in occ_pa variable
    # used for testing purposes:
    # stores: fitted function object in funct variable
    # stores: decision threshold function object in lim variable
    
    # generate best fitted HOST model for minimal flow (continuous) data:
    flow_bf, flow_pr, flow_pa, funct_flow = host.train_flow(training)
    # stores: best_fitted function distribution statistics in flow_bf variable
    # stores: predicted values for best_fitted function in flow_pr variable
    # stores: best_fitted function parameters in flow_pa variable
    # used for testing purposes:
    # stores: fitted function object in funct variable  
    
    # saves outputs to results overview tables for training set:
    # for occurrence data:
    occ_bf_res.loc[column] = [occ_bf['accuracy'],occ_bf['best_type'],
                              occ_bf['f1score'],occ_bf['precision'],
                              occ_bf['recall']]
    # for flow data:
    flow_bf_res.loc[column] = [flow_bf['best_type'],flow_bf['RMSE'],
                              flow_bf['corr'],flow_bf['chi2_stat'],
                              flow_bf['MWU_stat']]  
    
    # performing accuracy check for found models on test sets:
    # for occurrence data:
    test_occ = host.test_occurrence(testing, funct, lim, testx)
    # for flow data:
    test_flow = host.test_flow(testing, funct, testx)

    # saves outputs to results overview tables for testing set  :
    # for occurrence data:
    occ_test_res.loc[column] = [test_occ['accuracy'],test_occ['precision'],
                                 test_occ['recall'],test_occ['f1score']]
    # and flow data:
    flow_test_res.loc[column] = [test_flow['MWU_stat'],test_flow['MWU_p-val'],
                                 test_flow['chi2_stat'],test_flow['chi2_p-val'],
                                 test_flow['corr'],test_flow['corr_p-val'],
                                 test_flow['MSE'],test_flow['RMSE']]   