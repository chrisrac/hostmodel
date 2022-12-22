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

# data block:
# import data file:
raw_data = pd.read_csv('datasets/example_data.csv')
raw_data.drop('Unnamed: 0', axis=1, inplace=True)

# choose any column from dataframe as example, in this case "flow1":
flow = raw_data.flow1

# HOST usage block:
# use low flow occurrence (categorical) data and generate first 5 harmonics:
occ_harm = host.occ_harmonics(flow, 'lf', beginning='2-1-1979', functions=5)

# use minimal flow (continuous) data and generate first 3 harmonics:
flow_harm = host.flow_harmonics(flow, 'lf', beginning='2-1-1979', functions=3)

# generate best fitted HOST model for low flow occurrence (categorical) data:
occ_bf, occ_pr, occ_pa = host.host_occ(flow, 'lf', beginning='2-1-1979')
# stores: best_fitted function accuracy statistics in occ_bf variable
# stores: predicted values for best_fitted function in occ_pr variable
# stores: best_fitted function parameters in occ_pa variable

# generate best fitted HOST model for minimal flow (continuous) data:
flow_bf, flow_pr, flow_pa = host.host_flow(flow, 'lf', beginning='2-1-1979')
# stores: best_fitted function distribution statistics in flow_bf variable
# stores: predicted values for best_fitted function in flow_pr variable
# stores: best_fitted function parameters in flow_pa variable