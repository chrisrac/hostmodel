"""
This script contains extended description on using HOST Model and accessing 
the results.

Please refer to package repository for citation and updates:
https://github.com/chrisrac/hostmodel

Future functionalities will be added in later development.
Please reffer to documentation for information on current stage of development.

@authors: Krzysztof Raczynski

Table of content:
1. Calculating HOST model for occurrence
    1.1. creating host model object
    1.2. fitting the model
    1.3. accessing best-fit model results
    1.4. accessing specific model
    1.5. accessing model function
    1.6. accessing decomposed data
    1.7. controlling optimization errors
2. Calculating HOST model for flow
    2.1. creating host model object
    2.2. fitting the model
    2.3. accessing best-fit model results
    2.4. changing efficiency measure
"""

# imports
import pandas as pd
import numpy as np
import host

# importing the data. Example dataset can be downloaded from datasets folder 
# in main branch of package, available here: https://github.com/chrisrac/hostmodel
raw_data = pd.read_csv('/datasets/example_data.csv')
# removing the empty column containing artificially generated indexes
raw_data = raw_data.set_index('Unnamed: 0', drop=True)

# assigning data into valiable 'flow' for further usage, in this case seventh column
flow = raw_data['flow7']

#   1. Calculating HOST model for occurrence
#    1.1. creating host model object
#       in order to create HOST model for the data stored in 'flow' variable,
#       a Host object must be created. This object uses provided data 
#       and a set of initial parameters to store its properties, and after
#       the models are fitted and best-fitting one is found, its results.
#       To create Host object, simply call it from host package, with a 
#       set of parameters to be used:
    
#       in general for flow Host model objects are called as follows:
#       var_name = host.Host(input_data, event_type, 'occurrence', data_begin_date, 
#                            input_data_step, output_aggregation_interval, 
#                            train-test_split, threshold_method)
host_model = host.Host(flow, 'lf', 'occurrence', '1-2-1979', 'D', 'M', 0.8, 'median') 
#       as you can see the structude is similar to the Harmonics object, but
#       instead number of harmonics to fit, a train/test split is defined.
#       The train/test split allows to train the model on training data, and then
#       perform validation of the results on test data. In the provided example 
#       80% of data will be used for trainig and 20% for testing. 

#       The defined model properties can be shown by printing the object: 
host_model

#    1.2. fitting the model
#       after Host object is created, in order to fit the model and receive the
#       results, a fit() method must be called. This performs the fitting 
#       instructions:
host_model.fit()
#       depending on the size of the data, this might take a moment.

#    1.3. accessing best-fit model results
#       the Host model results are divided into a few categories and can be 
#       accessed by multiple approaches. 
#       In order to see which model was fitted to data, call .model, which returns
#       string description of the model, as follows: 'tf' - simple trend model,
#       'ts' - sloped trend model, 'sf' - seasonality model, 'cf' - simple combined
#       model, 'cs' - sloped combined model
host_model.model

#       you can return the set of accurracy statistics describing the best-fit
#       model for trainig set by calling .train
host_model.train

#       or you can access specific variable
host_model.train['recall']

#    1.4. accessing specific model
#       currently implemented version of the framework containf five harmonic
#       models that are fitted to the dataset, and a best-fit model is chosen 
#       based on the f1 score value. However you can access each of the models
#       statistics individually by calling theri abbreviation on the object
host_model.ts
#       which returns a set of model parameters describing found function. The
#       r2 value refers to the data component that the specific model is trained
#       on.
#       Additionally each of the parameters might be called individually
host_model.sf.offset

#       Similarly to trainig results, you can view trainig result by calling
host_model.test
#       note that in this example the precision and recall are 0. This is due 
#       to fact that the model predicted occurrence of low flow conditions 14
#       times during testing time, while in reality, no low flow conditions
#       occurred during that time. You should always choose train/test split 
#       carefully to avoid such situations, where the test series lacks any
#       information to compare model to.

#       In order to return a full set of results  describing the model, call 
#       .parameters. Please nope, that the output contains the predicted values
#       of the model as well.
host_model.parameters

#       You can access a specific variable by calling its description. For
#       example, to find model omega, call
host_model.parameters['omega']
#       and to access model predicted values call
host_model.parameters['predictions']

#       When calling predictions values you will note that these are the model
#       values and not model predictions about occurrence or non-occurrence
#       of the studied event. In order to receive the descriptive information
#       the prediction must be compared against decision threshold. You can
#       access this value by calling
host_model.parameters['threshold']

#       keep note that this is the value of threshold in the first day of the 
#       model run, and if the sloped model is the one that is characterized by
#       the best fit, a value of threshold in n-th day must be calculated by
#       including this slope value.
host_model.parameters['threshold']+host_model.parameters['slope']*376

#       since the predictions can be sliced, or specific day can be called,
#       you can compare the predicted value against the threshold to determine
#       the occurrence of event during that day

month_270 = host_model.parameters['predictions'][270]
limit_270 = host_model.parameters['threshold']+host_model.parameters['slope']*270
month_270 >= limit_270
#       indicates occurrence of event in month 270 (aggregation interval was set to 'M')
month_350 = host_model.parameters['predictions'][350]
limit_350 = host_model.parameters['threshold']+host_model.parameters['slope']*350
month_350 >= limit_350
#       indicates no event in month 350

#    1.5. accessing model function
#       By accessing model function object you can create your own object that
#       will allow for generating model output value at any given time step. 
#       You can access model function in two ways, either by calling it from
#       parameters:
host_function = host_model.parameters['function']
print(host_function(450))

#       or by calling .function() method:
host_function = host_model.function()
print(host_function(379))

#    1.6. accessing decomposed data
#       When calling Host model, the input data is decomposed using STL method,
#       and the initial decomposition period is found automatically, by 
#       calculating first harmonic function on data. You can access raw, 
#       decomposed data by simply calling them 
trend = np.array(host_model.trend)
season = np.array(host_model.seasonality)

import matplotlib.pyplot as plt
plt.plot(trend)
plt.plot(season)
plt.show()

#    1.7. controlling optimization errors
#       Sometimes, due to data quality or the problem difficulty, the function
#       won't be optimized with default 10,000 calls. If this happens, the model
#       raises Exception informing about the issue. If the input data is correct,
#       you can modify the number of optimization calls, by changing 'repeats'
#       argument within .fit() method:
host_model.fit(repeats=3)    
#       the above raises exception. Increase number of calls:
host_model.fit(repeats=15000)    


#   2. Calculating HOST model for flow
#    2.1. creating host model object
#       Working on flows instead of occurrences is similar, with a few exceptions.
#       The generalized flow object structure is as follows:
#       var_name = host.Host(input_data, event_type, 'flow', data_begin_date, 
#                            input_data_step, output_aggregation_interval, 
#                            train-test_split)
host_model = host.Host(flow, 'lf', 'flow', '1-2-1979', 'D', 'M', 0.8)

#    2.2. fitting the model
#       Similarly, to fit the model, call .fit() method
host_model.fit()

#       if exception is returned, you can increase number of function calls,
#       same way as for occurrence model:
host_model.fit(repeats=15000)     

#    2.3. accessing best-fit model results
#       Best-fit model information is stored in .model
host_model.model

#       and the parameters can be accessed the same way
host_model.parameters
host_model.parameters.omega

#       similarly, specific models outputs
host_model.ts
host_model.sf.r2

#    2.4. changing efficiency measure
#       The model for flow data provides two options for choosing the best-fit
#       Host model. The selection can be based on Kling-Gupta Efficiency (KGE)
#       measure (by default), or Nash-Sutcliffe Efficiency (NSE). You can 
#       change the method durring fitting stage. If you wish to use NSE, call  
host_model.fit(flow_statistic='nse')
#       if you don't provide any flow_statistic, a default KGE is used.

#       All other methods and parameters access ways are the same as for 
#       occurrence model provided above.
