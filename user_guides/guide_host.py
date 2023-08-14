"""
This script contains extended description on using HOST Model and accessing 
the results.

Please refer to package repository for citation and updates:
https://github.com/chrisrac/hostmodel
v1.0

@authors: Krzysztof Raczynski

Table of content:
1. Calculating HOST model for occurrence
    1.1. creating host model object
    1.2. fitting the model
    1.3. accessing model results
    1.4. accessing model equation
    1.5. accessing model function
    1.6. accessing decomposed data
    1.7. controlling optimization errors
2. Calculating HOST model for flow
    2.1. creating host model object
    2.2. fitting the model
    2.3. accessing model results
    2.4. changing efficiency measure
3. Calculating HOST model for magnitude
4. Additional object parameters (version v1.0+)
5. Fitting modifications (version v1.0+)
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
    
#       in general for occurrence (binary classification) objects are called as follows:
#       var_name = host.Host(input_data, event_type, 'occurrence', data_begin_date, 
#                            input_data_step, output_aggregation_interval, 
#                            train-test_split, threshold_method)
host_model = host.Host(flow, 'low', 'occurrence', '1-2-1979', 'D', 'M', 0.8, 'median') 
#       as you can see the structude is similar to the Harmonics object, but
#       instead number of harmonics to fit, a train/test split is defined.
#       The train/test split allows to train the model on training data, and then
#       perform validation of the results on test data. In the provided example 
#       80% of data will be used for trainig and 20% for testing.  

#       The defined model properties can be shown by printing the object: 
host_model

#    1.2. fitting the model
#       after Host object is created, in order to fit the model and receive the
#       results, a .fit() method must be called. This performs the fitting 
#       instructions:
host_model.fit()
#       depending on the size of the data, this might take a moment.

#    1.3. accessing model results
#       the Host model results are divided into a few categories and can be 
#       accessed by multiple approaches. 
#       In order to see model fitted to data, call .model, which returns
#       the information on the resultant (combined) model
host_model.model

#       you can return the set of statistics describing the model 
#       for trainig set by calling .train
host_model.train

#       or you can access specific variable
host_model.train['r2']

#    1.4. accessing model equation
#       once the model is built, aside from a set of its parameters, 
#       a mathematical formula representing the model can be displayed  
#       by calling .equation
host_model.equation

#       in order to access long-term or short-term models separatelly, 
#       they can be called individually:
host_model.trendmodel
host_model.seasonalmodel
#       which can be further explored with their fields: .parameters, .predictions,
#       .function, .equation. 
#       Additionally .models contains information on each of the models fitted,
#       before best-fit model was selected.

#       Similarly to trainig results, you can view testing result by calling
host_model.test
#       note that in some cases the precision and recall can be 0. In this case 
#       this is due to fact that the model predicted occurrence of event 
#       conditions 14 times during testing time, while in reality, no event
#       occurred during that time. You should always choose train/test split 
#       carefully to avoid such situations, where the test series lacks any
#       information to compare model to.

#       to access model predicted values call (on set you want to use)
host_model.train['predictions']

#       When calling predictions you will note that these are the model
#       values and not model predictions about occurrence or non-occurrence
#       of the studied event. In order to receive the descriptive information
#       the prediction must be compared against decision threshold. You can
#       access this value by calling
host_model.train['threshold']

#       keep note that this is the value of threshold in the first day of the 
#       model run, and if the sloped model is the one that is characterized by
#       the best fit, a value of threshold in n-th day must be calculated by
#       including this slope value.
host_model.train['threshold']+host_model.train['slope']*376

#       since the predictions can be sliced, or specific day can be called,
#       you can compare the predicted value against the threshold to determine
#       the occurrence of event during that day

month_270 = host_model.train['predictions'][270]
limit_270 = host_model.train['threshold']+host_model.train['slope']*270
month_270 >= limit_270
#       indicates occurrence of event in month 270 (aggregation interval was set to 'M')
month_350 = host_model.train['predictions'][350]
limit_350 = host_model.train['threshold']+host_model.train['slope']*350
month_350 >= limit_350
#       indicates no event in month 350

#    1.5. accessing model function
#       By accessing model function object you can create your own object that
#       will allow for generating model output value at any given time step. 
#       You can access model function by calling:
host_function = host_model.function
print(host_function(379))

#    1.6. accessing decomposed data
#       When calling Host model, the input data is decomposed using STL method,
#       and the initial decomposition period is found automatically, by 
#       calculating first harmonic function on data. You can access raw, 
#       decomposed data by simply calling them 
cleandata = np.array(host_model.raw_data)
trend = np.array(host_model.trend_data)
season = np.array(host_model.seasonal_data)

#       plotting results:
import matplotlib.pyplot as plt
plt.plot(cleandata)
plt.plot(trend)
plt.plot(season)
plt.show()

#    1.7. controlling optimization errors
#       Sometimes, due to data quality or the problem difficulty, the function
#       won't be optimized with default calls. If this happens, the model
#       raises Exception informing about the issue. If the input data is correct,
#       you can modify the number of optimization calls, by changing 'repeats'
#       argument within .fit() method:
host_model.fit(repeats=3)    
#       the above raises exception. Increase number of calls:
host_model.fit(repeats=1500000)    


#   2. Calculating HOST model for flow
#    2.1. creating host model object
#       Working on direct data instead of occurrences is similar, with a few exceptions.
#       The generalized signal object structure is as follows:
#       var_name = host.Host(input_data, event_type, 'signal', data_begin_date, 
#                            input_data_step, output_aggregation_interval, 
#                            train-test_split)
host_model = host.Host(flow, 'low', 'signal', '1-2-1979', 'D', 'M', 0.8)

#    2.2. fitting the model
#       Similarly, to fit the model, call .fit() method
host_model.fit()

#       if exception is returned, you can increase number of function calls,
#       same way as for occurrence model:
host_model.fit(repeats=1500000)     

#    2.3. accessing model results
#       the final model information is stored in .model
host_model.model

#       and the parameters can be accessed the same way
host_model.train
host_model.test

#       similarly, specific models outputs
host_model.test.r2

#    2.4. changing efficiency measure
#       The model for signal data provides two options for efficiency metric. 
#       The selection can be made between Kling-Gupta Efficiency (KGE)
#       measure (by default), or Nash-Sutcliffe Efficiency (NSE). You can 
#       change the method durring fitting stage. If you wish to use NSE, call:  
host_model.fit(efficiency='nse')
#       if you don't provide any metric, a default KGE is used.

#       All other methods and parameters access ways are the same as for 
#       occurrence model provided above.


#   3. Calculating HOST model for flow
#       Creating a magnitude based model is similar.
#       The generalized magnitude object structure is as follows:
#       var_name = host.Host(input_data, event_type, 'magnitude', data_begin_date, 
#                            input_data_step, output_aggregation_interval, 
#                            train-test_split)
host_model = host.Host(flow, 'low', 'magnitude', '1-2-1979', 'D', 'M', 0.8)
#       fitting the model and accessing results is performed the same way.


#   4. Additional object parameters (version v1.0+)
#       in release v1.0 three new attributes were included: 
#       threshold_overwrite, signal_method and area. These allow for additional control
#       over the experiment. 'threshold_overwrite' argument allows to overwrite the default
#       objective_threshold method and use user provided threshold. 'signal_method' controls
#       how the aggregation for signal analysis is performed, with available steps as:
#       mean, sum, min and max. Finally "area" argument allows to scale data based on the value
#       provided here. In this case values like magnitude will be devided by this factor 
#       allowing the comparison between data comming from different sources, by scaling them to
#       unitary values (assuming correct divider is provided).
#       Example below presents using Numpy to calculate 10th percentile and using it as threshold,
#       with aggregation to monthly totals, and use of division factor of 1568:
import numpy as np
harmonic_model = host.Harmonics(flow, 'low', 5, 'signal', '1-1-1979', 'D', 'M', 
                                'median', threshold_overwrite=np.quantile(flow, 0.1),
                                signal_method='sum', area=1568)


#   5. Fitting modifications (version v1.0+)
#       in release v1.0 new modification for .fit() method were implemented for HOST models.
#       These include following, optional attributes passed to .fit() method:
#       multiplier, include_damped, decision_statistic, binary_occurrence.
#       The order and default values are as follows:
.fit(self, repeats=1000000, multiplier=1, include_damped=True, decision_statistic='r2', efficiency='kge', binary_occurrence=True)

#       Modyfing multiplier value changes the period applied for STL decomposition. 
#       F.e. if the found period is 5, and the multiplier=3 is passed, the STL
#       decomposition period will be 15. This affects the periods of decomposed
#       components used for modeling and can improve results.
#       The example of automatic best-fit multiplier value (betwee 1 and 10) finder is presented below:
for i in [1,2,3,4,5,6,7,8,9,10]:
    occ_model.fit(multiplier=i)  
    temp_mag[i] = occ_model.test['accuracy']
    temp_res[i] = occ_model
occ_model = temp_res[max(temp_mag, key=temp_mag.get)]

#       If damped models are undesired, user can turn off fitting of all damped models
#       by setting include_damped argument:
occ_model.fit(include_damped=False)
#       this will include only non-damped models in the fitting process.

#       Additionally, decision_statistic arguments allows for choosing, what parameter
#       is used to choose best-fit model: r2 (by default), or chosen efficiency (regulated
#       with efficiency argument).

#       Lastly binary_occurrence allows for changing the binary classified occurrence model
#       to weighted occurrence model, when set to False. This modifies found function distribution,
#       by time-relevance factor inclusion.

