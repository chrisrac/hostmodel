"""
This script contains extended description on using HOST Model and accessing 
the results.

Please refer to package repository for citation and updates:
https://github.com/chrisrac/hostmodel

Future functionalities will be added in later development.
Please reffer to documentation for information on current stage of development.

@authors: Krzysztof Raczynski

Table of content:
1. Calculating raw harmonic functions for flow data
    1.1. creating harmonic object    
    1.2. fitting the model
    1.3. accessing full results
    1.4. accessing single harmonic results
    1.5. accessing values of harmonic model
2. Calculating raw harmonic functions of occurrence
    2.1. creating harmonic object  
    2.2. fitting the model
    2.3. accessing full results
    2.4. accessing single harmonic results
    2.5. accessing values of harmonic model
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

# assigning first data column into valiable 'flow' for further usage
flow = raw_data['flow1']

#   1. Calculating raw harmonic functions for flow data
#    1.1. creating harmonic object
#       in order to find n harmonics on the data stored in 'flow' variable,
#       a Harmonics object must be created. This object uses provided data 
#       and a set of initial parameters to store its properties, and after
#       the harmonic functions are fitted, its results.
#       To create Harmonics object, simply call it from host package, with a 
#       set of parameters to be used:
    
#       in general for flow harmonics the objects are called as follows:
#       var_name = host.Harmonics(input_data, event_type, number_of_harmonics,
#                                 'flow', data_begin_date, input_data_step,
#                                 output_aggregation_interval)    
harmonic_model = host.Harmonics(flow, 'lf', 5, 'flow', '1-2-1979', 'D', 'M')

#       in the above example, 'flow' variable values will be used as input data,
#       the 'lf' stand for 'low flow' and indicates, the analysis is performed
#       for streamflow droughts. Number 5 indicates, that 5 consecutive 
#       harmonic functions will be calculated from data. 'flow' determines,
#       that minimal flows will be used in the analysis. The provided date
#       indicates the start time for the initial data, which is 1st February 1979.
#       String 'D' indicates that data is provided in daily format, and output
#       from model should be generated for monthly 'M' aggregated data.

#       The defined model properties can be shown by printing the object: 
harmonic_model

#    1.2. fitting the model
#       after harmonic object is created, in order to receive the defined number
#       of harmonics, a fit() method must be called. This performs the fitting 
#       instructions:
harmonic_model.fit()

#    1.3. accessing full results
#       results of the harmonic functions fitted to data can be accessed in a 
#       few ways. The simples is calling the .results on the object, which will 
#       return a summary table of the results for all fitted harmonics:
harmonic_model.results 
    
#    1.4. accessing single harmonic results
#       the results however can be also returned for specific harmonic calculated.
#       For example in order to access only the results for second calculated
#       function, call:
harmonic_model.results['harmonic: 2']

#       and for fourth:
harmonic_model.results['harmonic: 4']

#       specific value of the results can be accessed by calling it on the
#       desired object:
harmonic_model.results['harmonic: 2'].offset

#       the results are returned in for of pandas DataFrame, therefore slicing
#       the table works same as for pandas Frames.  For example to return r2
#       values of all the functions, simply call:
harmonic_model.results.loc['r2']    

#    1.5. accessing values of harmonic model
#       to access values of harmonics calculated call values() method on Harmonics
#       object:
harmonic_model.values() 

#       by default however the values are not returned to save memory and instead
#       SyntaxError is returned with instructions. 
#       If you wish, that your harmonics have their predictions included, set
#       include_preds argument to True, when fitting the model: 
#       .fit(include_preds=True) or simply .fit(True)
harmonic_model.fit(True)

#       now you can access the values the functions values:
harmonic_model.values()

#       you can also access specific function values, by calling 
#       .value[which_harmonic]:
harmonic_model.value['harmonic: 3']   

#       or to access a specific value of all functions
harmonic_model.value.loc[349]  
    

#   2. Calculating raw harmonic functions of occurrence
#    2.1. creating harmonic object
#       using occurrence classification instead of flows have the same workflow
#       and all the differences are calculated during .fit() stage, however
#       in order to correctly assign the 'occurrence' type, it have to be 
#       stated during object creation stage. The structure is similar, with
#       one additional argument, controlling the threshold behaviour, if the
#       data is highly discretized:
    
#       for occurrence harmonics the objects are called as follows:
#       var_name = host.Harmonics(input_data, event_type, number_of_harmonics,
#                                 'occurrence', data_begin_date, input_data_step,
#                                 output_aggregation_interval, threshold_behavior)
harmonic_model = host.Harmonics(flow, 'lf', 6, 'occurrence', '1-2-1979', 'D', 'M', 'median')  
#       in the above example, 'occurrence' variable values will be used as input data,
#       the 'lf' stand for 'low flow' and indicates, the analysis is performed
#       for streamflow droughts. Number 6 indicates, that 6 consecutive 
#       harmonic functions will be calculated from data. 'occurrence' determines,
#       that event occurrence fact will be used in the analysis. The provided date
#       indicates the start time for the initial data, which is 1st February 1979.
#       String 'D' indicates that data is provided in daily format, and output
#       from model should be generated for monthly 'M' aggregated data. The 'median'
#       statement controls the behavior of objective_threshold package in case
#       data have multiple same values, that affect the breakpoint analysis.
#       Please refer to objective_thresholds documentation for more information:
#       https://github.com/chrisrac/objective_thresholds
#       Keep in mind that current 'occurrence' implementation includes limitations
#       that are further discussed in the documentation.

#       The defined model properties can be shown by printing the object: 
harmonic_model

#    1.2. fitting the model
#       after harmonic object is created, in order to receive the defined number
#       of harmonics, a fit() method must be called. This performs the fitting 
#       instructions:
harmonic_model.fit()

#    1.3. accessing full results
#       results of the harmonic functions fitted to data can be accessed in a 
#       few ways. The simples is calling the .results on the object, which will 
#       return a summary table of the results for all fitted harmonics:
harmonic_model.results 
    
#    1.4. accessing single harmonic results
#       the results however can be also returned for specific harmonic calculated.
#       For example in order to access only the results for second calculated
#       function, call:
harmonic_model.results['harmonic: 2']

#       and for fourth:
harmonic_model.results['harmonic: 4']

#       specific value of the results can be accessed by calling it on the
#       desired object:
harmonic_model.results['harmonic: 2'].amp

#       the results are returned in for of pandas DataFrame, therefore slicing
#       the table works same as for pandas Frames.  For example to return r2
#       values of all the functions, simply call:
harmonic_model.results.loc['r2']    

#    1.5. accessing values of harmonic model
#       to access values of harmonics calculated call values() method on Harmonics
#       object:
harmonic_model.values() 

#       by default however the values are not returned to save memory and instead
#       SyntaxError is returned with instructions. 
#       If you wish, that your harmonics have their predictions included, set
#       include_preds argument to True, when fitting the model: 
#       .fit(include_preds=True) or simply .fit(True)
harmonic_model.fit(True)

#       now you can access the values the functions values:
harmonic_model.values()

#       you can also access specific function values, by calling 
#       .value[which_harmonic]:
harmonic_model.value['harmonic: 3']   

#       or to access a specific value of all functions
harmonic_model.value.loc[499] 

#       please note that the harmonic function values for the occurrence model 
#       does not include the comparison against decision threshold. In order to
#       receive full model with the capabilities for interpretation of the
#       values scope refer to guide_host file.