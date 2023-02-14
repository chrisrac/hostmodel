# -*- coding: utf-8 -*-
"""
This is Harmonic Oscillator Seasonal Trend (HOST) Model framework module  
for hydrological extremes pattern identification and analysis.

Please reffer to package repository for citation and updates:
https://github.com/chrisrac/hostmodel

Future functionalities will be added in later development.
Please reffer to documentation for information on current stage of development.

@authors: Krzysztof Raczynski
"""

# imports block:
# import HOST model framework modules:
import framework
import preprocessor
# suppress all warnings 
import warnings
warnings.filterwarnings("ignore")
import numpy as np

# functions block:
def occ_harmonics(data, functions=6):
    '''
    Occurrence harmonics analysis. Performs data processing starting from 
    'beginning' date provided. Uses objective_thresholds for event threshold
    calculation. Determines and aggregates data in provided timestep. Generates
    determined number of harmonic functions.

    Parameters
    ----------
    data : array-like
        The Iterable sequence of numbers (int/float) representing flow data 
        to be used, f.e.: list, pd.Series, np.array or pd.DataFrame slice.
    event : string
        String representing type of studied event. Available options are:
            "lf" or "lowflow" : for streamflow drought/low flow analysis;
            "fd" or "flood"   : for flood analysis.
    beginning : date, optional
        Date for timeseries to strart from. The default is '1-1-1979'.
    functions : int, optional
        Number of harmonics to extract from data. Each harmonic is extracted 
        from data before next is fitted. Optional, default value is 6.

    Returns
    -------
    harmonics : pandas DataFrame
        The DataFrame object in form of summary table containing all function
        parameters, r2 value and predicted values.
    '''
    

    # calculate consecutive harmonics:
    harmonics = framework.harmonic_occ(data, no_functions=functions)

    return harmonics


def flow_harmonics(data, functions=6):
    '''
    Flow harmonics analysis. Performs data processing starting from 
    'beginning' date provided. Determines and aggregates data in provided 
    timestep. Generates determined number of harmonic functions.

    Parameters
    ----------
    data : array-like
        The Iterable sequence of numbers (int/float) representing flow data 
        to be used, f.e.: list, pd.Series, np.array or pd.DataFrame slice.
    event : string
        String representing type of studied event. Available options are:
            "lf" or "lowflow" : for minimal flow analysis;
            "fd" or "flood"   : for maximal flow analysis.
    beginning : date, optional
        Date for timeseries to strart from. The default is '1-1-1979'.
    functions : int, optional
        Number of harmonics to extract from data. Each harmonic is extracted 
        from data before next is fitted. Optional, default value is 6.

    Returns
    -------
    harmonics : pandas DataFrame
        The DataFrame object in form of summary table containing all function
        parameters, r2 value and predicted values.
    '''
    
    # calculate consecutive harmonics:   
    harmonics = framework.harmonic_flow(data, no_functions=functions)

    return harmonics

            
def train_occurrence(data):
    '''
    HOST model calling function for occurrence type analysis. Performs series 
    of transformations to generate five harmonic models: tf (trend), 
    ts (trend sloped), sf (seasonal), cf (combined), cs (combined sloped). 
    Returns best fitted model parameters, statistics and predicted values. 
    If combined model is best fitted, components parameters are returned.

    Parameters
    ----------
    data : array-like
        The iterable sequence of numbers (int/float) representing flow data 
        to be used, f.e.: list, pd.Series, np.array or pd.DataFrame slice.
    event : string
        String representing type of studied event. Available options are:
            "lf" or "lowflow" : for streamflow drought/low flow analysis;
            "fd" or "flood"   : for flood analysis.
    beginning : date, optional
        Date for timeseries to strart from. The default is '1-1-1979'.

    Returns
    -------
    occ_bestfit : dict
        Dictionary of best fitted model accuracy statistics.
    preds : array
        The iterable sequence of numbers (int/float) representing predicted 
        values of best fitted model: tf - trend, ts - trend sloped, 
        sf - seasonal, cf - combined, cs - combined sloped.
    params : dict
        Dictionary containing all parameters of best fitted model (amplitude, omega,
        phase, offset, frequency, period, r2, y_pred): tf - trend, ts - trend sloped,
        sf - seasonal, cf - combined, cs - combined sloped. 
    '''
    
    # create five HOST models:
    occ_pred, occ_params = framework.host_body(data['occurrence'])
    # perform topological analysis for each model:
    topo_ts = framework.topology(occ_pred['ts'], data['occurrence'])
    topo_tf = framework.topology(occ_pred['tf'], data['occurrence'])
    topo_sf = framework.topology(occ_pred['sf'], data['occurrence'])
    topo_cf = framework.topology(occ_pred['cf'], data['occurrence'])
    topo_cs = framework.topology(occ_pred['cs'], data['occurrence'])
    # find best fitted model:
    occ_bestfit = framework.selector_cat(topo_tf, topo_ts, topo_sf, topo_cf, topo_cs)

    
    # create best fitted model output:
    if occ_bestfit['best_type']=='ts':
        params = occ_params['ts']
        preds = occ_pred['ts']
        function, limits = framework.function_return('ts', 
                                                     occ_bestfit['threshold'],
                                                     True,
                                                     params['slope'], 
                                                     params['amp'], 
                                                     params['omega'], 
                                                     params['phase'], 
                                                     params['offset'])
        
    elif occ_bestfit['best_type']=='tf':
        params = occ_params['tf']
        preds = occ_pred['tf']   
        function, limits = framework.function_return('tf', 
                                                     occ_bestfit['threshold'],
                                                     True,
                                                     params['amp'], 
                                                     params['omega'], 
                                                     params['phase'], 
                                                     params['offset'])
        
    elif occ_bestfit['best_type']=='sf':
        params = occ_params['sf']
        preds = occ_pred['sf']   
        function, limits = framework.function_return('sf', 
                                                     occ_bestfit['threshold'],
                                                     True,
                                                     params['amp'], 
                                                     params['omega'], 
                                                     params['phase'], 
                                                     params['offset'])
        
    # if combined model is best fitted, include component models parameters:
    elif occ_bestfit['best_type']=='cf':
        preds = occ_pred['cf'] 
        temp1 = occ_params['sf']
        postfix = '_sf'
        temp1 = {str(key) + postfix: val for key, val in temp1.items()}
        temp2 = occ_params['tf']
        postfix = '_tf'
        temp2 = {str(key) + postfix: val for key, val in temp2.items()}
        params = {**temp1, **temp2}
        function, limits = framework.function_return('cf', 
                                                     occ_bestfit['threshold'], 
                                                     True,
                                                     params['amp_tf'], 
                                                     params['omega_tf'],
                                                     params['phase_tf'], 
                                                     params['offset_tf'], 
                                                     params['amp_sf'], 
                                                     params['omega_sf'], 
                                                     params['phase_sf'], 
                                                     params['offset_sf'])
    elif occ_bestfit['best_type']=='cs':
        preds = occ_pred['cs'] 
        temp1 = occ_params['sf']
        postfix = '_sf'
        temp1 = {str(key) + postfix: val for key, val in temp1.items()}
        temp2 = occ_params['ts']
        postfix = '_ts'
        temp2 = {str(key) + postfix: val for key, val in temp2.items()}
        params = {**temp1, **temp2}
        function, limits = framework.function_return('cs', 
                                                     occ_bestfit['threshold'], 
                                                     True,
                                                     params['slope_ts'], 
                                                     params['amp_ts'], 
                                                     params['omega_ts'], 
                                                     params['phase_ts'], 
                                                     params['offset_ts'], 
                                                     params['amp_sf'], 
                                                     params['omega_sf'], 
                                                     params['phase_sf'], 
                                                     params['offset_sf'])
    
    return occ_bestfit, preds, params, function, limits


def test_occurrence(data, function, limit, test_x):
    '''
    Function used for testing model performance on test dataset. Requires as
    input test dataset, function used to generate model values, decision 
    threshold and x values for test set time-range. Can be used for categorical
    data only.

    Parameters
    ----------
    data : array-like
        Test dataset prepared by preprocessor.preprocess() function.
    function : function object
        Function object used to generate model values in any time-range with
        parameters preprogrammed, returned by train_occurrence function.
    limit : array-like
        Values array representing decision threshold at each given time step.
        Returned by train_occurrence function.
    test_x : array-like
        Values array representing x time-steps for test dataset. Returned by
        preprocessor.preprocess() function.

    Returns
    -------
    dict
        Dictionary containing accurracy statistics for test dataset: accuracy,
        precision, recall, f1score.
    '''
    
    y_pred = []
    for x in test_x:
        y_pred.append(function(x))
    limits = []
    for x in test_x:
        limits.append(limit(x))
    data['comp'] = 0
    data['comp'].loc[data['occurrence'] >= 1] = 1
    data['data'] = y_pred
    data['limit'] = limits
    data['predict'] = data.apply(framework.event_predict, axis=1)
    data['cat'] = data.apply(framework.event_cat, axis=1)
    
    cont, acc, precision, recall, f1score = framework.contingency_tab(data)
    
    return {'accuracy' : acc, 'precision' : precision, 
            'recall' : recall, 'f1score' : f1score}



def train_flow(data):
    '''
    HOST model calling function for flow type analysis. Performs series 
    of transformations to generate five harmonic models: tf (trend), 
    ts (trend sloped), sf (seasonal), cf (combined), cs (combined sloped). 
    Returns best fitted model parameters, statistics and predicted values. 
    If combined model is best fitted, components parameters are returned.

    Parameters
    ----------
    data : array-like
        The iterable sequence of numbers (int/float) representing flow data 
        to be used, f.e.: list, pd.Series, np.array or pd.DataFrame slice.
    event : string
        String representing type of studied event. Available options are:
            "lf" or "lowflow" : for minimal flow analysis;
            "fd" or "flood"   : for maximal flow analysis.
    beginning : date, optional
        Date for timeseries to strart from. The default is '1-1-1979'.

    Returns
    -------
    flow_bestfit : dict
        Dictionary of best fitted model distribution statistics.
    preds : array
        The iterable sequence of numbers (int/float) representing predicted 
        values of best fitted model: tf - trend, ts - trend sloped, 
        sf - seasonal, cf - combined, cs - combined sloped.
    params : dict
        Dictionary containing all parameters of best fitted model (amplitude, omega,
        phase, offset, frequency, period, r2, y_pred): tf - trend, ts - trend sloped,
        sf - seasonal, cf - combined, cs - combined sloped. 
    '''

    # create five HOST models:
    flow_pred, flow_params = framework.host_body(data['flow'])
    # perform distribution analysis for each model:
    dist_ts = framework.accuracy_cont(data['flow'], flow_pred['ts'])
    dist_tf = framework.accuracy_cont(data['flow'], flow_pred['tf'])
    dist_sf = framework.accuracy_cont(data['flow'], flow_pred['sf'])
    dist_cf = framework.accuracy_cont(data['flow'], flow_pred['cf'])
    dist_cs = framework.accuracy_cont(data['flow'], flow_pred['cs'])
    # find best fitted model:
    flow_bestfit = framework.selector_cont(dist_tf, dist_ts, dist_sf, dist_cf, dist_cs)
    
    # create best fitted model output:
    if flow_bestfit['best_type']=='ts':
        params = flow_params['ts']
        preds = flow_pred['ts']
        function = framework.function_return('ts', 
                                             0,
                                             False,
                                             params['slope'], 
                                             params['amp'], 
                                             params['omega'], 
                                             params['phase'], 
                                             params['offset'])
        
    elif flow_bestfit['best_type']=='tf':
        params = flow_params['tf']
        preds = flow_pred['tf']  
        function = framework.function_return('tf', 
                                             0,
                                             False,
                                             params['amp'], 
                                             params['omega'], 
                                             params['phase'], 
                                             params['offset'])
        
    elif flow_bestfit['best_type']=='sf':
        params = flow_params['sf']
        preds = flow_pred['sf'] 
        function = framework.function_return('sf', 
                                             0,
                                             False,
                                             params['amp'], 
                                             params['omega'], 
                                             params['phase'], 
                                             params['offset'])        
        
    # if combined model is best fitted, include component models parameters:
    elif flow_bestfit['best_type']=='cf':
        preds = flow_pred['cf'] 
        temp1 = flow_params['sf']
        postfix = '_sf'
        temp1 = {str(key) + postfix: val for key, val in temp1.items()}
        temp2 = flow_params['tf']
        postfix = '_tf'
        temp2 = {str(key) + postfix: val for key, val in temp2.items()}
        params = {**temp1, **temp2}
        function = framework.function_return('cf', 
                                             0, 
                                             False,
                                             params['amp_tf'], 
                                             params['omega_tf'],
                                             params['phase_tf'], 
                                             params['offset_tf'], 
                                             params['amp_sf'], 
                                             params['omega_sf'], 
                                             params['phase_sf'], 
                                             params['offset_sf'])        
        
    elif flow_bestfit['best_type']=='cs':
        preds = flow_pred['cs'] 
        temp1 = flow_params['sf']
        postfix = '_sf'
        temp1 = {str(key) + postfix: val for key, val in temp1.items()}
        temp2 = flow_params['ts']
        postfix = '_ts'
        temp2 = {str(key) + postfix: val for key, val in temp2.items()}
        params = {**temp1, **temp2}
        function = framework.function_return('cs', 
                                             0, 
                                             False,
                                             params['slope_ts'], 
                                             params['amp_ts'], 
                                             params['omega_ts'], 
                                             params['phase_ts'], 
                                             params['offset_ts'], 
                                             params['amp_sf'], 
                                             params['omega_sf'], 
                                             params['phase_sf'], 
                                             params['offset_sf'])
        
    return flow_bestfit, preds, params, function


def test_flow(data, function, test_x):
    '''
    Function used for testing model performance on test dataset. Requires as
    input test dataset, function used to generate model values, and x values 
    for test set time-range. Can be used for continuous data only.

    Parameters
    ----------
    data : array-like
        Test dataset prepared by preprocessor.preprocess() function.
    function : function object
        Function object used to generate model values in any time-range with
        parameters preprogrammed, returned by train_occurrence function.
    test_x : array-like
        Values array representing x time-steps for test dataset. Returned by
        preprocessor.preprocess() function.

    Returns
    -------
    dict
        Dictionary containing accurracy statistics for test dataset: accuracy,
        precision, recall, f1score.
    '''
    
    y_pred = []
    for x in test_x:
        y_pred.append(function(x))
    data['predicted']=y_pred
    results = framework.accuracy_cont(data['flow'], data['predicted'])
    
    return results
    