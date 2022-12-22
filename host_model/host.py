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


# functions block:
def occ_harmonics(data, event, beginning='1-1-1979', functions=6):
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
    
    # process input data:
    processed = preprocessor.data_processor(data,start_date=beginning)
    # generate occurrences:
    event_d, aggr_d = preprocessor.variable_generator(processed, event)
    # calculate consecutive harmonics:
    harmonics = framework.harmonic_occ(aggr_d, no_functions=functions)

    return harmonics


def flow_harmonics(data, event, beginning='1-1-1979', functions=6):
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
    
    # process input data:
    processed = preprocessor.data_processor(data,start_date=beginning)
    # generate flows:
    event_d, aggr_d = preprocessor.variable_generator(processed, event)
    # calculate consecutive harmonics:   
    harmonics = framework.harmonic_flow(aggr_d, no_functions=functions)

    return harmonics


def host_occ(data, event, beginning='1-1-1979'):
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
    
    # process data:
    processed = preprocessor.data_processor(data,start_date=beginning)
    # generate occurrences:
    event_d, aggr_d = preprocessor.variable_generator(processed, event)
    # create five HOST models:
    occ_pred, occ_params = framework.host_body(aggr_d['occurrence'])
    # perform topological analysis for each model:
    topo_ts = framework.topology(occ_pred['ts'], aggr_d['occurrence'])
    topo_tf = framework.topology(occ_pred['tf'], aggr_d['occurrence'])
    topo_sf = framework.topology(occ_pred['sf'], aggr_d['occurrence'])
    topo_cf = framework.topology(occ_pred['cf'], aggr_d['occurrence'])
    topo_cs = framework.topology(occ_pred['cs'], aggr_d['occurrence'])
    # find best fitted model:
    occ_bestfit = framework.selector_cat(topo_tf, topo_ts, topo_sf, topo_cf, topo_cs)

    # create best fitted model output:
    if occ_bestfit['best_type']=='ts':
        params = occ_params['ts']
        preds = occ_pred['ts']
    elif occ_bestfit['best_type']=='tf':
        params = occ_params['tf']
        preds = occ_pred['tf']        
    elif occ_bestfit['best_type']=='sf':
        params = occ_params['sf']
        preds = occ_pred['sf']     
    # if combined model is best fitted, include component models parameters:
    elif occ_bestfit['best_type']=='cf':
        preds = occ_pred['cf'] 
        temp1 = occ_params['sf']
        temp1['amp_sf'] = temp1.pop('amp')
        temp1['omega_sf'] = temp1.pop('omega')
        temp1['phase_sf'] = temp1.pop('phase')
        temp1['offset_sf'] = temp1.pop('offset')
        temp1['freq_sf'] = temp1.pop('freq')
        temp1['period_sf'] = temp1.pop('period')
        temp1['r2_sf'] = temp1.pop('r2')
        temp2 = occ_params['tf']
        temp2['amp_tf'] = temp2.pop('amp')
        temp2['omega_tf'] = temp2.pop('omega')
        temp2['phase_tf'] = temp2.pop('phase')
        temp2['offset_tf'] = temp2.pop('offset')
        temp2['freq_tf'] = temp2.pop('freq')
        temp2['period_tf'] = temp2.pop('period')
        temp2['r2_tf'] = temp2.pop('r2')
        params = {**temp1, **temp2}
    elif occ_bestfit['best_type']=='cs':
        preds = occ_pred['cs'] 
        temp1 = occ_params['sf']
        temp1['amp_sf'] = temp1.pop('amp')
        temp1['omega_sf'] = temp1.pop('omega')
        temp1['phase_sf'] = temp1.pop('phase')
        temp1['offset_sf'] = temp1.pop('offset')
        temp1['freq_sf'] = temp1.pop('freq')
        temp1['period_sf'] = temp1.pop('period')
        temp1['r2_sf'] = temp1.pop('r2')
        temp2 = occ_params['ts']
        temp2['amp_ts'] = temp2.pop('amp')
        temp2['omega_ts'] = temp2.pop('omega')
        temp2['phase_ts'] = temp2.pop('phase')
        temp2['offset_ts'] = temp2.pop('offset')
        temp2['freq_ts'] = temp2.pop('freq')
        temp2['period_ts'] = temp2.pop('period')
        temp2['r2_ts'] = temp2.pop('r2')
        temp2['slope_ts'] = temp2.pop('slope')
        params = {**temp1, **temp2}
        
    return occ_bestfit, preds, params
    

def host_flow(data, event, beginning='1-1-1979'):
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

    # process data:
    processed = preprocessor.data_processor(data,start_date=beginning)
    # generate flows:
    event_d, aggr_d = preprocessor.variable_generator(processed, event)
    # create five HOST models:
    flow_pred, flow_params = framework.host_body(aggr_d['flow'])
    # perform distribution analysis for each model:
    dist_ts = framework.accuracy_cont(flow_pred['ts'], aggr_d['flow'])
    dist_tf = framework.accuracy_cont(flow_pred['tf'], aggr_d['flow'])
    dist_sf = framework.accuracy_cont(flow_pred['sf'], aggr_d['flow'])
    dist_cf = framework.accuracy_cont(flow_pred['cf'], aggr_d['flow'])
    dist_cs = framework.accuracy_cont(flow_pred['cs'], aggr_d['flow'])
    # find best fitted model:
    flow_bestfit = framework.selector_cont(dist_tf, dist_ts, dist_sf, dist_cf, dist_cs)
    
    # create best fitted model output:
    if flow_bestfit['best_type']=='ts':
        params = flow_params['ts']
        preds = flow_pred['ts']
    elif flow_bestfit['best_type']=='tf':
        params = flow_params['tf']
        preds = flow_pred['tf']        
    elif flow_bestfit['best_type']=='sf':
        params = flow_params['sf']
        preds = flow_pred['sf'] 
    # if combined model is best fitted, include component models parameters:
    elif flow_bestfit['best_type']=='cf':
        preds = flow_pred['cf'] 
        temp1 = flow_params['sf']
        temp1['amp_sf'] = temp1.pop('amp')
        temp1['omega_sf'] = temp1.pop('omega')
        temp1['phase_sf'] = temp1.pop('phase')
        temp1['offset_sf'] = temp1.pop('offset')
        temp1['freq_sf'] = temp1.pop('freq')
        temp1['period_sf'] = temp1.pop('period')
        temp1['r2_sf'] = temp1.pop('r2')
        temp2 = flow_params['tf']
        temp2['amp_tf'] = temp2.pop('amp')
        temp2['omega_tf'] = temp2.pop('omega')
        temp2['phase_tf'] = temp2.pop('phase')
        temp2['offset_tf'] = temp2.pop('offset')
        temp2['freq_tf'] = temp2.pop('freq')
        temp2['period_tf'] = temp2.pop('period')
        temp2['r2_tf'] = temp2.pop('r2')
        params = {**temp1, **temp2}
    elif flow_bestfit['best_type']=='cs':
        preds = flow_pred['cs'] 
        temp1 = flow_params['sf']
        temp1['amp_sf'] = temp1.pop('amp')
        temp1['omega_sf'] = temp1.pop('omega')
        temp1['phase_sf'] = temp1.pop('phase')
        temp1['offset_sf'] = temp1.pop('offset')
        temp1['freq_sf'] = temp1.pop('freq')
        temp1['period_sf'] = temp1.pop('period')
        temp1['r2_sf'] = temp1.pop('r2')
        temp2 = flow_params['ts']
        temp2['amp_ts'] = temp2.pop('amp')
        temp2['omega_ts'] = temp2.pop('omega')
        temp2['phase_ts'] = temp2.pop('phase')
        temp2['offset_ts'] = temp2.pop('offset')
        temp2['freq_ts'] = temp2.pop('freq')
        temp2['period_ts'] = temp2.pop('period')
        temp2['r2_ts'] = temp2.pop('r2')
        temp2['slope_ts'] = temp2.pop('slope')
        params = {**temp1, **temp2}
        
    return flow_bestfit, preds, params