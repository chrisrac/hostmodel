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
# import modules
import pandas as pd


# functions block:
def data_processor(data, start_date='1-1-1979', interval = 'D'):
    '''
    This function prepares data accordingly for further processing. Required 
    inputs are data in series like format and optionally start date for 
    time series mark. 
    Returns Pandas Series object.
    
    
    Parameters
    ----------
    data : array-like
        The Iterable sequence of numbers (int/float) to be used, f.e.: list,
        pd.Series, np.array or pd.DataFrame slice.
    start_date : date, optional
        Date for timeseries to strart from. The default is '1-1-1979'.
    interval : string: 'D', 'M' or 'Y', optional
        The interval of data. Use 'D' for daily, 'M' for monthly and
        'Y' for annual (year). Default is 'D'.

    Returns
    -------
    indexed_data : pandas Series
        Time indexed pandas Series object.
    '''
    
    # generate datetime index:
    indexes = pd.date_range(start_date, periods=len(data), freq=interval)
    # change data to pandas DataFrame (for future functionalities):
    indexed_data = data.to_frame()
    # assign datetime index and clean:
    indexed_data['indexes'] = indexes
    indexed_data = indexed_data.set_index('indexes',drop=True)
    # revert format change:
    indexed_data = indexed_data.squeeze()
    
    return indexed_data


def variable_generator(data, event_type, interval='M', threshold_method='leave'):
    '''
    Function used to generate a representation of the event parameter in form
    of variable array-like object and data aggregated in research scale 
    (monthly, annual). Uses objective threshold for TLM identification method. 
    Required inputs are data (array-like) and event type. Optional 
    parameter controlls aggregation period and the objective threshold method feedback.

    Parameters
    ----------
    data : array-like
        The Iterable sequence of numbers (int/float) to be used, f.e.: list,
        pd.Series, np.array or pd.DataFrame slice.
    event_type : string
        String representing type of studied event. 
        Available options are:
            "lf" or "lowflow" : for streamflow drought/low flow analysis;
            "fd" or "flood"   : for flood analysis.
    interval : string: 'M' or 'Y', optional
        The interval of data for aggregation. Use 'M' for monthly and
        'Y' for annual (year) aggregation. Default is 'M'.
    threshold_method : string, optional
        Used to controll objective method feedback on high discretized data, 
        that results in less than five unique values in the series. 
        Available options are:
            "leave"  : breaks the computation, no threshold is returned, default;
            "min"    : minimal value of data is used as threshold;
            "median" : data median is used as threshold.

    Returns
    -------
    event_data : pandas DataFrame
        Time indexed pandas DataFrame object with event oocurrence identified.
    aggreg_data : pandas DataFrame
        Time indexed pandas DataFrame object with event occurrence identified
        and data aggregated in provided interval.
    '''
    
    # decide aggregation frame based on provided interval:
    if interval == 'M':
        step = ['year', 'month']
    elif interval == 'Y':
        step = ['year']   
    
    # low flow / streamflow analysis:
    if event_type == 'lf' or event_type == 'lowflow':
        # import objective_thresholds low flow analysis module:
        import objective_thresholds.lowflow as olf
        # calculate threshold:
        threshold_value = olf.threshold(data, method=threshold_method)
        # convert data to frame:
        event_data = data.to_frame()
        event_data.columns = ['flow']
        # detect periods of drought occurrence based on calculated threshold:
        event_data.loc[event_data.flow <= threshold_value, 'occurrence'] = 1
        # prepare aggregation intervals:
        event_data['month'] = event_data.index.month
        event_data['year'] = event_data.index.year
        # aggregate occurrence data by sum for occurrence:        
        aggreg_data = event_data.groupby(step)['occurrence'].sum().to_frame()
        # aggregate flow data by minimal value in aggregation step for flow:    
        aggreg_data['flow'] = event_data.groupby(step)['flow'].min()         
    
    # flood analysis:
    elif event_type == 'fd' or event_type == 'flood':
        # import objective_thresholds flood analysis module:
        import objective_thresholds.flood as ofd
        # calculate threshold:
        threshold_value = ofd.threshold(data, method=threshold_method)
        # convert data to frame:
        event_data = data.to_frame()
        event_data.columns = ['flow']
        # detect periods of flood occurrence based on calculated threshold:
        event_data.loc[event_data.flow >= threshold_value, 'occurrence'] = 1
        # prepare aggregation intervals:
        data['month'] = data.index.month
        data['year'] = data.index.year
        # aggregate occurrence data by sum for occurrence:
        aggreg_data = data.groupby(step)['occurrence'].sum().to_frame()
        # aggregate flow data by maximal value in aggregation step for flow: 
        aggreg_data['flow'] = data.groupby(step)['flow'].max()  
        
    else:
        raise SyntaxError('Parameter event_type accepts string of:'+'\n'+
                          '"lf" or "lowflow" for low flow / streamflow drought studies'+'\n'+
                          '"fd" or "flood" for flood studies')

    
    return event_data, aggreg_data