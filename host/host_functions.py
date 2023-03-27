# -*- coding: utf-8 -*-
"""
This is Harmonic Oscillator Seasonal Trend (HOST) Model functions module  
that is required for HOST model to run.

Please refer to package repository for citation and updates:
https://github.com/chrisrac/hostmodel

Future functionalities will be added in later development.
Please reffer to documentation for information on current stage of development.

@authors: Krzysztof Raczynski
"""

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('mode.chained_assignment', None)
import scipy.optimize
from statsmodels.tsa.seasonal import STL


# functions block:
    
def preprocess(data, event, htype, beginning, step, interval, threshold_method):
    '''
    Preprocessing function used to prepare data, identify analyzed events (flood,
    streamflow drought, maximal or minimal flows) and aggregate them to analyzed
    interval. Used for both Harmonic as well as HOST classes.

    Parameters
    ----------
    data : array-like
        The Iterable sequence of numbers (int/float) to be used, f.e.: list,
        pd.Series, np.array or pd.DataFrame slice.
    event : string
        string representing type of studied event. Available options are:
            "lf" or "lowflow" : for streamflow drought/low flow analysis;
            "fd" or "flood"   : for flood/high flow analysis.
    htype : string
        string representing the type of analysis. Available options are:
            "flow"       : for analysis of min/max flows
            "occurrence" : for analysis of flood/drought occurrence. Please
                           refer to current documentation for limitations 
                           explanation.
    beginning : date
        date for timeseries to start from, eg.: '01-01-1979'.
    step : string
        the step of input data. Use 'D' for daily data, 'M' for monthly data or 
        'Y' for annual data.
    interval : string
        the interval of data to be agglomerated to. Use 'D' for daily, 
        'M' for monthly or 'Y' for annual (year).
    threshold_method : string
        Used to controll objective method feedback on high discretized data, 
        that results in less than five unique values in the series. 
        Available options are:
            "leave"  : breaks the computation, no threshold is returned;
            "min"    : minimal value of data is used as threshold;
            "median" : data median is used as threshold.

    Raises
    ------
    SyntaxError
        if event is missdefinied. Event string must be either "lf" or "lowflow" 
        for drought/low flow studies or "fd" or "flood" for flood/max flow 
        analysis.

    Returns
    -------
    aggreg_data : array-like
        time indexed array-like object with events identified and data 
        aggregated to defined interval.
    '''
    
    # change data to pandas DataFrame:
    indexed_data = data.to_frame()
    # assign datetime index and clean:
    indexed_data['indexes'] = pd.date_range(beginning,
                                            periods=len(data), 
                                            freq=step)
    indexed_data = indexed_data.set_index('indexes',drop=True)
    indexed_data = indexed_data.squeeze()
    
    if interval == 'M':
        step = ['year', 'month']
    elif interval == 'Y':
        step = ['year']   
    
    # low flow / streamflow analysis:
    if event == 'lf' or event == 'lowflow':
        # import objective_thresholds low flow analysis module:
        import objective_thresholds.lowflow as olf
        # calculate threshold:
        threshold_value = olf.threshold(indexed_data, method=threshold_method)
        # convert data to frame:
        indexed_data = indexed_data.to_frame()
        indexed_data.columns = ['flow']
        # detect periods of drought occurrence based on calculated threshold:
        indexed_data.loc[indexed_data.flow <= threshold_value, 'occurrence'] = 1
        # prepare aggregation intervals:
        indexed_data['month'] = indexed_data.index.month
        indexed_data['year'] = indexed_data.index.year
        # aggregate occurrence data by sum for occurrence:    
        if htype == 'occurrence':
            aggreg_data = indexed_data.groupby(step)['occurrence'].sum().to_frame()
            aggreg_data.loc[aggreg_data['occurrence'] > 0] = 1
        elif htype == 'flow':
            aggreg_data = indexed_data.groupby(step)['flow'].min().to_frame()      
    
    # flood analysis:
    elif event == 'fd' or event == 'flood':
        # import objective_thresholds flood analysis module:
        import objective_thresholds.flood as ofd
        # calculate threshold:
        threshold_value = ofd.threshold(data, method=threshold_method)
        # convert data to frame:
        indexed_data = indexed_data.to_frame()
        indexed_data.columns = ['flow']
        # detect periods of flood occurrence based on calculated threshold:
        indexed_data.loc[indexed_data.flow >= threshold_value, 'occurrence'] = 1
        # prepare aggregation intervals:
        indexed_data['month'] = indexed_data.index.month
        indexed_data['year'] = indexed_data.index.year
        # aggregate occurrence data by sum for occurrence:
        if htype == 'occurrence':    
            aggreg_data = indexed_data.groupby(step)['occurrence'].sum().to_frame() 
            aggreg_data.loc[aggreg_data['occurrence'] > 0] = 1
        elif htype == 'flow':
            aggreg_data = data.groupby(step)['flow'].max().to_frame()  
       
    else:
        raise SyntaxError('Parameter event_type accepts string of:'+'\n'+
                          '"lf" or "lowflow" for low flow / streamflow drought studies'+'\n'+
                          '"fd" or "flood" for flood studies')
    
    return aggreg_data
    
     
def fit_simple(x, y, repeats = 10000):
    '''
    Function to fit simple harmonic oscillator to the input data using 
    Fast Fourier Transform. Requires preprocessed, aggregated data. 
    Returns function parameters and statistics. 

    Parameters
    ----------
    x : array of int
        x-axis time factor.
    y : array of float/int
        y-axis variable like flow or occurrence information.
    repeats : int, optional
        maximal number of function calls to fit harmonic to data. Increase to
        try to fit to complicated data. May increase computation cost and time.
        Default is 10,000.
        
    Raises
    ------
    Exception    
        if repeats is to low and the function can't be fit to dataset raises
        information exception
    
    Returns
    -------
    dict
        the dictionary containing fitted function parameters: 'amp', 'omega', 
        'phase', 'offset', 'freq', 'period', together with statistics 
        'r2' (percentage of original data variance explained by model), 
        'y_pred' (predicted values) and 'function' (fitted function object).
    '''
    
    # prepare initial parameters
    x = np.array(x)
    y = np.array(y)
    ff = np.fft.fftfreq(len(x), (x[1]-x[0]))
    Fyy = abs(np.fft.fft(y))
    # guessing initial parameters to lower optimization cost
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])
    guess_amp = np.std(y) * 2.**0.5
    guess_offset = np.mean(y)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])
    
    # define simple harmonic function
    def sinfunc(x, A, w, p, c):  return A * np.sin(w*(x-p)) + c
    # fit function curve
    try:
        popt, pcov = scipy.optimize.curve_fit(sinfunc, x, y, p0=guess, maxfev = repeats)
    except:
        raise Exception("function can't be optimized with defined function calls \
                        you can try increasing 'repeats' parameter; \n \
                        however, it is recommended to check the data first, \
                        as some situations might increase optimization difficulty, \
                        f.e. constant data. \n \
                        Please refer to 'maxfev' parameter in 'scipy.optimize.curve_fit' \
                        for more information.")
    
    A, w, p, c = popt
    f = w/(2.*np.pi)
    # generate predicted values
    y_pred = sinfunc(x, *popt)
    # calculate variance explained by model as r2
    residuals = y - sinfunc(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # return control to generate function and/or predictions
    function = lambda x: c + A * np.sin(w*(x-p))
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f,
            "period": 1./f, "r2": r_squared, "y_pred": y_pred, "function": function}
    

def fit_sloped(x, y, repeats = 10000):
    '''
    Function to fit sloped harmonic oscillator to the input data using 
    Fast Fourier Transform. Requires preprocessed, aggregated data. 
    Returns function parameters and statistics. 

    Parameters
    ----------
    x : array of int
        x-axis time factor.
    y : array of float/int
        y-axis variable like flow or occurrence information.
    repeats : int, optional
        maximal number of function calls to fit harmonic to data. Increase to
        try to fit to complicated data. May increase computation cost and time.
        Default is 10,000.
        
    Raises
    ------
    Exception    
        if repeats is to low and the function can't be fit to dataset raises
        information exception
    
    Returns
    -------
    dict
        the dictionary containing fitted function parameters: 'amp', 'omega', 
        'phase', 'offset', 'freq', 'period', together with statistics 
        'r2' (percentage of original data variance explained by model), 
        'y_pred' (predicted values) and 'function' (fitted function object).
    '''
    
    # prepare initial parameters
    x = np.array(x)
    y = np.array(y)
    ff = np.fft.fftfreq(len(x), (x[1]-x[0]))
    Fyy = abs(np.fft.fft(y))
    # guessing initial parameters to lower optimization cost    
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])
    guess_amp = np.std(y) * 2.**0.5
    guess_offset = np.mean(y)
    guess_slope = 0
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset, guess_slope])

    # define sloped harmonic function
    def sinfunc(x, A, w, p, c, s):  return c + s * x + A * np.sin(w*(x-p))
    try:
        popt, pcov = scipy.optimize.curve_fit(sinfunc, x, y, p0=guess, maxfev = repeats)
    except:
        raise Exception("function can't be optimized with defined function calls \
                        you can try increasing 'repeats' parameter; \n \
                        however, it is recommended to check the data first, \
                        as some situations might increase optimization difficulty, \
                        f.e. constant data. \n \
                        Please refer to 'maxfev' parameter in 'scipy.optimize.curve_fit' \
                        for more information.")
                        
    A, w, p, c, s = popt
    f = w/(2.*np.pi)
    # generate predicted values    
    y_pred = sinfunc(x, *popt)
    # calculate variance explained by model as r2
    residuals = y - sinfunc(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # return control to generate function and/or predictions
    function = lambda x: c + A * np.sin(w*(x-p))
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "slope": s,
            "period": 1./f, "r2": r_squared, "y_pred": y_pred, "function": function}
    
    
def harmonic(data, htype, no_functions, include_preds):
    '''
    Function to fit first n harmonics to the occurrence data. Requires 
    preprocessed, aggregated data. Returns functions parameters and statistics 
    in form of summary table. 

    Parameters
    ----------
    data : array-like
        time indexed and aggregated array object representing model input.
    htype : string
        string representing the type of analysis. Available options are:
            "flow"       : for analysis of min/max flows
            "occurrence" : for analysis of flood/drought occurrence. Please
                           refer to current documentation for limitations 
                           explanation.
    no_functions : int
        number of harmonics to extract from data. Each harmonic is extracted 
        from data before next is fitted.
    include_preds : bool
        controls size of result table by including (when True) predicted 
        values. If False, predictions must be calculated manually based on the
        output parameters for each function.

    Returns
    -------
    harmonic_res : pandas DataFrame
        the summary table containing all harmonic functions parameters and r2 
        value.
    harmonic_preds : array-like
        series of function values for each fitted function. Returned only if
        include_pres == True.
    '''
    
    if htype=='flow':
        data_t = data[[htype]]
    elif htype=='occurrence':
        data_t = data[[htype]]
        # convert sums of event occurrence to occurrence flag:
        # 1 - event occurrence, 0 - lack of event        
        data_t.loc[data_t[htype] > 0] = 1
        data_t.loc[data_t[htype] <= 0] = 0        

    # calculate consecutive harmonics until no_functions reached, with each
    # harmonic subtracted from data. Return parameters and explained variance
    harmonic_res = pd.DataFrame()   
    if include_preds==True:
        harmonic_preds = pd.DataFrame()
        
    for i in range(1,no_functions+1):
        if i == 1:
            data = data_t[htype]
        x = range(0,len(data))
        res = fit_simple(x, data)
        data = data - res['y_pred']
        harmonic_res['harmonic: '+str(i)] = [res['amp'],res['omega'],res['phase'],
                                                res['offset'],res['freq'],res['period'],
                                                res['r2']]
        harmonic_res.index = ['amp','omega','phase','offset','freq','period',
                              'r2']
        if include_preds==True:
            harmonic_preds['harmonic: '+str(i)] = res['y_pred']

    # include predictions in return if include_preds=True
    # Allows to reduce output size symmetrically to series length
    if include_preds==True:
        return harmonic_res, harmonic_preds
    else:
        return harmonic_res
    

def data_split(data, train_size):
    '''
    Function used for train-test split of dataset.

    Parameters
    ----------
    data : array-like
        the iterable sequence of numbers (int/float) to be used, f.e.: list,
        pd.Series, np.array or pd.DataFrame slice.
    train_size : float
        Size of the training sample in nounit format. Value provided should be 
        no higher than 1 (equalt to 100%) and higher than 0. For example 0.8 
        reserves 80% of data for training and 20% for testing purposes.

    Returns
    -------
    train_y : array
        array of function values for trainig purposes, constituting
        of (train_size) part of input series.
    train_x : array
        array of function arguments representing x-axis, for the training 
        purposes. Length is same as train_y.
    test_y : array
        array of function values for testing purposes, constituting
        of (1 - train_size) part of input series. 
    test_x : array
        array of function arguments representing x-axis, for the testing 
        purposes. Length is same as test_x.        
    split_index : int
        index representing the moment of data split to training and testing 
        series in the original dataset.
    '''
    
    split_index = int(len(data) * train_size)
    if train_size < 1:
        train_y = data[:split_index]
        test_y = data[split_index:]
        train_x = np.arange(split_index)
        test_x = np.arange(split_index, len(data))
    else:
        train_y = data
        train_x = np.arange(len(data))
        test_y = np.array([])
        test_x = np.array([])
        
    return train_y, train_x, test_y, test_x, split_index


def stl_calc(data):
    '''
    Function calculates trend, seasonal and residual components based on STL
    decomposition method. Period length is assumed based on first harmonic and
    is not lower than two. 

    Parameters
    ----------
    data : array
        array of values to be decomposed. Might represent original data or
        train/test splitted data.

    Returns
    -------
    trend : array-like
        The iterable sequence of numbers (int/float) representing trend 
        component decomposed from data.
    seasonal : array-like
        The iterable sequence of numbers (int/float) representing seasonal
        component decomposed from data.
    resid : array-like
        The iterable sequence of numbers (int/float) representing residual
        component decomposed from data.
    '''

    x = range(0, len(data))
    # calculate initial period length for STL decomposition input
    res = fit_simple(x, data)
    periods = int(abs(res['period']))
    # control to low period values:
    if periods < 2:
        periods = 2
    # perform STL decomposition
    stl = STL(data, period=periods)
    res = stl.fit()
    
    return res.trend, res.seasonal, res.resid 


def topology(predictions, original, slope=0):
    '''
    Performs topological analysis to find best fitted decision threshold for
    harmonical model. Decides which part of function is interpretted as 
    'occurrence' vs 'non-occurrence' based on the highest f1 score of all
    thresholds. Used only if htype of object is set to 'occurrence'.
    
    Parameters
    ----------
    predictions : array-like
        model predictions. Must be same length as original.
    original : array-like
        original, raw data. Must be same length as predicted.
    slope : int/float, optional
        slope value for sloped models only. Affects the slope of decision 
        threshold. If simple model is compared, slope must be set to 0.

    Returns
    -------
    dict
        dictionary of results:
        threshold : float
            decision threshold for best fitted model.
        contingency : list-of-lists
            contingency table in form of list of lists.
        accuracy : float
            accuracy of the model based on found threshold.
        precision : float
            precision of the model based on found threshold.
        recall : float
            recall of the model based on found threshold.
        f1score : float
            f1 score of the model based on found threshold.    
    '''
    
    # sort harmonic values
    data_sorted = -np.sort(-predictions)
    
    # prepare decision list for highest f1 score
    f1s = []
    thrs = []
    # calculate accuracy statistics for each threshold: 
    for thr in data_sorted:
        results = contingency(predictions, thr, original, slope)
        f1s.append(results['f1score'])
        thrs.append(thr)

    # find highest f1 score:
    max_index = f1s.index(max(f1s))
    # find decision threshold based on highest f1 score
    threshold = thrs[max_index]
    # make accuracy statistics for best fitted model
    results = contingency(predictions, threshold, original, slope)       
            
    return {'threshold':threshold, 'contingency':results['contingency'],
            'accuracy':results['accuracy'], 'precision':results['precision'],
            'recall':results['recall'], 'f1score':results['f1score']}


def contingency(predictions, threshold, observed, slope):
    '''
    Contingency calculator used to determine contingency statistics for model
    data based on original observations, using threshold found during topological
    analysis.
    '''
    
    x_range = range(0,len(predictions))
    if slope != 0:
        limits = []
        for i in x_range:         
            limits.append(threshold + slope * i)
    else:
        limits = [threshold]*len(predictions)
    limits = np.array(limits)
    predicted = event_predict(predictions, limits) 
    results = contingency_stat(predicted, observed)
    
    return results

    
def event_predict(data, limit):
    '''
    Function used to compare model data versus the threshold.
    Flags data as 1 if event occurs (harmonic prediction is above threshold)
    or 0 for non-occurrence.
    '''
    return np.where(data >= limit, 1, 0)
    

def contingency_stat(predictions, original):
    '''
    Function calculates contingency table and statistics.

    Parameters
    ----------
    predictions : array
        model predictions values processed with event_predict function.
    original : array
        original observations data.

    Returns
    -------
    dict
        a dictionary of results, containing contingency table under 'contingency'
        and a set of accurracy statistics: 'accuracy', 'precision', 'recall', 
        'f1score'.
    '''
    
    count11 = np.count_nonzero((original==1) & (predictions==1))
    count01 = np.count_nonzero((original==0) & (predictions==1))
    count10 = np.count_nonzero((original==1) & (predictions==0))
    count00 = np.count_nonzero((original==0) & (predictions==0))
    
    contingency = np.array([[count11,count10],[count01,count00]])
    accuracy = (count11+count00)/(count11+count01+count10+count00)
    precision = count11/(count11+count01) if count11+count01 > 0 else 0
    recall = count11/(count11+count10) if count11+count10 > 0 else 0
    f1score = 2*(precision*recall)/(precision+recall) if precision+recall > 0 else 0
    return {'contingency':contingency, 'accuracy':accuracy, 
            'precision':precision, 'recall':recall, 'f1score':f1score}


def efficiency_stat(predictions, observations, statistic):
    '''
    Function to calculate efficiency value for continuous model data. Used if
    htype of object is set to 'flow'.

    Parameters
    ----------
    predictions : array
        model predicted values of min/max flows.
    observations : array
        original observations data representing min/max flows.
    statistic : string
        determines which efficiency measure to be used: 
            'nse' : for NAsh-Sutcliffe Efficiency (Knoben et al. 2019),
            'kge' : for modified Kling-Gupta Efficiency (Gupta et al. 2009; 
                                                         Kling et. al., 2012).
            
    Returns
    -------
    parameter : float
        the efficiency metric value.

    References
    -------
    Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). 
        Decomposition of the mean squared error and NSE performance criteria: 
        Implications for improving hydrological modelling. 
        Journal of Hydrology, 377(1-2), 80-91.
    Kling H., Fuchs M., Paulin M., (2012). 
        Runoff conditions in the upper Danube basin under an ensemble of 
        climate change scenarios.
        Journal of Hydrology, 424–425, 264-277.
    Knoben, W. J. M., Freer, J. E., Woods, R. A., (2019). 
        Technical note: Inherent benchmark or not? Comparing Nash-Sutcliffe 
        and Kling-Gupta efficiency scores. 
        Hydrology and Earth System Sciences, 23, 4323–4331.
    '''
    
    if statistic=='nse':
        mean_obs = observations.mean()
        parameter = 1 - (np.sum((observations - predictions) ** 2) / (np.sum(observations - mean_obs) ** 2))
    elif statistic=='kge':
        mean_pred = predictions.mean()
        mean_obs = observations.mean()
        r_num = np.sum((predictions - mean_pred) * (observations - mean_obs))
        r_den = np.sqrt(np.sum((predictions - mean_pred) ** 2) * np.sum((observations - mean_obs) ** 2))
        r = r_num / r_den
        alpha = (predictions.std() / mean_pred) / (observations.std() / mean_obs)
        beta = mean_pred / mean_obs
        parameter = 1 - np.sqrt(((r - 1) ** 2) + ((alpha - 1) ** 2) + ((beta - 1) ** 2))
    return parameter

