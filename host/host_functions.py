# -*- coding: utf-8 -*-
"""
This is Harmonic Oscillator Seasonal Trend (HOST) Model functions module  
that is required for HOST model to run.

Please refer to package repository for citation and updates:
https://github.com/chrisrac/hostmodel

Future functionalities will be added in later development.
Please reffer to documentation for information on current stage of development.

v. 1.0

@authors: Krzysztof Raczynski
"""

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('mode.chained_assignment', None)
from statsmodels.tsa.seasonal import STL
import host_models


def preprocess(data, event, htype, beginning, datastep, interval, 
               threshold_method, threshold_overwrite, signal_method, area,
               binary_occurrence=True):
    '''
    Preprocessing function used to prepare data, identify analyzed events (flood,
    streamflow drought, maximal or minimal flows) and aggregate them to analyzed
    interval. Used for both Harmonic as well as HOST classes.

    Parameters
    ----------
    data : array-like
        the Iterable sequence of numbers (int/float) to be used, f.e.: list,
        pd.Series, np.array or pd.DataFrame slice.
    event : string
        string representing type of studied event. Available options are:
            "value" : for direct values analysis. Use signal_method for 
                       aggregation control
            "low"   : for low/minimal values in aggregation step analysis;
            "high"  : for high/maximial values in aggregation step analysis.
    htype : string
        string representing the type of analysis. Available options are:
            "signal"     : for non-parametric values analysis;
            "occurrence" : for analysis of event occurrence (binary classification);
            "magnitude"  : for analysis of magnitudes of events in aggregation
                           interval, presented as sum of volumes by number of 
                           days in interval. Use area optional parameter to
                           resize the magnitude to unitary information.
    beginning : date
        date for timeseries to start from, eg.: '01-01-1979'.
    datastep : string
        the step of input data. Use 'D' for daily data, 'M' for monthly data or 
        'Y' for annual data.
    interval : string
        the interval of data to be agglomerated to. Use 'D' for daily, 
        'M' for monthly or 'Y' for annual (year).
    threshold_method : string
        used to controll objective method feedback on high discretized data, 
        that results in less than five unique values in the series. 
        Available options are:
            "leave"  : breaks the computation, no threshold is returned;
            "min"    : minimal value of data is used as threshold;
            "median" : data median is used as threshold.
    threshold_overwrite : int/float
        threshold value to be used instead of automatic use of objective 
        threshold method.
    signal_method : string
        allows to control the aggregation method when signal event is used. 
        Available options are:
            "mean" : mean values are computed from signal when aggregating;
            "sum" : totals are calculated from signal when aggregating;
            "min" : minimal signal values are used in aggregation step;
            "max" : maximal signal values are used in aggregation step;
    area : float / int
        if Area is provided and htype='magnitude' is used, the output will
        contain unitary magnitudes as product (magnitude / Area) instead of
        direct magnitudes. This allows to compare the magnitude of event between
        catchments of different sizes. Leave the default value (as 0) to not
        include transform to unit values.
        Default is 0.
    
    Raises
    ------
    SyntaxError
        if event is missdefinied. Event string must be either "signal", "min" 
        or "max" depending on user needs.
    Exception
        if non-standard data step is used. Only daily ("D"), monthly ("M") and
        annual ("Y") steps are supported.

    Returns
    -------
    aggreg_data : pandas DataFrame
        time indexed DataFrame with defined variable or parameters computed and
        aggregated to defined interval.
    '''
    
    # change data to pandas DataFrame:
    indexed_data = pd.DataFrame(data)
    # assign datetime index and clean:
    indexed_data['indexes'] = pd.date_range(beginning,
                                            periods=len(data), 
                                            freq=datastep)
    indexed_data = indexed_data.set_index('indexes',drop=True)
    indexed_data = indexed_data.squeeze()
    # prepare aggregation indexing space
    if interval == 'D':
        step = ['year', 'month', 'day']
    elif interval == 'M':
        step = ['year', 'month']
    elif interval == 'Y':
        step = ['year']   
    
    indexed_data = indexed_data.to_frame()
    indexed_data.columns = ['signal']
    indexed_data['volume'] = 0
    
    if event == 'value':
        indexed_data['occurrence'] = 1
        indexed_data.loc[pd.isnull(indexed_data['signal'])] = 0
        if datastep == 'D':
            indexed_data['volume'] = indexed_data.signal * 86400 
        elif datastep == 'M':
            indexed_data['volume'] = indexed_data.signal * \
                (indexed_data.index.days_in_month*86400)
        elif datastep == 'Y':
            indexed_data['volume'] = indexed_data.signal * (365*86400) 
        else:
            raise Exception('''Non-standard step (daily, monthly or annual) 
                            input data is not supported.''')            
    elif event == 'low':
        if threshold_overwrite != None:
            threshold_value = threshold_overwrite
        else:
            import objective_thresholds.lowflow as olf
            threshold_value = olf.threshold(indexed_data['signal'], method=threshold_method)
        indexed_data.loc[indexed_data.signal <= threshold_value, 'occurrence'] = 1
        if datastep == 'D':
            indexed_data['volume'].loc[indexed_data.signal <= threshold_value] = \
                (threshold_value - indexed_data.signal)*86400 
        elif datastep == 'M':
            indexed_data['volume'].loc[indexed_data.signal <= threshold_value] = \
                (threshold_value - indexed_data.signal) * \
                (indexed_data.index.days_in_month*86400)
        elif datastep == 'Y':
            indexed_data['volume'].loc[indexed_data.signal <= threshold_value] = \
                (threshold_value - indexed_data.signal) * (365*86400)   
        else:
            raise Exception('''Non-standard step (daily, monthly or annual) 
                            input data is not supported.''')
    elif event == 'high':
        if threshold_overwrite != None:
            threshold_value = threshold_overwrite
        else:
            import objective_thresholds.flood as ofd
            threshold_value = ofd.threshold(indexed_data['signal'], method=threshold_method)  
        indexed_data.loc[indexed_data.signal >= threshold_value, 'occurrence'] = 1
        if datastep == 'D':
            indexed_data['volume'].loc[indexed_data.signal >= threshold_value] = \
                (indexed_data.signal - threshold_value)*86400 
        elif datastep == 'M':
            indexed_data['volume'].loc[indexed_data.signal >= threshold_value] = \
                (indexed_data.signal - threshold_value) * \
                (indexed_data.index.days_in_month*86400)
        elif datastep == 'Y':
            indexed_data['volume'].loc[indexed_data.signal >= threshold_value] = \
                (indexed_data.signal - threshold_value) * (365*86400)        
        else:
            raise Exception('''Non-standard step (daily, monthly or annual) 
                            input data is not supported.''')
    else:
        raise SyntaxError('''Event argument requires one of the following options:
                          value - for analysis of raw signal data (all time occurrence,
                                  unless no data is present);
                          low - for analysis of minimal values or when event is
                                defined below threshold level;
                          high - for analysis of maximal values or when event is
                                defined above threshold level.''')
 
    # prepare aggregation intervals:
    indexed_data['month'] = indexed_data.index.month
    indexed_data['day'] = indexed_data.index.day 
    indexed_data['year'] = indexed_data.index.year
    # perform aggregation
    if htype == 'signal':
        if event == 'value':
            if signal_method=='mean':
                aggreg_data = indexed_data.groupby(step)['signal'].mean().to_frame()  
            elif signal_method=='sum':
                aggreg_data = indexed_data.groupby(step)['signal'].sum().to_frame()  
            elif signal_method=='min':
                aggreg_data = indexed_data.groupby(step)['signal'].min().to_frame()  
            elif signal_method=='max':
                aggreg_data = indexed_data.groupby(step)['signal'].max().to_frame()  
            else:
                raise Exception('''Non-standard method for signal processing is 
                                not supported. Use signal_method to set it to
                                mean (default), sum, min, max according to 
                                agglomeration needs.''')  
        if event == 'low':
            aggreg_data = indexed_data.groupby(step)['signal'].min().to_frame()  
        elif event == 'high':
            aggreg_data = indexed_data.groupby(step)['signal'].max().to_frame()        
    elif htype == 'occurrence':
        aggreg_data = indexed_data.groupby(step)['occurrence'].sum().to_frame()
        if binary_occurrence == True:
            aggreg_data.loc[aggreg_data['occurrence'] > 0] = 1       
    elif htype == 'magnitude':
        aggreg_data = indexed_data.groupby(step).agg({'month':'size', 'volume':'sum'}).rename(columns={'month':'count','volume':'sum'})
        aggreg_data['magnitude'] = aggreg_data['sum'] / aggreg_data['count']
        if area != 0:
            aggreg_data['magnitude'] = aggreg_data['magnitude'] / area    
        aggreg_data = aggreg_data[['magnitude']]

    else:
        raise SyntaxError('''Unrecognized htype. Accepted are: signal, 
                          occurrence, or magnitude.''') 
    
    return aggreg_data
    
    
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
            "variable"   : for analysis of variable values
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
    
    data_t = data[[htype]]        
    if htype=='occurrence':
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
        res = host_models.fit_sine(x, data)
        data = data - res['predictions']
        harmonic_res['harmonic: '+str(i)] = [res['amp'],res['freq'],res['phase'],
                                                res['offset'],res['period'],
                                                res['r2']]
        harmonic_res.index = ['amp','freq','phase','offset','period',
                              'r2']
        if include_preds==True:
            harmonic_preds['harmonic: '+str(i)] = res['predictions']


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


def stl_calc(data, multiplier, repeats, interval):
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
    try:
        res = host_models.fit_sine(x=x, y=data, repeats=repeats)
        periods = int(abs(res['period'])) * multiplier
    except:
        if interval=='M':
            periods=12
        elif interval=='D':
            periods=365
        else:
            periods=1
    # control to low period values:
    if periods < 2:
        periods = 2 * multiplier
    # perform STL decomposition
    stl = STL(data, period=periods)
    res = stl.fit()
    
    return res.trend, res.seasonal, res.resid 


def topology(predictions, original, slope=0):
    '''
    Performs topological analysis to find best fitted decision threshold for
    harmonic model. Decides which part of function is interpretted as 
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
    results, predictions = contingency(predictions, threshold, original, slope, True)       
            
    return {'threshold':threshold, 'contingency':results['contingency'],
            'accuracy':results['accuracy'], 'precision':results['precision'],
            'recall':results['recall'], 'f1score':results['f1score'], 
            'predictions':predictions}


def magnitude_topology(predictions, original, slope=0, threshold=None):
    '''
    Performs topological analysis to find best fitted decision threshold for
    harmonic model. Decides which part of function is interpretted as 
    best-fitting to magnitude distribution based on Kling-Gupta Efficiency, 
    then performs 'occurrence' vs 'non-occurrence' assesment based on this 
    magnitude fitted function. Used only if htype of object is set to 
    'magnitude'.
    
    Parameters
    ----------
    predictions : array-like
        model predictions. Must be same length as original.
    original : array-like
        original, raw data. Must be same length as predicted.
    slope : int/float, optional
        slope value for sloped models only. Affects the slope of decision 
        threshold. If simple model is compared, slope must be set to 0.
    threshold : float/int
        parameter to control behaviour between training and testing sets. 
        Unchanged means training dataset is analyzed and best-fit threshold
        must be found. Once testing/validationg, provide threshold computed
        in training dataset to use it in interpretation.

    Returns
    -------
    dict
        dictionary of results:
        threshold : float
            decision threshold for best fitted model.
        occurrence contingency : list-of-lists
            contingency table in form of list of lists for occurrence 
            assessment.
        occurrence accuracy : float
            accuracy of the occurrence model based on found threshold.
        occurrence precision : float
            precision of the occurrence model based on found threshold.
        occurrence recall : float
            recall of the occurrence model based on found threshold.
        occurrence f1score : float
            f1 score of the occurrence model based on found threshold. 
        occurrence predictions : array-like
            an array of predicted occurrences for magnitude-based model.
        'magnitude efficiency' : float
            Kling-Gupta efficiency of magnitude model.
        'magnitude predictions' : array-like
            an array of model predicted magnitudes.
    '''
    
    x_range = range(0,len(predictions))
    if threshold == None:
        # sort harmonic values
        data_sorted = -np.sort(-predictions)
        # prepare decision list for highest f1 score
        effs = []
        thrs = []
        # calculate accuracy statistics for each threshold:   
        for thr in data_sorted:
            limits = []
            if slope != 0:
                for i in x_range:         
                    limits.append(thr + slope * i)
            else:
                limits = [thr]*len(predictions) 
            mag_predictions = np.where(predictions > limits, predictions, 0)
            r2 = rsquared(original, mag_predictions)
            effs.append(r2)
            thrs.append(thr)
            #eff = efficiency_stat(mag_predictions, original, 'kge')
            #effs.append(eff)
            #thrs.append(thr)
            #results = contingency(predictions, thr, original, slope)
               
        # find highest f1 score:
        max_index = effs.index(max(effs))
        # find decision threshold based on highest f1 score
        threshold = thrs[max_index] 
    else:
        threshold = threshold
    limits = []
    if slope != 0:
        for i in x_range:         
            limits.append(threshold + slope * i)
    else:
        limits = [threshold]*len(predictions) 
    predictions = np.where(predictions > limits, predictions, 0)
    
    occ_prediction = np.where(predictions > 0, 1, 0)
    occ_original = np.where(original > 0, 1, 0)   
    occ_results = contingency_stat(occ_prediction, occ_original)
    
    mag_results, mag_predictions = efficiency_stat(predictions, original, 'kge', True)
    
    return {'threshold': threshold,
            'occurrence accuracy':occ_results['accuracy'], 
            'occurrence contingency':occ_results['contingency'], 
            'occurrence precision':occ_results['precision'], 
            'occurrence recall':occ_results['recall'], 
            'occurrence f1score':occ_results['f1score'], 
            'occurrence predictions':occ_results['predictions'],
            'magnitude predictions':mag_predictions,
            'magnitude efficiency':mag_results}   


def contingency(predictions, threshold, observed, slope, include_predictions=False):
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
    
    if include_predictions==False:
        return results
    else:
        return results, predicted


def rsquared(observations, predictions):
    observations = np.array(observations)
    predictions = np.array(predictions)
    residuals = observations - predictions
    square_res = np.sum(residuals**2)
    square_tot = np.sum((observations - np.mean(observations))**2)
    r2 = 1 - (square_res / square_tot)    
    return r2

    
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
            'precision':precision, 'recall':recall, 'f1score':f1score, 
            'predictions':predictions}


def efficiency_stat(predictions, observations, statistic, include_predictions=False):
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
    if include_predictions==True:
        return parameter, predictions
    else:
        return parameter