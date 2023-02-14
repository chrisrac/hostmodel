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
# suppress all warnings 
import warnings
warnings.filterwarnings("ignore")
# import needed modules:
import math
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('mode.chained_assignment', None)
import scipy.optimize
from statsmodels.tsa.seasonal import STL
from scipy.stats import mannwhitneyu
from scipy.stats import chisquare
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error


# functions block:
def fit_simple(x, y, maxfev_t = 10000, return_function=False):
    '''
    Inner function to fit simple harmonic oscillator to the flow data using 
    Fast Fourier Transform. Requires preprocessed, aggregated data. 
    Returns function parameters and statistics. 

    Parameters
    ----------
    x : int
        X-axis time factor.
    y : float / int
        Y-axis variable like flow or occurrence information.
    maxfev : int, optional
        Maximal number of function calls to fit harmonic to data. Increase to
        try to fit to complicated data. May increase computation cost and time.
        Default is 10,000.
    return_function : bool
        Controlls the return option for the function object. If True, returns 
        additional key "function" with fitted function object.

    Returns
    -------
    dict
        The dictionary containing fitted function parameters: 'amp', 'omega', 
        'phase', 'offset', 'freq', 'period', together with statistics 
        'r2' (percentage of variance explained by model), 'y_pred' (predicted
        values) and 'function' .
    '''
    
    # prepare initial parameters:
    x = np.array(x)
    y = np.array(y)
    ff = np.fft.fftfreq(len(x), (x[1]-x[0]))
    Fyy = abs(np.fft.fft(y))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])
    guess_amp = np.std(y) * 2.**0.5
    guess_offset = np.mean(y)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])
    
    # define simple harmonic function:
    def sinfunc(x, A, w, p, c):  return A * np.sin(w*(x-p)) + c
    # fit function curve:
    popt, pcov = scipy.optimize.curve_fit(sinfunc, x, y, p0=guess, maxfev = maxfev_t)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    # generate predicted values
    y_pred = sinfunc(x, *popt)
    # calculate variance explained by model as r2:
    residuals = y - sinfunc(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # return control to generate function object for future calculation
    if return_function==True:
        function = lambda x: c + A * np.sin(w*(x-p))
        return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f,
                "period": 1./f, "r2": r_squared, "y_pred": y_pred, "function": function}
    else:
        return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f,
                "period": 1./f, "r2": r_squared, "y_pred": y_pred}
    
    
def fit_sloped(x, y, maxfev_t = 10000, return_function=False):
    '''
    Inner function to fit sloped harmonic oscillator to the flow data using 
    Fast Fourier Transform. Requires preprocessed, aggregated data. 
    Returns function parameters and statistics. 

    Parameters
    ----------
    x : int
        X-axis time factor.
    y : float / int
        Y-axis variable like flow or occurrence information.
    maxfev : int, optional
        Maximal number of function calls to fit harmonic to data. Increase to
        try to fit to complicated data. May increase computation cost and time.
        Default is 10,000.
    return_function : bool
        Controlls the return option for the function object. If True, returns 
        additional key "function" with fitted function object.

    Returns
    -------
    dict
        The dictionary containing fitted function parameters: 'amp', 'omega', 
        'phase', 'offset', 'freq', 'period','slope', together with statistics 
        'r2' (percentage of variance explained by model) and 'y_pred' (predicted
        values).
    '''
    
    # prepare initial parameters:
    x = np.array(x)
    y = np.array(y)
    ff = np.fft.fftfreq(len(x), (x[1]-x[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(y))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(y) * 2.**0.5
    guess_offset = np.mean(y)
    guess_slope = 0
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset, guess_slope])

    # define simple harmonic function:
    def sinfunc(x, A, w, p, c, s):  return c + s * x + A * np.sin(w*(x-p))
    popt, pcov = scipy.optimize.curve_fit(sinfunc, x, y, p0=guess, maxfev = maxfev_t)
    A, w, p, c, s = popt
    f = w/(2.*np.pi)
    # generate predicted values    
    y_pred = sinfunc(x, *popt)
    # calculate variance explained by model as r2:
    residuals = y - sinfunc(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # return control to generate function object for future calculation    
    if return_function==True:
        function = lambda x: c + s * x + A * np.sin(w*(x-p))
        return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "slope": s,
                "period": 1./f, "r2": r_squared, "y_pred": y_pred, "fitfunc": function}
    else:
        return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "slope": s,
                "period": 1./f, "r2": r_squared, "y_pred": y_pred} 


def harmonic_occ(aggreg_data, no_functions=6, include_preds=True):
    '''
    Function to fit first n harmonics to the occurrence data. 
    Requires preprocessed, aggregated data. Returns functions parameters and 
    statistics in form of summary table. 

    Parameters
    ----------
    aggreg_data : pandas DataFrame
        Time indexed pandas DataFrame object with event occurrence identified
        and data aggregated in provided interval (produced by variable_generator).
    no_functions : int, optional
        Number of harmonics to extract from data. Each harmonic is extracted 
        from data before next is fitted. Optional, default value is 6.
    include_preds : bool, optional
        Controls size of result table by including (when True) function predicted 
        values. If False, predictions must be calculated manually based on the
        output parameters for each function.

    Returns
    -------
    harmonic_res : pandas DataFrame
        The DataFrame object in form of summary table containing all function
        parameters, r2 value and predicted values, if include_pres == True.
    '''
    
    # convert sums of event occurrence to occurrence flag: 
    # 1 - event occurrence, 0 - lack of event
    data_t = aggreg_data[['occurrence']]
    data_t.loc[data_t['occurrence'] > 0] = 1
    data_t.loc[data_t['occurrence'] <= 0] = 0

    # calculate consecutive harmonics until no_functions reached, with each
    # harmonic subtracted from data. Return parameters and explained variance:
    harmonic_res = pd.DataFrame()      
    for i in range(1,no_functions+1):
        if i == 1:
            data = data_t['occurrence']
        x = range(0,len(data))
        res = fit_simple(x, data)
        data = data - res['y_pred']
        if include_preds==True:
            harmonic_res['harmonic_'+str(i)] = [res['amp'],res['omega'],res['phase'],
                                                res['offset'],res['freq'],res['period'],
                                                res['r2'],res['y_pred']]
        else:
            harmonic_res['harmonic_'+str(i)] = [res['amp'],res['omega'],res['phase'],
                                                res['offset'],res['freq'],res['period'],
                                                res['r2']]

    # include predictions in return if include_preds=True (default).
    # Allows to reduce output size symmetrically to series length:
    if include_preds==True:
        harmonic_res.index = ['amp','omega','phase','offset','freq','period',
                              'r2','y_pred']
    else:
        harmonic_res.index = ['amp','omega','phase','offset','freq','period',
                              'r2']
    
    return harmonic_res


def harmonic_flow(aggreg_data, no_functions=6, include_preds=True):
    '''
    Function to fit first n harmonics to the (min/max) flow data. 
    Requires preprocessed, aggregated data. Returns functions parameters and 
    statistics in form of summary table. 

    Parameters
    ----------
    aggreg_data : pandas DataFrame
        Time indexed pandas DataFrame object with flow data aggregated in 
        provided interval (produced by variable_generator).
    no_functions : int, optional
        Number of harmonics to extract from data. Each harmonic is extracted 
        from data before next is fitted. Optional, default value is 6.
    include_preds : bool, optional
        Controls size of result table by including (when True) function predicted 
        values. If False, predictions must be calculated manually based on the
        output parameters for each function.

    Returns
    -------
    harmonic_res : pandas DataFrame
        The DataFrame object in form of summary table containing all function
        parameters, r2 value and predicted values, if include_pres == True.
    '''
    
    # create data copy for manipulation:
    data_t = aggreg_data[['flow']]

    # calculate consecutive harmonics until no_functions reached, with each
    # harmonic subtracted from data. Return parameters and explained variance:
    harmonic_res = pd.DataFrame()      
    for i in range(1,no_functions+1):
        if i == 1:
            data = data_t['flow']
        x = range(0,len(data))
        res = fit_simple(x, data)
        data = data - res['y_pred']
        if include_preds==True:
            harmonic_res['harmonic_'+str(i)] = [res['amp'],res['omega'],res['phase'],
                                            res['offset'],res['freq'],res['period'],
                                            res['r2'],res['y_pred']]
        else:
            harmonic_res['harmonic_'+str(i)] = [res['amp'],res['omega'],res['phase'],
                                            res['offset'],res['freq'],res['period'],
                                            res['r2']]            

    # include predictions in return if include_preds=True (default).
    # Allows to reduce output size symmetrically to series length:
    if include_preds==True:
        harmonic_res.index = ['amp','omega','phase','offset','freq','period',
                              'r2','y_pred']
    else:
        harmonic_res.index = ['amp','omega','phase','offset','freq','period',
                              'r2']
    
    return harmonic_res


def stl_calc(data):
    '''
    Function calculates trend, seasonal and residual components based on STL
    decomposition method. Period length is assumed based on first harmonic.
    Requires one data input that communicates with host_body function. 

    Parameters
    ----------
    data : pandas DataFrame
        Pandas DataFrame linked with host_body function.

    Returns
    -------
    trend : list
        The iterable sequence of numbers (int/float) representing trend 
        component decomposed from data.
    seasonal : list
        The iterable sequence of numbers (int/float) representing seasonal
        component decomposed from data.
    resid : list
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


def host_body(data, include_params=True):
    '''
    The main function to calculate five harmonic models based on the HOST framework
    assumptions. Accepts one input in form of aggregated, timestamped dataframe,
    preprocessed by variable_generator.

    Parameters
    ----------
    data : pandas DataFrame
        Aggregated, timestamped dataframe, preprocessed by variable_generator.
    include_params : bool
        Controls function output. Use False to return only model predictions,
        without parameters dictionary.

    Returns
    -------
    pred_dict : dict
        Dictionary containing all predicted values of fitted models (amplitude, omega,
        phase, offset, frequency, period, r2, y_pred): tf - trend, ts - trend sloped,
        sf - seasonal, cf - combined, cs - combined sloped.
    params_dict : dict
        Dictionary containing all parameters of fitted models (amplitude, omega,
        phase, offset, frequency, period, r2, y_pred): tf - trend, ts - trend sloped,
        sf - seasonal, cf - combined, cs - combined sloped. Returned only when 
        include_params=True
    '''
    
    # generate trend, seasonal and residuals using STL decomposition:
    trend, seasonality, residuals = stl_calc(data)
    x = range(0,len(trend))

    # prepare trend model, catch predictions and remove them from 
    # parameter dict:
    tf = fit_simple(x, trend)
    predicted_tf = tf['y_pred']
    del tf['y_pred']
    # prepare sloped trend model, catch predictions and remove them from
    # parameter dict:        
    ts = fit_sloped(x, trend)
    predicted_ts = ts['y_pred']
    del ts['y_pred']
    # prepare seasonal model, catch predictions and remove them from
    # parameter dict:    
    sf = fit_simple(x, seasonality)
    predicted_sf = sf['y_pred']
    del sf['y_pred']
    # create predictions for combined model:
    predicted_cf = predicted_tf + predicted_sf
    # create predictions for combined sloped model:
    predicted_cs = predicted_ts + predicted_sf
    
    # create parameters dictionary for output
    params_dict = {'tf' : tf, 'ts' : ts, 'sf' : sf}
    # create predictions dictionary for output
    pred_dict = {'tf' : predicted_tf, 'ts' : predicted_ts, 'sf' : predicted_sf,
                 'cf' : predicted_cf, 'cs' : predicted_cs}
    
    # controls output by include_params=True. If include_params=False, do not
    # include parameters dictionary in output (reduces output size):
    if include_params==False:
        return pred_dict
    else:
        return pred_dict, params_dict, 
 

def event_predict(df):
    '''Inner function for occurrence flag in topological anaysis'''
    
    if (df['data'] >= df['limit']):
        return 1
    else:
        return 0
    
    
def event_cat(df):
    '''
    Categorizer function for inner use. Marks combinations for contingency
    table according to rule:
    0 - real was 0, model predict 0,
    1 - real was 1, model predict 1,
    2 - real was 0, model predict 1,
    3 - real was 1, model predict 0
    '''
    
    if (df['comp'] == 0) & (df['predict'] == 0):
        return 0
    elif (df['comp'] == 1) & (df['predict'] == 1):
        return 1
    elif (df['comp'] == 0) & (df['predict'] == 1):
        return 2
    elif (df['comp'] == 1) & (df['predict'] == 0):
        return 3


def chi_sq(predicted, original):
    '''
    Function for distribution analysis for continuous data comparison between
    model outputs and original data. Must be provided with both model and 
    original data. Returns distribution comparison statistics.

    Parameters
    ----------
    predicted : array-like
        Model predictions of continuous data. Must be same length as original.
    original : array-like
        Original data. Must be same length as predicted.

    Returns
    -------
    stat : float
        Chi2 statistics for both series comparison.
    p_value : float
        Chi2 statistics p-value.
    r_corr : float
        Spearman rank correlation coefficient between model and original data.
    r_corr_p : float
        Spearman rank correlation coefficient p-value.
    MSE : float
        Mean square error.
    RMSE : float
        Root mean square error.
    '''
    
    df_bins = pd.DataFrame()
    # create q=20 equal bins based on quantile division and classify data:
    _, bins = pd.qcut(original, q=20, retbins=True, duplicates='drop')
    # calculates bins populations for original distribution:
    df_bins['bin'] = pd.cut(original, bins=bins).value_counts().index
    df_bins['org'] = pd.cut(original, bins=bins).value_counts().values
    # calculates bins populations for predicted data distribution:
    df_bins['pred'] = pd.cut(predicted, bins=bins).value_counts().values
    # calculates expected distribution:
    df_bins['expc'] = df_bins['org'] / np.sum(df_bins['org']) * np.sum(df_bins['pred'])
    # calculate chi2 statistic:
    stat, p_value = chisquare(df_bins['pred'], df_bins['expc'])
    # calculates correlation between original and predicted data:
    r_corr, r_corr_p = spearmanr(original, predicted)
    # calculate square errors:
    MSE = mean_squared_error(original, predicted)
    RMSE = math.sqrt(MSE)

    return stat, p_value, r_corr, r_corr_p, MSE, RMSE


def contingency(data, index, comparable, sloped=0):
    '''
    Contingency calculator function based of chi2 contingency.
    Inner function for use within accuracy finder in topological analysis.
    Returns contingency statistics. Must be runned in loop, with index accepted
    as iterator.
    '''

    # assume threshold on provided index sorted data level:
    limit = data_s[index]
    # prepare outputs:
    table = pd.DataFrame()
    # create comparable series based on original data:
    table['comp'] = comparable
    # convert sums of occurrence to occurrence flag:
    table['comp'].loc[table['comp'] >= 1] = 1
    # add model data:
    table['data'] = data
    # if sloped function is given, make sloped threshold instead of continuous,
    # keep slope value consistent with model slope:
    table['limit'] = [limit+sloped*t for t in range(0,len(table['data']))]
    # flag topological occurrence based on decision threshold:
    table['predict'] = table.apply(event_predict, axis=1)
    # mark category for contingency table:
    table['cat'] = table.apply(event_cat, axis=1)
    
    cont, acc, precision, recall, f1score = contingency_tab(table)
    
    return cont, acc, precision, recall, f1score


def contingency_tab(data):
    '''
    Contingency table calculator.
    Inner function for use within contingency functuion and test dataset 
    functions. Returns contingency statistics.
    '''
    
    # create contingency table:
    count00 = len(data['cat'].loc[data['cat']==0])
    count11 = len(data['cat'].loc[data['cat']==1])
    count01 = len(data['cat'].loc[data['cat']==2])
    count10 = len(data['cat'].loc[data['cat']==3])
    cont_in = [[count11,count10],[count01,count00]]
    
    # calculate accuracy:
    acc_in = (count11+count00)/len(data)
    # and accuracy statistics, try if no null division:
    try:
        precision_in = count11/(count11+count01)
    except:
        precision_in = 0
    try:
        recall_in = count11/(count11+count10)
    except:
        recall_in = 0
    try:
        f1score_in = 2*(precision_in*recall_in)/(precision_in+recall_in)
    except:
        f1score_in = 0
    
    return cont_in, acc_in, precision_in, recall_in, f1score_in    


def topology(input_data, comparable, slope=0):
    '''
    Performs topological analysis to find best fitted decision threshold for
    armonical model. Decides which part of function is interpretted as occurrence
    vs non-occurrence.
    
    Parameters
    ----------
    input_data : array-like
        Model predictions of continuous data. Must be same length as original.
    comparable : array-like
        Original data. Must be same length as predicted.
    slope : int/float
        Slope value for sloped models.

    Returns
    -------
    dict
        Dictionary containing accuracy statistics:
        threshold : float
            Best fitted decision threshold.
        contingency : list
            Contingency table in form of list of lists.
        accuracy : float
            Model best accuracy based on found threshold.
        precision : float
            Model best precision based on found threshold.
        recall : float
            Model best recall based on found threshold.
        f1score : float
            Model best f1 score based on found threshold.    
    '''
    
    # sort harmonic values:
    global data_s
    data_s = -np.sort(-input_data)
    
    # prepare decision list for highest f1 score:
    f1s = []
    # calculate accuracy statistics for each threshold: 
    for i in range(0,int(len(input_data))):
        cont, acc, precision, recall, f1score = contingency(input_data, i, comparable, slope)
        f1s.append(f1score)

    # find highest f1 score:
    max_index = f1s.index(max(f1s))
    # find decision threshold based on highest f1 score:
    best_limit = data_s[max_index]
    # make accuracy statistics for best fitted model:
    cont, acc, precision, recall, f1score = contingency(input_data, max_index, comparable, slope)  
    
    return {'threshold' : best_limit, 'contingency' : cont, 'accuracy' : acc, 
            'precision' : precision, 'recall' : recall, 'f1score' : f1score}

        
def selector_cat(tf, ts, sf, cf, cs):
    '''
    Selector function used on all models outputs from HOST model for categorical
    data to choose best fitted model. Accepts all five considered models.

    Parameters
    ----------
    tf : model-data
        Trend fitted model data, as output from topology.
    ts : TYPE
        Trend sloped model data, as output from topology.
    sf : TYPE
        Seasonality fitted model data, as output from topology.
    cf : TYPE
        Combined fitted model data, as output from topology.
    cs : TYPE
        Combined sloped model data, as output from topology.

    Returns
    -------
    res : dict
        Dictionary of best fitted model accuracy statistics.
    '''
    
    # compare f1 scores and find model with highest value:
    selector = [tf['f1score'], ts['f1score'], sf['f1score'], cf['f1score'], cs['f1score']] 
    best_index = selector.index(max(selector))
    # compile models:
    parameters = [tf, ts, sf, cf, cs]
    names = ['tf', 'ts', 'sf', 'cf', 'cs']
    # generate output of accuracy statistics for best fitted model:
    res = {'best_type':names[best_index],  
           'threshold':parameters[best_index]['threshold'],
           'accuracy' : parameters[best_index]['accuracy'], 
           'precision' : parameters[best_index]['precision'], 
           'recall' : parameters[best_index]['recall'], 
           'f1score' : parameters[best_index]['f1score']}
    
    return res
    

def accuracy_cont(model_data, original_data):
    '''Calling function for continuous data distribution analysis.'''
    
    # perform Mann-Whitney U test:
    mwstat, mwp_value = mannwhitneyu(original_data, model_data)
    # calculate distribution statistics for compared original and model data:
    chstat, chp, corr, corr_p, mse, rmse = chi_sq(model_data,original_data)

    return {'MWU_stat' : mwstat, 
            'MWU_p-val' : mwp_value, 
            'chi2_stat' : chstat, 
            'chi2_p-val' : chp, 
            'corr' : corr, 
            'corr_p-val' : corr_p, 
            'MSE' :  mse, 
            'RMSE' : rmse}


def selector_cont(tf, ts, sf, cf, cs):
    '''
    Selector function used on all models outputs from HOST model for continuous
    data to choose best fitted model. Accepts all five considered models.

    Parameters
    ----------
    tf : model-data
        Trend fitted model data, output from host_body.
    ts : model-data
        Trend sloped model data, output from host_body.
    sf : model-data
        Seasonality fitted model data, output from host_body.
    cf : model-data
        Combined fitted model data, output from host_body.
    cs : model-data
        Combined sloped model data, output from host_body.

    Returns
    -------
    res : dict
        Dictionary of best fitted model distribution statistics.
    '''
    
    # compare RMSE and find model with lowest value:    
    selector = [tf['RMSE'], ts['RMSE'], sf['RMSE'], cf['RMSE'], cs['RMSE']] 
    best_index = selector.index(min(selector))
    # compile models:
    parameters = [tf, ts, sf, cf, cs]
    names = ['tf', 'ts', 'sf', 'cf', 'cs']
    # generate output of distribution statistics for best fitted model:
    res = {'best_type':names[best_index],  
           'RMSE' : parameters[best_index]['RMSE'], 
           'corr' : parameters[best_index]['corr'], 
           'chi2_stat' : parameters[best_index]['chi2_stat'], 
           'MWU_stat' : parameters[best_index]['MWU_stat']}
    
    return res


def function_return(model, threshold, include_topo, *args):
    '''Function used to return function objects according to provided parameters
    and model type, allows for further generation of function values to use in 
    forcasting and prediction.
    
    Parameters
    ----------
    model : string
        String representing adjusted model. Accepted are: 'tf' (for trend), 
        'ts' (for trend sloped), 'sf' (for seasonal), 'cf' (for combined), 
        'cs' (for combined sloped).
    threshold : int/float
        Value representing found decision threshold for categorical data.
    include_topo : bool
        Boolean output modificator. If categorical data is used and decision
        threshold is present, this should be set to True to return threshold
        limit for further use with testing set. If continuous data (flow) is
        used, this should be set to False, to exclude threshold analysis.
    *args : int/float
        The model parameters provided for unpacking to returned function. The 
        general oder is as follows: slope (if applicable), amplitude, omega,
        phase, offset. If combined model is used provide first set of trend
        model parameters and then seasonal model parameters in same order. 
        See details below for further explanation:
        for ts model: slope, amp, omega, phase, offset
        for tf or sf model: amp, omega, phase, offset
        for cs model:   slope, amp (of trend), omega (of trend), 
                        phase (of trend), offset (of trend), amp (of season), 
                        omega (of season), phase (of season), offset (of season)
        for cf model:   amp (of trend), omega (of trend), phase (of trend), 
                        offset (of trend), amp (of season), omega (of season), 
                        phase (of season), offset (of season)

    Returns
    -------
    funcfit : function
        Function object built on provided model type and parameters. Function
        will require providing time variable when called. Remaining parameters
        are included based on the information provided in *args.
    limit : int/float
        Decision threshold initial value. Used only with include_topo=True.

    Exmple
    -------
    # calling trend sloped model (ts) with slope of 0.1, phase of 2 and 
    # remaining parameters as 1:   
    test = function_return('ts', 0.1, 1, 1, 2, 1)
    # generating function values in time step from 1 to 100 and creating list:
    res = []
    for i in range(1,100):
        res.append(test(i))
    # plotting the results (requires import matplotlib.pyplot as plt)
    plt.plot(res)
    plt.show()    
        
    '''     
    
    if model == 'ts':
        'parameters order: slope, amp, omega, phase, offset'
        st, At, wt, pt, ct = args
        def funcfit(t): return st * t + At * np.sin(wt*(t-pt))  + ct
        def limit(t): return st * t + threshold
        if include_topo==False:
            return funcfit
        else:
            return funcfit, limit
        
    elif model == 'tf' or model == 'sf':
        'parameters order: amp, omega, phase, offset'
        A, w, p, c = args
        def funcfit(t): return A * np.sin(w*(t-p)) + c
        def limit(t): return threshold
        if include_topo==False:
            return funcfit
        else:
            return funcfit, limit

    elif model == 'cs':
        'parameters order: slope, amp_t, omega_t, phase_t, offset_t, amp_s, omega_s, phase_s, offset_s'
        st, At, wt, pt, ct, As, ws, ps, cs = args
        def funcfit(t): return st * t + At * np.sin(wt*(t-pt)) + ct + As * np.sin(ws*(t-ps)) + cs
        def limit(t): return st * t + threshold
        if include_topo==False:
            return funcfit
        else:
            return funcfit, limit
        
    elif model == 'cf':
        'parameters order: amp_t, omega_t, phase_t, offset_t, amp_s, omega_s, phase_s, offset_s'
        At, wt, pt, ct, As, ws, ps, cs = args
        def funcfit(t): return At * np.sin(wt*(t-pt)) + ct + As * np.sin(ws*(t-ps)) + cs
        def limit(t): return threshold
        if include_topo==False:
            return funcfit
        else:
            return funcfit, limit