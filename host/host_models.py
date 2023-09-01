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
import scipy.optimize
import host_functions

def fit_sine(x, y, include_slope=False, repeats = 500000, efficiency='kge',use_bounds=False):
    '''
    Function to fit simple harmonic oscillator to the input data. 
    Requires preprocessed, aggregated data. 
    Returns function parameters and statistics. 

    Parameters
    ----------
    x : array of int
        x-axis time factor.
    y : array of float/int
        y-axis variable like flow or occurrence information.
    include_slope : bool
        True/False value determining if calculated function should include slope
        factor. 
        Cefault is False.
    repeats : int, optional
        maximal number of function calls to fit harmonic to data. Increase to
        try to fit to complicated data. May increase computation cost and time.
        Default is 500000.
    efficiency: str, default: 'kge'
        the efficiency statistic to use when comparing flow distributions.
        The default 'kge' calls for Kling-Gupta efficiency. Other accepted
        option is 'nse' for Nash-Sutcliffe efficiency. 
    use_bounds: bool, default: False
        boolean flag allowing to control fitting process. Default False means
        all parameter values are possible, including negative frequencies 
        (periods). Set to True to lock to only positive frequencies. Using 
        bounds might slower the optimization and lower fit efficiencies. 
        
    Raises
    ------
    Exception    
        if repeats is to low and the function can't be fit to dataset raises
        information exception.
    
    Returns
    -------
    dict
        the dictionary containing fitted function parameters: 'amp', 'phase', 
        'offset', 'freq', 'period', 'slope' (optional) as well as fit statistics 
        'r2' (percentage of original data variance explained by model), 
        'kge' (Kling-Gupta Efficiency), and 'predictions' (predicted values), 
        and 'function' (fitted function object).
    '''
    
    # prepare initial parameters
    x = np.array(x)
    y = np.array(y)
    # try to guess initial amplitude
    guess_a = np.std(y) * 2**0.5     
    # try to guess initial frequency
    ff = np.fft.fftfreq(len(x), (x[1]-x[0]))
    Fyy = abs(np.fft.fft(y))    
    guess_f = abs(ff[np.argmax(Fyy[1:])+1])    
    # try to guess initial phase
    try:
        guess_p = scipy.signal.find_peaks(y)[0][0]/len(x)
    except:
        guess_p = 0
    # try to guess initial offset
    guess_c = np.mean(y)
    # control result function by including/excluding slope 
    if include_slope==True:
        # try to guess initial slope
        guess_s = np.polyfit(x,y,1)[0]
        # array of initial parameters including slope
        guess = np.array([guess_a, guess_f, guess_p, guess_c, guess_s])
        # define simple harmonic function
        def sine(x, A, f, p, c, s):  
            return s * x + A * np.sin(2 * np.pi * f * x + p) + c  
        bound_low = [-(max(y)),0,-len(y),-max(y),-np.inf]
        bound_high = [(max(y)),10000,len(y),max(y),np.inf]
    else:
        # array of initial parameters
        guess = np.array([guess_a, guess_f, guess_p, guess_c])  
        def sine(x, A, f, p, c):  
            return A * np.sin(2 * np.pi * f * x + p) + c       
        bound_low = [-(max(y)),0,-len(y),-max(y)]
        bound_high = [(max(y)),10000,len(y),max(y)]
    # fit function curve and optimize parameters by least squares method
    try:
        if use_bounds==True:
            parameters, covariance = scipy.optimize.curve_fit(sine, x, y, p0=guess, 
                                                              maxfev = repeats, 
                                                              bounds=(bound_low,bound_high))
        else:
            parameters, covariance = scipy.optimize.curve_fit(sine, x, y, p0=guess, 
                                                              maxfev = repeats)
    except:
        raise Exception("function can't be optimized with defined function calls \
                        you can try increasing 'repeats' parameter; \n \
                        however, it is recommended to check the data first, \
                        as some situations might increase optimization difficulty, \
                        f.e. constant data. \n \
                        Please refer to 'maxfev' parameter in 'scipy.optimize.curve_fit' \
                        for more information.")
    # asign parameters values to variables
    if include_slope==True:
        A, f, p, c, s = parameters
    else:
        A, f, p, c = parameters
    # generate predicted values
    predictions = sine(x, *parameters)
    # calculate variance explained by model as r2
    residuals = y - predictions
    square_res = np.sum(residuals**2)
    square_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (square_res / square_tot)
    # calculate efficiency
    if efficiency == 'kge':
        eff_value = host_functions.efficiency_stat(predictions, y, 'kge')
    elif efficiency == 'nse':
        eff_value = host_functions.efficiency_stat(predictions, y, 'nse')
    else:
        raise SyntaxError('Only "kge" or "nse" are supported as valid efficiency metrics')
    
    results = {"amp": A, "freq": f, "phase": p, "offset": c, "period": 1/f, "r2": r2, "predictions": predictions}
    if efficiency == 'kge':
        results["kge"] = eff_value
    else:
        results["nse"] = eff_value
        
    # return control to generate function and/or predictions
    if include_slope==True:
        function = lambda x: s * x + A * np.sin(2 * np.pi * f * x + p) + c
        equation = 's·x+A·sin(2·\u03C0·f·x+p)+c'
        results["slope"] = s
    else:
        function = lambda x: A * np.sin(2 * np.pi * f * x + p) + c
        equation = 'A·sin(2·\u03C0·f·x+p)+c'        
        
    results["equation"] = equation
    results["function"] = function
        
    return results
    

def fit_damped(x, y, include_slope=False, repeats = 500000, efficiency='kge',use_bounds=False):
    '''
    Function to fit dumped sine to the input data. Requires preprocessed, 
    aggregated data. Returns function parameters and statistics. 

    Parameters
    ----------
    x : array of int
        x-axis time factor.
    y : array of float/int
        y-axis variable like flow or occurrence information.
    include_slope : bool
        True/False value determining if calculated function should include slope
        factor. 
        Cefault is False.
    repeats : int, optional
        maximal number of function calls to fit harmonic to data. Increase to
        try to fit to complicated data. May increase computation cost and time.
        Default is 500000.
    efficiency: str, default: 'kge'
        the efficiency statistic to use when comparing flow distributions.
        The default 'kge' calls for Kling-Gupta efficiency. Other accepted
        option is 'nse' for Nash-Sutcliffe efficiency. 
    use_bounds: bool, default: False
        boolean flag allowing to control fitting process. Default False means
        all parameter values are possible, including negative frequencies 
        (periods). Set to True to lock to only positive frequencies. Using 
        bounds might slower the optimization and lower fit efficiencies. 
        
    Raises
    ------
    Exception    
        if repeats is to low and the function can't be fit to dataset raises
        information exception
    
    Returns
    -------
    dict
        the dictionary containing fitted function parameters: 'amp', 'phase', 
        'offset', 'freq', 'period', 'damping_factor', 'slope' (optional) 
        as well as fit statistics 'r2' (percentage of original data variance explained by model), 
        'kge' (Kling-Gupta Efficiency), and 'predictions' (predicted values), 
        and 'function' (fitted function object).
    '''

    # prepare initial parameters
    x = np.array(x)
    y = np.array(y)
    # try to guess initial amplitude
    guess_a = np.std(y) * 2**0.5     
    # try to guess initial frequency
    ff = np.fft.fftfreq(len(x), (x[1]-x[0]))
    Fyy = abs(np.fft.fft(y))    
    guess_f = abs(ff[np.argmax(Fyy[1:])+1])    
    # try to guess initial phase
    try:
        guess_p = scipy.signal.find_peaks(y)[0][0]/len(x)
    except:
        guess_p = 0
    # try to guess initial offset
    guess_c = np.mean(y)
    # try to guess initial damping factor
    guess_d = guess_f*np.log(guess_a/(np.std(y[int(len(y)/2):])*2**0.5))
    # control result function by including/excluding slope 
    if include_slope==True:
        # try to guess initial slope
        guess_s = np.polyfit(x,y,1)[0]
        # array of initial parameters including slope
        guess = np.array([guess_a, guess_f, guess_p, guess_d, guess_c, guess_s])
        # define simple harmonic function
        def sinedamped(x, A, f, p, d, c, s):  
            return s * x + A * np.exp((-d) * x) * np.sin(2 * np.pi * f * x + p) + c
        bound_low = [-(max(y)),0,-len(y),-np.inf,-max(y),-np.inf]
        bound_high = [(max(y)),10000,len(y),np.inf,max(y),np.inf]
    else:
        # array of initial parameters
        guess = np.array([guess_a, guess_f, guess_p, guess_d, guess_c])  
        def sinedamped(x, A, f, p, d, c):  
            return A * np.exp((-d) * x) * np.sin(2 * np.pi * f * x + p) + c
        bound_low = [-(max(y)),0,-len(y),-np.inf,-max(y)]
        bound_high = [(max(y)),10000,len(y),np.inf,max(y)]
    # fit function curve and optimize parameters by least squares method
    try:
        if use_bounds==True:
            parameters, covariance = scipy.optimize.curve_fit(sinedamped, x, y, 
                                                          p0=guess, 
                                                          maxfev = repeats,
                                                          bounds=(bound_low,bound_high))
        else:
            parameters, covariance = scipy.optimize.curve_fit(sinedamped, x, y, 
                                                          p0=guess, 
                                                          maxfev = repeats)
    except:
        raise Exception("function can't be optimized with defined function calls \
                        you can try increasing 'repeats' parameter; \n \
                        however, it is recommended to check the data first, \
                        as some situations might increase optimization difficulty, \
                        f.e. constant data. \n \
                        Please refer to 'maxfev' parameter in 'scipy.optimize.curve_fit' \
                        for more information.")
    # asign parameters values to variables
    if include_slope==True:
        A, f, p, d, c, s = parameters
    else:
        A, f, p, d, c = parameters
    # generate predicted values
    predictions = sinedamped(x, *parameters)
    # calculate variance explained by model as r2
    residuals = y - predictions
    square_res = np.sum(residuals**2)
    square_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (square_res / square_tot)
    # calculate efficiency
    if efficiency == 'kge':
        eff_value = host_functions.efficiency_stat(predictions, y, 'kge')
    elif efficiency == 'nse':
        eff_value = host_functions.efficiency_stat(predictions, y, 'nse')
    else:
        raise SyntaxError('Only "kge" or "nse" are supported as valid efficiency metrics')

    results = {"amp": A, "freq": f, "phase": p, "offset": c, "damping_factor": d, "period": 1/f, "r2": r2, "predictions": predictions} 
    if efficiency == 'kge':
        results["kge"] = eff_value
    else:
        results["nse"] = eff_value

    # return control to generate function and/or predictions
    if include_slope==True:
        function = lambda x: s * x + A * np.exp((-d) * x) * np.sin(2 * np.pi * f * x + p) + c
        equation = 's·x+A·exp((-d)·x)·sin(2·\u03C0·f·x+p)+c'
        results["slope"] = s        
    else:
        function = lambda x: A * np.exp((-d) * x) * np.sin(2 * np.pi * f * x + p) + c
        equation = 'A·exp((-d)·x)·sin(2·\u03C0·f·x+p)+c'

    results["equation"] = equation
    results["function"] = function
        
    return results


def fit_amplitude_mod(x, y, include_slope=False, repeats = 500000, efficiency='kge',use_bounds=False):
    '''
    Function to fit amplitude modulated sine to the input data. 
    Requires preprocessed, aggregated data. 
    Returns function parameters and statistics. 

    Parameters
    ----------
    x : array of int
        x-axis time factor.
    y : array of float/int
        y-axis variable like flow or occurrence information.
    include_slope : bool
        True/False value determining if calculated function should include slope
        factor. 
        Cefault is False.
    repeats : int, optional
        maximal number of function calls to fit harmonic to data. Increase to
        try to fit to complicated data. May increase computation cost and time.
        Default is 500000.
    efficiency: str, default: 'kge'
        the efficiency statistic to use when comparing flow distributions.
        The default 'kge' calls for Kling-Gupta efficiency. Other accepted
        option is 'nse' for Nash-Sutcliffe efficiency. 
    use_bounds: bool, default: False
        boolean flag allowing to control fitting process. Default False means
        all parameter values are possible, including negative frequencies 
        (periods). Set to True to lock to only positive frequencies. Using 
        bounds might slower the optimization and lower fit efficiencies. 
        
    Raises
    ------
    Exception    
        if repeats is to low and the function can't be fit to dataset raises
        information exception.
    
    Returns
    -------
    dict
        the dictionary containing fitted function parameters: 'amp', 'beta', 
        'offset', 'freq', 'period', 'slope' (optional) as well as fit statistics 
        'r2' (percentage of original data variance explained by model), 
        'kge' (Kling-Gupta Efficiency), and 'predictions' (predicted values), 
        and 'function' (fitted function object).
    '''
    
    # prepare initial parameters
    x = np.array(x)
    y = np.array(y)
    # try to guess initial amplitude
    guess_a = np.std(y) * 2**0.5     
    # try to guess initial frequency
    ff = np.fft.fftfreq(len(x), (x[1]-x[0]))
    Fyy = abs(np.fft.fft(y))    
    guess_f = abs(ff[np.argmax(Fyy[1:])+1])    
    # try to guess initial offset
    guess_c = np.mean(y)
    # control result function by including/excluding slope 
    if include_slope==True:
        # try to guess initial slope
        guess_s = np.polyfit(x,y,1)[0]
        # array of initial parameters including slope
        guess = np.array([guess_a, guess_f, 0, guess_c, guess_s])
        # define simple harmonic function
        def ampmod(x, A, f, B, c, s):  
            return s * x + A * np.sin(f * x) * (1 + B * np.sin(2 * np.pi * f * x)) + c
        bound_low = [-(max(y)),0,-len(y),-max(y),-np.inf]
        bound_high = [(max(y)),10000,len(y),max(y),np.inf]
    else:
        # array of initial parameters
        guess = np.array([guess_a, guess_f, 0, guess_c])  
        def ampmod(x, A, f, B, c):  
            return A * np.sin(f * x) * (1 + B * np.sin(2 * np.pi * f * x)) + c
        bound_low = [-(max(y)),0,-len(y),-max(y)]
        bound_high = [(max(y)),10000,len(y),max(y)]
    # fit function curve and optimize parameters by least squares method
    try:
        if use_bounds==True:
            parameters, covariance = scipy.optimize.curve_fit(ampmod, x, y, p0=guess, 
                                                          maxfev = repeats,
                                                          bounds=(bound_low,bound_high))
        else:
            parameters, covariance = scipy.optimize.curve_fit(ampmod, x, y, p0=guess, 
                                                          maxfev = repeats)
    except:
        raise Exception("function can't be optimized with defined function calls \
                        you can try increasing 'repeats' parameter; \n \
                        however, it is recommended to check the data first, \
                        as some situations might increase optimization difficulty, \
                        f.e. constant data. \n \
                        Please refer to 'maxfev' parameter in 'scipy.optimize.curve_fit' \
                        for more information.")
    # asign parameters values to variables
    if include_slope==True:
        A, f, B, c, s = parameters
    else:
        A, f, B, c = parameters
    # generate predicted values
    predictions = ampmod(x, *parameters)
    # calculate variance explained by model as r2
    residuals = y - predictions
    square_res = np.sum(residuals**2)
    square_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (square_res / square_tot)
    # calculate efficiency
    if efficiency == 'kge':
        eff_value = host_functions.efficiency_stat(predictions, y, 'kge')
    elif efficiency == 'nse':
        eff_value = host_functions.efficiency_stat(predictions, y, 'nse')
    else:
        raise SyntaxError('Only "kge" or "nse" are supported as valid efficiency metrics')
    
    results = {"amp": A, "freq": f, "beta": B, "offset": c, "period": 1/f, "r2": r2, "predictions": predictions}
    if efficiency == 'kge':
        results["kge"] = eff_value
    else:
        results["nse"] = eff_value    
    
    # return control to generate function and/or predictions
    if include_slope==True:
        function = lambda x: s * x + A * np.sin(f * x) * (1 + B * np.sin(2 * np.pi * f * x)) + c
        equation = 's·x+A·sin(f·x)·(1+B·sin(2·\u03C0·f·x))+c'
        results["slope"] = s  
    else:
        function = lambda x: A * np.sin(f * x) * (1 + B * np.sin(2 * np.pi * f * x)) + c
        equation = 'A·sin(f·x)·(1+B·sin(2·\u03C0·f·x))+c'

    results["equation"] = equation
    results["function"] = function
        
    return results


def fit_frequency_mod(x, y, include_slope=False, repeats = 500000, efficiency='kge',use_bounds=False):
    '''
    Function to fit frequency modulated sine to the input data. 
    Requires preprocessed, aggregated data. 
    Returns function parameters and statistics. 

    Parameters
    ----------
    x : array of int
        x-axis time factor.
    y : array of float/int
        y-axis variable like flow or occurrence information.
    include_slope : bool
        True/False value determining if calculated function should include slope
        factor. 
        Cefault is False.
    repeats : int, optional
        maximal number of function calls to fit harmonic to data. Increase to
        try to fit to complicated data. May increase computation cost and time.
        Default is 500000.
    efficiency: str, default: 'kge'
        the efficiency statistic to use when comparing flow distributions.
        The default 'kge' calls for Kling-Gupta efficiency. Other accepted
        option is 'nse' for Nash-Sutcliffe efficiency. 
    use_bounds: bool, default: False
        boolean flag allowing to control fitting process. Default False means
        all parameter values are possible, including negative frequencies 
        (periods). Set to True to lock to only positive frequencies. Using 
        bounds might slower the optimization and lower fit efficiencies. 
        
    Raises
    ------
    Exception    
        if repeats is to low and the function can't be fit to dataset raises
        information exception.
    
    Returns
    -------
    dict
        the dictionary containing fitted function parameters: 'amp', 'phase', 
        'offset', 'freq', 'period', 'modulation_index': m, 'slope' (optional) 
        as well as fit statistics 'r2' (percentage of original data variance 
        explained by model), 'kge' (Kling-Gupta Efficiency), and 'predictions' 
        (predicted values), and 'function' (fitted function object).
    '''
    
    # prepare initial parameters
    x = np.array(x)
    y = np.array(y)
    # try to guess initial amplitude
    guess_a = np.std(y) * 2**0.5     
    # try to guess initial frequency
    ff = np.fft.fftfreq(len(x), (x[1]-x[0]))
    Fyy = abs(np.fft.fft(y))    
    guess_f = abs(ff[np.argmax(Fyy[1:])+1])  
    # try to guess initial phase
    try:
        guess_p = scipy.signal.find_peaks(y)[0][0]/len(x)
    except:
        guess_p = 0
    # try to guess initial offset
    guess_c = np.mean(y)
    # control result function by including/excluding slope 
    if include_slope==True:
        # try to guess initial slope
        guess_s = np.polyfit(x,y,1)[0]
        # array of initial parameters including slope
        guess = np.array([guess_a, guess_f, guess_p, 0, guess_c, guess_s])
        # define simple harmonic function
        def freqmod(x, A, f, p, m, c, s):  
            return s * x + A * np.sin(2 * np.pi * (f * x + m * np.cos(2 * np.pi * f * x) + p)) + c
        bound_low = [-(max(y)),0,-len(y),-np.inf,-max(y),-np.inf]
        bound_high = [(max(y)),10000,len(y),np.inf,max(y),np.inf]
    else:
        # array of initial parameters
        guess = np.array([guess_a, guess_f, guess_p, 0, guess_c])  
        def freqmod(x, A, f, p, m, c):  
            return A * np.sin(2 * np.pi * (f * x + m * np.cos(2 * np.pi * f * x) + p)) + c
        bound_low = [-(max(y)),0,-len(y),-np.inf,-max(y)]
        bound_high = [(max(y)),10000,len(y),np.inf,max(y)]
    # fit function curve and optimize parameters by least squares method
    try:
        if use_bounds==True:
            parameters, covariance = scipy.optimize.curve_fit(freqmod, x, y, p0=guess, 
                                                          maxfev = repeats,
                                                          bounds=(bound_low,bound_high))
        else:
            parameters, covariance = scipy.optimize.curve_fit(freqmod, x, y, p0=guess, 
                                                              maxfev = repeats)
    except:
        raise Exception("function can't be optimized with defined function calls \
                        you can try increasing 'repeats' parameter; \n \
                        however, it is recommended to check the data first, \
                        as some situations might increase optimization difficulty, \
                        f.e. constant data. \n \
                        Please refer to 'maxfev' parameter in 'scipy.optimize.curve_fit' \
                        for more information.")
    # asign parameters values to variables
    if include_slope==True:
        A, f, p, m, c, s = parameters
    else:
        A, f, p, m, c = parameters
    # generate predicted values
    predictions = freqmod(x, *parameters)
    # calculate variance explained by model as r2
    residuals = y - predictions
    square_res = np.sum(residuals**2)
    square_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (square_res / square_tot)
    # calculate efficiency
    if efficiency == 'kge':
        eff_value = host_functions.efficiency_stat(predictions, y, 'kge')
    elif efficiency == 'nse':
        eff_value = host_functions.efficiency_stat(predictions, y, 'nse')
    else:
        raise SyntaxError('Only "kge" or "nse" are supported as valid efficiency metrics')
    
    results = {"amp": A, "freq": f, "phase": p, "offset": c, 
            "modulation_index": m, "period": 1/f, "r2": r2, 
            "predictions": predictions}
    if efficiency == 'kge':
        results["kge"] = eff_value
    else:
        results["nse"] = eff_value    
    
    # return control to generate function and/or predictions
    if include_slope==True:
        function = lambda x: s * x + A * np.sin(2 * np.pi * (f * x + m * np.cos(2 * np.pi * f * x) + p)) + c
        equation = 's·x+A·sin(2·\u03C0·(f·x+m·np.cos(2·\u03C0·f·x)+p))+c'
        results["slope"] = s
    else:
        function = lambda x: A * np.sin(2 * np.pi * (f * x + m * np.cos(2 * np.pi * f * x) + p)) + c
        equation = 'A·sin(2·\u03C0·(f·x+m·np.cos(2·\u03C0·f·x)+p))+c'

    results["equation"] = equation
    results["function"] = function
        
    return results


def fit_modulated(x, y, include_slope=False, repeats = 500000, efficiency='kge',use_bounds=False):
    '''
    Function to fit amplitude and frequency modulated sine to the input data. 
    Requires preprocessed, aggregated data. 
    Returns function parameters and statistics. 

    Parameters
    ----------
    x : array of int
        x-axis time factor.
    y : array of float/int
        y-axis variable like flow or occurrence information.
    include_slope : bool
        True/False value determining if calculated function should include slope
        factor. 
        Cefault is False.
    repeats : int, optional
        maximal number of function calls to fit harmonic to data. Increase to
        try to fit to complicated data. May increase computation cost and time.
        Default is 500000.
    efficiency: str, default: 'kge'
        the efficiency statistic to use when comparing flow distributions.
        The default 'kge' calls for Kling-Gupta efficiency. Other accepted
        option is 'nse' for Nash-Sutcliffe efficiency. 
    use_bounds: bool, default: False
        boolean flag allowing to control fitting process. Default False means
        all parameter values are possible, including negative frequencies 
        (periods). Set to True to lock to only positive frequencies. Using 
        bounds might slower the optimization and lower fit efficiencies. 
        
    Raises
    ------
    Exception    
        if repeats is to low and the function can't be fit to dataset raises
        information exception.
    
    Returns
    -------
    dict
        the dictionary containing fitted function parameters: 'amp', 'phase', 
        'offset', 'freq', 'beta', 'modulation_index', 'period', 'slope' (optional) 
        as well as fit statistics 'r2' (percentage of original data variance 
        explained by model), 'kge' (Kling-Gupta Efficiency), and 'predictions' 
        (predicted values), and 'function' (fitted function object).
    '''

    # prepare initial parameters
    x = np.array(x)
    y = np.array(y)
    # try to guess initial amplitude
    guess_a = np.std(y) * 2**0.5     
    # try to guess initial frequency
    ff = np.fft.fftfreq(len(x), (x[1]-x[0]))
    Fyy = abs(np.fft.fft(y))    
    guess_f = abs(ff[np.argmax(Fyy[1:])+1])  
    # try to guess initial phase
    try:
        guess_p = scipy.signal.find_peaks(y)[0][0]/len(x)
    except:
        guess_p = 0
    # try to guess initial offset
    guess_c = np.mean(y)
    # control result function by including/excluding slope 
    if include_slope==True:
        # try to guess initial slope
        guess_s = np.polyfit(x,y,1)[0]
        # array of initial parameters including slope
        guess = np.array([guess_a, guess_f, guess_p, 0, 0, guess_c, guess_s])
        # define simple harmonic function
        def modulated(x, A, f, p, B, m, c, s):  
            return s * x + A * np.sin(f * x) * (1 + B * np.sin(2 * np.pi * (f * x + m * np.cos(2 * np.pi * f * x) + p))) + c
        bound_low = [-(max(y)),0,-len(y),-np.inf,-np.inf,-max(y),-np.inf]
        bound_high = [(max(y)),10000,len(y),np.inf,np.inf,max(y),np.inf]
    else:
        # array of initial parameters
        guess = np.array([guess_a, guess_f, guess_p, 0, 0, guess_c])  
        def modulated(x, A, f, p, B, m, c):  
            return A * np.sin(f * x) * (1 + B * np.sin(2 * np.pi * (f * x + m * np.cos(2 * np.pi * f * x) + p))) + c
        bound_low = [-(max(y)),0,-len(y),-np.inf,-np.inf,-max(y)]
        bound_high = [(max(y)),10000,len(y),np.inf,np.inf,max(y)]
    # fit function curve and optimize parameters by least squares method
    try:
        if use_bounds==True:
            parameters, covariance = scipy.optimize.curve_fit(modulated, x, y, p0=guess, 
                                                          maxfev = repeats,
                                                          bounds=(bound_low,bound_high))
        else:
            parameters, covariance = scipy.optimize.curve_fit(modulated, x, y, p0=guess, 
                                                          maxfev = repeats)
    except:
        raise Exception("function can't be optimized with defined function calls \
                        you can try increasing 'repeats' parameter; \n \
                        however, it is recommended to check the data first, \
                        as some situations might increase optimization difficulty, \
                        f.e. constant data. \n \
                        Please refer to 'maxfev' parameter in 'scipy.optimize.curve_fit' \
                        for more information.")
    # asign parameters values to variables
    if include_slope==True:
        A, f, p, B, m, c, s = parameters
    else:
        A, f, p, B, m, c = parameters
    # generate predicted values
    predictions = modulated(x, *parameters)
    # calculate variance explained by model as r2
    residuals = y - predictions
    square_res = np.sum(residuals**2)
    square_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (square_res / square_tot)
    # calculate efficiency
    if efficiency == 'kge':
        eff_value = host_functions.efficiency_stat(predictions, y, 'kge')
    elif efficiency == 'nse':
        eff_value = host_functions.efficiency_stat(predictions, y, 'nse')
    else:
        raise SyntaxError('Only "kge" or "nse" are supported as valid efficiency metrics')

    results = {"amp": A, "freq": f, "phase": p, "beta": B, "modulation_index": m, "offset": c, "period": 1/f, "r2": r2, "predictions": predictions}
    if efficiency == 'kge':
        results["kge"] = eff_value
    else:
        results["nse"] = eff_value
    
    # return control to generate function and/or predictions
    if include_slope==True:
        function = lambda x: s * x + A * np.sin(f * x) * (1 + B * np.sin(2 * np.pi * (f * x + m * np.cos(2 * np.pi * f * x) + p))) + c
        equation = 's·x+A·sin(f·x)·(1+B·sin(2·\u03C0·(f·x+m·cos(2·\u03C0·f·x)+p)))+c'
        results["slope"] = s       
    else:
        function = lambda x: A * np.sin(f * x) * (1 + B * np.sin(2 * np.pi * (f * x + m * np.cos(2 * np.pi * f * x) + p))) + c
        equation = 'A·sin(f·x)·(1+B·sin(2·\u03C0·(f·x+m·cos(2·\u03C0·f·x)+p)))+c'
 
    results["equation"] = equation
    results["function"] = function
        
    return results


def fit_damped_mod(x, y, include_slope=False, repeats = 500000, efficiency='kge',use_bounds=False):
    '''
    Function to fit damped amplitude and frequency modulated sine to the input data. 
    Requires preprocessed, aggregated data. 
    Returns function parameters and statistics. 

    Parameters
    ----------
    x : array of int
        x-axis time factor.
    y : array of float/int
        y-axis variable like flow or occurrence information.
    include_slope : bool
        True/False value determining if calculated function should include slope
        factor. 
        Cefault is False.
    repeats : int, optional
        maximal number of function calls to fit harmonic to data. Increase to
        try to fit to complicated data. May increase computation cost and time.
        Default is 500000.
    efficiency: str, default: 'kge'
        the efficiency statistic to use when comparing flow distributions.
        The default 'kge' calls for Kling-Gupta efficiency. Other accepted
        option is 'nse' for Nash-Sutcliffe efficiency. 
    use_bounds: bool, default: False
        boolean flag allowing to control fitting process. Default False means
        all parameter values are possible, including negative frequencies 
        (periods). Set to True to lock to only positive frequencies. Using 
        bounds might slower the optimization and lower fit efficiencies. 
        
    Raises
    ------
    Exception    
        if repeats is to low and the function can't be fit to dataset raises
        information exception.
    
    Returns
    -------
    dict
        the dictionary containing fitted function parameters: 'amp', 'phase', 
        'offset', 'freq', 'beta', 'modulation_index', 'damping_factor', 'period', 'slope' (optional) 
        as well as fit statistics 'r2' (percentage of original data variance 
        explained by model), 'kge' (Kling-Gupta Efficiency), and 'predictions' 
        (predicted values), and 'function' (fitted function object).
    '''

    # prepare initial parameters
    x = np.array(x)
    y = np.array(y)
    # try to guess initial amplitude
    guess_a = np.std(y) * 2**0.5     
    # try to guess initial frequency
    ff = np.fft.fftfreq(len(x), (x[1]-x[0]))
    Fyy = abs(np.fft.fft(y))    
    guess_f = abs(ff[np.argmax(Fyy[1:])+1])  
    # try to guess initial phase
    try:
        guess_p = scipy.signal.find_peaks(y)[0][0]/len(x)
    except:
        guess_p = 0
    # try to guess initial offset
    guess_c = np.mean(y)
    # try to guess initial damping factor
    guess_d = guess_f*np.log(guess_a/(np.std(y[int(len(y)/2):])*2**0.5))
    # control result function by including/excluding slope 
    if include_slope==True:
        # try to guess initial slope
        guess_s = np.polyfit(x,y,1)[0]
        # array of initial parameters including slope
        guess = np.array([guess_a, guess_f, guess_p, guess_d, 0, 0, guess_c, guess_s])
        # define simple harmonic function
        def dampedmod(x, A, f, p, d, B, m, c, s):  
            return s * x + A * np.exp((-d) * x) * np.sin(f * x) * (1 + B * np.sin(2 * np.pi * (f * x + m * np.cos(2 * np.pi * f * x) + p))) + c
        bound_low = [-(max(y)),0,-len(y),-np.inf,-np.inf,-np.inf,-max(y),-np.inf]
        bound_high = [(max(y)),10000,len(y),np.inf,np.inf,np.inf,max(y),np.inf]
    else:
        # array of initial parameters
        guess = np.array([guess_a, guess_f, guess_p, guess_d, 0, 0, guess_c])  
        def dampedmod(x, A, f, p, d, B, m, c):  
            return A * np.exp((-d) * x) * np.sin(f * x) * (1 + B * np.sin(2 * np.pi * (f * x + m * np.cos(2 * np.pi * f * x) + p))) + c
        bound_low = [-(max(y)),0,-len(y),-np.inf,-np.inf,-np.inf,-max(y)]
        bound_high = [(max(y)),10000,len(y),np.inf,np.inf,np.inf,max(y)]
    # fit function curve and optimize parameters by least squares method
    try:
        if use_bounds==True:
            parameters, covariance = scipy.optimize.curve_fit(dampedmod, x, y, p0=guess, 
                                                          maxfev = repeats,
                                                          bounds=(bound_low,bound_high))
        else:
            parameters, covariance = scipy.optimize.curve_fit(dampedmod, x, y, p0=guess, 
                                                          maxfev = repeats)
    except:
        raise Exception("function can't be optimized with defined function calls \
                        you can try increasing 'repeats' parameter; \n \
                        however, it is recommended to check the data first, \
                        as some situations might increase optimization difficulty, \
                        f.e. constant data. \n \
                        Please refer to 'maxfev' parameter in 'scipy.optimize.curve_fit' \
                        for more information.")
    # asign parameters values to variables
    if include_slope==True:
        A, f, p, d, B, m, c, s = parameters
    else:
        A, f, p, d, B, m, c = parameters
    # generate predicted values
    predictions = dampedmod(x, *parameters)
    # calculate variance explained by model as r2
    residuals = y - predictions
    square_res = np.sum(residuals**2)
    square_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (square_res / square_tot)
    # calculate efficiency
    if efficiency == 'kge':
        eff_value = host_functions.efficiency_stat(predictions, y, 'kge')
    elif efficiency == 'nse':
        eff_value = host_functions.efficiency_stat(predictions, y, 'nse')
    else:
        raise SyntaxError('Only "kge" or "nse" are supported as valid efficiency metrics')

    results = {"amp": A, "freq": f, "phase": p, "beta": B, "modulation_index": m, "damping_factor": d, "offset": c, "period": 1/f, "r2": r2, "predictions": predictions}
    if efficiency == 'kge':
        results["kge"] = eff_value
    else:
        results["nse"] = eff_value
    
    # return control to generate function and/or predictions
    if include_slope==True:
        function = lambda x: s * x + A * np.exp((-d) * x) * np.sin(f * x) * (1 + B * np.sin(2 * np.pi * (f * x + m * np.cos(2 * np.pi * f * x) + p))) + c
        equation = 's·x+A·exp((-d)·x)·sin(f·x)·(1+B·sin(2·\u03C0·(f·x+m·cos(2·\u03C0·f·x)+p)))+c'
        results["slope"] = s       
    else:
        function = lambda x: A * np.exp((-d) * x) * np.sin(f * x) * (1 + B * np.sin(2 * np.pi * (f * x + m * np.cos(2 * np.pi * f * x) + p))) + c
        equation = 'A·exp((-d)·x)·sin(f·x)·(1+B·sin(2·\u03C0·(f·x+m·cos(2·\u03C0·f·x)+p)))+c'

    results["equation"] = equation
    results["function"] = function
        
    return results    
    

def fit_incdec_mod(x, y, include_slope=False, repeats = 500000, efficiency='kge',use_bounds=False):
    '''
    Function to fit amplitude and frequency modulated period changing sine to 
    the input data. 
    Requires preprocessed, aggregated data. 
    Returns function parameters and statistics. 

    Parameters
    ----------
    x : array of int
        x-axis time factor.
    y : array of float/int
        y-axis variable like flow or occurrence information.
    include_slope : bool
        True/False value determining if calculated function should include slope
        factor. 
        Cefault is False.
    repeats : int, optional
        maximal number of function calls to fit harmonic to data. Increase to
        try to fit to complicated data. May increase computation cost and time.
        Default is 500000.
    efficiency: str, default: 'kge'
        the efficiency statistic to use when comparing flow distributions.
        The default 'kge' calls for Kling-Gupta efficiency. Other accepted
        option is 'nse' for Nash-Sutcliffe efficiency. 
    use_bounds: bool, default: False
        boolean flag allowing to control fitting process. Default False means
        all parameter values are possible, including negative frequencies 
        (periods). Set to True to lock to only positive frequencies. Using 
        bounds might slower the optimization and lower fit efficiencies. 
        
    Raises
    ------
    Exception    
        if repeats is to low and the function can't be fit to dataset raises
        information exception.
    
    Returns
    -------
    dict
        the dictionary containing fitted function parameters: 'amp', 'phase', 
        'offset', 'freq', 'beta', 'modulation_index', 'damping_factor', 'period', 'slope' (optional) 
        as well as fit statistics 'r2' (percentage of original data variance 
        explained by model), 'kge' (Kling-Gupta Efficiency), and 'predictions' 
        (predicted values), and 'function' (fitted function object).
    '''

    # prepare initial parameters
    x = np.array(x)
    y = np.array(y)
    # try to guess initial amplitude
    guess_a = np.std(y) * 2**0.5     
    # try to guess initial frequency
    ff = np.fft.fftfreq(len(x), (x[1]-x[0]))
    Fyy = abs(np.fft.fft(y))    
    guess_f = abs(ff[np.argmax(Fyy[1:])+1])  
    # try to guess initial phase
    try:
        guess_p = scipy.signal.find_peaks(y)[0][0]/len(x)
    except:
        guess_p = 0
    # try to guess initial offset
    guess_c = np.mean(y)
    # control result function by including/excluding slope 
    if include_slope==True:
        # try to guess initial slope
        guess_s = np.polyfit(x,y,1)[0]
        # array of initial parameters including slope
        guess = np.array([guess_a, guess_f, guess_p, guess_c, guess_s, 0, 0, 0])
        # define simple harmonic function
        def sinedecreasing(x, A, f, p, c, s, B, m, a):  
            return s * x + A * np.sin(f * x) * (1 + B * np.sin(2 * np.pi * (f * x + m * np.cos(2 * np.pi * f * np.exp(-a * x) * x) + p))) + c 
        def sineincreasing(x, A, f, p, c, s, B, m, a):  
            return s * x + A * np.sin(f * x) * (1 + B * np.sin(2 * np.pi * (f * x + m * np.cos(2 * np.pi * (f+a*x) * x) + p))) + c 
        bound_low = [-max(y),0,-len(y),-max(y),-np.inf,-np.inf,-np.inf,-np.inf]
        bound_high = [max(y),10000,len(y),max(y),np.inf,np.inf,np.inf,np.inf]
    else:
        # array of initial parameters
        guess = np.array([guess_a, guess_f, guess_p, guess_c, 0, 0, 0])  
        def sinedecreasing(x, A, f, p, c, B, m, a):  
            return A * np.sin(f * x) * (1 + B * np.sin(2 * np.pi * (f * x + m * np.cos(2 * np.pi * f * np.exp(-a * x) * x) + p))) + c  
        def sineincreasing(x, A, f, p, c, B, m, a):  
            return A * np.sin(f * x) * (1 + B * np.sin(2 * np.pi * (f * x + m * np.cos(2 * np.pi * (f+a*x) * x) + p))) + c   
        bound_low = [-max(y),0,-len(y),-max(y),-np.inf,-np.inf,-np.inf]
        bound_high = [max(y),10000,len(y),max(y),np.inf,np.inf,np.inf]
    # fit function curve and optimize parameters by least squares method
    try:
        if use_bounds==True:
            parameters_dec, covariance_dec = scipy.optimize.curve_fit(sinedecreasing, 
                                                                  x, y, 
                                                                  p0=guess, 
                                                                  maxfev = repeats,
                                                                  bounds=(bound_low,bound_high))
            parameters_inc, covariance_inc = scipy.optimize.curve_fit(sineincreasing, 
                                                                  x, y, 
                                                                  p0=guess, 
                                                                  maxfev = repeats,
                                                                  bounds=(bound_low,bound_high))   
        else:
            parameters_dec, covariance_dec = scipy.optimize.curve_fit(sinedecreasing, 
                                                                      x, y, 
                                                                      p0=guess, 
                                                                      maxfev = repeats)
            parameters_inc, covariance_inc = scipy.optimize.curve_fit(sineincreasing, 
                                                                      x, y, 
                                                                      p0=guess, 
                                                                      maxfev = repeats)   
    except:
        raise Exception("function can't be optimized with defined function calls \
                        you can try increasing 'repeats' parameter; \n \
                        however, it is recommended to check the data first, \
                        as some situations might increase optimization difficulty, \
                        f.e. constant data. \n \
                        Please refer to 'maxfev' parameter in 'scipy.optimize.curve_fit' \
                        for more information.")
    
    # generate predicted values
    predictions_inc = sineincreasing(x, *parameters_inc)
    predictions_dec = sinedecreasing(x, *parameters_dec)
    # calculate variance explained by model as r2
    residuals_inc = y - predictions_inc
    residuals_dec = y - predictions_dec
    square_res_inc = np.sum(residuals_inc**2)
    square_res_dec = np.sum(residuals_dec**2)
    square_tot = np.sum((y - np.mean(y))**2)
    r2_inc = 1 - (square_res_inc / square_tot)
    r2_dec = 1 - (square_res_dec / square_tot)
    
    if r2_inc > r2_dec:
        predictions = predictions_inc
        r2 = r2_inc
        if include_slope==True:
            A, f, p, c, s, B, m, a = parameters_inc
            function = lambda x: s * x + A * np.sin(f * x) * (1 + B * np.sin(2 * np.pi * (f * x + m * np.cos(2 * np.pi * (f+a*x) * x) + p))) + c
            equation = 's·x+A·sin(f·x)·(1+B·sin(2·\u03C0·(f·x+m·cos(2·\u03C0·(f+a·x)·x)+p)))+c'
        else:
            A, f, p, c, B, m, a = parameters_inc   
            function = lambda x: A * np.sin(f * x) * (1 + B * np.sin(2 * np.pi * (f * x + m * np.cos(2 * np.pi * (f+a*x) * x) + p))) + c
            equation = 'A·sin(f·x)·(1+B·sin(2·\u03C0·(f·x+m·cos(2·\u03C0·(f+a·x)·x)+p)))+c'
    else:
        predictions = predictions_dec
        r2 = r2_dec
        if include_slope==True:
            A, f, p, c, s, B, m, a = parameters_dec
            function = lambda x: s * x + A * np.sin(f * x) * (1 + B * np.sin(2 * np.pi * (f * x + m * np.cos(2 * np.pi * f * np.exp(-a * x) * x) + p))) + c 
            equation = 's·x+A·sin(f·x)·(1+B·sin(2·\u03C0·(f·x+m·cos(2·\u03C0·f·exp(-a·x)·x)+p)))+c'
        else:
            A, f, p, c, B, m, a = parameters_dec    
            function = lambda x: A * np.sin(f * x) * (1 + B * np.sin(2 * np.pi * (f * x + m * np.cos(2 * np.pi * f * np.exp(-a*x)*x) + p))) + c 
            equation = 'A·sin(f·x)·(1+B·sin(2·\u03C0·(f·x+m·cos(2·\u03C0·f·exp(-a·x)·x)+p)))+c'
            
    # calculate efficiency
    if efficiency == 'kge':
        eff_value = host_functions.efficiency_stat(predictions, y, 'kge')
    elif efficiency == 'nse':
        eff_value = host_functions.efficiency_stat(predictions, y, 'nse')
    else:
        raise SyntaxError('Only "kge" or "nse" are supported as valid efficiency metrics')

    results = {"amp": A, "freq": f, "phase": p, "beta": B, "modulation_index": m, 
            "period_factor": a, "offset": c, "period": 1/f, 
            "r2": r2, "predictions": predictions, "function": function, "equation": equation}   
    if efficiency == 'kge':
        results["kge"] = eff_value
    else:
        results["nse"] = eff_value        

    # return control to generate function and/or predictions
    if include_slope==True:
        results["slope"] = s       
  
    results["equation"] = equation
    results["function"] = function
        
    return results    