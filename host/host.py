# -*- coding: utf-8 -*-
"""
This is Harmonic Oscillator Seasonal Trend (HOST) Model module  
for hydrological extremes pattern identification and analysis.

Please refer to package repository for citation and updates:
https://github.com/chrisrac/hostmodel

Future functionalities will be added in later development.
Please reffer to documentation for information on current stage of development.

v. 1.0

@authors: Krzysztof Raczynski
"""
# imports
import numpy as np
import warnings
import host_functions
import host_models
from collections.abc import Iterable


# primary Classes
class IterationWarning(UserWarning):
    pass


class Harmonics:
    ''' Class defining raw harmonic functions based on provided data '''
    
    
    __slots__ = ['data','event','no_functions', 'htype', 'beginning','step',
                 'interval', 'threshold_method', 'threshold_overwrite',
                 'signal_method','area','results','value','raw_data']
    
    
    def __init__(self, data, event, no_functions=5, htype='signal', 
                 beginning='1-1-1979', step='D', interval='M', 
                 threshold_method='median', threshold_overwrite=None,
                 signal_method='mean', area=0):
        ''' Harmonics class constructor '''
        # data integrity check and handle warnings
        if not isinstance(data, Iterable):
            raise TypeError('The Iterable sequence must be used as data input.')
        if isinstance(data, (str, bytes)):
            raise TypeError('Sequence of numbers int or float must be used.')        
        if event not in ['value','low','high']:
            raise SyntaxError('Specified event does not exist, availabe options are: '+'\n'+
                              '"value", "low", or "high".')  
        if not isinstance(no_functions, int):
            raise TypeError('Int value must be used as no_functions.')            
        if htype not in ['signal','occurrence','magnitude']:
            raise SyntaxError('Specified htype does not exist, availabe options are: '+'\n'+
                      '"signal", "occurrence", or "magnitude".')        
        if not isinstance(beginning, str):
            raise TypeError('Date in beginning attribute have to be string, eg. "1-1-1979"')
        if step not in ['D','M','Y']:
            raise SyntaxError('Custom data steps are not handled, use: '+'\n'+
                      '"D" for daily, "M" for monthly, "Y" for annual.') 
        if interval not in ['D','M','Y']:
            raise SyntaxError('Custom output interval is not handled, use: '+'\n'+
                      '"D" for daily, "M" for monthly, "Y" for annual.')                
        if threshold_method not in ['leave' , 'max', 'median']:
            raise SyntaxError('Specified threshold_method does not exist, availabe options are: '+'\n'+
                              '"leave", "max", or "median".')   
        if signal_method not in ['mean' , 'sum', 'min', 'max']:
            raise SyntaxError('Specified signal_method does not exist, availabe options are: '+'\n'+
                              '"mean" , "sum", "min", "max".')   
        if not isinstance(area, (float, int)):
            raise TypeError('Area must be a number.')        
        # constructor calls
        self.data = data
        self.event = event
        self.no_functions = no_functions
        self.htype = htype
        self.beginning = beginning
        self.step = step
        self.interval = interval
        self.threshold_method = threshold_method
        self.threshold_overwrite = threshold_overwrite
        self.signal_method = signal_method
        self.area = area
        # results slots filled after calling .fit()
        self.results = None
        self.value = None
        self.raw_data = None
        
        
    def __repr__(self):
        ''' Harmonics class representation '''
        
        return 'Raw harmonic function object:\n \
        event type: {0} \n \
        harmonic type: {1} \n \
        number of functions to fit: {2} \n \
        data beginning: {3} \n \
        data step: {4} \n \
        aggregation interval: {5} \n \
        threshold behavior condition: {6}'.format(self.event, self.htype, 
        self.no_functions, self.beginning, self.step, self.interval, 
        self.threshold_method)
    
    
 
    def fit(self, include_predictions=False, binary_occurrence=True):
        ''' Fitting method for finding the harmonic function.
        
        Parameters
        ----------
        include_predictions : bool, default: False
            allows to control returned values of the found function. Must be
            set to True for values to be accessible through .values() method.
            
        Slots set
        -------
        results: DataFrame
            results DataFrame contains consecutive functions in columns and 
            their parameters in rows: amplitude, omega, phase, offset, 
            frequency, period, r2. Can be accessed by calling .results on 
            model object.
        value: DataFrame
            function values frame is returned only if include_predictions=True,
            when calling .fit method. Values can be accessed by calling .values()
            method.
        '''
        # performing data preprocessing, including identification of periods
        # with event occurring and aggregation to provided interval 
        aggregated = host_functions.preprocess(self.data, 
                                               self.event, 
                                               self.htype,
                                               self.beginning, 
                                               self.step, 
                                               self.interval, 
                                               self.threshold_method,
                                               self.threshold_overwrite,
                                               self.signal_method,
                                               self.area,
                                               binary_occurrence)
        self.raw_data = aggregated
        # generating results for default case without predictions
        if include_predictions==False:
            self.results = host_functions.harmonic(aggregated, 
                                                   self.htype,
                                                   self.no_functions, 
                                                   include_predictions)
        # providing predictions into value slot
        else:
            self.results, self.value = host_functions.harmonic(aggregated, 
                                                               self.htype,
                                                               self.no_functions, 
                                                               include_predictions)
    
    # controlling the way predictions are returned, with exception raised 
    # explaining steps to get the values
    def values(self):
        ''' Method returning values of the found harmonic function.
                   
        Returns
        -------
        value: DataFrame
            values of the harmonic function in frame if include_predictions=True,
            when calling .fit method, otherwise raise SyntaxError. 
        '''
        
        if self.value is None:
            raise SyntaxError('In order to return predicted values of the function \n \
                              use include_predictions argument in fit: \n \
                                  .fit(include_preds=True) or simply .fit(True)')
        else:
            return self.value



class Host:
    ''' Class defining HOST model '''
    
    
    #__slots__ = ['data','event','htype', 'beginning','step','interval', 
    #             'train_size', 'threshold_method']
    

    def __init__(self, data, event, htype='signal', beginning='1-1-1979', 
                 step='D', interval='M', train_size=0.8, threshold_method='median', 
                 threshold_overwrite=None, signal_method='mean', area=0):
        ''' Host class constructor '''
        # data integrity check and handle warnings
        if not isinstance(data, Iterable):
            raise TypeError('The Iterable sequence must be used as data input.')
        if isinstance(data, (str, bytes)):
            raise TypeError('Sequence of numbers int or float must be used.')        
        if event not in ['value','low','high']:
            raise SyntaxError('Specified event does not exist, availabe options are: '+'\n'+
                              '"value", "low", or "high".')  
        if htype not in ['signal','occurrence','magnitude']:
            raise SyntaxError('Specified htype does not exist, availabe options are: '+'\n'+
                      '"signal", "occurrence", or "magnitude".')        
        if not isinstance(beginning, str):
            raise TypeError('Date in beginning attribute have to be string, eg. "1-1-1979"')
        if step not in ['D','M','Y']:
            raise SyntaxError('Custom data steps are not handled, use: '+'\n'+
                      '"D" for daily, "M" for monthly, "Y" for annual.') 
        if interval not in ['D','M','Y']:
            raise SyntaxError('Custom output interval is not handled, use: '+'\n'+
                      '"D" for daily, "M" for monthly, "Y" for annual.')                
        if not isinstance(train_size, float) or train_size > 1 or train_size < 0:
            raise TypeError('Train_size must be a float between 0 and 1.')        
        if threshold_method not in ['leave' , 'max', 'median']:
            raise SyntaxError('Specified threshold_method does not exist, availabe options are: '+'\n'+
                              '"leave", "max", or "median".')   
        if signal_method not in ['mean' , 'sum', 'min', 'max']:
            raise SyntaxError('Specified signal_method does not exist, availabe options are: '+'\n'+
                              '"mean" , "sum", "min", "max".')   
        if not isinstance(area, (float, int)):
            raise TypeError('Area must be a number.')        
        # constructor calls
        self.data = data
        self.event = event
        self.htype = htype
        self.beginning = beginning
        self.step = step
        self.interval = interval
        self.train_size = train_size
        self.threshold_method = threshold_method
        self.threshold_overwrite = threshold_overwrite
        self.signal_method = signal_method
        self.area = area
        # results slots filled after calling .fit()
        self.train = None
        self.test = None
        self.trend_data = None
        self.seasonal_data = None
        self.raw_data = None
        self.trendmodel = None
        self.seasonalmodel = None
        self.model = None
        self.function = None
        self.equation = None
    
    def __repr__(self):
        ''' Host class representation '''
        
        return 'Occurrence object:\n \
        event type: {0} \n \
        harmonic type: {1} \n \
        data beginning: {2} \n \
        data step: {3} \n \
        aggregation interval: {4} \n \
        train / test split: {5} \n \
        threshold behavior condition: {6}'.format(self.event, 
        self.htype, 
        self.beginning, 
        self.step, 
        self.interval, 
        (str(round(self.train_size*100,2))+'% / '+(str(round((1-self.train_size)*100,2)))+'%'),
        self.threshold_method)
        
                
    def fit(self, repeats=1000000, multiplier=1, include_damped=True, decision_statistic='r2', 
            efficiency='kge', binary_occurrence=True, use_bounds=False):
        ''' Fitting method for finding the Host model.
        
        Parameters
        ----------
        repeats: int, default: 1000000
            integer representing the maximum number of function calls. 
            Increasing this number significantly might lower the performance.
        multiplier: int, default: 1
            integer value to be used as multiplier of period used during STL
            decomposition. By default decomposition uses first period of function
            found by Fast Fourier Transform. this period might be increased by
            multiplier argument. Only positive integer numbers are accepted. 
            Default of 1 means no multiplication. 
        include_damped: bool, default: True
            boolean flag to modify fitted models. If damped models are not desired,
            setting this to False will exclude them from fitting.
        decision_statistic: str, default: 'r2'
            string determining which characteristic should be used when choosing
            best fitted model. Default 'r2' might be changed to used effciency.
        efficiency: str, default: 'kge'
            the efficiency statistic to use when comparing flow distributions.
            The default 'kge' calls for Kling-Gupta efficiency. Other accepted
            option is 'nse' for Nash-Sutcliffe efficiency. 
        binary_occurrence: bool, default: True
            boolean flag to modify binary occurrence fitter. Default setting True
            means occurrence parameter is using binary classification. If set to 
            False will cause weighted occurrence calculation.
        use_bounds: bool, default: False
            boolean flag allowing to control fitting process. Default False means
            all parameter values are possible, including negative frequencies 
            (periods). Set to True to lock to only positive frequencies. Using 
            bounds might slower the optimization and lower fit efficiencies. 
            
            
        Slots set
        -------
        trend_data: array-like
            an array representing decomposed values of long-term change, 
            aggregated in 'interval' provided to object.
        seasonal_data: array-like
            an array representing decomposed values of short-term change, 
            aggregated in 'interval' provided to object.  
        raw_data: array-like
            an array of aggregated in interval preprocessed data.
        trendmodel: Trend class object
            a Trend class object of best-fitted model for trend data. Includes 
            .parameters, .predictions, .function and .equation.
        seasonalmodel: Seasonality class object
            a Seasonality class object of best-fitted model for seasonal data. 
            Includes .parameters, .predictions, .function and .equation.
        model: Combined class object
            a Combined class object of resultant model presented as waveform
            synthesis of trend and seasonal models. Includes .parameters, 
            .predictions, .function and .equation   .         
        function: function object
            a function object that represents mathematical formula of the
            resultant model. Can be used to compute result for any x provided.
        equation: string
            a UTF-8 coded string representing mathematical formula of the
            resultant model.
        train: dict
            a dictionary of best-fited model parameters applied to train data. 
            Keys include: 
            for occurrence: model, contingency, accuracy, precision, 
            recall, f1score;
            for flow: model, efficiency, efficiency statistic.
        test: dict
            a dictionary of best-fited model parameters applied to test data. 
            Keys include: 
            for occurrence: contingency, accuracy, precision, 
            recall, f1score;
            for flow: efficiency, efficiency statistic.
        '''
        
        if not isinstance(multiplier, int) or multiplier < 1:
            raise TypeError('multiplier must be a positive integer (of value 1 or higher).')      
            
        # performing data preprocessing, including identification of periods
        # with event occurring and aggregation to provided interval         
        aggregated = host_functions.preprocess(self.data, 
                                               self.event, 
                                               self.htype,
                                               self.beginning, 
                                               self.step, 
                                               self.interval, 
                                               self.threshold_method,
                                               self.threshold_overwrite,
                                               self.signal_method,
                                               self.area,
                                               binary_occurrence)
        #if self.htype=='magnitude':
        #    aggregated = aggregated.drop(['count','sum'], axis=1)
        self.raw_data = aggregated
        # data decomposition using STL 
        trend, seasonal, residual = host_functions.stl_calc(aggregated[self.htype], multiplier, repeats, self.interval)
        # assigninig decomposed series to slots for accessing once model is generated
        self.trend_data = trend
        self.seasonal_data = seasonal
        # splitting the data into train and test series
        t_train_y, t_train_x, t_test_y, t_test_x, split_index = host_functions.data_split(trend, self.train_size)
        s_train_y, s_train_x, s_test_y, s_test_x, split_index = host_functions.data_split(seasonal, self.train_size)
        train_original = np.array(aggregated[self.htype].iloc[:split_index])
        test_original = np.array(aggregated[self.htype].iloc[split_index:])        
        # fitting the models using their Classes
        trend_model = Trend(t_train_x, t_train_y)
        trend_model.fit(repeats, include_damped, decision_statistic, efficiency, use_bounds)
        self.trendmodel = trend_model
        seasonal_model = Seasonality(s_train_x, s_train_y)
        seasonal_model.fit(repeats, include_damped, decision_statistic, efficiency, use_bounds)   
        self.seasonalmodel = seasonal_model
        combined_model = Combined(trend_model, seasonal_model)
        combined_model.fit()
        self.model = combined_model

        # controlling fitting based on the type of harmonic: occurrence or flow
        if self.htype=='occurrence':
            # topological analysis returns accurracy statistics and best threshold
            if 'slope' in trend_model.parameters:
                slope = trend_model.parameters['slope']
            else:
                slope = 0
            train_statistics = host_functions.topology(combined_model.predictions, train_original, slope=slope)
            # assigning results to slots for accessing once model is generated    
            r2 = host_functions.rsquared(train_original, train_statistics['predictions'])
            train_statistics['r2'] = r2
            self.train = train_statistics
            self.function = combined_model.function
            self.equation =  combined_model.equation
            
            # test series block
            test_pred = []
            limits = []
            # generating predictions and limits for test series
            for x in t_test_x:         
                test_pred.append(self.function(x))
                limits.append(train_statistics['threshold']+slope*x)
            test_pred = np.array(test_pred)
            limits = np.array(limits)
            # evaluation if prediction is higher than limit to indicate occurrence
            test_predictions = host_functions.event_predict(test_pred, limits)
            # creating contingency table and calculating accurracy statistics
            test_results = host_functions.contingency_stat(test_predictions, test_original)
            # assigning results to slots for accessing once model is generated
            r2 = host_functions.rsquared(test_original, test_results['predictions'])
            test_results['r2'] = r2
            self.test = test_results

        elif self.htype=='signal':
            # calculating efficiency based on defined statistic kge/nse
            train_statistics, train_predictions = host_functions.efficiency_stat(combined_model.predictions, train_original, efficiency, True)
            # calculate variance explained by model as r2
            r2 = host_functions.rsquared(train_original, train_predictions)
            #residuals =  - train_predictions
            #square_res = np.sum(residuals**2)
            #square_tot = np.sum((train_original - np.mean(train_original))**2)
            #r2 = 1 - (square_res / square_tot)
            # assigning results to slots for accessing once model is generated
            self.train = {'efficiency':train_statistics, 'r2':r2, 'predictions': train_predictions}
            self.function = combined_model.function
            self.equation =  combined_model.equation
            
            # creating dictionaries of models results for selection of best model
            # assigning results to slots for accessing once model is generated         
            # test series block
            test_pred = []
            for x in t_test_x:         
                test_pred.append(self.function(x))
            test_pred = np.array(test_pred)
            # calculating efficiency statistic for test data based on selected model
            test_efficiency, test_predictions = host_functions.efficiency_stat(test_pred, test_original, efficiency, True) 
            # calculate variance explained by model as r2
            r2 = host_functions.rsquared(test_original, test_predictions)
            #residuals = test_original - test_predictions
            #square_res = np.sum(residuals**2)
            #square_tot = np.sum((test_original - np.mean(test_original))**2)
            #r2 = 1 - (square_res / square_tot)
            # assigning results to slots for accessing once model is generated
            self.test = {'efficiency':test_efficiency, 'r2':r2, 'predictions': test_predictions}
        
        elif self.htype=='magnitude':
            if 'slope' in trend_model.parameters:
                slope = trend_model.parameters['slope']
            else:
                slope = 0
            # calculating efficiency based on defined statistic kge/nse
            train_results = host_functions.magnitude_topology(combined_model.predictions, train_original, slope=slope)
            # calculate variance explained by model as r2
            r2 = host_functions.rsquared(train_original, train_results['magnitude predictions'])
            #residuals = train_original - train_results['magnitude predictions']
            #square_res = np.sum(residuals**2)
            #square_tot = np.sum((train_original - np.mean(train_original))**2)
            #r2 = 1 - (square_res / square_tot)
            train_results['r2'] = r2
            self.train = train_results
            self.function = combined_model.function
            self.equation =  combined_model.equation    
            
            # creating dictionaries of models results for selection of best model
            # assigning results to slots for accessing once model is generated         
            # test series block
            test_pred = []
            for x in t_test_x:         
                test_pred.append(self.function(x))
            test_pred = np.array(test_pred)            
            test_results = host_functions.magnitude_topology(test_pred, test_original, slope=slope, threshold=train_results['threshold'])
            # calculate variance explained by model as r2
            r2 = host_functions.rsquared(test_original, test_results['magnitude predictions'])
            #residuals = test_original - test_results['magnitude predictions']
            #square_res = np.sum(residuals**2)
            #square_tot = np.sum((test_original - np.mean(test_original))**2)
            #r2 = 1 - (square_res / square_tot)            
            # assigning results to slots for accessing once model is generated
            test_results['r2'] = r2
            self.test = test_results
            
        else:
            raise SyntaxError("Unrecognized type of parameter. Define the HOST \
                              object with: htype='flow' for flow analysis or \
                              htype='occurrence' for occurrence analysis or \
                              htype='magnitude' for magnitude analysis.")
        
    # method for generating results summary    
    def results(self):
        ''' Method returning summary of results for the found model.
                   
        Returns
        -------
        results: dict
            summary of results for best-fited model.
        '''
        
        if self.htype=='occurrence':
            return {'model equation':self.equation,
                    'trainig accuracy':self.train['accuracy'],
                    'testing accuracy':self.test['accuracy'],
                    'trend component parameters':self.model.parameters['trend'],
                    'seasonal component parameters':self.model.parameters['seasonal'],
                    'decision threshold at day one':self.train['threshold']
                    }
        
        elif self.htype=='signal':
            return {'model equation':self.equation,
                    'trainig efficiency':self.train['efficiency'],
                    'testing accuracy':self.test['efficiency'],
                    'trend component parameters':self.model.parameters['trend'],
                    'seasonal component parameters':self.train['seasonal'],
                    }
        elif self.htype=='magnitude':
            return {'model equation':self.equation,
                    'trainig efficiency':self.train['efficiency'],
                    'testing accuracy':self.test['efficiency'],
                    'trend component parameters':self.model.parameters['trend'],
                    'seasonal component parameters':self.train['seasonal'],
                    }
        else:
            raise SyntaxError('''Unrecognized htype. Accepted are: signal, 
                              occurrence, or magnitude.''') 
             


class Trend:
    ''' Class describing trend model parameters '''
    
    #__slots__ = ['data','x','y','tpredictions','tfunction','tparameters','tequation']
    
    def __init__(self, x, y):
        ''' Trend class constructor '''
        self.x = x
        self.y = y
        # results slots filled after calling .fit()
        self.parameters = None
        self.predictions = None
        self.function = None
        self.equation = None
        self.models = None

    def __repr__(self):
        ''' Trend class representation '''
        if self.parameters == None:
            return 'Trend HOST model object. Use .fit() to solve.'
        else:
            strpar = ', '.join("{!s} = {!r}".format(k,v) for (k,v) in self.parameters.items())
            return 'Trend HOST model object: {0} \n \
                    Parameters: {1}'.format(self.equation, strpar)

    def fit(self, repeats, include_damped, statistic, efficiency, use_bounds):
        ''' Fitting method for finding the best-fit function. 
        Chooses from all models available in host_models and autofind best-fit.
        To find specific model parameters use model-specific classes.
        
        Parameters
        ----------     
        repeats: int
            integer representing the maximum number of function calls. 
            Increasing this number significantly might lower the performance.
        include_damped: bool, default: True
            boolean flag to modify fitted models. If damped models are not desired,
            setting this to False will exclude them from fitting.
        statistic: str, default: 'r2'
            string determining which characteristic should be used when choosing
            best fitted model. Default 'r2' might be changed to used effciency.
        efficiency: str, default: 'kge'
            the efficiency statistic to use when comparing flow distributions.
            The default 'kge' calls for Kling-Gupta efficiency. Other accepted
            option is 'nse' for Nash-Sutcliffe efficiency. 
        use_bounds: bool, default: False
            boolean flag allowing to control fitting process. Default False means
            all parameter values are possible, including negative frequencies 
            (periods). Set to True to lock to only positive frequencies. Using 
            bounds might slower the optimization and lower fit efficiencies. 
                
            
        Slots set
        -------
        parameters: dict
            each parameter value for the model.
        equation: string
            a string representation of mathematical formula for the model.
        predictions: array
            predicted values of the model based on calculated function.
        function: function object
            function object for the found model.
        '''
        # fit the fixed function to data
        warnmsg1 = ' trend model skipped due to fail in converge with '
        warnmsg2 = ' repeats. Trying different models. Increase repeats argument to include.'
        
        try:
            tsine = host_models.fit_sine(self.x, self.y, False, repeats, efficiency, use_bounds)
        except:
            tsine = {statistic: np.nan, efficiency: np.nan}
            warnings.warn('Sine'+warnmsg1+str(repeats)+warnmsg2, IterationWarning)
        try:            
            tsines = host_models.fit_sine(self.x, self.y, True, repeats, efficiency, use_bounds)
        except:
            tsines = {statistic: np.nan, efficiency: np.nan}
            warnings.warn('Sine sloped'+warnmsg1+str(repeats)+warnmsg2, IterationWarning)
        try:
            tampmod = host_models.fit_amplitude_mod(self.x, self.y, False, repeats, efficiency, use_bounds)
        except:
            tampmod = {statistic: np.nan, efficiency: np.nan}
            warnings.warn('Ampmod'+warnmsg1+str(repeats)+warnmsg2, IterationWarning)
        try:
            tampmods = host_models.fit_amplitude_mod(self.x, self.y, True, repeats, efficiency, use_bounds)
        except:
            tampmods = {statistic: np.nan, efficiency: np.nan}
            warnings.warn('Ampmod sloped'+warnmsg1+str(repeats)+warnmsg2, IterationWarning)
        try:
            tfreqmod = host_models.fit_frequency_mod(self.x, self.y, False, repeats, efficiency, use_bounds)
        except:
            tfreqmod = {statistic: np.nan, efficiency: np.nan}
            warnings.warn('Freqmod'+warnmsg1+str(repeats)+warnmsg2, IterationWarning)
        try:
            tfreqmods = host_models.fit_frequency_mod(self.x, self.y, True, repeats, efficiency, use_bounds)
        except:
            tfreqmods = {statistic: np.nan, efficiency: np.nan}
            warnings.warn('Freqmod sloped'+warnmsg1+str(repeats)+warnmsg2, IterationWarning)
        try:
            tmodul = host_models.fit_modulated(self.x, self.y, False, repeats, efficiency, use_bounds)
        except:
            tmodul = {statistic: np.nan, efficiency: np.nan}
            warnings.warn('Modulated'+warnmsg1+str(repeats)+warnmsg2, IterationWarning)
        try:
            tmoduls = host_models.fit_modulated(self.x, self.y, True, repeats, efficiency, use_bounds)
        except:
            tmoduls = {statistic: np.nan, efficiency: np.nan}
            warnings.warn('Modulated sloped'+warnmsg1+str(repeats)+warnmsg2, IterationWarning)
        try:
            tincdec = host_models.fit_incdec_mod(self.x, self.y, False, repeats, efficiency, use_bounds)
        except:
            tincdec = {statistic: np.nan, efficiency: np.nan}
            warnings.warn('Incdec'+warnmsg1+str(repeats)+warnmsg2, IterationWarning)
        try:
            tincdecs = host_models.fit_incdec_mod(self.x, self.y, True, repeats, efficiency, use_bounds)
        except:
            tincdecs = {statistic: np.nan, efficiency: np.nan}
            warnings.warn('Incdec sloped'+warnmsg1+str(repeats)+warnmsg2, IterationWarning)
            
        self.models = {'sine':tsine, 'sine sloped': tsines, 'ampmod': tampmod, 
                       'ampmod sloped': tampmods, 'freqmod': tfreqmod,
                       'freqmod sloped': tfreqmods, 'modulated': tmodul,
                       'modulated sloped': tmoduls, 'incdec': tincdec,
                       'incdec sloped': tincdecs}
        
        scores = [tsine[statistic], tsines[statistic], tampmod[statistic], tampmods[statistic], 
                          tfreqmod[statistic], tfreqmods[statistic], tmodul[statistic], 
                          tmoduls[statistic],  tincdec[statistic], tincdecs[statistic]]
        models = [tsine, tsines, tampmod, tampmods, tfreqmod, tfreqmods, 
                   tmodul, tmoduls, tincdec, tincdecs]      
        
        if include_damped==True:
            try:
                tdamped = host_models.fit_damped(self.x, self.y, False, repeats, efficiency, use_bounds)
            except:
                tdamped = {statistic: np.nan, efficiency: np.nan}
                warnings.warn('Damped'+warnmsg1+str(repeats)+warnmsg2, IterationWarning)
            try:
                tdampeds = host_models.fit_damped(self.x, self.y, True, repeats, efficiency, use_bounds)
            except:
                tdampeds = {statistic: np.nan, efficiency: np.nan}
                warnings.warn('Damped sloped'+warnmsg1+str(repeats)+warnmsg2, IterationWarning)
            try:
                tdampmod = host_models.fit_damped_mod(self.x, self.y, False, repeats, efficiency, use_bounds)
            except:
                tdampmod = {statistic: np.nan, efficiency: np.nan}
                warnings.warn('Dampmod'+warnmsg1+str(repeats)+warnmsg2, IterationWarning)
            try:
                tdampmods = host_models.fit_damped_mod(self.x, self.y, True, repeats, efficiency, use_bounds)        
            except:
                tdampmods = {statistic: np.nan, efficiency: np.nan}
                warnings.warn('Dampmod sloped'+warnmsg1+str(repeats)+warnmsg2, IterationWarning)
                
            self.models['damped'] = tdamped
            self.models['damped sloped'] = tdampeds
            self.models['dampmod'] = tdampmod
            self.models['dampmod sloped'] = tdampmods
            
            scores.append(tdamped[statistic])
            scores.append(tdampeds[statistic])
            scores.append(tdampmod[statistic])
            scores.append(tdampmods[statistic])
            models.append(tdamped)
            models.append(tdampeds)
            models.append(tdampmod)
            models.append(tdampmods)
        
        scores = np.array(scores)
        if np.isnan(scores).all():
            raise ValueError('None of the trend models was optimized with {0} repeats. \
                             Increase argument value in .fit(repeats=).'.format(repeats))
        else:
            models = np.array(models)
            maxscore = np.argmax(scores)
            model = models[maxscore]
        
        # assigning results to slots for accessing once model is generated
        try:
            self.predictions = model['predictions']
            self.function = model['function']
            self.equation = model['equation']      
            self.parameters = {key: value for key, value in model.items() if key not in ["function", "equation", "predictions"]}
        except:
            raise RuntimeError('Model failed to converge with '+str(repeats)+' repeats. Increase repeats.')
        
        
class Seasonality:
    ''' Class describing seasonality model parameters ''' 
    
    #__slots__ = ['data','x','y','spredictions','sfunction','sparameters','sequation']
    
    def __init__(self, x, y):
        ''' Seasonality class constructor '''
        self.x = x
        self.y = y
        # results slots filled after calling .fit()
        self.parameters = None
        self.predictions = None
        self.function = None
        self.equation = None
        self.models = None

    def __repr__(self):
        ''' Seasonality class representation '''
        if self.parameters == None:
            return 'Seasonality HOST model object. Use .fit() to solve.'
        else:
            strpar = ', '.join("{!s} = {!r}".format(k,v) for (k,v) in self.parameters.items())
            return 'Seasonality HOST model object: {0} \n \
                    Parameters: {1}'.format(self.equation, strpar)

    def fit(self, repeats, include_damped, statistic, efficiency, use_bounds):
        ''' Fitting method for finding the best-fit function. 
        Chooses from all models available in host_models and autofind best-fit.
        To find specific model parameters use model-specific classes.
        
        Parameters
        ----------
        repeats: int
            integer representing the maximum number of function calls. 
            Increasing this number significantly might lower the performance.
        include_damped: bool, default: True
            boolean flag to modify fitted models. If damped models are not desired,
            setting this to False will exclude them from fitting.
        statistic: str, default: 'r2'
            string determining which characteristic should be used when choosing
            best fitted model. Default 'r2' might be changed to used effciency.
        efficiency: str, default: 'kge'
            the efficiency statistic to use when comparing flow distributions.
            The default 'kge' calls for Kling-Gupta efficiency. Other accepted
            option is 'nse' for Nash-Sutcliffe efficiency. 
        use_bounds: bool, default: False
            boolean flag allowing to control fitting process. Default False means
            all parameter values are possible, including negative frequencies 
            (periods). Set to True to lock to only positive frequencies. Using 
            bounds might slower the optimization and lower fit efficiencies. 
            
            
        Slots set
        -------
        parameters: dict
            each parameter value for the model.
        equation: string
            a string representation of mathematical formula for the model.
        predictions: array
            predicted values of the model based on calculated function.
        function: function object
            function object for the found model.
        '''
        # fit the fixed function to data
        warnmsg1 = ' seasonal model skipped due to fail in converge with '
        warnmsg2 = ' repeats. Trying different models. Increase repeats argument to include.'
        
        try:
            tsine = host_models.fit_sine(self.x, self.y, False, repeats, efficiency, use_bounds)
        except:
            tsine = {statistic: np.nan, efficiency: np.nan}
            warnings.warn('Sine'+warnmsg1+str(repeats)+warnmsg2, IterationWarning)
        try:
            tampmod = host_models.fit_amplitude_mod(self.x, self.y, False, repeats, efficiency, use_bounds)
        except:
            tampmod = {statistic: np.nan, efficiency: np.nan}
            warnings.warn('Ampmod'+warnmsg1+str(repeats)+warnmsg2, IterationWarning)
        try:
            tfreqmod = host_models.fit_frequency_mod(self.x, self.y, False, repeats, efficiency, use_bounds)
        except:
            tfreqmod = {statistic: np.nan, efficiency: np.nan}
            warnings.warn('Freqmod'+warnmsg1+str(repeats)+warnmsg2, IterationWarning)
        try:
            tmodul = host_models.fit_modulated(self.x, self.y, False, repeats, efficiency, use_bounds)
        except:
            tmodul = {statistic: np.nan, efficiency: np.nan}
            warnings.warn('Modulated'+warnmsg1+str(repeats)+warnmsg2, IterationWarning)
        try:
            tincdec = host_models.fit_incdec_mod(self.x, self.y, False, repeats, efficiency, use_bounds)
        except:
            tincdec = {statistic: np.nan, efficiency: np.nan}
            warnings.warn('Incdec'+warnmsg1+str(repeats)+warnmsg2, IterationWarning)
            
        self.models = {'sine':tsine, 'ampmod': tampmod, 'freqmod': tfreqmod, 
                       'modulated': tmodul, 'incdec': tincdec}
        
        scores = [tsine[statistic], tampmod[statistic], tfreqmod[statistic], tmodul[statistic],  
                          tincdec[statistic]]
        models = [tsine, tampmod, tfreqmod, tmodul, tincdec]
        
        if include_damped==True:
            try:
                tdamped = host_models.fit_damped(self.x, self.y, False, repeats, efficiency, use_bounds)
            except:
                tdamped = {statistic: np.nan, efficiency: np.nan}
                warnings.warn('Damped'+warnmsg1+str(repeats)+warnmsg2, IterationWarning)
            try:
                tdampmod = host_models.fit_damped_mod(self.x, self.y, False, repeats, efficiency, use_bounds)
            except:
                tdampmod = {statistic: np.nan, efficiency: np.nan}
                warnings.warn('Dampmod'+warnmsg1+str(repeats)+warnmsg2, IterationWarning)
            
            self.models['damped'] = tdamped
            self.models['dampmod'] = tdampmod
            scores.append(tdamped[statistic])
            scores.append(tdampmod[statistic])
            models.append(tdamped)
            models.append(tdampmod)
            
        scores = np.array(scores)
        if np.isnan(scores).all():
            raise ValueError('None of the seasonal models was optimized with {0} repeats. \
                             Increase argument value in .fit(repeats=).'.format(repeats))
        else:
            models = np.array(models)
            maxscore = np.argmax(scores)
            model = models[maxscore]
        
        # assigning results to slots for accessing once model is generated
        try:    
            self.predictions = model['predictions']
            self.function = model['function']
            self.equation = model['equation']      
            self.parameters = {key: value for key, value in model.items() if key not in ["function", "equation", "predictions"]}
        except:
            raise RuntimeError('Model failed to converge with '+str(repeats)+' repeats. Increase repeats.')
   
    
class Combined:
    ''' Class of the combined model object '''
    
    
    #__slots__ = ['trend_model','seasonal_model','cpredictions','cfunction','cparameters','cequation']
    
    
    def __init__(self, trend_model, seasonal_model):
        ''' Combined class constructor '''
        
        self.trend_model = trend_model
        self.seasonal_model = seasonal_model
        # results slots filled after calling .fit()
        self.parameters = None
        self.function = None
        self.predictions = None
        self.equation = None
        
        
    def __repr__(self):
        ''' Combined class representation '''
        if self.parameters == None:
            return 'Combined HOST model object. Use .fit() to solve.'
        else:
            strpar = ', '.join("{!s} = {!r}".format(k,v) for (k,v) in self.parameters.items())
            return 'Combined HOST model object: {0} \n \
                    Parameters: {1}'.format(self.equation, strpar)
            
    
    def fit(self):
        ''' Fitting method for finding the function. Used for trend and 
        seasonality data.
                    
        Slots set
        -------
        parameters: dict
            each parameter value for the model. Position [0] represent trend
            component and position [1] represents seasonal component.
        equation: string
            a string representation of mathematical formula for the model.
        predictions: array
            predicted values of the model based on calculated function.
        function: function object
            function object for the found model.
        '''
        
        # assigning results to slots for accessing once model is generated
        # combined model predictions are the sum of trend and seasonal models
        self.predictions = self.trend_model.predictions + self.seasonal_model.predictions
        # generating function object of the model
        self.function = lambda x: self.trend_model.function(x) + self.seasonal_model.function(x)
        # assigning function parameters in form of list, as two parameters are 
        # included in the model equation
        self.parameters = {'trend':self.trend_model.parameters, 'seasonal':self.seasonal_model.parameters}
        self.equation =  self.trend_model.equation + '+' + self.seasonal_model.equation