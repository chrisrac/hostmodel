# -*- coding: utf-8 -*-
"""
This is Harmonic Oscillator Seasonal Trend (HOST) Model module  
for hydrological extremes pattern identification and analysis.

Please refer to package repository for citation and updates:
https://github.com/chrisrac/hostmodel

Future functionalities will be added in later development.
Please reffer to documentation for information on current stage of development.

@authors: Krzysztof Raczynski
"""
# imports
import numpy as np
import host_functions


# primary Classes
class Harmonics:
    ''' Class defining raw harmonic functions based on provided data '''
    
    
    __slots__ = ['data','event','no_functions', 'htype', 'beginning','step',
                 'interval', 'threshold_method','results','value']
    
    
    def __init__(self, data, event, no_functions=5, htype='flow', 
                 beginning='1-1-1979', step='D', interval='M', 
                 threshold_method='median'):
        ''' Harmonics class constructor '''
        
        self.data = data
        self.event = event
        self.no_functions = no_functions
        self.htype = htype
        self.beginning = beginning
        self.step = step
        self.interval = interval
        self.threshold_method = threshold_method
        # results slots filled after calling .fit()
        self.results = None
        self.value = None
      
        
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
    
    
 
    def fit(self, include_predictions=False):
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
                                               self.threshold_method)
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
    
    
    __slots__ = ['data','event','htype', 'beginning','step','interval', 
                 'train_size', 'threshold_method','parameters','train', 
                 'test', 'model', 'ts', 'tf', 'sf', 'cf', 'cs', 'trend',
                 'seasonality']
    
    
    def __init__(self, data, event, htype='flow', beginning='1-1-1979', 
                 step='D', interval='M', train_size=0.8, threshold_method='median'):
        ''' Host class constructor '''
        
        self.data = data
        self.event = event
        self.htype = htype
        self.beginning = beginning
        self.step = step
        self.interval = interval
        self.train_size = train_size
        self.threshold_method = threshold_method
        # results slots filled after calling .fit()
        self.parameters = None
        self.train = None
        self.test = None
        self.model = None
        self.ts = None
        self.tf = None
        self.sf = None
        self.cf = None
        self.cs = None
        self.trend = None
        self.seasonality = None
      
        
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
        
                
    def fit(self, repeats=10000, flow_statistic='kge'):
        ''' Fitting method for finding the Host model.
        
        Parameters
        ----------
        repeats: int, default: 10000
            integer representing the maximum number of function calls. 
            Increasing this number significantly might lower the performance.
        flow_statistic: str, default: 'kge'
            the efficiency statistic to use when comparing flow distributions.
            The default 'kge' calls for Kling-Gupta efficiency. Other accepted
            option is 'nse' for Nash-Sutcliffe efficiency. This atribute is 
            called only if harmonic type of the object is set to 'flow'.
            
        Slots set
        -------
        trend: multi-indexed Series
            a Series representing decomposed elements of long-term change, 
            aggregated in 'interval' provided to object.
        seasonality: multi-indexed Series
            a Series representing decomposed elements of short-term change, 
            aggregated in 'interval' provided to object.        
        model: str
            a best-fitted model type: tf - trend fixed, ts - trend sloped, 
            sf - seasonal, cf - combined fixed, cs - combined sloped.
        tf: Fixed class object.
        ts: Sloped class object.
        sf: Fixed class object.
        cf: CombinedFixed class object.
        cs: CombinedSloped class object.
        parameters: dict
            a dictionary of best-fited model parameters for training data. 
            Keys include: amplitude, omega, phase, offset, frequency, slope,
            period, predictions, function,
            and odditionally for occurrence: threshold.
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
        # performing data preprocessing, including identification of periods
        # with event occurring and aggregation to provided interval         
        aggregated = host_functions.preprocess(self.data, 
                                               self.event, 
                                               self.htype,
                                               self.beginning, 
                                               self.step, 
                                               self.interval, 
                                               self.threshold_method)
        # data decomposition using STL 
        trend, seasonal, residual = host_functions.stl_calc(aggregated[self.htype])
        # assigninig decomposed series to slots for accessing once model is generated
        self.trend = trend
        self.seasonality = seasonal
        # splitting the data into train and test series
        t_train_y, t_train_x, t_test_y, t_test_x, split_index = host_functions.data_split(trend, self.train_size)
        s_train_y, s_train_x, s_test_y, s_test_x, split_index = host_functions.data_split(seasonal, self.train_size)
        train_original = np.array(aggregated[self.htype].iloc[:split_index])
        test_original = np.array(aggregated[self.htype].iloc[split_index:])        
        # fitting the models using their Classes
        tf = Fixed(t_train_x, t_train_y)
        tf.fit(repeats)
        ts = Sloped(t_train_x, t_train_y)
        ts.fit(repeats)        
        sf = Fixed(s_train_x, s_train_y)
        sf.fit(repeats)
        cf = CombinedFixed(tf, sf)
        cf.fit()
        cs = CombinedSloped(ts, sf)
        cs.fit()
        # controlling fitting based on the type of harmonic: occurrence or flow
        if self.htype=='occurrence':
            # topological analysis returns accurracy statistics and best threshold
            topo_ts = host_functions.topology(ts.predictions, train_original, slope=ts.slope)
            topo_tf = host_functions.topology(tf.predictions, train_original)
            topo_sf = host_functions.topology(sf.predictions, train_original)
            topo_cf = host_functions.topology(cf.predictions, train_original)
            topo_cs = host_functions.topology(cs.predictions, train_original, slope=cs.slope)
            # selecting best model based on highest f1 score
            scores = {'tf':topo_tf['f1score'], 'ts':topo_ts['f1score'],
                      'sf':topo_sf['f1score'], 'cf':topo_cf['f1score'],
                      'cs':topo_cs['f1score']}
            best_model = max(scores, key=scores.get)
            # assigning results to slots for accessing once model is generated
            self.model = best_model
            self.ts = ts
            self.tf = tf
            self.sf = sf
            self.cf = cf
            self.cs = cs
            # creating dictionaries of models results for selection of best model
            trainig_results = {
                'ts': {'model':'ts', 
                       'accuracy':topo_ts['accuracy'], 'precision':topo_ts['precision'], 
                       'recall':topo_ts['recall'], 'f1score':topo_ts['f1score'],
                       'contingency':topo_ts['contingency']},
                'tf': {'model':'tf', 
                       'accuracy':topo_tf['accuracy'], 'precision':topo_tf['precision'], 
                       'recall':topo_tf['recall'], 'f1score':topo_tf['f1score'],
                       'contingency':topo_tf['contingency']},
                'sf': {'model':'sf', 
                       'accuracy':topo_sf['accuracy'], 'precision':topo_sf['precision'], 
                       'recall':topo_sf['recall'], 'f1score':topo_sf['f1score'],
                       'contingency':topo_sf['contingency']},
                'cf': {'model':'cf', 
                       'accuracy':topo_cf['accuracy'], 'precision':topo_cf['precision'], 
                       'recall':topo_cf['recall'], 'f1score':topo_cf['f1score'],
                       'contingency':topo_cf['contingency']},
                'cs': {'model':'cs', 
                       'accuracy':topo_cs['accuracy'], 'precision':topo_cs['precision'], 
                       'recall':topo_cs['recall'], 'f1score':topo_cs['f1score'],
                       'contingency':topo_cs['contingency']}
            }
            training_parameters = {
                'ts': {'amplitude':ts.amp, 'omega':ts.omega, 'phase':ts.phase, 
                       'offset':ts.offset, 'frequency':ts.frequency, 
                       'slope':ts.slope, 'period':ts.period, 
                       'threshold':topo_ts['threshold'], 
                       'predictions':ts.predictions, 'function':ts.function},
                'tf': {'amplitude':tf.amp, 'omega':tf.omega, 'phase':tf.phase, 
                       'offset':tf.offset, 'frequency':tf.frequency, 
                       'slope':0, 'period':tf.period, 
                       'threshold':topo_tf['threshold'], 
                       'predictions':tf.predictions, 'function':tf.function},
                'sf': {'amplitude':sf.amp, 'omega':sf.omega, 'phase':sf.phase, 
                       'offset':sf.offset, 'frequency':sf.frequency, 
                       'slope':0, 'period':sf.period, 
                       'threshold':topo_sf['threshold'], 
                       'predictions':sf.predictions, 'function':sf.function},
                'cf': {'amplitude':cf.amp, 'omega':cf.omega, 'phase':cf.phase, 
                       'offset':cf.offset, 'frequency':cf.frequency, 
                       'slope':0, 'period':cf.period, 
                       'threshold':topo_cf['threshold'], 
                       'predictions':cf.predictions, 'function':cf.function},
                'cs': {'amplitude':cs.amp, 'omega':cs.omega, 'phase':cs.phase, 
                       'offset':cs.offset, 'frequency':cs.frequency, 
                       'slope':cs.slope, 'period':cs.period, 
                       'threshold':topo_cs['threshold'], 
                       'predictions':cs.predictions, 'function':cs.function}
            }
            # assigning results to slots for accessing once model is generated    
            self.train = trainig_results[best_model]
            self.parameters = training_parameters[best_model]
            # test series block
            test_predictions = []
            limits = []
            # generating predictions and limits for test series
            for x in t_test_x:         
                test_predictions.append(self.parameters['function'](x))
                limits.append(self.parameters['threshold']+self.parameters['slope']*x)
            test_predictions = np.array(test_predictions)
            limits = np.array(limits)
            # evaluation if prediction is higher than limit to indicate occurrence
            test_predictions = host_functions.event_predict(test_predictions, limits)
            # creating contingency table and calculating accurracy statistics
            test_results = host_functions.contingency_stat(test_predictions, test_original)
            # assigning results to slots for accessing once model is generated
            self.test = test_results

        elif self.htype=='flow':
            # calculating efficiency based on defined statistic kge/nse
            dist_ts = host_functions.efficiency_stat(ts.predictions, train_original, flow_statistic)
            dist_tf = host_functions.efficiency_stat(tf.predictions, train_original, flow_statistic)
            dist_sf = host_functions.efficiency_stat(sf.predictions, train_original, flow_statistic)
            dist_cf = host_functions.efficiency_stat(cf.predictions, train_original, flow_statistic)
            dist_cs = host_functions.efficiency_stat(cs.predictions, train_original, flow_statistic)
            # choosing best-fit model based on highest efficiency score 
            scores = {'tf':dist_tf, 'ts':dist_ts, 'sf':dist_sf, 'cf':dist_cf, 'cs':dist_cs}
            best_model = max(scores, key=scores.get)
            # assigning results to slots for accessing once model is generated
            self.model = best_model
            self.ts = ts
            self.tf = tf
            self.sf = sf
            self.cf = cf
            self.cs = cs
            # creating dictionaries of models results for selection of best model
            trainig_results = {
                'ts': {'model':'ts', 
                       'efficiency':dist_ts,
                       'efficiency statistic':flow_statistic},
                'tf': {'model':'tf', 
                       'efficiency':dist_tf,
                       'efficiency statistic':flow_statistic},
                'sf': {'model':'sf', 
                       'efficiency':dist_sf,
                       'efficiency statistic':flow_statistic},
                'cf': {'model':'cf', 
                       'efficiency':dist_cf,
                       'efficiency statistic':flow_statistic},
                'cs': {'model':'cs', 
                       'efficiency':dist_cs,
                       'efficiency statistic':flow_statistic}
            }
            training_parameters = {
                'ts': {'amplitude':ts.amp, 'omega':ts.omega, 'phase':ts.phase, 
                       'offset':ts.offset, 'frequency':ts.frequency, 
                       'slope':ts.slope, 'period':ts.period, 
                       'predictions':ts.predictions, 'function':ts.function},
                'tf': {'amplitude':tf.amp, 'omega':tf.omega, 'phase':tf.phase, 
                       'offset':tf.offset, 'frequency':tf.frequency, 
                       'slope':0, 'period':tf.period, 
                       'predictions':tf.predictions, 'function':tf.function},
                'sf': {'amplitude':sf.amp, 'omega':sf.omega, 'phase':sf.phase, 
                       'offset':sf.offset, 'frequency':sf.frequency, 
                       'slope':0, 'period':sf.period, 
                       'predictions':sf.predictions, 'function':sf.function},
                'cf': {'amplitude':cf.amp, 'omega':cf.omega, 'phase':cf.phase, 
                       'offset':cf.offset, 'frequency':cf.frequency, 
                       'slope':0, 'period':cf.period, 
                       'predictions':cf.predictions, 'function':cf.function},
                'cs': {'amplitude':cs.amp, 'omega':cs.omega, 'phase':cs.phase, 
                       'offset':cs.offset, 'frequency':cs.frequency, 
                       'slope':cs.slope, 'period':cs.period,  
                       'predictions':cs.predictions, 'function':cs.function}
            }
            # assigning results to slots for accessing once model is generated
            self.train = trainig_results[best_model]
            self.parameters = training_parameters[best_model]           
            # test series block
            test_predictions = []
            for x in t_test_x:         
                test_predictions.append(self.parameters['function'](x))
            test_predictions = np.array(test_predictions)
            # calculating efficiency statistic for test data based on selected model
            test_efficiency = host_functions.efficiency_stat(test_predictions, test_original, flow_statistic) 
            # assigning results to slots for accessing once model is generated
            self.test = {'efficiency':test_efficiency, 'efficiency statistic':flow_statistic}
            
        else:
            raise SyntaxError("Unrecognized type of parameter. Define the HOST \
                              object with: htype='flow' for flow analysis or \
                              htype='occurrence' for occurrence analysis. \
                              See documentation for current limitations.")
        
    # method for generating results summary    
    def results(self):
        ''' Method returning summary of results for the found model.
                   
        Returns
        -------
        results: dict
            summary of results for best-fited model.
        '''
        
        if self.htype=='occurrence':
            return {'model type':self.train['model'],
                    'trainig accuracy':self.train['accuracy'],
                    'testing accuracy':self.test['accuracy'],
                    'function amplitude':self.parameters['amplitude'],
                    'function period':self.parameters['period'],
                    'function offset':self.parameters['offset'],
                    'function slope':self.parameters['slope'],
                    'decision threshold at day one':self.parameters['threshold']
                    }
        
        elif self.htype=='flow':
            return {'model type':self.train['model'],
                    'trainig efficiency':self.train['efficiency'],
                    'testing accuracy':self.test['efficiency'],
                    'function amplitude':self.parameters['amplitude'],
                    'function period':self.parameters['period'],
                    'function offset':self.parameters['offset'],
                    'function slope':self.parameters['slope'],
                    }
        
        else:
            raise SyntaxError("Unrecognized type of parameter. Define the HOST \
                              object with: htype='flow' for flow analysis or \
                              htype='occurrence' for occurrence analysis. \
                              See documentation for current limitations.") 
             
    # method for generating function object, easier accessed than calling parameters dict                          
    def function(self):
        ''' Method returning function object of the found model.
                   
        Returns
        -------
        function: function
            function object for the found model. Can be also accessed 
            directly by calling .parameters['function']
        
        Examples
        -------
        host_model = test_class_bak.Host(flow, 'lf', 'occurrence')
        host_model.fit()
        host_function = host_model.function()
        print(host_function(379))
        >> -5.801
        '''  
        
        return self.parameters['function']


# Secondary Classes
class Fixed:
    ''' Class of the fixed trend model object '''
    
    
    __slots__ = ['data','x', 'y', 'amp','omega', 'phase','offset','frequency',
                 'period','r2','predictions', 'function']
    
    
    def __init__(self, x, y):
        ''' Fixed class constructor '''
        
        self.y = y
        self.x = x
        # results slots filled after calling .fit()
        self.amp = None
        self.omega = None
        self.phase = None
        self.offset = None
        self.frequency = None
        self.period = None
        self.r2 = None
        self.predictions = None
        self.function = None
        
        
    def __repr__(self):
        ''' Fixed class representation '''
        
        return 'Fixed HOST model:\n \
        amplitude: {0} \n \
        omega: {1} \n \
        phase: {2} \n \
        offset: {3} \n \
        frequency: {4} \n \
        period: {5} \n \
        r2: {6}'.format(self.amp, self.omega, 
        self.phase, self.offset, self.frequency, self.period, 
        self.r2)   
           
                
    def fit(self, repeats):
        ''' Fitting method for finding the fixed harmonic function. Used for
        trend and seasonality data.
        
        Parameters
        ----------
        repeats: int
            integer representing the maximum number of function calls. 
            Increasing this number significantly might lower the performance.
            
        Slots set
        -------
        amp: float
            amplitude of the model.
        omega: float 
            omega of the model.
        phase: float
            phase of the model.
        offset: float 
            offset of the model.
        frequency: float 
            frequency of the model.
        period: float 
            period of the model.
        r2: float 
            r2 of the model relative to initial decomposed data.
        predictions: array
            predicted values of the model based on calculated function.
        function: function object
            function object for the found model.
        '''
        # fit the fixed function to data
        fitted = host_functions.fit_simple(self.x, self.y, repeats)
        # assigning results to slots for accessing once model is generated
        self.amp = fitted['amp']
        self.omega = fitted['omega']
        self.phase = fitted['phase']
        self.offset = fitted['offset']
        self.frequency = fitted['freq']
        self.period = fitted['period']
        self.r2 = fitted['r2']
        self.predictions = fitted['y_pred']
        self.function = fitted['function']
    
    
        
class Sloped:
    ''' Class of the sloped trend model object '''
    
    
    __slots__ = ['data', 'x', 'y', 'amp', 'omega', 'phase', 'offset', 'frequency', 
                 'slope', 'period', 'r2', 'predictions', 'function']
    
    
    def __init__(self, x, y):
        ''' Sloped class constructor '''
        
        self.y = y
        self.x = x
        # results slots filled after calling .fit()
        self.amp = None
        self.omega = None
        self.phase = None
        self.offset = None
        self.frequency = None
        self.slope = None
        self.period = None
        self.r2 = None
        self.predictions = None
        self.function = None
    
    
    def __repr__(self):
        ''' Sloped class representation '''        
        return 'Sloped HOST model:\n \
        amplitude: {0} \n \
        omega: {1} \n \
        phase: {2} \n \
        offset: {3} \n \
        frequency: {4} \n \
        period: {5} \n \
        slope: {6} \n \
        r2: {7}'.format(self.amp, self.omega, 
        self.phase, self.offset, self.frequency, self.period, self.slope,
        self.r2)
        
        
    def fit(self, repeats):
        ''' Fitting method for finding the fixed harmonic function. Used for
        trend and seasonality data.
        
        Parameters
        ----------
        repeats: int
            integer representing the maximum number of function calls. 
            Increasing this number significantly might lower the performance.
            
        Slots set
        -------
        amp: float
            amplitude of the model.
        omega: float 
            omega of the model.
        phase: float
            phase of the model.
        offset: float 
            offset of the model.
        frequency: float 
            frequency of the model.
        slope: float
            slope of the model.
        period: float 
            period of the model.
        r2: float 
            r2 of the model relative to initial decomposed data.
        predictions: array
            predicted values of the model based on calculated function.
        function: function object
            function object for the found model.
        '''
        # fit the fixed function to data
        fitted = host_functions.fit_sloped(self.x, 
                                       self.y, 
                                       repeats)
        # assigning results to slots for accessing once model is generated
        self.amp = fitted['amp']
        self.omega = fitted['omega']
        self.phase = fitted['phase']
        self.offset = fitted['offset']
        self.frequency = fitted['freq']
        self.slope = fitted['slope']
        self.period = fitted['period']
        self.r2 = fitted['r2']
        self.predictions = fitted['y_pred']
        self.function = fitted['function']
    


class CombinedFixed:
    ''' Class of the combined fixed model object '''
    
    
    __slots__ = ['tf', 'sf', 'amp', 'omega', 'phase', 'offset', 'frequency', 
                 'period', 'r2', 'predictions', 'function']
    
    
    def __init__(self, tf, sf):
        ''' CombinedFixed class constructor '''
        
        self.tf = tf
        self.sf = sf
        # results slots filled after calling .fit()
        self.amp = None
        self.omega = None
        self.phase = None
        self.offset = None
        self.frequency = None
        self.period = None
        self.r2 = None
        self.predictions = None
        self.function = None


    def __repr__(self):
        ''' CombinedFixed class representation '''
        
        return 'Fixed HOST model:\n \
        amplitude: {0} \n \
        omega: {1} \n \
        phase: {2} \n \
        offset: {3} \n \
        frequency: {4} \n \
        period: {5} \n \
        r2: {6}'.format(self.amp, self.omega, 
        self.phase, self.offset, self.frequency, self.period, 
        self.r2)
            
    
    def fit(self):
        ''' Fitting method for finding the fixed harmonic function. Used for
        trend and seasonality data.
        
        Parameters
        ----------
        repeats: int
            integer representing the maximum number of function calls. 
            Increasing this number significantly might lower the performance.
            
        Slots set
        -------
        amp: list of floats
            amplitudes of the model, [0] for trend, [1] for seasonal component.
        omega: list of floats 
            omega of the model, [0] for trend, [1] for seasonal component.
        phase: list of floats
            phase of the model, [0] for trend, [1] for seasonal component.
        offset: list of floats 
            offset of the model, [0] for trend, [1] for seasonal component.
        frequency: list of floats 
            frequency of the model, [0] for trend, [1] for seasonal component.
        period: list of floats 
            period of the model, [0] for trend, [1] for seasonal component.
        r2: list of floats
            r2 of the models relative to initial decomposed data,
            [0] for trend, [1] for seasonal component.
        predictions: array
            predicted values of the model based on calculated function.
        function: function object
            function object for the found model.
        '''
        # assigning results to slots for accessing once model is generated
        # combined model predictions are the sum of trend and seasonal models
        self.predictions = self.tf.predictions + self.sf.predictions
        # generating function object of the model
        self.function = lambda x: (self.tf.amp * np.sin(self.tf.omega*(x-self.tf.phase))) + \
            (self.sf.amp * np.sin(self.sf.omega*(x-self.sf.phase))) + \
            (self.sf.offset +self. sf.offset)
        # assigning function parameters in form of list, as two parameters are 
        # included in the model equation
        self.amp = [self.tf.amp, self.sf.amp]
        self.omega = [self.tf.omega, self.sf.omega]
        self.phase = [self.tf.phase, self.sf.phase]
        self.offset = [self.tf.offset, self.sf.offset]
        self.frequency = [self.tf.frequency, self.sf.frequency]
        self.period = [self.tf.period, self.sf.period]
        self.r2 = [self.tf.r2, self.sf.r2]
        
        

class CombinedSloped:
    ''' Class of the combined sloped model object '''
    
    
    __slots__ = ['ts', 'sf', 'amp', 'omega', 'phase', 'offset', 'frequency', 
                 'slope', 'period', 'r2', 'predictions', 'function']
    
    
    def __init__(self, ts, sf):
        ''' CombinedSloped class constructor '''
        
        self.ts = ts
        self.sf = sf
        # results slots filled after calling .fit()
        self.amp = None
        self.omega = None
        self.phase = None
        self.offset = None
        self.frequency = None
        self.slope = None
        self.period = None
        self.r2 = None
        self.predictions = None
        self.function = None
        
        
    def __repr__(self):
        ''' CombinedSloped class representation '''
        
        return 'Sloped HOST model:\n \
        amplitude: {0} \n \
        omega: {1} \n \
        phase: {2} \n \
        offset: {3} \n \
        frequency: {4} \n \
        period: {5} \n \
        slope: {6} \n \
        r2: {7}'.format(self.amp, self.omega, 
        self.phase, self.offset, self.frequency, self.period, self.slope,
        self.r2) 
           
    
    def fit(self):
        ''' Fitting method for finding the fixed harmonic function. Used for
        trend and seasonality data.
        
        Parameters
        ----------
        repeats: int
            integer representing the maximum number of function calls. 
            Increasing this number significantly might lower the performance.
            
        Slots set
        -------
        amp: list of floats
            amplitudes of the model, [0] for trend, [1] for seasonal component.
        omega: list of floats 
            omega of the model, [0] for trend, [1] for seasonal component.
        phase: list of floats
            phase of the model, [0] for trend, [1] for seasonal component.
        offset: list of floats 
            offset of the model, [0] for trend, [1] for seasonal component.
        frequency: list of floats 
            frequency of the model, [0] for trend, [1] for seasonal component.
        slope: float
            slope of the model.
        period: list of floats 
            period of the model, [0] for trend, [1] for seasonal component.
        r2: list of floats
            r2 of the models relative to initial decomposed data,
            [0] for trend, [1] for seasonal component.
        predictions: array
            predicted values of the model based on calculated function.
        function: function object
            function object for the found model.
        '''
        # assigning results to slots for accessing once model is generated
        # combined model predictions are the sum of trend and seasonal models
        self.predictions = self.ts.predictions + self.sf.predictions
        # generating function object of the model
        self.function = lambda x: self.ts.slope * x + (self.ts.amp * np.sin(self.ts.omega*(x-self.ts.phase))) + \
            (self.sf.amp * np.sin(self.sf.omega*(x-self.sf.phase))) + \
            (self.sf.offset + self.sf.offset)
        # assigning function parameters in form of list, as two parameters are 
        # included in the model equation
        self.amp = [self.ts.amp, self.sf.amp]
        self.omega = [self.ts.omega, self.sf.omega]
        self.phase = [self.ts.phase, self.sf.phase]
        self.offset = [self.ts.offset, self.sf.offset]
        self.frequency = [self.ts.frequency, self.sf.frequency]
        self.slope = self.ts.slope
        self.period = [self.ts.period, self.sf.period]
        self.r2 = [self.ts.r2, self.sf.r2]