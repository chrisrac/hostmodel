## Harmonic Oscillator Seasonal Trend (HOST) Model Framework for temporal hydrological extremes pattern identification and analysis

> ***This software is currently under development!*** Important notes, current progress and known issues can be found below.


### Description
*hostmodel* is a Python package for hydrological extremes short- and long-term pattern identification and analysis based on harmonic oscillator theorem. 
This package contains tools for full HOST model calculation according to following sheme:
![Alt text](https://lh3.googleusercontent.com/pw/AMWts8CwwTiOw7kZUlNRi85wyTHh1FnQnp1u2wtrOudwD_Zp10-hKX3T0RMWIxIykp8236OPbi8L7baR0WZNoF50jSnbLWJ5Tc6zAE4rFr5miyQxnv62GIYAb2LUYUlJIjhY7ZOiayC05R0m5aFlmMj60rg=w720-h365-no "HOST model framework")



### How to cite. 
If you use this package in a scientific publication, please include the reference below:
> citation to be included on publication



### Website:
Official repository website address:
[https://github.com/chrisrac/hostmodel/](https://github.com/chrisrac/hostmodel/)



### Installation
Download the code from releases, unzip it and put it into you working directory for Python. 
Then use 'import host' within your code to use the framework. 
The pip installation for python, anaconda and other distributions will be released in the future, after adressing current limitations (see below).
Make sure you are using the latest release.



### Dependencies
The *hostmodel* package requires the following:
- numpy
- pandas
- scipy
- statsmodels
- objective_thresholds



### Usage
For detailed explanation of workflow and how to access results, please refer to [User Guides](https://github.com/chrisrac/hostmodel/tree/main/user_guides).

The general use of this model depends on the goal.
For calculation of Host models define the object as follows:
```
# for occurrence model, based on data saved in flow variable
# that uses low flow analysis, for daily data, starting from 
# '1-2-1979', aggregated to monthly outputs, and using 80%
# as trainig data:
host_model = host.Host(flow, 'lf', 'occurrence', '1-2-1979', 'D', 'M', 0.8, 'median') 
# or similarly for flow analysis:
host_model = host.Host(flow, 'lf', 'flow', '1-2-1979', 'D', 'M', 0.8)
```
For calculating Harmonic functions, similarly:
```
# for occurrence functions:
harmonic_model = host.Harmonics(flow, 'lf', 6, 'occurrence', '1-2-1979', 'D', 'M', 'median')  
# for flow analysis:
harmonic_model = host.Harmonics(flow, 'lf', 5, 'flow', '1-2-1979', 'D', 'M')
```
then use `.fit()` method on these objects to compute models or functions.


> Documentation will be published according to work timeframe.

**Errors and exceptions might occur at this stage.**



### Current task:
`magnitude solver, variate period handling, new models implementations, workflow alternation to adjust component r2`



### Project development timeframe:
- [x] `v0.0.1: 12/21/2022` pre-alpha code
- [x] `v0.0.1: 12/21/2022` pre-alpha in-module documentation
- [x] `v0.0.1: 12/21/2022` pre-alpha release (v.0.0.1)
- [x] `v0.0.1: 12/21/2022` data preprocessor
- [x] `v0.0.3: 2/14/2023` built in function returner to recreate and forecast points 
- [x] `v0.0.3: 2/14/2023` more output control in host modules
- [x] `v0.0.3: 2/14/2023` built in training/testing split and verification methods`
- [x] `v0.1: 3/27/2023` change in decision statistics for flow assesment to KGE/NSE 
- [x] `v0.1: 3/27/2023` code optimization and vectorization
- [x] `v0.1: 3/27/2023` testing and debugging
- [x] `v0.1: 3/27/2023` exceptions and errors handling
- [x] `v0.1: 3/27/2023` beta documentation
- [x] `v0.1: 3/27/2023` beta release
- [x] `v0.1.1: 3/31/2023` bugfix in sloped model lambda
- [ ] magnitude solver (current research)
- [ ] variate period handling (current research)
- [ ] new models implementations (current research)
- [ ] testing on workflow alternation to adjust component r2 (current research)
- [ ] testing and debugging
- [ ] full documentation
- [ ] full release v.1



### Known limitations:
- [ ] lack of magnitude solver (in current release, the occurrence is considered as boolean classification, therefore if data is aggregated in high resolution, like monthly or annual scales, single occurrence = multiple occurrence. This approach might lower the model precision. Magnitudes solver will be included in the next version of the software). 
- [ ] unable to predict inc/dec magnitudes (currently only sloped models account for increasing or decreasing changes, however, only in one direction. This might lower the accurracy/recall statistics in some cases. Decreasing or increasing temporal magnitudes will be included in the future versions of the software).
- [ ] no variate period handling (changing periods within function peaks affect the repeatability in varying environment conditions. This is especially visible due to human impact on data, and might lower precision and recall of the model. Solution will be included in the future versions of the software).
