## Harmonic Oscillator Seasonal Trend (HOST) Model Framework for temporal hydrological extremes pattern identification and analysis

> ***This software is currently under development!*** Important notes, current progress and known issues can be found below.

### Description
*hostmodel* is a Python package for hydrological extremes short- and long-term pattern identification and analysis based on harmonic oscillator theorem. 
This package contains tools for full HOST model calculation according to following sheme:
> Published workflow scheme goes here.
Documentation will be published according to work timeframe.


### How to cite. 
If you use this package in a scientific publication, please include the reference below:
> citation to be included on publication


### Website:
Official repository website address:
[https://github.com/chrisrac/hostmodel/](https://github.com/chrisrac/hostmodel/)


### Installation
With pip:
> to be included in pip on beta release


### Dependencies
The *hostmodel* package requires the following:
> to be included once installation is possible


### Usage
**This is unpublished pre-alpha version repository.**

**Please refer to run_script.py for running example.** 

> Documentation will be published according to work timeframe.

**Errors and exceptions might occur at this stage.**

### Current task:
`code optimization and vectorization`
to be implemented in: v0.0.4

### Project development timeframe:
- [x] `v0.0.1: 12/21/2022` pre-alpha code
- [x] `v0.0.1: 12/21/2022` pre-alpha in-module documentation
- [x] `v0.0.1: 12/21/2022` pre-alpha release (v.0.0.1)
- [x] `v0.0.1: 12/21/2022` data preprocessor
- [x] `v0.0.3: 2/14/2023` built in function returner to recreate and forecast points 
- [x] `v0.0.3: 2/14/2023` more output control in host modules
- [x] `v0.0.3: 2/14/2023` built in training/testing split and verification methods`
- [ ] `ready for publish in v0.0.4` change in decision statistics for flow assesment to KGE/NSE 
- [ ] code optimization and vectorization
- [ ] testing and debugging
- [ ] postprocessor for tables and graphs
- [ ] exceptions and errors handling
- [ ] stability finder
- [ ] testing and debugging
- [ ] beta documentation
- [ ] beta release (v.0.1)
- [ ] magnitude solver (research planned)
- [ ] variate period handling (research planned)
- [ ] testing and debugging
- [ ] full documentation
- [ ] full release v.1

### Known issues:
- [x] inconsistent naming `fixed v0.0.3: 2/14/2023`
- [x] broken execution link at topological analysis stage `fixed v0.0.1: 12/21/2022`
- [x] returns inconsistencies `fixed v0.0.1: 12/21/2022`

### Additional content:
- [ ] add decision for selector_cont if RMSE are same according to decision order: chi2, MWU
- [ ] function to recreate "y_pred" and function object from parameters returned if "include_pred==False"

### Known limitations:
- [ ] lack of magnitude solver
- [ ] unable to predict inc/dec magnitudes
- [ ] no variate period handling
