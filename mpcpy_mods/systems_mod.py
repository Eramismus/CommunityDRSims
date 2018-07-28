# -*- coding: utf-8 -*-
"""
``systems`` classes represent the controlled systems, with methods to collect 
measurements from or set control inputs to the system.  This representation 
can be real or emulated using a detailed simulation model.  A common interface
to the controlled system in both cases allows for algorithm development and
testing on a simulation with easy transition to the real system.  Measurement 
data can then be passed to ``models`` objects to estimate or validate model
parameters.  Measurement data has a specified variable organization in the form
of a Python dictionary in order to aid its use by other objects.  It is as 
follows:
.. _Key:
system.measurements = {"Measurement Variable Name" : {"Measurement Key" : mpcpy.Variables.Timeseries/Static}}.
The measurement variable name should match the variable that is measured in a 
model in the emulation case, or match the point name that is measured in a real
system case.  The measurement keys are from the following list:
- Simulated - timeseries variable for simulated measurement (yielded by ``models`` objects)
- Measured - timeseries variable for real measurement (yielded by ``systems`` objects)
- Sample - static variable for measurement sample rate
- SimulatedError - timeseries variable for simulated standard error
- MeasuredError - timeseries variable for measured standard error
=========
Emulation
=========
Emulation objects are used to simulate the performance of a real system and
collect the results of the simulation as measurements.  Models used for such 
simulations are often detailed physical models and are not necessarily the 
same as a model used for optimization.  A model for this purpose should be 
instantiated as a ``models`` object instead of a ``systems`` object.
Classes
=======
.. autoclass:: mpcpy.systems.EmulationFromFMU
    :members: collect_measurements, display_measurements, get_base_measurements
==== 
Real
====
Real objects are used to find and collect measurements from a real system.
Classes
=======
.. autoclass:: mpcpy.systems.RealFromCSV
    :members: collect_measurements, display_measurements, get_base_measurements
"""

from abc import ABCMeta, abstractmethod
from mpcpy import utility_mod as utility

#%% System class
class _System(utility._mpcpyPandas, utility._Building, utility._Measurements):
    '''Base class for representing systems.
    
    ''' 
    
    __metaclass__ = ABCMeta;

    @abstractmethod
    def collect_measurements():
        '''An object of the System class must gather system measurements.
        Yields
        ------
        Updates the ``'Measured'`` key for each measured variable in the 
        measurements dictionary attribute.
        '''
        
        pass;         
        
#%% System implementations
class _Emulation(_System):
    '''Base class for an emulated system.
    
    '''
    
    __metaclass__ = ABCMeta;

    def collect_measurements(self, start_time, final_time):
        '''Collect measurement data for the emulated system by simulation 
        using any given exodata inputs.
        
        Parameters
        ----------
        start_time : string
            Start time of measurements collection.
            Set to 'continue' in order to continue the emulation simulation
            from the final time of the previous simulation.  Exodata 
            input objects must contain values for the continuation timestamp.
            The measurements in a continued simulation replace previous values.
            They do not append to a previous simulation's measurements.
        final_time : string
            Final time of measurements collection.  Must be greater than the
            start time.
            
        Yields
        ------
        Updates the ``'Measured'`` key for each measured variable in the 
        measurements dictionary attribute.
        '''

        self._set_time_interval(start_time, final_time);
        self._simulate();
        for key in self.measurements.keys():
            self.measurements[key]['Measured'] = self.measurements[key]['Simulated']; 

class _Real(_System):
    '''Base class for a real system.
    
    '''
    
    def collect_measurements(self, start_time, final_time):
        '''Collect measurement data for the real system.
        
        Parameters
        ----------
        start_time : string
            Start time of measurements collection.
        final_time : string
            Final time of measurements collection.
            
        Yields
        ------
        Updates the ``'Measured'`` key for each measured variable in the 
        measurements dictionary attribute.
        
        '''

        self._set_time_interval(start_time, final_time);
        #self.total_elapsed_seconds = self.year_start_seconds;
        #self.total_elapsed_seconds = self.year_final_seconds+self.total_elapsed_seconds-2*86400; # This line because the fmus used have their own weather processors.
        self._collect_data();
        
#%% Source implementations
class EmulationFromFMU(_Emulation, utility._FMU):
    '''System emulation by FMU simulation.
    
    Parameters
    ----------
    measurements : dictionary
        {"Measurement Name" : {"Sample" : mpcpy.Variables.Static}}.
    fmupath : string, required if not moinfo
        FMU file path.
    moinfo : tuple or list, required if not fmupath
        (mopath, modelpath, libraries).  `mopath` is the path to the modelica file.
        `modelpath` is the path to the model to be compiled within the package specified in the modelica file.
        `libraries` is a list of paths directing to extra libraries required to compile the fmu.
    zone_names : list, optional
        List of zone name strings.
    weather_data : dictionary, optional
        ``exodata`` weather object data attribute.
    internal_data : dictionary, optional
        ``exodata`` internal object data attribute.
    control_data : dictionary, optional
        ``exodata`` control object data attribute.    
    other_inputs : dictionary, optional
        ``exodata`` other inputs object data attribute.    
    parameter_data : dictionary, optional
        ``exodata`` parameter object data attribute.
    tz_name : string, optional
        Name of timezone according to the package ``tzwhere``.  If 
        ``'from_geography'``, then geography kwarg is required.
    geography : list or tuple, optional
        List or tuple with (latitude, longitude) in degrees. 
    Attributes
    ----------
    measurements : dictionary
        {"Measurement Name" : {"Measurement Key_" : mpcpy.Variables.Timeseries/Static}}.
    fmu : pyfmi fmu object
        FMU respresenting the emulated system.
    fmupath : string
        Path to the FMU file.
    lat : numeric
        Latitude in degrees.  For timezone.
    lon : numeric
        Longitude in degrees.  For timezone.
        
    '''

    def __init__(self, measurements, **kwargs):
        '''Constructor of a system fmu simulation source.
        
        '''

        self.name = 'emulation_from_fmu';
        self._create_fmu(kwargs);
        self.measurements = measurements
        self.input_names = self._get_input_names();
        self._parse_building_kwargs(kwargs);
        self._parse_time_zone_kwargs(kwargs);
        
    def _simulate(self):
        '''Simulate the fmu.
        
        '''
        #self.total_elapsed_seconds = self.year_final_seconds+self.total_elapsed_seconds; # This line because the fmus used have their own weather processors.
        #self._simulate_fmu_emu();
        self._simulate_fmu();
        
class RealFromCSV(_Real, utility._DAQ):
    '''System measured data located in csv.
    
    Parameters
    ----------
    csv_file_path : string
        Path of csv file.
    measurements : dictionary
        {"Measurement Name" : {"Sample" : mpcpy.Variables.Static}}.
    variable_map : dictionary
        {"Column Header Name" : ("Measurement Variable Name", mpcpy.Units.unit)}.
    tz_name : string, optional
        Name of timezone according to the package ``tzwhere``.  If 
        ``'from_geography'``, then geography kwarg is required.
    geography : list or tuple, optional
        List or tuple with (latitude, longitude) in degrees. 
    Attributes
    ----------
    measurements : dictionary
        {"Measurement Variable Name" : {{"Measurement Key_" : mpcpy.Variables.Timeseries/Static}}.
    file_path : string
        Path of csv file. 
    lat : numeric
        Latitude in degrees.  For timezone.
    lon : numeric
        Longitude in degrees.  For timezone.
        
    '''

    def __init__(self, csv_file_path, measurements, variable_map, **kwargs):
        '''Constructor of a system fmu simulation source.
        
        '''

        self.name = 'real_from_csv';
        self.file_path = csv_file_path;
        self.measurements = measurements;
        self.variable_map = variable_map;
        self._parse_daq_kwargs(kwargs);
        self._parse_time_zone_kwargs(kwargs);
        
    def _collect_data(self):
        '''Collect data from csv into measurement dictionary.
        
        '''

        self._read_timeseries_from_csv();

    def _translate_variable_map(self):
        '''Translate csv column to measurement dictionary.
        
        '''

        varname = self.variable_map[self._key][0];
        unit = self.variable_map[self._key][1];
        self.measurements[varname]['Measured'] = self._dataframe_to_mpcpy_ts_variable(self._df, self._key, varname, unit, \
                                                                                      start_time=self.start_time, final_time=self.final_time, \
                                                                                      cleaning_type = self._cleaning_type, \
                                                                                      cleaning_args = self._cleaning_args);  