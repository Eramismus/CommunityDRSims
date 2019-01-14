# Import classes from mpcpy
from mpcpy import variables
from mpcpy import units
from mpcpy import exodata
from mpcpy import systems_mod as systems
from mpcpy import models_mod as models
from mpcpy import optimization_mod as optimization

# General python packages
#import dill
import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def get_weather(weather_file, start_time, final_time):
	# Function for getting weather-data
	print("%%%%%%---Getting weather data---%%%%%%%%%%%%%")
	weather = exodata.WeatherFromEPW(weather_file)
	
	return weather.collect_data(start_time, final_time)
	
def get_controlCSV(control_file, variable_map, start_time, final_time, timezone):
	# function to gather control data from csv
	print("%%%%%%%%%%%%%%%----Getting control data from CSV---%%%%%%%%%%%%%")
	control = exodata.ControlFromCSV(control_file,variable_map, tz_name = timezone)
	
	return control.collect_data(start_time, final_time)
	
def emulate_jmod(emulation, measured_var_dict, sample_rate, start_time, end_time):
	# function to emulate with J-Modelica

	print("%%%%%%%%%%%%%%%----Starting Emulation---%%%%%%%%%%%%%")
	for key in measured_var_dict.keys():
		var_str = 'sample_rate_' + key
		measured_var_dict[key]['Sample'] = variables.Static(var_str,
																sample_rate,
																units.s
																)
	
	return emulation.collect_measurements(start_time, end_time)
	
def estimate_params(model, start_time, end_time, est_param_list):

	print("%%%%%%%%%%%%%%%----Starting Estimation of MPC---%%%%%%%%%%%%%")
	# Do the training
	return model.estimate(start_time, end_time, est_param_list)

def validate_model(model,start_time, end_time, plot_str):
	print("%%%%%%%%%%%%%%%----Starting Validation ---%%%%%%%%%%%%%")
	# Perform validation against the measurements
	return model.validate(start_time,end_time,plot_str,plot=1)
	
def store_namespace(filename,class_name):
	with open(str(filename+'.pkl'), 'wb') as file:
		dill.dump(class_name, file)
	
def load_namespace(filename):
	with open(str(filename+'.pkl'), 'rb') as file:
		a = dill.load(file)
	return a
	
	
