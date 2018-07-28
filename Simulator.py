""" Second version, make the code easier and more modifiable """
# Define the main programme

#Import other required packages
import pandas as pd
import matplotlib.pyplot as plt
import dill
import os
import numpy as np
#import time

# Import classes from mpcpy
from mpcpy import variables
from mpcpy import units
from mpcpy import exodata
from mpcpy import systems_mod as systems
from mpcpy import models_mod as models
from mpcpy import optimization_mod as optimization

from funcs import emulate_jmod
#from funcs import simulate_mpc_JModelica
#from funcs import simulate_mpc_UKF
from funcs import store_namespace
from funcs import load_namespace

""" Class for handling the simulation workflow """

class SimHandler():
	def __init__(self, sim_start, sim_end, meas_sampl):
		print("Initializing SimHandler")
		
		# Measurement period for simulations
		self.sim_start = sim_start
		self.sim_end = sim_end
		
		# %%%%%%%%%%%% System and MPC models %%%%%%%%%%%%%%%%%%%%%%%%
		self.building = ""
		self.mod_path = "path to models"
		self.simu_path = "Simulator_git"
		
		self.fmupath_mpc = os.path.join(self.simu_path, 'Tutorial_RC.fmu')
		self.fmupath_emu = os.path.join(self.simu_path, 'ResidentialCommunity_Detached0_Detached0_Models_Detached0_House_mpc.fmu')
		
		
		# Model for emulation
		self.moinfo_emu = (os.path.join(self.mod_path, 'ResidentialCommunity\Detached0\Detached0_Models\Detached0_House_mpc.mo'),
						'ResidentialCommunity.Detached0.Detached0_Models.Detached0_House_mpc',
						{}
						) 
		
		# Model for MPC
		self.moinfo_mpc = (os.path.join(self.simu_path, 'Tutorial.mo'),
				'Tutorial.RC',
				{}
				)
		
		# Set parameters
		self.weather_file = os.path.join(self.simu_path, 'weather', 'Nottingham_TRY.epw')
		self.weather_file_mos = os.path.join(self.simu_path, 'weather', 'Nottingham_TRY.mos')

		self.price_varmap = {'Price': ('Price',units.cents_kWh)
			}
		self.flex_cost_varmap = {'FlexCost': ('FlexCost',units.cents_kWh)
			}
		self.ref_profile_varmap = {'RefProfile': ('RefProfile',units.cents_kWh)
			}
		
		self.contr_varmap = {'ConvGain2': ('ConvGain2',units.W)
			}
		
		self.other_varmap = {'ConvGain1': ('ConvGain1',units.W),
			'RadGain': ('RadGain',units.W)
			}
		
		self.control_file = os.path.join(self.simu_path,'csvs','ControlSignal.csv')
		self.price_file = os.path.join(self.simu_path,'csvs','PriceSignal.csv')
		self.param_file = os.path.join(self.simu_path,'csvs','Parameters.csv')
		self.constraint_csv = os.path.join(self.simu_path,'csvs','Constraints.csv')
		#self.meas_file = 'sim_results.csv'
		
		# Id period
		self.id_start = '1/1/2017 00:00:00'
		self.id_end = '1/3/2017 00:00:00'
		
		# Validation period
		self.val_start  = '1/3/2017 00:00:00'
		self.val_end = '1/4/2017 12:00:00'
		
		# Optimization period
		self.opt_start  = '1/4/2017 12:00:00'
		self.opt_end = '1/5/2017 00:00:00'

		# measurement sample rate in seconds
		self.meas_sampl = int(meas_sampl)
		
		self.meas_vars = {'TAir': {},
					}
		self.meas_vars['TAir']['Sample'] = variables.Static('sample_rate_Tzone',
						self.meas_sampl, # sample rate
						units.s
						) # units
						
		#self.meas_vars['ConvGain2']['Sample'] = variables.Static('sample_rate_heater',
		#				self.meas_sampl, # sample rate
		#				units.s
		#				) # units
						
		self.meas_varmap = {'TAir': ('TAir',units.K)}
		
		# List of estimation parameters
		self.est_params = ['TAir']
				
		# Constraints for optimization
		self.optcon_varmap = {
						'T_min': ('TAir', 'GTE', units.degC),
						'T_max': ('TAir', 'LTE', units.degC),
						'PHeater_min': ('ConvGain2', 'GTE', units.W),
						'PHeater_max': ('ConvGain2', 'LTE', units.W)
						}
						

		self.target_variable = 'ConvGain2'
		#self.ref_profile_series = load_namespace('ref_heatinput').resample(str(self.meas_sampl)+'S').mean()
		
		
		# %%%%%%%%%%%% Reference Models %%%%%%%%%%%%%%%%%%%%%%%%
		self.moinfo_emu_ref = (os.path.join(self.mod_path, 'ResidentialCommunity\Detached0\Detached0_Models'),
					'ResidentialCommunity.Detached0.Detached0_Models.Detached0_House_PI',
					{}
					)
		
		self.fmupath_ref = os.path.join(self.simu_path, 'refit_project_firstorder_ibpsa_Building01_Building01_Models_Building01_SingleDwelling_PI.fmu')
		
		self.meas_vars_ref = {'TAir': {}, 
							'HeatInput': {}
							}

		self.meas_vars_ref['TAir']['Sample'] = variables.Static('sample_rate_Tzone',
					self.meas_sampl, # sample rate
					units.s
					) # units
		
		self.meas_vars_ref['HeatInput']['Sample'] = variables.Static('sample_rate_Tzone',
					self.meas_sampl, # sample rate
					units.s
					) # units
		
		self.meas_sampl_ref = 900
		
		self.contr_varmap_ref = {'SetPoint': ('SetPoint',units.K)
		}
		
		self.start_temp = np.random.uniform(16,22)
		self.ref_profile_file = 'controlseq_Detached_0_R2CW_MinEne'
		
	# Get other inputs
	def get_other_input(self,start,end):
		
		index = pd.date_range(start, end, freq = str(self.meas_sampl)+'S')
		# Zero signal
		radgain = pd.Series(np.random.rand(len(index))*100,index=index)
		zero_signal = pd.Series(np.zeros(len(index)),index=index)
		#t_start = pd.Series(self.start_temp, index=index)
		
		mor_start = np.random.randint(5,9)
		mor_end = np.random.randint(9,12)
		eve_start =  np.random.randint(15,19)
		eve_end = np.random.randint(19,24)
		dem_temp = np.random.randint(18,23)
		ppl_nr = np.random.randint(1,5)
		
		for i in index:
			if i.hour >= mor_start and i.hour <= mor_end:
				radgain[i] = np.random.uniform(0,1)*150*np.random.randint(0,ppl_nr)
			if i.hour >= eve_start and i.hour <= eve_end:
				radgain[i] = np.random.uniform(0,1)*150*np.random.randint(0,ppl_nr)
		
		self.other_input = exodata.ControlFromCSV(self.control_file,
										self.other_varmap,
										tz_name = self.weather.tz_name)
		self.other_input.data = {"ConvGain1": variables.Timeseries('ConvGain1', zero_signal,units.W,tz_name=self.weather.tz_name),
			"RadGain": variables.Timeseries('RadGain', radgain,units.W,tz_name=self.weather.tz_name)
			}
		store_namespace('other_input_'+self.building,self.other_input)
	
		
	# Get control data from csv for identification
	def get_control(self):
		self.control = exodata.ControlFromCSV(self.control_file,
										self.contr_varmap,
										tz_name = self.weather.tz_name)
		index = pd.date_range(self.sim_start, self.sim_end, freq = str(self.meas_sampl)+'S')
		# Random control signal
		control_signal1 = pd.Series(np.random.rand(len(index))*3000,index=index)
		# Define control data
		self.control.data = {"ConvGain2": variables.Timeseries('ConvGain2', control_signal1,units.W,tz_name=self.weather.tz_name)
		}
		#pheat_max = pd.Series(10000,index=index)
		#self.control.collect_data(self.sim_start, self.sim_end)
		#print(self.control.display_data())
		store_namespace('control_'+self.building,self.control)
	
	# Get parameter data
	def get_params(self):
		self.parameters = exodata.ParameterFromCSV(self.param_file)
		self.parameters.collect_data()
		store_namespace('params_'+self.building,self.parameters)
	
	def get_constraints(self,start,end):	
		self.constraints = exodata.ConstraintFromCSV(self.constraint_csv, self.optcon_varmap)
		#print("%%%% constraint data %%%%")
		#print(self.constraints)
		#print(self.constraints.data)
		index = pd.date_range(start, end, freq = str(self.meas_sampl) +'S')
		# Set points based on UCL paper on set points!!! High variability? Commercial?
		mor_start = np.random.randint(5,9)
		mor_end = np.random.randint(9,12)
		eve_start =  np.random.randint(15,19)
		eve_end = np.random.randint(19,23)
		dem_temp = np.random.randint(18,23)
		t_min = pd.Series(14+273.15,index=index) # Contracted?
		t_max = pd.Series(28+273.15,index=index) # Contracted?
		for i in index:
			if i.hour >= mor_start and i.hour <= mor_end:
				t_min[i] = dem_temp-1+273.15
				t_max[i] = dem_temp+1+273.15
			if i.hour >= eve_start and i.hour <= eve_end:
				t_min[i] = dem_temp-1+273.15
				t_max[i] = dem_temp+1+273.15
		pheat_min = pd.Series(0,index=index)
		pheat_max = pd.Series(10000,index=index)
		#print(t_min)
		self.constraints.data = {'TAir': 
				{'GTE': variables.Timeseries('TAir_GTE', t_min, units.K,tz_name=self.weather.tz_name), 
				'LTE': variables.Timeseries('TAir_LTE', t_max, units.K,tz_name=self.weather.tz_name)},
				'ConvGain2': 
				{'GTE': variables.Timeseries('ConvGain2_GTE', pheat_min, units.W, tz_name=self.weather.tz_name), 
				'LTE': variables.Timeseries('ConvGain2_LTE', pheat_max, units.W,tz_name=self.weather.tz_name)}
				}
		store_namespace('constraints_'+self.building,self.constraints)
			
	def update_weather(self,start,end):
		# Next we get exogenous data from epw-file
		print("%%%%%%---Getting weather data---%%%%%%%%%%%%%")
		self.weather = exodata.WeatherFromEPW(self.weather_file)
		self.weather.collect_data(start, end)
		store_namespace('weather',self.weather.display_data())
	
	def update_other_input(self,start,end):
		print(" %%%%%%%%%%%%%%%% Updating other inputs %%%%%%%%%%%%%%%%")
		index = pd.date_range(start, end, freq = str(self.meas_sampl)+'S')
		# Zero signal 
		zero_signal = pd.Series(np.zeros(len(index)),index=index)
		#t_start = pd.Series(self.start_temp, index=index)
	
	
	def update_control(self,start,end):
		print("%%%%%%%%%%%%%%%% Updating control %%%%%%%%%%%%%%%%")
		index = pd.date_range(start, end, freq = str(self.meas_sampl)+'S')
		# Random control signal
		control_signal1 = pd.Series(np.random.rand(len(index))*0,index=index)
		# Define control data
		self.control.data = {"ConvGain2": variables.Timeseries('ConvGain2', control_signal1,units.W,tz_name=self.weather.tz_name)
		}
		#pheat_max = pd.Series(10000,index=index)
		#self.control.collect_data(self.sim_start, self.sim_end)
		#print(self.control.display_data())
		store_namespace('control_upd_'+self.building,self.control)
	
	def update_constraints(self,start,end):
		print(" %%%%%%%%%%%%%%%% Updating constraints %%%%%%%%%%%%%%%%")
		#print(self.constraints)
		#print(self.constraints.data)
		index = pd.date_range(start, end, freq = str(self.meas_sampl) +'S')
		mor_start = np.random.randint(5,9)
		mor_end = np.random.randint(9,12)
		eve_start =  np.random.randint(15,19)
		eve_end = np.random.randint(19,24)
		dem_temp = np.random.randint(18,23)
		t_min = pd.Series(15+273.15,index=index)
		t_max = pd.Series(26+273.15,index=index)
		for i in index:
			if i.hour >= mor_start and i.hour <= mor_end:
				t_min[i] = dem_temp-1+273.15
				t_max[i] = dem_temp+1+273.15
			if i.hour >= eve_start and i.hour <= eve_end:
				t_min[i] = dem_temp-1+273.15
				t_max[i] = dem_temp+1+273.15
		pheat_min = pd.Series(0,index=index)
		pheat_max = pd.Series(10000,index=index)
		#print(t_min)
		self.constraints.data = {'TAir': 
				{'GTE': variables.Timeseries('TAir_GTE', t_min, units.K,tz_name=self.weather.tz_name), 
				'LTE': variables.Timeseries('TAir_LTE', t_max, units.K,tz_name=self.weather.tz_name)},
				'ConvGain2': 
				{'GTE': variables.Timeseries('ConvGain2_GTE', pheat_min, units.W, tz_name=self.weather.tz_name), 
				'LTE': variables.Timeseries('ConvGain2_LTE', pheat_max, units.W,tz_name=self.weather.tz_name)}
				}
		#print(self.constraints.display_data())
		store_namespace('constraints_upd_'+self.building,self.constraints)
	
	def update_params(self, param, value, unit):
		# Add a new value fixed value to parameters
		self.parameters.data[param] = {'Free': variables.Static('FreeOrNot', False, units.boolean), 
		'Minimum': variables.Static(param, value, unit), 
		'Covariance': variables.Static('Covar', 0, unit), 
		'Value': variables.Static(param, value, unit), 
		'Maximum': variables.Static(param, value, unit)
		}
		store_namespace('params_upd_'+self.building, self.parameters)


	def init_models(self,use_ukf,use_fmu_mpc,use_fmu_emu):
		print("Initialising models")
		#self.emu = systems.RealFromCSV(self.meas_file,self.meas_vars,self.		meas_varmap, tz_name = self.weather.tz_name)
		if use_fmu_emu == 0:
			print("%%%%%%%%%%%%%%%%% Compiling new FMUs for emulation %%%%%%%%%%%%%%%%")
			self.emu = systems.EmulationFromFMU(self.meas_vars,
										moinfo=self.moinfo_emu,
										weather_data = self.weather.data,
										control_data = self.control.data,
										other_inputs = self.other_input.data,
										tz_name = self.weather.tz_name
										)
		else:
			print("%%%%%%%%%%%%%% Using existing FMUs for emulation %%%%%%%%%%%%%%%%%%%%%%%")
			print("Emulation FMU: " + str(self.fmupath_emu))
			self.emu = systems.EmulationFromFMU(self.meas_vars,
										fmupath=self.fmupath_emu,
										weather_data = self.weather.data,
										control_data = self.control.data,
										other_inputs = self.other_input.data,
										tz_name = self.weather.tz_name
										)
			#self.emu.collect_measurements(self.sim_start, self.sim_end)
		if use_ukf ==1:
			if use_fmu_mpc==1:
				print("%%%%%%%%%%%%%% Using existing MPC FMUs %%%%%%%%%%%%%%%%%%%%%%%")
				print("MPC FMU: " + str(self.fmupath_mpc))
				self.mpc = models.Modelica(models.UKF,
							models.RMSE,
							self.emu.measurements,
							fmupath = self.fmupath_mpc,
							parameter_data = self.parameters.data,
							weather_data = self.weather.data,
							control_data = self.control.data,
							other_inputs = self.other_input.data,
							tz_name = self.weather.tz_name,
							version = '1.0'
							)
				self.mpc.moinfo = self.moinfo_mpc
			else:
				print("%%%%%%%%%%%%%%%%% Compiling new FMUs %%%%%%%%%%%%%%%%")
				self.mpc = models.Modelica(models.UKF,
							models.RMSE,
							self.emu.measurements,
							moinfo = self.moinfo_mpc,
							parameter_data = self.parameters.data,
							weather_data = self.weather.data,
							control_data = self.control.data,
							other_inputs = self.other_input.data,
							tz_name = self.weather.tz_name,
							version = '1.0'
							)
		else:					
			if use_fmu_mpc==1:
				print("%%%%%%%%%%%%%% Using existing MPC FMUs %%%%%%%%%%%%%%%%%%%%%%%")
				print("MPC FMU: " + str(self.fmupath_mpc))
				self.mpc = models.Modelica(models.JModelica,
							models.RMSE,
							self.emu.measurements,
							fmupath = self.fmupath_mpc,
							parameter_data = self.parameters.data,
							weather_data = self.weather.data,
							control_data = self.control.data,
							other_inputs = self.other_input.data,
							tz_name = self.weather.tz_name,
							)
				self.mpc.moinfo = self.moinfo_mpc
			else:
				print("%%%%%%%%%%%%%%%%% Compiling new MPC FMUs %%%%%%%%%%%%%%%%")
				self.mpc = models.Modelica(models.JModelica,
							models.RMSE,
							self.emu.measurements,
							moinfo = self.moinfo_mpc,
							parameter_data = self.parameters.data,
							weather_data = self.weather.data,
							control_data = self.control.data,
							other_inputs = self.other_input.data,
							tz_name = self.weather.tz_name
							)

			#self.emu.collect_measurements(self.sim_start, self.sim_end)
					
					
	def sys_id(self):
		print("%%%% ---Starting system identification --- %%%%")
		# Emulate
		# Get measurements
		self.emu.collect_measurements(self.id_start, self.id_end)
						
		# Display the measurements
		print('%%%%---Measured variables---%%%%')
		print(self.emu.display_measurements('Measured'))
		
		#Simulate mpc
		self.mpc.simulate(self.id_start, self.id_end)
		#store_namespace('mpc.pik',self.mpc)
		print('%%%%---Simulated variables---%%%%')
		print(self.mpc.display_measurements('Simulated'))
		
		print('%%%%---Estimating parameters---%%%%')
		self.mpc.estimate(self.id_start,self.id_end, self.est_params)
		
		# Just print the parameters	
		print("%%%---Estimated Parameters----%%%%")
		for key in self.mpc.parameter_data.keys():
			print(key, self.mpc.parameter_data[key]['Value'].display_data())
		store_namespace('est_params_'+self.building,self.mpc.parameter_data)
	

			
	def validate(self, val_start, val_end, id , plotfile):
		print("%%%%%%%%%%%%%%%----Starting another validation with estimated parameters ---%%%%%%%%%%%%%")	
		# Validate against another time period
		self.emu.collect_measurements(val_start, val_end)
		self.mpc.measurements = self.emu.measurements
		self.mpc.validate(val_start, val_end, plotfile, plot=1)
		plt.clf() # clear the figure
		
		print("----The RMSE------")
		for key in self.mpc.RMSE.keys():
			print(self.mpc.RMSE[key].display_data())
		store_namespace('RMSE_'+id+'_'+self.building,self.mpc.RMSE)
		store_namespace('sim_'+id+'_'+self.building,self.mpc.display_measurements('Simulated'))
		store_namespace('meas_'+id+'_'+self.building,self.emu.display_measurements('Measured'))

			
	def opt_control_minEnergy(self):
		# Instantiate optimization problem
		self.opt_problem = optimization.Optimization(self.mpc,
												optimization.EnergyMin,
												optimization.JModelica,
												self.target_variable,
												constraint_data = self.constraints.data
												)
		
		self.opt_problem
		
		print("%%%%%%%%%%%%%%%----Starting Optimization of Control ---%%%%%%%%%%%%%")		
		self.opt_problem.optimize(self.opt_start,self.opt_end,res_control_step=self.meas_sampl)

		print("----Optimization stats----")
		print(self.opt_problem.get_optimization_statistics())

		print("----Optimization outcome----")
		self.opt_controlseq = self.opt_problem.Model.control_data # dictionary with mpcpy-timeseries variables of controls
		print(self.opt_controlseq)
		#print(opt_control['Qflow1'])
		#print(opt_control['Qflow2'])
		store_namespace('opt_control_minenergy_'+self.building,self.opt_controlseq)
		
	def opt_control_minCost(self):
		# Instantiate optimization problem
		self.opt_problem = optimization.Optimization(self.mpc,
												optimization.EnergyCostMin,
												optimization.JModelica,
												self.target_variable,
												constraint_data = self.constraints.data
												)

		print("%%%%%%%%%%%%%%%----Starting Optimization of Control ---%%%%%%%%%%%%%")		
		self.opt_problem.optimize(self.opt_start,self.opt_end,price_data = self.price.data)

		print("----Optimization stats----")
		print(self.opt_problem.get_optimization_statistics())

		print("----Optimization outcome----")
		self.opt_controlseq = self.opt_problem.Model.control_data # dictionary with mpcpy-timeseries variables of controls
		#print(self.opt_controlseq['ConvGain2'].display_data())
		#print(opt_control['Qflow1'])
		#print(opt_control['Qflow2'])
		store_namespace('opt_control_mincost_'+self.building,self.opt_controlseq)
	
	def get_DRinfo(self,start,end,**kwargs):
		self.price = exodata.PriceFromCSV(self.price_file,
										self.price_varmap,
										tz_name = self.weather.tz_name)
		self.flex_cost = exodata.PriceFromCSV(self.price_file,
										self.flex_cost_varmap,
										tz_name = self.weather.tz_name)
		self.ref_profile = exodata.ControlFromCSV(self.control_file,
										self.ref_profile_varmap,
										tz_name = self.weather.tz_name)
		index = pd.date_range(start, end, freq = str(self.meas_sampl)+'S')
		# Random control signal
		price_signal = pd.Series(np.random.rand(len(index))*40,index=index)
		flex_signal = pd.Series(0,index=index)
		k = pd.Series(1,index=index)
		ref_signal = pd.Series(25000,index=index)
		#ref_signal = load_namespace(self.ref_profile_file)[0]
		#print(type(ref_signal))
		for i in index:
			if i.hour >= 7 and i.hour <= 11:
				price_signal[i] = np.random.uniform(0.8,1)*60
			if i.hour >= 17 and i.hour <= 19:
				price_signal[i] = np.random.uniform(0.8,1)*90
			if i.hour >= 20 and i.hour <= 22:
				price_signal[i] = np.random.uniform(0.8,1)*60
		
		self.price.data = {"pi_e": variables.Timeseries('pi_e', price_signal,units.cents_kWh,tz_name=self.weather.tz_name)
		}
		self.flex_cost.data = {"flex_cost": variables.Timeseries('flex_cost', flex_signal,units.cents_kWh,tz_name=self.weather.tz_name)
		}
		self.ref_profile.data = {"ref_profile": variables.Timeseries('ref_profile', ref_signal, units.W,tz_name=self.weather.tz_name)
		}
		
		#pheat_max = pd.Series(10000,index=index)
		#self.control.collect_data(self.sim_start, self.sim_end)
		#print(self.price.display_data())
		#print(self.flex_cost.display_data())
		#print(self.ref_profile.display_data())
		store_namespace('price_'+self.building,self.price)
		store_namespace('flex_cost_'+self.building,self.flex_cost)
		store_namespace('ref_profile_'+self.building,self.ref_profile)

	def update_DRinfo(self, start, end,**kwargs):
		
		index = pd.date_range(start, end, freq = str(self.meas_sampl)+'S')
		# Random control signal
		flex_signal = pd.Series(0,index=index)
		#price_signal = pd.Series(np.random.rand(len(index))*5,index=index)
		if 'use_ref_file' in kwargs:
			ref_signal = load_namespace(kwargs['profile_file']).display_data()['ref_profile']
		else: 
			ref_signal = pd.Series(10000, index = index)
		#print(type(ref_signal))
		#print(ref_signal)
		
		flex_signal = pd.Series(0,index=index)
		if kwargs['DRevent_check'] == 1:
			print('%%%%%%%%% DR event triggered %%%%%%%%%%')
			for i in index:
				if i.hour >= kwargs['DRstart'] and i.hour < kwargs['DRend']:
					flex_signal.loc[i] = kwargs['flex_cost']
			flex_signal.sort_index()
		'''
		for i in index:
			if i.hour >= 7 and i.hour <= 10:
				price_signal[i] = np.random.uniform(0.8,1)*40
			if i.hour >= 17 and i.hour <= 23:
				price_signal[i] = np.random.uniform(0.8,1)*40
		'''		
		
		# Create DR event or not
		if 'DRevent_check' == 1 and use_ref_file == 0:
			print('%%%%%%%%% DR event triggered %%%%%%%%%%')
			#index = ref_signal.index.tz_convert(None)
			k = pd.Series(1,index=index)
			#flex_signal = pd.Series(0,index=index)
			#ref_signal.index = index

			if kwargs['DRevent_check'] == 1:
				for i in index:
					if i.hour >= kwargs['DRstart'] and i.hour <= kwargs['DRend']:
						k.loc[i] = 0.8
			print(k)
			print('Reference profile before modification:')
			print(ref_signal)
			# Sort the indices
			ref_signal.sort_index()
			k.sort_index()
			#flex_signal.sort_index()
			
			# Define the reference signal
			ref_signal = ref_signal * k * 30
			
			print('Reference profile after modification:')
			print(ref_signal)

		print('Flex cost:')
		print(flex_signal)
			
		#print(price_signal)
		# Define control data
		'''
		self.price.data = {"pi_e": variables.Timeseries('pi_e', price_signal,units.cents_kWh,tz_name=self.weather.tz_name)
		}
		'''
		self.flex_cost.data = {"flex_cost": variables.Timeseries('flex_cost', flex_signal,units.cents_kWh,tz_name=self.weather.tz_name)
		}
		self.ref_profile.data = {"ref_profile": variables.Timeseries('ref_profile', ref_signal, units.W,tz_name=self.weather.tz_name)
		}
		
		#pheat_max = pd.Series(10000,index=index)
		#self.control.collect_data(self.sim_start, self.sim_end)
		#print(self.price.display_data())
		#print(self.flex_cost.display_data())
		#print(self.ref_profile.display_data())
		#store_namespace('price_upd_'+self.building,self.price)
		store_namespace('flex_cost_upd_'+self.building,self.flex_cost)
		store_namespace('ref_profile_upd_'+self.building,self.ref_profile)	
		
	
	def opt_control_minDRCost(self):
		# Instantiate optimization problem
		self.opt_problem = optimization.Optimization(self.mpc,
												optimization.DRCostMin,
												optimization.JModelica,
												self.target_variable,
												constraint_data = self.constraints.data
												)

		print("%%%%%%%%%%%%%%%----Starting Optimization of Control ---%%%%%%%%%%%%%")		
		self.opt_problem.optimize(self.opt_start,self.opt_end,
									price_data = self.price.data,
									ref_profile = self.ref_profile.data,
									flex_cost = self.flex_cost.data
									)

		print("----Optimization stats----")
		print(self.opt_problem.get_optimization_statistics())

		print("----Optimization outcome----")
		self.opt_controlseq = self.opt_problem.Model.control_data # dictionary with mpcpy-timeseries variables of controls
		#print(opt_control)
		#print(opt_control['Qflow1'])
		#print(opt_control['Qflow2'])
		store_namespace('opt_control_DRcost_'+self.building,self.opt_controlseq)

	def emulate_opt(self,start,end):
		print("%%%---Emulating real system---%%%")
		self.emu.control_data = self.opt_controlseq
		emulate_jmod(self.emu, self.meas_vars, self.meas_sampl, start, end)
		print("Updating measurements")
		self.mpc.measurements = self.emu.measurements
		#print("Validating")
		#self.mpc.validate(start, end, 'val_opt', plot=1)
		#print(emu_opt.display_measurements('Measured')['TAir'])
		
		store_namespace('optemu_'+self.building,self.emu.display_measurements('Measured'))

			
	def init_refmodel(self, use_fmu, use_const):
		print("%%%---Initialising Reference Simulation---%%%")	
				
		# Define control profile for reference
		self.control_ref = exodata.ControlFromCSV(self.control_file,
										self.contr_varmap,
										tz_name = self.weather.tz_name)
		index = pd.date_range(self.sim_start, self.sim_end, freq = str(self.meas_sampl_ref) +'S')
		
		if use_const == 1:
			t_set = load_namespace(os.path.join('constraints_'+self.building)).data['TAir']['GTE'].get_base_data_data()+1
		else:
			t_set = pd.Series(20+273.15,index=index)
			for i in index:
				if i.hour >= 6 and i.hour <= 9:
					t_set[i] = 20 + 273.15
				if i.hour >= 17 and i.hour <= 23:
					t_set[i] = 20 + 273.15
		# Define control data
		self.control_ref.data = {"SetPoint": variables.Timeseries('SetPoint', t_set,units.K,tz_name=self.weather.tz_name)
		}
		
		if use_fmu == 0:
			self.emuref = systems.EmulationFromFMU(self.meas_vars_ref,
										moinfo=self.moinfo_emu_ref,
										weather_data = self.weather.data,
										control_data = self.control_ref.data,
										other_inputs = self.other_input.data,
										tz_name = self.weather.tz_name
										)
		else:
			self.emuref = systems.EmulationFromFMU(self.meas_vars_ref,
										fmupath=self.fmupath_ref,
										weather_data = self.weather.data,
										control_data = self.control_ref.data,
										other_inputs = self.other_input.data,
										tz_name = self.weather.tz_name
										)
	def run_reference(self, start, end):
		print("%%%---Running Reference Simulation---%%%")	
			
		emulate_jmod(self.emuref,self.meas_vars_ref, self.meas_sampl_ref, start, end)
				
		#print(self.emuref.display_measurements('Measured')['TAir'])	
		#print(self.emuref.display_measurements('Measured')['HeatInput'])
		
		store_namespace('ref_heatinput_'+self.building, self.emuref.display_measurements('Measured')['HeatInput'])
		
		store_namespace('ref_temp_'+self.building, self.emuref.display_measurements('Measured')['TAir'])
		
			
	def do_plot(self):
		print('%%%%%%%%%----Plot Optimized Control Signal and resulting temperature----%%%%%%')
		'''
		print("----Optimization temps based on MPC model----")
		b = opt_problem.Model.display_measurements('Simulated')['TAir']
		print(b)
		
		print("----Emulated temps----")
		c = opt_problem.Model.display_measurements('Measured')['TAir']
		
		a1_plot = opt_controlseq['ConvGain2'].display_data().plot(figsize=(11.69,8.27))
		#a2_plot = opt_controlseq['Qflow2'].display_data().plot(figsize=(11.69,8.27))
		a1_fig = a1_plot.get_figure()
		a1_fig.savefig("plot_convgain.pdf")
		a1_fig.clf()
		'''
		
		index = pd.date_range(self.opt_start, self.opt_end, freq = str(self.meas_sampl) +'S')
				
		sim = load_namespace('sim_val')[self.val_start:self.val_end]
		meas = load_namespace('meas_val')[self.val_start:self.val_end]
		
		opt_control_energy = load_namespace('opt_control_minenergy')
		opt_control_cost = load_namespace('opt_control_mincost')
		opt_control_DR = load_namespace('opt_control_DRcost')
		ref_heatinput = load_namespace('ref_heatinput')
		
		ref_temp = load_namespace('ref_temp')
		opt_temp = load_namespace('optemu')
		#print(opt_control['ConvGain2'].display_data())
		#print(ref_heatinput)
		opt_control_series_energy = opt_control_energy['ConvGain2'].display_data()
		opt_control_series_cost = opt_control_cost['ConvGain2'].display_data()
		opt_control_series_DR = opt_control_DR['ConvGain2'].display_data()
		
		
		#print(opt_temp)
		
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# Validation figure
		plt.figure(figsize=(11.69,8.27))
		plot_real = plt.plot(meas.index, meas['TAir'],'-x', label="Real")
		plot_mpc = plt.plot(sim.index, sim['TAir'], '--o', label="MPC model") 
		plt.legend(fontsize=14)
		plt.xlabel("Time",fontsize=18)
		plt.ylabel("Temperature [C]",fontsize=18)
		plt.title("Validation of MPC model",fontsize=22)
		plt.xticks(rotation=45)
		# We change the fontsize of minor ticks label 
		plt.tick_params(axis='both', which='major', labelsize=12)
		plt.tick_params(axis='both', which='minor', labelsize=12)
		plt.savefig("validation")
		plt.clf()
		
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# Optimization figure
		plt.figure(figsize=(11.69,8.27))
		plt.plot(opt_control_series_energy.index, opt_control_series_energy.values,'-o', label="MPC-EneMin")
		#plt.plot(opt_control_series_cost.index, opt_control_series_cost.values,'-+', label="MPC-CostMin")
		#plt.plot(opt_control_series_DR.index, opt_control_series_DR.values,'--o', label="MPC-DR")
		plt.plot(ref_heatinput.index, ref_heatinput.values,'--x', label="PI-ref") 		
		plt.legend(fontsize=14)
		plt.xlabel("Time",fontsize=18)
		plt.ylabel("Heat Input [W]",fontsize=18)
		plt.title("Optimized Control Sequences vs. PI controller",fontsize=22)
		plt.xticks(rotation=35)
		# We change the fontsize of minor ticks label 
		plt.tick_params(axis='both', which='major', labelsize=12)
		plt.tick_params(axis='both', which='minor', labelsize=12)
		plt.savefig("optimized_heatinput.pdf")
		plt.clf()
		
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# Optimization temps
		plt.figure(figsize=(11.69,8.27))
		plot_optimized = plt.plot(opt_temp.index, opt_temp.values,'-o', label="MPC")
		plot_ref = plt.plot(ref_temp.index, ref_temp.values,'-x', label="PI-ref") 		
		plt.legend(fontsize=14)
		plt.xlabel("Time",fontsize=18)
		plt.ylabel("Temperature [C]",fontsize=18)
		plt.title("Inside Temperatures with different controls",fontsize=22)
		plt.xticks(rotation=35)
		# We change the fontsize of minor ticks label 
		plt.tick_params(axis='both', which='major', labelsize=12)
		plt.tick_params(axis='both', which='minor', labelsize=12)
		plt.savefig("optimized_temps")
		plt.clf()
			
		#print(ref_heatinput)
		
		print('%%%% --- Optimised heat consumption (Wh) --- %%%%')
		print(sum(opt_control_energy['ConvGain2'].get_base_data().resample('3600S').mean()))
		
		print('%%%% --- Reference (PI) heat consumption (Wh) --- %%%%')
		print(sum(ref_heatinput.resample('3600S').mean()))


