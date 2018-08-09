""" Sixth version, make the code easier and more modifiable """
# Define the main programme

from funcs import store_namespace
from funcs import load_namespace
from funcs import emulate_jmod
import os
import datetime
import time
import pandas as pd
import numpy as np
from mpcpy import units
from mpcpy import variables

from Simulator import SimHandler

if __name__ == "__main__":	
	
	# Naming conventions for the simulation
	community = 'ResidentialCommunityUK'
	sim_id = 'MinDRCost_hierarch'
	model_id = 'R2CW'
	bldg_list = load_namespace(os.path.join('path to models'))
	bldg_index_start = 0
	bldg_index_end = 30
	emus = 10 # Amount of emulations
	dyn_price = 0 # Whether to apply dynamic pricing or not
	
	# Overall options
	start = '1/7/2017 13:00:00'
	end = '1/7/2017 19:00:00'
	meas_sampl = '1800'
	horizon = 10 #time horizon for optimization in multiples of the sample
	
	DRstart = 15 # hour to start DR - ramp down 30 mins before
	DRend = 18 # hour to end DR - ramp 30 mins later
	DR_call = 5 # Round of loop to implement the call
	ramp_down = len(bldg_list)*2000
	max_output = len(bldg_list)*1000
	flex_cost = 90 # Cost for flexibility
	
	sim_range = pd.date_range(start, end, freq = meas_sampl+'S')
	opt_start_str = start
	opt_end = datetime.datetime.strptime(end, '%m/%d/%Y %H:%M:%S') + datetime.timedelta(seconds = horizon*int(meas_sampl))
	opt_end_str = opt_end.strftime('%m/%d/%Y %H:%M:%S')
	
	index = pd.date_range(start, opt_end_str, freq = meas_sampl+'S')
	rand = np.random.choice(range(0,len(bldg_list)),emus, replace=False)
	print(rand)
	# Instantiate Simulator
	Sim_list = []
	i = 0
	j = 0
	for bldg in bldg_list:
		for number in rand:
			if i == number:
				bldg = bldg_list[i]
				Sim = SimHandler(sim_start = start,
							sim_end = end,
							meas_sampl = meas_sampl
							)
							
				Sim.moinfo_mpc = (os.path.join(Sim.simu_path, 'Tutorial_R2CW.mo'),
							'Tutorial_R2CW.R2CW',
							{}
							)
				
					
				Sim.building = bldg+'_'+model_id
				
				Sim.fmupath_mpc = os.path.join(Sim.simu_path, 'fmus', community, 'Tutorial_'+model_id+'_'+model_id+'.fmu')
				
				Sim.fmupath_emu = os.path.join(Sim.simu_path, 'fmus',community, community+'_'+bldg+'_'+bldg+'_Models_'+bldg+'_House_mpc.fmu')
				
				Sim.fmupath_ref = os.path.join(Sim.simu_path, 'fmus',community, community+'_'+bldg+'_'+bldg+'_Models_'+bldg+'_House_PI.fmu')
				
				Sim.moinfo_emu = (os.path.join(Sim.mod_path, community, bldg,bldg+'_Models',bldg+'_House_mpc.mo'),	community+'.'+bldg+'.'+bldg+'_Models.'+bldg+'_House_mpc',
				{}
				)
				
				Sim.moinfo_emu_ref = (os.path.join(Sim.mod_path, community, bldg,bldg+'_Models',bldg+'_House_PI.mo'),	community+'.'+bldg+'.'+bldg+'_Models.'+bldg+'_House_PI',
				{}
				)
				
				# Initialise exogenous data sources
				if j == 0:
					Sim.update_weather(start, opt_end_str)
					#Sim.control = load_namespace('control_Detached_0_R2CW')
					Sim.get_control()
					Sim.get_DRinfo(start,opt_end_str)
					Sim.price = load_namespace('sim_price')
					
					if dyn_price == 0:
						price_signal = pd.Series(np.random.rand(len(index))*40,index=index)
					
						Sim.price.data = {"pi_e": variables.Timeseries('pi_e', price_signal,units.cents_kWh,tz_name=Sim.weather.tz_name)
						}			
					
				else:
					Sim.weather = Sim_list[j-1].weather
					Sim.flex_cost = Sim_list[j-1].flex_cost
					Sim.price = Sim_list[j-1].price
					Sim.ref_profile = Sim_list[j-1].ref_profile
					Sim.get_control()
					
				
				Sim.other_input = load_namespace(os.path.join(Sim.simu_path, 'mincost vs minene', 'minene_opt results', 'other_input_'+Sim.building))
				Sim.constraints = load_namespace(os.path.join(Sim.simu_path, 'mincost vs minene', 'minene_opt results', 'constraints_'+Sim.building))
				
				
				Sim.param_file = os.path.join(Sim.simu_path,'csvs','Parameters_R2CW.csv')
				Sim.get_params()
				Sim.parameters.data = load_namespace(os.path.join(Sim.simu_path, 'jmod sysid ResidentialCommunityUK results', 'est_params_'+Sim.building))
				
				# Update parameter with measurements from the zone
				Sim.update_params('heatCapacitor.T.start',Sim.start_temp,units.degC)			
				Sim.update_params('heatCapacitor1.T.start',Sim.start_temp, units.degC)
				
				# Add to list of simulations
				Sim_list.append(Sim)
				
				# Initialise models
				if i == 0:
					Sim.init_models(use_ukf=0, use_fmu_emu=1, use_fmu_mpc=0) # Use for initialising models
				else:
					Sim.init_models(use_ukf=0, use_fmu_emu=1, use_fmu_mpc=0) # Use for initialising models
				#Sim.init_refmodel(use_fmu=1)
				j = j+1
		i = i+1
		
	# Instantiate Simulator for aggregated optimisation
	SimAggr = SimHandler(sim_start = start,
				sim_end = opt_end_str,
				meas_sampl = meas_sampl
				)
				
	SimAggr.moinfo_mpc = (os.path.join(SimAggr.simu_path, 'AggrMPC_ResUK.mo'),
				'AggrMPC_ResUK.Residential',
				{}
				)
		
	SimAggr.building = 'AggrMPC_ResidentialUK'
	SimAggr.target_variable = 'TotalHeatPower'

	# Not really used in this case ...
	SimAggr.fmupath_emu = os.path.join(SimAggr.simu_path, 'fmus', community, 'AggrMPC_ResUK_Residential.fmu')

	SimAggr.fmupath_mpc = os.path.join(SimAggr.simu_path, 'fmus', community,  'AggrMPC_ResUK_Residential.fmu')

	SimAggr.moinfo_emu = (os.path.join(SimAggr.simu_path, 'AggrMPC_ResUK.mo'),
				'AggrMPC_ResUK.Residential',
				{}
				)

	# Initialise aggregated model
	# Control
	bldg_control = load_namespace(os.path.join(SimAggr.simu_path, 'jmod sysid ResidentialCommunityUK results', 'control_Detached_0_R2CW'))
	# Constraints
	bldg_constraints = load_namespace(os.path.join(SimAggr.simu_path, 'jmod sysid ResidentialCommunityUK results', 'constraints_Detached_0_R2CW'))
	
	bldg_other_input = load_namespace(os.path.join(SimAggr.simu_path, 'jmod sysid ResidentialCommunityUK results', 'other_input_Detached_0_R2CW'))

	# Optimisation constraints variable map
	SimAggr.optcon_varmap = {}
	SimAggr.contr_varmap = {}
	SimAggr.other_varmap = {}
	for bldg in bldg_list:
		model_name = bldg+'_R2CW'
		for key in bldg_control.data:
			SimAggr.contr_varmap[key+'_'+bldg] = (key+'_'+bldg, bldg_control.data[key].get_display_unit())
		
		for key in bldg_constraints.data:
			for key1 in bldg_constraints.data[key]:
				SimAggr.optcon_varmap[model_name+'_'+key1] = (model_name+'.'+key, key1, bldg_constraints.data[key][key1].get_display_unit())
		
		for key in bldg_other_input.data:
			SimAggr.other_varmap[model_name+'.'+key] = (model_name+'.'+key, bldg_other_input.data[key].get_display_unit())
	
	
	SimAggr.optcon_varmap['TotalHeatPower_min'] = ('TotalHeatPower', 'GTE', units.W)
	SimAggr.optcon_varmap['TotalHeatPower_max'] = ('TotalHeatPower', 'LTE', units.W)

	
	index = pd.date_range(start, opt_end_str, freq = meas_sampl+'S')


	SimAggr.constraint_csv = os.path.join(SimAggr.simu_path,'csvs','Constraints_AggrRes.csv')
	SimAggr.control_file = os.path.join(SimAggr.simu_path,'csvs','ControlSignal_AggrRes.csv')
	SimAggr.price_file = os.path.join(SimAggr.simu_path,'csvs','PriceSignal.csv')
	SimAggr.param_file = os.path.join(SimAggr.simu_path,'csvs','Parameters.csv')


	# Initialise exogenous data sources
	SimAggr.weather = Sim_list[0].weather
	SimAggr.get_control()
	SimAggr.get_constraints(start,opt_end_str)
	SimAggr.get_other_input(start,opt_end_str)
	SimAggr.get_DRinfo(start,opt_end_str)
	SimAggr.get_params()

	# Empty old data
	SimAggr.parameters.data = {}
	SimAggr.control.data = {}
	SimAggr.constraints.data = {}
	SimAggr.meas_varmap = {}
	SimAggr.meas_vars = {}

	index = pd.date_range(start, end, freq = meas_sampl+'S', tz=SimAggr.weather.tz_name)
	for bldg in bldg_list:
		#Parameters from system id
		bldg_params = load_namespace(os.path.join(SimAggr.simu_path, 'jmod sysid ResidentialCommunityUK results', 'est_params_'+bldg+'_R2CW'))
		bldg_other_input = load_namespace(os.path.join(SimAggr.simu_path, 'mincost vs minene', 'minene_opt results', 'other_input_'+bldg+'_R2CW'))
		bldg_constraints = load_namespace(os.path.join(SimAggr.simu_path, 'mincost vs minene', 'minene_opt results', 'constraints_'+bldg+'_R2CW'))
		model_name = bldg+'_R2CW'

		for key in bldg_params:
			SimAggr.update_params(model_name+'.heatCapacitor.T.start',SimAggr.start_temp,unit=units.degC)
			SimAggr.update_params(model_name+'.heatCapacitor1.T.start',SimAggr.start_temp, unit=units.degC)
			SimAggr.update_params(model_name+'.'+key, bldg_params[key]['Value'].data, unit=bldg_params[key]['Value'].get_display_unit())
			
		for key in bldg_control.data:
			SimAggr.control.data[key+'_'+bldg] = variables.Timeseries(
				name = key+'_'+bldg,
				timeseries = pd.Series(np.random.rand(len(index))*3000,index=index),
				display_unit = bldg_control.data[key].get_display_unit(),
				tz_name = SimAggr.weather.tz_name
				)
				
		for key in bldg_other_input.data:
			print(bldg_other_input.data)
			SimAggr.other_input.data[model_name+'.'+key] = variables.Timeseries(
				name = model_name+'.'+key,
				timeseries = bldg_other_input.display_data().loc[index][key],
				display_unit = bldg_other_input.data[key].get_display_unit(),
				tz_name = SimAggr.weather.tz_name
				)
		
		for key in bldg_constraints.data:
			if key == 'ConvGain2':
				SimAggr.constraints.data[key+'_'+bldg] = {}
			else:
				SimAggr.constraints.data[model_name+'.'+key] = {}
			for key1 in bldg_constraints.data[key]:
				if key == 'ConvGain2':
					SimAggr.constraints.data[key+'_'+bldg][key1] = variables.Timeseries(
						name = key+'_'+bldg+'_'+key1, 
						timeseries = bldg_constraints.data[key][key1].display_data().loc[index], 
						display_unit = bldg_constraints.data[key][key1].get_display_unit(), 
						tz_name = SimAggr.weather.tz_name
						)
				else:
					SimAggr.constraints.data[model_name+'.'+key][key1] = variables.Timeseries(
						name = model_name+'_'+key+'_'+key1, 
						timeseries = bldg_constraints.data[key][key1].display_data().loc[index], 
						display_unit = bldg_constraints.data[key][key1].get_display_unit(), 
						tz_name = SimAggr.weather.tz_name
						)
		
		SimAggr.meas_varmap[model_name+'.'+'TAir'] = (model_name+'.'+'TAir', units.K)
		SimAggr.meas_vars[model_name+'.'+'TAir'] = {}
		SimAggr.meas_vars[model_name+'.'+'TAir']['Sample'] = variables.Static('sample_rate_TAir', int(meas_sampl), units.s)
	
	# Shape the load profile
	load_profile = pd.Series(len(bldg_list)*10000,index=index)
	flex_signal = pd.Series(0,index=index)
			
	# Shape the constraints
	for i in index:
		if i.hour >= DRstart and i.hour <= DRend:
			load_profile[i] = max_output
			flex_signal[i] = flex_cost
		if i.hour == DRstart-1 and i.minute == 30:
			load_profile[i] = ramp_down
			flex_signal[i] = flex_cost
		if i.hour == DRend and i.minute == 30:
			load_profile[i] = ramp_down
			flex_signal[i] = flex_cost
	
	SimAggr.constraints.data['TotalHeatPower'] = {
		'LTE': variables.Timeseries('TotalHeatPower_LTE', pd.Series(load_profile,index=index), units.W, tz_name = SimAggr.weather.tz_name),
		'GTE': variables.Timeseries('TotalHeatPower_GTE', pd.Series(0,index=index), units.W, tz_name = SimAggr.weather.tz_name)
			}
	
	SimAggr.get_DRinfo(start,opt_end_str)
	SimAggr.flex_cost.data['flex_cost'] = variables.Timeseries('flex_cost', flex_signal, units.cents_kWh, tz_name = SimAggr.weather.tz_name) 
	SimAggr.price = Sim_list[0].price
	
	store_namespace('params_'+SimAggr.building, SimAggr.parameters)
	store_namespace('control_'+SimAggr.building,SimAggr.control)
	store_namespace('other_input_'+SimAggr.building,SimAggr.other_input)
	store_namespace('constraints_'+SimAggr.building,SimAggr.constraints)
	store_namespace('flex_cost_'+SimAggr.building,SimAggr.flex_cost)


	# Initialise models
	SimAggr.init_models(use_ukf=0, use_fmu_emu=1, use_fmu_mpc=0) # Use for initialising models

		
	# Start the hourly loop
	i = 0
	emutemps = {}
	mpctemps = {}
	controlseq = {}
	opt_stats = {}
	emu_stats = {}
	refheat = []
	reftemps = []
	
	for Emu in Sim_list:
		emulate_jmod(Emu.emu, Emu.meas_vars, Emu.meas_sampl, '1/1/2017 00:00:00', start)
		Emu.start_temp = Emu.emu.display_measurements('Measured').values[1][-1]-273.15
		Emu.mpc.measurements = Emu.emu.measurements
	
	start_temps = []
	k = 0
	for bldg in bldg_list:
		controlseq[opt_start_str][bldg] = SimAggr.opt_controlseq['ConvGain2_'+bldg].display_data()
		start_temps.append(Emu_list[k].start_temp)
		k = k+1
	
	
	for simtime in sim_range:
		print('%%%%%%%%% IN LOOP: ' + str(i) + ' %%%%%%%%%%%%%%%%%') 
		i = i + 1
		if i == 1:
			simtime_str = 'continue'
		else:
			simtime_str = 'continue'
		opt_start_str = simtime.strftime('%m/%d/%Y %H:%M:%S')
		opt_end = simtime + datetime.timedelta(seconds = horizon*int(Sim.meas_sampl))
		emu_end = simtime + datetime.timedelta(seconds = int(Sim.meas_sampl))
		opt_end_str = opt_end.strftime('%m/%d/%Y %H:%M:%S')
		emu_end_str = emu_end.strftime('%m/%d/%Y %H:%M:%S')  		
		
		print('---- Simulation time: ' + str(simtime) + ' -------')
		print('---- Next time step: ' + str(emu_end) + ' -------')
		print('---- Optimisation horizon end: ' + str(opt_end) + ' -------')
		
		mpctemps[opt_start_str] = {}
		emutemps[opt_start_str] = {}
		controlseq[opt_start_str] = {}
		opt_stats[opt_start_str] = {}
		emu_stats[opt_start_str] = {}
		
		if i==DR_call:
			print('%%%%%%%%%%%%%%% DR event called %%%%%%%%%%%%%%%%%%%%%')
			SimAggr.opt_start = opt_start_str
			SimAggr.opt_end = opt_end_str
			
			j = 0
			for bldg in bldg_list:
				SimAggr.update_params(model_name+'.heatCapacitor.T.start',start_temps[j], units.degC)
				SimAggr.update_params(model_name+'.heatCapacitor1.T.start',start_temps[j], units.degC)
				j=j+1
			store_namespace('params_'+SimAggr.building, SimAggr.parameters)
			
			while True:
				try:
					#SimAggr.opt_control_minEnergy()
					SimAggr.opt_control_minCost()
					#SimAggr.opt_control_minDRCost()
					break
				except:
					print('%%%%%%%%%%%%%%%%%% Failed, trying again! %%%%%%%%%%%%%%%%%%%%%%')
					continue
				
			ref_profiles = load_namespace('opt_control_mincost_'+SimAggr.building)
			for key in ref_profiles.keys():
				print(ref_profiles[key].display_data())
			#print(ref_profiles.display_data())
			
			for Sim in Sim_list[0:len(Sim_list)]:
				print(Sim.building)
				print(Sim.building[:-(len(model_id)+1)])
				Sim.ref_profile.data['ref_profile'] = variables.Timeseries(
					'ref_profile', 
					ref_profiles['ConvGain2_'+Sim.building[:-(len(model_id)+1)]].display_data(), 
					ref_profiles['ConvGain2_'+Sim.building[:-(len(model_id)+1)]].get_display_unit(), 
					tz_name = Sim.weather.tz_name
					)
				
				Sim.flex_cost.data = SimAggr.flex_cost.data
					
				store_namespace('ref_profile_'+Sim.building, Sim.ref_profile)
				print(Sim.ref_profile.display_data())

		for Sim in Sim_list[0:len(Sim_list)]:
			while True:
				try:
					print('Optimising building ' + Sim.building)
					
					# Update parameter with measurements from the zone
					Sim.update_params('heatCapacitor.T.start',Sim.start_temp,units.degC)	
					Sim.update_params('heatCapacitor1.T.start',Sim.start_temp,units.degC)
					
					# Optimise for next time step
					print("%%%%%% --- Optimising --- %%%%%%")
					Sim.opt_start = opt_start_str
					Sim.opt_end = opt_end_str
					
					# Optimise
					#Sim.opt_control_minEnergy()
					#Sim.opt_control_minCost()
					Sim.opt_control_minDRCost()
					
					mpctemps[opt_start_str][Sim.building] = Sim.mpc.display_measurements('Simulated')
					opt_stats[opt_start_str][Sim.building] = Sim.opt_problem.get_optimization_statistics()
					print(Sim.mpc.display_measurements('Simulated'))
					
					print("Emulating response")
					#Update control and emulate effects
					Sim.emulate_opt(simtime_str,emu_end_str)
					
					# Collect measurements
					emutemps[opt_start_str][Sim.building] = Sim.emu.display_measurements('Measured').values[1][-1]
					
					# Update start temperature for next round
					Sim.start_temp = Sim.emu.display_measurements('Measured').values[1][-1]-273.15
					
					controlseq[opt_start_str][Sim.building] = Sim.opt_controlseq['ConvGain2'].display_data()
				
					#print(Sim.emu.display_measurements('Measured'))
					print(Sim.opt_controlseq['ConvGain2'].display_data())
					break
				
				except:
					print('%%%%%%%%%%%%%%%%%% Failed, trying again! %%%%%%%%%%%%%%%%%%%%%%')
					continue
					
		start_temps = []
		k = 0
		l = 0
		for bldg in bldg_list:
			for number in rand:
				if k == number:
					start_temps.append(Sim_list[l].emu.display_measurements('Measured').values[1][-1]-273.15)
					l = l + 1
				else:
					start_temps.append(Sim_list[np.random.randint(0,len(Sim_list))].emu.display_measurements('Measured').values[1][-1]-273.15)
			k = k+1
		
		while True:
			try:
				store_namespace('emutemps_'+sim_id, emutemps)
				store_namespace('mpctemps_'+sim_id, mpctemps)
				store_namespace('opt_stats_'+sim_id, opt_stats)
				store_namespace('controlseq_'+sim_id, controlseq)
				break
			except:
				print('%%%%%%%%%%%%%%%%%% Failed, trying again! %%%%%%%%%%%%%%%%%%%%%%')
				continue
	#Sim.do_plot()