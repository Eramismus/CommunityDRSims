""" Sixth version, make the code easier and more modifiable """
# Define the main programme

from funcs import store_namespace
from funcs import load_namespace
from funcs import emulate_jmod
import os
import datetime
import time
import pandas as pd
#rom multiprocessing import Pool
import numpy as np
import gc

from Simulator import SimHandler
from funcs import emulate_jmod
from mpcpy import units
from mpcpy import variables

# Naming conventions for the simulation
community = 'ResidentialCommunityUK'
sim_id = 'MinCost_Centr_Res'
bldg_list = load_namespace(os.path.join('path to models'))
#print(bldg_list)
# Overall options
start = '1/7/2017 13:00:00'
end = '1/7/2017 19:00:00'
meas_sampl = '1800'
horizon = 10 #time horizon for optimization, multiple of the measurement sample
use_previous = 0
emus = 10

sim_range = pd.date_range(start, end, freq = meas_sampl+'S')
opt_start_str = start
opt_end = datetime.datetime.strptime(end, '%m/%d/%Y %H:%M:%S') + datetime.timedelta(seconds = horizon*int(meas_sampl))
opt_end_str = opt_end.strftime('%m/%d/%Y %H:%M:%S')

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
SimAggr.fmupath_emu = os.path.join(SimAggr.simu_path, 'fmus', community,'AggrMPC_ResUK_Residential.fmu')

SimAggr.fmupath_mpc = os.path.join(SimAggr.simu_path, 'fmus', community,'AggrMPC_ResUK_Residential.fmu')

SimAggr.moinfo_emu = (os.path.join(SimAggr.simu_path, 'AggrMPC_ResUK.mo'),
			'AggrMPC_ResUK.Residential',
			{}
			)


# Initialise aggregated model
# Control
bldg_control = load_namespace(os.path.join(SimAggr.simu_path, 'jmod sysid ResidentialCommunityUK results', 'control_sysid'))
# Constraints
bldg_constraints = load_namespace(os.path.join(SimAggr.simu_path, 'mincost vs minene', 'minene_opt results', 'constraints_Detached_0_R2CW'))
	
bldg_other_input = load_namespace(os.path.join(SimAggr.simu_path, 'mincost vs minene', 'minene_opt results', 'other_input_Detached_0_R2CW'))

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


index = pd.date_range(start, opt_end_str, freq = meas_sampl+'S')



SimAggr.constraint_csv = os.path.join(SimAggr.simu_path,'csvs','Constraints_AggrRes.csv')
SimAggr.control_file = os.path.join(SimAggr.simu_path,'csvs','ControlSignal_AggrRes.csv')
SimAggr.price_file = os.path.join(SimAggr.simu_path,'csvs','PriceSignal.csv')
SimAggr.param_file = os.path.join(SimAggr.simu_path,'csvs','Parameters.csv')


# Initialise exogenous data sources
SimAggr.update_weather(start,opt_end_str)

SimAggr.get_control()

SimAggr.get_constraints(start,opt_end_str)
SimAggr.get_other_input(start,opt_end_str)

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
	
	#print(SimAggr.start_temp)
	
	for key in bldg_params:
		SimAggr.update_params(model_name+'.heatCapacitor.T.start',SimAggr.start_temp,unit=units.degC)
		SimAggr.update_params(model_name+'.heatCapacitor1.T.start',SimAggr.start_temp, unit=units.degC)
		SimAggr.update_params(model_name+'.'+key, bldg_params[key]['Value'].data, unit=bldg_params[key]['Value'].get_display_unit())
		#print(SimAggr.parameters.display_data())
		
	for key in bldg_control.data:
		SimAggr.control.data[key+'_'+bldg] = variables.Timeseries(
			name = key+'_'+bldg,
			timeseries = pd.Series(np.random.rand(len(index))*3000,index=index),
			display_unit = bldg_control.data[key].get_display_unit(),
			tz_name = SimAggr.weather.tz_name
			)
	
	for key in bldg_other_input.data:

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
	
if use_previous==1:
	SimAggr.parameters = load_namespace('params_'+SimAggr.building)
	SimAggr.control = load_namespace('control_'+SimAggr.building)
	SimAggr.other_input = load_namespace('other_input_'+SimAggr.building)
	SimAggr.constraints = load_namespace('constraints_'+SimAggr.building)

SimAggr.price = load_namespace('sim_price')

#print(SimAggr.meas_varmap)
store_namespace('params_'+SimAggr.building, SimAggr.parameters)
store_namespace('control_'+SimAggr.building,SimAggr.control)
store_namespace('other_input_'+SimAggr.building,SimAggr.other_input)
store_namespace('constraints_'+SimAggr.building,SimAggr.constraints)
store_namespace('price_'+SimAggr.building,SimAggr.price)



# Initialise models
SimAggr.init_models(use_ukf=0, use_fmu_emu=1, use_fmu_mpc=0) # Use for initialising models

Emu_list = []
i=0
j = 0
rand = np.random.choice(range(0,len(bldg_list)),emus, replace=False)
print(rand)
for bldg in bldg_list:
	for number in rand:
		if i == number:
			bldg = bldg_list[i]
			print('Instantiating emulation models, loop: ' + str(i))
			Sim = SimHandler(sim_start = start,
						sim_end = opt_end_str,
						meas_sampl = meas_sampl
						)
			
			model_id = 'R2CW'		
			
			Sim.moinfo_mpc = (os.path.join(Sim.simu_path, 'Tutorial_'+model_id+'.mo'),
						'Tutorial_'+model_id+'.'+model_id,
						{}
						)
			
			Sim.building = bldg+'_'+model_id
			
			Sim.fmupath_mpc = os.path.join(Sim.simu_path, 'fmus', community, 'Tutorial_'+model_id+'_'+model_id+'.fmu')
			
			Sim.fmupath_emu = os.path.join(Sim.simu_path, 'fmus', community, community+'_'+bldg+'_'+bldg+'_Models_'+bldg+'_House_mpc.fmu')
			
			Sim.fmupath_ref = os.path.join(Sim.simu_path, 'fmus', community, community+'_'+bldg+'_'+bldg+'_Models_'+bldg+'_House_PI.fmu')
			
			Sim.moinfo_emu = (os.path.join(Sim.mod_path, community, bldg,bldg+'_Models',bldg+'_House_mpc.mo'),	community+'.'+bldg+'.'+bldg+'_Models.'+bldg+'_House_mpc',
			{}
			)
			
			Sim.moinfo_emu_ref = (os.path.join(Sim.mod_path, community, bldg,bldg+'_Models',bldg+'_House_PI.mo'),	community+'.'+bldg+'.'+bldg+'_Models.'+bldg+'_House_PI',
			{}
			)
			
			Sim.weather = SimAggr.weather
			Sim.param_file = os.path.join(Sim.simu_path,'csvs','Parameters_R2CW.csv')
			Sim.weather = SimAggr.weather
			Sim.get_params()
			Sim.get_control()
	
			Sim.parameters.data = load_namespace(os.path.join(SimAggr.simu_path, 'jmod sysid ResidentialCommunityUK results', 'est_params_'+Sim.building))
			Sim.other_input = load_namespace(os.path.join(SimAggr.simu_path, 'mincost vs minene', 'minene_opt results', 'other_input_'+Sim.building))
			Sim.constraints = load_namespace(os.path.join(SimAggr.simu_path, 'mincost vs minene', 'minene_opt results', 'constraints_'+Sim.building))
	
			Sim.price = load_namespace('sim_price')
			
			Sim.init_models(use_ukf=0, use_fmu_emu = 1, use_fmu_mpc = 1)
			
			
			Emu_list.append(Sim)
	i = i+1
	
start_temps = []
k = 0
l = 0
for bldg in bldg_list:
	if k in rand:
		start_temps.append(Emu_list[l].start_temp)
		l = l+1
	else:
		start_temps.append(Emu_list[np.random.randint(0,len(Emu_list)-1)].start_temp)
	k = k+1


# Start the hourly loop
i = 0
emutemps = {}
mpctemps = {}
controlseq = {}
opt_stats = {}
emu_stats = {}
#refheat = []
#reftemps = []

gc.collect()

for Emu in Emu_list:
	# Needed to transfer the fmu to right place, it has its own weather processor
	emulate_jmod(Emu.emu, Emu.meas_vars, Emu.meas_sampl, '1/1/2017 00:00:00', start)

for simtime in sim_range:
	print('%%%%%%%%% IN SIMULATION LOOP: ' + str(i) + ' %%%%%%%%%%%%%%%%%') 
	i = i + 1
	if i == 1:
		#simtime_str = simtime.strftime('%m/%d/%Y %H:%M:%S')
		simtime_str = 'continue'
	else:
		simtime_str = 'continue'
	opt_start_str = simtime.strftime('%m/%d/%Y %H:%M:%S')
	opt_end = simtime + datetime.timedelta(seconds = horizon*int(SimAggr.meas_sampl))
	emu_end = simtime + datetime.timedelta(seconds = int(SimAggr.meas_sampl))
	opt_end_str = opt_end.strftime('%m/%d/%Y %H:%M:%S')
	emu_end_str = emu_end.strftime('%m/%d/%Y %H:%M:%S')  		
	
	print('---- Simulation time: ' + str(simtime) + ' -------')
	print('---- Next time step: ' + str(emu_end) + ' -------')
	print('---- Optimisation horizon end: ' + str(opt_end) + ' -------')
	
	mpctemps[opt_start_str] = {}
	emutemps[opt_start_str] = {}
	controlseq[opt_start_str] = {}
	emu_stats[opt_start_str] = {}
	
	
	# Update parameters with measurements from the zone
	j = 0
	for bldg in bldg_list:
		SimAggr.update_params(model_name+'.heatCapacitor.T.start',start_temps[j], units.degC)
		SimAggr.update_params(model_name+'.heatCapacitor1.T.start',start_temps[j], units.degC)
		j=j+1
	store_namespace('params_'+SimAggr.building, SimAggr.parameters)
	
	# Optimise for next time step
	print("%%%%%% --- Optimising --- %%%%%%")
	while True:
		try:
			SimAggr.opt_start = opt_start_str
			SimAggr.opt_end = opt_end_str
			
			# Optimise
			SimAggr.opt_control_minCost()
			opt_stats[opt_start_str] = SimAggr.opt_problem.get_optimization_statistics()
			mpctemps[opt_start_str][SimAggr.building] = SimAggr.mpc.display_measurements('Simulated')
			print(SimAggr.mpc.display_measurements('Simulated'))
			
			gc.collect()
			break
		except:
			print('%%%%%%%%%%%%%%%%%% Failed, trying again! %%%%%%%%%%%%%%%%%%%%%%')
			continue
			
	print("Emulating response")
	
	j=0
	flags = []
	for Emu in Emu_list:
		# Update control and emulate effects
		print('---- Emulating response of building: ' +  bldg + '  -------------')

		Emu.opt_controlseq = {'ConvGain2': variables.Timeseries(
								name = 'ConvGain2', 	
								timeseries = SimAggr.opt_controlseq['ConvGain2_'+Emu.building[:-(len(model_id)+1)]].display_data(),
								display_unit = SimAggr.opt_controlseq['ConvGain2_'+Emu.building[:-(len(model_id)+1)]].get_display_unit(), 
								tz_name = SimAggr.weather.tz_name
								)}
		
		try:
			Emu.emulate_opt(simtime_str,emu_end_str)
			flag = "Success"
			# Collect measurements
		except:
			print("NOTE: Emulation failed, using temps from other building")
			flag = "Fail"
			
		emutemps[opt_start_str][Emu.building] = Emu.emu.display_measurements('Measured').values[1][-1]-273.15
		flags.append(flag)
		j=j+1
		
	start_temps = []
	k = 0
	l = 0
	for bldg in bldg_list:
		controlseq[opt_start_str][bldg] = SimAggr.opt_controlseq['ConvGain2_'+bldg].display_data()
		if k in rand:
			start_temps.append(Emu_list[l].start_temp)
			l = l+1
		else:
			start_temps.append(Emu_list[np.random.randint(0,len(Emu_list)-1)].start_temp)
		k = k+1
	
	print(flags)
	print(start_temps)
	emu_stats[opt_start_str] = flags

	
	while True:
		try:
			store_namespace('emu_stats_'+SimAggr.building+'_'+sim_id, emu_stats)
			store_namespace('opt_stats_'+SimAggr.building+'_'+sim_id, opt_stats)
			store_namespace('emutemps_'+SimAggr.building+'_'+sim_id, emutemps)
			store_namespace('mpctemps_'+SimAggr.building+'_'+sim_id, mpctemps)
			store_namespace('controlseq_'+SimAggr.building+'_'+sim_id, controlseq)
			break
		except:
			print('%%%%%%%%%%%%%%%%%% Failed, trying again! %%%%%%%%%%%%%%%%%%%%%%')
			continue

#SimAggr.do_plot()