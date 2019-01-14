""" Sixth version, make the code easier and more modifiable """
# Define the main programme

from funcs import store_namespace
from funcs import load_namespace
from funcs import emulate_jmod
import os
import datetime
import time
import pandas as pd
from multiprocessing import Pool
from mpcpy import units

from Simulator import SimHandler

if __name__ == "__main__":	
	
	# Naming conventions for the simulation
	community = 'ResidentialCommunityUK'
	sim_id = 'MinEne'
	model_id = 'R2CW'
	bldg_list = load_namespace(os.path.join('path to models'))
	
	# Overall options
	start = '1/7/2017 13:00:00'
	end = '1/7/2017 19:00:00'
	meas_sampl = '1800'
	horizon = 10 #time horizon for optimization in multiples of the sample
	
	sim_range = pd.date_range(start, end, freq = meas_sampl+'S')
	opt_start_str = start
	opt_end = datetime.datetime.strptime(end, '%m/%d/%Y %H:%M:%S') + datetime.timedelta(seconds = horizon*int(meas_sampl))
	opt_end_str = opt_end.strftime('%m/%d/%Y %H:%M:%S')
	bldg_index_start = 20
	bldg_index_end = 30
	
	# Instantiate Simulator
	Sim_list = []
	i = 0
	for bldg in bldg_list[bldg_index_start:bldg_index_end]:
		i = i+1
		Sim = SimHandler(sim_start = start,
					sim_end = end,
					meas_sampl = meas_sampl
					)
					
		Sim.moinfo_mpc = (os.path.join(Sim.simu_path, 'Tutorial_R2CW.mo'),
					'Tutorial_R2CW.R2CW',
					{}
					)
		
			
		Sim.building = bldg+'_'+model_id
		
		Sim.fmupath_mpc = os.path.join(Sim.simu_path, 'fmus',community, 'Tutorial_'+model_id+'_'+model_id+'.fmu')
		
		Sim.fmupath_emu = os.path.join(Sim.simu_path, 'fmus', community, community+'_'+bldg+'_'+bldg+'_Models_'+bldg+'_House_mpc.fmu')
		
		Sim.fmupath_ref = os.path.join(Sim.simu_path, 'fmus', community, community+'_'+bldg+'_'+bldg+'_Models_'+bldg+'_House_PI.fmu')
		
		Sim.moinfo_emu = (os.path.join(Sim.mod_path, community, bldg,bldg+'_Models',bldg+'_House_mpc.mo'),	community+'.'+bldg+'.'+bldg+'_Models.'+bldg+'_House_mpc',
		{}
		)
		
		Sim.moinfo_emu_ref = (os.path.join(Sim.mod_path, community, bldg,bldg+'_Models',bldg+'_House_PI.mo'),	community+'.'+bldg+'.'+bldg+'_Models.'+bldg+'_House_PI',
		{}
		)
		
		# Initialise exogenous data sources
		if i == 1:
			Sim.update_weather(start, opt_end_str)
		else:
			Sim.weather = Sim_list[i-2].weather
		Sim.get_control()
		Sim.get_DRinfo(start,opt_end_str)
		
		Sim.param_file = os.path.join(Sim.simu_path,'csvs','Parameters_R2CW.csv')
		Sim.get_params()
		
		Sim.parameters.data = load_namespace(os.path.join(Sim.simu_path, 'jmod sysid ResidentialCommunityUK results', 'est_params_'+Sim.building))
		Sim.other_input = load_namespace(os.path.join(Sim.simu_path, 'mincost vs minene', 'minene_opt results', 'other_input_'+Sim.building))
		Sim.constraints = load_namespace(os.path.join(Sim.simu_path, 'mincost vs minene', 'minene_opt results', 'constraints_'+Sim.building))
		
		# Add to list of simulations
		Sim_list.append(Sim)
		
		# Initialise models
		if i == 1:
			Sim.init_models(use_ukf=0, use_fmu_emu=1, use_fmu_mpc=0) # Use for initialising models
		else:
			Sim.init_models(use_ukf=0, use_fmu_emu=1, use_fmu_mpc=0) # Use for initialising models
		
	# Start the hourly loop
	i = 0
	emutemps = {}
	mpctemps = {}
	controlseq = {}
	opt_stats = {}
	emu_stats = {}
	refheat = []
	reftemps = []

	for Sim in Sim_list:
		emulate_jmod(Sim.emu, Sim.meas_vars, Sim.meas_sampl, '1/1/2017 00:00:00', start)
		Sim.start_temp = Sim.emu.display_measurements('Measured').values[-1][0]-273.15
		Sim.mpc.measurements = Sim.emu.measurements
		print(Sim.start_temp)
		
	
	for simtime in sim_range:
		i = i + 1
		print('%%%%%%%%% IN LOOP: ' + str(i) + ' %%%%%%%%%%%%%%%%%') 
		if i == 1:
			#simtime_str = simtime.strftime('%m/%d/%Y %H:%M:%S')
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

		for Sim in Sim_list:
			while True:
				try:
					print('Optimising building ' + Sim.building)
					
					out_temp = Sim.weather.display_data()['weaTDryBul'].resample(meas_sampl+'S').ffill()[opt_start_str]
					Sim.update_params('heatCapacitor.T.start',Sim.start_temp,units.degC)	
					Sim.update_params('heatCapacitor1.T.start',(7*Sim.start_temp+out_temp)/8,units.degC)
					
					# Optimise for next time step
					print("%%%%%% --- Optimising --- %%%%%%")
					Sim.opt_start = opt_start_str
					Sim.opt_end = opt_end_str
					
					# Optimise
					Sim.opt_control_minEnergy()
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
		while True:
			try:
				store_namespace('emutemps_'+sim_id+'_'+str(bldg_index_start)+'-'+str(bldg_index_end), emutemps)
				store_namespace('mpctemps_'+sim_id+'_'+str(bldg_index_start)+'-'+str(bldg_index_end), mpctemps)
				store_namespace('opt_stats_'+sim_id+'_'+str(bldg_index_start)+'-'+str(bldg_index_end), opt_stats)
				store_namespace('controlseq_'+sim_id+'_'+str(bldg_index_start)+'-'+str(bldg_index_end), controlseq)
				break
			except:
				print('%%%%%%%%%%%%%%%%%% Failed, trying again! %%%%%%%%%%%%%%%%%%%%%%')
				continue