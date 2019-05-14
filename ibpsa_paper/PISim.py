from buildingspy.simulate.Simulator import Simulator
from buildingspy.io.outputfile import Reader

from Simulator_HP_mod3 import SimHandler

from mpcpy import variables
from mpcpy import units

from funcs import store_namespace
from funcs import load_namespace

import os
import datetime
import pandas as pd
import numpy as np

def main():
	community = 'ResidentialCommunityUK_rad_2elements'
	sim_id = 'RBC'
	model_id = 'R2CW_HP'
	bldg_list = load_namespace(os.path.join('path_to_models', 'teaser_bldgs_residentialUK_10bldgs_fallback'))
	folder = 'results'
	bldg_index_start = 0
	bldg_index_end = 10
	mon = 'mar'

	print(bldg_list)
	
	# Overall options
	start = '3/1/2017 12:00:00'
	end = '3/1/2017 19:10:00'
	meas_sampl = '300' # Daylight saving time hours 26/3 and 29/10
	
	index = pd.date_range(start, end, freq = meas_sampl+'S')
	
	SimJmod_list = []
	SimDym_list = []
	i = 0
	
	index = pd.date_range(start, end, freq = meas_sampl+'S')
	
	for bldg in bldg_list[bldg_index_start:bldg_index_end]:
		
		print('In loop:  ' + str(i))
		i = i+1
	
		# Then JModelica model
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
		
		if i == 1:
			Sim.update_weather(start, end)	
		else:
			Sim.weather = SimJmod_list[i-2].weather
		
		Sim.get_other_input(start,end)
		Sim.get_constraints(start,end)
		
		Sim.param_file = os.path.join(Sim.simu_path,'csvs','Parameters_R2CW.csv')
		Sim.get_params()
		
		Sim.parameters.data = load_namespace(os.path.join(Sim.simu_path, 'sysid', 'sysid_HPrad_2element_'+mon+'_600S','est_params_'+bldg+'_'+model_id))
		Sim.other_input = load_namespace(os.path.join(Sim.simu_path, 'ibpsa_paper', 'decentr_enemin_'+mon, 'other_input_'+bldg+'_'+model_id))
		Sim.constraints =  load_namespace(os.path.join(Sim.simu_path, 'ibpsa_paper', 'decentr_enemin_'+mon, 'constraints_'+bldg+'_'+model_id))
		
		store_namespace(os.path.join(folder, 'params_'+Sim.building), Sim.parameters)
		store_namespace(os.path.join(folder, 'constraints_'+Sim.building), Sim.constraints)
		store_namespace(os.path.join(folder, 'other_input_'+Sim.building), Sim.other_input)
		
		# Add to list of simulations
		SimJmod_list.append(Sim)
		
	for Sim in SimJmod_list:

		Sim.init_refmodel(use_fmu = 1, use_const = 1, const_path = os.path.join(Sim.simu_path,'ibpsa_paper', 'decentr_enemin_'+mon))
		store_namespace(os.path.join(folder, 'control_ref_'+Sim.building), Sim.control_ref)
	
	for Sim in SimJmod_list:

		Sim.run_reference(start, end)
main()
	