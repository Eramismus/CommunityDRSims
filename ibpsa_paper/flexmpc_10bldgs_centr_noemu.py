""" Sixth version, make the code easier and more modifiable """
# Define the main programme

from funcs import store_namespace
from funcs import load_namespace
from funcs import emulate_jmod
import os
import datetime
import time
import pandas as pd
#from multiprocessing import Pool
from mpcpy import units
from mpcpy import variables
from mpcpy import models_mod as models

from Simulator_HP_mod3 import SimHandler

if __name__ == "__main__":  
    
    # Naming conventions for the simulation
    community = 'ResidentialCommunityUK_rad_2elements'
    sim_id = 'MinEne'
    model_id = 'R2CW_HP'
    bldg_list = load_namespace(os.path.join('path_to_models', 'teaser_bldgs_residentialUK_10bldgs_fallback'))
    folder = 'results'
    bldg_index_start = 0
    bldg_index_end = 10
    
    # Overall options
    date = '11/20/2017 '
    start = date + '16:30:00'
    end = date + '19:00:00'
    meas_sampl = '300'
    horizon = 2*3600/float(meas_sampl) #time horizon for optimization in multiples of the sample
    mon = 'nov'
    
    DRstart = datetime.datetime.strptime(date + '17:30:00', '%m/%d/%Y %H:%M:%S') # hour to start DR - ramp down 30 mins before
    DRend = datetime.datetime.strptime(date + '18:30:00', '%m/%d/%Y %H:%M:%S') # hour to end DR - ramp 30 mins later
    DR_call_start = datetime.datetime.strptime(date + '17:00:00', '%m/%d/%Y %H:%M:%S') # Round of loop to implement the call
    DR_ramp_start = datetime.datetime.strptime(date + '17:30:00', '%m/%d/%Y %H:%M:%S')
    DR_ramp_end = datetime.datetime.strptime(date + '18:30:00', '%m/%d/%Y %H:%M:%S') # Round of loop to stop implementing the call
    flex_cost = 150 # Cost for flexibility

    compr_capacity=float(3000)
    
    ramp_modifier = float(2000/compr_capacity) # to further modify the load profile
    max_modifier = float(2000/compr_capacity)
    
    dyn_price = 1
    stat_cost = 50
    
    sim_range = pd.date_range(start, end, freq = meas_sampl+'S')
    opt_start_str = start
    opt_end = datetime.datetime.strptime(end, '%m/%d/%Y %H:%M:%S') + datetime.timedelta(seconds = horizon*int(meas_sampl))
    opt_end_str = opt_end.strftime('%m/%d/%Y %H:%M:%S')
    
    init_start = sim_range[0] - datetime.timedelta(seconds = 4.5*3600)
    init_start_str = init_start.strftime('%m/%d/%Y %H:%M:%S')
    print(init_start_str)
    
    

        # Instantiate Simulator for aggregated optimisation
    SimAggr = SimHandler(sim_start = start,
                sim_end = end,
                meas_sampl = meas_sampl
                )
                
    SimAggr.moinfo_mpc = (os.path.join(SimAggr.simu_path, 'AggrMPC_ResUK_10bldgs_heatpump_rad_fallback_test.mo'),
                'AggrMPC_ResUK_10bldgs_heatpump_rad_fallback_test.Residential',
                {}
                )
        
    SimAggr.building = 'AggrMPC_ResUK_10bldgs_heatpump_rad_fallback'
    SimAggr.target_variable = 'TotalHeatPower'

    # Not really used in this case ...
    SimAggr.fmupath_emu = os.path.join(SimAggr.simu_path, 'fmus', community, 'AggrMPC_ResUK_10bldgs_heatpump_rad_fallback_Residential.fmu')

    SimAggr.fmupath_mpc = os.path.join(SimAggr.simu_path, 'fmus', community, 'AggrMPC_ResUK_10bldgs_heatpump_rad_fallback_Residential.fmu')

    SimAggr.moinfo_emu = (os.path.join(SimAggr.simu_path, 'AggrMPC_ResUK_10bldgs_heatpump_rad_fallback'+'_test'+'.mo'),
                'AggrMPC_ResUK_10bldgs_heatpump_rad_fallback'+'_test'+'.Residential',
                {}
                )



    # Initialise aggregated model
    # Control
    bldg_control = load_namespace(os.path.join(SimAggr.simu_path, 'ibpsa_paper', '10bldgs_decentr_nodyn_jan', 'control_SemiDetached_2_R2CW_HP'))
    # Constraints
    bldg_constraints = load_namespace(os.path.join(SimAggr.simu_path, 'ibpsa_paper', '10bldgs_decentr_nodyn_jan', 'constraints_SemiDetached_2_R2CW_HP'))

    # Optimisation constraints variable map
    SimAggr.optcon_varmap = {}
    SimAggr.contr_varmap = {}
    SimAggr.addobj_varmap = {}
    SimAggr.slack_var = []
    
    for bldg in bldg_list:
        model_name = bldg+'_'+model_id+'_test'
        for key in bldg_control.data:
            SimAggr.contr_varmap[key+'_'+bldg] = (key+'_'+bldg, bldg_control.data[key].get_display_unit())
        
        for key in bldg_constraints.data:
            for key1 in bldg_constraints.data[key]:
                if key1 != 'Slack_GTE' and key1 != 'Slack_LTE':
                    SimAggr.optcon_varmap[model_name+'_'+key1] = (model_name+'.'+key, key1, bldg_constraints.data[key][key1].get_display_unit())
                else:
                    SimAggr.optcon_varmap[model_name+'_'+key1] = (model_name + '_TAir' + '_Slack', key1[-3:], units.degC)
                    
        
        SimAggr.slack_var.append(model_name +'_TAir'+ '_Slack')
        SimAggr.addobj_varmap[model_name + '_TAir' + '_Slack'] = (model_name + '_TAir' + '_Slack', units.unit1)

    index = pd.date_range(init_start, opt_end_str, freq = meas_sampl+'S')

    SimAggr.constraint_csv = os.path.join(SimAggr.simu_path,'csvs','Constraints_AggrRes.csv')
    SimAggr.control_file = os.path.join(SimAggr.simu_path,'csvs','ControlSignal_AggrRes.csv')
    SimAggr.price_file = os.path.join(SimAggr.simu_path,'csvs','PriceSignal.csv')
    SimAggr.param_file = os.path.join(SimAggr.simu_path,'csvs','Parameters.csv')

    # Initialise exogenous data sources
    SimAggr.update_weather(init_start_str,opt_end_str)
    SimAggr.get_DRinfo(init_start_str,opt_end_str)
    
    SimAggr.get_control()
    SimAggr.get_params()
    
    SimAggr.get_other_input(init_start_str,opt_end_str)
    SimAggr.get_constraints(init_start_str,opt_end_str,upd_control = 1)
    
    # Empty old data
    SimAggr.parameters.data = {}
    SimAggr.control.data = {}
    SimAggr.constraints.data = {}
    SimAggr.meas_varmap_mpc = {}
    SimAggr.meas_vars_mpc = {}
    SimAggr.other_input.data = {}
    
    index = pd.date_range(init_start_str, opt_end_str, freq = meas_sampl+'S', tz=SimAggr.weather.tz_name)
    for bldg in bldg_list:
        #Parameters from system id
        bldg_params = load_namespace(os.path.join(SimAggr.simu_path, 'sysid', 'sysid_HPrad_2element_'+mon+'_600S','est_params_'+bldg+'_'+model_id))
        bldg_other_input = load_namespace(os.path.join(SimAggr.simu_path, 'ibpsa_paper', 'decentr_enemin_'+mon, 'other_input_'+bldg+'_'+model_id))
        bldg_constraints = load_namespace(os.path.join(SimAggr.simu_path, 'ibpsa_paper', 'decentr_enemin_'+mon, 'constraints_'+bldg+'_'+model_id))
        
        model_name = bldg+'_'+model_id+'_test'
        
        pheat_min = pd.Series(0,index=index)
        pheat_max = pd.Series(1,index=index)
        bldg_constraints.data['HPPower']= {'GTE': variables.Timeseries('HPPower_GTE', pheat_min, units.W, tz_name=SimAggr.weather.tz_name), 
                    'LTE': variables.Timeseries('HPPower_LTE', pheat_max, units.W,tz_name=SimAggr.weather.tz_name)}

        for key in bldg_params:
            
            SimAggr.parameters.data[model_name+'.'+key] = {'Free': variables.Static('FreeOrNot', bldg_params[key]['Free'].data, units.boolean), 
                'Minimum': variables.Static('Min', bldg_params[key]['Minimum'].data, bldg_params[key]['Minimum'].get_display_unit()), 
                'Covariance': variables.Static('Covar', bldg_params[key]['Covariance'].data, bldg_params[key]['Covariance'].get_display_unit()), 
                'Value': variables.Static(model_name+'.'+key, bldg_params[key]['Value'].data, bldg_params[key]['Value'].get_display_unit()), 
                'Maximum': variables.Static('Max', bldg_params[key]['Maximum'].data, bldg_params[key]['Maximum'].get_display_unit())
                }
            
            SimAggr.update_params(model_name+'.heatCapacitor.T.start',SimAggr.start_temp,unit=units.degC)
            SimAggr.update_params(model_name+'.heatCapacitor1.T.start',SimAggr.start_temp, unit=units.degC)
        
        if dyn_price == 0:
            bldg_control = load_namespace(os.path.join(SimAggr.simu_path, 'ibpsa_paper', '10bldgs_decentr_'+'nodyn_'+mon, 'control_'+bldg+'_'+model_id))
        else:
            bldg_control = load_namespace(os.path.join(SimAggr.simu_path, 'ibpsa_paper', '10bldgs_decentr_'+'dyn_'+mon, 'control_'+bldg+'_'+model_id))
            
        for key in bldg_control.data:
            SimAggr.control.data[key+'_'+bldg] = variables.Timeseries(
                name = key+'_'+bldg,
                timeseries = bldg_control.data[key].display_data(),
                display_unit = bldg_control.data[key].get_display_unit(),
                tz_name = SimAggr.weather.tz_name
                )
        
        for key in bldg_constraints.data:
            if key == 'HPPower':
                SimAggr.constraints.data[key+'_'+bldg] = {}
            else:
                SimAggr.constraints.data[model_name+'.'+key] = {}
            for key1 in bldg_constraints.data[key]:
                if key == 'HPPower':
                    SimAggr.constraints.data[key+'_'+bldg][key1] = variables.Timeseries(
                        name = key+'_'+bldg+'_'+key1, 
                        timeseries = bldg_constraints.data[key][key1].display_data().loc[index], 
                        display_unit = bldg_constraints.data[key][key1].get_display_unit(), 
                        tz_name = SimAggr.weather.tz_name
                        )
                else:
                    if key1 == 'Slack_GTE' or key1 == 'Slack_LTE':
                        SimAggr.constraints.data[model_name+'.'+key][key1] = variables.Timeseries(
                            name = model_name+'_'+key+'_'+key1, 
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
        
        SimAggr.meas_varmap_mpc[model_name+'.'+'TAir'] = (model_name+'.'+'TAir', units.K)
        SimAggr.meas_varmap_mpc[model_name+'.'+'TAir1'] = (model_name+'.'+'TAir1', units.K)
        
        SimAggr.meas_vars_mpc[model_name+'.'+'TAir'] = {}
        SimAggr.meas_vars_mpc[model_name+'.'+'TAir']['Sample'] = variables.Static('sample_rate_TAir', int(meas_sampl), units.s)
        
        SimAggr.meas_vars_mpc[model_name+'.'+'TAir1'] = {}
        SimAggr.meas_vars_mpc[model_name+'.'+'TAir1']['Sample'] = variables.Static('sample_rate_TAir', int(meas_sampl), units.s)
    
        
    SimAggr.meas_varmap_emu = SimAggr.meas_varmap_mpc 
    SimAggr.meas_vars_emu = SimAggr.meas_vars_mpc
    
    
    index = pd.date_range(init_start_str, opt_end_str, freq = meas_sampl+'S')
    SimAggr.price = load_namespace(os.path.join('prices','sim_price_'+mon))

    if dyn_price == 0:
        index = pd.date_range(init_start_str, opt_end_str, freq = '1800S')
        price_signal = pd.Series(50,index=index)
        SimAggr.price.data = {"pi_e": variables.Timeseries('pi_e', price_signal,units.cents_kWh,tz_name=SimAggr.weather.tz_name)
        }
    
    store_namespace(os.path.join(folder, 'sim_price'), SimAggr.price)
        
    #print(SimAggr.meas_varmap)
    store_namespace(os.path.join(folder, 'params_'+SimAggr.building), SimAggr.parameters)
    store_namespace(os.path.join(folder, 'control_'+SimAggr.building), SimAggr.control)
    store_namespace(os.path.join(folder, 'constraints_'+SimAggr.building), SimAggr.constraints)
    
    SimAggr.init_models(use_ukf=0, use_fmu_emu=0, use_fmu_mpc=0) # Use for initialising models
    
    ''' Emulation list '''
    Emu_list = []
    i = 0
    for bldg in bldg_list[bldg_index_start:bldg_index_end]:
        i = i+1
        print('Instantiating emulation models, loop: ' + str(i))
        Sim = SimHandler(sim_start = start,
                    sim_end = end,
                    meas_sampl = meas_sampl
                    )
                    
        Sim.moinfo_mpc = (os.path.join(Sim.simu_path, 'Tutorial_'+model_id+'_test'+'.mo'),
                    'Tutorial_'+model_id+'_test'+'.'+model_id+'_test',
                    {}
                    )
        
            
        Sim.building = bldg+'_'+model_id
        
        Sim.fmupath_mpc = os.path.join(Sim.simu_path, 'fmus',community, 'Tutorial_'+model_id+'_test'+'_'+model_id+'_test'+'.fmu')
        
        Sim.fmupath_emu = os.path.join(Sim.simu_path, 'fmus', community, community+'_'+bldg+'_'+bldg+'_Models_'+bldg+'_House_mpc.fmu')
        
        Sim.fmupath_ref = os.path.join(Sim.simu_path, 'fmus', community, community+'_'+bldg+'_'+bldg+'_Models_'+bldg+'_House_PI.fmu')
        
        Sim.moinfo_emu = (os.path.join(Sim.mod_path, community, bldg,bldg+'_Models',bldg+'_House_mpc.mo'),  community+'.'+bldg+'.'+bldg+'_Models.'+bldg+'_House_mpc',
            {}
            )
        
        Sim.moinfo_emu_ref = (os.path.join(Sim.mod_path, community, bldg,bldg+'_Models',bldg+'_House_PI.mo'),   community+'.'+bldg+'.'+bldg+'_Models.'+bldg+'_House_PI',
        {}
        )
        
        Sim.meas_vars_emu = {'TAir': {}
                            }
                            
        Sim.meas_vars_emu['TAir']['Sample'] = variables.Static('sample_rate_Tzone',
                    Sim.meas_sampl, # sample rate
                    units.s
                    ) # units
        
        Sim.meas_varmap_emu = {'TAir': ('TAir',units.K)
                                }

        # Initialise exogenous data sources
        if i == 1:
            Sim.weather = SimAggr.weather
            Sim.get_DRinfo(init_start_str,opt_end_str)
            Sim.flex_cost = SimAggr.flex_cost
            Sim.price = SimAggr.price
            Sim.rho = SimAggr.rho
            #Sim.addobj = SimAggr.addobj
        else:
            Sim.weather = Emu_list[i-2].weather
            Sim.flex_cost = Emu_list[i-2].flex_cost
            Sim.price = Emu_list[i-2].price
            Sim.rho = Emu_list[i-2].rho
            Sim.addobj = Emu_list[i-2].addobj
        
        #Sim.sim_start= '1/1/2017 00:00'
        Sim.get_control()
        Sim.get_other_input(init_start_str,opt_end_str)
        Sim.get_constraints(init_start_str,opt_end_str,upd_control=1)

        Sim.param_file = os.path.join(Sim.simu_path,'csvs','Parameters_R2CW.csv')
        Sim.get_params()
        
        Sim.parameters.data = load_namespace(os.path.join(Sim.simu_path, 'sysid', 'sysid_HPrad_2element_'+mon+'_600S','est_params_'+Sim.building))
        Sim.other_input = load_namespace(os.path.join(Sim.simu_path, 'ibpsa_paper', 'decentr_enemin_'+mon, 'other_input_'+Sim.building))
        Sim.constraints = load_namespace(os.path.join(Sim.simu_path, 'ibpsa_paper', 'decentr_enemin_'+mon, 'constraints_'+Sim.building))
        
        # Add to list of simulations
        Emu_list.append(Sim)

        
    # Start the hourly loop
    i = 0
    emutemps = {}
    mpctemps = {}
    controlseq = {}
    power = {}
    opt_stats = {}
    refheat = []
    reftemps = []

    index = pd.date_range(start, opt_end_str, freq = meas_sampl+'S')
    flex_cost_signal = pd.Series(0,index=index)

    start_temps = []
    start_temp = {}
    wall_temp = {}
          
    out_temp=SimAggr.weather.display_data()['weaTDryBul'].resample(meas_sampl+'S').ffill()[start]
    
    # Initialise models
    for Sim in Emu_list:
        
        while True:
            try:
         
                Sim.init_models(use_ukf=1, use_fmu_mpc=1, use_fmu_emu=1) # Use for initialising 
                emulate_jmod(Sim.emu, Sim.meas_vars_emu, Sim.meas_sampl, init_start_str, start)
                
                Sim.start_temp = Sim.emu.display_measurements('Measured').values[-1][-1]-273.15
                print(Sim.emu.display_measurements('Measured'))

                start_temp[Sim.building] = Sim.start_temp
                wall_temp[Sim.building] = (7*Sim.start_temp+out_temp)/8
                print(out_temp)
                print(wall_temp[Sim.building])
                print(Sim.start_temp)
               
                SimAggr.update_params(Sim.building+'_test.heatCapacitor1.T.start', wall_temp[Sim.building]+273.15, units.K)
                SimAggr.update_params(Sim.building+'_test.heatCapacitor.T.start', start_temp[Sim.building]+273.15, units.K)
                break
            except:
                print('%%%%%%%%%%%%%%%%%% Failed, trying again! %%%%%%%%%%%%%%%%%%%%%%')
                continue
        
    Emu_list = [] # Not needed anymore

    print(sim_range)    
    for simtime in sim_range:
        i = i + 1
        print('%%%%%%%%% IN LOOP: ' + str(i) + ' %%%%%%%%%%%%%%%%%') 
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

        simtime_naive = simtime.replace(tzinfo=None)        
        
        print('---- Simulation time: ' + str(simtime) + ' -------')
        print('---- Next time step: ' + str(emu_end) + ' -------')
        print('---- Optimisation horizon end: ' + str(opt_end) + ' -------')
        
        while True:
            try:
                # Optimise for next time step
                print("%%%%%% --- Optimising --- %%%%%%")
                SimAggr.opt_start = opt_start_str
                SimAggr.opt_end = opt_end_str
                   
                out_temp=SimAggr.weather.display_data()['weaTDryBul'].resample(meas_sampl+'S').ffill()[opt_start_str]
                
                for bldg in bldg_list:  
                    building = bldg + '_' + model_id
                    SimAggr.update_params(building+'_test.heatCapacitor.T.start', start_temp[building]+273.15, units.K)
                    SimAggr.update_params(building+'_test.heatCapacitor1.T.start', wall_temp[building]+273.15, units.K)
                
                
                if simtime.hour == DR_call_start.hour and simtime.minute == DR_call_start.minute:
                    print('%%%%%%%%%%%%%%% DR event called - flexibility profile defined %%%%%%%%%%%%%%%%%%%%%')
                    flex_cost_signal = pd.Series(0,index=index)
                    
                    j = 0
                    for bldg in bldg_list:
                        if j == 0:
                            load_profiles = pd.Series(SimAggr.opt_controlseq['HPPower_' + bldg].display_data().values, index = SimAggr.opt_controlseq['HPPower_' + bldg].display_data().index)
                        else:
                            load_profile = pd.Series(SimAggr.opt_controlseq['HPPower_' + bldg].display_data().values, index = SimAggr.opt_controlseq['HPPower_' + bldg].display_data().index)
                            load_profiles = pd.concat([load_profiles, load_profile], axis=1)
                        j = j+1
                    
                    load_profile_aggr  = load_profiles.sum(axis=1)
                    
                     #Shape the profile if required
                    for t in load_profile.index:
                        t = t.replace(tzinfo = None)
                        if t >= DRstart and t < DRend:
                            print(load_profile[t])
                            load_profile[t] = load_profile[t]-max_modifier
                            print(load_profile[t])
                            flex_cost_signal[t] = flex_cost 
                        if load_profile[t] < 0:
                            load_profile[t] = 0
                    
                    SimAggr.flex_cost.data = {"flex_cost": variables.Timeseries('flex_cost', flex_cost_signal, units.cents_kWh,tz_name=Sim.weather.tz_name)}
                    
                    SimAggr.ref_profile.data['ref_profile'] = variables.Timeseries(
                                    'ref_profile', 
                                    load_profile_aggr, 
                                    SimAggr.opt_controlseq['HPPower_'+bldg_list[0]].get_display_unit(), 
                                    tz_name = Sim.weather.tz_name
                                    )
                    
                    SimAggr.flex_cost.data = {"flex_cost": variables.Timeseries('flex_cost', flex_cost_signal,units.cents_kWh,tz_name=Sim.weather.tz_name)
                    }
                    
                    store_namespace('ref_profile_'+SimAggr.building, SimAggr.ref_profile)
                    store_namespace('flex_cost_'+SimAggr.building, SimAggr.flex_cost)
                
                if simtime_naive >= DR_ramp_start and simtime_naive <= DR_ramp_end:
                    print('%%%%%%%%%%%%%%% Load-tracking %%%%%%%%%%%%%%%%%%%%%')
                else:
                    print('%%%%%%%%%%%%%%% No load_tracking %%%%%%%%%%%%%%%%%%%%%')
                    #Sim.constraints = Sim.constraints_reg
                
                # ### Optimise ### 
                SimAggr.opt_control_minDRCost()
                
                print(SimAggr.mpc.measurements)
                print(SimAggr.mpc.display_measurements('Simulated'))
                
                mpctemps = SimAggr.mpc.display_measurements('Simulated')
                opt_stats = SimAggr.opt_problem.get_optimization_statistics()


                for bldg in bldg_list:  
                    building = bldg + '_' + model_id
                    
                    power[building] = SimAggr.opt_controlseq['HPPower_' + bldg].display_data()[simtime]*3000
                    
                    controlseq[building] = SimAggr.opt_controlseq['HPPower_' + bldg].display_data()
                    
                    emutemps[building] = SimAggr.mpc.display_measurements('Simulated')['mpc_model.'+building+'_test.TAir'][emu_end]
                    
                    start_temp[building] = SimAggr.mpc.display_measurements('Simulated')['mpc_model.'+building+'_test.TAir'][emu_end]-273.15
                    
                    wall_temp[building] = SimAggr.mpc.display_measurements('Simulated')['mpc_model.'+building+'_test.TAir1'][emu_end]-273.15
                    
                    print(start_temp[building])
                    print(wall_temp[building])
                break
            except:
                print('%%%%%%%%%%%%%%%%%% Failed, trying again! %%%%%%%%%%%%%%%%%%%%%%')
                continue
        
        while True:
            try:
                store_namespace(os.path.join(folder,'opt_stats_'+SimAggr.building+'_'+sim_id+'_'+opt_start_str.replace('/','-').replace(':','-').replace(' ','-')), opt_stats)
                store_namespace(os.path.join(folder,'emutemps_'+SimAggr.building+'_'+sim_id+'_'+opt_start_str.replace('/','-').replace(':','-').replace(' ','-')), emutemps)
                store_namespace(os.path.join(folder,'mpctemps_'+SimAggr.building+'_'+sim_id+'_'+opt_start_str.replace('/','-').replace(':','-').replace(' ','-')), mpctemps)
                store_namespace(os.path.join(folder,'power_'+SimAggr.building+'_'+sim_id+'_'+opt_start_str.replace('/','-').replace(':','-').replace(' ','-')), power)
                store_namespace(os.path.join(folder,'controlseq_'+SimAggr.building+'_'+sim_id+'_'+opt_start_str.replace('/','-').replace(':','-').replace(' ','-')), controlseq)
                break
            except:
                print('%%%%%%%%%%%%%%%%%% Failed, trying again! %%%%%%%%%%%%%%%%%%%%%%')
                continue