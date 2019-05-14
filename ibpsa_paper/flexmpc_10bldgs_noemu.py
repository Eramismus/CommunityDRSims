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

from Simulator_HP_mod2 import SimHandler

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
    
    #down_weight = 0.5 # weights to define the reference profile from upper and lower limit
    #up_weight = 1-down_weight
    
    compr_capacity=float(3000)
    
    ramp_modifier = float(200/compr_capacity) # to further modify the load profile
    max_modifier = float(200/compr_capacity)
    
    dyn_price = 1
    stat_cost = 50
    set_change = 1
    
    sim_range = pd.date_range(start, end, freq = meas_sampl+'S')
    opt_start_str = start
    opt_end = datetime.datetime.strptime(end, '%m/%d/%Y %H:%M:%S') + datetime.timedelta(seconds = horizon*int(meas_sampl))
    opt_end_str = opt_end.strftime('%m/%d/%Y %H:%M:%S')
    
    init_start = sim_range[0] - datetime.timedelta(seconds = 4.5*3600)
    init_start_str = init_start.strftime('%m/%d/%Y %H:%M:%S')
    print(init_start_str)

    # Instantiate Simulator
    Sim_list = []
    i = 0
    for bldg in bldg_list[bldg_index_start:bldg_index_end]:
        i = i+1
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
        
        Sim.meas_vars_mpc = {'TAir': {}, 
                            'TAir1': {},
                            }
        Sim.meas_vars_mpc['TAir']['Sample'] = variables.Static('sample_rate_Tzone',
                    Sim.meas_sampl, # sample rate
                    units.s
                    ) # units
        Sim.meas_vars_mpc['TAir1']['Sample'] = variables.Static('sample_rate_Tzone1',
                    Sim.meas_sampl, # sample rate
                    units.s
                    ) # units
        
        Sim.meas_varmap_mpc = {'TAir': ('TAir',units.K), 
                                'TAir1': ('TAir1',units.K)
                            }
        
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
            Sim.update_weather(init_start_str, opt_end_str)
            Sim.get_DRinfo(init_start_str,opt_end_str)
            Sim.price = load_namespace(os.path.join(Sim.simu_path, 'ibpsa_paper', 'prices', 'sim_price_'+mon))
            index = pd.date_range(start, opt_end_str, freq = meas_sampl+'S', tz=Sim.weather.tz_name)
            if dyn_price == 0:
                price_signal = pd.Series(stat_cost, index)
                Sim.price.data = {"pi_e": variables.Timeseries('pi_e', price_signal,units.cents_kWh,tz_name=Sim.weather.tz_name)
                }
            store_namespace(os.path.join(folder, 'sim_price'), Sim.price)
        else:
            Sim.weather = Sim_list[i-2].weather
            Sim.ref_profile = Sim_list[i-2].ref_profile
            Sim.flex_cost = Sim_list[i-2].flex_cost
            Sim.price = Sim_list[i-2].price
            Sim.rho = Sim_list[i-2].rho
            Sim.addobj = Sim_list[i-2].addobj
        
        #Sim.sim_start= '1/1/2017 00:00'
        Sim.get_control()
        #Sim.sim_start= start
        Sim.get_other_input(init_start_str,opt_end_str)
        Sim.get_constraints(init_start_str,opt_end_str,upd_control=1)
        
        
        Sim.param_file = os.path.join(Sim.simu_path,'csvs','Parameters_R2CW.csv')
        Sim.get_params()
        
        if dyn_price == 0:
            Sim.control = load_namespace(os.path.join(Sim.simu_path, 'ibpsa_paper', '10bldgs_decentr_'+'nodyn_'+mon, 'control_'+bldg+'_'+model_id))
        else:
            Sim.control = load_namespace(os.path.join(Sim.simu_path, 'ibpsa_paper', '10bldgs_decentr_'+'dyn_'+mon, 'control_'+bldg+'_'+model_id))
        
        Sim.parameters.data = load_namespace(os.path.join(Sim.simu_path, 'sysid', 'sysid_HPrad_2element_'+mon+'_600S','est_params_'+Sim.building))
        Sim.other_input = load_namespace(os.path.join(Sim.simu_path, 'ibpsa_paper', 'decentr_enemin_'+mon, 'other_input_'+Sim.building))
        Sim.constraints = load_namespace(os.path.join(Sim.simu_path, 'ibpsa_paper', 'decentr_enemin_'+mon, 'constraints_'+Sim.building))
            
        # Add to list of simulations
        Sim_list.append(Sim)
        
        # Initialise models
        Sim.init_models(use_ukf=1, use_fmu_mpc=1, use_fmu_emu=1) # Use for initialising 
        
    # Start the hourly loop
    i = 0
    emutemps = {}
    mpctemps = {}
    controlseq = {}
    power = {}
    opt_stats = {}
    emu_stats = {}
    refheat = []
    reftemps = []
    wall_temp = {}
    
    index = pd.date_range(start, opt_end_str, freq = meas_sampl+'S')
    flex_cost_signal = pd.Series(0,index)
    
    for Sim in Sim_list:
        
        while True:
            try:
                emulate_jmod(Sim.emu, Sim.meas_vars_emu, Sim.meas_sampl, init_start_str, start)
                print(Sim.emu.display_measurements('Measured'))
                Sim.start_temp = Sim.emu.display_measurements('Measured').values[-1][-1]-273.15
                
                Sim.fmupath_emu = os.path.join(Sim.simu_path, 'fmus',community, 'Tutorial_'+model_id+'_test'+'_'+model_id+'_test'+'.fmu')
                
                Sim.meas_vars_emu = {'TAir': {}, 
                                    'TAir1': {},
                                    }
                Sim.meas_vars_emu['TAir']['Sample'] = variables.Static('sample_rate_Tzone',
                            Sim.meas_sampl, # sample rate
                            units.s
                            ) # units
                Sim.meas_vars_emu['TAir1']['Sample'] = variables.Static('sample_rate_Tzone1',
                            Sim.meas_sampl, # sample rate
                            units.s
                            ) # units
                
                Sim.meas_varmap_emu = {'TAir': ('TAir',units.K), 
                                        'TAir1': ('TAir1',units.K)
                                    }
                                    
                Sim.init_models(use_ukf=1, use_fmu_mpc=0, use_fmu_emu=1)
                
                out_temp=Sim.weather.display_data()['weaTDryBul'].resample(meas_sampl+'S').ffill()[init_start]
                
                wall_temp[Sim.building] = (7*Sim.start_temp+out_temp)/8
                
                print(Sim.start_temp)
                print(wall_temp[Sim.building])
        
                break
            except:
                print('%%%%%%%%%%%%%%%%%% Failed, trying again! %%%%%%%%%%%%%%%%%%%%%%')
                continue
        
        
        
    for simtime in sim_range:
        i = i + 1
        print('%%%%%%%%% IN LOOP: ' + str(i) + ' %%%%%%%%%%%%%%%%%') 
        if i == 1:
            simtime_str = 'continue'
        else:
            simtime_str = 'continue'
        opt_start_str = simtime.strftime('%m/%d/%Y %H:%M:%S')
        opt_end = simtime + datetime.timedelta(seconds = horizon*int(Sim.meas_sampl))
        emu_end = simtime + datetime.timedelta(seconds = int(Sim.meas_sampl))
        opt_end_str = opt_end.strftime('%m/%d/%Y %H:%M:%S')
        emu_end_str = emu_end.strftime('%m/%d/%Y %H:%M:%S')

        simtime_naive = simtime.replace(tzinfo=None)        
        
        print('---- Simulation time: ' + str(simtime) + ' -------')
        print('---- Next time step: ' + str(emu_end) + ' -------')
        print('---- Optimisation horizon end: ' + str(opt_end) + ' -------')
        
        emutemps = {}
        mpctemps = {}
        controlseq = {}
        power = {}
        opt_stats = {}
        emu_stats = {}
        
        for Sim in Sim_list:
            while True:
                try:
                       
                    out_temp=Sim.weather.display_data()['weaTDryBul'].resample(meas_sampl+'S').ffill()[opt_start_str]
                    
                    # Update parameter with measurements from the zone
                    Sim.update_params('C1start',Sim.start_temp+273.15,units.K)  
                    Sim.update_params('C2start',wall_temp[Sim.building]+273.15,units.K)
                    
                
                    # Optimise for next time step
                    print("%%%%%% --- Optimising --- %%%%%%")
                    Sim.opt_start = opt_start_str
                    Sim.opt_end = opt_end_str

                    # Shift to load tracking - down flexibility
                    if simtime.hour == DR_call_start.hour and simtime.minute == DR_call_start.minute:
                        print('%%%%%%%%%%%%%%% DR event called - flexibility profile defined %%%%%%%%%%%%%%%%%%%%%')
                        load_profile = Sim.opt_controlseq['HPPower'].display_data()
                        flex_cost_signal = pd.Series(0,index=index)
                        
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
                                
                        Sim.ref_profile.data['ref_profile'] = variables.Timeseries(
                        'ref_profile', 
                        load_profile, 
                        Sim.opt_controlseq['HPPower'].get_display_unit(), 
                        tz_name = Sim.weather.tz_name
                        )
                        
                        Sim.flex_cost.data = {"flex_cost": variables.Timeseries('flex_cost', flex_cost_signal,units.cents_kWh,tz_name=Sim.weather.tz_name)
                        }
                        
                        store_namespace('ref_profile_'+Sim.building, Sim.ref_profile)
                        store_namespace('flex_cost_'+Sim.building, Sim.flex_cost)
                    
                    if simtime_naive >= DR_ramp_start and simtime_naive <= DR_ramp_end:
                        print('%%%%%%%%%%%%%%% Load-tracking %%%%%%%%%%%%%%%%%%%%%')
                    else:
                        print('%%%%%%%%%%%%%%% No load_tracking %%%%%%%%%%%%%%%%%%%%%')
                        
                    Sim.opt_control_minDRCost()
                    load_profile = Sim.opt_controlseq['HPPower'].display_data()
                    
                    mpctemps[Sim.building] = Sim.mpc.display_measurements('Simulated')
                    print(Sim.mpc.display_measurements('Simulated'))
                    opt_stats[Sim.building] = Sim.opt_problem.get_optimization_statistics()
                    
                    print("Emulating response")
                    #Update control and emulate effects
                    Sim.control.data = Sim.opt_controlseq
                    Sim.mpc.control_data = Sim.opt_controlseq
                    
                    power[Sim.building] = Sim.opt_controlseq['HPPower'].display_data()[simtime]*3000
                    emutemps[Sim.building] = Sim.mpc.display_measurements('Simulated')['mpc_model.TAir'][emu_end]
                    
                    # Update start temperature for next round
                    Sim.start_temp = Sim.mpc.display_measurements('Simulated')['mpc_model.TAir'][emu_end]-273.15
                    wall_temp[Sim.building] = Sim.mpc.display_measurements('Simulated')['mpc_model.TAir1'][emu_end]-273.15
                    print(Sim.start_temp)
                    print(wall_temp[Sim.building])
                    controlseq[Sim.building] = Sim.opt_controlseq['HPPower'].display_data()

                    print("%%%%%%%%%%% Control Sequence %%%%%%%%%")
                    print(Sim.opt_controlseq['HPPower'].display_data()[simtime:])

                    break
                except:
                    print('%%%%%%%%%%%%%%%%%% Failed, trying again! %%%%%%%%%%%%%%%%%%%%%%')
                    continue
                    
            
            while True:
                try:
                    print('Storing all the stuff')
                    store_namespace(os.path.join(folder,'emutemps_'+sim_id+'_'+str(bldg_index_start)+'-'+str(bldg_index_end)+'_'+opt_start_str.replace('/','-').replace(':','-').replace(' ','-')), emutemps)
                    store_namespace(os.path.join(folder,'mpctemps_'+sim_id+'_'+str(bldg_index_start)+'-'+str(bldg_index_end)+'_'+opt_start_str.replace('/','-').replace(':','-').replace(' ','-')), mpctemps)
                    store_namespace(os.path.join(folder,'opt_stats_'+sim_id+'_'+str(bldg_index_start)+'-'+str(bldg_index_end)+'_'+opt_start_str.replace('/','-').replace(':','-').replace(' ','-')), opt_stats)
                    store_namespace(os.path.join(folder,'controlseq_'+sim_id+'_'+str(bldg_index_start)+'-'+str(bldg_index_end)+'_'+opt_start_str.replace('/','-').replace(':','-').replace(' ','-')), controlseq)
                    store_namespace(os.path.join(folder,'power_'+sim_id+'_'+str(bldg_index_start)+'-'+str(bldg_index_end)+'_'+opt_start_str.replace('/','-').replace(':','-').replace(' ','-')), power)
    
                    break
                except:
                    print('%%%%%%%%%%%%%%%%%% Failed, trying again! %%%%%%%%%%%%%%%%%%%%%%')
                    continue