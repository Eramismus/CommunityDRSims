
# coding: utf-8

# # PuLP testing

# In[32]:


import pulp
# Import PuLP modeler functions
from pulp import *

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

from pulp_funcs import *


# In[116]:


community = 'ResidentialCommunityUK_rad_2elements'
sim_id = 'MinEne'
model_id = 'R2CW_HP'
arx_model = 'ARX_lag_4_exog4'
bldg_list = load_namespace(os.path.join('path_to_folder', 'teaser_bldgs_residential'))
folder = 'path_to_folder'
bldg_index_start = 0
bldg_index_end = 30

# Overall options
date = '11/20/2017 '
start = date + '16:30:00'
end = date + '19:00:00'
meas_sampl = '300'
horizon = 3.0*3600.0/float(meas_sampl) #time horizon for optimization in multiples of the sample
mon = 'nov'

DRstart = datetime.datetime.strptime(date + '18:00:00', '%m/%d/%Y %H:%M:%S') # hour to start DR - ramp down 30 mins before
DRend = datetime.datetime.strptime(date + '18:30:00', '%m/%d/%Y %H:%M:%S') # hour to end DR - ramp 30 mins later
DR_call_start = datetime.datetime.strptime(date + '17:00:00', '%m/%d/%Y %H:%M:%S') # Round of loop to implement the call
DR_ramp_start = datetime.datetime.strptime(date + '17:30:00', '%m/%d/%Y %H:%M:%S')
DR_ramp_end = datetime.datetime.strptime(date + '18:00:00', '%m/%d/%Y %H:%M:%S') # Round of loop to stop implementing the call

# reduction of demand
compr_capacity_list=[float(4500.0)]*10+[float(3000.0)]*20
print(compr_capacity_list)
ramp_modifier = float(200.0) # to further modify the load profile
max_modifier = float(200.0)


# Pricing
dyn_price = 0
stat_cost = 50
flex_cost = 100 # Utilisation cost for flexibility
rho = 500 # Cost of comfort violations

lag = 13 # Number of delay terms to take from measurements
power_lag = 1 # Lag for last control action
temp_lag1 = 4 # Continuous immediate lag of temps for optimisation 
temp_lag2 = 13 # Lag from further away for temps

sim_range = pd.date_range(start, end, freq = meas_sampl+'S')
opt_start_str = start
opt_end = datetime.datetime.strptime(end, '%m/%d/%Y %H:%M:%S') + datetime.timedelta(seconds = horizon*int(meas_sampl))
opt_end_str = opt_end.strftime('%m/%d/%Y %H:%M:%S')

if mon == 'jan':
    init_start = sim_range[0] - datetime.timedelta(seconds = 1.5*3600)
else: 
    init_start = sim_range[0] - datetime.timedelta(seconds = 4.5*3600)
    
init_start_str = init_start.strftime('%m/%d/%Y %H:%M:%S')
print(init_start_str)


Sim_list = []
i = 0
for bldg in bldg_list[bldg_index_start:bldg_index_end]:
    i = i+1
    Sim = SimHandler(sim_start = start,
                sim_end = end,
                meas_sampl = meas_sampl
                )
    Sim.building = bldg+'_'+model_id
    
    Sim.compr_capacity = compr_capacity_list[i-1]

    Sim.fmupath_emu = os.path.join(Sim.simu_path, 'fmus', community, community+'_'+bldg+'_'+bldg+'_Models_'+bldg+'_House_mpc.fmu')

    Sim.fmupath_ref = os.path.join(Sim.simu_path, 'fmus', community, community+'_'+bldg+'_'+bldg+'_Models_'+bldg+'_House_PI.fmu')

    Sim.moinfo_emu = (os.path.join(Sim.mod_path, community, bldg,bldg+'_Models',bldg+'_House_mpc.mo'),  community+'.'+bldg+'.'+bldg+'_Models.'+bldg+'_House_mpc',
    {}
    )

    Sim.moinfo_emu_ref = (os.path.join(Sim.mod_path, community, bldg,bldg+'_Models',bldg+'_House_PI.mo'),   community+'.'+bldg+'.'+bldg+'_Models.'+bldg+'_House_PI',
    {}
    )

    # Initialise exogenous data sources
    if i == 1:
        Sim.update_weather(init_start_str, opt_end_str)
        Sim.get_DRinfo(init_start_str,opt_end_str)

        index = pd.date_range(start, opt_end_str, freq = meas_sampl+'S', tz=Sim.weather.tz_name)
        
        Sim.price = load_namespace(os.path.join(Sim.simu_path, 'JournalPaper', 'drcases', 'decentr_costmin_30bldgs_'+mon, 'sim_price'))

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
        #Sim.rho = Sim_list[i-2].rho
        #Sim.addobj = Sim_list[i-2].addobj

    #Sim.sim_start= '1/1/2017 00:00'
    Sim.get_control()
    #Sim.sim_start= start
    Sim.get_other_input(init_start_str,opt_end_str)
    Sim.get_constraints(init_start_str,opt_end_str,upd_control=1)

    #Sim.param_file = os.path.join(Sim.simu_path,'csvs','Parameters_R2CW.csv')
    #Sim.get_params()

    #Sim.parameters.data = load_namespace(os.path.join(Sim.simu_path, 'sysid', 'sysid_HPrad_2element_'+mon+'_600S','est_params_'+Sim.building))
    Sim.other_input = load_namespace(os.path.join(Sim.simu_path, 'JournalPaper', 'drcases', 'decentr_enemin_constr_'+mon, 'other_input_'+Sim.building))
    Sim.constraints = load_namespace(os.path.join(Sim.simu_path, 'JournalPaper', 'drcases', 'decentr_enemin_constr_'+mon, 'constraints_'+Sim.building))

    # Add to list of simulations
    Sim_list.append(Sim)

    # Initialise models
    Sim.init_models(use_ukf=1, use_fmu_mpc=0, use_fmu_emu=1) # Use for initialising 


# In[136]:


# Get ARX model
for Sim in Sim_list:
    Sim.ARX_model = load_namespace(os.path.join(Sim.simu_path, 'JournalPaper', 'drcases', 'results_sysid_new_'+mon, arx_model, 'sysid_ARXmodel_'+mon+'_'+Sim.building))


# In[137]:


# Initialise models
for Sim in Sim_list: 
   emulate_jmod(Sim.emu, Sim.meas_vars_emu, Sim.meas_sampl, init_start_str, start)

   Sim.start_temp = Sim.emu.display_measurements('Measured').values[-1][-1]
   print(Sim.emu.display_measurements('Measured'))
   print(Sim.start_temp-273.15)
           


# In[138]:


# Start the loop
i = 0
emutemps = {}
mpctemps = {}
controlseq = {}
power = {}
opt_stats = {}
emu_stats = {}

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
        '''
        while True:
            try:   
        ''' 
        
        # Optimise for next time step
        print("%%%%%% --- Optimising --- %%%%%%")
        Sim.opt_start = opt_start_str
        Sim.opt_end = opt_end_str
        
        opt_index = pd.date_range(Sim.opt_start, Sim.opt_end, freq = meas_sampl+'S')
        #opt_index = pd.date_range(Sim.opt_start, Sim.opt_end, freq = meas_sampl+'S').strftime('%m/%d/%Y %H:%M:%S')
        
        # Shift to load tracking - down flexibility
        if simtime == DR_call_start:
            print('%%%%%%%%%%%%%%% DR event called - flexibility profile defined %%%%%%%%%%%%%%%%%%%%%')
            load_profile = Sim.opt_controlseq['HPPower'].display_data()
            #load_profile = controlseq[opt_start_str_prev][Sim.building]
            #Sim.constraints = Sim.constraints_down
            load_profile.reindex(index).fillna(0)
            ref_profile = load_profile
            flex_cost_signal = pd.Series(float(0),index=index)
            
            DR_call_bin = 1
            
            #Shape the profile if required
            for t in index:
                t = t.replace(tzinfo = None)
                if t >= DRstart and t < DRend:
                    #print(load_profile[t])
                    ref_profile[t] = float(load_profile[t]) - (float(max_modifier) / Sim.compr_capacity)
                    #print(ref_profile[t])
                    if ref_profile[t] < 0:
                        ref_profile[t] = 0
                        
                if t > DRstart and t < DRend:
                    flex_cost_signal[t] = flex_cost 

            Sim.ref_profile.data['ref_profile'] = variables.Timeseries(
            'ref_profile', 
            ref_profile, 
            Sim.opt_controlseq['HPPower'].get_display_unit(), 
            tz_name = Sim.weather.tz_name
            )
            
            Sim.flex_cost.data = {"flex_cost": variables.Timeseries('flex_cost', flex_cost_signal,units.cents_kWh,tz_name=Sim.weather.tz_name)
            }
            
           # print(max_modifier)
            #print(load_profile)
            #print(ref_profile)
            #print(Sim.flex_cost.display_data())
            #exit()
            
            store_namespace('ref_profile_'+Sim.building, Sim.ref_profile)
            store_namespace('flex_cost_'+Sim.building, Sim.flex_cost)
    
        # Initialise the problem
        
        if i == 1:
            for j in range(0, lag):
                Sim.start_temps.append(Sim.emu.display_measurements('Measured').values[-j-1][-1]-273.15) # start 16:30
                Sim.start_powers.append(Sim.emu.display_measurements('Measured')['PowerCompr'].values[-j-1] / 1000.0 ) #
            print(Sim.start_temps)
            print(Sim.start_powers)
        else:
            Sim.start_temps.insert(0, Sim.emu.display_measurements('Measured').values[-1][-1]-273.15) # start 16:30
            Sim.start_powers.insert(0, Sim.emu.display_measurements('Measured')['PowerCompr'].values[-1] / 1000.0) #
            
            Sim.start_temps.pop(lag)
            Sim.start_powers.pop(lag)
            
            print(Sim.start_temps)
            print(Sim.start_powers)
        
        prob = None
        prob, hp_power, intemp, slack_1, slack_2,  dr_slack = lp_flexv1(Sim, opt_index, simtime, DR_call_start, DRstart, DRend, rho, Sim.compr_capacity)
        
        # Solve the thing
        #print(prob)
        prob.solve()
        print('%%%%%%%% -- Optimisation outcome -- %%%%%%')
        print(LpStatus[prob.status])
        
        prob_out = {}
        for time in opt_index:
            prob_out[time] = {
                'Power': hp_power[time].varValue,
                'InTemp': intemp[time].varValue,
                'Slack_1': slack_1[time].varValue,
                'Slack_2': slack_2[time].varValue,
                'dr_slack': dr_slack[time].varValue
                }

        result_df = pd.DataFrame.from_dict(prob_out, orient='index')
        result_df.index = opt_index
        
            
        #result_df = result_df.shift(periods=1)
        #result_df = result_df.fillna(method='ffill')
        #avg_power = (result_df['Power'][opt_index[0]] + result_df['Power'][opt_index[1]])/2.0
        #if avg_power <= 0.2:
        #    avg_power = 0
        #result_df['Power'][opt_index[0]] = avg_power
        #result_df['Power'][opt_index[1]] = avg_power
        print(result_df)
        
        Sim.opt_controlseq = {"HPPower": 
                              variables.Timeseries('HPPower', 
                                                   result_df['Power'].shift(periods=0) / (Sim.compr_capacity / 1000.0), 
                                                   units.W, 
                                                   tz_name=Sim.weather.tz_name)
                                }
        
        controlseq[Sim.building] = Sim.opt_controlseq['HPPower'].display_data()
        
        Sim.control.data = Sim.opt_controlseq
        
        print("%%%%%%%%%%% Control Sequence %%%%%%%%%")
        print(Sim.opt_controlseq['HPPower'].display_data()[simtime:])
        
        mpctemps[Sim.building] = result_df['InTemp']
        opt_stats[Sim.building] = LpStatus[prob.status]
        
        # Emulate the response
        Sim.emulate_opt(simtime_str,emu_end_str)
        
        emutemps[Sim.building] = Sim.emu.display_measurements('Measured').values[1][-1]-273.15
        power[Sim.building] = Sim.emu.display_measurements('Measured').values[1][0]
        # Collect measurements
        print(Sim.emu.display_measurements('Measured'))
        
        
        print('Storing all the stuff')
        store_namespace(os.path.join(folder,'emutemps_'+sim_id+'_'+str(bldg_index_start)+'-'+str(bldg_index_end)+'_'+opt_start_str.replace('/','-').replace(':','-').replace(' ','-')), emutemps)
        store_namespace(os.path.join(folder,'mpctemps_'+sim_id+'_'+str(bldg_index_start)+'-'+str(bldg_index_end)+'_'+opt_start_str.replace('/','-').replace(':','-').replace(' ','-')), mpctemps)
        store_namespace(os.path.join(folder,'opt_stats_'+sim_id+'_'+str(bldg_index_start)+'-'+str(bldg_index_end)+'_'+opt_start_str.replace('/','-').replace(':','-').replace(' ','-')), opt_stats)
        #store_namespace('refheat_'+sim_id, refheat)
        store_namespace(os.path.join(folder,'controlseq_'+sim_id+'_'+str(bldg_index_start)+'-'+str(bldg_index_end)+'_'+opt_start_str.replace('/','-').replace(':','-').replace(' ','-')), controlseq)
        store_namespace(os.path.join(folder,'opt_result_'+sim_id+'_'+str(bldg_index_start)+'-'+str(bldg_index_end)+'_'+opt_start_str.replace('/','-').replace(':','-').replace(' ','-')), result_df)
        store_namespace(os.path.join(folder,'power_'+sim_id+'_'+str(bldg_index_start)+'-'+str(bldg_index_end)+'_'+opt_start_str.replace('/','-').replace(':','-').replace(' ','-')), power)
        
        
        #exit()
        #print(prob)
        #print(index.strftime('%m/%d/%Y %H:%M:%S'))

