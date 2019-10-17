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

def lp_enemin(Sim, opt_index, rho, compr_capacity):

    prob = LpProblem("Minimise Energy Consumption", LpMinimize)
        
    # Initialise the variables
    # Power
    hp_power=LpVariable.dicts("hp_power", 
                                (time for time in opt_index), 
                                lowBound = 0,
                                upBound = Sim.compr_capacity/1000,
                                cat = 'Continuous')

    # Indoor temperature
    intemp=LpVariable.dicts("intemp", 
                                (time for time in opt_index), 
                                cat = 'Continuous'
                                   )
    
    
    slack_1=LpVariable.dicts("slack_1", 
                                (time for time in opt_index), 
                                lowBound = 0,
                                cat = 'Continuous'
                                   )
    
    slack_2=LpVariable.dicts("slack_2", 
                                (time for time in opt_index), 
                                lowBound = 0,
                                cat = 'Continuous'
                                   )
    
    #print(intemp)
    
    # Decentralised cost minimisation
    
    rho = rho
    prob += lpSum([[Sim.price.display_data().loc[time] * hp_power[time]] for time in opt_index] + [rho * slack_1[time] for time in opt_index] + [rho * slack_2[time] for time in opt_index]
                    )
    
    # Define constraints        
    # Temperature constraints
    j = 0
    for time in opt_index:
        prob += intemp[time] - slack_1[time] <= Sim.constraints.display_data().loc[time]['TAir_Slack_LTE'] - 273.15
        prob += intemp[time] + slack_2[time] >= Sim.constraints.display_data().loc[time]['TAir_Slack_GTE']- 273.15
        j += 1

    # Define relationship between hp_power and in_temp
    params = Sim.ARX_model.fit_results.params
    #print(Sim.start_temp+273.15)

    # Weather
    out_temp = Sim.weather.display_data()['weaTDryBul'].resample(str(Sim.meas_sampl)+'S').ffill()
    radiation = Sim.weather.display_data()['weaHGloHor'].resample(str(Sim.meas_sampl)+'S').ffill()

    j = 0
    for time in opt_index:
        if j == 0:
            prob += intemp[time] ==  Sim.start_temps[j]
            prob += hp_power[time] == Sim.start_powers[j]
            prev_time1 = time
        elif j == 1:
            prob += intemp[time] == (params['Intercept'] 
                                    + params['weaTDryBul_delay1'] * out_temp[prev_time1] 
                                    + params['weaHGloHor_delay1'] * radiation[prev_time1]
                                    + params['PowerCompr'] * hp_power[time]
                                    + params['PowerCompr_delay1'] * hp_power[prev_time1]
                                    + params['T_in_delay1'] * intemp[prev_time1]
                                    + params['T_in_delay2'] * Sim.start_temps[1]
                                    + params['T_in_delay3'] * Sim.start_temps[2]
                                    + params['T_in_delay4'] * Sim.start_temps[3]
                                    + params['T_in_delay13'] * Sim.start_temps[-2]
                                    )
            prev_time2 = prev_time1
            prev_time1 = time
        elif j == 2:
            prob += intemp[time] == (params['Intercept'] 
                                    + params['weaTDryBul_delay1'] * out_temp[prev_time1] 
                                    + params['weaHGloHor_delay1'] * radiation[prev_time1]
                                    + params['PowerCompr'] * hp_power[time]
                                    + params['PowerCompr_delay1'] * hp_power[prev_time1]
                                    + params['T_in_delay1'] * intemp[prev_time1]
                                    + params['T_in_delay2'] * intemp[prev_time2]
                                    + params['T_in_delay3'] * Sim.start_temps[1]
                                    + params['T_in_delay4'] * Sim.start_temps[2]
                                    + params['T_in_delay13'] * Sim.start_temps[-3]
                                    )
            prob += hp_power[time] == hp_power[prev_time1]
            prev_time3 = prev_time2
            prev_time2 = prev_time1
            prev_time1 = time
        elif j == 3:
            prob += intemp[time] == (params['Intercept'] 
                                    + params['weaTDryBul_delay1'] * out_temp[prev_time1] 
                                    + params['weaHGloHor_delay1'] * radiation[prev_time1]
                                    + params['PowerCompr'] * hp_power[time]
                                    + params['PowerCompr_delay1'] * hp_power[prev_time1]
                                    + params['T_in_delay1'] * intemp[prev_time1]
                                    + params['T_in_delay2'] * intemp[prev_time2]
                                    + params['T_in_delay3'] * intemp[prev_time3]
                                    + params['T_in_delay4'] * Sim.start_temps[1]
                                    + params['T_in_delay13'] * Sim.start_temps[-4]
                                    )
            prob += hp_power[time] == hp_power[prev_time1]
            prev_time4 = prev_time3
            prev_time3 = prev_time2
            prev_time2 = prev_time1
            prev_time1 = time
        elif j < 13:
            prob += intemp[time] == (params['Intercept'] 
                                    + params['weaTDryBul_delay1'] * out_temp[prev_time1] 
                                    + params['weaHGloHor_delay1'] * radiation[prev_time1]
                                    + params['PowerCompr'] * hp_power[time]
                                    + params['PowerCompr_delay1'] * hp_power[prev_time1]
                                    + params['T_in_delay1'] * intemp[prev_time1]
                                    + params['T_in_delay2'] * intemp[prev_time2]
                                    + params['T_in_delay3'] * intemp[prev_time3]
                                    + params['T_in_delay4'] * intemp[prev_time4]
                                    + params['T_in_delay13'] * Sim.start_temps[-j-1]
                                    )
            
            if j > 3 and ((j-1) % 3) != 0:
                prob += hp_power[time] == hp_power[prev_time1]
                
            #prob += hp_power[time] == hp_power[prev_time1]
            prev_time4 = prev_time3
            prev_time3 = prev_time2
            prev_time2 = prev_time1
            prev_time1 = time
        else:
            prev_time12 = opt_index[j-13]
            prob += intemp[time] == (params['Intercept'] 
                                    + params['weaTDryBul_delay1'] * out_temp[prev_time1] 
                                    + params['weaHGloHor_delay1'] * radiation[prev_time1]
                                    + params['PowerCompr'] * hp_power[time]
                                    + params['PowerCompr_delay1'] * hp_power[prev_time1]
                                    + params['T_in_delay1'] * intemp[prev_time1]
                                    + params['T_in_delay2'] * intemp[prev_time2]
                                    + params['T_in_delay3'] * intemp[prev_time3]
                                    + params['T_in_delay4'] * intemp[prev_time4]
                                    + params['T_in_delay13'] * intemp[prev_time12]
                                    )
                                    
            if j > 2 and ((j-1) % 2) != 0:
                prob += hp_power[time] == hp_power[prev_time1]
            if j > 33:
                prob += slack_1[time] == 0
                prob += slack_2[time] == 0
                prob += hp_power[time] == hp_power[prev_time1]
                
            prev_time4 = prev_time3
            prev_time3 = prev_time2
            prev_time2 = prev_time1
            prev_time1 = time
        j = j+1
    
    #print(prob)
    #exit()
    
    return prob, hp_power, intemp, slack_1, slack_2

def lp_flex(Sim, opt_index, simtime, DR_call_start, DRstart, DRend, incentive, rho, compr_capacity):
    prob = LpProblem("Minimise Cost", LpMinimize)
    
    # Initialise the variables
    # Power
    hp_power=LpVariable.dicts("hp_power", 
                                (time for time in opt_index), 
                                lowBound = 0,
                                upBound = compr_capacity/1000.0,
                                cat = 'Continuous')

    # Indoor temperature
    intemp=LpVariable.dicts("intemp", 
                                (time for time in opt_index), 
                                cat = 'Continuous'
                                   )
    
    
    slack_1=LpVariable.dicts("slack_1", 
                                (time for time in opt_index), 
                                lowBound = 0,
                                cat = 'Continuous'
                                   )
    
    slack_2=LpVariable.dicts("slack_2", 
                                (time for time in opt_index), 
                                lowBound = 0,
                                cat = 'Continuous'
                                   )
    
    # Binary variable to define answer to flexibility call
    flex_bin=LpVariable.dicts("flex_bin", 
                                (time for time in opt_index), 
                                cat = 'Binary'
                                )
    
    rho = rho
    ref_energy = Sim.ref_profile.display_data() * (compr_capacity / 1000.0)
    
    print('Reference profile')
    print(ref_energy) 
    #print(ref_energy)
    #ref_energy = ref_energy[opt_index]
    #ref_energy = ref_energy.fillna(method='ffill')
    
    
    #print(Sim.flex_cost.display_data().loc[opt_index[0]].values[0])
    #print(ref_energy.loc[opt_index[0]].values[0])
    #print(Sim.flex_cost.display_data())
    
    # Decentralised contracted flexibility function
    prob += lpSum([(Sim.price.display_data().loc[time] + Sim.flex_cost.display_data().loc[time].values[0]) * hp_power[time] for time in opt_index]  +
                      [- Sim.flex_cost.display_data().loc[time].values[0] * ref_energy.loc[time].values[0] * flex_bin[time]  for time in opt_index] + 
                       [- incentive * flex_bin[time] for time in opt_index] +
                       [rho * slack_1[time] for time in opt_index] + 
                       [rho * slack_2[time] for time in opt_index]
                     ) 
    
    
    # Define constraints        
    # Temperature constraints
    j = 0
    for time in opt_index:
        prob += intemp[time] - slack_1[time] <= Sim.constraints.display_data().loc[time]['TAir_Slack_LTE']-273.15 
        prob += intemp[time] + slack_2[time] >= Sim.constraints.display_data().loc[time]['TAir_Slack_GTE']-273.15
        j += 1

    # Flexibility binary variable updated at DR call
    
    if simtime < DR_call_start: # No DR in objective function
        for time in opt_index:
            prob += flex_bin[time] == 0
    
    if simtime == DR_call_start: # Freedom to choose but has to remain
        #print('Time for DR')
        j = 0
        
        for time in opt_index:
            print(time)
            if time >= DRstart and time < DRend:
                #print('Time for DR')
                if time == DRstart:
                    print('Pass')
                    pass
                else:
                    print('Add flex constr')
                    prob += flex_bin[time] == flex_bin[prev_time]
            else:
                prob += flex_bin[time] == 0
            prev_time = time
    if simtime > DR_call_start: # After the call use the binaries defined during DR call
        for time in opt_index:
            if time in pd.date_range(DRstart, DRend - datetime.timedelta(seconds = horizon*int(meas_sampl)), freq = meas_sampl+'S'):
                prob += flex_bin[time] == Sim.flex_bin[time]
            else:
                prob += flex_bin[time] == 0
    
    # Define relationship between hp_power and in_temp
    params = Sim.ARX_model.fit_results.params
    #print(Sim.start_temp+273.15)

    # Weather
    out_temp = Sim.weather.display_data()['weaTDryBul'].resample(str(Sim.meas_sampl)+'S').ffill()
    radiation = Sim.weather.display_data()['weaHGloHor'].resample(str(Sim.meas_sampl)+'S').ffill()

    j = 0
    for time in opt_index:
        if j == 0:
            prob += intemp[time] ==  Sim.start_temp
            prob += hp_power[time] == Sim.start_power
            prev_time1 = time
        elif j == 1:
            prob += intemp[time] == (params['Intercept'] 
                                    + params['weaTDryBul'] * out_temp[time] 
                                    + params['weaHGloHor'] * radiation[time]
                                    + params['PowerCompr'] * hp_power[time]
                                    + params['PowerCompr_delay1'] * hp_power[prev_time1]
                                    + params['T_in_delay1'] * intemp[prev_time1]
                                    + params['T_in_delay2'] * Sim.start_temp
                                    #+ params['T_in_delay3'] * Sim.start_temp1
                                    )
            prev_time2 = prev_time1
            prev_time1 = time
        elif j == 2:
            prob += intemp[time] == (params['Intercept'] 
                                    + params['weaTDryBul'] * out_temp[time] 
                                    + params['weaHGloHor'] * radiation[time]
                                    + params['PowerCompr'] * hp_power[time]  
                                    + params['PowerCompr_delay1'] * hp_power[prev_time1]
                                    + params['T_in_delay1'] * intemp[prev_time1]
                                    + params['T_in_delay2'] * intemp[prev_time2]
                                    #+ params['T_in_delay3'] * Sim.start_temp
                                    )
            prev_time3 = prev_time2
            prev_time2 = prev_time1
            prev_time1 = time
        else:
            prob += intemp[time] == (params['Intercept'] 
                                    + params['weaTDryBul'] * out_temp.loc[time] 
                                    + params['weaHGloHor'] * radiation.loc[time]
                                    + params['PowerCompr'] * hp_power[time]                            
                                    + params['PowerCompr_delay1'] * hp_power[prev_time1]
                                    + params['T_in_delay1'] * intemp[prev_time1]
                                    + params['T_in_delay2'] * intemp[prev_time2]
                                    #+ params['T_in_delay3'] * intemp[prev_time3]
                                    )
            prev_time3 = prev_time2
            prev_time2 = prev_time1
            prev_time1 = time
        j = j+1
    
    return prob, hp_power, intemp, slack_1, slack_2,  flex_bin
    
def lp_flexv1(Sim, opt_index, simtime, DR_call_start, DRstart, DRend, rho, compr_capacity):
    prob = LpProblem("Minimise Cost", LpMinimize)

    # Initialise the variables
    # Power
    hp_power=LpVariable.dicts("hp_power", 
                                (time for time in opt_index), 
                                lowBound = 0,
                                upBound = compr_capacity/1000.0,
                                cat = 'Continuous')

    # Indoor temperature
    intemp=LpVariable.dicts("intemp", 
                                (time for time in opt_index), 
                                cat = 'Continuous'
                                   )


    slack_1=LpVariable.dicts("slack_1", 
                                (time for time in opt_index), 
                                lowBound = 0,
                                cat = 'Continuous'
                                   )

    slack_2=LpVariable.dicts("slack_2", 
                                (time for time in opt_index), 
                                lowBound = 0,
                                cat = 'Continuous'
                                   )

    dr_slack=LpVariable.dicts("dr_slack", 
                                (time for time in opt_index), 
                                lowBound = 0,
                                cat = 'Continuous'
                                   )

    rho = rho
    ref_energy = Sim.ref_profile.display_data() * (compr_capacity / 1000.0)
    
    #print('Reference profile')
    #print(ref_energy) 
    #print('Reference profile')
    #print(ref_energy)
    #ref_energy = ref_energy[opt_index]
    #ref_energy = ref_energy.fillna(method='ffill')


    #print(Sim.flex_cost.display_data().loc[opt_index[0]].values[0])
    #print(ref_energy.loc[opt_index[0]].values[0])
    #print(Sim.flex_cost.display_data())

    # Decentralised contracted flexibility function
    prob += lpSum([Sim.price.display_data().loc[time] * hp_power[time] for time in opt_index]  +
                      [Sim.flex_cost.display_data().loc[time].values[0] * dr_slack[time]  for time in opt_index] +
                       [rho * slack_1[time] for time in opt_index] + 
                       [rho * slack_2[time] for time in opt_index]
                     ) 


    # Define constraints        
    # Temperature constraints
    j = 0
    for time in opt_index:
        prob += intemp[time] - slack_1[time] <= Sim.constraints.display_data().loc[time]['TAir_Slack_LTE']-273.15 
        prob += intemp[time] + slack_2[time] >= Sim.constraints.display_data().loc[time]['TAir_Slack_GTE']-273.15
        j += 1
    
    for time in opt_index:
        if time >= DRstart and time < DRend:
            print('Time for DR')
            print(Sim.flex_cost.display_data().loc[time].values[0])
            # Add flexibility tracking incentivising operation at or below reference and discouraging operation above it
            prob += hp_power[time] - dr_slack[time] <= ref_energy.loc[time].values[0]


    # Define relationship between hp_power and in_temp
    params = Sim.ARX_model.fit_results.params
    #print(Sim.start_temp+273.15)

    # Weather
    out_temp = Sim.weather.display_data()['weaTDryBul'].resample(str(Sim.meas_sampl)+'S').ffill()
    radiation = Sim.weather.display_data()['weaHGloHor'].resample(str(Sim.meas_sampl)+'S').ffill()
    
    j = 0
    for time in opt_index:
        if j == 0:
            prob += intemp[time] ==  Sim.start_temps[j]
            prob += hp_power[time] == Sim.start_powers[j]
            prev_time1 = time
        elif j == 1:
            prob += intemp[time] == (params['Intercept'] 
                                    + params['weaTDryBul_delay1'] * out_temp[prev_time1] 
                                    + params['weaHGloHor_delay1'] * radiation[prev_time1]
                                    + params['PowerCompr'] * hp_power[time]
                                    + params['PowerCompr_delay1'] * hp_power[prev_time1]
                                    + params['T_in_delay1'] * intemp[prev_time1]
                                    + params['T_in_delay2'] * Sim.start_temps[1]
                                    + params['T_in_delay3'] * Sim.start_temps[2]
                                    + params['T_in_delay4'] * Sim.start_temps[3]
                                    + params['T_in_delay13'] * Sim.start_temps[-2]
                                    )
            prev_time2 = prev_time1
            prev_time1 = time
        elif j == 2:
            prob += intemp[time] == (params['Intercept'] 
                                    + params['weaTDryBul_delay1'] * out_temp[prev_time1] 
                                    + params['weaHGloHor_delay1'] * radiation[prev_time1]
                                    + params['PowerCompr'] * hp_power[time]
                                    + params['PowerCompr_delay1'] * hp_power[prev_time1]
                                    + params['T_in_delay1'] * intemp[prev_time1]
                                    + params['T_in_delay2'] * intemp[prev_time2]
                                    + params['T_in_delay3'] * Sim.start_temps[1]
                                    + params['T_in_delay4'] * Sim.start_temps[2]
                                    + params['T_in_delay13'] * Sim.start_temps[-3]
                                    )
            prob += hp_power[time] == hp_power[prev_time1]
            prev_time3 = prev_time2
            prev_time2 = prev_time1
            prev_time1 = time
        elif j == 3:
            prob += intemp[time] == (params['Intercept'] 
                                    + params['weaTDryBul_delay1'] * out_temp[prev_time1] 
                                    + params['weaHGloHor_delay1'] * radiation[prev_time1]
                                    + params['PowerCompr'] * hp_power[time]
                                    + params['PowerCompr_delay1'] * hp_power[prev_time1]
                                    + params['T_in_delay1'] * intemp[prev_time1]
                                    + params['T_in_delay2'] * intemp[prev_time2]
                                    + params['T_in_delay3'] * intemp[prev_time3]
                                    + params['T_in_delay4'] * Sim.start_temps[1]
                                    + params['T_in_delay13'] * Sim.start_temps[-4]
                                    )
            prob += hp_power[time] == hp_power[prev_time1]
            prev_time4 = prev_time3
            prev_time3 = prev_time2
            prev_time2 = prev_time1
            prev_time1 = time
        elif j < 13:
            prob += intemp[time] == (params['Intercept'] 
                                    + params['weaTDryBul_delay1'] * out_temp[prev_time1] 
                                    + params['weaHGloHor_delay1'] * radiation[prev_time1]
                                    + params['PowerCompr'] * hp_power[time]
                                    + params['PowerCompr_delay1'] * hp_power[prev_time1]
                                    + params['T_in_delay1'] * intemp[prev_time1]
                                    + params['T_in_delay2'] * intemp[prev_time2]
                                    + params['T_in_delay3'] * intemp[prev_time3]
                                    + params['T_in_delay4'] * intemp[prev_time4]
                                    + params['T_in_delay13'] * Sim.start_temps[-j-1]
                                    )
                                    
            if j > 3 and ((j-1) % 3) != 0:
                prob += hp_power[time] == hp_power[prev_time1]
                
            #prob += hp_power[time] == hp_power[prev_time1]
            prev_time4 = prev_time3
            prev_time3 = prev_time2
            prev_time2 = prev_time1
            prev_time1 = time
        else:
            prev_time12 = opt_index[j-13]
            prob += intemp[time] == (params['Intercept'] 
                                    + params['weaTDryBul_delay1'] * out_temp[prev_time1] 
                                    + params['weaHGloHor_delay1'] * radiation[prev_time1]
                                    + params['PowerCompr'] * hp_power[time]
                                    + params['PowerCompr_delay1'] * hp_power[prev_time1]
                                    + params['T_in_delay1'] * intemp[prev_time1]
                                    + params['T_in_delay2'] * intemp[prev_time2]
                                    + params['T_in_delay3'] * intemp[prev_time3]
                                    + params['T_in_delay4'] * intemp[prev_time4]
                                    + params['T_in_delay13'] * intemp[prev_time12]
                                    )
                                    
            if j > 3 and ((j-1) % 3) != 0:
                prob += hp_power[time] == hp_power[prev_time1]
            if j > 33:
                prob += slack_1[time] == 0
                prob += slack_2[time] == 0
                prob += hp_power[time] == hp_power[prev_time1]
                
            prev_time4 = prev_time3
            prev_time3 = prev_time2
            prev_time2 = prev_time1
            prev_time1 = time
            
        j = j+1
        
    return prob, hp_power, intemp, slack_1, slack_2,  dr_slack
    

def lp_flexv1_centr(Sim_list, opt_index, simtime, DR_call_start, DRstart, DRend, rho, load_profile_aggr):
    prob = LpProblem("Minimise Cost", LpMinimize)
    
    bldg_list = []
    for Sim in Sim_list:
        bldg_list.append(Sim.building)
    
    time_list = []
    for time in opt_index:
        time_list.append(time.strftime('%m/%d/%Y %H:%M:%S'))
    # Initialise the variables
    # Power
    hp_power=LpVariable.dicts("hp_power", 
                                (opt_index, bldg_list),
                                cat = 'Continuous')
    #print(hp_power)

    # Indoor temperature
    intemp=LpVariable.dicts("intemp", 
                                (opt_index, bldg_list), 
                                cat = 'Continuous'
                                   )


    slack_1=LpVariable.dicts("slack_1", 
                                (opt_index, bldg_list),  
                                lowBound = 0,
                                cat = 'Continuous'
                                   )

    slack_2=LpVariable.dicts("slack_2", 
                                (opt_index, bldg_list), 
                                lowBound = 0,
                                cat = 'Continuous'
                                   )

    dr_slack=LpVariable.dicts("dr_slack", 
                                (opt_index, bldg_list), 
                                lowBound = 0,
                                cat = 'Continuous'
                                   )

    rho = rho
    
    
    
    #print('Reference profile')
    #print(ref_energy) 
    #print('Reference profile')
    #print(ref_energy)
    #ref_energy = ref_energy[opt_index]
    #ref_energy = ref_energy.fillna(method='ffill')


    #print(Sim.flex_cost.display_data().loc[opt_index[0]].values[0])
    #print(ref_energy.loc[opt_index[0]].values[0])
    #print(Sim.flex_cost.display_data())

    # Decentralised contracted flexibility function
    prob += lpSum([Sim_list[0].price.display_data().loc[time] * hp_power[time][bldg] for time in opt_index for bldg in bldg_list]  +
                      [Sim_list[0].flex_cost.display_data().loc[time].values[0] * dr_slack[time][bldg]  for time in opt_index for bldg in bldg_list] +
                       [rho * slack_1[time][bldg] for time in opt_index for bldg in bldg_list] + 
                       [rho * slack_2[time][bldg] for time in opt_index for bldg in bldg_list]
                     ) 


    # Define constraints        
    # Temperature constraints
    j = 0
    for time in opt_index:
        for Sim in Sim_list:
            bldg = Sim.building
            prob += intemp[time][bldg] - slack_1[time][bldg] <= Sim.constraints.display_data().loc[time]['TAir_Slack_LTE']-273.15 
            prob += intemp[time][bldg] + slack_2[time][bldg] >= Sim.constraints.display_data().loc[time]['TAir_Slack_GTE']-273.15
            
            prob += hp_power[time][bldg] <= (Sim.compr_capacity / 1000.0)
            prob += hp_power[time][bldg] >= 0
            
        j += 1
    
    load_profile_aggr = load_profile_aggr/1000.0
    #print(load_profile_aggr)
    
    for time in opt_index:
        for Sim in Sim_list:
            bldg = Sim.building
            
            if time >= DRstart and time < DRend:
                print('Time for DR')
                print(Sim.flex_cost.display_data().loc[time].values[0])
                # Add flexibility tracking incentivising operation at or below reference and discouraging operation above it
                prob += lpSum([hp_power[time][bldg] for bldg in bldg_list]) - dr_slack[time] <= load_profile_aggr.loc[time]



    # Weather
    out_temp = Sim.weather.display_data()['weaTDryBul'].resample(str(Sim.meas_sampl)+'S').ffill()
    radiation = Sim.weather.display_data()['weaHGloHor'].resample(str(Sim.meas_sampl)+'S').ffill()

    
    for Sim in Sim_list:
        j = 0
        bldg = Sim.building
        
        params = Sim.ARX_model.fit_results.params
        
        for time in opt_index:
            if j == 0:
                prob += intemp[time][bldg] == Sim.start_temps[j]
                prob += hp_power[time][bldg] == Sim.start_powers[j]
                prev_time1 = time
            elif j == 1:
                prob += intemp[time][bldg] == (params['Intercept'] 
                                        + params['weaTDryBul_delay1'] * out_temp[prev_time1] 
                                        + params['weaHGloHor_delay1'] * radiation[prev_time1]
                                        + params['PowerCompr'] * hp_power[time][bldg]
                                        + params['PowerCompr_delay1'] * hp_power[prev_time1][bldg]
                                        + params['T_in_delay1'] * intemp[prev_time1][bldg]
                                        + params['T_in_delay2'] * Sim.start_temps[1]
                                        + params['T_in_delay3'] * Sim.start_temps[2]
                                        + params['T_in_delay4'] * Sim.start_temps[3]
                                        + params['T_in_delay13'] * Sim.start_temps[-2]
                                        )
                prev_time2 = prev_time1
                prev_time1 = time
            elif j == 2:
                prob += intemp[time][bldg] == (params['Intercept'] 
                                        + params['weaTDryBul_delay1'] * out_temp[prev_time1]
                                        + params['weaHGloHor_delay1'] * radiation[prev_time1]
                                        + params['PowerCompr'] * hp_power[time][bldg]
                                        + params['PowerCompr_delay1'] * hp_power[prev_time1][bldg]
                                        + params['T_in_delay1'] * intemp[prev_time1][bldg]
                                        + params['T_in_delay2'] * intemp[prev_time2][bldg]
                                        + params['T_in_delay3'] * Sim.start_temps[1]
                                        + params['T_in_delay4'] * Sim.start_temps[2]
                                        + params['T_in_delay13'] * Sim.start_temps[-3]
                                        )
                prob += hp_power[time][bldg] == hp_power[prev_time1][bldg]
                prev_time3 = prev_time2
                prev_time2 = prev_time1
                prev_time1 = time
            elif j == 3:
                prob += intemp[time][bldg] == (params['Intercept'] 
                                        + params['weaTDryBul_delay1'] * out_temp[prev_time1] 
                                        + params['weaHGloHor_delay1'] * radiation[prev_time1]
                                        + params['PowerCompr'] * hp_power[time][bldg]
                                        + params['PowerCompr_delay1'] * hp_power[prev_time1][bldg]
                                        + params['T_in_delay1'] * intemp[prev_time1][bldg]
                                        + params['T_in_delay2'] * intemp[prev_time2][bldg]
                                        + params['T_in_delay3'] * intemp[prev_time3][bldg]
                                        + params['T_in_delay4'] * Sim.start_temps[1]
                                        + params['T_in_delay13'] * Sim.start_temps[-4]
                                        )
                prob += hp_power[time][bldg] == hp_power[prev_time1][bldg]
                prev_time4 = prev_time3
                prev_time3 = prev_time2
                prev_time2 = prev_time1
                prev_time1 = time
            elif j < 13:
                prob += intemp[time][bldg] == (params['Intercept'] 
                                        + params['weaTDryBul_delay1'] * out_temp[prev_time1]
                                        + params['weaHGloHor_delay1'] * radiation[prev_time1]
                                        + params['PowerCompr'] * hp_power[time][bldg]
                                        + params['PowerCompr_delay1'] * hp_power[prev_time1][bldg]
                                        + params['T_in_delay1'] * intemp[prev_time1][bldg]
                                        + params['T_in_delay2'] * intemp[prev_time2][bldg]
                                        + params['T_in_delay3'] * intemp[prev_time3][bldg]
                                        + params['T_in_delay4'] * intemp[prev_time4][bldg]
                                        + params['T_in_delay13'] * Sim.start_temps[-j-1]
                                        )
                                        
                if j > 3 and ((j-1) % 3) != 0:
                    prob += hp_power[time][bldg] == hp_power[prev_time1][bldg]
                    
                #prob += hp_power[time] == hp_power[prev_time1]
                prev_time4 = prev_time3
                prev_time3 = prev_time2
                prev_time2 = prev_time1
                prev_time1 = time
            else:
                prev_time12 = opt_index[j-13]
                prob += intemp[time][bldg] == (params['Intercept'] 
                                        + params['weaTDryBul_delay1'] * out_temp[prev_time1] 
                                        + params['weaHGloHor_delay1'] * radiation[prev_time1]
                                        + params['PowerCompr'] * hp_power[time][bldg]
                                        + params['PowerCompr_delay1'] * hp_power[prev_time1][bldg]
                                        + params['T_in_delay1'] * intemp[prev_time1][bldg]
                                        + params['T_in_delay2'] * intemp[prev_time2][bldg]
                                        + params['T_in_delay3'] * intemp[prev_time3][bldg]
                                        + params['T_in_delay4'] * intemp[prev_time4][bldg]
                                        + params['T_in_delay13'] * intemp[prev_time12][bldg]
                                        )
                                        
                if j > 3 and ((j-1) % 3) != 0:
                    prob += hp_power[time][bldg] == hp_power[prev_time1][bldg]
                if j > 33:
                    prob += slack_1[time][bldg] == 0
                    prob += slack_2[time][bldg] == 0
                    prob += hp_power[time][bldg] == hp_power[prev_time1][bldg]
                    
                prev_time4 = prev_time3
                prev_time3 = prev_time2
                prev_time2 = prev_time1
                prev_time1 = time
            j += 1
        
    #print(prob)
        
    return prob, hp_power, intemp, slack_1, slack_2, dr_slack
    
def lp_flexv2(Sim, opt_index, simtime, DR_call_start, DRstart, DRend, rho, compr_capacity):
    prob = LpProblem("Minimise Cost", LpMinimize)

    # Initialise the variables
    # Power
    hp_power=LpVariable.dicts("hp_power", 
                                (time for time in opt_index), 
                                lowBound = 0,
                                upBound = compr_capacity/1000.0,
                                cat = 'Continuous')

    # Indoor temperature
    intemp=LpVariable.dicts("intemp", 
                                (time for time in opt_index), 
                                cat = 'Continuous'
                                   )


    slack_1=LpVariable.dicts("slack_1", 
                                (time for time in opt_index), 
                                lowBound = 0,
                                cat = 'Continuous'
                                   )

    slack_2=LpVariable.dicts("slack_2", 
                                (time for time in opt_index), 
                                lowBound = 0,
                                cat = 'Continuous'
                                   )

    dr_slack=LpVariable.dicts("dr_slack", 
                                (time for time in opt_index), 
                                lowBound = 0,
                                cat = 'Continuous'
                                   )

    rho = rho
    ref_energy = Sim.ref_profile.display_data() * (compr_capacity / 1000.0)
    #print('Reference profile')
    #print(ref_energy)
    #ref_energy = ref_energy[opt_index]
    #ref_energy = ref_energy.fillna(method='ffill')


    #print(Sim.flex_cost.display_data().loc[opt_index[0]].values[0])
    #print(ref_energy.loc[opt_index[0]].values[0])
    #print(Sim.flex_cost.display_data())

    # Decentralised contracted flexibility function
    prob += lpSum([Sim.price.display_data().loc[time] * hp_power[time] for time in opt_index]  +
                      [Sim.flex_cost.display_data().loc[time].values[0] * dr_slack[time]  for time in opt_index] +
                       [- flex_bin[time] * incentive for time in opt_index] +
                       [rho * slack_1[time] for time in opt_index] + 
                       [rho * slack_2[time] for time in opt_index]
                     ) 


    # Define constraints        
    # Temperature constraints
    j = 0
    for time in opt_index:
        prob += intemp[time] - slack_1[time] <= Sim.constraints.display_data().loc[time]['TAir_Slack_LTE']-273.15 
        prob += intemp[time] + slack_2[time] >= Sim.constraints.display_data().loc[time]['TAir_Slack_GTE']-273.15
        j += 1
        

    if simtime == DR_call_start: # Freedom to choose but has to remain
        print('Time for DR')
        j = 0
        
        for time in opt_index:
            print(time)
            if time >= DRstart and time < DRend:
                print('Time for DR')
                
                # Add flexibility tracking incentivising operation below and discouraging operation above reference request
                prob += hp_power[time] - dr_slack[time] <= ref_energy.loc[time].values[0]
                
            prev_time = time


    # Define relationship between hp_power and in_temp
    params = Sim.ARX_model.fit_results.params
    #print(Sim.start_temp+273.15)

    # Weather
    out_temp = Sim.weather.display_data()['weaTDryBul'].resample(str(Sim.meas_sampl)+'S').ffill()
    radiation = Sim.weather.display_data()['weaHGloHor'].resample(str(Sim.meas_sampl)+'S').ffill()

    j = 0
    for time in opt_index:
        if j == 0:
            prob += intemp[time] ==  Sim.start_temp
            prob += hp_power[time] == Sim.start_power
            prev_time1 = time
        elif j == 1:
            prob += intemp[time] == (params['Intercept'] 
                                    + params['weaTDryBul'] * out_temp[time] 
                                    + params['weaHGloHor'] * radiation[time]
                                    + params['PowerCompr'] * hp_power[time]
                                    + params['PowerCompr_delay1'] * hp_power[prev_time1]
                                    + params['T_in_delay1'] * intemp[prev_time1]
                                    + params['T_in_delay2'] * Sim.start_temp
                                    #+ params['T_in_delay3'] * Sim.start_temp1
                                    )
            prev_time2 = prev_time1
            prev_time1 = time
        elif j == 2:
            prob += intemp[time] == (params['Intercept'] 
                                    + params['weaTDryBul'] * out_temp[time] 
                                    + params['weaHGloHor'] * radiation[time]
                                    + params['PowerCompr'] * hp_power[time]  
                                    + params['PowerCompr_delay1'] * hp_power[prev_time1]
                                    + params['T_in_delay1'] * intemp[prev_time1]
                                    + params['T_in_delay2'] * intemp[prev_time2]
                                    #+ params['T_in_delay3'] * Sim.start_temp
                                    )
            prev_time3 = prev_time2
            prev_time2 = prev_time1
            prev_time1 = time
        else:
            prob += intemp[time] == (params['Intercept'] 
                                    + params['weaTDryBul'] * out_temp.loc[time] 
                                    + params['weaHGloHor'] * radiation.loc[time]
                                    + params['PowerCompr'] * hp_power[time]                            
                                    + params['PowerCompr_delay1'] * hp_power[prev_time1]
                                    + params['T_in_delay1'] * intemp[prev_time1]
                                    + params['T_in_delay2'] * intemp[prev_time2]
                                    #+ params['T_in_delay3'] * intemp[prev_time3]
                                    )
            prev_time3 = prev_time2
            prev_time2 = prev_time1
            prev_time1 = time
        j = j+1
        
    return prob, hp_power, intemp, slack_1, slack_2,  dr_slack
            
            
        
    