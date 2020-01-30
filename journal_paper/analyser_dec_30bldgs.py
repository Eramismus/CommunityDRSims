# The analyser

import pandas as pd
import matplotlib.pyplot as plt
import dill
import os
import numpy as np
from funcs import store_namespace
from funcs import load_namespace
import datetime

from matplotlib.font_manager import FontProperties

from matplotlib import rc

community = 'ResidentialCommunity'
sim_ids = ['MinEne_0-30']
model_id = 'R2CW_HP'
bldg_list = load_namespace(os.path.join('file_path_to_folder', 'teaser_bldgs_residential'))
compr_capacity_list = [float(4500.0)]*10+[float(3000.0)]*20
print(bldg_list)
folder = 'results'
step = 300
nodynprice=0
mon = 'nov'
constr_folder = 'decentr_enemin_constr_'+mon
if mon == 'jan':
    start = '1/7/2017 16:30:00'
    end = '1/7/2017 19:00:00'
    controlseq_time = '01/07/2017 16:55:00'
elif mon == 'mar':
    start = '3/1/2017 16:30:00'
    end = '3/1/2017 19:00:00'
    controlseq_time = '03/01/2017 16:55:00'
elif mon=='nov':
    start = '11/20/2017 16:30:00'
    end = '11/20/2017 19:00:00'
    controlseq_time = '11/20/2017 16:55:00'

sim_range = pd.date_range(start, end, freq = str(step)+'S')

simu_path = "file_path_to_folder"

other_input = {}
price = {}
flex_cost = {}
ref_profile = {}
controlseq = {}
opt_control = {}
emutemps = {}
mpctemps = {}
opt_stats = {}
flex_down = {}
flex_up = {}
power = {}

i = 0            
for bldg in bldg_list:

    building = bldg+'_'+model_id
    
    flex_cost[building] = load_namespace(os.path.join(simu_path, folder, 'flex_cost_'+bldg+'_'+model_id)).data['flex_cost'].get_base_data()
    flex_cost[building] = flex_cost[building].resample(str(step)+'S').mean().shift(0, freq=str(step) + 'S')
    
    ref_profile[building] = load_namespace(os.path.join("'file_path_to_folder'", folder,'ref_profile_'+bldg+'_'+model_id)).data['ref_profile'].get_base_data()*compr_capacity_list[i]
    i += 1



for sim_id in sim_ids:
    opt_stats[sim_id] = {}
    controlseq[sim_id] = {}
    mpctemps[sim_id] = {}
    emutemps[sim_id] = {}
    power[sim_id] = {}
    
    for time_idx in sim_range:
        time_idx = time_idx.strftime('%m/%d/%Y %H:%M:%S')
        t = time_idx.replace('/','-').replace(':','-').replace(' ','-')
        
        opt_stats[sim_id][time_idx] = load_namespace(os.path.join(simu_path, folder, 'opt_stats_'+sim_id+'_'+t))

        emutemps[sim_id][time_idx] = load_namespace(os.path.join(simu_path, folder, 'emutemps_'+sim_id+'_'+t))

        mpctemps[sim_id][time_idx] = load_namespace(os.path.join(simu_path, folder, 'mpctemps_'+sim_id+'_'+t))

        controlseq[sim_id][time_idx] = load_namespace(os.path.join(simu_path, folder, 'controlseq_'+sim_id)+'_'+t)
        power[sim_id][time_idx] = load_namespace(os.path.join(simu_path, folder, 'power_'+sim_id)+'_'+t)

i=0
for sim_id in sim_ids:
    if i == 0:
        emutemps_df = pd.DataFrame.from_dict(emutemps[sim_id],orient='index')
        emutemps_df.index = pd.to_datetime(emutemps_df.index)
        emutemps_df.index = emutemps_df.index.shift(1, freq=str(step)+'S')
        
        power_df = pd.DataFrame.from_dict(power[sim_id],orient='index')
        power_df.index = pd.to_datetime(power_df.index).shift(1, freq=str(step)+'S')
        
        opt_stats_df = pd.DataFrame.from_dict(opt_stats[sim_id],orient='index')
        opt_stats_df.index = pd.to_datetime(opt_stats_df.index)
        #power_df.index = power_df.index.shift(1, freq=str(step)+'S')
    else:
        emutemps_df1 = pd.DataFrame.from_dict(emutemps[sim_id],orient='index')
        emutemps_df1.index = pd.to_datetime(emutemps_df1.index)
        emutemps_df1.index = emutemps_df1.index.shift(1, freq=str(step) + 'S')
        emutemps_df = pd.concat([emutemps_df, emutemps_df1])
        
        power_df1 = pd.DataFrame.from_dict(power[sim_id],orient='index')
        power_df1.index = pd.to_datetime(power_df1.index)
        power_df1.index = power_df1.index.shift(1, freq=str(step)+'S')
        power_df = pd.concat([power_df, power_df1])
        
        opt_stats_df1 = pd.DataFrame.from_dict(opt_stats[sim_id],orient='index')
        opt_stats_df1.index = pd.to_datetime(opt_stats_df1.index)
       
        opt_stats_df = pd.concat([opt_stats, opt_stats_df1])
       
    i = i+1
    
store_namespace(os.path.join(simu_path, folder,'emutemps'),emutemps_df)
store_namespace(os.path.join(simu_path, folder,'mpctemps'),mpctemps)
store_namespace(os.path.join(simu_path, folder,'opt_stats'),opt_stats_df)

constraints = {}
for bldg in bldg_list:
    setpoint_dict = load_namespace(os.path.join(simu_path, constr_folder, 'constraints_'+bldg+'_'+model_id)).data['TAir']
    constraints[bldg] = {}
    for param in setpoint_dict.keys():
        constraints[bldg]['hi'] = setpoint_dict['Slack_LTE'].display_data().resample(str(step)+'S').ffill()
        constraints[bldg]['lo'] = setpoint_dict['Slack_GTE'].display_data().resample(str(step)+'S').ffill()

constraints_df = pd.DataFrame.from_dict(constraints, orient = 'index')
#print(constraints_df['hi'].values)

weather = load_namespace(os.path.join(simu_path, folder, 'weather'))
price = load_namespace(os.path.join(simu_path, folder, 'sim_price'))
price = price.display_data()

if nodynprice==1:
    price = pd.Series(50, price.index,name='pi_e')
    
# """""""""""" Comfort violations """""""""""""""""""
violation = {}
#print(constraints_df.loc['Detached_0']['lo'])
for bldg in bldg_list:
    violation[bldg] = {} 
    for time in emutemps_df[bldg+'_'+model_id].index:
        #print(emutemps_df[bldg+'_'+model_id][time])
        emutemp = emutemps_df[bldg+'_'+model_id][time]

        constraint_hi = constraints_df.loc[bldg]['hi'][time]-273.15
        constraint_lo = constraints_df.loc[bldg]['lo'][time]-273.15
        
        if emutemp > constraint_hi:
            violation[bldg][time] = (emutemp - constraint_hi)*step/3600
        elif emutemp < constraint_lo:
            violation[bldg][time] = (constraint_lo-emutemp)*step/3600
        else:
            violation[bldg][time] = 0

violation_df = pd.DataFrame.from_dict(violation, orient = 'columns')
print(violation_df)
store_namespace(os.path.join(simu_path, folder,'violation_df'),violation_df)

ref_profile_df = pd.DataFrame.from_dict(ref_profile, orient = 'columns')
aggr_ref_df = ref_profile_df.sum(axis = 1)

aggr = {}
dt = []
#print(controlseq.keys())
for time in controlseq[sim_ids[0]].keys():

    control_start = datetime.datetime.strptime(time, '%m/%d/%Y %H:%M:%S')
    control_end = datetime.datetime.strptime(time, '%m/%d/%Y %H:%M:%S') + datetime.timedelta(seconds = 10*int(step))
    dt.append(control_start)
    aggr[time] = pd.DataFrame.from_dict(controlseq[sim_ids[0]][time],orient='columns')
    
dt = pd.DataFrame(dt,columns = ['Dates'])
dt = dt.set_index(pd.DatetimeIndex(dt['Dates']))
index = dt.index
index = index.tz_localize('UTC')
index = index.sort_values()
mast_index = index

last_str = index[-1].strftime('%m/%d/%Y %H:%M:%S')
real_cont = power_df

real_cont_aggr = real_cont.sum(axis=1)

aggrcom = {}
for time in aggr.keys():
    aggrcom[time] = aggr[time].sum(axis=1)
    
store_namespace(os.path.join(simu_path,folder,'real_cont'), real_cont)
store_namespace(os.path.join(simu_path,folder,'aggr'), aggr)
store_namespace(os.path.join(simu_path,folder,'aggrcom'), aggrcom)
store_namespace(os.path.join(simu_path,folder,'ref'), ref_profile_df)
store_namespace(os.path.join(simu_path,folder,'aggr_ref'), aggr_ref_df)

# --------------------- Flexibility factor and peak power ---------------
if mon == 'jan':
    ff_date = '01/07/2017'
if mon == 'mar':
    ff_date = '03/01/2017'
if mon == 'nov':
    ff_date = '11/20/2017'
hc_start = datetime.datetime.strptime(ff_date + ' 18:00:00', '%m/%d/%Y %H:%M:%S')
hc_end = index[-1]
lc_start = index[0]
lc_end = datetime.datetime.strptime(ff_date + ' 17:59:00', '%m/%d/%Y %H:%M:%S')

peak_comm = real_cont_aggr.max()
peak_time_comm = real_cont_aggr.idxmax()
peak_time_comm_hh = real_cont_aggr.resample(str(step)+'S').mean().idxmax()
peak_comm_hh = real_cont_aggr.resample(str(step)+'S').mean().max()

peak_comm =(peak_comm, peak_time_comm)
peak_comm_hh =(peak_comm_hh, peak_time_comm_hh)

print(peak_comm)
print(peak_comm_hh)

peak = {}
peak_hh = {}

cons_hc = real_cont[hc_start:hc_end]
cons_lc = real_cont[lc_start:lc_end]

print(cons_hc)

real_cont_hh = real_cont.resample(str(step)+'S').mean()

for bldg in bldg_list:
    bldg = bldg+'_'+model_id
    peak_val = real_cont.loc[:][bldg].max()
    peak_idx = real_cont.loc[:][bldg].idxmax()
    peak_hh_val = real_cont_hh.loc[:][bldg].max()
    peak_hh_idx = real_cont_hh.loc[:][bldg].idxmax()
    
    peak[bldg] = (peak_val, peak_idx)
    peak_hh[bldg] = (peak_hh_val, peak_hh_idx)
    
peak = pd.DataFrame.from_dict(peak, orient='index')
peak_hh = pd.DataFrame.from_dict(peak_hh, orient='index')
print(peak_hh)
print(peak)
# -----------------------------------------------


print('%%%%%%%%%---- Plots ----%%%%%%%')

fig_folder = os.path.join(simu_path, folder, 'figs')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Prices
fig = plt.figure(figsize=(11.69,8.27))
ax = fig.gca()
ax1 = ax.twinx()
i = 0
plot_times=[0,4,8,12,18]

for bldg in [bldg_list[0]]:
    ax.plot(ref_profile[bldg+'_'+model_id].index, ref_profile[bldg+'_'+model_id].values/1000,'--', label='ref_profile')
    ax.plot(real_cont.index, real_cont[bldg+'_'+model_id].values/1000,'-', label='ref_profile')
    
#resamp_index = index.asfreq('1800S')
ax.set_ylabel('Heat demand [kW]', fontsize=18)

ax1.plot(price.index, price.values, '--o', label="Price")
ax1.plot(flex_cost[bldg_list[0]+'_'+model_id].index, flex_cost[bldg_list[0]+'_'+model_id].values, '--o', label="Flex Cost")

handles,labels = [],[]
for ax in fig.axes:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)


ax1.set_ylabel(r'Price [pounds / kWh]', fontsize=18)
plt.xticks(rotation=35)
plt.xlabel("Time",fontsize=18)
plt.title("Decentralised Algorithm:\n Heat demand under dynamic pricing and loadshaping",fontsize=22)
# We change the fontsize of minor ticks label 
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)
plt.savefig(os.path.join(simu_path, folder, "mincost_price.png"))
plt.clf()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Outside temperature and optimised control sequence
fig = plt.figure(figsize=(11.69,8.27))
ax = fig.gca()
ax1 = ax.twinx()
plot_times=[0,4,8,12,18]

ax.plot(real_cont_aggr.index, real_cont_aggr.values/1000,'-x', label='realised')

ax.plot(aggr_ref_df.index, aggr_ref_df.values/1000,'--', label = 'aggr_reference')
ax.set_ylabel('Heat demand [kW]', fontsize=18)

ax1.plot(price.index, price.values, '--o', label="Price")
ax1.plot(flex_cost[bldg_list[0]+'_'+model_id].index, flex_cost[bldg_list[0]+'_'+model_id].values, '--o', label="Flex Cost")


handles,labels = [],[]
for ax in fig.axes:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

ax1.set_ylabel(r'Price [pounds / kWh]', fontsize=18)
plt.xticks(rotation=35)
plt.xlabel("Time",fontsize=18)
plt.title("Decentralised Algorithm:\n Power demand",fontsize=22)
# We change the fontsize of minor ticks label 
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)
plt.savefig(os.path.join(simu_path, folder, "mincost_price_aggr.png"))
plt.clf()



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Temperatures
fig = plt.figure(figsize=(11.69,8.27))
ax = fig.gca()
#ax1 = ax.twinx()
plot_bldgs = [0]
plot_times=[0,1,2,3,4]
i= 0
#print(emutemps)
for sim_id in sim_ids:
    i = 0
    for time in mpctemps[sim_id].keys():
        j = 0
        for bldg in mpctemps[sim_id][time].keys():
            if j in plot_bldgs:
                ax.plot(mpctemps[sim_id][time][bldg].index, mpctemps[sim_id][time][bldg].values-273.15, '-' , label='mpc_'+bldg)
            j = j+1
        i = i+1

        
handles,labels = [],[]
for ax in fig.axes:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)    

#ax.legend(fontsize=14)
plt.xlabel("Time",fontsize=18)
plt.ylabel(r"Temperature [$^\circ$C]",fontsize=18)
plt.title("Predicted Temperatures with Cost Minimisation",fontsize=22)
plt.xticks(rotation=35)
# We change the fontsize of minor ticks label 
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)
plt.legend(handles,labels, bbox_to_anchor = (1.04,0.5), loc ='center left')
plt.savefig(os.path.join(simu_path, folder, "temps_mpc.pdf"),bbox_inches="tight")
plt.savefig(os.path.join(simu_path, folder, "temps_mpc.png"),bbox_inches="tight")
plt.clf()
#print(ref_heatinput)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Temperatures

fig = plt.figure(figsize=(11.69,8.27))
ax = fig.gca()
#ax1 = ax.twinx()
plot_bldgs = [0]
i= 0
#print(emutemps)
for sim_id in sim_ids:
    if i == 0:
        emutemps_df = pd.DataFrame.from_dict(emutemps[sim_id],orient='index')
        emutemps_df.index = pd.to_datetime(emutemps_df.index)
        emutemps_df.index = emutemps_df.index.shift(1, freq=str(step)+'S')
    else:
        emutemps_df1 = pd.DataFrame.from_dict(emutemps[sim_id],orient='index')
        emutemps_df1.index = pd.to_datetime(emutemps_df1.index)
        emutemps_df1.index = emutemps_df1.index.shift(1, freq=str(step)+'S')
        emutemps_df = pd.concat([emutemps_df, emutemps_df1])
    i = i+1

#print(emutemps_df)

i = 0
for bldg in bldg_list:
    #if i in plot_bldgs:
        #print(emutemps_df.index)
    ax.plot(emutemps_df.index, emutemps_df.values, '-')
    i = i+1
    
handles,labels = [],[]
for ax in fig.axes:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)


#ax.legend(fontsize=14)
plt.xlabel("Time",fontsize=18)
plt.ylabel(r"Temperature [$^\circ$C]",fontsize=18)
plt.title("Decentralised Algorithm: Emulated Temperatures with Cost Minimisation",fontsize=22)
plt.xticks(rotation=35)
# We change the fontsize of minor ticks label 
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)
#plt.legend(handles,labels, bbox_to_anchor = (1.04,0.5), loc ='center left')
plt.savefig(os.path.join(simu_path, folder, "temps_emu.pdf"),bbox_inches="tight")
plt.savefig(os.path.join(simu_path, folder, "temps_emu.png"),bbox_inches="tight")
plt.clf()

print('%%%% --- Aggregated heat consumption (Wh) --- %%%%')
aggr_bldg = {}
aggr_comm = {}
aggr_total = {}

for time in aggr.keys():
    aggr_bldg[time] = aggr[time].resample('3600S').mean().sum(axis=0)
    aggr_comm[time] = aggr[time].resample('3600S').mean().sum(axis=1)
    aggr_total[time] = aggr_comm[time].resample('3600S').mean().sum(axis=0)

print('Projected energy consumption per building [Wh]')
print(aggr_bldg)
print('Projected energy consumption [Wh] over the community:')
print(aggr_comm)
print('Projected energy consumptions [Wh]:')
print(aggr_total)

print('Realised energy consumption over the community [Wh]:')
print(real_cont_aggr.resample('3600S').mean())

print('Realised energy consumption over the community [Wh]:')
print(real_cont_aggr.resample('3600S').mean().sum(axis=0))

price = price[mast_index[0]:mast_index[-1]]

#print(price)

real_cont_re = real_cont_aggr.resample(str(step)+'S').mean()
real_cont.index = real_cont.index.tz_localize(None)
ref_profile_df.index=ref_profile_df.index.tz_localize(None)
dr_diff = real_cont.resample(str(step)+'S').mean()[mast_index[0]:mast_index[-1]] - ref_profile_df.resample(str(step)+'S').mean()[mast_index[0]:mast_index[-1]]

i=0
for sim_id in sim_ids:
    if i == 0:
        df_wols = pd.DataFrame.from_dict(controlseq[sim_id][controlseq_time],orient='columns')
        df_wols.index = pd.to_datetime(df_wols.index)
    else:
        df_wols_df1 = pd.DataFrame.from_dict(controlseq[sim_id][controlseq_time],orient='columns')
        df_wols_df1.index = pd.to_datetime(df_wols_df1.index)
        #power_df1.index = power_df1.index.shift(1, freq=str(step)+'S')
        df_wols = pd.concat([df_wols, df_wols_df1])


compr_capacity_list = [float(4500.0)]*10+[float(3000.0)]*20
i=0
for column in df_wols:
    df_wols[column] = df_wols[column]*compr_capacity_list[i]
    i += 1
#print(df_wols)
df_wols.index = df_wols.index.tz_localize(None)

store_namespace(os.path.join(simu_path,folder,'ref_wols'), df_wols)

dr_diff_wols = real_cont.resample(str(step)+'S').mean()[mast_index[0]:mast_index[-1]] - df_wols.resample(str(step)+'S').mean()[mast_index[0]:mast_index[-1]]

print('%%%%%%%%%%%%%%%% Costs %%%%%%%%%%%%%%%%%%%%%%%%%%')

costs_flex = {}
costs_elec = {}
costs = {}

flex_cost[bldg_list[0]+'_'+model_id].index = flex_cost[bldg_list[0]+'_'+model_id].index.tz_localize(None)

price.index = price.index.tz_localize(None)
print(price)

for column in dr_diff.columns:  
    costs_flex[column] =  dr_diff[column].resample('1800S').mean()*flex_cost[bldg_list[0]+'_'+model_id] / 1000000 * 0.5
    if nodynprice == 1: 
        costs_elec[column] = real_cont[column].resample('1800S').mean() * price / 1000000 * 0.5 
        costs[column] = costs_flex[column] + costs_elec[column]
    else:
        costs_elec[column] = real_cont[column].resample('1800S').mean() * price['pi_e'] / 1000000 * 0.5 
        costs[column] = costs_flex[column] + costs_elec[column]
        
costs_flex = pd.DataFrame.from_dict(costs_flex,orient='columns')
costs_elec = pd.DataFrame.from_dict(costs_elec,orient='columns')
costs = pd.DataFrame.from_dict(costs,orient='columns')

print('----------------- Electricity costs [pounds]:')
print(costs_elec)
    
print('------------------- Flexibility costs [pounds]:')    
print(costs_flex)
#print(costs)

print('-------------- Total costs [pounds]:')   
print(costs)

print('------------------ Total hourly cost for community [pounds]: ')
print(costs.sum(axis=1))

print('------------------ Total costs over the community: ')
print(costs.sum(axis=1).sum(axis=0))

print('------------------ Electricity costs over the community: ')
print(costs_elec.sum(axis=1).sum(axis=0))

print('------------------ Flexibility costs over the community: ')
print(costs_flex.sum(axis=1).sum(axis=0))


print("-------------------Flexibility metrics and comfort-----------------")
print("Comfort violations [Ch]: ")
violation_bldg = violation_df.sum(axis=0)
violation_comm = violation_bldg.sum(axis=0)
print("Per building:")
print(violation_bldg)
print("Over the community:")
print(violation_comm)
print("Community Peak Consumption [kW]")
print(peak_comm)
print("Consumption high cost")
print(cons_hc.resample('3600S').mean().sum(axis=0).sum(axis=0))
print("Consumption low cost")
print(cons_lc.resample('3600S').mean().sum(axis=0).sum(axis=0))

store_namespace(os.path.join(simu_path,folder,'aggr_bldg'), aggr_bldg)
store_namespace(os.path.join(simu_path,folder,'aggr_comm'), aggr_comm)
store_namespace(os.path.join(simu_path,folder,'aggr_total'), aggr_total)
store_namespace(os.path.join(simu_path,folder,'peak_comm'), peak_comm)
store_namespace(os.path.join(simu_path,folder,'peak_comm_hh'), peak_comm_hh)
store_namespace(os.path.join(simu_path,folder,'cons_hc'), cons_hc)
store_namespace(os.path.join(simu_path,folder,'cons_lc'), cons_lc)
store_namespace(os.path.join(simu_path,folder,'costs'), costs.sum(axis=1))
store_namespace(os.path.join(simu_path,folder,'costs_flex'), costs_flex)
store_namespace(os.path.join(simu_path,folder,'costs_elec'), costs_elec)
store_namespace(os.path.join(simu_path,folder,'dr_diff'), dr_diff)
store_namespace(os.path.join(simu_path,folder,'dr_diff_wols'), dr_diff_wols)
