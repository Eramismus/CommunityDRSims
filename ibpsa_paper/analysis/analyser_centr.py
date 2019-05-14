# The analyser

import pandas as pd
import matplotlib.pyplot as plt
import dill
import os
import numpy as np
from funcs import store_namespace
from funcs import load_namespace
import datetime

community = 'ResidentialCommunity'
sim_id = 'MinEne'
model_id = 'R2CW_HP'
bldg_list = load_namespace(os.path.join('path_to_models', 'teaser_bldgs_residentialUK_10bldgs_fallback'))
print(bldg_list)
folder = '10bldgs_centr_dyn_fin_jan'
mon = 'jan'
nodynprice = 0
step = 300
constr_folder = 'decentr_enemin_'+mon

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
            
building = 'AggrMPC_ResUK_10bldgs_heatpump_rad_fallback'

simu_path = "path_to_simus"

price = load_namespace(os.path.join(simu_path, folder, 'sim_price'))

other_input = load_namespace(os.path.join(simu_path, folder, 'other_input_'+building))

flex_cost = load_namespace(os.path.join(simu_path, folder, 'flex_cost_'+building)).data['flex_cost'].get_base_data()
flex_cost = flex_cost.resample(str(step)+'S').mean()

ref_profile = load_namespace(os.path.join(simu_path, folder, 'ref_profile_'+building))


controlseq = {}
opt_control = {}
emutemps = {}
mpctemps = {}
opt_stats = {}
power = {}

for time_idx in sim_range:
    time_idx = time_idx.strftime('%m/%d/%Y %H:%M:%S')
    t = time_idx.replace('/','-').replace(':','-').replace(' ','-')
    
    controlseq[time_idx] = load_namespace(os.path.join(simu_path, folder, 'controlseq_'+building+'_'+sim_id+'_'+t))

    #opt_control = load_namespace(os.path.join(simu_path, folder, 'opt_control_DRCost_'+building))

    opt_stats[time_idx] = load_namespace(os.path.join(simu_path, folder, 'opt_stats_'+building+'_'+sim_id+'_'+t))

    emutemps[time_idx] = load_namespace(os.path.join(simu_path, folder, 'emutemps_'+building+'_'+sim_id+'_'+t))

    mpctemps[time_idx] = load_namespace(os.path.join(simu_path, folder, 'mpctemps_'+building+'_'+sim_id+'_'+t))

    power[time_idx] = load_namespace(os.path.join(simu_path, folder, 'power_'+building+'_'+sim_id+'_'+t))
    
opt_stats_df = pd.DataFrame.from_dict(opt_stats, orient='index')
opt_stats_df.index = pd.to_datetime(opt_stats_df.index)
    
store_namespace(os.path.join(simu_path, folder,'mpctemps'),mpctemps)
store_namespace(os.path.join(simu_path, folder,'emutemps'),emutemps)
store_namespace(os.path.join(simu_path, folder,'power'),power)
store_namespace(os.path.join(simu_path, folder,'opt_stats'),opt_stats_df)

#print(controlseq)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Optimization temps
emutemps_df = pd.DataFrame.from_dict(emutemps, orient='index')
emutemps_df.index = pd.to_datetime(emutemps_df.index)
emutemps_df.index = emutemps_df.index.shift(1, freq=str(step)+'S')

power_df = pd.DataFrame.from_dict(power,orient='index')
power_df.index = pd.to_datetime(power_df.index)
power_df.index = power_df.index.shift(1, freq=str(step)+'S')
#print(emutemps_df)
store_namespace(os.path.join(simu_path, folder,'emutemps'),emutemps_df)

constraints = {}
for bldg in bldg_list:
    setpoint_dict = load_namespace(os.path.join(simu_path, constr_folder, 'constraints_'+bldg+'_'+model_id)).data['TAir']
    constraints[bldg] = {}
    for param in setpoint_dict.keys():
        constraints[bldg]['hi'] = setpoint_dict['Slack_LTE'].display_data().resample(str(step)+'S').ffill()
        constraints[bldg]['lo'] = setpoint_dict['Slack_GTE'].display_data().resample(str(step)+'S').ffill()
    #power[sim_id] = load_namespace(os.path.join(simu_path, folder, 'power_'+sim_id))

constraints_df = pd.DataFrame.from_dict(constraints, orient = 'index')



weather = load_namespace(os.path.join(simu_path, folder, 'weather'))
price = load_namespace(os.path.join(simu_path, folder, 'sim_price'))
price = price.display_data()
if nodynprice==1:
    price = pd.Series(50, price.index,name='pi_e')

# """""""""""" Comfort violations """""""""""""""""""
#print(constraints_df)
#print(emutemps_df)
violation = {}
#print(constraints_df.loc['Detached_0']['lo'])
for bldg in emutemps_df.columns:
    violation[bldg] = {} 
    for time in emutemps_df.index:
        #print(emutemps_df[bldg+'_'+model_id][time])
        emutemp = emutemps_df[bldg][time]+273.15
        #emutemp = emutemp[time]
        
        bldg1 = bldg.split('_')
        bldg1 = bldg1[0] + '_' + bldg1[1]
        #emutemp = emutemp.values()
        #print(constraints_df)
        constraint_hi = constraints_df.loc[bldg1]['hi'][time]
        constraint_lo = constraints_df.loc[bldg1]['lo'][time]
        print(bldg1)
        print(time)
        print(constraint_hi)
        #print(constraint_hi)
        if emutemp > constraint_hi:
            violation[bldg][time] = (emutemp - constraint_hi)*step/3600
        elif emutemp < constraint_lo:
            violation[bldg][time] = (constraint_lo-emutemp)*step/3600
        else:
            violation[bldg][time] = 0

violation_df = pd.DataFrame.from_dict(violation, orient = 'columns')

store_namespace(os.path.join(simu_path, folder,'violation_df'),violation_df)


aggr = {}
dt = [] 
#time = pd.date_range(start, end, freq = meas_sampl+'S')
for time in controlseq.keys():
    #index = controlseq[time][controlseq[time].keys()[0]].display_data().index
    #df1 = pd.DataFrame(index = index)
    #df1 = df1.resample(str(step)+'S').mean()
    #print(df1)
    i = 0
    control_start = datetime.datetime.strptime(time, '%m/%d/%Y %H:%M:%S')
    control_end = datetime.datetime.strptime(time, '%m/%d/%Y %H:%M:%S') + datetime.timedelta(seconds = 10*int(1800))
    dt.append(control_start)
    for bldg in controlseq[time].keys():
        df = controlseq[time][bldg]
        #df = df.resample(str(step)+'S').mean()
        if i == 0:
            aggr[time] = df
        else:
            aggr[time] = pd.concat([aggr[time], df],axis=1)
            #aggr[time] = aggr[time].sum(axis=1)
        i=i+1

#print(aggr)
        
dt = pd.DataFrame(dt,columns = ['Dates'])
dt = dt.set_index(pd.DatetimeIndex(dt['Dates']))
index = dt.index
index = index.tz_localize('UTC')
index = index.sort_values()

mast_index = index

last_str = index[-1].strftime('%m/%d/%Y %H:%M:%S')

#real_cont = pd.DataFrame.from_dict(controlseq[last_str],orient='columns')[index[0]:index[-1]]
print(power_df)
real_cont = power_df
real_cont_aggr = real_cont.sum(axis=1)

aggrcom = {}
for time in aggr.keys():
    aggrcom[time] = aggr[time].sum(axis=1)
    
ref_prof_dict = {}
for bldg in ref_profile.data.keys():
    ref_prof_dict[bldg] = ref_profile.data[bldg].display_data()*3000
    
ref_profile_df = pd.DataFrame.from_dict(ref_prof_dict, orient = 'columns')
#print(ref_profile_df)

aggr_ref_df = ref_profile_df.sum(axis = 1)

store_namespace(os.path.join(simu_path,folder,'real_cont'), real_cont)
store_namespace(os.path.join(simu_path,folder,'aggr'), aggr)
store_namespace(os.path.join(simu_path,folder,'aggrcom'), aggrcom)
store_namespace(os.path.join(simu_path,folder,'ref'), ref_profile_df)


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

#print(peak_comm)
#print(peak_comm_hh)

peak = {}
peak_hh = {}

cons_hc = real_cont[hc_start:hc_end]
cons_lc = real_cont[lc_start:lc_end]

#print(cons_hc)

real_cont_hh = real_cont.resample(str(step)+'S').mean()

for bldg in bldg_list:
    bldg1 = bldg+'_'+model_id
    peak_val = real_cont.loc[:][bldg1].max()
    peak_idx = real_cont.loc[:][bldg1].idxmax()
    peak_hh_val = real_cont_hh.loc[:][bldg1].max()
    peak_hh_idx = real_cont_hh.loc[:][bldg1].idxmax()
    
    peak[bldg] = (peak_val, peak_idx)
    peak_hh[bldg] = (peak_hh_val, peak_hh_idx)
    
peak = pd.DataFrame.from_dict(peak, orient='index')
peak_hh = pd.DataFrame.from_dict(peak_hh, orient='index')
#print(peak_hh)
#print(peak)


print('%%%%%%%%%---- Plots ----%%%%%%%')
print(flex_cost)
#print(opt_stats)

#print(mpctemps)
#print(emutemps)

#print(controlseq)

#print(opt_control)

#print(price.display_data())

#print(flex_cost.display_data())

fig_folder = os.path.join(simu_path, folder)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Price and optimised control sequence
fig = plt.figure(figsize=(11.69,8.27))
ax = fig.gca()
ax1 = ax.twinx()

        
#print(aggr)
i=0
plot_times=[0,1,2,6]

for time in aggr.keys():
    if i in plot_times:
        #print(aggr[time])
        ax.plot(aggr[time].index, aggr[time].values/1000,'-', label='aggr ' + str(time))
    i = i+1

ax.set_ylabel('Heat Input [kW]', fontsize=18)

ax1.plot(price.index, price.values, '--o', label="Price")
ax1.plot(flex_cost.index, flex_cost.values, '--o', label="Flex Cost")

ax1.set_ylabel(r'Cost [pounds / MWh]', fontsize=18)
#ax.legend(fontsize=14)
#ax1.legend(fontsize=14)
#plt.legend(handles,labels, bbox_to_anchor = (1.04,0.5), loc ='center left')
plt.xticks(rotation=35)
plt.xlabel("Time",fontsize=18)
plt.title("Response of a residential community under dynamic pricing",fontsize=22)
# We change the fontsize of minor ticks label 
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)
plt.savefig(os.path.join(simu_path, folder, "control_price.png"),bbox_inches="tight")
plt.savefig(os.path.join(simu_path, folder, "control_price.pdf"),bbox_inches="tight")
plt.clf()

# ############### Aggregated 

fig = plt.figure(figsize=(11.69,8.27))
ax = fig.gca()
ax1 = ax.twinx()
#aggr = {}
i = 0
#price = price.display_data()
plot_times=[0,4,8,12,18]
#print(controlseq.keys())

i=0
#print(aggrcom)

#print(aggr_ref_df.index.freq)
#print(aggr_ref_df)

for time in aggrcom.keys():
    #aggrcom[time] = aggrcom[time].sum(axis=1)
    #if i in plot_times:
    #print(aggrcom[time].index.freq)
    #ax.plot(aggrcom[time].index, aggrcom[time].values,'-+', label=str(time)+' '+'output')
    i=i+1

ax.plot(real_cont_aggr.index, real_cont_aggr.values/1000,'-x', label='realised')

ax.plot(aggr_ref_df.index, aggr_ref_df.values/1000,'--', label = 'aggr_reference')
ax.set_ylabel('Heat demand [kW]', fontsize=18)

ax1.plot(price.index, price.values, '--o', label="Price")
ax1.plot(flex_cost.index, flex_cost.values, '--o', label="Flex Cost")


handles,labels = [],[]
for ax in fig.axes:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

ax1.set_ylabel(r'Price [pounds / kWh]', fontsize=18)
#ax.legend(fontsize=14, loc = 0)
#plt.legend(handles,labels, bbox_to_anchor = (1.04,0.5), loc ='center left')
plt.xticks(rotation=35)
plt.xlabel("Time",fontsize=18)
plt.title("Centralised Algorithm:\n Power demand",fontsize=22)
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

#print(mpctemps)
plot_times=[0,3,5]

i = 0
for time in mpctemps:
    for bldg in mpctemps[time]:
        if i in plot_times:
            index = mpctemps[time][bldg].index
            values = mpctemps[time][bldg].values-273.15
            ax.plot(index,values, '-')
    i = i+1
        
#print(emutemps_df)

handles,labels = [],[]
for ax in fig.axes:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

#ax.plot(emutemps_df.index, emutemps_df[0].values, '-x', label='emu_'+bldg)
plt.legend(handles,labels, bbox_to_anchor = (1.04,0.5), loc ='center left') 
plt.xlabel("Time",fontsize=18)
plt.ylabel(r"Temperature [$^\circ$C]",fontsize=18)
plt.title("Predicted Temperatures",fontsize=22)
plt.xticks(rotation=35)
# We change the fontsize of minor ticks label 
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)
plt.savefig(os.path.join(simu_path, folder, "temps_mpc.png"), bbox_inches="tight")
plt.clf()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Optimization temps
emutemps_df = pd.DataFrame.from_dict(emutemps, orient='index')
emutemps_df.index = pd.to_datetime(emutemps_df.index)
emutemps_df.index = emutemps_df.index.shift(1, freq=str(step)+'S')
#print(emutemps_df)


fig = plt.figure(figsize=(11.69,8.27))
ax = fig.gca()
#ax1 = ax.twinx()
plot_bldgs = [0, 10, 20, 29, 4, 12]
i= 0

ax.plot(emutemps_df.index, emutemps_df.values, '-')     

handles,labels = [],[]
for ax in fig.axes:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

plt.legend(handles,labels, bbox_to_anchor = (1.04,0.5), loc ='center left') 
plt.xlabel("Time",fontsize=18)
plt.ylabel(r"Temperature [$^\circ$ C]",fontsize=18)
plt.title("Emulated Temperatures",fontsize=22)
plt.xticks(rotation=35)
# We change the fontsize of minor ticks label 
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=12)
plt.savefig(os.path.join(simu_path, folder, "temps_emu.png"), bbox_inches="tight")
plt.clf()
    
#print(ref_heatinput)

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

#price = price.display_data()[mast_index[0]:mast_index[-1]]

#print(price)
#real_cont = real_cont_aggr.resample(str(step)+'S').mean()
#print(real_cont_aggr.index)
real_cont_aggr.index  = real_cont_aggr.index.tz_localize(None)
#print(real_cont_aggr.index)

real_cont.index = real_cont.index.tz_localize(None)
#print(real_cont_aggr)
#real_cont.replace(tzinfo=None)
ref_profile_df.index = ref_profile_df.index.tz_localize(None)
mast_index = mast_index.tz_localize(None)
#print(ref_profile_df)
#ref_profile_df.replace(tzinfo=None)
dr_diff = real_cont_aggr.resample(str(step)+'S').mean()[mast_index[0]:mast_index[-1]] - ref_profile_df['ref_profile'].resample(str(step)+'S').mean()[mast_index[0]:mast_index[-1]]


ref_wols = controlseq[controlseq_time]
ref_wols = pd.DataFrame.from_dict(ref_wols, orient='columns').sum(axis=1)*3000
ref_wols.index = ref_wols.index.tz_localize(None)
#print(ref_wols)

store_namespace(os.path.join(simu_path,folder,'ref_wols'), ref_wols)


dr_diff_wols = real_cont_aggr.resample(str(step)+'S').mean()[mast_index[0]:mast_index[-1]] - (ref_wols.resample(str(step)+'S').mean()[mast_index[0]:mast_index[-1]])
#print(dr_diff_wols)

#dr_diff = dr_diff.resample(str(step)+'S').mean()

#print(flex_cost)
#print(flex_cost[flex_index[0]:flex_index[-1]]['flex_cost'])

print('%%%%%%%%%%%%%%%% Costs %%%%%%%%%%%%%%%%%%%%%%%%%%')
#print(price.display_data()[index[0]:index[-1]])

#costs_elec = (real_cont_re * price['pi_e']) / 1000000 * 0.5  
#print(real_cont)

costs_flex = {}
costs_elec = {}
costs = {}
price.index = price.index.tz_localize(None)
flex_cost.index = flex_cost.index.tz_localize(None)
    
costs_flex[building] =  dr_diff.resample('1800S').mean() * flex_cost / 1000000 * 0.5

for bldg in bldg_list:
    bldg = bldg+'_'+model_id
    if nodynprice == 1: 
        costs_elec[bldg] = real_cont[bldg].resample('1800S').mean() * price / 1000000 * 0.5 
    else:
        costs_elec[bldg] = real_cont[bldg].resample('1800S').mean() * price['pi_e'] / 1000000 * 0.5
    

    
costs_flex = pd.DataFrame.from_dict(costs_flex,orient='columns')
costs_elec = pd.DataFrame.from_dict(costs_elec,orient='columns')

costs[building] = costs_elec.sum(axis=1)+costs_flex[building]

costs = pd.DataFrame.from_dict(costs,orient='columns')


#costs_flex = pd.DataFrame.from_dict(costs_flex,orient='columns')

#costs = pd.DataFrame.from_dict(costs,orient='columns')

print('----------------- Electricity costs [pounds]:')
print(costs_elec)
    
print('------------------- Flexibility costs [pounds]:')    
print(costs_flex)
#print(costs)

print('-------------- Total costs [pounds]:')   
print(costs)

print('------------------ Total hourly cost for community [pounds]: ')
#print(costs.sum(axis=1))

print('------------------ Total costs over the community: ')
print(costs.sum(axis=0))

print('------------------ Electricity costs over the community: ')
print(costs_elec.sum(axis=1).sum(axis=0))

print('------------------ Flexibility costs over the community: ')
print(costs_flex.sum(axis=0))

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
store_namespace(os.path.join(simu_path,folder,'costs'), costs)
store_namespace(os.path.join(simu_path,folder,'costs_flex'), costs_flex)
store_namespace(os.path.join(simu_path,folder,'costs_elec'), costs_elec)
store_namespace(os.path.join(simu_path,folder,'dr_diff'), dr_diff)
store_namespace(os.path.join(simu_path,folder,'dr_diff_wols'), dr_diff_wols)
#print(sum(opt_control_energy['ConvGain2'].get_base_data().resample('3600S').mean()))

#print('%%%% --- Reference (PI) heat consumption (Wh) --- %%%%')
#print(sum(ref_heatinput.resample('3600S').mean()))
