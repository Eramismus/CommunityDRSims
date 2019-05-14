import pandas as pd
import matplotlib.pyplot as plt
import dill
import os
import numpy as np
from funcs import store_namespace
from funcs import load_namespace
import datetime
import matplotlib.mlab as mlab
import matplotlib.dates as mdates

community = 'ResidentialCommunity'
sim_id = 'MinCost_Centr_Res'
model_id = 'R2CW_HP'
bldg_list = load_namespace(os.path.join('path to models', 'teaser_bldgs_residentialUK_10bldgs_fallback'))
print(bldg_list)
mons = ['jan', 'mar', 'nov']
nodynprice=0
folders = ['10bldgs_decentr_nodyn_fin', '10bldgs_decentr_dyn_fin', '10bldgs_centr_nodyn_fin', '10bldgs_centr_dyn_fin']

pricing = ['Static', 'Dynamic', 'Static', 'Dynamic']
structure = ['Decentralised', 'Decentralised', 'Centralised', 'Centralised']
            
building = 'AggrMPC_ResidentialUK'

simu_path = "path to sims"
analysis_folder = 'analysis_flex'

step = 300
horizon = 2*3600/step

costs = {}
aggrcom = {}
aggr = {}
real_cont = {}
real_cont_aggr = {}
ref_profile = {}
ref_profile_aggr = {}
violations = {}
peak_comm = {}
cons_hc = {}
cons_lc = {}
emutemps = {}
mpctemps = {}
costs_flex = {}
costs_elec = {}
dr_diff = {}
dr_diff_wols = {}
PI_diff = {}
enemin_diff = {}
PI_diff_aggr = {}
enemin_diff_aggr = {}
PIcons_lc = {}
PIcons_hc = {}
minene_decentr_hc = {}
minene_decentr_lc = {}
proj_cost = {}
proj_cost_dfs = {}
proj_flex_cost = {}
proj_flex_cost_dfs = {}

ref_projections = {}
diff_ref = {}
diff_ref_df = {}
projected_cost_ind = {}
projected_flexcost_ind = {}
proj_cost_ind_dfs = {}
proj_flexcost_ind_dfs = {}
proj_aggrcost_ind_dfs = {}
proj_aggrflexcost_ind_dfs = {}
proj_aggrcost_dfs = {}
proj_aggrflexcost_dfs = {}
proj_aggrtotcost_dfs = {}
proj_totcost_ind_dfs = {}

#print(refheat_aggr)
minenes = {}
refheats = {}
reftemps = {}

h=0
for mon in mons:
    if mon == 'jan':
        ff_date = '1/7/2017'
    if mon == 'mar':
        ff_date = '3/1/2017'
    if mon == 'nov':
        ff_date = '11/20/2017'
        
    DRstart = datetime.datetime.strptime(ff_date + ' 17:30:00', '%m/%d/%Y %H:%M:%S') # hour to start DR - ramp down 30 mins before
    DRend = datetime.datetime.strptime(ff_date + ' 18:30:00', '%m/%d/%Y %H:%M:%S') # hour to end DR - ramp 30 mins later
    DR_call_start = datetime.datetime.strptime(ff_date + ' 17:00:00', '%m/%d/%Y %H:%M:%S') # Round of loop to implement the call
    ref_time = datetime.datetime.strptime(ff_date + ' 16:55:00', '%m/%d/%Y %H:%M:%S')
    DR_call_end = datetime.datetime.strptime(ff_date + ' 18:30:00', '%m/%d/%Y %H:%M:%S') # Round of loop to
    sim_end = datetime.datetime.strptime(ff_date + ' 19:00:00', '%m/%d/%Y %H:%M:%S')
    minene_decentr = load_namespace(os.path.join(simu_path, '10bldgs_decentr_enemin_'+mon, "real_cont"))

    refheatinput_df = load_namespace(os.path.join(simu_path, 'analysis_ref','refheatinput_df_'+mon))
    reftemp_df = load_namespace(os.path.join(simu_path, 'analysis_ref', 'reftemp_df_'+mon))
    refheat_aggr = refheatinput_df.sum(axis=1)

    refheatinput_df.index = refheatinput_df.index.tz_localize(None)
    
    minenes[mon] = minene_decentr
    refheats[mon] = refheatinput_df
    reftemps[mon] = reftemp_df 
    
    
    index = minene_decentr.resample(str(step)+'S').mean().index
    hc_start = DRstart
    hc_end = index[-1]
    lc_start = datetime.datetime.strptime(ff_date + ' 16:30:00', '%m/%d/%Y %H:%M:%S')
    lc_end = DRstart - datetime.timedelta(seconds = 1)
    
    #print(lc_start)
    #print(hc_end)
    
    PIcons_hc[mon]=refheatinput_df[hc_start:hc_end]
    PIcons_lc[mon]=refheatinput_df[lc_start:lc_end]
    #print(PIcons_hc[mon].sum(axis=1))
    
    minene_decentr_hc[mon] = minene_decentr[hc_start:hc_end]
    minene_decentr_lc[mon] = minene_decentr[lc_start:lc_end]
    
    i = 0
    for folder in folders:
        case = folder+'_'+mon
    
        costs[case] = load_namespace(os.path.join(simu_path,case,'costs'))
        aggrcom[case] = load_namespace(os.path.join(simu_path,case,'aggrcom'))
        aggr[case] = load_namespace(os.path.join(simu_path,case,'aggr'))
        real_cont[case] = load_namespace(os.path.join(simu_path,case,'real_cont'))
        real_cont_aggr[case] =  real_cont[case].sum(axis=1)
        ref_profile[case] = load_namespace(os.path.join(simu_path,case,'ref')).resample(str(step)+'S').mean()
        violations[case] = load_namespace(os.path.join(simu_path,case,'violation_df'))
        peak_comm[case] = load_namespace(os.path.join(simu_path,case,'peak_comm'))
        cons_hc[case] = load_namespace(os.path.join(simu_path,case,'cons_hc'))
        cons_lc[case] = load_namespace(os.path.join(simu_path,case,'cons_lc'))
        emutemps[case] = load_namespace(os.path.join(simu_path,case,'emutemps'))
        mpctemps[case] = load_namespace(os.path.join(simu_path,case,'mpctemps'))
        costs_flex[case] = load_namespace(os.path.join(simu_path,case,'costs_flex'))
        costs_elec[case] = load_namespace(os.path.join(simu_path,case,'costs_elec'))
        dr_diff[case] = load_namespace(os.path.join(simu_path,case,'dr_diff'))
        dr_diff_wols[case] = load_namespace(os.path.join(simu_path,case,'dr_diff_wols'))
        
        #print(dr_diff[case])
        index = real_cont[case].resample(str(step)+'S').mean().index
        #print(index)
        
        real_cont[case].index = real_cont[case].index.tz_localize(None)
        
        if structure[i] in ['Centralised']:
            PI_diff[case] = {}
            for bldg in real_cont[case].columns:
                PI_diff[case][bldg] = real_cont[case].resample('10S').ffill()[DRstart:DRend][bldg]-refheatinput_df.resample('10S').ffill()[DR_call_start:DRend][bldg]
            PI_diff[case] = pd.DataFrame.from_dict(PI_diff[case],orient='columns')
            
        else:
            PI_diff[case] = real_cont[case].resample('10S').ffill()[DRstart:DRend]-refheatinput_df.resample('10S').ffill()[DR_call_start:DRend]
            
        PI_diff_aggr[case] = PI_diff[case].sum(axis=1)
        
        if structure[i] in ['Decentralised','Hierarchical']:
            enemin_diff[case] = real_cont[case].resample('10S').ffill()[DRstart:DRend]-minene_decentr[DR_call_start:DRend].resample('10S').ffill()
        else:
            enemin_diff[case] = real_cont[case].resample('10S').ffill()[DRstart:DRend]-minene_decentr[DR_call_start:DRend].resample('10S').ffill()  
        enemin_diff_aggr[case] = enemin_diff[case].sum(axis=1)
        
        #print(dr_diff[case])
        
        if structure[i] in ['Hierarchical', 'Decentralised']:
            index = dr_diff[case].resample(str(step)+'S').mean().index
            dr_diff[case] = dr_diff[case][DR_call_start:sim_end].sum(axis=1)
            dr_diff_wols[case] = dr_diff_wols[case][DR_call_start:sim_end].sum(axis=1)
        else:
            index = dr_diff[case].resample(str(step)+'S').mean().index
            dr_diff[case] = dr_diff[case][DR_call_start:sim_end]
            dr_diff_wols[case] = dr_diff_wols[case][DR_call_start:sim_end]
        #print(dr_diff[case])
        
        ref_profile_aggr[case] = ref_profile[case].sum(axis=1).resample(str(step)+'S').mean()
        
        cons_hc[case] = real_cont[case][hc_start:hc_end]
        cons_lc[case] = real_cont[case][lc_start:lc_end]        
        
        ### Projected costs
        time_idx = []
        for time_str in aggrcom[case].keys():
            time_idx.append(datetime.datetime.strptime(time_str, '%m/%d/%Y %H:%M:%S'))
            time_idx.sort()
            #print(time_idx)
        
        #print(time_idx)
        index_first = real_cont_aggr[case].index[0]
        
        price = load_namespace(os.path.join(simu_path, 'prices', 'sim_price_'+mon)).display_data()
        price.index = price.index.tz_localize(None)
        
        price_nodyn = pd.Series(50, price.index, name='pi_e')
        price_nodyn = pd.DataFrame(price_nodyn)
        #print(price_nodyn)
        
        if pricing[i] == 'Static':
            price = price_nodyn
        
        flex_cost = load_namespace(os.path.join(simu_path, folders[0]+'_'+mon, 'flex_cost_Terrace_2_R2CW_HP')).data['flex_cost'].get_base_data()
        flex_cost.index = flex_cost.index.tz_localize(None)
        
        proj_cost[case] = {}
        proj_flex_cost[case] = {}
        projected_cost_ind[case] = {}
        projected_flexcost_ind[case] = {}
        proj_cost_ind_dfs[case] = {}
        proj_flexcost_ind_dfs[case] = {}
        proj_totcost_ind_dfs[case] = {}
        proj_aggrcost_ind_dfs[case] = {}
        proj_aggrflexcost_ind_dfs[case] = {}
        
        # Reference profile
        ref_projections[case] = aggr[case][ref_time.strftime('%m/%d/%Y %H:%M:%S')][ref_time:sim_end]*3000
        ref_projections[case].index = ref_projections[case].index.tz_localize(None)
        
        for idx in time_idx:
            time_str = idx.strftime('%m/%d/%Y %H:%M:%S')
            end_idx = idx + datetime.timedelta(seconds = horizon*step)
            
            aggrcom[case][time_str].index = aggrcom[case][time_str].index.tz_localize(None)
            ref_profile_aggr[case].index = ref_profile_aggr[case].index.tz_localize(None)
            aggr[case][time_str].index = aggr[case][time_str].index.tz_localize(None)
            
            projected_cost_ts = aggrcom[case][time_str]*3000 * price['pi_e'].resample(str(step)+'S').ffill() / 1000000 * step/3600
            
            projected_flex_cost_ts = (aggrcom[case][time_str]*3000 - ref_profile_aggr[case]) * flex_cost.resample(str(step)+'S').ffill() / 1000000 * step/3600
            
            projected_cost = projected_cost_ts[idx:end_idx].sum()
            projected_flex_cost = projected_flex_cost_ts[idx:end_idx].sum()
            
            if idx < DR_call_start:
                projected_flex_cost = 0
            
            proj_cost[case][time_str] = projected_cost
            proj_flex_cost[case][time_str] = projected_flex_cost
            
            projected_cost_ind[case][time_str] = {}
            projected_flexcost_ind[case][time_str] = {}
            
            proj_cost_ind_dfs[case][time_str] = {}
            proj_flexcost_ind_dfs[case][time_str] = {}
            
            proj_aggrcost_ind_dfs[case][time_str] = {}
            proj_aggrflexcost_ind_dfs[case][time_str] = {}
            
            for bldg in bldg_list:
                bldg_key1 = bldg+'_R2CW_HP'
                if structure[i] == 'Centralised':
                    bldg_key = 'HPPower_'+bldg
                else:
                    bldg_key = bldg+'_R2CW_HP'
                
                # For each individual building
                projected_cost_ind[case][time_str][bldg_key1] = aggr[case][time_str][bldg_key][idx:end_idx]*3000 * price['pi_e'][idx:end_idx].resample(str(step)+'S').ffill() / 1000000 * step/3600
                
                if idx < DR_call_start:
                    projected_flexcost_ind[case][time_str][bldg_key1] = aggr[case][time_str][bldg_key][idx:end_idx]*0
                else: 
                    projected_flexcost_ind[case][time_str][bldg_key1] = (aggr[case][time_str][bldg_key][idx:end_idx]*3000 - ref_projections[case][bldg_key])  * flex_cost[idx:end_idx].resample(str(step)+'S').ffill() / 1000000 * step/3600
            
            proj_cost_ind_dfs[case][time_str] = pd.DataFrame.from_dict(projected_cost_ind[case][time_str], orient='columns').fillna(value = 0)
            proj_flexcost_ind_dfs[case][time_str] = pd.DataFrame.from_dict(projected_flexcost_ind[case][time_str], orient='columns').fillna(value = 0)
            
            proj_totcost_ind_dfs[case][time_str] = (proj_flexcost_ind_dfs[case][time_str] + proj_cost_ind_dfs[case][time_str]).dropna(axis = 0)
            
            proj_aggrcost_ind_dfs[case][time_str] = proj_cost_ind_dfs[case][time_str].sum(axis=0)
            proj_aggrflexcost_ind_dfs[case][time_str] = proj_flexcost_ind_dfs[case][time_str].sum(axis=0)
            
        proj_aggrcost_dfs[case] = pd.DataFrame.from_dict(proj_aggrcost_ind_dfs[case], orient='index')
        proj_aggrflexcost_dfs[case] = pd.DataFrame.from_dict(proj_aggrflexcost_ind_dfs[case], orient='index')
        proj_aggrtotcost_dfs[case] = proj_aggrflexcost_dfs[case] + proj_aggrcost_dfs[case]
        
        ### Difference to reference over Dr period for each building
        diff_ref[case] = {}
        for bldg in bldg_list:
            bldg_key1 = bldg+'_R2CW_HP'
            if structure[i] == 'Centralised':
                bldg_key = 'HPPower_'+bldg
            else:
                bldg_key = bldg+'_R2CW_HP'
            
            real_cont[case].index = real_cont[case].index.tz_localize(None)
            ref_projections[case].index = ref_projections[case].index.tz_localize(None)
            
            diff_ref[case][bldg_key1] = real_cont[case][bldg_key1][ref_time:sim_end] - ref_projections[case][bldg_key]
        
        diff_ref_df[case] = pd.DataFrame.from_dict(diff_ref[case], orient='columns')
            
        i=i+1

#print(proj_totcost_ind_dfs[case])
#exit()
        
violation_comm = {}
totreal_cont = {}
flexcosts = {}
elecosts = {}
totcons_hc = {}
totcons_lc = {}
totcosts = {}
comm_metric = {}

for mon in mons:
    price = load_namespace(os.path.join(simu_path, 'prices', 'sim_price_'+mon)).display_data()
    price.index = price.index.tz_localize(None)
    if mon == 'jan':
        ff_date = '1/7/2017'
    if mon == 'mar':
        ff_date = '3/1/2017'
    if mon == 'nov':
        ff_date = '11/20/2017'
        
    DRstart = datetime.datetime.strptime(ff_date + ' 17:30:00', '%m/%d/%Y %H:%M:%S') # hour to start DR - ramp down 30 mins before
    DRend = datetime.datetime.strptime(ff_date + ' 18:30:00', '%m/%d/%Y %H:%M:%S') # hour to end DR - ramp 
    DR_call_start = datetime.datetime.strptime(ff_date + ' 17:00:00', '%m/%d/%Y %H:%M:%S') # Round of loop to implement the call
    
    for folder in folders:
        case = folder+'_'+mon
        comm_metric[case] = {}
        comm_metric[case]['Total Consumption [kWh]'] = real_cont_aggr[case].resample(str(step)+'S').mean().sum(axis=0)*step/3600/1000
        
        case_cost = real_cont_aggr[case].resample('1800S').mean() * price['pi_e'] / 1000000 * 0.5
        
        comm_metric[case]['Violation Ch'] = violations[case].sum(axis=0).sum(axis=0)
        comm_metric[case]['totcons_hc [kWh]'] = cons_hc[case].resample(str(step)+'S').mean().sum(axis=0).sum(axis=0)*step/3600/1000
        comm_metric[case]['totcons_lc [kWh]'] = cons_lc[case].resample(str(step)+'S').mean().sum(axis=0).sum(axis=0)*step/3600/1000
        comm_metric[case]['peak consumption [kW]'] = peak_comm[case][0]/1000
        comm_metric[case]['peak time'] = peak_comm[case][1]
        comm_metric[case]['flex_costs'] = costs_flex[case].sum(axis=0).sum(axis=0)
        comm_metric[case]['elec_costs'] = case_cost.sum(axis=0)
        comm_metric[case]['Case Cost [pounds]'] = case_cost.sum(axis=0)+costs_flex[case].sum(axis=0).sum(axis=0)
        comm_metric[case]['dr-diff [kWh]'] = dr_diff[case][DRstart:DRend].resample(str(step)+'S').mean().sum(axis=0)*step/3600/1000
        comm_metric[case]['dr-diff-mean [W]'] = dr_diff[case][DRstart:DRend].mean()
        comm_metric[case]['dr-diff-wols [kWh]'] = dr_diff_wols[case][DRstart:DRend].resample(str(step)+'S').mean().sum(axis=0)*step/3600/1000
        comm_metric[case]['dr-diff-wols-mean [W]'] = dr_diff_wols[case][DRstart:DRend].mean()
        comm_metric[case]['enemin-diff [kWh]'] = enemin_diff_aggr[case][DRstart:DRend].resample(str(step)+'S').mean().sum(axis=0)*step/3600/1000
        comm_metric[case]['enemin-diff-mean [W]'] = enemin_diff_aggr[case][DRstart:DRend].mean()
        comm_metric[case]['PI-diff [kWh]'] = PI_diff_aggr[case][DRstart:DRend].resample(str(step)+'S').mean().sum(axis=0)*step/3600/1000
        comm_metric[case]['PI-diff-mean [W]'] = PI_diff_aggr[case][DRstart:DRend].mean()
        #comm_metric[case]['minene_centr_lc_'+mon] = minene_centr_lc[mon].resample('3600S').mean().sum(axis=0).sum(axis=0)/1000
        #comm_metric[case]['minene_centr_hc_'+mon] = minene_centr_hc[mon].resample('3600S').mean().sum(axis=0).sum(axis=0)/1000
        
        proj_cost_dfs[case] = pd.DataFrame.from_dict(proj_cost[case],orient='index').sort_index()
        proj_flex_cost_dfs[case] = pd.DataFrame.from_dict(proj_flex_cost[case],orient='index').sort_index()
        
        proj_cost_dfs[case].index = pd.to_datetime(proj_cost_dfs[case].index)
        proj_flex_cost_dfs[case].index = pd.to_datetime(proj_flex_cost_dfs[case].index)
        
        totreal_cont[case] = real_cont_aggr[case].resample('3600S').mean().sum(axis=0)
        violation_comm[case] = violations[case].sum(axis=0).sum(axis=0)
        totcons_hc[case] = cons_hc[case].resample(str(step)+'S').mean().sum(axis=0).sum(axis=0)*step/3600/1000
        totcons_lc[case] = cons_lc[case].resample(str(step)+'S').mean().sum(axis=0).sum(axis=0)*step/3600/1000
        totcosts[case] = costs[case].sum(axis=0) 
        flexcosts[case] = costs_flex[case].sum(axis=0)
        elecosts[case] = costs_elec[case].sum(axis=0)
    
    comm_metric['PIcons_'+mon] = {}
    comm_metric['minene_decentr_'+mon] = {}
    comm_metric['PIcons_'+mon]['totcons_lc [kWh]'] = PIcons_lc[mon].resample(str(step)+'S').mean().sum(axis=1).sum(axis=0)*step/3600/1000
    comm_metric['PIcons_'+mon]['totcons_hc [kWh]'] = PIcons_hc[mon].resample(str(step)+'S').mean().sum(axis=1).sum(axis=0)*step/3600/1000
    comm_metric['minene_decentr_'+mon]['totcons_lc [kWh]'] = minene_decentr_lc[mon].resample(str(step)+'S').mean().sum(axis=0).sum(axis=0)*step/3600/1000
    comm_metric['minene_decentr_'+mon]['totcons_hc [kWh]'] = minene_decentr_hc[mon].resample(str(step)+'S').mean().sum(axis=0).sum(axis=0)*step/3600/1000

#print(proj_flex_cost_dfs)
#exit()
    
comm_metric_df = pd.DataFrame.from_dict(comm_metric, orient='columns')
comm_metric_df.to_csv(os.path.join(simu_path, analysis_folder, 'comm_metrics_DRcases_dyn.csv'))
print(comm_metric_df)
#print(costs)

price = load_namespace(os.path.join(simu_path, 'sim_price_jan')).display_data()
price_dyn = load_namespace(os.path.join(simu_path, 'sim_price_jan')).display_data()
price_nodyn = pd.Series(50, price.index,name='pi_e')

#flex_cost = load_namespace(os.path.join(simu_path, 'sim_price')).display_data()

flex_cost = load_namespace(os.path.join(simu_path, folders[0]+'_'+mons[0], 'flex_cost_Terrace_2_R2CW_HP')).data['flex_cost'].get_base_data()

dt = []
for time in aggr[case].keys():
    control_start = datetime.datetime.strptime(time, '%m/%d/%Y %H:%M:%S')
    control_end = datetime.datetime.strptime(time, '%m/%d/%Y %H:%M:%S') + datetime.timedelta(seconds = horizon)
    dt.append(control_start)

dt = pd.DataFrame(dt,columns = ['Dates'])
dt = dt.set_index(pd.DatetimeIndex(dt['Dates']))
index = dt.index
index = index.tz_localize('UTC')
index = index.sort_values()

print('%%%%%%%%%---- Plots ----%%%%%%%')    
cmap = plt.get_cmap('viridis')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Differences to MPC projections
#fig = plt.figure(figsize=(11.69,8.27))
f, axarr = plt.subplots(nrows = 3, ncols = 2, sharex = 'row', sharey=False, figsize=(7.0, 8.5))

folders_plot = folders
structures_plot = structure

i = 0 
for mon in mons:
    
    if mon == 'jan':
        ff_date = '1/7/2017'
    if mon == 'mar':
        ff_date = '3/1/2017'
    if mon == 'nov':
        ff_date = '11/20/2017'
    
    DRstart = datetime.datetime.strptime(ff_date + ' 17:30:00', '%m/%d/%Y %H:%M:%S') # hour to start DR - ramp down 30 mins before
    DRend = datetime.datetime.strptime(ff_date + ' 18:30:00', '%m/%d/%Y %H:%M:%S') # hour to end DR - ramp 30 mins later
    DR_call_start = datetime.datetime.strptime(ff_date + ' 17:00:00', '%m/%d/%Y %H:%M:%S') # 
    
    #price = load_namespace(os.path.join(simu_path, 'prices', 'sim_price_'+mon)).display_data()
    k = 0
    j = 0
    for folder in folders_plot:
        case = folder+'_'+mon
        color = cmap(float(j)/int(len(folders_plot)/2))
        
        index = dr_diff[case].resample('10S').ffill().index
        values = dr_diff[case].resample('10S').ffill()[index[0]:index[-1]-datetime.timedelta(minutes = 6)].values/1000
        drdiff_plot = dr_diff[case].resample('10S').ffill()[index[0]:index[-1]-datetime.timedelta(minutes = 6)]
        index = dr_diff[case].resample('10S').ffill()[index[0]:index[-1]-datetime.timedelta(minutes = 6)].index
        #print(dr_diff_wols[case])
        
        if pricing[k] == 'Static':
            ax = axarr[i][0]
        if pricing[k] == 'Dynamic':
            ax = axarr[i][1]
            
        ax.plot(index,values, '-', label = structures_plot[k] if (i == 0 and k%2 == 0) else "", color=color, linewidth=1.5)
        
        fill_idx =  dr_diff[case].resample('10S').ffill()[DRstart:DRend].index
        fill_values = dr_diff[case].resample('10S').ffill()[DRstart:DRend].values/1000
        ax.fill_between(fill_idx, 0, fill_values, where=fill_values<0, facecolor=color, alpha=0.4)
        
        load_shape = pd.Series(float(0.0),index=index)
        ax.plot(index, load_shape.values, '-' ,color='black', linewidth = 1.0)
        
        ref1 = ref_profile_aggr[case][index[0]:index[-1]].resample('10S').ffill()
        ref2 = ref_projections[case][index[0]:index[-1]].resample('10S').ffill().sum(axis=1)
        
        for t in index:
            if t >= DRstart and t <= DRend:
                load_shape[t] = float((ref2[t]-ref1[t]))/float(1000.0)
                
                if (ref2[t]-ref1[t]) < 10 and drdiff_plot[t] < 0:
                    drdiff_plot[t] = 0
                    
        if mon == 'mar' and structure[k] =='Decentralised':
            print(load_shape[DRstart:DRend])
        
        if structure[k] == 'Decentralised':
            ax.plot(load_shape[DRstart:DRend].index, load_shape[DRstart:DRend].values, '--', color='black', linewidth = 1.0)
        
        ax.set_xlim(index[0]-datetime.timedelta(minutes = 1),index[-1])
        ax.grid(b=True, which='major', axis='both')
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,30], interval = 1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=10)
        
        ax.axvline(x=DR_call_start, ymin=0, ymax=1, linestyle = '-.', color='red', linewidth=1)
        ax.axvline(x=DRstart, ymin=0, ymax=1, linestyle = '-.', color='blue', linewidth=1)
        ax.axvline(x=DRend, ymin=0, ymax=1, linestyle = '-.', color='blue', linewidth=1)
        
        if i == 1 and k==2:
            ax.text(DR_call_start+datetime.timedelta(minutes = 1), -0.5, 'DR call',fontsize=12, color='red')
            ax.text(DRstart + datetime.timedelta(minutes = 3), 1.25, 'DR period',fontsize=12, color='blue')
            
        if k%2 != 0:
            j = j+1
        k=k+1
    i = i+1

f.suptitle('Differences to Requested Demand Profiles', y=0.98, fontsize=16)    

axarr[0][0].set_title("Static Pricing\nWinter 7.1.",fontsize=14)
axarr[0][1].set_title("Dynamic pricing\nWinter 7.1.",fontsize=14)   
axarr[1][0].set_title("Spring 1.3.",fontsize=14)
axarr[1][1].set_title("Spring 1.3.",fontsize=14)
axarr[2][0].set_title("Autumn 20.11.",fontsize=14)
axarr[2][1].set_title("Autumn 20.11.",fontsize=14)  

axarr[1][0].set_ylabel('Difference to request [kW]', fontsize=14)
axarr[2][0].set_xlabel("Time",fontsize=14)

handles,labels = [],[]
for ax in axarr[0][:]:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

axarr[2][0].legend(handles,labels, bbox_to_anchor=(1.0, -0.25), loc = 9, ncol=2, fontsize=14)

plt.subplots_adjust(hspace = 0.35, wspace = 0.2)    

f.savefig(os.path.join(simu_path, analysis_folder, "drdiff.png"), bbox_inches="tight")
f.savefig(os.path.join(simu_path, analysis_folder, "drdiff.pdf"), bbox_inches="tight")
plt.clf()

exit()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Differences to MPC projections
#fig = plt.figure(figsize=(11.69,8.27))
f, axarr = plt.subplots(nrows = 3, ncols = 1, sharex = False, sharey=False, figsize=(10, 7))

folders_plot = [folders[1]]
structures_plot = [structure[1]]

i = 0 
for mon in mons:
    
    if mon == 'jan':
        ff_date = '1/7/2017'
    if mon == 'mar':
        ff_date = '3/1/2017'
    if mon == 'nov':
        ff_date = '11/20/2017'
    
    DRstart = datetime.datetime.strptime(ff_date + ' 17:30:00', '%m/%d/%Y %H:%M:%S') # hour to start DR - ramp down 30 mins before
    DRend = datetime.datetime.strptime(ff_date + ' 18:30:00', '%m/%d/%Y %H:%M:%S') # hour to end DR - ramp 30 mins later
    DR_call_start = datetime.datetime.strptime(ff_date + ' 17:00:00', '%m/%d/%Y %H:%M:%S') # 
    
    #price = load_namespace(os.path.join(simu_path, 'prices', 'sim_price_'+mon)).display_data()
    k = 0
    j = 0
    for folder in folders_plot:
        case = folder+'_'+mon
        color = cmap(j)
        
        index = dr_diff[case].resample('10S').ffill().index
      
        if pricing[k] == 'Static':
            ax = axarr[i]
        if pricing[k] == 'Dynamic':
            ax = axarr[i]
            
        drdiff_plot = dr_diff[case].resample('10S').ffill()[index[0]:index[-1]-datetime.timedelta(minutes = 6)]
        
        index = dr_diff[case].resample('10S').ffill()[index[0]:index[-1]-datetime.timedelta(minutes = 6)].index
        #print(dr_diff_wols[case])

        load_shape = pd.Series(float(0.0),index=index)
        ax.plot(index, load_shape.values, '-' ,color='black', linewidth = 1.0)
        
        ref1 = ref_profile_aggr[case][index[0]:index[-1]].resample('10S').ffill()
        ref2 = ref_projections[case][index[0]:index[-1]].resample('10S').ffill().sum(axis=1)
        
        for t in index:
            if t >= DRstart and t <= DRend:
                load_shape[t] = float((ref2[t]-ref1[t]))/float(1000.0)
                
                if (ref2[t]-ref1[t]) < 10 and drdiff_plot[t] < 0:
                    drdiff_plot[t] = 0
                    
        #if mon == 'mar' and structure[k] =='Decentralised':
        #    print(load_shape[DRstart:DRend])
           
        values = drdiff_plot.resample('10S').ffill().values/1000
        
        ax.plot(index,values, '-', label = structures_plot[k] if (i == 0 and k%2 == 0) else "", color=color, linewidth=1.5)
        
        ax.plot(load_shape[DRstart:DRend].index, load_shape[DRstart:DRend].values, '--', color='black', linewidth = 1.0)
        
        fill_idx =  drdiff_plot.resample('10S').ffill()[DRstart:DRend].index
        fill_values = drdiff_plot.resample('10S').ffill()[DRstart:DRend].values/1000
        ax.fill_between(fill_idx, 0, fill_values, where=fill_values<0, facecolor=color, alpha=0.4)
        
        ax.set_xlim(index[0]-datetime.timedelta(minutes = 1),index[-1])
        ax.grid(b=True, which='major', axis='both')
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,10,20,30,40,50], interval = 1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.tick_params(axis='both', which='minor', labelsize=6)
        
        ax.axvline(x=DR_call_start, ymin=0, ymax=1, linestyle = '-.', color='red', linewidth=1)
        ax.axvline(x=DRstart, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1)
        ax.axvline(x=DRend, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1)
        
        if i == 1 and k==2:
            ax.text(DR_call_start+datetime.timedelta(minutes = 1), -0.95, r'Demand response call',fontsize=10, color='red')
            ax.text(DRstart + +datetime.timedelta(minutes = 3), 1.0, r'Demand response period',fontsize=10, color='green')
        
        if k%2 != 0:
            j = j+1
        k=k+1
    i = i+1

f.suptitle('Difference between Realised and Requested Demand Baseline\nDecentralised MPC, Afternoons, Dynamic Pricing', y=0.98, fontsize=15)    

#axarr[0][0].set_title("Static Pricing\nWinter 7.1.",fontsize=12)
axarr[0].set_title("Winter 7.1.",fontsize=12)   
#axarr[1][0].set_title("Spring 1.3.",fontsize=12)
axarr[1].set_title("Spring 1.3.",fontsize=12)
axarr[2].set_title("Autumn 20.11.",fontsize=12)
#axarr[2][1].set_title("Autumn 20.11.",fontsize=12)  

axarr[1].set_ylabel('Difference to MPC projection [kW]', fontsize=12)
axarr[2].set_xlabel("Time",fontsize=12)

handles,labels = [],[]
for h,l in zip(*ax.get_legend_handles_labels()):
    handles.append(h)
    labels.append(l)

#axarr[2].legend(handles,labels, bbox_to_anchor=(1.0, -0.25), loc = 9, ncol=2, fontsize=14)

plt.subplots_adjust(hspace = 0.3, wspace = 0.15)    

f.savefig(os.path.join(simu_path, analysis_folder, "drdiff_dyn_paper.png"), bbox_inches="tight")
f.savefig(os.path.join(simu_path, analysis_folder, "drdiff_dyn_paper.pdf"), bbox_inches="tight")
plt.clf()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Differences to MPC projections
#fig = plt.figure(figsize=(11.69,8.27))
f, axarr = plt.subplots(nrows = 2, ncols = 1, sharex = 'row', sharey=False, figsize=(10, 7))

folders_plot = [folders[0]]
structures_plot = [structure[0]]
bldg_plotlist = [bldg_list[3]]
mons_plot = ['nov']

i = 0 
for mon in mons_plot:
    
    if mon == 'jan':
        ff_date = '1/7/2017'
    if mon == 'mar':
        ff_date = '3/1/2017'
    if mon == 'nov':
        ff_date = '11/20/2017'
    
    DRstart = datetime.datetime.strptime(ff_date + ' 17:30:00', '%m/%d/%Y %H:%M:%S') # hour to start DR - ramp down 30 mins before
    DRend = datetime.datetime.strptime(ff_date + ' 18:30:00', '%m/%d/%Y %H:%M:%S') # hour to end DR - ramp 30 mins later
    DR_call_start = datetime.datetime.strptime(ff_date + ' 17:00:00', '%m/%d/%Y %H:%M:%S') # 
    
    ref_time_str = ff_date + ' 16:55:00'
    
    #price = load_namespace(os.path.join(simu_path, 'prices', 'sim_price_'+mon)).display_data()
    k = 0
    j = 0
    for folder in folders_plot:
        case = folder+'_'+mon        
        index = emutemps[case].index
        
        l = 0
        for bldg in bldg_plotlist:
            bldg_key = bldg+'_R2CW_HP'
            color = cmap(float(l)/int(len(bldg_plotlist)))
            
            bldg_key = bldg+'_R2CW_HP'
            
            ax = axarr[0]
            
            index = real_cont[case][bldg_key].resample('10S').ffill().index
            index1 = index
            values = real_cont[case][bldg_key].resample('10S').ffill().values
            ax.plot(index,values, '-', color=color, linewidth=2)
            
            #ax.set_xlim(index[0],index[-1])
            
            index = ref_profile[case][bldg_key].resample('10S').ffill().index
            values = ref_profile[case][bldg_key].resample('10S').ffill().values
            ax.plot(index,values, '--', color=color, linewidth=2)
            
            ax.set_ylabel('Power [W]', fontsize=12)
            
            ax.grid(b=True, which='major', axis='both')
            #ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,15,45], interval = 1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.tick_params(axis='both', which='minor', labelsize=8)
            
            ax.axvline(x=DR_call_start, ymin=0, ymax=1, linestyle = '-.', color='red', linewidth=1)
            ax.axvline(x=DRstart, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1)
            ax.axvline(x=DRend, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1)
            
            if structures_plot[k] == 'Centralised':
                values = emutemps[case][bldg_key].values 
                line_type = '--'
            else:
                values = emutemps[case][bldg_key].values-273.15
                line_type = '-'
            
            ax = axarr[1]
            
            index = emutemps[case][bldg_key].index
            index2 = index
            ax.plot(index,values, '-', label = 'Realised' if i==0 else "", color=color, linewidth=2)
            
            axarr[0].set_xlim(index2[0],index1[-1])
            axarr[1].set_xlim(index2[0],index1[-1])
            
            if structures_plot[k] == 'Centralised':
                index = mpctemps[case][ref_time_str]['mpc_model.'+bldg_key+'.TAir'].index
                values = mpctemps[case][ref_time_str]['mpc_model.'+bldg_key+'.TAir'].values -273.15
                line_type = '--'
            else:
                index = mpctemps[case]['MinEne_0-10'][ref_time_str][bldg_key].index
                values = mpctemps[case]['MinEne_0-10'][ref_time_str][bldg_key].values -273.15
                line_type = '-'

            ax.plot(index,values, '--', label = 'Projected', color=color, linewidth=2)
            
            ax.set_ylabel('Temperature [C]', fontsize=12)
            
            ax.grid(b=True, which='major', axis='both')
            #ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,15,45], interval = 1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.tick_params(axis='both', which='minor', labelsize=8)
            
            ax.axvline(x=DR_call_start, ymin=0, ymax=1, linestyle = '-.', color='red', linewidth=1)
            ax.axvline(x=DRstart, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1)
            ax.axvline(x=DRend, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1)
            
            ax.text(DR_call_start+datetime.timedelta(minutes = 1), 19.5, r'Demand response call',fontsize=10, color='red')
            ax.text(DRstart + +datetime.timedelta(minutes = 10), 19, r'Demand response period',fontsize=10, color='green')
            
            l += 1
    i = i+1

f.suptitle('Decentralised MPC - Autumn 20.11. - Static Pricing - SemiDetached_7', y=0.98, fontsize=16)   

axarr[0].set_title("Power profiles",fontsize=12)
axarr[1].set_title("Temperature profiles",fontsize=12)   

axarr[1].set_xlabel("Time",fontsize=12)

handles,labels = [],[]
for ax in axarr[:]:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

axarr[1].legend(handles,labels, bbox_to_anchor=(0.5, -0.25), loc = 9, ncol=2, fontsize=14)

plt.subplots_adjust(hspace = 0.3, wspace = 0.15)    

f.savefig(os.path.join(simu_path, analysis_folder, "temps_power_Semi7.png"), bbox_inches="tight")
f.savefig(os.path.join(simu_path, analysis_folder, "temps_power_Semi7.pdf"), bbox_inches="tight")
plt.clf()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Projected Costs - Dynamic Pricing
#fig = plt.figure(figsize=(11.69,8.27))    
f, axarr = plt.subplots(nrows = 3, sharey=True, figsize=(11.69,8.27))
folders_plot = [folders[0],folders[2]]
structures_plot = [structure[0], structure[2]]
#bldg = 'SemiDetached_3'

i = 0
for mon in mons:
    j = 0
    
    if mon == 'jan':
        ff_date = '1/7/2017'
    if mon == 'mar':
        ff_date = '3/1/2017'
    if mon == 'nov':
        ff_date = '11/20/2017'
    
    DRstart = datetime.datetime.strptime(ff_date + ' 17:30:00', '%m/%d/%Y %H:%M:%S') # hour to start DR - ramp down 30 mins before
    DRend = datetime.datetime.strptime(ff_date + ' 18:30:00', '%m/%d/%Y %H:%M:%S') # hour to end DR - ramp 30 mins later
    DR_call_start = datetime.datetime.strptime(ff_date + ' 17:00:00', '%m/%d/%Y %H:%M:%S') # 
    
    #price = load_namespace(os.path.join(simu_path, 'prices', 'sim_price_'+mon)).display_data()
    for folder in folders_plot:
        case = folder+'_'+mon
        color = cmap(float(j)/len(folders_plot))
        
        bldg_plotlist = [bldg_list[0], bldg_list[4], bldg_list[8]]
        bldg_key_list = []
        
        for item in bldg_plotlist:
            bldg_key_list.append(item+'_R2CW_HP')
        
        index = emutemps[case].index
        if structures_plot[j] == 'Centralised':
            values = emutemps[case][bldg_key_list].values
        else:
            values = emutemps[case][bldg_key_list].values -273.15
        
        axarr[i].plot(index,values, '-', label = structures_plot[j] if i==0 else "", color=color, linewidth=2)
        
        j += 1
    
    axarr[i].grid(b=True, which='major', axis='both')
    axarr[i].tick_params(axis='both', which='major', labelsize=10)
    axarr[i].tick_params(axis='both', which='minor', labelsize=10)
    axarr[i].set_xlim(emutemps[case].index[0], emutemps[case].index[-1])
    axarr[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axarr[i].xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    
    axarr[i].axvline(x=DR_call_start, ymin=0, ymax=1, linestyle = '-.', color='red', linewidth=1.5)

    axarr[i].axvline(x=DRstart, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1.5)
    
    axarr[i].axvline(x=DRend, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1.5)
    
    
    if i == 1:
        axarr[i].text(DR_call_start+datetime.timedelta(minutes = 1), 20, r'Demand response call',fontsize=14, color='red')
        axarr[i].text(DRstart + +datetime.timedelta(minutes = 10), 18, r'Demand response period',fontsize=14, color='green')
    
    i=i+1

f.suptitle('Emulated indoor temperature - Static Pricing: ' + bldg,y=0.96,fontsize=20)

axarr[0].set_title("Winter 7.1.",fontsize=18)
axarr[1].set_title("Spring 1.3.",fontsize=18)
axarr[2].set_title("Autumn 20.11.",fontsize=18)
axarr[2].set_xlabel("Time",fontsize=16)

axarr[1].set_ylabel('Temperature [C]', fontsize=16)

#axarr[0].text(index[8], 0.1, r'Demand response period',fontsize=10, color='green')
#axarr[0].text(index[4], 0.1, r'Demand response call',fontsize=10, color='red')

handles,labels = [],[]
for ax in axarr:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

lgd = plt.legend(handles,labels, bbox_to_anchor=(0.5, -0.3), loc = 9, ncol=4, fontsize=16)

f.subplots_adjust(hspace=0.35)

f.savefig(os.path.join(simu_path, analysis_folder, "emutemps_dr_stat_"+bldg+"_paper.png"), bbox_inches="tight")
f.savefig(os.path.join(simu_path, analysis_folder, "emutemps_dr_stat_"+bldg+"_paper.pdf"), bbox_inches="tight")
plt.close()

#exit()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Consumption profiles
#fig = plt.figure(figsize=(11.69,8.27))
f, axarr = plt.subplots(nrows = 3, sharey=True, figsize=(11.69,8.27))
folders_plot = [folders[0],folders[2]]
structures_plot = [structure[0], structure[2]]
i = 0
for mon in mons:
    j = 0
    
    if mon == 'jan':
        ff_date = '1/7/2017'
    if mon == 'mar':
        ff_date = '3/1/2017'
    if mon == 'nov':
        ff_date = '11/20/2017'
    
    DRstart = datetime.datetime.strptime(ff_date + ' 17:30:00', '%m/%d/%Y %H:%M:%S') # hour to start DR - ramp down 30 mins before
    DRend = datetime.datetime.strptime(ff_date + ' 18:30:00', '%m/%d/%Y %H:%M:%S') # hour to end DR - ramp 30 mins later
    DR_call_start = datetime.datetime.strptime(ff_date + ' 17:00:00', '%m/%d/%Y %H:%M:%S') # 
    
    #price = load_namespace(os.path.join(simu_path, 'prices', 'sim_price_'+mon)).display_data()
    for folder in folders_plot:
        case = folder+'_'+mon
        color = cmap(float(j)/len(folders_plot))
        index = real_cont_aggr[case].resample('10S').ffill().index
        values = real_cont_aggr[case].resample('10S').ffill().values/1000
        axarr[i].plot(index,values, '-', label = structures_plot[j] if i==0 else "", color=color, linewidth=2)
        
        index = ref_profile_aggr[case].resample('10S').ffill().index
        values = ref_profile_aggr[case].resample('10S').ffill().values/1000
        axarr[i].plot(index,values, '--', label = structures_plot[j]+'-ref' if i==0 else "", color=color, linewidth=2)
        j = j+1
    
    axarr[i].grid(b=True, which='major', axis='both')
    axarr[i].tick_params(axis='both', which='major', labelsize=10)
    axarr[i].tick_params(axis='both', which='minor', labelsize=10)
    axarr[i].set_xlim(real_cont_aggr[case].index[0],real_cont_aggr[case].index[-1])
    axarr[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axarr[i].xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    
    axarr[i].axvline(x=DR_call_start, ymin=0, ymax=1, linestyle = '-.', color='red', linewidth=1.5)

    axarr[i].axvline(x=DRstart, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1.5)
    
    axarr[i].axvline(x=DRend, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1.5)
    
    if i == 1:
        axarr[i].text(DR_call_start+datetime.timedelta(minutes = 1), 12, r'Demand response call',fontsize=14, color='red')
        axarr[i].text(DRstart + +datetime.timedelta(minutes = 10), 12, r'Demand response period',fontsize=14, color='green')
    
    i=i+1

f.suptitle('Demand and Reference Profiles - Static Electricity Pricing',y=0.96,fontsize=20)

axarr[0].set_title("Winter 7.1.",fontsize=18)
axarr[1].set_title("Spring 1.3.",fontsize=18)
axarr[2].set_title("Autumn 20.11.",fontsize=18)
axarr[2].set_xlabel("Time",fontsize=16)

axarr[1].set_ylabel('Power [kW]', fontsize=16)

#axarr[0].text(index[8], 0.1, r'Demand response period',fontsize=10, color='green')
#axarr[0].text(index[4], 0.1, r'Demand response call',fontsize=10, color='red')

handles,labels = [],[]
for ax in axarr:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

lgd = plt.legend(handles,labels, bbox_to_anchor=(0.5, -0.3), loc = 9, ncol=4, fontsize=16)

f.subplots_adjust(hspace=0.35)

f.savefig(os.path.join(simu_path, analysis_folder, "demand_plots_dr_stat.png"), bbox_inches="tight")
f.savefig(os.path.join(simu_path, analysis_folder, "demand_plots_dr_stat.pdf"), bbox_inches="tight")
plt.clf()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Consumption profiles
#fig = plt.figure(figsize=(11.69,8.27))
f, axarr = plt.subplots(nrows = 3, sharey=True, figsize=(11.69,8.27))
folders_plot = [folders[1],folders[3]]
structures_plot = [structure[1], structure[3]]
i = 0
for mon in mons:
    j = 0
    
    if mon == 'jan':
        ff_date = '1/7/2017'
    if mon == 'mar':
        ff_date = '3/1/2017'
    if mon == 'nov':
        ff_date = '11/20/2017'
    
    DRstart = datetime.datetime.strptime(ff_date + ' 17:30:00', '%m/%d/%Y %H:%M:%S') # hour to start DR - ramp down 30 mins before
    DRend = datetime.datetime.strptime(ff_date + ' 18:30:00', '%m/%d/%Y %H:%M:%S') # hour to end DR - ramp 30 mins later
    DR_call_start = datetime.datetime.strptime(ff_date + ' 17:00:00', '%m/%d/%Y %H:%M:%S') # 
    
    #price = load_namespace(os.path.join(simu_path, 'prices', 'sim_price_'+mon)).display_data()
    for folder in folders_plot:
        case = folder+'_'+mon
        color = cmap(float(j)/len(folders_plot))
        index = real_cont_aggr[case].resample(str(step)+'S').mean().resample('10S').ffill().index
        values = real_cont_aggr[case].resample(str(step)+'S').mean().resample('10S').ffill().values/1000
        axarr[i].plot(index,values, '-', label = structures_plot[j] if i==0 else "", color=color, linewidth=1.5)
        
        index = ref_profile_aggr[case].resample(str(step)+'S').mean().resample('10S').ffill().index
        values = ref_profile_aggr[case].resample(str(step)+'S').mean().resample('10S').ffill().values/1000
        axarr[i].plot(index,values, '--', label = structures_plot[j]+'-ref' if i==0 else "", color=color, linewidth=1.5)
        j = j+1
    
    axarr[i].grid(b=True, which='major', axis='both')
    axarr[i].tick_params(axis='both', which='major', labelsize=10)
    axarr[i].tick_params(axis='both', which='minor', labelsize=10)
    axarr[i].set_xlim(real_cont_aggr[case].index[0],real_cont_aggr[case].index[-1])
    axarr[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axarr[i].xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    
    axarr[i].axvline(x=DR_call_start, ymin=0, ymax=1, linestyle = '-.', color='red', linewidth=1)

    axarr[i].axvline(x=DRstart, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1)
    
    axarr[i].axvline(x=DRend, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1)
    
    if i == 1:
        axarr[i].text(DR_call_start+datetime.timedelta(minutes = 1), 20, r'Demand response call',fontsize=14, color='red')
        axarr[i].text(DRstart + +datetime.timedelta(minutes = 10), 12, r'Demand response period',fontsize=14, color='green')
    
    i=i+1
    
f.suptitle('Demand and Reference Profiles - Dynamic Electricity Pricing', y=0.96, fontsize=20)

axarr[0].set_title("Winter 7.1.",fontsize=18)
axarr[1].set_title("Spring 1.3.",fontsize=18)
axarr[2].set_title("Autumn 20.11.",fontsize=18)
axarr[2].set_xlabel("Time",fontsize=16)

axarr[1].set_ylabel('Power [kW]', fontsize=16)

#axarr[0].text(index[8], 0.1, r'Demand response period',fontsize=10, color='green')
#axarr[0].text(index[4], 0.1, r'Demand response call',fontsize=10, color='red')

handles,labels = [],[]
for ax in axarr:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

lgd = plt.legend(handles,labels, bbox_to_anchor=(0.5, -0.3), loc = 9, ncol=4, fontsize=16)

f.subplots_adjust(hspace=0.35)
f.savefig(os.path.join(simu_path, analysis_folder, "demand_plots_dr_dyn.png"), bbox_inches="tight")

f.savefig(os.path.join(simu_path, analysis_folder, "demand_plots_dr_dyn.pdf"), bbox_inches="tight")
plt.clf()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extracted flexibility - dynamic pricing
#fig = plt.figure(figsize=(11.69,8.27))
f, axarr = plt.subplots(nrows = 3, ncols = 2, sharex = 'row', sharey=True, figsize=(10, 7))

folders_plot = [folders[1],folders[3]]
structures_plot = [structure[1], structure[3]]

i = 0 
for mon in mons:
    j = 0
    
    if mon == 'jan':
        ff_date = '1/7/2017'
    if mon == 'mar':
        ff_date = '3/1/2017'
    if mon == 'nov':
        ff_date = '11/20/2017'
    
    DRstart = datetime.datetime.strptime(ff_date + ' 17:30:00', '%m/%d/%Y %H:%M:%S') # hour to start DR - ramp down 30 mins before
    DRend = datetime.datetime.strptime(ff_date + ' 18:30:00', '%m/%d/%Y %H:%M:%S') # hour to end DR - ramp 30 mins later
    DR_call_start = datetime.datetime.strptime(ff_date + ' 17:00:00', '%m/%d/%Y %H:%M:%S') # 
    
    #price = load_namespace(os.path.join(simu_path, 'prices', 'sim_price_'+mon)).display_data()
    k = 0
    for folder in folders_plot:
        case = folder+'_'+mon
        color = cmap(float(k)/len(folders_plot))
        
        index = PI_diff_aggr[case].resample('10S').ffill().index
        values = PI_diff_aggr[case].resample('10S').ffill().values/1000
        ax = axarr[i][0]
        
        ax.plot(index,values, '-', label = structures_plot[k], color=color, linewidth=1.5)
        ax.fill_between(index, 0, values, where=values<0, facecolor=color, alpha=0.4)
        
        ax.set_xlim(index[0],index[-1])
        ax.grid(b=True, which='major', axis='both')
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8) 
        
        index = enemin_diff_aggr[case].resample('10S').ffill().index
        values = enemin_diff_aggr[case].resample('10S').ffill().values/1000
        ax = axarr[i][1]
        
        ax.plot(index,values, '-', color=color, linewidth=1.5)
        ax.fill_between(index, 0, values, where=values<0, facecolor=color, alpha=0.4)
        
        ax.set_xlim(index[0],index[-1])
        ax.grid(b=True, which='major', axis='both')
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8) 
        
        k=k+1
    i = i+1

f.suptitle('Differences to RBC and Energy Minimisation - Dynamic Pricing', y=0.98, fontsize=16) 

axarr[0][0].set_title("Rule-based Control\nWinter 7.1.",fontsize=12)
axarr[0][1].set_title("Energy Minimisation\nWinter 7.1.",fontsize=12)   
axarr[1][0].set_title("Spring 1.3.",fontsize=12)
axarr[1][1].set_title("Spring 1.3.",fontsize=12)
axarr[2][0].set_title("Autumn 20.11.",fontsize=12)
axarr[2][1].set_title("Autumn 20.11.",fontsize=12)  
axarr[1][0].set_ylabel('Difference to reference [kW]', fontsize=12)
axarr[2][0].set_xlabel("Time",fontsize=12)

handles,labels = [],[]
for ax in axarr[0][:]:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

axarr[2][0].legend(handles,labels, bbox_to_anchor=(1.0, -0.25), loc = 9, ncol=2, fontsize=14)

plt.subplots_adjust(hspace = 0.3, wspace = 0.15)    

f.savefig(os.path.join(simu_path, analysis_folder, "flex_profiles_dyn.png"), bbox_inches="tight")
f.savefig(os.path.join(simu_path, analysis_folder, "flex_profiles_dyn.pdf"), bbox_inches="tight")
plt.clf()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extracted flexibility - static pricing
#fig = plt.figure(figsize=(11.69,8.27))
f, axarr = plt.subplots(nrows = 3, ncols = 2, sharex = 'row', sharey=True, figsize=(10, 7))

folders_plot = [folders[0],folders[2]]
structures_plot = [structure[0], structure[2]]

i = 0 
for mon in mons:
    j = 0
    
    if mon == 'jan':
        ff_date = '1/7/2017'
    if mon == 'mar':
        ff_date = '3/1/2017'
    if mon == 'nov':
        ff_date = '11/20/2017'
    
    DRstart = datetime.datetime.strptime(ff_date + ' 17:30:00', '%m/%d/%Y %H:%M:%S') # hour to start DR - ramp down 30 mins before
    DRend = datetime.datetime.strptime(ff_date + ' 18:30:00', '%m/%d/%Y %H:%M:%S') # hour to end DR - ramp 30 mins later
    DR_call_start = datetime.datetime.strptime(ff_date + ' 17:00:00', '%m/%d/%Y %H:%M:%S') # 
    
    #price = load_namespace(os.path.join(simu_path, 'prices', 'sim_price_'+mon)).display_data()
    k = 0
    for folder in folders_plot:
        case = folder+'_'+mon
        color = cmap(float(k)/len(folders_plot))
        
        index = PI_diff_aggr[case].resample('10S').ffill().index
        values = PI_diff_aggr[case].resample('10S').ffill().values/1000
        ax = axarr[i][0]
        
        ax.plot(index,values, '-', label = structures_plot[k], color=color, linewidth=1.5)
        ax.fill_between(index, 0, values, where=values<0, facecolor=color, alpha=0.4)
        
        ax.set_xlim(index[0],index[-1])
        ax.grid(b=True, which='major', axis='both')
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8) 
        
        index = enemin_diff_aggr[case].resample('10S').ffill().index
        values = enemin_diff_aggr[case].resample('10S').ffill().values/1000
        ax = axarr[i][1]
        
        ax.plot(index,values, '-', color=color, linewidth=1.5)
        ax.fill_between(index, 0, values, where=values<0, facecolor=color, alpha=0.4)
        
        ax.set_xlim(index[0],index[-1])
        ax.grid(b=True, which='major', axis='both')
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8) 
        
        k=k+1
    i = i+1

f.suptitle('Differences to RBC and Energy Minimisation - Static Pricing', y=0.98, fontsize=16)  

axarr[0][0].set_title("Rule-based Control\nWinter 7.1.",fontsize=12)
axarr[0][1].set_title("Energy Minimisation\nWinter 7.1.",fontsize=12)   
axarr[1][0].set_title("Spring 1.3.",fontsize=12)
axarr[1][1].set_title("Spring 1.3.",fontsize=12)
axarr[2][0].set_title("Autumn 20.11.",fontsize=12)
axarr[2][1].set_title("Autumn 20.11.",fontsize=12)  

axarr[1][0].set_ylabel('Difference to reference [kW]', fontsize=12)
axarr[2][0].set_xlabel("Time",fontsize=12)

handles,labels = [],[]
for ax in axarr[0][:]:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

axarr[2][0].legend(handles,labels, bbox_to_anchor=(1.0, -0.25), loc = 9, ncol=2, fontsize=14)

plt.subplots_adjust(hspace = 0.3, wspace = 0.15)    

f.savefig(os.path.join(simu_path, analysis_folder, "flex_profiles_stat.png"), bbox_inches="tight")
f.savefig(os.path.join(simu_path, analysis_folder, "flex_profiles_stat.pdf"), bbox_inches="tight")
plt.clf()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Extracted flexibility - ALL
#fig = plt.figure(figsize=(11.69,8.27))
f, axarr = plt.subplots(nrows = 3, ncols = 4, sharex = 'row', sharey=True, figsize=(10, 7))

folders_plot = folders
structures_plot = structure

i = 0 
for mon in mons:
    j = 0
    
    if mon == 'jan':
        ff_date = '1/7/2017'
    if mon == 'mar':
        ff_date = '3/1/2017'
    if mon == 'nov':
        ff_date = '11/20/2017'
    
    DRstart = datetime.datetime.strptime(ff_date + ' 17:30:00', '%m/%d/%Y %H:%M:%S') # hour to start DR - ramp down 30 mins before
    DRend = datetime.datetime.strptime(ff_date + ' 18:30:00', '%m/%d/%Y %H:%M:%S') # hour to end DR - ramp 30 mins later
    DR_call_start = datetime.datetime.strptime(ff_date + ' 17:00:00', '%m/%d/%Y %H:%M:%S') # 
    
    #price = load_namespace(os.path.join(simu_path, 'prices', 'sim_price_'+mon)).display_data()
    k = 0
    j = 0
    for folder in folders_plot:
        case = folder+'_'+mon
        color = cmap(float(j)/(len(folders_plot)/2))
        
        if k % 2 == 0:      
            index = PI_diff_aggr[case].resample('10S').ffill().index
            values = PI_diff_aggr[case].resample('10S').ffill().values/1000
            ax = axarr[i][0]
        
            ax.plot(index,values, '-', label = structures_plot[k], color=color, linewidth=1.5)
            ax.fill_between(index, 0, values, where=values<0, facecolor=color, alpha=0.4)
            
            ax.set_xlim(index[0],index[-1])
            ax.grid(b=True, which='major', axis='both')
            #ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
            ax.tick_params(axis='both', which='major', labelsize=8)
            #ax.tick_params(axis='both', which='minor', labelsize=8)    
            
            index = enemin_diff_aggr[case].resample('10S').ffill().index
            values = enemin_diff_aggr[case].resample('10S').ffill().values/1000
            ax = axarr[i][1]
            
            ax.plot(index,values, '-', color=color, linewidth=1.5)
            ax.fill_between(index, 0, values, where=values<0, facecolor=color, alpha=0.4)
            
            ax.set_xlim(index[0],index[-1])
            ax.grid(b=True, which='major', axis='both')
            #ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
            ax.tick_params(axis='both', which='major', labelsize=8)
            #ax.tick_params(axis='both', which='minor', labelsize=8)
            j = j+1
            if j > 1:
                j = 0
            
        else:   
            index = PI_diff_aggr[case].resample('10S').ffill().index
            values = PI_diff_aggr[case].resample('10S').ffill().values/1000
            ax = axarr[i][2]
        
            ax.plot(index,values, '-', color=color, linewidth=1.5)
            ax.fill_between(index, 0, values, where=values<0, facecolor=color, alpha=0.4)
            
            ax.set_xlim(index[0],index[-1])
            #ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
            ax.grid(b=True, which='major', axis='both')
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.tick_params(axis='both', which='minor', labelsize=8) 
            
            
            index = enemin_diff_aggr[case].resample('10S').ffill().index
            values = enemin_diff_aggr[case].resample('10S').ffill().values/1000
            ax = axarr[i][3]
            
            ax.plot(index,values, '-', color=color, linewidth=1.5)
            ax.fill_between(index, 0, values, where=values<0, facecolor=color, alpha=0.4)
            
            ax.set_xlim(index[0],index[-1])
            ax.grid(b=True, which='major', axis='both')
            #ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.tick_params(axis='both', which='minor', labelsize=8)     
            
        k=k+1
    i = i+1

f.suptitle('Differences to RBC and Energy Minimisation', y=0.98, fontsize=16)   

axarr[0][0].set_title("Winter 7.1. - Static Pricing\nRBC",fontsize=14)
axarr[0][1].set_title("Energy Min.",fontsize=14)
axarr[0][2].set_title("Dynamic Pricing\nRBC",fontsize=14)
axarr[0][3].set_title("Energy Min.",fontsize=14)
axarr[1][0].set_title("Spring 1.3.",fontsize=14)
axarr[2][0].set_title("Autumn 20.11.",fontsize=14)

axarr[1][0].set_ylabel('Difference to reference [kW]', fontsize=12)
axarr[2][0].set_xlabel("Time",fontsize=12)

handles,labels = [],[]
for ax in axarr[0][:]:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

axarr[2][0].legend(handles,labels, bbox_to_anchor=(2.4, -0.25), loc = 9, ncol=4, fontsize=14)

plt.subplots_adjust(hspace = 0.3, wspace = 0.25)    

f.savefig(os.path.join(simu_path, analysis_folder, "flex_profiles_all.png"), bbox_inches="tight")
f.savefig(os.path.join(simu_path, analysis_folder, "flex_profiles_all.pdf"), bbox_inches="tight")
plt.clf()



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Differences to MPC projections
#fig = plt.figure(figsize=(11.69,8.27))
f, axarr = plt.subplots(nrows = 3, ncols = 2, sharex = 'row', sharey=False, figsize=(10, 7))

folders_plot = folders
structures_plot = structure

i = 0 
for mon in mons:
    
    if mon == 'jan':
        ff_date = '1/7/2017'
    if mon == 'mar':
        ff_date = '3/1/2017'
    if mon == 'nov':
        ff_date = '11/20/2017'
    
    DRstart = datetime.datetime.strptime(ff_date + ' 17:30:00', '%m/%d/%Y %H:%M:%S') # hour to start DR - ramp down 30 mins before
    DRend = datetime.datetime.strptime(ff_date + ' 18:30:00', '%m/%d/%Y %H:%M:%S') # hour to end DR - ramp 30 mins later
    DR_call_start = datetime.datetime.strptime(ff_date + ' 17:00:00', '%m/%d/%Y %H:%M:%S') # 
    
    #price = load_namespace(os.path.join(simu_path, 'prices', 'sim_price_'+mon)).display_data()
    k = 0
    j = 0
    for folder in folders_plot:
        case = folder+'_'+mon
        color = cmap(float(j)/int(len(folders_plot)/2))
        
        index = dr_diff_wols[case].resample('10S').ffill().index
        values = dr_diff_wols[case].resample('10S').ffill().values/1000
        #print(dr_diff_wols[case])
        
        if pricing[k] == 'Static':
            ax = axarr[i][0]
        if pricing[k] == 'Dynamic':
            ax = axarr[i][1]
            
        ax.plot(index,values, '-', label = structures_plot[k] if (i == 0 and k%2 == 0) else "", color=color, linewidth=1.5)
        ax.fill_between(index, 0, values, where=values<0, facecolor=color, alpha=0.4)
        
        load_shape = pd.Series(0,index=index)
        ax.plot(index, load_shape.values, '-' ,color='black', linewidth = 1.0)
        
        for t in load_shape.index:
            if t >= DRstart and t <= DRend:
                load_shape[t] = -2
        ax.plot(index, load_shape.values, '--', color='black', linewidth = 1.0)
        
        ax.set_xlim(index[0]-datetime.timedelta(minutes = 1),index[-1]+datetime.timedelta(minutes = 1))
        ax.grid(b=True, which='major', axis='both')
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,10,20,30,40,50], interval = 1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        
        ax.axvline(x=DR_call_start, ymin=0, ymax=1, linestyle = '-.', color='red', linewidth=1)
        ax.axvline(x=DRstart, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1)
        ax.axvline(x=DRend, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1)
        
        if i == 1 and k==2:
            ax.text(DR_call_start+datetime.timedelta(minutes = 1), -1, r'Demand response call',fontsize=10, color='red')
            ax.text(DRstart + +datetime.timedelta(minutes = 10), 1.0, r'Demand response period',fontsize=10, color='green')
        
        if k%2 != 0:
            j = j+1
        k=k+1
    i = i+1

f.suptitle('Difference between Realised and Projected Demand', y=0.98, fontsize=16)   

axarr[0][0].set_title("Static Pricing\nWinter 7.1.",fontsize=12)
axarr[0][1].set_title("Dynamic pricing\nWinter 7.1.",fontsize=12)   
axarr[1][0].set_title("Spring 1.3.",fontsize=12)
axarr[1][1].set_title("Spring 1.3.",fontsize=12)
axarr[2][0].set_title("Autumn 20.11.",fontsize=12)
axarr[2][1].set_title("Autumn 20.11.",fontsize=12)  

axarr[1][0].set_ylabel('Difference to MPC projection [kW]', fontsize=12)
axarr[2][0].set_xlabel("Time",fontsize=12)

handles,labels = [],[]
for ax in axarr[0][:]:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

axarr[2][0].legend(handles,labels, bbox_to_anchor=(1.0, -0.25), loc = 9, ncol=2, fontsize=14)

plt.subplots_adjust(hspace = 0.3, wspace = 0.15)    

f.savefig(os.path.join(simu_path, analysis_folder, "drdiff_wols.png"), bbox_inches="tight")
f.savefig(os.path.join(simu_path, analysis_folder, "drdiff_wols.pdf"), bbox_inches="tight")
plt.clf()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Cost Projections - Dynamic Pricing
#fig = plt.figure(figsize=(11.69,8.27))
f, axarr = plt.subplots(nrows = 3, sharey=True, figsize=(11.69,8.27))
folders_plot = [folders[1],folders[3]]
structures_plot = [structure[1], structure[3]]
i = 0
for mon in mons:
    j = 0
    
    if mon == 'jan':
        ff_date = '1/7/2017'
    if mon == 'mar':
        ff_date = '3/1/2017'
    if mon == 'nov':
        ff_date = '11/20/2017'
    
    DRstart = datetime.datetime.strptime(ff_date + ' 17:30:00', '%m/%d/%Y %H:%M:%S') # hour to start DR - ramp down 30 mins before
    DRend = datetime.datetime.strptime(ff_date + ' 18:30:00', '%m/%d/%Y %H:%M:%S') # hour to end DR - ramp 30 mins later
    DR_call_start = datetime.datetime.strptime(ff_date + ' 17:00:00', '%m/%d/%Y %H:%M:%S') # 
    
    #price = load_namespace(os.path.join(simu_path, 'prices', 'sim_price_'+mon)).display_data()
    for folder in folders_plot:
        case = folder+'_'+mon
        color = cmap(float(j)/len(folders_plot))
        index = proj_cost_dfs[case].index
        values = proj_cost_dfs[case].values
        axarr[i].plot(index,values, '-', label = structures_plot[j] if i==0 else "", color=color, linewidth=2)
        j += 1
    
    axarr[i].grid(b=True, which='major', axis='both')
    axarr[i].tick_params(axis='both', which='major', labelsize=10)
    axarr[i].tick_params(axis='both', which='minor', labelsize=10)
    axarr[i].set_xlim(proj_cost_dfs[case].index[0],proj_cost_dfs[case].index[-1])
    axarr[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axarr[i].xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    
    axarr[i].axvline(x=DR_call_start, ymin=0, ymax=1, linestyle = '-.', color='red', linewidth=1.5)

    axarr[i].axvline(x=DRstart, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1.5)
    
    axarr[i].axvline(x=DRend, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1.5)
    
    if i == 1:
        axarr[i].text(DR_call_start+datetime.timedelta(minutes = 1), 1.2, r'Demand response call',fontsize=14, color='red')
        axarr[i].text(DRstart + +datetime.timedelta(minutes = 10), 1, r'Demand response period',fontsize=14, color='green')
    
    i=i+1

f.suptitle('Cost projections - Dynamic Electricity Pricing',y=0.96,fontsize=20)

axarr[0].set_title("Winter 7.1.",fontsize=18)
axarr[1].set_title("Spring 1.3.",fontsize=18)
axarr[2].set_title("Autumn 20.11.",fontsize=18)
axarr[2].set_xlabel("Time",fontsize=16)

axarr[1].set_ylabel('Price (pounds)', fontsize=16)

#axarr[0].text(index[8], 0.1, r'Demand response period',fontsize=10, color='green')
#axarr[0].text(index[4], 0.1, r'Demand response call',fontsize=10, color='red')

handles,labels = [],[]
for ax in axarr:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

lgd = plt.legend(handles,labels, bbox_to_anchor=(0.5, -0.3), loc = 9, ncol=4, fontsize=16)

f.subplots_adjust(hspace=0.35)

f.savefig(os.path.join(simu_path, analysis_folder, "projected_cost_plots_dr_dyn.png"), bbox_inches="tight")
f.savefig(os.path.join(simu_path, analysis_folder, "projected_cost_plots_dr_dyn.pdf"), bbox_inches="tight")
plt.clf()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# FlexCost Projections - Static Pricing
#fig = plt.figure(figsize=(11.69,8.27))
f, axarr = plt.subplots(nrows = 3, sharey=True, figsize=(11.69,8.27))
folders_plot = [folders[0],folders[2]]
structures_plot = [structure[0], structure[2]]
i = 0
for mon in mons:
    j = 0
    
    if mon == 'jan':
        ff_date = '1/7/2017'
    if mon == 'mar':
        ff_date = '3/1/2017'
    if mon == 'nov':
        ff_date = '11/20/2017'
    
    DRstart = datetime.datetime.strptime(ff_date + ' 17:30:00', '%m/%d/%Y %H:%M:%S') # hour to start DR - ramp down 30 mins before
    DRend = datetime.datetime.strptime(ff_date + ' 18:30:00', '%m/%d/%Y %H:%M:%S') # hour to end DR - ramp 30 mins later
    DR_call_start = datetime.datetime.strptime(ff_date + ' 17:00:00', '%m/%d/%Y %H:%M:%S') # 
    
    #price = load_namespace(os.path.join(simu_path, 'prices', 'sim_price_'+mon)).display_data()
    for folder in folders_plot:
        case = folder+'_'+mon
        color = cmap(float(j)/len(folders_plot))
        index = proj_flex_cost_dfs[case].index
        values = proj_flex_cost_dfs[case].values+proj_cost_dfs[case].values
        axarr[i].plot(index,values, '-', label = structures_plot[j] if i==0 else "", color=color, linewidth=2)
        j += 1
    
    axarr[i].grid(b=True, which='major', axis='both')
    axarr[i].tick_params(axis='both', which='major', labelsize=10)
    axarr[i].tick_params(axis='both', which='minor', labelsize=10)
    axarr[i].set_xlim(proj_flex_cost_dfs[case].index[0],proj_flex_cost_dfs[case].index[-1])
    axarr[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axarr[i].xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    
    axarr[i].axvline(x=DR_call_start, ymin=0, ymax=1, linestyle = '-.', color='red', linewidth=1.5)

    axarr[i].axvline(x=DRstart, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1.5)
    
    axarr[i].axvline(x=DRend, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1.5)
    
    if i == 1:
        axarr[i].text(DR_call_start+datetime.timedelta(minutes = 1), -0.4, r'Demand response call',fontsize=14, color='red')
        axarr[i].text(DRstart + +datetime.timedelta(minutes = 10), -0.7, r'Demand response period',fontsize=14, color='green')
    
    i=i+1

f.suptitle('Total Cost Projections - Static Electricity Pricing',y=0.96,fontsize=20)

axarr[0].set_title("Winter 7.1.",fontsize=18)
axarr[1].set_title("Spring 1.3.",fontsize=18)
axarr[2].set_title("Autumn 20.11.",fontsize=18)
axarr[2].set_xlabel("Time",fontsize=16)

axarr[1].set_ylabel('Price (pounds)', fontsize=16)

#axarr[0].text(index[8], 0.1, r'Demand response period',fontsize=10, color='green')
#axarr[0].text(index[4], 0.1, r'Demand response call',fontsize=10, color='red')

handles,labels = [],[]
for ax in axarr:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

lgd = plt.legend(handles,labels, bbox_to_anchor=(0.5, -0.3), loc = 9, ncol=4, fontsize=16)

f.subplots_adjust(hspace=0.35)

f.savefig(os.path.join(simu_path, analysis_folder, "projected_totcost_plots_dr_stat.png"), bbox_inches="tight")
f.savefig(os.path.join(simu_path, analysis_folder, "projected_totcost_plots_dr_stat.pdf"), bbox_inches="tight")
plt.clf()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# FlexCost Projections - Static Pricing
#fig = plt.figure(figsize=(11.69,8.27))
f, axarr = plt.subplots(nrows = 3, sharey=True, figsize=(11.69,8.27))
folders_plot = [folders[0],folders[2]]
structures_plot = [structure[0], structure[2]]
i = 0
for mon in mons:
    j = 0
    
    if mon == 'jan':
        ff_date = '1/7/2017'
    if mon == 'mar':
        ff_date = '3/1/2017'
    if mon == 'nov':
        ff_date = '11/20/2017'
    
    DRstart = datetime.datetime.strptime(ff_date + ' 17:30:00', '%m/%d/%Y %H:%M:%S') # hour to start DR - ramp down 30 mins before
    DRend = datetime.datetime.strptime(ff_date + ' 18:30:00', '%m/%d/%Y %H:%M:%S') # hour to end DR - ramp 30 mins later
    DR_call_start = datetime.datetime.strptime(ff_date + ' 17:00:00', '%m/%d/%Y %H:%M:%S') # 
    
    #price = load_namespace(os.path.join(simu_path, 'prices', 'sim_price_'+mon)).display_data()
    for folder in folders_plot:
        case = folder+'_'+mon
        color = cmap(float(j)/len(folders_plot))
        index = proj_cost_dfs[case].index
        values = proj_cost_dfs[case].values
        axarr[i].plot(index,values, '-', label = structures_plot[j] if i==0 else "", color=color, linewidth=2)
        j += 1
    
    axarr[i].grid(b=True, which='major', axis='both')
    axarr[i].tick_params(axis='both', which='major', labelsize=10)
    axarr[i].tick_params(axis='both', which='minor', labelsize=10)
    axarr[i].set_xlim(proj_flex_cost_dfs[case].index[0],proj_flex_cost_dfs[case].index[-1])
    axarr[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axarr[i].xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    
    axarr[i].axvline(x=DR_call_start, ymin=0, ymax=1, linestyle = '-.', color='red', linewidth=1.5)

    axarr[i].axvline(x=DRstart, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1.5)
    
    axarr[i].axvline(x=DRend, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1.5)
    
    if i == 1:
        axarr[i].text(DR_call_start+datetime.timedelta(minutes = 1), -0.2, r'Demand response call',fontsize=14, color='red')
        axarr[i].text(DRstart + +datetime.timedelta(minutes = 10), -0.5, r'Demand response period',fontsize=14, color='green')
    
    i=i+1

f.suptitle('Electricity Cost Projections - Static Electricity Pricing',y=0.96,fontsize=20)

axarr[0].set_title("Winter 7.1.",fontsize=18)
axarr[1].set_title("Spring 1.3.",fontsize=18)
axarr[2].set_title("Autumn 20.11.",fontsize=18)
axarr[2].set_xlabel("Time",fontsize=16)

axarr[1].set_ylabel('Cost (pounds)', fontsize=16)

#axarr[0].text(index[8], 0.1, r'Demand response period',fontsize=10, color='green')
#axarr[0].text(index[4], 0.1, r'Demand response call',fontsize=10, color='red')

handles,labels = [],[]
for ax in axarr:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

lgd = plt.legend(handles,labels, bbox_to_anchor=(0.5, -0.3), loc = 9, ncol=4, fontsize=16)

f.subplots_adjust(hspace=0.35)

f.savefig(os.path.join(simu_path, analysis_folder, "projected_cost_plots_dr_stat.png"), bbox_inches="tight")
f.savefig(os.path.join(simu_path, analysis_folder, "projected_cost_plots_dr_stat.pdf"), bbox_inches="tight")
plt.clf()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# FlexCost Projections - Static Pricing
#fig = plt.figure(figsize=(11.69,8.27))
f, axarr = plt.subplots(nrows = 3, sharey=True, figsize=(11.69,8.27))
folders_plot = [folders[0],folders[2]]
structures_plot = [structure[0], structure[2]]
i = 0
for mon in mons:
    j = 0
    
    if mon == 'jan':
        ff_date = '1/7/2017'
    if mon == 'mar':
        ff_date = '3/1/2017'
    if mon == 'nov':
        ff_date = '11/20/2017'
    
    DRstart = datetime.datetime.strptime(ff_date + ' 17:30:00', '%m/%d/%Y %H:%M:%S') # hour to start DR - ramp down 30 mins before
    DRend = datetime.datetime.strptime(ff_date + ' 18:30:00', '%m/%d/%Y %H:%M:%S') # hour to end DR - ramp 30 mins later
    DR_call_start = datetime.datetime.strptime(ff_date + ' 17:00:00', '%m/%d/%Y %H:%M:%S') # 
    
    #price = load_namespace(os.path.join(simu_path, 'prices', 'sim_price_'+mon)).display_data()
    for folder in folders_plot:
        case = folder+'_'+mon
        color = cmap(float(j)/len(folders_plot))
        index = proj_flex_cost_dfs[case].index
        values = proj_flex_cost_dfs[case].values
        axarr[i].plot(index,values, '-', label = structures_plot[j] if i==0 else "", color=color, linewidth=2)
        j += 1
    
    axarr[i].grid(b=True, which='major', axis='both')
    axarr[i].tick_params(axis='both', which='major', labelsize=10)
    axarr[i].tick_params(axis='both', which='minor', labelsize=10)
    axarr[i].set_xlim(proj_flex_cost_dfs[case].index[0],proj_flex_cost_dfs[case].index[-1])
    axarr[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axarr[i].xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    
    axarr[i].axvline(x=DR_call_start, ymin=0, ymax=1, linestyle = '-.', color='red', linewidth=1.5)

    axarr[i].axvline(x=DRstart, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1.5)
    
    axarr[i].axvline(x=DRend, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1.5)
    
    if i == 1:
        axarr[i].text(DR_call_start+datetime.timedelta(minutes = 1), -0.2, r'Demand response call',fontsize=14, color='red')
        axarr[i].text(DRstart + +datetime.timedelta(minutes = 10), -0.5, r'Demand response period',fontsize=14, color='green')
    
    i=i+1

f.suptitle('Flexibility Cost Projections - Static Electricity Pricing',y=0.96,fontsize=20)

axarr[0].set_title("Winter 7.1.",fontsize=18)
axarr[1].set_title("Spring 1.3.",fontsize=18)
axarr[2].set_title("Autumn 20.11.",fontsize=18)
axarr[2].set_xlabel("Time",fontsize=16)

axarr[1].set_ylabel('Cost (pounds)', fontsize=16)

#axarr[0].text(index[8], 0.1, r'Demand response period',fontsize=10, color='green')
#axarr[0].text(index[4], 0.1, r'Demand response call',fontsize=10, color='red')

handles,labels = [],[]
for ax in axarr:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

lgd = plt.legend(handles,labels, bbox_to_anchor=(0.5, -0.3), loc = 9, ncol=4, fontsize=16)

f.subplots_adjust(hspace=0.35)

f.savefig(os.path.join(simu_path, analysis_folder, "projected_flexcost_plots_dr_stat.png"), bbox_inches="tight")
f.savefig(os.path.join(simu_path, analysis_folder, "projected_flexcost_plots_dr_stat.pdf"), bbox_inches="tight")
plt.clf()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Differences to MPC projections
#fig = plt.figure(figsize=(11.69,8.27))
f, axarr = plt.subplots(nrows = 3, ncols = 2, sharex = 'row', sharey=False, figsize=(10, 7))

folders_plot = folders
structures_plot = structure

i = 0 
for mon in mons:
    
    if mon == 'jan':
        ff_date = '1/7/2017'
    if mon == 'mar':
        ff_date = '3/1/2017'
    if mon == 'nov':
        ff_date = '11/20/2017'
    
    DRstart = datetime.datetime.strptime(ff_date + ' 17:30:00', '%m/%d/%Y %H:%M:%S') # hour to start DR - ramp down 30 mins before
    DRend = datetime.datetime.strptime(ff_date + ' 18:30:00', '%m/%d/%Y %H:%M:%S') # hour to end DR - ramp 30 mins later
    DR_call_start = datetime.datetime.strptime(ff_date + ' 17:00:00', '%m/%d/%Y %H:%M:%S') # 
    
    #price = load_namespace(os.path.join(simu_path, 'prices', 'sim_price_'+mon)).display_data()
    k = 0
    j = 0
    for folder in folders_plot:
        case = folder+'_'+mon
        color = cmap(float(j)/int(len(folders_plot)/2))
        
        index = proj_cost_dfs[case].index
        values = proj_cost_dfs[case].values + proj_flex_cost_dfs[case].values
        #print(dr_diff_wols[case])
        
        if pricing[k] == 'Static':
            ax = axarr[i][0]
        if pricing[k] == 'Dynamic':
            ax = axarr[i][1]
            
        ax.plot(index,values, '-', label = structures_plot[k] if (i == 0 and k%2 == 0) else "", color=color, linewidth=1.5)
        #ax.fill_between(index, 0, values, where=values<0, facecolor=color, alpha=0.4)
        
        #load_shape = pd.Series(0,index=index)
        #ax.plot(index, load_shape.values, '-' ,color='black', linewidth = 1.0)
        
        #for t in load_shape.index:
        #    if t >= DRstart and t <= DRend:
        #        load_shape[t] = -2
        #ax.plot(index, load_shape.values, '--', color='black', linewidth = 1.0)
        
        ax.set_xlim(index[0]-datetime.timedelta(minutes = 1),index[-1]+datetime.timedelta(minutes = 1))
        ax.grid(b=True, which='major', axis='both')
        #ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,10,20,30,40,50], interval = 1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        
        ax.axvline(x=DR_call_start, ymin=0, ymax=1, linestyle = '-.', color='red', linewidth=1)
        ax.axvline(x=DRstart, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1)
        ax.axvline(x=DRend, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1)
        
        if i == 1 and k==2:
            ax.text(DR_call_start+datetime.timedelta(minutes = 1), 0.8, r'Demand response call',fontsize=10, color='red')
            ax.text(DRstart + +datetime.timedelta(minutes = 10), 0.4, r'Demand response period',fontsize=10, color='green')
        
        if k%2 != 0:
            j = j+1
        k=k+1
    i = i+1

f.suptitle('Projections of total cost', y=0.98, fontsize=16)   

axarr[0][0].set_title("Static Pricing\nWinter 7.1.",fontsize=12)
axarr[0][1].set_title("Dynamic pricing\nWinter 7.1.",fontsize=12)   
axarr[1][0].set_title("Spring 1.3.",fontsize=12)
axarr[1][1].set_title("Spring 1.3.",fontsize=12)
axarr[2][0].set_title("Autumn 20.11.",fontsize=12)
axarr[2][1].set_title("Autumn 20.11.",fontsize=12)  

axarr[1][0].set_ylabel('Cost [pounds]', fontsize=12)
axarr[2][0].set_xlabel("Time",fontsize=12)

handles,labels = [],[]
for ax in axarr[0][:]:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

axarr[2][0].legend(handles,labels, bbox_to_anchor=(1.0, -0.25), loc = 9, ncol=2, fontsize=14)

plt.subplots_adjust(hspace = 0.3, wspace = 0.15)    

f.savefig(os.path.join(simu_path, analysis_folder, "totcosts.png"), bbox_inches="tight")
f.savefig(os.path.join(simu_path, analysis_folder, "totcosts.pdf"), bbox_inches="tight")
plt.clf()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Differences to MPC projections
#fig = plt.figure(figsize=(11.69,8.27))
f, axarr = plt.subplots(nrows = 3, ncols = 2, sharex = 'row', sharey=False, figsize=(10, 7))

folders_plot = folders
structures_plot = structure
bldg_plotlist = [bldg_list[0], bldg_list[3], bldg_list[6],bldg_list[9]]

i = 0 
for mon in mons:
    
    if mon == 'jan':
        ff_date = '1/7/2017'
    if mon == 'mar':
        ff_date = '3/1/2017'
    if mon == 'nov':
        ff_date = '11/20/2017'
    
    DRstart = datetime.datetime.strptime(ff_date + ' 17:30:00', '%m/%d/%Y %H:%M:%S') # hour to start DR - ramp down 30 mins before
    DRend = datetime.datetime.strptime(ff_date + ' 18:30:00', '%m/%d/%Y %H:%M:%S') # hour to end DR - ramp 30 mins later
    DR_call_start = datetime.datetime.strptime(ff_date + ' 17:00:00', '%m/%d/%Y %H:%M:%S') # 
    
    #price = load_namespace(os.path.join(simu_path, 'prices', 'sim_price_'+mon)).display_data()
    k = 0
    j = 0
    for folder in folders_plot:
        case = folder+'_'+mon        
        index = emutemps[case].index
        
        if pricing[k] == 'Static':
                ax = axarr[i][0]
        if pricing[k] == 'Dynamic':
                ax = axarr[i][1]
        
        l = 0
        for bldg in bldg_plotlist:
            bldg_key = bldg+'_R2CW_HP'
            color = cmap(float(l)/int(len(bldg_plotlist)))
            if structures_plot[k] == 'Centralised':
                values = emutemps[case][bldg_key].values 
                line_type = '--'
            else:
                values = emutemps[case][bldg_key].values -273.15
                line_type = '-'
            
            ax.plot(index,values, line_type, label = structures_plot[k] + '-' + bldg if (i == 0 and k%2 == 0) else "", color=color, linewidth=1.5)
            
            l += 1
        
        ax.set_xlim(index[0],index[-1])
        ax.grid(b=True, which='major', axis='both')
        #ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,15,45], interval = 1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        
        ax.axvline(x=DR_call_start, ymin=0, ymax=1, linestyle = '-.', color='red', linewidth=1)
        ax.axvline(x=DRstart, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1)
        ax.axvline(x=DRend, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1)
        
        if i == 1 and k==2:
            ax.text(DR_call_start+datetime.timedelta(minutes = 1), 21.5, r'Demand response call',fontsize=10, color='red')
            ax.text(DRstart + +datetime.timedelta(minutes = 10), 21, r'Demand response period',fontsize=10, color='green')
        
        if k%2 != 0:
            j = j+1
        k=k+1
    i = i+1

f.suptitle('Emulated indoor temperatures', y=0.98, fontsize=16)   

axarr[0][0].set_title("Static Pricing\nWinter 7.1.",fontsize=12)
axarr[0][1].set_title("Dynamic pricing\nWinter 7.1.",fontsize=12)   
axarr[1][0].set_title("Spring 1.3.",fontsize=12)
axarr[1][1].set_title("Spring 1.3.",fontsize=12)
axarr[2][0].set_title("Autumn 20.11.",fontsize=12)
axarr[2][1].set_title("Autumn 20.11.",fontsize=12)  

axarr[1][0].set_ylabel('Temperature [C]', fontsize=12)
axarr[2][0].set_xlabel("Time",fontsize=12)

handles,labels = [],[]
for ax in axarr[0][:]:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

axarr[2][0].legend(handles,labels, bbox_to_anchor=(1.0, -0.25), loc = 9, ncol=2, fontsize=14)

plt.subplots_adjust(hspace = 0.3, wspace = 0.15)    

f.savefig(os.path.join(simu_path, analysis_folder, "emutemps_paper.png"), bbox_inches="tight")
f.savefig(os.path.join(simu_path, analysis_folder, "emutemps_paper.pdf"), bbox_inches="tight")
plt.clf()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Differences to MPC projections
#fig = plt.figure(figsize=(11.69,8.27))
f, axarr = plt.subplots(nrows = 3, ncols = 2, sharex = 'row', sharey=False, figsize=(10, 7))

folders_plot = folders
structures_plot = structure
bldg_plotlist = [bldg_list[0], bldg_list[3], bldg_list[6],bldg_list[9]]

i = 0 
for mon in mons:
    
    if mon == 'jan':
        ff_date = '01/07/2017'
    if mon == 'mar':
        ff_date = '03/01/2017'
    if mon == 'nov':
        ff_date = '11/20/2017'
    
    DRstart = datetime.datetime.strptime(ff_date + ' 17:30:00', '%m/%d/%Y %H:%M:%S') # hour to start DR - ramp down 30 mins before
    DRend = datetime.datetime.strptime(ff_date + ' 18:30:00', '%m/%d/%Y %H:%M:%S') # hour to end DR - ramp 30 mins later
    DR_call_start = datetime.datetime.strptime(ff_date + ' 17:00:00', '%m/%d/%Y %H:%M:%S') # 
    
    ref_time_str = ff_date + ' 16:55:00'
    plot_time_str = ff_date + ' 17:25:00'
    
    #price = load_namespace(os.path.join(simu_path, 'prices', 'sim_price_'+mon)).display_data()
    k = 0
    j = 0
    for folder in folders_plot:
        case = folder+'_'+mon        
        index = emutemps[case].index
        
        if pricing[k] == 'Static':
                ax = axarr[i][0]
        if pricing[k] == 'Dynamic':
                ax = axarr[i][1]
        
        l = 0
        for bldg in bldg_plotlist:
            bldg_key = bldg+'_R2CW_HP'
            color = cmap(float(l)/int(len(bldg_plotlist)))
            #print(mpctemps[case].keys())            
            if structures_plot[k] == 'Centralised':
                index1 = mpctemps[case][ref_time_str]['mpc_model.'+bldg_key+'.TAir'].index
                values = mpctemps[case][ref_time_str]['mpc_model.'+bldg_key+'.TAir'].values -273.15
                line_type = '--'
            else:
                index1 = mpctemps[case]['MinEne_0-10'][ref_time_str][bldg_key].index
                values = mpctemps[case]['MinEne_0-10'][ref_time_str][bldg_key].values -273.15
                line_type = '-'
            
            ax.plot(index1,values, line_type, label = structures_plot[k] + '-' + bldg if (i == 0 and k%2 == 0) else "", color=color, linewidth=1.5)
            
            
            if structures_plot[k] == 'Centralised':
                index2 = mpctemps[case][plot_time_str]['mpc_model.'+bldg_key+'.TAir'].index
                values = mpctemps[case][plot_time_str]['mpc_model.'+bldg_key+'.TAir'].values -273.15
                line_type = '--'
            else:
                index2 = mpctemps[case]['MinEne_0-10'][plot_time_str][bldg_key].index
                values = mpctemps[case]['MinEne_0-10'][plot_time_str][bldg_key].values -273.15
                line_type = '-'
            
            ax.plot(index2,values, line_type, color=color, linewidth=1.5)
            
            l += 1
        
        ax.set_xlim(index1[0],index2[-1])
        ax.grid(b=True, which='major', axis='both')
        #ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,15,45], interval = 1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        
        ax.axvline(x=DR_call_start, ymin=0, ymax=1, linestyle = '-.', color='red', linewidth=1)
        ax.axvline(x=DRstart, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1)
        ax.axvline(x=DRend, ymin=0, ymax=1, linestyle = '-.', color='green', linewidth=1)
        
        if i == 1 and k==2:
            ax.text(DR_call_start+datetime.timedelta(minutes = 1), 21.5, r'Demand response call',fontsize=10, color='red')
            ax.text(DRstart + +datetime.timedelta(minutes = 10), 21, r'Demand response period',fontsize=10, color='green')
        
        if k%2 != 0:
            j = j+1
        k=k+1
    i = i+1

f.suptitle('Projected indoor temperatures', y=0.98, fontsize=16)   

axarr[0][0].set_title("Static Pricing\nWinter 7.1.",fontsize=12)
axarr[0][1].set_title("Dynamic pricing\nWinter 7.1.",fontsize=12)   
axarr[1][0].set_title("Spring 1.3.",fontsize=12)
axarr[1][1].set_title("Spring 1.3.",fontsize=12)
axarr[2][0].set_title("Autumn 20.11.",fontsize=12)
axarr[2][1].set_title("Autumn 20.11.",fontsize=12)  

axarr[1][0].set_ylabel('Temperature [C]', fontsize=12)
axarr[2][0].set_xlabel("Time",fontsize=12)

handles,labels = [],[]
for ax in axarr[0][:]:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

axarr[2][0].legend(handles,labels, bbox_to_anchor=(1.0, -0.25), loc = 9, ncol=2, fontsize=14)

plt.subplots_adjust(hspace = 0.3, wspace = 0.15)    

f.savefig(os.path.join(simu_path, analysis_folder, "mpctemps_paper.png"), bbox_inches="tight")
f.savefig(os.path.join(simu_path, analysis_folder, "mpctemps_paper.pdf"), bbox_inches="tight")
plt.clf()



#exit()