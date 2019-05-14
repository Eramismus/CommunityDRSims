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

from bokeh.plotting import figure, output_file, show

community = 'ResidentialCommunity'
sim_id = 'MinEne'
model_id = 'R2CW_HP'
bldg_list = load_namespace(os.path.join('path to models', 'teaser_bldgs_residentialUK_10bldgs_fallback'))
print(bldg_list)
mons = ['jan', 'mar', 'nov']
folders = ['10bldgs_decentr_enemin']
enemin_folder = 'decentr_enemin'

cases = ['Energy Minimisation']
			
building = 'AggrMPC_ResidentialUK'

simu_path = "path to sims"
analysis_folder = 'analysis_ref'
step = 300
horizon = 2*3600/step



costs = {}
aggrcom = {}
aggr = {}
real_cont = {}
real_cont_aggr = {}
violations = {}
peak_comm = {}
cons_hc = {}
cons_lc = {}
emutemps = {}
mpctemps = {}
refheats = {}
refheataggrs = {}
reftemps = {}
refviolations = {}

for mon in mons:
	for folder in folders:
		case = folder+'_'+mon
		price = load_namespace(os.path.join(simu_path, 'prices', 'sim_price_'+mon)).display_data()
		costs[case] = load_namespace(os.path.join(simu_path,case,'costs'))
		aggrcom[case] = load_namespace(os.path.join(simu_path,case,'aggrcom'))
		aggr[case] = load_namespace(os.path.join(simu_path,case,'aggr'))
		real_cont[case] = load_namespace(os.path.join(simu_path,case,'real_cont'))
		real_cont_aggr[case] =  real_cont[case].sum(axis=1)
		violations[case] = load_namespace(os.path.join(simu_path,case,'violation_df'))
		peak_comm[case] = load_namespace(os.path.join(simu_path,case,'peak_comm'))
		cons_hc[case] = load_namespace(os.path.join(simu_path,case,'cons_hc'))
		cons_lc[case] = load_namespace(os.path.join(simu_path,case,'cons_lc'))
		emutemps[case] = load_namespace(os.path.join(simu_path,case,'emutemps'))
		mpctemps[case] = load_namespace(os.path.join(simu_path,case,'mpctemps'))

	refheatinput = {}	
	reftemp = {}
	for bldg in bldg_list:
		PIfolder = 'PI_folder_'+mon
		bldg = bldg+'_'+model_id
		
		df = load_namespace(os.path.join(PIfolder, 'ref_power_'+bldg))
		df = df[~df.index.duplicated(keep='last')]
		df = df.resample(str(10)+'S').mean().ffill()
		refheatinput[bldg] = df
	
	
		df = load_namespace(os.path.join(PIfolder, 'ref_temp_'+bldg))
		
		df = df[~df.index.duplicated(keep='last')]
		df = df.resample(str(10)+'S').mean().interpolate()
		reftemp[bldg] = df
		#print(reftemp[bldg])
	
	refheatinput_df = pd.DataFrame.from_dict(refheatinput, orient='columns')
	reftemp_df = pd.DataFrame.from_dict(reftemp, orient='columns')
	refheat_aggr = refheatinput_df.sum(axis=1)
	
	refheats[mon] = refheatinput_df
	reftemps[mon] = reftemp_df
	refheataggrs[mon] = refheat_aggr
	#print(refheat_aggr)
	#print(reftemp_df)
	print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
	#print(reftemp_df)
	store_namespace(os.path.join(simu_path, analysis_folder, 'refheatinput_df_'+mon),refheatinput_df)
	store_namespace(os.path.join(simu_path, analysis_folder, 'reftemp_df_' + mon),reftemp_df)

# """""""""""" Comfort violations for reference """""""""""""""""""
for mon in mons:
	constr_folder = enemin_folder+'_'+mon
	index = emutemps[folders[0]+'_'+mon].index
	#print(index)
	reftemp_df = reftemps[mon]
	#print(reftemp_df)
	
	constraints = {}
	for bldg in bldg_list:
		setpoint_dict = load_namespace(os.path.join(simu_path, constr_folder, 'constraints_'+bldg+'_'+model_id)).data['TAir']
		constraints[bldg] = {}
		for param in setpoint_dict.keys():
			constraints[bldg]['hi'] = setpoint_dict['Slack_LTE'].display_data()
			constraints[bldg]['lo'] = setpoint_dict['Slack_GTE'].display_data()
			
	constraints_df = pd.DataFrame.from_dict(constraints, orient = 'index')
			
	#print(constraints_df)
	#print(emutemps_df)
	violation = {}
	#print(constraints_df.loc['Detached_0']['lo'])
	for bldg in bldg_list:
		violation[bldg] = {} 
		for time in index:
			emutemp = reftemp_df.resample(str(step)+'S').mean()[bldg+'_'+model_id][time]
			#print(emutemp)
			
			#emutemp = emutemp.values()
			#print(emutemp)
			constraint_hi = constraints_df.loc[bldg]['hi'].resample(str(step)+'S').ffill()
			constraint_lo = constraints_df.loc[bldg]['lo'].resample(str(step)+'S').ffill()
			
			constraint_hi = constraint_hi[time]
			constraint_lo = constraint_lo[time]
			#print(constraint_hi)
			
			if emutemp > constraint_hi:
				violation[bldg][time] = (emutemp - constraint_hi)*step/3600
			elif emutemp < constraint_lo:
				violation[bldg][time] = (constraint_lo-emutemp)*step/3600
			else:
				violation[bldg][time] = 0

	refviolation_df = pd.DataFrame.from_dict(violation, orient = 'columns')
	#print(refviolation_df)
	refviolations[mon] = refviolation_df
	store_namespace(os.path.join(simu_path,analysis_folder,'refviolation_df_'+mon),refviolation_df)
	#print(refviolation_df)
	#print(reftemp_df.resample(str(step)+'S').mean())

#print(mpctemps)	

violation_comm = {}
totreal_cont = {}
totcons_hc = {}
totcons_lc = {}
totcosts = {}
comm_metric = {}

for mon in mons:
	
	price = load_namespace(os.path.join(simu_path, 'prices', 'sim_price_'+mon)).display_data()
	price.index = price.index.tz_localize(None)
	
	for folder in folders:
		case = folder+'_'+mon
		
		if mon == 'jan':
			ff_date = '01/07/2017'
		if mon == 'mar':
			ff_date = '03/01/2017'
		if mon == 'nov':
			ff_date = '11/20/2017'
		index = real_cont[case].index
		
		DRstart = datetime.datetime.strptime(ff_date + ' 17:30:00', '%m/%d/%Y %H:%M:%S') # hour to start DR - ramp down 30 mins before
		DRend = datetime.datetime.strptime(ff_date + ' 18:30:00', '%m/%d/%Y %H:%M:%S') # hour to end DR - ramp 30 mins later
		DR_call_start = datetime.datetime.strptime(ff_date + ' 17:00:00', '%m/%d/%Y %H:%M:%S') # Round of loop to implement the call
		DR_call_end = datetime.datetime.strptime(ff_date + ' 18:30:00', '%m/%d/%Y %H:%M:%S') # Round of loop to
		
		hc_start = datetime.datetime.strptime(ff_date + ' 17:30:00', '%m/%d/%Y %H:%M:%S')
		hc_start = DRstart
		hc_end = index[-1]
		lc_start = datetime.datetime.strptime(ff_date + ' 16:30:00', '%m/%d/%Y %H:%M:%S')
		lc_end = DRstart - datetime.timedelta(seconds = 1)
		#print(index[0])
		print(index[-1])
		#lc_end = datetime.datetime.strptime(ff_date + ' 17:29:59', '%m/%d/%Y %H:%M:%S')
		
		cons_hc[case] = real_cont[case][hc_start:hc_end]
		cons_lc[case] = real_cont[case][lc_start:lc_end]
		#print(cons_hc[case].sum(axis=1))
		#print(cons_lc[case].sum(axis=1))
		
		comm_metric[case] = {}
		comm_metric[case]['Total Consumption [kWh]'] = real_cont_aggr[case].resample(str(step)+'S').mean().sum(axis=0)*step/3600/1000
		index = real_cont_aggr[case].index
		#real_cont_aggr[case].index.tz = None
		case_cost = real_cont_aggr[case].resample('1800S').mean() * price['pi_e'] / 1000000 * 0.5
		comm_metric[case]['Total Cost [pounds]'] = case_cost.sum(axis=0)
		comm_metric[case]['Violation Ch'] = violations[case].sum(axis=0).sum(axis=0)
		comm_metric[case]['totcons_hc [kWh]'] = cons_hc[case].resample(str(step)+'S').mean().sum(axis=0).sum(axis=0)*step/3600/1000
		comm_metric[case]['totcons_lc [kWh]'] = cons_lc[case].resample(str(step)+'S').mean().sum(axis=0).sum(axis=0)*step/3600/1000
		comm_metric[case]['peak consumption [kW]'] = peak_comm[case][0]/1000
		comm_metric[case]['peak consumption time'] = peak_comm[case][1]
		
		totreal_cont[case] = real_cont_aggr[case].resample(str(step)+'S').mean().sum(axis=0)
		violation_comm[case] = violations[case].sum(axis=0).sum(axis=0)
		totcons_hc[case] = cons_hc[case].sum(axis=0).sum(axis=0)
		totcons_lc[case] = cons_lc[case].sum(axis=0).sum(axis=0)
		totcosts[case] = costs[case].sum(axis=0)

	refheat_aggr = refheataggrs[mon]
	refviolation_df = refviolations[mon]
	print(refheat_aggr[hc_start:hc_end])
	refheat_aggr.index = refheat_aggr.index.tz_localize(None)
	comm_metric['PI_'+mon] = {}
	comm_metric['PI_'+mon]['Total Consumption [kWh]'] = refheat_aggr[index[0]:index[-1]].resample(str(step)+'S').mean().sum(axis=0)*step/3600/1000
	refcost = refheat_aggr[index[0]:index[-1]].resample('1800S').mean() * price['pi_e'] / 1000000 * 0.5
	comm_metric['PI_'+mon]['Total Cost [pounds]'] = refcost.sum(axis=0)
	comm_metric['PI_'+mon]['totcons_hc [kWh]'] = refheat_aggr[hc_start:hc_end].resample(str(step)+'S').mean().sum(axis=0)*step/3600/1000
	comm_metric['PI_'+mon]['totcons_lc [kWh]'] = refheat_aggr[lc_start:lc_end].resample(str(step)+'S').mean().sum(axis=0)*step/3600/1000
	comm_metric['PI_'+mon]['peak consumption [kW]'] = refheat_aggr.max()/1000
	comm_metric['PI_'+mon]['peak consumption time'] = refheat_aggr.idxmax()
	comm_metric['PI_'+mon]['Violation Ch'] = refviolation_df.sum(axis=0).sum(axis=0)

comm_metric_df = pd.DataFrame.from_dict(comm_metric, orient='columns')
comm_metric_df.to_csv(os.path.join(simu_path, analysis_folder, 'comm_metrics.csv'))
print(comm_metric_df)
	
#print(real_cont)	
#real_cont = pd.concat([real_cont[folders[0]], real_cont[folders[1]], real_cont[folders[2]]], axis=1)	

dt = []
for time in aggr[case].keys():
	control_start = datetime.datetime.strptime(time, '%m/%d/%Y %H:%M:%S')
	control_end = datetime.datetime.strptime(time, '%m/%d/%Y %H:%M:%S') + datetime.timedelta(seconds = horizon)
	dt.append(control_start)

dt = pd.DataFrame(dt,columns = ['Dates'])
dt = dt.set_index(pd.DatetimeIndex(dt['Dates']))
index = dt.index
index = index.tz_localize(None)
index = index.sort_values()

print('%%%%%%%%% ---- Plots ---- %%%%%%%')	
cmap = plt.get_cmap('jet')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Consumption profile
#fig = plt.figure(figsize=(11.69,8.27))
f, axarr = plt.subplots(nrows = 3, sharey=True, figsize=(11.69,8.27))
#ax = fig.gca()
#ax1 = ax.twinx()
#print(mpctemps)
#plot_times=[0,3,5]
i = 0
for mon in mons:
	j = 0
	price = load_namespace(os.path.join(simu_path, 'prices', 'sim_price_'+mon)).display_data()
	for folder in folders:
		case = folder+'_'+mon
		color = cmap(float(j)/len(folders))
		index = real_cont_aggr[case].resample('10S').ffill().index
		values = real_cont_aggr[case].resample('10S').ffill().values/1000
		axarr[i].plot(index,values, '-', label = cases[j] if i==0 else "", color=color, linewidth=2)
		j = j+1
	
	index = refheataggrs[mon][index[0]:index[-1]].index
	values = refheataggrs[mon][index[0]:index[-1]].values/1000
	axarr[i].plot(index,values, '-.', label = 'RBC' if i==0 else "", color='red', linewidth=2)
	
	ax1 = axarr[i].twinx()
	index = price[index[0]:index[-1]].resample('10S').ffill().index
	values = price[index[0]:index[-1]].resample('10S').ffill().values
	ax1.plot(index,values, '--', label = 'Price' if i==0 else "", color='black', linewidth=1.5)
	if i == 1:
		ax1.set_ylabel('Electricity Price [GBP/MWh]', fontsize=18)
	line = ax1.get_lines()[0]
	
	axarr[i].grid(b=True, which='major', axis='both')
	axarr[i].tick_params(axis='both', which='major', labelsize=12)
	axarr[i].tick_params(axis='both', which='minor', labelsize=12)
	axarr[i].set_xlim(real_cont_aggr[case].index[0],real_cont_aggr[case].index[-1])
	axarr[i].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
	axarr[i].xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
	i=i+1

f.suptitle('Reference Demand Profiles', y=0.98, fontsize=20)	
axarr[0].set_title("Winter 7.1.",fontsize=18)
axarr[1].set_title("Spring 1.3.",fontsize=18)
axarr[2].set_title("Autumn 20.11.",fontsize=18)
axarr[2].set_xlabel("Time",fontsize=18)

axarr[1].set_ylabel('Power [kW]', fontsize=16)	

handles,labels = [],[]
for ax in axarr:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

handles.append(line)
labels.append('Price')

lgd = plt.legend(handles,labels, bbox_to_anchor=(0.5, -0.3), loc = 9, ncol=4, fontsize=16)

f.subplots_adjust(hspace=0.35)

f.savefig(os.path.join(simu_path, analysis_folder, "cons_subplots-ref.png"), bbox_inches="tight")
f.savefig(os.path.join(simu_path, analysis_folder, "cons_subplots-ref.pdf"), bbox_inches="tight")
plt.clf()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# FF figure

fig = plt.figure(figsize=(11.69,8.27))
ax = fig.gca()
#ax1 = ax.twinx()
#plot_bldgs = [0, 10, 20, 29, 4, 12]

mu = 0
variance = 1
sigma = np.sqrt(variance)
x = np.linspace(0, mu + 3*sigma, 200)
y1= mlab.normpdf(x, mu, sigma)*5
y2= np.exp(x)
y3= np.full(len(x), 1)

color = cmap(0.5)
ax.plot(x, y3, '-', color='black', label='Consumption Profile 1' , linewidth=2)
ax.fill_between(x, 0, y3, where=x >= 1.5, facecolor=cmap(0.5), alpha=0.4, interpolate=True)
ax.fill_between(x, 0, y3, where=x <= 1.5, facecolor=cmap(1), alpha=0.4, interpolate=True) 

color = cmap(1)

ax.plot(x, y1, '--', label='Consumption Profile 2', color='black', linewidth=2)
ax.fill_between(x, 0, y1, where=x >= 1.5, facecolor=cmap(0.5), alpha=0.4, interpolate=True)
ax.fill_between(x, 0, y1, where=x <= 1.5, facecolor=cmap(1), alpha=0.4, interpolate=True)
#ax.plot(x, y2, '-', label='consumption profile 2')
#ax.fill_between(x, 0, y2, where=x >= 0.5, facecolor='blue', alpha=0.5, interpolate=True)


ax.set_xlim(x[0],x[-1])
ax.set_ylim(y1[-1],y1[0]+0.1)
	
handles,labels = [],[]
for ax in fig.axes:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

plt.axvline(x=1.5,linestyle = '-',color='red',linewidth=3)
#ax.legend(fontsize=14)
plt.xlabel("Time",fontsize=18)
plt.ylabel(r"Energy Consumption",fontsize=18)
plt.title("Flexibility Factor Demonstration",fontsize=22)
#plt.xticks(rotation=35)
# We change the fontsize of minor ticks label 

ax.set_yticklabels([])
ax.set_xticklabels([])

ax.text(0.25, 0.8, r'Time Period 1',fontsize=14, color='white', bbox={'facecolor':cmap(1), 'alpha':0.6, 'pad':10})
ax.text(1.75, 0.8, r'Time Period 2',fontsize=14, bbox={'facecolor':cmap(0.5), 'alpha':0.6, 'pad':10})

ax.text(2, 1.5, r'$FF_1 < FF_2$', fontsize=16, bbox={'facecolor':'white', 'alpha':0.6, 'pad':10})
#plt.tick_params(axis='both', which='major', labelsize=12)
#plt.tick_params(axis='both', which='minor', labelsize=12)
plt.legend(handles,labels,fontsize=14)
#plt.legend(handles,labels, bbox_to_anchor = (1.04,0.5), loc ='center left')
plt.savefig(os.path.join(simu_path, "ff.pdf"),bbox_inches="tight")
plt.savefig(os.path.join(simu_path,  "ff.png"),bbox_inches="tight")
plt.clf()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MPC reference profile demo figure

fig = plt.figure(figsize=(11.69,5))
ax = fig.gca()
#ax1 = ax.twinx()
#plot_bldgs = [0, 10, 20, 29, 4, 12]

mu = 0
variance = 1
sigma = np.sqrt(variance)
x = np.linspace(0, mu + 3*sigma, 200)
y1= mlab.normpdf(x, mu, sigma)
#y2= np.exp(x)
y3= np.full(len(x), 1)*np.random.rand(len(x))

color = cmap(0.5)
ax.plot(x, y1, '-', color='black', label='Demand projection - no load-shaping' , linewidth=2)
#ax.fill_between(x, 0, y3, where=x >= 1.5, facecolor=cmap(0.5), alpha=0.4, interpolate=True)
#ax.fill_between(x, 0, y3, where=x <= 1.5, facecolor=cmap(1), alpha=0.4, interpolate=True) 
#print(x)
y2 = mlab.normpdf(x, mu, sigma)
k1 = 1
k2 = 2

i=0
for y in y2:
	if x[i] >= k1-0.5 and x[i] <= k1:
		y2[i] = 0.75*y
	if x[i] >= k1 and x[i] <= k2:
		y2[i] = 0.5*y 
	if x[i] >= k2 and x[i] <= k2+0.5:
		y2[i] = 0.75*y
	i=i+1

color = cmap(1)
ax.plot(x, y2, '--', color='black', label='Demand projection - load-shaping' , linewidth=2)

ax.fill_between(x, y2, y1, facecolor='red', alpha=0.4, interpolate=True)

ax.set_xlim(x[0],x[-1])
ax.set_ylim(y1[-1],y1[0]+0.1)
	
handles,labels = [],[]
for ax in fig.axes:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

plt.axvline(x=k1-0.5,linestyle = '-',color='red',linewidth=3)
plt.axvline(x=k2+0.5,linestyle = '-',color='red',linewidth=3)
#ax.legend(fontsize=14)
plt.xlabel("Time",fontsize=18)
plt.ylabel(r"Energy Consumption",fontsize=18)
plt.title("Reference Profile and Load-Shaping Demonstration",fontsize=22)
#plt.xticks(rotation=35)
# We change the fontsize of minor ticks label 

ax.set_yticklabels([])
ax.set_xticklabels([])

ax.text(k1-0.45, 0.35, r'DR start',fontsize=16, color='red')
ax.text(k2+0.25, 0.35, r'DR end',fontsize=16, color='red')

ax.text(1.5, 0.17, 'Requested \nDemand Reduction', fontsize=14, bbox={'facecolor':'red', 'alpha':0.4, 'pad':10})
#ax.text(1.75, 0.8, r'Time Period 2',fontsize=14, bbox={'facecolor':cmap(0.5), 'alpha':0.6, 'pad':10})

#ax.text(2, 1.5, r'$FF_1 < FF_2$', fontsize=16, bbox={'facecolor':'white', 'alpha':0.6, 'pad':10})
#plt.tick_params(axis='both', which='major', labelsize=12)
#plt.tick_params(axis='both', which='minor', labelsize=12)
plt.legend(handles,labels,fontsize=14)
#plt.legend(handles,labels, bbox_to_anchor = (1.04,0.5), loc ='center left')
plt.savefig(os.path.join(simu_path, analysis_folder, "load_shaping.pdf"),bbox_inches="tight")
plt.savefig(os.path.join(simu_path, analysis_folder,  "load_shaping.png"),bbox_inches="tight")
plt.clf()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MPC reference profile demo figure

fig = plt.figure(figsize=(11.69,4))
ax = fig.gca()
#ax1 = ax.twinx()
#plot_bldgs = [0, 10, 20, 29, 4, 12]

mu = 0
variance = 1
sigma = np.sqrt(variance)
x = np.linspace(0, 200, 200)
y1= np.ones(200)
y2= np.ones(200)
#y3= np.full(len(x), 1)*np.random.rand(len(x))

color = cmap(0.5)
ax.plot(x, y1, '-', color='black', label='Demand projection' , linewidth=2)
#ax.fill_between(x, 0, y3, where=x >= 1.5, facecolor=cmap(0.5), alpha=0.4, interpolate=True)
#ax.fill_between(x, 0, y3, where=x <= 1.5, facecolor=cmap(1), alpha=0.4, interpolate=True) 
#print(x)
k1 = 75
k2 = 150
k3 = 25

i=0
for y in y1:
	if x[i] >= k1 and x[i] <= k2:
		y2[i] = 0.75*y
	if x[i] >= k3 and x[i] <= k1:
		y2[i] = 1.0*y
	i=i+1

color = cmap(1)
ax.plot(x, y2, '--', color='black', label='Requested Demand Reduction' , linewidth=1.8)

ax.fill_between(x, y2, y1, facecolor='red', alpha=0.4, interpolate=True)
ax.axvspan(k3, k1, alpha=0.2, color='green')
ax.axvspan(k1, k2, alpha=0.2, color='red')

ax.set_xlim(x[0],x[-1])
ax.set_ylim(0,2)
	
handles,labels = [],[]
for ax in fig.axes:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

plt.axvline(x=k1,linestyle = '-',color='red',linewidth=3)
plt.axvline(x=k2,linestyle = '-',color='red',linewidth=3)

plt.axvline(x=k3,linestyle = '-',color='green',linewidth=3)
#ax.legend(fontsize=14)
plt.xlabel("Time",fontsize=18)
plt.ylabel(r"Energy Consumption",fontsize=18)
plt.title("Principle of Contracted Flexibility",fontsize=22)
#plt.xticks(rotation=35)
# We change the fontsize of minor ticks label 

ax.set_yticklabels([])
ax.set_xticklabels([])

#ax.text(k1-0.45, 0.35, r'DR start',fontsize=14, color='red')
#ax.text(k2+0.25, 0.35, r'DR end',fontsize=14, color='red')

ax.text(80, 1.2, 'Requested \nDemand Reduction, e', fontsize=14, bbox={'facecolor':'red', 'alpha':0.4, 'pad':10})

ax.text(k3+5, 1.5, 'Notice period, n', fontsize=14, bbox={'facecolor':'green', 'alpha':0.4, 'pad':10})
#ax.text(1.75, 0.8, r'Time Period 2',fontsize=14, bbox={'facecolor':cmap(0.5), 'alpha':0.6, 'pad':10})

#ax.text(2, 1.5, r'$FF_1 < FF_2$', fontsize=16, bbox={'facecolor':'white', 'alpha':0.6, 'pad':10})
#plt.tick_params(axis='both', which='major', labelsize=12)
#plt.tick_params(axis='both', which='minor', labelsize=12)
plt.legend(handles,labels,fontsize=14)
#plt.legend(handles,labels, bbox_to_anchor = (1.04,0.5), loc ='center left')
plt.savefig(os.path.join(simu_path, analysis_folder, "shape-load.pdf"),bbox_inches="tight")
plt.savefig(os.path.join(simu_path, analysis_folder,  "shape-load.png"),bbox_inches="tight")
plt.clf()
