""" Sixth version, make the code easier and more modifiable """
# Define the main programme

from funcs import store_namespace
from funcs import load_namespace
from funcs import emulate_jmod
import os
import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from multiprocessing import Pool
from mpcpy import units
from mpcpy import variables
from mpcpy import models_mod as models

from scipy.optimize import curve_fit
from scipy.linalg import expm
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error, r2_score

from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

from Simulator_HP_mod3 import SimHandler

if __name__ == "__main__":  
    
    # Naming conventions for the simulation
    community = 'ResidentialCommunityUK_rad_2elements'
    sim_id = 'MinEne'
    model_id = 'R2CW_HP'
    bldg_list = load_namespace(os.path.join('file_path_to_folder', 'teaser_bldgs_residential'))
   
    bldg_index_start = 0
    bldg_index_end = 30
    
    emulate = 0 # Emulate or not?
    old_sim_param = 1 # Choose initial guesses
    
    # Overall options
    date = '1/1/2017 '
    start = '1/1/2017 00:00:00'
    end = '1/9/2017 00:00:00'
    
    train_start = start
    valid_start = '1/6/2017 00:00:00'
    train_end = valid_start
    valid_end = '1/9/2017 00:00:00'
    
    meas_sampl = '300'
    mon = 'jan'
    
    folder = 'path_to_file\\results_sysid_new_'+mon
    
    # Features to use in training
    exog = ['weaTDryBul_delay1', 'weaHGloHor_delay1','PowerCompr', 'PowerCompr_delay1', 'T_in_delay13']
    target = 'TAir'
    features_dict = {}
    exog_list = []
    
    j = 0
    for item in exog:
        exog_list.append(item)
        ind_exog = item
        
        ar = []
        for i in range(5):
            ar.append('T_in_delay'+str(i+1))
            features_dict['ARX_lag_'+str(i+1)+'_exog'+str(j)] = exog_list + ar
            features_dict['ARX_lag_'+str(i+1)+'_'+ind_exog] = [ind_exog] + ar
        j += 1    
            
    # Instantiate Simulator
    Sim_list = []
    i = 0
    for bldg in bldg_list[bldg_index_start:bldg_index_end]:
        i = i+1
        Sim = SimHandler(sim_start = start,
                    sim_end = end,
                    meas_sampl = meas_sampl
                    )
                    
        Sim.building = bldg+'_'+model_id
        
        #Sim.fmupath_mpc = os.path.join(Sim.simu_path, 'fmus',community, 'Tutorial_'+model_id+'_'+model_id+'.fmu')
        
        Sim.fmupath_emu = os.path.join(Sim.simu_path, 'fmus', community, community+'_'+bldg+'_'+bldg+'_Models_'+bldg+'_House_mpc.fmu')
        
        Sim.fmupath_ref = os.path.join(Sim.simu_path, 'fmus', community, community+'_'+bldg+'_'+bldg+'_Models_'+bldg+'_House_PI.fmu')
        
        Sim.moinfo_emu = (os.path.join(Sim.mod_path, community, bldg,bldg+'_Models',bldg+'_House_mpc.mo'),  community+'.'+bldg+'.'+bldg+'_Models.'+bldg+'_House_mpc',
        {}
        )
        
        Sim.moinfo_emu_ref = (os.path.join(Sim.mod_path, community, bldg,bldg+'_Models',bldg+'_House_PI.mo'),   community+'.'+bldg+'.'+bldg+'_Models.'+bldg+'_House_PI',
        {}
        )
        
        if emulate == 1:
            # Initialise exogenous data sources
            if i == 1:
                Sim.update_weather(start, end)
                index = pd.date_range(start, end, freq = meas_sampl+'S', tz=Sim.weather.tz_name)
            else:
                Sim.weather = Sim_list[i-2].weather
        
            #Sim.sim_start= '1/1/2017 00:00'
            Sim.get_control()
            #Sim.sim_start= start
            Sim.get_other_input(start,end)
            Sim.get_constraints(start,end,upd_control=1)
            
            Sim.param_file = os.path.join(Sim.simu_path,'csvs','Parameters_R2CW.csv')
            Sim.get_params()
        
            if i > 1:
                Sim.control = Sim_list[i-2].control
                store_namespace(os.path.join(folder, 'sysid_control_'+Sim.building+'_'+mon), Sim.control)
            else:
                store_namespace(os.path.join(folder, 'sysid_control_'+Sim.building+'_'+mon), Sim.control)
          
                
            # Initialise models
            Sim.init_models(use_ukf=1, use_fmu_mpc=1, use_fmu_emu=1) # Use for initialising
        
        # Add to list of simulations
        Sim_list.append(Sim)
    
    
    index = pd.date_range(start, end, freq = meas_sampl+'S')
    train_dict = {}
    test_dict = {}
    
    results_dict = {}
    
    for Sim in Sim_list:
        if emulate == 1:
            # Emlate  to get data
            emulate_jmod(Sim.emu, Sim.meas_vars_emu, Sim.meas_sampl, start, end)
            
            # Handle data
            print(Sim.emu.display_measurements('Measured'))
    
            measurements = Sim.emu.display_measurements('Measured')
    
            index = pd.to_datetime(measurements.index)
            measurements.index = index 

            weather = Sim.weather.display_data().resample(meas_sampl+'S').ffill()
            #print(weather)
            weather.index = index
            
            df = pd.concat([measurements, weather],axis=1)[start:end]
            
            df['PowerCompr'] = df['PowerCompr']/1000.0
            
            df['TAir'] = df['TAir']-273.15
            
            for j in range(1,20):
                df['T_in_delay'+str(j)] = df['TAir'].shift(periods=j)
                df['PowerCompr_delay'+str(j)] = df['PowerCompr'].shift(periods=j)
                df['weaTDryBul_delay'+str(j)] = df['weaTDryBul'].shift(periods=j)
                df = df.fillna(method='bfill')
            
            # Remove the lags from the beginning
            train_start = datetime.datetime.strptime(start, '%m/%d/%Y %H:%M:%S') + datetime.timedelta(seconds = 10*float(meas_sampl))
            
            # Split the dataset
            df_train = df[train_start:train_end]
            df_test = df[valid_start:valid_end]
  
            train_dict[bldg] = df_train
            test_dict[bldg] = df_test
            
            store_namespace(os.path.join(folder, 'all_data_'+mon+'_'+Sim.building), df)
            store_namespace(os.path.join(folder, 'train_data_'+mon+'_'+Sim.building), df_train)
            store_namespace(os.path.join(folder, 'test_data_'+mon+'_'+Sim.building), df_test)

        else:
            df = load_namespace(os.path.join('results_sysid_test_'+mon, 'all_data_'+mon+'_'+Sim.building))
            
            # Remove the lags from the beginning
            #train_start = start.strftime('%m/%d/%Y %H:%M:%S')
            train_start = datetime.datetime.strptime(start, '%m/%d/%Y %H:%M:%S') + datetime.timedelta(seconds = 10*float(meas_sampl))
            
            for j in range(1,20):
                df['T_in_delay'+str(j)] = df['TAir'].shift(periods=j)
                df['PowerCompr_delay'+str(j)] = df['PowerCompr'].shift(periods=j)
                df['weaTDryBul_delay'+str(j)] = df['weaTDryBul'].shift(periods=j)
                df['weaHGloHor_delay'+str(j)] = df['weaHGloHor'].shift(periods=j)
                df = df.fillna(method='bfill')
            
            # Split the dataset
            df_train = df[train_start:train_end]
            df_test = df[valid_start:valid_end]
  
            train_dict[bldg] = df_train
            test_dict[bldg] = df_test
            
            store_namespace(os.path.join(folder, 'train_data_'+mon+'_'+Sim.building), df_train)
            store_namespace(os.path.join(folder, 'test_data_'+mon+'_'+Sim.building), df_test)
            
                        
            #print(df_train['weaTDryBul'])

        
        '''Identify parameters '''
        
        train_data = df_train

        i = 0
        for case in features_dict.keys():
            while True:
                try:
                    features = features_dict[case]
                    
                    feats = features + [target]
                    
                    
                    Sim.init_ARX_model(features, target, train_data)
                    Sim.ARX_model.evaluate()
                   
                    # Make some predictions
                    test_x = df_train[features].values 
                    Sim.ARX_model.predict(test_x)
                    preds = Sim.ARX_model.predictions
                    
                    if not os.path.exists(os.path.join(folder,case)):
                        os.makedirs(os.path.join(folder,case))
                    
                    store_namespace(os.path.join(folder, case, 'sysid_ARXmodel_'+mon+'_'+Sim.building), Sim.ARX_model)
                    store_namespace(os.path.join(folder, case, 'sysid_ARXparams_results_'+mon+'_'+Sim.building), Sim.ARX_model.fit_results.params)
                    store_namespace(os.path.join(folder, case, 'sysid_ARXparams_IDpreds_'+mon+'_'+Sim.building), preds)
                    
                    results_dict[case+'_'+Sim.building] = {}
                    results_dict[case+'_'+Sim.building]['AIC'] = Sim.ARX_model.fit_results.aic
                    results_dict[case+'_'+Sim.building]['BIC'] = Sim.ARX_model.fit_results.bic 
                    results_dict[case+'_'+Sim.building]['MSE-total'] = Sim.ARX_model.fit_results.mse_total 
                    results_dict[case+'_'+Sim.building]['MSE-model'] = Sim.ARX_model.fit_results.mse_model 
                    results_dict[case+'_'+Sim.building]['MSE-resid'] = Sim.ARX_model.fit_results.mse_resid 
                    results_dict[case+'_'+Sim.building]['R2-ID'] = Sim.ARX_model.fit_results.rsquared 
                    
                    ''' Plot for checks 
                    plt.figure(figsize=(12,6))
                    plot_index = df_train.index
                    
                    plt.plot(plot_index, preds,label='predictions_'+Sim.building)
                    plt.plot(plot_index, df_train['TAir'].values, label='true_'+Sim.building)
                    plt.legend()
                    plt.savefig(os.path.join(folder, case, Sim.building+'_'+mon+"_id.png"), bbox_inches="tight")
                    #plt.savefig(os.path.join(folder, analysis_folder, Sim.building+ "_valid.pdf"), bbox_inches="tight")
                    plt.close()
                    '''
                    '''
                    if i == 0:
                        Plot for checks - ACF 
                        plt.figure(figsize=(12,6))
                        #plt.plot(Sim.ARX_model.acf)
                        plot_acf(train_data[target].values)
                        plt.savefig(os.path.join(folder, Sim.building+'_'+mon+"_acf.png"), bbox_inches="tight")
                        #plt.savefig(os.path.join(folder, analysis_folder, Sim.building+ "_valid.pdf"), bbox_inches="tight")
                        #plt.show()
                        plt.close()
                        
                        Plot for checks - PACF 
                        plt.figure(figsize=(12,6))
                                
                        plot_pacf(train_data[target].values, lags=40, method='ols')
                        #plt.plot(Sim.ARX_model.pacf)
                        #plt.plot(Sim.ARX_model.pacf_confint, color='red')
                        plt.savefig(os.path.join(folder, Sim.building+'_'+mon+"_pacf.png"), bbox_inches="tight")
                        #plt.savefig(os.path.join(folder, analysis_folder, Sim.building+ "_valid.pdf"), bbox_inches="tight")
                        #plt.show()
                        plt.close()
                        '''
                    
                    
                    '''Validate'''
                    # Make some predictions
                    test_x = df_test[features].values 
                    Sim.ARX_model.predict(test_x)
                    preds = Sim.ARX_model.predictions
                    
                    mse = mean_squared_error(df_test['TAir'].values, preds)
                    rscore = r2_score(df_test['TAir'].values, preds)
                    
                    ''' Plot for checks 
                    plt.figure(figsize=(12,6))
                    plot_index = df_test.index
                    
                    plt.plot(plot_index, preds,label='predictions_'+Sim.building)
                    plt.plot(plot_index, df_test['TAir'].values, label='true_'+Sim.building)
                    plt.legend()
                    plt.savefig(os.path.join(folder, Sim.building+'_'+mon+"_valid.png"), bbox_inches="tight")
                    plt.close()
                    '''
                    #plt.savefig(os.path.join(folder, Sim.building+ "_valid.pdf"), bbox_inches="tight")
                    
                    print('Mean squared error - validation')
                    print(mean_squared_error(df_test['TAir'].values, preds))

                    print('R^2 score - validation')
                    print(r2_score(df_test['TAir'].values, preds))
                    
                    
                    results_dict[case+'_'+Sim.building]['MSE-model-valid'] = mse
                    results_dict[case+'_'+Sim.building]['R2-valid'] = rscore 
                    
                    store_namespace(os.path.join(folder, case, 'sysid_ARXparams_validpreds_'+mon+'_'+Sim.building), preds)
                    store_namespace(os.path.join(folder, case, 'sysid_ARXparams_validMSE_'+mon+'_'+Sim.building), mse)
                    store_namespace(os.path.join(folder, case, 'sysid_ARXparams_validR2_'+mon+'_'+Sim.building), rscore)
                       
                    break
                except:
                    print('%%%%%%%%%%%%%%%%%% Failed, trying again! %%%%%%%%%%%%%%%%%%%%%%')
                    continue
                    
    
    results_pd = pd.DataFrame.from_dict(results_dict, orient='index')
    results_pd.to_csv(os.path.join(folder, 'model_selection_all.csv'))
    
        
        
        

