""" Second version, make the code easier and more modifiable """
# Define the main programme

#Import other required packages
import pandas as pd
import matplotlib.pyplot as plt
import dill
import os
import numpy as np
#import time

# Import classes from mpcpy
from mpcpy import variables
from mpcpy import units
from mpcpy import exodata
from mpcpy import systems_mod as systems
from mpcpy import models_mod as models
from mpcpy import optimization_mod3 as optimization

from funcs import emulate_jmod
#from funcs import simulate_mpc_JModelica
#from funcs import simulate_mpc_UKF
from funcs import store_namespace
from funcs import load_namespace

""" Class for handling the simulation workflow """

class SimHandler():
    def __init__(self, sim_start, sim_end, meas_sampl):
        print("Initializing SimHandler")
        
        # Measurement period for simulations
        self.sim_start = sim_start
        self.sim_end = sim_end
        
        # %%%%%%%%%%%% System and MPC models %%%%%%%%%%%%%%%%%%%%%%%%
        self.building = ""
        self.mod_path = "path_to_models"
        self.simu_path = "path_to_sims"
        
        self.fmupath_mpc = os.path.join(self.simu_path, 'Tutorial_RC.fmu')
        self.fmupath_emu = os.path.join(self.simu_path, 'ResidentialCommunity_Detached0_Detached0_Models_Detached0_House_mpc.fmu')
        
        
        # Model for emulation
        self.moinfo_emu = (os.path.join(self.mod_path, 'ResidentialCommunityUK_new\Detached0\Detached0_Models\Detached0_House_mpc.mo'),
                        'ResidentialCommunity.Detached0.Detached0_Models.Detached0_House_mpc',
                        {}
                        ) 
        
        # Model for MPC
        self.moinfo_mpc = (os.path.join(self.simu_path, 'Tutorial_R2CW.mo'),
                'Tutorial_R2CW.R2CW_HP',
                {}
                )
        
        # Set parameters
        self.weather_file = os.path.join(self.simu_path, 'weather', 'Nottingham_TRY.epw')
        self.weather_file_mos = os.path.join(self.simu_path, 'weather', 'Nottingham_TRY.mos')

        self.price_varmap = {'Price': ('Price',units.cents_kWh)
            }
        self.flex_cost_varmap = {'FlexCost': ('FlexCost',units.cents_kWh)
            }
        self.ref_profile_varmap = {'RefProfile': ('RefProfile',units.cents_kWh)
            }
        
        self.rho_varmap = {'rho': ('rho',units.unit1)
            }
        
        self.addobj_varmap = {'TAir_Slack': ('TAir_Slack',units.unit1)
            }
        
        self.contr_varmap = {'HPPower': ('HPPower',units.W)
            }
        
        self.other_varmap = {'ConvGain1': ('ConvGain1',units.W),
            'RadGain': ('RadGain',units.W)
            }
        
        self.control_file = os.path.join(self.simu_path,'csvs','ControlSignal_HP.csv')
        self.price_file = os.path.join(self.simu_path,'csvs','PriceSignal.csv')
        self.param_file = os.path.join(self.simu_path,'csvs','Parameters_R2CW.csv')
        self.constraint_csv = os.path.join(self.simu_path,'csvs','Constraints_HP.csv')
        #self.meas_file = 'sim_results.csv'
        
        # Id period
        self.id_start = '1/1/2017 00:00:00'
        self.id_end = '1/6/2017 00:00:00'
        
        # Validation period
        self.val_start  = '1/1/2017 00:00:00'
        self.val_end = '1/4/2017 12:00:00'
        
        # Optimization period
        self.opt_start  = '1/4/2017 12:00:00'
        self.opt_end = '1/5/2017 00:00:00'

        # measurement sample rate in seconds
        self.meas_sampl = int(meas_sampl)
        
        self.meas_vars_emu = {'TAir': {}, 
                            'PowerCompr': {}
                            }

        self.meas_vars_mpc = {'TAir': {}, 
                            }

        self.meas_vars_emu['TAir']['Sample'] = variables.Static('sample_rate_Tzone',
                    self.meas_sampl, # sample rate
                    units.s
                    ) # units
        
        self.meas_vars_emu['PowerCompr']['Sample'] = variables.Static('sample_rate_compr',
                    self.meas_sampl, # sample rate
                    units.s
                    ) # units
                    
        self.meas_vars_mpc['TAir']['Sample'] = variables.Static('sample_rate_Tzone',
                    self.meas_sampl, # sample rate
                    units.s
                    ) # units
                        
        self.meas_varmap_emu = {'TAir': ('TAir',units.K), 
                            'PowerCompr': ('PowerCompr',units.W)
                            }
                            
        self.meas_varmap_mpc = {'TAir': ('TAir',units.K), 
                            }
        
        # List of estimation parameters
        self.est_params = ['TAir']
                
        # Constraints for optimization
        self.optcon_varmap = {
                        'T_min': ('Slack', 'GTE', units.degC),
                        'T_max': ('Slack', 'LTE', units.degC),
                        'HPPower_min': ('HPPower', 'GTE', units.W),
                        'HPPower_max': ('HPPower', 'LTE', units.W)
                        }
                        
        self.target_variable = 'HPPower'
        self.slack_var = ['TAir_Slack']
        
        # %%%%%%%%%%%% Reference Models %%%%%%%%%%%%%%%%%%%%%%%%
        self.moinfo_emu_ref = (os.path.join(self.mod_path, 'ResidentialCommunity\Detached0\Detached0_Models'),
                    'ResidentialCommunity.Detached0.Detached0_Models.Detached0_House_PI',
                    {}
                    )
        
        self.fmupath_ref = os.path.join(self.simu_path, 'refit_project_firstorder_ibpsa_Building01_Building01_Models_Building01_SingleDwelling_PI.fmu')
        
        self.meas_vars_ref = {'TAir': {}, 
                            'PowerCompr': {}
                            }

        self.meas_vars_ref['TAir']['Sample'] = variables.Static('sample_rate_Tzone',
                    self.meas_sampl, # sample rate
                    units.s
                    ) # units
        
        self.meas_vars_ref['PowerCompr']['Sample'] = variables.Static('sample_rate_compr',
                    self.meas_sampl, # sample rate
                    units.s
                    ) # units
        
        self.meas_sampl_ref = 3600
        
        self.contr_varmap_ref = {'SetPoint': ('SetPoint',units.K)
        }
        
        self.start_temp = np.random.uniform(18,22)
        self.ref_profile_file = 'controlseq_Detached_0_R2CW_MinEne'
        
    # Get other inputs
    def get_other_input(self,start,end):
        
        index = pd.date_range(start, end, freq = str(self.meas_sampl)+'S')
        # Zero signal
        radgain = pd.Series(np.random.rand(len(index))*100,index=index)
        zero_signal = pd.Series(np.zeros(len(index)),index=index)
        
        self.other_input = exodata.ControlFromCSV(self.control_file,
                                        self.other_varmap,
                                        tz_name = self.weather.tz_name)
        self.other_input.data = {"ConvGain1": variables.Timeseries('ConvGain1', zero_signal,units.W,tz_name=self.weather.tz_name),
            "RadGain": variables.Timeseries('RadGain', radgain,units.W,tz_name=self.weather.tz_name)
            }
        store_namespace('other_input_'+self.building,self.other_input)
    
        
    # Get control data from csv for identification
    def get_control(self):
        self.control = exodata.ControlFromCSV(self.control_file,
                                        self.contr_varmap,
                                        tz_name = self.weather.tz_name)
        index = pd.date_range(self.sim_start, self.sim_end, freq = str(self.meas_sampl)+'S')
        # Random control signal
        control_signal1 = pd.Series(np.random.rand(len(index))*0.1,index=index)
        # Define control data
        self.control.data = {"HPPower": variables.Timeseries('HPPower', control_signal1,units.W,tz_name=self.weather.tz_name)
        }

        store_namespace('control_'+self.building,self.control)
    
    # Get parameter data
    def get_params(self):
        self.parameters = exodata.ParameterFromCSV(self.param_file)
        self.parameters.collect_data()
        store_namespace('params_'+self.building,self.parameters)
    
    def get_constraints(self,start,end, **kwargs):  
        self.constraints = exodata.ConstraintFromCSV(self.constraint_csv, self.optcon_varmap)
        index = pd.date_range(start, end, freq = str(self.meas_sampl) +'S')
        # Set points based on UCL paper on set points!!! High variability? Commercial?
        mor_start = np.random.randint(7,8)
        mor_end = np.random.randint(7,8)
        eve_start =  np.random.randint(17,18)
        eve_end = np.random.randint(19,23)
        dem_temp = np.random.randint(19,21)
        t_min = pd.Series(16+273.15,index=index) # Contracted?
        t_max = pd.Series(24+273.15,index=index) # Contracted?
        
        ppl_nr = np.random.randint(1,5)
        radgain = pd.Series(np.random.rand(len(index))*100,index=index)
        zero_signal = pd.Series(np.zeros(len(index)),index=index)
        control_signal1 = pd.Series(np.random.rand(len(index))*0,index=index)
        
        for i in index:
            if i.hour >= mor_start and i.hour <= mor_end:
                t_min[i] = dem_temp-1+273.15
                t_max[i] = dem_temp+1+273.15
                radgain[i] = np.random.uniform(0,1)*50*np.random.randint(0,ppl_nr)
                control_signal1[i] = np.random.uniform(0.2,0.6)
            if i.hour >= eve_start and i.hour <= eve_end:
                t_min[i] = dem_temp-1+273.15
                t_max[i] = dem_temp+1+273.15
                radgain[i] = np.random.uniform(0,1)*50*np.random.randint(0,ppl_nr)
                control_signal1[i] = np.random.uniform(0.2,0.6)
        pheat_min = pd.Series(0,index=index)
        pheat_max = pd.Series(1,index=index)
        
        if 'set_change' in kwargs.keys():
            set_change = kwargs['set_change']
        else:
            set_change = 1
        #print(t_min)
        
        
        self.constraints.data = {
                'HPPower': 
                {'GTE': variables.Timeseries('HPPower_GTE', pheat_min, units.W, tz_name=self.weather.tz_name), 
                'LTE': variables.Timeseries('HPPower_LTE', pheat_max, units.W,tz_name=self.weather.tz_name)},
                'TAir': 
                {'Slack_GTE': variables.Timeseries('TAir_Slack_GTE', t_min, units.K, tz_name=self.weather.tz_name),
                'Slack_LTE': variables.Timeseries('TAir_Slack_LTE', t_max, units.K, tz_name=self.weather.tz_name)}
                }
        
        self.constraints_reg = self.constraints # Regular constraints
       
        self.other_input.data = {"ConvGain1": variables.Timeseries('ConvGain1', zero_signal,units.W,tz_name=self.weather.tz_name),
            "RadGain": variables.Timeseries('RadGain', radgain,units.W,tz_name=self.weather.tz_name)
            }
        
        if "upd_control" in kwargs.keys():
            self.control.data = {"HPPower": variables.Timeseries('HPPower', control_signal1,units.W,tz_name=self.weather.tz_name)}
            store_namespace('control_'+self.building,self.control)
        
        store_namespace('constraints_'+self.building,self.constraints)
        store_namespace('other_input_'+self.building,self.other_input)
    
    def get_DRinfo(self,start,end,**kwargs):
        self.price = exodata.PriceFromCSV(self.price_file,
                                        self.price_varmap,
                                        tz_name = self.weather.tz_name)
        self.flex_cost = exodata.PriceFromCSV(self.price_file,
                                        self.flex_cost_varmap,
                                        tz_name = self.weather.tz_name)
        self.rho = exodata.PriceFromCSV(self.price_file,
                                        self.rho_varmap,
                                        tz_name = self.weather.tz_name)
        self.ref_profile = exodata.ControlFromCSV(self.control_file,
                                        self.ref_profile_varmap,
                                        tz_name = self.weather.tz_name)
        self.addobj = exodata.ControlFromCSV(self.price_file,
                                        self.addobj_varmap,
                                        tz_name = self.weather.tz_name)
        index = pd.date_range(start, end, freq = str(self.meas_sampl)+'S')
        # Random control signal
        index = pd.date_range(start, end, freq = '1800S') #half-hourly price signal
        price_signal = pd.Series(np.random.rand(len(index))*40,index=index)
        
        for i in index:
            if i.hour >= 7 and i.hour <= 11:
                price_signal[i] = np.random.uniform(0.8,1)*60
            if i.hour >= 18 and i.hour <= 19:
                price_signal[i] = np.random.uniform(0.9,1)*90
            if i.hour >= 20 and i.hour <= 22:
                price_signal[i] = np.random.uniform(0.8,1)*60
        
        index = pd.date_range(start, end, freq = str(self.meas_sampl)+'S')
        flex_signal = pd.Series(0,index=index)
        k = pd.Series(1,index=index)
        ref_signal = pd.Series(0.3,index=index)
        rho_signal = pd.Series(1000,index=index)
        #ref_signal = load_namespace(self.ref_profile_file)[0]
        #print(type(ref_signal))
        
        self.price.data = {"pi_e": variables.Timeseries('pi_e', price_signal,units.cents_kWh,tz_name=self.weather.tz_name)
        }
        
        self.flex_cost.data = {"flex_cost": variables.Timeseries('flex_cost', flex_signal,units.cents_kWh,tz_name=self.weather.tz_name)
        }
        
        self.ref_profile.data = {"ref_profile": variables.Timeseries('ref_profile', ref_signal, units.W,tz_name=self.weather.tz_name)
        }
        
        self.rho.data = {"rho": variables.Timeseries('rho', rho_signal, units.unit1,tz_name=self.weather.tz_name)
        }
        
        self.addobj.data = {}
        
        for item in self.slack_var:
            self.addobj.data[item] = variables.Timeseries(item, pd.Series(0,index=index), units.unit1,tz_name=self.weather.tz_name)
            
        #print(self.addobj.data)
        
        store_namespace('price_'+self.building,self.price)
        store_namespace('flex_cost_'+self.building,self.flex_cost)
        store_namespace('ref_profile_'+self.building,self.ref_profile)
        store_namespace('rho_'+self.building,self.rho)

            
    def update_weather(self,start,end):
        # Next we get exogenous data from epw-file
        print("%%%%%%---Getting weather data---%%%%%%%%%%%%%")
        self.weather = exodata.WeatherFromEPW(self.weather_file)
        self.weather.collect_data(start, end)
        store_namespace('weather',self.weather.display_data())
    
    def update_params(self, param, value, unit):
        # Add a new value fixed value to parameters
        
        self.parameters.data[param] = {'Free': variables.Static('FreeOrNot', False, units.boolean), 
        'Minimum': variables.Static('Min', value, unit), 
        'Covariance': variables.Static('Covar', 0, unit), 
        'Value': variables.Static(param, value, unit), 
        'Maximum': variables.Static('Max', value, unit)
        }
        
        #self.other_input.data[param] = variables.Static(param, value, unit)
        store_namespace('params_upd_'+self.building, self.parameters)


    def init_models(self,use_ukf,use_fmu_mpc,use_fmu_emu):
        print("Initialising models")
        if use_fmu_emu == 0:
            print("%%%%%%%%%%%%%%%%% Compiling new FMUs for emulation %%%%%%%%%%%%%%%%")
            self.emu = systems.EmulationFromFMU(self.meas_vars_emu,
                                        moinfo=self.moinfo_emu,
                                        weather_data = self.weather.data,
                                        control_data = self.control.data,
                                        other_inputs = self.other_input.data,
                                        tz_name = self.weather.tz_name
                                        )
        else:
            print("%%%%%%%%%%%%%% Using existing FMUs for emulation %%%%%%%%%%%%%%%%%%%%%%%")
            print("Emulation FMU: " + str(self.fmupath_emu))
            self.emu = systems.EmulationFromFMU(self.meas_vars_emu,
                                        fmupath=self.fmupath_emu,
                                        weather_data = self.weather.data,
                                        control_data = self.control.data,
                                        other_inputs = self.other_input.data,
                                        tz_name = self.weather.tz_name
                                        )
        if use_ukf ==1:
            if use_fmu_mpc==1:
                print("%%%%%%%%%%%%%% Using existing FMUs for MPC %%%%%%%%%%%%%%%%%%%%%%%")
                print("MPC FMU: " + str(self.fmupath_mpc))
                self.mpc = models.Modelica(models.UKF,
                            models.RMSE,
                            self.emu.measurements,
                            fmupath = self.fmupath_mpc,
                            parameter_data = self.parameters.data,
                            weather_data = self.weather.data,
                            control_data = self.control.data,
                            other_inputs = self.other_input.data,
                            tz_name = self.weather.tz_name,
                            version = '1.0'
                            )
                self.mpc.moinfo = self.moinfo_mpc
            else:
                print("%%%%%%%%%%%%%%%%% Compiling new FMUs for MPC %%%%%%%%%%%%%%%%")
                self.mpc = models.Modelica(models.UKF,
                            models.RMSE,
                            self.emu.measurements,
                            moinfo = self.moinfo_mpc,
                            parameter_data = self.parameters.data,
                            weather_data = self.weather.data,
                            control_data = self.control.data,
                            other_inputs = self.other_input.data,
                            tz_name = self.weather.tz_name,
                            version = '1.0'
                            )
        else:                   
            if use_fmu_mpc==1:
                print("%%%%%%%%%%%%%% Using existing MPC FMUs for MPC %%%%%%%%%%%%%%%%%%%%%%%")
                print("MPC FMU: " + str(self.fmupath_mpc))
                self.mpc = models.Modelica(models.JModelica,
                            models.RMSE,
                            self.emu.measurements,
                            fmupath = self.fmupath_mpc,
                            parameter_data = self.parameters.data,
                            weather_data = self.weather.data,
                            control_data = self.control.data,
                            other_inputs = self.other_input.data,
                            tz_name = self.weather.tz_name,
                            )
                self.mpc.moinfo = self.moinfo_mpc
            else:
                print("%%%%%%%%%%%%%%%%% Compiling new MPC FMUs %%%%%%%%%%%%%%%%")
                self.mpc = models.Modelica(models.JModelica,
                            models.RMSE,
                            self.emu.measurements,
                            moinfo = self.moinfo_mpc,
                            parameter_data = self.parameters.data,
                            weather_data = self.weather.data,
                            control_data = self.control.data,
                            other_inputs = self.other_input.data,
                            tz_name = self.weather.tz_name
                            )

            #self.emu.collect_measurements(self.sim_start, self.sim_end)
                    
                    
    def sys_id(self):
        print("%%%% ---Starting system identification --- %%%%")
        # Emulate
        print('%%%%---Measured variables---%%%%')
        print(self.emu.display_measurements('Measured'))
        
        #Simulate mpc
        self.mpc.simulate(self.id_start, self.id_end)
        print('%%%%---Simulated variables---%%%%')
        print(self.mpc.display_measurements('Simulated'))
        
        print('%%%%---Estimating parameters---%%%%')
        self.mpc.estimate(self.id_start,self.id_end, self.est_params)
        
        # Just print the parameters 
        print("%%%---Estimated Parameters----%%%%")
        for key in self.mpc.parameter_data.keys():
            print(key, self.mpc.parameter_data[key]['Value'].display_data())
        store_namespace('est_params_'+self.building,self.mpc.parameter_data)
    

            
    def validate(self, val_start, val_end, id , plotfile):
        print("%%%%%%%%%%%%%%%----Starting another validation with estimated parameters ---%%%%%%%%%%%%%")  
        # Validate against another time period
        self.emu.collect_measurements(val_start, val_end)
        self.mpc.measurements = self.emu.measurements
        self.mpc.validate(val_start, val_end, plotfile, plot=1)
        plt.clf() # clear the figure
        
        print("----The RMSE------")
        for key in self.mpc.RMSE.keys():
            print(self.mpc.RMSE[key].display_data())
        store_namespace('RMSE_'+id+'_'+self.building,self.mpc.RMSE)
        store_namespace('sim_'+id+'_'+self.building,self.mpc.display_measurements('Simulated'))
        store_namespace('meas_'+id+'_'+self.building,self.emu.display_measurements('Measured'))

            
    def opt_control_minEnergy(self):
        # Instantiate optimization problem

        self.opt_problem = optimization.Optimization(self.mpc,
                                                optimization.EnergyMin,
                                                optimization.JModelica,
                                                self.target_variable,
                                                constraint_data = self.constraints.data
                                                )
        
        self.opt_options = {}
        self.opt_options['blocking_factors'] = [1]*10
        self.opt_problem.set_optimization_options(self.opt_options)
        #self.opt_problem
        
        print("%%%%%%%%%%%%%%%----Starting Optimization of Control ---%%%%%%%%%%%%%")
        self.opt_problem.optimize(self.opt_start,self.opt_end,res_control_step=self.meas_sampl)

        print("----Optimization stats----")
        print(self.opt_problem.get_optimization_statistics())

        print("----Optimization outcome----")
        self.opt_controlseq = self.opt_problem.Model.control_data # dictionary with mpcpy-timeseries variables of controls
        print(self.opt_controlseq)
        store_namespace('opt_control_minenergy_'+self.building,self.opt_controlseq)
        
    def opt_control_minCost(self):
        # Instantiate optimization problem
        
        self.opt_problem = optimization.Optimization(self.mpc,
                                                optimization.EnergyCostMin,
                                                optimization.JModelica,
                                                self.target_variable,
                                                constraint_data = self.constraints.data,
                                                add_objective = self.slack_var
                                                )

        print("%%%%%%%%%%%%%%%----Starting Optimization of Control ---%%%%%%%%%%%%%")
        self.opt_options = {}
        self.opt_options['blocking_factors'] = [1]*10
        self.opt_problem.set_optimization_options(self.opt_options)     
        self.opt_problem.optimize(self.opt_start,self.opt_end,
                                price_data = self.price.data, 
                                rho = self.rho.data,
                                addobj = self.addobj.data
                                )

        print("----Optimization stats----")
        print(self.opt_problem.get_optimization_statistics())

        print("----Optimization outcome----")
        self.opt_controlseq = self.opt_problem.Model.control_data # dictionary with mpcpy-timeseries variables of controls
        store_namespace('opt_control_mincost_'+self.building,self.opt_controlseq)
        
    
    def opt_control_minDRCost(self):
        # Instantiate optimization problem
        self.opt_problem = optimization.Optimization(self.mpc,
                                                optimization.DRCostMin,
                                                optimization.JModelica,
                                                self.target_variable,
                                                constraint_data = self.constraints.data,
                                                add_objective = self.slack_var
                                                )

        print("%%%%%%%%%%%%%%%----Starting Optimization of Control ---%%%%%%%%%%%%%")       
        self.opt_options = {}
        self.opt_options['IPOPT_options']={}
        self.opt_options['IPOPT_options']['obj_scaling_factor'] = 100
        self.opt_options['n_cp'] = 1
        self.opt_problem.set_optimization_options(self.opt_options)
        self.opt_problem.optimize(self.opt_start,self.opt_end,
                                    price_data = self.price.data,
                                    ref_profile = self.ref_profile.data,
                                    flex_cost = self.flex_cost.data,
                                    rho = self.rho.data,
                                    addobj = self.addobj.data
                                    )

        print("----Optimization stats----")
        print(self.opt_problem.get_optimization_statistics())

        print("----Optimization outcome----")
        self.opt_controlseq = self.opt_problem.Model.control_data # dictionary with mpcpy-timeseries variables of controls
        store_namespace('opt_control_DRcost_'+self.building,self.opt_controlseq)

    def emulate_opt(self,start,end):
        print("%%%---Emulating real system---%%%")
        self.emu.control_data = self.opt_controlseq
        emulate_jmod(self.emu, self.meas_vars_emu, self.meas_sampl, start, end)
        print("Updating measurements")
        self.mpc.measurements = self.emu.measurements

        store_namespace('optemu_'+self.building,self.emu.display_measurements('Measured'))

            
    def init_refmodel(self, use_fmu, use_const, const_path):
        print("%%%---Initialising Reference Simulation---%%%")  
                
        # Define control profile for reference
        self.control_ref = exodata.ControlFromCSV(self.control_file,
                                        self.contr_varmap,
                                        tz_name = self.weather.tz_name)
        index = pd.date_range(self.sim_start, self.sim_end, freq = str(self.meas_sampl_ref) +'S')
        
        if use_const == 1:
            t_set = load_namespace(os.path.join(const_path, 'constraints_'+self.building)).data['TAir']['Slack_GTE'].get_base_data()
        else:
            t_set = pd.Series(20+273.15,index=index)
            for i in index:
                if i.hour >= 6 and i.hour <= 9:
                    t_set[i] = 20 + 273.15
                if i.hour >= 17 and i.hour <= 23:
                    t_set[i] = 20 + 273.15
        # Define control data
        self.control_ref.data = {"SetPoint": variables.Timeseries('SetPoint', t_set,units.K,tz_name=self.weather.tz_name)
        }
        
        if use_fmu == 0:
            self.emuref = systems.EmulationFromFMU(self.meas_vars_ref,
                                        moinfo=self.moinfo_emu_ref,
                                        weather_data = self.weather.data,
                                        control_data = self.control_ref.data,
                                        other_inputs = self.other_input.data,
                                        tz_name = self.weather.tz_name
                                        )
        else:
            self.emuref = systems.EmulationFromFMU(self.meas_vars_ref,
                                        fmupath=self.fmupath_ref,
                                        weather_data = self.weather.data,
                                        control_data = self.control_ref.data,
                                        other_inputs = self.other_input.data,
                                        tz_name = self.weather.tz_name
                                        )
    def run_reference(self, start, end):
        print("%%%---Running Reference Simulation---%%%")   
            
        emulate_jmod(self.emuref,self.meas_vars_ref, self.meas_sampl_ref, start, end)
                
        #print(self.emuref.display_measurements('Measured')['TAir'])    
        #print(self.emuref.display_measurements('Measured')['HeatInput'])
        
        store_namespace('ref_power_'+self.building, self.emuref.display_measurements('Measured')['PowerCompr'])
        
        store_namespace('ref_temp_'+self.building, self.emuref.display_measurements('Measured')['TAir'])
        