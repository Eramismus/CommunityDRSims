# Research simulation repository

This repository is for publishing code used for my research into model-predictive control and building energy flexibility. The research was conducted under London-Loughborough Centre for Doctoral Training in Energy Demand, funded by EPSRC ref. EP/L01517X/1. 

The repository is split based on research outputs:

1. A journal paper, accepted for publication in Applied Energy. [Link to L'boro University Repository](https://repository.lboro.ac.uk/articles/Contracted_energy_flexibility_characteristics_of_communities_Analysis_of_a_control_strategy_for_demand_response/11770464/1)
2. A conference paper, accepted and presented in IBPSA Building Simulation 2019 conference in Rome. [Link to L'boro University Repository](https://repository.lboro.ac.uk/articles/Delivery_of_contracted_energy_flexibility_from_communities/9459845) 
3. Master of Research Thesis, submitted 2018 for Loughborough University.

## Journal paper

#### Contracted energy flexibility characteristics of communities: Analysis of a control strategy for demand response

The folder is called journal_paper, which presents the code for the journal paper accepted for Applied Energy. Simulator_HP_mod3.py contains python functions for running different types of simulations and to initialise the models from the fmus. The models are in the fmu-folder. Constraints for optimisation are stored as pickle-files in the constraints-folder. funcs.py contains functions to perform basic operations such as read and save variables. R2C2_model.py is class used to handle the ARX-models used for the MPC. 

Below is more information about the different steps.

### System Identification
sys_id_arx_jan.py provides an example of how the ARX-model identification was performed for January case day. A set of models evaluated for each endogenous lag configuration. The combinations were formed by forward addition of different exogenous parameters in order to each endogenous lag configuration. In total 40 ARX-model configurations were evaluated for each building.

### Demand Response Cases
1. Decentralised, demand response, pulp_dr_30bldgs.py
2. Centralised, demand response, pulp_dr_centr_30bldgs.py

### Reference Cases
1. Minimise energy consumption, decentralised, pulp_enemin_30bldgs.py
2. PI control in each house, PISim.py.

### Analysis
Analysis is done in two phases: 
1. Analyse each case including references are analysed with analyser_cen_30bldgs.py, analyser_dec_30bldgs.py or analyser_dec_enemin_30bldgs.py
2. Analyse all cases together and generate plots and csvs for communication, results in the results-folder.

### Results
Under results-folder key tables and figures used in the paper are included as files.

Final note: the code is poorly commented as it was written "on the fly" for research purposes. For any questions, you can just contact the author, Eramismus.

## IBPSA paper

The folder is ibpsa_paper. Simulator_HP_mod2.py and Simulator_HP_mod3.py contains python functions for running different types of simulations and to initialise the models from the fmus or mo-files.

funcs.py contains functions to perform basic operations such as read and save variables etc.

### Demand Response Cases
1. Decentralised, demand response, flexmpc_10bldgs.py
2. Centralised, demand response, flexmpc_10bldgs_centr.py
3. Decentralised, demand response without real building emulation, flexmpc_10bldgs_noemu.py
4. Centralised, demand response without real building emulation, flexmpc_10bldgs_centr_noemu.py

### Reference Cases
1. Minimise energy consumption, decentralised, flexmpc_10bldgs_enemin.py
2. PI control in each house, PISim.py.

### Analysis
Analysis is done in two phases: 
1. Analyse each case including references are analysed with analyser_decentr_case.py, analyser_centr_case or analyser_decentr_enemin_case.py
2. Analyse all cases together and generate plots and csvs for communication with analyser_drcases.py and analyser_ref.py

To save space only results for January are in the folder.

Final note: the code is poorly commented and it is currently in as it was written "on the fly". For any questions, just contact the author, Eramismus.

## Thesis

Simulator.py contains python functions for running different types of simulations.

funcs.py contains functions to perform basic operations such as read and save variables etc.

Simulation cases (structure, objective function, file):
1. Decentralised, minimise energy consumption, minene_opt.py
2. Centralised, minimise energy consumption, minene_centr_opt.py
3. Decentralised, minimise cost, mincost_opt.py
4. Centralised, minimise cost, mincost_centr_opt.py
5. Decentralised, minimise cost with DR, minDRcost_decentr_opt.py
6. Centralised, minimise cost with DR, minDRcost_centr_opt.py
7. Hierarchical, minimise cost with DR, minDRcost_hierarchical_opt.py

## General Remarks
Working environment used (follow installation instructions of respective software):
MPCPy-0.1.0
Python 2.7 32-bit
JModelica.org 2.1
Windows 10

Modifications made into standard distribution of MPCPy included under mpcpy_mods. Currently the paths in this repository are not relative, i.e. need to be adjusted manually. Sorry for that and the messy code :)

For more information you can contact me, Eramismus.