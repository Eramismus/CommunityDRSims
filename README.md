# IBPSA paper

Simulator_HP_mod2.py and Simulator_HP_mod3.py contains python functions for running different types of simulations and to initialise the models from the fmus or mo-files.

funcs.py contains functions to perform basic operations such as read and save variables etc.

## Demand Response Cases
1. Decentralised, demand response, flexmpc_10bldgs.py
2. Centralised, demand response, flexmpc_10bldgs_centr.py
3. Decentralised, demand response without real building emulation, flexmpc_10bldgs_noemu.py
4. Centralised, demand response without real building emulation, flexmpc_10bldgs_centr_noemu.py

## Reference Cases
1. Minimise energy consumption, decentralised, flexmpc_10bldgs_enemin.py
2. PI control in each house, PISim.py.

## Analysis
Analysis is done in two phases: 
1. Analyse each case including references are analysed with analyser_decentr_case.py, analyser_centr_case or analyser_decentr_enemin_case.py
2. Analyse all cases together and generate plots and csvs for communication with analyser_drcases.py and analyser_ref.py

To save space only results for January are in the folder.

Final note: the code is poorly commented and it is currently in as it was written "on the fly". For any questions, just contact the author, Eramismus.

#Thesis

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

# General Remarks
Working environment used (follow installation instructions of respective software):
MPCPy-0.1.0
Python 2.7 32-bit
JModelica.org 2.1
Windows 10

Modifications made into standard distribution of MPCPy included under mpcpy_mods. Currently the paths in this repository are not relative, i.e. need to be adjusted manually. Sorry for that and the messy code :)

For more information you can contact me, Eramismus.