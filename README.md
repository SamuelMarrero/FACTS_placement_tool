# FACTS_placement_tool
A FACTS placement tool based on hourly demand data and Pareto optimality

This tool divided into two different parts. The first one is intended to automate PSS-E to perform power flow simultations (FACTS_placement_Demand-variation.py). The second one is intended to compute the indicators and execute the decision making algorithm (FACT_placer\pareto_calculation.py).

Two configuration files need to be provided. These files must have the same name, which is the name of the "study". The file whose extension is ".test" is intended to configure the P-V study and the FACTS device implementation. The one whose extension is ".scen" includes the information needed to create the demand scenarios. Demand data shoud be provided in a ".csv" file placed in the proper directory. A set of 8760 substation demand scenarios is provided, which have been obtained from (URL: https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Distribution-zone-substation-data).

The results of the simulations will be stored in "\studies\name_of_the_study". A ".sav" file is created for each simulation, as well as data file. Data files are used by paret_calculation.py to compute indices and perform the placement selection.

# FACTS_placement_Demand-variation.py configuration

The name of the study must be provided at the begining of the script. 
Two flags may be used. "Redicrect_silence = 1" reduces the amount of information printed from PSS-E to accelerate the computation. "peak_valley = 1" may be used to simulate only peak and valley scenarios. 
Two types of devices may be placed by alternatively choosing "FACTS" or "GENERATOR" in the ".test" file. The FACTS device is simulated as a synchronous compensator to emulate a STATCOM. Other types of FACTS devices may be simulated providing adequate parameters values.

# FACT_placer\pareto_calculation.py configuration

The name of the study, the path where it is placed and the name of the ".sav" file used in the simulations need to be provided at the begining of the script.
The number of demand scenarios to be considered needs to be provided. Different sampling methods may be used.
Voltage deviation, loading margin, active and reactive power losses, voltage angle deviation, greenhouse emissions and operation costs are calculated. Coefficient values need to be provided for greenhouse gases emissions and operation costs estimation using polinomial equations.
An index selection method is used to select two indices according to the Mutual Information they share.
The results of the placement process in terms of the selected indices are scatter plotted. The Pareto set is also plotted.

