import psse34,psspy,os,csv
import numpy as np
import copy
from itertools import groupby

coma = ','
semicolomn = ';'


from demand_sorter import Sorter

cwd = os.getcwd()

def load_scenarios_config_data(program_path,data_path,study_name):
    os.chdir(program_path)
    study_file = os.path.join('', study_name + '.scen')
    scenarios_config_file = open(study_file, 'r')
    scenarios_config_data = scenarios_config_file.readlines()

    demand_scenarios_file_name = scenarios_config_data[3].rstrip('\n')
    wind_gen_scenarios_file_name = scenarios_config_data[6].rstrip('\n')

    size_scenarios = []
    number_of_size_scenarios = int(scenarios_config_data[9].rstrip('\n'))
    len_line = len(scenarios_config_data[12])
    config_data_line = scenarios_config_data[12].rstrip('\n')
    j = 0
    vector_element = 0
    for h in range(number_of_size_scenarios):
        vector = []
        partial_data = ''
        for q in range(6):
            if j <= len_line - 1:
                if config_data_line[j] != (','):
                    partial_data += config_data_line[j]
                    j += 1
                elif config_data_line[j] == (','):
                    vector.append(int(partial_data))
                    j += 1
                    vector_element += 1
                    break
            else:
                break
        size_scenarios.append(vector)

    scenarios_config_file.close()

    demand_data, demand_values= data_loader(data_path, demand_scenarios_file_name, coma )
    wind_gen_data, wind_gen_values = data_loader(data_path, wind_gen_scenarios_file_name, semicolomn)

    return demand_data,demand_values,wind_gen_data,wind_gen_values,number_of_size_scenarios,size_scenarios


def data_loader(data_path,data_file_name,delimiter):
    os.chdir(data_path)
    data = []
    with open(data_file_name, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for row in reader:
            data.append(row)

    data_np = np.array(data)
    nodes_names = data_np[0, :]
    demand_tstamps = data_np[:, 0]
    data_type = np.dtype('S8')
    data_values = np.array(data_np, dtype=data_type)
    #data_values = np.delete(data_values, (0), axis=0)
    data_values = np.delete(data_values, (0), axis=1)
    data_values = data_values.astype(np.float)

    return data, data_values

def WPP_scenarios_preparer(WPP_data,plant_power):

    np_data = np.array(WPP_data)
    data_max = np.max(np_data)
    WPP_data_scaled = np_data/data_max * plant_power


    return WPP_data_scaled

def demand_scenarios_preparer(demand_data,nodes_load,load_nodes,distribution):
    demand_data_size = demand_data.shape
    num_nodes_system = np.array(nodes_load).size
    num_nodes_demand = demand_data_size[1]
    exceeding_demand_nodes = num_nodes_demand - num_nodes_system
    for exceeding_node in range(exceeding_demand_nodes):
        demand_data = np.delete(demand_data,(num_nodes_demand - exceeding_node-1),axis = 1)

    nodes_active_power =[]
    nodes_PQ_ratio = []
    node = 0
    for nodes in nodes_load:
        nodes_active_power.append(nodes.real)
        nodes_PQ_ratio.append(nodes.imag/nodes.real)
        node += 1

    total_system_demand = np.sum(nodes_active_power)

    if distribution == 'fixed':
        load_share_data = np.divide(nodes_active_power,total_system_demand)
        total_demand_data = np.sum(demand_data, axis=1)
        peak_demand_data = np.max(total_demand_data)
        total_scaled_demand = total_demand_data*total_system_demand/peak_demand_data
        scaled_demand_data = np.transpose([node_load_share*total_scaled_demand for node_load_share in load_share_data])

    elif distribution == 'data':
        demand_data_sorted = Sorter.sort(demand_data, load_nodes, nodes_load)
        peak_node_demand = np.max(demand_data_sorted, axis = 0)
        peak_value = sum(peak_node_demand)
        scaling_coefficients = np.divide(nodes_active_power, peak_node_demand)
        scaled_demand_data = demand_data_sorted * scaling_coefficients
        total_scaled_demand = np.sum(scaled_demand_data, axis=1)
        peak_scaled = np.max(total_scaled_demand)
        load_share_data = np.divide(scaled_demand_data,total_scaled_demand[:, None])

    min_demand_pos = np.argmin(total_scaled_demand)
    max_demand_pos = np.argmax(total_scaled_demand)

    return scaled_demand_data,min_demand_pos,max_demand_pos,nodes_PQ_ratio,total_scaled_demand,load_share_data

def scenarios_frequencies_calculator(demand_data,len_sample,load_steps):

    P_valley = np.min(demand_data, axis=0)
    P_peak = np.max(demand_data, axis=0)

    loading_margin_data = np.divide(np.subtract(demand_data, P_valley), P_valley)
    loading_margin_max = np.max(loading_margin_data, axis=0)
    load_increment_step_percent = loading_margin_max / float(load_steps)
    load_increment_step_MVA = (P_peak - P_valley)/ float(load_steps)
    interval_width = (loading_margin_max / load_steps)
    values = np.zeros(load_steps)
    for step in range(load_steps):
        a = (step + 1)/float(load_steps)
        interval_max = ( a * loading_margin_max)
        interval_min = interval_max - interval_width
        for element in range(len_sample):
            value = loading_margin_data[element]
            if interval_min < value and value <= interval_max:
                values[step] += 1

    frequencies = np.divide(values,len_sample)

    return frequencies, load_increment_step_percent,load_increment_step_MVA

def generation_scenario_creator(xarray_wmod,xarray_bus,xarray_id,xarray_mach,generation_scenario,generation_scenarios,len_wmod,total_data):

    no_wind_turbines = 0
    wind_turbines_number_vector = []
    wind_turbines_pgen_vector = []
    wind_turbines_id_vector = []
    machine = 0
    # Busar que maquinas son aerogeneradores
    for j in range(len_wmod):
        if xarray_wmod[0][machine]!= 0:
            wind_turbines_number_vector.append( xarray_bus[0][machine])
            wind_turbines_id_vector.append(xarray_id[0][machine])
            wind_turbines_pgen_vector.append( xarray_mach[0][machine])
        machine += 1
     # Calcular la potencia renovable total
    len_wind_turbines = len(wind_turbines_pgen_vector)
    bsys_wind_turbines_number_vector = []
    total_ren_power = 0
    wind_turbine = 0
    for j in range(len_wind_turbines):
        total_ren_power += wind_turbines_pgen_vector[wind_turbine]
        bsys_wind_turbines_number_vector.append(wind_turbines_number_vector[wind_turbine])
        wind_turbine +=1
    # Calular la penetracion renovable
    if len_wind_turbines > 0:
        initial_conventional_generation = total_data[1] - total_ren_power
        desired_ren_fraction_percent = int(generation_scenarios[generation_scenario][0])
        desired_ren_fraction = desired_ren_fraction_percent*100**(-1)
        new_ren_pgen_MVA = desired_ren_fraction * initial_conventional_generation/(1-desired_ren_fraction)
        # Ajustar la generacion renovable a un valor de penetracion determinado

        ierr_subsystem = psspy.bsys(3, 0, [0.0], 0, [], len_wind_turbines, bsys_wind_turbines_number_vector, 0, [], 0, [])
        sid = 3
        value_all = 0
        apiopt = 0
        status = [0,1,0,0,2]
        scalval = [0.0,new_ren_pgen_MVA,0.0,0.0,0.0,0.0,0.0]
        ierr_scal2,totals,moto = psspy.scal_2 (sid,value_all,apiopt,status,scalval)
    elif len_wind_turbines == 0:
        no_wind_turbines = 1

    return no_wind_turbines


    # load = 0
    # subsys_load_MVA = 0
    # for k in range(len(subsys_load_vector[0])):
    #     subsys_load_MVA += subsys_load_vector[0][load]
    #     load += 1
    #
    # #   Calcular el porcentaje de demanda de la zona frente a la demanda total
    # initial_non_zone_load = total_sys_load - subsys_load_MVA
    # initial_zone_load_perunit = subsys_load_MVA/total_sys_load

    #   Calcular la nueva demanda de la zona
    #desired_zone_load_perunit = initial_zone_load_perunit + 0.05
    #new_zone_load = abs(initial_non_zone_load * desired_zone_load_perunit/(1-desired_zone_load_perunit))




