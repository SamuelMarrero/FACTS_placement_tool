import psse34,psspy,os,csv
import numpy as np
from itertools import groupby
cwd = os.getcwd()

def load_scenarios_config_data(program_path,data_path,study_name):
    os.chdir(program_path)
    study_file = os.path.join('', study_name + '.scen')
    scenarios_config_file = open(study_file, 'r')
    scenarios_config_data = scenarios_config_file.readlines()

    demand_scenarios_file_name = scenarios_config_data[3].rstrip('\n')

    number_of_size_scenarios = int(scenarios_config_data[6].rstrip('\n'))

    size_scenarios = []

    len_line = len(scenarios_config_data[9])
    config_data_line = scenarios_config_data[9].rstrip('\n')
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
    os.chdir(data_path)
    demand_data = []
    with open(demand_scenarios_file_name,'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        for row in reader:
            demand_data.append(row)

    demand_data_np = np.array(demand_data)
    nodes_names = demand_data_np[0,:]
    demand_tstamps = demand_data_np[:,0]
    data_type = np.dtype('S8')
    demand_values = np.array(demand_data_np,dtype = data_type)
    demand_values = np.delete(demand_values, (0), axis = 0)
    demand_values = np.delete(demand_values, (0), axis = 1)

    demand_values = demand_values.astype(np.float)


    return demand_data,demand_values,size_scenarios,number_of_size_scenarios

def demand_scenarios_preparer(demand_data,nodes_load):
    demand_data_size = demand_data.shape
    num_nodes_system = np.array(nodes_load).size
    num_nodes_demand = demand_data_size[1]
    exceeding_demand_nodes = num_nodes_demand - num_nodes_system
    for exceeding_node in range(exceeding_demand_nodes):
        demand_data = np.delete(demand_data,(exceeding_node),axis = 1)

    mean_node_demand = np.mean(demand_data, axis = 0)

    nodes_active_power =[]
    nodes_PQ_ratio = []
    node = 0
    for nodes in nodes_load:
        nodes_active_power.append(nodes.real)
        nodes_PQ_ratio.append(nodes.imag/nodes.real)
        node += 1

    scaling_coefficients = np.divide(nodes_active_power,mean_node_demand)
    scaled_demand_data = demand_data * scaling_coefficients

    total_scaled_demand = np.sum(scaled_demand_data, axis=1)
    min_demand_value = np.min(total_scaled_demand)
    min_demand_pos = np.argmin(total_scaled_demand)

    load_share_data = np.divide(scaled_demand_data,total_scaled_demand[:, None])

    demand_scenarios_data = load_share_data * min_demand_value

    total_demand_scenarios = np.sum(demand_scenarios_data, axis=1)
    mean_load_share = np.mean(load_share_data, axis = 0)

    # min_load_share = np.min(load_share_data, axis = 0)
    # max_load_share = np.max(load_share_data, axis = 0)
    # mean_load_share = np.mean(load_share_data, axis = 0)
    # delta_load_share = (max_load_share - min_load_share)/mean_load_share
    return demand_scenarios_data,min_demand_pos,nodes_PQ_ratio,total_scaled_demand

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



program_path demand_data_path,study_name

nodes_load =


demand_data,demand_values,size_scenarios,len_size_scenarios = load_scenarios_config_data(program_path,demand_data_path,study_name)
demand_scenarios_data, min_demand_position, nodes_PQ_ratio, total_scaled_demand_data = demand_scenarios_preparer(demand_values,nodes_load[0])
