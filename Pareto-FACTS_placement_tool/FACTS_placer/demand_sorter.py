import numpy as np

class Sorter:

    @staticmethod
    def sort(demand_data, nodes_names, nodes_demand):
        std_values = np.std(demand_data,axis=0)
        med_values = np.mean(demand_data,axis=0)
        med= np.sum(med_values)
        index_values = np.divide(std_values,med_values)
        std_asc_order = np.argsort(med_values,axis=-1)
        demand_dsc_order = np.argsort(nodes_demand,axis=-1)
        map = {}
        for i, demand_index_value in enumerate(demand_dsc_order):
            column_index = std_asc_order[i]
            map[nodes_names[demand_index_value]] = demand_data[:,column_index]
        return Sorter.__to_matrix(nodes_names, map)

    @staticmethod
    def __to_matrix(nodes_names, mapa):
        arr = []
        nodes_names_sorted = sorted(nodes_names)
        for node_name in nodes_names_sorted:
            arr.append(mapa[node_name])

        arr = np.transpose(arr,axes=None)
        return arr
