import numpy as np

def data_filter(data):

    data_filtered = []
    len_generation_scenario = len(data)
    generation_scenario = 0
    for i in range(len_generation_scenario):
        len_demand_zones = len(data[generation_scenario])
        data_filtered.append([])
        demand_zone = 0
        for j in range(len_demand_zones):
            len_facts = len(data[generation_scenario][demand_zone])
            data_filtered[generation_scenario].append([])
            facts = 0
            if len_facts > 0:
                for k in range(len_facts):
                    data_filtered[generation_scenario][demand_zone].append([])
                    filtered_value = 0
                    if len_facts > 0:
                        iter_gen_scenario = 0
                        num_elements_mean = 0
                        for j in range(3):
                            iter_demand_zone = 0
                            for h in range (3):
                                gen_scenario_mean = generation_scenario - 1 + iter_gen_scenario
                                demand_zone_mean = demand_zone - 1 + iter_demand_zone
                                if gen_scenario_mean >= 0 and demand_zone_mean >= 0:
                                    try:
                                        filtered_value += data[gen_scenario_mean][demand_zone_mean][facts]
                                        num_elements_mean += 1
                                    except:
                                        pass
                                iter_demand_zone += 1
                            iter_gen_scenario += 1
                        try:
                            data_filtered[generation_scenario][demand_zone][facts] = float(filtered_value) / float(num_elements_mean)
                        except:
                            data_filtered[generation_scenario][demand_zone][facts] = 0
                    facts += 1
            demand_zone += 1
        generation_scenario += 1

    return data_filtered

def data_handler (indices_data, selected_index):
    index_recovery_data = []
    len_generation_scenarios = len(indices_data)
    generation_scenario = 0
    for j in range(len_generation_scenarios):
        index_recovery_data.append([])
        demand_zone = 0
        len_demand_zones = len(indices_data[generation_scenario])
        for k in range(len_demand_zones):
            len_facts_positions = len(indices_data[generation_scenario][demand_zone][selected_index])
            index_recovery_data[generation_scenario].append([])
            facts_position = 0
            for l in range(len_facts_positions):
                index_recovery_data[generation_scenario][demand_zone].append(indices_data[generation_scenario][demand_zone]
                                                                                            [selected_index][facts_position])
                facts_position += 1
            demand_zone += 1
        generation_scenario += 1
    return index_recovery_data

def indices_recalculator(index_data,subindex_data, best_facts_position, best_facts_pos_lambda,facts_parameters):

    gen_scen = 0
    demand_variation_step = 0

    demand_zone = 0
    for f in range(10):
        try:
            demand_study_init = best_facts_position[0][demand_zone][2][0]
            break
        except:
            pass
        demand_zone += 1

    demand_variation_step = demand_study_init

    best_facts_position = []
    best_facts_pos_lambda = []
    best_facts_pos_ordered = []
    best_facts_pos_lambda_ordered = []
    best_facts_FPI_value_ordered = []
    best_facts_lambda_value_ordered = []

    number_of_facts = len(facts_parameters[0]) + 1
    for d in range(number_of_facts):
        best_facts_pos_ordered.append([])
        best_facts_pos_lambda_ordered.append([])
        best_facts_FPI_value_ordered.append([])
        best_facts_lambda_value_ordered.append([])

    len_generation_scenarios = len(index_data)
    load_percent_zone_1 =  float(demand_study_init)/100
    generation_scenario = 0
    for j in range(len_generation_scenarios):
        best_facts_position.append([])
        best_facts_pos_lambda.append([])
        len_demand_zones = len(index_data[generation_scenario])
        load_percent_zone_2 =  float(demand_study_init)/100
        demand_zone = 0
        for k in range(len_demand_zones):
    #   Calculo del maximo valor de entre los minimos de los indices relativos a las distitnas ubicaciones

            best_facts_position[generation_scenario].append([0, 0])
            best_facts_pos_lambda[generation_scenario].append([0, 0])
            facts_position = 0
            len_indices = len(index_data[generation_scenario][demand_zone])

            result = []
            result_lambda = []
            for m in range(len_indices):
                if index_data[generation_scenario][demand_zone][facts_position] > best_facts_position[generation_scenario][demand_zone][1]:
                    if facts_position == 0:
                        best_facts_position[generation_scenario][demand_zone] = [0,index_data[generation_scenario][demand_zone][facts_position]]
                        result = [0,index_data[generation_scenario][demand_zone][facts_position]]
                    if facts_position > 0:
                        best_facts_position[generation_scenario][demand_zone] = [facts_parameters[1][facts_position - 1][0],index_data[generation_scenario][demand_zone][facts_position]]
                        result = [facts_parameters[1][facts_position - 1][0],index_data[generation_scenario][demand_zone][facts_position]]

                if subindex_data[generation_scenario][demand_zone][facts_position] > best_facts_pos_lambda[generation_scenario][demand_zone][1]:
                    if facts_position == 0:
                        best_facts_pos_lambda[generation_scenario][demand_zone] = [0,subindex_data[generation_scenario][demand_zone][facts_position]]
                        result_lambda = [0,
                                     subindex_data[generation_scenario][demand_zone][facts_position]]
                    if facts_position > 0:
                        best_facts_pos_lambda[generation_scenario][demand_zone] = [
                                facts_parameters[1][facts_position - 1][0],
                                subindex_data[generation_scenario][demand_zone][facts_position]]
                        result_lambda = [facts_parameters[1][facts_position - 1][0],
                                             subindex_data[generation_scenario][demand_zone][facts_position]]
                facts_position += 1


            result.append([load_percent_zone_1, load_percent_zone_2])
            result_lambda.append([load_percent_zone_1, load_percent_zone_2])

            best_facts_position[generation_scenario][demand_zone] = result
            best_facts_pos_lambda[generation_scenario][demand_zone] = result_lambda


            bus_number = 0
            for j in range(number_of_facts):

                if result[0] == 0:
                    best_facts_pos_ordered[bus_number].append([load_percent_zone_1, load_percent_zone_2])
                    best_facts_FPI_value_ordered[bus_number].append(best_facts_position[generation_scenario][demand_zone][1])
                    break
                elif bus_number > 0 and result[0] == facts_parameters[1][bus_number-1][0]:
                    best_facts_pos_ordered[bus_number].append([load_percent_zone_1, load_percent_zone_2])
                    best_facts_FPI_value_ordered[bus_number].append(best_facts_position[generation_scenario][demand_zone][1])
                    break
                bus_number += 1

            bus_number = 0
            for j in range(number_of_facts):
                if result_lambda[0] == 0:
                    best_facts_pos_lambda_ordered[bus_number].append([load_percent_zone_1, load_percent_zone_2])
                    best_facts_lambda_value_ordered[bus_number].append(
                                            best_facts_pos_lambda[generation_scenario][demand_zone][1])
                    break
                elif bus_number > 0 and result_lambda[0] == facts_parameters[1][bus_number-1][0]:
                    best_facts_pos_lambda_ordered[bus_number].append([load_percent_zone_1, load_percent_zone_2])
                    best_facts_lambda_value_ordered[bus_number].append(
                                            best_facts_pos_lambda[generation_scenario][demand_zone][1])
                    break
                bus_number += 1

            load_percent_zone_2 += float(demand_variation_step)
            demand_zone += 1

        load_percent_zone_1 += float(demand_variation_step)
        generation_scenario += 1

    return best_facts_pos_ordered, best_facts_pos_lambda_ordered, best_facts_FPI_value_ordered, best_facts_lambda_value_ordered, best_facts_position, best_facts_pos_lambda

