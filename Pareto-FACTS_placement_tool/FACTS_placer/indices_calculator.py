def parameter_indices(indices_i_temp, placement_study_data, len_load_increment_base, len_load_increment_lm, len_buses,
                      vref, facts_iteration, len_gens, len_loads, generation_scenario, demand_zone):
    #   Calculo del margen de carga para la ubicacion del FACTS
    try:
        max_load = \
        placement_study_data[generation_scenario][demand_zone][facts_iteration][7][len_load_increment_lm - 1][1]
        init_load = placement_study_data[generation_scenario][demand_zone][0][7][0][1]
        loading_margin = (max_load - init_load) / init_load
        if loading_margin < 0:
            loading_margin = 0
    except:
        loading_margin = 0

    #   Calculo de la sumatoria de las desviaciones de la tension de los nodos fuera de limite para la ubicacion del FACTS
    bus = 0
    volt_dev_sum = 0
    q_loss = 0
    q_gen = 0
    q_load = 0
    for i in range(len_buses):
        # vref = placement_study_data[generation_scenario][demand_zone][facts_iteration][0][0][0][bus]
        bus_voltage = \
        placement_study_data[generation_scenario][demand_zone][facts_iteration][0][len_load_increment_lm - 1][0][bus]
        if bus_voltage < 0.95 or bus_voltage > 1.05:
            volt_dev = abs((vref - bus_voltage) / vref)
            volt_dev_sum += volt_dev
        bus += 1

        # data = 0
        # for k in range(num_of_zones):
        #     q_loss += PV_data[facts_iteration][9][len_load_increment-1][1][data]
        #     data += 1

    #   Calculo de las perdidas de reactiva para la ubicacion del FACTS
    gen = 0
    for k in range(len_gens):
        q_gen += float(
            placement_study_data[generation_scenario][demand_zone][facts_iteration][1][len_load_increment_lm - 1][1][
                gen])
        gen += 1
    load = 0
    for k in range(len_loads):
        q_load += float(
            placement_study_data[generation_scenario][demand_zone][facts_iteration][2][len_load_increment_lm - 1][0][
                load].imag)
        load += 1

    q_loss = q_gen - q_load

    #   Creacion de la matriz de datos
    indices_i_temp[0].append(loading_margin)
    indices_i_temp[1].append(volt_dev_sum)
    indices_i_temp[2].append(q_loss)
    return indices_i_temp


def membership_function(indices, indices_i, indices_ii, best_facts_position, best_facts_pos_lambda,
                        best_facts_pos_lambda_ordered,
                        best_facts_lambda_value_ordered, facts_placement_parameters, demand_scenario, size_scenario,
                        best_facts_pos_ordered, best_facts_FPI_value_ordered):
    #   Calculo de la funcion de pertenencia relativa a cada indice del escenario
    index = 0
    index_I_min = []
    index_I_max = []
    len_indices_i = len(indices_i[demand_scenario][size_scenario])
    for j in range(len_indices_i):
        index_I_min.append(float(min(indices_i[demand_scenario][size_scenario][index])))
        index_I_max.append(float(max(indices_i[demand_scenario][size_scenario][index])))

        index_I_iteration = 0
        for k in range(len(indices_i[demand_scenario][size_scenario][index])):
            index_i_current_value = indices_i[demand_scenario][size_scenario][index][index_I_iteration]
            index_i_zero_value = indices_i[demand_scenario][size_scenario][index][0]
            if index == 0:
                indices_ii[demand_scenario][size_scenario][index].append(
                    (float(index_i_current_value) - index_I_min[index]) / (index_I_max[index] - index_I_min[index]))
            if index > 0:
                indices_ii[demand_scenario][size_scenario][index].append(
                    (float(index_I_max[index]) - float(index_i_current_value)) / (
                    float(index_I_max[index]) - index_I_min[index]))
            index_I_iteration += 1
        index += 1

    #   Optimizacion de la ubicacion en base a las funciones de pertenencia del escenario
    #   Calculo del minimo entre los indices relativos a las distitnas ubicaciones


    facts_position = 0
    len_indices_ii = len(indices_ii[demand_scenario][size_scenario][0])
    for l in range(len_indices_ii):
        index = 0
        indices[demand_scenario][size_scenario].append([1])
        for k in range(len(indices_ii[demand_scenario][size_scenario])):
            if indices_ii[demand_scenario][size_scenario][index][facts_position] < \
                    indices[demand_scenario][size_scenario][facts_position]:
                indices[demand_scenario][size_scenario][facts_position] = \
                indices_ii[demand_scenario][size_scenario][index][facts_position]
            index += 1
        facts_position += 1

    # Calculo del maximo valor de entre los minimos de los indices relativos a las distitnas ubicaciones
    facts_position = 0
    best_facts_position[demand_scenario][size_scenario] = [0, 0]
    best_facts_pos_lambda[demand_scenario][size_scenario] = [0, 0]
    len_indices = len(indices[demand_scenario][size_scenario])
    for m in range(len_indices):
        if indices[demand_scenario][size_scenario][facts_position] > \
                best_facts_position[demand_scenario][size_scenario][1]:
            if facts_position == 0:
                best_facts_position[demand_scenario][size_scenario] = [0, indices[demand_scenario][size_scenario][
                    facts_position]]
                result = [0, indices[demand_scenario][size_scenario][facts_position]]
            elif facts_position > 0:
                best_facts_position[demand_scenario][size_scenario] = [
                    facts_placement_parameters[1][facts_position][0],
                    indices[demand_scenario][size_scenario][facts_position]]
                result = [facts_placement_parameters[1][facts_position][0],
                          indices[demand_scenario][size_scenario][facts_position]]

        if indices_i[demand_scenario][size_scenario][0][facts_position] > \
                best_facts_pos_lambda[demand_scenario][size_scenario][1]:
            if facts_position == 0:
                best_facts_pos_lambda[demand_scenario][size_scenario] = [0,
                                                                           indices_i[demand_scenario][size_scenario][0][facts_position]]
                result_lambda = [0, indices_i[demand_scenario][size_scenario][0][facts_position]]
            elif facts_position > 0:
                best_facts_pos_lambda[demand_scenario][size_scenario] = [
                    facts_placement_parameters[1][facts_position][0],
                    indices_i[demand_scenario][size_scenario][0][
                        facts_position]]
                result_lambda = [facts_placement_parameters[1][facts_position][0],
                                 indices_i[demand_scenario][size_scenario][0][facts_position]]
        facts_position += 1

    result.append([demand_scenario, size_scenario])
    result_lambda.append([demand_scenario, size_scenario])

    best_facts_position[demand_scenario][size_scenario] = result
    best_facts_pos_lambda[demand_scenario][size_scenario] = result_lambda

    bus_number = 0
    for j in range(len_indices):
        if result[0] == 0:
            best_facts_pos_ordered[bus_number].append([demand_scenario, size_scenario])
            best_facts_FPI_value_ordered[bus_number].append(best_facts_position[demand_scenario][size_scenario][1])
            break
        elif bus_number > 0 and result[0] == facts_placement_parameters[1][bus_number - 1][0]:
            best_facts_pos_ordered[bus_number].append([demand_scenario, size_scenario])
            best_facts_FPI_value_ordered[bus_number].append(best_facts_position[demand_scenario][size_scenario][1])
            break
        bus_number += 1

    bus_number = 0
    for j in range(len_indices):
        if result_lambda[0] == 0:
            best_facts_pos_lambda_ordered[bus_number].append([demand_scenario, size_scenario])
            best_facts_lambda_value_ordered[bus_number].append(
                best_facts_pos_lambda[demand_scenario][size_scenario][1])
            break
        elif bus_number > 0 and result_lambda[0] == facts_placement_parameters[1][bus_number - 1][0]:
            best_facts_pos_lambda_ordered[bus_number].append([demand_scenario, size_scenario])
            best_facts_lambda_value_ordered[bus_number].append(
                best_facts_pos_lambda[demand_scenario][size_scenario][1])
            break
        bus_number += 1

    return indices, indices_ii, best_facts_pos_ordered, best_facts_pos_lambda_ordered, best_facts_lambda_value_ordered, best_facts_position, best_facts_pos_lambda, best_facts_FPI_value_ordered
