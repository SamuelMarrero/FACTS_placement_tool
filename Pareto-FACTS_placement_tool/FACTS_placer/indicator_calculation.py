import numpy as np
from pareto_functions import pareto_data_constructor

def calculator(raw_data,facts_parameters, gen_emission_coeffs,gen_cost_coeffs, S_cost_coeff,scenarios):
    error = 0

    len_generators = len(gen_cost_coeffs)
    len_escenarios_1 = len(raw_data)
    len_escenarios_ubicaciones = len(facts_parameters[0]) + 1
    facts_number = len(raw_data[0][0])
    len_buses = len(raw_data[0][0][1][0][0][0]) - 1

    gen_1_power = [[] for facts in xrange(facts_number - 1)]
    gen_2_power = [[] for facts in xrange(facts_number - 1)]
    gen_3_power = [[] for facts in xrange(facts_number - 1)]
    valores_voltage_deviation = [[] for facts in xrange(facts_number - 1)]
    valores_voltage_angle_deviation = [[] for facts in xrange(facts_number - 1)]
    lambda_absolute_values = [[] for facts in xrange(facts_number)]
    gen_costs_per_unit = [[] for generators in xrange(len_generators)]
    valores_emissions = [[] for facts in xrange(facts_number - 1)]
    valores_pgen = [[] for facts in xrange(facts_number - 1)]
    voltage_data = [[] for facts in xrange(facts_number)]
    valores_lambda = [[] for facts in xrange(facts_number - 1)]
    valores_qloss = [[] for facts in xrange(facts_number - 1)]
    valores_ploss = [[] for facts in xrange(facts_number - 1)]
    valores_Sloss = [[] for facts in xrange(facts_number - 1)]
    critic_nodes = []
    valores_costs = [[] for facts in xrange(facts_number - 1)]

    valores_costs_gen1 = [[] for facts in xrange(facts_number - 1)]
    valores_costs_gen2 = [[] for facts in xrange(facts_number - 1)]
    valores_emissions_gen1 = [[] for facts in xrange(facts_number - 1)]
    valores_emissions_gen2 = [[] for facts in xrange(facts_number - 1)]

    valores_load_share = [[]]
    valores_pload = [[]]
    valores_lambda_ini = [[]]
    valores_voltage_ini = []
    valores_ploss_ini = [[]]
    valores_qloss_ini = [[]]

    for demand_scenario in scenarios:
        try:
            len_escenarios_2 = len(raw_data[demand_scenario])
        except:
            pass

        for size_scenario in range(len_escenarios_2):
            lambda_old = 0
            break_code = 0
            len_base_case = len(raw_data[demand_scenario][size_scenario][0][0])
            if len_base_case == 0:
                error += 1
                continue

            for load_step in range(10):
                for facts in range(len_escenarios_ubicaciones):
                    try:
                        x = raw_data[demand_scenario][size_scenario][facts][1][load_step][0]
                        lambda_present = round(raw_data[demand_scenario][size_scenario][facts][1][load_step][0], 3)
                        if facts == 0:
                            lambda_old = lambda_present
                            lambda_ini_pos = load_step
                        if facts > 0 and lambda_present == lambda_old:
                            lambda_ref = lambda_present
                            lambda_ref_pos = load_step
                            lambda_old = lambda_present
                        elif facts > 0 and lambda_present != lambda_old:
                            break_code = 1
                            break
                    except:
                        break_code = 1
                        break
                if break_code == 1:
                    break
            try:
                critic_nodes.append(np.argmin(raw_data[demand_scenario][size_scenario][0][0][-1][0]))
            except:
                pass

            for facts in range(len_escenarios_ubicaciones):
                len_load_increment_lm_base = len(raw_data[demand_scenario][size_scenario][0][0]) - 1
                len_load_increment_lm = len(raw_data[demand_scenario][size_scenario][facts][0]) - 1

                try:
                    volt = raw_data[demand_scenario][size_scenario][facts][0][0][0][0]
                    power = raw_data[demand_scenario][size_scenario][facts][1][0][0]
                    loss = raw_data[demand_scenario][size_scenario][facts][2][0][0]
                    q_gen = raw_data[demand_scenario][size_scenario][facts][3][0]
                    p_gen = raw_data[demand_scenario][size_scenario][facts][4][0][1][0][0]
                except:
                    if facts >= 1:
                        indice_lambda = -0.1
                        indice_qloss = -0.1
                        indice_volt_dev = -0.1
                        indice_Sloss = -0.1
                        indice_costs = -0.1
                        indice_emissions = -0.1
                        valores_lambda[facts - 1].append(indice_lambda)
                        valores_qloss[facts - 1].append(indice_qloss)
                        valores_voltage_deviation[facts - 1].append(indice_volt_dev)
                        valores_Sloss[facts - 1].append(indice_Sloss)
                        valores_costs[facts - 1].append(indice_costs)
                        valores_emissions[facts - 1].append(indice_emissions)
                        voltage_data[facts].append([0 for facts in xrange(len_buses)])
                    continue

                voltage_data[facts].append(raw_data[demand_scenario][size_scenario][facts][0][0][0][0:len_buses + 1])

                lambda_ini = raw_data[demand_scenario][size_scenario][0][1][len_load_increment_lm_base][0]
                lambda_comp = raw_data[demand_scenario][size_scenario][facts][1][len_load_increment_lm][0]
                diferencia_lambda = lambda_comp - lambda_ini
                try:
                    indice_lambda = (diferencia_lambda / (lambda_ini))
                except:
                    indice_lambda = 0
                if facts >= 1:
                    valores_lambda[facts - 1].append(indice_lambda)

                qloss_ini = raw_data[demand_scenario][size_scenario][0][2][0][1][0]
                qloss_comp = raw_data[demand_scenario][size_scenario][facts][2][0][1][0]

                indice_qloss = index_calculator(qloss_ini, qloss_comp)

                valor_volt_dev_ini = 0
                valor_volt_dev_comp = 0
                valor_volt_ang_dev_ini = 0
                valor_volt_ang_dev_comp = 0
                load_ini = 0
                load_comp = 0
                volt_ini_buses = []
                for bus in range(len_buses):
                    volt_ini = raw_data[demand_scenario][size_scenario][0][0][0][0][bus]
                    volt_comp = raw_data[demand_scenario][size_scenario][facts][0][0][0][bus]

                    volt_ini_buses.append(volt_ini)
                    valor_volt_dev_ini += abs(1.00 - volt_ini)
                    valor_volt_dev_comp += abs(1.00 - volt_comp)

                    valor_volt_ang_dev_ini += abs(raw_data[demand_scenario][size_scenario][0][0][0][1][bus])
                    valor_volt_ang_dev_comp += abs(raw_data[demand_scenario][size_scenario][facts][0][0][1][bus])

                    try:
                        load_temp_ini = raw_data[demand_scenario][size_scenario][0][5][0][0][bus]
                        load_temp_comp = raw_data[demand_scenario][size_scenario][facts][5][0][0][bus]

                        load_ini += load_temp_ini.real
                        load_comp += load_temp_comp.real
                    except:
                        pass

                buses_load = raw_data[demand_scenario][size_scenario][0][5][0][0]
                load_share = np.divide(buses_load,load_ini).astype('float')
                indice_load_share = np.min(load_share)/np.max(load_share)
                indice_volt_dev = index_calculator(valor_volt_dev_ini, valor_volt_dev_comp)
                indice_volt_ang_dev = index_calculator(valor_volt_ang_dev_ini, valor_volt_ang_dev_comp)

                ploss_ini = raw_data[demand_scenario][size_scenario][0][2][0][0][0]
                ploss_comp = raw_data[demand_scenario][size_scenario][facts][2][0][0][0]
                S_losses_ini = np.sqrt(ploss_ini ** 2 + qloss_ini ** 2)
                S_losses_comp = np.sqrt(ploss_comp ** 2 + qloss_comp ** 2)

                indice_Sloss = index_calculator(S_losses_ini, S_losses_comp)
                indice_Ploss = index_calculator(ploss_comp, ploss_ini)

                generation_costs_ini = 0
                generation_costs_comp = 0
                system_emissions_ini = 0
                system_emissions_comp = 0
                total_gen_power_ini = 0
                total_gen_power_comp = 0
                for generator in range(len_generators):
                    try:
                        try:
                            gen_power_ini = raw_data[demand_scenario][size_scenario][0][4][0][1][0][generator]
                        except:
                            gen_power_ini = 0
                        gen_power_comp = raw_data[demand_scenario][size_scenario][facts][4][0][1][0][generator]
                        generator_cost_ini = gen_power_ini ** 2 * gen_cost_coeffs[generator][0] + gen_power_ini * \
                                                                                                  gen_cost_coeffs[
                                                                                                      generator][
                                                                                                      1] + \
                                             gen_cost_coeffs[generator][2]
                        generator_cost_comp = gen_power_comp ** 2 * gen_cost_coeffs[generator][0] + gen_power_comp * \
                                                                                                    gen_cost_coeffs[
                                                                                                        generator][
                                                                                                        1] + \
                                              gen_cost_coeffs[generator][2]

                        generator_emissions_ini = gen_power_ini ** 2 * gen_emission_coeffs[generator][
                            0] + gen_power_ini * \
                                 gen_emission_coeffs[generator][1] + gen_emission_coeffs[generator][2]
                        generator_emissions_comp = gen_power_comp ** 2 * gen_emission_coeffs[generator][
                            0] + gen_power_comp * \
                                 gen_emission_coeffs[generator][1] + gen_emission_coeffs[generator][2]

                        generation_costs_ini += generator_cost_ini
                        generation_costs_comp += generator_cost_comp
                        system_emissions_ini += generator_emissions_ini
                        system_emissions_comp += generator_emissions_comp
                        total_gen_power_ini += gen_power_ini
                        total_gen_power_comp += gen_power_comp

                        if generator == 0:
                            indice_costs_gen1 = index_calculator(generator_cost_ini, generator_cost_comp)
                            indice_emissions_gen1 = index_calculator(generator_emissions_ini,
                                                                     generator_emissions_comp)
                            indice_power_gen1 = index_calculator(gen_power_ini, gen_power_comp)
                        if generator == 1:
                            indice_costs_gen2 = index_calculator(generator_cost_ini, generator_cost_comp)
                            indice_emissions_gen2 = index_calculator(generator_emissions_ini,
                                                                     generator_emissions_comp)
                            indice_power_gen2 = index_calculator(gen_power_ini, gen_power_comp)
                        if generator == 2:
                            indice_power_gen3 = index_calculator(gen_power_ini, gen_power_comp)

                        if facts == 0:
                            gen_costs_per_unit[generator].append(generation_costs_ini / gen_power_ini)

                    except:
                        generation_costs_ini += 0
                        generation_costs_comp += 0
                        system_emissions_ini += 0
                        system_emissions_comp += 0

                S_losses_costs_ini = S_losses_ini * S_cost_coeff
                S_losses_costs_comp = S_losses_comp * S_cost_coeff

                transmission_costs_ini = total_gen_power_ini * S_cost_coeff
                transmission_costs_comp = total_gen_power_comp * S_cost_coeff

                valor_cost_ini = generation_costs_ini + transmission_costs_ini
                valor_cost_comp = generation_costs_comp + transmission_costs_comp

                indice_costs = index_calculator(valor_cost_ini, valor_cost_comp)
                indice_pgen = index_calculator(total_gen_power_ini, total_gen_power_comp)
                indice_emissions = index_calculator(system_emissions_ini, system_emissions_comp)

                if facts >= 1:
                    valores_voltage_deviation[facts - 1].append(indice_volt_dev)
                    valores_voltage_angle_deviation[facts - 1].append(indice_volt_ang_dev)
                    valores_Sloss[facts - 1].append(indice_Sloss)
                    valores_ploss[facts - 1].append(indice_Ploss)
                    valores_qloss[facts - 1].append(indice_qloss)
                    valores_pgen[facts - 1].append(indice_pgen)
                    valores_costs[facts - 1].append(indice_costs)
                    valores_emissions[facts - 1].append(indice_emissions)
                    valores_costs_gen1[facts - 1].append(indice_costs_gen1)
                    valores_emissions_gen1[facts - 1].append(indice_emissions_gen1)
                    valores_costs_gen2[facts - 1].append(indice_costs_gen1)
                    valores_emissions_gen2[facts - 1].append(indice_emissions_gen1)
                    gen_1_power[facts - 1].append(indice_power_gen1)
                    gen_2_power[facts - 1].append(indice_power_gen2)
                    gen_3_power[facts - 1].append(indice_power_gen3)
                if facts == 1:
                    valores_load_share[0].append(indice_load_share)
                    valores_pload[0].append(load_ini)
                    valores_lambda_ini[0].append(lambda_ini)
                    valores_voltage_ini.append(volt_ini_buses)
                    # valores_ploss_ini.append(ploss_ini)
                    valores_qloss_ini[0].append(qloss_ini)

                lambda_absolute_values[facts].append(lambda_comp)
    voltage_ini = [[buses_voltage[-1] for buses_voltage in valores_voltage_ini]]
    data = {
        #"volt_ini": voltage_ini,
        "q_loss_ini": valores_qloss_ini,
        "Voltage deviation" : valores_voltage_deviation,
        "Voltage angle deviation" : valores_voltage_angle_deviation,
        "Loading margin" : valores_lambda,
        "Active power losses" : valores_ploss,
        "Reactive power losses" : valores_qloss,
        "Generated power" : valores_pgen,
        "Costs": valores_costs,
        "GHG emissions": valores_emissions,
        "p_load" : valores_pload,
        "load_share" : valores_load_share,
        "lambda_ini" : valores_lambda_ini
    }

    return data


def MI_indicator_filter(data,indicators):
    filtered_data = []
    filtered_indicators = []
    iteration = 0
    for indicator in indicators:
        mean = np.mean(data[indicator],axis=1)
        std = np.std(data[indicator],axis=1)
        check = True
        for iteration in range(len(mean)):
            if abs(mean[iteration])< 0.005 and std[iteration] < 0.025:
                check = False
        if check == True: #
            filtered_data.append(data[indicator])
            filtered_indicators.append(indicator)
            iteration += 1
    filtered_data = np.transpose(filtered_data,axes=[0,2,1])
    return filtered_data,filtered_indicators

def med_std_calculator(data,indicators):
    results_med_std ={}
    for indicator in indicators:
        med_std = pareto_data_constructor(data[indicator])
        dict = { indicator : med_std}
        results_med_std.update(dict)
    return results_med_std

def index_calculator(ini_value, comp_value):
    difference = ini_value - comp_value
    index = difference / ini_value
    return index