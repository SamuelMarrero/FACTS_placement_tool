import os
cwd = os.getcwd()
#   Abrir archivo de configuracion
def load (study_name):
    study_file = os.path.join('', study_name + '.test')
    config_file = open(study_file,'r')
    config_data = config_file.readlines()

#   ENTRADAS
    psse_data_path = config_data[4].rstrip('\n')
    psse_file_name_short = config_data[7].rstrip('.sav\n')

#   Peticion de datos para configuracion del estudio
    #   Subsistema
    sid = int(config_data[11].rstrip('\n'))
    num_of_zones = int(config_data[14].rstrip('\n'))
    j = 0
    k = 0
    zones = ['']
    for n in range(num_of_zones):
        zones[k] = int(config_data[17][j])
        j += 2
        k += 1

    num_of_areas = int(config_data[20].rstrip('\n'))
    areas= config_data[23].rstrip('\n')
    if areas == '':
        areas = []
    owner = 1
    tolerance_val = float(config_data[26].rstrip('\n'))
    neg_tolerance_val = - float(tolerance_val)

    #   Para calculo de flujo de carga
    fnsl_options = [[],[],[],[],[],[],[],[]]
    j=0
    k=0
    for i in range(8):
        fnsl_options[k] = int(config_data[38][j])
        j += 2
        k += 1
    if fnsl_options[6]== 9:
        fnsl_options[6]= -1

    #  Para el estudio PV
    percent_mode = int(config_data[29].rstrip('\n'))
    load_increment_step_percent = int(config_data[32].rstrip('\n'))
    load_increment_step_MVA = int(config_data[35].rstrip('\n'))

    #   Para la API SCAL_2
    value_all = int(config_data[41].rstrip('\n'))
    apiopt = int(config_data[44].rstrip('\n'))
    status = [[],[],[],[],[]]

    j=0
    k=0
    for i in range(5):
        status[k] = int(config_data[47][j])
        j += 2
        k += 1

    scalval = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]

    number_of_facts = int(config_data[51].rstrip('\n'))

    type_of_device = str(config_data[54].rstrip('\n'))

    facts_number = 0
    facts_placement_parameters = [[],[]]
    for o in range(number_of_facts):
        facts_placement_parameters[0].append(config_data[57 + facts_number].rstrip('\n'))
        # vector = []
        # j = 0
        # k = 0
        # vector_element = 0
        # number_of_vector_elements = 6
        # file_line_number = 57 + number_of_facts - 1 + facts_number
        # len_line = len(config_data[file_line_number])
        # config_data_line = config_data[file_line_number].rstrip('\n')
        # for p in range(number_of_vector_elements):
        #     partial_data = ''
        #     for q in range(6):
        #         if j <= len_line-1:
        #             if config_data_line[j] != (','):
        #                 partial_data += config_data_line[j]
        #                 j += 1
        #             elif config_data_line[j] == (','):
        #                 vector.append(int(partial_data))
        #                 j += 1
        #                 vector_element += 1
        #                 break
        #         else:
        #             break
        # facts_placement_parameters[1].append(vector)

        vector = []
        j = 0
        k = 0
        variable_names = ['ibus', 'jbus', 'Mbase', 'WMOD', 'RemoteBus', 'Vref', 'VSref', 'Pmax', 'Pmin', 'Qmax',
                          'Qmin', 'Rsource', 'Zsource']
        vector_element = 0
        number_of_vector_elements = 13
        file_line_number = 60 + (number_of_facts - 1) + facts_number
        len_line = len(config_data[file_line_number])
        config_data_line = config_data[file_line_number].rstrip('\n')
        for p in range(number_of_vector_elements):
            partial_data = ''
            for q in range(7):
                if j <= len_line-1:
                    if config_data_line[j] != (','):
                        partial_data += config_data_line[j]
                        j += 1
                    elif config_data_line[j] == (','):
                        if (vector_element == 0) or (vector_element == 1) or (vector_element == 3) or (vector_element == 4) or (vector_element == 6):
                            vector.append(int(partial_data))
                        else:
                            vector.append(float(partial_data))
                        j += 1
                        vector_element += 1
                        break
                else:
                    break
        facts_placement_parameters[1].append(vector)

        facts_number += 1
    config_file.close()
    return psse_data_path, psse_file_name_short, sid, num_of_zones, zones, num_of_areas, areas, owner, tolerance_val,\
           neg_tolerance_val, fnsl_options, percent_mode, load_increment_step_MVA, load_increment_step_percent, value_all,\
           apiopt, status, scalval, number_of_facts, type_of_device, facts_placement_parameters, variable_names


# facts_number = 0
#     facts_placement_parameters = [[],[],[]]
#     for o in range(number_of_facts):
#         facts_placement_parameters[0].append(config_data[54 + facts_number].rstrip('\n'))
#         vector = []
#         j = 0
#         k = 0
#         vector_element = 0
#         number_of_vector_elements = 6
#         file_line_number = 57 + number_of_facts - 1 + facts_number
#         len_line = len(config_data[file_line_number])
#         config_data_line = config_data[file_line_number].rstrip('\n')
#         for p in range(number_of_vector_elements):
#             partial_data = ''
#             for q in range(6):
#                 if j <= len_line-1:
#                     if config_data_line[j] != (','):
#                         partial_data += config_data_line[j]
#                         j += 1
#                     elif config_data_line[j] == (','):
#                         vector.append(int(partial_data))
#                         j += 1
#                         vector_element += 1
#                         break
#                 else:
#                     break
#         facts_placement_parameters[1].append(vector)
#
#         vector = []
#         j = 0
#         k = 0
#         vector_element = 0
#         number_of_vector_elements = 13
#         file_line_number = 60 + 2*(number_of_facts - 1) + facts_number
#         len_line = len(config_data[file_line_number])
#         config_data_line = config_data[file_line_number].rstrip('\n')
#         for p in range(number_of_vector_elements):
#             partial_data = ''
#             for q in range(7):
#                 if j <= len_line-1:
#                     if config_data_line[j] != (','):
#                         partial_data += config_data_line[j]
#                         j += 1
#                     elif config_data_line[j] == (','):
#                         if (vector_element == 0) or (vector_element == 1) or (vector_element == 3) or (vector_element == 4):
#                             vector.append(int(partial_data))
#                         else:
#                             vector.append(float(partial_data))
#                         j += 1
#                         vector_element += 1
#                         break
#                 else:
#                     break
#         facts_placement_parameters[2].append(vector)
#
#         facts_number += 1