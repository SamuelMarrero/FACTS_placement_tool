import os
import cPickle as pickle
import numpy as np
import time
import matplotlib.pyplot as plt

execution_time = time.clock()

psse_file_name_short = 'IEEE 14 bus-NoShunt'
os.chdir('d:\Users\smarrero\Documents\Doctorado\Estudios\PSS-E\FACTS_placement_tool\FACTS_placer\studies\IEEE-14-prueba\\results')

demand_scenarios = 100
size_scenarios = 1
raw_data = []

# for scenario in xrange(1000,2000):
#     data_object_name = os.path.join('', psse_file_name_short + '-D-' + str(scenario) + '.obj')
#     with open(data_object_name,'r') as data_file:
#         temp_data = pickle.load(data_file)
#     with open(data_object_name, 'wb') as data_file:
#         pickle.dump(temp_data, data_file, protocol=-1)

iteration = 0
for demand_scenario in xrange(4200,4500):
    data_object_name = os.path.join('', psse_file_name_short + '-D-' + str(demand_scenario) + '.obj')
    raw_data.append([])
    #print data_object_name
    with open(data_object_name,'rb') as data_file:
        temp_data = pickle.load(data_file)
        raw_data[iteration].append([])
        for size_scenario in xrange(len(temp_data)):
            volts = []
            total_p = []
            total_loss = []
            for facts in xrange(len(temp_data[size_scenario])):
                volts = temp_data[size_scenario][facts][0]
                total_p = temp_data[size_scenario][facts][7]
                p_gen = temp_data[size_scenario][facts][1]
                total_loss = temp_data[size_scenario][facts][9]
                raw_data[iteration][size_scenario].append([volts, total_p, p_gen, total_loss])
    iteration += 1

with open('reduced_raw_data.obj', 'wb') as save_raw_data:
    pickle.dump(raw_data, save_raw_data, protocol=-1)

execution_time = time.clock()
#



# losses = []
# loading =[]
# voltages =[]
# for demand_scenario in range(len(raw_data)-1):
#     losses.append([])
#     loading.append([])
#     voltages.append([])
#     break_code = 0
#     for size_scenario in range(len(raw_data[demand_scenario])):
#         losses[demand_scenario].append([])
#         loading[demand_scenario].append([])
#         voltages[demand_scenario].append([])
#         for facts in range(len(raw_data[demand_scenario][size_scenario])):
#             len_load_increment_lm = len(raw_data[demand_scenario][size_scenario][facts][0])-1
#
#             for load_step in range(10):
#                 for facts in range(len(raw_data[demand_scenario][size_scenario])):
#                     try:
#                         x = raw_data[demand_scenario][size_scenario][facts][1][load_step][0]
#                         lambda_present = round(raw_data[demand_scenario][size_scenario][facts][1][load_step][0], 3)
#                         if facts == 0:
#                             lambda_old = lambda_present
#                             lambda_ini_pos = load_step
#                         if facts > 0 and lambda_present == lambda_old:
#                             lambda_ref = lambda_present
#                             lambda_ref_pos = load_step
#                             lambda_old = lambda_present
#                         elif facts > 0 and lambda_present != lambda_old:
#                             break_code = 1
#                             break
#                     except:
#                         break_code = 1
#                         break
#                 if break_code == 1:
#                     break
#
#             try:
#                 max_load = raw_data[demand_scenario][size_scenario][facts][1][len_load_increment_lm][1]
#                 init_load = raw_data[demand_scenario][size_scenario][0][1][0][1]
#                 loading_margin = (max_load - init_load) / init_load
#                 if loading_margin < 0:
#                     loading_margin = 0
#             except:
#                 loading_margin = 0
#             loading[demand_scenario][size_scenario].append([loading_margin,lambda_ref_pos])
#             losses[demand_scenario][size_scenario].append(raw_data[demand_scenario][size_scenario][facts][2])
#             voltages[demand_scenario][size_scenario].append(raw_data[demand_scenario][size_scenario][facts][0])





# len_ubicaciones = 9
# len_buses = len(raw_data[5][0][size_scenario][0][0][0])
# fig_2 = plt.figure()
# fig_2.suptitle('Curvas PV, escenario 5-0, sin y con STATCOM')
# plt.axis()
# for j in range(len_ubicaciones):
#     plt.subplot(3, 3, j + 1)
#     for k in range(len( raw_data[5][0][j][1])):
#         for l in range(len_buses):
#             try:
#                 x = raw_data[5][0][j][1][k][0]
#                 y = raw_data[5][0][j][0][k][0][l]
#                 color=['black','blue','green','red','pink','violet','grey','brown','orange','darkred','lawngreen','gold','purple','aqua']
#                 plt.scatter(x,y,color = color[l])
#             except:
#                 pass
#
# plt.show()





a = 0

#for load_step in range(10):
        #     for facts in range(len(raw_data[demand_scenario][size_scenario])):
        #         try:
        #             x = raw_data[demand_scenario][size_scenario][facts][1][load_step][0]
        #             lambda_present = round(raw_data[demand_scenario][size_scenario][facts][1][load_step][0],3)
        #             if facts == 0:
        #                 lambda_old = lambda_present
        #                 lambda_ini_pos = load_step
        #             if facts > 0 and lambda_present == lambda_old:
        #                 lambda_ref = lambda_present
        #                 lambda_ref_pos = load_step
        #                 lambda_old = lambda_present
        #             elif facts > 0 and lambda_present != lambda_old:
        #                 break_code = 1
        #                 break
        #         except:
        #             break_code = 1
        #             break
        #     if break_code == 1:
        #         break