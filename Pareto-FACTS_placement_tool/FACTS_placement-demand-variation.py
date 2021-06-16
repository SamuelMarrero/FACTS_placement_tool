#############################################VARIABLES##################################################################

study_name = 'IEEE-14-prueba-100MVA-paper'
redirect_silence = 1
peak_valley = 0

#	Cargar paquetes
import os,sys,time,logging, logging.handlers
import cPickle as pickle
import psutil
import numpy as np
import matplotlib.pyplot as plt

process = psutil.Process(os.getpid())
execution_time = time.clock()

## CONFIGURAR PSSE PARA INICIALIZAR
PSSE_PATH = r'C:\Program Files (x86)\PTI\PSSE33\PSSBIN'
PSSEVERSION = 33

if PSSEVERSION==34:
   import psse34               # it sets new path for psspy
else:
   sys.path.append(PSSE_PATH)

"""
PYTHONLIB = r'C:\Program Files (x86)\PTI\PSSE34EXplore\PSSLIB'
PYTHONPRM = r'C:\Program Files (x86)\PTI\PSSE34EXplore\PSSPRM'
MODELFOLDER = r'C:\IEEE39'

sys.path.append(PYTHONLIB)
sys.path.append(PYTHONPRM)
"""

from FACTS_placer import load_parameters, load_case, scenario_creator, PV_study, demand_data_visualization

### CREAR ARBOL DE CARPETAS

program_path, psse_file_name_short, sid, num_of_zones, zones, num_of_areas, areas, owner, tolerance_val, neg_tolerance_val,\
fnsl_options, percent_mode, load_steps_MVA, load_steps_percent, value_all, apiopt, status, scalval, \
number_of_facts, type_of_device, facts_placement_parameters, placement_variable_names = load_parameters.load(study_name)

#   Creacion del arbol de carpetas
demand_data_path = os.path.join('', program_path + '\\demand_data')

test_systems_path = os.path.join('', program_path + '\\' + 'test_systems')

if not os.path.exists('studies'):
    os.makedirs('studies')
studies_path = os.path.join('', program_path + '\\' + 'studies')

os.chdir(studies_path)
if not os.path.exists(study_name):
    os.makedirs(study_name)
study_path = os.path.join('', studies_path + '\\' + study_name)

os.chdir(study_path)
if not os.path.exists('psse_files'):
    os.makedirs('psse_files')
psse_files_path = os.path.join('', study_path + '\\psse_files')

if not os.path.exists('results'):
    os.makedirs('results')
results_path = os.path.join('', study_path + '\\results')

if not os.path.exists('log'):
    os.makedirs('log')
log_path =  os.path.join('', study_path + '\\log')

## INICIAR LOG
log_on = 0
if log_on == 1:
    os.chdir(log_path)
    log_file = os.path.join('', study_name + '.log')

    script_logger = logging.getLogger(study_name)
    script_logger.setLevel(logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : %(levelname)s : %(message)s', filename=log_file,
                        filemode='w', )
    log_file_handler = logging.handlers.TimedRotatingFileHandler(filename=log_file, when="m", interval=1, backupCount=5)
    log_file_formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                           datefmt='%y-%m-%d %H:%M:%S')
    log_file_handler.setFormatter(log_file_formatter)
    script_logger.addHandler(log_file_handler)
    script_logger.info('Log for study %s (only errors)', study_name)

total_MVA_load = 0
ok_iter_value = 0.0

ierr_accc = 0
voltage_delta = 0
placement_study_data = []
PV_bus_data_matrix = []
data_load_step = [[],[],[],[],[],[]]

    #   Para recuperacion de datos
flag = 1
flag_load = 1
flag_gen = 1
nstr = 1

    #   Para crear configurar el estudio de PSSE
dfx_options = [1,0,0]
sub_file_name = 'IEEE-14-PV.sub'
mon_file_name = 'IEEE-14-PV.mon'
con_file_name = 'IEEE-14-PV.con'
dfx_file_name = 'IEEE-14-PV.dfx'

volt_control_deadband = 0.01
neg_volt_control_deadband = - volt_control_deadband

#    Para estudio PV
load_increment_value = 0
load_increment = 0
load_increment_percent = 0
gen_increment_percent = 0
load_increment_MVA = 0
gen_increment_MVA = 0
sav_format = '.sav'
sid_vref = 4
#sid_all = 9
# sid_gen = 5
# sid_load = 6
no_convergence = []
fail = 0
demand_variation_step = 5
demand_study_init = demand_variation_step
demand_study_end = 100
voltage_limit_low = 0.90
voltage_limit_high = 1.1
vref = 1
vref_min = 1

################################################   PROGRAMA    #########################################################
#   Inicializacion PSSE
sys.path.append(PSSE_PATH)
os.environ['PATH'] += ';' + PSSE_PATH

import psspy,redirect

#	Redirigir la salida de PSSE a Python
redirect.psse2py()


#	Inicializar PSSE
psspy.psseinit(1000)
os.chdir(test_systems_path)
psse_file_name = os.path.join('', psse_file_name_short + sav_format)

if redirect_silence == 1:
    psspy.report_output(6,'',[])
    psspy.progress_output(6,'',[])
    psspy.alert_output(6,'',[])
    psspy.prompt_output(6,'',[])

rdef = psspy.getdefaultreal()
idef = psspy.getdefaultint()
cdef = psspy.getdefaultchar()

#   Cargar el caso
tolerance_val, ierr_case = load_case.load(test_systems_path, psse_file_name,num_of_zones,zones, num_of_areas, areas, sid,
                                   dfx_options, sub_file_name, mon_file_name, con_file_name,dfx_file_name)
#tolerance = load_case.Case_configuration(num_of_zones, zones, num_of_areas, areas, sid)

os.chdir(psse_files_path )
ierr_opf_opt_file = psspy.write_opf_options_file()

#   Obtener datos del caso
nodes_load,load_buses,load_ids,xarray_mach,xarray_bus,xarray_id,xarray_wmod,len_buses,len_loads,len_gens,len_wmod,case_data = load_case.case_data_retriever(sid,flag,owner,flag_load,flag_gen)
ierr_buses, buses = psspy.abusint(-1,1,'NUMBER')

#   Cargar datos de escenarios de generacion/demanda
demand_data,demand_values,wind_gen_data,wind_gen_values,len_size_scenarios,size_scenarios = scenario_creator.load_scenarios_config_data(program_path,demand_data_path,study_name)
demand_scenarios_data, min_demand_position, max_demand_position, nodes_PQ_ratio, total_scaled_demand_data,load_share_data = \
    scenario_creator.demand_scenarios_preparer(demand_values,nodes_load[0],load_buses[0],distribution='data') # distribution = 'data' or 'fixed'

if peak_valley == 1:
    demand_scenarios_data = np.array([demand_scenarios_data[max_demand_position],demand_scenarios_data[min_demand_position]])
len_demand_scenarios = len(demand_scenarios_data)

WPP_scaled_data = scenario_creator.WPP_scenarios_preparer(wind_gen_values,facts_placement_parameters[1][0][2])

if percent_mode == 1:
    demand_scenarios_frequencies, load_increment_step_percent, load_increment_step_MVA = scenario_creator.scenarios_frequencies_calculator(total_scaled_demand_data,len_demand_scenarios,load_steps_percent)
else:
    demand_scenarios_frequencies, load_increment_step_percent, load_increment_step_MVA = scenario_creator.scenarios_frequencies_calculator(
        total_scaled_demand_data, len_demand_scenarios, load_steps_MVA)

####################### DEMAND SCENARIOS PLOT FOR TESTING ########################################
# demand_data_II = np.array(np.delete(demand_data,0,axis = 1))
# total_demand_data = np.sum(demand_values, axis=1)
# len_loads = len(demand_values[1])
# load_share_raw = np.divide(demand_values,total_demand_data[:, None])
# load_buses = [[i for i in range(1,len_loads+1)]]
#
# demand_map = demand_data_visualization.plot_demand_map(load_share_data,total_scaled_demand_data,len_loads,load_buses)
# demand_histogram = demand_data_visualization.plot_demand_histogram(total_scaled_demand_data[6000:8000])

#trans_demand_data = np.transpose(demand_scenarios_data)
#demand_data_visualization.nodes_load_histogram(trans_demand_data, bins=50, ylim=1000)
#WPP_gen_histogram = demand_data_visualization.plot_demand_histogram(WPP_scaled_data)
##################################################################################################

os.chdir(results_path)
with open('demand_frequencies.obj', 'wb') as save_frequencies_data:
    pickle.dump(demand_scenarios_frequencies, save_frequencies_data, protocol=-1)
with open('legend_data_file.obj', 'wb') as save_legend_data:
    pickle.dump(facts_placement_parameters, save_legend_data, protocol=-1)

if log_on == 1:
    script_logger.info('Number of scenarios in study = %d', len_demand_scenarios)

#   Crear escenarios generacion/demanda
facts_iteration = 0
no_wind_turbines = 0

load_values = [[],[]]
for demand_scenario in range(len_demand_scenarios): #

    placement_study_data = []

    psse_file_name = os.path.join('', psse_file_name_short + sav_format)
    tolerance, ierr_case = load_case.load(test_systems_path, psse_file_name, num_of_zones, zones, num_of_areas, areas, sid, dfx_options,
                   sub_file_name, mon_file_name, con_file_name,
                   dfx_file_name)

    if ierr_case != 0 and log_on == 1:
        script_logger.debug('Error code when loading case - %s = %d', psse_file_name,ierr_case)

    for load in range(len_loads):
        P_load = demand_scenarios_data[demand_scenario][load]
        Q_load = demand_scenarios_data[demand_scenario][load]*nodes_PQ_ratio[load]
        ierr_load_change = psspy.load_chng_4(load_buses[0][load],load_ids[0][load],[idef,idef,idef,idef,idef,idef],[P_load,Q_load,0.0,0.0,0.0,0.0])
        if ierr_load_change != 0 and log_on == 1:
            script_logger.warning('Error code when changing bus load = %d', ierr_load_change)

    #new_generation_data = scaled_demand_data[0:8760, 0]
    #generation_coefficients = xarray_mach[0]/totals
    #scaled_new_generation_data = np.transpose(np.array([np.divide(xarray_mach[0][row], new_generation_data) for row in range(len(xarray_mach[0]))]))
    #scaled_demand_data = np.subtract(scaled_demand_data, scaled_new_generation_data)
    for gen in range(len_gens):

        pass
    ierr_fnsl = psspy.fnsl(fnsl_options)

    load_case.opf_configurator()
    ierr_opf = psspy.nopf(sid,1)

    os.chdir(results_path)
    data_object_name = os.path.join('', psse_file_name_short + '-D-' + str(demand_scenario) + '.obj')
    with open(data_object_name, 'wb') as save_raw_data:

        for size_scenario in range(len_size_scenarios):

            os.chdir(psse_files_path)
            psse_file_name = os.path.join('', psse_file_name_short + '-D-' + str(demand_scenario) + '-S-' + str(size_scenario) +
                                          '-F-' + str(0) + sav_format)
            psspy.save(psse_file_name)

            placement_study_data.append([])
#   Establecer ubicacion del dispositivo FACTS
            facts = 0
            facts_iteration = 0
            for facts_iteration in range(number_of_facts + 1):
                facts_placement_parameters[1][facts][2] = size_scenarios[size_scenario][0]
                ierr_sid_subsystem = psspy.bsys(sid=sid, numbus=len_buses, buses=buses[0])

                if facts_iteration>0:
                    parameters = {}
                    for parameter in range(len(placement_variable_names)):
                        parameters.update(
                            {placement_variable_names[parameter]: facts_placement_parameters[1][facts][parameter]})

                    tolerance, ierr_case = load_case.load(psse_files_path, psse_file_name, num_of_zones, zones, num_of_areas, areas, sid, dfx_options,
                        sub_file_name, mon_file_name, con_file_name,dfx_file_name)
                    facts_bus = facts_placement_parameters[1][facts][0]
                    ierr_bus_volt, facts_bus_voltage = psspy.busdat(facts_bus, 'PU')

                    if ierr_case != 0 and log_on == 1:
                        script_logger.debug('Error code when loading case %s - = %d', psse_file_name, ierr_case)

                    ierr_sid_subsystem = psspy.bsys(sid=sid, numbus=len_buses, buses=buses[0])
                    # if facts_bus_voltage >= vref_min:
                    #     facts_placement_parameters[2][facts][2] = facts_bus_voltage
                    # else:
                    #     facts_placement_parameters[2][facts][2] = vref_min

                    if type_of_device == 'GENERATOR':
                        fake_bus_number = int(buses[0][-1]+1)
                        ierr_busbaskv, bus_baskv = psspy.busdat(parameters['ibus'], 'BASE')
                        ierr_busdat, gen_bus_volt = psspy.busdat(ibus=parameters['ibus'], string='PU')
                        ierr_bus_data = psspy.bus_data_3( fake_bus_number, realar1=bus_baskv,name='WPP')
                        ierr_branch_data = psspy.branch_data(parameters['ibus'], fake_bus_number,'1',realar1=0.0001,realar2=0.005)
                        ierr_plant = psspy.plant_data(fake_bus_number, intgar1=parameters['ibus'],realar1=parameters['Vref'])
                        ierr_ren_gen = psspy.machine_data_2(i=fake_bus_number,id='1', intgar6 = parameters['WMOD'],realar1 = WPP_scaled_data[demand_scenario],\
                                        realar2 = 0,realar3 = parameters['Qmax'],realar4 = parameters['Qmin'],\
                                        realar5 = parameters['Pmax'],realar6 = parameters['Pmin'],realar7 = parameters['Mbase'],\
                                        realar8 = parameters['Rsource'],realar9 = parameters['Zsource'],realari17 =  parameters['WMOD'] )
                        ierr_bus_type = psspy.bus_data(fake_bus_number, [2], [])


                    if type_of_device == 'FACTS':
                        ierr_facts, facts_placement_data = psspy.facts_data_2(name= facts_placement_parameters[0][facts],\
                                intgar1=parameters['ibus'],intgar3=parameters['WMOD'],intgar5=parameters['VSref'],\
                                intgar6=parameters['RemoteBus'],realari3= parameters['Vref'],realari4=parameters['Mbase'])

                    ierr_fnsl = psspy.fnsl(fnsl_options)

                    opf_buses = []
                    for bus in range(len_buses):
                        if buses[0][bus] != facts_placement_parameters[1][facts][0]:
                            opf_buses.append(buses[0][bus])

                    load_case.opf_configurator()
                    ierr_vref_subsystem = psspy.bsys(sid = sid_vref, numbus = len_buses -1, buses = opf_buses)
                    ierr_opf = psspy.nopf(sid=sid_vref, all =0)

                    # if type_of_device == 'GENERATOR':
                    #     ierr_plant = psspy.plant_data(ibus = parameters['ibus'], intgar1=parameters['ibus'] , realar1=gen_bus_volt)
                    #     ierr_fnsl = psspy.fnsl(fnsl_options)


                total_data = load_case.total_data_calculator(flag_load, flag_gen, sid, len_gens)
                ierr_busdata, bus_data_matrix = psspy.abusreal(sid, flag, ['PU', 'ANGLED'])
                ierr_zone_data, zone_loss_data_matrix = psspy.azonereal(-1, 2, ['PLOSS', 'QLOSS'])
                ierr_machdata, machine_power_matrix = psspy.amachreal(sid, flag, ['PGEN', 'QGEN'])
                ierr_maxdata, machine_max_power = psspy.amachreal(sid, flag, ['PMAX', 'PMIN'])
                machine_data_matrix = [xarray_bus[0],xarray_id[0],machine_power_matrix[0],zone_loss_data_matrix,machine_max_power]

                #   Estudio PV
                PV_data = [[],[],[],[],[],[],[],[],[],[],[],[]]
                load_increment_value = 0
                load_increment = 0
                scal_moto = 0
                ierr_scal = 0
                normative_violation = 0

                if percent_mode == 1:
                    [PV_data_scenario, ierr_scal] = PV_study.PV_study_percent(psse_files_path,psse_file_name,machine_data_matrix,
                                                        load_increment,psse_file_name_short,demand_scenario,size_scenario,
                                                        sav_format,apiopt, status, tolerance_val, neg_tolerance_val,
                                                        fnsl_options, value_all, dfx_options,sub_file_name, mon_file_name,
                                                        con_file_name, dfx_file_name,load_increment_percent,load_increment_step_percent,
                                                        sid, flag, ok_iter_value, facts, number_of_facts, PV_data, scal_moto,
                                                        ierr_scal, facts_placement_parameters, facts_iteration, total_data,
                                                        data_load_step, len_gens, flag_gen, scalval,True,normative_violation,voltage_limit_low,
                                                        voltage_limit_high)
                    placement_study_data[size_scenario].append(PV_data_scenario)

                if percent_mode == 0:
                    placement_study_data[demand_scenario][size_scenario].append(PV_study.PV_study_MVA(psse_files_path,
                                                        psse_file_name, num_of_zones, zones, num_of_areas,
                                                        areas,load_increment,psse_file_name_short,demand_scenario,size_scenario,
                                                        sav_format, apiopt, status, tolerance_val, neg_tolerance_val, fnsl_options,
                                                        value_all, dfx_options,sub_file_name, mon_file_name, con_file_name,
                                                        dfx_file_name,load_increment_MVA, load_increment_step_MVA,
                                                        sid, flag, ok_iter_value, facts,number_of_facts,PV_data,scal_moto,
                                                        ierr_scal,facts_placement_parameters, facts_iteration, total_data,
                                                        data_load_step, len_gens, flag_gen, scalval,True,normative_violation,voltage_limit_low,
                                                        voltage_limit_high))
                if ierr_scal != 0 and log_on == 1:
                    script_logger.debug('Error code when scaling load = %d', ierr_scal)
                if facts <= number_of_facts and facts_iteration > 0:
                    facts += 1
                print(psse_file_name)
        try:
            pickle.dump(placement_study_data, save_raw_data,protocol = -1)
        except:
            save_raw_data.close()
        save_raw_data.close()

os.chdir(results_path)

execution_time = time.clock()

a=0


