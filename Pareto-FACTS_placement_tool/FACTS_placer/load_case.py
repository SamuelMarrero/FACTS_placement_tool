import os, psspy
def  Case_configuration(num_of_zones,zones, num_of_areas, areas, sid, dfx_options, sub_file_name, mon_file_name, con_file_name, dfx_file_name): #
    ierr_subsystem = psspy.bsys(sid,0,[0.0],num_of_areas,areas,0,[],0,[],num_of_zones,zones)
        #   Crear archivo dfx
    #ierr_dfx = psspy.dfax_2(dfx_options, sub_file_name, mon_file_name, con_file_name, dfx_file_name)
    ierr_new_tol, tolerance_val = psspy.newton_tolerance()
    return tolerance_val

def load (psse_data_path, psse_file_name,num_of_zones,zones, num_of_areas, areas,sid, dfx_options, sub_file_name, mon_file_name, con_file_name,
                                     dfx_file_name):
    os.chdir(psse_data_path)
    ierr_case = psspy.case(psse_file_name)
    tolerance_val = Case_configuration(num_of_zones,zones, num_of_areas, areas, sid, dfx_options, sub_file_name, mon_file_name, con_file_name,dfx_file_name)#

    return tolerance_val,ierr_case

def data(sid,flag,owner):
    case_data = []
    ierr_busnum, bus_number_vector = psspy.abusint(sid,flag,'NUMBER')
    ierr_machnum, machine_number_vector = psspy.amachint(sid,flag,'NUMBER')
    ierr_machid, machine_id_vector = psspy.amachchar(sid, flag, 'ID')
    ierr_loadnum, load_number_vector = psspy.aloadint(sid,flag,'NUMBER')
    ierr_linenum, line_number_matrix = psspy.abrnint(sid,owner,1,flag,1,['FROMNUMBER','TONUMBER'])
    ierr_transnum, transformer_number_matrix = psspy.atrnint(sid,owner,1,flag,1,['FROMNUMBER','TONUMBER'])

    case_data.append(bus_number_vector)
    case_data.append(machine_number_vector)
    case_data.append(machine_id_vector)
    case_data.append(load_number_vector)
    case_data.append(line_number_matrix)
    case_data.append(transformer_number_matrix)
    return case_data

def total_data_calculator(flag_load,flag_gen,sid,len_gens):
    total_MVA_load = 0.00
    total_MW_gen = 0.00
    totals_data = []
    load = 0
    gen = 0
    ierr_load, MVA_load =  psspy.aloadcplx(sid, flag_load,'MVAACT')
    ierr_gen, MW_gen = psspy.amachcplx(sid,flag_gen,'PQGEN')
    len_loads = len(MVA_load[0])
    for h in range (len_loads):
        total_MVA_load += MVA_load[0][load]
        load += 1
    for i in range (len_gens):
        total_MW_gen += MW_gen[0][gen]
        gen += 1
    totals_data.append(total_MVA_load)
    totals_data.append(total_MW_gen)
    return totals_data

def case_data_retriever(sid,flag,owner,flag_load,flag_gen):
    ierr, load_values = psspy.aloadcplx(sid, flag, 'MVAACT')
    ierr_bus_numbers, load_buses = psspy.alodbusint(sid,flag,'NUMBER')
    ierr_bus_id, load_ids = psspy.aloadchar(sid,flag,'ID')
    ierr_mach, xarray_mach = psspy.amachreal(sid, flag, 'PGEN')
    ierr_mach, xarray_bus = psspy.amachint(sid, flag, 'NUMBER')
    ierr_mach, xarray_id = psspy.amachchar(sid, flag, 'ID')
    ierr_wmod, xarray_wmod = psspy.amachint(sid, flag, 'WMOD')

    #   Recabar datos del sistema
    case_data = data(sid, flag, owner)
    len_buses = len(case_data[0][0])
    len_loads = len(case_data[3][0])
    len_gens = len(case_data[1][0])
    len_wmod = len(xarray_wmod[0])

    return load_values,load_buses,load_ids,xarray_mach,xarray_bus,xarray_id,xarray_wmod,len_buses,len_loads,len_gens,len_wmod,case_data

def opf_configurator():
    ieer_opf_fuel_costs = psspy.minimize_fuel_cost(1)
    ierr_opf_min_reactances = psspy.minimize_series_comp(0)
    ier_opf_bus_shunts = psspy.minimize_adj_bus_shunts(0)
    ierr_opf_plosses = psspy.minimize_p_losses(0)
    ierr_opf_qlosses = psspy.minimize_q_losses(0)
    ierr_opf_bus_volt_feasibility = psspy.open_bus_voltage_limits(1)
    ierr_opf_sched_volt = psspy.opf_use_generator_vsched(1)
    ierr_Q_loss_coeff = psspy.q_losses_cost_coeff(0.837)
    ierr = psspy.p_losses_cost_coeff(0.837)
    ierr_fix_generators = psspy.opf_fix_all_generators(0)
