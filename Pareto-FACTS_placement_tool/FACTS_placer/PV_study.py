import psspy,os
from FACTS_placer import load_case

def power_flow (sid, flag, PV_data, value_all, apiopt, status, scalval, fnsl_options, data_load_step, total_data, len_gens, flag_gen,machine_data_matrix):
    #ierr_accc = psspy.accc_with_dsp_3(accc_missmatch,accc_options,'',dfx_file_name,accc_file_name,'','','')
    ierr_scal, scal_totals, scal_moto = psspy.scal_2(sid, value_all, apiopt, status, scalval)
    data_load_step = data_retrieval(sid, flag, flag_gen, data_load_step)
    generation_dispatcher(total_data, scalval, len_gens, machine_data_matrix)
    ierr_ordr = psspy.ordr(1)
    ierr_fnsl = psspy.fnsl(fnsl_options)
    data_load_step = data_retrieval(sid, flag, flag_gen, data_load_step)
    #voltage_control = gen_voltage_control(len_gens, case_data,fnsl_options)
    return ierr_scal,scal_moto,data_load_step

def mismatch_controller(sid, flag, PV_data, value_all, apiopt, status, scalval, fnsl_options, tolerance_val, neg_tolerance_val,data_load_step, total_data, len_gens, flag_gen,machine_data_matrix):
     mismatch_correction_iteration = 0

     for h in range(8):
         ierr_ordr = psspy.ordr(1)
         ierr_scal,scal_moto,data_load_step = power_flow(sid, flag, PV_data, value_all, apiopt, status, scalval, fnsl_options,data_load_step, total_data, len_gens, flag_gen,machine_data_matrix)

         ierr_mismatch, bus_max_mismatch, max_mismatch = psspy.maxmsm()
         if max_mismatch.real >= tolerance_val and scal_moto == 0 and ierr_scal == 0:
             scalval[1] -= 2
             mismatch_correction_iteration += 1
         if max_mismatch.real <= neg_tolerance_val and scal_moto == 0 and ierr_scal == 0:
             scalval[1] += 2
             mismatch_correction_iteration += 1
         else:
             break
     return max_mismatch,scal_moto, ierr_scal,scalval


def scalval_percent(load_increment_percent, total_data,scalval):
     gen_increment_percent = load_increment_percent*1.01
     load_increment_value = total_data[0].real* (1 + load_increment_percent)
     gen_increment_value = total_data[1].real* (1 + gen_increment_percent)
     scalval[0] = load_increment_value
     scalval[1] = gen_increment_value
     return scalval

def scalval_MVA(load_increment_MVA,total_data,scalval):
    gen_increment_MVA= load_increment_MVA*1.01
    load_increment_value = abs(total_data[0] + load_increment_MVA)
    gen_increment_value = total_data[1] + gen_increment_MVA
    scalval[0] = load_increment_value
    scalval[1] = gen_increment_value
    return scalval

def data_retrieval(sid, flag, flag_gen, data_load_step):
    ierr_gen_bus, gen_bus_list = psspy.amachint(sid, flag, 'NUMBER')
    ierr_gen_id, gen_id_list = psspy.amachchar(sid, flag, 'ID')
    ierr_gen_id, gen_owner_list = psspy.amachint(sid, flag, 'OWN1')
    ierr_gen, MW_gen_list = psspy.amachreal(sid, flag_gen, 'PGEN')
    ierr_gen, MWmax_gen_list = psspy.amachreal(sid, flag_gen, 'PMAX')
    ierr_machstat, gen_status_list = psspy.amachint(sid, flag, 'STATUS')


    data_load_step[0]= gen_bus_list
    data_load_step[1]= gen_id_list
    data_load_step[2]= gen_owner_list
    data_load_step[3]= MW_gen_list
    data_load_step[4]= MWmax_gen_list
    data_load_step[5]= gen_status_list
    return data_load_step


def generation_dispatcher(total_data,scalval,len_gens,generators_data):
    gen = 0
    for i in range(len_gens):
        generator_power_fraction = generators_data[2][gen]/total_data[1].real
        Pgen = scalval[1] * generator_power_fraction
        ierr_mach_data = psspy.machine_data_2(i=generators_data[0][gen],id=str(generators_data[1][gen]),realar1=Pgen)
        gen += 1


def setdata(PV_data, sid, flag, max_mismatch,load_increment_percent,scalval,facts_placement_parameters,facts, facts_iteration):
    ierr_busdata, bus_data_matrix = psspy.abusreal(sid, flag, ['PU', 'ANGLED'])
    ierr_machdata, machine_data_matrix = psspy.amachreal(sid, flag, ['PGEN', 'QGEN'])
    ierr_machbus, machine_bus_matrix = psspy.amachint(sid, flag, 'NUMBER')
    ierr_machvoltdata, machine_voltage_data_matrix = psspy.abusreal(sid, flag, ['PU', 'ANGLED'])
    ierr_loaddata, load_data_matrix = psspy.aloadcplx(sid, flag, ['MVAACT'])
    ierr_lineflow_data, line_flow_data_matrix = psspy.aflowreal(sid, 1, 1, flag, ['P', 'Q'])
    ierr_lineloss_data, line_loss_data_matrix = psspy.aflowreal(sid, 1, 1, flag, ['PLOSS', 'QLOSS'])
    ierr_transflow_data, transformer_flow_data_matrix = psspy.atrnreal(sid, 1, 1, flag, 1, ['P', 'Q'])
    ierr_trasnsloss_data, transformer_loss_data_matrix = psspy.atrnreal(sid, 1, 1, flag, 1, ['PLOSS', 'QLOSS'])
    ierr_zone_data, zone_loss_data_matrix = psspy.azonereal(-1,2,['PLOSS','QLOSS'])

    ierr_device_data, device_data_matrix = psspy.fcddat_2(facts_placement_parameters[0][facts],'QSHNT')

    if ierr_device_data != 0:
        bus = facts_placement_parameters[1][facts][0]
        ierr_device_data, device_data_matrix = psspy.macdat(ibus = bus, id = '1' , string = 'P')

    mismatch = max_mismatch

    PV_data[0].append(bus_data_matrix)
    PV_data[1].append([machine_bus_matrix,machine_data_matrix])
    PV_data[2].append(load_data_matrix)
    PV_data[3].append(line_flow_data_matrix)
    PV_data[4].append(line_loss_data_matrix)
    PV_data[5].append(transformer_flow_data_matrix)
    PV_data[6].append(transformer_loss_data_matrix)
    PV_data[7].append([load_increment_percent, scalval[0], scalval[1]])
    PV_data[8].append(mismatch)
    PV_data[9].append(zone_loss_data_matrix)
    PV_data[10].append(device_data_matrix)
    return PV_data


def PV_study_percent(psse_data_path, psse_file_name,machine_data_matrix,load_increment,psse_file_name_short,generation_scenario,demand_zone,sav_format,
                     apiopt, status, tolerance_val,neg_tolerance_val, fnsl_options, value_all, dfx_options,
                     sub_file_name,mon_file_name, con_file_name,dfx_file_name, load_increment_percent,
                     load_increment_step_percent, sid, flag,ok_iter_value,facts,number_of_facts,PV_data,scal_moto,ierr_scal,
                     facts_placement_parameters,facts_iteration,total_data, data_load_step, len_gens, flag_gen,scalval,
                    first_time,normative_violation,voltage_limit_low,voltage_limit_high):

     if load_increment_step_percent <= 0.01 or scal_moto != 0 or ierr_scal != 0:
         psspy.close_powerflow()
         return PV_data,ierr_scal
     else:
         normative_violation = 0

         scalval = scalval_percent(load_increment_percent, total_data,scalval)

         data_load_step = data_retrieval(sid, flag, flag_gen, data_load_step)
         max_mismatch, scal_moto, ierr_scal, scalval = mismatch_controller(sid, flag, PV_data, value_all, apiopt, status, scalval, fnsl_options, tolerance_val,
                                 neg_tolerance_val, data_load_step, total_data, len_gens, flag_gen,
                                 machine_data_matrix)


         ierr_voltage, voltages = psspy.abusreal(-1, flag, 'PU')
         ierr_line_flows, line_flows =  psspy.aflowreal(-1, 1, 1, flag, ['P', 'Q'])
         S_line_flows = []
         ierr_line_rate, line_rates_A = psspy.aflowreal(-1, 1, 1, flag, 'RATEA')
         ierr_line_rate, line_rates = psspy.aflowreal(-1, 1, 1, flag, 'PCTMVARATEA')

         termal_violation = 0
         try:
             max_rate = max(line_rates[0])
             min_rate = min(line_rates[0])
             max_voltage = max(voltages[0])
             min_voltage = min(voltages[0])
         except:
             max_rate = 0
             min_rate = 0
             max_voltage = 0
             min_voltage = 0
         if min_voltage < voltage_limit_low or max_voltage > voltage_limit_high or max_rate > 100 or min_rate < -100:
             normative_violation = 1

         if first_time == True:
             psse_file_name = os.path.join('', psse_file_name_short + '-D-' + str(generation_scenario) + '-S-' + str(demand_zone) +
                                          '-F-' + str(facts_iteration) + sav_format)
             psspy.save(psse_file_name)

         if abs(max_mismatch.real) <= tolerance_val and abs(max_mismatch.imag)<= tolerance_val and normative_violation == 0:

            setdata(PV_data, -1, flag, max_mismatch,load_increment_percent,scalval,facts_placement_parameters,
                                            facts, facts_iteration)


            PV_study_percent(psse_data_path, psse_file_name, machine_data_matrix, load_increment,psse_file_name_short,generation_scenario,demand_zone,sav_format,
                             apiopt, status, tolerance_val,neg_tolerance_val, fnsl_options,value_all, dfx_options,
                             sub_file_name,mon_file_name, con_file_name,dfx_file_name, load_increment_percent+load_increment_step_percent,
                             load_increment_step_percent, sid, flag,ok_iter_value,facts,number_of_facts,PV_data,scal_moto,ierr_scal,
                             facts_placement_parameters,facts_iteration,total_data, data_load_step, len_gens, flag_gen,scalval,False,normative_violation,voltage_limit_low,voltage_limit_high)

         else:
            PV_study_percent(psse_data_path, psse_file_name, machine_data_matrix, load_increment,psse_file_name_short,generation_scenario,demand_zone,sav_format,
                                apiopt, status, tolerance_val,neg_tolerance_val, fnsl_options,value_all, dfx_options,
                                sub_file_name,mon_file_name, con_file_name,dfx_file_name, load_increment_percent-load_increment_step_percent/2,
                                load_increment_step_percent/4, sid, flag,ok_iter_value,facts,number_of_facts,PV_data,scal_moto,ierr_scal,
                                facts_placement_parameters,facts_iteration,total_data, data_load_step, len_gens, flag_gen,scalval,False,normative_violation,voltage_limit_low,voltage_limit_high)

     return PV_data,ierr_scal


def PV_study_MVA(psse_data_path, psse_file_name, num_of_zones,zones, num_of_areas, areas,load_increment,psse_file_name_short,generation_scenario,demand_zone,sav_format,
                     apiopt, status, tolerance_val,neg_tolerance_val, fnsl_options, value_all, dfx_options,
                     sub_file_name,mon_file_name, con_file_name,dfx_file_name, load_increment_MVA,
                     load_increment_step_MVA, sid, flag,ok_iter_value,facts,number_of_facts,PV_data,scal_moto,ierr_scal,
                     facts_placement_parameters,facts_iteration,total_data, data_load_step, len_gens, flag_gen,scalval,first_time,normative_violation,voltage_limit_low,voltage_limit_high,case_data):


    if load_increment_step_MVA <= 1 or scal_moto != 0 or ierr_scal != 0:
        psspy.close_powerflow()
        return PV_data
    else:
        case_calculator = load_case.load(psse_data_path, psse_file_name, num_of_zones,zones, num_of_areas, areas, sid, dfx_options, sub_file_name,mon_file_name, con_file_name,dfx_file_name)

        scalval = scalval_MVA(load_increment_MVA, total_data, scalval)
        data_load_step = data_retrieval(sid, flag, flag_gen, data_load_step)
        max_mismatch,scal_moto,ierr_scal = mismatch_controller(sid, flag, PV_data, value_all, apiopt, status, scalval, fnsl_options,
                                           tolerance_val, neg_tolerance_val, data_load_step, total_data, len_gens,
                                           flag_gen)

        ierr_voltage, voltages = psspy.abusreal(sid, flag, 'PU')
        ierr_line_flows, line_flows = psspy.aflowreal(sid, 1, 1, flag, ['P', 'Q'])
        S_line_flows = []
        ierr_line_rate, line_rates = psspy.aflowreal(sid, 1, 1, flag, 'PCTRATE')
        termal_violation = 0
        try:
            max_rate = max(line_rates[0])
            min_rate = min(line_rates[0])
            max_voltage = max(voltages[0])
            min_voltage = min(voltages[0])
        except:
            max_rate = 0
            min_rate = 0
            max_voltage = 0
            min_voltage = 0
        if min_voltage < voltage_limit_low or max_voltage > voltage_limit_high or max_rate > 100 or min_rate < -100:
            normative_violation = 1

        if first_time == True:
            psse_file_name = os.path.join('', psse_file_name_short + '-D-' + str(generation_scenario) + '-S-' + str(
                demand_zone) + '-F-' + str(facts_iteration) + sav_format)
            psspy.save(psse_file_name)


        if abs(max_mismatch.real) <= tolerance_val and abs(max_mismatch.imag) <= tolerance_val:
            if ok_iter_value != scalval[0]:
                setdata(PV_data, sid, flag, max_mismatch,load_increment_MVA,scalval,facts_placement_parameters,facts, facts_iteration)
                ok_iter_value = scalval[0]

            PV_study_MVA(psse_data_path, psse_file_name, num_of_zones,zones, num_of_areas, areas,load_increment,psse_file_name_short,generation_scenario,demand_zone,sav_format,
                     apiopt, status, tolerance_val,neg_tolerance_val, fnsl_options, value_all, dfx_options,
                     sub_file_name,mon_file_name, con_file_name,dfx_file_name, load_increment_MVA+load_increment_step_MVA,
                     load_increment_step_MVA, sid, flag,ok_iter_value,facts,number_of_facts,PV_data,scal_moto,ierr_scal,
                     facts_placement_parameters,facts_iteration,total_data, data_load_step, len_gens, flag_gen,scalval,False,normative_violation,voltage_limit_low,voltage_limit_high)
        else:
            PV_study_MVA(psse_data_path, psse_file_name, num_of_zones,zones, num_of_areas, areas,load_increment,psse_file_name_short,generation_scenario,demand_zone,sav_format,
                     apiopt, status, tolerance_val,neg_tolerance_val, fnsl_options, value_all, dfx_options,
                     sub_file_name,mon_file_name, con_file_name,dfx_file_name, load_increment_MVA+load_increment_step_MVA,
                     load_increment_step_MVA/2, sid, flag,ok_iter_value,facts,number_of_facts,PV_data,scal_moto,ierr_scal,
                     facts_placement_parameters,facts_iteration,total_data, data_load_step, len_gens, flag_gen,scalval,False,normative_violation,voltage_limit_low,voltage_limit_high)
    return PV_data

