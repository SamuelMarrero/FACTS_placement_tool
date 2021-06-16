import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import cPickle as pickle
import csv
from pareto_functions import MI_calculator,MI_decision,pareto_frontier_plot,sensitivity_tester,week_random_number_generator,\
                                quartile_scenarios_calculator,day_random_number_generator
from plot_results import bins_tester_plot,MI_comparison_locations,comparative_Pareto,bar_mean_std_plot
import indicator_calculation
from mpl_toolkits.axisartist.axislines import SubplotZero

psse_file_name_short = 'IEEE 14 bus-NoShunt-paper'
study_name = "IEEE-14-prueba-100MVA-paper"

study_file_path = 'd:\Users\smarrero\Documents\Doctorado\Estudios\PSS-E\Pareto-FACTS_placement_tool\studies\\'+ study_name
data_file_path = study_file_path +'\\results'

len_demand_scenarios = 80
annual_scenarios = [scenario for scenario in xrange(0,len_demand_scenarios,1)]
reduced_scenarios = np.random.choice(range(8760), size = 400, replace= True)
reduced_scenarios_20 = np.random.choice(range(8760), size = 20, replace= True)
size_scenarios = 1
MI_test_flag = 0

labels = ['4','5','9','10','11','12','13','14'] #['.98','.99','1.00','1.01','1.02','1.03','1.04','1.05']

IEEE_gen_cost_coeffs = [[0.005, 2.45, 105],[0.005, 3.5, 44.1],[0.005, 3.89, 40.6],[0,0,0],[0,0,0],[0,6,0]]
paper_gen_cost_coeffs = [[0.03546, 38.30553, 1234.5311],[0.02111, 36.32782, 1658.5696],[0.01799, 38.27041, 1356.6592], [0,0,0], [0,0,0],[0,6,0]]
gen_emission_coeffs_paper = [[0.00683, -0.54551, 40.26690],[0.00461, -0.51160, 42.89553],[0.00461, -0.51160, 42.89553], [0,0,0],[0,0,0],[0,0,0]]

gen_cost_coeffs_Balakrishnan = [[0.0015, 7.92, 561],[0.00194, 7.85, 310],[0.00482, 7.97, 78], [0,0,0], [0,0,0],[0,6,0]]
gen_emission_coeffs_Balakrishnan = [[0.0126, -1.355, 22.983],[0.01375, -1.249, 137.37],[0.00765, -0.805, 363.7], [0,0,0], [0,0,0],[0,6,0]]

S_cost_coeff = 0.837
generated_power_price = 6
gen_cost_coeffs = paper_gen_cost_coeffs
gen_emission_coeffs = gen_emission_coeffs_paper

pareto_criterion = 'distance' # = 'distance', 'area' or 'consensus'

raw_data = []
os.chdir(data_file_path)

with open('demand_frequencies.obj', 'rb') as open_frequencies_data:
    load_step_freq_data = pickle.load(open_frequencies_data)

with open('legend_data_file.obj', 'rb') as open_legend_data:
    facts_parameters = pickle.load(open_legend_data)


for demand_scenario in xrange(len_demand_scenarios):
    data_object_name = os.path.join('', psse_file_name_short + '-D-' + str(demand_scenario) + '.obj')
    raw_data.append([])
    # print data_object_name
    with open(data_object_name, 'rb') as data_file:
        temp_data = pickle.load(data_file)
        raw_data[demand_scenario].append([])
        for size_scenario in xrange(len(temp_data)):
            volts = []
            total_p = []
            total_loss = []
            q_facts = []
            p_gen = []
            load = []
            for facts in xrange(len(temp_data[size_scenario])): #
                try:
                    volts = temp_data[size_scenario][facts][0]
                    total_p = temp_data[size_scenario][facts][7]
                    total_loss = temp_data[size_scenario][facts][9]
                    q_facts = temp_data[size_scenario][facts][10]
                    p_gen = temp_data[size_scenario][facts][1]
                    load = temp_data[size_scenario][facts][2]
                    raw_data[demand_scenario][size_scenario].append([volts, total_p, total_loss, q_facts,p_gen,load])
                except:
                    pass

indicators = ["Loading margin","Voltage deviation","Reactive power losses","Active power losses","Generated power","Costs","GHG emissions","p_load"]

annual_results = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff, annual_scenarios )
annual_med_std = indicator_calculation.med_std_calculator(annual_results,indicators)

indicators_reduced = ["Loading margin","Voltage deviation","Reactive power losses","Active power losses","Generated power","Costs","GHG emissions"]

annual_indicator_comparison = {indicator : annual_results[indicator]for indicator in indicators_reduced}
annual_indicator_comparison, indicators_filtered = indicator_calculation.MI_indicator_filter(annual_indicator_comparison,indicators_reduced)

indices_labels = indicators_reduced

if MI_test_flag == 1:
    sensitivity_values = [0.01,0.02,0.03,0.04,0.05,0.06]
    min_bins_values = [20,40,60,80]
    bins_tester_plot(sensitivity_values=sensitivity_values,min_bins=20,data=annual_indicator_comparison,MI_calculation='D')
    bins_tester_plot(sensitivity_values=sensitivity_values, min_bins=40, data=annual_indicator_comparison,
                     MI_calculation='D')
    bins_tester_plot(sensitivity_values=sensitivity_values, min_bins=60, data=annual_indicator_comparison,
                     MI_calculation='D')
    bins_tester_plot(sensitivity_values=sensitivity_values, min_bins=80, data=annual_indicator_comparison,
                     MI_calculation='D')
    MI_sensitivity_errors = sensitivity_tester(sensitivity_values=sensitivity_values, min_bins=min_bins_values,data=annual_indicator_comparison)
    x_argmin = np.argmin(np.min(MI_sensitivity_errors,axis=0))
    y_argmin = np.argmin(np.min(MI_sensitivity_errors,axis=1))

min_bins = 60
sensitivity = 0.04
MI_matrix, H_matrix, D_matrix, MI, MIh = MI_calculator(annual_indicator_comparison, sensitivity=sensitivity, min_bins=min_bins)
MI_val, MI_pos, MI_2d, MI_semi_val, MI_semi_pos, MI_val_sum, MI_set = MI_decision(D_matrix, desired_set_size=2, max=False)

indicators = [indicators_filtered[pos] for pos in MI_set]


# peak_scenario = [np.argmax(annual_results["p_load"])]
# valley_scenario = [np.argmin(annual_results["p_load"])]
#
# peak_results = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,peak_scenario)
# peak_med_std = indicator_calculation.med_std_calculator(peak_results,indicators)
#
# valley_results = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,valley_scenario)
# valley_med_std = indicator_calculation.med_std_calculator(valley_results,indicators)
#
# peak_scenario_lambda = [np.argmin(annual_results["lambda_ini"])]
# valley_scenario_lambda = [np.argmax(annual_results["lambda_ini"])]
#
# peak_results_lambda = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,peak_scenario_lambda)
# peak_med_std_lambda = indicator_calculation.med_std_calculator(peak_results_lambda,indicators)
#
# valley_results_lambda = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,valley_scenario_lambda)
# valley_med_std_lambda = indicator_calculation.med_std_calculator(valley_results_lambda,indicators)
#
pareto_front_data, pareto_front_positions = pareto_frontier_plot(annual_med_std[indicators[1]][0],annual_med_std[indicators[0]][0], indicators[1] + " decrease (p.u.)", indicators[0] + " increase (p.u.)",facts_parameters[0],maxX=True,maxY=True)
# pareto_front_data, pareto_front_positions = pareto_frontier_plot(peak_med_std["volt_dev"][0],peak_med_std["lambda"][0], "Voltage Deviation decrease (p.u.)","Loading Margin increase (p.u.)",facts_parameters[0],maxX=True,maxY=True)
# pareto_front_data, pareto_front_positions = pareto_frontier_plot(peak_med_std_lambda["volt_dev"][0],peak_med_std_lambda["lambda"][0], "Voltage Deviation decrease (p.u.)","Loading Margin increase (p.u.)",facts_parameters[0],maxX=True,maxY=True)
#
a = 0

# colors = ['red','gold','purple','violet','black'] #
# data_x = [peak_med_std["volt_dev"],valley_med_std["volt_dev"],peak_med_std_lambda["volt_dev"],valley_med_std_lambda["volt_dev"],annual_med_std["volt_dev"]] #
# data_y = [peak_med_std["lambda"],valley_med_std["lambda"],peak_med_std_lambda["lambda"],valley_med_std_lambda["lambda"],annual_med_std["lambda"]] #
# axes_labels = ["Voltage Deviation decrease (p.u.)","Loading Margin increase (p.u.)"]
# legend_labels = ['Peak', 'Valley', 'min-lambda', 'max-lambda', 'Annual']
#
# comparative_Pareto(data_x,data_y,labels=labels,colors=colors,axes_labels=axes_labels,legend_labels=legend_labels)
#
a=0

Q1_scenarios, Q2_scenarios, Q3_scenarios, Q4_scenarios = quartile_scenarios_calculator(annual_results['p_load'][0])

Q1_results = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,Q1_scenarios.astype('int'))
Q2_results = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,Q2_scenarios.astype('int'))
Q3_results = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,Q3_scenarios.astype('int'))
Q4_results = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,Q4_scenarios.astype('int'))

Q1_med_std = indicator_calculation.med_std_calculator(Q1_results,indicators)
Q2_med_std = indicator_calculation.med_std_calculator(Q2_results,indicators)
Q3_med_std = indicator_calculation.med_std_calculator(Q3_results,indicators)
Q4_med_std = indicator_calculation.med_std_calculator(Q4_results,indicators)

colors = ['gold','yellowgreen','limegreen','darkgreen','black'] #
data_x = [Q1_med_std[indicators[1]],Q2_med_std[indicators[1]],Q3_med_std[indicators[1]],Q4_med_std[indicators[1]],annual_med_std[indicators[1]]] #
data_y = [Q1_med_std[indicators[0]],Q2_med_std[indicators[0]],Q3_med_std[indicators[0]],Q4_med_std[indicators[0]],annual_med_std[indicators[0]]] #
axes_labels = [indicators[1] + " decrease (p.u.)",indicators[0] + " increase (p.u.)"]
legend_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Annual']

comparative_Pareto(data_x,data_y,labels=labels,colors=colors,axes_labels=axes_labels,legend_labels=legend_labels)

a=1
# Q1_scenarios, Q2_scenarios, Q3_scenarios, Q4_scenarios = quartile_scenarios_calculator(annual_results['load_share'])
#
# Q1_results = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,Q1_scenarios.astype('int'))
# Q2_results = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,Q2_scenarios.astype('int'))
# Q3_results = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,Q3_scenarios.astype('int'))
# Q4_results = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,Q4_scenarios.astype('int'))
#
# Q1_med_std = indicator_calculation.med_std_calculator(Q1_results,indicators)
# Q2_med_std = indicator_calculation.med_std_calculator(Q2_results,indicators)
# Q3_med_std = indicator_calculation.med_std_calculator(Q3_results,indicators)
# Q4_med_std = indicator_calculation.med_std_calculator(Q4_results,indicators)
#
# colors = ['yellow','yellowgreen','limegreen','darkgreen','black'] #
# data_x = [Q1_med_std["volt_dev"],Q2_med_std["volt_dev"],Q3_med_std["volt_dev"],Q4_med_std["volt_dev"],annual_med_std["volt_dev"]] #
# data_y = [Q1_med_std["lambda"],Q2_med_std["lambda"],Q3_med_std["lambda"],Q4_med_std["lambda"],annual_med_std["lambda"]] #
# axes_labels = ["Voltage Deviation decrease (p.u.)","Loading Margin increase (p.u.)"]
# legend_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Annual']
#
# comparative_Pareto(data_x,data_y,labels=labels,colors=colors,axes_labels=axes_labels,legend_labels=legend_labels)

# a = 2
#
# pareto_front_data, pareto_front_positions = pareto_frontier_plot(reduced_med_std["volt_dev"][0],reduced_med_std["lambda"][0], "Voltage Deviation decrease (p.u.)","Loading Margin increase (p.u.)",facts_parameters[0],maxX=True,maxY=True)
#
# colors = ['red','green','black'] #
# data_x = [reduced_med_std_20["volt_dev"], reduced_med_std["volt_dev"],annual_med_std["volt_dev"]] #
# data_y = [reduced_med_std_20["lambda"],reduced_med_std["lambda"],annual_med_std["lambda"]] #
# axes_labels = ["Voltage Deviation reduction (p.u.)","Loading Margin enhancement (p.u.)"]
# legend_labels = ['20 scenarios','400 scenarios', '8760 scenarios']
#
# comparative_Pareto(data_x,data_y,labels=labels,colors=colors,axes_labels=axes_labels,legend_labels=legend_labels)

def number_scenarios_optimiser(data,indicators_red,from_scenarios,to_scenarios,step_scenarios,num_of_samples,number_of_bins,range_of_samples):
    scenarios_reduced = []
    reduced_mean = []
    reduced_std = []
    sampling_data = []

    iter = 0
    for a in range(from_scenarios,to_scenarios,step_scenarios):
        #lognormal_scenarios = np.random.lognormal(mean=0.4, sigma=1.1, size=200) * 100
        bins_range = range(0, step_scenarios, int(step_scenarios*range_of_samples/number_of_bins))
        uniform_scenarios = np.random.uniform(size=num_of_samples) * int(step_scenarios*range_of_samples)
        test_scenarios = [int(i) for i in uniform_scenarios]
        scenarios_to_choose = range(a, a + step_scenarios, 1)
        total_data = indicator_calculation.calculator(data, facts_parameters, gen_emission_coeffs, gen_cost_coeffs,
                                                       S_cost_coeff, scenarios_to_choose)
        total_indicators = indicator_calculation.med_std_calculator(total_data, indicators_red)
        del(total_data)
        scenarios_reduced.append({indicator:[[],[]] for indicator in indicators_red})
        reduced_mean.append({indicator:[] for indicator in indicators_red})
        reduced_std.append({indicator:[] for indicator in indicators_red})
        sampling_data.append({indicator:[[] for bin in range(number_of_bins+2)] for indicator in indicators_red})
        for num_of_scenarios in test_scenarios: #20, len_demand_scenarios, 20
            scenarios = np.random.choice(scenarios_to_choose, size = num_of_scenarios, replace= True)
            sample_data = indicator_calculation.calculator(data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,scenarios)
            sample_indicators = indicator_calculation.med_std_calculator(sample_data,indicators_red)
            del (sample_data)
            for indicator in indicators_red:
                error = abs((total_indicators[indicator][0] - sample_indicators[indicator][0])/total_indicators[indicator][0])
                error_mean = abs(np.mean(error))
                scenarios_reduced[iter][indicator][0].append(num_of_scenarios)
                scenarios_reduced[iter][indicator][1].append(error_mean)

        for indicator in indicators_red:
            digit = np.digitize(scenarios_reduced[iter][indicator][0],bins_range)
            for sample in range(num_of_samples):
                bin = digit[sample]-1
                try:
                    sampling_data[iter][indicator][bin].append(scenarios_reduced[iter][indicator][1][sample])
                except:
                    pass
            for bin in range(number_of_bins-1):
                if len(sampling_data[iter][indicator][bin]) > 0:
                    reduced_mean[iter][indicator].append(np.mean(sampling_data[iter][indicator][bin]))
                    reduced_std[iter][indicator].append(np.std(sampling_data[iter][indicator][bin]))

            del(digit)

        # fig = plt.figure()
        # fig_1 = fig.add_subplot(2,1,1)
        # fig_2 = fig.add_subplot(2, 1, 2)

        fig_01 = plt.figure()
        fig_02 = plt.figure()
        fig_1 = fig_01.add_subplot(1,1,1)
        fig_2 = fig_02.add_subplot(1,1,1)
        for indicator in indicators_red:
            data_y = reduced_mean[iter][indicator]
            data_x = [bins_range[number] for number in range(len(reduced_mean[iter][indicator]))]
            std = reduced_std[iter][indicator]
            fig_1.errorbar(data_x,data_y, yerr= std, label=indicator)
            fig_2.scatter(scenarios_reduced[iter][indicator][0], scenarios_reduced[iter][indicator][1], label=indicator)

        fig_1.set_xlabel('Number of scenarios')
        fig_2.set_xlabel('Number of scenarios')
        fig_1.set_ylabel('Mean Relative Error by number of sceanrios (p.u.)')
        fig_2.set_ylabel('Mean Relative Error by candidate solutions (p.u.)')
        fig_1.legend()
        fig_2.legend()
        plt.show()
        plt.close()
        iter += 1

# Q1_scenarios, Q2_scenarios, Q3_scenarios, Q4_scenarios = quartile_scenarios_calculator(annual_results['p_load'][0])
#
# Q1_data = [raw_data[scenario] for scenario in Q1_scenarios]
# Q4_data = [raw_data[scenario] for scenario in Q4_scenarios]
#
# len_Q4 = len(Q4_scenarios)
#
# indicators_red = ['lambda','volt_dev','p_load'] #,'q_loss_ini'
# from_scenarios = 0
# num_of_samples = 200
# number_of_bins = 20
# range_of_samples = 0.2
# to_scenarios = len_Q4
# step_scenarios = len_Q4
#
# number_scenarios_optimiser(Q4_data,indicators_red,from_scenarios,to_scenarios,step_scenarios,num_of_samples,number_of_bins,range_of_samples)
# number_scenarios_optimiser(Q1_data,indicators_red,from_scenarios,to_scenarios,step_scenarios,num_of_samples,number_of_bins,range_of_samples)


def quartiles_calculator(data):
    Q1_value = np.percentile(data, 25)
    Q2_value = np.percentile(data, 50)
    Q3_value = np.percentile(data, 75)
    return [Q1_value,Q2_value,Q3_value]



# colors = ['gold','darkorange','red','lime','limegreen','darkgreen','black'] #
# data_x = [random_1_week_med_std["volt_dev"],random_10_week_med_std["volt_dev"]] #
# data_y = [random_1_week_med_std["lambda"],random_10_week_med_std["lambda"]] #
# axes_labels = ["Voltage Deviation reduction (p.u.)","Loading Margin enhancement (p.u.)"]
# legend_labels = ['1 week by weeks', '10 weeks by weeks', '20 weeks by weeks', '1 week by hours', '10 weeks by hours', '20 weeks by hours', 'Annual']
#
# comparative_Pareto(data_x,data_y,labels=labels,colors=colors,axes_labels=axes_labels,legend_labels=legend_labels)

# random_1_weeks = week_random_number_generator(51,1)
# random_10_weeks = week_random_number_generator(51,10)
# random_20_weeks = week_random_number_generator(51,20)
#
# random_1_weeks_hours = np.random.choice(len_demand_scenarios, size = 168, replace= False)
# random_10_weeks_hours = np.random.choice(len_demand_scenarios, size = 1680, replace= False)
# random_20_weeks_hours = np.random.choice(len_demand_scenarios, size = 3360, replace= False)
#
# random_1_week_results = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,random_1_weeks)
# random_1_week_results_hours = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,random_1_weeks_hours)
# random_10_week_results = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,random_10_weeks)
# random_10_week_results_hours = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,random_10_weeks_hours)
# random_20_week_results = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,random_20_weeks)
# random_20_week_results_hours = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,random_20_weeks_hours)
#
# random_1_week_med_std = indicator_calculation.med_std_calculator(random_1_week_results,indicators)
# random_1_week_med_std_hours = indicator_calculation.med_std_calculator(random_1_week_results_hours,indicators)
# random_10_week_med_std = indicator_calculation.med_std_calculator(random_10_week_results,indicators)
# random_10_week_med_std_hours = indicator_calculation.med_std_calculator(random_10_week_results_hours,indicators)
# random_20_week_med_std = indicator_calculation.med_std_calculator(random_20_week_results,indicators)
# random_20_week_med_std_hours = indicator_calculation.med_std_calculator(random_20_week_results_hours,indicators)
#
# colors = ['gold','darkorange','red','lime','limegreen','darkgreen','black'] #
# data_x = [random_1_week_med_std["volt_dev"],random_10_week_med_std["volt_dev"],random_20_week_med_std["volt_dev"],random_1_week_med_std_hours["volt_dev"],random_10_week_med_std_hours["volt_dev"],random_20_week_med_std_hours["volt_dev"],annual_med_std["volt_dev"]] #
# data_y = [random_1_week_med_std["lambda"],random_10_week_med_std["lambda"],random_20_week_med_std["lambda"],random_1_week_med_std_hours["lambda"],random_10_week_med_std_hours["lambda"],random_20_week_med_std_hours["lambda"],annual_med_std["lambda"]] #
# axes_labels = ["Voltage Deviation reduction (p.u.)","Loading Margin enhancement (p.u.)"]
# legend_labels = ['1 week by weeks', '10 weeks by weeks', '20 weeks by weeks', '1 week by hours', '10 weeks by hours', '20 weeks by hours', 'Annual']
#
# comparative_Pareto(data_x,data_y,labels=labels,colors=colors,axes_labels=axes_labels,legend_labels=legend_labels)
#
# ######################################################################################################################3
# summer_scenarios = [scenario for scenario in xrange(49,2065,1)]
# fall_scenarios = [scenario for scenario in xrange(2066,4082,1)]
# winter_scenarios = [scenario for scenario in xrange(4587,6603,1)]
# spring_scenarios = [scenario for scenario in xrange(6604,8620,1)]
# alternating_scenarios = [scenario for scenario in xrange(696,8760,4)]
#
# summer_results = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,summer_scenarios)
# fall_results = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,fall_scenarios)
# winter_results = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,winter_scenarios)
# spring_results = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,spring_scenarios)
# alternating_results = indicator_calculation.calculator(raw_data,facts_parameters, gen_emission_coeffs, gen_cost_coeffs, S_cost_coeff,alternating_scenarios)
#
# summer_med_std = indicator_calculation.med_std_calculator(summer_results,["volt_dev","volt_angle_dev","lambda","p_loss","q_loss","costs","emissions"])
# fall_med_std = indicator_calculation.med_std_calculator(fall_results,["volt_dev","volt_angle_dev","lambda","p_loss","q_loss","costs","emissions"])
# winter_med_std = indicator_calculation.med_std_calculator(winter_results,["volt_dev","volt_angle_dev","lambda","p_loss","q_loss","costs","emissions"])
# spring_med_std = indicator_calculation.med_std_calculator(spring_results,["volt_dev","volt_angle_dev","lambda","p_loss","q_loss","costs","emissions"])
# alternating_med_std = indicator_calculation.med_std_calculator(alternating_results,["volt_dev","volt_angle_dev","lambda","p_loss","q_loss","costs","emissions"])

# colors = ['orange','red','green','grey','blue','black'] #
# data_x = [spring_med_std["volt_dev"],summer_med_std["volt_dev"],fall_med_std["volt_dev"],winter_med_std["volt_dev"],annual_med_std["volt_dev"],alternating_med_std['volt_dev']] #
# data_y = [spring_med_std["lambda"],summer_med_std["lambda"],fall_med_std["lambda"],winter_med_std["lambda"],annual_med_std["lambda"],alternating_med_std['lambda']] #
# labels = ['4','5','9','10','11','12','13','14']
# axes_labels = ["Voltage Deviation reduction (p.u.)","Loading Margin enhancement (p.u.)"]
#
# comparative_Pareto(data_x,data_y,labels=labels,colors=colors,axes_labels=axes_labels)

#critic_node, critic_nodes_freq = mode(np.array(critic_nodes), axis = 0)

facts_buses = np.array(facts_parameters[0])
#bar_mean_std_plot(annual_med_std['volt_dev'][0], annual_med_std['volt_dev'][1], facts_buses,"Voltage deviation decrease")
# bar_mean_std_plot(annual_med_std['volt_angle_dev'][0], annual_med_std['volt_angle_dev'][1], facts_buses,"Voltage angle deviation reduction")
# bar_mean_std_plot(annual_med_std['lambda'][0], annual_med_std['lambda'][1], facts_buses,"Loading margin increase")
# bar_mean_std_plot(annual_med_std['emissions'][0], annual_med_std['emissions'][1], facts_buses,"GHG emissions reduction")
# bar_mean_std_plot(annual_med_std['p_loss'][0], annual_med_std['p_loss'][1], facts_buses,"Active power loss decrease")
# bar_mean_std_plot(annual_med_std['q_loss'][0], annual_med_std['q_loss'][1], facts_buses,"Reactive power loss reduction")
# bar_mean_std_plot(annual_med_std['costs'][0], annual_med_std['costs'][1], facts_buses,"Operating costs reduction")
# bar_mean_std_plot(generated_power_med_std[0], generated_power_med_std[2], facts_buses,"Generated active power reduction")
# bar_mean_std_plot(p_gen1_med_std[0], p_gen1_med_std[2], facts_buses,"Gen 1 active power reduction")
# bar_mean_std_plot(p_gen2_med_std[0], p_gen2_med_std[2], facts_buses,"Gen 2 active power reduction")
# bar_mean_std_plot(p_gen3_med_std[0], p_gen3_med_std[2], facts_buses,"Gen 3 active power reduction")

#MI_plot_3d(D_matrix, indices_labels)

# pareto_front_data, pareto_front_positions = pareto_frontier_plot(annual_med_std["volt_dev"][0],annual_med_std["lambda"][0], "Voltage Deviation decrease (p.u.)","Loading Margin increase (p.u.)",facts_parameters[0],maxX=True,maxY=True)
# pareto_front_data, pareto_front_positions = pareto_frontier_plot(emissions_med_std[0], costs_med_std[0], "E (%)","AC (%)",facts_parameters[0],maxX=True,maxY=True)

# number_of_features = len(indices_results)
# MI_comparison_locations(number_of_features,facts,MI,MIh)
# MI_comparison_single(number_of_features,5,MI,MIh)
# MI_comparison_single(number_of_features,4,MI,MIh)
# MI_comparison_single(number_of_features,3,MI,MIh)

os.chdir(study_file_path)

export_2d_matrices = open('2d-matrices-min-FACTS-filtered.csv', 'w')
with export_2d_matrices:
    writer = csv.writer(export_2d_matrices, delimiter=';')
    writer.writerow(indices_labels)
    writer.writerow(['2d MI matrix'])
    writer.writerows(MI_matrix)
    writer.writerow(['2d H matrix'])
    writer.writerows(H_matrix)
    writer.writerow(['2d D matrix'])
    writer.writerows(D_matrix)
    # writer.writerow(['ECM MI matrix'])
    # writer.writerow(sensitivity_values)
    # writer.writerow(min_bins_values)
    # writer.writerows(MI_sensitivity_errors)

export_MI_2d = open('D-2d-min-FACTS-filtered.csv', 'w')
with export_MI_2d:
    writer = csv.writer(export_MI_2d, delimiter=';')
    writer.writerow(indices_labels)
    writer.writerow(['Max Values from 3d matrix'])
    writer.writerows(MI_2d)
    writer.writerow(['Max Values from 2d matrix'])
    writer.writerow(MI_semi_val)
    writer.writerow(MI_semi_pos)
    writer.writerow(['Aggregated Values'])
    writer.writerow(MI_val[0])
    writer.writerow(MI_pos)
    writer.writerow(['Summated Values'])
    writer.writerow(MI_val_sum[0])
    writer.writerow(['Feature Set'])
    writer.writerow(MI_set)

x_pos = np.arange(len(MI_val[0]))
fig = plt.subplot(1,1,1)
fig.bar(x_pos,MI_val[0],align='center')
fig.set_xlabel('Index')
fig.set_ylabel('Aggregated MI score')
plt.show()

a = 0

######################################################################################################

#voltage_plot = bar_mean_std_plot(volt_dev_med_std[0], volt_dev_med_std[2], facts_buses,'VD')
#qloss_plot = bar_mean_std_plot(qloss_med_std[0], qloss_med_std[2], facts_buses, 'Reactive power losses')
#costs_plot = bar_mean_std_plot(costs_med_std[0], costs_med_std[2], facts_buses, 'AC')
#emissions_plot = bar_mean_std_plot(emissions_med_std[0], emissions_med_std[2], facts_buses, 'E')

#voltage_profile_mean = np.mean(np.array(voltage_data),axis=1)
#voltage_profile_std = np.std(np.array(voltage_data),axis=1)
#voltage_profile_plot(voltage_profile_mean[1:], voltage_profile_std[1:], [5,6] ,facts_buses)


#####################################################################################################

#device_placement = pareto_final_decision(pareto_front_data, best_positions_lambda, best_positions_qloss, 0.5, pareto_front_positions, 'distance')
#LCOE_digging_plot(LCOE_diff,cost_diff, power_diff,facts_buses)

#####################################################################################################

# def results_sorter(vector,buses):
#     positions= [1,2,3,4,5,6,7,8]
#     vector_order = np.flip(np.argsort(vector, axis=-1),axis=0)
#     buses_sorted = buses[vector_order[:]]
#     ranks = np.empty_like(vector)
#     ranks[vector_order] = np.arange(len(vector))
#     return buses_sorted,ranks
#
# sorted_bus_labmda, order_lambda = results_sorter(lambda_med_std[0],facts_buses)
# sorted_bus_volt_dev, order_volt_dev = results_sorter(volt_dev_med_std[0],facts_buses)
# sorted_bus_qloss, order_qloss = results_sorter(qloss_med_std[0],facts_buses)
# sorted_bus_costs, order_costs = results_sorter(costs_med_std[0],facts_buses)
# sorted_bus_emissions, order_emissions = results_sorter(emissions_med_std[0],facts_buses)
#
# ordered_indices_results = np.array([order_lambda, order_volt_dev, order_qloss, order_costs, order_emissions])
#
# ordered_indices_array = [order_lambda, order_volt_dev, order_qloss, order_costs, order_emissions]

#plot = plot_indices_comparison(ordered_indices_results,facts_buses,["LM","VD","QLoss","Costs","Emi"])


####################################################################################################

demand_increment = 0.035
interest_rate = 0.5
transmitted_power_price = 8.37
investment = 2969617

# annual_business_value = transmitted_power_price*sumatorio_p*8
# annual_qloss_cost = transmitted_power_price*0.1*sumatorio_p*8

#npv_positions, npv_values = npv_decision(pareto_front_positions,lambda_absolute_min,Sloss_med_std[1],demand_increment,interest_rate,annual_business_value,annual_qloss_cost,investment)

#scenarios_plot(valores_diferencia_lambda,resultado_med_std,facts_parameters)

# LCOE = np.divide(np.add(system_costs,device_costs),np.add(system_power,device_power))
# LCOE_diff = np.array([(LCOE[0] - LCOE[i])/LCOE[0] for i in xrange(1,facts_number)])
# LCOE_order = np.argsort(LCOE_diff, axis=0)
# LCOE_sorted = facts_buses[LCOE_order[:]]
# LCOE_inverse = np.divide(1,LCOE)

# best_positions_lambda = np.array(best_placement_frequencies(valores_lambda))
# best_positions_qloss = np.array(best_placement_frequencies(valores_qloss))
# best_positions_Sloss = np.array(best_placement_frequencies(valores_Sloss))
# best_positions_costs = np.array(best_placement_frequencies(valores_costs))
#
# best_positions_lambda_data = stats.mode(best_positions_lambda)
# best_positions_qloss_data = stats.mode(best_positions_qloss)
# best_positions_Sloss_data = stats.mode(best_positions_Sloss)

# pareto_3d_data = pareto_3d_data_peparer(lambda_med_std[0],costs_med_std[0],emissions_med_std[0])
# pareto_3d_set = pareto_frontier_multi(pareto_3d_data)
# pareto_3d_plot(pareto_3d_data,pareto_3d_set,facts_buses)

# lambda_absolute_med = np.mean(np.array(lambda_absolute_values),axis=1)
# lambda_absolute_min = np.min(np.array(lambda_absolute_values),axis=1)
# lambda_absolute_min_pos = np.argmin(np.array(lambda_absolute_values),axis=1)

# critic_nodes_vector = np.array(critic_nodes)
# mode_critic_nodes = stats.mode(critic_nodes_vector)
# critic_nodes_frequencies = np.bincount(critic_nodes_vector)

##################################################################################################################

# final_solution = []
# index = []
# weights = []
# iteration = 0
# for criteria_weight in range(0,11,1):
#     criteria_weight = float(criteria_weight)/10
#     final_solution.append([])
#     weights.append([criteria_weight])
#     index.append([])
#     final_solution[iteration],index[iteration] = pareto_final_decision(pareto_front_data, best_positions_lambda_data,
#                                             best_positions_qloss_data, criteria_weight,
#                                             pareto_front_positions, pareto_criterion,[lambda_sum_norm, qloss_sum_norm],[lambda_med_std[0],qloss_med_std[0]])
#     iteration += 1
#
# index_transp = np.transpose(np.array(index))
# for solution in range(len(pareto_front_positions)):
#     name = facts_parameters[0][pareto_front_positions[solution]]
#     x = weights
#     y = index_transp[solution]
#     plt.plot(x, y,label=name)
# plt.legend()
# plt.xlabel("Prioridad de Lambda frente a Qloss")
# plt.ylabel("Distancia al punto optimo")
# plt.show()



