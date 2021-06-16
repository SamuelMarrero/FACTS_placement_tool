import os
import numpy as np
from numpy import mean, absolute
#from statsmodels import robust
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
#from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.axisartist.axislines import SubplotZero
from sklearn import feature_selection, metrics
import scipy, operator, itertools

def week_random_number_generator(number_of_weeks,number_of_samples):

    numbers = np.random.choice(number_of_weeks , size = number_of_samples,replace = False)

    random_scenarios = np.array([])
    for number in numbers:
        if number > 0:
            random_week_scenario = int(49 + number * 168)
        elif number == 0:
            random_week_scenario = int(49)
        week_scenarios = range(random_week_scenario, random_week_scenario + 168, 1)
        random_scenarios = np.append(random_scenarios, week_scenarios)
    return random_scenarios.astype(int)

def day_random_number_generator(number_of_days,number_of_samples):
    numbers = np.random.choice(number_of_days, size=number_of_samples, replace=False)

    random_scenarios = np.array([])
    for number in numbers:
        random_day_scenario = int(number * 24)
        week_scenarios = range(random_day_scenario, random_day_scenario + 24, 1)
        random_scenarios = np.append(random_scenarios, week_scenarios)
    return random_scenarios.astype(int)

def quartile_scenarios_calculator (data):
    len_data = len(data)
    Q1_value = np.percentile(data,25)
    Q2_value = np.percentile(data,50)
    Q3_value = np.percentile(data,75)

    Q1_scenarios = np.array([])
    Q2_scenarios = np.array([])
    Q3_scenarios = np.array([])
    Q4_scenarios = np.array([])

    for scenario in range(len_data):
        if data[scenario] < Q1_value:
            Q1_scenarios = np.append(Q1_scenarios,int(scenario))
        if data[scenario] > Q1_value and data[scenario] < Q2_value:
            Q2_scenarios = np.append(Q2_scenarios,int(scenario))
        if data[scenario] > Q2_value and data[scenario] < Q3_value:
            Q3_scenarios = np.append(Q3_scenarios,int(scenario))
        if data[scenario] > Q3_value:
            Q4_scenarios = np.append(Q4_scenarios,int(scenario))
    return Q1_scenarios.astype(int), Q2_scenarios.astype(int), Q3_scenarios.astype(int), Q4_scenarios.astype(int)

def data_preparer(vector):
    array = np.array(vector)
    array_maximum = np.ndarray.max(array,axis=0)
    array_minimum = np.ndarray.min(array,axis=0)

    array_normalized = np.array([])
    for i in range(len(array)):
        array_normalized = np.append(array_normalized,(array[i]-array_minimum)/(array_maximum-array_minimum))

    return array, array_maximum, array_minimum,array_normalized

def pareto_data_constructor(vector):
    vector_numpy = np.array(vector)
    media = np.array([])
    std = np.array([])
    mad = np.array([])
    for j in range(len(vector_numpy)):
        if len(vector_numpy[j]) == 0:
            vector_numpy[j] = np.append(vector_numpy[j],0)
        media = np.append(media,np.mean(vector_numpy[j]))
        std = np.append(std,np.std(vector_numpy[j]))

    # dispersion_datos = []
    # for j in range(len(vector_numpy)):
    #     dispersion_datos.append([])
    #     for k in range(len(vector_numpy[j])):
    #         dispersion_datos[j].append(abs(vector[j][k]-media[j]))
    #
    # dispersion_datos_np = np.array(dispersion_datos)
    # max_dispersion = []
    # min_dispersion = []
    # for k in range(len(dispersion_datos_np)):
    #     max_dispersion.append(np.max(dispersion_datos_np[k]))
    #     min_dispersion.append(np.min(dispersion_datos_np[k]))
    #
    # std_norm = []
    # mad_norm = []
    # dispersion_datos_norm = []
    # for j in range(len(dispersion_datos)):
    #     dispersion_datos_norm.append([0])
    #     for k in range(len(dispersion_datos[j])):
    #         dispersion_datos_norm[j] += (1-(dispersion_datos[j][k] - min_dispersion[j])/(max_dispersion[j] - min_dispersion[j]))
    #     mad_norm.append(dispersion_datos_norm[j]/(len(dispersion_datos[j])-1))
    #     std_norm.append(dispersion_datos_norm[j]/(len(dispersion_datos[j])-1))

    med_std = [media,std]
    return med_std

def best_placement_frequencies(data):
    len_scenarios = len(data[0])
    len_positions = len(data)
    best_positions = []
    for scenario in range(len_scenarios):
        #best_positions.append([])
        value_old = 0
        for position in range(len_positions):
            value = data[position][scenario]
            if value > value_old:
                best_positions.append(position)

    return best_positions

def pareto_frontier_plot(Xs, Ys,x_label,y_label,labels, maxX=True, maxY=True):

    '''Pareto frontier selection process'''
    sorted_list = sorted([[Xs[i], Ys[i]] for i in range(0,len(Xs))], reverse=maxY)
    sorted_positions = np.argsort(Xs)
    sorted_positions = np.flip(sorted_positions,axis=0)
    pareto_front = [sorted_list[0]]
    pareto_front_position = [sorted_positions[0]]
    position = 1
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] >= pareto_front[-1][1]:
                pareto_front.append(pair)
                pareto_front_position.append(sorted_positions[position])
        else:
            if pair[1] <= pareto_front[-1][1]:
                pareto_front.append(pair)
                pareto_front_position.append(sorted_positions[position])
        position += 1

    '''Plotting process'''
    plt.rc('font', size=11)
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=11)
    plt.rc('ytick', labelsize=11)
    plt.rc('legend', fontsize=11)
    fig_1 = plt.figure()
    figure_1 = SubplotZero(fig_1,111)
    fig_1.add_subplot(figure_1)
    for i in range(len(labels)):
        if Xs[i] >= 0 and Ys[i]>=0:
            figure_1.scatter(Xs[i], Ys[i], color='grey')
        else:
            figure_1.scatter(Xs[i], Ys[i], color='red')

    pf_X = [pair[0] for pair in pareto_front if pair[0] >= 0 and pair[1] >= 0]
    pf_Y = [pair[1] for pair in pareto_front if pair[0] >= 0 and pair[1] >= 0]
    figure_1.plot(pf_X, pf_Y,color='green')
    figure_1.scatter(pf_X, pf_Y, color='green')
    #figure_1.plot()
    iteration = 0
    for j in range(len(labels)):
        label = labels[iteration]
        x = Xs[iteration]
        y = Ys[iteration]
        figure_1.annotate(
            label,
            xy=(x, y), xytext=(20, -30),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='lightgrey', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        iteration += 1

    # for direction in ["xzero", "yzero"]:
    #     # adds X and Y-axis from the origin
    #     figure_1.axis[direction].set_visible(True)
    #
    # for direction in ["left", "right", "bottom", "top"]:
    #     # hides borders
    #     figure_1.axis[direction].set_visible(False)

    #plt.title( x_label + ' vs. ' + y_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
   # plt.fill([0,0.1,0.1,0],[0,0,0.04,0.04],'b', alpha=0.2)
    plt.grid()
    plt.show()
    return pareto_front, pareto_front_position

def is_pareto_efficient_dumb(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
    return is_efficient

def pareto_3d_data_peparer(data1, data2, data3):
    len_points = len(data1)
    data = np.zeros((len_points,3))
    for i in range(len_points):
        data[i][0] = data1[i]
        data[i][1] = data2[i]
        data[i][2]= data3[i]
    return data

def pareto_frontier_multi(myArray):
    # Sort on first dimension
    myArray = myArray[myArray[:,0].argsort()]
    # Add first row to pareto_frontier
    pareto_frontier = myArray[0:1,:]
    # Test next row against the last row in pareto_frontier
    for row in myArray[1:,:]:
        if sum([row[x] >= pareto_frontier[-1][x]
                for x in range(len(row))]) == len(row):
            # If it is better on all features add the row to pareto_frontier
            pareto_frontier = np.concatenate((pareto_frontier, [row]))
    return pareto_frontier

def pareto_final_decision(pareto_set, lambda_frequencies, qloss_frequencies, criteria_weight,pareto_front_positions,criterion):
    len_pareto_set = len(pareto_set)

    # lambda_freq_postions = lambda_frequencies.mode
    # lambda_freq_value = lambda_frequencies.count
    #
    # qloss_freq_postions = qloss_frequencies.mode
    # qloss_freq_value = qloss_frequencies.count

    index = []
    if criterion == 'distance':
        max = np.amax(pareto_set, axis = 0)
        min = np.amin(pareto_set, axis = 0)
        for non_dominated_solution in range(len_pareto_set):
            x_norm = np.divide((pareto_set[non_dominated_solution][0]-min[0]), (max[0]-min[0]))
            y_norm = np.divide((pareto_set[non_dominated_solution][1]-min[1]), (max[1]-min[1]))
            x = pareto_set[non_dominated_solution][0]
            y = pareto_set[non_dominated_solution][1]
            index.append(np.sqrt((criteria_weight*(1-x_norm))**2 + (((1-criteria_weight)*(1-y_norm))**2)))
    elif criterion == 'area':
        for non_dominated_solution in range(len_pareto_set):
            index.append(abs(pareto_set[non_dominated_solution][0]*pareto_set[non_dominated_solution][1]))
    elif criterion == 'consensus':
        for non_dominated_solution in pareto_front_positions:

            index.append(lambda_frequencies[non_dominated_solution] + qloss_frequencies[non_dominated_solution])

    values_order = np.argsort(index,axis=0)
    values_order = np.flip(values_order,axis=0)
    sorted_positions = [x for _, x in sorted(zip(values_order,pareto_front_positions))]

    return sorted_positions, index

def npv_decision(pareto_set,loading_margin,delta_qloss_comp,demand_increment,interest_rate,annual_business_value,annual_qloss_cost,investment):
    len_pareto_front = len(pareto_set)
    no_facts_cash_flow = npv_preparer(loading_margin[0], delta_qloss_comp[0], demand_increment,
                          annual_business_value, annual_qloss_cost, 0)
    npv_values = []
    for solution in range(len_pareto_front):
        facts = pareto_set[solution]
        facts_cash_flow = npv_preparer(loading_margin[facts], delta_qloss_comp[facts], demand_increment,
                            annual_business_value, annual_qloss_cost, investment)
        cash_flow = [a - b for a, b in zip(facts_cash_flow,no_facts_cash_flow)]
        npv_values.append(np.npv(interest_rate, cash_flow))

    npv_order = np.argsort(npv_values, axis=0)
    npv_order = np.flip(npv_order, axis=0)
    sorted_positions = [x for _, x in sorted(zip(npv_order, pareto_set))]
    return sorted_positions,npv_values

def npv_preparer(loading_margin,delta_qloss_comp,demand_increment,annual_business_value,annual_qloss_cost,investment):
    time_interval = float(loading_margin/demand_increment)
    time_decimal = time_interval-int(time_interval)
    time_interval = int(time_interval)+2

    investment_year = time_interval - 10
    # if time_interval >= 10:
    #     time_interval = 11
    #     time_decimal = 1

    maintenance_cost = investment*0.05

    result = []
    income = []
    costs = []
    for year in range(time_interval):
        if year == 0:
            income.append(0)
            costs.append(investment)
        elif year == 1:
            income.append(annual_business_value)
            costs.append(annual_qloss_cost*(1-delta_qloss_comp)+maintenance_cost)
        elif year >= 2 and year < time_interval:
            income.append(income[year-1]*(1+demand_increment))
            costs.append(costs[year-1]*(1+demand_increment)+maintenance_cost)
        if year == investment_year:
            income.append(income[year - 1] * (1 + demand_increment))
            costs.append(costs[year - 1] * (1 + demand_increment) + maintenance_cost + investment)
        elif year == time_interval:
            income.append(income[year - 1] * demand_increment*time_decimal)
            costs.append((costs[year - 1] * (1+demand_increment)+maintenance_cost)*time_decimal)

        result.append(income[year]-costs[year])

    #plt.plot(income,color='blue')
    #plt.plot(costs,color='red')
    #plt.plot(result,color='black')
    #plt.show()
    return result

def mode(a, axis):
    scores = np.unique(np.ravel(a))       # get ALL unique values
    testshape = list(a.shape)
    testshape[axis] = 1
    oldmostfreq = np.zeros(testshape)
    oldcounts = np.zeros(testshape)

    for score in scores:
        template = (a == score)
        counts = np.expand_dims(np.sum(template, axis),axis)
        mostfrequent = np.where(counts > oldcounts, score, oldmostfreq)
        oldcounts = np.maximum(counts, oldcounts)
        oldmostfreq = mostfrequent

    return mostfrequent, oldcounts

def bins_calculator(sample,sensitivity,min_bins):
    max_sample = np.max(sample)
    min_sample = np.min(sample)
    bins = int(np.round((max_sample - min_sample) / sensitivity, decimals=0))
    if bins < min_bins:
        bins = min_bins
    return bins

def joint_entropy(sample, sensitivity,min_bins):

    len_sample = len(sample)
    bins_x = bins_calculator(sample[:,0],sensitivity=sensitivity,min_bins=min_bins)
    bins_y = bins_calculator(sample[:,1], sensitivity=sensitivity, min_bins=min_bins)

    hist_2d = np.histogramdd(sample, bins=(bins_x,bins_y))
    freq_2d = np.divide(hist_2d[0], len_sample)
    freq_x = np.sum(freq_2d, axis=0)
    freq_y = np.sum(freq_2d, axis=1)
    H = -np.sum(freq_2d * np.log2(np.where(freq_2d>0,freq_2d,1)) ) # Use masking to replace log(0) with 0
    Hx = -np.sum(freq_x * np.ma.log2(freq_x).filled(0))
    Hy = -np.sum(freq_y * np.ma.log2(freq_y).filled(0))
    MI = 0
    # count = 0
    # for x in range(bins_x):
    #     for y in range(bins_y):
    #         pxpy = freq_x[x]*freq_y[y]
    #         freq = freq_2d[x][y]
    #         if pxpy > 0 and freq > 0:
    #             pdiv = np.log2(freq / pxpy)
    #             MI += freq*pdiv
    #             count += 1
    # count_2d = np.count_nonzero(freq_2d)
    MIh = Hx + Hy - H
    return H, MIh


def MI_calculator(data,sensitivity,min_bins):
    number_of_features = len(data)
    number_of_experiments = len(data[0][0])
    result_I = np.zeros(shape=(number_of_features, number_of_features,number_of_experiments))
    result_Ip = np.zeros(shape=(number_of_features, number_of_features, number_of_experiments))
    result_H = np.zeros(shape=(number_of_features, number_of_features, number_of_experiments))
    for experiment in range(number_of_experiments):
        for i in range(0,number_of_features):
            for j in range(1+i,number_of_features):
                X = data[i,:,experiment]
                Y = data[j,:,experiment]
                sample = np.transpose([X, Y])
                H, MI_propia = joint_entropy(sample, sensitivity,min_bins)
                result_H[i, j, experiment] = H
                MI = feature_selection.mutual_info_regression(X.reshape(-1, 1),Y,discrete_features= False, n_neighbors=3, copy=True, random_state=None)
                #MI_score = metrics.mutual_info_score(X,Y, contingency = None)
                result_I[i,j,experiment] = MI
                result_Ip[i, j, experiment] = MI_propia

    D = np.subtract(np.ones((number_of_features,number_of_features,number_of_experiments)),np.ma.divide(result_Ip,result_H).filled(0))
    max_MI_values = np.max(result_I,axis=2)
    max_MI_locations = np.argmax(result_I, axis=2)
    mean_MI_values= np.mean(result_I,axis=2)
    gmean_MI_values = scipy.stats.gmean(result_I,axis=2)

    max_MIp_values = np.max(result_Ip, axis=2)
    max_MIp_locations = np.argmax(result_Ip, axis=2)

    min_D_values = np.min(D,axis=2)
    min_D_locations = np.argmin(result_I, axis=2)

    max_H_values = np.zeros((number_of_features,number_of_features))
    for x in range(len(max_MI_values)):
        for y in range(len(max_MI_values[x])):
            pos = max_MI_locations[x][y]
            max_H_values[x][y] = result_H[x][y][pos]

    #max_H_values = np.max(result_H, axis=2)
    mean_H_values = np.mean(result_H, axis=2)
    gmean_H_values = scipy.stats.gmean(result_H, axis=2)

    D_matrix = np.subtract(np.ones((number_of_features,number_of_features)),np.ma.divide(max_MI_values,max_H_values).filled(0))
    return max_MI_values, max_H_values, min_D_values, result_I, result_Ip

def sensitivity_tester(sensitivity_values, min_bins,data):
    sensitivity_errors=[]
    MI_sensitivity = []
    iter=0
    for value in sensitivity_values:
        sensitivity_errors.append([])
        MI_sensitivity.append([])
        for num in min_bins:
            MI_matrix, H_matrix, D_matrix, MI, MIh = MI_calculator(data, sensitivity=value,min_bins=num)
            n = (len(MI) * len(MI[0]))
            SME = np.sum((MI - MIh) ** 2, axis=None) / n
            sensitivity_errors[iter].append(SME)
            MI_sensitivity[iter].append(MIh)
        iter+=1
    return sensitivity_errors


def MI_decision(MI_matrix, desired_set_size, max):
    number_of_features = len(MI_matrix)
    for x in range(number_of_features):
        for y in range(x+1,number_of_features):
            key = str(x)+ str(y)
            if x==0 and y==1:
                MI_dict = {key: MI_matrix[x][y]}
            else:
                MI_dict.update({key: MI_matrix[x][y]})

    result_I_sum = np.zeros((1, number_of_features))
    for i in range(number_of_features):
        for j in range(number_of_features):
            if MI_matrix[i][j] >= 0:
                result_I_sum[0][i] += MI_matrix[i][j]
                result_I_sum[0][j] += MI_matrix[i][j]

    number_of_discards = number_of_features-desired_set_size
    discarded_indices = []
    for nominated in range(number_of_discards):

        if max==True:
            max_MI = max(MI_dict.iteritems(), key=operator.itemgetter(1))[0]
            keys = MI_dict.keys()
            nominated_a = max_MI[0]
            nominated_b = max_MI[1]

            a_score = result_I_sum[0][int(nominated_a)]
            b_score = result_I_sum[0][int(nominated_b)]

            if a_score > b_score:
                discarded_indices.append(nominated_a)
                for key in range(len(keys)):
                    if nominated_a in keys[key]:
                        del(MI_dict[keys[key]])
            if a_score < b_score:
                discarded_indices.append(nominated_b)
                for key in range(len(keys)):
                    if nominated_b in keys[key]:
                        del(MI_dict[keys[key]])
        else:
            min_MI = min(MI_dict.iteritems(), key=operator.itemgetter(1))[0]
            keys = MI_dict.keys()
            nominated_a = min_MI[0]
            nominated_b = min_MI[1]

            a_score = result_I_sum[0][int(nominated_a)]
            b_score = result_I_sum[0][int(nominated_b)]

            if a_score < b_score:
                discarded_indices.append(nominated_a)
                for key in range(len(keys)):
                    if nominated_a in keys[key]:
                        del (MI_dict[keys[key]])
            if a_score > b_score:
                discarded_indices.append(nominated_b)
                for key in range(len(keys)):
                    if nominated_b in keys[key]:
                        del (MI_dict[keys[key]])

    discarded_indices=np.flip(np.sort(discarded_indices,axis=0),axis=0)

    feature_set = [range(number_of_features)][0]
    for index in range(len(discarded_indices)):
        del(feature_set[int(discarded_indices[index])])

    max_results_val = np.zeros((2,number_of_features))
    max_results_pos = np.zeros((2, number_of_features))
    max_results_val[0,:] = np.max(MI_matrix, axis=0)
    max_results_val[1,:] = np.max(MI_matrix, axis=1)
    max_results_pos[0,:] = np.argmax(MI_matrix, axis=0)
    max_results_pos[1,:] = np.argmax(MI_matrix, axis=1)

    final_result_val = np.max(max_results_val,axis=0)
    final_result_order = np.argmax(max_results_val,axis=0)
    final_result_pos =[]
    for i in range(len(final_result_order)):
        final_result_pos.append(int(max_results_pos[final_result_order[i]][i]))

    final_result_aggregated = np.zeros((1,len(final_result_pos)))
    for j in range(len(final_result_pos)):
        index=final_result_pos[j]
        final_result_aggregated[0][index]+= final_result_val[j]

    indices_order = np.argsort(final_result_aggregated[0])

    return final_result_aggregated, indices_order, MI_matrix, final_result_val, final_result_pos, result_I_sum, feature_set