import matplotlib.pyplot as plt
def plot_demand_map(X_data,Ys,len_loads,labels):
    color=['silver','blue','green','red','pink','violet','dimgrey','brown','orange','darkred','lawngreen','gold','purple','aqua']
    plt.rc('font', size=11)
    plt.rc('axes', labelsize=15)
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.rc('legend', fontsize=13)
    fig_1 = plt.figure()
    figure_1 = fig_1.add_subplot(111)
    for bus in range(len_loads):
        Xs = X_data[:, bus]
        figure_1.scatter(Xs, Ys,color = color[bus],edgecolors='none',alpha=.4,label=str(labels[0][bus]))

    #plt.title('Demand Scenarios map')
    plt.xlabel("Load Share (%)")
    plt.ylabel("Total Demand (MW)")
    plt.legend(title='Load buses',bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=len_loads, mode="expand", borderaxespad=0.,handletextpad=0.01)
    plt.show()


def plot_demand_histogram(Ys):
    fig_2 = plt.figure()
    figure_2 = fig_2.add_subplot(111)
    figure_2.hist(Ys,100,color='crimson')
    #plt.title('Total demand histogarm')
    plt.xlabel("Total demand (MW)")
    plt.ylabel("Number of occurrences")
    plt.show()


def nodes_load_histogram(data, bins, ylim):
    fig_1, axs = plt.subplots(2)
    labels = [[], []]
    for bus in range(len(data)):
        if bus == 0 or bus == 1 or bus == 2 or bus == 5 or bus == 6 or bus == 10:
            axs[0].hist(data[bus], bins=bins, histtype='step')
            labels[0].append(str(bus))

        else:
            axs[1].hist(data[bus], bins=bins, histtype='step')
            labels[1].append(str(bus))
    axs[0].set_ylim(0, ylim)
    axs[1].set_ylim(0, ylim)
    axs[0].legend(['Load bus ' + label for label in labels[0]])
    axs[1].legend(['Load bus ' + label for label in labels[1]])
    axs[0].set_xlabel('Power (MW)')
    axs[0].set_ylabel('# of occurences')
    axs[1].set_xlabel('Power (MW)')
    axs[1].set_ylabel('# of occurences')
    plt.show()