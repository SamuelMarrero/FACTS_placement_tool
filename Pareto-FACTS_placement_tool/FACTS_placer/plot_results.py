import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import proj3d,Axes3D
from mpl_toolkits.axisartist.axislines import SubplotZero
from pareto_functions import MI_calculator, MI_decision

def bar_mean_std_plot(mean_data, std_data, labels,index):

    y_pos = np.arange(len(labels))
    plt.rcdefaults()
    plt.rc('font', size=11)
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=11)
    plt.rc('ytick', labelsize=11)
    plt.rc('legend', fontsize=11)
    fig, ax = plt.subplots()
    ax.barh(y_pos, mean_data, xerr=std_data, align='center',
            color='limegreen', ecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel(index + ' (mean and standard deviation (p.u.))')
    #ax.set_title('How fast do you want to go today?')

    plt.show()

def voltage_profile_plot(mean_data, std_data, rows_to_plot,labels):
    x_pos = np.arange(len(mean_data[0]))
    fig_2 = plt.figure()
    plt.rcdefaults()
    len_plots = len(rows_to_plot)
    for j in range(len_plots):
        ax = plt.subplot(len_plots,1,j+1)
        #fig, ax = plt.subplots()
        row = rows_to_plot[j]
        ax.bar(x_pos, mean_data[row], xerr=std_data[row], align='center',
                color='green', ecolor='black')
        ax.set_xticks(x_pos)
        ax.set_ylim([1,1.06])
        ax.set_ylabel(labels[row])
    ax.set_xlabel('Voltage mean and standard deviation')

    plt.show()

def scenario_volt_dev_plot(datos,media,etiquetas):
    len_ubicaciones = len(datos)
    len_escenarios = len(datos[0])
    fig_2 = plt.figure()
    plt.axis()
    for j in range(len_ubicaciones):
        if j ==1:
            plt.title('Desviaciones de la tension para todos los escenarios y valor medio')
        plt.subplot(len_ubicaciones,1,j+1)
        plt.plot(range(len_escenarios),datos[j],color = 'black')
        plt.axhline(media[0][j], color='grey')
        plt.axhline(0,color = 'black')
        plt.ylabel(etiquetas[1][j][0])

    plt.show()

def scenario_PV_plot(raw_data,escenario):
    len_buses = len(raw_data[escenario][0][0][0][0][0])
    len_ubicaciones = len(raw_data[escenario][0])
    fig_2 = plt.figure()
    fig_2.suptitle('Curvas PV, escenario 1810 - 0, sin y con STATCOM')
    plt.axis()
    for j in range(len_ubicaciones):
        plt.subplot(3, 3, j + 1)
        for k in range(len( raw_data[escenario][0][j][1])):  # len( raw_data[escenario][0][j][1])
            for l in range(len_buses):
                try:
                    x = raw_data[escenario][0][j][1][k][0]
                    y = raw_data[escenario][0][j][0][k][0][l]
                    color = ['black', 'blue', 'green', 'red', 'pink', 'violet', 'grey', 'brown', 'orange', 'darkred',
                             'lawngreen', 'gold', 'purple', 'aqua']
                    plt.scatter(x, y, color=color[l])
                except:
                    pass

    plt.show()

def MI_plot_3d(raw_data,labels):
    # setup the figure and axes
    len_data = len(raw_data)
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(111, projection='3d')

    data = np.zeros((len_data,len_data))
    for i in range(len_data):
        for j in range(i+1,len_data):
            data[i,j] = raw_data[i,j]

    data = np.flip(data, axis=0)
    data = np.flip(data, axis=1)

    _x = np.arange(len_data)
    _y = np.arange(len_data)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = data.ravel()
    width = depth = 0.8
    bottom = np.zeros_like(top)
    ax1.bar3d(x, y, bottom, width, depth, top, color='lightgreen')
    ax1.set_zlabel('D-distance')
    plt.show()
    a = 0

def LCOE_digging_plot(LCOE,costs,power,labels):
    width = 0.20
    x = np.arange(len(LCOE))
    fig, ax = plt.subplots()
    LCOE_plot = ax.bar(x, LCOE,width, label ='LCOE')
    costs_plot = ax.bar(x - width, costs, width, label='Costs')
    power_plot = ax.bar(x + width, power, width, label='Power')

    ax.set_ylabel('Improvement (%)')
    ax.set_title('LCOE comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()

def pareto_3d_plot(data,pareto_set,labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], c='b', marker='o')
    ax.plot(pareto_set[:,0], pareto_set[:,1], pareto_set[:,2], c = 'r', marker = 'o')

    iteration = 0
    for j in range(len(labels)):
        label = labels[iteration]
        x = data[:,0]
        y = data[:,1]
        z = data[:,2]

        x2, y2, _ = proj3d.proj_transform(1, 1, 1, ax.get_proj())
        ax.annotate(
            label,
            xy=(x2, y2), xytext=(20, -30),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='lightgrey', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        iteration += 1

    ax.set_xlabel('Lambda')
    ax.set_ylabel('Costs')
    ax.set_zlabel('Emissions')

    plt.show()

    def plot_indices_comparison(data):
        # data = np.random.randn(6, 6)
        y = ["Prod. {}".format(i) for i in range(10, 70, 10)]
        x = ["Cycle {}".format(i) for i in range(1, 7)]

        fig = plt.subplots(1, 1)
        qrates = np.array(list("ABCDEFGH"))
        norm = matplotlib.colors.BoundaryNorm(np.linspace(-3.5, 3.5, 8), 7)
        fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: qrates[::-1][norm(x)])

        im, _ = heatmap(data, y, x,
                        cmap=plt.get_cmap("PiYG", 7), norm=norm,
                        cbar_kw=dict(ticks=np.arange(-3, 4), format=fmt),
                        cbarlabel="Quality Rating")

        annotate_heatmap(im, valfmt=fmt, size=9, fontweight="bold", threshold=-1,
                         textcolors=("red", "black"))

        plt.tight_layout()
        plt.show()

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def plot_indices_comparison(data,x_labels,y_labels):
    data = data + 1
    #original_data = data
    #data = np.flip(data, axis=1)
    y = y_labels
    x = x_labels

    fig = plt.subplots(1,1)
    qrates = np.array(list("12345678"))
    norm = matplotlib.colors.BoundaryNorm(np.linspace(0.5, 8.5, num=9), 8)
    numeros = np.linspace(0.5, 8.5, num=9)
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: qrates[::][norm(x)])

    im, _ = heatmap(data, y, x,
                    cmap=plt.get_cmap("hot", 8), norm=norm,
                    cbar_kw=dict(ticks=np.arange(0.5, 8.5), format=fmt),
                    cbarlabel="Location order according to index performance")

    annotate_heatmap(im, valfmt=fmt, size=8, fontweight="bold", threshold=1,
                     textcolors=("red", "grey"))

    plt.tight_layout()
    plt.show()
    a = 0

def bins_tester_plot(sensitivity_values,min_bins,data, MI_calculation):
    bins_test = []
    len_bins = len(sensitivity_values)
    for test in range(len_bins):
        MI_matrix, H_matrix, D_matrix, MI,MIh =MI_calculator(data,sensitivity=sensitivity_values[test],min_bins=min_bins)
        if MI_calculation == 'MI':
            matrix = MI_matrix
        elif MI_calculation == 'D':
            matrix = D_matrix
        elif MI_calculation == 'H':
            matrix = H_matrix
        MI_val, MI_pos, MI_2d, MI_semi_val, MI_semi_pos, MI_val_sum, MI_set = MI_decision(matrix,3,max=False)
        bins_test.append(MI_2d)
    number_of_features = len(data)
    for i in range(0, number_of_features):
        for j in range(1 + i, number_of_features):
            x= sensitivity_values
            y= []
            for k in range(len_bins):
                y.append(bins_test[k][i][j])
            plt.scatter(x,y)
    #plt.title('D vs. Sensitivity (min_bins='+str(min_bins)+')')
    plt.xlabel('Granularity (p.u.)')
    plt.ylabel(MI_calculation +' value')
    plt.show()

def MI_comparison_locations(number_of_features,facts,MI,MIh):
    figure1 = plt.figure()
    for i in range(0, number_of_features):
        for j in range(1 + i, number_of_features):
            x = range(facts)
            y_MI = []
            y_MIh = []
            for k in range(facts):
                y_MI.append(MI[i][j][k])
                y_MIh.append(MIh[i][j][k])
            plt.scatter(x, y_MI,color='blue')
            plt.scatter(x, y_MIh, color='red')
    plt.xlabel('FACTS location')
    plt.ylabel('MI value')
    plt.title('MI comparison (sensitivity='+ str(sensitivity)+')')
    plt.show()

def MI_comparison_single(number_of_features,facts,MI,MIh):
    figure1 = plt.figure()
    x = []
    y = []
    for i in range(0, number_of_features):
        for j in range(1 + i, number_of_features):
            x.append(MI[i][j][facts])
            y.append(MIh[i][j][facts])

    plt.plot(x)
    plt.plot(y)
    plt.xlabel('Index combination')
    plt.ylabel('MI value')
    plt.title('MI comparison (sensitivity='+ str(sensitivity)+')')
    plt.show()

def comparative_Pareto(x_data,y_data,labels,colors,axes_labels,legend_labels):
    plt.rc('font', size=11)
    plt.rc('axes', labelsize=15)
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.rc('legend', fontsize=13)
    fig_1 = plt.figure()
    figure_1 = SubplotZero(fig_1,111)
    fig_1.add_subplot(figure_1)
    x_neg = False
    y_neg = False

    for iter in range(len(x_data)):
        figure_1.scatter(x_data[iter][0],y_data[iter][0], color=colors[iter], marker = 'o', s=70,label=legend_labels[iter])
        #figure_1.plot(x_data[iter][0],y_data[iter][0],color=colors[iter])
        iteration = 0
        for j in range(len(labels)):
            label = labels[iteration]
            x = x_data[iter][0][iteration]
            y = y_data[iter][0][iteration]
            figure_1.annotate(
                label,
                xy=(x, y), xytext=(20, -30),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgrey', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            if x < 0:
                x_neg = True
            if y < 0:
                y_neg = True
            iteration += 1

    if x_neg == True :
        plt.axvline(x=0, color='k')
    if y_neg == True :
        plt.axhline(y=0, color='k')

    plt.xlabel(axes_labels[0])
    plt.ylabel(axes_labels[1])
    plt.legend(loc=0) #bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5, mode="expand", borderaxespad=0., handletextpad=0.01

    plt.grid()
    plt.show()

