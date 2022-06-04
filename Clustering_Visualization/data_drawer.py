import numpy as np
import matplotlib.pyplot as plt


def draw_LPs_new(x, LPs, LP_type, ymax=None, pure=False, cluster_center=None, lt_title=None,
                 title=None, filepath=None, formats=None, color='#C8C8FF', c_color='#8B0000', dpi=300, fontsize=25):
    assert LP_type == 'DLP' or LP_type == 'WLP' or LP_type == 'MLP' or LP_type == 'YLP'
    xlabel_dict = {'DLP': 'Hour of the day', 'WLP': 'Day of the week',
                   'MLP': 'Day of the month', 'YLP': 'Month of the year'}
    LP_width_dict = {'DLP': 0.5, 'WLP': 0.5, 'MLP': 0.5, 'YLP': 0.3}
    C_width_dict = {'DLP': 8, 'WLP': 3, 'MLP': 2, 'YLP': 0.4}

    fig = plt.figure(figsize=(8, 7), dpi=dpi)

    plt.rcParams['font.family'] = ['Times New Roman']
    label_font = {
        # 'weight': 'bold',
        'size': fontsize,
        'family': 'Times New Roman'
    }

    LPs = np.array(LPs)
    if ymax == None:
        ymax = int(LPs.max()) + 1

    x = np.array(x)
    assert len(x) == LPs.shape[1]

    for LP in LPs:
        if pure:
            plt.plot(x, LP, color=color, linewidth=LP_width_dict[LP_type])
        else:
            plt.plot(x, LP, linewidth=LP_width_dict[LP_type])
    ax = fig.gca()
    if pure:
        if np.any(cluster_center) != None:
            plt.plot(x, cluster_center, color=c_color,
                     linewidth=C_width_dict[LP_type], label='Cluster Center')
        else:
            plt.plot(x, LPs.mean(axis=0), color=c_color,
                     linewidth=C_width_dict[LP_type], label='Cluster Center')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    # plt.minorticks_on()
    plt.xlabel(xlabel_dict[LP_type], fontdict=label_font)
    plt.ylabel('Power consumption (kWh)', fontdict=label_font)
    if LP_type == 'WLP':
        plt.tick_params(which='major', axis='x', width=1,
                        length=5, direction='in', color='#C0C0C0')
        plt.tick_params(which='minor', axis='x', width=1,
                        length=5, direction='in', color='#C0C0C0')
    else:
        plt.tick_params(which='major', axis='x', width=1,
                        length=8, direction='out')
        plt.tick_params(which='minor', axis='x', width=1,
                        length=5, direction='out')

    plt.tick_params(which='major', axis='y', width=1, length=8)
    plt.tick_params(which='minor', axis='y', width=1, length=0)

    def get_step(n):
        if n > 10:
            return (n+5) // 10
        elif n > 5:
            return 1
        else:
            return 0.5

    ymax = int(ymax)
    if ymax % 2 == 0:
        ymax += 2
    else:
        ymax += 1
    yticks_labels = [str(i) for i in np.arange(
        0, ymax+1, get_step(ymax))]  
    plt.yticks(np.arange(0, ymax+1, get_step(ymax)),
               yticks_labels, fontsize=fontsize)
    plt.xlim((0, len(x)-1))
    plt.ylim((-ymax/60, ymax))

    if LP_type == 'DLP':
        plt.xticks(np.arange(0, len(x), len(x)/6),
                   ['0', '4', '8', '12', '16', '20'], fontsize=fontsize)
    elif LP_type == 'WLP':
        xticks_labels = ['Sun', 'Mon', 'Tues',
                         'Wed', 'Thu', 'Fri', 'Sat']
        fg = np.linspace(0, len(x), len(xticks_labels)+1)
        for vx in fg[1:-1]:
            plt.vlines(vx, -ymax/60, ymax, linestyle='dashed', color='#C0C0C0')
        # plt.grid(True, linestyle='dashed', axis='x', color='#C0C0C0')
        plt.minorticks_on()
        delta = (fg[1] - fg[0]) / 2
        plt.xticks(fg[0:-1]+delta, xticks_labels, fontsize=fontsize)
        # plt.xticks(np.linspace(len(x)/(7*2), len(x)-len(x)/(7*2), len(xticks_labels)),
        #            xticks_labels, fontsize=fontsize)
        # xticks_labels = ['Sun', 'Tues', 'Thu', 'Sat']
        # plt.xticks(np.arange(0, len(x), len(x)*2/7),
        #            xticks_labels, fontsize=fontsize)
    elif LP_type == 'MLP':
        xticks_labels = ['1', '5', '9', '13', '17',
                         '21', '25', '29']
        plt.xticks(np.linspace(len(x)/(31*2), len(x)-len(x)/(31*2), len(xticks_labels)),
                   xticks_labels, fontsize=fontsize)
    elif LP_type == 'YLP':
        plt.xticks(np.linspace(len(x)/(12*2), len(x)-len(x)/(12*2), 12),
                   ['Jan',  'Feb', 'Mar',  'Apr',
                    'May', 'Jun', 'Jul', 'Aug',
                    'Sept',  'Oct',  'Nov', 'Dec'], fontsize=fontsize-6)

        # plt.xticks(np.arange(0, len(x), len(x)*2/12),
        #            ['Jan',  'Mar', 'May',  'Jul', 'Sept',  'Nov', ], fontsize=fontsize)

    if title is not None:
        plt.title(title, fontdict=label_font)
    if lt_title is not None:
        plt.text(0.11, 0.93, lt_title, fontdict=dict(fontsize=fontsize,
                                                     color='r', family='DejaVu Sans',
                                                     weight='bold'),
                 ha='center', va='center', transform=ax.transAxes)
    if filepath is not None and formats is not None:
        for format in formats:
            plt.savefig(filepath+'.'+format, format=format, dpi=dpi)
    # plt.show()
    
def draw_center_LPs_new(x, center_LPs, LP_type, ymax=None, label='Cluster',
                        title=None, filepath=None, formats=None, dpi=300, fontsize=25):
    assert LP_type == 'DLP' or LP_type == 'WLP' or LP_type == 'MLP' or LP_type == 'YLP'
    xlabel_dict = {'DLP': 'Time', 'WLP': 'Day', 'MLP': 'Day', 'YLP': 'Month'}
    LP_width_dict = {'DLP': 3, 'WLP': 2.5, 'MLP': 2, 'YLP': 1.5}

    plt.rcParams['font.family'] = ['Times New Roman']
    label_font = {
        # 'weight': 'bold',
        'size': fontsize,
        'family': 'Times New Roman'
    }
    center_LPs = np.array(center_LPs)
    x = np.array(x)
    if ymax == None:
        ymax = int(center_LPs.max()) + 1

    fig = plt.figure(figsize=(8, 7), dpi=dpi)

    plt.rcParams['font.family'] = ['Times New Roman']
    label_font = {
        # 'weight': 'bold',
        'size': fontsize,
        'family': 'Times New Roman'
    }


    assert len(x) == center_LPs.shape[1]

    for i, LP in enumerate(center_LPs):
        plt.plot(x, LP, label=label+" "+str(i+1),
                 linewidth=LP_width_dict[LP_type])

    ax = plt.gca()

    plt.legend(prop={'size': fontsize-5},  bbox_to_anchor=(0.5, 1.2), frameon=False,
               loc='upper center', ncol=3, labelspacing=0.4, columnspacing=0.4)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)  
    ax.spines['right'].set_linewidth(2) 
    ax.spines['top'].set_linewidth(2) 

    # plt.minorticks_on()
    plt.xlabel(xlabel_dict[LP_type], fontdict=label_font)
    plt.ylabel('Power consumption (kWh)', fontdict=label_font)
    if LP_type == 'WLP':
        plt.tick_params(which='major', axis='x', width=1,
                        length=5, direction='in', color='#C0C0C0')
        plt.tick_params(which='minor', axis='x', width=1,
                        length=5, direction='in', color='#C0C0C0')
    else:
        plt.tick_params(which='major', axis='x', width=1,
                        length=8, direction='out')
        plt.tick_params(which='minor', axis='x', width=1,
                        length=5, direction='out')

    plt.tick_params(which='major', axis='y', width=1, length=8)
    plt.tick_params(which='minor', axis='y', width=1, length=0)

    ymax = int(ymax)
    if ymax % 2 == 0:
        ymax += 2
    else:
        ymax += 1
    yticks_labels = [str(i) for i in range(0, ymax+1, 2)]
    plt.yticks(np.arange(0, ymax+1, 2),
               yticks_labels, fontsize=fontsize)
    plt.xlim((0, len(x)-1))
    plt.ylim((-0.5, ymax))
    

    if LP_type == 'DLP':
        plt.xticks(np.arange(0, len(x), len(x)/6),
                   ['0', '4', '8', '12', '16', '20'], fontsize=fontsize)
    elif LP_type == 'WLP':
        xticks_labels = ['Sun', 'Mon', 'Tues',
                         'Wed', 'Thu', 'Fri', 'Sat']
        fg = np.linspace(0, len(x), len(xticks_labels)+1)
        for vx in fg[1:-1]:
            plt.vlines(vx ,-0.5, ymax, linestyle='dashed',color='#C0C0C0')
        # plt.grid(True, linestyle='dashed', axis='x', color='#C0C0C0')
        plt.minorticks_on()
        delta = (fg[1] - fg[0]) / 2
        plt.xticks(fg[0:-1]+delta, xticks_labels, fontsize=fontsize)
        # plt.xticks(np.linspace(len(x)/(7*2), len(x)-len(x)/(7*2), len(xticks_labels)),
        #            xticks_labels, fontsize=fontsize)
        # xticks_labels = ['Sun', 'Tues', 'Thu', 'Sat']
        # plt.xticks(np.arange(0, len(x), len(x)*2/7),
        #            xticks_labels, fontsize=fontsize)
    elif LP_type == 'MLP':
        xticks_labels = ['1', '5', '9', '13', '17',
                         '21', '25', '29']
        plt.xticks(np.linspace(len(x)/(31*2), len(x)-len(x)/(31*2), len(xticks_labels)),
                   xticks_labels, fontsize=fontsize)
    elif LP_type == 'YLP':
        plt.xticks(np.linspace(len(x)/(12*2), len(x)-len(x)/(12*2), 12),
                   ['Jan',  'Feb', 'Mar',  'Apr',
                    'May', 'Jun', 'Jul', 'Aug',
                    'Sept',  'Oct',  'Nov', 'Dec'], fontsize=fontsize-6)

    if title is not None:
        plt.title(title, fontdict=label_font)
    if filepath is not None and formats is not None:
        for format in formats:
            plt.savefig(filepath+'.'+format, format=format, dpi=dpi)
    plt.show()
    


def draw_center_LPs(x, center_LPs, LP_type, ymax=None, label='Cluster',
                    title=None, filepath=None, formats=None):
    assert LP_type == 'DLP' or LP_type == 'WLP' or LP_type == 'MLP' or LP_type == 'YLP'
    xlabel_dict = {'DLP': 'Time', 'WLP': 'Day', 'MLP': 'Day', 'YLP': 'Month'}
    LP_width_dict = {'DLP': 2, 'WLP': 1.5, 'MLP': 1, 'YLP': 0.5}

    plt.rcParams['font.family'] = ['Times New Roman']
    label_font = {
        # 'weight': 'bold',
        'size': 18,
        'family': 'Times New Roman'
    }

    plt.figure(figsize=(8, 6))
    center_LPs = np.array(center_LPs)
    if ymax == None:
        ymax = int(center_LPs.max()) + 1
    x = np.array(x)
    assert len(x) == center_LPs.shape[1]

    for i, LP in enumerate(center_LPs):
        plt.plot(x, LP, label=label+" "+str(i+1),
                 linewidth=LP_width_dict[LP_type])

    ax = plt.gca()

    plt.legend(loc=1, prop={'size': 15})
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.xlim((-1, len(x)))
    plt.ylim((0, ymax))
    plt.xlabel(xlabel_dict[LP_type], fontdict=label_font)
    plt.ylabel('Power consumption (kWh)', fontdict=label_font)
    plt.tick_params(axis='x')
    plt.tick_params(axis='y')
    ynew_ticks = np.linspace(0, ymax, 10)
    plt.yticks(ynew_ticks, fontsize=13)

    if LP_type == 'DLP':
        plt.xticks(np.arange(len(x)/24, len(x) + len(x)/24, len(x)/12),
                   ['01:00',  '03:00', '05:00',  '07:00',
                    '09:00', '11:00', '13:00', '15:00',
                    '17:00',  '19:00',  '21:00', '23:00'], fontsize=13)
    elif LP_type == 'WLP':
        xticks_labels = ['Sun', 'Mon', 'Tues',
                         'Wed', 'Thu', 'Fri', 'Sat']
        plt.xticks(np.linspace(12, len(x)-12, len(xticks_labels)),
                   xticks_labels, fontsize=13)
    elif LP_type == 'MLP':
        xticks_labels = ['1', '3', '5', '7', '9', '11',
                         '13', '15', '17', '19', '21', '23',
                         '25', '27', '29', '31']
        plt.xticks(np.linspace(0, len(x), len(xticks_labels)),
                   xticks_labels, fontsize=13)
    elif LP_type == 'YLP':
        plt.xticks(np.linspace(48, len(x), 12),
                   ['Jan',  'Feb', 'Mar',  'Apr',
                    'May', 'Jun', 'Jul', 'Aug',
                    'Sept',  'Oct',  'Nov', 'Dec'], fontsize=13)

    if title is not None:
        plt.title(title, fontdict=label_font)
    if filepath is not None and formats is not None:
        for format in formats:
            plt.savefig(filepath+'.'+format, format=format)
    plt.show()


def draw_choose_K(x, silhouette_scores, calinski_harabasz_scores,
                  davies_bouldin_scores, inertias=None, title=None, filepath=None, formats=None):
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.figure(figsize=(8, 6))
    if np.any(inertias) != None:
        inertias = np.array(inertias)
    silhouette_scores = np.array(silhouette_scores)
    calinski_harabasz_scores = np.array(calinski_harabasz_scores)
    davies_bouldin_scores = np.array(davies_bouldin_scores)
    if np.any(inertias) != None:
        inertias -= inertias.min()
        inertias /= (inertias.max() - inertias.min())
    silhouette_scores -= silhouette_scores.min()
    silhouette_scores /= (silhouette_scores.max() - silhouette_scores.min())
    calinski_harabasz_scores -= calinski_harabasz_scores.min()
    calinski_harabasz_scores /= (calinski_harabasz_scores.max() -
                                 calinski_harabasz_scores.min())
    davies_bouldin_scores -= davies_bouldin_scores.min()
    davies_bouldin_scores /= (davies_bouldin_scores.max() -
                              davies_bouldin_scores.min())

    plt.grid(linestyle="--")
    ax = plt.gca()

    if np.any(inertias) != None:
        plt.plot(x, inertias, marker='o', color="blue",
                 label="inertia", linewidth=1.5)
    plt.plot(x, silhouette_scores, marker='o', color="green",
             label="SI", linewidth=1.5)
    plt.plot(x, calinski_harabasz_scores, marker='o', color="red",
             label="CH", linewidth=1.5)
    plt.plot(x, davies_bouldin_scores, marker='o', color="yellow",
             label='DB', linewidth=1.5)

    group_labels = x 
    plt.xticks(x, group_labels, fontsize=12, fontweight='bold') 
    plt.yticks(fontsize=12, fontweight='bold')
    if title != None:
        plt.title(title, fontsize=12, fontweight='bold')  
    plt.xlabel("Number of clusters", fontsize=13, fontweight='bold')
    plt.ylabel("CVI scores", fontsize=13, fontweight='bold')
    plt.xlim(1, len(x)+2)  
    plt.ylim(0, 1)
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold') 

    if filepath is not None and formats is not None:
        for format in formats:
            plt.savefig(filepath+'.'+format, format=format)
    plt.show()


def draw_choose_K_alone(x, scores, choice=None, ylabel="CVI scores", title=None, filepath=None, formats=None):
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.figure(figsize=(6, 4))

    plt.grid(linestyle="--")  
    ax = plt.gca()

    if choice is not None:
        plt.plot(x, scores, color="#0462b2", linewidth=1.5)
        for i in range(len(x)):
            if i == choice:
                plt.plot(x[choice], scores[choice], marker='d',
                         color='#ec1f24', markersize=11)
            else:
                plt.plot(x[i], scores[i], marker='s',
                         color="#0462b2", markersize=8)
    else:
        plt.plot(x, scores, marker='s', color="#0462b2",
                 linewidth=1.5, markersize=8)
    group_labels = x 
    plt.xticks(x, group_labels, fontsize=15, fontweight='bold')  
    plt.yticks(fontsize=15, fontweight='bold')
    if title != None:
        plt.title(title, fontsize=15, fontweight='bold') 
    plt.xlabel("Number of clusters", fontsize=15, fontweight='bold')
    plt.ylabel(ylabel, fontsize=15, fontweight='bold')
    plt.xlim(1, len(x)+2) 

    if filepath is not None and formats is not None:
        for format in formats:
            plt.savefig(filepath+'.'+format, format=format)
    plt.show()


def draw_PCA(dim, x2, explained_variance_ratio: list, ch=None, title=None, filepath=None, formats=('svg')):
    plt.rcParams['font.family'] = ['Times New Roman']
    assert dim == len(explained_variance_ratio)
    x = np.arange(0, dim+1)
    explained_variance_ratio = [0] + explained_variance_ratio
    for i in range(1, len(explained_variance_ratio)):
        explained_variance_ratio[i] += explained_variance_ratio[i - 1]

    plt.figure(figsize=(8, 4))
    # plt.style.use(['science'])

    plt.grid(linestyle="--") 
    ax = plt.gca()

    plt.plot(x, explained_variance_ratio, color="red", linewidth=1)

    y2 = []

    for i in x2:
        y2.append(explained_variance_ratio[i])
    if ch != None:
        plt.plot(x2[ch], y2[ch], 'd', color='red', markersize=8)
        x2.remove(x2[ch])
        y2.remove(y2[ch])
    plt.plot(x2, y2, 's', linewidth=2)

    if title != None:
        plt.title(title, fontsize=12)  
    plt.xlabel("Number of principal components",
               fontsize=13)
    plt.ylabel("Explained variance ratio", fontsize=13)
    plt.xlim(0, len(x)-1)  
    plt.ylim(0, 1)

    if filepath != None:
        for format in formats:
            plt.savefig(filepath+'.'+format, format=format)
    plt.show()


def draw_pie(datas, labels, label_title=None, title=None, filepath=None, formats=('svg')):
    # with plt.style.context('seaborn-paper'):

    plt.rcParams['font.family'] = ['Times New Roman']
    label_font = {
        # 'weight': 'bold',
        'size': 18,
        'family': 'Times New Roman'
    }
    fig, ax = plt.subplots(figsize=(7, 4), subplot_kw=dict(aspect="equal"))
    explode = [0] * len(labels)
    maxi = 0
    for i in range(len(datas)):
        if datas[i] > datas[maxi]:
            maxi = i
    explode[maxi] = 0.1
    wedges, texts, autotexts = ax.pie(datas, autopct='%1.1f%%',
                                      textprops={'color': "w"},
                                      explode=explode,
                                      shadow=True, startangle=90)

    ax.legend(wedges, labels,
              title=label_title,
              loc=1,
              bbox_to_anchor=(1, 0, 0.5, 1), prop={'size': 15})

    plt.setp(autotexts, size=16, weight="bold")

    if title != None:
        ax.set_title(title)

    if filepath != None:
        for format in formats:
            plt.savefig(filepath+'.'+format, format=format)
    plt.show()
