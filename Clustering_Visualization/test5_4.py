from data_drawer import draw_choose_K, draw_choose_K_alone
import sys
from sklearn import metrics
import numpy as np
import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

user_info = pd.read_csv(BASIC_PATH)
user_info.set_index('CONS_NO', inplace=True)
tmp_X = user_info.to_numpy()
X = []
cnttt = 0
for ttt in tmp_X:
    if ttt.sum() == 12:
        X.append(ttt)
    else:
        cnttt += 1
print('cnttt:', cnttt)

Y = distance.pdist(np.array(X), metric='euclidean')

Z = hierarchy.linkage(Y, method='average', metric='euclidean')
plt.figure(figsize=(12, 10))
plt.rcParams['font.family'] = ['Times New Roman']
fontsize = 25
label_font = {
    # 'weight': 'bold',
    'size': fontsize,
    'family': 'Times New Roman'
}
plt.title('Hierarchical Clustering Dendrogram', fontdict=label_font)
plt.xlabel('Users', fontdict=label_font)
plt.ylabel('Distance', fontdict=label_font)
hierarchy.dendrogram(Z, labels=None, color_threshold=Y.max()*0.315,
                     leaf_rotation=0, linewidth=3)

plt.axhline(y=Y.max()*0.315, color='#000000', linestyle='dashed', linewidth=3)
plt.tick_params(which='major', axis='y', width=2, length=8)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
plt.xticks([])
plt.yticks(fontsize=fontsize)
plt.xlim([-50, len(Z)*10+50])
plt.savefig(SAVE_PATH+'test5_4_szt.tif', format='tif', dpi=300)
plt.show()


SIs = []  # silhouette scores
CHs = []  # calinski harabasz scores
DBs = []  # davies bouldin scores

for tt in [0.7, 0.5, 0.315, 0.29, 0.27, 0.235, 0.205, 0.2, 0.16]:
    res = hierarchy.fcluster(Z, t=Y.max()*tt, criterion='distance')
    SIs.append(metrics.silhouette_score(
        X, res, metric='euclidean'))
    CHs.append(metrics.calinski_harabasz_score(X, res))
    DBs.append(metrics.davies_bouldin_score(X, res))
    print(res.max(), end=" ")

sys.path.append('..')

choice = 2
x = np.arange(2, 11)
draw_choose_K(x, SIs, CHs,
              DBs, filepath=SAVE_PATH+'zl',
              formats=('svg', 'png', 'tif'))

draw_choose_K_alone(x, SIs, choice=choice,
                    ylabel='Silhouette scores', filepath=SAVE_PATH+'lkxs', formats=('svg', 'png', 'tif'))

draw_choose_K_alone(x, CHs, choice=choice,
                    ylabel='Calinski harabasz scores', filepath=SAVE_PATH+'CHI', formats=('svg', 'png', 'tif'))

draw_choose_K_alone(x, DBs, choice=choice,
                    ylabel='Davies bouldin scores', filepath=SAVE_PATH+'DBI', formats=('svg', 'png', 'tif'))


def draw_pie(datas, labels, label_title=None, title=None, filepath=None, formats=('svg')):

    # with plt.style.context('seaborn-paper'):
    plt.rcParams['font.family'] = ['Times New Roman']
    # label_font = {
    #     # 'weight': 'bold',
    #     'size': 18,
    #     'family': 'Times New Roman'
    # }
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(aspect="equal"))
    explode = [0.02] * len(labels)
    # maxi = 0
    # for i in range(len(datas)):
    #     if datas[i] > datas[maxi]:
    #         maxi = i
    # explode[maxi] = 0.1
    colors = plt.get_cmap('Set1').colors
    wedges, texts, autotexts = ax.pie(datas, autopct='%1.1f%%',
                                      #   textprops={'color': "w"},
                                      explode=explode,
                                      #   shadow=True,
                                      colors=colors,
                                      startangle=90)

    ax.legend(wedges, labels,
              title=label_title,
              #   loc="lower center",
              #   loc=1,
              frameon=False,
              ncol=3,
              handletextpad=0.1,
              columnspacing=0.7,
              bbox_to_anchor=(0, 0, 0.95, 1.1),
              prop={'size': 25})  # , 'weight': 'bold'

    plt.setp(autotexts, size=20, weight="bold")

    if title != None:
        ax.set_title(title)

    if filepath != None:
        for format in formats:
            plt.savefig(filepath+'.'+format, format=format, dpi=300)
    plt.show()


def draw_center_users_barv(datas, label1='Cluster', label2='Cluster',
                           title=None, filepath=None, formats=None, label_font=None):
    plt.figure(figsize=(12, 10))
    if label_font is None:
        label_font = {
            # 'weight': 'bold',
            'size': 25,
            'family': 'Times New Roman'
        }
    datas = np.array(datas).T
    plt.rcParams['font.family'] = ['Times New Roman']
    labels = [label1 + " " + str(i+1) for i in range(len(datas[0]))]

    x = range(len(labels))
    width = 0.5

    bottom_y = np.zeros(len(labels))

    sums = np.sum(datas, axis=0)
    for i, data in enumerate(datas):
        y = data / sums
        plt.bar(x, y, width, bottom=bottom_y, label=label2+' '+str(i+1))
        bottom_y = y + bottom_y
    if len(datas) > 4:
        plt.legend(ncol=3, bbox_to_anchor=(0.5, 1.16), frameon=False,
                   columnspacing=0.7,
                   loc='upper center', prop={'size': 25})
    else:
        plt.legend(ncol=len(datas[0]), bbox_to_anchor=(0.5, 1.1), frameon=False,
                   columnspacing=0.7,
                   loc='upper center', prop={'size': 25})
    plt.xticks(x, labels, fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylabel('Proportion', fontsize=25)
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    if title is not None:
        plt.title(title, fontdict=label_font)
    if filepath is not None and formats is not None:
        for format in formats:
            plt.savefig(filepath+'.'+format, format=format, dpi=300)
    plt.show()


def draw_center_users_barh(datas, label1='Cluster', label2='Cluster',
                           title=None, filepath=None, formats=None, label_font=None):
    datas = np.array(datas)
    if label_font is None:
        label_font = {
            # 'weight': 'bold',
            'size': 25,
            'family': 'Times New Roman'
        }
    assert len(datas) > 0
    plt.rcParams['font.family'] = ['Times New Roman']
    labels = [label1+' '+str(i+1) for i in range(len(datas))]
    category_names = [label2 + ' '+str(i+1) for i in range(len(datas[0]))]
    datas = datas / datas.mean(axis=1, keepdims=True)
    datas_cum = datas.cumsum(axis=1)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.invert_yaxis()
    # ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(datas, axis=1).max())

    for i, colname in enumerate(category_names):
        widths = datas[:, i]
        starts = datas_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname)  # , color=color

    if len(datas[0]) > 4:
        plt.legend(ncol=3, bbox_to_anchor=(0.5, 1.16), frameon=False,
                   columnspacing=0.7,
                   loc='upper center', prop={'size': 25})
    else:
        plt.legend(ncol=len(datas[0]), bbox_to_anchor=(0.5, 1.1), frameon=False,
                   columnspacing=0.7,
                   loc='upper center', prop={'size': 25})

    plt.xticks(np.linspace(0, datas.sum(axis=1).min(), 6),
               ['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=25)
    plt.yticks(fontsize=25)
    ax.set_xlabel('Proportion', fontsize=25)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)

    if title is not None:
        plt.title(title, fontdict=label_font)
    if filepath is not None and formats is not None:
        for format in formats:
            plt.savefig(filepath+'.'+format, format=format, dpi=300)
    plt.show()


def draw_center_users(K_user, K_load, center_users, ymax=None, label1='Cluster', label2='Cluster',
                      title=None, filepath=None, formats=None, label_font=None):
    if label_font is None:
        label_font = {
            # 'weight': 'bold',
            'size': 25,
            'family': 'Times New Roman'
        }
    assert K_user == len(center_users)
    plt.rcParams['font.family'] = ['Times New Roman']

    if ymax == None:
        ymax = int(center_users.max()) + 1
    plt.figure(figsize=(12, 10))
    center_users = np.array(center_users)
    x = np.arange(1, K_load+1, dtype=np.int)
    assert len(x) == center_users.shape[1]

    for i, user in enumerate(center_users):
        plt.plot(x, user, label=label1+" "+str(i+1),
                 linewidth=4)

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    plt.legend(loc=1, prop={'size': 25})
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.xlim((0.5, len(x)+0.5))
    plt.ylim((0, ymax))
    # plt.xlabel(xlabel_dict[user_type], fontdict=label_font)
    plt.ylabel('Number of typical load patterns', fontdict=label_font)
    plt.tick_params(axis='x')
    plt.tick_params(axis='y')
    plt.xticks(x,
               [label2+" "+str(i) for i in range(1, K_load+1)], fontsize=25)
    ynew_ticks = np.arange(0, ymax, max(1, ymax//10))
    plt.yticks(ynew_ticks, fontsize=25)

    if title is not None:
        plt.title(title, fontdict=label_font)
    if filepath is not None and formats is not None:
        for format in formats:
            plt.savefig(filepath+'.'+format, format=format, dpi=300)
    plt.show()


labels = hierarchy.fcluster(Z, t=Y.max()*0.315, criterion='distance')
labels -= 1
labels = labels.astype(np.int32)
k = labels.max()
print(k+1)

user_clusters = [[] for i in range(k+1)]

for i in range(len(labels)):
    user_clusters[labels[i]].append(X[i])

user_mean = []
nums_cluster = []

for i, user_cluster in enumerate(user_clusters):
    print('cluster' + str(i + 1) + ":")
    nums_cluster.append(len(user_cluster))
    print('num: '+str(len(user_cluster)))
    user_cluster = np.array(user_cluster)
    user_mean.append(np.mean(user_cluster, axis=0))

print(user_mean)

draw_pie(nums_cluster, labels=['MMLP ' + str(i) for i in range(1, k + 2)],
         filepath=SAVE_PATH+'test5_4_2_pie', formats=('svg', 'png', 'tif'))

draw_center_users(k+1, 4, center_users=np.array(user_mean), ymax=None, label1='MMLP', label2='TMLP',
                  filepath=SAVE_PATH+'test5_4_2_user', formats=('svg', 'png', 'tif'))
draw_center_users_barv(np.array(user_mean), label1='MMLP', label2='TMLP',
                       filepath=SAVE_PATH+'test5_4_2_user_cz', formats=('svg', 'png', 'tif'))
draw_center_users_barh(np.array(user_mean), label1='MMLP', label2='TMLP',
                       filepath=SAVE_PATH+'test5_4_2_user_sp', formats=('svg', 'png', 'tif'))
