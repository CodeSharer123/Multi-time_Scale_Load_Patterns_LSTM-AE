from data_drawer import draw_choose_K, draw_choose_K_alone
import sys
import numpy as np
import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn import metrics


SIs = []  # silhouette scores
CHs = []  # calinski harabasz scores
DBs = []  # davies bouldin scores

for tt in [0.5, 0.48, 0.45, 0.4, 0.38, 0.35, 0.32, 0.3, 0.28]:
    res = hierarchy.fcluster(Z, t=Y.max()*tt, criterion='distance')
    SIs.append(metrics.silhouette_score(
        X, res, metric='euclidean'))
    CHs.append(metrics.calinski_harabasz_score(X, res))
    DBs.append(metrics.davies_bouldin_score(X, res))
    print(res.max(), end=" ")


sys.path.append('..')

choice = 1
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
    fig, ax = plt.subplots(figsize=(7, 4), subplot_kw=dict(aspect="equal"))
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
              loc=1,
              frameon=False,
              ncol=4,
              handletextpad=0.1,
              columnspacing=0.7,
              bbox_to_anchor=(0.6, 0, 0.5, 1),
              prop={'size': 10, 'weight': 'bold'})

    plt.setp(autotexts, size=15, weight="bold")

    if title != None:
        ax.set_title(title)

    if filepath != None:
        for format in formats:
            plt.savefig(filepath+'.'+format, format=format, dpi=300)
    plt.show()


def draw_center_users_barv(datas, label1='Cluster', label2='Cluster',
                           title=None, filepath=None, formats=None, label_font=None):
    plt.figure(figsize=(8, 6))
    if label_font is None:
        label_font = {
            # 'weight': 'bold',
            'size': 18,
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
        plt.legend(ncol=4, bbox_to_anchor=(0.5, 1.26), frameon=False,
                   columnspacing=0.7,
                   loc='upper center', prop={'size': 22})
    else:
        plt.legend(ncol=len(datas[0]), bbox_to_anchor=(0.5, 1.22), frameon=False,
                   columnspacing=0.7,
                   loc='upper center', prop={'size': 22})
    plt.xticks(x, labels, fontsize=25)
    plt.yticks(fontsize=25)
    plt.ylabel('Proportion', fontsize=25)
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
            'size': 18,
            'family': 'Times New Roman'
        }
    assert len(datas) > 0
    plt.rcParams['font.family'] = ['Times New Roman']
    labels = [label1+' '+str(i+1) for i in range(len(datas))]
    category_names = [label2 + ' '+str(i+1) for i in range(len(datas[0]))]
    datas = datas / datas.mean(axis=1, keepdims=True)
    datas_cum = datas.cumsum(axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.invert_yaxis()
    ax.set_xlim(0, np.sum(datas, axis=1).max())

    for i, colname in enumerate(category_names):
        widths = datas[:, i]
        starts = datas_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname)  # , color=color

    if len(datas[0]) > 4:
        plt.legend(ncol=4, bbox_to_anchor=(0.5, 1.26), frameon=False,
                   columnspacing=0.7,
                   loc='upper center', prop={'size': 22})
    else:
        plt.legend(ncol=len(datas[0]), bbox_to_anchor=(0.5, 1.2), frameon=False,
                   columnspacing=0.7,
                   loc='upper center', prop={'size': 22})

    plt.xticks(np.linspace(0, datas.sum(axis=1).min(), 6),
               ['0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=25)
    plt.yticks(fontsize=25)
    ax.set_xlabel('Proportion', fontsize=25)

    if title is not None:
        plt.title(title, fontdict=label_font)
    if filepath is not None and formats is not None:
        for format in formats:
            plt.savefig(filepath+'.'+format, format=format)
    plt.show()


labels = hierarchy.fcluster(Z, t=Y.max()*0.48, criterion='distance')
labels -= 1
labels = labels.astype(np.int32)
k = labels.max()


user_clusters = [[] for i in range(k+1)]

for i in range(user_info.shape[0]):
    user_clusters[labels[i]].append(user_info.iloc[i, :])

user_mean = []
nums_cluster = []

for i, user_cluster in enumerate(user_clusters):
    print('cluster' + str(i + 1) + ":")
    nums_cluster.append(len(user_cluster))
    print('num: '+str(len(user_cluster)))
    user_cluster = np.array(user_cluster)
    user_mean.append(np.mean(user_cluster, axis=0))

print(user_mean)

draw_pie(nums_cluster, labels=['MDTP ' + str(i) for i in range(1, k + 2)],
         filepath=SAVE_PATH+'test4_4_2_pie', formats=('svg', 'png', 'tif'))

draw_center_users(k+1, 6, center_users=np.array(user_mean), ymax=None, label1='MDTP', label2='DLTP',
                  filepath=SAVE_PATH+'test4_4_2_user', formats=('svg', 'png', 'tif'))
draw_center_users_barv(np.array(user_mean), label1='MDTP', label2='DLTP',
                       filepath=SAVE_PATH+'test4_4_2_user_cz', formats=('svg', 'png', 'tif'))
draw_center_users_barh(np.array(user_mean), label1='MDTP', label2='DLTP',
                       filepath=SAVE_PATH+'test4_4_2_user_sp', formats=('svg', 'png', 'tif'))

np.array(user_mean).shape[1]

K1 = 6
user_info = pd.read_csv(BASIC_PATH)
user_info.set_index('CONS_NO', inplace=True)
X = user_info.to_numpy()
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
hierarchy.dendrogram(Z, labels=None, color_threshold=Y.max()*0.4,
                     leaf_rotation=0, linewidth=3)

plt.axhline(y=Y.max()*0.4, color='#000000', linestyle='dashed', linewidth=3)

ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
plt.tick_params(which='major', axis='y', width=2, length=8)
plt.xticks([])
plt.yticks(fontsize=fontsize)
plt.xlim([-50, len(Z)*10+50])
plt.show()
