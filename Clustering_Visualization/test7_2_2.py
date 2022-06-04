# -*- coding: utf-8 -*-
from data_drawer import draw_LPs, draw_LPs_new, draw_center_LPs, draw_pie
import time
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import sys
sys.path.append("..")


dim = 8
WLPs = None
for choice in ['LOW', 'BASE', 'HIGH']:
    Loads = pd.read_csv(BASIC_PATH+f'resident_WLPs_{choice}.csv')
    Loads.set_index('Start_Date', inplace=True)
    if WLPs is None:
        WLPs = copy.deepcopy(Loads.to_numpy())
    else:
        WLPs = np.concatenate(
            [WLPs, copy.deepcopy(Loads.to_numpy())], axis=0)


np.random.seed(233)
np.random.shuffle(WLPs)
np.random.seed(int(time.time()))


pipe_lr = Pipeline([('standard_scaler1', StandardScaler()),
                    ('pca', PCA(n_components=dim))
                    ])
new_datas = pipe_lr.fit_transform(WLPs)


k = 4
print(f'K={k}')
km = KMeans(n_clusters=k)
labels = km.fit_predict(new_datas)
labels = np.array(labels)
km_cluster_centers = pipe_lr.inverse_transform(km.cluster_centers_)
np.savetxt(SAVE_PATH+'test7_2_1_cluster_centers_' +
           str(k) + '.txt', km_cluster_centers)
np.savetxt(SAVE_PATH+'test7_2_1_labels_' + str(k) + '.txt', labels)

color_list = [('c', 'r'), ('tan', 'r'), ('palegreen', 'r'), ('royalblue', 'r'),
              ('gold', 'r'), ('plum', 'r'), ('darkblue', 'r'), ('silver', 'r')]

labels = np.loadtxt(LABEL_PATH+'test7_2_1_labels_' + str(K) + '.txt')
labels = labels.astype(np.int32)


cluster_centers = np.loadtxt(
    LABEL_PATH+'test7_2_1_cluster_centers_' + str(K) + '.txt')


k = int(labels.max() + 1)
assert k == K, "K is not equal to k!!!"
WLP_clusters = [[] for i in range(k)]

for i in range(WLPs.shape[0]):
    WLP_clusters[labels[i]].append(WLPs[i, :])

WLP_mean = []
nums_cluster = []
LEN = 24 * 7
x = np.arange(0, LEN)
ymax = 1
for WLP_cluster in WLP_clusters:
    ymax = max(ymax, int(np.array(WLP_cluster).max()+1))
for i, WLP_cluster in enumerate(WLP_clusters):
    print('cluster' + str(i + 1) + ":")
    nums_cluster.append(len(WLP_cluster))
    print('num: '+str(len(WLP_cluster)))
    WLP_cluster = np.array(WLP_cluster)
    draw_LPs(x, WLP_cluster, LP_type='WLP', pure=True, color=color_list[i][0], c_color=color_list[i][1],
             ymax=ymax, lt_title="WTLP "+str(i+1),
             filepath=SAVE_PATH+'test7_2_2_cluster'+str(i+1)+'_' + str(K), formats=('svg', 'png', 'tif'))
    WLP_mean.append(np.mean(WLP_cluster, axis=0))

draw_pie(nums_cluster, labels=['Cluster ' + str(i) for i in range(1, k + 1)],
         filepath=SAVE_PATH+'test7_2_2_pie'+'_' + str(K), formats=('svg', 'png', 'tif'))
draw_center_LPs(x, WLP_mean, LP_type='WLP', ymax=ymax, label="WTLP", filepath=SAVE_PATH +
                'test7_2_2_cluster_center'+'_' + str(K), formats=('svg', 'png', 'tif'))

color_list = [('c', 'r'), ('tan', 'r'), ('palegreen', 'r'), ('royalblue', 'r'),
              ('gold', 'r'), ('plum', 'r'), ('darkblue', 'r'), ('silver', 'r')]

labels = np.loadtxt(LABEL_PATH+'test7_2_1_labels_' + str(K) + '.txt')
labels = labels.astype(np.int32)


cluster_centers = np.loadtxt(
    LABEL_PATH+'test7_2_1_cluster_centers_' + str(K) + '.txt')


k = int(labels.max() + 1)
assert k == K, "K is not equal to k!!!"
WLP_clusters = [[] for i in range(k)]

for i in range(WLPs.shape[0]):
    WLP_clusters[labels[i]].append(WLPs[i, :])

WLP_mean = []
nums_cluster = []
LEN = 24 * 7
x = np.arange(0, LEN)
ymax = 1
for WLP_cluster in WLP_clusters:
    ymax = max(ymax, int(np.array(WLP_cluster).max()+1))
for i, WLP_cluster in enumerate(WLP_clusters):
    print('cluster' + str(i + 1) + ":")
    nums_cluster.append(len(WLP_cluster))
    print('num: '+str(len(WLP_cluster)))

nums_cluster = np.array(nums_cluster)
nums_cluster = nums_cluster / nums_cluster.sum()

for i, WLP_cluster in enumerate(WLP_clusters):
    WLP_cluster = np.array(WLP_cluster)
    draw_LPs_new(x, WLP_cluster, LP_type='WLP', pure=True, color=color_list[i][0], c_color=color_list[i][1],
                 ymax=ymax,
                 filepath=SAVE_PATH+'test7_2_2_cluster'+str(i+1)+'_' + str(K),
                 title=f'TWLP{i+1}: {100*nums_cluster[i]:.2f}%',
                 formats=('svg', 'png', 'tif'))
    WLP_mean.append(np.mean(WLP_cluster, axis=0))
