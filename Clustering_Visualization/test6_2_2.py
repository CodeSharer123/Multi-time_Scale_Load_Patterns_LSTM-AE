# -*- coding: utf-8 -*-
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn import metrics
from data_drawer import draw_LPs_new
import sys
sys.path.append("..")


dim = 74
YLPs = None
for choice in ['LOW', 'BASE', 'HIGH']:
    Loads = pd.read_csv(BASIC_PATH+f'resident_YLPs_{choice}.csv')
    Loads.set_index('Number', inplace=True)
    if YLPs is None:
        YLPs = copy.deepcopy(Loads.to_numpy())
    else:
        YLPs = np.concatenate(
            [YLPs, copy.deepcopy(Loads.to_numpy())], axis=0)

np.random.seed(233)
np.random.shuffle(YLPs)
np.random.seed(int(time.time()))

pipe_lr = Pipeline([('standard_scaler1', StandardScaler()),
                    ('pca', PCA(n_components=dim))
                    ])
new_datas = pipe_lr.fit_transform(YLPs)

k = 3
print(f'K={k}')
km = KMeans(n_clusters=k)
labels = km.fit_predict(new_datas)
labels = np.array(labels)
km_cluster_centers = pipe_lr.inverse_transform(km.cluster_centers_)
np.savetxt(SAVE_PATH+'test6_2_1_cluster_centers_' +
           str(k) + '.txt', km_cluster_centers)
np.savetxt(SAVE_PATH+'test6_2_1_labels_' + str(k) + '.txt', labels)


K = 3


color_list = [('c', 'r'), ('tan', 'r'), ('palegreen', 'r'), ('royalblue', 'r'),
              ('gold', 'r'), ('plum', 'r'), ('darkblue', 'r'), ('silver', 'r')]

labels = np.loadtxt(LABEL_PATH+'test6_2_1_labels_' + str(K) + '.txt')
labels = labels.astype(np.int32)

cluster_centers = np.loadtxt(
    LABEL_PATH+'test6_2_1_cluster_centers_' + str(K) + '.txt')


k = int(labels.max() + 1)
assert k == K, "K is not equal to k!!!"
YLP_clusters = [[] for i in range(k)]

for i in range(YLPs.shape[0]):
    YLP_clusters[labels[i]].append(YLPs[i, :])

YLP_mean = []
nums_cluster = []
LEN = 24 * 365
x = np.arange(0, LEN)
ymax = 1
for YLP_cluster in YLP_clusters:
    ymax = max(ymax, int(np.array(YLP_cluster).max()+1))
for i, YLP_cluster in enumerate(YLP_clusters):
    print('cluster' + str(i + 1) + ":")
    nums_cluster.append(len(YLP_cluster))
    print('num: '+str(len(YLP_cluster)))
    YLP_cluster = np.array(YLP_cluster)
    draw_LPs(x, YLP_cluster, LP_type='YLP', pure=True, color=color_list[i][0], c_color=color_list[i][1],
             ymax=ymax, lt_title="YTLP "+str(i+1),
             filepath=SAVE_PATH+'test6_2_2_cluster'+str(i+1)+'_' + str(K), formats=('svg', 'png', 'tif'))
    YLP_mean.append(np.mean(YLP_cluster, axis=0))


draw_pie(nums_cluster, labels=['Cluster ' + str(i) for i in range(1, k + 1)],
         filepath=SAVE_PATH+'test6_2_2_pie'+'_' + str(K), formats=('svg', 'png', 'tif'))
draw_center_LPs(x, YLP_mean, LP_type='YLP', ymax=ymax, label="YTLP", filepath=SAVE_PATH +
                'test6_2_2_cluster_center'+'_' + str(K), formats=('svg', 'png', 'tif'))

K = 3
color_list = [('c', 'r'), ('tan', 'r'), ('palegreen', 'r'), ('royalblue', 'r'),
              ('gold', 'r'), ('plum', 'r'), ('darkblue', 'r'), ('silver', 'r')]

labels = np.loadtxt(LABEL_PATH+'test6_2_1_labels_' + str(K) + '.txt')
labels = labels.astype(np.int32)

cluster_centers = np.loadtxt(
    LABEL_PATH+'test6_2_1_cluster_centers_' + str(K) + '.txt')


k = int(labels.max() + 1)
assert k == K, "K is not equal to k!!!"
YLP_clusters = [[] for i in range(k)]

for i in range(YLPs.shape[0]):
    YLP_clusters[labels[i]].append(YLPs[i, :])

YLP_mean = []
nums_cluster = []
LEN = 24 * 365
x = np.arange(0, LEN)
ymax = 1
for YLP_cluster in YLP_clusters:
    ymax = max(ymax, int(np.array(YLP_cluster).max()+1))
for i, YLP_cluster in enumerate(YLP_clusters):
    print('cluster' + str(i + 1) + ":")
    nums_cluster.append(len(YLP_cluster))
    print('num: '+str(len(YLP_cluster)))

nums_cluster = np.array(nums_cluster)
nums_cluster = nums_cluster / nums_cluster.sum()

for i, YLP_cluster in enumerate(YLP_clusters):
    YLP_cluster = np.array(YLP_cluster)
    draw_LPs_new(x, YLP_cluster, LP_type='YLP', pure=True, color=color_list[i][0], c_color=color_list[i][1],
                 ymax=ymax, title=f'TYLP{i+1}: {100*nums_cluster[i]:.2f}%',
                 filepath=SAVE_PATH+'test6_2_2_cluster'+str(i+1)+'_' + str(K), formats=('svg', 'png', 'tif'))
    YLP_mean.append(np.mean(YLP_cluster, axis=0))
