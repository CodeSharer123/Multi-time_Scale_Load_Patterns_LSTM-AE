# -*- coding: utf-8 -*-
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
# from data_drawer import draw_LPs, draw_center_LPs, draw_pie


dim = 22
MLPs = None
for choice in ['LOW', 'BASE', 'HIGH']:
    Loads = pd.read_csv(BASIC_PATH+f'resident_MLPs_{choice}.csv')
    Loads.set_index('Month', inplace=True)
    if MLPs is None:
        MLPs = copy.deepcopy(Loads.to_numpy())
    else:
        MLPs = np.concatenate(
            [MLPs, copy.deepcopy(Loads.to_numpy())], axis=0)

np.random.seed(233)
np.random.shuffle(MLPs)
np.random.seed(int(time.time()))


pipe_lr = Pipeline([('standard_scaler1', StandardScaler()),
                    ('pca', PCA(n_components=dim))
                    ])
new_datas = pipe_lr.fit_transform(MLPs)
k = 4
print(f'K={k}')
km = KMeans(n_clusters=k)
labels = km.fit_predict(new_datas)
labels = np.array(labels)
km_cluster_centers = pipe_lr.inverse_transform(km.cluster_centers_)
np.savetxt(SAVE_PATH+'test5_2_1_cluster_centers_' +
           str(k) + '.txt', km_cluster_centers)
np.savetxt(SAVE_PATH+'test5_2_1_labels_' + str(k) + '.txt', labels)

K = 4
color_list = [('c', 'r'), ('tan', 'r'), ('palegreen', 'r'), ('royalblue', 'r'),
              ('gold', 'r'), ('plum', 'r'), ('darkblue', 'r'), ('silver', 'r')]
dim = 22
MLPs = None
for choice in ['LOW', 'BASE', 'HIGH']:
    Loads = pd.read_csv(BASIC_PATH+f'resident_MLPs_{choice}.csv')
    Loads.set_index('Month', inplace=True)
    if MLPs is None:
        MLPs = copy.deepcopy(Loads.to_numpy())
    else:
        MLPs = np.concatenate(
            [MLPs, copy.deepcopy(Loads.to_numpy())], axis=0)

np.random.seed(233)
np.random.shuffle(MLPs)
np.random.seed(int(time.time()))

labels = np.loadtxt(LABEL_PATH+'test5_2_1_labels_' + str(K) + '.txt')
labels = labels.astype(np.int32)


cluster_centers = np.loadtxt(
    LABEL_PATH+'test5_2_1_cluster_centers_' + str(K) + '.txt')


k = int(labels.max() + 1)
assert k == K, "K is not equal to k!!!"
MLP_clusters = [[] for i in range(k)]

for i in range(MLPs.shape[0]):
    MLP_clusters[labels[i]].append(MLPs[i, :])

MLP_mean = []
nums_cluster = []
LEN = 24 * 31
x = np.arange(0, LEN)
ymax = 1
for MLP_cluster in MLP_clusters:
    ymax = max(ymax, int(np.array(MLP_cluster).max()+1))
for i, MLP_cluster in enumerate(MLP_clusters):
    print('cluster' + str(i + 1) + ":")
    nums_cluster.append(len(MLP_cluster))
    print('num: '+str(len(MLP_cluster)))
    MLP_cluster = np.array(MLP_cluster)
    draw_LPs(x, MLP_cluster, LP_type='MLP', pure=True, color=color_list[i][0], c_color=color_list[i][1],
             ymax=ymax, lt_title="MTLP "+str(i+1),
             filepath=SAVE_PATH+'test5_2_2_cluster'+str(i+1)+'_' + str(K), formats=('svg', 'png', 'tif'))
    MLP_mean.append(np.mean(MLP_cluster, axis=0))

draw_pie(nums_cluster, labels=['Cluster ' + str(i) for i in range(1, k + 1)],
         filepath=SAVE_PATH+'test5_2_2_pie'+'_' + str(K), formats=('svg', 'png', 'tif'))
draw_center_LPs(x, MLP_mean, LP_type='MLP', ymax=ymax, label="MTLP", filepath=SAVE_PATH +
                'test5_2_2_cluster_center'+'_' + str(K), formats=('svg', 'png', 'tif'))
