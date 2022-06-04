
from data_drawer import draw_center_LPs_new
from data_drawer import draw_LPs_new
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from data_drawer import draw_DLPs, draw_center_DLPs, draw_pie
import time


# %%
K = 6
DLPs = None
for choice in ['LOW', 'BASE', 'HIGH']:
    Loads = pd.read_csv(BASIC_PATH+f'resident_DLPs_{choice}.csv')
    Loads.set_index('Date', inplace=True)
    if DLPs is None:
        DLPs = copy.deepcopy(Loads.to_numpy())
    else:
        DLPs = np.concatenate(
            [DLPs, copy.deepcopy(Loads.to_numpy())], axis=0)

np.random.seed(233)
np.random.shuffle(DLPs) 
np.random.seed(int(time.time()))
labels = np.loadtxt(LABEL_PATH+'test4_2_1_labels_' + str(K) + '.txt')
labels = labels.astype(np.int32)


cluster_centers = np.loadtxt(
    LABEL_PATH+'test4_2_1_cluster_centers_' + str(K) + '.txt')


k = int(labels.max() + 1)
assert k == K, "K is not equal to k!!!"
DLP_clusters = [[] for i in range(k)]

for i in range(DLPs.shape[0]):
    DLP_clusters[labels[i]].append(DLPs[i, :])

DLP_mean = []
nums_cluster = []
LEN = 24
x = np.arange(0, LEN)
ymax = 1
for DLP_cluster in DLP_clusters:
    ymax = max(ymax, int(np.array(DLP_cluster).max()+1))
for i, DLP_cluster in enumerate(DLP_clusters): 
    print('cluster' + str(i + 1) + ":")
    nums_cluster.append(len(DLP_cluster))
    print('num: '+str(len(DLP_cluster)))
    DLP_cluster = np.array(DLP_cluster)
    draw_DLPs(x, DLP_cluster, ymax=ymax, title='Cluster '+str(i+1),
              filepath=SAVE_PATH+'test4_2_2_cluster'+str(i+1)+'_' + str(K), formats=('svg', 'png', 'tif'))
    DLP_mean.append(np.mean(DLP_cluster, axis=0))

draw_pie(nums_cluster, labels=['Cluster ' + str(i) for i in range(1, k + 1)],
         filepath=SAVE_PATH+'test4_2_2_pie'+'_' + str(K), formats=('svg', 'png', 'tif'))
draw_center_DLPs(x, DLP_mean, ymax=ymax, filepath=SAVE_PATH +
                 'test4_2_2_cluster_center'+'_' + str(K), formats=('svg', 'png', 'tif'))


K = 4
color_list = [('c', 'r'), ('tan', 'r'), ('palegreen', 'r'), ('royalblue', 'r'),
              ('gold', 'r'), ('plum', 'r'), ('darkblue', 'r'), ('silver', 'r')]
DLPs = None
for choice in ['LOW', 'BASE', 'HIGH']:
    Loads = pd.read_csv(BASIC_PATH+f'resident_DLPs_{choice}.csv')
    Loads.set_index('Date', inplace=True)
    if DLPs is None:
        DLPs = copy.deepcopy(Loads.to_numpy())
    else:
        DLPs = np.concatenate(
            [DLPs, copy.deepcopy(Loads.to_numpy())], axis=0)

K = 6
color_list = [('c', 'r'), ('tan', 'r'), ('palegreen', 'r'), ('royalblue', 'r'),
              ('gold', 'r'), ('plum', 'r'), ('darkblue', 'r'), ('silver', 'r')]
DLPs = None
for choice in ['LOW', 'BASE', 'HIGH']:
    Loads = pd.read_csv(BASIC_PATH+f'resident_DLPs_{choice}.csv')
    Loads.set_index('Date', inplace=True)
    if DLPs is None:
        DLPs = copy.deepcopy(Loads.to_numpy())
    else:
        DLPs = np.concatenate(
            [DLPs, copy.deepcopy(Loads.to_numpy())], axis=0)

np.random.seed(233)
np.random.shuffle(DLPs) 
np.random.seed(int(time.time()))
labels = np.loadtxt(LABEL_PATH+'test4_2_1_labels_' + str(K) + '.txt')
labels = labels.astype(np.int32)


cluster_centers = np.loadtxt(
    LABEL_PATH+'test4_2_1_cluster_centers_' + str(K) + '.txt')


k = int(labels.max() + 1)
assert k == K, "K is not equal to k!!!"
DLP_clusters = [[] for i in range(k)]

for i in range(DLPs.shape[0]):
    DLP_clusters[labels[i]].append(DLPs[i, :])

nums_cluster = []
LEN = 24
x = np.arange(0, LEN)
ymax = 1
for DLP_cluster in DLP_clusters:
    ymax = max(ymax, int(np.array(DLP_cluster).max()+1))
for i, DLP_cluster in enumerate(DLP_clusters):  
    print('cluster' + str(i + 1) + ":")
    nums_cluster.append(len(DLP_cluster))
    print('num: '+str(len(DLP_cluster)))

nums_cluster = np.array(nums_cluster)
nums_cluster = nums_cluster / nums_cluster.sum()

for i, DLP_cluster in enumerate(DLP_clusters):
    draw_LPs_new(x, DLP_cluster, LP_type='DLP', pure=True, color=color_list[i][0], c_color=color_list[i][1],
                 ymax=ymax, title=f'TDLP{i+1}: {100*nums_cluster[i]:.2f}%',
                 filepath=SAVE_PATH+f'test4_2_2_cluster{i+1}_{K}_new', formats=('svg', 'png', 'tif'))

