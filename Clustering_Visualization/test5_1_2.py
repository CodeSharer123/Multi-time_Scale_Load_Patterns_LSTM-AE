# -*- coding: utf-8 -*-
import copy
import time
from data_drawer import draw_choose_K, draw_choose_K_alone
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
sys.path.append("..")

PICTURE_NAME = 'test5_1_2'
test_num = 5

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

SIs = []  # silhouette scores
CHs = []  # calinski harabasz scores
DBs = []  # davies bouldin scores
inertias = []
for i in range(test_num):
    print(i)
    SI = []
    CH = []
    DB = []
    inertia = []
    for k in range(2, 12):
        print(f'K={k}')
        km = KMeans(n_clusters=k)
        labels = km.fit_predict(new_datas)
        # labels = np.array(labels)
        # km_cluster_centers = pipe_lr.inverse_transform(km.cluster_centers_)
        # np.savetxt(SAVE_PATH+'test5_2_1_cluster_centers_' +
        #         str(k) + '.txt', km_cluster_centers)
        # np.savetxt(SAVE_PATH+'test5_2_1_labels_' + str(k) + '.txt', labels)
        inertia.append(km.inertia_)
        SI.append(metrics.silhouette_score(
            new_datas, labels, metric='euclidean'))
        CH.append(metrics.calinski_harabasz_score(new_datas, labels))
        DB.append(metrics.davies_bouldin_score(new_datas, labels))
    inertias.append(inertia)
    SIs.append(SI)
    CHs.append(CH)
    DBs.append(DB)

mean_inertia = np.array(inertias).mean(axis=0)
mean_SI = np.array(SIs).mean(axis=0)
mean_CH = np.array(CHs).mean(axis=0)
mean_DB = np.array(DBs).mean(axis=0)
x = np.arange(2, 12)
print('mean_inertia', mean_inertia)
draw_choose_K_alone(x, mean_inertia)
print('mean_SI', mean_SI)
draw_choose_K_alone(x, mean_SI)
print('mean_CH', mean_CH)
draw_choose_K_alone(x, mean_CH)
print('mean_DB', mean_DB)
draw_choose_K_alone(x, mean_DB)
draw_choose_K(x, mean_SI, mean_CH, mean_DB,
              filepath=SAVE_PATH+PICTURE_NAME)
