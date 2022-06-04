# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn import metrics
from data_drawer import draw_choose_K, draw_choose_K_alone
import time
import copy
import sys
sys.path.append("..")


PICTURE_NAME = 'test7_1_2'
test_num = 5

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
choice = 2
x = np.arange(2, 12)
print('mean_inertia')
print(mean_inertia)
draw_choose_K_alone(x, mean_inertia, choice=choice, ylabel='Cost function J', filepath=SAVE_PATH+'zbfz',
                    formats=('svg', 'png', 'tif'))
print('mean_SI')
print(mean_SI)
draw_choose_K_alone(x, mean_SI, choice=choice,
                    ylabel='Silhouette scores', filepath=SAVE_PATH+'lkxs', formats=('svg', 'png', 'tif'))
print('mean_CH')
print(mean_CH)
draw_choose_K_alone(x, mean_CH, choice=choice,
                    ylabel='Calinski harabasz scores', filepath=SAVE_PATH+'CHI', formats=('svg', 'png', 'tif'))
print('mean_DB')
print(mean_DB)
draw_choose_K_alone(x, mean_DB, choice=choice,
                    ylabel='Davies bouldin scores', filepath=SAVE_PATH+'DBI', formats=('svg', 'png', 'tif'))
draw_choose_K(x, mean_SI, mean_CH, mean_DB,
              filepath=SAVE_PATH+'gl', formats=('svg', 'png', 'tif'))
