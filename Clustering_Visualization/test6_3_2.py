# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from data_drawer import draw_choose_K, draw_choose_K_alone
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
import sys
sys.path.append('..')

K = 3
test_num = 10
PICTURE_NAME = 'test6_3_2.svg'


user_info = pd.read_csv(BASIC_PATH)
user_info.set_index('CONS_NO', inplace=True)
new_datas = user_info.to_numpy()

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
        km = KMeans(n_clusters=k)
        labels = km.fit_predict(new_datas)
        SI.append(metrics.silhouette_score(
            new_datas, labels, metric='euclidean'))
        CH.append(metrics.calinski_harabasz_score(new_datas, labels))
        DB.append(metrics.davies_bouldin_score(new_datas, labels))
        inertia.append(km.inertia_)
    SIs.append(SI)
    CHs.append(CH)
    DBs.append(DB)
    inertias.append(inertia)

silhouette_scores = np.array(SIs).mean(axis=0)
calinski_harabasz_scores = np.array(CHs).mean(axis=0)
davies_bouldin_scores = np.array(DBs).mean(axis=0)
inertias = np.array(inertias).mean(axis=0)

choice = 5
x = np.arange(2, 12)
draw_choose_K(x, silhouette_scores, calinski_harabasz_scores,
              davies_bouldin_scores, filepath=SAVE_PATH+'zl',
              formats=('svg', 'png', 'tif'))
draw_choose_K_alone(x, inertias, choice=choice, ylabel='Cost function J', filepath=SAVE_PATH+'zbfz',
                    formats=('svg', 'png', 'tif'))

draw_choose_K_alone(x, silhouette_scores, choice=choice,
                    ylabel='Silhouette scores', filepath=SAVE_PATH+'lkxs', formats=('svg', 'png', 'tif'))

draw_choose_K_alone(x, calinski_harabasz_scores, choice=choice,
                    ylabel='Calinski harabasz scores', filepath=SAVE_PATH+'CHI', formats=('svg', 'png', 'tif'))

draw_choose_K_alone(x, davies_bouldin_scores, choice=choice,
                    ylabel='Davies bouldin scores', filepath=SAVE_PATH+'DBI', formats=('svg', 'png', 'tif'))
