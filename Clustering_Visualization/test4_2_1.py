
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from data_drawer import draw_choose_K, draw_choose_K_alone
import time
import copy
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import sys
sys.path.append("..")


def test4_2_1(K=4, dim=2):

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

    pipe_lr = Pipeline([('standard_scaler1', StandardScaler()),
                        ('pca', PCA(n_components=dim))
                        ])
    new_datas = pipe_lr.fit_transform(DLPs)

    km = KMeans(n_clusters=K)
    labels = km.fit_predict(new_datas)
    labels = np.array(labels)
    km_cluster_centers = pipe_lr.inverse_transform(km.cluster_centers_)
    np.savetxt(SAVE_PATH+'test4_2_1_cluster_centers_' +
               str(K) + '.txt', km_cluster_centers)
    np.savetxt(SAVE_PATH+'test4_2_1_labels_' + str(K) + '.txt', labels)
    print(f"K = {K}")
    print("SI", metrics.silhouette_score(
        new_datas, labels, metric='euclidean'))
    print("CH", metrics.calinski_harabasz_score(new_datas, labels))
    print("DB", metrics.davies_bouldin_score(new_datas, labels))
    print("inertia", km.inertia_)

x = np.arange(2, 9)
choice = 4


def test4_2_1_2(K=4, dim=2):
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

    pipe_lr = Pipeline([('standard_scaler1', StandardScaler()),
                        ('pca', PCA(n_components=dim))
                        ])
    new_datas = pipe_lr.fit_transform(DLPs)

    km = KMeans(n_clusters=K)
    labels = km.fit_predict(new_datas)
    print(f"K = {K}")
    return km.inertia_
inertias = []
for i in range(2, 13):
    inertias.append(test4_2_1_2(K=i, dim=2))
plt.plot(inertias)

print(inertias)
