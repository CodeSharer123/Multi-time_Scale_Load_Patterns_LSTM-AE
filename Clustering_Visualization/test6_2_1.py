# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from data_drawer import draw_choose_K
import pickle
import time
import os

debug = False
choices = ['LOW', 'BASE', 'HIGH']
K = {'LOW': 2, 'BASE': 2, 'HIGH': 2} 
dims = {'LOW': 77, 'BASE': 115, 'HIGH': 169} 
for choice in choices:
    print('For', choice)
    print('Debug is', debug)

    YLPs = pd.read_csv(BASIC_PATH+'resident_YLPs_' + choice + '.csv')
    YLPs.set_index('Unnamed: 0', inplace=True)
    YLPs = YLPs.to_numpy()

    np.random.seed(233)
    np.random.shuffle(YLPs)  
    np.random.seed(int(time.time()))

    pipe_lr = Pipeline([('standard_scaler1', StandardScaler()),
                        ('pca', PCA(n_components=dims[choice])),
                        ('MinMax_scaler2', MinMaxScaler()),
                        ])
    new_datas = pipe_lr.fit_transform(YLPs)

    if os.path.exists('./test6_2_1_labels_'+choice+'.txt') or os.path.exists(SAVE_PATH+'test6_2_1_labels_'+choice+'.txt'):
        pass
    else:
        km = KMeans(n_clusters=K[choice])
        labels = km.fit_predict(new_datas)
        labels = np.array(labels)
        km_cluster_centers = pipe_lr.inverse_transform(km.cluster_centers_)
        if debug:
            np.savetxt('./test6_2_1_cluster_centers_' +
                       choice+'.txt', km_cluster_centers)
            np.savetxt('./test6_2_1_labels_'+choice+'.txt', labels)
        else:
            np.savetxt(SAVE_PATH+'test6_2_1_cluster_centers_' +
                       choice+'.txt', km_cluster_centers)
            np.savetxt(SAVE_PATH+'test6_2_1_labels_'+choice+'.txt', labels)
