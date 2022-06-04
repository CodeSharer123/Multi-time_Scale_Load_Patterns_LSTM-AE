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
from data_drawer import draw_PCA
import copy
import time


YLPs = None
for choice in ['LOW', 'BASE', 'HIGH']:
    Loads = pd.read_csv(BASIC_PATH+f'resident_YLPs_{choice}.csv')
    Loads.set_index('Number', inplace=True)
    if YLPs is None:
        YLPs = copy.deepcopy(Loads.to_numpy())
    else:
        YLPs = np.concatenate(
            [YLPs, copy.deepcopy(Loads.to_numpy())], axis=0)
PICTURE_NAME = 'test6_1_1'

np.random.seed(233)
np.random.shuffle(YLPs) 
np.random.seed(int(time.time()))

dim = 900
ss = StandardScaler()
new_datas = ss.fit_transform(YLPs)
pca = PCA(n_components=dim)
pca.fit(new_datas)

pca_80 = PCA(n_components=0.80)
dim_80 = pca_80.fit_transform(new_datas).shape[1]
print(dim_80)

pca_90 = PCA(n_components=0.90)
dim_90 = pca_90.fit_transform(new_datas).shape[1]
print(dim_90)

pca_95 = PCA(n_components=0.95)
dim_95 = pca_95.fit_transform(new_datas).shape[1]
print(dim_95)

pca_99 = PCA(n_components=0.99)
dim_99 = pca_99.fit_transform(new_datas).shape[1]
print(dim_99)

draw_PCA(dim, [dim_90, dim_95, dim_99],
         pca.explained_variance_ratio_.tolist(), ch=1, filepath=SAVE_PATH+PICTURE_NAME, formats=('svg', 'png', 'tif'))
