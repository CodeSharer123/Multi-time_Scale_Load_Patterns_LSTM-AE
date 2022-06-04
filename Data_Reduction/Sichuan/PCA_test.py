# %%
from utils import MAPPE
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import copy
import sys
sys.path.append('..')
keep = 0.95
choices = ['DLPs', 'WLPs', 'MLPs', 'YLPs']

for choice in choices:
    print(keep, choice)
    old_datas = pd.read_csv(f'./{choice}_toy_datas.csv')
    old_datas.set_index('NO', inplace=True)
    pure_datas = old_datas.to_numpy()
    pure_datas = StandardScaler().fit_transform(pure_datas)
    X_train = pure_datas
    X_test = pure_datas

    pca = PCA(n_components=keep)
    pca.fit(X_train)
    y_tmp = pca.transform(X_test)
    print('latent_dim:', y_tmp.shape[1])
    yhat = pca.inverse_transform(y_tmp)
    print('RMSE:', metrics.mean_squared_error(X_test, yhat, squared=False))
    print('MSE:', metrics.mean_squared_error(X_test, yhat, squared=True))
    print('MAE:', metrics.mean_absolute_error(X_test, yhat))
    print('MAPE:', metrics.mean_absolute_percentage_error(X_test, yhat))
    print('MAPPE:', MAPPE(X_test, yhat))
