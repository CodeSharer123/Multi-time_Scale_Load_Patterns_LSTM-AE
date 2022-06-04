from utils import MAPPE
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import copy
import sys
sys.path.append('..')
keep = 0.95
choices = ['DLPs', 'WLPs', 'MLPs', 'YLPs']


for choice in choices:
    print(keep, choice)
    pure_datas = None
    for choice2 in ['LOW', 'BASE', 'HIGH']:
        filepath = r""

        Loads = pd.read_csv(filepath)
        Loads.drop(['Date'], axis=1, inplace=True)
        if pure_datas is None:
            pure_datas = copy.deepcopy(Loads.to_numpy())
        else:
            pure_datas = np.concatenate(
                [pure_datas, copy.deepcopy(Loads.to_numpy())], axis=0)
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
