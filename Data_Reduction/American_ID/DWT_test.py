import numpy as np
import pandas as pd
import pywt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import copy
import sys
sys.path.append('..')
from utils import MAPPE

choices = ['DLPs', 'WLPs', 'MLPs', 'YLPs']

mask_boys = {"DLPs":2, "WLPs": 4, "MLPs":4, "YLPs":5}
for choice in choices:
    pure_datas = np.load(f"com_American_ID_{choice}.npy")
    pure_datas = StandardScaler().fit_transform(pure_datas)
    X_train = pure_datas
    X_test = pure_datas
    y_tmp = pywt.wavedec(X_train, 'haar') # dsad
    print(len(y_tmp), y_tmp[mask_boys[choice]].shape)
    for i in range(len(y_tmp)):
        if i >= mask_boys[choice]:
            y_tmp[i] = np.zeros_like(y_tmp[i])
    yhat = pywt.waverec(y_tmp, 'haar')
    print('RMSE:', metrics.mean_squared_error(X_test, yhat, squared=False))
    print('MSE:', metrics.mean_squared_error(X_test, yhat, squared=True))
    print('MAE:', metrics.mean_absolute_error(X_test, yhat))
    print('MAPE:', metrics.mean_absolute_percentage_error(X_test, yhat))
    print('MAPPE:', MAPPE(X_test, yhat))