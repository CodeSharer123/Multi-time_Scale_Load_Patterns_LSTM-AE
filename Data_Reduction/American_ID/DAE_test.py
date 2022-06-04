from utils import MAPPE
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import copy
import datetime
import sys
sys.path.append('..')

choices = ['DLPs', 'WLPs', 'MLPs', 'YLPs']
latent_dims = [19, 15, 17, 16]
input_dims = [96, 96*7, 96*31, 96*365]

# # DLPs


class Deep_AutoEncoder(Model):
    def __init__(self, input_dim, latent_dim):
        super(Deep_AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Dense(48, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(12, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(12, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(48, activation='relu'),
            layers.Dense(input_dim)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


for no, choice in enumerate(choices):
    if choice != 'DLPs':
        continue
    print(choice)
    old_datas = pd.read_csv(f'./{choice}_toy_datas.csv')
    old_datas.set_index('NO', inplace=True)
    pure_datas = old_datas.to_numpy()
    pure_datas = StandardScaler().fit_transform(pure_datas)
    print(pure_datas.shape)
    # pure_datas = pure_datas.reshape((*pure_datas.shape, 1))
    # print(pure_datas.shape)
    X_train = pure_datas
    X_test = pure_datas
    print(input_dims[no], latent_dims[no])

    log_dir = "logs\\DAE_fit_"+choice+"\\" + \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
                                                      patience=20, verbose=0, mode='auto',
                                                      baseline=None, restore_best_weights=True)

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam')

    model = Deep_AutoEncoder(input_dims[no], latent_dim=latent_dims[no])
    model.compile(optimizer=opt, loss='mse')
    model.fit(X_train, X_train, epochs=200, batch_size=32,
              callbacks=[tensorboard_callback, early_stopping])

    yhat = model.predict(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1)
    yhat = yhat.reshape(yhat.shape[0], -1)
    print('RMSE:', metrics.mean_squared_error(X_test, yhat, squared=False))
    print('MSE:', metrics.mean_squared_error(X_test, yhat, squared=True))
    print('MAE:', metrics.mean_absolute_error(X_test, yhat))
    print('MAPE:', metrics.mean_absolute_percentage_error(X_test, yhat))
    print('MAPPE:', MAPPE(X_test, yhat))
    model.save(f'./models/DAE_{choice}_Sichuan')

# # WLP


class Deep_AutoEncoder(Model):
    def __init__(self, input_dim, latent_dim):
        super(Deep_AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Dense(48*7, activation='relu'),
            layers.Dense(24*7, activation='relu'),
            # layers.Dense(12*7, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            # layers.Dense(12*7, activation='relu'),
            layers.Dense(24*7, activation='relu'),
            layers.Dense(48*7, activation='relu'),
            layers.Dense(input_dim)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


for no, choice in enumerate(choices):
    if choice != 'WLPs':
        continue
    print(choice)
    old_datas = pd.read_csv(f'./{choice}_toy_datas.csv')
    old_datas.set_index('NO', inplace=True)
    pure_datas = old_datas.to_numpy()
    pure_datas = StandardScaler().fit_transform(pure_datas)
    print(pure_datas.shape)
    # pure_datas = pure_datas.reshape((*pure_datas.shape, 1))
    # print(pure_datas.shape)
    X_train = pure_datas
    X_test = pure_datas
    print(input_dims[no], latent_dims[no])

    log_dir = "logs\\DAE_fit_"+choice+"\\" + \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
                                                      patience=30, verbose=0, mode='auto',
                                                      baseline=None, restore_best_weights=True)

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam')

    model = Deep_AutoEncoder(input_dims[no], latent_dim=latent_dims[no])
    model.compile(optimizer=opt, loss='mse')
    model.fit(X_train, X_train, epochs=200, batch_size=32,
              callbacks=[tensorboard_callback, early_stopping])

    yhat = model.predict(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1)
    yhat = yhat.reshape(yhat.shape[0], -1)
    print('RMSE:', metrics.mean_squared_error(X_test, yhat, squared=False))
    print('MSE:', metrics.mean_squared_error(X_test, yhat, squared=True))
    print('MAE:', metrics.mean_absolute_error(X_test, yhat))
    print('MAPE:', metrics.mean_absolute_percentage_error(X_test, yhat))
    print('MAPPE:', MAPPE(X_test, yhat))
    model.save(f'./models/DAE_{choice}_Sichuan')

# # MLP


class Deep_AutoEncoder(Model):
    def __init__(self, input_dim, latent_dim):
        super(Deep_AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Dense(48*31, activation='relu'),
            layers.Dense(24*31, activation='relu'),
            layers.Dense(12*31, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(12*31, activation='relu'),
            layers.Dense(24*31, activation='relu'),
            layers.Dense(48*31, activation='relu'),
            layers.Dense(input_dim)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


for no, choice in enumerate(choices):
    if choice != 'MLPs':
        continue
    print(choice)
    old_datas = pd.read_csv(f'./{choice}_toy_datas.csv')
    old_datas.set_index('NO', inplace=True)
    pure_datas = old_datas.to_numpy()
    pure_datas = StandardScaler().fit_transform(pure_datas)
    print(pure_datas.shape)
    # pure_datas = pure_datas.reshape((*pure_datas.shape, 1))
    # print(pure_datas.shape)
    X_train = pure_datas
    X_test = pure_datas
    print(input_dims[no], latent_dims[no])

    log_dir = "logs\\DAE_fit_"+choice+"\\" + \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
                                                      patience=30, verbose=0, mode='auto',
                                                      baseline=None, restore_best_weights=True)

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam')

    model = Deep_AutoEncoder(input_dims[no], latent_dim=latent_dims[no])
    model.compile(optimizer=opt, loss='mse')
    model.fit(X_train, X_train, epochs=200, batch_size=32,
              callbacks=[tensorboard_callback, early_stopping])

    yhat = model.predict(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1)
    yhat = yhat.reshape(yhat.shape[0], -1)
    print('RMSE:', metrics.mean_squared_error(X_test, yhat, squared=False))
    print('MSE:', metrics.mean_squared_error(X_test, yhat, squared=True))
    print('MAE:', metrics.mean_absolute_error(X_test, yhat))
    print('MAPE:', metrics.mean_absolute_percentage_error(X_test, yhat))
    print('MAPPE:', MAPPE(X_test, yhat))
    model.save(f'./models/DAE_{choice}_Sichuan')

# # YLPs


class Deep_AutoEncoder(Model):
    def __init__(self, input_dim, latent_dim):
        super(Deep_AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Dense(48*200, activation='relu'),
            layers.Dense(24*100, activation='relu'),
            layers.Dense(12*50, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(12*50, activation='relu'),
            layers.Dense(24*100, activation='relu'),
            layers.Dense(48*200, activation='relu'),
            layers.Dense(input_dim)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


for no, choice in enumerate(choices):
    if choice != 'YLPs':
        continue
    print(choice)
    old_datas = pd.read_csv(f'./{choice}_toy_datas.csv')
    old_datas.set_index('NO', inplace=True)
    pure_datas = old_datas.to_numpy()
    pure_datas = StandardScaler().fit_transform(pure_datas)
    print(pure_datas.shape)
    # pure_datas = pure_datas.reshape((*pure_datas.shape, 1))
    # print(pure_datas.shape)
    X_train = pure_datas
    X_test = pure_datas
    print(input_dims[no], latent_dims[no])

    log_dir = "logs\\DAE_fit_"+choice+"\\" + \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
                                                      patience=30, verbose=0, mode='auto',
                                                      baseline=None, restore_best_weights=True)

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam')

    model = Deep_AutoEncoder(input_dims[no], latent_dim=latent_dims[no])
    model.compile(optimizer=opt, loss='mse')
    model.fit(X_train, X_train, epochs=200, batch_size=32,
              callbacks=[tensorboard_callback, early_stopping])

    yhat = model.predict(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1)
    yhat = yhat.reshape(yhat.shape[0], -1)
    print('RMSE:', metrics.mean_squared_error(X_test, yhat, squared=False))
    print('MSE:', metrics.mean_squared_error(X_test, yhat, squared=True))
    print('MAE:', metrics.mean_absolute_error(X_test, yhat))
    print('MAPE:', metrics.mean_absolute_percentage_error(X_test, yhat))
    print('MAPPE:', MAPPE(X_test, yhat))
    model.save(f'./models/DAE_{choice}_Sichuan')
