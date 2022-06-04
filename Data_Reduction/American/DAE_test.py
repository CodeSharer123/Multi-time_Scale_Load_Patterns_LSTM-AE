import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import copy
import datetime
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class Deep_AutoEncoder(Model):

    def __init__(self, input_dim, latent_dim):
        super(Deep_AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Dense(12, activation='relu'),
            layers.Dense(6, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(6, activation='relu'),
            layers.Dense(12, activation='relu'),
            layers.Dense(input_dim)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# # DLPs

choices = ['DLPs', 'WLPs', 'MLPs', 'YLPs']
latent_dims = [2, 8, 22, 74]
input_dims = [24, 24*7, 24*31, 24*365]
for no, choice in enumerate(choices):
    print(choice)
    old_datas = pd.read_csv(f'./{choice}_toy_datas.csv')
    old_datas.set_index('Unnamed: 0', inplace=True)
    pure_datas = old_datas.to_numpy()
    pure_datas = StandardScaler().fit_transform(pure_datas)
    print(pure_datas.shape)
    # pure_datas = pure_datas.reshape((*pure_datas.shape, 1))
    # print(pure_datas.shape)
    X_train = pure_datas
    X_test = pure_datas
    print(input_dims[no], latent_dims[no])

    # log_dir = "logs\\LSTM_fit_"+choice+"\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(
    #     log_dir=log_dir, histogram_freq=1)
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
    #                                                 patience=20, verbose=0, mode='auto',
    #                                                 baseline=None, restore_best_weights=False)

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam')

    model = Deep_AutoEncoder(input_dims[no], latent_dim=latent_dims[no])
    model.compile(optimizer=opt, loss='mse')
    model.fit(X_train, X_train, epochs=100,
              validation_split=0.1, batch_size=32)

    yhat = model.predict(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1)
    yhat = yhat.reshape(yhat.shape[0], -1)
    print('RMSE:', metrics.mean_squared_error(X_test, yhat, squared=False))
    print('MSE:', metrics.mean_squared_error(X_test, yhat, squared=True))
    print('MAE:', metrics.mean_absolute_error(X_test, yhat))
    print('MAPE:', metrics.mean_absolute_percentage_error(X_test, yhat))


model.fit(X_train, X_train, epochs=100, validation_split=0.1, batch_size=32)

yhat = model.predict(X_test)
X_test = X_test.reshape(X_test.shape[0], -1)
yhat = yhat.reshape(yhat.shape[0], -1)
print('RMSE:', metrics.mean_squared_error(X_test, yhat, squared=False))
print('MSE:', metrics.mean_squared_error(X_test, yhat, squared=True))
print('MAE:', metrics.mean_absolute_error(X_test, yhat))
print('MAPE:', metrics.mean_absolute_percentage_error(X_test, yhat))

model.save('./models/DAE_DLP_American')

# # WLP


class Deep_AutoEncoder(Model):
    def __init__(self, input_dim, latent_dim):
        super(Deep_AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Dense(12*7, activation='relu'),
            layers.Dense(6*7, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(6*7, activation='relu'),
            layers.Dense(12*7, activation='relu'),
            layers.Dense(input_dim)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


choices = ['DLPs', 'WLPs']
latent_dims = [2, 8, 22, 74]
input_dims = [24, 24*7, 24*31, 24*365]
for no, choice in enumerate(choices):
    if choice != 'WLPs':
        continue
    print(choice)
    old_datas = pd.read_csv(f'./{choice}_toy_datas.csv')
    old_datas.set_index('Unnamed: 0', inplace=True)
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
                                                      baseline=None, restore_best_weights=False)

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam')

    model = Deep_AutoEncoder(input_dims[no], latent_dim=latent_dims[no])
    model.compile(optimizer=opt, loss='mse')
    model.fit(X_train, X_train, epochs=100, validation_split=0.1,
              batch_size=32, callbacks=[tensorboard_callback, early_stopping])

    yhat = model.predict(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1)
    yhat = yhat.reshape(yhat.shape[0], -1)
    print('RMSE:', metrics.mean_squared_error(X_test, yhat, squared=False))
    print('MSE:', metrics.mean_squared_error(X_test, yhat, squared=True))
    print('MAE:', metrics.mean_absolute_error(X_test, yhat))
    print('MAPE:', metrics.mean_absolute_percentage_error(X_test, yhat))
    model.save(f'./models/DAE_{choice}_American')

# # MLP


class Deep_AutoEncoder(Model):
    def __init__(self, input_dim, latent_dim):
        super(Deep_AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Dense(12*31, activation='relu'),
            layers.Dense(6*31, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(6*31, activation='relu'),
            layers.Dense(12*31, activation='relu'),
            layers.Dense(input_dim)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


choices = ['DLPs', 'WLPs', 'MLPs']
latent_dims = [2, 8, 22, 74]
input_dims = [24, 24*7, 24*31, 24*365]
for no, choice in enumerate(choices):
    if choice != 'MLPs':
        continue
    print(choice)
    old_datas = pd.read_csv(f'./{choice}_toy_datas.csv')
    old_datas.set_index('Unnamed: 0', inplace=True)
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
                                                      baseline=None, restore_best_weights=False)

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam')

    model = Deep_AutoEncoder(input_dims[no], latent_dim=latent_dims[no])
    model.compile(optimizer=opt, loss='mse')
    model.fit(X_train, X_train, epochs=100, validation_split=0.1,
              batch_size=32, callbacks=[tensorboard_callback, early_stopping])

    yhat = model.predict(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1)
    yhat = yhat.reshape(yhat.shape[0], -1)
    print('RMSE:', metrics.mean_squared_error(X_test, yhat, squared=False))
    print('MSE:', metrics.mean_squared_error(X_test, yhat, squared=True))
    print('MAE:', metrics.mean_absolute_error(X_test, yhat))
    print('MAPE:', metrics.mean_absolute_percentage_error(X_test, yhat))
    model.save(f'./models/DAE_{choice}_American')

# # YLPs


class Deep_AutoEncoder(Model):
    def __init__(self, input_dim, latent_dim):
        super(Deep_AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Dense(12*128, activation='relu'),
            layers.Dense(6*64, activation='relu'),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(6*64, activation='relu'),
            layers.Dense(12*128, activation='relu'),
            layers.Dense(input_dim)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


choices = ['DLPs', 'WLPs', 'MLPs', 'YLPs']
latent_dims = [2, 8, 22, 74]
input_dims = [24, 24*7, 24*31, 24*365]
for no, choice in enumerate(choices):
    if choice != 'YLPs':
        continue
    print(choice)
    old_datas = pd.read_csv(f'./{choice}_toy_datas.csv')
    old_datas.set_index('Unnamed: 0', inplace=True)
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
    model.fit(X_train, X_train, epochs=100, batch_size=32,
              callbacks=[tensorboard_callback, early_stopping])

    yhat = model.predict(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1)
    yhat = yhat.reshape(yhat.shape[0], -1)
    print('RMSE:', metrics.mean_squared_error(X_test, yhat, squared=False))
    print('MSE:', metrics.mean_squared_error(X_test, yhat, squared=True))
    print('MAE:', metrics.mean_absolute_error(X_test, yhat))
    print('MAPE:', metrics.mean_absolute_percentage_error(X_test, yhat))
    model.save(f'./models/DAE_{choice}_American')
