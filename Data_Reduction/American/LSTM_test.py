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
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


# # DLP
class LSTM_AutoEncoder(Model):
    def __init__(self, input_dim, latent_dim=100, boy=64):
        super(LSTM_AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=input_dim),
            layers.LSTM(boy, activation='tanh', return_sequences=True),
            layers.LSTM(latent_dim, activation='tanh')
        ])
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=latent_dim),
            layers.RepeatVector(input_dim[0]),
            layers.LSTM(boy, activation='tanh', return_sequences=True),
            layers.TimeDistributed(layers.Dense(1))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


choices = ['DLPs', 'WLPs', 'MLPs', 'YLPs']
latent_dims = [2, 8, 22, 74]
input_dims = [24, 24*7, 24*31, 24*365]
boys = [128, 128, 128, 128]


# # WLP

class LSTM_AutoEncoder(Model):
    def __init__(self, input_dim, latent_dim=100, boy=64):
        super(LSTM_AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=input_dim),
            layers.LSTM(boy, activation='tanh', return_sequences=True),
            layers.LSTM(latent_dim, activation='tanh')
        ])
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=latent_dim),
            layers.RepeatVector(input_dim[0]),
            layers.LSTM(boy, activation='tanh', return_sequences=True),
            layers.TimeDistributed(layers.Dense(1))
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
    old_datas.set_index('Unnamed: 0', inplace=True)
    pure_datas = old_datas.to_numpy()
    pure_datas = StandardScaler().fit_transform(pure_datas)
    print(pure_datas.shape)
    pure_datas = pure_datas.reshape((*pure_datas.shape, 1))
    print(pure_datas.shape)
    X_train = pure_datas
    X_test = pure_datas
    print(input_dims[no], latent_dims[no])

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam', clipvalue=0.5)

    log_dir = "logs\\LSTM-AE_fit_"+choice+"\\" + \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
                                                      patience=30, verbose=0, mode='auto',
                                                      baseline=None, restore_best_weights=True)

    model = LSTM_AutoEncoder(
        (input_dims[no], 1), latent_dim=latent_dims[no], boy=boys[no])
    model.compile(optimizer=opt, loss='mse')
    model.fit(X_train, X_train, epochs=300, batch_size=32,
              callbacks=[tensorboard_callback, early_stopping])

    yhat = model.predict(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1)
    yhat = yhat.reshape(yhat.shape[0], -1)
    print('RMSE:', metrics.mean_squared_error(X_test, yhat, squared=False))
    print('MSE:', metrics.mean_squared_error(X_test, yhat, squared=True))
    print('MAE:', metrics.mean_absolute_error(X_test, yhat))
    print('MAPE:', metrics.mean_absolute_percentage_error(X_test, yhat))
    print('MAPPE:', MAPPE(X_test, yhat))
    # model.save(f'./models/LSTM-AE_{choice}_American')


model.save(f'./models/LSTM-AE_{choice}_American')


# # MLP


class LSTM_AutoEncoder(Model):
    def __init__(self, input_dim, latent_dim=100, boy=64):
        super(LSTM_AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_dim),
            layers.LSTM(latent_dim, activation='tanh')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=latent_dim),
            layers.RepeatVector(input_dim[0]),
            layers.LSTM(boy, activation='tanh', return_sequences=True),
            layers.TimeDistributed(layers.Dense(input_dim[1]))
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
    old_datas.set_index('Unnamed: 0', inplace=True)
    pure_datas = old_datas.to_numpy()[0:1000]
    pure_datas = StandardScaler().fit_transform(pure_datas)
    print(pure_datas.shape)
    assert pure_datas.shape[1] % 24 == 0
    pure_datas = pure_datas.reshape(
        (pure_datas.shape[0], 24, pure_datas.shape[1] // 24))
    print(pure_datas.shape)
    print(pure_datas.shape)
    X_train = pure_datas
    X_test = pure_datas
    print(input_dims[no], latent_dims[no])

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam')  # , clipvalue=0.5

    log_dir = "logs\\LSTM-AE_fit_"+choice+"\\" + \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
                                                      patience=30, verbose=0, mode='auto',
                                                      baseline=None, restore_best_weights=True)

    model = LSTM_AutoEncoder(
        (24, input_dims[no] // 24), latent_dim=latent_dims[no], boy=256)
    model.compile(optimizer=opt, loss='mse')
    model.fit(X_train, X_train, epochs=300, batch_size=32,
              callbacks=[tensorboard_callback, early_stopping])

    yhat = model.predict(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1)
    yhat = yhat.reshape(yhat.shape[0], -1)
    print('RMSE:', metrics.mean_squared_error(X_test, yhat, squared=False))
    print('MSE:', metrics.mean_squared_error(X_test, yhat, squared=True))
    print('MAE:', metrics.mean_absolute_error(X_test, yhat))
    print('MAPE:', metrics.mean_absolute_percentage_error(X_test, yhat))
    print('MAPPE:', MAPPE(X_test, yhat))
    # model.save(f'./models/LSTM-AE_{choice}_American')

# %%


class LSTM_AutoEncoder(Model):
    def __init__(self, input_dim, latent_dim=100, boy=64):
        super(LSTM_AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_dim),
            layers.LSTM(boy, activation='tanh', return_sequences=True),
            layers.LSTM(latent_dim, activation='tanh')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=latent_dim),
            layers.RepeatVector(input_dim[0]),
            layers.LSTM(boy, activation='tanh', return_sequences=True),
            layers.TimeDistributed(layers.Dense(input_dim[1]))
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
    old_datas.set_index('Unnamed: 0', inplace=True)
    pure_datas = old_datas.to_numpy()
    pure_datas = StandardScaler().fit_transform(pure_datas)
    print(pure_datas.shape)
    assert pure_datas.shape[1] % 24 == 0
    pure_datas = pure_datas.reshape(
        (pure_datas.shape[0], 24, pure_datas.shape[1] // 24))
    print(pure_datas.shape)
    print(pure_datas.shape)
    X_train = pure_datas
    X_test = pure_datas
    print(input_dims[no], latent_dims[no])

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam')  # , clipvalue=0.5

    log_dir = "logs\\LSTM-AE_fit_"+choice+"\\" + \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
                                                      patience=30, verbose=0, mode='auto',
                                                      baseline=None, restore_best_weights=True)

    model = LSTM_AutoEncoder(
        (24, input_dims[no] // 24), latent_dim=latent_dims[no], boy=256)
    model.compile(optimizer=opt, loss='mse')
    model.fit(X_train, X_train, epochs=300, batch_size=32,
              callbacks=[tensorboard_callback, early_stopping])

    yhat = model.predict(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1)
    yhat = yhat.reshape(yhat.shape[0], -1)
    print('RMSE:', metrics.mean_squared_error(X_test, yhat, squared=False))
    print('MSE:', metrics.mean_squared_error(X_test, yhat, squared=True))
    print('MAE:', metrics.mean_absolute_error(X_test, yhat))
    print('MAPE:', metrics.mean_absolute_percentage_error(X_test, yhat))
    print('MAPPE:', MAPPE(X_test, yhat))
    model.save(f'./models/LSTM-AE_{choice}_American')

# # YLP


class LSTM_AutoEncoder(Model):
    def __init__(self, input_dim, latent_dim=100, boy=64):
        super(LSTM_AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_dim),
            layers.LSTM(boy, activation='tanh', return_sequences=True),
            layers.LSTM(latent_dim, activation='tanh')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=latent_dim),
            layers.RepeatVector(input_dim[0]),
            layers.LSTM(boy, activation='tanh', return_sequences=True),
            layers.TimeDistributed(layers.Dense(input_dim[1]))
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
    old_datas.set_index('Unnamed: 0', inplace=True)
    pure_datas = old_datas.to_numpy()
    pure_datas = StandardScaler().fit_transform(pure_datas)
    print(pure_datas.shape)
    assert pure_datas.shape[1] % 24 == 0
    pure_datas = pure_datas.reshape(
        (pure_datas.shape[0], 24, pure_datas.shape[1] // 24))
    print(pure_datas.shape)
    print(pure_datas.shape)
    X_train = pure_datas
    X_test = pure_datas
    print(input_dims[no], latent_dims[no])

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam')  # , clipvalue=0.5

    log_dir = "logs\\LSTM-AE_fit_"+choice+"\\" + \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
                                                      patience=30, verbose=0, mode='auto',
                                                      baseline=None, restore_best_weights=True)

    model = LSTM_AutoEncoder(
        (24, input_dims[no] // 24), latent_dim=latent_dims[no], boy=512)
    model.compile(optimizer=opt, loss='mse')
    model.fit(X_train, X_train, epochs=300, batch_size=32,
              callbacks=[tensorboard_callback, early_stopping])

    yhat = model.predict(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1)
    yhat = yhat.reshape(yhat.shape[0], -1)
    print('RMSE:', metrics.mean_squared_error(X_test, yhat, squared=False))
    print('MSE:', metrics.mean_squared_error(X_test, yhat, squared=True))
    print('MAE:', metrics.mean_absolute_error(X_test, yhat))
    print('MAPE:', metrics.mean_absolute_percentage_error(X_test, yhat))
    print('MAPPE:', MAPPE(X_test, yhat))
    model.save(f'./models/LSTM-AE_{choice}_American')


class LSTM_AutoEncoder(Model):
    def __init__(self, input_dim, latent_dim=100, boy=64):
        super(LSTM_AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_dim),
            layers.LSTM(boy, activation='tanh', return_sequences=True),
            layers.LSTM(latent_dim, activation='tanh')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=latent_dim),
            layers.RepeatVector(input_dim[0]),
            layers.LSTM(boy, activation='tanh', return_sequences=True),
            layers.TimeDistributed(layers.Dense(input_dim[1]))
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
    old_datas.set_index('Unnamed: 0', inplace=True)
    pure_datas = old_datas.to_numpy()
    pure_datas = StandardScaler().fit_transform(pure_datas)
    print(pure_datas.shape)
    assert pure_datas.shape[1] % 24 == 0
    pure_datas = pure_datas.reshape(
        (pure_datas.shape[0], 24, pure_datas.shape[1] // 24))
    print(pure_datas.shape)
    print(pure_datas.shape)
    X_train = pure_datas
    X_test = pure_datas
    print(input_dims[no], latent_dims[no])

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam')  # , clipvalue=0.5

    log_dir = "logs\\LSTM-AE_fit_"+choice+"\\" + \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
                                                      patience=30, verbose=0, mode='auto',
                                                      baseline=None, restore_best_weights=True)

    model = LSTM_AutoEncoder(
        (24, input_dims[no] // 24), latent_dim=latent_dims[no], boy=1024)
    model.compile(optimizer=opt, loss='mse')
    model.fit(X_train, X_train, epochs=300, batch_size=32,
              callbacks=[tensorboard_callback, early_stopping])

    yhat = model.predict(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1)
    yhat = yhat.reshape(yhat.shape[0], -1)
    print('RMSE:', metrics.mean_squared_error(X_test, yhat, squared=False))
    print('MSE:', metrics.mean_squared_error(X_test, yhat, squared=True))
    print('MAE:', metrics.mean_absolute_error(X_test, yhat))
    print('MAPE:', metrics.mean_absolute_percentage_error(X_test, yhat))
    print('MAPPE:', MAPPE(X_test, yhat))
    model.save(f'./models/LSTM-AE_{choice}_1024_American')

choice = 'YLPs'
old_datas = pd.read_csv(f'./{choice}_toy_datas.csv')
old_datas.set_index('Unnamed: 0', inplace=True)
pure_datas = old_datas.to_numpy()
pure_datas = StandardScaler().fit_transform(pure_datas)
print(pure_datas.shape)
assert pure_datas.shape[1] % 24 == 0
pure_datas = pure_datas.reshape(
    (pure_datas.shape[0], 24, pure_datas.shape[1] // 24))
print(pure_datas.shape)
print(pure_datas.shape)
X_train = pure_datas
X_test = pure_datas
print(input_dims[no], latent_dims[no])

opt = tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')  # , clipvalue=0.5

log_dir = "logs\\LSTM-AE_fit_"+choice+"\\" + \
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
                                                  patience=30, verbose=0, mode='auto',
                                                  baseline=None, restore_best_weights=True)

model = tf.keras.models.load_model(f'./models/LSTM-AE_{choice}_American')
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
print('MAPPE:', MAPPE(X_test, yhat))
# model.save(f'./models/LSTM-AE_{choice}_American')

# %%


class LSTM_AutoEncoder(Model):
    def __init__(self, input_dim, latent_dim=100, boy=64):
        super(LSTM_AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_dim),
            layers.LSTM(boy, activation='tanh', return_sequences=True),
            layers.LSTM(latent_dim, activation='tanh')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=latent_dim),
            layers.RepeatVector(input_dim[0]),
            layers.LSTM(boy, activation='tanh', return_sequences=True),
            layers.TimeDistributed(layers.Dense(input_dim[1]))
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
    old_datas.set_index('Unnamed: 0', inplace=True)
    pure_datas = old_datas.to_numpy()
    pure_datas = StandardScaler().fit_transform(pure_datas)
    print(pure_datas.shape)
    assert pure_datas.shape[1] % 24 == 0
    pure_datas = pure_datas.reshape(
        (pure_datas.shape[0], 24, pure_datas.shape[1] // 24))
    print(pure_datas.shape)
    print(pure_datas.shape)
    X_train = pure_datas
    X_test = pure_datas
    print(input_dims[no], latent_dims[no])

    opt = tf.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam')  # , clipvalue=0.5

    log_dir = "logs\\LSTM-AE_fit_"+choice+"\\" + \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
                                                      patience=30, verbose=0, mode='auto',
                                                      baseline=None, restore_best_weights=True)

    model = LSTM_AutoEncoder(
        (24, input_dims[no] // 24), latent_dim=latent_dims[no], boy=256)
    model.compile(optimizer=opt, loss='mse')
    model.fit(X_train, X_train, epochs=300, batch_size=32,
              callbacks=[tensorboard_callback, early_stopping])

    yhat = model.predict(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1)
    yhat = yhat.reshape(yhat.shape[0], -1)
    print('RMSE:', metrics.mean_squared_error(X_test, yhat, squared=False))
    print('MSE:', metrics.mean_squared_error(X_test, yhat, squared=True))
    print('MAE:', metrics.mean_absolute_error(X_test, yhat))
    print('MAPE:', metrics.mean_absolute_percentage_error(X_test, yhat))
    print('MAPPE:', MAPPE(X_test, yhat))
    model.save(f'./models/LSTM-AE_{choice}_American')
