import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import math

 

class SCSAE(Model):
    def __init__(self, input_dim, latent_dim):
        super(SCSAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=input_dim), # like (4, 4, 6)
            layers.Conv2D(16, kernel_size=3, activation='relu',
                          padding='same', strides=2), # (2, 2, 16)
            layers.Conv2D(32, kernel_size=3, activation='relu',
                          padding='same', strides=1), # (2, 2, 32)
            layers.SeparableConv2D(64, kernel_size=3, activation='relu',
                padding='same', strides=1),           # (2, 2, 64) 
            layers.MaxPool2D(pool_size=2, strides=2), # (1, 1, 64) 
            layers.Flatten(),
            layers.Dense(latent_dim)
        ])
        boy = input_dim[0] * input_dim[1] * 16
        self.decoder = tf.keras.Sequential([
            layers.Dense(boy, activation='relu'),
            layers.Reshape((input_dim[0]//2, input_dim[1]//2, 64)), 
            layers.Conv2DTranspose(64, kernel_size=3, strides=1,
                                   activation='relu', padding='same'),
            layers.Conv2DTranspose(32, kernel_size=3, strides=1,
                                   activation='relu', padding='same'),
            layers.Conv2DTranspose(16, kernel_size=3, activation='relu',
                          padding='same', strides=2),
            layers.Conv2D(input_dim[2], kernel_size=1, padding='same')
        ])


    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == '__main__':
    # DLP (8, 12, 1) 
    # WLP (8, 12, 7) 
    # MLP (8, 12, 31)
    # YLP (8, 12, 365) 
    toy_data = tf.random.uniform((5,8, 12, 365) )
    model = SCSAE((8, 12, 365) , 8)
    opt = tf.keras.optimizers.Adam(
    learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')
    model.compile(optimizer=opt, loss='mse')
    model.fit(toy_data, toy_data)
    y_hat = model.predict(toy_data)
    print(y_hat.shape)
