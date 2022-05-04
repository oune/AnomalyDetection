from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def autoencoder_model(input_shape : tuple):
    model = Sequential([
        Dense(4, activation='elu', input_shape=input_shape), 
        Dense(16, activation='elu'),
        Dense(8, activation='elu'),
        Dense(4, activation='elu'),
        Dense(2, activation='elu'),
        Dense(4, activation='elu'),
        Dense(8, activation='elu'),
        Dense(16, activation='elu'),
        Dense(4, activation='elu')
    ])

    return model