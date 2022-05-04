from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

## input_shape(n, n)
def autoencoder_model(input_shape : tuple):
    inputs = Input(shape=input_shape)
    L1 = LSTM(64, activation='relu', return_sequences=True,
            kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(32, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(input_shape[0])(L2)
    L4 = LSTM(32, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(64, activation='relu', return_sequences=True)(L4)

    output = TimeDistributed(Dense(input_shape[1]))(L5)    
    model = Model(inputs=inputs, outputs=output)

    return model