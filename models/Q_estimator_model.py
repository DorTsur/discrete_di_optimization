from tensorflow.keras.layers import LSTM, Dense, Input, Lambda
from tensorflow.keras.models import Model
import tensorflow as tf

def Q_model(config):
    """
    building the Q-graph estimation model
    """
    def build(config):
        input_shape = [config.batch_size, config.bptt, 1]
        lstm_layer = LSTM(units=config.q_lstm_units, return_sequences=True, name="Q_lstm", stateful=True)
        softmax = Dense(units=config.s_alphabet, activation="softmax")
        noise_layer = Lambda(lambda x: x + tf.random.normal(shape=tf.shape(x),mean=0,stddev=config.noise_layer_q_std, dtype='float64'))

        in_ = Input(batch_shape=input_shape)
        states = lstm_layer(in_)
        out = softmax(states)

        model = Model(inputs=in_, outputs=out)
        return model

    return build(config)


def Q_model_noise(config, training):
    """
    building the Q-graph estimation model
    """
    def build(config, training):
        input_shape = [config.batch_size, config.bptt, 1]
        lstm_layer = LSTM(units=config.q_lstm_units, return_sequences=True, name="Q_lstm", stateful=True)
        softmax = Dense(units=config.s_alphabet, activation="softmax")
        noise_layer = Lambda(lambda x: x + tf.random.normal(shape=tf.shape(x),mean=0,stddev=config.noise_layer_q_std, dtype='float64'))
        fcn = Dense(units=32, activation="elu")

        in_ = Input(batch_shape=input_shape)
        states = lstm_layer(in_)
        ###
        if training:
            # noise layer:
            states = noise_layer(states)
        ###

        ### try:
        states = fcn(states)
        ###
        out = softmax(states)

        model = Model(inputs=in_, outputs=out)
        return model

    return build(config, training)
