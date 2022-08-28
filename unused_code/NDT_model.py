import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, BatchNormalization, ELU, \
    Reshape, Softmax, Lambda, Concatenate, Input, Dropout, LSTM

tf.keras.backend.set_floatx('float64')
from models.layers import SamplingLayer, SamplingLayer_gen_alphabet
from models.channel_methods import clean_channel, bsc_channel, z_channel, s_channel, bec_channel


##########################
#  Encoder Models:
##########################
def NDTModel(config):
    """
    Channel encoder model - generating [x,y,p]
    """

    def build(config):
        def build_ff(config):
            """
            FF model case
            """
            # Initializers:
            initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.1)

            encoder_transform = Sequential([
                keras.layers.LSTM(config.enc_hidden[0], return_sequences=True, name="LSTM_ndt", stateful=True,
                                  batch_input_shape=[config.batch_size, 1, config.x_dim + config.y_dim],
                                  dropout=config.ndt_dropout, recurrent_dropout=config.ndt_dropout),
                Dense(config.enc_hidden[1], activation="elu"),
                Dense(config.enc_last_hidden[2], activation="elu"),
                Dense(config.x_dim, activation="sigmoid", bias_initializer=initializer),
                constraint])

            enc_out = list()  # list of X values
            channel_out = list()  # list of Y values

            enc_in_stable = keras.layers.Input(shape=[config.bptt, config.x_dim])
            enc_split = tf.split(enc_in_stable, num_or_size_splits=config.bptt, axis=1)
            enc_in_feedback = keras.layers.Input(shape=[1, config.x_dim])
            enc_in_0 = tf.concat([enc_split[0], enc_in_feedback], axis=-1)

            for t in range(config.bptt):
                if t == 0:
                    enc_out.append(encoder_transform(enc_in_0))
                else:
                    enc_in_t = tf.concat([enc_split[t], enc_out[t - 1]], axis=-1)
                    enc_out.append(
                        constraint_layer(encoder_transform(enc_in_t)))  # SHOULD THE CONSTRAINT BE APPLIED TWICE?!

                channel_out.append(channel(enc_out[t]))  # calculate y_t(x_t)

            channel_out = tf.concat(channel_out, axis=1)  # transform y list into tensor
            enc_out = tf.concat(enc_out, axis=1)  # transform x list into tensor

            encoder = keras.models.Model(inputs=[enc_in_stable, enc_in_feedback], outputs=[enc_out, channel_out])
            return encoder

        def build_fb(config):
            """
            FB model case
            """
            # Initializers:
            initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.1)

            encoder_transform = Sequential([
                keras.layers.LSTM(config.enc_hidden[0], return_sequences=True, name="LSTM_ndt", stateful=True,
                                  batch_input_shape=[config.batch_size, 1, config.x_dim + config.y_dim + 1],
                                  dropout=config.ndt_dropout, recurrent_dropout=config.ndt_dropout),
                Dense(config.enc_hidden[1], activation="elu"),
                Dense(config.enc_hidden[2], activation="elu"),
                Dense(config.x_dim, activation="sigmoid", bias_initializer=initializer),
                constraint])

            enc_out = list()  # list of X values
            channel_out = list()  # list of Y values

            enc_in_stable = keras.layers.Input(shape=[config.bptt, config.x_dim])
            enc_split = tf.split(enc_in_stable, num_or_size_splits=config.bptt, axis=1)
            enc_in_feedback = keras.layers.Input(shape=[1, 2 * config.x_dim])
            enc_in_0 = tf.concat([enc_split[0], enc_in_feedback], axis=-1)

            for t in range(config.bptt):
                if t == 0:
                    enc_out.append(encoder_transform(enc_in_0))
                else:
                    enc_in_t = tf.concat([enc_split[t], enc_out[t - 1], channel_out[t - 1]], axis=-1)
                    enc_out.append(
                        constraint_layer(encoder_transform(enc_in_t)))  # SHOULD THE CONSTRAINT BE APPLIED TWICE?!

                channel_out.append(channel(enc_out[t]))  # calculate y_t(x_t)

            channel_out = tf.concat(channel_out, axis=1)  # transform y list into tensor
            enc_out = tf.concat(enc_out, axis=1)  # transform x list into tensor

            encoder = keras.models.Model(inputs=[enc_in_stable, enc_in_feedback], outputs=[enc_out, channel_out])
            return encoder

        channel = Cont_Channel_Layer(config=config)  # define channel layer
        # channel = Sequential([channel_layer])

        if config.constraint_name == "norm":
            constraint_layer = keras.layers.Lambda(
                lambda x: tf.cast(tf.divide(x, tf.sqrt(tf.reduce_mean(tf.square(x)))), 'float64')
                          * tf.cast(tf.sqrt(config.ndt_power_constraint), 'float64'), name="norm_constraint")
        elif config.constraint_name == "amplitude":
            constraint_layer = keras.layers.Lambda(
                lambda x: tf.clip_by_value(
                    x, -config.ndt_amplitude_constraint, config.ndt_amplitude_constraint, name="clip_constraint"))
        else:
            raise ValueError("'{}' is an invalid constraint name")

        constraint = Sequential([constraint_layer])  # sampling model of x from p

        if config.feedback == 1:
            model = build_fb(config)  # build channel model with feedback, with axis(-1) = 3
        else:
            model = build_ff(config)  # build channel model without feedback, with axis(-1) = 2

        return model

    enc_model = build(config)

    return enc_model


##########################
#  Channel Layers:
##########################
class Cont_Channel_Layer(tf.keras.layers.Layer):  # creates channel outputs from chanel inputs
    def __init__(self, config):
        """
        Unifying class for all channel operations options
        :param config: configuration list of parameters values
        """
        super(Cont_Channel_Layer, self).__init__()
        self.channel = config.channel_name
        self.batch_size = config.batch_size
        self.x_dim = config.x_dim
        self.bptt = config.bptt
        self.awgn_std = np.sqrt(config.awgn_variance)
        self.shape = [self.batch_size, 1, self.x_dim]

        self.state = tf.Variable(tf.zeros(shape=[self.batch_size, 1, config.x_dim], dtype='float64'), trainable=False)
        self.y = tf.Variable(tf.zeros(shape=[self.batch_size, 1, config.x_dim], dtype='float64'), trainable=False)


    def call(self, x, mask=None, training=False):
        if self.channel == "awgn":
            return self.call_awgn(x)
        elif self.channel == "AR_GN":
            return self.call_awgn(x)
        elif self.channel == "MA_GN":
            return self.call_awgn(x)
        else:
            raise ValueError("'{}' is an invalid channel name")

    def call_awgn(self, x):
        y = x + tf.random.normal(self.shape, mean=0.0, stddev=self.awgn_std, dtype='float64')
        return y