import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def DVMINEModel(config):
    """
    DV Model For MINE simluation
    """
    def build_DV(input_shape, config):
        randN_05 = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
        bias_init = randN_05

        dense0 = layers.Dense(config.DV_hidden[0], bias_initializer=bias_init, kernel_initializer=randN_05, activation="elu")
        dense1 = layers.Dense(config.DV_hidden[1], bias_initializer=bias_init, kernel_initializer=randN_05,
                              activation="elu")
        dense2 = layers.Dense(1, bias_initializer=bias_init, kernel_initializer=randN_05, activation=None)
        batch_norm = layers.BatchNormalization()

        in_ = t = layers.Input(batch_shape=input_shape)
        t = batch_norm(t)
        t = dense0(t)
        t = dense1(t)
        t = dense2(t)
        model = keras.models.Model(inputs=in_, outputs=t)
        return model

    in_shape = [config.batch_size, config.x_dim + config.y_dim]  # x_dim + y_dim

    model = build_DV(in_shape, config)

    return model


def PMFMINEModel(config):
    """
    PMF model for MINE simulations
    """
    def build_model():
        randN_05 = keras.initializers.RandomNormal(mean=0.0, stddev=0.08, seed=None)
        initializer = randN_05

        logit_out = layers.Dense(units=config.x_alphabet, activation=None, bias_initializer=initializer, kernel_initializer=initializer, name="logits_output")

        # no depth:
        model_in = keras.layers.Input(shape=[1])
        model_out = logit_out(model_in)

        model = keras.models.Model(inputs=model_in, outputs=model_out)

        return model


    model = build_model()

    return model


def SamplerMINEModel(config):
    """
    Sampling model for MINE simulations - sampling from modulations
    """
    if config.constellation == "1d":
        sampler_layer = SamplingLayer1D(config)
    elif config.constellation == "qam_p_norm":
        sampler_layer = SamplingLayer_QAM_P_norm(config)
    else:
        raise ValueError("'{}' is an invalid constellation")
    model = Sequential([sampler_layer])  # sampling model of x from
    return model


############# Sampler Implementations ###############

class SamplingLayer_QAM_P_norm(tf.keras.layers.Layer):  # samples x from probability tensor
    """
    Sampler for QAM constellations with fixed size.
    Contains several additional commented normalization options.
    """
    def __init__(self, config):
        """
        Keras layer class to sample QAM from a probability tensor
        """
        super(SamplingLayer_QAM_P_norm, self).__init__()
        self.batch_size = config.batch_size
        self.M = config.x_alphabet
        self.k = np.sqrt(self.M)
        if not(self.k % 2 == 0):
            raise ValueError("'{}' is an invalid qam constellation order")
        self.gen_QAM_constellation()


    def call(self, logits, mask=None):
        # Sample indices from logits:
        indices = tf.random.categorical(logits=logits, num_samples=1)  # sample x~p for each p_t element, shape [B, 1]
        # map into one-hot vectors:
        ind = tf.cast(tf.one_hot(indices=tf.reshape(indices, [-1]), depth=int(self.M)), 'float64')

        # option 1: Constant constellation (divided by normalization factor) for a pmf regularization option
        # x = tf.matmul(ind, self.normed_mat)   # by some normalization factor
        x = tf.matmul(ind, self.QAM_mat)  # predetermined QAM mat (-1,1)


        # option 2: Direct constellation normalization (sampling from normalized constellation):
        # logits_ = logits.numpy()
        # p = tf.nn.softmax(logits_[0, :])  # obtain pmf
        # m = tf.reduce_sum(tf.square(self.QAM_mat), axis=-1)  # obtain constellation norm
        # norm_factor = tf.sqrt(tf.reduce_sum(m*p))
        # normed_constellation = self.QAM_mat / norm_factor  # normalize energy to 1
        # x = tf.matmul(ind, normed_constellation)


        # option 3: same as before but without p.numpy() copy
        # p = tf.nn.softmax(logits[0, :])  # obtain pmf
        # m = tf.reduce_sum(tf.square(self.QAM_mat), axis=-1)  # obtain constellation norm
        # norm_factor = tf.sqrt(tf.reduce_sum(m*p))
        # self.normed_constellation = self.QAM_mat / norm_factor  # normalize energy to 1
        # x = tf.matmul(ind, self.normed_constellation)



        ##########################
        ##########################
        ##########################
        # shortened norm - original for experiments:
        # logits_ = logits.numpy()
        # p = tf.nn.softmax(logits_[0, :])
        # m = tf.reduce_sum(tf.square(self.QAM_mat), axis=-1)
        # norm_factor = tf.sqrt(tf.reduce_sum(m*p))
        # self.normed_mat = self.QAM_mat / norm_factor
        # x = tf.matmul(ind, self.normed_mat)

        # shortened norm - taking into consideration the normalization:
        # p = tf.nn.softmax(logits[0, :])
        # m = tf.reduce_sum(tf.square(self.QAM_mat), axis=-1)
        # norm_factor = tf.sqrt(tf.reduce_sum(m*p))
        # x = tf.matmul(ind, self.QAM_mat)/norm_factor

        # new norm implementation over all batch elements:
        # p = tf.nn.softmax(logits)
        # m = tf.reduce_sum(tf.square(self.QAM_mat), axis=-1, keepdims=True)
        # norm_factor = tf.sqrt(tf.matmul(p, m))
        # x_unnormed = tf.matmul(ind, self.QAM_mat)
        # x = x_unnormed/norm_factor

        # original p_norm:
        # norm_factor = tf.matmul(tf.transpose(tf.expand_dims(p, axis=-1)), tf.expand_dims(tf.reduce_sum(tf.square(self.QAM_mat), axis=-1), axis=-1))
        # self.normed_mat = self.QAM_mat/np.sqrt(norm_factor)
        # x = tf.matmul(ind, self.normed_mat)



        # constant norm factor for debugging:
        # norm_factor = 10.0
        # self.normed_mat = self.QAM_mat/np.sqrt(norm_factor)
        # x = tf.matmul(ind, self.normed_mat)


        # print('norm factor={}'.format(norm_factor))

        # tryingg with tf.norm:
        # m = tf.norm(self.QAM_mat, axis=-1)
        # norm_factor = tf.reduce_sum(m * p)
        # self.normed_mat = self.QAM_mat / norm_factor

        # x = tf.matmul(ind, self.normed_mat)

        return tf.concat([tf.cast(indices, 'float64'), x], axis=-1)

    def gen_QAM_constellation(self):
        axe = np.linspace(start=-1, stop=1, num=int(self.k))
        Q0, Q1 = np.meshgrid(axe, axe)
        self.QAM_mat = tf.cast(np.stack([Q0.flatten(), Q1.flatten()], axis=-1), 'float64')
        self.uniform_norm_factor = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(self.QAM_mat), axis=-1)))
        self.normed_mat = self.QAM_mat / self.uniform_norm_factor


class SamplingLayer1D(tf.keras.layers.Layer):   # samples x from probability tensor
    """
    Sampler of 1D constellations [-A,A] by sampling in [0,1] and applying a linear transformation
    """
    def __init__(self, config):
        """
        Keras layer class to sample from a probability tensor
        """
        super(SamplingLayer1D, self).__init__()
        if config.constellation_by_A:
            self.A = np.sqrt(10 ** (config.snr / 10))
        else:
            self.A = 1
        self.gen_alphabet_params(config)
        self.batch_size = config.batch_size



    def call(self, logits, mask=None):
        indices = tf.cast(tf.random.categorical(logits=logits, num_samples=1), 'float64')  # sample x~p for each p_t element, shape [B, 1]
        x = self.c_line + ( (self.d_line-self.c_line)/(self.b_line-self.a_line) )*(indices-self.a_line)  # map to [-A,A]

        return tf.concat([indices, x], axis=-1)

    def gen_alphabet_params(self, config):
        self.a_line = 0.
        self.b_line = config.x_alphabet-1.
        # for assymetric edges:
        # self.c_line = config.A_min
        # self.d_line = config.A_max
        # for symmetric:
        self.c_line = -self.A
        self.d_line = self.A
