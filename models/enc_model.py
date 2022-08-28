import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, BatchNormalization, ELU, \
    Reshape, Softmax, Lambda, Concatenate, Input, Dropout, LSTM
tf.keras.backend.set_floatx('float64')
from models.layers import SamplingLayer, SamplingLayer_gen_alphabet
from models.channel_methods import clean_channel, bsc_channel, z_channel, s_channel, bec_channel



##########################
#  Encoder Models:
##########################
def PMFModel(config):
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

            # Encoder transformation with sigmoid output:
            encoder_transform = Sequential([
                keras.layers.LSTM(config.enc_hidden[0], return_sequences=True, name="LSTM_enc", stateful=True,
                                  batch_input_shape=[config.batch_size, 1, config.x_dim],
                                  dropout=config.enc_dropout, recurrent_dropout=config.enc_dropout),
                Dense(config.enc_hidden[1], activation="elu"),
                # Dense(config.enc_last_hidden, activation="elu"),
                Dense(config.x_dim, activation="sigmoid", bias_initializer=initializer)])  # WITH SIGMOID AS LAST LAYER

            p_out = list()  # list of calculated P(X|H)
            enc_out = list()  # list of X values
            channel_out = list()  # list of Y values

            enc_in_feedback = keras.layers.Input(shape=[1, config.x_dim])  # Input layer
            enc_in_0 = enc_in_feedback

            for t in range(config.bptt):
                if t == 0:
                    p_out.append(encoder_transform(enc_in_0))  # append transformation of first input
                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append sampled x
                else:
                    enc_in_t = enc_out[t - 1]  # prepare input for time t
                    p_out.append(encoder_transform(enc_in_t))  # calculate p_t(in_t)
                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append x_t

                channel_out.append(channel(enc_out[t]))  # calculate y_t(x_t)

            channel_out = tf.concat(channel_out, axis=1)  # transform y list into tensor
            enc_out = tf.concat(enc_out, axis=1)  # transform x list into tensor
            p_out = tf.concat(p_out, axis=1)  # transform p list into tensor

            encoder = Model(inputs=[enc_in_feedback], outputs=[enc_out, channel_out, p_out])
            return encoder


        def build_fb(config):
            """
            FF model case
            """
            # Initializers:
            initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.1)

            encoder_transform = Sequential([
                keras.layers.LSTM(config.enc_hidden[0], return_sequences=True, name="LSTM_enc", stateful=True,
                                  batch_input_shape=[config.batch_size, 1, config.x_dim + config.y_dim],
                                  dropout=config.enc_dropout, recurrent_dropout=config.enc_dropout),
                Dense(config.enc_hidden[1], activation="elu"),
                # Dense(config.enc_last_hidden, activation="elu"),
                Dense(config.x_dim, activation="sigmoid", bias_initializer=initializer)])

            p_out = list()  # list of calculated P(X|H)
            enc_out = list()  # list of X values
            channel_out = list()  # list of Y values

            enc_in_feedback = keras.layers.Input(shape=[1, config.y_dim + config.x_dim])  # Input layer
            enc_in_0 = enc_in_feedback

            for t in range(config.bptt):
                if t == 0:
                    p_out.append(encoder_transform(enc_in_0))  # append transformation of first input
                    # for debug:
                    # p = encoder_transform(enc_in_0)
                    # p_out.append(0.5*tf.ones(shape=[config.batch_size,1,1], dtype='float64')+0*p)
                    ##
                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append sampled x
                else:
                    enc_in_t = tf.concat([enc_out[t - 1], channel_out[t-1]], axis=-1)  # prepare input for time t
                    p_out.append(encoder_transform(enc_in_t))  # calculate p_t(in_t)
                    # for debug:
                    # p = encoder_transform(enc_in_0)
                    # p_out.append(0.5 * tf.ones(shape=[config.batch_size, 1, 1], dtype='float64') + 0 * p)
                    ##

                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append x_t

                channel_out.append(channel(enc_out[t]))  # calculate y_t(x_t)

            channel_out = tf.concat(channel_out, axis=1)  # transform y list into tensor
            enc_out = tf.concat(enc_out, axis=1)  # transform x list into tensor
            p_out = tf.concat(p_out, axis=1)  # transform p list into tensor

            encoder = Model(inputs=[enc_in_feedback], outputs=[enc_out, channel_out, p_out])
            return encoder

        channel = Channel_Layer(config=config)  # define channel layer
        sampler_layer = SamplingLayer()
        sampler = Sequential([sampler_layer])  # sampling model of x from p

        if config.feedback == 1:
            model = build_fb(config)  # build channel model with feedback, with axis(-1) = 3
        else:
            model = build_ff(config)  # build channel model without feedback, with axis(-1) = 2

        return model

    enc_model = build(config)

    return enc_model


def PMFModelQ_train(config):
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
                keras.layers.LSTM(config.enc_hidden[0], return_sequences=True, name="LSTM_enc", stateful=True,
                                  batch_input_shape=[config.batch_size, 1, config.x_dim],
                                  dropout=config.enc_dropout, recurrent_dropout=config.enc_dropout),
                Dense(config.enc_hidden[1], activation="elu"),
                Dense(config.x_dim, activation="sigmoid", bias_initializer=initializer)])

            p_out = list()  # list of calculated P(X|H)
            enc_out = list()  # list of X values
            channel_out = list()  # list of Y values
            channel_state = list()

            enc_in_feedback = keras.layers.Input(shape=[1, config.x_dim])  # Input layer
            enc_in_0 = enc_in_feedback

            for t in range(config.bptt):
                if t == 0:
                    p_out.append(encoder_transform(enc_in_0))  # append transformation of first input
                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append sampled x
                else:
                    p_out.append(encoder_transform(enc_out[t - 1]))  # calculate p_t(in_t)

                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append x_t

                [y_t, s_t] = channel(enc_out[t])
                channel_out.append(y_t)
                channel_state.append(s_t)



            channel_out = tf.concat(channel_out, axis=1)  # transform y list into tensor
            enc_out = tf.concat(enc_out, axis=1)  # transform x list into tensor
            p_out = tf.concat(p_out, axis=1)  # transform p list into tensor
            channel_state = tf.concat(channel_state, axis=1)
            encoder = Model(inputs=[enc_in_feedback], outputs=[enc_out, channel_out, p_out, channel_state])

            return encoder

        def build_fb(config):
            """
            FF model case
            """
            # Initializers:
            initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.1)

            encoder_transform = Sequential([
                keras.layers.LSTM(config.enc_hidden[0], return_sequences=True, name="LSTM_enc", stateful=True,
                                  batch_input_shape=[config.batch_size, 1, config.x_dim + config.y_dim],
                                  dropout=config.enc_dropout, recurrent_dropout=config.enc_dropout),
                Dense(config.enc_hidden[1], activation="elu"),
                # Dense(config.enc_last_hidden, activation="elu"),
                Dense(config.x_dim, activation="sigmoid", bias_initializer=initializer)])

            p_out = list()  # list of calculated P(X|H)
            enc_out = list()  # list of X values
            channel_out = list()  # list of Y values
            channel_state = list()

            enc_in_feedback = keras.layers.Input(shape=[1, config.y_dim + config.x_dim])  # Input layer
            enc_in_0 = enc_in_feedback

            for t in range(config.bptt):
                if t == 0:
                    p_out.append(encoder_transform(enc_in_0))  # append transformation of first input
                    # for debug:
                    # p = encoder_transform(enc_in_0)
                    # p_out.append(0.5*tf.ones(shape=[config.batch_size,1,1], dtype='float64')+0*p)
                    ##
                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append sampled x
                else:
                    enc_in_t = tf.concat([enc_out[t - 1], channel_out[t-1]], axis=-1)  # prepare input for time t
                    p_out.append(encoder_transform(enc_in_t))  # calculate p_t(in_t)
                    # for debug:
                    # p = encoder_transform(enc_in_0)
                    # p_out.append(0.5 * tf.ones(shape=[config.batch_size, 1, 1], dtype='float64') + 0 * p)
                    ##

                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append x_t

                [y_t, s_t] = channel(enc_out[t])
                channel_out.append(y_t)
                channel_state.append(s_t)



            channel_out = tf.concat(channel_out, axis=1)  # transform y list into tensor
            enc_out = tf.concat(enc_out, axis=1)  # transform x list into tensor
            p_out = tf.concat(p_out, axis=1)  # transform p list into tensor
            channel_state = tf.concat(channel_state, axis=1)
            encoder = Model(inputs=[enc_in_feedback], outputs=[enc_out, channel_out, p_out, channel_state])

            return encoder

        channel = Channel_Layer_no_p_new(config=config)  # define channel layer
        sampler_layer = SamplingLayer()
        sampler = Sequential([sampler_layer])  # sampling model of x from p

        if config.feedback == 1:
            model = build_fb(config)  # build channel model with feedback, with axis(-1) = 3
        else:
            model = build_ff(config)  # build channel model without feedback, with axis(-1) = 2

        return model

    enc_model = build(config)

    return enc_model


def PMFModelQ_eval(config):
    """
    Channel encoder model - generating [x,y,p]
    """

    def build(config):
        def build_ff(config, fsc_flag):
            """
            FF model case
            """
            # Initializers:
            initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.1)

            # Encoder transformation with sigmoid output:
            encoder_transform = Sequential([
                keras.layers.LSTM(config.enc_hidden[0], return_sequences=True, name="LSTM_enc", stateful=True,
                                  batch_input_shape=[config.batch_size, 1, config.x_dim],
                                  dropout=config.enc_dropout, recurrent_dropout=config.enc_dropout),
                Dense(config.enc_hidden[1], activation="elu"),
                # Dense(config.enc_last_hidden, activation="elu"),
                Dense(config.x_dim, activation="sigmoid", bias_initializer=initializer)])  # WITH SIGMOID AS LAST LAYER

            p_out = list()  # list of calculated P(X|H)
            enc_out = list()  # list of X values
            channel_out = list()  # list of Y values
            channel_state = list()

            enc_in_feedback = keras.layers.Input(shape=[1, config.x_dim])  # Input layer
            enc_in_0 = enc_in_feedback

            for t in range(config.bptt):
                if t == 0:
                    p_out.append(encoder_transform(enc_in_0))  # append transformation of first input
                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append sampled x
                else:
                    enc_in_t = enc_out[t - 1]  # prepare input for time t
                    p_out.append(encoder_transform(enc_in_t))  # calculate p_t(in_t)
                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append x_t

                if fsc_flag:
                    [y_t, s_t] = channel(enc_out[t])
                    channel_out.append(y_t)
                    channel_state.append(s_t)
                else:
                    channel_out.append(channel(enc_out[t]))  # calculate y_t(x_t)

            channel_out = tf.concat(channel_out, axis=1)  # transform y list into tensor
            enc_out = tf.concat(enc_out, axis=1)  # transform x list into tensor
            p_out = tf.concat(p_out, axis=1)  # transform p list into tensor

            if fsc_flag:
                channel_state = tf.concat(channel_state, axis=1)
                encoder = Model(inputs=[enc_in_feedback], outputs=[enc_out, channel_out, p_out, channel_state])
                return encoder
            else:
                encoder = Model(inputs=[enc_in_feedback], outputs=[enc_out, channel_out, p_out])
                return encoder

        def build_fb(config, fsc_flag):
            """
            FF model case
            """
            # Initializers:
            initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.1)
            concat_layer = Lambda(lambda x: tf.concat(x, axis=1))

            encoder_transform = Sequential([
                keras.layers.LSTM(config.enc_hidden[0], return_sequences=True, name="LSTM_enc", stateful=True,
                                  batch_input_shape=[config.batch_size, 1, config.x_dim + config.y_dim],
                                  dropout=config.enc_dropout, recurrent_dropout=config.enc_dropout),
                Dense(config.enc_hidden[1], activation="elu"),
                Dense(config.enc_last_hidden, activation="elu"),
                Dense(config.x_dim, activation="sigmoid", bias_initializer=initializer)])

            p_out = list()  # list of calculated P(X|H)
            enc_out = list()  # list of X values
            channel_out = list()  # list of Y values
            channel_state = list()

            enc_in_feedback = keras.layers.Input(shape=[1, config.y_dim + config.x_dim])  # Input layer
            enc_in_0 = enc_in_feedback

            for t in range(config.bptt):
                if t == 0:
                    p_out.append(encoder_transform(enc_in_0))  # append transformation of first input
                    # for debug:
                    # p = encoder_transform(enc_in_0)
                    # p_out.append(0.5*tf.ones(shape=[config.batch_size,1,1], dtype='float64')+0*p)
                    ##
                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append sampled x
                else:
                    enc_in_t = tf.concat([enc_out[t - 1], channel_out[t-1]], axis=-1)  # prepare input for time t
                    p_out.append(encoder_transform(enc_in_t))  # calculate p_t(in_t)
                    # for debug:
                    # p = encoder_transform(enc_in_0)
                    # p_out.append(0.5 * tf.ones(shape=[config.batch_size, 1, 1], dtype='float64') + 0 * p)
                    ##

                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append x_t

                # calculate y_t(x_t)
                if fsc_flag:
                    [y_t, s_t] = channel(enc_out[t])
                    channel_out.append(y_t)
                    channel_state.append(s_t)
                else:
                    channel_out.append(channel(enc_out[t]))

            channel_out = tf.concat(channel_out, axis=1)  # transform y list into tensor
            enc_out = tf.concat(enc_out, axis=1)  # transform x list into tensor
            p_out = tf.concat(p_out, axis=1)  # transform p list into tensor
            # channel_state = tf.concat(channel_state, axis=1)

            # channel_out = concat_layer(channel_out)  # transform y list into tensor
            # enc_out = concat_layer(enc_out)  # transform x list into tensor
            # p_out = concat_layer(p_out)  # transform p list into tensor
            # channel_state = concat_layer(channel_state)
            if fsc_flag:
                channel_state = tf.concat(channel_state, axis=1)
                encoder = Model(inputs=[enc_in_feedback], outputs=[enc_out, channel_out, p_out, channel_state])
                return encoder
            else:
                encoder = Model(inputs=[enc_in_feedback], outputs=[enc_out, channel_out, p_out])
                return encoder


        fsc_flag = config.channel_name in ['ising','trapdoor','post','GE']

        if fsc_flag:
            channel = Channel_Layer_with_channel_states(config=config)  # define channel layer
        else:
            channel = Channel_Layer(config=config)  # define channel layer

        sampler_layer = SamplingLayer()
        sampler = Sequential([sampler_layer])  # sampling model of x from p


        if config.feedback == 1:
            model = build_fb(config, fsc_flag)  # build channel model with feedback, with axis(-1) = 3
        else:
            model = build_ff(config, fsc_flag)  # build channel model without feedback, with axis(-1) = 2

        return model

    enc_model = build(config)

    return enc_model





# OLD:
def EncModel(config):
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

            # Encoder transformation with sigmoid output:
            encoder_transform = Sequential([
                keras.layers.LSTM(config.enc_hidden, return_sequences=True, name="LSTM_enc", stateful=True,
                                  batch_input_shape=[config.batch_size, 1, config.x_dim + 1],
                                  dropout=config.enc_dropout, recurrent_dropout=config.enc_dropout),
                Dense(config.enc_hidden, activation="elu"),
                Dense(config.enc_last_hidden, activation="elu"),
                Dense(config.x_dim, activation="sigmoid", bias_initializer=initializer)])  # WITH SIGMOID AS LAST LAYER

            p_out = list()  # list of calculated P(X|H)
            enc_out = list()  # list of X values
            channel_out = list()  # list of Y values

            enc_in_feedback = keras.layers.Input(shape=[1, config.x_dim+1])  # Input layer
            enc_in_0 = enc_in_feedback

            for t in range(config.bptt):
                if t == 0:
                    p_out.append(encoder_transform(enc_in_0))  # append transformation of first input
                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append sampled x
                else:
                    enc_in_t = tf.concat([p_out[t-1], enc_out[t - 1]], axis=-1)  # prepare input for time t
                    p_out.append(encoder_transform(enc_in_t))  # calculate p_t(in_t)
                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append x_t

                channel_out.append(channel(enc_out[t]))  # calculate y_t(x_t)

            channel_out = tf.concat(channel_out, axis=1)  # transform y list into tensor
            enc_out = tf.concat(enc_out, axis=1)  # transform x list into tensor
            p_out = tf.concat(p_out, axis=1)  # transform p list into tensor

            encoder = Model(inputs=[enc_in_feedback], outputs=[enc_out, channel_out, p_out])
            return encoder

        def build_fb_old(config):
            """
            FF model case
            """
            # Initializers:
            initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.1)

            encoder_transform = Sequential([
                keras.layers.LSTM(config.enc_hidden, return_sequences=True, name="LSTM_enc", stateful=True,
                                  batch_input_shape=[config.batch_size, 1, config.x_dim + config.y_dim+1],
                                  dropout=config.enc_dropout, recurrent_dropout=config.enc_dropout),
                Dense(config.enc_hidden, activation="elu"),
                Dense(config.enc_last_hidden, activation="elu"),
                Dense(config.x_dim, activation="sigmoid", bias_initializer=initializer)])

            p_out = list()  # list of calculated P(X|H)
            enc_out = list()  # list of X values
            channel_out = list()  # list of Y values

            enc_in_feedback = keras.layers.Input(shape=[1, config.y_dim + config.x_dim+1])  # Input layer
            enc_in_0 = enc_in_feedback

            for t in range(config.bptt):
                if t == 0:
                    p_out.append(encoder_transform(enc_in_0))  # append transformation of first input
                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append sampled x
                else:
                    enc_in_t = tf.concat([p_out[t-1], enc_out[t - 1], channel_out[t-1]], axis=-1)  # prepare input for time t
                    p_out.append(encoder_transform(enc_in_t))  # calculate p_t(in_t)
                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append x_t

                channel_out.append(channel(enc_out[t]))  # calculate y_t(x_t)

            channel_out = tf.concat(channel_out, axis=1)  # transform y list into tensor
            enc_out = tf.concat(enc_out, axis=1)  # transform x list into tensor
            p_out = tf.concat(p_out, axis=1)  # transform p list into tensor

            encoder = Model(inputs=[enc_in_feedback], outputs=[enc_out, channel_out, p_out])
            return encoder

        def build_fb(config):
            """
            FF model case
            """
            # Initializers:
            initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.1)

            encoder_transform = Sequential([
                keras.layers.LSTM(config.enc_hidden, return_sequences=True, name="LSTM_enc", stateful=True,
                                  batch_input_shape=[config.batch_size, 1, config.x_dim + config.y_dim+1],
                                  dropout=config.enc_dropout, recurrent_dropout=config.enc_dropout),
                Dense(config.enc_hidden, activation="elu"),
                Dense(config.enc_last_hidden, activation="elu"),
                Dense(config.x_dim, activation="sigmoid", bias_initializer=initializer)])

            p_out = list()  # list of calculated P(X|H)
            enc_out = list()  # list of X values
            channel_out = list()  # list of Y values

            enc_in_feedback = keras.layers.Input(shape=[1, config.y_dim + config.x_dim+1])  # Input layer
            enc_in_0 = enc_in_feedback

            for t in range(config.bptt):
                if t == 0:
                    p_out.append(encoder_transform(enc_in_0))  # append transformation of first input
                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append sampled x
                else:
                    enc_in_t = tf.concat([p_out[t-1], enc_out[t - 1], channel_out[t-1]], axis=-1)  # prepare input for time t
                    p_out.append(encoder_transform(enc_in_t))  # calculate p_t(in_t)
                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append x_t

                channel_out.append(channel(enc_out[t]))  # calculate y_t(x_t)

            channel_out = tf.concat(channel_out, axis=1)  # transform y list into tensor
            enc_out = tf.concat(enc_out, axis=1)  # transform x list into tensor
            p_out = tf.concat(p_out, axis=1)  # transform p list into tensor

            encoder = Model(inputs=[enc_in_feedback], outputs=[enc_out, channel_out, p_out])
            return encoder

        channel = Channel_Layer(config=config)  # define channel layer
        sampler_layer = SamplingLayer()
        sampler = Sequential([sampler_layer])  # sampling model of x from p

        if config.feedback == 1:
            model = build_fb(config)  # build channel model with feedback, with axis(-1) = 3
        else:
            model = build_ff(config)  # build channel model without feedback, with axis(-1) = 2

        return model

    enc_model = build(config)

    return enc_model


def EncModel_no_p(config):
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

            # Encoder transformation with sigmoid output:
            encoder_transform = Sequential([
                keras.layers.LSTM(config.enc_hidden[0], return_sequences=True, name="LSTM_enc", stateful=True,
                                  batch_input_shape=[config.batch_size, 1, config.x_dim],
                                  dropout=config.enc_dropout, recurrent_dropout=config.enc_dropout),
                Dense(config.enc_hidden[1], activation="elu"),
                # Dense(config.enc_last_hidden, activation="elu"),
                Dense(config.x_dim, activation="sigmoid", bias_initializer=initializer)])  # WITH SIGMOID AS LAST LAYER

            p_out = list()  # list of calculated P(X|H)
            enc_out = list()  # list of X values
            channel_out = list()  # list of Y values

            enc_in_feedback = keras.layers.Input(shape=[1, config.x_dim])  # Input layer
            enc_in_0 = enc_in_feedback

            for t in range(config.bptt):
                if t == 0:
                    p_out.append(encoder_transform(enc_in_0))  # append transformation of first input
                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append sampled x
                else:
                    enc_in_t = enc_out[t - 1]  # prepare input for time t
                    p_out.append(encoder_transform(enc_in_t))  # calculate p_t(in_t)
                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append x_t

                channel_out.append(channel(enc_out[t]))  # calculate y_t(x_t)

            channel_out = tf.concat(channel_out, axis=1)  # transform y list into tensor
            enc_out = tf.concat(enc_out, axis=1)  # transform x list into tensor
            p_out = tf.concat(p_out, axis=1)  # transform p list into tensor

            encoder = Model(inputs=[enc_in_feedback], outputs=[enc_out, channel_out, p_out])
            return encoder


        def build_fb(config):
            """
            FF model case
            """
            # Initializers:
            initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.1)

            encoder_transform = Sequential([
                keras.layers.LSTM(config.enc_hidden[0], return_sequences=True, name="LSTM_enc", stateful=True,
                                  batch_input_shape=[config.batch_size, 1, config.x_dim + config.y_dim],
                                  dropout=config.enc_dropout, recurrent_dropout=config.enc_dropout),
                Dense(config.enc_hidden[1], activation="elu"),
                # Dense(config.enc_last_hidden, activation="elu"),
                Dense(config.x_dim, activation="sigmoid", bias_initializer=initializer)])

            p_out = list()  # list of calculated P(X|H)
            enc_out = list()  # list of X values
            channel_out = list()  # list of Y values

            enc_in_feedback = keras.layers.Input(shape=[1, config.y_dim + config.x_dim])  # Input layer
            enc_in_0 = enc_in_feedback

            for t in range(config.bptt):
                if t == 0:
                    p_out.append(encoder_transform(enc_in_0))  # append transformation of first input
                    # for debug:
                    # p = encoder_transform(enc_in_0)
                    # p_out.append(0.5*tf.ones(shape=[config.batch_size,1,1], dtype='float64')+0*p)
                    ##
                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append sampled x
                else:
                    enc_in_t = tf.concat([enc_out[t - 1], channel_out[t-1]], axis=-1)  # prepare input for time t
                    p_out.append(encoder_transform(enc_in_t))  # calculate p_t(in_t)
                    # for debug:
                    # p = encoder_transform(enc_in_0)
                    # p_out.append(0.5 * tf.ones(shape=[config.batch_size, 1, 1], dtype='float64') + 0 * p)
                    ##

                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append x_t

                channel_out.append(channel(enc_out[t]))  # calculate y_t(x_t)

            channel_out = tf.concat(channel_out, axis=1)  # transform y list into tensor
            enc_out = tf.concat(enc_out, axis=1)  # transform x list into tensor
            p_out = tf.concat(p_out, axis=1)  # transform p list into tensor

            encoder = Model(inputs=[enc_in_feedback], outputs=[enc_out, channel_out, p_out])
            return encoder

        channel = Channel_Layer(config=config)  # define channel layer
        sampler_layer = SamplingLayer()
        sampler = Sequential([sampler_layer])  # sampling model of x from p

        if config.feedback == 1:
            model = build_fb(config)  # build channel model with feedback, with axis(-1) = 3
        else:
            model = build_ff(config)  # build channel model without feedback, with axis(-1) = 2

        return model

    enc_model = build(config)

    return enc_model


def EncModel_no_p_new(config):
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
                keras.layers.LSTM(config.enc_hidden[0], return_sequences=True, name="LSTM_enc", stateful=True,
                                  batch_input_shape=[config.batch_size, 1, config.x_dim],
                                  dropout=config.enc_dropout, recurrent_dropout=config.enc_dropout),
                Dense(config.enc_hidden[1], activation="elu"),
                Dense(config.x_dim, activation="sigmoid", bias_initializer=initializer)])

            p_out = list()  # list of calculated P(X|H)
            enc_out = list()  # list of X values
            channel_out = list()  # list of Y values
            channel_state = list()

            enc_in_feedback = keras.layers.Input(shape=[1, config.x_dim])  # Input layer
            enc_in_0 = enc_in_feedback

            for t in range(config.bptt):
                if t == 0:
                    p_out.append(encoder_transform(enc_in_0))  # append transformation of first input
                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append sampled x
                else:
                    p_out.append(encoder_transform(enc_out[t - 1]))  # calculate p_t(in_t)

                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append x_t

                [y_t, s_t] = channel(enc_out[t])
                channel_out.append(y_t)
                channel_state.append(s_t)



            channel_out = tf.concat(channel_out, axis=1)  # transform y list into tensor
            enc_out = tf.concat(enc_out, axis=1)  # transform x list into tensor
            p_out = tf.concat(p_out, axis=1)  # transform p list into tensor
            channel_state = tf.concat(channel_state, axis=1)
            encoder = Model(inputs=[enc_in_feedback], outputs=[enc_out, channel_out, p_out, channel_state])

            return encoder

        def build_fb(config):
            """
            FF model case
            """
            # Initializers:
            initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.1)

            encoder_transform = Sequential([
                keras.layers.LSTM(config.enc_hidden[0], return_sequences=True, name="LSTM_enc", stateful=True,
                                  batch_input_shape=[config.batch_size, 1, config.x_dim + config.y_dim],
                                  dropout=config.enc_dropout, recurrent_dropout=config.enc_dropout),
                Dense(config.enc_hidden[1], activation="elu"),
                # Dense(config.enc_last_hidden, activation="elu"),
                Dense(config.x_dim, activation="sigmoid", bias_initializer=initializer)])

            p_out = list()  # list of calculated P(X|H)
            enc_out = list()  # list of X values
            channel_out = list()  # list of Y values
            channel_state = list()

            enc_in_feedback = keras.layers.Input(shape=[1, config.y_dim + config.x_dim])  # Input layer
            enc_in_0 = enc_in_feedback

            for t in range(config.bptt):
                if t == 0:
                    p_out.append(encoder_transform(enc_in_0))  # append transformation of first input
                    # for debug:
                    # p = encoder_transform(enc_in_0)
                    # p_out.append(0.5*tf.ones(shape=[config.batch_size,1,1], dtype='float64')+0*p)
                    ##
                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append sampled x
                else:
                    enc_in_t = tf.concat([enc_out[t - 1], channel_out[t-1]], axis=-1)  # prepare input for time t
                    p_out.append(encoder_transform(enc_in_t))  # calculate p_t(in_t)
                    # for debug:
                    # p = encoder_transform(enc_in_0)
                    # p_out.append(0.5 * tf.ones(shape=[config.batch_size, 1, 1], dtype='float64') + 0 * p)
                    ##

                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append x_t

                [y_t, s_t] = channel(enc_out[t])
                channel_out.append(y_t)
                channel_state.append(s_t)



            channel_out = tf.concat(channel_out, axis=1)  # transform y list into tensor
            enc_out = tf.concat(enc_out, axis=1)  # transform x list into tensor
            p_out = tf.concat(p_out, axis=1)  # transform p list into tensor
            channel_state = tf.concat(channel_state, axis=1)
            encoder = Model(inputs=[enc_in_feedback], outputs=[enc_out, channel_out, p_out, channel_state])

            return encoder

        channel = Channel_Layer_no_p_new(config=config)  # define channel layer
        sampler_layer = SamplingLayer()
        sampler = Sequential([sampler_layer])  # sampling model of x from p

        if config.feedback == 1:
            model = build_fb(config)  # build channel model with feedback, with axis(-1) = 3
        else:
            model = build_ff(config)  # build channel model without feedback, with axis(-1) = 2

        return model

    enc_model = build(config)

    return enc_model


# Encoder model that yields the channel states as well for an FSC
def EncModel_PDINE(config):
    """
    Channel encoder model - generating [x,y,p]
    """

    def build(config):
        def build_ff(config, fsc_flag):
            """
            FF model case
            """
            # Initializers:
            initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.1)

            # Encoder transformation with sigmoid output:
            encoder_transform = Sequential([
                keras.layers.LSTM(config.enc_hidden[0], return_sequences=True, name="LSTM_enc", stateful=True,
                                  batch_input_shape=[config.batch_size, 1, config.x_dim],
                                  dropout=config.enc_dropout, recurrent_dropout=config.enc_dropout),
                Dense(config.enc_hidden[1], activation="elu"),
                # Dense(config.enc_last_hidden, activation="elu"),
                Dense(config.x_dim, activation="sigmoid", bias_initializer=initializer)])  # WITH SIGMOID AS LAST LAYER

            p_out = list()  # list of calculated P(X|H)
            enc_out = list()  # list of X values
            channel_out = list()  # list of Y values
            channel_state = list()

            enc_in_feedback = keras.layers.Input(shape=[1, config.x_dim])  # Input layer
            enc_in_0 = enc_in_feedback

            for t in range(config.bptt):
                if t == 0:
                    p_out.append(encoder_transform(enc_in_0))  # append transformation of first input
                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append sampled x
                else:
                    enc_in_t = enc_out[t - 1]  # prepare input for time t
                    p_out.append(encoder_transform(enc_in_t))  # calculate p_t(in_t)
                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append x_t

                if fsc_flag:
                    [y_t, s_t] = channel(enc_out[t])
                    channel_out.append(y_t)
                    channel_state.append(s_t)
                else:
                    channel_out.append(channel(enc_out[t]))  # calculate y_t(x_t)

            channel_out = tf.concat(channel_out, axis=1)  # transform y list into tensor
            enc_out = tf.concat(enc_out, axis=1)  # transform x list into tensor
            p_out = tf.concat(p_out, axis=1)  # transform p list into tensor

            if fsc_flag:
                channel_state = tf.concat(channel_state, axis=1)
                encoder = Model(inputs=[enc_in_feedback], outputs=[enc_out, channel_out, p_out, channel_state])
                return encoder
            else:
                encoder = Model(inputs=[enc_in_feedback], outputs=[enc_out, channel_out, p_out])
                return encoder

        def build_fb(config, fsc_flag):
            """
            FF model case
            """
            # Initializers:
            initializer = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.1)
            concat_layer = Lambda(lambda x: tf.concat(x, axis=1))

            encoder_transform = Sequential([
                keras.layers.LSTM(config.enc_hidden[0], return_sequences=True, name="LSTM_enc", stateful=True,
                                  batch_input_shape=[config.batch_size, 1, config.x_dim + config.y_dim],
                                  dropout=config.enc_dropout, recurrent_dropout=config.enc_dropout),
                Dense(config.enc_hidden[1], activation="elu"),
                Dense(config.enc_last_hidden, activation="elu"),
                Dense(config.x_dim, activation="sigmoid", bias_initializer=initializer)])

            p_out = list()  # list of calculated P(X|H)
            enc_out = list()  # list of X values
            channel_out = list()  # list of Y values
            channel_state = list()

            enc_in_feedback = keras.layers.Input(shape=[1, config.y_dim + config.x_dim])  # Input layer
            enc_in_0 = enc_in_feedback

            for t in range(config.bptt):
                if t == 0:
                    p_out.append(encoder_transform(enc_in_0))  # append transformation of first input
                    # for debug:
                    # p = encoder_transform(enc_in_0)
                    # p_out.append(0.5*tf.ones(shape=[config.batch_size,1,1], dtype='float64')+0*p)
                    ##
                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append sampled x
                else:
                    enc_in_t = tf.concat([enc_out[t - 1], channel_out[t-1]], axis=-1)  # prepare input for time t
                    p_out.append(encoder_transform(enc_in_t))  # calculate p_t(in_t)
                    # for debug:
                    # p = encoder_transform(enc_in_0)
                    # p_out.append(0.5 * tf.ones(shape=[config.batch_size, 1, 1], dtype='float64') + 0 * p)
                    ##

                    x = sampler(p_out[t])  # sample x using the estimated conditional pmf
                    enc_out.append(x)  # append x_t

                # calculate y_t(x_t)
                if fsc_flag:
                    [y_t, s_t] = channel(enc_out[t])
                    channel_out.append(y_t)
                    channel_state.append(s_t)
                else:
                    channel_out.append(channel(enc_out[t]))

            channel_out = tf.concat(channel_out, axis=1)  # transform y list into tensor
            enc_out = tf.concat(enc_out, axis=1)  # transform x list into tensor
            p_out = tf.concat(p_out, axis=1)  # transform p list into tensor
            # channel_state = tf.concat(channel_state, axis=1)

            # channel_out = concat_layer(channel_out)  # transform y list into tensor
            # enc_out = concat_layer(enc_out)  # transform x list into tensor
            # p_out = concat_layer(p_out)  # transform p list into tensor
            # channel_state = concat_layer(channel_state)
            if fsc_flag:
                channel_state = tf.concat(channel_state, axis=1)
                encoder = Model(inputs=[enc_in_feedback], outputs=[enc_out, channel_out, p_out, channel_state])
                return encoder
            else:
                encoder = Model(inputs=[enc_in_feedback], outputs=[enc_out, channel_out, p_out])
                return encoder


        fsc_flag = config.channel_name in ['ising','trapdoor','post','GE']

        if fsc_flag:
            channel = Channel_Layer_with_channel_states(config=config)  # define channel layer
        else:
            channel = Channel_Layer(config=config)  # define channel layer

        sampler_layer = SamplingLayer()
        sampler = Sequential([sampler_layer])  # sampling model of x from p


        if config.feedback == 1:
            model = build_fb(config, fsc_flag)  # build channel model with feedback, with axis(-1) = 3
        else:
            model = build_ff(config, fsc_flag)  # build channel model without feedback, with axis(-1) = 2

        return model

    enc_model = build(config)

    return enc_model




##########################
#  Channel Layers:
##########################
class Channel_Layer(tf.keras.layers.Layer):  # creates channel outputs from chanel inputs
    def __init__(self, config):
        """
        Unifying class for all channel operations options
        :param config: configuration list of parameters values
        """
        super(Channel_Layer, self).__init__()
        self.channel = config.channel_name
        self.batch_size = config.batch_size
        self.bptt = config.bptt
        self.state = tf.Variable(tf.zeros(shape=[self.batch_size, 1, config.x_dim], dtype='float64'), trainable=False)
        self.y = tf.Variable(tf.zeros(shape=[self.batch_size, 1, config.x_dim], dtype='float64'), trainable=False)
        self.eta_post = config.eta_post

        if self.channel == "GE":
            self.g = config.GE_g
            self.b = config.GE_b
            self.pb = config.GE_p_b
            self.pg = config.GE_p_g
            self.calculate_GE_logits()
            self.initial = True
            self.calculate_GE_state()  # calculate initial states batch
        else:
            self.p_bsc = config.p_bsc
            self.p_z = config.p_z
            self.p_bec = config.p_bec
            self.p_ising = config.p_ising
            self.p_post = config.p_post
            self.p_trapdoor = config.p_trapdoor

            if self.channel == "clean":
                self.channel_fn = clean_channel
            elif self.channel == "BSC":
                self.channel_fn = bsc_channel
            elif self.channel == "z_ch":
                self.channel_fn = z_channel
            elif self.channel == "s_ch":
                self.channel_fn = s_channel
            elif self.channel == "BEC":
                self.channel_fn = bec_channel

    def call(self, x, mask=None, training=False):
        """
        *** Currently implemnted for 1D samples ***
        """
        if self.channel == "GE":
            return self.call_GE(x)
        elif self.channel == "ising":
            return self.call_Ising(x)
        elif self.channel == "post":
            return self.call_post(x)
        elif self.channel == "noisy_post":
            return self.call_noisy_post(x)
        elif self.channel == "trapdoor":
            return self.call_Trapdoor(x)
        else:
            return self.call_DMC(x)


    # GE methods:
    def calculate_GE_logits(self):
        # Calculation of all logits required for the GE channel
        # States Markov Chain:
        b_t = self.b * tf.ones(shape=[self.batch_size, 1])
        b_bar_t = tf.ones_like(b_t) - b_t
        self.logits_b = tf.math.log(tf.concat([b_bar_t, b_t], axis=1))  # define logits

        g_t = self.g * tf.ones(shape=[self.batch_size, 1])
        g_bar_t = tf.ones_like(g_t) - g_t
        self.logits_g = tf.math.log(tf.concat([g_bar_t, g_t], axis=1))  # define logits

        pb_t = self.pb * tf.ones(shape=[self.batch_size, 1])
        pb_bar_t = tf.ones_like(pb_t) - pb_t
        self.logits_pb = tf.math.log(tf.concat([pb_bar_t, pb_t], axis=1))  # define logits

        pg_t = self.pg * tf.ones(shape=[self.batch_size, 1])
        pg_bar_t = tf.ones_like(pg_t) - pg_t
        self.logits_pg = tf.math.log(tf.concat([pg_bar_t, pg_t], axis=1))  # define logits

    def calculate_GE_state(self):
        """
        calculate markov chain state
        :param initial:
        :return:
        """
        if self.initial:  # first state od sequence - drawn from stationary distribution
            g_0 = (self.g / (self.g + self.b)) * tf.ones(shape=[self.batch_size, 1])  # stationary distribution
            b_0 = tf.ones_like(g_0) - g_0
            logits = tf.math.log(tf.concat([b_0, g_0], axis=1))  # define logits
            state = tf.cast(tf.random.categorical(logits=logits, num_samples=1), 'float64')  # sample state
            self.state.assign(tf.expand_dims(state, axis=1))  # add bptt dimension
            self.initial = False
        else:  # at an intermediate stage of the markov chain
            # obtain s_b ~ ber(g):
            z_b = tf.expand_dims(tf.cast(tf.random.categorical(logits=self.logits_g, num_samples=1), 'float64'), axis=1)
            s_b = tf.math.floormod(self.state + z_b, 2)

            # obtain s_g ~ ber(b):
            z_g = tf.expand_dims(tf.cast(tf.random.categorical(logits=self.logits_b, num_samples=1), 'float64'), axis=1)
            s_g = tf.math.floormod(self.state + z_g, 2)

            # obtain new state
            self.state.assign(tf.where(tf.equal(self.state, 1), s_g, s_b))

    def call_GE(self, x):
        """
        operation of the GE channel:
        y_i = x_i+z_i(mod2) with z_i~ber(P(s_i)), s_i is a stationary Markov chain.
        """
        z_b = tf.expand_dims(tf.cast(tf.random.categorical(logits=self.logits_pb, num_samples=1), 'float64'), axis=1)
        y_b = tf.math.floormod(x + z_b, 2)  # input transition for state b
        z_g = tf.expand_dims(tf.cast(tf.random.categorical(logits=self.logits_pg, num_samples=1), 'float64'), axis=1)
        y_g = tf.math.floormod(x + z_g, 2)  # input transition for state g

        # obtain GE channel output:
        y = tf.where(tf.equal(self.state, 1), y_g, y_b)
        self.calculate_GE_state()
        return y

    # Ising Method:
    def call_Ising(self, x):
        """
        operation of the Ising channel
        s_i = x_{i-1}
        -> if x_i = s_i than y_i = x-i
        -> else: y_i~Ber(0.5)
        """
        B = tf.shape(x)[0]  # acquire catch size
        p_t = self.p_ising * tf.ones(shape=[B, 1])  # create logits
        p_bar_t = 1 - p_t
        logits = tf.math.log(
            tf.concat([p_bar_t, p_t], axis=1))  # create transition probability logits of shape ([B,1])
        y_ber = tf.cast(tf.random.categorical(logits=logits, num_samples=1), 'float64')
        y_ber = tf.expand_dims(y_ber, axis=-1)
        # y = tf.where(tf.equal(x, self.state), x, y_ber)
        y = tf.where(tf.equal(y_ber, 1), x, self.state)
        self.state.assign(x)

        return y

    # POST Method:
    def call_post(self, x):
        """
        operation of the post channel
        s_i = y_{i-1}
        -> if x_i = s_i than y_i = x-i
        -> else: y_i~Ber(0.5)
        """
        B = tf.shape(x)[0]  # acquire catch size
        p_t = self.p_post * tf.ones(shape=[B, 1])  # create logits
        p_bar_t = tf.ones_like(p_t) - p_t
        logits = tf.math.log(
            tf.concat([p_bar_t, p_t], axis=1))  # create transition probability logits of shape ([B*T,1])
        y_ber = tf.cast(tf.random.categorical(logits=logits, num_samples=1), 'float64')
        y_ber = tf.expand_dims(y_ber, axis=-1)
        y_o = tf.math.floormod(x + y_ber, 2)
        # y = tf.where(tf.equal(x, self.state), x, y_ber)
        y = tf.where(tf.equal(x, self.state), x, y_o)
        self.state.assign(y)

        return y

    def call_Trapdoor(self, x):
        """
        operation of the Ising channel
        s_i = x_{i-1}
        -> if x_i = s_i than y_i = x-i
        -> else: y_i~Ber(0.5)
        """
        B = tf.shape(x)[0]  # acquire catch size
        p_t = self.p_trapdoor * tf.ones(shape=[B, 1])  # create logits
        p_bar_t = 1 - p_t
        logits = tf.math.log(
            tf.concat([p_bar_t, p_t], axis=1))  # create transition probability logits of shape ([B,1])
        y_ber = tf.cast(tf.random.categorical(logits=logits, num_samples=1), 'float64')
        y_ber = tf.expand_dims(y_ber, axis=-1)
        y = tf.where(tf.equal(x, self.state), x, y_ber)  #right
        # self.y.assign(y)
        s = tf.math.floormod(self.state + x + y , 2)
        self.state.assign(s)

        return y

    def call_noisy_post(self,x):
        B = tf.shape(x)[0]  # acquire catch size
        p_t = self.p_post * tf.ones(shape=[B, 1])  # create logits
        p_bar_t = tf.ones_like(p_t) - p_t
        logits = tf.math.log(
            tf.concat([p_bar_t, p_t], axis=1))  # create transition probability logits of shape ([B*T,1])
        y_ber = tf.cast(tf.random.categorical(logits=logits, num_samples=1), 'float64')
        y_ber = tf.expand_dims(y_ber, axis=-1)
        y_o = tf.math.floormod(x + y_ber, 2)
        # y = tf.where(tf.equal(x, self.state), x, y_ber)
        y = tf.where(tf.equal(x, self.state), x, y_o)

        p_s = self.eta_post * tf.ones(shape=[B, 1])  # create logits
        p_bar_s = tf.ones_like(p_s) - p_s
        logits_eta = tf.math.log(
            tf.concat([p_bar_s, p_s], axis=1))
        y_eta = tf.cast(tf.random.categorical(logits=logits_eta, num_samples=1), 'float64')
        y_eta = tf.expand_dims(y_eta, axis=-1)
        s = tf.where(tf.equal(y, 0), y, y_eta)
        self.state.assign(s)

        return y

    # DMC methods:
    def call_DMC(self, x):
        """
        Operation of a DMC function, implemented with a function from channel_methods
        :param x:
        :return:
        """
        return self.channel_fn(x=x, p_bsc=self.p_bsc, p_z=self.p_z, p_bec=self.p_bec, batch_size=self.batch_size)

    def reset_states(self):
        self.state.assign(tf.zeros(shape=[self.batch_size, 1, 1], dtype='float64'))


class Channel_Layer_with_channel_states(tf.keras.layers.Layer):
    """
    Unifying class for all FSC - to collect channel state as well
    :param config: configuration list of parameters values
    """
    def __init__(self, config):
        super(Channel_Layer_with_channel_states, self).__init__()
        self.channel = config.channel_name
        self.batch_size = config.batch_size
        self.bptt = config.bptt
        self.state = tf.Variable(tf.zeros(shape=[self.batch_size, 1, config.x_dim], dtype='float64'), trainable=False)
        self.y = tf.Variable(tf.zeros(shape=[self.batch_size, 1, config.x_dim], dtype='float64'), trainable=False)

        if config.channel_name not in ['ising','trapdoor','post','GE']:
            raise Exception("Channel name does not fit the chosen encoder model")

        if self.channel == "GE":
            self.g = config.GE_g
            self.b = config.GE_b
            self.pb = config.GE_p_b
            self.pg = config.GE_p_g
            self.calculate_GE_logits()
            self.initial = True
            self.calculate_GE_state()  # calculate initial states batch
        else:
            self.p_ising = config.p_ising
            self.p_post = config.p_post
            self.p_trapdoor = config.p_trapdoor

    def call(self, x, mask=None, training=False):
        """
        *** Currently implemnted for 1D samples ***
        """
        if self.channel == "GE":
            return self.call_GE(x)
        elif self.channel == "ising":
            return self.call_Ising(x)
        elif self.channel == "post":
            return self.call_post(x)
        elif self.channel == "noisy_post":
            return self.call_noisy_post(x)
        elif self.channel == "trapdoor":
            return self.call_Trapdoor(x)
        else:
            raise Exception("Channel name does not fit possible call methods")

    # GE methods:
    def calculate_GE_logits(self):
        # Calculation of all logits required for the GE channel
        # States Markov Chain:
        b_t = self.b * tf.ones(shape=[self.batch_size, 1])
        b_bar_t = tf.ones_like(b_t) - b_t
        self.logits_b = tf.math.log(tf.concat([b_bar_t, b_t], axis=1))  # define logits

        g_t = self.g * tf.ones(shape=[self.batch_size, 1])
        g_bar_t = tf.ones_like(g_t) - g_t
        self.logits_g = tf.math.log(tf.concat([g_bar_t, g_t], axis=1))  # define logits

        pb_t = self.pb * tf.ones(shape=[self.batch_size, 1])
        pb_bar_t = tf.ones_like(pb_t) - pb_t
        self.logits_pb = tf.math.log(tf.concat([pb_bar_t, pb_t], axis=1))  # define logits

        pg_t = self.pg * tf.ones(shape=[self.batch_size, 1])
        pg_bar_t = tf.ones_like(pg_t) - pg_t
        self.logits_pg = tf.math.log(tf.concat([pg_bar_t, pg_t], axis=1))  # define logits

    def calculate_GE_state(self):
        """
        calculate markov chain state
        :param initial:
        :return:
        """
        if self.initial:  # first state od sequence - drawn from stationary distribution
            g_0 = (self.g / (self.g + self.b)) * tf.ones(shape=[self.batch_size, 1])  # stationary distribution
            b_0 = tf.ones_like(g_0) - g_0
            logits = tf.math.log(tf.concat([b_0, g_0], axis=1))  # define logits
            state = tf.cast(tf.random.categorical(logits=logits, num_samples=1), 'float64')  # sample state
            self.state.assign(tf.expand_dims(state, axis=1))  # add bptt dimension
            self.initial = False
        else:  # at an intermediate stage of the markov chain
            # obtain s_b ~ ber(g):
            z_b = tf.expand_dims(tf.cast(tf.random.categorical(logits=self.logits_g, num_samples=1), 'float64'), axis=1)
            s_b = tf.math.floormod(self.state + z_b, 2)

            # obtain s_g ~ ber(b):
            z_g = tf.expand_dims(tf.cast(tf.random.categorical(logits=self.logits_b, num_samples=1), 'float64'), axis=1)
            s_g = tf.math.floormod(self.state + z_g, 2)

            # obtain new state
            self.state.assign(tf.where(tf.equal(self.state, 1), s_g, s_b))

    def call_GE(self, x):
        """
        operation of the GE channel:
        y_i = x_i+z_i(mod2) with z_i~ber(P(s_i)), s_i is a stationary Markov chain.
        """
        z_b = tf.expand_dims(tf.cast(tf.random.categorical(logits=self.logits_pb, num_samples=1), 'float64'), axis=1)
        y_b = tf.math.floormod(x + z_b, 2)  # input transition for state b
        z_g = tf.expand_dims(tf.cast(tf.random.categorical(logits=self.logits_pg, num_samples=1), 'float64'), axis=1)
        y_g = tf.math.floormod(x + z_g, 2)  # input transition for state g

        # obtain GE channel output:
        y = tf.where(tf.equal(self.state, 1), y_g, y_b)
        self.calculate_GE_state()
        return [y,self.state]

    # Ising Method:
    def call_Ising(self, x):
        """
        operation of the Ising channel
        s_i = x_{i-1}
        -> if x_i = s_i than y_i = x-i
        -> else: y_i~Ber(0.5)
        """
        B = tf.shape(x)[0]  # acquire catch size
        p_t = self.p_ising * tf.ones(shape=[B, 1])  # create logits
        p_bar_t = 1 - p_t
        logits = tf.math.log(
            tf.concat([p_bar_t, p_t], axis=1))  # create transition probability logits of shape ([B,1])
        y_ber = tf.cast(tf.random.categorical(logits=logits, num_samples=1), 'float64')
        y_ber = tf.expand_dims(y_ber, axis=-1)
        # y = tf.where(tf.equal(x, self.state), x, y_ber)
        y = tf.where(tf.equal(y_ber, 1), x, self.state)
        self.state.assign(x)

        return [y,self.state]

    # POST Method:
    def call_post(self, x):
        """
        operation of the post channel
        s_i = y_{i-1}
        -> if x_i = s_i than y_i = x-i
        -> else: y_i~Ber(0.5)
        """
        B = tf.shape(x)[0]  # acquire catch size
        p_t = self.p_post * tf.ones(shape=[B, 1])  # create logits
        p_bar_t = tf.ones_like(p_t) - p_t
        logits = tf.math.log(
            tf.concat([p_bar_t, p_t], axis=1))  # create transition probability logits of shape ([B*T,1])
        y_ber = tf.cast(tf.random.categorical(logits=logits, num_samples=1), 'float64')
        y_ber = tf.expand_dims(y_ber, axis=-1)
        y_o = tf.math.floormod(x + y_ber, 2)
        # y = tf.where(tf.equal(x, self.state), x, y_ber)
        y = tf.where(tf.equal(x, self.state), x, y_o)
        self.state.assign(y)

        return [y,self.state]

    # Trapdoor Method:
    def call_Trapdoor(self, x):
        """
        operation of the Ising channel
        s_i = x_{i-1}
        -> if x_i = s_i than y_i = x-i
        -> else: y_i~Ber(0.5)
        """
        B = tf.shape(x)[0]  # acquire catch size
        p_t = self.p_trapdoor * tf.ones(shape=[B, 1])  # create logits
        p_bar_t = 1 - p_t
        logits = tf.math.log(
            tf.concat([p_bar_t, p_t], axis=1))  # create transition probability logits of shape ([B,1])
        y_ber = tf.cast(tf.random.categorical(logits=logits, num_samples=1), 'float64')
        y_ber = tf.expand_dims(y_ber, axis=-1)
        y = tf.where(tf.equal(x, self.state), x, y_ber)  #right
        # self.y.assign(y)
        s = tf.math.floormod(self.state + x + y , 2)
        self.state.assign(s)

        return [y,self.state]

    def call_noisy_post(self,x):
        B = tf.shape(x)[0]  # acquire catch size
        p_t = self.p_post * tf.ones(shape=[B, 1])  # create logits
        p_bar_t = tf.ones_like(p_t) - p_t
        logits = tf.math.log(
            tf.concat([p_bar_t, p_t], axis=1))  # create transition probability logits of shape ([B*T,1])
        y_ber = tf.cast(tf.random.categorical(logits=logits, num_samples=1), 'float64')
        y_ber = tf.expand_dims(y_ber, axis=-1)
        y_o = tf.math.floormod(x + y_ber, 2)
        # y = tf.where(tf.equal(x, self.state), x, y_ber)
        y = tf.where(tf.equal(x, self.state), x, y_o)

        p_s = (1-self.eta_post) * tf.ones(shape=[B, 1])  # create logits
        p_bar_s = tf.ones_like(p_s) - p_s
        logits_eta = tf.math.log(
            tf.concat([p_bar_s, p_s], axis=1))
        y_eta = tf.cast(tf.random.categorical(logits=logits_eta, num_samples=1), 'float64')
        y_eta = tf.expand_dims(y_eta, axis=-1)
        s = tf.where(tf.equal(y, 0), y, y_eta)
        self.state.assign(s)

        return [y,self.state]

    def reset_states(self):
        if self.channel == "GE":
            self.initial = True
        else:
            self.state.assign(tf.zeros(shape=[self.batch_size, 1, 1], dtype='float64'))


class Channel_Layer_no_p_new(tf.keras.layers.Layer):  # creates channel outputs from chanel inputs
    def __init__(self, config):
        """
        Unifying class for all channel operations options
        :param config: configuration list of parameters values
        """
        super(Channel_Layer_no_p_new, self).__init__()
        self.channel = config.channel_name
        self.batch_size = config.batch_size
        self.bptt = config.bptt
        self.state = tf.Variable(tf.zeros(shape=[self.batch_size, 1, config.x_dim], dtype='float64'), trainable=False)
        self.y = tf.Variable(tf.zeros(shape=[self.batch_size, 1, config.x_dim], dtype='float64'), trainable=False)
        self.eta_post = config.eta_post

        if self.channel == "GE":
            self.g = config.GE_g
            self.b = config.GE_b
            self.pb = config.GE_p_b
            self.pg = config.GE_p_g
            self.calculate_GE_logits()
            self.initial = True
            self.calculate_GE_state()  # calculate initial states batch
        else:
            self.p_bsc = config.p_bsc
            self.p_z = config.p_z
            self.p_bec = config.p_bec
            self.p_ising = config.p_ising
            self.p_post = config.p_post
            self.p_trapdoor = config.p_trapdoor

            if self.channel == "clean":
                self.channel_fn = clean_channel
            elif self.channel == "BSC":
                self.channel_fn = bsc_channel
            elif self.channel == "z_ch":
                self.channel_fn = z_channel
            elif self.channel == "s_ch":
                self.channel_fn = s_channel
            elif self.channel == "BEC":
                self.channel_fn = bec_channel

    def call(self, x, mask=None, training=False):
        """
        *** Currently implemnted for 1D samples ***
        """
        if self.channel == "GE":
            return self.call_GE(x)
        elif self.channel == "ising":
            return self.call_Ising(x)
        elif self.channel == "post":
            return self.call_post(x)
        elif self.channel == "noisy_post":
            return self.call_noisy_post(x)
        elif self.channel == "trapdoor":
            return self.call_Trapdoor(x)
        else:
            return self.call_DMC(x)


    # GE methods:
    def calculate_GE_logits(self):
        # Calculation of all logits required for the GE channel
        # States Markov Chain:
        b_t = self.b * tf.ones(shape=[self.batch_size, 1])
        b_bar_t = tf.ones_like(b_t) - b_t
        self.logits_b = tf.math.log(tf.concat([b_bar_t, b_t], axis=1))  # define logits

        g_t = self.g * tf.ones(shape=[self.batch_size, 1])
        g_bar_t = tf.ones_like(g_t) - g_t
        self.logits_g = tf.math.log(tf.concat([g_bar_t, g_t], axis=1))  # define logits

        pb_t = self.pb * tf.ones(shape=[self.batch_size, 1])
        pb_bar_t = tf.ones_like(pb_t) - pb_t
        self.logits_pb = tf.math.log(tf.concat([pb_bar_t, pb_t], axis=1))  # define logits

        pg_t = self.pg * tf.ones(shape=[self.batch_size, 1])
        pg_bar_t = tf.ones_like(pg_t) - pg_t
        self.logits_pg = tf.math.log(tf.concat([pg_bar_t, pg_t], axis=1))  # define logits

    def calculate_GE_state(self):
        """
        calculate markov chain state
        :param initial:
        :return:
        """
        if self.initial:  # first state od sequence - drawn from stationary distribution
            g_0 = (self.g / (self.g + self.b)) * tf.ones(shape=[self.batch_size, 1])  # stationary distribution
            b_0 = tf.ones_like(g_0) - g_0
            logits = tf.math.log(tf.concat([b_0, g_0], axis=1))  # define logits
            state = tf.cast(tf.random.categorical(logits=logits, num_samples=1), 'float64')  # sample state
            self.state.assign(tf.expand_dims(state, axis=1))  # add bptt dimension
            self.initial = False
        else:  # at an intermediate stage of the markov chain
            # obtain s_b ~ ber(g):
            z_b = tf.expand_dims(tf.cast(tf.random.categorical(logits=self.logits_g, num_samples=1), 'float64'), axis=1)
            s_b = tf.math.floormod(self.state + z_b, 2)

            # obtain s_g ~ ber(b):
            z_g = tf.expand_dims(tf.cast(tf.random.categorical(logits=self.logits_b, num_samples=1), 'float64'), axis=1)
            s_g = tf.math.floormod(self.state + z_g, 2)

            # obtain new state
            self.state.assign(tf.where(tf.equal(self.state, 1), s_g, s_b))

    def call_GE(self, x):
        """
        operation of the GE channel:
        y_i = x_i+z_i(mod2) with z_i~ber(P(s_i)), s_i is a stationary Markov chain.
        """
        z_b = tf.expand_dims(tf.cast(tf.random.categorical(logits=self.logits_pb, num_samples=1), 'float64'), axis=1)
        y_b = tf.math.floormod(x + z_b, 2)  # input transition for state b
        z_g = tf.expand_dims(tf.cast(tf.random.categorical(logits=self.logits_pg, num_samples=1), 'float64'), axis=1)
        y_g = tf.math.floormod(x + z_g, 2)  # input transition for state g

        # obtain GE channel output:
        y = tf.where(tf.equal(self.state, 1), y_g, y_b)
        self.calculate_GE_state()
        return [y, self.state]

    # Ising Method:
    def call_Ising(self, x):
        """
        operation of the Ising channel
        s_i = x_{i-1}
        -> if x_i = s_i than y_i = x-i
        -> else: y_i~Ber(0.5)
        """
        B = tf.shape(x)[0]  # acquire catch size
        p_t = self.p_ising * tf.ones(shape=[B, 1])  # create logits
        p_bar_t = 1 - p_t
        logits = tf.math.log(
            tf.concat([p_bar_t, p_t], axis=1))  # create transition probability logits of shape ([B,1])
        y_ber = tf.cast(tf.random.categorical(logits=logits, num_samples=1), 'float64')
        y_ber = tf.expand_dims(y_ber, axis=-1)
        # y = tf.where(tf.equal(x, self.state), x, y_ber)
        y = tf.where(tf.equal(y_ber, 1), x, self.state)
        self.state.assign(x)

        return [y, x]

    # POST Method:
    def call_post(self, x):
        """
        operation of the post channel
        s_i = y_{i-1}
        -> if x_i = s_i than y_i = x-i
        -> else: y_i~Ber(0.5)
        """
        B = tf.shape(x)[0]  # acquire catch size
        p_t = self.p_post * tf.ones(shape=[B, 1])  # create logits
        p_bar_t = tf.ones_like(p_t) - p_t
        logits = tf.math.log(
            tf.concat([p_bar_t, p_t], axis=1))  # create transition probability logits of shape ([B*T,1])
        y_ber = tf.cast(tf.random.categorical(logits=logits, num_samples=1), 'float64')
        y_ber = tf.expand_dims(y_ber, axis=-1)
        y_o = tf.math.floormod(x + y_ber, 2)
        y = tf.where(tf.equal(x, self.state), x, y_o)
        self.state.assign(y)

        return [y, self.state]

    def call_Trapdoor(self, x):
        """
        operation of the Ising channel
        s_i = x_{i-1}
        -> if x_i = s_i than y_i = x-i
        -> else: y_i~Ber(0.5)
        """
        B = tf.shape(x)[0]  # acquire catch size
        p_t = self.p_trapdoor * tf.ones(shape=[B, 1])  # create logits
        p_bar_t = 1 - p_t
        logits = tf.math.log(
            tf.concat([p_bar_t, p_t], axis=1))  # create transition probability logits of shape ([B,1])
        y_ber = tf.cast(tf.random.categorical(logits=logits, num_samples=1), 'float64')
        y_ber = tf.expand_dims(y_ber, axis=-1)
        y = tf.where(tf.equal(x, self.state), x, y_ber)  #right
        # self.y.assign(y)
        s = tf.math.floormod(self.state + x + y , 2)
        self.state.assign(s)

        return [y,s]

    def call_noisy_post(self,x):
        B = tf.shape(x)[0]  # acquire catch size
        p_t = self.p_post * tf.ones(shape=[B, 1])  # create logits
        p_bar_t = tf.ones_like(p_t) - p_t
        logits = tf.math.log(
            tf.concat([p_bar_t, p_t], axis=1))  # create transition probability logits of shape ([B*T,1])
        y_ber = tf.cast(tf.random.categorical(logits=logits, num_samples=1), 'float64')
        y_ber = tf.expand_dims(y_ber, axis=-1)
        y_o = tf.math.floormod(x + y_ber, 2)
        # y = tf.where(tf.equal(x, self.state), x, y_ber)
        y = tf.where(tf.equal(x, self.state), x, y_o)

        p_s = self.eta_post * tf.ones(shape=[B, 1])  # create logits
        p_bar_s = tf.ones_like(p_s) - p_s
        logits_eta = tf.math.log(
            tf.concat([p_bar_s, p_s], axis=1))
        y_eta = tf.cast(tf.random.categorical(logits=logits_eta, num_samples=1), 'float64')
        y_eta = tf.expand_dims(y_eta, axis=-1)
        s = tf.where(tf.equal(y, 0), y, y_eta)
        self.state.assign(s)

        return [y, s]

    # DMC methods:
    def call_DMC(self, x):
        """
        Operation of a DMC function, implemented with a function from channel_methods
        :param x:
        :return:
        """
        return [self.channel_fn(x=x, p_bsc=self.p_bsc, p_z=self.p_z, p_bec=self.p_bec, batch_size=self.batch_size), tf.zeros_like(x)]

    def reset_states(self):
        self.state.assign(tf.zeros(shape=[self.batch_size, 1, 1], dtype='float64'))
