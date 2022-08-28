import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Embedding, BatchNormalization, ELU, \
    Reshape, Softmax, Lambda, Concatenate, Input, Dropout, LSTM
from models.layers import LSTMNew, ContrastiveNoiseLayer
tf.keras.backend.set_floatx('float64')


def DVModel(config, bptt, batch_size, dims=(1, 1), contrastive_duplicates=None):
    def build(name, input_shape, split_input=None):
        """
        Building the DV estimators models - The amount of T outputs is equal to the amount of samples inserted
        output shape: [batch_size, bptt, 1] for y and [batch_size, bptt, contrastive_duplicates, 1]
        """
        # Random kernel and bias initializers:
        randN_05 = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        bias_init = keras.initializers.Constant(0.01)

        if config.compression_flag == 1:
            DV_hidden = config.hidden_size_compression
        else:
            DV_hidden = config.hidden_size

        # Custom layers definition:
        max_norm = config.max_norm_y if split_input is None else config.max_norm_xy
        lstm = LSTMNew(DV_hidden[0], return_sequences=True, name="{}_lstm".format(name), stateful=True,
                       dropout=config.dropout, recurrent_dropout=config.dropout, contrastive_duplicates=config.contrastive_duplicates)
        split = Lambda(tf.split, arguments={'axis': -2, 'num_or_size_splits': [1, config.contrastive_duplicates]})
        dense0 = Dense(DV_hidden[1], bias_initializer=bias_init, kernel_initializer=randN_05, activation="relu")
        dense1 = Dense(DV_hidden[2], bias_initializer=bias_init, kernel_initializer=randN_05, activation="relu")
        dense2 = Dense(1, bias_initializer=bias_init, kernel_initializer=randN_05, activation=None,
                       kernel_constraint=tf.keras.constraints.MaxNorm(max_value=max_norm))
        split_input = split_input if split_input is None else dims
        contrastive_noise = ContrastiveNoiseLayer(config, split_input=split_input)
        exp_layer = Lambda(lambda x: tf.math.exp(x))
        noise_layer = Lambda(lambda x: x + tf.random.normal(shape=tf.shape(x), mean=0, stddev=0.05, dtype=x.dtype))
        dense_compress = Dense(2, bias_initializer=bias_init, kernel_initializer=randN_05, activation="relu")


        # Architecture definition:

        in_ = t = Input(batch_shape=input_shape)  # input layer
        t = contrastive_noise(t)  # create LSTM input, sample y_tild
        t = lstm(t)  # modified LSTM layer
        t_1, t_2 = split(t)  # split into T and T_bar counterparts
        t_1, t_2 = (dense0(x) for x in (t_1, t_2))  # post-processing
        t_1, t_2 = (dense1(x) for x in (t_1, t_2))
        t_1, t_2 = (dense2(x) for x in (t_1, t_2))
        t_2 = exp_layer(t_2)  # yield exp(T_bar)

        model = keras.models.Model(inputs=in_, outputs=[t_1, t_2])  # encapsulate in a Keras model
        return model

    xy_model = build("LSTM_xy",  [batch_size, bptt, dims[0] + dims[1]], split_input=True)
    y_model = build("LSTM_y",  [batch_size, bptt, dims[1]])


    model = {'y': y_model,
             'xy': xy_model}

    return model


def DVModel_with_states(config, bptt, batch_size, dims=(1, 1), contrastive_duplicates=None):
    def build(name, input_shape, split_input=None):
        """
        Building the DV estimators models - The amount of T outputs is equal to the amount of samples inserted
        output shape: [batch_size, bptt, 1] for y and [batch_size, bptt, contrastive_duplicates, 1]
        """
        # Random kernel and bias initializers:
        randN_05 = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        bias_init = keras.initializers.Constant(0.01)

        if config.compression_flag == 1:
            DV_hidden = config.hidden_size_compression
        else:
            DV_hidden = config.hidden_size

        # Custom layers definition:
        max_norm = config.max_norm_y if split_input is None else config.max_norm_xy
        lstm = LSTMNew(DV_hidden[0], return_sequences=True, name="{}_lstm".format(name), stateful=True,
                       dropout=config.dropout, recurrent_dropout=config.dropout, contrastive_duplicates=config.contrastive_duplicates)
        split = Lambda(tf.split, arguments={'axis': -2, 'num_or_size_splits': [1, config.contrastive_duplicates]})
        dense0 = Dense(DV_hidden[1], bias_initializer=bias_init, kernel_initializer=randN_05, activation="relu")
        dense1 = Dense(DV_hidden[2], bias_initializer=bias_init, kernel_initializer=randN_05, activation="relu")
        dense2 = Dense(1, bias_initializer=bias_init, kernel_initializer=randN_05, activation=None,
                       kernel_constraint=tf.keras.constraints.MaxNorm(max_value=max_norm))
        split_input = split_input if split_input is None else dims
        contrastive_noise = ContrastiveNoiseLayer(config, split_input=split_input)
        exp_layer = Lambda(lambda x: tf.math.exp(x))
        noise_layer = Lambda(lambda x: x + tf.random.normal(shape=tf.shape(x), mean=0, stddev=0.05, dtype=x.dtype))
        dense_compress = Dense(2, bias_initializer=bias_init, kernel_initializer=randN_05, activation="relu")


        # Architecture definition:

        in_ = t = Input(batch_shape=input_shape)  # input layer
        t = contrastive_noise(t)  # create LSTM input, sample y_tild
        t = lstm(t)  # modified LSTM layer
        t_1_, t_2 = split(t)  # split into T and T_bar counterparts
        t_1, t_2 = (dense0(x) for x in (t_1_, t_2))  # post-processing
        t_1, t_2 = (dense1(x) for x in (t_1, t_2))
        t_1, t_2 = (dense2(x) for x in (t_1, t_2))
        t_2 = exp_layer(t_2)  # yield exp(T_bar)

        # t_1_ are the states of the LSTM for representation analysis if required

        model = keras.models.Model(inputs=in_, outputs=[t_1, t_2, t_1_])  # encapsulate in a Keras model
        return model

    y_model = build("LSTM_y",  [batch_size, bptt, dims[1]])
    xy_model = build("LSTM_xy",  [batch_size, bptt, dims[0] + dims[1]], split_input=True)

    model = {'y': y_model,
             'xy': xy_model}

    return model





###############
# For new pdine version
def DVModel_pdine(config, mode='train'):
    def build(name, hidden_sizes, input_shape, pertubate_states=0, compress_representation=0, split_input=None):
        """
        Building the DV estimators models - The amount of T outputs is equal to the amount of samples inserted
        output shape: [batch_size, bptt, 1] for y and [batch_size, bptt, contrastive_duplicates, 1]
        """
        # Random kernel and bias initializers:
        randN_05 = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        bias_init = keras.initializers.Constant(0.01)


        # Custom layers definition:
        max_norm = config.max_norm_y if split_input is None else config.max_norm_xy
        lstm = LSTMNew(config.lstm_dv_size, return_sequences=True, name="{}_lstm".format(name), stateful=True,
                       dropout=config.dropout, recurrent_dropout=config.dropout, contrastive_duplicates=config.contrastive_duplicates)
        split = Lambda(tf.split, arguments={'axis': -2, 'num_or_size_splits': [1, config.contrastive_duplicates]}, name="split_layer")
        split_input = split_input if split_input is None else dims
        contrastive_noise = ContrastiveNoiseLayer(config, split_input=split_input)
        exp_layer = Lambda(lambda x: tf.math.exp(x), name="Exponent_layer")
        noise_layer = Lambda(lambda x: x + tf.random.normal(shape=tf.shape(x), mean=0, stddev=0.05, dtype=x.dtype))
        dense_compress = Dense(2, bias_initializer=bias_init, kernel_initializer=randN_05, activation="relu")

        # define the list of dense layers
        dense_list = []
        for i,size in enumerate(hidden_sizes):
            if size == 1:
                dense_list.append(Dense(1, bias_initializer=bias_init, kernel_initializer=randN_05, activation=None,
                      kernel_constraint=tf.keras.constraints.MaxNorm(max_value=max_norm), name="DV_"+name+"_Dense_{}".format(i+1)))
            else:
                dense_list.append(Dense(size, bias_initializer=bias_init, kernel_initializer=randN_05,
                                        activation="relu", name="DV_"+name+"_Dense_{}".format(i+1)))

        # Architecture definition:
        in_ = t = Input(batch_shape=input_shape, name="DV_"+name+"_input")  # input layer
        t = contrastive_noise(t)  # create LSTM input, sample y_tild
        t = lstm(t)  # modified LSTM layer
        t_1, t_2 = split(t)  # split into T and T_bar counterparts
        if pertubate_states:
            t_1, t_2 = (noise_layer(x) for x in (t_1, t_2)) # pertubate t_2 to obtain a better separation of the lstm states
        if compress_representation:
            t_1 = dense_compress(t_1)
            t_1_comp = t_1
            for layer in dense_list[1:]:
                t_1, t_2 = (layer(x) for x in (t_1, t_2))
        else:
            for layer in dense_list:
                t_1, t_2 = (layer(x) for x in (t_1, t_2))
        t_2 = exp_layer(t_2)


        if compress_representation:
            model = keras.models.Model(inputs=in_, outputs=[t_1, t_2, t_1_comp])  # encapsulate in a Keras model
        else:
            model = keras.models.Model(inputs=in_, outputs=[t_1, t_2])  # encapsulate in a Keras model
        return model

    dims = (config.x_dim, config.y_dim)
    pertubate_states = config.pertubate_dv_flag and mode =='train'
    compress_representation = config.compress_dv_flag

    y_model = build("LSTM_y", hidden_sizes=config.hidden_sizes_dv_y,
                    input_shape=[config.batch_size, config.bptt, config.y_dim],
                    pertubate_states=pertubate_states,
                    compress_representation=compress_representation
                    )
    xy_model = build("LSTM_xy", hidden_sizes=config.hidden_sizes_dv_xy,
                    input_shape=[config.batch_size, config.bptt, config.y_dim+config.x_dim],
                    pertubate_states=pertubate_states,
                    compress_representation=compress_representation,
                    split_input=True
                    )

    model = {'y': y_model,
             'xy': xy_model}

    return model


def DVModel_denses(config, bptt, batch_size, dims=(1, 1), contrastive_duplicates=None):
    def build(name, input_shape, split_input=None):
        """
        Building the DV estimators models - The amount of T outputs is equal to the amount of samples inserted
        output shape: [batch_size, bptt, 1] for y and [batch_size, bptt, contrastive_duplicates, 1]
        """
        # Random kernel and bias initializers:
        randN_05 = keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        bias_init = keras.initializers.Constant(0.01)

        # Custom layers definition:
        max_norm = config.max_norm_y if split_input is None else config.max_norm_xy
        lstm = LSTMNew(config.hidden_size[0], return_sequences=True, name="{}_lstm".format(name), stateful=True,
                       dropout=config.dropout, recurrent_dropout=config.dropout, contrastive_duplicates=config.contrastive_duplicates)
        split = Lambda(tf.split, arguments={'axis': -2, 'num_or_size_splits': [1, config.contrastive_duplicates]})
        dense_list = []
        for i in range(len(config.hidden_size)-1):
            dense_list.append(Dense(config.hidden_size[i+1], bias_initializer=bias_init, kernel_initializer=randN_05,
                                    activation="relu", name="dense_"+str(i+1)))
        dense_out = Dense(1, bias_initializer=bias_init, kernel_initializer=randN_05, activation=None,
                       kernel_constraint=tf.keras.constraints.MaxNorm(max_value=max_norm))
        split_input = split_input if split_input is None else dims
        contrastive_noise = ContrastiveNoiseLayer(config, split_input=split_input)
        exp_layer = Lambda(lambda x: tf.math.exp(x))
        noise_layer = Lambda(lambda x: x + tf.random.normal(shape=tf.shape(x), mean=0, stddev=0.25, dtype=x.dtype))


        # Architecture definition:

        in_ = t = Input(batch_shape=input_shape)  # input layer
        t = contrastive_noise(t)  # create LSTM input, sample y_tild
        t = lstm(t)  # modified LSTM layer
        t_1, t_2 = split(t)  # split into T and T_bar counterparts
        # adding noise
        # t_1, t_2 = (noise_layer(x) for x in (t_1, t_2))
        for i in range(len(dense_list)):
            t_1, t_2 = (dense_list[i](x) for x in (t_1, t_2))  # post-processing
        t_1, t_2 = (dense_out(x) for x in (t_1, t_2))
        t_2 = exp_layer(t_2)  # yield exp(T_bar)

        model = keras.models.Model(inputs=in_, outputs=[t_1, t_2])  # encapsulate in a Keras model
        return model

    y_model = build("LSTM_y",  [batch_size, bptt, dims[1]])
    xy_model = build("LSTM_xy",  [batch_size, bptt, dims[0] + dims[1]], split_input=True)

    model = {'y': y_model,
             'xy': xy_model}

    return model
