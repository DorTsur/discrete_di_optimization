import tensorflow as tf


def load_data(config):
    if config.data_name == "encoder":
        return load_data_encoder_input(config)  # in this option we have an encoder which is fed with zeros
    elif config.data_name == "cont_ndt":
        return load_data_awgn(config)
    else:
        raise ValueError("'{}' is invalid data name".format(config.data_name))


def load_data_encoder_input(config):
    """
    Creates an input batch of zeros for the sequential PMF generator as the shape of an input x.
    """
    def data_gen():
        """
        Data Generator - creates the encoder inputs
        """
        noise = tf.zeros(shape=[config.batch_size, 1, config.x_dim], dtype="float64")
        yield noise

    train = tf.data.Dataset.from_generator(data_gen,
                                           tf.float64,
                                           output_shapes=tf.TensorShape([config.batch_size, 1, config.x_dim]))

    # define a data loader for each case (vary by length)
    data = {'train': lambda: train.take(config.train_epoch_len).repeat(config.batches),
            'eval': lambda: train.take(config.eval_epoch_len).repeat(10*config.batches),
            'long_eval': lambda: train.take(config.long_eval_epoch_len).repeat(100*config.batches)}
    return data

def load_data_awgn(config):
    """
    Creates an input batch of zeros for the sequential PMF generator as the shape of an input x.
    """
    def data_gen():
        """
        Data Generator - creates the encoder inputs
        """
        noise = tf.random.normal(shape=[config.batch_size, config.bptt, config.x_dim], mean=0, stddev=1, dtype="float64")
        yield noise

    train = tf.data.Dataset.from_generator(data_gen,
                                           tf.float64,
                                           output_shapes=tf.TensorShape([config.batch_size, config.bptt, config.x_dim]))

    # define a data loader for each case (vary by length)
    data = {'train': lambda: train.take(config.train_epoch_len).repeat(config.batches),
            'eval': lambda: train.take(config.eval_epoch_len).repeat(10*config.batches),
            'long_eval': lambda: train.take(config.long_eval_epoch_len).repeat(100*config.batches)}
    return data
