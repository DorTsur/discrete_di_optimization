import tensorflow as tf

############## Function Layer Functions ################

def clean_channel(x, p_bsc=None, p_z=None, p_bec=None, batch_size=1):
    return x


def bsc_channel(x, p_bsc, p_z=None, p_bec=None, batch_size=1):
    z = tf.cast(
        tf.reshape(tf.random.categorical(logits=tf.math.log([[1 - p_bsc, p_bsc]]), num_samples=batch_size),
                   shape=tf.shape(x)), 'float64')
    return tf.math.floormod(x + z, 2)


def z_channel(x, p_bsc=None, p_z=None, p_bec=None, batch_size=1):
    B = tf.shape(x)[0]  # acquire catch size
    T = tf.shape(x)[1]  # acquire memory length
    p_z_t = p_z*tf.ones(shape=[B*T, 1])  # create logits
    p_z_bar_t = tf.ones_like(p_z_t) - p_z_t
    logits = tf.math.log(tf.concat([p_z_bar_t, p_z_t], axis=1))  # create transition probability logits of shape ([B*T,1])
    y = tf.cast(tf.random.categorical(logits=logits, num_samples=1), 'float64')  # create bernoulli samples
    y = tf.reshape(y, shape=tf.shape(x))  # reshape into (B,T,1)
    y = tf.where(tf.equal(x, 0), x, y)
    return tf.cast(y, 'float64')


def s_channel(x, p_bsc=None, p_z=None, p_bec=None, batch_size=1):
    B = tf.shape(x)[0]  # acquire catch size
    T = tf.shape(x)[1]  # acquire memory length
    p_z_t = p_z*tf.ones(shape=[B*T, 1])  # create logits
    p_z_bar_t = tf.ones_like(p_z_t) - p_z_t
    logits = tf.math.log(tf.concat([p_z_bar_t, p_z_t], axis=1))  # create transition probability logits of shape ([B*T,1])
    y = tf.cast(tf.random.categorical(logits=logits, num_samples=1), 'float64')  # create bernoulli samples
    y = tf.reshape(y, shape=tf.shape(x))  # reshape into (B,T,1)
    y = tf.where(tf.equal(x, 1), x, y)
    return tf.cast(y, 'float64')


def bec_channel(x, p_bsc=None, p_z=None, p_bec=None, batch_size=None):
    """
    Implementation of the BEC channel, p_bec is thee error probability of transmission.
    The error symbol is mapped to the number '2'
    """
    B = tf.shape(x)[0]  # obtain batch size
    T = tf.shape(x)[1]  # acquire memory length
    p_bec_t = p_bec * tf.ones(shape=[B*T, 1])
    p_bec_bar_t = tf.ones_like(p_bec_t) - p_bec_t
    logits = tf.math.log(tf.concat([p_bec_bar_t, p_bec_t], axis=1))  # create transition probability logits
    y_0 = tf.cast(tf.random.categorical(logits=logits, num_samples=1),
                'float64')
    y_1 = tf.cast(tf.random.categorical(logits=logits, num_samples=1),
                'float64')  # defining two independent bec outputs samples tensors
    y_0 = 2*y_0  # mapping e to '2'
    y_1 = tf.ones_like(y_1) + y_1
    y_0 = tf.reshape(y_0, shape=tf.shape(x))  # reshape into (B,T,1)
    y_1 = tf.reshape(y_1, shape=tf.shape(x))  # reshape into (B,T,1)
    y = tf.where(tf.equal(x, 0), y_0, y_1)
    return y

