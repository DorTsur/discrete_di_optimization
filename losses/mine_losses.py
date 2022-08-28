import tensorflow as tf
from tensorflow.keras import backend as K

class DV_Loss(tf.keras.losses.Loss):
    def __init__(self, name='dv_loss'):
        super(DV_Loss, self).__init__(name=name, reduction='none')

    def call(self, t1, t2, **kwargs):
        N = tf.cast(tf.reduce_prod(t1.shape[:-1]), tf.float64)
        N_ref = tf.cast(tf.reduce_prod(t2.shape[:-1]), tf.float64)
        loss_t = K.sum(t1 / N)
        loss_et = K.sum(t2 / N_ref)
        return -(loss_t - K.log(loss_et))


class DV_Loss_regularized(tf.keras.losses.Loss):
    def __init__(self, reg_coef=0.01, name='dv_loss'):
        super(DV_Loss_regularized, self).__init__(name=name, reduction='none')
        self.reg_coef = reg_coef

    def call(self, t1, t2, **kwargs):
        N = tf.cast(tf.reduce_prod(t1.shape[:-1]), tf.float64)
        N_ref = tf.cast(tf.reduce_prod(t2.shape[:-1]), tf.float64)
        loss_t = K.sum(t1 / N)
        loss_et = K.sum(t2 / N_ref)
        pdf_regularizer = self.reg_coef*tf.square(loss_et-1)
        return -(loss_t - K.log(loss_et) - pdf_regularizer)


class PMF_Loss(tf.keras.losses.Loss):
    def __init__(self, config, name='dv_loss'):
        super(PMF_Loss, self).__init__(name=name, reduction='none')
        self.batch_size = config.batch_size

    def call(self, p, in2, **kwargs):
        [t, ind] = in2
        n_ind = tf.cast(tf.expand_dims(tf.linspace(start=0, stop=self.batch_size-1, num=self.batch_size), axis=-1), 'int64')
        x_ind = tf.concat([n_ind, tf.cast(ind, 'int64')], axis=-1)
        # p = tf.concat([p] * self.batch_size, axis=0)

        p_loss = tf.math.log(tf.gather_nd(indices=x_ind, params=p))

        # p = p / p.numpy()
        # p_loss = (tf.gather_nd(indices=x_ind, params=p))

        loss_t = tf.math.reduce_mean(t*p_loss)
        return -(loss_t)
