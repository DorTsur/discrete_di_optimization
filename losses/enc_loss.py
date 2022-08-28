import tensorflow as tf
from tensorflow.keras import backend as K


class Model_enc_loss(tf.keras.losses.Loss):

    def __init__(self, subtract=None, name='model_loss',  **kwargs):
        super(Model_enc_loss, self).__init__(name=name, reduction='none')
        self.subtract = subtract
        self.loss_fn = {"y": Enc_Loss("y"),
                        "xy": Enc_Loss("xy")}

    def call(self, t_y, t_xy, **kwargs):
        loss_y = self.loss_fn["y"](t_y[0], t_y[1])
        loss_xy = self.loss_fn["xy"](t_xy[0], t_xy[1])
        return loss_xy - loss_y

        # # enc loss without log(rhs part on each dv):
        # Ny = tf.cast(tf.reduce_prod(t_y[0].shape[:-1]), tf.float64)
        # Nxy = tf.cast(tf.reduce_prod(t_xy[0].shape[:-1]), tf.float64)
        # return K.sum(t_y[0] / Ny) - K.sum(t_xy[0] / Nxy)



class Enc_Loss(tf.keras.losses.Loss):
    def __init__(self, name='enc_loss'):
        super(Enc_Loss, self).__init__(name=name, reduction='none')

    def call(self, t1, t2, **kwargs):
        N = tf.cast(tf.reduce_prod(t1.shape[:-1]), tf.float64)
        N_ref = tf.cast(tf.reduce_prod(t2.shape[:-1]), tf.float64)
        loss_t = K.sum(t1 / N)
        loss_et = K.sum(t2 / N_ref)
        # return -(loss_t - loss_et)
        return -(loss_t)
    # enc loss without log(rhs part on each dv):


    def call_new(self, t1, t2, **kwargs):
        N = tf.cast(tf.reduce_prod(t1.shape[:-1]), tf.float64)
        N_ref = tf.cast(tf.reduce_prod(t2.shape[:-1]), tf.float64)
        loss_t = K.sum(t1 / N)
        loss_et = K.sum(t2 / N_ref)
        # return -(loss_t - loss_et)
        return -(loss_t)




