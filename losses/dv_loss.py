import tensorflow as tf
from tensorflow.keras import backend as K


class Model_loss(tf.keras.losses.Loss):

    def __init__(self, subtract=None, name='model_loss',  **kwargs):
        super(Model_loss, self).__init__(name=name, reduction='none')
        self.subtract = subtract
        self.loss_fn = {"y": DV_Loss("y"),
                        "xy": DV_Loss("xy")}

    def call(self, t_y, t_xy, **kwargs):
        loss_y = self.loss_fn["y"](t_y[0], t_y[1])
        loss_xy = self.loss_fn["xy"](t_xy[0], t_xy[1])
        if self.subtract is None:  # original DV case - to train each statistics network
            return tf.stack([loss_y, loss_xy], axis=0)
        else:  # encoder case - the difference is the loss
            return loss_xy - loss_y


class DV_Loss(tf.keras.losses.Loss):
    def __init__(self, name='dv_loss'):
        super(DV_Loss, self).__init__(name=name, reduction='none')

    def call(self, t1, t2, **kwargs):
        N = tf.cast(tf.reduce_prod(t1.shape[:-1]), tf.float64)
        N_ref = tf.cast(tf.reduce_prod(t2.shape[:-1]), tf.float64)
        loss_t = K.sum(t1 / N)
        loss_et = K.sum(t2 / N_ref)
        return -(loss_t - K.log(loss_et))




