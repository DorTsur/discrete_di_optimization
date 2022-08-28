import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import CategoricalCrossentropy
import logging
import math
logger = logging.getLogger("logger")


class ModelMetrics(tf.keras.metrics.Metric):

    def __init__(self, writer, name='', **kwargs):
        super(ModelMetrics, self).__init__(name=name, **kwargs)
        self.writer = writer
        self.metric_pool = [DV(name='dv_xy_{}'.format(name)),
                            DV(name='dv_y_{}'.format(name)),
                            DI(name='di_{}'.format(name)),
                            DI_bits(name='di_bits')]

    def update_state(self, t_y, t_xy, **kwargs):
        self.metric_pool[0].update_state(t_xy[0], t_xy[1])
        self.metric_pool[1].update_state(t_y[0], t_y[1])
        self.metric_pool[2].update_state(t_y[0], t_y[1], t_xy[0], t_xy[1])
        self.metric_pool[3].update_state(t_y[0], t_y[1], t_xy[0], t_xy[1])

    def result(self):
        return [metric.result() for metric in self.metric_pool]

    def reset_states(self):
        for metric in self.metric_pool:
            metric.reset_states()
        return

    def log_metrics(self, epoch, model_name):
        # log to tensorboard
        with self.writer.as_default():
            for metric in self.metric_pool:
                tf.summary.scalar(metric.name, metric.result(), epoch)

        # print to terminal
        msg = ["{} Epoch: {:05d}\t".format(self.name, epoch)]
        for metric in self.metric_pool:
            if np.isnan(metric.result()):
                raise ValueError("NaN appeared in metric {}".format(metric.name))
            msg.append("{:s} {:3.6f}\t".format(metric.name, float(metric.result())))
        msg.append(model_name)
        logger.info("\t".join(msg))


class DV(tf.keras.metrics.Metric):  # estimated DV loss calcaultion metric class
    def __init__(self, name='dv_loss', **kwargs):
        super(DV, self).__init__(name=name, **kwargs)
        self.T = self.add_weight(name='t', initializer='zeros')
        self.exp_T_bar = self.add_weight(name='exp_t_bar', initializer='zeros')
        self.global_counter = self.add_weight(name='n', initializer='zeros')
        self.global_counter_ref = self.add_weight(name='n_ref', initializer='zeros')

    def update_state(self, T, T_bar, **kwargs):
        self.T.assign(self.T + tf.reduce_sum(T))
        self.exp_T_bar.assign(self.exp_T_bar + tf.reduce_sum(T_bar))
        self.global_counter.assign(self.global_counter + tf.cast(tf.reduce_prod(T.shape[:-1]), dtype=tf.float64))
        self.global_counter_ref.assign(self.global_counter_ref + tf.cast(tf.reduce_prod(T_bar.shape[:-1]), dtype=tf.float64))

    def result(self):
        loss = self.T / self.global_counter - K.log(self.exp_T_bar / self.global_counter_ref)
        return loss


class DI(tf.keras.metrics.Metric):  # estimated DI calcaultion metric class
    def __init__(self, name='dv_loss', **kwargs):
        super(DI, self).__init__(name=name, **kwargs)
        self.c_T = self.add_weight(name='c_t', initializer='zeros')
        self.c_exp_T_bar = self.add_weight(name='c_exp_t_bar', initializer='zeros')
        self.xc_T = self.add_weight(name='xc_t', initializer='zeros')
        self.xc_exp_T_bar = self.add_weight(name='xc_exp_t_bar', initializer='zeros')
        self.global_counter = self.add_weight(name='n', initializer='zeros')
        self.global_counter_ref = self.add_weight(name='n_ref', initializer='zeros')

    def update_state(self, c_T, c_T_bar, xc_T, xc_T_bar, **kwargs):
        self.c_T.assign(self.c_T + tf.reduce_sum(c_T))
        self.c_exp_T_bar.assign(self.c_exp_T_bar + tf.reduce_sum(c_T_bar))

        self.xc_T.assign(self.xc_T + tf.reduce_sum(xc_T))
        self.xc_exp_T_bar.assign(self.xc_exp_T_bar + tf.reduce_sum(xc_T_bar))

        self.global_counter.assign(self.global_counter + c_T.shape[0]*c_T.shape[1])
        self.global_counter_ref.assign(self.global_counter_ref + c_T_bar.shape[0]*c_T_bar.shape[1]*c_T_bar.shape[2])

    def result(self):
        loss_y = self.c_T / self.global_counter - K.log(self.c_exp_T_bar / self.global_counter_ref)
        loss_xy = self.xc_T / self.global_counter - K.log(self.xc_exp_T_bar / self.global_counter_ref)
        return loss_xy - loss_y


class DI_bits(tf.keras.metrics.Metric):  # estimated DI calcaultion metric class in bits
    def __init__(self, name='dv_loss', **kwargs):
        super(DI_bits, self).__init__(name=name, **kwargs)
        self.c_T = self.add_weight(name='c_t', initializer='zeros')
        self.c_exp_T_bar = self.add_weight(name='c_exp_t_bar', initializer='zeros')
        self.xc_T = self.add_weight(name='xc_t', initializer='zeros')
        self.xc_exp_T_bar = self.add_weight(name='xc_exp_t_bar', initializer='zeros')
        self.global_counter = self.add_weight(name='n', initializer='zeros')
        self.global_counter_ref = self.add_weight(name='n_ref', initializer='zeros')

    def update_state(self, c_T, c_T_bar, xc_T, xc_T_bar, **kwargs):
        self.c_T.assign(self.c_T + tf.reduce_sum(c_T))
        self.c_exp_T_bar.assign(self.c_exp_T_bar + tf.reduce_sum(c_T_bar))

        self.xc_T.assign(self.xc_T + tf.reduce_sum(xc_T))
        self.xc_exp_T_bar.assign(self.xc_exp_T_bar + tf.reduce_sum(xc_T_bar))

        self.global_counter.assign(self.global_counter + c_T.shape[0]*c_T.shape[1])
        self.global_counter_ref.assign(self.global_counter_ref + c_T_bar.shape[0]*c_T_bar.shape[1]*c_T_bar.shape[2])

    def result(self):
        loss_y = self.c_T / self.global_counter - K.log(self.c_exp_T_bar / self.global_counter_ref)
        loss_xy = self.xc_T / self.global_counter - K.log(self.xc_exp_T_bar / self.global_counter_ref)
        return (loss_xy - loss_y)/math.log(2)


class PMF(tf.keras.metrics.Metric):  # estimated DV loss calcaultion metric class
    def __init__(self, name='dv_loss', **kwargs):
        super(PMF, self).__init__(name=name, **kwargs)
        self.p = self.add_weight(name='p', initializer='zeros')
        self.global_counter = self.add_weight(name='n', initializer='zeros')

    def update_state(self, p, **kwargs):
        self.p.assign(self.p + tf.reduce_sum(p))
        self.global_counter.assign(self.global_counter + tf.cast(tf.reduce_prod(p.shape[:-1]), dtype=tf.float64))

    def result(self):
        result = self.p / self.global_counter
        return result


class ModelWithEncMetrics(ModelMetrics):

    def __init__(self, writer, name='', **kwargs):
        super(ModelMetrics, self).__init__(name=name, **kwargs)
        self.writer = writer
        self.metric_pool = [DV(name='dv_xy_{}'.format(name)),
                            DV(name='dv_y_{}'.format(name)),
                            DI(name='di_{}'.format(name)),
                            PMF(name='p_mean'),
                            DI_bits(name='di_bits_{}'.format(name))]


    def update_state(self, data, **kwargs):
        [t_y,t_xy,p] = data
        self.metric_pool[0].update_state(t_xy[0], t_xy[1])
        self.metric_pool[1].update_state(t_y[0], t_y[1])
        self.metric_pool[2].update_state(t_y[0], t_y[1], t_xy[0], t_xy[1])
        self.metric_pool[3].update_state(p)
        self.metric_pool[4].update_state(t_y[0], t_y[1], t_xy[0], t_xy[1])


    def result(self):
        return [metric.result() for metric in self.metric_pool]

    def reset_states(self):
        for metric in self.metric_pool:
            metric.reset_states()
        return

    def log_metrics(self, epoch, model_name):
        # log to tensorboard
        with self.writer.as_default():
            for metric in self.metric_pool:
                tf.summary.scalar(metric.name, metric.result(), epoch)

        # print to terminal
        msg = ["{} Epoch: {:05d}\t".format(self.name, epoch)]
        for metric in self.metric_pool:
            if np.isnan(metric.result()):
                raise ValueError("NaN appeared in metric {}".format(metric.name))
            msg.append("{:s} {:3.6f}\t".format(metric.name, float(metric.result())))
        msg.append(model_name)
        logger.info("\t".join(msg))


class PDINE_metrics(ModelMetrics):

    def __init__(self, writer, name='', **kwargs):
        super(ModelMetrics, self).__init__(name=name, **kwargs)
        self.writer = writer
        self.metric_pool = [DV(name='dv_xy_{}'.format(name)),
                            DV(name='dv_y_{}'.format(name)),
                            DI(name='di_{}'.format(name)),
                            PMF(name='p_mean'),
                            DI_bits(name='di_bits_{}'.format(name)),
                            CategoricalCrossentropy(name="Q_est_CE")]

    def update_state(self, data, **kwargs):
        """
        receives data_list
        data_list = [t_y,t_xy,p,q_est,s_one_hot]
        """
        [t_y, t_xy, p, q_est, s_onehot] = data

        self.metric_pool[0].update_state(t_xy[0], t_xy[1])
        self.metric_pool[1].update_state(t_y[0], t_y[1])
        self.metric_pool[2].update_state(t_y[0], t_y[1], t_xy[0], t_xy[1])
        self.metric_pool[3].update_state(p)
        self.metric_pool[4].update_state(t_y[0], t_y[1], t_xy[0], t_xy[1])
        self.metric_pool[5].update_state(y_true=s_onehot, y_pred=q_est)

    def result(self):
        return [metric.result() for metric in self.metric_pool]

    def reset_states(self):
        for metric in self.metric_pool:
            metric.reset_states()
        return

    def log_metrics(self, epoch, model_name):
        # log to tensorboard
        with self.writer.as_default():
            for metric in self.metric_pool:
                tf.summary.scalar(metric.name, metric.result(), epoch)

        # print to terminal
        msg = ["{} Epoch: {:05d}\t".format(self.name, epoch)]
        for metric in self.metric_pool:
            if np.isnan(metric.result()):
                raise ValueError("NaN appeared in metric {}".format(metric.name))
            msg.append("{:s} {:3.6f}\t".format(metric.name, float(metric.result())))
        msg.append(model_name)
        logger.info("\t".join(msg))


class Q_est_metrics(tf.keras.metrics.Metric):

    def __init__(self, writer, name='', **kwargs):
        super(Q_est_metrics, self).__init__(name=name, **kwargs)
        self.writer = writer
        self.metric_pool = [CategoricalCrossentropy(name='Q_CE')]

    def update_state(self, data, **kwargs):
        [q, s] = data
        # self.metric_pool[0].update_state(q, s)
        self.metric_pool[0].update_state(tf.reshape(q, [tf.reduce_prod(tf.shape(q)[:-1]),tf.shape(q)[-1]]), tf.reshape(s, [tf.reduce_prod(tf.shape(s)[:-1]),tf.shape(s)[-1]]))

    def result(self):
        return [metric.result() for metric in self.metric_pool]

    def reset_states(self):
        for metric in self.metric_pool:
            metric.reset_states()
        return

    def log_metrics(self, epoch):
        with self.writer.as_default():
            for metric in self.metric_pool:
                tf.summary.scalar(metric.name, metric.result(), epoch)

        # print to terminal
        msg = ["{} Epoch: {:05d}\t".format(self.name, epoch)]
        for metric in self.metric_pool:
            if np.isnan(metric.result()):
                raise ValueError("NaN appeared in metric {}".format(metric.name))
            msg.append("{:s} {:3.6f}\t".format(metric.name, float(metric.result())))
        logger.info("\t".join(msg))


# For new pdine version
class pdine_metrics(ModelMetrics):
    def __init__(self, writer, name='', **kwargs):
        super(pdine_metrics, self).__init__(name=name, writer=writer, **kwargs)
        self.writer = writer
        self.metric_pool = [DV(name='dv_xy_{}'.format(name)),
                            DV(name='dv_y_{}'.format(name)),
                            DI_bits(name='di_bits')]

    def update_state(self, data, **kwargs):
        """
        data is a dictionary
        """
        self.metric_pool[0].update_state(data["t_xy"][0], data["t_xy"][1])
        self.metric_pool[1].update_state(data["t_y"][0], data["t_y"][1])
        self.metric_pool[2].update_state(data["t_y"][0], data["t_y"][1], data["t_xy"][0], data["t_xy"][1])



### For MINE based simulations:
class MINE_PMF_metrics(tf.keras.metrics.Metric):

    def __init__(self, writer, name='', **kwargs):
        super(MINE_PMF_metrics, self).__init__(name=name, **kwargs)
        self.writer = writer
        self.metric_pool = [DVMINE_metric(name='dv_{}'.format(name)),
                            DVMINE_bits_metric(name='dv_bits_{}'.format(name)),
                            PMFMINE_loss_metric(name='pmf_loss_{}'.format(name))]


    def update_state(self, metric_input, **kwargs):
        [t, t_, p, ind] = metric_input
        self.metric_pool[0].update_state(t, t_)
        self.metric_pool[1].update_state(t, t_)
        self.metric_pool[2].update_state(t, p, ind)

    def result(self):
        return [metric.result() for metric in self.metric_pool]

    def reset_states(self):
        for metric in self.metric_pool:
            metric.reset_states()
        return

    def log_metrics(self, epoch, model_name):
        # log to tensorboard
        with self.writer.as_default():
            for metric in self.metric_pool:
                tf.summary.scalar(metric.name, metric.result(), epoch)

        # print to terminal
        msg = ["{} Epoch: {:05d}\t".format(self.name, epoch)]
        for metric in self.metric_pool:
            if np.isnan(metric.result()):
                raise ValueError("NaN appeared in metric {}".format(metric.name))
            msg.append("{:s} {:3.6f}\t".format(metric.name, float(metric.result())))
        msg.append(model_name)
        logger.info("\t".join(msg))


class DVMINE_metric(tf.keras.metrics.Metric):  # estimated DV loss calcaultion metric class
    def __init__(self, name='dv_loss', **kwargs):
        super(DVMINE_metric, self).__init__(name=name, **kwargs)
        self.T = self.add_weight(name='t', initializer='zeros')
        self.exp_T_bar = self.add_weight(name='exp_t_bar', initializer='zeros')
        self.global_counter = self.add_weight(name='n', initializer='zeros')
        self.global_counter_ref = self.add_weight(name='n_ref', initializer='zeros')

    def update_state(self, T, T_bar, **kwargs):
        self.T.assign(self.T + tf.reduce_sum(T))
        self.exp_T_bar.assign(self.exp_T_bar + tf.reduce_sum(T_bar))
        self.global_counter.assign(self.global_counter + tf.cast(tf.reduce_prod(T.shape[:-1]), dtype=tf.float64))
        self.global_counter_ref.assign(self.global_counter_ref + tf.cast(tf.reduce_prod(T_bar.shape[:-1]), dtype=tf.float64))

    def result(self):
        loss = self.T / self.global_counter - K.log(self.exp_T_bar / self.global_counter_ref)
        return loss

class DVMINE_bits_metric(tf.keras.metrics.Metric):  # estimated DV loss calcaultion metric class
    def __init__(self, name='dv_loss', **kwargs):
        super(DVMINE_bits_metric, self).__init__(name=name, **kwargs)
        self.T = self.add_weight(name='t', initializer='zeros')
        self.exp_T_bar = self.add_weight(name='exp_t_bar', initializer='zeros')
        self.global_counter = self.add_weight(name='n', initializer='zeros')
        self.global_counter_ref = self.add_weight(name='n_ref', initializer='zeros')

    def update_state(self, T, T_bar, **kwargs):
        self.T.assign(self.T + tf.reduce_sum(T))
        self.exp_T_bar.assign(self.exp_T_bar + tf.reduce_sum(T_bar))
        self.global_counter.assign(self.global_counter + tf.cast(tf.reduce_prod(T.shape[:-1]), dtype=tf.float64))
        self.global_counter_ref.assign(self.global_counter_ref + tf.cast(tf.reduce_prod(T_bar.shape[:-1]), dtype=tf.float64))

    def result(self):
        loss = self.T / self.global_counter - K.log(self.exp_T_bar / self.global_counter_ref)
        return loss/math.log(2)

class PMFMINE_loss_metric(tf.keras.metrics.Metric):
    def __init__(self, name='pmf_loss', **kwargs):
        super(PMFMINE_loss_metric, self).__init__(name=name, **kwargs)
        self.T = self.add_weight(name='t', initializer='zeros')
        self.global_counter = self.add_weight(name='n', initializer='zeros')


    def update_state(self, T, p, ind, **kwargs):
        batch_size = tf.reduce_prod(T.shape[:-1]).numpy()
        # n_ind = tf.cast(tf.expand_dims(tf.linspace(start=0, stop=batch_size - 1, num=batch_size), axis=-1),
        #                 'int64')
        # x_ind = tf.concat([n_ind, tf.cast(ind, 'int64')], axis=-1)
        n_ind = tf.cast(tf.expand_dims(np.linspace(start=0, stop=batch_size - 1, num=batch_size), axis=-1),
                        'int64')
        x_ind = tf.concat([n_ind, tf.cast(ind, 'int64')], axis=-1)
        p_loss = (T-tf.reduce_mean(T))*tf.math.log(tf.gather_nd(indices=x_ind, params=p))
        self.T.assign(self.T + tf.reduce_sum(p_loss))
        self.global_counter.assign(self.global_counter + tf.cast(tf.reduce_prod(T.shape[:-1]), dtype=tf.float64))

    def result(self):
        loss = self.T / self.global_counter
        return loss
