import numpy as np
import tensorflow as tf
from scipy.stats import bernoulli

class Ising_Data(object):
    def __init__(self, config):
        self.config = config
        self.p_x = 0.4503
        self.p_ch = 0.5
        self.batch_size = config.batch_size
        self.bptt = config.bptt
        self.ising_ch_logits = self.gen_logits(self.p_ch)
        # self.ising_x_logits = self.gen_logits(self.p_x)
        self.ising_x_logits = self.gen_logits(1-self.p_x)
        self.initialize_channel()

    def gen_logits(self,p):
        p_t = p * tf.ones(shape=[self.batch_size, 1])  # create logits for Ber(p) samples
        p_bar_t = tf.ones_like(p_t) - p_t
        logits = tf.math.log(
            tf.concat([p_bar_t, p_t], axis=1))
        return logits

    def gen_data(self):
        y_l = []
        x_l = []

        for t in range(self.bptt):
            self.encoder()
            x_l.append(self.x)
            self.channel()
            y_l.append(self.y)

        x = tf.concat(x_l, axis=1)
        y = tf.concat(y_l, axis=1)

        return x, y

    def initialize_channel(self):
        for step in [0,1]:
            self.encoder(step)
            self.channel(step)

    def encoder(self, step=None):
        if step == 0 :
            self.s_past = tf.zeros(shape=[self.batch_size, 1, 1], dtype='float64')  # s_0
            self.x = tf.cast(tf.random.categorical(logits=self.ising_x_logits, num_samples=1), dtype='float64')  # x_0
            self.x = tf.expand_dims(self.x, axis=-1)
        elif step == 1:
            # z = tf.cast(tf.random.categorical(logits=self.ising_x_logits, num_samples=1), dtype='float64')  # x_0
            # z = tf.expand_dims(z, axis=-1)
            # self.x = tf.math.floormod(self.x + z, 2)
            self.x = self.x
        else:
            z = tf.cast(tf.random.categorical(logits=self.ising_x_logits, num_samples=1), dtype='float64')  # x_0
            z = tf.expand_dims(z, axis=-1)
            x_p = tf.math.floormod(self.x + z, 2)

            x_new = tf.where(tf.equal(self.y, self.s_past), self.s, x_p)  # x(t) = f(y_{t-1}, s_{t-2})
            self.x = x_new

    def channel(self, step=None):
        z = tf.cast(tf.random.categorical(logits=self.ising_ch_logits, num_samples=1), dtype='float64')
        z = tf.expand_dims(z, axis=-1)
        if step == 0:
            self.s = self.s_past
        self.y = tf.where(tf.equal(z,0), self.x, self.s)
        self.s_past = self.s
        self.s = self.x


class Ising(object):
    def __init__(self, config):
        self.p_enc = 1-0.4503
        # self.p_enc = 0.4503
        self.p_ch = 0.5
        self.batch_size = config.batch_size
        self.bptt = config.bptt
        self.logits_enc = self.gen_logits(self.p_enc)
        self.logits_ch = self.gen_logits(self.p_ch)
        self.initialize()

    def initialize(self):
        # self.s_past = self.gen_ber(self.p_ch)  # gen s_0 as ber(0.5)
        # self.s = tf.zeros(shape=[self.batch_size,1,1])
        # self.y = self.channel_t(self.s)

        # self.s = self.gen_ber(self.logits_ch)
        # self.y = self.channel_t(tf.zeros(shape=[self.batch_size,1,1], dtype='float64'))
        # self.change_flag = tf.equal(self.s,self.s+tf.ones_like(self.s))  # define a logical true tensor as a beginning
        self.s_past = tf.zeros(shape=[self.batch_size,1,1],dtype='float64')
        self.s = tf.zeros(shape=[self.batch_size,1,1],dtype='float64')
        self.y = tf.zeros(shape=[self.batch_size,1,1],dtype='float64')
        self.flag = tf.equal(self.s,self.s+tf.ones_like(self.s))  # set initial flag to false

    def gen_logits(self, p):
        p_t = p * tf.ones(shape=[self.batch_size, 1])  # create logits for Ber(p) samples
        p_bar_t = tf.ones_like(p_t) - p_t
        logits = tf.math.log(
            tf.concat([p_bar_t, p_t], axis=1))
        return logits

    def channel_t(self, x):
        y_ber = self.gen_ber(self.logits_ch)  # generate ising output where x != s
        y = tf.where(tf.equal(x,self.s), x, y_ber)
        self.past_s = self.s
        self.s = x
        return y

    def enc_t(self):
        # flag_y = tf.equal(self.y, self.past_s)
        # self.change_flag = tf.math.logical_and(flag_y, tf.math.logical_not(self.change_flag))
        #
        # z = self.gen_ber(self.logits_enc)
        # x_bar = tf.math.floormod(self.s + z, 2)
        #
        # x = tf.where(self.change_flag, self.s, x_bar)

        flag_y = tf.equal(self.s_past, self.y)  # check if y_{t-1} = s_{t-2}
        cond = tf.math.logical_and(self.flag, flag_y)  # check the transmission condition
        self.flag = tf.math.logical_not(cond)  # new flag val is according to cond
        z = self.gen_ber(self.logits_enc)
        x_new = tf.where(tf.equal(z, 0),self.s, 1-self.s )
        # x_new = tf.math.floormod(self.s + z, 2)  # generate bitflip x for places where cond is 0

        x = tf.where(cond, self.s, x_new)  # set value of x_t
        return x

    def gen_data(self):
        x_l = []
        y_l = []

        for t in range(self.bptt):
            x_ = self.enc_t()
            x_l.append(x_)
            y_l.append(self.channel_t(x_))

        x = tf.concat(x_l, axis=1)
        y = tf.concat(y_l, axis=1)

        return x,y

    def gen_ber(self,logits):
        ber = tf.expand_dims(tf.cast(tf.random.categorical(logits=logits, num_samples=1), dtype='float64'),axis=-1)
        return ber


class Ising_seq(object):
    def __init__(self, config):
        self.p_enc = 1-0.4503
        self.p_ch = 0.5
        self.batch_size = config.batch_size
        self.bptt = config.bptt
        self.initialize()

    def initialize(self):
        self.s_past = 0
        self.s = 0
        self.y = 0
        self.flag = False

    def channel_t(self, x):
        y_ber = bernoulli.rvs(self.p_ch, size=1)  # generate ising output where x != s
        if self.s == x:
            y = x
        else:
            y = y_ber

        self.past_s = self.s
        self.s = x
        return y

    def enc_t(self):
        z = bernoulli.rvs(self.p_enc, size=1)
        x_new = (self.s + z) % 2
        if self.flag:
            if self.y == self.past_s:
                x = x_new
                self.flag = True
            else:
                x = self.s
                self.flag = False
        else:
            x = x_new
            self.flag = True

        return x

    def gen_data(self):
        x_b = []
        y_b = []
        for b in range(self.batch_size):
            x_l = []
            y_l = []
            for t in range(self.bptt):
                x_ = self.enc_t()
                x_l.append(np.expand_dims(x_,axis=0))
                y_l.append(np.expand_dims(self.channel_t(x_),axis=0))

            x = np.concatenate(x_l, axis=1)
            y = np.concatenate(y_l, axis=1)
            x_b.append(np.expand_dims(x, axis=2))
            y_b.append(np.expand_dims(y, axis=2))

        x =np.concatenate(x_b, axis=0)
        y =np.concatenate(y_b, axis=0)

        return x,y


class IsingChannel_ziv(object):
    def __init__(self, input_shape=(1000, 1), dtype=tf.int64, **kwargs):
        self.shape = input_shape
        self.dtype = dtype
        self.logits = tf.concat([0.4503 * tf.ones(shape=input_shape), (1 - 0.4503) * tf.ones(shape=input_shape)], axis=1)

    def _iter_(self):
        batch_size = self.shape[0]
        cur_logits = tf.gather_nd(self.logits,
                                  tf.stack((tf.range(batch_size, dtype=tf.int64), tf.squeeze(tf.cast(self.s,dtype='int64'))), axis=1))
        cur_logits = tf.stack([cur_logits, 1 - cur_logits], axis=1)
        new_symbol = tf.random.categorical(logits=cur_logits, num_samples=1, dtype=tf.int64)
        # dor:
        # new_symbol = tf.cast(tf.expand_dims(new_symbol, axis=-1),dtype='float64')
        #
        x = tf.where(tf.equal(self.q, 0), self.s, new_symbol)
        channel_noise = tf.random.uniform(shape=self.shape, minval=0, maxval=2, dtype=tf.int64)
        # dor:
        # channel_noise = tf.expand_dims(channel_noise, axis=-1)
        #
        y = tf.where(tf.equal(channel_noise, 1), x, self.s)
        s_plus = x
        q_plus = tf.where(tf.equal(self.q, 1),
                          tf.where(tf.equal(self.s, y), tf.constant(0, tf.int64), tf.constant(1, tf.int64)),
                          tf.constant(1, tf.int64))
        # ziv:
        # yield x, y
        # self.s = s_plus
        # self.q = q_plus
        # dor:
        self.s = s_plus
        self.q = q_plus
        return [x, y]

    def _call_(self):
        self.s = tf.zeros(shape=self.shape, dtype=tf.int64)
        self.q = tf.ones(shape=self.shape, dtype=tf.int64)
        # dor:
        # self.s = tf.cast(tf.expand_dims(self.s, axis=-1),dtype='float64')
        # self.q = tf.cast(tf.expand_dims(self.q, axis=-1), dtype='float64')
        #
        return self

    def _gen_(self):
        x_l = []
        y_l = []
        for t in range(5):
            out = self._iter_()
            x_l.append(out[0])
            y_l.append(out[1])
        x_l = tf.expand_dims(tf.concat(x_l, axis=1),axis=-1)
        y_l = tf.expand_dims(tf.concat(y_l, axis=1),axis=-1)

        return x_l,y_l

class IsingChannel_state(object):
    def __init__(self, bptt, input_shape=(1000, 1), dtype=tf.int64, **kwargs):
        self.bptt = bptt
        self.shape = input_shape
        self.dtype = dtype
        self.logits = tf.concat([0.4503 * tf.ones(shape=input_shape), (1 - 0.4503) * tf.ones(shape=input_shape)], axis=1)

    def _iter_(self):
        batch_size = self.shape[0]
        cur_logits = tf.gather_nd(self.logits,
                                  tf.stack((tf.range(batch_size, dtype=tf.int64), tf.squeeze(tf.cast(self.s,dtype='int64'))), axis=1))
        cur_logits = tf.stack([cur_logits, 1 - cur_logits], axis=1)
        new_symbol = tf.random.categorical(logits=cur_logits, num_samples=1, dtype=tf.int64)
        # dor:
        # new_symbol = tf.cast(tf.expand_dims(new_symbol, axis=-1),dtype='float64')
        #
        x = tf.where(tf.equal(self.q, 0), self.s, new_symbol)
        channel_noise = tf.random.uniform(shape=self.shape, minval=0, maxval=2, dtype=tf.int64)
        # dor:
        # channel_noise = tf.expand_dims(channel_noise, axis=-1)
        #
        y = tf.where(tf.equal(channel_noise, 1), x, self.s)
        s_plus = x
        q_plus = tf.where(tf.equal(self.q, 1),
                          tf.where(tf.equal(self.s, y), tf.constant(0, tf.int64), tf.constant(1, tf.int64)),
                          tf.constant(1, tf.int64))
        # ziv:
        # yield x, y
        # self.s = s_plus
        # self.q = q_plus
        # dor:
        self.s = s_plus
        self.q = q_plus
        return [x, y]

    def _call_(self):
        self.s = tf.zeros(shape=self.shape, dtype=tf.int64)
        self.q = tf.ones(shape=self.shape, dtype=tf.int64)
        # dor:
        # self.s = tf.cast(tf.expand_dims(self.s, axis=-1),dtype='float64')
        # self.q = tf.cast(tf.expand_dims(self.q, axis=-1), dtype='float64')
        #
        return self

    def _gen_(self):
        x_l = []
        y_l = []
        for t in range(self.bptt):
            out = self._iter_()
            x_l.append(out[0])
            y_l.append(out[1])
        x_l = tf.expand_dims(tf.concat(x_l, axis=1),axis=-1)
        y_l = tf.expand_dims(tf.concat(y_l, axis=1),axis=-1)

        return x_l,y_l