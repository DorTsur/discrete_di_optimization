import numpy as np
import tensorflow as tf

class Clean_Channel(object):
    def __init__(self,config):
        self.bptt = config.bptt
        self.batch_size = config.batch_size
        self.gen_logits()

    def gen_data(self):
        x = tf.cast(tf.random.categorical(logits=self.logits, num_samples=1), dtype='float64')
        x = tf.reshape(x, shape=[self.batch_size,self.bptt,1])
        y=x
        return x,y

    def gen_logits(self):
        p_t = 0.5*tf.ones(shape=[self.batch_size*self.bptt,1])
        p_bar_t = tf.ones_like(p_t) - p_t
        self.logits = tf.math.log(
            tf.concat([p_bar_t, p_t], axis=1))