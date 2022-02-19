import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
import logging
import os
from losses.dv_loss import Model_loss
from metrics.metrics import ModelWithEncMetrics, ModelMetrics
from trainers.savers import ContDINESaver
from optimizers.lr import exp_dec_lr


logger = logging.getLogger("logger")


class DI_Trainer_NDT(object):
    def __init__(self, model, data, config):
        """
        Constructor of trainer class, inherits from DINumericalTrainer
        :param model: a dictionary consisting of all models relevant to this trainer
        :param data: data loader instance, not used in this class.
        :param config: configuration parameters obtains for the relevant .json file
        """
        self.model = model  # the DINE model
        self.data_iterators = data  # the data generator
        self.config = config  # the configuration
        self.loss_fn = Model_loss()  # the DINE model loss function
        self.learning_rate = config.lr  # the model's learning rate
        if config.decay == 1:
            self.learning_rate_dv = exp_dec_lr(config, data, config.lr)
            self.learning_rate_ndt = exp_dec_lr(config, data, config.lr/4)
        else:
            self.learning_rate_dv = config.lr
            self.learning_rate_ndt = config.lr/4
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.bptt = config.bptt
        if config.optimizer == "adam":
            self.optimizer = {"dv_y": Adam(amsgrad=True, learning_rate=self.learning_rate_dv),
                              "dv_xy": Adam(amsgrad=True, learning_rate=self.learning_rate_dv),
                              "enc": Adam(amsgrad=True, learning_rate=self.learning_rate_ndt)}  # the model's optimizers
        else:
            self.optimizer = {"dv_y": SGD(learning_rate=config.lr_SGD),
                              "dv_xy": SGD(learning_rate=config.lr_SGD),
                              "ndt": SGD(learning_rate=config.lr_SGD/4)
                              }  # the model's optimizers

        self.saver = ContDINESaver(config)
        self.global_step_enc = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.metrics = {"train": ModelMetrics(config.train_writer, name='dv_train'),
                        "eval": ModelMetrics(config.test_writer, name='dv_eval')}  # the trainer's metrics
        self.feedback = config.feedback

    def train_epoch(self, epoch):
        self.metrics["train"].reset_states()
        self.reset_model_states()  # reset model and metrics states
        if np.random.rand() > 0 or epoch < 200:  # train DINE
            model_name = "DV epoch"
            for sample in self.data_iterators["train"]():
                output = self.train_dine_step(sample)  # calculate model outputs and perform a training step
                self.metrics["train"].update_state(output[0], output[1])  # update trainer metrics
        else:  # train Enc.
            model_name = "Encoder epoch"
            for sample in self.data_iterators["train"]():
                output = self.train_enc_step(sample)  # calculate model outputs and perform a training step
                self.metrics["train"].update_state(output[0], output[1])  # update trainer metrics

        self.metrics["train"].log_metrics(epoch, model_name=model_name)  # log updated metrics

    def train_dine_step(self, sample):
        gradients_dv_y, gradients_dv_xy, t_y, t_xy = self.compute_dine_grads(sample)  # calculate gradients
        self.apply_grads(gradients_dv_y, gradients_dv_xy)  # apply gradients
        return [t_y, t_xy]

    def train_enc_step(self, sample):
        # gradients_enc, t_y, t_xy, p = self.compute_enc_grads(sample)  # calculate gradients
        gradients_enc, t_y, t_xy = self.compute_enc_grads(sample)  # calculate gradients
        self.apply_enc_grads(gradients_enc)  # apply gradients
        return [t_y, t_xy]

    # @tf.function
    def compute_dine_grads(self, sample):  # gradients computation method
        with tf.GradientTape(persistent=True) as tape:
            model_in = [sample, tf.convert_to_tensor(np.zeros([self.config.batch_size, 1, 2*self.config.x_dim]))]
            # [x, y] = self.model['enc'](model_in, training=False)  # obtain samples through model
            ## for debug:
            x = tf.random.normal(shape=[self.config.batch_size, self.config.bptt, 1],mean=0,stddev=1)
            y = x + tf.random.normal(shape=[self.config.batch_size, self.config.bptt, 1],mean=0,stddev=1)
            ##
            xy = tf.concat([x, y], axis=-1)  # t_xy model input
            t_y = self.model['dv']['y'](y, training=True)
            t_xy = self.model['dv']['xy'](xy, training=True)  # calculate models outputs
            loss = self.loss_fn(t_y, t_xy)  # calculate loss for each model
            loss_y, loss_xy = tf.split(loss, num_or_size_splits=2, axis=0)

        gradients_dv_y = tape.gradient(loss_y, self.model['dv']['y'].trainable_weights)
        gradients_dv_xy = tape.gradient(loss_xy, self.model['dv']['xy'].trainable_weights)  # calculate gradients

        gradients_dv_y, grad_norm_dv_y = tf.clip_by_global_norm(gradients_dv_y, self.config.clip_grad_norm)
        gradients_dv_xy, grad_norm_dv_xy = tf.clip_by_global_norm(gradients_dv_xy, self.config.clip_grad_norm)  # normalize gradients

        with self.config.train_writer.as_default():  # update trainer metrics
            tf.summary.scalar("grad_norm_dv_y", grad_norm_dv_y, self.global_step)
            tf.summary.scalar("grad_norm_dv_xy", grad_norm_dv_xy, self.global_step)
            self.global_step.assign_add(1)

        # if self.feedback:
        #     x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
        #     y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
        #     self.fb_input = tf.concat([x_, y_], axis=-1)
        # else:
        #     x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
        #     self.fb_input = x_


        return gradients_dv_y, gradients_dv_xy, t_y, t_xy

    # @tf.function
    def compute_enc_grads(self, sample):  # gradients computation method
        with tf.GradientTape(persistent=True) as tape:
            model_in = [sample, tf.convert_to_tensor(np.zeros([self.config.batch_size, 1, 2 * self.config.x_dim]))]
            [x, y] = self.model['enc'](model_in, training=True)  # obtain samples through model
            xy = tf.concat([x, y], axis=-1)  # t_xy model input
            t_y = self.model['dv']['y'](y, training=False)
            t_xy = self.model['dv']['xy'](xy, training=False)  # calculate models outputs
            loss = self.loss_fn(t_y, t_xy)  # calculate loss for each model

        gradients_enc = tape.gradient(loss, self.model['enc'].trainable_weights)

        gradients_enc, grad_norm_enc = tf.clip_by_global_norm(gradients_enc, self.config.clip_grad_norm)

        with self.config.train_writer.as_default():  # update trainer metrics
            tf.summary.scalar("grad_norm_enc", grad_norm_enc, self.global_step)
            self.global_step.assign_add(1)

        # if self.feedback:
        #     x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
        #     y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
        #     self.fb_input = tf.concat([x_, y_], axis=-1)
        # else:
        #     x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
        #     self.fb_input = x_

        return gradients_enc, t_y, t_xy

    @tf.function
    def apply_enc_grads(self, gradients_enc):
        self.optimizer['enc'].apply_gradients(zip(gradients_enc, self.model['enc'].trainable_weights))

    def sync_eval_model(self):
        # sync DV:
        w_y = self.model['dv']['y'].get_weights()
        w_xy = self.model['dv']['xy'].get_weights()
        self.model['dv_eval']['y'].set_weights(w_y)
        self.model['dv_eval']['xy'].set_weights(w_xy)
        # sync enc:
        w_enc = self.model['enc'].get_weights()  # similarly sync encoder model
        self.model['enc_eval'].set_weights(w_enc)

    def evaluate(self, epoch, iterator="eval"):
        self.sync_eval_model()
        self.metrics["eval"].reset_states()
        self.reset_model_states()
        self.saver.reset_state()

        x = []
        t_y = []


        for input_batch in self.data_iterators[iterator]():
            output = self.eval_step(input_batch)
            x.append(output[2])  #
            t_y.append(output[0][0])
            self.metrics["eval"].update_state(output[0], output[1])

        self.metrics["eval"].log_metrics(epoch, model_name="Evaluation")
        x_t = tf.concat(x, axis=1)
        t_y_t = tf.concat(t_y, axis=1)
        with self.config.test_writer.as_default():
            tf.summary.histogram(name="x_hist", data=x_t, step=self.global_step, buckets=100)
            tf.summary.histogram(name="t_y_hist", data=t_y_t, step=self.global_step, buckets=100)

        self.saver.save(None if epoch > 0 else 'begin.mat')
        if self.config.trainer_name == "di_with_enc_states":
            self.saver.save_models(models=self.model, path=os.path.join(self.config.tensor_board_dir, "models_weights"))

    # @tf.function
    def eval_step(self, input_batch):
        model_in = [input_batch, tf.convert_to_tensor(np.zeros([self.config.batch_size, 1, 2*self.config.x_dim]))]
        [x, y] = self.model['enc'](model_in, training=False)
        # ## for debug:
        # x = tf.random.normal(shape=[self.config.batch_size, self.config.bptt, 1], mean=0, stddev=1)
        # y = x + tf.random.normal(shape=[self.config.batch_size, self.config.bptt, 1], mean=0, stddev=1)
        # ##
        xy = tf.concat([x, y], axis=-1)
        t_y = self.model['dv_eval']['y'](y, training=False)
        t_xy = self.model['dv_eval']['xy'](xy, training=False)
        # addition with states (now output is [t,et,s]:
        t_y = [t_y[0], t_y[1]]
        t_xy = [t_xy[0], t_xy[1]]

        self.saver.update_state(t_y, t_xy, x, y)

        # if self.feedback:
        #     x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
        #     y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
        #     self.fb_input = tf.concat([x_, y_], axis=-1)
        # else:
        #     x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
        #     self.fb_input = x_

        return t_y, t_xy, x

    def train(self):
        for epoch in range(self.config.num_epochs):
            if epoch % self.config.eval_freq == 0:
                self.evaluate(epoch)  # perform evaluation step
                continue
            self.train_epoch(epoch)  # perform a training epoch

        self.evaluate(self.config.num_epochs, iterator="long_eval")  # perform a long evaluation

    def reset_model_states(self):
        def reset_recursively(models):
            for model in models.values():
                if isinstance(model, dict):
                    reset_recursively(model)
                else:
                    model.reset_states()

        reset_recursively(self.model)
        if self.config.channel_name != "awgn":
            self.model['enc'].layers[3].reset_states()
            self.model['enc_eval'].layers[3].reset_states()

    @tf.function
    def apply_grads(self, gradients_dv_y, gradients_dv_xy):
        self.optimizer['dv_y'].apply_gradients(zip(gradients_dv_y, self.model['dv']['y'].trainable_weights))
        self.optimizer['dv_xy'].apply_gradients(zip(gradients_dv_xy, self.model['dv']['xy'].trainable_weights))

