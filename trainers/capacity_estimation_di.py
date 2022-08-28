import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
import os
from losses.dv_loss import Model_loss
from losses.enc_loss import Model_enc_loss
from metrics.metrics import ModelWithEncMetrics
from trainers.savers import DVEncoderVisualizer_perm1
from optimizers.lr import exp_dec_lr


class CapEstDI(object):
    def __init__(self, model, data, config):
        """
        Constructor of trainer class, for the purpose of optimizing and estimating DI
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
            self.learning_rate_enc = exp_dec_lr(config, data, config.lr/4)
        else:
            self.learning_rate_dv = config.lr
            self.learning_rate_enc = config.lr/4
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.bptt = config.bptt
        if config.optimizer == "adam":
            self.optimizer = {"dv_y": Adam(amsgrad=True, learning_rate=self.learning_rate_dv),
                              "dv_xy": Adam(amsgrad=True, learning_rate=self.learning_rate_dv),
                              "enc": Adam(amsgrad=True, learning_rate=self.learning_rate_enc)}  # the model's optimizers
        else:
            self.optimizer = {"dv_y": SGD(learning_rate=config.lr_SGD),
                              "dv_xy": SGD(learning_rate=config.lr_SGD),
                              "enc": SGD(learning_rate=config.lr_SGD/4)
                              }  # the model's optimizers
        self.saver = DVEncoderVisualizer_perm1(config)
        self.enc_loss_fn = Model_enc_loss(subtract=True)
        self.global_step_enc = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.metrics = {"train": ModelWithEncMetrics(config.train_writer, name='dv_train'),
                        "eval": ModelWithEncMetrics(config.test_writer, name='dv_eval')}  # the trainer's metrics
        self.feedback = (config.feedback == 1)
        self.T = config.T
        if self.feedback:
            self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim + config.y_dim])
        else:
            self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim])

    def train_epoch(self, epoch):
        self.metrics["train"].reset_states()
        self.reset_model_states()  # reset model and metrics states
        if np.random.rand() > 0.3 or epoch < 20:  # train DINE
            model_name = "DV epoch"

            for _ in range(self.config.batches):
                output = self.train_dine_step(self.fb_input)  # calculate model outputs and perform a training step
                self.metrics["train"].update_state(output)  # update trainer metrics
        else:  # train Enc.
            model_name = "Encoder epoch"
            for _ in range(self.config.batches):
                output = self.train_enc_step(self.fb_input)  # calculate model outputs and perform a training step
                self.metrics["train"].update_state(output)  # update trainer metrics

        self.metrics["train"].log_metrics(epoch, model_name=model_name)  # log updated metrics

    def train_dine_step(self, sample):
        gradients_dv_y, gradients_dv_xy, t_y, t_xy, p = self.compute_dine_grads(sample)  # calculate gradients
        self.apply_grads(gradients_dv_y, gradients_dv_xy)  # apply gradients
        return [t_y, t_xy, p]

    def train_enc_step(self, sample):
        gradients_enc, t_y, t_xy, p = self.compute_pmf_grads_reinforce(sample)  # calculate gradients
        self.apply_pmf_grads(gradients_enc)  # apply gradients
        return [t_y, t_xy, p]

    # @tf.function
    def compute_dine_grads(self, sample):  # gradients computation method
        with tf.GradientTape(persistent=True) as tape:
            [x, y, p] = self.model['enc'](sample, training=False)  # obtain samples through model
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

        if self.feedback:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = tf.concat([x_, y_], axis=-1)
        else:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = x_


        return gradients_dv_y, gradients_dv_xy, t_y, t_xy, p

    def compute_pmf_grads_reinforce(self, sample):  # gradients computation method
        with tf.GradientTape(persistent=True) as tape:
            [x, y, p] = self.model['enc'](sample, training=True)  # obtain samples through model
            xy = tf.concat([x, y], axis=-1)  # t_xy model input
            t_y = self.model['dv']['y'](y, training=False)
            t_xy = self.model['dv']['xy'](xy, training=False)  # calculate models outputs
            #### reinforce idea ####
            lp = self.enc_lp(x,p)
            r = t_xy[0] - t_y[0]
            r = tf.squeeze(r, axis=-1)
            rho = tf.reduce_mean(r)
            T = self.T
            sum_list = [tf.reduce_sum(tf.slice(r, [0,k,0], [tf.shape(r)[0],T,tf.shape(r)[-1]]), axis=1, keepdims=True)
                        for k in range(tf.shape(r)[1]-T)]  # [B,1,1]
            rt = tf.concat(sum_list, axis=1)   # [B,bptt-T,1]
            [lp,_] = tf.split(lp, axis=1, num_or_size_splits=[self.bptt-T,T])
            loss = -tf.reduce_mean(lp*(rt-rho))
            ##### reinforce idea #####

        gradients_enc = tape.gradient(loss, self.model['enc'].trainable_weights)

        gradients_enc, grad_norm_enc = tf.clip_by_global_norm(gradients_enc, self.config.clip_grad_norm_enc)

        with self.config.train_writer.as_default():  # update trainer metrics
            tf.summary.scalar("grad_norm_enc", grad_norm_enc, self.global_step)
            tf.summary.scalar("mdp_loss", loss, self.global_step)
            self.global_step.assign_add(1)

        if self.feedback:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = tf.concat([x_, y_], axis=-1)
        else:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = x_

        return gradients_enc, t_y, t_xy, p

    def enc_lp(self, x, p):
        p_out = tf.math.log(tf.where(tf.equal(x, 1), p, 1-p))
        return p_out

    @tf.function
    def apply_pmf_grads(self, gradients_enc):
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
        p_t = []
        t_y_t = []
        states = []

        if iterator == "eval":
            eval_batches = 10
        elif iterator == "long_eval":
            eval_batches = 100
        else:
            eval_batches = 1


        for _ in range(self.config.batches*eval_batches):
            input_batch = self.fb_input
            output = self.eval_step(input_batch)
            p_t.append(output[2])
            t_y_t.append(output[0][0])
            states.append(output[3])
            self.metrics["eval"].update_state(output[:-1])

        self.metrics["eval"].log_metrics(epoch, model_name="Evaluation")
        p_t = tf.concat(p_t, axis=1)
        t_y_t = tf.concat(t_y_t, axis=1)
        states = tf.concat(states, axis=1)
        with self.config.test_writer.as_default():
            tf.summary.histogram(name="p_hist", data=p_t, step=self.global_step, buckets=20)
            tf.summary.histogram(name="t_y_hist", data=t_y_t, step=self.global_step, buckets=20)
            tf.summary.histogram(name="states_y_hist", data=states, step=self.global_step, buckets=20)

        self.saver.save(None if epoch > 0 else 'begin.mat')
        self.saver.save_models(models=self.model, path=os.path.join(self.config.tensor_board_dir, "models_weights"))

    # @tf.function
    def eval_step(self, input_batch):
        [x, y, p] = self.model['enc'](input_batch, training=False)
        xy = tf.concat([x, y], axis=-1)
        t_y = self.model['dv_eval']['y'](y, training=False)
        t_xy = self.model['dv_eval']['xy'](xy, training=False)
        # addition with states (now output is [t,et,s]:
        states_y = t_y[2]
        t_y = [t_y[0], t_y[1]]
        t_xy = [t_xy[0], t_xy[1]]

        self.saver.update_state(t_y, t_xy, x, y, p, states_y)

        if self.feedback:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = tf.concat([x_, y_], axis=-1)
        else:
            x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = x_

        return t_y, t_xy, p, states_y

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
        if (self.config.channel_name == "ising") or (self.config.channel_name == "trapdoor"):
            self.model['enc'].layers[3].reset_states()
            self.model['enc_eval'].layers[3].reset_states()

    @tf.function
    def apply_grads(self, gradients_dv_y, gradients_dv_xy):
        self.optimizer['dv_y'].apply_gradients(zip(gradients_dv_y, self.model['dv']['y'].trainable_weights))
        self.optimizer['dv_xy'].apply_gradients(zip(gradients_dv_xy, self.model['dv']['xy'].trainable_weights))


class CapEstDIChkpt(CapEstDI):
    def __init__(self, model, data, config):
        super().__init__(model, data, config)
        self.load_enc_weights()

    def load_enc_weights(self):
        """
        model to set weights is encoder
        """
        filepath = os.path.join("./pretrained_models", self.config.channel_name + "_ff", "1", "weights")
        self.model['enc'].load_weights(filepath)




# Old implementation without inheritance:
# class CapEstDIChkpt(object):
#     def __init__(self, model, data, config):
#         """
#         Constructor of trainer class, for the purpose of training DI from checkpoint models.
#         :param model: a dictionary consisting of all models relevant to this trainer
#         :param data: data loader instance, not used in this class.
#         :param config: configuration parameters obtains for the relevant .json file
#         """
#         self.model = model  # the DINE model
#         self.data_iterators = data  # the data generator
#         self.config = config  # the configuration
#         self.load_enc_weights()
#
#         self.loss_fn = Model_loss()  # the DINE model loss function
#         self.learning_rate = config.lr  # the model's learning rate
#         if config.decay == 1:
#             self.learning_rate_dv = exp_dec_lr(config, data, config.lr)
#             self.learning_rate_enc = exp_dec_lr(config, data, config.lr/4)
#         else:
#             self.learning_rate_dv = config.lr
#             self.learning_rate_enc = config.lr/4
#         self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
#         self.bptt = config.bptt
#         if config.optimizer == "adam":
#             self.optimizer = {"dv_y": Adam(amsgrad=True, learning_rate=self.learning_rate_dv),
#                               "dv_xy": Adam(amsgrad=True, learning_rate=self.learning_rate_dv),
#                               "enc": Adam(amsgrad=True, learning_rate=self.learning_rate_enc)}  # the model's optimizers
#         else:
#             self.optimizer = {"dv_y": SGD(learning_rate=config.lr_SGD),
#                               "dv_xy": SGD(learning_rate=config.lr_SGD),
#                               "enc": SGD(learning_rate=config.lr_SGD/4)
#                               }  # the model's optimizers
#         self.saver = DVEncoderVisualizer_perm1(config)
#         self.enc_loss_fn = Model_enc_loss(subtract=True)
#         self.global_step_enc = tf.Variable(0, trainable=False, dtype=tf.int64)
#         self.metrics = {"train": ModelWithEncMetrics(config.train_writer, name='dv_train'),
#                         "eval": ModelWithEncMetrics(config.test_writer, name='dv_eval')}  # the trainer's metrics
#         self.feedback = (config.feedback == 1)
#         self.T = config.T
#         if self.feedback:
#             self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim + config.y_dim])
#         else:
#             self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim])
#
#     def load_enc_weights(self):
#         """
#         model to set weights is encoder
#         """
#         filepath = os.path.join("./pretrained_models", self.config.channel_name + "_ff", "1", "weights")
#         self.model['enc'].load_weights(filepath)
#
#     def train_epoch(self, epoch):
#         self.metrics["train"].reset_states()
#         self.reset_model_states()  # reset model and metrics states
#         i = 0
#         if np.random.rand() > 0.3 or epoch < 20:  # train DINE
#             model_name = "DV epoch"
#             for sample in self.data_iterators["train"]():
#                 if i >= 0:
#                     sample = self.fb_input
#                 output = self.train_dine_step(sample)  # calculate model outputs and perform a training step
#                 self.metrics["train"].update_state(output)  # update trainer metrics
#         else:  # train Enc.
#             model_name = "Encoder epoch"
#             for sample in self.data_iterators["train"]():
#                 if sample is None:
#                     self.reset_model_states()  # in case we deal with a dataset
#                     continue
#                 output = self.train_enc_step(self.fb_input)  # calculate model outputs and perform a training step
#                 self.metrics["train"].update_state(output)  # update trainer metrics
#
#         self.metrics["train"].log_metrics(epoch, model_name=model_name)  # log updated metrics
#
#     def train_dine_step(self, sample):
#         gradients_dv_y, gradients_dv_xy, t_y, t_xy, p = self.compute_dine_grads(sample)  # calculate gradients
#         self.apply_grads(gradients_dv_y, gradients_dv_xy)  # apply gradients
#         return [t_y, t_xy, p]
#
#     def train_enc_step(self, sample):
#         gradients_enc, t_y, t_xy, p = self.compute_pmf_grads_reinforce(sample)  # calculate gradients
#         self.apply_pmf_grads(gradients_enc)  # apply gradients
#         return [t_y, t_xy, p]
#
#     # @tf.function
#     def compute_dine_grads(self, sample):  # gradients computation method
#         with tf.GradientTape(persistent=True) as tape:
#             [x, y, p] = self.model['enc'](sample, training=False)  # obtain samples through model
#             # ## for debug:
#             # p = 0.7 * tf.ones(shape=[self.config.batch_size, self.config.bptt, 1])
#             # x, y = self.model['sampler'](p, training=False)
#             # ##
#             xy = tf.concat([x, y], axis=-1)  # t_xy model input
#             t_y = self.model['dv']['y'](y, training=True)
#             t_xy = self.model['dv']['xy'](xy, training=True)  # calculate models outputs
#             loss = self.loss_fn(t_y, t_xy)  # calculate loss for each model
#             loss_y, loss_xy = tf.split(loss, num_or_size_splits=2, axis=0)
#
#         gradients_dv_y = tape.gradient(loss_y, self.model['dv']['y'].trainable_weights)
#         gradients_dv_xy = tape.gradient(loss_xy, self.model['dv']['xy'].trainable_weights)  # calculate gradients
#
#         gradients_dv_y, grad_norm_dv_y = tf.clip_by_global_norm(gradients_dv_y, self.config.clip_grad_norm)
#         gradients_dv_xy, grad_norm_dv_xy = tf.clip_by_global_norm(gradients_dv_xy, self.config.clip_grad_norm)  # normalize gradients
#
#         with self.config.train_writer.as_default():  # update trainer metrics
#             tf.summary.scalar("grad_norm_dv_y", grad_norm_dv_y, self.global_step)
#             tf.summary.scalar("grad_norm_dv_xy", grad_norm_dv_xy, self.global_step)
#             tf.summary.scalar("x_mean", tf.math.reduce_mean(x), self.global_step)
#             tf.summary.scalar("y_mean", tf.math.reduce_mean(y), self.global_step)
#             self.global_step.assign_add(1)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = x_
#
#
#         return gradients_dv_y, gradients_dv_xy, t_y, t_xy, p
#
#     def compute_pmf_grads_reinforce(self, sample):  # gradients computation method
#         with tf.GradientTape(persistent=True) as tape:
#             [x, y, p] = self.model['enc'](sample, training=True)  # obtain samples through model
#             xy = tf.concat([x, y], axis=-1)  # t_xy model input
#             t_y = self.model['dv']['y'](y, training=False)
#             t_xy = self.model['dv']['xy'](xy, training=False)  # calculate models outputs
#             #### reinforce idea ####
#             lp = self.enc_lp(x,p)
#             r = t_xy[0] - t_y[0]
#             r = tf.squeeze(r, axis=-1)
#             rho = tf.reduce_mean(r)
#             T = self.T
#             sum_list = [tf.reduce_sum(tf.slice(r, [0,k,0], [tf.shape(r)[0],T,tf.shape(r)[-1]]), axis=1, keepdims=True)
#                         for k in range(tf.shape(r)[1]-T)]  # [B,1,1]
#             rt = tf.concat(sum_list, axis=1)   # [B,bptt-T,1]
#             [lp,_] = tf.split(lp, axis=1, num_or_size_splits=[self.bptt-T,T])
#             loss = -tf.reduce_mean(lp*(rt-rho))
#             ##### reinforce idea #####
#
#         gradients_enc = tape.gradient(loss, self.model['enc'].trainable_weights)
#
#         gradients_enc, grad_norm_enc = tf.clip_by_global_norm(gradients_enc, self.config.clip_grad_norm_enc)
#
#         with self.config.train_writer.as_default():  # update trainer metrics
#             tf.summary.scalar("grad_norm_enc", grad_norm_enc, self.global_step)
#             tf.summary.scalar("mdp_loss", loss, self.global_step)
#             self.global_step.assign_add(1)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = x_
#
#         return gradients_enc, t_y, t_xy, p
#
#     # @tf.function
#     def calc_enc_loss_input(self, t_l, x, p):
#         p_et = tf.stack(self.config.contrastive_duplicates * [p], axis=2)
#         p_bar_et = tf.ones_like(p_et) - p_et
#         p_t = tf.expand_dims(p, axis=2)
#         p_bar_t = tf.ones_like(p_t) - p_t  # obtain p_bar values
#         [t, et] = t_l
#         et_mean = tf.reduce_mean(et)
#
#         t_0, t_1 = t * tf.math.log(p_bar_t), t * tf.math.log(p_t)  # create tensor to draw values from
#         et_0, et_1 = et * tf.math.log(p_bar_et), et * tf.math.log(p_et)
#
#         x_t = tf.expand_dims(x, axis=2)
#         x_et = tf.stack(self.config.contrastive_duplicates * [x], axis=2)
#
#         t_enc = tf.where(tf.equal(x_t, 1), t_1, t_0)  # choose according to binary criteria
#         et_enc = tf.where(tf.equal(x_et, 1), et_1, et_0)
#
#         return [t_enc, et_enc / et_mean]
#
#     def enc_lp(self, x, p):
#         p_out = tf.math.log(tf.where(tf.equal(x,1), p, 1-p))
#         return p_out
#
#     @tf.function
#     def apply_pmf_grads(self, gradients_enc):
#         self.optimizer['enc'].apply_gradients(zip(gradients_enc, self.model['enc'].trainable_weights))
#
#     def sync_eval_model(self):
#         # sync DV:
#         w_y = self.model['dv']['y'].get_weights()
#         w_xy = self.model['dv']['xy'].get_weights()
#         self.model['dv_eval']['y'].set_weights(w_y)
#         self.model['dv_eval']['xy'].set_weights(w_xy)
#         # sync enc:
#         w_enc = self.model['enc'].get_weights()  # similarly sync encoder model
#         self.model['enc_eval'].set_weights(w_enc)
#
#     def evaluate(self, epoch, iterator="eval"):
#         self.sync_eval_model()
#         self.metrics["eval"].reset_states()
#         self.reset_model_states()
#         self.saver.reset_state()
#         i=0
#         p_t = []
#         t_y_t = []
#         states = []
#
#         for input_batch in self.data_iterators[iterator]():
#             input_batch = self.fb_input
#             output = self.eval_step(input_batch)
#             p_t.append(output[2])
#             t_y_t.append(output[0][0])
#             states.append(output[3])
#             self.metrics["eval"].update_state(output[:-1])
#
#         self.metrics["eval"].log_metrics(epoch, model_name="Evaluation")
#         p_t = tf.concat(p_t, axis=1)
#         t_y_t = tf.concat(t_y_t, axis=1)
#         states = tf.concat(states, axis=1)
#         with self.config.test_writer.as_default():
#             tf.summary.histogram(name="p_hist", data=p_t, step=self.global_step, buckets=20)
#             tf.summary.histogram(name="t_y_hist", data=t_y_t, step=self.global_step, buckets=20)
#             tf.summary.histogram(name="states_y_hist", data=states, step=self.global_step, buckets=20)
#
#         self.saver.save(None if epoch > 0 else 'begin.mat')
#         self.saver.save_models(models=self.model, path=os.path.join(self.config.tensor_board_dir, "models_weights"))
#
#     # @tf.function
#     def eval_step(self, input_batch):
#         [x, y, p] = self.model['enc'](input_batch, training=False)
#         xy = tf.concat([x, y], axis=-1)
#         t_y = self.model['dv_eval']['y'](y, training=False)
#         t_xy = self.model['dv_eval']['xy'](xy, training=False)  # output is [t,et,s]
#         states_y = t_y[2]
#         t_y = [t_y[0], t_y[1]]
#         t_xy = [t_xy[0], t_xy[1]]
#
#         self.saver.update_state(t_y, t_xy, x, y, p, states_y)
#
#         if self.feedback:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = tf.concat([x_, y_], axis=-1)
#         else:
#             x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
#             self.fb_input = x_
#
#         return t_y, t_xy, p, states_y
#
#     def train(self):
#         for epoch in range(self.config.num_epochs):
#             if epoch % self.config.eval_freq == 0:
#                 self.evaluate(epoch)  # perform evaluation step
#                 continue
#             self.train_epoch(epoch)  # perform a training epoch
#
#         self.evaluate(self.config.num_epochs, iterator="long_eval")  # perform a long evaluation
#
#     def reset_model_states(self):
#         def reset_recursively(models):
#             for model in models.values():
#                 if isinstance(model, dict):
#                     reset_recursively(model)
#                 else:
#                     model.reset_states()
#
#         reset_recursively(self.model)
#         if (self.config.channel_name == "ising") or (self.config.channel_name == "trapdoor"):
#             self.model['enc'].layers[3].reset_states()
#             self.model['enc_eval'].layers[3].reset_states()
#
#     @tf.function
#     def apply_grads(self, gradients_dv_y, gradients_dv_xy):
#         self.optimizer['dv_y'].apply_gradients(zip(gradients_dv_y, self.model['dv']['y'].trainable_weights))
#         self.optimizer['dv_xy'].apply_gradients(zip(gradients_dv_xy, self.model['dv']['xy'].trainable_weights))