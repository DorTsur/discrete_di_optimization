import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
import logging
import os
from losses.dv_loss import Model_loss
from metrics.metrics import pdine_metrics
from trainers.savers import pdine_saver
from optimizers.lr import exp_dec_lr



logger = logging.getLogger("logger")


# For new pdine version
class pdine_trainer(object):
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
        self.set_hyper_params(config, data)

    def set_hyper_params(self, config, data):
        self.learning_rate = config.lr  # the model's learning rate

        if config.decay == 1:  # set lr decay
            self.learning_rate_dv = exp_dec_lr(config, data, config.lr)
            self.learning_rate_enc = exp_dec_lr(config, data, config.lr / 4)
        else:
            self.learning_rate_dv = config.lr
            self.learning_rate_enc = config.lr / 4

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.bptt = config.bptt

        # set optimizers
        if config.optimizer == "adam":
            self.optimizer = {"dv_y": Adam(amsgrad=True, learning_rate=self.learning_rate_dv),
                              "dv_xy": Adam(amsgrad=True, learning_rate=self.learning_rate_dv),
                              "enc": Adam(amsgrad=True, learning_rate=self.learning_rate_enc)}  # the model's optimizers
        else:
            self.optimizer = {"dv_y": SGD(learning_rate=config.lr_SGD),
                              "dv_xy": SGD(learning_rate=config.lr_SGD),
                              "enc": SGD(learning_rate=config.lr_SGD / 4)
                              }

        # set visualizer, metrics, etc.
        self.saver = pdine_saver(config)
        self.global_step_enc = tf.Variable(0, trainable=False, dtype=tf.int64)

        self.metrics = {"train": pdine_metrics(config.train_writer, name='dv_train'),
                        "eval": pdine_metrics(config.test_writer, name='dv_eval')}  # the trainer's metrics
        self.feedback = (config.feedback == 1)
        self.T = config.T

        if self.feedback:
            self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim + config.y_dim])
        else:
            self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim])

    def train(self):
        for epoch in range(self.config.num_epochs):
            if epoch % self.config.eval_freq == 0:
                self.evaluate(epoch)  # perform evaluation step
                continue
            self.train_epoch(epoch)  # perform a training epoch

        self.evaluate(self.config.num_epochs, iterator="long_eval")  # perform a long evaluation

    def train_epoch(self, epoch):
        self.metrics["train"].reset_states()
        self.reset_model_states()  # reset model and metrics states
        if np.random.rand() > 0.3 or epoch < 20:  # train DINE
            model_name = "DV epoch"
            for sample in self.data_iterators["train"]():
                sample = self.fb_input
                output = self.train_dine_step(sample)  # calculate model outputs and perform a training step
                self.metrics["train"].update_state(output)  # update trainer metrics
        else:  # train Enc.
            model_name = "Encoder epoch"
            for sample in self.data_iterators["train"]():
                output = self.train_enc_step(self.fb_input)  # calculate model outputs and perform a training step
                self.metrics["train"].update_state(output)  # update trainer metrics

        self.metrics["train"].log_metrics(epoch, model_name=model_name)  # log updated metrics

    def train_dine_step(self, sample):
        outputs = self.compute_dine_grads(sample)  # calculate gradients
        self.apply_grads(outputs['grad_dv_y'], outputs['grad_dv_xy'])  # apply gradients
        return outputs

    def train_enc_step(self, sample):
        outputs = self.compute_enc_grads(sample)  # calculate gradients
        self.apply_enc_grads(outputs['grad_enc'])  # apply gradients
        return outputs

    def compute_dine_grads(self, sample):  # gradients computation method
        outputs = {}
        with tf.GradientTape(persistent=True) as tape:
            [outputs["x"], outputs["y"], outputs["p"], outputs["s"]] = self.model['enc'](sample, training=False)  # obtain samples through model
            xy = tf.concat([outputs["x"], outputs["y"]], axis=-1)  # t_xy model input
            ##
            (outputs["x"], outputs["y"]) = (tf.ones_like(outputs["x"]),tf.ones_like(outputs["y"]))
            ##
            outputs["t_y"] = self.model['dv']['y'](outputs["y"], training=True)
            outputs["t_xy"] = self.model['dv']['xy'](xy, training=True)  # calculate models outputs
            loss = self.loss_fn(outputs["t_y"], outputs["t_xy"])  # calculate loss for each model
            loss_y, loss_xy = tf.split(loss, num_or_size_splits=2, axis=0)


        # print("loss = {}".format(loss_xy-loss_y))
        gradients_dv_y = tape.gradient(loss_y, self.model['dv']['y'].trainable_weights)
        gradients_dv_xy = tape.gradient(loss_xy, self.model['dv']['xy'].trainable_weights)  # calculate gradients

        outputs['grad_dv_y'], outputs['grad_norm_dv_y'] = tf.clip_by_global_norm(gradients_dv_y, self.config.clip_grad_norm)
        outputs['grad_dv_xy'], outputs['grad_norm_dv_xy'] = tf.clip_by_global_norm(gradients_dv_xy, self.config.clip_grad_norm)  # normalize gradients

        with self.config.train_writer.as_default():  # update trainer metrics
            tf.summary.scalar("grad_norm_dv_y", outputs['grad_norm_dv_y'], self.global_step)
            tf.summary.scalar("grad_norm_dv_xy", outputs['grad_norm_dv_xy'], self.global_step)
            self.global_step.assign_add(1)

        # Check if this is allowed in tf.function:
        if self.feedback:
            x_ = tf.split(outputs["x"], axis=1, num_or_size_splits=self.config.bptt)[-1]
            y_ = tf.split(outputs["y"], axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = tf.concat([x_, y_], axis=-1)
        else:
            x_ = tf.split(outputs["x"], axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = x_


        return outputs

    def compute_enc_grads(self, sample):  # gradients computation method
        outputs = {}
        with tf.GradientTape(persistent=True) as tape:
            [outputs["x"], outputs["y"], outputs["p"], outputs["s"]] = self.model['enc'](sample, training=True)  # obtain samples through model
            xy = tf.concat([outputs["x"], outputs["y"]], axis=-1)  # t_xy model input
            outputs["t_y"] = self.model['dv']['y'](outputs["y"], training=False)
            outputs["t_xy"] = self.model['dv']['xy'](xy, training=False)  # calculate models outputs
            loss = self.calc_reinforce(outputs["x"], outputs["p"], outputs["t_xy"], outputs["t_y"])

        gradients_enc = tape.gradient(loss, self.model['enc'].trainable_weights)

        outputs['grad_enc'], outputs['grad_norm_enc'] = tf.clip_by_global_norm(gradients_enc, self.config.clip_grad_norm_enc)

        with self.config.train_writer.as_default():  # update trainer metrics
            tf.summary.scalar("grad_norm_enc", outputs['grad_norm_enc'], self.global_step)
            tf.summary.scalar("mdp_loss", loss, self.global_step)
            self.global_step.assign_add(1)

        # Check if this is allowed in tf.function:
        if self.feedback:
            x_ = tf.split(outputs["x"], axis=1, num_or_size_splits=self.config.bptt)[-1]
            y_ = tf.split(outputs["y"], axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = tf.concat([x_, y_], axis=-1)
        else:
            x_ = tf.split(outputs["x"], axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = x_

        return outputs

    def calc_reinforce(self,x,p,t_xy,t_y):
        """
        Calculating the REINFORCE objective function
        """
        lp = tf.math.log(tf.where(tf.equal(x,1), p, 1-p))
        r = t_xy[0] - t_y[0]
        r = tf.squeeze(r, axis=-1)
        rho = tf.reduce_mean(r)
        T = self.T
        sum_list = [tf.reduce_sum(tf.slice(r, [0, k, 0], [tf.shape(r)[0], T, tf.shape(r)[-1]]), axis=1, keepdims=True)
                    for k in range(tf.shape(r)[1] - T)]  # [B,1,1]
        rt = tf.concat(sum_list, axis=1)  # [B,bptt-T,1]
        [lp, _] = tf.split(lp, axis=1, num_or_size_splits=[self.bptt - T, T])
        loss = -tf.reduce_mean(lp * (rt - rho))
        return loss

    def apply_enc_grads(self, gradients_enc):
        self.optimizer['enc'].apply_gradients(zip(gradients_enc, self.model['enc'].trainable_weights))

    def apply_grads(self, gradients_dv_y, gradients_dv_xy):
        self.optimizer['dv_y'].apply_gradients(zip(gradients_dv_y, self.model['dv']['y'].trainable_weights))
        self.optimizer['dv_xy'].apply_gradients(zip(gradients_dv_xy, self.model['dv']['xy'].trainable_weights))

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

    def evaluate(self, epoch, iterator="eval"):
        self.sync_eval_model()
        self.metrics["eval"].reset_states()
        self.reset_model_states()
        self.saver.reset_state()

        outputs_eval = {"p": [],
                   "t_y": [],
                   "t_xy":[],
                   "states":[]}

        for input_batch in self.data_iterators[iterator]():
            input_batch = self.fb_input
            output = self.eval_step(input_batch)
            outputs_eval["p"].append(output["p"])
            outputs_eval["t_y"].append(output["t_y"])
            outputs_eval["t_xy"].append(output["t_xy"])
            outputs_eval["states"].append(output["states_t_y"])
            self.metrics["eval"].update_state(output)

        self.metrics["eval"].log_metrics(epoch, model_name="Evaluation")
        p_t = tf.concat(outputs_eval["p"], axis=1)
        t_y_t = tf.concat([d[0] for d in outputs_eval["t_y"]], axis=1)
        if self.config.compress_dv_flag:
            states = tf.concat(outputs_eval["states"], axis=1)

        with self.config.test_writer.as_default():
            tf.summary.histogram(name="p_hist", data=p_t, step=self.global_step, buckets=50)
            tf.summary.histogram(name="t_y_hist", data=t_y_t, step=self.global_step, buckets=50)
            if self.config.compress_dv_flag:
                tf.summary.histogram(name="compressed_t_y_hist", data=states, step=self.global_step, buckets=50)

        self.saver.save(None if epoch > 0 else 'begin.mat')
        self.saver.save_models(models=self.model, path=os.path.join(self.config.tensor_board_dir, "models_weights"))

    def eval_step(self, input_batch):
        outputs = {}
        [outputs["x"], outputs["y"], outputs["p"], outputs["s"]] = self.model['enc'](input_batch, training=False)
        xy = tf.concat([outputs["x"], outputs["y"]], axis=-1)
        outputs["t_y"] = self.model['dv_eval']['y'](outputs["y"], training=False)
        outputs["t_xy"] = self.model['dv_eval']['xy'](xy, training=False)
        if self.config.compress_dv_flag:
            # In this case t_y = [t,et,states]
            outputs["states_t_y"] = outputs["t_y"][2]
        else:
            outputs["states_t_y"] = 0
        # outputs["t_y"] = [outputs["t_y"][0], outputs["t_y"][1]]
        # outputs["t_xy"] = [outputs["t_xy"][0], outputs["t_xy"][1]]

        self.saver.update_state(outputs)

        if self.feedback:
            x_ = tf.split(outputs["x"], axis=1, num_or_size_splits=self.config.bptt)[-1]
            y_ = tf.split(outputs["y"], axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = tf.concat([x_, y_], axis=-1)
        else:
            x_ = tf.split(outputs["x"], axis=1, num_or_size_splits=self.config.bptt)[-1]
            self.fb_input = x_

        return outputs

    def sync_eval_model(self):
        # sync DV:
        w_y = self.model['dv']['y'].get_weights()
        w_xy = self.model['dv']['xy'].get_weights()
        self.model['dv_eval']['y'].set_weights(w_y)
        self.model['dv_eval']['xy'].set_weights(w_xy)
        # sync enc:
        w_enc = self.model['enc'].get_weights()  # similarly sync encoder model
        self.model['enc_eval'].set_weights(w_enc)

