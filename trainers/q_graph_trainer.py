import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
import os
from metrics.metrics import  Q_est_metrics
from trainers.savers import Q_est_saver
from data.Ising_FB_data_gen import IsingChannel_state
from optimizers.lr import exp_dec_lr
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical


class QgraphTrainer(object):
    def __init__(self, model, data, config):
        self.config = config
        self.model = model
        self.channel_name = config.channel_name
        self.load_enc_weights()
        self.data_iterator = data
        self.loss_function = CategoricalCrossentropy()  # q-graph estimator loss function
        self.feedback = config.feedback

        # learning rate:
        self.learning_rate = config.lr  # the model's learning rate
        if config.decay == 1:
            self.learning_rate = exp_dec_lr(config, data, config.lr / 4)
        else:
            self.learning_rate = config.lr / 4

        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.bptt = config.bptt
        if config.optimizer == "adam":
            self.optimizer = {"q_graph": Adam(amsgrad=True, learning_rate=self.learning_rate)}
        else:
            self.optimizer = {"q_graph": SGD(learning_rate=config.lr_SGD / 4)}  # the model's optimizers

        self.saver = Q_est_saver(config)

        # metrics:
        self.global_step_enc = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.metrics = {"train": Q_est_metrics(config.train_writer, name='dv_train'),
                            "eval": Q_est_metrics(config.test_writer, name='dv_eval')}

        if self.feedback:
            self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim + config.y_dim])
        else:
            self.fb_input = tf.zeros(shape=[config.batch_size, 1, config.x_dim])


        # external data generator:
        self.external_gen = False
        if self.external_gen:
            self.data_gen = IsingChannel_state(bptt=config.bptt, input_shape=[config.batch_size,1])
            self.data_gen._call_()

    def evaluate(self, epoch, iterator="eval"):
        self.sync_eval_model()
        self.metrics["eval"].reset_states()
        self.reset_model_states()
        self.saver.reset_state()
        s_t = []
        q_t = []

        if iterator == "eval":
            eval_batches = 10
        elif iterator == "long_eval":
            eval_batches = 100
        else:
            eval_batches = 1

        for _ in range(self.config.batches*eval_batches):
            input_batch = self.fb_input
            output = self.eval_step(input_batch)  # output = [q_est,s_onehot]
            q_t.append(output[0])
            s_t.append(output[1])
            self.metrics["eval"].update_state(output)

        self.metrics["eval"].log_metrics(epoch)
        q_t = tf.concat(q_t, axis=1)
        s_t = tf.concat(s_t, axis=1)
        with self.config.test_writer.as_default():
            tf.summary.histogram(name="Q_0 hist", data=q_t[:-1,0], step=self.global_step, buckets=50)
            tf.summary.histogram(name="Q_1 hist", data=q_t[:-1,1], step=self.global_step, buckets=50)

        name = None if epoch > 0 else 'begin.mat'
        self.saver.save(name=name)
        self.saver.save_models(models=self.model, path=self.config.tensor_board_dir)

    def eval_step(self, input_batch):
        if self.external_gen:
            [x, y] = self.data_gen._gen_()
            s = x
        else:
            [x, y, p, s] = self.model['enc'](input_batch, training=False)  # obtain samples through model
        q_est = self.model['q_graph_test'](y, training=True)
        s_onehot = to_categorical(s, num_classes=2)
        # self.saver.histogram(q_est)

        self.saver.update_state([q_est, s_onehot, y])

        x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
        y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]

        if self.feedback:
            self.fb_input = tf.concat([x_, y_], axis=-1)
        else:
            self.fb_input = x_

        return [q_est, s_onehot]

    def load_enc_weights(self):
        """
        model to set weights is encoder
        """
        # if self.channel_name == 'ising':
        #     p_val = self.config.p_ising
        # elif self.channel_name == 'trapdoor':
        #     p_val = self.config.p_trapdoor
        # elif self.channel_name == 'post':
        #     p_val = self.config.p_post
        # else:
        #     p_val = 0.5
        # dir_name = "enc_" + self.config.channel_name + str(p_val)
        # dir_name = os.path.join("pretrained_models_weights","enc_ising_01/enc.index")
        # weights_path = os.path.join(os.getcwd(),dir_name)
        # self.model['enc'].load_weights(weights_path)
        # for loading model:
        # loading model:
        # self.model['enc'] = None
        # filepath = os.path.join("./pretrained_models",self.config.channel_name+"_fb","p_05_4037_47k")
        # self.model['enc'] = tf.keras.models.load_model(filepath=filepath)

    #     loading weights:
    #     filepath = os.path.join("./pretrained_models", self.config.channel_name + "_fb", "p_02_emp_","weights")
        filepath = os.path.join("./pretrained_models", self.config.channel_name + "_ff", "1", "weights")
        self.model['enc'].load_weights(filepath)

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
        for _ in range(self.config.batches):
            sample = self.fb_input
            output = self.train_step(sample)  # calculate model outputs and perform a training step
            self.metrics["train"].update_state(output)  # update trainer metrics

        self.metrics["train"].log_metrics(epoch)  # log updated metrics

    def train_step(self, sample):
        gradients, data_metrics = self.compute_grads(sample)
        self.apply_grads(gradients)
        return data_metrics

    def compute_grads(self, sample):  # gradients computation method
        with tf.GradientTape(persistent=True) as tape:
            if self.external_gen:
                # external gen implemenets ising
                [x, y] = self.data_gen._gen_()
                s = x
            else:
                [x, y, p, s] = self.model['enc'](sample, training=False)  # obtain samples through model
            q_est = self.model['q_graph'](y, training=True)
            s_onehot = to_categorical(s, num_classes=2)
            loss = self.loss_function(s_onehot, q_est)

        gradients = tape.gradient(loss, self.model['q_graph'].trainable_weights)
        gradients, grad_norm = tf.clip_by_global_norm(gradients, self.config.clip_grad_norm)

        with self.config.train_writer.as_default():  # update trainer metrics
            tf.summary.scalar("grad_norm", grad_norm, self.global_step)
            tf.summary.scalar("QE_loss", loss, self.global_step)
            self.global_step.assign_add(1)

        x_ = tf.split(x, axis=1, num_or_size_splits=self.config.bptt)[-1]
        y_ = tf.split(y, axis=1, num_or_size_splits=self.config.bptt)[-1]

        if self.feedback:
            self.fb_input = tf.concat([x_, y_], axis=-1)
        else:
            self.fb_input = x_

        return [gradients, [q_est, s_onehot]]

    def apply_grads(self, gradients):
        self.optimizer['q_graph'].apply_gradients(zip(gradients, self.model['q_graph'].trainable_weights))

    def reset_model_states(self):
        def reset_recursively(models):
            for model in models.values():
                if isinstance(model, dict):
                    reset_recursively(model)
                else:
                    model.reset_states()

        reset_recursively(self.model)
        # if (self.config.channel_name == "ising") or (self.config.channel_name == "trapdoor"):
        #     self.model['enc'].layers[3].reset_states()
        #     self.model['enc_eval'].layers[3].reset_states()

    def sync_eval_model(self):
        # sync with evaluation model model:
        w = self.model['q_graph'].get_weights()
        self.model['q_graph_test'].set_weights(w)
